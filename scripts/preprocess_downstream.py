"""
Downstream preprocessing script for event-based windowing with labels.

This script processes EEG datasets for downstream tasks (e.g., motor imagery,
seizure detection) by extracting event-locked windows with labels.

Usage:
    python scripts/preprocess_downstream.py --config configs/downstream/eegmmidb.yaml
    python scripts/preprocess_downstream.py --config configs/downstream/chbmit.yaml
"""

import os
import glob
import logging
import warnings
from pathlib import Path
from typing import List, Union, Optional

import mne
from tqdm import tqdm
from jsonargparse import CLI

from speed.cache import load_cache, save_cache, compute_input_hash, compute_config_hash, is_cached, update_cache
from speed.pipeline import PretrainPipeline
from speed.provenance import save_provenance
from speed.utils import save_hdf5_with_labels


def configure_logging(filename: str, level: str = "INFO"):
    """Configure logging to file."""
    # Create log directory if it doesn't exist
    log_path = Path(filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=filename,
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )


def suppress_warnings():
    """Suppress common MNE and library warnings."""
    mne.set_log_level("ERROR")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=ModuleNotFoundError)
    warnings.filterwarnings("ignore", message=r".*pymatreader cannot import.*", category=UserWarning)


def discover_files(dataset_path: str, file_extensions: Union[str, List[str]] = ".edf") -> List[Path]:
    """Find all source files to process from directory or text file."""
    if os.path.isfile(dataset_path) and dataset_path.endswith(".txt"):
        with open(dataset_path, "r") as f:
            return [Path(line.strip()) for line in f if line.strip()]

    if os.path.isdir(dataset_path):
        extensions = [file_extensions] if isinstance(file_extensions, str) else file_extensions
        paths = []
        for ext in extensions:
            pattern = f"*{ext}" if ext.startswith(".") else f"*.{ext}"
            paths.extend(glob.glob(os.path.join(dataset_path, "**", pattern), recursive=True))
        return [Path(p) for p in paths]

    raise ValueError(f"Invalid dataset path: {dataset_path}. Must be a directory or .txt file.")


def preprocess_downstream(
    pipeline: PretrainPipeline,
    dataset_path: str,
    out_path: str,
    log_path: str,
    file_extension: Union[str, List[str]] = ".edf",
    batch_size: int = 100,
    n_jobs: int = 1,
    overwrite: bool = False,
    shuffle_files: bool = False,
    log_level: str = "INFO",
    use_cache: bool = True,
) -> None:
    """
    Main preprocessing function for downstream tasks.

    Parameters
    ----------
    pipeline : PretrainPipeline
        Pipeline configured with event_windowing=True
    dataset_path : str
        Path to dataset directory or .txt file with file paths
    out_path : str
        Output directory for HDF5 files
    log_path : str
        Path to log file
    file_extension : str or list of str
        File extension(s) to process (e.g., ".edf")
    batch_size : int
        Number of windows per HDF5 file (approximate)
    n_jobs : int
        Number of parallel workers (currently sequential only)
    overwrite : bool
        If True, overwrite existing files
    shuffle_files : bool
        If True, randomly shuffle file order
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    suppress_warnings()

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    configure_logging(log_path, log_level)
    logging.info(f"Starting downstream preprocessing")
    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"Output: {out_path}")
    logging.info(f"Event windowing: {pipeline.event_windowing}")
    logging.info(f"Event labels: {pipeline.event_labels}")

    if n_jobs > 1:
        logging.warning(
            f"n_jobs={n_jobs} specified but parallel processing is not yet "
            f"implemented for downstream. Processing will be sequential."
        )

    # Validate pipeline configuration
    if not pipeline.event_windowing:
        raise ValueError("Pipeline must have event_windowing=True for downstream preprocessing")

    # Discover files
    file_paths = discover_files(dataset_path, file_extension)
    logging.info(f"Found {len(file_paths)} files to process")

    if len(file_paths) == 0:
        logging.warning("No files found to process")
        return

    # Cache-based filtering
    cache = {}
    cfg_hash = None
    if use_cache and not overwrite and file_paths:
        cache = load_cache(str(out_path))
        cfg_hash = compute_config_hash(pipeline)
        original_count = len(file_paths)
        filtered = []
        for p in file_paths:
            try:
                ih = compute_input_hash(str(p))
                if not is_cached(cache, str(p), ih, cfg_hash):
                    filtered.append(p)
            except OSError:
                filtered.append(p)
        cache_skipped = original_count - len(filtered)
        if cache_skipped:
            logging.info(f"Skipping {cache_skipped} files via cache.")
        file_paths = filtered

    if shuffle_files:
        import random
        random.shuffle(file_paths)
        logging.info("Shuffled file order")

    # Process files one by one, accumulating windows in batches
    # Track source path per window directly (simpler than index mapping)
    all_raws, all_labels, all_times, all_src_per_window = [], [], [], []
    batch_id = 0
    total_windows = 0

    def _save_batch(raws, labels, times, src_per_window, batch_id):
        """Save accumulated windows to an HDF5 batch file."""
        hdf5_path = out_path / f"batch_{batch_id:05d}.hdf5"

        if hdf5_path.exists() and not overwrite:
            # Verify existing batch is valid and has the expected number of windows
            try:
                import h5py
                with h5py.File(hdf5_path, 'r') as f:
                    if 'data' in f and f['data'].shape[0] == len(raws):
                        logging.info(f"Batch {batch_id}: Already exists and valid ({len(raws)} windows), skipping.")
                        return
                    else:
                        logging.warning(f"Batch {batch_id}: Exists but invalid/incomplete. Overwriting.")
            except Exception:
                logging.warning(f"Batch {batch_id}: Exists but corrupted. Overwriting.")

        label_descriptions = list(pipeline.label_mapping.keys())

        # Build unique source paths list (order-preserving)
        unique_src_paths = list(dict.fromkeys(src_per_window))
        indices = [unique_src_paths.index(p) for p in src_per_window]

        save_hdf5_with_labels(
            raws=raws,
            labels=labels,
            label_descriptions=label_descriptions,
            src_paths=unique_src_paths,
            times=times,
            indices=indices,
            dest_path=hdf5_path
        )

        logging.info(f"Batch {batch_id}: Saved {len(raws)} windows to {hdf5_path}")

    def _commit_batch_cache(cache_dict, src_per_window):
        """Persist cache entries for every file contributing to the just-saved batch.

        Called only after _save_batch succeeds, so the cache on disk reflects
        exactly the files whose windows are in persisted HDF5 files. Cheap:
        one atomic JSON write (~1 MB for a completed CCD run) per batch, i.e.
        roughly once per 100 windows / ~5 files.
        """
        nonlocal cfg_hash
        if not use_cache or not src_per_window:
            return
        if cfg_hash is None:
            cfg_hash = compute_config_hash(pipeline)
        # dict.fromkeys preserves insertion order while deduplicating.
        for fp in dict.fromkeys(src_per_window):
            try:
                ih = compute_input_hash(str(fp))
                update_cache(cache_dict, str(fp), ih, cfg_hash, str(out_path))
            except OSError:
                pass
        save_cache(str(out_path), cache_dict)

    for file_path in tqdm(file_paths, desc="Processing files"):
        try:
            result = pipeline.run([str(file_path)])

            if not isinstance(result, tuple) or len(result) != 4:
                logging.error(
                    f"Expected 4-tuple from pipeline.run() for {file_path.name}, "
                    f"got {type(result).__name__}"
                    f"(len={len(result) if hasattr(result, '__len__') else 'N/A'})"
                )
                continue

            raws, times, indices, labels = result

            if len(raws) == 0:
                logging.info(f"No windows extracted from {file_path.name}")
                continue

            # Accumulate windows
            all_raws.extend(raws)
            all_labels.extend(labels)
            all_times.extend(times)
            all_src_per_window.extend([file_path] * len(raws))

            logging.info(f"{file_path.name}: Extracted {len(raws)} windows")

            # Save batch when full. Cache entries for files whose windows end
            # up in this batch are persisted here (see _commit_batch_cache) so
            # that a SIGTERM between batch flushes cannot leave the cache
            # claiming a file is done while its windows are still in memory.
            if len(all_raws) >= batch_size:
                _save_batch(all_raws, all_labels, all_times, all_src_per_window, batch_id)
                total_windows += len(all_raws)
                _commit_batch_cache(cache, all_src_per_window)
                batch_id += 1
                all_raws, all_labels, all_times, all_src_per_window = [], [], [], []

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            continue

    # Save remaining windows and persist cache entries for their files.
    if len(all_raws) > 0:
        _save_batch(all_raws, all_labels, all_times, all_src_per_window, batch_id)
        total_windows += len(all_raws)
        _commit_batch_cache(cache, all_src_per_window)

    logging.info(f"Preprocessing complete. Total windows: {total_windows}")
    logging.info(f"Created {batch_id + 1} HDF5 files in {out_path}")

    # Save provenance
    try:
        config_dict = {
            "pipeline": {
                "class_path": type(pipeline).__module__ + "." + type(pipeline).__name__,
                "init_args": {
                    k: v for k, v in vars(pipeline).items()
                    if not k.startswith("_")
                },
            },
            "dataset_path": dataset_path,
            "out_path": str(out_path),
            "file_extension": file_extension,
            "batch_size": batch_size,
            "overwrite": overwrite,
        }
        n_output = batch_id + 1 if total_windows > 0 else 0
        save_provenance(str(out_path), config_dict, len(file_paths), n_output)
    except Exception as e:
        logging.warning(f"Failed to save provenance: {e}")


def main(
    pipeline: PretrainPipeline,
    dataset_path: str,
    out_path: str,
    log_path: str,
    file_extension: Union[str, List[str]] = ".edf",
    batch_size: int = 100,
    n_jobs: int = 1,
    overwrite: bool = False,
    shuffle_files: bool = False,
    log_level: str = "INFO",
    use_cache: bool = True,
) -> None:
    """
    CLI entry point for downstream preprocessing.

    Parameters
    ----------
    pipeline : PretrainPipeline
        Pipeline configured with event_windowing=True
    dataset_path : str
        Path to dataset directory or .txt file with file paths
    out_path : str
        Output directory for HDF5 files
    log_path : str
        Path to log file
    file_extension : str or list of str
        File extension(s) to process
    batch_size : int
        Number of windows per HDF5 file
    n_jobs : int
        Number of parallel workers
    overwrite : bool
        If True, overwrite existing files
    shuffle_files : bool
        If True, randomly shuffle file order
    log_level : str
        Logging level
    """
    # Validate pipeline configuration
    if not pipeline.event_windowing:
        raise ValueError("Pipeline must have event_windowing=True for downstream preprocessing")

    # Run preprocessing
    preprocess_downstream(
        pipeline=pipeline,
        dataset_path=dataset_path,
        out_path=out_path,
        log_path=log_path,
        file_extension=file_extension,
        batch_size=batch_size,
        n_jobs=n_jobs,
        overwrite=overwrite,
        shuffle_files=shuffle_files,
        log_level=log_level,
        use_cache=use_cache,
    )


if __name__ == "__main__":
    CLI(main)
