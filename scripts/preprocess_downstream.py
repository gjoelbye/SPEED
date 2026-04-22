"""
Downstream preprocessing script for event-based windowing with labels.

Processes EEG datasets for downstream tasks (motor imagery, seizure detection,
HBN CCD/CBCL/rest/symbolSearch/surroundSupp) by extracting event-locked windows
with labels. When ``n_jobs > 1``, files are processed concurrently via a
``ProcessPoolExecutor`` using the ``forkserver`` start method so each worker is
a single-threaded-BLAS process; BLAS thread env vars are pinned to 1 at the top
of this module so they are latched before numpy / MNE are imported in either
the parent or the workers (over-subscription silently kills the speedup).
"""

# Must run before any numpy / mne / pyprep / meegkit imports in this module OR
# in the forkserver helper (which re-imports this module). Safe if already set.
import os
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import glob
import logging
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union

import h5py
import mne
import numpy as np
from jsonargparse import CLI
from tqdm import tqdm

from speed.cache import (
    compute_config_hash,
    compute_input_hash,
    is_cached,
    load_cache,
    save_cache,
    update_cache,
)
from speed.pipeline import PretrainPipeline
from speed.provenance import save_provenance
from speed.utils import save_hdf5_with_labels


def configure_logging(filename: str, level: str = "INFO"):
    """Configure logging to file."""
    log_path = Path(filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=filename,
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
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


# -------------------------------------------------------------------------
# Worker-side state: populated by _init_worker at pool startup so the pipeline
# object is pickled ONCE per worker instead of once per file submission.
# -------------------------------------------------------------------------
_WORKER_PIPELINE: Optional[PretrainPipeline] = None


def _init_worker(pipeline: PretrainPipeline, log_path: str, log_level: str) -> None:
    """ProcessPoolExecutor initializer — runs in each worker after forkserver fork."""
    global _WORKER_PIPELINE
    _WORKER_PIPELINE = pipeline

    # Per-worker log file so multi-process writes don't interleave. Named next
    # to the main log, e.g. log.txt → log.worker_12345.txt.
    pid = os.getpid()
    lp = Path(log_path)
    worker_log = lp.parent / f"{lp.stem}.worker_{pid}{lp.suffix}"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(worker_log),
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - [pid=%(process)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
    )
    suppress_warnings()


def _process_file(file_path_str: str, pipeline: Optional[PretrainPipeline] = None) -> dict:
    """Run the full pipeline on one file, returning plain data (no MNE Raw objects).

    Usable both from the main process (serial fallback, ``pipeline`` passed in)
    and inside a ProcessPoolExecutor worker (``pipeline`` defaults to the
    module-global populated by ``_init_worker``).
    """
    if pipeline is None:
        pipeline = _WORKER_PIPELINE
    # Buffer is instance-level and may hold metrics from an earlier task that
    # ran in this same worker process. Clear so the returned slice is per-file.
    pipeline._quality_metrics_buffer.clear()

    try:
        result = pipeline.run([file_path_str])
        if not isinstance(result, tuple) or len(result) != 4:
            return {
                "file": file_path_str,
                "error": (
                    f"Expected 4-tuple from pipeline.run(), got {type(result).__name__}"
                    f"(len={len(result) if hasattr(result, '__len__') else 'N/A'})"
                ),
                "data": [], "labels": [], "times": [],
                "ch_names": [], "sfreq": 0.0, "quality_metrics": [],
            }

        raws, times, indices, labels = result
        data_arrays = [np.ascontiguousarray(r._data, dtype=np.float32) for r in raws]
        ch_names = list(raws[0].ch_names) if raws else []
        sfreq = float(raws[0].info["sfreq"]) if raws else 0.0
        metrics = list(pipeline._quality_metrics_buffer)

        return {
            "file": file_path_str,
            "error": None,
            "data": data_arrays,
            "labels": labels,
            "times": list(times),
            "ch_names": ch_names,
            "sfreq": sfreq,
            "quality_metrics": metrics,
        }
    except Exception as e:
        return {
            "file": file_path_str,
            "error": f"{type(e).__name__}: {e}",
            "data": [], "labels": [], "times": [],
            "ch_names": [], "sfreq": 0.0, "quality_metrics": [],
        }


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
    """Main preprocessing function for downstream tasks.

    Parameters
    ----------
    n_jobs : int
        Number of worker processes. If ``<= 1`` the serial fallback is used
        (preferred for tests and local debugging). Otherwise files are
        dispatched to a ``ProcessPoolExecutor`` with the ``forkserver`` start
        method; BLAS threads are pinned to 1 per worker so 12 workers on a
        12-CPU shard don't collectively spawn 144 BLAS threads.
    """
    suppress_warnings()

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    configure_logging(log_path, log_level)
    logging.info("Starting downstream preprocessing")
    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"Output: {out_path}")
    logging.info(f"Event windowing: {pipeline.event_windowing}")
    logging.info(f"Event labels: {pipeline.event_labels}")
    logging.info(f"n_jobs: {n_jobs}")

    if not pipeline.event_windowing:
        raise ValueError("Pipeline must have event_windowing=True for downstream preprocessing")

    file_paths = discover_files(dataset_path, file_extension)
    logging.info(f"Found {len(file_paths)} files to process")

    if len(file_paths) == 0:
        logging.warning("No files found to process")
        return

    # Cache-based filtering — unchanged from round-4 resume protocol.
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
    # Otherwise, natural input-file order. A longest-first sort was tried but
    # OOM-killed every shard — with 12 workers starting the 12 longest files
    # in lockstep, meegkit DSS + pyprep NoisyChannels transient copies peaked
    # at >48 GB collectively. Alphabetical-by-subject order mixes durations
    # across pool slots and avoids the collision.

    # Accumulators for the batch flush logic. Identical semantics to the
    # previous serial loop — just fed by either `_process_file` (serial) or
    # `as_completed` (parallel).
    all_raws: List[np.ndarray] = []
    all_labels: List = []
    all_times: List = []
    all_src_per_window: List[Path] = []
    batch_id = 0
    total_windows = 0

    def _save_batch(data_arrays, labels, times, src_per_window, batch_id):
        hdf5_path = out_path / f"batch_{batch_id:05d}.hdf5"

        if hdf5_path.exists() and not overwrite:
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    if 'data' in f and f['data'].shape[0] == len(data_arrays):
                        logging.info(
                            f"Batch {batch_id}: Already exists and valid "
                            f"({len(data_arrays)} windows), skipping."
                        )
                        return
                    logging.warning(f"Batch {batch_id}: Exists but invalid/incomplete. Overwriting.")
            except Exception:
                logging.warning(f"Batch {batch_id}: Exists but corrupted. Overwriting.")

        label_descriptions = list(pipeline.label_mapping.keys())
        unique_src_paths = list(dict.fromkeys(src_per_window))
        indices = [unique_src_paths.index(p) for p in src_per_window]

        save_hdf5_with_labels(
            data_arrays=data_arrays,
            labels=labels,
            label_descriptions=label_descriptions,
            src_paths=unique_src_paths,
            times=times,
            indices=indices,
            dest_path=hdf5_path,
        )

        logging.info(f"Batch {batch_id}: Saved {len(data_arrays)} windows to {hdf5_path}")

    def _commit_batch_cache(cache_dict, src_per_window):
        """Persist cache for every file contributing to the just-saved batch."""
        nonlocal cfg_hash
        if not use_cache or not src_per_window:
            return
        if cfg_hash is None:
            cfg_hash = compute_config_hash(pipeline)
        for fp in dict.fromkeys(src_per_window):
            try:
                ih = compute_input_hash(str(fp))
                update_cache(cache_dict, str(fp), ih, cfg_hash, str(out_path))
            except OSError:
                pass
        save_cache(str(out_path), cache_dict)

    def _handle_result(result: dict) -> None:
        nonlocal batch_id, total_windows
        nonlocal all_raws, all_labels, all_times, all_src_per_window

        file_path = Path(result["file"])

        if result["error"]:
            logging.error(f"Error processing {file_path.name}: {result['error']}")
            return
        if not result["data"]:
            logging.info(f"No windows extracted from {file_path.name}")
            return

        all_raws.extend(result["data"])
        all_labels.extend(result["labels"])
        all_times.extend(result["times"])
        all_src_per_window.extend([file_path] * len(result["data"]))
        # Preserve the quality-metrics contract: main-process pipeline buffer
        # accumulates per-file metrics so get_quality_metrics() works after.
        pipeline._quality_metrics_buffer.extend(result["quality_metrics"])

        logging.info(f"{file_path.name}: Extracted {len(result['data'])} windows")

        if len(all_raws) >= batch_size:
            _save_batch(all_raws, all_labels, all_times, all_src_per_window, batch_id)
            total_windows += len(all_raws)
            _commit_batch_cache(cache, all_src_per_window)
            batch_id += 1
            all_raws, all_labels, all_times, all_src_per_window = [], [], [], []

    if n_jobs <= 1 or len(file_paths) <= 1:
        # Serial fallback. Keeps tests/debug simple and avoids pool startup cost
        # when there's nothing to parallelise.
        for file_path in tqdm(file_paths, desc="Processing files"):
            result = _process_file(str(file_path), pipeline=pipeline)
            _handle_result(result)
    else:
        # forkserver: fork+BLAS deadlocks on OpenBLAS; spawn pays a ~3-5s
        # per-worker import cost. forkserver imports the script ONCE in a
        # helper process, then forks cheaply.
        ctx = mp.get_context("forkserver")
        logging.info(f"Spawning ProcessPoolExecutor (forkserver, max_workers={n_jobs})")
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(pipeline, log_path, log_level),
        ) as ex:
            futures = {ex.submit(_process_file, str(fp)): fp for fp in file_paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                fp = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    logging.error(f"Worker crashed processing {fp.name}: {type(e).__name__}: {e}")
                    continue
                _handle_result(result)

    # Flush the tail.
    if len(all_raws) > 0:
        _save_batch(all_raws, all_labels, all_times, all_src_per_window, batch_id)
        total_windows += len(all_raws)
        _commit_batch_cache(cache, all_src_per_window)

    logging.info(f"Preprocessing complete. Total windows: {total_windows}")
    logging.info(f"Created {batch_id + 1} HDF5 files in {out_path}")

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
    """CLI entry point for downstream preprocessing."""
    if not pipeline.event_windowing:
        raise ValueError("Pipeline must have event_windowing=True for downstream preprocessing")

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
