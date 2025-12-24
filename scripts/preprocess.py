import os
import glob
import logging
import warnings
import fcntl
from pathlib import Path
from time import sleep
from typing import List, Union, Optional, Dict, Any

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from jsonargparse import CLI

from speed.pipeline import PretrainPipeline
from speed.utils import (
    save_hdf5,
    write_bdf_from_raw,
    write_raw_to_file,
    copy_non_eeg_files,
    DEFAULT_EXCLUDE_PATTERNS
)


def configure_logging(filename: str, level: str = "INFO"):
    """Configure logging to file."""
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


def save_quality_metrics(metrics: List[dict], metrics_path: Path) -> None:
    """Append quality metrics to CSV with thread-safe file locking."""
    if not metrics:
        return
    
    metrics_path = Path(metrics_path)
    if metrics_path.suffix == '.csv':
        metrics_path = metrics_path.parent
    
    metrics_path.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_path / "quality_metrics.csv"
    lock_path = csv_path.with_suffix('.csv.lock')
    
    df = pd.DataFrame(metrics)
    lock_file = None
    
    try:
        lock_file = open(lock_path, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    except Exception as e:
        logging.warning(f"Error writing quality metrics: {e}")
    finally:
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass


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


def get_already_processed(
    out_path: str,
    output_format: str = "hdf5",
    file_extensions: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """Get list of already processed file stems."""
    processed = []
    
    if output_format == "hdf5":
        import h5py
        for hdf5_file in glob.glob(f"{out_path}/**/*.hdf5", recursive=True):
            with h5py.File(hdf5_file, "r") as f:
                processed.extend(f.attrs["files"].tolist())
    else:
        extensions = {".edf", ".bdf", ".set"}
        if file_extensions:
            exts = [file_extensions] if isinstance(file_extensions, str) else file_extensions
            extensions.update(exts)
        
        for ext in extensions:
            pattern = f"*{ext}" if ext.startswith(".") else f"*.{ext}"
            for f in glob.glob(os.path.join(out_path, "**", pattern), recursive=True):
                processed.append(Path(f).stem)
    
    return processed


def compute_dest_path(src_path: Path, dataset_root: Path, out_path: Path) -> Path:
    """Compute destination path preserving folder structure."""
    try:
        rel_path = src_path.relative_to(dataset_root)
    except ValueError:
        rel_path = Path(src_path.name)
    return out_path / rel_path


def create_dest_paths(out_path: Path, n_batches: int) -> List[Path]:
    """Create non-conflicting HDF5 destination paths."""
    paths = []
    idx = 1
    while len(paths) < n_batches:
        candidate = out_path / f"data_{idx}.hdf5"
        if not candidate.exists():
            paths.append(candidate)
        idx += 1
    return paths


def preprocess_batch(
    pipeline: PretrainPipeline,
    src_paths: List[Path],
    dest_path: Path,
    conf_log,
    save_as_hdf5: bool = True,
    sleep_after: float = 5.0,
    metrics_path: Optional[Path] = None
) -> None:
    """Process a batch of files (pretrain mode)."""
    suppress_warnings()
    conf_log()
    
    logging.debug(f"Starting preprocessing of {len(src_paths)} files...")
    
    raws, times, indices = pipeline(src_paths)
    
    if metrics_path:
        quality_metrics = pipeline.get_quality_metrics()
        if quality_metrics:
            save_quality_metrics(quality_metrics, metrics_path)
    
    if not raws:
        logging.debug("No valid data to save. Skipping batch.")
        return
    
    logging.debug("Saving preprocessed data...")
    
    if save_as_hdf5:
        save_hdf5(raws, src_paths, times, indices, dest_path)
    else:
        if len(raws) != 1:
            raise ValueError("Per-file output requires exactly one window per file.")
        write_bdf_from_raw(raws[0], str(dest_path))
    
    logging.debug(f"Saved to {dest_path}. File size: {dest_path.stat().st_size / 1e6:.2f} MB.")
    
    del raws, times, indices
    sleep(sleep_after)


def process_single_file(
    pipeline: PretrainPipeline,
    src_path: Path,
    dest_path: Path,
    conf_log,
    output_format: str = "auto",
    fallback_format: str = "edf",
    overwrite: bool = False,
    skip_on_quality_fail: bool = False,
    metrics_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Process a single EEG file (downstream mode)."""
    suppress_warnings()
    conf_log()
    
    result = {
        "src_path": str(src_path),
        "dest_path": str(dest_path),
        "status": "unknown",
        "quality_passed": None,
        "error": None
    }
    
    if dest_path.exists() and not overwrite:
        result["status"] = "skipped"
        logging.info(f"Skipping {src_path.name}: output already exists.")
        return result
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        raw, quality_metrics, quality_passed = pipeline.process_single(
            src_path, skip_on_quality_fail=skip_on_quality_fail
        )
        
        result["quality_passed"] = quality_passed
        
        if raw is None:
            result["status"] = "quality_failed" if not quality_passed else "failed"
            return result
        
        # Determine output format
        if output_format == "auto":
            out_fmt = src_path.suffix.lower().lstrip('.')
            if out_fmt not in ("edf", "bdf", "set"):
                out_fmt = fallback_format
        else:
            out_fmt = output_format
        
        # Adjust extension
        ext_map = {"edf": ".edf", "bdf": ".bdf", "set": ".set"}
        expected_ext = ext_map.get(out_fmt, f".{fallback_format}")
        if dest_path.suffix.lower() != expected_ext:
            dest_path = dest_path.with_suffix(expected_ext)
            result["dest_path"] = str(dest_path)
        
        actual_path = write_raw_to_file(
            raw, str(dest_path), format=out_fmt,
            write_annotations=True, fallback_format=fallback_format
        )
        result["dest_path"] = actual_path
        result["status"] = "success"
        
        logging.info(f"Saved {src_path.name} -> {Path(actual_path).name}")
        
        if metrics_path and quality_metrics:
            save_quality_metrics([{
                "filename": str(src_path),
                "oha": quality_metrics[0] if quality_metrics else None,
                "thv": quality_metrics[1] if quality_metrics else None,
                "chv": quality_metrics[2] if quality_metrics else None,
                "bcr": quality_metrics[3] if quality_metrics else None,
                "quality_passed": quality_passed
            }], metrics_path)
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logging.error(f"Error processing {src_path.name}: {e}")
    
    return result


def preprocess_dataset(
    pipeline: PretrainPipeline,
    dataset_path: str,
    out_path: str,
    log_path: str,
    overwrite: bool = False,
    shuffle_files: bool = True,
    n_jobs: int = 6,
    file_extension: Union[str, List[str]] = ".edf",
    log_level: str = "INFO",
    metrics_path: Optional[str] = None,
    batch_size: int = 10,
    save_as_hdf5: bool = True,
    sleep_after_batch: float = 5.0,
    preserve_structure: bool = False,
    output_format: str = "auto",
    fallback_format: str = "edf",
    skip_on_quality_fail: bool = False,
    copy_other_files: bool = False,
    exclude_patterns: Optional[List[str]] = None
):
    """
    Preprocess an EEG dataset.
    
    Parameters
    ----------
    pipeline : PretrainPipeline
        Configured preprocessing pipeline instance.
    dataset_path : str
        Path to dataset directory or .txt file with paths.
    out_path : str
        Output directory for preprocessed files.
    log_path : str
        Path to log file.
    overwrite : bool
        If False, skip already processed files.
    shuffle_files : bool
        Whether to randomly shuffle file order.
    n_jobs : int
        Number of parallel workers.
    file_extension : str or list of str
        File extensions to process.
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR.
    metrics_path : str, optional
        Directory to save quality metrics CSV.
    batch_size : int
        Number of files per HDF5 batch (pretrain mode).
    save_as_hdf5 : bool
        If True, save batched HDF5. If False, save individual BDF (pretrain mode).
    sleep_after_batch : float
        Seconds to sleep after each batch.
    preserve_structure : bool
        If True, preserve folder structure (downstream mode).
    output_format : str
        Output format for downstream mode: "auto", "edf", "bdf", or "set".
    fallback_format : str
        Fallback format if requested format unavailable.
    skip_on_quality_fail : bool
        Skip files that fail quality checks (downstream mode).
    copy_other_files : bool
        Copy non-EEG files to preserve folder structure (downstream mode).
    exclude_patterns : list of str, optional
        Patterns to exclude when copying files.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    conf_log = lambda: configure_logging(log_path, log_level)
    conf_log()
    
    src_paths = discover_files(dataset_path, file_extension)
    msg = f"Discovered {len(src_paths)} files to process."
    logging.info(msg)
    print(msg)
    
    if not src_paths:
        logging.info("No files to process. Exiting.")
        print("No files to process. Exiting.")
        return
    
    out_path_obj = Path(out_path)
    out_path_obj.mkdir(parents=True, exist_ok=True)
    
    dataset_root = (
        Path(dataset_path) if os.path.isdir(dataset_path)
        else Path(os.path.commonpath([str(p.parent) for p in src_paths]))
    )
    
    # Filter already processed
    if not overwrite:
        output_fmt = "auto" if preserve_structure else ("hdf5" if save_as_hdf5 else "bdf")
        processed = get_already_processed(str(out_path_obj), output_fmt, file_extension)
        original_count = len(src_paths)
        src_paths = [p for p in src_paths if p.stem not in processed]
        skipped = original_count - len(src_paths)
        if skipped:
            msg = f"Skipping {skipped} already processed files."
            logging.info(msg)
            print(msg)
    
    msg = f"Files to process: {len(src_paths)}"
    logging.info(msg)
    print(msg)
    
    if not src_paths:
        logging.info("All files already processed.")
        print("All files already processed.")
    else:
        if shuffle_files:
            src_paths = np.random.permutation(src_paths).tolist()
        
        metrics_path_obj = Path(metrics_path) if metrics_path else None
        if metrics_path_obj:
            if metrics_path_obj.suffix == '.csv':
                metrics_path_obj = metrics_path_obj.parent
            metrics_path_obj.mkdir(parents=True, exist_ok=True)
            csv_path = metrics_path_obj / "quality_metrics.csv"
            if overwrite and csv_path.exists():
                csv_path.unlink()
        
        if preserve_structure:
            # Downstream mode: preserve structure, single files
            print(f"Processing {len(src_paths)} files with {n_jobs} workers (downstream mode)...")
            
            dest_paths = [compute_dest_path(src, dataset_root, out_path_obj) for src in src_paths]
            
            jobs = [
                delayed(process_single_file)(
                    pipeline, src, dest, conf_log,
                    output_format, fallback_format, overwrite,
                    skip_on_quality_fail, metrics_path_obj
                )
                for src, dest in zip(src_paths, dest_paths)
            ]
            
            results = Parallel(n_jobs=n_jobs)(
                tqdm(jobs, total=len(jobs), desc="Preprocessing", smoothing=0.05)
            )
            
            status_counts: Dict[str, int] = {}
            for r in results:
                status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))
            logging.info(f"Processing complete. {summary}")
            print(f"\nProcessing complete. {summary}")
        
        else:
            # Pretrain mode: batch into HDF5/BDF
            if not save_as_hdf5:
                batch_size = 1
            
            batches = [src_paths[i:i + batch_size] for i in range(0, len(src_paths), batch_size)]
            logging.info(f"Created {len(batches)} batches (batch_size={batch_size}).")
            print(f"Processing {len(src_paths)} files in {len(batches)} batches with {n_jobs} workers...")
            
            if save_as_hdf5:
                dest_paths = create_dest_paths(out_path_obj, len(batches))
                jobs = [
                    delayed(preprocess_batch)(
                        pipeline, batch, dest, conf_log, True, sleep_after_batch, metrics_path_obj
                    )
                    for batch, dest in zip(batches, dest_paths)
                ]
            else:
                jobs = [
                    delayed(preprocess_batch)(
                        pipeline, batch, out_path_obj / f"{batch[0].stem}.bdf",
                        conf_log, False, sleep_after_batch, metrics_path_obj
                    )
                    for batch in batches
                ]
            
            results = Parallel(n_jobs=n_jobs)(
                tqdm(jobs, total=len(jobs), desc="Preprocessing", smoothing=0.05)
            )
            
            print(f"Completed {len(results)} / {len(jobs)} batches.")
            logging.info(f"Completed {len(results)} / {len(jobs)} batches.")
    
    # Copy non-EEG files (downstream mode only)
    if copy_other_files and preserve_structure and os.path.isdir(dataset_path):
        print("\nCopying non-EEG files...")
        logging.info("Copying non-EEG files to preserve folder structure...")
        
        eeg_extensions = [file_extension] if isinstance(file_extension, str) else list(file_extension)
        patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS.copy()
        
        copy_stats = copy_non_eeg_files(dataset_root, out_path_obj, eeg_extensions, patterns, overwrite)
        
        logging.info(f"Copied {len(copy_stats['copied'])} files.")
        print(f"Copied {len(copy_stats['copied'])} files.")
        print(f"Skipped {len(copy_stats['skipped'])} existing files.")
        print(f"Excluded {len(copy_stats['excluded'])} files by pattern.")
    
    print("\nDone!")
    logging.info("Preprocessing complete.")


if __name__ == "__main__":
    mne.set_log_level("CRITICAL")
    CLI(preprocess_dataset)
