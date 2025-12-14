import os
import glob
import logging
import warnings
import fcntl
from pathlib import Path
from time import sleep
from typing import List, Union

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from jsonargparse import CLI

from speed.pipeline import Pipeline
from speed.utils import save_hdf5, write_bdf_from_raw


# =============================================================================
# Logging Configuration
# =============================================================================

def configure_logging(filename: str, level: str = "INFO"):
    """Configure logging to file."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        filename=filename,
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )


# =============================================================================
# Batch Processing
# =============================================================================

def preprocess_batch(
    pipeline: Pipeline,
    src_paths: List[Path],
    dest_path: Path,
    conf_log,
    save_as_hdf5: bool = True,
    sleep_after: float = 5.0,
    metrics_path: Path = None
) -> None:
    """
    Process a batch of files.
    
    Parameters
    ----------
    pipeline : Pipeline
        The preprocessing pipeline instance.
    src_paths : list of Path
        Source file paths.
    dest_path : Path
        Output file path.
    conf_log : callable
        Logging configuration function.
    save_as_hdf5 : bool
        If True, save as HDF5 batch. If False, save as individual BDF.
    sleep_after : float
        Seconds to sleep after processing.
    metrics_path : Path, optional
        Directory to save quality metrics CSV.
    """
    mne.set_log_level("ERROR")
    conf_log()
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=ModuleNotFoundError)
    warnings.filterwarnings("ignore", message=r".*pymatreader cannot import.*", category=UserWarning)
    
    logging.debug(f"Starting preprocessing of {len(src_paths)} files...")
    
    raws, times, indices = pipeline(src_paths)
    
    # Save quality metrics if available
    if metrics_path is not None:
        quality_metrics = pipeline.get_quality_metrics()
        if quality_metrics:
            save_quality_metrics(quality_metrics, metrics_path)
    
    if len(raws) == 0:
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


def save_quality_metrics(metrics: List[dict], metrics_path: Path) -> None:
    """
    Append quality metrics to CSV file with thread-safe file locking.
    
    This function is thread-safe for multiprocessing. It uses file locking
    to ensure that multiple processes can safely append to the same CSV file.
    
    Parameters
    ----------
    metrics : list of dict
        Quality metric entries.
    metrics_path : Path
        Directory to save the CSV file.
    """
    if not metrics:
        return
    
    metrics_path = Path(metrics_path)
    
    # If metrics_path points to a file (e.g., quality_metrics.csv), get its parent directory
    if metrics_path.exists() and metrics_path.is_file():
        metrics_path = metrics_path.parent
    elif metrics_path.suffix == '.csv':
        # Path ends with .csv but file doesn't exist yet, treat as file path
        metrics_path = metrics_path.parent
    
    # Ensure the directory exists
    metrics_path.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_path / "quality_metrics.csv"
    
    df = pd.DataFrame(metrics)
    
    # Use file locking for thread-safe writing in multiprocessing
    lock_path = csv_path.with_suffix('.csv.lock')
    
    # Acquire lock file
    lock_file = None
    try:
        # Create lock file if it doesn't exist
        lock_file = open(lock_path, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        
        # Check if file exists after acquiring lock (to avoid race condition)
        file_exists = csv_path.exists()
        
        # Append mode: write header only if file doesn't exist
        df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
        
    except Exception as e:
        logging.warning(f"Error writing quality metrics: {e}")
    finally:
        # Release lock
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                # Clean up lock file
                if lock_path.exists():
                    lock_path.unlink()
            except Exception:
                pass


# =============================================================================
# File Discovery
# =============================================================================

def discover_files(
    dataset_path: str,
    file_extensions: Union[str, List[str]] = ".edf",
    list_file: str = None
) -> List[Path]:
    """
    Find all source files to process.
    
    Parameters
    ----------
    dataset_path : str
        Path to dataset directory or .txt file with paths.
    file_extensions : str or list of str
        File extensions to search for.
    list_file : str, optional
        Path to text file with explicit file paths.
    
    Returns
    -------
    list of Path
        All discovered source file paths.
    """
    if list_file is not None:
        with open(list_file, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        return [Path(p) for p in paths]
    
    if os.path.isfile(dataset_path) and dataset_path.endswith(".txt"):
        with open(dataset_path, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        return [Path(p) for p in paths]
    
    if os.path.isdir(dataset_path):
        if isinstance(file_extensions, str):
            extensions = [file_extensions]
        else:
            extensions = file_extensions
        
        src_paths = []
        for ext in extensions:
            pattern = f"*{ext}" if ext.startswith(".") else ext
            src_paths.extend(glob.glob(os.path.join(dataset_path, "**", pattern), recursive=True))
        
        return [Path(p) for p in src_paths]
    
    raise ValueError(f"Invalid dataset path: {dataset_path}. Must be a directory or .txt file.")


def get_already_processed(out_path: str, save_as_hdf5: bool = True) -> List[str]:
    """Get list of already processed file stems."""
    import h5py
    
    processed = []
    
    if save_as_hdf5:
        for hdf5_file in glob.glob(f"{out_path}/*.hdf5"):
            with h5py.File(hdf5_file, "r") as f:
                processed.extend(f.attrs["files"].tolist())
    else:
        for bdf_file in glob.glob(f"{out_path}/*.bdf"):
            processed.append(Path(bdf_file).stem)
    
    return processed


def create_dest_paths(out_path: Path, n_batches: int) -> List[Path]:
    """Create non-conflicting HDF5 destination paths."""
    dest_paths = []
    idx = 1
    while len(dest_paths) < n_batches:
        candidate = out_path / f"data_{idx}.hdf5"
        if not candidate.exists():
            dest_paths.append(candidate)
        idx += 1
    return dest_paths


# =============================================================================
# Main Entry Point
# =============================================================================

def preprocess_dataset(
    pipeline: Pipeline,
    dataset_path: str,
    out_path: str,
    log_path: str,
    overwrite: bool = False,
    shuffle_files: bool = True,
    batch_size: int = 10,
    n_jobs: int = 6,
    file_extension: Union[str, List[str]] = ".edf",
    list_file: str = None,
    save_as_hdf5: bool = True,
    log_level: str = "INFO",
    sleep_after_batch: float = 5.0,
    metrics_path: str = None
):
    """
    Preprocess an entire EEG dataset.
    
    Parameters
    ----------
    pipeline : Pipeline
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
    batch_size : int
        Number of files per HDF5 batch.
    n_jobs : int
        Number of parallel workers.
    file_extension : str or list of str
        File extensions to process.
    list_file : str, optional
        Path to text file with explicit file paths.
    save_as_hdf5 : bool
        If True, save batched HDF5 files. If False, save individual BDF files.
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR.
    sleep_after_batch : float
        Seconds to sleep after each batch.
    metrics_path : str, optional
        Directory to save quality metrics CSV.
    """
    if os.path.exists(log_path) and overwrite:
        raise ValueError("Log file already exists. Set overwrite=False or use a different path.")
    
    conf_log = lambda: configure_logging(log_path, log_level)
    conf_log()
    
    src_paths = discover_files(dataset_path, file_extension, list_file)
    logging.info(f"Discovered {len(src_paths)} files to process.")
    
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not overwrite:
        processed = get_already_processed(str(out_path), save_as_hdf5)
        original_count = len(src_paths)
        src_paths = [p for p in src_paths if p.stem not in processed]
        logging.info(f"Skipping {original_count - len(src_paths)} already processed files.")
    
    logging.info(f"Files to process: {len(src_paths)}")
    
    if len(src_paths) == 0:
        logging.info("No files to process. Exiting.")
        return
    
    if shuffle_files:
        src_paths = np.random.permutation(src_paths).tolist()
    
    if not save_as_hdf5:
        batch_size = 1
    
    batches = [src_paths[i:i + batch_size] for i in range(0, len(src_paths), batch_size)]
    logging.info(f"Created {len(batches)} batches (batch_size={batch_size}).")
    print(f"Processing {len(src_paths)} files in {len(batches)} batches with {n_jobs} workers...")
    
    metrics_path_obj = Path(metrics_path) if metrics_path else None
    
    # Handle metrics file overwrite logic
    if metrics_path_obj is not None:
        # If metrics_path points to a file (e.g., quality_metrics.csv), get its parent directory
        if metrics_path_obj.exists() and metrics_path_obj.is_file():
            metrics_path_obj = metrics_path_obj.parent
        elif metrics_path_obj.suffix == '.csv':
            # Path ends with .csv but file doesn't exist yet, treat as file path
            metrics_path_obj = metrics_path_obj.parent
        
        # Ensure the directory exists
        metrics_path_obj.mkdir(parents=True, exist_ok=True)
        csv_path = metrics_path_obj / "quality_metrics.csv"
        if overwrite and csv_path.exists():
            csv_path.unlink()
            logging.info("Removed existing quality_metrics.csv (overwrite=True).")
    
    if save_as_hdf5:
        dest_paths = create_dest_paths(out_path, len(batches))
        jobs = [
            delayed(preprocess_batch)(
                pipeline, batch, dest, conf_log, True, sleep_after_batch, metrics_path_obj
            )
            for batch, dest in zip(batches, dest_paths)
        ]
    else:
        jobs = [
            delayed(preprocess_batch)(
                pipeline, batch, out_path / f"{batch[0].stem}.bdf",
                conf_log, False, sleep_after_batch, metrics_path_obj
            )
            for batch in batches
        ]
    
    results = Parallel(n_jobs=n_jobs)(
        tqdm(jobs, total=len(jobs), desc="Preprocessing", smoothing=0.05)
    )
    
    print(f"Completed {len(results)} / {len(jobs)} batches.")
    logging.info(f"Completed {len(results)} / {len(jobs)} batches.")


if __name__ == "__main__":
    mne.set_log_level("CRITICAL")
    CLI(preprocess_dataset)
