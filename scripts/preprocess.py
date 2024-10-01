#from typing import Tuple
from jsonargparse import CLI
import h5py
import glob
import numpy as np
import os
import mne
from tqdm import tqdm
import logging
import os, glob
from joblib import Parallel, delayed
from src.pipeline import Pipeline
import warnings
from time import sleep
from pathlib import Path

def configure_logging(filename):
    # Configure logging to file with a specific format
    logging.basicConfig(filename=filename,
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - [%(processName)s] -\t%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',  # Added to format timestamp without milliseconds
                        force=True)


def preprocess(pipeline: Pipeline, src_paths: list[Path], dest_path: str, conf_log: str):
    # Set logging level for mne    
    mne.set_log_level("ERROR")
    conf_log()    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=ModuleNotFoundError)
    
    logging.debug("Starting preprocessing...")    
    
    raws, times, indices = pipeline(src_paths)
    
    logging.debug("Saving preprocessed data...")
    # Save with hp5 format
        
    with h5py.File(dest_path, "w") as file:            
        file.attrs["files"] = [path.stem for path in src_paths]
        file.attrs['file_idxs'] = indices
        file.attrs['time_slices'] = times
        file.create_dataset("data", data=np.array([raw._data for raw in raws]), dtype='float32')
            
    logging.debug(f"Saved to {dest_path}. File size: {Path(dest_path).stat().st_size / 1e6:.2f} MB.")            
    
    # Delete the raws to save memory and clean up memory
    del raws, times, indices
    
    # Sleep for 5 seconds to avoid memory issues
    sleep(5)

def preprocess_dataset(pipeline: Pipeline, dataset_path: str, out_path: str, log_path: str,
                        overwrite: bool = False, shuffle_files: bool = True, batch_size: int = 10, n_jobs: int = 6):
    
    
    if os.path.exists(log_path):
        if overwrite:
            raise ValueError("Log file already exists. Please provide a different path.")
        else:
            print("Log file already exists. Appending to the existing log file.")
    
    conf_log = lambda: configure_logging(log_path)
    conf_log()
        
    if os.path.isdir(dataset_path):
        src_paths = glob.glob(os.path.join(dataset_path, "**/*.edf"), recursive=True)
    elif os.path.isfile(dataset_path) and dataset_path.endswith(".txt"):
        with open(dataset_path, "r") as file:
            src_paths = [path.strip() for path in file.readlines()]
    else:
        raise ValueError("Invalid dataset path. Please provide a valid path to the dataset.")
    
    src_paths = [Path(src_path) for src_path in src_paths]
    
    # Create output directory if it does not exist.
    os.makedirs(out_path, exist_ok=True)
    
    # Get all the files that have been saved    
    if not overwrite:
        processed_edf_files = []
        data_files = glob.glob(f"{out_path}/*.hdf5")
        for file_path in data_files:
            with h5py.File(file_path, "r") as file:
                processed_edf_files.extend(file.attrs["files"].tolist())
                      
        # Remove already processed files
        src_paths = [src_path for src_path in src_paths if src_path.stem not in processed_edf_files]
    
    # Shuffle the files
    if shuffle_files:    
        src_paths = np.random.permutation(src_paths).tolist()

    # Split the EDF files into batches
    src_paths_batches = [src_paths[i:i + batch_size] for i in range(0, len(src_paths), batch_size)]
    
    # Create destination files
    des_paths = []
    
    idx = 1
    while len(des_paths) < len(src_paths_batches):
        file_name = f"data_{idx}.hdf5"
        if not os.path.exists(os.path.join(out_path, file_name)):
            des_paths.append(Path(os.path.join(out_path, file_name)))
        idx += 1
    
    assert len(des_paths) == len(src_paths_batches), "Number of destination files should be equal to the number of batches"
    
    logging.debug(f"Total files: {len(src_paths)}")
    logging.debug(f"Batch size: {batch_size}")
    logging.debug(f"Total batches: {len(src_paths_batches)}")
    logging.debug(f"Number of jobs: {n_jobs}")
    
    _ = Parallel(n_jobs=n_jobs)(delayed(preprocess)(pipeline, src_path_batch, des_path, conf_log) \
        for src_path_batch, des_path in tqdm(zip(src_paths_batches, des_paths),total=len(src_paths_batches), desc='Preprocessing files'))

if __name__ == "__main__":
    mne.set_log_level("CRITICAL")
    CLI(preprocess_dataset)