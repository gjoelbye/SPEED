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
from speed.pipeline_challenge import Pipeline
import warnings
from time import sleep
from pathlib import Path
# from mne.export import export_eeglab


def configure_logging(filename):
    # Configure logging to file with a specific format
    logging.basicConfig(filename=filename,
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - [%(processName)s] -\t%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',  # Added to format timestamp without milliseconds
                        force=True)


def preprocess(pipeline: Pipeline, src_paths: list[Path], dest_path: str, conf_log: str, save_as_hdf5: bool = True):
    mne.set_log_level("ERROR")
    conf_log()    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=ModuleNotFoundError)
    
    logging.debug("Starting preprocessing...")    
    raws, times, indices = pipeline(src_paths)
    
    if len(raws) == 0:
        logging.debug("No valid data to save. Skipping this batch.")
        return
    logging.debug("Saving preprocessed data...")
    if save_as_hdf5:
        with h5py.File(dest_path, "w") as file:
            file.attrs["files"] = [path.stem for path in src_paths]
            file.attrs['file_idxs'] = indices
            file.attrs['time_slices'] = times
            file.create_dataset("data", data=np.array([raw._data for raw in raws]), dtype='float32')
    else:
        if len(raws) != 1:
            raise ValueError("(not .hdf5 output) requires exactly one file per call.")  
        raws[0].save(dest_path, overwrite=True)
        # export_eeglab(dest_path, raws[0])
        # raws[0].export(dest_path, fmt="edf")

    logging.debug(f"Saved to {dest_path}. File size: {Path(dest_path).stat().st_size / 1e6:.2f} MB.")            
    
    # Delete the raws to save memory and clean up memory
    del raws, times, indices
    # Sleep for 5 seconds to avoid memory issues
    sleep(5)

def preprocess_dataset(pipeline: Pipeline, dataset_path: str, out_path: str, log_path: str,
                        overwrite: bool = False, shuffle_files: bool = True, batch_size: int = 10, n_jobs: int = 6, file_extension: str = ".edf", save_as_hdf5 : bool = True):
    
    if os.path.exists(log_path):
        if overwrite:
            raise ValueError("Log file already exists. Please provide a different path.")
        else:
            print("Log file already exists. Appending to the existing log file.")
    
    conf_log = lambda: configure_logging(log_path)
    conf_log()
       
    if os.path.isdir(dataset_path):
        # src_paths = glob.glob(os.path.join(dataset_path, "**/*.edf"), recursive=True)
        pattern = f"*{file_extension}"
        src_paths = glob.glob(os.path.join(dataset_path, "**", pattern), recursive=True)
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
        processed_files = []
        if save_as_hdf5:
            data_files = glob.glob(f"{out_path}/*.hdf5")
            for file_path in data_files:
                with h5py.File(file_path, "r") as file:
                    processed_files.extend(file.attrs["files"].tolist())
        else:
            data_files = glob.glob(f"{out_path}/*.fif")
            processed_files = [Path(f).stem for f in data_files]
        # Remove already processed files
    src_paths = [src for src in src_paths if src.stem not in processed_files]
    logging.info(f"Total files to process: {len(src_paths)}")

    # Shuffle the files
    if shuffle_files:    
        src_paths = np.random.permutation(src_paths).tolist()

    # Split the EDF files into batches
    if not save_as_hdf5:
        batch_size = 1  # force one file per batch for .set/.edf/ .fif
    src_paths_batches = [src_paths[i:i + batch_size] for i in range(0, len(src_paths), batch_size)]
    
    # # Create destination files
    if save_as_hdf5:
        des_paths = []
        idx = 1
        while len(des_paths) < len(src_paths_batches):
            candidate = Path(out_path) / f"data_{idx}.hdf5"
            if not candidate.exists():
                des_paths.append(candidate)
            idx += 1

    processed_count = 0
    # if save_as_hdf5:
    #     for src_path_batch, dest_file in zip(src_paths_batches, des_paths):
    #         preprocess(pipeline, src_path_batch, dest_file, conf_log, save_as_hdf5=True)
    #         processed_count += len(src_path_batch)
    #         logging.info(f"Processed {processed_count}/{len(src_paths)} files")
    if save_as_hdf5:
        for src_path_batch, dest_file in tqdm(zip(src_paths_batches, des_paths), total=len(des_paths), desc="Batches"):
            preprocess(pipeline, src_path_batch, dest_file, conf_log, save_as_hdf5=True)
            processed_count += len(src_path_batch)
            tqdm.write(f"Processed {processed_count}/{len(src_paths)} files")

    else:
        for src_path_batch in src_paths_batches:
            dest_file = Path(out_path) / f"{src_path_batch[0].stem}.fif"
            preprocess(pipeline, src_path_batch, dest_file, conf_log, save_as_hdf5=False)
            processed_count += len(src_path_batch)
            logging.info(f"Processed {processed_count}/{len(src_paths)} files")


if __name__ == "__main__":
    mne.set_log_level("CRITICAL")
    CLI(preprocess_dataset)
    