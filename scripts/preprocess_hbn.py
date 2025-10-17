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
import pyedflib
from pyedflib import FILETYPE_BDFPLUS
from datetime import datetime
# from mne.export import export_eeglab

def write_bdf_from_raw(raw: mne.io.BaseRaw, out_path: str, write_annotations: bool = True):
    # pick channels
    picks = mne.pick_types(
        raw.info,
        eeg=True, eog=True, ecg=True, emg=True, misc=True, stim=True,
        seeg=True, dbs=True, ecog=True, resp=True, bio=True, exci=True, ias=True, syst=True
    )
    if len(picks) == 0:
        raise RuntimeError("No writable channels were found.")

    # data in volts, then to microvolts
    data_v = raw.get_data(picks).astype(np.float64)    # (n_ch, n_samp)
    data_uv = data_v * 1e6
    ch_names = [raw.ch_names[i] for i in picks]
    sfreq = float(raw.info["sfreq"])

    # helper to round limits so their string fits 8 characters
    def round_to_edf8(x: float) -> float:
        s = "-" if x < 0 else ""
        int_digits = len(str(int(abs(x))))
        max_decimals = max(0, 7 - len(s) - int_digits)
        if max_decimals == 0:
            return float(f"{int(x)}")
        return float(f"{x:.{max_decimals}f}")

    # compute limits with margin
    MARGIN_PCT = 0.01
    pmins = data_uv.min(axis=1)
    pmaxs = data_uv.max(axis=1)
    spans = np.maximum(pmaxs - pmins, 1.0)  # at least 1 uV span before rounding
    pmins = pmins - MARGIN_PCT * spans
    pmaxs = pmaxs + MARGIN_PCT * spans

    # round for shorter strings
    pmins = np.array([round_to_edf8(v) for v in pmins], dtype=float)
    pmaxs = np.array([round_to_edf8(v) for v in pmaxs], dtype=float)

    # SAFETY: ensure min < max after rounding
    # if equal or invalid, enforce a symmetric default window
    DEFAULT_WINDOW_UV = 1.0
    for i in range(len(pmins)):
        if not np.isfinite(pmins[i]) or not np.isfinite(pmaxs[i]) or pmaxs[i] <= pmins[i]:
            pmins[i] = -DEFAULT_WINDOW_UV
            pmaxs[i] = +DEFAULT_WINDOW_UV

    # build headers
    sig_headers = []
    for name, pmin, pmax in zip(ch_names, pmins, pmaxs):
        sig_headers.append({
            "label": name[:16],
            "dimension": "uV",
            "sample_frequency": sfreq,
            "physical_min": float(pmin),
            "physical_max": float(pmax),
            "digital_min": -8388608,   # 24 bit BDF
            "digital_max":  8388607
        })

    # start time
    md = raw.info.get("meas_date")
    if md is not None and isinstance(md, (tuple, list)):
        md = md[0]
    startdate = md.replace(tzinfo=None) if md is not None else datetime.now()

    # write BDF+
    f = pyedflib.EdfWriter(str(out_path), n_channels=len(sig_headers), file_type=FILETYPE_BDFPLUS)
    f.setSignalHeaders(sig_headers)
    f.setStartdatetime(startdate)

    # write all channels in one call
    f.writeSamples([data_uv[i] for i in range(len(sig_headers))])

    # annotations
    if write_annotations and len(raw.annotations) > 0:
        first_time = float(raw.first_time)
        for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            f.writeAnnotation(float(onset + first_time), float(duration), str(desc))

    f.close()


def configure_logging(filename):
    # Configure logging to file with a specific format
    logging.basicConfig(filename=filename,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(processName)s] -\t%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',  # Added to format timestamp without milliseconds
                        force=True)


def preprocess(pipeline: Pipeline, src_paths: list[Path], dest_path: str, conf_log: str, save_as_hdf5: bool = True):
    mne.set_log_level("ERROR")
    conf_log()    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=ModuleNotFoundError)
    warnings.filterwarnings("ignore", message=r".*pymatreader cannot import Matlab string variables.*", category=UserWarning)
    
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
        # raws[0].save(dest_path, overwrite=True)
        # export_eeglab(dest_path, raws[0])
        ########################################################
        # data = raws[0].get_data()  
        # for i, ch_name in enumerate(raws[0].ch_names):
        #     n_nans = np.isnan(data[i]).sum()
        #     if n_nans > 0:
        #         print(f"Channel {ch_name} has {n_nans} NaNs")
        #raws[0].export(dest_path, fmt="edf")
        dest_path = str(dest_path)
        write_bdf_from_raw(raws[0], dest_path.replace(".edf", ".bdf"))

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
            data_files = glob.glob(f"{out_path}/*.edf")
            processed_files = [Path(f).stem for f in data_files]
        # Remove already processed files
    src_paths = [src for src in src_paths if src.stem not in processed_files]
    logging.info(f"Total files to process: {len(src_paths)}")

    # Shuffle the files
    if shuffle_files:    
        src_paths = np.random.permutation(src_paths).tolist()

    # Split the EDF files into batches
    if not save_as_hdf5:
        batch_size = 1  # force one file per batch for .set/.edf/ .edf
    src_paths_batches = [src_paths[i:i + batch_size] for i in range(0, len(src_paths), batch_size)]
    print(f"Num batches {len(src_paths_batches)}")
    
    # # Create destination files
    if save_as_hdf5:
        des_paths = []
        idx = 1
        while len(des_paths) < len(src_paths_batches):
            candidate = Path(out_path) / f"data_{idx}.hdf5"
            if not candidate.exists():
                des_paths.append(candidate)
            idx += 1

    # processed_count = 0
    # if save_as_hdf5:
    #     for src_path_batch, dest_file in tqdm(zip(src_paths_batches, des_paths), total=len(des_paths), desc="Batches"):
    #         preprocess(pipeline, src_path_batch, dest_file, conf_log, save_as_hdf5=True)
    #         processed_count += len(src_path_batch)
    #         tqdm.write(f"Processed {processed_count}/{len(src_paths)} files")

    # else:
    #     for src_path_batch in src_paths_batches:
    #         dest_file = Path(out_path) / f"{src_path_batch[0].stem}.edf"
    #         preprocess(pipeline, src_path_batch, dest_file, conf_log, save_as_hdf5=False)
    #         processed_count += len(src_path_batch)
    #         logging.info(f"Processed {processed_count}/{len(src_paths)} files")


    # --- Prepare jobs ---
    if save_as_hdf5:
        jobs = [
            delayed(preprocess)(pipeline, src_batch, dest_file, conf_log, save_as_hdf5=True)
            for src_batch, dest_file in zip(src_paths_batches, des_paths)
        ]
    else:
        jobs = [
            delayed(preprocess)(
                pipeline,
                src_batch,
                Path(out_path) / f"{src_batch[0].stem}.bdf",
                conf_log,
                save_as_hdf5=False
            )
            for src_batch in src_paths_batches
        ]

    # --- Run jobs in parallel ---
    results = Parallel(n_jobs=n_jobs)(
        tqdm(jobs, total=len(jobs), desc="Batches", smoothing=0.05)
    )

    print(f"Completed {len(results)} / {len(jobs)} batches.")


if __name__ == "__main__":
    mne.set_log_level("CRITICAL")
    CLI(preprocess_dataset)
    