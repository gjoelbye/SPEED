import mne
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import pandas as pd
import logging
# import h5py
from collections import OrderedDict
from datetime import datetime
from time import sleep
from speed.utils import split_raw, make_tuh_montage
from speed.methods import PreprocessMethods
from filelock import FileLock
import warnings

# Import typing
from typing import Tuple, List, Optional, Dict
import traceback


# Make ABC abstract class
from abc import ABC, abstractmethod

class Pipeline(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
    
class BasePipeline(Pipeline): 
    """
    A class for preprocessing EEG data files.

    Parameters:
    - src_paths (List[str]): Paths to the source EEG files.
    - window_length (int): Length of the window for processing, in seconds. Default is 60.
    - hp_freq (float): High pass filter frequency. Default is 1.0.
    - lp_freq (float): Low pass filter frequency. Default is 100.0.
    - line_freqs (List[float]): Frequencies to be removed by the notch filter. Default is [60.0].
    - iclabel_threshold (float): Threshold for IC label classification. Default is 0.7.
    - min_nchans (int): Minimum number of channels. Default is 10.
    - do_ica (bool): Whether to perform ICA. Default is True.
    - included_components (List[str]): Components to include. Default is ["brain", "other"].
    - memory_efficient (bool): If True, uses a memory-efficient approach. Default is True.
    """
    def __init__(
            self, 
            window_length: int = 60,
            shift_seconds: Optional[float] = None,
            sfreq: float = 256.0,
            hp_freq: Optional[float] = 0.5, 
            lp_freq: Optional[float] = 100.0,
            line_freqs: List[float] = [60.0], 
            iclabel_threshold: float = 0.7,
            # quality_check: bool = True,
            min_nchans: int = 10, 
            do_ica: bool = True, 
            included_components: List[str] = ["brain", "other"], 
            memory_efficient: bool = True,
            montage_name: str = None, #str = "tuh",
            channels: List[str] = None, #aka chs later in script #['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
            channels_to_remove: List[str] = None,
            channels_rename: Optional[Dict[str, str]] = None,
            metrics_path: Path = None,
            return_quality_metrics: bool = False,
            drop_bad_quality: bool = True,
            fit_to_hbn_montage: bool = False,
        ):
        
        mne.set_log_level('ERROR')
        
        self.channels_rename = channels_rename
        self.channels_to_remove = channels_to_remove
        
        self.sfreq = sfreq
        self.interpolation_mode = "accurate"
        self.ransac = True
        # self.quality_check = quality_check
        self.return_quality_metrics = return_quality_metrics
        self.drop_bad_quality = drop_bad_quality
        self.fit_to_hbn_montage = fit_to_hbn_montage
        
        # Quality check thresholds
        self.oha_threshold = 40e-6
        self.thv_threshold = 40e-6
        self.chv_threshold = 80e-6
        self.min_unique_ratio=0.001
                            
        self.memory_efficient = memory_efficient        
        self._setup_montage_and_channels(montage_name, channels)
        self._setup_filters(hp_freq, lp_freq, line_freqs)
        self._setup_ica(iclabel_threshold, included_components, do_ica)

        self.min_nchans = min_nchans
        self.window_length = window_length # Seconds 
        self.shift_seconds = shift_seconds

        self.metrics_path = metrics_path
        metrics_path.mkdir(exist_ok=True, parents=True) if metrics_path is not None else None
        
    def _setup_montage_and_channels(self, montage_name, chs):
        """Setup montage and channels."""
        # gib error if channels is none or empty: 
        if not chs or len(chs) == 0:
            raise ValueError("Channels must be provided and be in required order.")
        self.chs = chs # all passed channels in order
        self.montage_name = montage_name
        
        if montage_name is None: 
            self.montage = None
            return
        elif self.montage_name == "tuh":
            montage = make_tuh_montage()
            self.montage = montage            
        else:
            montage = mne.channels.make_standard_montage(self.montage_name)       
            self.montage = montage
        
    def _setup_filters(self, hp_freq, lp_freq, line_freqs):
        """Configure filters and window length."""
        self.hp_freq = hp_freq
        self.lp_freq = lp_freq
        self.line_freqs = np.sort([line_freqs]).flatten()
        
    def _setup_ica(self, iclabel_threshold, included_components, do_ica):
        """
        Setup Independent Component Analysis (ICA) to identify and exclude artifacts.
        """
        self._possible_iclabels = ["brain", "muscle", "eye", "heart", "line noise", "channel noise", "other"]
        self.included_components = included_components
        assert all([label in self._possible_iclabels for label in self.included_components])
        
        self.iclabel_threshold = iclabel_threshold
        self.do_ica = do_ica
        
    # Add typing
    def _split_raw(self, raw: mne.io.Raw):
        return split_raw(raw, self.window_length, self.shift_seconds)
    
    def _set_montage(self, raw: mne.io.Raw):
        return PreprocessMethods.set_montage(raw, self.montage)
    
    def _to_standard_names(self, raw: mne.io.Raw):
        return PreprocessMethods.to_standard_names(raw)
    
    def _average_reference(self, raw: mne.io.Raw):
        return mne.set_eeg_reference(raw, ref_channels='average', projection=False, copy=False, verbose=False)
        
    def _evaluate_quality(self, raw: mne.io.Raw):
        # if not self.quality_check:
        #     return True
        # else:
        return PreprocessMethods.evaluate_quality(raw, self.oha_threshold, self.thv_threshold, self.chv_threshold, self.min_unique_ratio,
                              self.min_nchans, self.line_freqs, self.hp_freq, self.lp_freq)
                    
    def _interpolate_nearest(self, raw: mne.io.Raw):     
        return PreprocessMethods.interpolate_nearest(raw, self.sfreq)
        
    def _drop_bad_channels(self, raw: mne.io.Raw):       
        return PreprocessMethods.find_bad_channels(raw, ransac = self.ransac, drop = True)

    def _filter(self, raw: mne.io.Raw):
        return PreprocessMethods.filter(raw, self.hp_freq, self.lp_freq, [])
    
    def _remove_line_noise(self, raw: mne.io.Raw):
        return PreprocessMethods.filter(raw, None, None, self.line_freqs, do_detrend=False)        

    def _ica_clean(self, raw: mne.io.Raw):
        return PreprocessMethods.ica_clean(raw, self.iclabel_threshold, self.included_components)       

    def _interpolate_missing(self, raw: mne.io.Raw):
        return PreprocessMethods.interpolate_missing(raw, self.chs, self.channels_to_remove, self.montage, mode=self.interpolation_mode)
    
    def _interpolate_to_hbn(self, raw: mne.io.Raw):
        return PreprocessMethods.interpolate_to_hbn(raw)
    
    # def _drop_extra_and_reorder(self, raw: mne.io.Raw):
    #     return PreprocessMethods.drop_extra_and_reorder(raw, self.chs)

    def _drop_extra_channels(self, raw: mne.io.Raw):
        return PreprocessMethods.drop_extra_channels(raw, self.chs)
    
    def _reorder_channels(self, raw: mne.io.Raw):
        return PreprocessMethods.reorder_chans(raw, self.chs)
    
    def _drop_channels_manually(self, raw: mne.io.Raw):
        return PreprocessMethods.drop_channels_manually(raw, self.channels_to_remove)


class PretrainPipeline(BasePipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, src_paths: List[str]) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]]:
        return self.run(src_paths)    

    def run(self, src_paths):
        logging.debug("Loading EDF files...")
        src_paths = [Path(src_path) for src_path in src_paths]

        raws = []
        times = []
        indices = []
        
        logging.debug("Splitting raws...")   
        for i, src_path in enumerate(src_paths):
            try:
                if src_path.suffix == ".edf":
                    raw_orig = mne.io.read_raw_edf(src_path, preload=True, verbose=False)
                elif src_path.suffix == ".set":
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*'boundary' events.*",
                            category=RuntimeWarning,
                        )
                        raw_orig = mne.io.read_raw_eeglab(src_path, preload=True, verbose=False)
                else:
                    raise ValueError(f"Unsupported file type: {src_path.suffix}")

                # Drop manually specified channels
                if self.channels_to_remove is not None:
                    drop_chs = self._drop_channels_manually(raw_orig)
                    logging.info(f"File: {src_paths[i].stem}.\tManually dropped {len(drop_chs)} channels: {drop_chs}.")

                # Rename channels with channel_rename
                if self.channels_rename is not None:
                    raw_orig.rename_channels(self.channels_rename)
                    logging.info(f"File: {src_paths[i].stem}.\tRenamed channels: {self.channels_rename}.")

                if self.montage_name == "tuh":
                    self._to_standard_names(raw_orig)
                    
                if self.montage_name is not None:
                    drop_chs = self._set_montage(raw_orig)
                    logging.info(f"File: {src_paths[i].stem}.\tDropped {len(drop_chs)} channels when setting montage: {drop_chs}.")
                
                # Skip if raw_orig is too short
                if self.window_length is not None:
                    if raw_orig.times[-1] < 1.5*self.window_length:
                        logging.info(f"File: {src_path.stem}.\tDuration too short. Skipping file.")
                        continue
            except Exception as e:
                logging.error(f"Dropping file: {src_path.stem}.\tError: {e}")
                continue
                
            if self.window_length is None:
                raws.append(raw_orig)
                times.append((raw_orig.times[0], raw_orig.times[-1]))
                indices.append(i)
                raw_orig._original_info = raw_orig.info.copy()
                continue

            raws_split, times_split = self._split_raw(raw_orig)
            if self.montage_name is None:
                for r in raws_split:
                    r._original_info = raw_orig.info.copy()     
                          
            raws.extend(raws_split)
            times.extend(times_split)
            indices.extend([i] * len(raws_split)) 
            
        total_windows = len(raws)
        logging.debug(f"Total windows: {total_windows}")
                       
        for i, raw in enumerate(raws):
            start_time = round(times[i][0], 1)
            end_time = round(times[i][1], 1)
            filename = src_paths[indices[i]] #.stem  
            try:
                raws[i] = self.run_single(raw, start_time, end_time, filename)
            except Exception as e:
                logging.error(f"File: {filename}.\tTime: {(start_time, end_time)}.\tError: {e}\nTraceback:\n{traceback.format_exc()}") # {e}")
                raws[i] = None
                
            # Log progress every N windows
            if (i + 1) % 10 == 0:
                logging.debug(f"Processed {i + 1}/{total_windows} windows.")
        
        mask = [True if raw is not None else False for raw in raws]
        raws = [raw for raw, m in zip(raws, mask) if m]
        times = [time for time, m in zip(times, mask) if m]
        indices = [idx for idx, m in zip(indices, mask) if m]
        
        return raws, times, indices
        
    def run_single(self, raw, start_time, end_time, filename) -> Optional[mne.io.Raw]:
        window_info_str = f"File: {filename}.\tTime: {(start_time, end_time)}."

        # --- First quality check
        ok, metrics1 = self._run_quality_check(raw, window_info_str, stage=1)
        metrics1 = metrics1 or (None, None, None, None)
        if not ok:
            if self.return_quality_metrics:
                self._save_quality_metrics(
                    filename, start_time, end_time,
                    *(metrics1), None, None, None, None)
            return None

        # --- Preprocessing
        self._remove_line_noise(raw)
        bad_chs = self._drop_bad_channels(raw)
        logging.info(f"{window_info_str}\tFound {len(bad_chs)} bad channels: {bad_chs}.")

        # # --- Second quality check
        # ok, metrics2 = self._run_quality_check(raw, window_info_str, stage=2)
        # metrics2 = metrics2 or (None, None, None, None)
        # if not ok:
        #     if self.return_quality_metrics:
        #         self._save_quality_metrics(
        #             filename, start_time, end_time,
        #             *(metrics1), *(metrics2)  # stage 2 metrics are empty
        # )
        #     return None
        metrics2 = (None, None, None, None)


        # --- Save metrics only if requested
        if self.return_quality_metrics:
            self._save_quality_metrics(
                filename, start_time, end_time,
                *(metrics1), *(metrics2)
            )

        # --- Continue preprocessing
        self._filter(raw)
        self._average_reference(raw)
                
        if self.do_ica:
            excluded_idxs, labels, y_proba = self._ica_clean(raw)
            logging.info(f"{window_info_str}\tExcluding {len(excluded_idxs)} components: {excluded_idxs}.")
            logging.info(f"{window_info_str}\tLabels: {labels}.")
            logging.info(f"{window_info_str}\tProbabilities: {[round(prob, 2) for prob in y_proba]}.")
            
            bad_chs = self._drop_bad_channels(raw)
            logging.info(f"{window_info_str}\tFound {len(bad_chs)} bad channels: {bad_chs}.")

        logging.info(f"{window_info_str}\tFitting to HBN montage: {self.fit_to_hbn_montage}.")
        if self.fit_to_hbn_montage:
            raw = self._interpolate_to_hbn(raw)
        else:
            missing_chs = self._interpolate_missing(raw)
            logging.info(f"{window_info_str}\tIntepolating {len(missing_chs)} channels: {missing_chs}.")
            extra_chs = self._drop_extra_channels(raw)
            if len(extra_chs) > 0:
                logging.info(f"{window_info_str}\tRemoving {len(extra_chs)} extra channels: {extra_chs}.")
            self._reorder_channels(raw)
        
        self._interpolate_nearest(raw) # why not normal resampling?
        
        return raw

    def _run_quality_check(
        self, raw, window_info_str: str, stage: int
    ):
        """
        Evaluate quality. Returns: (quality_ok: bool, metrics: tuple or None)
        """
        if not (self.return_quality_metrics or self.drop_bad_quality):
            return True, None  # skip completely if not needed

        quality, metrics = self._evaluate_quality(raw)

        if self.drop_bad_quality and not quality:
            logging.info(f"{window_info_str}.\tQuality check {stage} failed. Dropping window.")
            return False, metrics

        return True, metrics # True even if quality is bad if not dropping bc not used anyways, just need to continue


    def _save_quality_metrics(
        self, filename: str, start_time: str, end_time: str,
        oha1: float = None, thv1: float = None, chv1: float = None, bcr1: float = None,
        oha2: float = None, thv2: float = None, chv2: float = None, bcr2: float = None
    ) -> None:
        if self.metrics_path is None:
            return
        
        fname = self.metrics_path / "quality_metrics.csv"
        lock = FileLock(str(fname) + ".lock")
        with lock:
            pd.DataFrame([{
                "filename": filename,
                "window_start_time": start_time,
                "window_end_time": end_time,
                "oha1": oha1, "thv1": thv1, "chv1": chv1, "bcr1": bcr1,
            "oha2": oha2, "thv2": thv2, "chv2": chv2, "bcr2": bcr2,
            }]).to_csv(fname, mode="a", header=not fname.is_file(), index=False)
