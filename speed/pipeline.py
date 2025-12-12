"""
Unified EEG preprocessing pipeline.

A configurable, dataset-agnostic pipeline for preprocessing EEG data.
Supports various montage formats, quality checking, ICA, and flexible I/O.
"""
import mne
import numpy as np
import logging
import traceback
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Union

from speed.utils import split_raw, load_montage
from speed.methods import PreprocessMethods


class Pipeline(ABC):
    """Abstract base class for all preprocessing pipelines."""
    
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def __call__(self, src_paths: List[str]) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]]:
        pass


class BasePipeline(Pipeline):
    """
    Base pipeline class with all configurable preprocessing parameters.
    
    Parameters
    ----------
    window_length : int, optional
        Length of processing windows in seconds. None = process full file.
    shift_seconds : float, optional
        Window shift for overlapping windows. None = no overlap.
    sfreq : float
        Target sampling frequency after resampling.
    hp_freq : float, optional
        High-pass filter cutoff frequency in Hz.
    lp_freq : float, optional
        Low-pass filter cutoff frequency in Hz.
    line_freqs : list of float
        Power line frequencies to notch filter.
    iclabel_threshold : float
        Probability threshold for ICA component rejection (0-1).
    min_nchans : int
        Minimum number of channels required to process a file.
    do_ica : bool
        Whether to perform ICA artifact rejection.
    included_components : list of str
        ICA component labels to keep.
    montage : str, optional
        Montage source: "tuh", standard MNE name, path to .fif file, or None.
    channels : list of str
        Required output channels in desired order.
    channels_to_remove : list of str, optional
        Channels to drop before processing.
    channels_rename : dict, optional
        Mapping of old channel names to new names.
    
    Quality Control Parameters
    --------------------------
    oha_threshold : float
        Overall high amplitude threshold (Volts).
    thv_threshold : float
        Temporal high variance threshold (Volts).
    chv_threshold : float
        Channel high variance threshold (Volts).
    min_unique_ratio : float
        Minimum ratio of unique samples per channel (0-1).
    drop_bad_quality : bool
        Whether to drop windows that fail quality check.
    return_quality_metrics : bool
        Whether to compute and return quality metrics.
    
    Interpolation Parameters
    ------------------------
    target_montage : str, optional
        Path to target montage .fif file for spatial interpolation.
        If set, channels will be interpolated to this montage.
    interpolation_mode : str
        MNE interpolation mode: "accurate" or "fast".
    use_ransac : bool
        Whether to use RANSAC for bad channel detection.
    """
    
    def __init__(
        self,
        # Window settings
        window_length: Optional[int] = 60,
        shift_seconds: Optional[float] = None,
        sfreq: float = 256.0,
        # Filter settings
        hp_freq: Optional[float] = 0.5,
        lp_freq: Optional[float] = 100.0,
        line_freqs: List[float] = [60.0],
        # ICA settings
        iclabel_threshold: float = 0.7,
        do_ica: bool = True,
        included_components: List[str] = ["brain", "other"],
        # Channel settings
        min_nchans: int = 10,
        montage: Optional[str] = None,
        channels: Optional[List[str]] = None,
        channels_to_remove: Optional[List[str]] = None,
        channels_rename: Optional[Dict[str, str]] = None,
        # Quality check settings
        oha_threshold: float = 40e-6,
        thv_threshold: float = 40e-6,
        chv_threshold: float = 80e-6,
        min_unique_ratio: float = 0.001,
        drop_bad_quality: bool = True,
        return_quality_metrics: bool = False,
        # Quality decision thresholds
        oha_limit: float = 0.8,
        thv_limit: float = 0.5,
        chv_limit: float = 0.5,
        bcr_limit: float = 0.8,
        # Interpolation settings
        target_montage: Optional[str] = None,
        interpolation_mode: str = "accurate",
        use_ransac: bool = True,
        # Channel naming
        standardize_channel_names: bool = False,
    ):
        mne.set_log_level('ERROR')
        
        # Validate channels
        if channels is None or len(channels) == 0:
            raise ValueError("channels must be provided as a non-empty list.")
        
        # Channel configuration
        self.channels_rename = channels_rename
        self.channels_to_remove = channels_to_remove
        self.chs = channels
        self.standardize_channel_names = standardize_channel_names
        
        # Sampling and interpolation
        self.sfreq = sfreq
        self.interpolation_mode = interpolation_mode
        self.use_ransac = use_ransac
        
        # Quality check thresholds
        self.oha_threshold = oha_threshold
        self.thv_threshold = thv_threshold
        self.chv_threshold = chv_threshold
        self.min_unique_ratio = min_unique_ratio
        self.drop_bad_quality = drop_bad_quality
        self.return_quality_metrics = return_quality_metrics
        
        # Quality decision limits
        self.oha_limit = oha_limit
        self.thv_limit = thv_limit
        self.chv_limit = chv_limit
        self.bcr_limit = bcr_limit
        
        # Processing parameters
        self.min_nchans = min_nchans
        self.window_length = window_length
        self.shift_seconds = shift_seconds
        
        # Setup montage
        self.montage_source = montage
        self.montage = load_montage(montage)
        
        # Setup filters
        self.hp_freq = hp_freq
        self.lp_freq = lp_freq
        self.line_freqs = np.sort(np.atleast_1d(line_freqs)).flatten()
        
        # Setup ICA
        self._setup_ica(iclabel_threshold, included_components, do_ica)
        
        # Setup target montage for interpolation
        self.target_montage = None
        if target_montage is not None:
            self.target_montage = mne.channels.read_dig_fif(str(target_montage))
        
        # Quality metrics buffer
        self._quality_metrics_buffer = []
    
    def _setup_ica(self, threshold: float, included: List[str], do_ica: bool):
        """Configure ICA artifact rejection."""
        valid_labels = ["brain", "muscle", "eye", "heart", "line noise", "channel noise", "other"]
        if not all(label in valid_labels for label in included):
            raise ValueError(f"Invalid ICA labels. Valid options: {valid_labels}")
        
        self.iclabel_threshold = threshold
        self.included_components = included
        self.do_ica = do_ica
    
    # =========================================================================
    # Preprocessing Method Wrappers
    # =========================================================================
    
    def _split_raw(self, raw: mne.io.Raw):
        return split_raw(raw, self.window_length, self.shift_seconds)
    
    def _set_montage(self, raw: mne.io.Raw):
        return PreprocessMethods.set_montage(raw, self.montage)
    
    def _to_standard_names(self, raw: mne.io.Raw):
        return PreprocessMethods.to_standard_names(raw)
    
    def _average_reference(self, raw: mne.io.Raw):
        return mne.set_eeg_reference(
            raw, ref_channels='average', projection=False, copy=False, verbose=False
        )
    
    def _evaluate_quality(self, raw: mne.io.Raw):
        return PreprocessMethods.evaluate_quality(
            raw, self.oha_threshold, self.thv_threshold, self.chv_threshold,
            self.min_unique_ratio, self.min_nchans, self.line_freqs.tolist(),
            self.hp_freq, self.lp_freq,
            self.oha_limit, self.thv_limit, self.chv_limit, self.bcr_limit
        )
    
    def _interpolate_nearest(self, raw: mne.io.Raw):
        return PreprocessMethods.interpolate_nearest(raw, self.sfreq)
    
    def _drop_bad_channels(self, raw: mne.io.Raw):
        return PreprocessMethods.find_bad_channels(raw, ransac=self.use_ransac, drop=True)
    
    def _filter(self, raw: mne.io.Raw):
        return PreprocessMethods.filter(raw, self.hp_freq, self.lp_freq, [])
    
    def _remove_line_noise(self, raw: mne.io.Raw):
        return PreprocessMethods.filter(raw, None, None, self.line_freqs.tolist(), do_detrend=False)
    
    def _ica_clean(self, raw: mne.io.Raw):
        return PreprocessMethods.ica_clean(raw, self.iclabel_threshold, self.included_components)
    
    def _interpolate_missing(self, raw: mne.io.Raw):
        return PreprocessMethods.interpolate_missing(
            raw, self.chs, self.channels_to_remove, self.montage, mode=self.interpolation_mode
        )
    
    def _interpolate_to_target_montage(self, raw: mne.io.Raw):
        return PreprocessMethods.interpolate_to_target_montage(raw, self.target_montage)
    
    def _drop_extra_channels(self, raw: mne.io.Raw):
        return PreprocessMethods.drop_extra_channels(raw, self.chs)
    
    def _reorder_channels(self, raw: mne.io.Raw):
        return PreprocessMethods.reorder_channels(raw, self.chs)
    
    def _drop_channels_manually(self, raw: mne.io.Raw):
        return PreprocessMethods.drop_channels_manually(raw, self.channels_to_remove)
    
    def _zero_missing(self, raw: mne.io.Raw):
        return PreprocessMethods.zero_missing(raw, self.chs, self.montage)
    
    # =========================================================================
    # Quality Metrics
    # =========================================================================
    
    def get_quality_metrics(self) -> List[dict]:
        """Return collected quality metrics and clear buffer."""
        metrics = self._quality_metrics_buffer.copy()
        self._quality_metrics_buffer.clear()
        return metrics
    
    def _add_quality_metric(
        self,
        filename: Union[str, Path],
        start_time: float,
        end_time: float,
        metrics: Optional[Tuple]
    ) -> None:
        """Add a quality metric entry to the buffer."""
        oha, thv, chv, bcr = metrics if metrics else (None, None, None, None)
        self._quality_metrics_buffer.append({
            "filename": str(filename),
            "window_start_time": start_time,
            "window_end_time": end_time,
            "oha": oha,
            "thv": thv,
            "chv": chv,
            "bcr": bcr,
        })


class PretrainPipeline(BasePipeline):
    """
    Main preprocessing pipeline for EEG data.
    
    Supports configurable preprocessing with quality checking, ICA artifact
    rejection, montage handling, and flexible channel management.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, src_paths: List[str]) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]]:
        """Process a batch of files and return preprocessed windows."""
        return self.run(src_paths)
    
    def run(self, src_paths: List[str]) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]]:
        """
        Load and preprocess EEG files.
        
        Parameters
        ----------
        src_paths : list of str or Path
            Paths to source EEG files (.edf or .set).
        
        Returns
        -------
        raws : list of mne.io.Raw
            Preprocessed raw objects.
        times : list of tuple
            (start_time, end_time) for each window.
        indices : list of int
            Index of source file for each window.
        """
        src_paths = [Path(p) for p in src_paths]
        logging.debug(f"Loading {len(src_paths)} files...")
        
        raws, times, indices = [], [], []
        
        for i, src_path in enumerate(src_paths):
            try:
                raw_orig = self._load_raw_file(src_path)
                if raw_orig is None:
                    continue
                
                self._preprocess_channels(raw_orig, src_path)
                
                if not self._check_duration(raw_orig, src_path):
                    continue
                
                windows, window_times = self._extract_windows(raw_orig)
                
                raws.extend(windows)
                times.extend(window_times)
                indices.extend([i] * len(windows))
                
            except Exception as e:
                logging.error(f"Dropping file: {src_path.stem}. Error: {e}")
                continue
        
        total_windows = len(raws)
        logging.debug(f"Total windows: {total_windows}")
        
        for i in range(len(raws)):
            start_time = round(times[i][0], 1)
            end_time = round(times[i][1], 1)
            filename = src_paths[indices[i]]
            
            try:
                raws[i] = self.run_single(raws[i], start_time, end_time, filename)
            except Exception as e:
                logging.error(
                    f"File: {filename}. Time: {(start_time, end_time)}. "
                    f"Error: {e}\n{traceback.format_exc()}"
                )
                raws[i] = None
            
            if (i + 1) % 10 == 0:
                logging.debug(f"Processed {i + 1}/{total_windows} windows.")
        
        mask = [raw is not None for raw in raws]
        raws = [r for r, m in zip(raws, mask) if m]
        times = [t for t, m in zip(times, mask) if m]
        indices = [idx for idx, m in zip(indices, mask) if m]
        
        return raws, times, indices
    
    def _load_raw_file(self, src_path: Path) -> Optional[mne.io.Raw]:
        """Load a raw EEG file based on its extension."""
        suffix = src_path.suffix.lower()
        
        if suffix == ".edf":
            return mne.io.read_raw_edf(src_path, preload=True, verbose=False)
        elif suffix == ".set":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r".*'boundary' events.*", category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore", message=r".*pymatreader cannot import.*", category=UserWarning
                )
                return mne.io.read_raw_eeglab(src_path, preload=True, verbose=False)
        else:
            logging.warning(f"Unsupported file type: {suffix}. Skipping {src_path.stem}")
            return None
    
    def _preprocess_channels(self, raw: mne.io.Raw, src_path: Path):
        """Apply initial channel preprocessing: removal, renaming, montage."""
        if self.channels_to_remove:
            dropped = self._drop_channels_manually(raw)
            if dropped:
                logging.info(f"File: {src_path.stem}. Manually dropped {len(dropped)} channels.")
        
        if self.channels_rename:
            raw.rename_channels(self.channels_rename)
            logging.info(f"File: {src_path.stem}. Renamed channels.")
        
        if self.standardize_channel_names:
            self._to_standard_names(raw)
        
        if self.montage is not None:
            dropped = self._set_montage(raw)
            logging.info(f"File: {src_path.stem}. Dropped {len(dropped)} channels when setting montage.")
    
    def _check_duration(self, raw: mne.io.Raw, src_path: Path) -> bool:
        """Check if file meets minimum duration requirements."""
        if self.window_length is not None:
            min_duration = 1.5 * self.window_length
            if raw.times[-1] < min_duration:
                logging.info(
                    f"File: {src_path.stem}. Duration too short "
                    f"({raw.times[-1]:.1f}s < {min_duration:.1f}s). Skipping."
                )
                return False
        return True
    
    def _extract_windows(self, raw: mne.io.Raw) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]]]:
        """Split raw into windows or return full file."""
        if self.window_length is None:
            raw._original_info = raw.info.copy()
            return [raw], [(raw.times[0], raw.times[-1])]
        
        windows, times = self._split_raw(raw)
        
        if self.montage is None:
            for w in windows:
                w._original_info = raw.info.copy()
        
        return windows, times
    
    def run_single(
        self,
        raw: mne.io.Raw,
        start_time: float,
        end_time: float,
        filename: Union[str, Path]
    ) -> Optional[mne.io.Raw]:
        """
        Process a single window.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data window.
        start_time : float
            Window start time in seconds.
        end_time : float
            Window end time in seconds.
        filename : str or Path
            Source filename for logging.
        
        Returns
        -------
        mne.io.Raw or None
            Preprocessed raw object, or None if dropped.
        """
        info_str = f"File: {filename}. Time: ({start_time}, {end_time})."
        
        # Quality check
        passed, metrics = self._run_quality_check(raw, info_str)
        
        if self.return_quality_metrics:
            self._add_quality_metric(filename, start_time, end_time, metrics)
        
        if not passed:
            return None
        
        # Line noise removal
        self._remove_line_noise(raw)
        
        # Find and drop bad channels
        bad_chs = self._drop_bad_channels(raw)
        logging.info(f"{info_str} Found {len(bad_chs)} bad channels: {bad_chs}.")
        
        # Bandpass filter
        self._filter(raw)
        
        # Average reference
        self._average_reference(raw)
        
        # ICA artifact rejection
        if self.do_ica:
            excluded, labels, proba = self._ica_clean(raw)
            logging.info(f"{info_str} Excluding {len(excluded)} ICA components.")
            logging.info(f"{info_str} Labels: {labels}.")
            logging.info(f"{info_str} Probabilities: {[round(p, 2) for p in proba]}.")
            
            bad_chs = self._drop_bad_channels(raw)
            logging.info(f"{info_str} Found {len(bad_chs)} bad channels after ICA.")
        
        # Channel finalization
        if self.target_montage is not None:
            logging.info(f"{info_str} Interpolating to target montage.")
            raw = self._interpolate_to_target_montage(raw)
        else:
            missing = self._interpolate_missing(raw)
            if missing:
                logging.info(f"{info_str} Interpolated {len(missing)} missing channels.")
            
            extra = self._drop_extra_channels(raw)
            if extra:
                logging.info(f"{info_str} Removed {len(extra)} extra channels.")
            
            self._reorder_channels(raw)
        
        # Resample to target frequency
        self._interpolate_nearest(raw)
        
        return raw
    
    def _run_quality_check(
        self,
        raw: mne.io.Raw,
        info_str: str
    ) -> Tuple[bool, Optional[Tuple]]:
        """
        Run quality check on raw data.
        
        Returns
        -------
        passed : bool
            Whether the window passed quality checks.
        metrics : tuple or None
            Quality metrics (oha, thv, chv, bcr) if computed.
        """
        if not (self.return_quality_metrics or self.drop_bad_quality):
            return True, None
        
        quality, metrics = self._evaluate_quality(raw)
        
        if self.drop_bad_quality and not quality:
            logging.info(f"{info_str} Quality check failed. Dropping window.")
            return False, metrics
        
        return True, metrics
