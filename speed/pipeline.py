import mne
import numpy as np
import logging
import traceback
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Union

from speed.utils import split_raw, split_raw_annotations, load_montage
from speed.methods import PreprocessMethods
from speed.annotations import (
    parse_chbmit_summary, generate_non_seizure_windows,
    convert_chbmit_bipolar_to_monopolar, generate_tiled_annotations,
    parse_eegmat_annotations, parse_hmc_sleepscoring, parse_isruc_annotations,
    parse_mumtaz2016_annotations, parse_tuev_annotations, parse_tuab_annotations,
    parse_bcic_iv_2a_events, parse_shu_mi_events, parse_siena_seizures,
)


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
        include_ok_quality: bool = True,
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
        # Bipolar conversion
        bipolar_to_monopolar: bool = False,
        # Normalization
        normalize: Optional[str] = None,
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
        self.bipolar_to_monopolar = bipolar_to_monopolar
        
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
        self.include_ok_quality = include_ok_quality
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
        
        # Normalization
        self.normalize = normalize

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
    
    def _resample(self, raw: mne.io.Raw):
        return PreprocessMethods.resample(raw, self.sfreq)

    def _normalize(self, raw: mne.io.Raw):
        return PreprocessMethods.normalize(raw, self.normalize)
    
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
        metrics: Optional[Tuple],
        quality_rating: Optional[str] = None,
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
            "quality_rating": quality_rating,
        })


class PretrainPipeline(BasePipeline):
    """
    Main preprocessing pipeline for EEG data.

    Supports configurable preprocessing with quality checking, ICA artifact
    rejection, montage handling, and flexible channel management.

    Can operate in three modes:
    - Batch mode (default): Process multiple files, return windows for HDF5 batching
    - Single-file mode: Process one file at a time with metadata preservation
    - Event-based mode: Extract event-locked windows with labels for downstream tasks

    Parameters
    ----------
    All parameters from BasePipeline, plus:

    preserve_metadata : bool
        If True, restore original metadata (subject info, dates, annotations)
        after preprocessing. Useful for downstream tasks.
    event_windowing : bool
        If True, use event-based windowing instead of fixed windowing.
    event_labels : list of str, optional
        Labels to extract (e.g., ['T1', 'T2', 'T3', 'T4'] for EEGMMIDB).
    event_tmin : float
        Window start relative to event onset in seconds.
    event_tlen : float
        Window length in seconds.
    label_mapping : dict, optional
        Map event labels to integer class indices (e.g., {'T1': 0, 'T2': 1}).
    annotation_format : str
        Annotation format: 'eegmmidb', 'chbmit', or 'auto'.
    """

    def __init__(
        self,
        preserve_metadata: bool = False,
        event_windowing: bool = False,
        event_labels: Optional[List[str]] = None,
        event_tmin: float = 0.0,
        event_tlen: float = 5.0,
        label_mapping: Optional[Dict[str, Union[int, float]]] = None,
        annotation_format: str = "auto",
        task_type: str = "classification",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.preserve_metadata = preserve_metadata
        self.event_windowing = event_windowing
        self.event_labels = event_labels or []
        self.event_tmin = event_tmin
        self.event_tlen = event_tlen
        self.label_mapping = label_mapping or {}
        self.annotation_format = annotation_format
        self.task_type = task_type
    
    def __call__(self, src_paths: List[str]) -> Union[
        Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]],
        Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int], List[int]]
    ]:
        """Process a batch of files and return preprocessed windows."""
        return self.run(src_paths)
    
    def run(self, src_paths: List[str]) -> Union[
        Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]],
        Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int], List[int]]
    ]:
        """
        Load and preprocess EEG files (batch mode).

        Parameters
        ----------
        src_paths : list of str or Path
            Paths to source EEG files (.edf, .bdf, or .set).

        Returns
        -------
        raws : list of mne.io.Raw
            Preprocessed raw objects.
        times : list of tuple
            (start_time, end_time) for each window.
        indices : list of int
            Index of source file for each window.
        labels : list of int (only if event_windowing=True)
            Integer class labels for each window.
        """
        src_paths = [Path(p) for p in src_paths]
        logging.debug(f"Loading {len(src_paths)} files...")

        raws, times, indices = [], [], []
        labels = [] if self.event_windowing else None
        original_infos = []  # Store for metadata preservation

        for i, src_path in enumerate(src_paths):
            try:
                raw_orig = self._load_raw_file(src_path)
                if raw_orig is None:
                    continue

                # Store original info if preserving metadata
                orig_info = raw_orig.info.copy() if self.preserve_metadata else None
                orig_annot = raw_orig.annotations.copy() if self.preserve_metadata and raw_orig.annotations else None

                raw_orig = self._preprocess_channels(raw_orig, src_path)

                if not self._check_duration(raw_orig, src_path):
                    continue

                if self.event_windowing:
                    # Preprocess full recording first, then slice into event windows
                    raw_processed = self._run_single(
                        raw_orig, 0.0, raw_orig.times[-1], src_path,
                        orig_info, orig_annot
                    )
                    if raw_processed is None:
                        continue
                    windows, window_labels, window_times = self._extract_event_windows(
                        raw_processed, src_path
                    )
                    labels.extend(window_labels)
                    raws.extend(windows)
                    times.extend(window_times)
                    indices.extend([i] * len(windows))
                else:
                    # Fixed windowing: preprocess full recording, then slice
                    raw_processed = self._run_recording_level(raw_orig, src_path)
                    if raw_processed is None:
                        continue
                    windows, window_times = self._extract_windows(raw_processed)
                    raws.extend(windows)
                    times.extend(window_times)
                    indices.extend([i] * len(windows))
                    original_infos.extend([(orig_info, orig_annot)] * len(windows))

            except Exception as e:
                logging.error(f"Dropping file: {src_path.stem}. Error: {e}")
                continue

        total_windows = len(raws)
        logging.debug(f"Total windows: {total_windows}")

        if not self.event_windowing:
            # Per-window preprocessing (ICA, channel finalization, resampling)
            for i in range(len(raws)):
                start_time = round(times[i][0], 1)
                end_time = round(times[i][1], 1)
                filename = src_paths[indices[i]]
                orig_info, orig_annot = original_infos[i] if self.preserve_metadata else (None, None)

                try:
                    raws[i] = self._run_window_level(
                        raws[i], start_time, end_time, filename,
                        orig_info, orig_annot
                    )
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

        if self.event_windowing:
            labels = [l for l, m in zip(labels, mask) if m]
            return raws, times, indices, labels
        else:
            return raws, times, indices
    
    def process_single(
        self,
        src_path: str,
        skip_on_quality_fail: bool = False
    ) -> Tuple[Optional[mne.io.Raw], Optional[Tuple], bool]:
        """
        Process a single EEG file (single-file mode).
        
        Useful for downstream tasks where you want one output per input file.
        
        Parameters
        ----------
        src_path : str or Path
            Path to the source EEG file.
        skip_on_quality_fail : bool
            If True, return None for files that fail quality checks.
            If False, process them anyway.
        
        Returns
        -------
        raw : mne.io.Raw or None
            Preprocessed raw object, or None if processing failed.
        quality_metrics : tuple or None
            Quality metrics (oha, thv, chv, bcr) if computed.
        quality_passed : bool
            Whether the file passed quality checks.
        """
        src_path = Path(src_path)
        info_str = f"File: {src_path.name}"
        
        try:
            raw = self._load_raw_file(src_path)
            if raw is None:
                logging.error(f"{info_str} Unsupported file format.")
                return None, None, False
            
            # Store original info for metadata preservation
            orig_info = raw.info.copy() if self.preserve_metadata else None
            orig_annot = raw.annotations.copy() if self.preserve_metadata and raw.annotations else None
            
            raw = self._preprocess_channels(raw, src_path)

            # Run quality check first (if needed for metrics or skip logic)
            quality_passed = True
            quality_metrics = None
            quality_rating = None
            did_quality_check = False

            if self.return_quality_metrics or skip_on_quality_fail:
                quality_rating, quality_metrics = self._evaluate_quality(raw)
                quality_passed = quality_rating != "bad"
                did_quality_check = True

                if self.return_quality_metrics:
                    self._add_quality_metric(src_path, 0.0, raw.times[-1], quality_metrics, quality_rating)

                if not quality_passed and skip_on_quality_fail:
                    logging.info(f"{info_str} Quality rating: {quality_rating}. Skipping.")
                    return None, quality_metrics, False
            
            # Process (skip quality check only if we already did it above)
            raw = self._run_single(
                raw, 0.0, raw.times[-1], src_path, orig_info, orig_annot,
                skip_quality_check=did_quality_check
            )
            
            if raw is None:
                return None, quality_metrics, quality_passed
            
            logging.info(f"{info_str} Processing complete.")
            return raw, quality_metrics, quality_passed
            
        except Exception as e:
            logging.error(f"{info_str} Error: {e}")
            traceback.print_exc()
            return None, None, False
    
    def _load_raw_file(self, src_path: Path) -> Optional[mne.io.Raw]:
        """Load a raw EEG file based on its extension."""
        suffix = src_path.suffix.lower()
        
        if suffix in (".edf", ".rec"):
            return mne.io.read_raw_edf(src_path, preload=True, verbose=False)
        elif suffix == ".bdf":
            return mne.io.read_raw_bdf(src_path, preload=True, verbose=False)
        elif suffix == ".set":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r".*'boundary' events.*", category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore", message=r".*pymatreader cannot import.*", category=UserWarning
                )
                return mne.io.read_raw_eeglab(src_path, preload=True, verbose=False)
        elif suffix == ".gdf":
            return mne.io.read_raw_gdf(str(src_path), preload=True, verbose=False)
        else:
            logging.warning(f"Unsupported file type: {suffix}. Skipping {src_path.stem}")
            return None
    
    def _preprocess_channels(self, raw: mne.io.Raw, src_path: Path) -> mne.io.Raw:
        """Apply initial channel preprocessing: removal, renaming, montage.

        Returns the (potentially replaced) raw object.
        """
        # Bipolar-to-monopolar conversion (e.g., CHBMIT, TUEV, TUAB)
        # Must happen before standardize_channel_names and set_montage
        if self.bipolar_to_monopolar:
            raw = convert_chbmit_bipolar_to_monopolar(raw)
            logging.info(
                f"File: {src_path.stem}. Converted bipolar to "
                f"{len(raw.ch_names)} monopolar channels."
            )

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

        return raw
    
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
    
    def _extract_event_windows(
        self,
        raw: mne.io.Raw,
        src_path: Path
    ) -> Tuple[List[mne.io.Raw], List[int], List[Tuple[float, float]]]:
        """
        Extract event-locked windows with labels.

        Parameters
        ----------
        raw : mne.io.Raw
            Preprocessed raw EEG data
        src_path : Path
            Path to source file (for annotation loading)

        Returns
        -------
        windows : List[mne.io.Raw]
            List of windowed Raw objects
        labels : List[int]
            List of integer class labels
        time_slices : List[Tuple[float, float]]
            List of (start, end) time tuples
        """
        # 1. Load annotations based on format
        if self.annotation_format == 'eegmmidb':
            # EEGMMIDB uses EDF+ format — annotations (T0, T1, T2, etc.) are
            # already embedded in the file and loaded into raw.annotations by
            # _load_raw_file(). No need to re-read them.
            pass

        elif self.annotation_format == 'chbmit':
            # Parse seizure events from summary file
            summary_file = src_path.parent / f"{src_path.parent.name}-summary.txt"

            if not summary_file.exists():
                logging.warning(f"Summary file not found: {summary_file}")
                return [], [], []

            seizure_events = parse_chbmit_summary(summary_file, src_path.name)

            # Generate non-seizure windows
            recording_duration = raw.times[-1]
            non_seizure_events = generate_non_seizure_windows(
                recording_duration,
                seizure_events,
                window_length=self.event_tlen,
                stride=self.event_tlen,  # Non-overlapping
                margin=60.0
            )

            # Add all events to annotations
            for onset, duration in seizure_events:
                raw.annotations.append(onset, duration, 'seizure')
            for onset, label in non_seizure_events:
                raw.annotations.append(onset, self.event_tlen, label)

        elif self.annotation_format == 'eegmat':
            # MentalArithmetic: label from filename, tile into windows
            events = parse_eegmat_annotations(src_path, self.event_tlen, raw.times[-1])
            for onset, duration, label in events:
                raw.annotations.append(onset, duration, label)

        elif self.annotation_format == 'hmc':
            # HMC: parse companion sleep scoring file
            events = parse_hmc_sleepscoring(src_path)
            for onset, duration, label in events:
                raw.annotations.append(onset, duration, label)

        elif self.annotation_format == 'isruc':
            # ISRUC: parse companion annotation file
            events = parse_isruc_annotations(src_path)
            for onset, duration, label in events:
                raw.annotations.append(onset, duration, label)

        elif self.annotation_format == 'mumtaz2016':
            # Mumtaz2016: label from directory structure, tile into windows
            events = parse_mumtaz2016_annotations(src_path, self.event_tlen, raw.times[-1])
            for onset, duration, label in events:
                raw.annotations.append(onset, duration, label)

        elif self.annotation_format == 'tuev':
            # TUEV: parse .tse annotation files
            events = parse_tuev_annotations(src_path)
            for onset, duration, label in events:
                raw.annotations.append(onset, duration, label)

        elif self.annotation_format == 'tuab':
            # TUAB: label from directory path, tile into windows
            events = parse_tuab_annotations(src_path, self.event_tlen, raw.times[-1])
            for onset, duration, label in events:
                raw.annotations.append(onset, duration, label)

        elif self.annotation_format == 'bcic_iv_2a':
            # BCIC-IV-2a: extract GDF motor imagery event markers
            events = parse_bcic_iv_2a_events(raw)
            for onset, duration, label in events:
                raw.annotations.append(onset, duration, label)

        elif self.annotation_format == 'shu_mi':
            # SHU-MI: parse companion events TSV file
            events = parse_shu_mi_events(src_path, self.sfreq)
            for onset, duration, label in events:
                raw.annotations.append(onset, duration, label)

        elif self.annotation_format == 'siena':
            # Siena: parse seizure times from Seizures-list-PNxx.txt
            seizure_events = parse_siena_seizures(src_path)
            recording_duration = raw.times[-1]
            non_seizure_events = generate_non_seizure_windows(
                recording_duration,
                seizure_events,
                window_length=self.event_tlen,
                stride=self.event_tlen,
                margin=60.0
            )
            for onset, duration in seizure_events:
                raw.annotations.append(onset, duration, 'seizure')
            for onset, label in non_seizure_events:
                raw.annotations.append(onset, self.event_tlen, label)

        # 2. Extract windows using split_raw_annotations utility
        windows, time_slices, descriptions = split_raw_annotations(
            raw,
            labels=self.event_labels,
            tmin=self.event_tmin,
            tlen=self.event_tlen,
            verbose=True
        )

        # 3. Map labels: classification uses label_mapping, regression parses from description
        filtered_windows, filtered_labels, filtered_times = [], [], []
        if self.task_type == 'regression':
            for window, desc, time_slice in zip(windows, descriptions, time_slices):
                try:
                    parts = desc.split('_')
                    target_values = [float(p) for p in parts[1:]]
                    if len(target_values) == 1:
                        target_values = target_values[0]
                except (ValueError, IndexError):
                    continue
                filtered_windows.append(window)
                filtered_labels.append(target_values)
                filtered_times.append(time_slice)
        else:
            for window, desc, time_slice in zip(windows, descriptions, time_slices):
                if desc not in self.label_mapping:
                    logging.warning(
                        f"Annotation '{desc}' not in label_mapping "
                        f"{list(self.label_mapping.keys())}. Skipping window."
                    )
                    continue
                filtered_windows.append(window)
                filtered_labels.append(self.label_mapping[desc])
                filtered_times.append(time_slice)

        return filtered_windows, filtered_labels, filtered_times

    def _extract_windows(self, raw: mne.io.Raw) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]]]:
        """Split raw into windows or return full file."""
        if self.window_length is None:
            return [raw], [(raw.times[0], raw.times[-1])]

        return self._split_raw(raw)
    
    def _run_recording_level(
        self,
        raw: mne.io.Raw,
        filename: Union[str, Path],
        skip_quality_check: bool = False
    ) -> Optional[mne.io.Raw]:
        """
        Recording-level preprocessing (should run on full recording before windowing).

        Includes quality check, line noise removal, bad channel detection,
        bandpass filtering, and average re-referencing.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data (full recording).
        filename : str or Path
            Source filename for logging.
        skip_quality_check : bool
            If True, skip quality check.

        Returns
        -------
        mne.io.Raw or None
            Preprocessed raw object, or None if quality check failed.
        """
        info_str = f"File: {filename}."

        # Quality check
        if not skip_quality_check:
            passed, metrics, rating = self._run_quality_check(raw, info_str)

            if self.return_quality_metrics:
                self._add_quality_metric(filename, 0.0, raw.times[-1], metrics, rating)

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

        return raw

    def _run_window_level(
        self,
        raw: mne.io.Raw,
        start_time: float,
        end_time: float,
        filename: Union[str, Path],
        original_info: Optional[mne.Info] = None,
        original_annotations: Optional[mne.Annotations] = None,
    ) -> Optional[mne.io.Raw]:
        """
        Window-level preprocessing (runs per-window after windowing).

        Includes ICA artifact rejection, channel finalization,
        resampling, and metadata restoration.

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
        original_info : mne.Info, optional
            Original info to restore (if preserve_metadata=True).
        original_annotations : mne.Annotations, optional
            Original annotations to restore.

        Returns
        -------
        mne.io.Raw or None
            Preprocessed raw object.
        """
        info_str = f"File: {filename}. Time: ({start_time}, {end_time})."

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
        self._resample(raw)

        # Normalize
        if self.normalize is not None:
            self._normalize(raw)

        # Restore metadata if requested
        if self.preserve_metadata and original_info is not None:
            self._restore_metadata(raw, original_info, original_annotations)

        return raw

    def _run_single(
        self,
        raw: mne.io.Raw,
        start_time: float,
        end_time: float,
        filename: Union[str, Path],
        original_info: Optional[mne.Info] = None,
        original_annotations: Optional[mne.Annotations] = None,
        skip_quality_check: bool = False
    ) -> Optional[mne.io.Raw]:
        """
        Process a single window/file (full pipeline).

        Composes recording-level and window-level preprocessing.
        Used by process_single() and the event-windowing path where the
        full recording is processed as one unit.

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
        original_info : mne.Info, optional
            Original info to restore (if preserve_metadata=True).
        original_annotations : mne.Annotations, optional
            Original annotations to restore.
        skip_quality_check : bool
            If True, skip quality check (used when already done by caller).

        Returns
        -------
        mne.io.Raw or None
            Preprocessed raw object, or None if dropped.
        """
        raw = self._run_recording_level(raw, filename, skip_quality_check)
        if raw is None:
            return None

        return self._run_window_level(
            raw, start_time, end_time, filename,
            original_info, original_annotations
        )
    
    def _run_quality_check(
        self,
        raw: mne.io.Raw,
        info_str: str
    ) -> Tuple[bool, Optional[Tuple], Optional[str]]:
        """
        Run quality check on raw data.

        Returns
        -------
        passed : bool
            Whether the window passed quality checks.
        metrics : tuple or None
            Quality metrics (oha, thv, chv, bcr) if computed.
        rating : str or None
            Quality rating: "good", "ok", or "bad".
        """
        if not (self.return_quality_metrics or self.drop_bad_quality):
            return True, None, None

        rating, metrics = self._evaluate_quality(raw)

        if self.drop_bad_quality and rating == "bad":
            logging.info(f"{info_str} Quality rating: bad. Dropping window.")
            return False, metrics, rating

        if self.drop_bad_quality and not self.include_ok_quality and rating == "ok":
            logging.info(f"{info_str} Quality rating: ok. Dropping window (include_ok_quality=False).")
            return False, metrics, rating

        return True, metrics, rating
    
    def _restore_metadata(
        self,
        raw: mne.io.Raw,
        original_info: mne.Info,
        original_annotations: Optional[mne.Annotations]
    ) -> None:
        """Restore non-signal metadata from the original file."""
        if original_info.get('meas_date') is not None:
            with raw.info._unlock():
                raw.info['meas_date'] = original_info['meas_date']
        
        if original_info.get('subject_info') is not None:
            with raw.info._unlock():
                raw.info['subject_info'] = original_info['subject_info']
        
        if original_info.get('device_info') is not None:
            with raw.info._unlock():
                raw.info['device_info'] = original_info['device_info']
        
        if original_info.get('experimenter') is not None:
            with raw.info._unlock():
                raw.info['experimenter'] = original_info['experimenter']
        
        if original_info.get('description') is not None:
            with raw.info._unlock():
                raw.info['description'] = original_info['description']
        
        if original_annotations is not None and len(original_annotations) > 0:
            raw.set_annotations(original_annotations)
