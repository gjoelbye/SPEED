"""
EEG preprocessing methods.

Static methods for signal processing, quality evaluation, channel handling,
and artifact rejection.
"""
import contextlib
import io
import warnings

import mne
import numpy as np
import pyprep
from mne.preprocessing import ICA
from mne_icalabel import label_components
from meegkit.detrend import detrend
from meegkit.dss import dss_line_iter

from speed.utils import create_channel_type_dict, heuristic_resolution

warnings.simplefilter(action='ignore', category=RuntimeWarning)


class PreprocessMethods:
    """Static methods for EEG preprocessing."""
    
    # =========================================================================
    # Channel Name Standardization
    # =========================================================================
    
    @staticmethod
    def to_standard_names(raw: mne.io.Raw) -> None:
        """
        Convert channel names to standardized format.
        
        Applies heuristic resolution and renames T3/T4/P7/P8 to T7/T8/T5/T6.
        """
        channel_type_dict = create_channel_type_dict(raw)
        channel_names = list(channel_type_dict.keys())
        revised_channel_types = heuristic_resolution(channel_type_dict)
        rename_map = {
            orig: new.lower().capitalize()
            for orig, new in zip(channel_names, revised_channel_types)
        }
        raw.rename_channels(rename_map)
        
        # Rename channels with same positions in standard 10-5 system
        rename_map = {'T3': 'T7', 'T4': 'T8', 'P7': 'T5', 'P8': 'T6'}
        rename_map = {k: v for k, v in rename_map.items() if k in raw.ch_names}
        raw.rename_channels(rename_map)
    
    # =========================================================================
    # Quality Evaluation
    # =========================================================================
    
    @staticmethod
    def evaluate_quality(
        raw: mne.io.Raw,
        oha_threshold: float = 40e-6,
        thv_threshold: float = 40e-6,
        chv_threshold: float = 80e-6,
        min_unique_ratio: float = 0.001,
        min_nchans: int = 10,
        line_freqs: list = [60],
        hp_freq: float = 0.5,
        lp_freq: float = 100,
        oha_limit: float = 0.8,
        thv_limit: float = 0.5,
        chv_limit: float = 0.5,
        bcr_limit: float = 0.8
    ) -> tuple:
        """
        Evaluate EEG recording quality.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw EEG data to evaluate.
        oha_threshold : float
            Amplitude threshold for high-amplitude detection (Volts).
        thv_threshold : float
            Temporal variance threshold (Volts).
        chv_threshold : float
            Channel variance threshold (Volts).
        min_unique_ratio : float
            Minimum fraction of unique samples per channel.
        min_nchans : int
            Minimum number of channels required.
        line_freqs : list
            Power line frequencies for notch filtering.
        hp_freq, lp_freq : float
            High-pass and low-pass filter frequencies.
        oha_limit, thv_limit, chv_limit, bcr_limit : float
            Thresholds for passing quality check (fraction).
        
        Returns
        -------
        rating : str
            Quality rating: "good", "ok", or "bad".
        metrics : tuple
            (oha, thv, chv, bcr) quality metrics.
        """
        n_chans = raw.info['nchan']
        if n_chans < min_nchans:
            return "bad", (1.0, 1.0, 1.0, 1.0)

        raw = raw.copy()

        # Detect discrete/flat channels
        min_unique = int(min_unique_ratio * raw.n_times)
        unique_counts = np.array([len(np.unique(chan_data)) for chan_data in raw._data])
        discrete_channels = np.array(raw.ch_names)[unique_counts < min_unique].tolist()

        # Detect noisy channels
        noisychannels = pyprep.NoisyChannels(raw)
        noisychannels.find_bad_by_SNR()
        noisychannels.find_bad_by_correlation()
        noisychannels.find_bad_by_deviation()
        noisychannels.find_bad_by_hfnoise()
        noisychannels.find_bad_by_nan_flat()

        # Apply simple filtering for metric calculation
        # Cap filter frequencies to Nyquist frequency
        sfreq = raw.info['sfreq']
        nyquist = sfreq / 2.0

        # Filter line frequencies that are below Nyquist
        valid_line_freqs = [f for f in line_freqs if f < nyquist]
        if valid_line_freqs:
            raw.notch_filter(valid_line_freqs, verbose=False)

        # Cap filter frequencies to be less than Nyquist
        effective_hp_freq = hp_freq if hp_freq is None or hp_freq < nyquist else None
        effective_lp_freq = lp_freq if lp_freq is None or lp_freq < nyquist else nyquist * 0.95

        if effective_hp_freq is not None or effective_lp_freq is not None:
            raw.filter(effective_hp_freq, effective_lp_freq, verbose=False)

        # Calculate quality metrics
        oha = np.mean(np.abs(raw._data) > oha_threshold)
        thv = np.mean(np.std(raw._data, axis=0) > thv_threshold)
        chv = np.mean(np.std(raw._data, axis=1) > chv_threshold)

        # Bad channel ratio
        bad_channels = list(set(discrete_channels + noisychannels.get_bads()))
        bcr = len(bad_channels) / n_chans

        # Three-tier rating:
        #   "good" — all metrics within strict limits
        #   "bad"  — any metric exceeds 2x the strict limit
        #   "ok"   — in between
        strict_pass = (oha < oha_limit) and (thv < thv_limit) and (chv < chv_limit) and (bcr < bcr_limit)
        relaxed_pass = (
            (oha < 2 * oha_limit) and (thv < 2 * thv_limit)
            and (chv < 2 * chv_limit) and (bcr < 2 * bcr_limit)
        )

        if strict_pass:
            rating = "good"
        elif relaxed_pass:
            rating = "ok"
        else:
            rating = "bad"

        return rating, (oha, thv, chv, bcr)
    
    # =========================================================================
    # Resampling
    # =========================================================================
    
    @staticmethod
    def resample(raw: mne.io.Raw, sfreq: float = 256.0) -> None:
        """
        Resample data using MNE's anti-aliased resampling.

        Uses a FIR anti-aliasing filter before decimation to prevent
        frequency aliasing artifacts.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw data to resample (modified in-place).
        sfreq : float
            Target sampling frequency.
        """
        if raw.info['sfreq'] != sfreq:
            raw.resample(sfreq, verbose=False)
    
    # =========================================================================
    # Normalization
    # =========================================================================

    @staticmethod
    def normalize(raw: mne.io.Raw, method: str = "zscore") -> None:
        """
        Normalize EEG data in-place.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw data to normalize (modified in-place).
        method : str
            Normalization method: "zscore", "minmax", or "robust".
        """
        data = raw._data
        eps = 1e-8

        if method == "zscore":
            mean = data.mean(axis=-1, keepdims=True)
            std = data.std(axis=-1, keepdims=True)
            raw._data = (data - mean) / (std + eps)
        elif method == "minmax":
            dmin = data.min(axis=-1, keepdims=True)
            dmax = data.max(axis=-1, keepdims=True)
            raw._data = (data - dmin) / (dmax - dmin + eps)
        elif method == "robust":
            median = np.median(data, axis=-1, keepdims=True)
            q75 = np.percentile(data, 75, axis=-1, keepdims=True)
            q25 = np.percentile(data, 25, axis=-1, keepdims=True)
            iqr = q75 - q25
            raw._data = (data - median) / (iqr + eps)
        else:
            raise ValueError(f"Unknown normalization method: {method!r}. "
                             f"Choose from 'zscore', 'minmax', 'robust'.")

    # =========================================================================
    # Montage and Channel Management
    # =========================================================================
    
    @staticmethod
    def set_montage(raw: mne.io.Raw, montage) -> list:
        """
        Set montage on raw data, dropping channels not in montage.
        
        Returns list of dropped channel names.
        """
        if montage is None:
            return []
        drop_chs = [ch for ch in raw.ch_names if ch not in montage.ch_names]
        raw.drop_channels(drop_chs)
        raw.set_montage(montage)
        return drop_chs
    
    @staticmethod
    def find_bad_channels(raw: mne.io.Raw, ransac: bool = True, drop: bool = True) -> list:
        """
        Find and optionally drop bad channels using pyprep.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw data to check.
        ransac : bool
            Whether to use RANSAC for bad channel detection.
        drop : bool
            Whether to drop detected bad channels.
        
        Returns
        -------
        bad_chs : list
            List of bad channel names.
        """
        noisychannels = pyprep.NoisyChannels(raw)
        noisychannels.find_bad_by_deviation(deviation_threshold=5.0)
        noisychannels.find_bad_by_hfnoise()
        noisychannels.find_bad_by_correlation(frac_bad=0.05)
        noisychannels.find_bad_by_SNR()
        
        if ransac and raw.info['nchan'] >= 16:
            try:
                noisychannels.find_bad_by_ransac()
            except Exception:
                pass
        
        bad_chs = noisychannels.get_bads()
        if drop:
            raw.drop_channels(bad_chs)
        return bad_chs
    
    @staticmethod
    def drop_extra_channels(raw: mne.io.Raw, target_channels: list) -> list:
        """Drop channels not in target list."""
        extra_ch = [c for c in raw.ch_names if c not in target_channels]
        raw.drop_channels(extra_ch)
        return extra_ch
    
    @staticmethod
    def reorder_channels(raw: mne.io.Raw, target_channels: list) -> None:
        """Reorder channels to match target order."""
        new_ch_order = [ch for ch in target_channels if ch in raw.ch_names]
        raw.reorder_channels(new_ch_order)
    
    @staticmethod
    def drop_channels_manually(raw: mne.io.Raw, channels_to_remove: list) -> list:
        """Drop specified channels from raw data."""
        if channels_to_remove is None:
            return []
        existing = [ch for ch in channels_to_remove if ch in raw.ch_names]
        raw.drop_channels(existing)
        return existing
    
    # =========================================================================
    # Filtering
    # =========================================================================
    
    @staticmethod
    def filter(
        raw: mne.io.Raw,
        hp_freq: float,
        lp_freq: float,
        line_freqs: list,
        do_detrend: bool = True
    ) -> None:
        """
        Apply filtering to raw data.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw data to filter (modified in-place).
        hp_freq : float
            High-pass filter frequency. None to skip.
        lp_freq : float
            Low-pass filter frequency. None to skip.
        line_freqs : list
            Power line frequencies for DSS notch filtering.
        do_detrend : bool
            Whether to apply detrending before filtering.
        """
        sfreq = raw.info['sfreq']
        nyquist = sfreq / 2.0
        
        # Cap filter frequencies to Nyquist
        effective_hp_freq = min(hp_freq, nyquist * 0.95) if hp_freq is not None else None
        effective_lp_freq = min(lp_freq, nyquist * 0.95) if lp_freq is not None else None
        
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if do_detrend:
                raw._data = detrend(raw._data.T, 1)[0].T
            
            for line_freq in line_freqs:
                if line_freq < nyquist:
                    if effective_lp_freq is not None and line_freq >= effective_lp_freq:
                        continue
                    raw._data = dss_line_iter(raw._data.T, line_freq, sfreq)[0].T
        
        if effective_hp_freq is not None or effective_lp_freq is not None:
            raw.filter(effective_hp_freq, effective_lp_freq, verbose=False)
    
    # =========================================================================
    # ICA Artifact Rejection
    # =========================================================================
    
    @staticmethod
    def ica_clean(
        raw: mne.io.Raw,
        iclabel_threshold: float,
        included_components: list
    ) -> tuple:
        """
        Apply ICA-based artifact rejection using ICLabel.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw data to clean (modified in-place).
        iclabel_threshold : float
            Probability threshold for component exclusion.
        included_components : list
            Component labels to keep (e.g., ["brain", "other"]).
        
        Returns
        -------
        exclude_idx : list
            Indices of excluded components.
        labels : list
            Labels for all components.
        y_proba : array
            Probability scores for all components.
        """
        ica = ICA(
            n_components=np.linalg.matrix_rank(raw._data),
            max_iter=1000,
            method="infomax",
            fit_params=dict(extended=True),
            verbose=False
        )
        ica.fit(raw, verbose=False)
        
        ic_labels = label_components(raw, ica, method="iclabel")
        labels = ic_labels["labels"]
        y_proba = ic_labels["y_pred_proba"]
        
        exclude_idx = [
            idx for idx, (label, y_prob) in enumerate(zip(labels, y_proba))
            if label not in included_components and y_prob > iclabel_threshold
        ]
        
        ica.apply(raw, exclude=exclude_idx, verbose=False)
        return exclude_idx, labels, y_proba
    
    # =========================================================================
    # Channel Interpolation
    # =========================================================================
    
    @staticmethod
    def interpolate_missing(
        raw: mne.io.Raw,
        target_channels: list,
        exclude_channels: list = None,
        montage=None,
        mode: str = "accurate"
    ) -> list:
        """
        Interpolate missing channels to match target channel set.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw data (modified in-place).
        target_channels : list
            List of required channel names.
        exclude_channels : list, optional
            Channels to exclude from interpolation.
        montage : DigMontage, optional
            Montage with channel locations.
        mode : str
            Interpolation mode: "accurate" or "fast".
        
        Returns
        -------
        missing_ch : list
            List of interpolated channel names.
        """
        exclude_channels = exclude_channels or []
        missing_ch = list(set(target_channels) - set(raw.ch_names) - set(exclude_channels))
        if len(missing_ch) == 0:
            return missing_ch
        
        # Add placeholder channels with NaN values
        new_channel_data = np.nan * np.zeros((len(missing_ch), raw._data.shape[1]))
        new_channel_info = mne.create_info(
            missing_ch, sfreq=raw.info['sfreq'], ch_types='eeg'
        )
        raw.add_channels(
            [mne.io.RawArray(new_channel_data, new_channel_info, verbose=False)],
            force_update_info=True
        )
        raw.info['bads'] = missing_ch
        
        # Set channel locations
        if montage is not None:
            raw.set_montage(montage, verbose=False)
        elif hasattr(raw, "_original_info"):
            orig_info = raw._original_info
            for ch_name in missing_ch:
                if ch_name in orig_info["ch_names"]:
                    src_idx = orig_info["ch_names"].index(ch_name)
                    dst_idx = raw.ch_names.index(ch_name)
                    raw.info["chs"][dst_idx]["loc"] = orig_info["chs"][src_idx]["loc"].copy()
        
        raw.interpolate_bads(reset_bads=True, mode=mode, verbose=False)
        return missing_ch
    
    @staticmethod
    def interpolate_to_target_montage(raw: mne.io.Raw, target_montage) -> mne.io.Raw:
        """
        Interpolate raw data to a target montage using spline interpolation.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw data to interpolate.
        target_montage : DigMontage
            Target montage with desired channel locations.
        
        Returns
        -------
        raw : mne.io.Raw
            Interpolated raw data.
        """
        return raw.interpolate_to(sensors=target_montage, method='spline')
    
    @staticmethod
    def zero_missing(raw: mne.io.Raw, target_channels: list, montage=None) -> list:
        """
        Add missing channels as zeros.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw data (modified in-place).
        target_channels : list
            List of required channel names.
        montage : DigMontage, optional
            Montage with channel locations.
        
        Returns
        -------
        missing_ch : list
            List of added channel names.
        """
        missing_ch = [c for c in target_channels if c not in raw.ch_names]
        if len(missing_ch) == 0:
            return missing_ch
        
        new_channel_data = np.zeros((len(missing_ch), raw._data.shape[1]))
        new_channel_info = mne.create_info(
            missing_ch, sfreq=raw.info['sfreq'], ch_types='eeg'
        )
        raw.add_channels(
            [mne.io.RawArray(new_channel_data, new_channel_info, verbose=False)],
            force_update_info=True
        )
        
        if montage is not None:
            raw.set_montage(montage, verbose=False)
        
        return missing_ch


# Backward compatibility aliases
PreprocessMethods.reorder_chans = PreprocessMethods.reorder_channels
PreprocessMethods.interpolate_to_hbn = PreprocessMethods.interpolate_to_target_montage
PreprocessMethods.interpolate_nearest = PreprocessMethods.resample
