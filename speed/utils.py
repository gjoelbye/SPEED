"""
Utility functions for EEG preprocessing.

Includes:
- Channel name resolution and standardization
- Raw data windowing and splitting
- Montage creation
- I/O functions for HDF5 and BDF formats
"""
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import mne
import numpy as np
import h5py


# =============================================================================
# Channel Name Resolution
# =============================================================================

def _heuristic_eeg_resolution(eeg_ch_name: str):
    return eeg_ch_name if len(eeg_ch_name) > 0 else None


def _heuristic_eog_resolution(eog_channel_name):
    return eog_channel_name


def _heuristic_ref_resolution(ref_channel_name: str):
    if ref_channel_name.find('A1') != -1:
        return 'A1'
    elif ref_channel_name.find('A2') != -1:
        return 'A2'
    if ref_channel_name.find('L') != -1:
        return 'A1'
    elif ref_channel_name.find('R') != -1:
        return 'A2'
    else:
        return "REF"


def _heuristic_ecg_resolution(ecg_ch_name: str):
    n = ''.join([i for i in ecg_ch_name if i.isdigit()])
    return "EKG" + n


def _heuristic_extra_resolution(extra_ch_name: str):
    return extra_ch_name


def _clean_channel_name(channel_name: str):
    channel_name = channel_name.upper()
    if channel_name == "REF":
        return channel_name
    channel_name = channel_name.replace('EEG', '')
    channel_name = channel_name.replace('LE', '')
    channel_name = channel_name.replace('REF', '')
    channel_name = channel_name.replace('EAR', '')
    channel_name = ''.join(c for c in channel_name if c.isalnum())
    return channel_name


def create_channel_type_dict(raw: mne.io.Raw) -> dict:
    """Create a dictionary mapping channel names to their types."""
    type_dict = dict()
    for k, v in dict(zip(raw.info['ch_names'], raw.get_channel_types())).items():
        if any([x in k for x in ["A1", "A2"]]):
            type_dict[k] = 'ref'
        elif 'ECG' in k or 'EKG' in k:
            type_dict[k] = 'ecg'
        elif any([x in k for x in ['PHOTIC', 'IBI', 'BURSTS', 'SUPPR']]):
            type_dict[k] = 'extra'
        elif any([x in k for x in ['ROC', 'LOC']]):
            type_dict[k] = 'eog'
        else:
            type_dict[k] = v
    return type_dict


def heuristic_resolution(old_type_dict: OrderedDict) -> OrderedDict:
    """Resolve channel names to standardized names based on channel type."""
    resolver = {
        'eeg': _heuristic_eeg_resolution,
        'eog': _heuristic_eog_resolution,
        'ref': _heuristic_ref_resolution,
        'ecg': _heuristic_ecg_resolution,
        'extra': _heuristic_extra_resolution
    }
    
    new_type_dict = OrderedDict()
    
    for old_name, ch_type in old_type_dict.items():
        if ch_type is None:
            new_type_dict[old_name] = None
            continue
        
        new_name = _clean_channel_name(old_name)
        new_name = resolver[ch_type](new_name)
        
        if new_name is None:
            new_type_dict[new_name] = None
        else:
            while new_name in new_type_dict.keys():
                new_name = new_name + '-COPY'
            new_type_dict[new_name] = old_type_dict[old_name]
    
    assert len(new_type_dict) == len(old_type_dict)
    return new_type_dict


# =============================================================================
# Raw Data Windowing
# =============================================================================

def split_raw(
    raw: mne.io.Raw,
    window_length: float = 60,
    shift_seconds: Optional[float] = None
) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]]]:
    """
    Split raw EEG data into fixed-length windows.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to split.
    window_length : float
        Length of each window in seconds.
    shift_seconds : float, optional
        Shift between consecutive windows. None = no overlap.
    
    Returns
    -------
    windows : list of mne.io.Raw
        List of windowed raw objects.
    time_slices : list of tuple
        (start_time, end_time) for each window.
    """
    sfreq = raw.info['sfreq']
    window_samples = int(window_length * sfreq)
    
    if shift_seconds is None or shift_seconds == 0:
        shift_seconds = window_length
    shift_samples = int(shift_seconds * sfreq)
    
    windows = []
    time_slices = []
    start_sample = 0
    
    while start_sample + window_samples <= raw.n_times:
        end_sample = start_sample + window_samples
        window, times = raw[:, start_sample:end_sample]
        
        info = mne.create_info(
            ch_names=raw.info['ch_names'],
            sfreq=sfreq,
            ch_types=raw.get_channel_types()
        )
        window_raw = mne.io.RawArray(window, info, verbose=False)
        window_raw.set_montage(raw.get_montage())
        
        windows.append(window_raw)
        time_slices.append((times[0], times[-1]))
        start_sample += shift_samples
    
    return windows, time_slices


def get_unannotated_raw(raw: mne.io.Raw, resting_state: List[str] = ['T0']) -> mne.io.Raw:
    """Extract non-annotated segments from raw data."""
    non_annotated_data_segments = []
    sfreq = raw.info['sfreq']
    
    annotations = sorted(raw.annotations, key=lambda ann: ann['onset'])
    last_end_sample = 0
    
    for annotation in annotations:
        if annotation['description'] not in resting_state:
            onset_sample = int(annotation['onset'] * sfreq)
            end_sample = int((annotation['onset'] + annotation['duration']) * sfreq)
            
            if onset_sample > last_end_sample:
                segment_data, _ = raw[:, last_end_sample:onset_sample]
                non_annotated_data_segments.append(segment_data)
            
            last_end_sample = end_sample
    
    if last_end_sample < raw.n_times:
        segment_data, _ = raw[:, last_end_sample:raw.n_times]
        non_annotated_data_segments.append(segment_data)
    
    concatenated_data = np.concatenate(non_annotated_data_segments, axis=1)
    info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=sfreq, ch_types='eeg')
    return mne.io.RawArray(concatenated_data, info)


def split_raw_annotations(
    raw: mne.io.Raw,
    labels: List[str],
    tmin: float = -0.5,
    tlen: float = 5.0,
    verbose: bool = True
) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[str]]:
    """Split raw data based on annotations."""
    windows = []
    time_slices = []
    descriptions = []
    
    for annotation in raw.annotations:
        onset, duration, description, _ = annotation.values()
        
        if description not in labels:
            continue
        
        sfreq = raw.info['sfreq']
        start_sample = round((onset + tmin) * sfreq)
        
        if start_sample < 0:
            if verbose:
                print(f'Skipping {description} at {onset:.2f} s')
            continue
        
        tlen_sample = round(tlen * sfreq)
        end_sample = start_sample + tlen_sample
        
        if end_sample > raw.n_times:
            if verbose:
                print(f'Skipping {description} at {onset:.2f} s')
            continue
        
        window, times = raw[:, start_sample:end_sample]
        
        info = mne.create_info(
            ch_names=raw.info['ch_names'],
            sfreq=sfreq,
            ch_types=raw.get_channel_types()
        )
        window_raw = mne.io.RawArray(window, info, verbose=False)
        window_raw.set_montage(raw.get_montage())
        
        windows.append(window_raw)
        time_slices.append((times[0], times[-1]))
        descriptions.append(description)
    
    return windows, time_slices, descriptions


# =============================================================================
# Montage Creation
# =============================================================================

def make_tuh_montage() -> mne.channels.DigMontage:
    """
    Create a custom montage for TUH dataset.
    
    Combines standard_1005 with T1/T2 positions from standard_postfixed.
    """
    standard_1005 = mne.channels.make_standard_montage('standard_1005')
    standard_postfixed = mne.channels.make_standard_montage('standard_postfixed')
    
    t1_pos = standard_postfixed.get_positions()['ch_pos']['T1']
    t2_pos = standard_postfixed.get_positions()['ch_pos']['T2']
    
    pos = standard_1005.get_positions()
    ch_pos = pos['ch_pos']
    ch_pos['T1'] = t1_pos
    ch_pos['T2'] = t2_pos
    
    ch_pos = OrderedDict({ch.lower().capitalize(): p for ch, p in ch_pos.items()})
    
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=pos['nasion'],
        lpa=pos['lpa'],
        rpa=pos['rpa'],
        hsp=pos['hsp'],
        hpi=pos['hpi'],
        coord_frame=pos['coord_frame']
    )
    return montage


def load_montage(montage_source: str) -> Optional[mne.channels.DigMontage]:
    """
    Load a montage from various sources.
    
    Parameters
    ----------
    montage_source : str
        Either:
        - "tuh" for TUH custom montage
        - A standard MNE montage name (e.g., "standard_1005")
        - A path to a .fif montage file
    
    Returns
    -------
    montage : DigMontage or None
    """
    if montage_source is None:
        return None
    elif montage_source == "tuh":
        return make_tuh_montage()
    elif Path(montage_source).exists() and montage_source.endswith('.fif'):
        return mne.channels.read_dig_fif(montage_source)
    else:
        return mne.channels.make_standard_montage(montage_source)


# =============================================================================
# I/O Functions
# =============================================================================

def save_hdf5(
    raws: List[mne.io.Raw],
    src_paths: List[Path],
    times: List[Tuple[float, float]],
    indices: List[int],
    dest_path: Path
) -> None:
    """
    Save preprocessed data as HDF5 batch file.
    
    Parameters
    ----------
    raws : list of mne.io.Raw
        Preprocessed raw objects.
    src_paths : list of Path
        Original source file paths.
    times : list of tuple
        (start_time, end_time) for each window.
    indices : list of int
        Index of source file for each window.
    dest_path : Path
        Output HDF5 file path.
    """
    with h5py.File(dest_path, "w") as f:
        f.attrs["files"] = [p.stem for p in src_paths]
        f.attrs["file_idxs"] = indices
        f.attrs["time_slices"] = times
        f.create_dataset("data", data=np.array([r._data for r in raws]), dtype='float32')


def write_bdf_from_raw(
    raw: mne.io.Raw,
    out_path: str,
    write_annotations: bool = True
) -> None:
    """
    Export MNE Raw object to BDF+ format using pyedflib.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to export.
    out_path : str
        Output file path.
    write_annotations : bool
        Whether to include annotations in the output file.
    
    Raises
    ------
    ImportError
        If pyedflib is not installed.
    RuntimeError
        If no writable channels are found.
    """
    try:
        import pyedflib
        from pyedflib import FILETYPE_BDFPLUS
    except ImportError:
        raise ImportError("pyedflib is required for BDF export. Install with: pip install pyedflib")
    
    picks = mne.pick_types(
        raw.info,
        eeg=True, eog=True, ecg=True, emg=True, misc=True, stim=True,
        seeg=True, dbs=True, ecog=True, resp=True, bio=True, exci=True, ias=True, syst=True
    )
    if len(picks) == 0:
        raise RuntimeError("No writable channels were found.")
    
    data_v = raw.get_data(picks).astype(np.float64)
    data_uv = data_v * 1e6
    ch_names = [raw.ch_names[i] for i in picks]
    sfreq = float(raw.info["sfreq"])
    
    def round_to_edf8(x: float) -> float:
        """Round value to fit in 8-character EDF field."""
        s = "-" if x < 0 else ""
        int_digits = len(str(int(abs(x))))
        max_decimals = max(0, 7 - len(s) - int_digits)
        if max_decimals == 0:
            return float(f"{int(x)}")
        return float(f"{x:.{max_decimals}f}")
    
    MARGIN_PCT = 0.01
    pmins = data_uv.min(axis=1)
    pmaxs = data_uv.max(axis=1)
    spans = np.maximum(pmaxs - pmins, 1.0)
    pmins = pmins - MARGIN_PCT * spans
    pmaxs = pmaxs + MARGIN_PCT * spans
    
    pmins = np.array([round_to_edf8(v) for v in pmins], dtype=float)
    pmaxs = np.array([round_to_edf8(v) for v in pmaxs], dtype=float)
    
    DEFAULT_WINDOW_UV = 1.0
    for i in range(len(pmins)):
        if not np.isfinite(pmins[i]) or not np.isfinite(pmaxs[i]) or pmaxs[i] <= pmins[i]:
            pmins[i] = -DEFAULT_WINDOW_UV
            pmaxs[i] = +DEFAULT_WINDOW_UV
    
    sig_headers = []
    for name, pmin, pmax in zip(ch_names, pmins, pmaxs):
        sig_headers.append({
            "label": name[:16],
            "dimension": "uV",
            "sample_frequency": sfreq,
            "physical_min": float(pmin),
            "physical_max": float(pmax),
            "digital_min": -8388608,
            "digital_max": 8388607
        })
    
    md = raw.info.get("meas_date")
    if md is not None and isinstance(md, (tuple, list)):
        md = md[0]
    startdate = md.replace(tzinfo=None) if md is not None else datetime.now()
    
    f = pyedflib.EdfWriter(str(out_path), n_channels=len(sig_headers), file_type=FILETYPE_BDFPLUS)
    f.setSignalHeaders(sig_headers)
    f.setStartdatetime(startdate)
    f.writeSamples([data_uv[i] for i in range(len(sig_headers))])
    
    if write_annotations and len(raw.annotations) > 0:
        first_time = float(raw.first_time)
        for onset, duration, desc in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description
        ):
            f.writeAnnotation(float(onset + first_time), float(duration), str(desc))
    
    f.close()
