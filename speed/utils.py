"""Utility functions for EEG preprocessing."""

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import shutil
import fnmatch

import mne
import numpy as np
import h5py


# --- Channel Name Resolution ---

def _clean_channel_name(name: str) -> str:
    """Normalize channel name: uppercase, remove common prefixes/suffixes."""
    name = name.upper()
    if name == "REF":
        return name
    for substr in ('EEG', 'LE', 'REF', 'EAR'):
        name = name.replace(substr, '')
    return ''.join(c for c in name if c.isalnum())


_CH_TYPE_MARKERS = {
    'ref': (("A1", "A2"),),
    'ecg': (('ECG', 'EKG'),),
    'extra': (('PHOTIC', 'IBI', 'BURSTS', 'SUPPR'),),
    'eog': (('ROC', 'LOC'),),
}


def _resolve_channel_name(name: str, ch_type: str) -> Optional[str]:
    """Map channel name to standardized form based on type."""
    name = _clean_channel_name(name)
    
    if ch_type == 'eeg':
        return name or None
    if ch_type == 'eog':
        return name
    if ch_type == 'ref':
        if 'A1' in name or 'L' in name:
            return 'A1'
        if 'A2' in name or 'R' in name:
            return 'A2'
        return "REF"
    if ch_type == 'ecg':
        return "EKG" + ''.join(c for c in name if c.isdigit())
    return name


def create_channel_type_dict(raw: mne.io.Raw) -> Dict[str, str]:
    """Create mapping of channel names to their types."""
    type_dict = {}
    for name, ch_type in zip(raw.info['ch_names'], raw.get_channel_types()):
        for override_type, (markers,) in _CH_TYPE_MARKERS.items():
            if any(m in name for m in markers):
                type_dict[name] = override_type
                break
        else:
            type_dict[name] = ch_type
    return type_dict


def heuristic_resolution(old_type_dict: OrderedDict) -> OrderedDict:
    """Resolve channel names to standardized names based on channel type."""
    new_type_dict = OrderedDict()
    
    for old_name, ch_type in old_type_dict.items():
        if ch_type is None:
            new_type_dict[old_name] = None
            continue
        
        new_name = _resolve_channel_name(old_name, ch_type)
        if new_name is None:
            new_type_dict[old_name] = None
        else:
            while new_name in new_type_dict:
                new_name += '-COPY'
            new_type_dict[new_name] = ch_type
    
    assert len(new_type_dict) == len(old_type_dict)
    return new_type_dict


# --- Raw Data Windowing ---

def _create_window_raw(
    data: np.ndarray,
    raw: mne.io.Raw,
    montage: Optional[mne.channels.DigMontage]
) -> mne.io.Raw:
    """Create a RawArray from windowed data, preserving channel info."""
    info = mne.create_info(
        ch_names=raw.info['ch_names'],
        sfreq=raw.info['sfreq'],
        ch_types=raw.get_channel_types()
    )
    window_raw = mne.io.RawArray(data, info, verbose=False)
    if montage is not None:
        window_raw.set_montage(montage)
    return window_raw


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
    shift_samples = int((shift_seconds or window_length) * sfreq)
    montage = raw.get_montage()
    
    windows, time_slices = [], []
    start = 0
    
    while start + window_samples <= raw.n_times:
        data, times = raw[:, start:start + window_samples]
        windows.append(_create_window_raw(data, raw, montage))
        time_slices.append((times[0], times[-1]))
        start += shift_samples
    
    return windows, time_slices


def get_unannotated_raw(
    raw: mne.io.Raw,
    resting_state: List[str] = None
) -> mne.io.Raw:
    """Extract non-annotated segments from raw data."""
    if resting_state is None:
        resting_state = ['T0']
    
    sfreq = raw.info['sfreq']
    annotations = sorted(raw.annotations, key=lambda a: a['onset'])
    segments = []
    last_end = 0
    
    for ann in annotations:
        if ann['description'] in resting_state:
            continue
        onset = int(ann['onset'] * sfreq)
        end = int((ann['onset'] + ann['duration']) * sfreq)
        
        if onset > last_end:
            segments.append(raw[:, last_end:onset][0])
        last_end = end
    
    if last_end < raw.n_times:
        segments.append(raw[:, last_end:raw.n_times][0])
    
    data = np.concatenate(segments, axis=1)
    info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=sfreq, ch_types='eeg')
    return mne.io.RawArray(data, info)


def split_raw_annotations(
    raw: mne.io.Raw,
    labels: List[str],
    tmin: float = -0.5,
    tlen: float = 5.0,
    verbose: bool = True
) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[str]]:
    """Split raw data based on annotations."""
    sfreq = raw.info['sfreq']
    montage = raw.get_montage()
    windows, time_slices, descriptions = [], [], []

    for ann in raw.annotations:
        # MNE annotations: extract onset, duration, description safely
        onset = ann['onset']
        duration = ann['duration']
        description = ann['description']

        if labels and description not in labels:
            continue
        if not labels and description.startswith('BAD'):
            continue

        start = round((onset + tmin) * sfreq)
        end = start + round(tlen * sfreq)

        if start < 0 or end > raw.n_times:
            if verbose:
                print(f'Skipping {description} at {onset:.2f} s')
            continue

        data, times = raw[:, start:end]
        windows.append(_create_window_raw(data, raw, montage))
        time_slices.append((times[0], times[-1]))
        descriptions.append(description)

    return windows, time_slices, descriptions


# --- Montage Creation ---

def make_tuh_montage() -> mne.channels.DigMontage:
    """Create a custom montage for TUH dataset."""
    standard = mne.channels.make_standard_montage('standard_1005')
    postfixed = mne.channels.make_standard_montage('standard_postfixed')
    
    pos = standard.get_positions()
    ch_pos = pos['ch_pos'].copy()
    
    # Add T1/T2 from postfixed montage
    postfixed_pos = postfixed.get_positions()['ch_pos']
    ch_pos['T1'] = postfixed_pos['T1']
    ch_pos['T2'] = postfixed_pos['T2']
    
    # Normalize case (e.g., 'FP1' -> 'Fp1')
    ch_pos = OrderedDict((ch.lower().capitalize(), p) for ch, p in ch_pos.items())
    
    return mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=pos['nasion'],
        lpa=pos['lpa'],
        rpa=pos['rpa'],
        hsp=pos['hsp'],
        hpi=pos['hpi'],
        coord_frame=pos['coord_frame']
    )


def load_montage(montage_source: Optional[str]) -> Optional[mne.channels.DigMontage]:
    """
    Load a montage from various sources.
    
    Parameters
    ----------
    montage_source : str or None
        Either "tuh", a standard MNE montage name, or path to .fif file.
    """
    if montage_source is None:
        return None
    if montage_source == "tuh":
        return make_tuh_montage()
    if montage_source.endswith('.fif') and Path(montage_source).exists():
        return mne.channels.read_dig_fif(montage_source)
    return mne.channels.make_standard_montage(montage_source)


# --- I/O Functions ---

def save_hdf5(
    raws: List[mne.io.Raw],
    src_paths: List[Path],
    times: List[Tuple[float, float]],
    indices: List[int],
    dest_path: Path
) -> None:
    """Save preprocessed data as HDF5 batch file."""
    with h5py.File(dest_path, "w") as f:
        f.attrs["files"] = [p.stem for p in src_paths]
        f.attrs["file_idxs"] = indices
        f.attrs["time_slices"] = times
        f.create_dataset("data", data=np.array([r._data for r in raws]), dtype='float32')


def save_hdf5_with_labels(
    raws: List[mne.io.Raw],
    labels,
    label_descriptions: List[str],
    src_paths: List[Path],
    times: List[Tuple[float, float]],
    indices: List[int],
    dest_path: Path,
    quality_metrics: Optional[List[Dict]] = None
) -> None:
    """
    Save preprocessed windows with labels to HDF5 for downstream tasks.

    HDF5 Structure:
        data: (N, C, T) - EEG windows
        labels: (N,) or (N, T) - integer class labels or float regression targets
        file_idxs: (N,) - source file index
        files: (F,) - source filenames
        time_slices: (N, 2) - (start, end) times
        attrs['descriptions']: label descriptions
        attrs['quality_metrics']: optional quality data

    Parameters
    ----------
    raws : List[mne.io.Raw]
        Preprocessed raw objects
    labels : list of int, float, or list/array
        Class labels (int) or regression targets (float). For multi-target
        regression, each element can be a list/array of floats.
    label_descriptions : List[str]
        String descriptions for each label class
    src_paths : List[Path]
        Source file paths
    times : List[Tuple[float, float]]
        (start_time, end_time) for each window
    indices : List[int]
        Index of source file for each window
    dest_path : Path
        Output HDF5 file path
    quality_metrics : List[Dict], optional
        Quality metrics for each window
    """
    # Convert to numpy arrays
    data = np.array([r._data for r in raws], dtype='float32')
    labels_arr = np.array(labels)
    # Auto-detect dtype: float for regression, int for classification
    if labels_arr.dtype.kind == 'f':
        labels_dtype = np.float32
    else:
        labels_dtype = np.int32
    labels_arr = labels_arr.astype(labels_dtype)
    file_idxs_arr = np.array(indices, dtype=np.int32)
    time_slices_arr = np.array(times, dtype=np.float32)
    files_arr = np.array([p.stem for p in src_paths], dtype=h5py.string_dtype())

    # Save to HDF5
    with h5py.File(dest_path, "w") as f:
        # Main datasets (matching HDF5CombinerDownstream format)
        f.create_dataset("data", data=data, dtype='float32', fletcher32=True)
        f.create_dataset("labels", data=labels_arr, dtype=labels_dtype, fletcher32=True)
        f.create_dataset("file_idxs", data=file_idxs_arr, dtype=np.int32, fletcher32=True)
        f.create_dataset("files", data=files_arr, dtype=h5py.string_dtype())
        f.create_dataset("time_slices", data=time_slices_arr, dtype=np.float32, fletcher32=True)

        # Attributes
        f.attrs['descriptions'] = label_descriptions

        # Optional quality metrics
        if quality_metrics is not None:
            f.attrs['quality_metrics'] = str(quality_metrics)


def _round_to_edf8(x: float) -> float:
    """Round value to fit in 8-character EDF/BDF physical min/max field."""
    sign = "-" if x < 0 else ""
    int_digits = len(str(int(abs(x))))
    decimals = max(0, 7 - len(sign) - int_digits)
    return float(int(x)) if decimals == 0 else round(x, decimals)


def _prepare_edf_data(raw: mne.io.Raw) -> Tuple[np.ndarray, List[str], float, np.ndarray, np.ndarray]:
    """Prepare data and compute physical ranges for EDF/BDF export."""
    picks = mne.pick_types(
        raw.info,
        eeg=True, eog=True, ecg=True, emg=True, misc=True, stim=True,
        seeg=True, dbs=True, ecog=True, resp=True, bio=True, exci=True,
        ias=True, syst=True
    )
    if len(picks) == 0:
        raise RuntimeError("No writable channels were found.")
    
    data_uv = raw.get_data(picks).astype(np.float64) * 1e6
    ch_names = [raw.ch_names[i] for i in picks]
    sfreq = float(raw.info["sfreq"])
    
    # Add 1% margin to physical ranges
    pmins, pmaxs = data_uv.min(axis=1), data_uv.max(axis=1)
    margin = 0.01 * np.maximum(pmaxs - pmins, 1.0)
    pmins -= margin
    pmaxs += margin
    
    pmins = np.array([_round_to_edf8(v) for v in pmins])
    pmaxs = np.array([_round_to_edf8(v) for v in pmaxs])
    
    # Fix invalid ranges
    invalid = ~np.isfinite(pmins) | ~np.isfinite(pmaxs) | (pmaxs <= pmins)
    pmins[invalid], pmaxs[invalid] = -1.0, 1.0
    
    return data_uv, ch_names, sfreq, pmins, pmaxs


def _write_edf_bdf(raw: mne.io.Raw, out_path: str, write_annotations: bool, is_bdf: bool) -> None:
    """Write EDF/BDF file using pyedflib."""
    try:
        import pyedflib
    except ImportError as e:
        raise ImportError("pyedflib required for EDF/BDF export: pip install pyedflib") from e
    
    data_uv, ch_names, sfreq, pmins, pmaxs = _prepare_edf_data(raw)
    
    digital_min, digital_max = (-8388608, 8388607) if is_bdf else (-32768, 32767)
    file_type = pyedflib.FILETYPE_BDFPLUS if is_bdf else pyedflib.FILETYPE_EDFPLUS
    
    sig_headers = [
        {
            "label": name[:16],
            "dimension": "uV",
            "sample_frequency": sfreq,
            "physical_min": float(pmin),
            "physical_max": float(pmax),
            "digital_min": digital_min,
            "digital_max": digital_max
        }
        for name, pmin, pmax in zip(ch_names, pmins, pmaxs)
    ]
    
    md = raw.info.get("meas_date")
    if isinstance(md, (tuple, list)):
        md = md[0]
    startdate = md.replace(tzinfo=None) if md else datetime.now()
    
    f = pyedflib.EdfWriter(str(out_path), n_channels=len(sig_headers), file_type=file_type)
    try:
        f.setSignalHeaders(sig_headers)
        f.setStartdatetime(startdate)
        f.writeSamples(list(data_uv))
        
        if write_annotations and len(raw.annotations) > 0:
            first_time = float(raw.first_time)
            for onset, dur, desc in zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description
            ):
                f.writeAnnotation(float(onset + first_time), float(dur), str(desc))
    finally:
        f.close()


def write_bdf_from_raw(raw: mne.io.Raw, out_path: str, write_annotations: bool = True) -> None:
    """Export MNE Raw object to BDF+ format."""
    _write_edf_bdf(raw, out_path, write_annotations, is_bdf=True)


def write_edf_from_raw(raw: mne.io.Raw, out_path: str, write_annotations: bool = True) -> None:
    """Export MNE Raw object to EDF+ format."""
    _write_edf_bdf(raw, out_path, write_annotations, is_bdf=False)


def write_set_from_raw(raw: mne.io.Raw, out_path: str, write_annotations: bool = True) -> None:
    """Export MNE Raw object to EEGLAB .set format."""
    try:
        import eeglabio  # noqa: F401
        raw.export(out_path, fmt='eeglab', overwrite=True)
    except ImportError as e:
        raise ImportError("eeglabio required for SET export: pip install eeglabio") from e


_FORMAT_WRITERS = {
    "edf": (write_edf_from_raw, ".edf"),
    "bdf": (write_bdf_from_raw, ".bdf"),
    "set": (write_set_from_raw, ".set"),
}


def write_raw_to_file(
    raw: mne.io.Raw,
    out_path: str,
    format: str = "auto",
    write_annotations: bool = True,
    fallback_format: str = "edf"
) -> str:
    """
    Write MNE Raw object to file, preserving format when possible.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to export.
    out_path : str
        Output file path. Extension determines format if format="auto".
    format : str
        Output format: "auto", "edf", "bdf", or "set".
    write_annotations : bool
        Whether to include annotations in the output file.
    fallback_format : str
        Format to use if the requested format is not available.
    
    Returns
    -------
    str
        The actual output path used.
    """
    out_path = Path(out_path)
    
    if format == "auto":
        ext_map = {".edf": "edf", ".bdf": "bdf", ".set": "set"}
        format = ext_map.get(out_path.suffix.lower(), fallback_format)
    
    writer, expected_ext = _FORMAT_WRITERS.get(format, _FORMAT_WRITERS[fallback_format])
    
    if out_path.suffix.lower() != expected_ext:
        out_path = out_path.with_suffix(expected_ext)
    
    try:
        writer(raw, str(out_path), write_annotations)
        return str(out_path)
    except ImportError as e:
        logging.warning(f"Cannot write {format} format: {e}. Using {fallback_format}.")
        writer, expected_ext = _FORMAT_WRITERS[fallback_format]
        out_path = out_path.with_suffix(expected_ext)
        writer(raw, str(out_path), write_annotations)
        return str(out_path)


# --- File Copy Utilities ---

DEFAULT_EXCLUDE_PATTERNS = [
    # Version control
    ".git", ".git/**", ".gitignore", ".gitattributes",
    ".svn", ".svn/**", ".hg", ".hg/**",
    # Python
    "__pycache__", "__pycache__/**", "*.pyc", "*.pyo", "*.pyd",
    ".pytest_cache", ".pytest_cache/**",
    "*.egg-info", "*.egg-info/**",
    ".eggs", ".eggs/**",
    "venv", "venv/**", ".venv", ".venv/**",
    "env", "env/**", ".env",
    # IDE/Editor
    ".idea", ".idea/**", ".vscode", ".vscode/**",
    "*.swp", "*.swo", "*~",
    ".project", ".pydevproject", ".settings", ".settings/**",
    # OS
    ".DS_Store", "Thumbs.db", "desktop.ini",
    # Build/Output
    "build", "build/**", "dist", "dist/**",
    # Temp files
    "*.tmp", "*.temp", "*.bak", "*.log",
    # Lock files
    "*.lock", ".*.lock",
]


def should_exclude_file(file_path: Path, base_path: Path, patterns: List[str]) -> bool:
    """Check if a file should be excluded based on glob patterns."""
    try:
        rel_path = file_path.relative_to(base_path)
    except ValueError:
        rel_path = file_path
    
    rel_str = str(rel_path)
    name = file_path.name
    
    for pattern in patterns:
        if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(name, pattern):
            return True
        # Check parent directories
        if any(fnmatch.fnmatch(str(p), pattern) or fnmatch.fnmatch(p.name, pattern) for p in rel_path.parents):
            return True
    return False


def copy_non_eeg_files(
    src_dir: Path,
    dest_dir: Path,
    eeg_extensions: List[str],
    exclude_patterns: Optional[List[str]] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Copy all non-EEG files from source to destination, preserving structure.
    
    Returns dict with keys: copied, skipped, excluded, errors
    """
    patterns = exclude_patterns if exclude_patterns is not None else DEFAULT_EXCLUDE_PATTERNS.copy()
    eeg_exts = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in eeg_extensions}
    
    stats: Dict[str, Any] = {"copied": [], "skipped": [], "excluded": [], "errors": []}
    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    
    for src_file in src_dir.rglob("*"):
        if not src_file.is_file() or src_file.suffix.lower() in eeg_exts:
            continue
        
        if should_exclude_file(src_file, src_dir, patterns):
            stats["excluded"].append(str(src_file))
            continue
        
        dest_file = dest_dir / src_file.relative_to(src_dir)
        
        if dest_file.exists() and not overwrite:
            stats["skipped"].append(str(dest_file))
            continue
        
        try:
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            stats["copied"].append(str(dest_file))
        except Exception as e:
            stats["errors"].append((str(src_file), str(e)))
    
    return stats
