"""
Annotation parsing utilities for downstream EEG datasets.

This module provides parsers for dataset-specific annotation formats:
- EEGMMIDB: Motor imagery event annotations from EDF+ files
- CHBMIT: Seizure annotations from summary text files
- EEGMAT: Mental arithmetic task labels from filenames
- HMC: Sleep stage annotations from companion scoring files
- ISRUC: Sleep stage annotations from companion text files
- Mumtaz2016: Depression labels from subject metadata
- TUEV: TUH EEG event annotations from .tse files
- TUAB: TUH abnormal/normal labels from directory structure
- BCIC-IV-2a: Motor imagery events from GDF event markers
"""

import csv
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Tuple, Optional
import mne
import numpy as np
import re


def parse_eegmmidb_events(
    edf_path: Path,
    event_labels: Optional[List[str]] = None
) -> List[Tuple[float, float, str]]:
    """
    Parse EEGMMIDB motor imagery event annotations from EDF+ file.

    EEGMMIDB uses EDF+ format with annotations embedded in the file.
    Events are labeled as T0 (rest), T1 (left fist), T2 (right fist),
    T3 (both feet), T4 (both fists).

    Parameters
    ----------
    edf_path : Path
        Path to the .edf file
    event_labels : List[str], optional
        List of event labels to extract (e.g., ['T1', 'T2', 'T3', 'T4']).
        If None, extracts all events.

    Returns
    -------
    events : List[Tuple[float, float, str]]
        List of (onset_seconds, duration_seconds, event_label) tuples

    Examples
    --------
    >>> events = parse_eegmmidb_events(Path('S001R03.edf'), ['T1', 'T2'])
    >>> print(events[0])
    (4.5, 4.1, 'T1')
    """
    # Read EDF file with annotations
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)

    # Extract annotations
    events = []
    for ann in raw.annotations:
        onset, duration, description = ann['onset'], ann['duration'], ann['description']

        # Filter by labels if specified
        if event_labels is None or description in event_labels:
            events.append((float(onset), float(duration), description))

    return events


def parse_chbmit_summary(
    summary_path: Path,
    edf_filename: str
) -> List[Tuple[float, float]]:
    """
    Parse CHBMIT seizure annotations from summary text file.

    CHBMIT summary files contain seizure onset and end times in seconds
    for each EDF recording in a subject's directory.

    Parameters
    ----------
    summary_path : Path
        Path to the subject summary file (e.g., 'chb01-summary.txt')
    edf_filename : str
        Name of the specific EDF file to extract seizures for
        (e.g., 'chb01_03.edf')

    Returns
    -------
    seizures : List[Tuple[float, float]]
        List of (onset_seconds, duration_seconds) tuples for seizures

    Examples
    --------
    >>> seizures = parse_chbmit_summary(Path('chb01-summary.txt'), 'chb01_03.edf')
    >>> print(seizures[0])
    (2996.0, 40.0)
    """
    with open(summary_path, 'r') as f:
        content = f.read()

    # Find the section for this specific file
    # Pattern: "File Name: chb01_03.edf" followed by seizure info
    file_pattern = rf'File Name: {re.escape(edf_filename)}.*?(?=File Name:|$)'
    file_match = re.search(file_pattern, content, re.DOTALL)

    if not file_match:
        return []  # File not found in summary

    file_section = file_match.group(0)

    # Extract number of seizures
    num_seizures_match = re.search(r'Number of Seizures in File:\s*(\d+)', file_section)
    if not num_seizures_match or int(num_seizures_match.group(1)) == 0:
        return []  # No seizures in this file

    # Extract seizure start and end times
    seizures = []
    start_times = re.findall(r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds', file_section)
    end_times = re.findall(r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds', file_section)

    for start, end in zip(start_times, end_times):
        onset = float(start)
        duration = float(end) - onset
        seizures.append((onset, duration))

    return seizures


def generate_non_seizure_windows(
    recording_duration: float,
    seizure_intervals: List[Tuple[float, float]],
    window_length: float = 60.0,
    stride: float = 60.0,
    margin: float = 60.0,
    seed: Optional[int] = None
) -> List[Tuple[float, str]]:
    """
    Generate non-seizure window annotations via sliding window.

    Extracts non-overlapping windows from inter-ictal periods, excluding
    seizure times plus a safety margin before and after each seizure.

    Parameters
    ----------
    recording_duration : float
        Total duration of the recording in seconds
    seizure_intervals : List[Tuple[float, float]]
        List of (onset, duration) tuples for seizures in the recording
    window_length : float, default=60.0
        Length of each window in seconds
    stride : float, default=60.0
        Stride between windows in seconds (60.0 = non-overlapping)
    margin : float, default=60.0
        Safety margin to exclude before/after seizures in seconds
    seed : int, optional
        Random seed (for future random sampling strategies)

    Returns
    -------
    windows : List[Tuple[float, str]]
        List of (onset_seconds, 'non-seizure') tuples

    Examples
    --------
    >>> windows = generate_non_seizure_windows(
    ...     recording_duration=3600,
    ...     seizure_intervals=[(2996, 40)],
    ...     window_length=60,
    ...     stride=60,
    ...     margin=60
    ... )
    >>> len(windows)
    48  # Approximately (3600 - 40 - 2*60) / 60 = 56 windows
    """
    # Convert seizure intervals to excluded periods (with margin)
    excluded_periods = []
    for onset, duration in seizure_intervals:
        start = max(0, onset - margin)
        end = min(recording_duration, onset + duration + margin)
        excluded_periods.append((start, end))

    # Merge overlapping excluded periods
    if excluded_periods:
        excluded_periods.sort()
        merged = [excluded_periods[0]]
        for start, end in excluded_periods[1:]:
            if start <= merged[-1][1]:
                # Overlapping, merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        excluded_periods = merged

    # Generate windows from valid inter-ictal periods
    windows = []
    current_time = 0.0

    while current_time + window_length <= recording_duration:
        # Check if window overlaps with any excluded period
        window_end = current_time + window_length
        is_valid = True

        for excl_start, excl_end in excluded_periods:
            # Check for overlap
            if not (window_end <= excl_start or current_time >= excl_end):
                is_valid = False
                # Skip to after this excluded period
                current_time = excl_end
                break

        if is_valid:
            windows.append((current_time, 'non-seizure'))
            current_time += stride
        # else: current_time already updated to skip excluded period

    return windows


def _parse_bipolar_name(ch_name: str) -> Optional[Tuple[str, str]]:
    """
    Parse a bipolar channel name into its electrode pair.

    Parameters
    ----------
    ch_name : str
        Channel name (e.g., 'FP1-F7', 'EEG FP1-F7')

    Returns
    -------
    pair : tuple of str or None
        (electrode_a, electrode_b) in uppercase, or None if not bipolar
    """
    name = ch_name.strip()
    # Remove common EEG prefixes
    for prefix in ('EEG ', 'EEG-'):
        if name.upper().startswith(prefix):
            name = name[len(prefix):]

    parts = name.split('-')
    if len(parts) != 2:
        return None

    a = parts[0].strip().upper()
    b = parts[1].strip().upper()

    if not a or not b:
        return None

    return (a, b)


def convert_chbmit_bipolar_to_monopolar(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Convert CHBMIT bipolar montage to monopolar (average reference) signals.

    Each bipolar channel 'A-B' records V_A - V_B. This function recovers the
    monopolar electrode voltages by solving the linear system y = M*x via
    pseudoinverse, where M is the bipolar mixing matrix.

    The pseudoinverse yields the minimum-norm solution, which is equivalent
    to average reference (sum of all monopolar signals ≈ 0).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with bipolar channel names (e.g., 'FP1-F7').

    Returns
    -------
    raw_mono : mne.io.Raw
        Raw EEG data with monopolar channel names in average reference.

    Notes
    -----
    If the bipolar channel graph has disconnected components, the relative
    scaling between components cannot be determined from the bipolar data alone.
    For standard CHBMIT (23 channels), FZ/CZ/PZ form a separate component
    from the other 18 electrodes. The pseudoinverse gives the minimum-norm
    solution, but the inter-component scaling is approximate. This is typically
    acceptable for classification tasks, especially after high-pass filtering.
    """
    # 1. Parse bipolar channel names
    bipolar_channels = []
    bipolar_indices = []
    for i, ch_name in enumerate(raw.ch_names):
        parsed = _parse_bipolar_name(ch_name)
        if parsed is not None:
            bipolar_channels.append(parsed)
            bipolar_indices.append(i)

    if len(bipolar_channels) == 0:
        # No bipolar channels — already monopolar (e.g., chb12).
        # Return unchanged so the rest of the pipeline can proceed.
        logging.info(
            "No bipolar channels found; assuming already monopolar. "
            f"Channel names: {raw.ch_names[:5]}..."
        )
        return raw

    # 2. Identify unique electrodes (preserve discovery order)
    seen = set()
    unique_electrodes = []
    for a, b in bipolar_channels:
        for electrode in (a, b):
            if electrode not in seen:
                seen.add(electrode)
                unique_electrodes.append(electrode)

    n_bipolar = len(bipolar_channels)
    n_mono = len(unique_electrodes)
    electrode_to_idx = {e: i for i, e in enumerate(unique_electrodes)}

    # 3. Build bipolar mixing matrix M: shape (n_bipolar, n_mono)
    #    Row i has +1 at electrode_a's column, -1 at electrode_b's column
    M = np.zeros((n_bipolar, n_mono))
    for row, (a, b) in enumerate(bipolar_channels):
        M[row, electrode_to_idx[a]] = 1.0
        M[row, electrode_to_idx[b]] = -1.0

    # 4. Check graph connectivity — disconnected components introduce
    #    reconstruction ambiguity (relative scaling between components is lost)
    adj = defaultdict(set)
    for a, b in bipolar_channels:
        adj[a].add(b)
        adj[b].add(a)

    visited = set()
    components = []
    for node in unique_electrodes:
        if node not in visited:
            comp = []
            queue = deque([node])
            while queue:
                n = queue.popleft()
                if n not in visited:
                    visited.add(n)
                    comp.append(n)
                    queue.extend(nb for nb in adj[n] if nb not in visited)
            components.append(comp)

    if len(components) > 1:
        comp_strs = [', '.join(c) for c in components]
        logging.warning(
            f"Bipolar graph has {len(components)} disconnected components. "
            f"Relative scaling between components is ambiguous. "
            f"Components: [{'] / ['.join(comp_strs)}]"
        )

    # 5. Compute pseudoinverse and recover monopolar signals
    M_pinv = np.linalg.pinv(M)  # shape (n_mono, n_bipolar)
    bipolar_data = raw.get_data(picks=bipolar_indices)  # (n_bipolar, n_times)
    monopolar_data = M_pinv @ bipolar_data               # (n_mono, n_times)

    # 6. Create new Raw object with monopolar channels
    info = mne.create_info(
        ch_names=unique_electrodes,
        sfreq=raw.info['sfreq'],
        ch_types='eeg'
    )
    raw_mono = mne.io.RawArray(monopolar_data, info, verbose=False)

    # 7. Preserve annotations
    if raw.annotations is not None and len(raw.annotations) > 0:
        raw_mono.set_annotations(raw.annotations)

    return raw_mono


def generate_tiled_annotations(
    recording_duration: float,
    window_length: float,
    label: str,
    stride: Optional[float] = None
) -> List[Tuple[float, float, str]]:
    """
    Generate non-overlapping fixed-length annotations spanning a recording.

    Used for datasets where the entire file has one label and must be sliced
    into fixed-length windows (e.g., MentalArithmetic, Mumtaz2016, TUAB).

    Parameters
    ----------
    recording_duration : float
        Total duration of the recording in seconds.
    window_length : float
        Length of each window in seconds.
    label : str
        Annotation label for all windows.
    stride : float, optional
        Stride between windows in seconds. Defaults to window_length
        (non-overlapping).

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset_seconds, duration_seconds, label) tuples.
    """
    if stride is None:
        stride = window_length

    annotations = []
    onset = 0.0
    while onset + window_length <= recording_duration:
        annotations.append((onset, window_length, label))
        onset += stride

    return annotations


# =========================================================================
# MentalArithmetic (EEGMAT) Annotations
# =========================================================================

def parse_eegmat_annotations(
    edf_path: Path,
    window_length: float = 5.0,
    recording_duration: Optional[float] = None
) -> List[Tuple[float, float, str]]:
    """
    Generate annotations for Mental Arithmetic dataset from filename.

    Labels are determined by filename convention:
    - Subject##_1.edf → "baseline" (resting, no mental stress)
    - Subject##_2.edf → "arithmetic" (during mental arithmetic task)

    Parameters
    ----------
    edf_path : Path
        Path to the .edf file.
    window_length : float
        Length of each window in seconds (default 5.0).
    recording_duration : float, optional
        Duration of the recording in seconds. If None, reads from file.

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset, duration, label) tuples.
    """
    stem = edf_path.stem  # e.g., "Subject00_1" or "Subject00_2"
    if stem.endswith('_1'):
        label = 'baseline'
    elif stem.endswith('_2'):
        label = 'arithmetic'
    else:
        logging.warning(f"Cannot determine label from filename: {stem}")
        return []

    if recording_duration is None:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        recording_duration = raw.times[-1]

    return generate_tiled_annotations(recording_duration, window_length, label)


# =========================================================================
# HMC Sleep Staging Annotations
# =========================================================================

def parse_hmc_sleepscoring(
    src_path: Path
) -> List[Tuple[float, float, str]]:
    """
    Parse HMC sleep scoring from companion *_sleepscoring.txt file.

    The scoring file is CSV with columns:
    Date, Time, Recording onset, Duration, Annotation, Linked channel

    Only sleep stage annotations are extracted (Sleep stage W/N1/N2/N3/R).

    Parameters
    ----------
    src_path : Path
        Path to the EEG .edf file. The scoring file is found by replacing
        the extension with '_sleepscoring.txt'.

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset_seconds, duration_seconds, stage_label) tuples.
    """
    scoring_path = src_path.parent / (src_path.stem + '_sleepscoring.txt')
    if not scoring_path.exists():
        logging.warning(f"Sleep scoring file not found: {scoring_path}")
        return []

    sleep_stages = {'Sleep stage W', 'Sleep stage N1', 'Sleep stage N2',
                    'Sleep stage N3', 'Sleep stage R'}
    annotations = []

    with open(scoring_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        for row in reader:
            if len(row) < 5:
                continue
            try:
                onset = float(row[2].strip())
                duration = float(row[3].strip())
            except (ValueError, IndexError):
                continue
            annotation = row[4].strip()
            if annotation in sleep_stages:
                annotations.append((onset, duration, annotation))

    return annotations


# =========================================================================
# ISRUC Sleep Staging Annotations
# =========================================================================

def parse_isruc_annotations(
    src_path: Path
) -> List[Tuple[float, float, str]]:
    """
    Parse ISRUC sleep stage annotations from companion text file.

    ISRUC annotation files contain one integer label per line, each
    corresponding to a 30-second epoch. Labels: 0=W, 1=N1, 2=N2, 3=N3, 5=REM.

    Parameters
    ----------
    src_path : Path
        Path to the EEG file. The annotation file is searched in the same
        directory with common naming patterns.

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset_seconds, 30.0, stage_label) tuples.
    """
    # ISRUC annotation files follow various naming patterns
    parent = src_path.parent
    possible_names = [
        src_path.stem + '_1.txt',
        src_path.stem + '_2.txt',
        src_path.stem + '.txt',
        'Annotations.txt',
    ]

    annotation_path = None
    for name in possible_names:
        candidate = parent / name
        if candidate.exists():
            annotation_path = candidate
            break

    if annotation_path is None:
        logging.warning(f"ISRUC annotation file not found for: {src_path}")
        return []

    label_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'R'}
    annotations = []

    with open(annotation_path, 'r') as f:
        epoch_count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                stage_int = int(line)
            except ValueError:
                epoch_count += 1
                continue
            label = label_map.get(stage_int)
            if label is not None:
                onset = epoch_count * 30.0
                annotations.append((onset, 30.0, label))
            epoch_count += 1

    return annotations


# =========================================================================
# Mumtaz2016 Depression Detection Annotations
# =========================================================================

def parse_mumtaz2016_annotations(
    src_path: Path,
    window_length: float = 5.0,
    recording_duration: Optional[float] = None
) -> List[Tuple[float, float, str]]:
    """
    Generate annotations for Mumtaz2016 depression detection dataset.

    Labels are determined from the filename or directory structure:
    - Filenames starting with 'MDD' → depressed
    - Filenames starting with 'H ' → healthy
    - Directory paths containing 'mdd' or 'hc'/'h' also work as fallback

    Parameters
    ----------
    src_path : Path
        Path to the .edf file.
    window_length : float
        Length of each window in seconds (default 5.0).
    recording_duration : float, optional
        Duration of the recording in seconds. If None, reads from file.

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset, duration, label) tuples.
    """
    # Determine label — first try filename, then directory
    filename = src_path.name
    if filename.upper().startswith('MDD'):
        label = 'depressed'
    elif filename.upper().startswith('H ') or filename.upper().startswith('HC'):
        label = 'healthy'
    else:
        # Fallback: check directory path
        path_lower = str(src_path).lower()
        if '/mdd/' in path_lower or 'mdd' in src_path.parent.name.lower():
            label = 'depressed'
        elif '/h/' in path_lower or '/hc/' in path_lower or 'healthy' in path_lower:
            label = 'healthy'
        else:
            # Final fallback: files without MDD prefix in Figshare 3385168
            # are healthy controls (e.g., "S1 EC.edf")
            if re.match(r'^S\d+', filename):
                label = 'healthy'
            else:
                logging.warning(f"Cannot determine MDD/HC label from: {src_path}")
                return []

    if recording_duration is None:
        raw = mne.io.read_raw_edf(str(src_path), preload=False, verbose=False)
        recording_duration = raw.times[-1]

    return generate_tiled_annotations(recording_duration, window_length, label)


# =========================================================================
# TUEV Event Detection Annotations
# =========================================================================

def parse_tuev_annotations(
    src_path: Path
) -> List[Tuple[float, float, str]]:
    """
    Parse TUH EEG Events (TUEV) annotations from .tse files.

    TSE files are tab-separated with format:
        start_time  end_time  label  probability

    Parameters
    ----------
    src_path : Path
        Path to the .edf file. The .tse file is found by replacing
        the extension.

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset_seconds, duration_seconds, event_label) tuples.
    """
    tse_path = src_path.with_suffix('.tse')
    if not tse_path.exists():
        # Try alternate naming: .tse_bi
        tse_path = src_path.with_suffix('.tse_bi')
    if not tse_path.exists():
        logging.warning(f"TSE annotation file not found for: {src_path}")
        return []

    annotations = []
    with open(tse_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('version'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[2]
                    duration = end - start
                    if duration > 0:
                        annotations.append((start, duration, label))
                except ValueError:
                    continue

    return annotations


# =========================================================================
# TUAB Abnormality Detection Annotations
# =========================================================================

def parse_tuab_annotations(
    src_path: Path,
    window_length: float = 10.0,
    recording_duration: Optional[float] = None
) -> List[Tuple[float, float, str]]:
    """
    Generate annotations for TUH Abnormal EEG (TUAB) dataset.

    Labels are determined from the directory structure:
    - Files under 'abnormal/' directory → 'abnormal'
    - Files under 'normal/' directory → 'normal'

    Parameters
    ----------
    src_path : Path
        Path to the .edf file.
    window_length : float
        Length of each window in seconds (default 10.0).
    recording_duration : float, optional
        Duration of the recording in seconds. If None, reads from file.

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset, duration, label) tuples.
    """
    path_parts = [p.lower() for p in src_path.parts]
    if 'abnormal' in path_parts:
        label = 'abnormal'
    elif 'normal' in path_parts:
        label = 'normal'
    else:
        logging.warning(f"Cannot determine normal/abnormal label from path: {src_path}")
        return []

    if recording_duration is None:
        raw = mne.io.read_raw_edf(str(src_path), preload=False, verbose=False)
        recording_duration = raw.times[-1]

    return generate_tiled_annotations(recording_duration, window_length, label)


# =========================================================================
# BCIC-IV-2a Motor Imagery Annotations
# =========================================================================

def parse_bcic_iv_2a_events(
    raw: mne.io.Raw
) -> List[Tuple[float, float, str]]:
    """
    Parse BCI Competition IV Dataset 2a motor imagery events from GDF events.

    GDF event codes for BCIC-IV-2a:
    - 769 (0x0301): Left hand
    - 770 (0x0302): Right hand
    - 771 (0x0303): Both feet
    - 772 (0x0304): Tongue

    Parameters
    ----------
    raw : mne.io.Raw
        Raw GDF data with embedded events.

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset_seconds, duration_seconds, label) tuples.
    """
    event_id_map = {
        769: 'left_hand',
        770: 'right_hand',
        771: 'both_feet',
        772: 'tongue',
    }

    events, ann_map = mne.events_from_annotations(raw, verbose=False)
    # Invert: event_id → description string
    id_to_desc = {v: k for k, v in ann_map.items()}

    annotations = []
    sfreq = raw.info['sfreq']
    for event in events:
        sample, _, event_id = event
        desc = id_to_desc.get(event_id, '')
        # Try to match by event code in the description or by numeric ID
        try:
            code = int(desc)
        except ValueError:
            code = event_id

        label = event_id_map.get(code)
        if label is not None:
            onset = sample / sfreq
            annotations.append((onset, 0.0, label))

    return annotations


# =========================================================================
# SHU-MI Motor Imagery Annotations
# =========================================================================

def parse_shu_mi_events(
    src_path: Path,
    sfreq: float = 250.0
) -> List[Tuple[float, float, str]]:
    """
    Parse SHU-MI motor imagery events from companion TSV files.

    Event TSV format (BIDS-style):
        onset  duration  trial_type  response_time  sample  value
        1.00   1000.00   left        n/a            1.00    1.00

    Onset and duration are in SAMPLES (not seconds).

    Parameters
    ----------
    src_path : Path
        Path to the .edf file. The events TSV is found by replacing
        '_eeg.edf' with '_events.tsv'.
    sfreq : float
        Sampling frequency (default 250 Hz) for sample→seconds conversion.

    Returns
    -------
    annotations : List[Tuple[float, float, str]]
        List of (onset_seconds, duration_seconds, trial_type) tuples.
    """
    # Find companion events TSV
    events_path = Path(str(src_path).replace('_eeg.edf', '_events.tsv'))
    if not events_path.exists():
        logging.warning(f"Events TSV not found: {events_path}")
        return []

    annotations = []
    with open(events_path, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            onset_samples = float(parts[0].strip())
            duration_samples = float(parts[1].strip())
            trial_type = parts[2].strip()
            onset_sec = onset_samples / sfreq
            duration_sec = duration_samples / sfreq
            annotations.append((onset_sec, duration_sec, trial_type))

    return annotations


# =========================================================================
# Siena Scalp EEG Seizure Detection Annotations
# =========================================================================

def _parse_clock_time(time_str: str) -> Optional[float]:
    """Parse HH.MM.SS or HH:MM:SS to seconds since midnight. Returns None on failure."""
    time_str = time_str.strip().replace(':', '.')
    parts = time_str.split('.')
    if len(parts) < 3:
        return None
    try:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    except ValueError:
        return None


def parse_siena_seizures(
    src_path: Path,
) -> List[Tuple[float, float]]:
    """
    Parse Siena Scalp EEG seizure annotations from Seizures-list-PNxx.txt.

    Each seizure block contains the file name, registration start/end,
    and seizure start/end in clock time. We match by file name and compute
    onset relative to registration start.

    Parameters
    ----------
    src_path : Path
        Path to the .edf file (e.g., PN00-1.edf).

    Returns
    -------
    seizures : List[Tuple[float, float]]
        List of (onset_seconds, duration_seconds) relative to recording start.
    """
    subject_dir = src_path.parent
    subject_id = subject_dir.name
    seizure_file = subject_dir / f"Seizures-list-{subject_id}.txt"

    if not seizure_file.exists():
        logging.warning(f"Seizure list not found: {seizure_file}")
        return []

    edf_name = src_path.name

    with open(seizure_file, 'r') as f:
        content = f.read()

    # Split into per-seizure blocks (each starts with "Seizure n X")
    blocks = re.split(r'Seizure\s+n\s*\d+', content)

    edf_name = src_path.name
    edf_stem = src_path.stem  # e.g., "PN00-1"

    seizures = []
    for block in blocks[1:]:  # Skip header
        # Extract file name for this seizure block
        file_match = re.search(r'File\s*name:\s*(\S+)', block)
        if file_match:
            block_file = file_match.group(1).replace('.edf', '')
            # Match: exact or prefix (PN01 matches PN01-1)
            if block_file != edf_stem and not edf_stem.startswith(block_file):
                continue
        else:
            # No file ref in this block — check the header block
            header_file = re.search(r'File\s*name:\s*(\S+)', blocks[0])
            if header_file:
                hf = header_file.group(1).replace('.edf', '')
                if hf != edf_stem and not edf_stem.startswith(hf):
                    continue

        # Parse registration start time (look in this block first, then header)
        reg_match = re.search(r'Registration\s+start\s+time:\s*([\d.:]+)', block)
        if not reg_match:
            reg_match = re.search(r'Registration\s+start\s+time:\s*([\d.:]+)', blocks[0])
        if not reg_match:
            continue
        reg_start = _parse_clock_time(reg_match.group(1))
        if reg_start is None:
            continue

        # Parse seizure start/end (handle "Seizure start time:" and "Start time:")
        sz_start = re.search(r'[Ss]eizure\s+start\s+time:\s*([\d.:]+)', block)
        if not sz_start:
            sz_start = re.search(r'[Ss]tart\s+time:\s*([\d.:]+)', block)
        sz_end = re.search(r'[Ss]eizure\s+end\s+time:\s*([\d.:]+)', block)
        if not sz_end:
            sz_end = re.search(r'[Ee]nd\s+time:\s*([\d.:]+)', block)
        if not sz_start or not sz_end:
            continue

        seizure_start = _parse_clock_time(sz_start.group(1))
        seizure_end = _parse_clock_time(sz_end.group(1))
        if seizure_start is None or seizure_end is None:
            continue

        onset = seizure_start - reg_start
        if onset < 0:
            onset += 86400
        duration = seizure_end - seizure_start
        if duration < 0:
            duration += 86400

        # Sanity check: real seizures are < 10 minutes
        if duration > 600:
            logging.warning(
                f"Siena: Skipping implausible seizure in {edf_name} "
                f"(onset={onset:.0f}s, duration={duration:.0f}s). Likely annotation typo."
            )
            continue

        seizures.append((onset, duration))

    return seizures


# Backward compatibility alias
convert_chbmit_bipolar_to_average = convert_chbmit_bipolar_to_monopolar
