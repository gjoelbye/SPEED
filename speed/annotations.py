"""
Annotation parsing utilities for downstream EEG datasets.

This module provides parsers for dataset-specific annotation formats:
- EEGMMIDB: Motor imagery event annotations from EDF+ files
- CHBMIT: Seizure annotations from summary text files
"""

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


# Backward compatibility alias
convert_chbmit_bipolar_to_average = convert_chbmit_bipolar_to_monopolar
