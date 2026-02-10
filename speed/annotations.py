"""
Annotation parsing utilities for downstream EEG datasets.

This module provides parsers for dataset-specific annotation formats:
- EEGMMIDB: Motor imagery event annotations from EDF+ files
- CHBMIT: Seizure annotations from summary text files
"""

from pathlib import Path
from typing import List, Tuple, Optional
import mne
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


def convert_chbmit_bipolar_to_average(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Convert CHBMIT bipolar montage to average reference.

    CHBMIT uses bipolar derivations (e.g., 'FP1-F7' represents FP1 referenced
    to F7). This function converts to common average reference for compatibility
    with standard montage interpolation.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with bipolar channel names

    Returns
    -------
    raw_avg : mne.io.Raw
        Raw EEG data with average reference

    Notes
    -----
    This is a placeholder implementation. The actual conversion requires
    reconstructing monopolar signals from bipolar derivations, which may not
    always be possible depending on the bipolar configuration. For CHBMIT,
    an alternative approach is to work directly with the bipolar montage or
    use MNE's montage interpolation capabilities.
    """
    # TODO: Implement proper bipolar to average reference conversion
    # For now, this is a placeholder that sets average reference
    # The actual implementation should:
    # 1. Parse channel names to identify electrode pairs
    # 2. Reconstruct monopolar signals where possible
    # 3. Set average reference on monopolar signals

    # Simple approach: set average reference (may not be mathematically correct)
    raw_copy = raw.copy()
    raw_copy.set_eeg_reference('average', projection=False, verbose=False)

    return raw_copy
