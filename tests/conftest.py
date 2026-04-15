"""Shared test fixtures for SPEED pipeline tests."""

import tempfile
from pathlib import Path

import h5py
import mne
import numpy as np
import pytest


@pytest.fixture
def synthetic_raw():
    """Create a 30-second, 19-channel MNE Raw object with known frequency content.

    Contains sinusoidal components at 10 Hz, 50 Hz, and 90 Hz plus white noise.
    Sampling rate: 256 Hz. Channel names match the standard TUH 19-channel montage.
    """
    sfreq = 256.0
    duration = 30.0
    n_samples = int(sfreq * duration)
    n_channels = 19
    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "T5", "P3", "Pz", "P4", "T6",
        "O1", "O2",
    ]

    rng = np.random.RandomState(42)
    t = np.arange(n_samples) / sfreq

    # Build signal: 10 Hz + 50 Hz + 90 Hz + noise
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        data[ch] = (
            1e-5 * np.sin(2 * np.pi * 10 * t)
            + 0.5e-5 * np.sin(2 * np.pi * 50 * t)
            + 0.3e-5 * np.sin(2 * np.pi * 90 * t)
            + 0.1e-5 * rng.randn(n_samples)
        )

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
    return raw


@pytest.fixture
def synthetic_raw_short():
    """Create a 5-second, 19-channel MNE Raw object for fast tests."""
    sfreq = 256.0
    duration = 5.0
    n_samples = int(sfreq * duration)
    n_channels = 19
    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "T5", "P3", "Pz", "P4", "T6",
        "O1", "O2",
    ]

    rng = np.random.RandomState(0)
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        data[ch] = 1e-5 * np.sin(2 * np.pi * 10 * t) + 0.1e-5 * rng.randn(n_samples)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
    return raw


@pytest.fixture
def tmp_hdf5(tmp_path):
    """Create a temporary HDF5 file with mock data/labels/metadata.

    Simulates 5 subjects, 4 samples each (20 total).
    """
    hdf5_path = tmp_path / "test_data.hdf5"
    n_subjects = 5
    samples_per_subject = 4
    n_total = n_subjects * samples_per_subject
    n_channels = 19
    n_times = 256 * 5  # 5 seconds at 256 Hz

    rng = np.random.RandomState(42)
    data = rng.randn(n_total, n_channels, n_times).astype(np.float32)
    labels = np.array([i % 3 for i in range(n_total)], dtype=np.int32)

    # Source filenames: S001R01, S001R02, ..., S005R04
    source_files = []
    file_idxs = np.zeros(n_total, dtype=np.int32)
    for subj in range(n_subjects):
        for rec in range(samples_per_subject):
            fname = f"S{subj + 1:03d}R{rec + 1:02d}"
            source_files.append(fname)
            file_idxs[subj * samples_per_subject + rec] = subj * samples_per_subject + rec

    time_slices = np.array(
        [(i * 5.0, (i + 1) * 5.0) for i in range(n_total)], dtype=np.float32
    )

    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("labels", data=labels)
        f.create_dataset("file_idxs", data=file_idxs)
        f.create_dataset(
            "files",
            data=np.array(source_files, dtype=h5py.string_dtype()),
        )
        f.create_dataset("time_slices", data=time_slices)
        f.attrs["descriptions"] = ["class_0", "class_1", "class_2"]

    return hdf5_path
