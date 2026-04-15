"""
Convert SEED-VIG vigilance estimation dataset from .mat to EDF format.

SEED-VIG: 23 subjects, 17 EEG channels at 200 Hz, regression task.
Continuous PERCLOS-derived vigilance targets (one float per 8-second window).

The converted EDF files embed regression target values as annotations
in the format 'perclos_<value>' (e.g., 'perclos_0.42').

Usage:
    python scripts/converters/convert_seed_vig.py --input_dir /path/to/SEEDVIG/raw --output_dir /path/to/SEEDVIG/edf
"""

import argparse
import logging
from pathlib import Path

import mne
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SEED-VIG 17-channel layout (BCMI lab)
CHANNELS_17 = [
    'FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8', 'P3', 'PZ', 'P4', 'O1'
]


def load_mat_file(mat_path: Path):
    """Load .mat file."""
    try:
        from scipy.io import loadmat
        return loadmat(str(mat_path), squeeze_me=True)
    except NotImplementedError:
        import h5py
        data = {}
        with h5py.File(str(mat_path), 'r') as f:
            for key in f.keys():
                if not key.startswith('#'):
                    data[key] = np.array(f[key])
        return data


def convert_subject(mat_path: Path, output_dir: Path, sfreq: float = 200.0, window_length: float = 8.0):
    """Convert a single subject's .mat file to EDF with PERCLOS annotations."""
    logger.info(f"Converting {mat_path.name}")
    data = load_mat_file(mat_path)

    eeg_keys = ['EEG', 'data', 'eeg', 'de_LDS', 'X']
    target_keys = ['perclos', 'PERCLOS', 'label', 'labels', 'target', 'targets', 'y']

    eeg_data = None
    for key in eeg_keys:
        if key in data:
            eeg_data = np.array(data[key], dtype=np.float64)
            break

    targets = None
    for key in target_keys:
        if key in data:
            targets = np.array(data[key], dtype=np.float64).flatten()
            break

    if eeg_data is None:
        logger.warning(f"No EEG data in {mat_path.name}. Keys: {list(data.keys())}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if eeg_data.ndim == 2:
        n_channels = min(eeg_data.shape[0], len(CHANNELS_17))
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T  # (samples, channels) → (channels, samples)
            n_channels = min(eeg_data.shape[0], len(CHANNELS_17))

        ch_names = CHANNELS_17[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data[:n_channels] * 1e-6, info, verbose=False)

        # Add PERCLOS annotations as 8-second windows
        if targets is not None:
            for win_idx, target_val in enumerate(targets):
                onset = win_idx * window_length
                if onset + window_length > raw.times[-1]:
                    break
                raw.annotations.append(onset, window_length, f'perclos_{target_val:.6f}')
        else:
            logger.warning(f"No targets found for {mat_path.name}")

    elif eeg_data.ndim == 3:
        # (n_windows, n_channels, n_samples)
        n_windows, n_channels, n_samples = eeg_data.shape
        full_data = np.concatenate([eeg_data[i] for i in range(n_windows)], axis=1)
        ch_names = CHANNELS_17[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(full_data * 1e-6, info, verbose=False)

        if targets is not None:
            trial_duration = n_samples / sfreq
            for win_idx in range(min(n_windows, len(targets))):
                onset = win_idx * trial_duration
                raw.annotations.append(onset, trial_duration, f'perclos_{targets[win_idx]:.6f}')
    else:
        logger.warning(f"Unexpected shape: {eeg_data.shape}")
        return

    edf_path = output_dir / f"{mat_path.stem}.edf"
    mne.export.export_raw(str(edf_path), raw, fmt='edf', overwrite=True, verbose=False)
    logger.info(f"Saved {edf_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert SEED-VIG to EDF')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sfreq', type=float, default=200.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    mat_files = sorted(input_dir.rglob('*.mat'))
    logger.info(f"Found {len(mat_files)} .mat files")

    for mat_path in mat_files:
        try:
            rel_path = mat_path.relative_to(input_dir)
            out_subdir = output_dir / rel_path.parent
            convert_subject(mat_path, out_subdir, args.sfreq)
        except Exception as e:
            logger.error(f"Failed: {mat_path.name}: {e}")


if __name__ == '__main__':
    main()
