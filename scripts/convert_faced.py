"""
Convert FACED emotion recognition dataset from .mat to EDF format.

FACED: 123 subjects, 32 EEG channels at 250 Hz, 9-class emotion recognition.
Each .mat file contains EEG data and emotion labels for video stimuli.

The converted EDF files have embedded annotations marking each trial
with the corresponding emotion label.

Usage:
    python scripts/convert_faced.py --input_dir /path/to/FACED/raw --output_dir /path/to/FACED/edf
"""

import argparse
import logging
from pathlib import Path

import mne
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# FACED 32-channel 10-20 extended layout
CHANNELS_32 = [
    'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO3', 'PO4', 'O1', 'Oz', 'O2'
]

# 9 emotion classes in FACED
EMOTION_LABELS = [
    'amusement', 'inspiration', 'joy', 'tenderness',  # positive
    'anger', 'disgust', 'fear', 'sadness',             # negative
    'neutral'
]


def load_mat_file(mat_path: Path):
    """Load a FACED .mat file using scipy or h5py (for v7.3 format)."""
    try:
        from scipy.io import loadmat
        data = loadmat(str(mat_path), squeeze_me=True)
        return data
    except NotImplementedError:
        import h5py
        data = {}
        with h5py.File(str(mat_path), 'r') as f:
            for key in f.keys():
                if not key.startswith('#'):
                    data[key] = np.array(f[key])
        return data


def convert_subject(mat_path: Path, output_dir: Path, sfreq: float = 250.0):
    """Convert a single subject's .mat file to EDF with annotations."""
    logger.info(f"Converting {mat_path.name}")
    data = load_mat_file(mat_path)

    # FACED .mat structure varies — common keys: 'EEG', 'data', 'label', 'labels'
    # Adjust key names based on actual dataset structure
    eeg_keys = ['EEG', 'data', 'eeg', 'Data']
    label_keys = ['label', 'labels', 'Label', 'Labels']

    eeg_data = None
    for key in eeg_keys:
        if key in data:
            eeg_data = np.array(data[key], dtype=np.float64)
            break

    labels = None
    for key in label_keys:
        if key in data:
            labels = np.array(data[key]).flatten()
            break

    if eeg_data is None:
        logger.warning(f"No EEG data found in {mat_path.name}. Keys: {list(data.keys())}")
        return
    if labels is None:
        logger.warning(f"No labels found in {mat_path.name}. Keys: {list(data.keys())}")
        return

    # Ensure shape is (n_channels, n_samples) or handle trials
    if eeg_data.ndim == 3:
        # Shape: (n_trials, n_channels, n_samples)
        n_trials, n_channels, n_samples = eeg_data.shape
        trial_duration = n_samples / sfreq

        # Concatenate trials and create annotations
        full_data = eeg_data.reshape(n_trials * n_channels // n_channels,
                                      n_channels, n_samples)
        # Actually, reshape by concatenating along time axis
        full_data = np.concatenate([eeg_data[i] for i in range(n_trials)], axis=1)

        ch_names = CHANNELS_32[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(full_data * 1e-6, info, verbose=False)  # Convert µV to V

        # Add trial annotations
        for trial_idx in range(n_trials):
            onset = trial_idx * trial_duration
            label_idx = int(labels[trial_idx]) if trial_idx < len(labels) else 0
            label_str = EMOTION_LABELS[label_idx] if label_idx < len(EMOTION_LABELS) else f'emotion_{label_idx}'
            raw.annotations.append(onset, trial_duration, label_str)

    elif eeg_data.ndim == 2:
        # Shape: (n_channels, n_samples) — continuous recording
        n_channels = min(eeg_data.shape[0], len(CHANNELS_32))
        ch_names = CHANNELS_32[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data[:n_channels] * 1e-6, info, verbose=False)

        # Add per-trial annotations based on labels and assumed trial duration (10s)
        trial_duration = 10.0
        for trial_idx, label_val in enumerate(labels):
            onset = trial_idx * trial_duration
            if onset >= raw.times[-1]:
                break
            label_idx = int(label_val)
            label_str = EMOTION_LABELS[label_idx] if label_idx < len(EMOTION_LABELS) else f'emotion_{label_idx}'
            raw.annotations.append(onset, trial_duration, label_str)
    else:
        logger.warning(f"Unexpected data shape: {eeg_data.shape}")
        return

    # Save as EDF
    output_dir.mkdir(parents=True, exist_ok=True)
    edf_path = output_dir / f"{mat_path.stem}.edf"
    mne.export.export_raw(str(edf_path), raw, fmt='edf', overwrite=True, verbose=False)
    logger.info(f"Saved {edf_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert FACED .mat to EDF')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with .mat files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for EDF files')
    parser.add_argument('--sfreq', type=float, default=250.0, help='Sampling frequency')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    mat_files = sorted(input_dir.rglob('*.mat'))
    logger.info(f"Found {len(mat_files)} .mat files in {input_dir}")

    for mat_path in mat_files:
        try:
            # Preserve relative directory structure
            rel_path = mat_path.relative_to(input_dir)
            out_subdir = output_dir / rel_path.parent
            convert_subject(mat_path, out_subdir, args.sfreq)
        except Exception as e:
            logger.error(f"Failed to convert {mat_path.name}: {e}")


if __name__ == '__main__':
    main()
