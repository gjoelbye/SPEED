"""
Convert BCI Competition 2020 Dataset IV-3 (Imagined Speech) to EDF format.

BCIC2020-IV-3: 15 subjects, 64 EEG channels at 256 Hz, 5-class imagined speech.
Each subject performs imagined speech of 5 different words/phrases.

Usage:
    python scripts/converters/convert_bcic2020_iv_3.py --input_dir /path/to/BCIC2020/raw --output_dir /path/to/BCIC2020/edf
"""

import argparse
import logging
from pathlib import Path

import mne
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 64-channel 10-20 extended layout (typical for BCI2020)
CHANNELS_64 = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'CB1', 'CB2', 'Iz'
]

# 5 imagined speech classes
SPEECH_LABELS = ['hello', 'help_me', 'stop', 'thank_you', 'yes']


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


def convert_subject(mat_path: Path, output_dir: Path, sfreq: float = 256.0):
    """Convert a single subject's data to EDF with speech trial annotations."""
    logger.info(f"Converting {mat_path.name}")
    data = load_mat_file(mat_path)

    eeg_keys = ['eeg', 'data', 'EEG', 'X', 'rawEEG']
    label_keys = ['label', 'labels', 'y', 'Y', 'classlabel']

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
        logger.warning(f"No EEG data in {mat_path.name}. Keys: {list(data.keys())}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if eeg_data.ndim == 3:
        n_trials, n_channels, n_samples = eeg_data.shape
        trial_duration = n_samples / sfreq

        full_data = np.concatenate([eeg_data[i] for i in range(n_trials)], axis=1)
        ch_names = CHANNELS_64[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(full_data * 1e-6, info, verbose=False)

        for trial_idx in range(n_trials):
            onset = trial_idx * trial_duration
            label_val = int(labels[trial_idx]) if labels is not None and trial_idx < len(labels) else 0
            label_str = SPEECH_LABELS[label_val] if 0 <= label_val < len(SPEECH_LABELS) else f'speech_{label_val}'
            raw.annotations.append(onset, trial_duration, label_str)

    elif eeg_data.ndim == 2:
        n_channels = min(eeg_data.shape[0], len(CHANNELS_64))
        ch_names = CHANNELS_64[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data[:n_channels] * 1e-6, info, verbose=False)

        if labels is not None:
            trial_duration = 3.0
            for trial_idx, label_val in enumerate(labels):
                onset = trial_idx * trial_duration
                if onset >= raw.times[-1]:
                    break
                label_str = SPEECH_LABELS[int(label_val)] if 0 <= int(label_val) < len(SPEECH_LABELS) else f'speech_{int(label_val)}'
                raw.annotations.append(onset, trial_duration, label_str)
    else:
        logger.warning(f"Unexpected shape: {eeg_data.shape}")
        return

    edf_path = output_dir / f"{mat_path.stem}.edf"
    mne.export.export_raw(str(edf_path), raw, fmt='edf', overwrite=True, verbose=False)
    logger.info(f"Saved {edf_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert BCIC2020-IV-3 to EDF')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sfreq', type=float, default=256.0)
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
