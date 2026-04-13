"""
Convert SEED-V emotion recognition dataset from .mat to EDF format.

SEED-V: 16 subjects, 62 EEG channels at 1000 Hz, 5-class emotion recognition.
Each session has multiple trials with video stimuli. Labels per trial.

The converted EDF files have embedded annotations marking each 1-second
segment with the corresponding emotion label.

Usage:
    python scripts/convert_seedv.py --input_dir /path/to/SEEDV/raw --output_dir /path/to/SEEDV/edf
"""

import argparse
import logging
from pathlib import Path

import mne
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SEED-V 62-channel layout (BCMI lab NeuroScan ESI system)
# Standard 10-20 extended positions
CHANNELS_62 = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1',
    'Oz', 'O2', 'CB2'
]

# SEED-V emotion labels
EMOTION_LABELS = ['happy', 'sad', 'disgust', 'neutral', 'fear']


def load_mat_file(mat_path: Path):
    """Load a SEED-V .mat file."""
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


def convert_session(mat_path: Path, label_path: Path, output_dir: Path, sfreq: float = 1000.0):
    """Convert a single session's .mat file to EDF with annotations."""
    logger.info(f"Converting {mat_path.name}")
    data = load_mat_file(mat_path)

    # Load labels
    if label_path.exists():
        label_data = load_mat_file(label_path)
        labels_key = [k for k in label_data.keys() if not k.startswith('_')]
        labels = np.array(label_data[labels_key[0]]).flatten() if labels_key else None
    else:
        labels = None

    # SEED-V stores trials as separate variables (e.g., 'eeg_1', 'eeg_2', ...)
    # or as a single matrix. Detect format.
    trial_keys = sorted([k for k in data.keys() if not k.startswith('_') and k.startswith(('eeg', 'de_', 'csd'))])

    if not trial_keys:
        # Try numeric keys or 'data' key
        trial_keys = sorted([k for k in data.keys() if not k.startswith('_')])

    output_dir.mkdir(parents=True, exist_ok=True)

    for trial_idx, key in enumerate(trial_keys):
        trial_data = np.array(data[key], dtype=np.float64)
        if trial_data.ndim != 2:
            continue

        n_channels = min(trial_data.shape[0], len(CHANNELS_62))
        if trial_data.shape[0] < trial_data.shape[1]:
            # (channels, samples)
            eeg = trial_data[:n_channels]
        else:
            # (samples, channels) — transpose
            eeg = trial_data[:, :n_channels].T

        ch_names = CHANNELS_62[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg * 1e-6, info, verbose=False)

        # Add 1-second window annotations with trial label
        if labels is not None and trial_idx < len(labels):
            label_idx = int(labels[trial_idx])
            label_str = EMOTION_LABELS[label_idx] if 0 <= label_idx < len(EMOTION_LABELS) else f'emotion_{label_idx}'
        else:
            label_str = f'unknown_{trial_idx}'

        # Tile 1-second annotations over the trial
        for onset in np.arange(0, raw.times[-1], 1.0):
            if onset + 1.0 <= raw.times[-1]:
                raw.annotations.append(onset, 1.0, label_str)

        edf_path = output_dir / f"{mat_path.stem}_trial{trial_idx:02d}.edf"
        mne.export.export_raw(str(edf_path), raw, fmt='edf', overwrite=True, verbose=False)
        logger.info(f"Saved {edf_path} ({label_str})")


def main():
    parser = argparse.ArgumentParser(description='Convert SEED-V .mat to EDF')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Directory with label .mat files (default: same as input_dir)')
    parser.add_argument('--sfreq', type=float, default=1000.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    label_dir = Path(args.label_dir) if args.label_dir else input_dir

    mat_files = sorted(input_dir.rglob('*.mat'))
    logger.info(f"Found {len(mat_files)} .mat files")

    for mat_path in mat_files:
        try:
            # Look for corresponding label file
            label_path = label_dir / (mat_path.stem + '_label.mat')
            if not label_path.exists():
                label_path = label_dir / 'label.mat'

            rel_path = mat_path.relative_to(input_dir)
            out_subdir = output_dir / rel_path.parent
            convert_session(mat_path, label_path, out_subdir, args.sfreq)
        except Exception as e:
            logger.error(f"Failed to convert {mat_path.name}: {e}")


if __name__ == '__main__':
    main()
