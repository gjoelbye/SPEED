"""
Convert WBCIC SHU Motor Imagery dataset from .mat to EDF format.

SHU-MI: 52 subjects (2-class subset), 58 EEG channels at 250 Hz.
Each .mat has: data=(58, 1000, 200) and labels=(200,).
  - 58 channels, 1000 samples per trial (4s at 250 Hz), 200 trials
  - Labels: 1=left hand, 2=right hand

Data is organized as:
  shu_mi/sub-XXX/ses-YY/eeg/sub-XXX_ses-YY_task-motorimagery_eeg.mat

Usage:
    python scripts/convert_shu_mi.py \\
        --input_dir /scratch/agjma/SPEED/Original/shu_mi \\
        --output_dir /scratch/agjma/SPEED/Original/shu_mi/edf
"""

import argparse
import logging
from pathlib import Path

import mne
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 59 channels from channels.tsv, but Pz is the reference and not in the data.
# The data has 58 channels — all except Pz.
CHANNELS_ALL = [
    'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2',
    'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3',
    'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4',
    'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
    'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz',
    'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2'
]
CHANNELS_58 = [ch for ch in CHANNELS_ALL if ch != 'Pz']

MI_LABELS = {1: 'left_hand', 2: 'right_hand'}
SFREQ = 250.0


def convert_session(mat_path: Path, output_dir: Path):
    """Convert a single session .mat to EDF with trial annotations."""
    from scipy.io import loadmat

    logger.info(f"Converting {mat_path}")
    d = loadmat(str(mat_path), squeeze_me=True)

    data = np.array(d['data'], dtype=np.float64)   # (58, 1000, 200)
    labels = np.array(d['labels']).flatten()         # (200,)

    n_channels, n_samples_per_trial, n_trials = data.shape
    trial_duration = n_samples_per_trial / SFREQ  # 4.0 seconds

    # Concatenate trials along time axis: (58, 1000*200)
    full_data = data.reshape(n_channels, -1)

    ch_names = CHANNELS_58[:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(full_data * 1e-6, info, verbose=False)  # µV → V

    # Add trial annotations
    for trial_idx in range(n_trials):
        onset = trial_idx * trial_duration
        label_int = int(labels[trial_idx])
        label_str = MI_LABELS.get(label_int, f'class_{label_int}')
        raw.annotations.append(onset, trial_duration, label_str)

    output_dir.mkdir(parents=True, exist_ok=True)
    edf_path = output_dir / f"{mat_path.stem}.edf"
    mne.export.export_raw(str(edf_path), raw, fmt='edf', overwrite=True, verbose=False)
    logger.info(f"Saved {edf_path} ({n_channels}ch, {n_trials} trials)")


def main():
    parser = argparse.ArgumentParser(description='Convert WBCIC SHU-MI .mat to EDF')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root dir with sub-XXX/ses-YY/eeg/*.mat')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for EDF files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    mat_files = sorted(input_dir.rglob('*_eeg.mat'))
    logger.info(f"Found {len(mat_files)} session .mat files")

    for mat_path in mat_files:
        try:
            convert_session(mat_path, output_dir)
        except Exception as e:
            logger.error(f"Failed: {mat_path}: {e}")


if __name__ == '__main__':
    main()
