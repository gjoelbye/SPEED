"""
Convert MoBI gait prediction dataset from text files to EDF format.

MoBI: 8 subjects (SL01-SL08), 3 trials each, 64 EEG/EOG channels at 100 Hz.
12 joint-angle targets: 6 goniometer-measured (GHR, GKR, GAR, GHL, GKL, GAL)
+ 6 BCI-predicted (PHR, PKR, PAR, PHL, PKL, PAL).
Hardware: ActiCap system (Brain Products GmbH), NOT BioSemi.

Data format:
- eeg.txt: header "64 channels", then tab-separated rows (timestamp + 64 ch + extra)
- joints.txt: 2 header lines, then tab-separated (timestamp + 6 measured + 6 predicted)

The converted EDF files embed regression targets as annotations in the
format 'gait_<v1>_<v2>_..._<v12>' for each 2-second window.

Usage:
    python scripts/convert_mobi.py \\
        --input_dir /scratch/agjma/SPEED/Original/mobi \\
        --output_dir /scratch/agjma/SPEED/Original/mobi_edf
"""

import argparse
import logging
from pathlib import Path

import mne
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ActiCap 60 EEG channels (from standard 10-20 extended layout)
# The dataset has 64 total channels (60 EEG + 4 EOG)
CHANNELS_64 = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'EOG1', 'EOG2', 'EOG3', 'EOG4'  # 4 EOG channels
]  # 60 EEG + 4 EOG = 64 (Fpz is ground, not recorded)

# Joint angle target names from joints.txt header:
# GHR=goniometer hip right, GKR=knee right, GAR=ankle right,
# GHL=hip left, GKL=knee left, GAL=ankle left,
# PHR=predicted hip right, PKR=predicted knee right, etc.
TARGET_NAMES = ['GHR', 'GKR', 'GAR', 'GHL', 'GKL', 'GAL',
                'PHR', 'PKR', 'PAR', 'PHL', 'PKL', 'PAL']


def load_eeg_txt(eeg_path: Path):
    """Load eeg.txt: header line, then tab-separated (time + 64 channels)."""
    # Files have trailing tabs and \r — use pandas for robust parsing
    import pandas as pd
    df = pd.read_csv(eeg_path, sep='\t', skiprows=1, header=None,
                     engine='c', on_bad_lines='skip')
    df = df.dropna(axis=1, how='all')  # Drop empty trailing column
    data = df.values.astype(np.float64)
    timestamps = data[:, 0]
    n_channels = min(data.shape[1] - 1, 64)
    eeg_data = data[:, 1:n_channels + 1]
    return timestamps, eeg_data.T  # (n_channels, n_samples)


def load_joints_txt(joints_path: Path):
    """Load joints.txt: 2 header lines, then tab-separated (time + 12 angles)."""
    import pandas as pd
    df = pd.read_csv(joints_path, sep='\t', skiprows=2, header=None,
                     engine='c', on_bad_lines='skip')
    df = df.dropna(axis=1, how='all')
    data = df.values.astype(np.float64)
    timestamps = data[:, 0]
    n_targets = min(data.shape[1] - 1, 12)
    targets = data[:, 1:n_targets + 1]
    return timestamps, targets


def convert_session(session_dir: Path, output_dir: Path, sfreq: float = 100.0,
                    window_length: float = 2.0, stride: float = 0.05):
    """Convert a single session's txt files to EDF with gait target annotations."""
    eeg_path = session_dir / 'eeg.txt'
    joints_path = session_dir / 'joints.txt'

    if not eeg_path.exists():
        logger.warning(f"eeg.txt not found in {session_dir}")
        return

    logger.info(f"Converting {session_dir.name}")

    # Load EEG data
    timestamps, eeg_data = load_eeg_txt(eeg_path)
    n_channels, n_samples = eeg_data.shape

    # Use actual channel names or fallback
    ch_names = CHANNELS_64[:n_channels]
    ch_types = ['eeg'] * min(n_channels, 60) + ['eog'] * max(0, n_channels - 60)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # EEG data is in microvolts, convert to volts
    raw = mne.io.RawArray(eeg_data * 1e-6, info, verbose=False)

    # Load joint angle targets and build annotations in batch
    if joints_path.exists():
        joint_timestamps, targets = load_joints_txt(joints_path)

        n_win_samples = int(window_length * sfreq)
        stride_samples = int(stride * sfreq)

        onsets, durations, descriptions = [], [], []
        for win_start in range(0, n_samples - n_win_samples + 1, stride_samples):
            if win_start >= len(targets):
                break
            win_end = min(win_start + n_win_samples, len(targets))
            target_vals = np.mean(targets[win_start:win_end], axis=0) / 90.0
            target_str = '_'.join(f'{v:.4f}' for v in target_vals)
            onsets.append(win_start / sfreq)
            durations.append(window_length)
            descriptions.append(f'gait_{target_str}')

        raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
        logger.info(f"  {len(onsets)} annotations created")
    else:
        logger.warning(f"joints.txt not found in {session_dir}")

    # Save as EDF
    output_dir.mkdir(parents=True, exist_ok=True)
    edf_path = output_dir / f"{session_dir.name}.edf"
    mne.export.export_raw(str(edf_path), raw, fmt='edf', overwrite=True, verbose=False)
    logger.info(f"Saved {edf_path} ({n_channels}ch, {n_samples} samples, {raw.times[-1]:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description='Convert MoBI txt to EDF')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with SLxx-Tyy/ session folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for EDF files')
    parser.add_argument('--sfreq', type=float, default=100.0)
    parser.add_argument('--window_length', type=float, default=2.0)
    parser.add_argument('--stride', type=float, default=0.05)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find session directories (SLxx-Tyy pattern)
    session_dirs = sorted([d for d in input_dir.iterdir()
                          if d.is_dir() and d.name.startswith('SL')])
    logger.info(f"Found {len(session_dirs)} session directories")

    from tqdm import tqdm
    for session_dir in tqdm(session_dirs, desc="Converting sessions"):
        try:
            convert_session(session_dir, output_dir, args.sfreq,
                          args.window_length, args.stride)
        except Exception as e:
            logger.error(f"Failed: {session_dir.name}: {e}")


if __name__ == '__main__':
    main()
