# SPEED
# Scalable Preprocessing of EEG Data For Self-Supervised Learning

## Overview
This repository contains the source code and resources for the paper "[SPEED: Scalable Preprocessing of EEG Data for Self-Supervised Learning](https://arxiv.org/abs/2408.08065)" presented at the 2024 IEEE International Workshop on Machine Learning for Signal Processing. The pipeline is designed to efficiently preprocess large-scale EEG data for self-supervised learning models, improving stability and performance on downstream tasks.

![Lightning McQueen](speed.gif)

### Key Features:
- **Scalable Preprocessing:** Efficient handling of large EEG datasets with parallel processing and HDF5 batching.
- **Anti-Aliased Resampling:** Proper FIR-based resampling via MNE (no aliasing artifacts).
- **17 Downstream Datasets:** Ready-to-use configs for EEGMMIDB, CHB-MIT, TUAB, TUEV, ISRUC, HMC, SEED-V, and more.
- **Subject-Wise Splitting:** Prevent data leakage with `subject_wise_split()` for proper train/val/test separation.
- **Evaluation Metrics:** Built-in balanced accuracy, AUROC, F1, Cohen's Kappa for benchmarking.
- **Configurable Normalization:** Z-score, min-max, or robust normalization as a pipeline option.
- **Comprehensive Quality Assessment:** Bad channel detection (RANSAC), line noise removal, ICA artifact rejection.
- **Test Suite:** 29 tests with CI/CD via GitHub Actions.

## Repository Structure

- `speed/`: Core preprocessing package (pipeline, methods, dataloader, metrics, annotations).
- `configs/pretrain/`: Pretrain configs (TUH, HBN, example).
- `configs/downstream/`: Downstream dataset configs (17 datasets).
- `scripts/`: Main preprocessing scripts (`preprocess.py`, `preprocess_downstream.py`, `hdf5_combiner.py`).
- `scripts/converters/`: Dataset format conversion scripts (MAT/TXT to EDF).
- `tests/`: Test suite (pytest).
- `examples/`: Usage examples for data loading and evaluation.
- `docs/`: Downstream dataset reference documentation.
- `resources/`: Montage files.
- `slurm/`: SLURM job scripts for HPC environments.

## How to Run the Pipeline

### 1. Clone the Repository
```bash
git clone https://github.com/AndersGMadsen/SPEED.git
cd SPEED
```

### 2. Install Dependencies
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3. Preprocess the Data
The preprocessing script handles large EEG datasets such as TUH EEG, HBN, and MMIDB. Configure paths and parameters in `configs/`.

```bash
python scripts/preprocess.py --config configs/pretrain/tuh.yaml
```

This script will:
- Standardize channels and detect bad channels
- Apply high-pass and low-pass filters
- Run ICA for artifact removal (optional)
- Interpolate missing channels and resample to target frequency
- Export preprocessed data as batched HDF5 files

The script runs in parallel and exports the preprocessed data into multiple HDF5 files.

### 4. Combine Preprocessed Data (Optional)
To combine preprocessed data to fewer files:
```bash
python scripts/hdf5_combiner.py {input_folder} {output_folder}
```
Use `--file_size` to set maximum filesize in MB (default: 2000).

---

## Downstream Preprocessing

For downstream tasks where you need to preserve the original file structure, naming, and format, use the **downstream mode** by setting `preserve_structure: true`.

### What it does:
- Processes full files (no windowing)
- Preserves original folder structure
- Preserves original file naming  
- Preserves original file format (EDF → EDF, BDF → BDF, SET → SET)
- Preserves metadata (subject info, recording date, annotations)
- Optionally copies non-EEG files to maintain complete folder structure

### Usage

```bash
python scripts/preprocess.py --config configs/downstream/example.yaml
```

### Example Configuration

```yaml
pipeline:
  class_path: speed.pipeline.PretrainPipeline
  init_args:
    preserve_metadata: true         # Key: restore metadata after preprocessing
    window_length: null             # Key: process full files
    sfreq: 256.0
    hp_freq: 0.5
    lp_freq: 100.0
    line_freqs: [60.0]
    do_ica: true
    montage: tuh
    channels: [Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, T5, P3, Pz, P4, T6, O1, O2]
    standardize_channel_names: true
    return_quality_metrics: true

dataset_path: /path/to/input/dataset
out_path: /path/to/output
log_path: /path/to/output/log.txt
metrics_path: /path/to/output/

# Key downstream options
preserve_structure: true            # Enable downstream mode
output_format: auto                 # Preserve original format
skip_on_quality_fail: false         # Process all files, even if quality check fails
copy_other_files: true              # Copy non-EEG files (JSON, CSV, etc.)

file_extension: ".edf"
n_jobs: 4
```

### Downstream-Specific Options

| Parameter | Description |
|-----------|-------------|
| `preserve_structure` | If `true`, preserve folder structure (downstream mode). If `false`, batch into HDF5 (pretrain mode). |
| `preserve_metadata` | Pipeline option to restore original metadata after preprocessing. |
| `skip_on_quality_fail` | If `true`, skip files that fail quality checks. If `false`, process them anyway. |
| `output_format` | `"auto"` = preserve original format, or force `"edf"`, `"bdf"`, `"set"` |
| `fallback_format` | Format to use if requested format is unavailable (`"edf"` or `"bdf"`) |
| `copy_other_files` | Copy non-EEG files to preserve complete folder structure |
| `exclude_patterns` | Patterns to exclude when copying (default excludes `.git`, `__pycache__`, etc.) |

### Example: Process TUH dataset preserving structure

```bash
# Input structure:
# /data/tuh_raw/
#   ├── subject_001/
#   │   ├── session_01/
#   │   │   ├── recording.edf
#   │   │   └── metadata.json
#   │   └── session_02/
#   │       └── recording.edf
#   └── subject_002/
#       └── session_01/
#           └── recording.edf

python scripts/preprocess.py \
    --pipeline.init_args.preserve_metadata true \
    --pipeline.init_args.window_length null \
    --pipeline.init_args.channels "[Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8]" \
    --dataset_path /data/tuh_raw/ \
    --out_path /data/tuh_processed/ \
    --preserve_structure true \
    --output_format auto \
    --copy_other_files true \
    --n_jobs 8

# Output structure (preserved):
# /data/tuh_processed/
#   ├── subject_001/
#   │   ├── session_01/
#   │   │   ├── recording.edf    (preprocessed)
#   │   │   └── metadata.json    (copied)
#   │   └── session_02/
#   │       └── recording.edf    (preprocessed)
#   └── subject_002/
#       └── session_01/
#           └── recording.edf    (preprocessed)
```

See `configs/downstream/example.yaml` for a complete reference with all parameters.

---

## Data Loading & Evaluation

### Loading Preprocessed Data

```python
from speed import DownstreamDataset, get_dataloader

dataset = DownstreamDataset('/path/to/processed_data')
print(f"Samples: {len(dataset)}, Labels: {dataset.get_label_counts()}")

loader = get_dataloader('/path/to/processed_data', batch_size=64)
for data, labels in loader:
    print(f"Batch: {data.shape}")  # (64, n_channels, n_samples)
    break
```

### Subject-Wise Splitting (Preventing Data Leakage)

```python
from speed import DownstreamDataset, subject_wise_split, SUBJECT_EXTRACTORS

dataset = DownstreamDataset('/path/to/processed_data')

# Use dataset-specific subject extractor (or None for auto-detection)
train, val, test = subject_wise_split(
    dataset,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    subject_extractor=SUBJECT_EXTRACTORS['eegmmidb'],
    seed=42
)
# No subject appears in more than one split
```

### Evaluation Metrics

```python
from speed import balanced_accuracy, auroc, f1_score, cohens_kappa, classification_report

# After training and getting predictions:
report = classification_report(y_true, y_pred, y_score=y_probs)
# Returns: {'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'cohens_kappa', 'auroc'}
```

### Normalization

Add `normalize` to your config to normalize per-channel after resampling:

```yaml
pipeline:
  init_args:
    normalize: zscore   # Options: null, zscore, minmax, robust
```

---

## Important Files

### `scripts/preprocess.py`
Orchestrates the entire preprocessing workflow. Takes configuration from `configs/` and processes EEG data.

- **--config:** Config file (`YAML`) specifying datasets, channels, filters, output directory and more.

### `scripts/hdf5_combiner.py`
Combines multiple HDF5 files from preprocessing into larger batches.

### `speed/dataloader.py`
PyTorch Dataset and DataLoader for downstream tasks. Includes `subject_wise_split()` for data leakage prevention.

### `speed/metrics.py`
Evaluation metrics for downstream benchmarking: balanced accuracy, AUROC, F1-score, Cohen's Kappa.

## Configuring the Preprocessing Pipeline

The SPEED pipeline is configured using YAML files. See `configs/pretrain/example.yaml` for all options.

### Example Configuration

```yaml
pipeline:
  class_path: speed.pipeline.PretrainPipeline
  init_args:
    # Window settings
    window_length: 60
    shift_seconds: 30
    sfreq: 200.0
    
    # Filter settings
    hp_freq: 0.5
    lp_freq: 50.0
    line_freqs: [60.0]
    
    # ICA settings
    do_ica: false
    
    # Channel configuration
    montage: tuh
    channels: [Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, T5, P3, Pz, P4, T6, O1, O2]
    standardize_channel_names: true
    
    # Quality thresholds
    oha_threshold: 0.00004
    thv_threshold: 0.00004
    chv_threshold: 0.00008
    min_unique_ratio: 0.001
    
    # Quality decision limits
    oha_limit: 0.8
    thv_limit: 0.5
    chv_limit: 0.5
    bcr_limit: 0.8
    
    drop_bad_quality: true
    return_quality_metrics: true

dataset_path: /path/to/dataset
out_path: /path/to/output
log_path: /path/to/output/log.txt
metrics_path: /path/to/output/quality_metrics.csv

batch_size: 4
n_jobs: 16
file_extension: ".edf"
save_as_hdf5: true
```

### Configuration Parameters

**Pipeline Settings:**
- `window_length`: Window length in seconds. `null` = full file.
- `shift_seconds`: Window shift for overlap. `null` = no overlap.
- `sfreq`: Target sampling frequency after resampling.
- `hp_freq` / `lp_freq`: High-pass and low-pass filter cutoffs. `null` = skip.
- `line_freqs`: Power line frequencies for notch filtering (e.g., `[60.0]`).
- `do_ica`: Enable ICA artifact rejection.
- `iclabel_threshold`: Probability threshold for IC classification (0-1).
- `included_components`: IC types to retain (e.g., `["brain", "other"]`).
- `normalize`: Normalization method after resampling: `null` (none), `"zscore"`, `"minmax"`, or `"robust"`.

**Channel Settings:**
- `montage`: `"tuh"`, standard MNE name (e.g., `"standard_1020"`), path to `.fif` file, or `null`.
- `channels`: Required output channels in order.
- `channels_to_remove`: Channels to drop before processing.
- `channels_rename`: Dictionary for renaming channels.
- `standardize_channel_names`: Apply TUH-style channel name standardization.
- `min_nchans`: Minimum channels required to process a file.

**Quality Control:**
- `oha_threshold`: Overall high amplitude threshold (Volts).
- `thv_threshold`: Temporal high variance threshold (Volts).
- `chv_threshold`: Channel high variance threshold (Volts).
- `min_unique_ratio`: Minimum ratio of unique samples per channel.
- `oha_limit` / `thv_limit` / `chv_limit` / `bcr_limit`: Decision thresholds for quality metrics.
- `drop_bad_quality`: Drop windows failing quality checks.
- `return_quality_metrics`: Collect and export quality metrics.

**Interpolation:**
- `target_montage`: Path to target montage `.fif` file for spatial interpolation.
- `interpolation_mode`: `"accurate"` or `"fast"`.
- `use_ransac`: Use RANSAC for bad channel detection.

**Processing Options:**
- `dataset_path`: Directory or `.txt` file with paths.
- `out_path`: Output directory.
- `log_path`: Log file path.
- `metrics_path`: Directory for `quality_metrics.csv`.
- `overwrite`: Overwrite existing files.
- `shuffle_files`: Randomly shuffle file order.
- `batch_size`: Files per HDF5 batch.
- `n_jobs`: Parallel workers.
- `file_extension`: File extensions to process (`.edf`, `.set`, or list).
- `save_as_hdf5`: `true` = batch HDF5, `false` = individual BDF files.

See `configs/pretrain/example.yaml` for a complete reference with all parameters.

## Supported Datasets

### Pretrain Datasets
- **[TUH EEG Corpus (TUEG)](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)** — Largest publicly available EEG dataset (26,846 recordings).
- **[Healthy Brain Network (HBN)](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)** — Large pediatric neuroimaging dataset.

### Downstream Datasets (17 configs)
Motor imagery, seizure detection, sleep staging, emotion recognition, abnormality detection, and more. Each has a ready-to-use config in `configs/downstream/`.

See `docs/downstream_datasets.md` for the full reference with channels, sampling rates, labels, and processing notes.

## Links

- **Paper**: [Link to Paper](https://arxiv.org/abs/2408.08065)
- **GitHub**: [Repository](https://github.com/AndersGMadsen/SPEED)
- **Data**: [TUEG](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/), [HBN](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/), [MMIDB](https://www.physionet.org/content/eegmmidb/1.0.0/), [BCIC](https://www.kaggle.com/c/inria-bci-challenge/data)

## Citation

If you use this code for your research, please cite the following paper:

@inproceedings{Gjoelbye2024SPEED,\
   &nbsp;&nbsp;&nbsp;&nbsp;title={SPEED: Scalable Preprocessing of EEG Data for Self-Supervised Learning},\
   &nbsp;&nbsp;&nbsp;&nbsp;author={Anders Gjølbye, Lina Skerath, William Lehn-Schiøler, Nicolas Langer, Lars Kai Hansen},\
   &nbsp;&nbsp;&nbsp;&nbsp;booktitle={IEEE International Workshop on Machine Learning for Signal Processing},\
   &nbsp;&nbsp;&nbsp;&nbsp;year={2024}\
}

## License
This project is licensed under the CC BY 4.0 License - see the [LICENSE](LICENSE) file for details.