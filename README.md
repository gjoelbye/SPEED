# SPEED
# Scalable Preprocessing of EEG Data For Self-Supervised Learning

## Overview
This repository contains the source code and resources for the paper "[SPEED: Scalable Preprocessing of EEG Data for Self-Supervised Learning](https://arxiv.org/abs/2408.08065)" presented at the 2024 IEEE International Workshop on Machine Learning for Signal Processing. The pipeline is designed to efficiently preprocess large-scale EEG data for self-supervised learning models, improving stability and performance on downstream tasks.

![Lightning McQueen](speed.gif)

### Key Features:
- **Scalable Preprocessing:** Efficient handling of large EEG datasets, such as the Temple University Hospital EEG Corpus.
- **Self-Supervised Learning Compatibility:** Optimized for SSL frameworks to enhance model performance on various downstream tasks.
- **Comprehensive Quality Assessment:** Includes several quality checks, such as bad channel detection, artifact removal (e.g., line noise), and ICA for component classification.
- **Support for Multiple EEG Datasets:** Preprocessing steps tailored for TUH EEG, HBN, MMIDB, and other datasets.

## Repository Structure

- `configs/`: Configuration files to customize the preprocessing pipeline (e.g., datasets, channels, filtering options).
- `examples/`: Examples of how to use the SPEED pipeline, analyze results and load preprocessed data.
- `notebooks/`: Development notebooks for pipeline testing and analysis.
- `scripts/`: Utility scripts for preprocessing and data management.
- `speed/`: Core preprocessing pipeline and methods.
- `resources/`: Montage files and other resources.
- `slurm/`: SLURM job scripts for HPC environments.
- `requirements.txt`: Python package dependencies.
- `requirements_dev.txt`: Additional dependencies for development.

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
python scripts/preprocess.py --config configs/tuh.yaml
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

## Important Files

### `scripts/preprocess.py`
Orchestrates the entire preprocessing workflow. Takes configuration from `configs/` and processes EEG data.

- **--config:** Config file (`YAML`) specifying datasets, channels, filters, output directory and more.

### `scripts/hdf5_combiner.py`
Combines multiple HDF5 files from preprocessing into larger batches.

### `examples/data_loader.ipynb`
PyTorch dataloader class optimized for large-scale preprocessed data.

## Configuring the Preprocessing Pipeline

The SPEED pipeline is configured using YAML files. See `configs/example.yaml` for all options.

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
- `list_file`: Text file with explicit file paths.
- `save_as_hdf5`: `true` = batch HDF5, `false` = individual BDF files.

See `configs/example.yaml` for a complete reference with all parameters.

## Datasets Used

1. **[TUH EEG Corpus (TUEG)](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)**
   - Largest publicly available EEG dataset with 26,846 recordings.
   - Used for pretraining and fine-tuning.

2. **[Healthy Brain Network (HBN)](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)**
   - Large pediatric neuroimaging dataset.
   - Supports `.set` file format.

3. **[Motor Movement/Imagery Dataset (MMIDB)](https://www.physionet.org/content/eegmmidb/1.0.0/)**
   - Used for downstream benchmarking tasks like motor imagery classification.

4. **[BCI Challenge @ NER 2015 (BCIC)](https://www.kaggle.com/c/inria-bci-challenge/data)**
   - Smaller dataset for classification tasks involving feedback.

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