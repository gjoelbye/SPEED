# SPEED
# EEG Preprocessing Pipeline for Self-Supervised Learning (SPEED)

## Overview
This repository contains the source code and resources for the paper "[SPEED: Scalable Preprocessing of EEG Data for Self-Supervised Learning](https://arxiv.org/abs/2408.08065)" presented at the 2024 IEEE International Workshop on Machine Learning for Signal Processing. The pipeline is designed to efficiently preprocess large-scale EEG data for self-supervised learning models, improving stability and performance on downstream tasks.

![Lightning McQueen](speed.gif)

### Key Features:
- **Scalable Preprocessing:** Efficient handling of large EEG datasets, such as the Temple University Hospital EEG Corpus.
- **Self-Supervised Learning Compatibility:** Optimized for SSL frameworks to enhance model performance on various downstream tasks.
- **Comprehensive Quality Assessment:** Includes several quality checks, such as bad channel detection, artifact removal (e.g., line noise), and ICA for component classification.
- **Support for Multiple EEG Datasets:** Preprocessing steps tailored for TUH EEG, MMIDB, and other datasets.

## Repository Structure

- `configs/`: Contains configuration files to customize the preprocessing pipeline (e.g., datasets, channels, filtering options).
- `examples/`: Examples of how to use the SPEED pipeline, analyze of results and example of how to load the preprocessed data.
- `scripts/`: Utility scripts to automate various tasks such as data download, preprocessing, and model training.
- `src/`: Contains all the scripts responsible for the core preprocessing of EEG data (e.g., filtering, bad channel detection, ICA).
- `requirements.txt`: Lists the necessary Python packages to run the pipeline.
- `requirements_dev.txt`: Additional dependencies for development purposes (e.g., testing, linting).

## How to Run the Pipeline

### 1. Clone the Repository
```bash
git clone https://github.com/AndersGMadsen/SPEED.git
cd SPEED
```

### 2. Install Dependencies
It is recommended to use a virtual environment to manage dependencies:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 3. Preprocess the Data
The preprocessing script is designed to handle large EEG datasets such as the **TUH EEG Corpus** and **MMIDB** dataset. Ensure the paths to these datasets are configured in the `configs/` directory.

Run the preprocessing as follows:
```bash
python scripts/preprocess.py --config configs/preprocessing_config.yaml
```

This script will process the EEG data using the methods described in the paper, including:
- Standardizing channels and detecting bad channels
- Applying filters (high-pass and low-pass)
- Running Independent Component Analysis (ICA) for artifact removal (optional)
- Interpolating missing channels and resampling to a uniform rate

You can modify preprocessing steps (e.g., include or skip ICA) by updating the configuration files in the `configs/` folder.

The script is able to run in parallel and exports the preprocessed data into multiple HDF5 files.

### 4. Combine Preprocessed Data (Optional)
To combine your preprocessed data to fewer files, run the provided script:
```bash
python scripts/hdf5_combiner.py {input_folder} {output_folder}
```
You can modify the maximum filesize by the argument ```--file_size```. The default value is 2000 (MB). 

<!-- ### 5. Training a Self-Supervised Learning Model
After preprocessing, the data can be fed into SSL models such as BENDR or custom models for pretraining:
```bash
python scripts/train_ssl_model.py --data-dir preprocessed_data/ --model bendr
``` -->

## Important Files

### `scripts/preprocess.py`
This script orchestrates the entire preprocessing workflow. It takes the configuration from `configs/` and processes the EEG data according to the steps defined in the paper.

- **--config:** Config file (`YAML`) specifying datasets, channels, filters, output directory and more.

### `scripts/hdf5_combiner.py`
This script combines the many HDF5 files that comes from a result of the preprocessing.

### `examples/data_loader.ipynb`
This notebook contains a PyTorch dataloader class that is designed for the preprocessed data and which is optimized for large scale.


<!-- ### `src/pipeline.py`
Contains class for the preprocessing pipeline.

### `src/methods.py`

### `src/utils.py` -->

## Configuring the Preprocessing Pipeline

The SPEED pipeline can be customized using YAML configuration files, allowing you to control key preprocessing parameters like filter frequencies, ICA settings, and more.

### Example Configuration File

```yaml
pipeline:
  class_path: preprocessing.pipeline.DynamicPipeline
  init_args:
    lp_freq: 75
    do_ica: False
    line_freqs: [60]
dataset_path: {DATASET_PATH}
out_path: {OUTPUT_PATH}
log_path: {LOG_PATH}
overwrite: False
shuffle_files: True
batch_size: 4
n_jobs: 16
```

### Explanation of Configuration Parameters

- **`pipeline`**:
  - **`class_path`**: `str`  
    Path to the pipeline class. Default is `preprocessing.pipeline.DynamicPipeline`.
  - **`init_args`**:  
    Initialization arguments for the pipeline. Key parameters include:
    - **`window_length`**: `int`, optional, default=60  
      Length of the processing window in seconds.
    - **`sfreq`**: `float`, optional, default=256.0  
      Target sampling frequency after resampling.
    - **`hp_freq`**: `float or None`, optional, default=0.5  
      High-pass filter frequency. Set to `None` to disable high-pass filtering.
    - **`lp_freq`**: `float or None`, optional, default=100.0  
      Low-pass filter frequency. Set to `None` to disable low-pass filtering.
    - **`line_freqs`**: `list of float`, optional, default=[60.0]  
      Frequencies for notch filtering, typically used to remove power line noise (e.g., 60 Hz).
    - **`iclabel_threshold`**: `float`, optional, default=0.7  
      Threshold for Independent Component (IC) classification during ICA.
    - **`quality_check`**: `bool`, optional, default=True  
      If `True`, performs quality checks like bad channel detection.
    - **`min_nchans`**: `int`, optional, default=10  
      Minimum number of channels required for processing a recording.
    - **`do_ica`**: `bool`, optional, default=True  
      If `True`, performs Independent Component Analysis (ICA) for artifact removal.
    - **`included_components`**: `list of str`, optional, default=["brain", "other"]  
      List of IC types to retain after ICA. Common values include "brain" and "other".
    - **`memory_efficient`**: `bool`, optional, default=True  
      If `True`, uses a memory-efficient approach, which can reduce resource usage.
    - **`montage_name`**: `str`, optional, default="tuh"  
      Name of the montage to use for the dataset (e.g., `"tuh"` or `"standard_1020"`).
    - **`channels`**: `list of str`  
      List of EEG channels to include in the analysis.
    - **`channels_rename`**: `dict or None`, optional, default=None  
      Dictionary for renaming channels, if necessary.

- **`dataset_path`**: `str`  
  Path to the raw EEG dataset.
- **`out_path`**: `str`  
  Directory where the preprocessed data will be saved.
- **`log_path`**: `str`  
  Path to the log file that records the preprocessing steps.
- **`overwrite`**: `bool`, optional, default=False  
  If `True`, overwrites any existing preprocessed files in the output directory.
- **`shuffle_files`**: `bool`, optional, default=True  
  If `True`, shuffles files before processing to balance load across batches.
- **`batch_size`**: `int`, optional, default=4  
  Number of files to process simultaneously, depending on the system's memory.
- **`n_jobs`**: `int`, optional, default=16  
  Number of CPU cores to use for parallel processing. More cores will speed up processing but require more resources.

Customize these parameters based on your dataset and system configuration, and run the pipeline using your configuration file. See `src/pipeline.py` for more details.


## Datasets Used

1. **[TUH EEG Corpus (TUEG)](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)**
   - Largest publicly available EEG dataset with 26,846 recordings.
   - Used for pretraining and fine-tuning.

2. **[Motor Movement/Imagery Dataset (MMIDB)](https://www.physionet.org/content/eegmmidb/1.0.0/)**
   - Used for downstream benchmarking tasks like motor imagery classification.

3. **[BCI Challenge @ NER 2015 (BCIC)](https://www.kaggle.com/c/inria-bci-challenge/data)**
   - A smaller dataset used for classification tasks involving feedback.

## Links

- **Paper**: [Link to Paper](https://arxiv.org/abs/2408.08065)
- **GitHub**: [Repository](https://github.com/AndersGMadsen/SPEED)
- **Data**: [TUEG](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/), [MMIDB](https://www.physionet.org/content/eegmmidb/1.0.0/), [BCIC](https://www.kaggle.com/c/inria-bci-challenge/data)

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

