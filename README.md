# SPEED
# EEG Preprocessing Pipeline for Self-Supervised Learning (SPEED)

## Overview
This repository contains the source code and resources for the paper "[SPEED: Scalable Preprocessing of EEG Data for Self-Supervised Learning](https://arxiv.org/abs/2408.08065)" presented at the 2024 IEEE International Workshop on Machine Learning for Signal Processing. The pipeline is designed to efficiently preprocess large-scale EEG data for self-supervised learning models, improving stability and performance on downstream tasks.

### Key Features:
- **Scalable Preprocessing:** Efficient handling of large EEG datasets, such as the Temple University Hospital EEG Corpus.
- **Self-Supervised Learning Compatibility:** Optimized for SSL frameworks to enhance model performance on various downstream tasks.
- **Comprehensive Quality Assessment:** Includes several quality checks, such as bad channel detection, artifact removal (e.g., line noise), and ICA for component classification.
- **Support for Multiple EEG Datasets:** Preprocessing steps tailored for TUH EEG, MMIDB, and other datasets.

## Repository Structure

- `configs/`: Contains configuration files to customize the preprocessing pipeline (e.g., datasets, channels, filtering options).
- `preprocessing/`: Contains all the scripts responsible for the core preprocessing of EEG data (e.g., filtering, bad channel detection, ICA).
- `scripts/`: Utility scripts to automate various tasks such as data download, preprocessing, and model training.
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

### 4. Combine Preprocessed Data (Optional)
To combine your preprocessed data to fewer files, run the provided script:
```bash
python scripts/hdf5_combiner.py {input_folder} {output_folder}
```
You can modify the maximum filesize by the argument ```--file_size```. The default value is 2000 (MB). 

### 5. Training a Self-Supervised Learning Model
After preprocessing, the data can be fed into SSL models such as BENDR or custom models for pretraining:
```bash
python scripts/train_ssl_model.py --data-dir preprocessed_data/ --model bendr
```

## Important Functions

### `scripts/preprocess.py`
This script orchestrates the entire preprocessing workflow. It takes the configuration from `configs/` and processes the EEG data according to the steps defined in the paper.

- **Input:**
  - Config file (`YAML`) specifying datasets, channels, and filters.
- **Output:**
  - Preprocessed EEG data, quality logs, and bad channel statistics.

### `scripts/hdf5_combiner.py`
This module performs quality control on the data by checking for artifacts, line noise, and bad channels.
- **Input:**
  - Raw EEG data (after channel standardization).
- **Output:**
  - Metrics and logs for bad windows and channels, used to filter low-quality data.

### `src/pipeline.py`
This optional step applies Independent Component Analysis (ICA) and classifies components using ICLabel for artifact removal.
- **Input:**
  - EEG data with bad channels removed.
- **Output:**
  - Clean EEG data with artifacts (e.g., muscle, blink) removed.

### `src/methods.py`

### `src/utils.py`

## Datasets Used

1. **TUH EEG Corpus (TUEG)**
   - Largest publicly available EEG dataset with 26,846 recordings.
   - Used for pretraining and fine-tuning.

2. **Motor Movement/Imagery Dataset (MMIDB)**
   - Used for downstream benchmarking tasks like motor imagery classification.

3. **BCI Challenge @ NER 2015**
   - A smaller dataset used for classification tasks involving feedback.

## Links

- **Paper**: [Link to Paper](https://arxiv.org/abs/2408.08065)
- **GitHub**: [Repository](https://github.com/AndersGMadsen/SPEED)
- **Data**: Links to datasets if publicly available

## Citation

If you use this code for your research, please cite the following paper:

@inproceedings{Gjølbye2024SPEED,
  title={SPEED: Scalable Preprocessing of EEG Data for Self-Supervised Learning},
  author={Anders Gjølbye, Lina Skerath, William Lehn-Schiøler, Nicolas Langer, Lars Kai Hansen},
  booktitle={IEEE International Workshop on Machine Learning for Signal Processing},
  year={2024}
}

## License
This project is licensed under the CC BY 4.0 License - see the [LICENSE](LICENSE) file for details.

