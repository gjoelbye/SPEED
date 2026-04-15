Quick Start
===========

Preprocessing EEG Data
----------------------

1. Configure your pipeline in a YAML file (see ``configs/pretrain/example.yaml``).

2. Run the preprocessing script:

.. code-block:: bash

   python scripts/preprocess.py --config configs/pretrain/tuh.yaml

Loading Preprocessed Data
-------------------------

.. code-block:: python

   from speed import DownstreamDataset, get_dataloader

   dataset = DownstreamDataset('/path/to/processed_data')
   loader = get_dataloader('/path/to/processed_data', batch_size=64)

   for data, labels in loader:
       print(data.shape)  # (64, n_channels, n_samples)
       break

Subject-Wise Splitting
----------------------

.. code-block:: python

   from speed import DownstreamDataset, subject_wise_split, SUBJECT_EXTRACTORS

   dataset = DownstreamDataset('/path/to/processed_data')
   train, val, test = subject_wise_split(
       dataset,
       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
       subject_extractor=SUBJECT_EXTRACTORS['eegmmidb'],
       seed=42
   )

Evaluation
----------

.. code-block:: python

   from speed import classification_report

   report = classification_report(y_true, y_pred, y_score=y_probs)
