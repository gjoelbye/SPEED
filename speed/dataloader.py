"""
PyTorch Dataset and DataLoader for downstream tasks with labels.

This module provides efficient loading of preprocessed EEG windows with labels
from HDF5 files created by the downstream preprocessing pipeline.
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DownstreamDataset(Dataset):
    """
    PyTorch Dataset for downstream tasks with labels.

    Loads preprocessed EEG windows with labels from HDF5 files created by
    the downstream preprocessing pipeline. Supports multi-file datasets,
    label filtering, and optional transforms.

    Parameters
    ----------
    path : str or Path
        Directory containing HDF5 files or path to single HDF5 file
    transform : callable, optional
        Optional transform to apply to data (e.g., normalization, augmentation)
    label_filter : list of int, optional
        Filter to specific class labels (e.g., [0, 1] for binary classification).
        If None, include all labels.
    return_metadata : bool, default=False
        If True, return (data, label, metadata_dict) instead of (data, label)

    Attributes
    ----------
    label_descriptions : list of str
        String descriptions for each label class (loaded from HDF5 attributes)
    paths : list of Path
        List of HDF5 file paths
    index : list of tuple
        Index mapping global dataset index to (file_idx, local_idx)

    Examples
    --------
    >>> # Basic usage
    >>> dataset = DownstreamDataset('/path/to/processed_data')
    >>> data, label = dataset[0]
    >>> print(f"Data shape: {data.shape}, Label: {label}")

    >>> # With metadata
    >>> dataset = DownstreamDataset('/path/to/processed_data', return_metadata=True)
    >>> data, label, metadata = dataset[0]
    >>> print(f"Source file: {metadata['filename']}")

    >>> # Filter to specific classes
    >>> dataset = DownstreamDataset(
    ...     '/path/to/processed_data',
    ...     label_filter=[0, 1]  # Binary classification
    ... )

    >>> # With normalization transform
    >>> def z_normalize(x):
    ...     return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
    >>> dataset = DownstreamDataset('/path/to/processed_data', transform=z_normalize)
    """

    def __init__(
        self,
        path: Union[str, Path],
        transform: Optional[Callable] = None,
        label_filter: Optional[List[int]] = None,
        return_metadata: bool = False
    ):
        self.path = Path(path)
        self.transform = transform
        self.label_filter = label_filter
        self.return_metadata = return_metadata

        # Discover HDF5 files
        if self.path.is_dir():
            self.paths = sorted([
                self.path / f for f in os.listdir(self.path)
                if f.endswith('.hdf5') or f.endswith('.h5')
            ])
        elif self.path.is_file():
            self.paths = [self.path]
        else:
            raise ValueError(f"Path does not exist: {path}")

        if len(self.paths) == 0:
            raise ValueError(f"No HDF5 files found in {path}")

        # Build index and load metadata
        self.label_descriptions = None
        self.index = []
        self._build_index()

    def _build_index(self):
        """Build index of (file_idx, local_idx) for each valid sample."""
        for file_idx, hdf5_path in enumerate(self.paths):
            with h5py.File(hdf5_path, 'r') as f:
                n_samples = f['data'].shape[0]

                # Load label descriptions from first file
                if self.label_descriptions is None and 'descriptions' in f.attrs:
                    self.label_descriptions = list(f.attrs['descriptions'])

                # Load labels if filtering
                if self.label_filter is not None and 'labels' in f:
                    labels = f['labels'][:]
                    valid_indices = [i for i in range(n_samples) if labels[i] in self.label_filter]
                else:
                    valid_indices = list(range(n_samples))

                # Add to index
                for local_idx in valid_indices:
                    self.index.append((file_idx, local_idx))

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.index)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        data : torch.Tensor
            EEG data window of shape (n_channels, n_samples)
        label : torch.Tensor
            Integer class label (scalar tensor)
        metadata : dict (only if return_metadata=True)
            Dictionary with keys: 'file_idx', 'filename', 'time_slice'
        """
        file_idx, local_idx = self.index[idx]
        hdf5_path = self.paths[file_idx]

        # Load data and label from HDF5
        with h5py.File(hdf5_path, 'r') as f:
            data = f['data'][local_idx]
            label = f['labels'][local_idx] if 'labels' in f else -1

            if self.return_metadata:
                file_idx_in_file = f['file_idxs'][local_idx]
                filename = f['files'][file_idx_in_file]

                # Handle both string and bytes
                if isinstance(filename, bytes):
                    filename = filename.decode('utf-8')

                metadata = {
                    'file_idx': int(file_idx_in_file),
                    'filename': filename,
                    'time_slice': tuple(f['time_slices'][local_idx])
                }

        # Convert to tensors
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)

        # Apply transform
        if self.transform:
            data = self.transform(data)

        if self.return_metadata:
            return data, label, metadata
        return data, label

    def get_label_counts(self) -> Dict[int, int]:
        """
        Get count of samples for each label class.

        Returns
        -------
        counts : dict
            Mapping from label index to count
        """
        counts = {}

        # Group index entries by file to batch-read labels
        from collections import defaultdict
        file_groups = defaultdict(list)
        for file_idx, local_idx in self.index:
            file_groups[file_idx].append(local_idx)

        for file_idx, local_indices in file_groups.items():
            with h5py.File(self.paths[file_idx], 'r') as f:
                if 'labels' in f:
                    all_labels = f['labels'][:]
                    for local_idx in local_indices:
                        label = int(all_labels[local_idx])
                        counts[label] = counts.get(label, 0) + 1
        return counts

    def get_label_name(self, label: int) -> str:
        """
        Get string description for a label index.

        Parameters
        ----------
        label : int
            Label index

        Returns
        -------
        description : str
            Label description
        """
        if self.label_descriptions is None:
            return str(label)
        if 0 <= label < len(self.label_descriptions):
            return self.label_descriptions[label]
        return str(label)


def get_dataloader(
    path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader from HDF5 directory or file.

    Parameters
    ----------
    path : str or Path
        Directory containing HDF5 files or path to single HDF5 file
    batch_size : int, default=32
        Batch size
    shuffle : bool, default=True
        Whether to shuffle data
    num_workers : int, default=4
        Number of worker processes for data loading
    pin_memory : bool, default=True
        If True, pin memory for faster GPU transfer
    **dataset_kwargs
        Additional arguments passed to DownstreamDataset

    Returns
    -------
    dataloader : DataLoader
        PyTorch DataLoader

    Examples
    --------
    >>> # Basic dataloader
    >>> loader = get_dataloader('/path/to/processed_data', batch_size=64)
    >>> for data, labels in loader:
    ...     print(f"Batch shape: {data.shape}, Labels: {labels.shape}")
    ...     break

    >>> # With custom settings
    >>> loader = get_dataloader(
    ...     '/path/to/processed_data',
    ...     batch_size=32,
    ...     shuffle=False,
    ...     num_workers=8,
    ...     label_filter=[0, 1]  # Passed to DownstreamDataset
    ... )
    """
    dataset = DownstreamDataset(path, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def get_weighted_sampler(dataset: DownstreamDataset) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a weighted sampler for balanced class sampling.

    Useful for handling class imbalance (e.g., CHBMIT seizure vs non-seizure).

    Parameters
    ----------
    dataset : DownstreamDataset
        Dataset to create sampler for

    Returns
    -------
    sampler : WeightedRandomSampler
        Sampler that balances classes

    Examples
    --------
    >>> dataset = DownstreamDataset('/path/to/imbalanced_data')
    >>> sampler = get_weighted_sampler(dataset)
    >>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """
    # Get label counts
    label_counts = dataset.get_label_counts()

    # Compute class weights (inverse frequency)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}

    # Build a label lookup per sample by batch-reading labels per file
    from collections import defaultdict
    file_groups = defaultdict(list)
    for global_idx, (file_idx, local_idx) in enumerate(dataset.index):
        file_groups[file_idx].append((global_idx, local_idx))

    sample_weights = [1.0] * len(dataset.index)
    for file_idx, entries in file_groups.items():
        with h5py.File(dataset.paths[file_idx], 'r') as f:
            if 'labels' in f:
                all_labels = f['labels'][:]
                for global_idx, local_idx in entries:
                    label = int(all_labels[local_idx])
                    sample_weights[global_idx] = class_weights[label]

    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
