"""
PyTorch Dataset and DataLoader for downstream tasks with labels.

This module provides efficient loading of preprocessed EEG windows with labels
from HDF5 files created by the downstream preprocessing pipeline.
"""

import os
import re
from collections import defaultdict
from glob import glob as _glob
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


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
        return_metadata: bool = False,
        target_key: str = "labels",
    ):
        """Parameters (in addition to the class docstring):

        target_key : str, default "labels"
            Which HDF5 dataset to return as the ``label`` element of each sample.
            The default reads the top-level ``labels`` dataset (backward compat).
            Pass e.g. ``"targets/ccd_correct"`` to read a nested per-window
            target written by the multi-target pipeline / migration path. If
            the key is missing from a file, that file's labels fall back to
            sentinel −1 (int) / NaN (float).
        """
        self.path = Path(path)
        self.transform = transform
        self.label_filter = label_filter
        self.return_metadata = return_metadata
        self.target_key = target_key

        # Discover HDF5 files. Recursive so sharded layouts
        # (out_path/shard_XX/batch_*.hdf5) work as transparently as the flat
        # (out_path/batch_*.hdf5) layout. Both are produced by SPEED's
        # downstream SLURM scripts.
        if self.path.is_dir():
            patterns = (
                str(self.path / "**" / "*.hdf5"),
                str(self.path / "**" / "*.h5"),
            )
            self.paths = sorted({
                Path(p) for pat in patterns for p in _glob(pat, recursive=True)
            })
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

                # Load labels if filtering (filter always looks at the selected
                # target_key, not just top-level 'labels').
                if self.label_filter is not None and self.target_key in f:
                    labels = f[self.target_key][:]
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
            # target_key drives which dataset is returned as the label. When
            # missing, fall back to top-level 'labels' (older HDF5s without
            # targets/ group) or −1.
            if self.target_key in f:
                label = f[self.target_key][local_idx]
            elif 'labels' in f:
                label = f['labels'][local_idx]
            else:
                label = -1

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
        # Auto-detect regression (float) vs classification (int) labels
        if isinstance(label, np.floating) or (isinstance(label, np.ndarray) and label.dtype.kind == 'f'):
            label = torch.tensor(label, dtype=torch.float)
        else:
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
                src = self.target_key if self.target_key in f else (
                    'labels' if 'labels' in f else None
                )
                if src is None:
                    continue
                all_labels = f[src][:]
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

    def get_subject_groups(
        self,
        subject_extractor: Optional[Callable[[str], str]] = None
    ) -> Dict[str, List[int]]:
        """
        Group sample indices by subject ID.

        Parameters
        ----------
        subject_extractor : callable, optional
            Function mapping a source filename (str) to a subject ID (str).
            If None, uses a default heuristic: first token before '_' or '-'.

        Returns
        -------
        groups : dict
            Mapping from subject ID to list of global sample indices.
        """
        if subject_extractor is None:
            subject_extractor = _default_subject_extractor

        # Build mapping. Prefer a stored per-window ``targets/subject_id`` when
        # present (written by the multi-target pipeline / migration) —
        # guaranteed correct even for pooled-task buckets (e.g. movies) where
        # stem-regex extraction can be brittle. Fall back to extractor-on-stem.
        groups = defaultdict(list)

        for file_idx, hdf5_path in enumerate(self.paths):
            with h5py.File(hdf5_path, 'r') as f:
                stored_subject = None
                if 'targets/subject_id' in f:
                    stored_subject = [
                        s.decode('utf-8') if isinstance(s, bytes) else s
                        for s in f['targets/subject_id'][:]
                    ]
                    source_files = None
                    file_idxs = None
                elif 'files' in f:
                    source_files = [
                        s.decode('utf-8') if isinstance(s, bytes) else s
                        for s in f['files'][:]
                    ]
                    file_idxs = f['file_idxs'][:] if 'file_idxs' in f else None
                else:
                    source_files = None
                    file_idxs = None

            # Map global indices to subjects for this file.
            for gi, (fi, local_idx) in enumerate(self.index):
                if fi != file_idx:
                    continue
                if stored_subject is not None:
                    subject_id = stored_subject[local_idx] or hdf5_path.stem
                elif source_files is not None:
                    if file_idxs is not None:
                        src_name = source_files[file_idxs[local_idx]]
                    else:
                        src_name = hdf5_path.stem
                    subject_id = subject_extractor(src_name)
                else:
                    subject_id = hdf5_path.stem
                groups[subject_id].append(gi)

        return dict(groups)


def list_targets(path: Union[str, Path]) -> Tuple[Dict[str, Tuple[str, tuple]], Dict[str, Any]]:
    """Inspect the ``targets/`` group of an HDF5 bucket.

    Parameters
    ----------
    path : str or Path
        Directory containing HDF5 files, or a single HDF5 file. The first
        file found is inspected — ``HDF5CombinerDownstream`` guarantees all
        files in a bucket share the same target schema.

    Returns
    -------
    targets : dict[str, (str, tuple)]
        Mapping from target key (e.g. ``"targets/ccd_correct"``) to
        ``(dtype_str, shape)``. Empty dict if no ``targets/`` group present.
    attrs : dict
        Attributes attached to the ``targets/`` group (e.g.
        ``event_tmin_used``, ``generator_version``). Empty dict if no group.
    """
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("**/*.hdf5")) + sorted(path.glob("**/*.h5"))
        if not candidates:
            raise FileNotFoundError(f"No HDF5 files in {path}")
        h5_path = candidates[0]
    else:
        h5_path = path

    out: Dict[str, Tuple[str, tuple]] = {}
    attrs: Dict[str, Any] = {}
    with h5py.File(h5_path, 'r') as f:
        if 'targets' not in f:
            return out, attrs
        grp = f['targets']
        for k, v in grp.items():
            out[f"targets/{k}"] = (str(v.dtype), tuple(v.shape))
        for k, v in grp.attrs.items():
            attrs[k] = v.item() if hasattr(v, 'item') else v
    return out, attrs


def _default_subject_extractor(filename: str) -> str:
    """Extract subject ID from filename using first token before '_' or '-'."""
    # Try common patterns: S001R03 -> S001, chb01_03 -> chb01, sub-001_task -> sub-001
    match = re.match(r'^((?:sub-)?[A-Za-z]*\d+)', filename)
    if match:
        return match.group(1)
    # Fallback: first token split by '_'
    return filename.split('_')[0]


# Per-dataset subject extractors
SUBJECT_EXTRACTORS = {
    'eegmmidb': lambda f: re.match(r'^(S\d+)', f).group(1) if re.match(r'^(S\d+)', f) else f.split('R')[0],
    'chbmit': lambda f: f.split('_')[0],
    'tuab': lambda f: '_'.join(f.split('_')[:3]),
    'tuev': lambda f: '_'.join(f.split('_')[:3]),
    'isruc': lambda f: f.split('_')[0],
    'hmc': lambda f: f.split('_')[0],
    'siena': lambda f: f.split('_')[0],
    'eegmat': lambda f: f.split('_')[0],
    'seedv': lambda f: f.split('_')[0],
    'seed_vig': lambda f: f.split('_')[0],
    'faced': lambda f: f.split('_')[0],
    'mumtaz2016': lambda f: f.split('_')[0],
    'shu_mi': lambda f: f.split('_')[0],
    'mobi': lambda f: f.split('_')[0],
    'bcic_iv_2a': lambda f: f.split('_')[0],
    'bcic2020_iv_3': lambda f: f.split('_')[0],
    # HBN stems look like "sub-NDARAC904DMU_task-Y_[run-Z_]eeg"; the subject
    # id is the leading "sub-<ALPHANUM>" segment.
    'hbn': lambda f: re.match(r'^(sub-[A-Z0-9]+)', f).group(1) if re.match(r'^(sub-[A-Z0-9]+)', f) else f.split('_')[0],
}


def subject_wise_split(
    dataset: 'DownstreamDataset',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    subject_extractor: Optional[Callable[[str], str]] = None,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset ensuring no subject appears in multiple splits.

    Parameters
    ----------
    dataset : DownstreamDataset
        Dataset to split.
    train_ratio : float
        Fraction of subjects for training.
    val_ratio : float
        Fraction of subjects for validation.
    test_ratio : float
        Fraction of subjects for testing.
    subject_extractor : callable, optional
        Function mapping filename to subject ID. If None, uses default heuristic.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_subset, val_subset, test_subset : Subset
        Non-overlapping subsets with subject-level isolation.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    groups = dataset.get_subject_groups(subject_extractor)
    subjects = sorted(groups.keys())

    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
    # test gets the remainder
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    train_indices = [idx for s in train_subjects for idx in groups[s]]
    val_indices = [idx for s in val_subjects for idx in groups[s]]
    test_indices = [idx for s in test_subjects for idx in groups[s]]

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


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
