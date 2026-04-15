"""
Example usage of the downstream dataloader for EEG classification tasks.

This script demonstrates how to:
1. Load preprocessed EEG data with labels from HDF5 files
2. Create PyTorch DataLoaders for training/validation/test
3. Apply transforms (normalization)
4. Handle class imbalance with weighted sampling
5. Iterate over batches for model training
"""

import torch
import torch.nn as nn
from pathlib import Path

from speed.dataloader import (
    DownstreamDataset,
    get_dataloader,
    get_weighted_sampler
)


# =============================================================================
# Example 1: Basic Usage - EEGMMIDB Motor Imagery
# =============================================================================

def example_eegmmidb_basic():
    """Basic dataloader for EEGMMIDB 4-class motor imagery."""
    print("\n=== Example 1: EEGMMIDB Basic Usage ===")

    # Path to preprocessed data
    data_path = "/scratch/agjma/MMIDB_processed"

    # Create dataloader
    train_loader = get_dataloader(
        path=data_path,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    # Iterate over batches
    for batch_idx, (data, labels) in enumerate(train_loader):
        # data: (batch_size, 19, 1536) - 19 channels, 1536 samples (6s at 256 Hz)
        # labels: (batch_size,) - class indices 0-3
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels: {labels}")
        print(f"  Label distribution: {labels.unique(return_counts=True)}")

        if batch_idx >= 2:  # Just show first 3 batches
            break


# =============================================================================
# Example 2: With Metadata - Inspect Sample Sources
# =============================================================================

def example_with_metadata():
    """Load data with metadata to trace sample origins."""
    print("\n=== Example 2: Loading with Metadata ===")

    data_path = "/scratch/agjma/MMIDB_processed"

    # Create dataset with metadata
    dataset = DownstreamDataset(
        path=data_path,
        return_metadata=True
    )

    # Examine first sample
    data, label, metadata = dataset[0]

    print(f"Sample 0:")
    print(f"  Data shape: {data.shape}")
    print(f"  Label: {label} ({dataset.get_label_name(label)})")
    print(f"  Source file: {metadata['filename']}")
    print(f"  Time slice: {metadata['time_slice']}")

    # Print label distribution
    label_counts = dataset.get_label_counts()
    print(f"\nLabel distribution:")
    for label_idx, count in sorted(label_counts.items()):
        print(f"  {dataset.get_label_name(label_idx)} (class {label_idx}): {count} samples")


# =============================================================================
# Example 3: With Normalization Transform
# =============================================================================

def example_with_transform():
    """Apply z-score normalization transform."""
    print("\n=== Example 3: With Normalization Transform ===")

    data_path = "/scratch/agjma/MMIDB_processed"

    # Define z-score normalization
    def z_score_normalize(x):
        """Z-score normalize each channel independently."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    # Create dataset with transform
    dataset = DownstreamDataset(
        path=data_path,
        transform=z_score_normalize
    )

    # Check normalization
    data, label = dataset[0]
    print(f"Sample 0 after normalization:")
    print(f"  Data shape: {data.shape}")
    print(f"  Mean per channel: {data.mean(dim=-1)}")
    print(f"  Std per channel: {data.std(dim=-1)}")
    print(f"  Should be close to 0 mean, 1 std")


# =============================================================================
# Example 4: Binary Classification - Filter Classes
# =============================================================================

def example_binary_classification():
    """Filter to specific classes for binary classification."""
    print("\n=== Example 4: Binary Classification (2 classes) ===")

    data_path = "/scratch/agjma/MMIDB_processed"

    # Filter to only classes 0 and 1 (T1: left fist, T2: right fist)
    dataset = DownstreamDataset(
        path=data_path,
        label_filter=[0, 1]
    )

    print(f"Total samples after filtering: {len(dataset)}")

    # Check label distribution
    label_counts = dataset.get_label_counts()
    for label_idx, count in sorted(label_counts.items()):
        print(f"  Class {label_idx} ({dataset.get_label_name(label_idx)}): {count} samples")


# =============================================================================
# Example 5: Handle Class Imbalance - CHBMIT Seizure Detection
# =============================================================================

def example_chbmit_imbalanced():
    """Handle severe class imbalance with weighted sampling."""
    print("\n=== Example 5: CHBMIT with Weighted Sampling ===")

    data_path = "/scratch/agjma/CHBMIT_processed"

    # Create dataset
    dataset = DownstreamDataset(path=data_path)

    # Check class imbalance
    label_counts = dataset.get_label_counts()
    print(f"Class distribution (imbalanced):")
    for label_idx, count in sorted(label_counts.items()):
        print(f"  {dataset.get_label_name(label_idx)} (class {label_idx}): {count} samples")

    total = sum(label_counts.values())
    if len(label_counts) == 2:
        ratio = max(label_counts.values()) / min(label_counts.values())
        print(f"Imbalance ratio: {ratio:.1f}:1")

    # Create weighted sampler for balanced batches
    sampler = get_weighted_sampler(dataset)

    # Create dataloader with sampler (don't shuffle when using sampler)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Check batch balance
    print(f"\nSampling {3} balanced batches:")
    for batch_idx, (data, labels) in enumerate(loader):
        unique, counts = labels.unique(return_counts=True)
        print(f"Batch {batch_idx}: {dict(zip(unique.tolist(), counts.tolist()))}")

        if batch_idx >= 2:
            break


# =============================================================================
# Example 6: Simple Training Loop
# =============================================================================

def example_training_loop():
    """Demonstrate simple training loop."""
    print("\n=== Example 6: Simple Training Loop ===")

    data_path = "/scratch/agjma/MMIDB_processed"

    # Define a simple model
    class SimpleEEGClassifier(nn.Module):
        def __init__(self, n_channels=19, n_samples=1536, n_classes=4):
            super().__init__()
            self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=25, stride=1)
            self.pool = nn.MaxPool1d(4)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=25, stride=1)
            self.fc1 = nn.Linear(64 * 374, 128)  # Calculated based on conv/pool dims
            self.fc2 = nn.Linear(128, n_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleEEGClassifier(n_channels=19, n_samples=1536, n_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataloader
    train_loader = get_dataloader(
        path=data_path,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    # Training loop (1 epoch for demo)
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print(f"Training on device: {device}")
    print(f"Training for 1 epoch ({len(train_loader)} batches)...")

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, Acc={100. * correct / total:.2f}%")

        if batch_idx >= 49:  # Just show first 50 batches
            break

    print(f"\nFinal stats: Loss={total_loss / (batch_idx + 1):.4f}, "
          f"Acc={100. * correct / total:.2f}%")


# =============================================================================
# Example 7: Subject-Wise Splitting (Preventing Data Leakage)
# =============================================================================

def example_subject_wise_split():
    """Split dataset by subject so no subject appears in multiple splits."""
    from speed import DownstreamDataset, subject_wise_split, SUBJECT_EXTRACTORS

    data_path = "/scratch/agjma/SPEED/Processed/eegmmidb"

    dataset = DownstreamDataset(data_path)
    print(f"Total samples: {len(dataset)}")

    # Split by subject (70% train, 15% val, 15% test)
    train, val, test = subject_wise_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        subject_extractor=SUBJECT_EXTRACTORS['eegmmidb'],
        seed=42
    )

    print(f"Train: {len(train)} samples")
    print(f"Val:   {len(val)} samples")
    print(f"Test:  {len(test)} samples")

    # Verify no subject leakage
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader = DataLoader(val, batch_size=64)
    test_loader = DataLoader(test, batch_size=64)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples (comment out examples you don't want to run)."""

    # Basic examples (work with any dataset)
    # example_eegmmidb_basic()
    # example_with_metadata()
    # example_with_transform()
    # example_binary_classification()

    # Dataset-specific examples
    # example_chbmit_imbalanced()  # Requires CHBMIT processed data

    # Subject-wise splitting
    # example_subject_wise_split()  # Requires EEGMMIDB processed data

    # Training example
    # example_training_loop()  # Requires EEGMMIDB processed data and PyTorch

    print("\n=== Examples Complete ===")
    print("\nUncomment the examples you want to run in the main() function.")
    print("Make sure you've preprocessed the data first using:")
    print("  python scripts/preprocess_downstream.py --config configs/downstream/eegmmidb.yaml")
    print("  python scripts/preprocess_downstream.py --config configs/downstream/chbmit.yaml")


if __name__ == "__main__":
    main()
