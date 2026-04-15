"""Tests for speed.dataloader subject-wise splitting."""

import numpy as np
import pytest

from speed.dataloader import (
    DownstreamDataset,
    subject_wise_split,
    SUBJECT_EXTRACTORS,
    _default_subject_extractor,
)


class TestDefaultSubjectExtractor:
    """Tests for the default filename-to-subject heuristic."""

    def test_eegmmidb_pattern(self):
        assert _default_subject_extractor("S001R03") == "S001"

    def test_chbmit_pattern(self):
        assert _default_subject_extractor("chb01_03") == "chb01"

    def test_sub_prefix_pattern(self):
        assert _default_subject_extractor("sub-001_task-rest") == "sub-001"

    def test_numeric_only(self):
        assert _default_subject_extractor("12345_session1") == "12345"

    def test_fallback_underscore_split(self):
        assert _default_subject_extractor("patient_001_rec_01") == "patient"


class TestSubjectExtractors:
    """Tests for dataset-specific subject extractors."""

    def test_eegmmidb_extractor(self):
        ext = SUBJECT_EXTRACTORS["eegmmidb"]
        assert ext("S001R03") == "S001"
        assert ext("S042R14") == "S042"

    def test_chbmit_extractor(self):
        ext = SUBJECT_EXTRACTORS["chbmit"]
        assert ext("chb01_03") == "chb01"
        assert ext("chb24_01") == "chb24"


class TestSubjectWiseSplit:
    """Tests for subject-wise dataset splitting."""

    def test_no_subject_leakage(self, tmp_hdf5):
        """No subject should appear in more than one split."""
        dataset = DownstreamDataset(tmp_hdf5, return_metadata=True)

        train, val, test = subject_wise_split(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
        )

        # Collect subject IDs per split
        def get_subjects(subset):
            subjects = set()
            for idx in subset.indices:
                _, _, meta = dataset[idx]
                # Extract subject from filename
                fname = meta["filename"]
                subj = _default_subject_extractor(fname)
                subjects.add(subj)
            return subjects

        train_subjects = get_subjects(train)
        val_subjects = get_subjects(val)
        test_subjects = get_subjects(test)

        # No overlap between any pair
        assert train_subjects.isdisjoint(val_subjects), (
            f"Train/val overlap: {train_subjects & val_subjects}"
        )
        assert train_subjects.isdisjoint(test_subjects), (
            f"Train/test overlap: {train_subjects & test_subjects}"
        )
        assert val_subjects.isdisjoint(test_subjects), (
            f"Val/test overlap: {val_subjects & test_subjects}"
        )

    def test_all_samples_accounted_for(self, tmp_hdf5):
        """All samples should appear in exactly one split."""
        dataset = DownstreamDataset(tmp_hdf5)
        train, val, test = subject_wise_split(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        total = len(train.indices) + len(val.indices) + len(test.indices)
        assert total == len(dataset)

    def test_invalid_ratios_raise(self, tmp_hdf5):
        dataset = DownstreamDataset(tmp_hdf5)
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            subject_wise_split(dataset, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_reproducibility(self, tmp_hdf5):
        """Same seed should produce the same split."""
        dataset = DownstreamDataset(tmp_hdf5)
        train1, val1, test1 = subject_wise_split(dataset, seed=42)
        train2, val2, test2 = subject_wise_split(dataset, seed=42)
        assert train1.indices == train2.indices
        assert val1.indices == val2.indices
        assert test1.indices == test2.indices

    def test_different_seeds_differ(self, tmp_hdf5):
        """Different seeds should produce different splits."""
        dataset = DownstreamDataset(tmp_hdf5)
        train1, _, _ = subject_wise_split(dataset, seed=42)
        train2, _, _ = subject_wise_split(dataset, seed=99)
        # With 5 subjects, different seeds should give different orderings
        assert train1.indices != train2.indices


class TestDownstreamDataset:
    """Tests for DownstreamDataset basic functionality."""

    def test_load_and_length(self, tmp_hdf5):
        dataset = DownstreamDataset(tmp_hdf5)
        assert len(dataset) == 20  # 5 subjects x 4 samples

    def test_getitem_shapes(self, tmp_hdf5):
        dataset = DownstreamDataset(tmp_hdf5)
        data, label = dataset[0]
        assert data.shape == (19, 256 * 5)
        assert label.dim() == 0  # scalar

    def test_label_filter(self, tmp_hdf5):
        dataset = DownstreamDataset(tmp_hdf5, label_filter=[0])
        # Only class 0 samples
        for i in range(len(dataset)):
            _, label = dataset[i]
            assert label.item() == 0

    def test_metadata(self, tmp_hdf5):
        dataset = DownstreamDataset(tmp_hdf5, return_metadata=True)
        data, label, meta = dataset[0]
        assert "filename" in meta
        assert "file_idx" in meta
        assert "time_slice" in meta

    def test_label_counts(self, tmp_hdf5):
        dataset = DownstreamDataset(tmp_hdf5)
        counts = dataset.get_label_counts()
        assert sum(counts.values()) == len(dataset)
