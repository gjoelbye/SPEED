"""Tests for speed/downstream_targets.py + the HDF5 targets/ round-trip.

Focused on:

- ``compute_all_targets`` returns the expected keys per task
- ``_match_onsets_to_events`` tolerance / sentinel semantics
- ``save_hdf5_with_labels`` + ``add_targets_to_existing_hdf5`` round-trip
- ``HDF5CombinerDownstream`` propagates the ``targets/`` group on merge
- ``DownstreamDataset(target_key=...)`` reads the right dataset
- ``subject_wise_split`` prefers stored ``targets/subject_id`` over stem regex
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from speed.dataloader import (
    DownstreamDataset,
    SUBJECT_EXTRACTORS,
    list_targets,
    subject_wise_split,
)
from speed.downstream_targets import (
    HARD_TOL_SEC,
    SOFT_TOL_SEC,
    TargetMismatchError,
    _match_onsets_to_events,
)
from speed.utils import add_targets_to_existing_hdf5, save_hdf5_with_labels
from hdf5_combiner import HDF5CombinerDownstream


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dummy_bucket(tmp_path: Path) -> Path:
    """Two HDF5 files emulating two subjects (sub-A, sub-B), each 3 windows."""
    for i, subject in enumerate(["sub-A", "sub-B"]):
        save_hdf5_with_labels(
            data_arrays=[np.full((4, 10), i, dtype=np.float32) for _ in range(3)],
            labels=[0, 1, 0],
            label_descriptions=["cls0", "cls1"],
            src_paths=[Path(f"{subject}_task-symbolSearch_eeg.set")],
            times=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
            indices=[0] * 3,
            dest_path=tmp_path / f"sh_{i:02d}.hdf5",
            extra_targets={
                "age":         np.array([7.0, 7.0, 7.0] if i == 0 else [9.0, 9.0, 9.0], dtype=np.float32),
                "subject_id":  np.array([subject] * 3, dtype=object),
                "sym_target_present": np.array([1, 0, 1], dtype=np.int8),
            },
            targets_attrs={"event_tmin_used": -1.0, "generator_version": "test"},
        )
    return tmp_path


# =============================================================================
# Unit tests
# =============================================================================

def test_match_onsets_to_events_basic():
    events = np.array([1.0, 5.0, 10.0])
    windows = np.array([1.00003, 5.0, 9.999])  # all within SOFT_TOL_SEC
    matched = _match_onsets_to_events(windows, events, Path("fake"), "test")
    assert list(matched) == [0, 1, 2]


def test_match_onsets_to_events_hard_fail():
    events = np.array([1.0, 5.0])
    windows = np.array([1.0, 100.0])  # second is way out of range
    with pytest.raises(TargetMismatchError):
        _match_onsets_to_events(windows, events, Path("fake"), "test")


def test_match_onsets_to_events_empty():
    """No candidate events → all windows unmatched (sentinel -1), no raise."""
    matched = _match_onsets_to_events(
        np.array([1.0, 2.0]), np.array([]), Path("fake"), "test"
    )
    assert list(matched) == [-1, -1]


def test_targets_roundtrip_write_then_read(tmp_path: Path):
    p = tmp_path / "t.h5"
    save_hdf5_with_labels(
        data_arrays=[np.zeros((4, 10), dtype=np.float32) for _ in range(3)],
        labels=[0, 1, 2],
        label_descriptions=["a", "b", "c"],
        src_paths=[Path("/x/foo.set"), Path("/x/bar.set"), Path("/x/baz.set")],
        times=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
        indices=[0, 1, 2],
        dest_path=p,
        extra_targets={
            "age": np.array([7.5, 8.2, 9.0], dtype=np.float32),
            "sex": np.array([1, 0, -1], dtype=np.int8),
            "subject_id": np.array(["sub-A", "sub-B", "sub-C"], dtype=object),
        },
        targets_attrs={"event_tmin_used": -0.5, "generator_version": "1"},
    )
    with h5py.File(p, "r") as f:
        assert set(f["targets"].keys()) == {"age", "sex", "subject_id"}
        assert np.allclose(f["targets/age"][:], [7.5, 8.2, 9.0])
        assert list(f["targets/sex"][:]) == [1, 0, -1]
        ids = [s.decode() if isinstance(s, bytes) else s for s in f["targets/subject_id"][:]]
        assert ids == ["sub-A", "sub-B", "sub-C"]
        assert f["targets"].attrs["event_tmin_used"] == -0.5


def test_add_targets_skip_existing(tmp_path: Path):
    """Default behaviour: pre-existing keys are skipped, not overwritten."""
    p = tmp_path / "t.h5"
    save_hdf5_with_labels(
        data_arrays=[np.zeros((4, 10), dtype=np.float32) for _ in range(3)],
        labels=[0, 1, 0],
        label_descriptions=["a", "b"],
        src_paths=[Path("/x/a.set")],
        times=[(0.0, 1.0)] * 3,
        indices=[0] * 3,
        dest_path=p,
        extra_targets={"age": np.array([7.0, 7.0, 7.0], dtype=np.float32)},
    )
    add_targets_to_existing_hdf5(p, {"age": np.array([99, 99, 99], dtype=np.float32)})
    with h5py.File(p, "r") as f:
        assert np.allclose(f["targets/age"][:], [7.0, 7.0, 7.0])


def test_add_targets_force_overwrite(tmp_path: Path):
    p = tmp_path / "t.h5"
    save_hdf5_with_labels(
        data_arrays=[np.zeros((4, 10), dtype=np.float32) for _ in range(3)],
        labels=[0, 1, 0],
        label_descriptions=["a", "b"],
        src_paths=[Path("/x/a.set")],
        times=[(0.0, 1.0)] * 3,
        indices=[0] * 3,
        dest_path=p,
        extra_targets={"age": np.array([7.0, 7.0, 7.0], dtype=np.float32)},
    )
    add_targets_to_existing_hdf5(
        p, {"age": np.array([99, 99, 99], dtype=np.float32)}, force=True
    )
    with h5py.File(p, "r") as f:
        assert np.allclose(f["targets/age"][:], [99.0, 99.0, 99.0])


def test_add_targets_length_mismatch(tmp_path: Path):
    p = tmp_path / "t.h5"
    save_hdf5_with_labels(
        data_arrays=[np.zeros((4, 10), dtype=np.float32) for _ in range(3)],
        labels=[0, 1, 0],
        label_descriptions=["a", "b"],
        src_paths=[Path("/x/a.set")],
        times=[(0.0, 1.0)] * 3,
        indices=[0] * 3,
        dest_path=p,
    )
    with pytest.raises(ValueError, match="length 2 != data length 3"):
        add_targets_to_existing_hdf5(p, {"age": np.array([1, 2], dtype=np.float32)})


def test_list_targets(dummy_bucket: Path):
    keys, attrs = list_targets(dummy_bucket)
    assert "targets/age" in keys
    assert "targets/subject_id" in keys
    assert "targets/sym_target_present" in keys
    assert attrs["event_tmin_used"] == -1.0


def test_dataset_target_key_default(dummy_bucket: Path):
    ds = DownstreamDataset(dummy_bucket)
    _x, y = ds[0]
    # First window of sub-A: label=0 (int) via legacy labels
    assert int(y.item()) == 0


def test_dataset_target_key_override(dummy_bucket: Path):
    ds = DownstreamDataset(dummy_bucket, target_key="targets/sym_target_present")
    # First window of sub-A: sym_target_present=1
    _x, y = ds[0]
    assert int(y.item()) == 1


def test_dataset_target_key_float_regression(dummy_bucket: Path):
    ds = DownstreamDataset(dummy_bucket, target_key="targets/age")
    _x, y = ds[0]  # sub-A, age=7.0
    assert abs(y.item() - 7.0) < 1e-5


def test_subject_wise_split_uses_stored_subject_id(dummy_bucket: Path):
    ds = DownstreamDataset(dummy_bucket)
    tr, va, _te = subject_wise_split(ds, 0.5, 0.5, 0.0)
    tr_ids = {_read_subject(ds, i) for i in tr.indices}
    va_ids = {_read_subject(ds, i) for i in va.indices}
    assert tr_ids.isdisjoint(va_ids), "Subject IDs must be disjoint across splits"
    assert tr_ids | va_ids == {"sub-A", "sub-B"}


def test_combiner_propagates_targets(dummy_bucket: Path, tmp_path: Path):
    out_dir = tmp_path / "merged"
    out_dir.mkdir()
    paths = sorted(str(p) for p in dummy_bucket.glob("*.hdf5"))
    HDF5CombinerDownstream(paths, str(out_dir), max_file_size=1).combine()
    merged = sorted(out_dir.glob("combined_*.hdf5"))
    assert merged, "no merged file produced"
    with h5py.File(merged[0], "r") as f:
        assert set(f["targets"].keys()) == {"age", "subject_id", "sym_target_present"}
        # Subject IDs: first 3 from sub-A, last 3 from sub-B
        subj = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["targets/subject_id"][:]
        ]
        assert subj == ["sub-A"] * 3 + ["sub-B"] * 3
        # Attributes propagated
        assert f["targets"].attrs["event_tmin_used"] == -1.0


def test_combiner_handles_partial_migration(tmp_path: Path):
    """A source without a targets/ group should get sentinel-filled rows."""
    save_hdf5_with_labels(
        data_arrays=[np.zeros((4, 10), dtype=np.float32) for _ in range(2)],
        labels=[0, 1],
        label_descriptions=["a", "b"],
        src_paths=[Path("sub-A_task-x_eeg.set")],
        times=[(0.0, 1.0)] * 2,
        indices=[0] * 2,
        dest_path=tmp_path / "with.h5",
        extra_targets={
            "age": np.array([7.0, 7.0], dtype=np.float32),
            "subject_id": np.array(["sub-A"] * 2, dtype=object),
        },
    )
    save_hdf5_with_labels(
        data_arrays=[np.zeros((4, 10), dtype=np.float32) for _ in range(2)],
        labels=[1, 0],
        label_descriptions=["a", "b"],
        src_paths=[Path("sub-B_task-x_eeg.set")],
        times=[(0.0, 1.0)] * 2,
        indices=[0] * 2,
        dest_path=tmp_path / "without.h5",
        # no extra_targets → no targets/ group
    )
    out = tmp_path / "merged"
    out.mkdir()
    HDF5CombinerDownstream(
        [str(tmp_path / "with.h5"), str(tmp_path / "without.h5")],
        str(out),
        max_file_size=1,
    ).combine()
    merged = sorted(out.glob("combined_*.hdf5"))
    with h5py.File(merged[0], "r") as f:
        subj = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["targets/subject_id"][:]
        ]
        assert subj == ["sub-A", "sub-A", "", ""]  # partial migration sentinel
        ages = f["targets/age"][:]
        assert ages[0] == 7.0 and ages[1] == 7.0
        assert np.isnan(ages[2]) and np.isnan(ages[3])


# =============================================================================
# Helpers
# =============================================================================

def _read_subject(ds: DownstreamDataset, global_idx: int) -> str:
    """Read stored subject_id for a given global dataset index."""
    fi, li = ds.index[global_idx]
    with h5py.File(ds.paths[fi], "r") as f:
        s = f["targets/subject_id"][li]
    return s.decode() if isinstance(s, bytes) else s
