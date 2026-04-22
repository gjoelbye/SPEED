"""Tests for scripts.hdf5_combiner.HDF5CombinerDownstream.

Covers the three label layouts produced by the downstream pipeline:
  - int32 (N,)    — classification
  - float32 (N,)  — scalar regression (e.g. hbn_ccd_rt reaction time)
  - float32 (N,K) — multi-target regression (e.g. hbn_cbcl)

Plus the empty-combine fallback.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# scripts/ is not a package; add it to sys.path so we can import the combiner.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hdf5_combiner import HDF5CombinerDownstream  # noqa: E402


def _write_source(
    path: Path,
    data: np.ndarray,
    labels: np.ndarray,
    files: np.ndarray = None,
    file_idxs: np.ndarray = None,
    time_slices: np.ndarray = None,
    descriptions=None,
) -> None:
    """Emit a minimal source HDF5 in the layout produced by save_hdf5_with_labels."""
    n = data.shape[0]
    if files is None:
        files = np.array([f"rec_{path.stem}"], dtype=h5py.string_dtype())
    if file_idxs is None:
        file_idxs = np.zeros(n, dtype=np.int32)
    if time_slices is None:
        time_slices = np.zeros((n, 2), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data.astype(np.float32))
        f.create_dataset("labels", data=labels)
        f.create_dataset("file_idxs", data=file_idxs.astype(np.int32))
        f.create_dataset("files", data=files)
        f.create_dataset("time_slices", data=time_slices.astype(np.float32))
        f.attrs["descriptions"] = descriptions if descriptions is not None else []


def _combine(src_paths, out_dir):
    HDF5CombinerDownstream(
        [str(p) for p in src_paths], str(out_dir), max_file_size=2000
    ).combine()
    outs = sorted(out_dir.glob("combined_*.hdf5"))
    assert len(outs) == 1, f"expected 1 combined file, got {len(outs)}"
    return outs[0]


class TestHDF5CombinerDownstream:
    def test_int32_scalar_labels(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        labels_a = np.array([0, 1, 0], dtype=np.int32)
        labels_b = np.array([1, 1], dtype=np.int32)
        data_a = np.zeros((3, 4, 8), dtype=np.float32)
        data_b = np.ones((2, 4, 8), dtype=np.float32)
        _write_source(src_dir / "a.hdf5", data_a, labels_a, descriptions=["neg", "pos"])
        _write_source(src_dir / "b.hdf5", data_b, labels_b, descriptions=["neg", "pos"])

        combined = _combine([src_dir / "a.hdf5", src_dir / "b.hdf5"], out_dir)
        with h5py.File(combined, "r") as f:
            lbl = f["labels"][:]
            assert lbl.dtype == np.int32
            assert lbl.shape == (5,)
            np.testing.assert_array_equal(lbl, np.concatenate([labels_a, labels_b]))
            assert list(f.attrs["descriptions"]) == ["neg", "pos"]

    def test_float32_scalar_regression(self, tmp_path):
        """hbn_ccd_rt style — the case that the old int32 cast corrupted."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        labels_a = np.array([1.72, 0.84], dtype=np.float32)
        labels_b = np.array([2.13, 1.96, 1.61], dtype=np.float32)
        data_a = np.zeros((2, 4, 8), dtype=np.float32)
        data_b = np.zeros((3, 4, 8), dtype=np.float32)
        _write_source(src_dir / "a.hdf5", data_a, labels_a)
        _write_source(src_dir / "b.hdf5", data_b, labels_b)

        combined = _combine([src_dir / "a.hdf5", src_dir / "b.hdf5"], out_dir)
        with h5py.File(combined, "r") as f:
            lbl = f["labels"][:]
            assert lbl.dtype == np.float32
            assert lbl.shape == (5,)
            # Fractional values must survive — they would be truncated to ints
            # by the old hardcoded-int32 combiner.
            expected = np.concatenate([labels_a, labels_b])
            np.testing.assert_array_equal(lbl, expected)
            assert lbl[0] == pytest.approx(1.72)

    def test_float32_multitarget_regression(self, tmp_path):
        """hbn_cbcl style — (N, 4) float labels."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        labels_a = np.array(
            [[-0.603, -0.446, 1.248, 0.325], [-0.258, -0.425, 1.006, 0.509]],
            dtype=np.float32,
        )
        labels_b = np.array(
            [[0.1, 0.2, 0.3, 0.4], [-1.1, -1.2, -1.3, -1.4], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        data_a = np.zeros((2, 4, 8), dtype=np.float32)
        data_b = np.zeros((3, 4, 8), dtype=np.float32)
        _write_source(src_dir / "a.hdf5", data_a, labels_a)
        _write_source(src_dir / "b.hdf5", data_b, labels_b)

        combined = _combine([src_dir / "a.hdf5", src_dir / "b.hdf5"], out_dir)
        with h5py.File(combined, "r") as f:
            lbl = f["labels"][:]
            assert lbl.dtype == np.float32
            assert lbl.shape == (5, 4)
            expected = np.concatenate([labels_a, labels_b], axis=0)
            np.testing.assert_array_equal(lbl, expected)

    def test_empty_combine_float_multitarget(self, tmp_path):
        """Source file with zero windows → combined labels have correct empty shape."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Source with 0 windows but still float32 (N,4) dtype+trailing-shape info.
        _write_source(
            src_dir / "a.hdf5",
            data=np.zeros((0, 4, 8), dtype=np.float32),
            labels=np.zeros((0, 4), dtype=np.float32),
            files=np.array([], dtype=h5py.string_dtype()),
            file_idxs=np.zeros(0, dtype=np.int32),
            time_slices=np.zeros((0, 2), dtype=np.float32),
        )
        combined = _combine([src_dir / "a.hdf5"], out_dir)
        with h5py.File(combined, "r") as f:
            lbl = f["labels"][:]
            assert lbl.dtype == np.float32
            assert lbl.shape == (0, 4)
