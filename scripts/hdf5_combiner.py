"""
HDF5 Combiner - Combines multiple HDF5 files into larger batched files.

Optimized for performance with list-based accumulation, 64MB chunk cache,
and single-pass array conversion.
"""

from pathlib import Path
from os.path import join
from glob import glob
from typing import List, Optional, Dict, Any

import numpy as np
import h5py
from tqdm import tqdm


class HDF5Combiner:
    """
    Combines multiple HDF5 files into larger batched files.
    
    Subclass and override _get_extra_metadata(), _init_extra_lists(),
    _extend_extra_lists(), and _save_extra_datasets() for additional fields.
    """
    
    HDF5_CACHE_BYTES = 64 * 1024 * 1024
    HDF5_CACHE_SLOTS = 10007
    
    def __init__(
        self,
        src_paths: List[str],
        out_dir: str,
        max_file_size: int = 2000
    ):
        self.src_paths = src_paths
        self.out_dir = out_dir
        self.max_file_size = max_file_size  # MB
        
        self._file_idx = 0
        self._current_size = 0.0  # MB
        self._data_idx = 0
        self._file: Optional[h5py.File] = None
        
        self._shape = self._get_shape()
        self._n_windows = int(
            max_file_size * 1e6 / (np.prod(self._shape) * np.float32().itemsize)
        )
        
        # Core metadata lists
        self._file_idxs: List[int] = []
        self._files: List[str] = []
        self._time_slices: List[List[float]] = []
        
        self._get_extra_metadata()
    
    def _get_shape(self) -> tuple:
        with h5py.File(self.src_paths[0], 'r') as f:
            return f['data'].shape[1:]
    
    def _get_extra_metadata(self) -> None:
        """Override to load additional metadata from source files."""
        pass
    
    def _init_extra_lists(self) -> None:
        """Override to initialize additional list buffers."""
        pass
    
    def _extend_extra_lists(self, src_file: h5py.File) -> None:
        """Override to extend additional lists from source file data."""
        pass
    
    def _save_extra_datasets(self) -> None:
        """Override to save additional datasets/attributes to output file."""
        pass
    
    def _remap_file_indices(
        self,
        files: List[str],
        idxs: List[int]
    ) -> tuple:
        """Remove unused files and remap indices accordingly."""
        used = set(idxs)
        to_remove = [i for i in range(len(files)) if i not in used]
        
        # Build index map
        idx_map = {}
        offset = 0
        for i in range(len(files)):
            if i in to_remove:
                offset += 1
            else:
                idx_map[i] = i - offset
        
        # Remove unused files (reverse order to preserve indices)
        for i in reversed(to_remove):
            del files[i]
        
        remapped = [idx_map[i] for i in idxs if i in idx_map]
        
        return (
            np.array(files, dtype=h5py.string_dtype()),
            np.array(remapped, dtype=np.int32)
        )
    
    def _create_file(self, initial_size: Optional[int] = None) -> None:
        if self._file is not None:
            self._file.close()
        
        path = join(self.out_dir, f'combined_{self._file_idx:05d}.hdf5')
        if Path(path).exists():
            raise FileExistsError(f"Output file exists: {path}")
        
        self._file = h5py.File(
            path, 'w',
            rdcc_nbytes=self.HDF5_CACHE_BYTES,
            rdcc_nslots=self.HDF5_CACHE_SLOTS
        )
        
        size = initial_size or self._n_windows
        self._file.create_dataset(
            "data",
            shape=(size, *self._shape),
            chunks=(1, *self._shape),
            maxshape=(None, *self._shape),
            dtype=np.float32,
            fletcher32=True
        )
        
        self._file_idx += 1
        self._current_size = 0.0
        self._data_idx = 0
        
        self._file_idxs = []
        self._files = []
        self._time_slices = []
        self._init_extra_lists()
    
    def _save_and_close(self) -> None:
        self._file['data'].resize((self._data_idx, *self._shape))
        
        file_idxs = np.array(self._file_idxs, dtype=np.int32) if self._file_idxs else np.empty(0, dtype=np.int32)
        time_slices = np.vstack(self._time_slices) if self._time_slices else np.empty((0, 2), dtype=np.float32)
        
        files_arr, file_idxs = self._remap_file_indices(self._files, file_idxs.tolist())
        
        assert len(time_slices) == len(file_idxs) == self._data_idx, \
            f"Length mismatch: time_slices={len(time_slices)}, file_idxs={len(file_idxs)}, data={self._data_idx}"
        
        self._file.create_dataset("file_idxs", data=file_idxs, dtype=np.int32, fletcher32=True)
        self._file.create_dataset("files", data=files_arr, dtype=h5py.string_dtype())
        self._file.create_dataset("time_slices", data=time_slices, dtype=np.float32, fletcher32=True)
        
        self._save_extra_datasets()
    
    def _add_data(
        self,
        data: np.ndarray,
        file_idxs: np.ndarray,
        files: np.ndarray,
        time_slices: np.ndarray
    ) -> None:
        size_mb = data.nbytes / 1e6
        
        if self._current_size + size_mb > self.max_file_size:
            self._save_and_close()
            self._create_file()
        
        n = data.shape[0]
        capacity = self._file['data'].shape[0]
        
        if self._data_idx + n > capacity:
            new_cap = max(capacity + self._n_windows, self._data_idx + n)
            self._file['data'].resize((new_cap, *self._shape))
        
        self._file['data'][self._data_idx:self._data_idx + n] = data
        self._data_idx += n
        
        # Offset indices by current file count
        offset = len(self._files)
        if offset > 0:
            file_idxs = file_idxs + offset
        
        self._file_idxs.extend(file_idxs.tolist() if isinstance(file_idxs, np.ndarray) else file_idxs)
        self._files.extend(files.tolist() if isinstance(files, np.ndarray) else files)
        self._time_slices.extend(time_slices.tolist() if isinstance(time_slices, np.ndarray) else time_slices)
        
        self._current_size += size_mb
    
    def _read_field(self, f: h5py.File, name: str) -> np.ndarray:
        """Read field from dataset or attrs (backwards compatibility)."""
        return f[name][:] if name in f else f.attrs[name]
    
    def combine(self) -> None:
        self._create_file()
        failed = []
        
        for path in tqdm(self.src_paths):
            try:
                with h5py.File(path, 'r') as f:
                    self._add_data(
                        f['data'][:],
                        self._read_field(f, 'file_idxs'),
                        self._read_field(f, 'files'),
                        self._read_field(f, 'time_slices')
                    )
                    self._extend_extra_lists(f)
            except Exception as e:
                tqdm.write(f"ERROR: {path}\n       {e}")
                failed.append(path)
        
        if self._file is not None:
            self._save_and_close()
            self._file.close()
            self._file = None
        
        print(f"Combined into {self._file_idx} file(s).")
        if failed:
            print(f"WARNING: {len(failed)} file(s) failed:")
            for f in failed:
                print(f"  - {f}")


class HDF5CombinerDownstream(HDF5Combiner):
    """HDF5Combiner with labels + descriptions + optional ``targets/`` group.

    Labels may be:
      - int32 (N,) for classification,
      - float32 (N,) for scalar regression (e.g. reaction time),
      - float32 (N, K) for multi-target regression (e.g. CBCL 4-vectors).

    Dtype and trailing shape are inferred from the first source file and
    preserved end-to-end; labels from each source file are appended as arrays
    and concatenated on save, so no per-element cast ever happens.

    If the first source file contains a ``targets/`` group (written by the
    multi-target pipeline or the migration script), every per-window target
    dataset inside it is propagated through the merge, along with any
    group-level attributes. Sources missing a key that the first file has get
    dtype-appropriate sentinel fill (``-1`` for int, NaN for float, ``""``
    for string) so partially-migrated source trees still merge cleanly.
    """

    def _get_extra_metadata(self) -> None:
        with h5py.File(self.src_paths[0], 'r') as f:
            self._descriptions = f.attrs['descriptions']
            ds = f['labels']
            self._labels_dtype = ds.dtype
            self._labels_trailing_shape = ds.shape[1:]  # () for 1-D, (K,) for multi-target

            # Optional ``targets/`` group — infer schema from first source.
            self._targets_schema: Dict[str, tuple] = {}  # key -> (dtype, trailing_shape)
            self._targets_attrs: Dict[str, Any] = {}
            if 'targets' in f:
                grp = f['targets']
                for k, v in grp.items():
                    self._targets_schema[k] = (v.dtype, tuple(v.shape[1:]))
                for k, v in grp.attrs.items():
                    self._targets_attrs[k] = v

    def _init_extra_lists(self) -> None:
        # One array per source file — concat preserves dtype + shape.
        self._label_arrays: List[np.ndarray] = []
        # One list per target key.
        self._target_arrays: Dict[str, List[np.ndarray]] = {
            k: [] for k in self._targets_schema
        }

    def _extend_extra_lists(self, src_file: h5py.File) -> None:
        self._label_arrays.append(src_file['labels'][:])
        if not self._targets_schema:
            return
        n = src_file['data'].shape[0]
        src_grp = src_file.get('targets')
        for key, (dtype, trailing) in self._targets_schema.items():
            if src_grp is not None and key in src_grp:
                self._target_arrays[key].append(src_grp[key][:])
            else:
                # Partial migration: fill with sentinel of the expected shape.
                self._target_arrays[key].append(_sentinel_fill((n,) + trailing, dtype))

    def _save_extra_datasets(self) -> None:
        if self._label_arrays:
            labels = np.concatenate(self._label_arrays, axis=0)
        else:
            labels = np.empty(
                (0,) + self._labels_trailing_shape, dtype=self._labels_dtype
            )

        assert labels.shape[0] == self._data_idx, (
            f"Labels length mismatch: {labels.shape[0]} vs {self._data_idx}"
        )

        self._file.create_dataset(
            "labels", data=labels, dtype=self._labels_dtype, fletcher32=True
        )
        self._file.attrs['descriptions'] = self._descriptions

        # Write the ``targets/`` group if we have one.
        if self._targets_schema:
            grp = self._file.create_group('targets')
            for key, arrays in self._target_arrays.items():
                if arrays:
                    concat = np.concatenate(arrays, axis=0)
                else:
                    dtype, trailing = self._targets_schema[key]
                    concat = np.empty((0,) + trailing, dtype=dtype)
                assert concat.shape[0] == self._data_idx, (
                    f"Target {key!r} length {concat.shape[0]} != {self._data_idx}"
                )
                if concat.dtype.kind in ('O', 'U', 'S'):
                    # Variable-length strings can't be fletcher32-compressed.
                    str_arr = np.array(
                        [s.decode('utf-8') if isinstance(s, bytes)
                         else ("" if s is None else str(s))
                         for s in concat],
                        dtype=h5py.string_dtype(),
                    )
                    grp.create_dataset(key, data=str_arr, dtype=h5py.string_dtype())
                else:
                    grp.create_dataset(
                        key, data=concat, dtype=concat.dtype, fletcher32=True,
                    )
            for k, v in self._targets_attrs.items():
                grp.attrs[k] = v

        self._label_arrays = []  # free memory before the next output file
        self._target_arrays = {k: [] for k in self._targets_schema}


def _sentinel_fill(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Return a sentinel-filled array for a target missing from some source."""
    kind = dtype.kind
    if kind == 'f':
        out = np.empty(shape, dtype=dtype)
        out[...] = np.nan
        return out
    if kind in ('i', 'u'):
        return np.full(shape, -1, dtype=dtype)
    if kind in ('O', 'U', 'S'):
        return np.full(shape, "", dtype=dtype)
    if kind == 'b':
        return np.zeros(shape, dtype=dtype)
    raise TypeError(f"Cannot build sentinel for dtype {dtype!r}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine HDF5 files')
    parser.add_argument('src_dir', help='Directory containing HDF5 files')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--max_file_size', type=int, default=2000, help='Max file size (MB)')
    parser.add_argument('--downstream', action='store_true', help='Use downstream combiner with labels')
    args = parser.parse_args()
    
    paths = sorted(glob(join(args.src_dir, '**/*.hdf5'), recursive=True))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    cls = HDF5CombinerDownstream if args.downstream else HDF5Combiner
    cls(paths, args.out_dir, args.max_file_size).combine()


if __name__ == "__main__":
    main()
