"""Public exports for the speed package."""

from speed.pipeline import Pipeline, BasePipeline, PretrainPipeline
from speed.utils import (
    save_hdf5,
    write_bdf_from_raw,
    write_edf_from_raw,
    write_set_from_raw,
    write_raw_to_file,
    load_montage,
    copy_non_eeg_files,
    DEFAULT_EXCLUDE_PATTERNS,
)
from speed.methods import PreprocessMethods

__all__ = [
    # Pipelines
    "Pipeline",
    "BasePipeline",
    "PretrainPipeline",
    # I/O functions
    "save_hdf5",
    "write_bdf_from_raw",
    "write_edf_from_raw",
    "write_set_from_raw",
    "write_raw_to_file",
    "load_montage",
    # File utilities
    "copy_non_eeg_files",
    "DEFAULT_EXCLUDE_PATTERNS",
    # Methods
    "PreprocessMethods",
]
