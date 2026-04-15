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
from speed.dataloader import (
    DownstreamDataset,
    subject_wise_split,
    SUBJECT_EXTRACTORS,
    get_dataloader,
    get_weighted_sampler,
)
from speed.metrics import (
    balanced_accuracy,
    auroc,
    f1_score,
    cohens_kappa,
    classification_report,
)
from speed.provenance import save_provenance
from speed.report import generate_report

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
    # Data loading and splitting
    "DownstreamDataset",
    "subject_wise_split",
    "SUBJECT_EXTRACTORS",
    "get_dataloader",
    "get_weighted_sampler",
    # Evaluation metrics
    "balanced_accuracy",
    "auroc",
    "f1_score",
    "cohens_kappa",
    "classification_report",
    # Provenance and reports
    "save_provenance",
    "generate_report",
]
