# Changelog

## [2.0.0] - 2026-04-15

### Fixed
- Resampling now uses MNE's anti-aliased FIR filter instead of nearest-neighbor interpolation, preventing frequency aliasing artifacts when downsampling.
- Pretrain mode (fixed windowing) now applies bandpass filtering to full recordings before windowing, eliminating filter edge artifacts on short windows.

### Added
- Subject-wise dataset splitting (`subject_wise_split()`) to prevent data leakage between train/val/test sets. Includes per-dataset subject extractors (`SUBJECT_EXTRACTORS`) for 16 datasets.
- Configurable normalization (`normalize: zscore/minmax/robust`) as a first-class pipeline option, applied per-channel after resampling.
- Evaluation metrics module (`speed.metrics`): `balanced_accuracy`, `auroc`, `f1_score`, `cohens_kappa`, `classification_report`.
- `DownstreamDataset` and `get_dataloader` exported from `speed` package for convenient data loading.
- Test suite with 29 tests covering resampling, normalization, subject-wise splitting, and annotation parsing.
- CI/CD via GitHub Actions (Python 3.9, 3.10, 3.11).
- 17 downstream dataset configurations with annotation parsers.

### Changed
- Configs reorganized: `configs/pretrain/` (3 configs) and `configs/downstream/` (17 configs).
- Convert scripts moved to `scripts/converters/`.
- Checkpoint files removed from version control.
- All runtime dependencies declared in `pyproject.toml`.
- Pipeline internals split into `_run_recording_level()` and `_run_window_level()` for clearer separation of full-recording vs per-window operations.

### Improved
- Annotation parsers now log unparseable rows at DEBUG level instead of silently dropping them, with INFO-level summary counts.

## [0.1.0] - 2024-08-15

Initial release accompanying the SPEED paper at IEEE MLSP 2024.
