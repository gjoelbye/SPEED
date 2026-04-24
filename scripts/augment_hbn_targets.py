#!/usr/bin/env python
"""
Append a ``targets/`` group to already-preprocessed HBN HDF5 buckets.

Works identically on post-merge buckets
(``/scratch/agjma/HBN_SPEED_downstream_merged/<task>/combined_*.hdf5``) and
on per-shard trees (``/work3/agjma/HBN_SPEED_downstream/<task>/shard_*/combined_*.hdf5``):
it just walks every ``.hdf5`` under ``--bucket``.

Matches windows back to their original events.tsv rows by
``time_slices[:, 0] - event_tmin`` and computes every target defined in
``speed/downstream_targets.py`` for the task. Subject-level biometrics +
identity (sex, age, ehq_total, 4 CBCL factors, subject_id, task_name,
release_number, recording_onset_sec) are always added regardless of task.

Idempotent: skips any ``targets/<key>`` that already exists unless
``--force`` is passed.

Example — migrate the existing cyclopes ccd_rt bucket (event_tmin=+0.5):
    python scripts/augment_hbn_targets.py \\
        --bucket /scratch/agjma/HBN_SPEED_downstream_merged/ccd_rt \\
        --task ccd_rt \\
        --event-tmin 0.5 \\
        --raw-root /dtu-compute/EEG_at_scale/HBN_EEG/hbn_eeg_original \\
        --n-jobs 8

Migrate one DTU shard tree (event_tmin=-0.5 from hbn_ccd_correct.yaml):
    python scripts/augment_hbn_targets.py \\
        --bucket /work3/agjma/HBN_SPEED_downstream/ccd_correct \\
        --task ccd_correct \\
        --event-tmin -0.5 \\
        --raw-root /dtu-compute/EEG_at_scale/HBN_EEG/hbn_eeg_original
"""

import argparse
import logging
import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# Make speed/ importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from speed.downstream_targets import (
    GENERATOR_VERSION,
    TargetMismatchError,
    compute_all_targets,
)
from speed.utils import add_targets_to_existing_hdf5


# =============================================================================
# Stem index — map an HDF5 ``files`` entry back to a raw .set path
# =============================================================================

def build_stem_index(raw_root: Path, extension: str = ".set") -> Dict[str, Path]:
    """Walk ``raw_root`` and build ``{file_stem → full_path}``.

    HBN stems (``sub-NDARXXXXX_task-Y[_run-N]_eeg``) are globally unique across
    releases, so a single dict suffices. For ~3,600 subjects × ~10 tasks,
    walking the tree takes ~15 s on cyclopes (SSD-backed NFS).
    """
    logging.info(f"Scanning {raw_root} for {extension} files …")
    index: Dict[str, Path] = {}
    count = 0
    for dirpath, _dirs, files in os.walk(raw_root):
        for f in files:
            if f.endswith(extension):
                stem = f[: -len(extension)]
                index[stem] = Path(dirpath) / f
                count += 1
    logging.info(f"Indexed {count} raw files under {raw_root}")
    return index


def load_or_build_stem_index(
    raw_root: Path,
    cache_path: Optional[Path],
    extension: str = ".set",
) -> Dict[str, Path]:
    if cache_path and cache_path.is_file():
        try:
            with open(cache_path, "rb") as f:
                cached_root, index = pickle.load(f)
            if Path(cached_root) == raw_root:
                logging.info(f"Loaded stem index from cache: {cache_path} ({len(index)} entries)")
                return index
            else:
                logging.info(
                    f"Stem-index cache was built for {cached_root}, not {raw_root} — rebuilding."
                )
        except Exception as e:
            logging.warning(f"Stem-index cache at {cache_path} unreadable ({e}); rebuilding.")
    index = build_stem_index(raw_root, extension=extension)
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump((str(raw_root), index), f)
            logging.info(f"Saved stem index to {cache_path}")
        except Exception as e:
            logging.warning(f"Could not write stem-index cache: {e}")
    return index


# =============================================================================
# Per-HDF5 work
# =============================================================================

def augment_one_hdf5(
    h5_path: Path,
    task: str,
    event_tmin: float,
    event_tlen: float,
    movie_window_stride: Optional[float],
    stem_index: Dict[str, Path],
    force: bool,
    dry_run: bool,
) -> Tuple[Path, bool, Optional[str]]:
    """Augment a single HDF5 file. Returns (path, succeeded, error_msg)."""
    try:
        with h5py.File(h5_path, "r") as f:
            n = int(f["data"].shape[0])
            files = [
                s.decode("utf-8") if isinstance(s, bytes) else s
                for s in f["files"][:]
            ]
            file_idxs = f["file_idxs"][:]
            time_slices = f["time_slices"][:]
            existing_targets = set(f["targets"].keys()) if "targets" in f else set()

        # Short-circuit if every expected key already exists and not forcing.
        # We don't know all keys without computing — but if 'subject_id' already
        # exists and force is False, skip the file entirely as an optimisation.
        if existing_targets and not force:
            logging.info(
                f"{h5_path.name}: existing targets ({len(existing_targets)} keys) — "
                f"skip (use --force to recompute)."
            )
            return (h5_path, True, None)

        if dry_run:
            logging.info(f"[dry-run] would augment {h5_path} ({n} windows)")
            return (h5_path, True, None)

        # Compute once per source file (groups windows).
        window_event_onsets = time_slices[:, 0].astype(np.float64) - float(event_tmin)
        recording_onset_sec = time_slices[:, 0].astype(np.float32)

        merged: Dict[str, np.ndarray] = {}

        # Resolve each source stem → real raw .set path.
        for src_idx, stem in enumerate(files):
            mask = (file_idxs == src_idx)
            if not mask.any():
                continue
            if stem not in stem_index:
                raise FileNotFoundError(
                    f"Source stem {stem!r} (from {h5_path.name}) not in raw-root "
                    f"index. Re-run with a matching --raw-root or delete the stale "
                    f"--stem-index cache."
                )
            src_path = stem_index[stem]
            rows = np.where(mask)[0]
            per_src = compute_all_targets(
                src_path=src_path,
                window_onsets=window_event_onsets[rows],
                recording_onset_sec=recording_onset_sec[rows],
                task=task,
            )
            for key, arr in per_src.items():
                if key not in merged:
                    if arr.dtype.kind in ("U", "O", "S"):
                        merged[key] = np.full(n, "", dtype=object)
                    elif arr.dtype.kind in ("i", "u", "b"):
                        merged[key] = np.full(n, -1, dtype=arr.dtype)
                    else:
                        merged[key] = np.full(n, np.nan, dtype=np.float32)
                merged[key][rows] = arr

        attrs = {
            "event_tmin_used": float(event_tmin),
            "event_tlen_used": float(event_tlen),
            "generator_version": GENERATOR_VERSION,
        }
        if movie_window_stride is not None:
            attrs["movie_window_stride"] = float(movie_window_stride)

        add_targets_to_existing_hdf5(h5_path, merged, attrs=attrs, force=force)
        return (h5_path, True, None)
    except TargetMismatchError as e:
        return (h5_path, False, f"TargetMismatch: {e}")
    except Exception as e:
        return (h5_path, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# =============================================================================
# CLI
# =============================================================================

# Default event_tmin per task — matches the yaml configs already in use.
# Tasks not in this table require --event-tmin explicitly.
_TASK_DEFAULT_EVENT_TMIN: Dict[str, float] = {
    "ccd":           -0.5,  # new unified config
    "ccd_rt":         0.5,  # legacy
    "ccd_rt_4s":     -0.5,
    "ccd_correct":   -0.5,
    "symbolsearch":  -1.0,
    "surroundsupp":   0.0,
    "rest_ec_eo":     0.0,
    "seqlearning6":  -0.1,
    "seqlearning8":  -0.1,
    "movies":         0.0,
    "cbcl":           0.0,
}


def main():
    ap = argparse.ArgumentParser(
        description="Append per-window targets/ group to HBN HDF5 buckets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--bucket", required=True, type=Path,
                    help="Directory holding *.hdf5 files (walked recursively).")
    ap.add_argument("--task", required=True, choices=sorted(_TASK_DEFAULT_EVENT_TMIN),
                    help="Task tag — chooses which target extractor in speed/downstream_targets.py to call.")
    ap.add_argument("--event-tmin", type=float, default=None,
                    help="event_tmin used during preprocessing (window_start = event_onset + event_tmin). "
                         "Defaults to the canonical value for --task if unspecified.")
    ap.add_argument("--event-tlen", type=float, default=None,
                    help="event_tlen used during preprocessing (informational; written to targets.attrs).")
    ap.add_argument("--movie-window-stride", type=float, default=None,
                    help="Only for --task movies; written to targets.attrs.")
    ap.add_argument("--raw-root", type=Path,
                    default=Path("/dtu-compute/EEG_at_scale/HBN_EEG/hbn_eeg_original"),
                    help="Root of the HBN raw-data tree (for events.tsv + participants.tsv lookup).")
    ap.add_argument("--stem-index", type=Path, default=Path("/tmp/hbn_stem_index.pkl"),
                    help="Cache path for the {stem → raw .set path} lookup.")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="Parallel workers across HDF5 files.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing targets/<key> datasets.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List files that would be augmented; write nothing.")
    ap.add_argument("--log-level", default="INFO",
                    choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.bucket.is_dir():
        raise SystemExit(f"--bucket {args.bucket} is not a directory")
    if not args.raw_root.is_dir():
        raise SystemExit(f"--raw-root {args.raw_root} is not a directory")

    event_tmin = args.event_tmin if args.event_tmin is not None else _TASK_DEFAULT_EVENT_TMIN[args.task]
    event_tlen = args.event_tlen if args.event_tlen is not None else float("nan")

    # Enumerate HDF5 files under the bucket (sharded or flat).
    h5_files = sorted(args.bucket.rglob("*.hdf5")) + sorted(args.bucket.rglob("*.h5"))
    h5_files = [p for p in h5_files if p.is_file()]
    if not h5_files:
        raise SystemExit(f"No *.hdf5 files found under {args.bucket}")

    logging.info(
        f"Augment: bucket={args.bucket}  task={args.task}  event_tmin={event_tmin}  "
        f"files={len(h5_files)}  n_jobs={args.n_jobs}  force={args.force}  dry_run={args.dry_run}"
    )

    stem_index = load_or_build_stem_index(args.raw_root, args.stem_index)

    results: List[Tuple[Path, bool, Optional[str]]] = []
    if args.n_jobs <= 1 or args.dry_run:
        for p in h5_files:
            res = augment_one_hdf5(
                p, args.task, event_tmin, event_tlen, args.movie_window_stride,
                stem_index, args.force, args.dry_run,
            )
            results.append(res)
            _log_result(res)
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futures = {
                ex.submit(
                    augment_one_hdf5, p, args.task, event_tmin, event_tlen,
                    args.movie_window_stride, stem_index, args.force, args.dry_run,
                ): p
                for p in h5_files
            }
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                _log_result(res)

    n_ok = sum(1 for _p, ok, _e in results if ok)
    n_fail = len(results) - n_ok
    logging.info(f"Done. Succeeded: {n_ok}/{len(results)}  Failed: {n_fail}")
    if n_fail:
        for p, ok, err in results:
            if not ok:
                logging.error(f"FAILED {p}: {err}")
        raise SystemExit(1)


def _log_result(res: Tuple[Path, bool, Optional[str]]):
    p, ok, err = res
    if ok:
        logging.info(f"OK {p.name}")
    else:
        logging.error(f"FAIL {p}: {err}")


if __name__ == "__main__":
    main()
