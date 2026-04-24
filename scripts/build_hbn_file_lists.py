"""
Build per-task .set file lists for HBN downstream preprocessing.

The HBN-EEG dataset lives on an NFS mount where recursive globs across all
12 releases are slow and prone to stalling. This helper iterates each
release *one at a time* and walks a single subject's eeg/ directory,
which is markedly faster and more reliable.

For each configured downstream task, the script writes a .txt file listing
the absolute paths of the matching .set files. Each downstream yaml config
(configs/downstream/hbn_*.yaml) points `dataset_path` at its corresponding
.txt list.

Subjects whose participants.tsv availability flag for the task equals
"unavailable" are skipped. The --include-caution flag (default: true)
controls whether caution-flagged recordings are included.

Usage
-----
    python scripts/build_hbn_file_lists.py \\
        --dataset_root /dtu-compute/EEG_at_scale/HBN_EEG/hbn_eeg_original \\
        --out_dir /scratch/agjma/HBN_SPEED/lists

Writes:
    <out_dir>/ccd.txt            — all contrastChangeDetection_run-* files
    <out_dir>/rest.txt           — all RestingState files
    <out_dir>/surroundsupp.txt   — all surroundSupp_run-* files
    <out_dir>/symbolsearch.txt   — all symbolSearch files
    <out_dir>/seqlearning6.txt   — all seqLearning6target files
    <out_dir>/seqlearning8.txt   — all seqLearning8target files
    <out_dir>/movies.txt         — pooled DespicableMe + FunwithFractals +
                                   ThePresent + DiaryOfAWimpyKid files
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Task-name → (availability-column glob, file-suffix glob)
# Some tasks have multiple runs, each tracked by a separate availability column.
TASK_SPECS: Dict[str, Dict] = {
    "ccd": {
        "availability_cols": [
            "contrastChangeDetection_1",
            "contrastChangeDetection_2",
            "contrastChangeDetection_3",
        ],
        "file_patterns": [
            "_task-contrastChangeDetection_run-1_eeg.set",
            "_task-contrastChangeDetection_run-2_eeg.set",
            "_task-contrastChangeDetection_run-3_eeg.set",
        ],
    },
    "rest": {
        "availability_cols": ["RestingState"],
        "file_patterns": ["_task-RestingState_eeg.set"],
    },
    "surroundsupp": {
        "availability_cols": ["surroundSupp_1", "surroundSupp_2"],
        "file_patterns": [
            "_task-surroundSupp_run-1_eeg.set",
            "_task-surroundSupp_run-2_eeg.set",
        ],
    },
    "symbolsearch": {
        "availability_cols": ["symbolSearch"],
        "file_patterns": ["_task-symbolSearch_eeg.set"],
    },
    "seqlearning6": {
        "availability_cols": ["seqLearning6target"],
        "file_patterns": ["_task-seqLearning6target_eeg.set"],
    },
    "seqlearning8": {
        "availability_cols": ["seqLearning8target"],
        "file_patterns": ["_task-seqLearning8target_eeg.set"],
    },
    # Pooled 4-movie bucket — each movie has its own availability column and
    # filename; downstream hbn_movies.yaml treats the combined list as one
    # dataset with movie_name as the primary label.
    "movies": {
        "availability_cols": [
            "DespicableMe", "FunwithFractals", "ThePresent", "DiaryOfAWimpyKid",
        ],
        "file_patterns": [
            "_task-DespicableMe_eeg.set",
            "_task-FunwithFractals_eeg.set",
            "_task-ThePresent_eeg.set",
            "_task-DiaryOfAWimpyKid_eeg.set",
        ],
    },
}


def _availability_ok(value: str, include_caution: bool) -> bool:
    if value == "available":
        return True
    if value == "caution":
        return include_caution
    return False  # 'unavailable' or anything unexpected


def _collect_release(
    release_dir: Path,
    include_caution: bool,
) -> Dict[str, List[Path]]:
    """Walk one release directory and collect paths per task."""
    out: Dict[str, List[Path]] = defaultdict(list)

    participants_tsv = release_dir / "participants.tsv"
    if not participants_tsv.is_file():
        logging.warning("No participants.tsv in %s; skipping release.", release_dir)
        return out

    participants = pd.read_csv(participants_tsv, sep="\t")
    if "participant_id" not in participants.columns:
        logging.warning("participants.tsv in %s lacks participant_id column", release_dir)
        return out
    participants = participants.set_index("participant_id")

    for subject_id, row in participants.iterrows():
        subject_eeg = release_dir / subject_id / "eeg"
        if not subject_eeg.is_dir():
            continue
        for task_name, spec in TASK_SPECS.items():
            for avail_col, suffix in zip(spec["availability_cols"], spec["file_patterns"]):
                if avail_col not in participants.columns:
                    continue
                flag = str(row.get(avail_col, "unavailable"))
                if not _availability_ok(flag, include_caution):
                    continue
                candidate = subject_eeg / f"{subject_id}{suffix}"
                if candidate.is_file():
                    out[task_name].append(candidate)
                # Do not warn when missing — availability flag can be
                # "available" without the file being present on this mount.
    return out


def build_lists(
    dataset_root: Path,
    out_dir: Path,
    include_caution: bool = True,
    releases: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Generate per-task .txt lists; return {task: count}."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if releases is None:
        releases = sorted(p.name for p in dataset_root.glob("ds*") if p.is_dir())
    logging.info("Scanning %d release(s) under %s", len(releases), dataset_root)

    per_task: Dict[str, List[Path]] = defaultdict(list)
    for rel in releases:
        release_dir = dataset_root / rel
        if not release_dir.is_dir():
            logging.warning("Release dir not found: %s", release_dir)
            continue
        logging.info("  %s ...", rel)
        found = _collect_release(release_dir, include_caution)
        for task, paths in found.items():
            per_task[task].extend(paths)

    counts = {}
    for task, paths in per_task.items():
        paths.sort()
        out_path = out_dir / f"{task}.txt"
        with open(out_path, "w") as f:
            for p in paths:
                f.write(str(p) + "\n")
        logging.info("  wrote %d paths → %s", len(paths), out_path)
        counts[task] = len(paths)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/dtu-compute/EEG_at_scale/HBN_EEG/hbn_eeg_original"),
        help="Root dir containing ds005505..ds005516",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to write <task>.txt files into",
    )
    parser.add_argument(
        "--include-caution",
        dest="include_caution",
        action="store_true",
        default=True,
        help="Include caution-flagged recordings (default: true)",
    )
    parser.add_argument(
        "--exclude-caution",
        dest="include_caution",
        action="store_false",
        help="Exclude caution-flagged recordings",
    )
    parser.add_argument(
        "--releases",
        nargs="+",
        default=None,
        help="Subset of release dirs to scan (e.g. ds005505 ds005509)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    counts = build_lists(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        include_caution=args.include_caution,
        releases=args.releases,
    )
    total = sum(counts.values())
    logging.info("Done. Totals: %s (grand total %d files)", counts, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
