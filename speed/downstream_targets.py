"""
Per-window downstream targets for HBN recordings.

This module is the single source of truth for "given a preprocessed HBN window,
what labels can we derive from events.tsv + participants.tsv?". Both the
preprocessing pipeline (forward path) and the migration script
(scripts/augment_hbn_targets.py, post-hoc path) call into these functions, so
forward and migrated HDF5s end up byte-identical in their ``targets/`` group.

Every function in this module has the same shape:

    task_targets(src_path: Path, window_onsets: np.ndarray, **kw) -> Dict[str, np.ndarray]

where ``window_onsets`` is the array of per-window event onsets (in seconds,
relative to recording start) derived as::

    window_onsets = time_slices[:, 0] - event_tmin

The returned dict has one entry per target key. Shapes are ``(N,)``. Dtype and
sentinel convention (missing / not-applicable values):

- int dtypes → ``-1``
- float dtypes → ``np.nan``
- string dtypes → ``""``

Plus two task-agnostic helpers that are unioned in for *every* HBN bucket:

- :func:`hbn_identity_targets` — ``subject_id``, ``task_name``, ``release_number``,
  ``recording_onset_sec`` (per-window).
- :func:`hbn_biometrics_targets` — ``sex``, ``age``, ``ehq_total``,
  ``p_factor``, ``attention``, ``internalizing``, ``externalizing`` (broadcast
  from participants.tsv to per-window).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from speed.annotations import (
    _HBN_CBCL_COLUMNS,
    _hbn_ccd_pair_trials,
    _hbn_find_participants_tsv,
    _hbn_load_participants,
    _hbn_read_events,
    _hbn_resolve_events_tsv,
    _hbn_subject_id,
)


# =============================================================================
# Constants + exceptions
# =============================================================================

# How close a window onset must be to its matching event.tsv onset before we
# consider it "matched". time_slices are stored as float32 and windowing rounds
# to the nearest sample — at 200 Hz that's 5 ms/sample, so a sub-sample ~2 ms
# gap is normal. A mismatch > HARD_TOL_SEC means the HDF5 was produced by a
# different config / different events.tsv → fail loudly rather than silently
# write wrong targets.
SOFT_TOL_SEC = 0.006   # 6 ms — > 1 sample at 200 Hz
HARD_TOL_SEC = 0.010   # 10 ms — hard fail

GENERATOR_VERSION = "1"

# Canonical string labels for movies — derived from filename "task-<X>_eeg".
HBN_MOVIE_NAMES = ("DespicableMe", "FunwithFractals", "ThePresent", "DiaryOfAWimpyKid")


class TargetMismatchError(ValueError):
    """Raised when window onsets don't align with any event row within HARD_TOL_SEC."""


# =============================================================================
# Helpers
# =============================================================================

def _match_onsets_to_events(
    window_onsets: np.ndarray,
    event_onsets: np.ndarray,
    src_path: Path,
    label: str,
) -> np.ndarray:
    """Return ``matched_idx`` — for each window, the index into event_onsets of the
    nearest event, or −1 if no event within ``HARD_TOL_SEC``.

    Emits a warning when gap > SOFT_TOL_SEC (useful signal about preprocessing drift);
    raises when gap > HARD_TOL_SEC (schema mismatch — migration would fill wrong labels).
    """
    if event_onsets.size == 0:
        # No candidate events — return all −1 (all windows unmatched).
        # Do NOT raise here: some HBN recordings genuinely lack the expected
        # events rows (e.g. symbolSearch session where the participant timed
        # out before any trialResponse).
        return np.full(window_onsets.shape[0], -1, dtype=np.int64)

    order = np.argsort(event_onsets)
    event_onsets = event_onsets[order]
    matched = np.full(window_onsets.shape[0], -1, dtype=np.int64)
    max_gap = 0.0
    for i, w in enumerate(window_onsets):
        j = int(np.searchsorted(event_onsets, w))
        candidates = []
        if j > 0:
            candidates.append(j - 1)
        if j < event_onsets.size:
            candidates.append(j)
        best_j, best_gap = -1, np.inf
        for c in candidates:
            gap = abs(event_onsets[c] - w)
            if gap < best_gap:
                best_gap, best_j = gap, c
        if best_gap > HARD_TOL_SEC:
            raise TargetMismatchError(
                f"{label}: window {i} @ {w:.6f}s has no event within "
                f"{HARD_TOL_SEC*1000:.0f} ms (nearest gap {best_gap*1000:.2f} ms). "
                f"Source: {src_path}"
            )
        matched[i] = order[best_j]  # restore original (unsorted) event index
        if best_gap > max_gap:
            max_gap = best_gap
    if max_gap > SOFT_TOL_SEC:
        logging.warning(
            f"{label}: max onset-match gap {max_gap*1000:.2f} ms > "
            f"{SOFT_TOL_SEC*1000:.0f} ms (src: {src_path.name})"
        )
    return matched


def _parse_stem_task_name(stem: str) -> str:
    """Extract ``<task>`` from an HBN filename like ``sub-XXX_task-<task>[_run-N]_eeg``."""
    m = re.search(r"task-([^_]+)", stem)
    return m.group(1) if m else ""


def _parse_stem_subject_id(stem: str) -> str:
    """Extract ``sub-NDARXXXXX`` from an HBN filename stem."""
    m = re.match(r"^(sub-[A-Z0-9]+)", stem)
    return m.group(1) if m else ""


def _safe_float(x) -> float:
    try:
        return float(x) if x not in ("", None) and not pd.isna(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _safe_sex(x) -> int:
    """``M`` → 1, ``F`` → 0, anything else → −1."""
    if x in ("M", "Male"):
        return 1
    if x in ("F", "Female"):
        return 0
    return -1


def _release_from_path(src_path: Path) -> str:
    """Walk up src_path looking for a ``ds00551X`` directory (HBN releases).

    Returns the release id (e.g. ``ds005511``) or ``""`` if not found.
    The yaml-declared participants.tsv column is ``release_number`` which uses
    ``R7``, ``R8``, … — we return that when available, falling back to the
    directory id.
    """
    for p in src_path.parents:
        if p.name.startswith("ds005"):
            return p.name
    return ""


# =============================================================================
# Identity + biometrics (applied to every HBN bucket regardless of task)
# =============================================================================

def hbn_identity_targets(
    src_path: Path,
    n_windows: int,
    recording_onset_sec: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-window subject/task/release identifiers."""
    stem = src_path.stem
    # ``stem`` may or may not include the _eeg suffix depending on the caller.
    # All our HBN recordings end in _eeg.set, so the stored h5 ``files`` entry
    # ends with ``_eeg``. Normalise.
    subject = _parse_stem_subject_id(stem) or _hbn_subject_id(src_path) or ""
    task = _parse_stem_task_name(stem)
    release = ""
    # Try participants.tsv first (authoritative for R7..R11 labels).
    try:
        ptsv = _hbn_find_participants_tsv(src_path)
        df = _hbn_load_participants(ptsv)
        if subject in df.index and "release_number" in df.columns:
            rel = df.loc[subject, "release_number"]
            if isinstance(rel, str):
                release = rel
    except (FileNotFoundError, ValueError):
        pass
    if not release:
        release = _release_from_path(src_path)

    subj_arr = np.full(n_windows, subject, dtype=object)
    task_arr = np.full(n_windows, task, dtype=object)
    rel_arr = np.full(n_windows, release, dtype=object)
    return {
        "subject_id": subj_arr,
        "task_name": task_arr,
        "release_number": rel_arr,
        "recording_onset_sec": recording_onset_sec.astype(np.float32, copy=False),
    }


def hbn_biometrics_targets(
    src_path: Path,
    n_windows: int,
) -> Dict[str, np.ndarray]:
    """Per-window broadcast of participants.tsv phenotype columns.

    Missing participant row or NaN values produce the sentinel for each dtype.
    """
    sex = -1
    age = np.nan
    ehq = np.nan
    p_factor = np.nan
    attention = np.nan
    internalizing = np.nan
    externalizing = np.nan

    subject = ""
    try:
        subject = _parse_stem_subject_id(src_path.stem) or _hbn_subject_id(src_path)
        ptsv = _hbn_find_participants_tsv(src_path)
        df = _hbn_load_participants(ptsv)
        if subject in df.index:
            row = df.loc[subject]
            sex = _safe_sex(row.get("sex", ""))
            age = _safe_float(row.get("age", np.nan))
            ehq = _safe_float(row.get("ehq_total", np.nan))
            p_factor = _safe_float(row.get("p_factor", np.nan))
            attention = _safe_float(row.get("attention", np.nan))
            internalizing = _safe_float(row.get("internalizing", np.nan))
            externalizing = _safe_float(row.get("externalizing", np.nan))
        else:
            logging.info(
                f"participants.tsv has no row for {subject} at {ptsv}; "
                f"biometrics → sentinel values."
            )
    except (FileNotFoundError, ValueError) as e:
        logging.info(f"Biometrics lookup failed for {src_path.name}: {e}")

    return {
        "sex": np.full(n_windows, sex, dtype=np.int8),
        "age": np.full(n_windows, age, dtype=np.float32),
        "ehq_total": np.full(n_windows, ehq, dtype=np.float32),
        "p_factor": np.full(n_windows, p_factor, dtype=np.float32),
        "attention": np.full(n_windows, attention, dtype=np.float32),
        "internalizing": np.full(n_windows, internalizing, dtype=np.float32),
        "externalizing": np.full(n_windows, externalizing, dtype=np.float32),
    }


# =============================================================================
# Task-specific: CCD
# =============================================================================

def hbn_ccd_targets(
    src_path: Path,
    window_onsets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """All six CCD targets.

    Expects ``window_onsets`` to be the original stim onsets (i.e.
    ``time_slices[:,0] - event_tmin``) — matches what the CCD parsers emit as
    annotation onsets.
    """
    events_tsv = _hbn_resolve_events_tsv(src_path)
    events_df = _hbn_read_events(events_tsv)

    trials = _hbn_ccd_pair_trials(events_df)  # (t0, stim_on, resp_on, fb) tuples
    stim_onsets = np.array([t[1] for t in trials], dtype=np.float64)

    matched = _match_onsets_to_events(
        window_onsets, stim_onsets, src_path, label="ccd"
    )

    N = window_onsets.shape[0]
    rt = np.full(N, np.nan, dtype=np.float32)
    correct = np.full(N, -1, dtype=np.int8)
    feedback3 = np.full(N, -1, dtype=np.int8)
    target_side = np.full(N, -1, dtype=np.int8)
    button_side = np.full(N, -1, dtype=np.int8)
    non_target = np.full(N, -1, dtype=np.int8)

    # Need side info from events — pre-compute maps from stim onset to side.
    stim_rows = events_df[events_df["value"].isin(["left_target", "right_target"])]
    stim_side_by_onset = {
        float(o): (0 if v == "left_target" else 1)
        for o, v in zip(stim_rows["onset"].to_numpy(), stim_rows["value"].to_numpy())
    }
    resp_rows = events_df[events_df["value"].isin(["left_buttonPress", "right_buttonPress"])]
    resp_side_by_onset = {
        float(o): (0 if v == "left_buttonPress" else 1)
        for o, v in zip(resp_rows["onset"].to_numpy(), resp_rows["value"].to_numpy())
    }

    for i, tidx in enumerate(matched):
        if tidx == -1:
            continue
        _t0, stim_on, resp_on, fb = trials[tidx]
        target_side[i] = stim_side_by_onset.get(stim_on, -1)
        # Only set RT / button side / correctness when we have a response.
        if resp_on is not None and not np.isnan(resp_on):
            rt[i] = float(resp_on - stim_on)
            button_side[i] = resp_side_by_onset.get(resp_on, -1)
        if fb == "smiley_face":
            correct[i] = 1
            feedback3[i] = 1
            non_target[i] = 0
        elif fb == "sad_face":
            correct[i] = 0
            feedback3[i] = 0
            non_target[i] = 0
        elif fb == "non_target":
            correct[i] = -1
            feedback3[i] = 2
            non_target[i] = 1

    return {
        "ccd_feedback_3class": feedback3,
        "ccd_correct": correct,
        "ccd_rt": rt,
        "ccd_target_side": target_side,
        "ccd_button_side": button_side,
        "ccd_non_target": non_target,
    }


# =============================================================================
# Task-specific: symbolSearch
# =============================================================================

def hbn_symbolsearch_targets(
    src_path: Path,
    window_onsets: np.ndarray,
) -> Dict[str, np.ndarray]:
    events_tsv = _hbn_resolve_events_tsv(src_path)
    events_df = _hbn_read_events(events_tsv)
    resp = events_df[events_df["value"] == "trialResponse"]
    event_onsets = resp["onset"].to_numpy(dtype=np.float64)
    ua = pd.to_numeric(resp["user_answer"], errors="coerce").to_numpy()
    ca = pd.to_numeric(resp["correct_answer"], errors="coerce").to_numpy()

    matched = _match_onsets_to_events(
        window_onsets, event_onsets, src_path, label="symbolsearch"
    )

    N = window_onsets.shape[0]
    user_correct = np.full(N, -1, dtype=np.int8)
    target_present = np.full(N, -1, dtype=np.int8)
    user_answer = np.full(N, -1, dtype=np.int8)

    for i, tidx in enumerate(matched):
        if tidx == -1:
            continue
        u, c = ua[tidx], ca[tidx]
        if pd.isna(u) or pd.isna(c):
            continue
        user_correct[i] = 1 if u == c else 0
        target_present[i] = int(c)
        user_answer[i] = int(u)

    return {
        "sym_user_correct": user_correct,
        "sym_target_present": target_present,
        "sym_user_answer": user_answer,
    }


# =============================================================================
# Task-specific: surroundSupp
# =============================================================================

def hbn_surroundsupp_targets(
    src_path: Path,
    window_onsets: np.ndarray,
) -> Dict[str, np.ndarray]:
    events_tsv = _hbn_resolve_events_tsv(src_path)
    events_df = _hbn_read_events(events_tsv)
    stim = events_df[events_df["value"] == "stim_ON"]
    event_onsets = stim["onset"].to_numpy(dtype=np.float64)
    cond = pd.to_numeric(stim["stimulus_cond"], errors="coerce").to_numpy()
    bg = pd.to_numeric(stim["background"], errors="coerce").to_numpy()
    fg = pd.to_numeric(stim["foreground_contrast"], errors="coerce").to_numpy()

    matched = _match_onsets_to_events(
        window_onsets, event_onsets, src_path, label="surroundsupp"
    )

    # Map foreground contrast values to class indices for discrete version.
    FG_TO_CLASS = {0.0: 0, 0.3: 1, 0.6: 2, 1.0: 3}

    N = window_onsets.shape[0]
    stim_cond = np.full(N, -1, dtype=np.int8)
    background = np.full(N, -1, dtype=np.int8)
    fg_contrast = np.full(N, np.nan, dtype=np.float32)
    fg_class = np.full(N, -1, dtype=np.int8)

    for i, tidx in enumerate(matched):
        if tidx == -1:
            continue
        if not pd.isna(cond[tidx]):
            c = int(cond[tidx])
            if c in (1, 2, 3):
                stim_cond[i] = c
        if not pd.isna(bg[tidx]):
            background[i] = int(bg[tidx])
        if not pd.isna(fg[tidx]):
            fv = float(fg[tidx])
            fg_contrast[i] = fv
            fg_class[i] = FG_TO_CLASS.get(round(fv, 1), -1)

    return {
        "surr_stimulus_cond": stim_cond,
        "surr_background": background,
        "surr_foreground_contrast": fg_contrast,
        "surr_fg_contrast_class": fg_class,
    }


# =============================================================================
# Task-specific: RestingState EC/EO
# =============================================================================

def hbn_rest_ec_eo_targets(
    src_path: Path,
    window_onsets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Derive EC/EO for each window.

    Existing windows are locked to ``instructed_toCloseEyes`` /
    ``instructed_toOpenEyes`` instruction onsets (+ fixed offsets) — NOT to
    individual dot events. So ``rest_dot_number`` / ``rest_dot_on_off`` come
    out as sentinels for the current configs (placeholder for a future
    dot-locked variant). EC/EO is inferred by nearest preceding instruction.
    """
    events_tsv = _hbn_resolve_events_tsv(src_path)
    events_df = _hbn_read_events(events_tsv)
    instr = events_df[events_df["value"].isin(
        ["instructed_toCloseEyes", "instructed_toOpenEyes"]
    )].copy()
    instr_onsets = instr["onset"].to_numpy(dtype=np.float64)
    instr_kind = np.array(
        [0 if v == "instructed_toCloseEyes" else 1 for v in instr["value"]],
        dtype=np.int8,
    )

    N = window_onsets.shape[0]
    ec_eo = np.full(N, -1, dtype=np.int8)
    if instr_onsets.size:
        order = np.argsort(instr_onsets)
        instr_onsets = instr_onsets[order]
        instr_kind = instr_kind[order]
        for i, w in enumerate(window_onsets):
            j = int(np.searchsorted(instr_onsets, w, side="right")) - 1
            if j >= 0:
                ec_eo[i] = int(instr_kind[j])

    # Placeholders; non-dot-locked windows keep sentinels.
    dot_number = np.full(N, -1, dtype=np.int8)
    dot_on_off = np.full(N, -1, dtype=np.int8)

    return {
        "rest_ec_eo": ec_eo,
        "rest_dot_number": dot_number,
        "rest_dot_on_off": dot_on_off,
    }


# =============================================================================
# Task-specific: seqLearning
# =============================================================================

def hbn_seqlearning_targets(
    src_path: Path,
    window_onsets: np.ndarray,
    n_targets: int = 6,
) -> Dict[str, np.ndarray]:
    """Targets for seqLearning{6,8}target.

    Windows are expected to be locked to ``dot_noK_ON`` events (codes 11..16
    or 11..18). Derives:

    - ``seq_dot_position`` (1..n_targets)
    - ``seq_learning_block`` (1..5) by walking events.tsv tracking most
      recent ``learningBlock_N`` event
    - ``seq_dot_on_off`` — 1 for these windows (placeholder for a future
      ON+OFF variant)
    - ``seq_target_count`` — constant == n_targets for every window
    - ``seq_sequence_hamming_acc`` — per-block ``user_answer``-vs-
      ``correct_answer`` Hamming agreement ∈ [0, 1], broadcast to every
      dot window within that block; NaN if the block has no user_answer
    """
    events_tsv = _hbn_resolve_events_tsv(src_path)
    events_df = _hbn_read_events(events_tsv)

    dot_on_vals = [f"dot_no{k}_ON" for k in range(1, n_targets + 1)]
    dot_rows = events_df[events_df["value"].isin(dot_on_vals)].copy()
    dot_rows["pos"] = dot_rows["value"].apply(lambda v: int(v.split("_no")[1].split("_")[0]))
    event_onsets = dot_rows["onset"].to_numpy(dtype=np.float64)

    # Walk events.tsv tracking (learning_block, per-block hamming acc).
    block_boundaries: List[Tuple[float, int, float]] = []  # (onset, block, hamming_acc)
    current_block = -1
    for _, row in events_df.iterrows():
        v = str(row["value"])
        if v.startswith("learningBlock_"):
            try:
                current_block = int(v.split("_")[1])
            except (IndexError, ValueError):
                continue
            hamming = _compute_block_hamming(row, events_df, n_targets)
            block_boundaries.append((float(row["onset"]), current_block, hamming))

    matched = _match_onsets_to_events(
        window_onsets, event_onsets, src_path, label=f"seqlearning{n_targets}"
    )

    N = window_onsets.shape[0]
    dot_position = np.full(N, -1, dtype=np.int8)
    learning_block = np.full(N, -1, dtype=np.int8)
    dot_on_off = np.full(N, 1, dtype=np.int8)  # all ON by construction
    target_count = np.full(N, n_targets, dtype=np.int8)
    hamming_acc = np.full(N, np.nan, dtype=np.float32)

    positions = dot_rows["pos"].to_numpy(dtype=np.int64)
    for i, tidx in enumerate(matched):
        if tidx == -1:
            continue
        dot_position[i] = int(positions[tidx])
        wo = float(window_onsets[i])
        # Find the last block boundary at or before this window.
        blk, ham = -1, np.nan
        for bo, b, h in block_boundaries:
            if bo <= wo:
                blk, ham = b, h
            else:
                break
        learning_block[i] = blk
        hamming_acc[i] = ham

    return {
        "seq_dot_position": dot_position,
        "seq_learning_block": learning_block,
        "seq_dot_on_off": dot_on_off,
        "seq_target_count": target_count,
        "seq_sequence_hamming_acc": hamming_acc,
    }


def _compute_block_hamming(
    block_row: pd.Series,
    events_df: pd.DataFrame,
    n_targets: int,
) -> float:
    """Parse ``user_answer`` / ``correct_answer`` strings from a learningBlock row
    and return Hamming agreement in [0, 1].

    Both values are dash-separated sequences (e.g. ``"1-4-2-3-6-5-6"``). We only
    compare up to ``min(len(user), len(correct))`` — trailing mismatches count
    as agreement=0 only when the lengths differ. Returns NaN if parsing fails.
    """
    ua = block_row.get("user_answer", "")
    ca = block_row.get("correct_answer", "")
    if pd.isna(ua) or pd.isna(ca) or ua == "n/a" or ca == "n/a":
        return float("nan")
    try:
        u = [int(x) for x in str(ua).split("-") if x]
        c = [int(x) for x in str(ca).split("-") if x]
    except ValueError:
        return float("nan")
    if not c:
        return float("nan")
    k = min(len(u), len(c))
    matches = sum(1 for i in range(k) if u[i] == c[i])
    return float(matches) / float(len(c))


# =============================================================================
# Task-specific: movies
# =============================================================================

def hbn_movie_targets(
    src_path: Path,
    window_onsets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Targets for DespicableMe / FunwithFractals / ThePresent / DiaryOfAWimpyKid.

    Windows are fixed strided (not event-locked), but we still need the
    ``video_start`` onset to compute per-window offset into the film.
    """
    movie_name = _parse_stem_task_name(src_path.stem) or ""
    if movie_name not in HBN_MOVIE_NAMES:
        logging.warning(
            f"movie_targets: unexpected task_name '{movie_name}' for {src_path.name}"
        )

    # video_start is always present once per recording.
    video_start = 0.0
    try:
        events_tsv = _hbn_resolve_events_tsv(src_path)
        events_df = _hbn_read_events(events_tsv)
        vs = events_df[events_df["value"] == "video_start"]
        if len(vs):
            video_start = float(vs["onset"].iloc[0])
    except FileNotFoundError:
        # Some movie recordings have no events.tsv; treat video_start as 0.
        logging.info(f"movie_targets: no events.tsv for {src_path.name}, video_start=0")

    N = window_onsets.shape[0]
    name_arr = np.full(N, movie_name, dtype=object)
    offset = (window_onsets - video_start).astype(np.float32)
    return {
        "movie_name": name_arr,
        "movie_window_offset_sec": offset,
    }


# =============================================================================
# Dispatcher
# =============================================================================

_TASK_TO_FN: Dict[str, Callable[..., Dict[str, np.ndarray]]] = {
    "ccd": hbn_ccd_targets,
    "ccd_rt": hbn_ccd_targets,
    "ccd_rt_4s": hbn_ccd_targets,  # legacy 4 s variant — same events, wider window
    "ccd_correct": hbn_ccd_targets,
    "symbolsearch": hbn_symbolsearch_targets,
    "surroundsupp": hbn_surroundsupp_targets,
    "rest_ec_eo": hbn_rest_ec_eo_targets,
    "seqlearning6": lambda p, o: hbn_seqlearning_targets(p, o, n_targets=6),
    "seqlearning8": lambda p, o: hbn_seqlearning_targets(p, o, n_targets=8),
    "movies": hbn_movie_targets,
    "cbcl": None,  # no task-specific extras; identity + biometrics only
}


def compute_all_targets(
    src_path: Path,
    window_onsets: np.ndarray,
    recording_onset_sec: np.ndarray,
    task: str,
) -> Dict[str, np.ndarray]:
    """Return ``{key → (N,) array}`` for every target applicable to ``task``.

    ``task`` is the short task tag as understood by the migration script and
    pipeline (``ccd`` / ``symbolsearch`` / ``cbcl`` / …). Always includes
    identity + biometrics. Raises ``KeyError`` for unknown tasks.
    """
    if task not in _TASK_TO_FN:
        raise KeyError(f"Unknown task {task!r}; known: {sorted(_TASK_TO_FN)}")
    n = window_onsets.shape[0]
    out: Dict[str, np.ndarray] = {}
    # Identity + biometrics first (available even if task-specific parse fails).
    out.update(hbn_identity_targets(src_path, n, recording_onset_sec))
    out.update(hbn_biometrics_targets(src_path, n))
    # Task-specific last.
    task_fn = _TASK_TO_FN[task]
    if task_fn is not None:
        out.update(task_fn(src_path, window_onsets))
    return out
