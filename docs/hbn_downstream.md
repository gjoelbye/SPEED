# HBN downstream tasks in SPEED

The Healthy Brain Network (HBN-EEG) dataset is already supported as a
pretraining source via `configs/pretrain/hbn.yaml`. This doc describes the
six downstream task configs in `configs/downstream/hbn_*.yaml`, the
per-task label derivations they use, and how to preprocess the dataset
end-to-end.

## Dataset at a glance

HBN-EEG is curated as a BIDS dataset split across **12 OpenNeuro releases**
(`ds005505`…`ds005516`) containing **~3,602 subjects** (ages 5–22 y; 2,309 M /
1,293 F) and **~30k `.set` recordings**. All recordings were made at
500 Hz with 129 EEG channels (EGI Geodesic layout `E1`…`E128` + `Cz`
reference).

Every subject dir follows the BIDS layout

```
<release>/participants.tsv
<release>/sub-XXX/eeg/sub-XXX_task-<T>[_run-<N>]_eeg.set
<release>/sub-XXX/eeg/sub-XXX_task-<T>[_run-<N>]_events.tsv
<release>/sub-XXX/eeg/sub-XXX_task-<T>[_run-<N>]_channels.tsv
```

There are 10 BIDS task identifiers:

| BIDS task | Paradigm | Typical duration | Notes |
|---|---|---|---|
| `RestingState` | Eyes-open / eyes-closed | ~360 s | 5 EC + 6 EO instructions per run |
| `DespicableMe`, `DiaryOfAWimpyKid`, `FunwithFractals`, `ThePresent` | Passive movie watching | 121 / 173 / 167 / 205 s | fixed film length (mostly) |
| `contrastChangeDetection_run-{1,2,3}` | Active — flickering-disk detection | 180–870 s | ~25 trials per run |
| `surroundSupp_run-{1,2}` | Active — contrast suppression | 255–1080 s | 64 × 2.4 s stim_ON per run |
| `seqLearning6target`, `seqLearning8target` | Active — sequence recall | 220–1220 s | 5 learning blocks per run |
| `symbolSearch` | Active — symbol matching | 150–510 s | 6–30 trialResponse events |

`participants.tsv` attaches per-subject metadata (age, sex, handedness,
CBCL psychopathology scores) and per-task `available`/`caution`/`unavailable`
flags.

## CBCL subject-level labels

Every `participants.tsv` row carries four Child Behavior Checklist–derived
scores obtained from a bifactor model:

- `p_factor` — general psychopathology factor
- `attention` — attention dimension (ADHD-like)
- `internalizing` — mood / anxiety / withdrawal
- `externalizing` — impulsivity / conduct

Empirically, the values are **z-scored** (mean ≈ 0, SD ≈ 1, observed range
roughly [−2.5, +3.5]). Approximately **1.5 % of subjects (55/3,602) have
NaN CBCL scores**; those are skipped by `parse_hbn_cbcl`.

## Downstream configs

Each config copies the pretrain channel + filter block verbatim so the
downstream tensor shape matches the pretrain one. Resampling to 200 Hz
yields **(105 channels, 400 samples)** for the 2 s windows.

| File | `annotation_format` | Task type | Metric | tmin / tlen | Label |
|---|---|---|---|---|---|
| `hbn_ccd_rt.yaml` | `hbn_ccd_rt` | regression (1-D) | normalized RMSE | +0.5 / 2.0 s @ stimulus | `rt_from_stimulus` (seconds) |
| `hbn_ccd_correct.yaml` | `hbn_ccd_correct` | binary classification | balanced accuracy / AUROC | −0.5 / 2.0 s @ stimulus | 1 if feedback=smiley_face else 0 |
| `hbn_cbcl.yaml` | `hbn_cbcl` | regression (4-D) | normalized RMSE per dim | 0 / **4.0 s, 2 s stride** | `[p_factor, attention, internalizing, externalizing]` |
| `hbn_rest_ec_eo.yaml` | `hbn_rest_ec_eo` | binary classification | balanced accuracy | 0 / 2.0 s | `eyes_closed`=0, `eyes_open`=1 |
| `hbn_surroundsupp.yaml` | `hbn_surroundsupp` | 3-class classification | balanced accuracy | 0 / **2.4 s** @ `stim_ON` | `stimulus_cond ∈ {1,2,3}` → {0,1,2} |
| `hbn_symbolsearch.yaml` | `hbn_symbolsearch` | binary classification (imbalanced) | balanced accuracy / AUROC | −1.0 / 2.0 s @ `trialResponse` | 1 if user_answer == correct_answer else 0 |

### Why these window lengths

The window length for each config was chosen from the task's event timing,
not a default. Empirical evidence (ds005505, 15–20 subjects) behind each
choice:

| Config | Choice | Rationale |
|---|---|---|
| `hbn_ccd_rt` | **2.0 s** @ stim+0.5 | Matches `eeg2025/startkit/challenge_1.py` (`WINDOW_LEN=2.0`). CCD trial RT median is 1.6 s (p95 2.17 s) so the [stim+0.5, stim+2.5] window captures both stimulus and response in almost every valid trial. Inter-stim gap is median 6.8 s (p5 5.2 s) so even 4 s would fit — sticking with 2 s for starter-kit parity. |
| `hbn_ccd_correct` | **2.0 s** @ stim−0.5 | Same trial structure; window [stim−0.5, stim+1.5] centres on the stimulus and covers the median response. |
| `hbn_cbcl` | **4.0 s, 2 s stride** | Matches `eeg2025/startkit/challenge_2.py`, which samples 4 s windows and crops to 2 s at training time. Overlapping 2 s stride gives ~116 windows per 234 s CCD recording. |
| `hbn_rest_ec_eo` | **2.0 s**, 8 windows per EC/EO | EC cycle is 40 s, EO 20 s (very consistent). EEGDash's `hbn_ec_ec_reannotation` extracts 8 × 2 s windows at offsets {15,17,…,29} s (EC) and {5,7,…,19} s (EO), skipping instruction transients. Yields 40 EC + 48 EO = **88 windows per file**. Alternative designs (1 × 20 s or 4 × 4 s) are viable but have no starter-kit baseline. |
| `hbn_surroundsupp` | **2.4 s** @ `stim_ON` | `stim_ON` duration is always 2.4 s; 2 s would truncate the last 0.4 s of the stimulus. stim→next_stim gap is ~3.42 s so 2.4 s windows are non-overlapping. |
| `hbn_symbolsearch` | **2.0 s** centred on response | trialResponse → next-trial gap: p1 = 2.01 s, median 6.14 s. A 2 s window at response−1 is safe for > 99 % of trials. |

All windows are resampled from 500 Hz to 200 Hz (matching the pretrain
config), so the `data` tensor per window is `(105, tlen × 200)` — that is
`(105, 400)` for 2 s, `(105, 480)` for 2.4 s, and `(105, 800)` for 4 s.

### Label derivation details

**CCD reaction-time (`hbn_ccd_rt`)** — mirrors the eeg2025 Challenge-1
baseline. Each `contrastTrial_start` row defines a trial window bounded by
the next trial (or the recording end; `end_experiment` is empirically
never present). Within the trial we pair the first
`left_target`/`right_target` (stimulus onset) with the first subsequent
`left_buttonPress`/`right_buttonPress` (response onset). RT =
response_onset − stimulus_onset. Trials without either event (~17 % of
trials) and trials with RT < 0.25 s (observed p1 = 0.45 s; faster values
are dominated by anticipation / accidental presses) are dropped.

**CCD correctness (`hbn_ccd_correct`)** — the `feedback` column has three
values (empirical counts across 1,625 trials): `smiley_face` (1,225,
correct hit), `sad_face` (124, wrong button for a target), `non_target`
(343, button press outside a proper target window). Only smiley/sad trials
are emitted; `non_target` and missing-response trials are skipped.

**CBCL regression (`hbn_cbcl`)** — subject-level labels from
`participants.tsv`, attached to fixed-stride 2 s windows spanning the full
recording. The description format `cbcl_{p}_{att}_{int}_{ext}` parses
through SPEED's existing `_`-separated regression decoder as a 4-D float.
By default CBCL is attached to CCD recordings (matches eeg2025
Challenge-2) — point `dataset_path` at `rest.txt` / `surroundsupp.txt` /
etc. to attach CBCL to a different task's recordings instead.

**RestingState EC/EO (`hbn_rest_ec_eo`)** — mirrors EEGDash's
`hbn_ec_ec_reannotation`. After each `instructed_toCloseEyes` the parser
emits 2 s windows at offsets `{15, 17, 19, 21, 23, 25, 27, 29}` s. After
each `instructed_toOpenEyes`, 2 s windows at `{5, 7, 9, 11, 13, 15, 17,
19}` s. The typical layout of 5 EC + 6 EO instructions yields 40 + 48 =
**88 windows per file**. Windows that would extend past the recording end
are clipped (only relevant for the shortest ~343 s recordings).

**Surround suppression (`hbn_surroundsupp`)** — one 2 s window per
`stim_ON` row (64 per run). The label is the `stimulus_cond` column
(values 1, 2, 3 — verified present and complete empirically).

**symbolSearch (`hbn_symbolsearch`)** — one 2 s window per `trialResponse`
row (6–30 per recording, median 17). The label compares the numeric
`user_answer` and `correct_answer` columns. Expect heavy class imbalance:
subject-level accuracy ranges 57–100 % (median 92 %), so the trial-level
positive-class rate is roughly 90 %.

### What is NOT included

- **Passive movies** (`DespicableMe`, `DiaryOfAWimpyKid`, `FunwithFractals`,
  `ThePresent`) — no official downstream label exists. Keep them in the
  pretraining pool (`configs/pretrain/hbn.yaml`).
- **Sequence learning** (`seqLearning6target`, `seqLearning8target`) — the
  task emits only ~5 labelled recall events per file and the natural
  target is the sequence similarity between `user_answer` and
  `correct_answer` (e.g. Levenshtein distance). There is no starter-kit
  precedent and the metric choice is non-trivial; not included in the
  initial converter.

## Usage

### 1. Generate per-task file lists

Recursive globbing over NFS across 12 releases is unreliable. Use the
helper script, which walks one release at a time and consults each
subject's availability flags:

```bash
python scripts/build_hbn_file_lists.py \
    --dataset_root /dtu-compute/EEG_at_scale/HBN_EEG/hbn_eeg_original \
    --out_dir /scratch/agjma/HBN_SPEED/lists
```

Produces:

```
/scratch/agjma/HBN_SPEED/lists/
    ccd.txt
    rest.txt
    surroundsupp.txt
    symbolsearch.txt
```

Pass `--exclude-caution` to drop caution-flagged recordings (the
`symbolSearch` flag applies to ~41.8 % of its recordings — heavy filter).
Pass `--releases ds005505 ds005509` to run a subset.

### 2. Run downstream preprocessing per task

```bash
for cfg in hbn_ccd_rt hbn_ccd_correct hbn_cbcl hbn_rest_ec_eo \
           hbn_surroundsupp hbn_symbolsearch; do
    python scripts/preprocess_downstream.py \
        --config configs/downstream/${cfg}.yaml
done
```

Each run writes HDF5 batches to the corresponding `out_path`
(`/scratch/agjma/HBN_SPEED_downstream/<task>/`). A batch has:

- `data`: `(N, 105, 400)` float32 windows
- `labels`: `(N,)` int32 for classification, `(N,)` float32 for CCD RT, or
  `(N, 4)` float32 for CBCL
- `files`, `file_idxs`, `time_slices`: window-level provenance
- attr `descriptions`: human-readable label names

### 3. Train with `DownstreamDataset`

```python
from speed.dataloader import DownstreamDataset, subject_wise_split, SUBJECT_EXTRACTORS

ds = DownstreamDataset("/scratch/agjma/HBN_SPEED_downstream/ccd_rt")
train, val, test = subject_wise_split(
    ds,
    subject_extractor=SUBJECT_EXTRACTORS["hbn"],
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
)
```

The `hbn` subject extractor pulls `sub-NDAR<XXX>` out of every HDF5 file
stem, guaranteeing no cross-split subject leakage.

## Known edge cases

- **Concatenated paradigms.** A few recordings (notably in ds005505)
  contain trailing events from an adjacent paradigm (e.g. `seqLearning_start`
  late inside a `RestingState` file). All current parsers either read only
  task-specific event values or use fixed offsets that fall safely before
  the concatenation boundary.
- **EEGLAB `boundary` markers.** Every HBN branch in the pipeline
  dispatcher replaces the `.set` file's pre-existing annotations (via
  `raw.set_annotations`), so `boundary` / `break cnt` markers do not
  become spurious windows.
- **Short recordings.** The RestingState min observed is 343 s; the last
  EO window at +19 s reaches ≤ 367 s, so all standard windows fit. Any
  window that would overflow is clipped by the parser.
- **CBCL NaN** — `parse_hbn_cbcl` returns an empty list when any of the 4
  CBCL columns is NaN for this subject, so the dispatcher emits 0 windows
  and SPEED's downstream script logs the skip.
- **CCD trials without response.** Roughly 17 % of CCD trials lack a
  paired button press. `parse_hbn_ccd_rt` and `parse_hbn_ccd_correct` skip
  them silently.
