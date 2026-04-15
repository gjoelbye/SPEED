# Downstream Datasets Reference

All datasets are processed to **200 Hz** with SPEED and saved as batched HDF5 files.
Each dataset preserves its native channel set (SPEED finds bad channels and interpolates).

## Quick Reference

| # | Dataset | Task | Classes | Duration | Channels | Native Hz | Line Hz | Format | Downloaded |
|---|---------|------|---------|----------|----------|-----------|---------|--------|------------|
| 1 | PhysioNet-MI | Motor Imagery | 4 | 4 s | 64 | 160 | 60 | .edf | Yes |
| 2 | CHB-MIT | Seizure Detection | 2 | 10 s | 19* | 256 | 60 | .edf | Yes |
| 3 | MentalArithmetic | Workload | 2 | 5 s | 19 | 500 | 50 | .edf | Yes |
| 4 | HMC | Sleep Staging | 5 | 30 s | 4 | 256 | 50 | .edf | Yes |
| 5 | ISRUC | Sleep Staging | 5 | 30 s | 6 | 200 | 50 | .edf | Yes |
| 6 | Mumtaz2016 | Depression | 2 | 5 s | 19 | 256 | 50 | .edf | Yes (179/191) |
| 7 | BCIC-IV-2a | Motor Imagery | 4 | 4 s | 22 | 250 | 50 | .gdf | Yes |
| 8 | SHU-MI | Motor Imagery | 2 | 4 s | 58 | 250 | 50 | .mat* | Yes |
| 9 | MoBI | Gait Prediction | reg.(12) | 2 s | 64 | 100 | 50 | .txt* | Yes |
| 10 | FACED | Emotion | 9 | 10 s | 32 | 250 | 50 | .mat* | No |
| 11 | SEED-V | Emotion | 5 | 1 s | 62 | 1000 | 50 | .mat* | No |
| 12 | BCIC2020-IV-3 | Imagined Speech | 5 | 3 s | 64 | 256 | 60 | .mat* | No |
| 13 | SEED-VIG | Vigilance | reg. | 8 s | 17 | 200 | 50 | .mat* | No |
| 14 | TUEV | Event Detection | 6 | 5 s | 19* | 256 | 60 | .edf | No |
| 15 | TUAB | Abnormality | 2 | 10 s | 19* | 256 | 60 | .edf | No |

\* after bipolar-to-monopolar conversion or .mat/.txt conversion to EDF required

---

## Downloaded Datasets (verified)

All at `/scratch/agjma/SPEED/Original/`

### 1. PhysioNet-MI / EEGMMIDB (Motor Imagery)

- **Path:** `Original/eegmmidb/` (1353 files, 1.2 GB)
- **Download:** `wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/`
- **Subjects:** 109 (70 train / 19 val / 20 test)
- **Channels:** 64 (BCI2000 system). Raw names have EDF padding dots (`Fc5.`, `C3..`), handled by `standardize_channel_names: true`
- **Sfreq:** 160 Hz
- **Labels:** EDF+ embedded annotations (T0=rest, T1=left fist, T2=right fist, T3=both feet, T4=both fists)
- **Config:** `configs/downstream/eegmmidb.yaml`
- **Status:** Ready to process

### 2. CHB-MIT (Seizure Detection)

- **Path:** `Original/chbmit/` (312 files, 17 GB)
- **Download:** `wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/`
- **Subjects:** 23 (19 train / 2 val / 2 test)
- **Channels:** 23 bipolar (FP1-F7, F7-T7, ...) → 19 monopolar after `bipolar_to_monopolar: true`
- **Sfreq:** 256 Hz
- **Labels:** `chb##-summary.txt` seizure annotations parsed by `parse_chbmit_summary()`
- **Config:** `configs/downstream/chbmit.yaml`
- **Notes:** Severe class imbalance (~1:287 seizure:non-seizure), use weighted sampling
- **Status:** Ready to process

### 3. MentalArithmetic / EEGMAT (Mental Workload)

- **Path:** `Original/eegmat/` (77 files, 174 MB)
- **Download:** `wget -r -N -c -np https://physionet.org/files/eeg-during-mental-arithmetic-tasks/1.0.0/`
- **Subjects:** 36 (28 train / 4 val / 4 test)
- **Channels:** 21 raw (`EEG Fp1`, ..., `EEG A2-A1`, `ECG ECG`). After `standardize_channel_names`: 19 EEG (A2-A1 and ECG dropped by montage, T3→T7, T4→T8)
- **Sfreq:** 500 Hz
- **Labels:** From filename: `Subject##_1.edf` = baseline, `Subject##_2.edf` = arithmetic. Tiled into 5s windows by `parse_eegmat_annotations()`
- **Config:** `configs/downstream/eegmat.yaml`
- **Status:** Ready to process (verified parser produces 36 windows from 182s recording)

### 4. HMC (Sleep Staging)

- **Path:** `Original/hmc/` (460 files, 16 GB)
- **Download:** `wget -r -N -c -np https://physionet.org/files/hmc-sleep-staging/1.1/`
- **Subjects:** 151 (complete)
- **Channels:** 8 raw (`EEG F4-M1`, `EEG C4-M1`, `EEG O2-M1`, `EEG C3-M2`, `EMG chin`, `EOG E1-M2`, `EOG E2-M2`, `ECG`). After `channels_rename`: 4 EEG (F4, C4, O2, C3), non-EEG dropped by montage
- **Sfreq:** 256 Hz
- **Labels:** `*_sleepscoring.txt` CSV companion files. Parsed by `parse_hmc_sleepscoring()`. 5 stages: W, N1, N2, N3, R
- **Config:** `configs/downstream/hmc.yaml`
- **Status:** Ready to process (verified parser: 854 epochs for SN001)

### 5. ISRUC (Sleep Staging)

- **Path:** `Original/isruc/` (500 files, 14 GB)
- **Download:** sleeptight.isr.uc.pt (ISRUC-SLEEP, Subgroup I)
- **Subjects:** 100 (80 train / 10 val / 10 test), Subgroup I
- **Channels:** 19 raw (`LOC-A2`, `ROC-A1`, `F3-A2`, `C3-A2`, `O1-A2`, `F4-A1`, `C4-A1`, `O2-A1`, `X1`-`X8`, `SaO2`, `DC3`, `DC8`). After `channels_rename`: 6 EEG (F3, C3, O1, F4, C4, O2), rest dropped by montage
- **Sfreq:** 200 Hz (no resampling needed)
- **Labels:** `{subject}_1.txt` companion files, one integer per line per 30s epoch (0=W, 1=N1, 2=N2, 3=N3, 5=REM). Scorer 1 used.
- **Config:** `configs/downstream/isruc.yaml`
- **Notes:** Original `.rec` files renamed to `.edf`. `standardize_channel_names` disabled (breaks referential names); uses `channels_rename` instead
- **Status:** Ready to process

### 6. Mumtaz2016 (Depression Detection)

- **Path:** `Original/mumtaz2016/` (179 files, 763 MB)
- **Download:** Figshare item 4244171
- **Subjects:** 64 — 34 MDD + 30 HC. 12 session files failed to download on Figshare.
- **Channels:** HC: 22 raw (`EEG Fp1-LE`, ..., `EEG A2-A1`, `EEG 23A-23R`, `EEG 24A-24R`). MDD: 20 raw (missing 23A-23R, 24A-24R). After `standardize_channel_names`: 19 EEG in both (extras dropped by montage, T3→T7, T4→T8)
- **Sfreq:** 256 Hz
- **Labels:** From filename prefix: `MDD S##` = depressed, `H S##` = healthy. Parsed by `parse_mumtaz2016_annotations()`. Only EC (eyes-closed) and EO (eyes-open) used; TASK excluded.
- **Config:** `configs/downstream/mumtaz2016.yaml`
- **Notes:** Channel count differs between groups (22 vs 20) but all 19 EEG channels present in both
- **Status:** Ready to process (179/191 files, all 64 subjects represented)

### 7. BCIC-IV-2a (Motor Imagery)

- **Path:** `Original/bcic_iv_2a/` (36 files, 576 MB)
- **Download:** BCI Competition IV (bbci.de/competition/iv/), eval labels from BNCI Horizon 2020
- **Subjects:** 9 (5 train / 2 val / 2 test), 2 sessions each (T=training, E=evaluation)
- **Channels:** 25 raw (`EEG-Fz`, `EEG-0`..`EEG-16`, `EEG-C3`, `EEG-Cz`, `EEG-C4`, `EEG-Pz`, `EOG-left/central/right`). GDF uses generic numbered names. After `channels_rename`: 22 EEG (Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz), 3 EOG dropped by montage
- **Sfreq:** 250 Hz
- **Labels:** GDF event codes: 769=left hand, 770=right hand, 771=both feet, 772=tongue. Training files have true labels; evaluation files have code 783 (unknown) but true labels in companion `.mat` files. Parsed by `parse_bcic_iv_2a_events()`
- **Config:** `configs/downstream/bcic_iv_2a.yaml`
- **Notes:** `event_tmin=2.0` (MI period starts 2s post-cue), `event_tlen=4.0`
- **Status:** Ready to process (verified: 288 events/session, 72 per class)

### 8. SHU-MI (Motor Imagery)

- **Path:** `Original/shu_mi/` (159 files, 6.2 GB)
- **Download:** WBCIC SHU Motor Imagery dataset (Figshare)
- **Subjects:** 52 (2-class subset), 3 sessions each
- **Channels:** 58 (Pz used as reference, not in data). Channel names from `task-motorimagery_channels.tsv`: Fpz, Fp1, Fp2, ..., O1, O2 (59 total minus Pz = 58 data channels)
- **Sfreq:** 250 Hz (inferred: 1000 samples / 4s MI period)
- **Labels:** In `.mat` file: `labels` array with values {1=left hand, 2=right hand}
- **Format:** `.mat` with `data=(58, 1000, 200)` and `labels=(200,)`. Requires conversion: `python scripts/converters/convert_shu_mi.py`
- **Config:** `configs/downstream/shu_mi.yaml`
- **Status:** Needs conversion to EDF before processing

### 9. MoBI (Gait Prediction)

- **Path:** `Original/mobi/` (153 files, 1 GB)
- **Download:** Figshare article 5807511
- **Subjects:** 8 (SL01-SL08), 3 trials each (T01-T03)
- **Channels:** 64 (60 EEG + 4 EOG, ActiCap system). No channel names in data files — assigned from standard 10-20 extended layout
- **Sfreq:** 100 Hz (inferred from timestamps)
- **Format:** `eeg.txt` (65 cols: timestamp + 64 channels) + `joints.txt` (13 cols: timestamp + 12 angles). Requires conversion: `python scripts/converters/convert_mobi.py`
- **Targets:** 12 joint angles — 6 goniometer-measured (GHR, GKR, GAR, GHL, GKL, GAL) + 6 BCI-predicted (PHR, PKR, PAR, PHL, PKL, PAL). Normalized by /90.
- **Config:** `configs/downstream/mobi.yaml`
- **Notes:** Multi-target regression. 2s windows with 50ms stride = ~27,400 windows per session. Regression targets are parsed from annotation description strings (`gait_<v1>_..._<v12>` → 12 floats). Uses `task_type: regression` in config.
- **Status:** Converted to EDF. Ready to process.

---

## Datasets Not Yet Downloaded

| Dataset | Source | Notes |
|---------|--------|-------|
| FACED | Author request / OpenNeuro | .mat, 9-class emotion, 32ch |
| SEED-V | BCMI lab application | .mat, 5-class emotion, 62ch |
| BCIC2020-IV-3 | BCI Competition 2020 | .mat, 5-class imagined speech, 64ch |
| SEED-VIG | BCMI lab application | .mat, regression (PERCLOS), 17ch |
| TUEV | NEDC credentials | .edf + .tse, 6-class events, 19ch (bipolar) |
| TUAB | NEDC credentials | .edf, binary normal/abnormal, 19ch (bipolar) |

---

## Processing Commands

```bash
# Datasets that can be processed directly (EDF/GDF):
python scripts/preprocess_downstream.py --config configs/downstream/<name>.yaml

# Datasets needing conversion first (.mat/.txt → EDF):
python scripts/converters/convert_shu_mi.py --input_dir /scratch/agjma/SPEED/Original/shu_mi --output_dir /scratch/agjma/SPEED/Original/shu_mi/edf
python scripts/converters/convert_mobi.py --input_dir /scratch/agjma/SPEED/Original/mobi --output_dir /scratch/agjma/SPEED/Original/mobi_edf
```

### 10. Siena Scalp EEG (Seizure Detection)

- **Path:** `Original/siena-scalp-eeg/1.0.0/` (18 GB)
- **Download:** PhysioNet (siena-scalp-eeg/1.0.0)
- **Subjects:** 14 adult patients (ages 20-71)
- **Channels:** 29 EEG (monopolar 10-20 extended) → 19 after montage. Non-EEG (EKG, SPO2, HR) dropped.
- **Sfreq:** 512 Hz
- **Labels:** Binary seizure vs non-seizure. 42 seizures parsed from `Seizures-list-PNxx.txt` clock-time annotations. Non-seizure windows via sliding window with 60s margin.
- **Config:** `configs/downstream/siena.yaml`
- **Notes:**
  - Used in EEG-FM-Bench (2025) benchmark
  - Complements CHB-MIT (adult vs pediatric, monopolar vs bipolar, EU vs US)
  - PN00-3 has a confirmed annotation typo (seizure end `19.29.29` should be `18.29.29`) — skipped automatically
  - Annotation format is messy (mixed separators, Italian text, varying file naming) — parser handles most cases (42/47 seizures)
  - BIDS version available on Zenodo (10640762) with clean TSV annotations and 19ch at 256 Hz
  - `standardize_channel_names: true` handles `EEG` prefix and T3→T7, T4→T8 renaming

---

## Data Locations

- **Original data:** `/scratch/agjma/SPEED/Original/<name>/`
- **Processed output:** `/scratch/agjma/SPEED/Processed/<name>/`
- **Configs:** `/home/agjma/SPEED/configs/downstream/<name>.yaml`
