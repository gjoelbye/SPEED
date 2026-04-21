"""Tests for speed.annotations error logging."""

import logging
import tempfile
import textwrap
from pathlib import Path

import pytest


class TestAnnotationLogging:
    """Verify that parsers log on unparseable input instead of silently dropping."""

    def test_isruc_logs_on_bad_line(self, caplog, tmp_path):
        """ISRUC parser should log unparseable lines at DEBUG level."""
        from speed.annotations import parse_isruc_annotations

        # ISRUC expects: EEG file path, looks for <stem>.txt in same dir
        eeg_path = tmp_path / "subject1.edf"
        eeg_path.touch()  # dummy EEG file
        annot_path = tmp_path / "subject1_1.txt"
        annot_path.write_text(
            "not_a_number\n"  # bad line
            "0\n"  # valid: W
            "1\n"  # valid: N1
        )

        with caplog.at_level(logging.DEBUG):
            annotations = parse_isruc_annotations(eeg_path)

        assert len(annotations) == 2  # two valid lines
        assert any("Skipping unparseable line" in msg for msg in caplog.messages)

    def test_tuev_logs_on_bad_line(self, caplog, tmp_path):
        """TUEV parser should log unparseable TSE lines at DEBUG level."""
        from speed.annotations import parse_tuev_annotations

        tse_path = tmp_path / "annotations.tse"
        tse_path.write_text(
            "version = tse_v1.0.0\n"
            "not_a_number 10.0 spsw\n"  # bad start time
            "0.0 10.0 spsw\n"  # valid
        )

        with caplog.at_level(logging.DEBUG):
            annotations = parse_tuev_annotations(tse_path)

        assert len(annotations) == 1
        assert any("Skipping unparseable TSE line" in msg for msg in caplog.messages)


def _make_hbn_fake_layout(tmp_path: Path, task: str, subject: str = "sub-NDARABC123XYZ"):
    """Build a minimal BIDS-style layout for HBN parser tests.

    Returns (set_path, events_tsv_path, release_dir).
    """
    release_dir = tmp_path / "ds005505"
    eeg_dir = release_dir / subject / "eeg"
    eeg_dir.mkdir(parents=True)
    # The .set file does not need to be a valid EEGLAB file — the parsers only
    # touch its sibling events.tsv. We still create a placeholder so
    # _hbn_resolve_events_tsv's suffix check passes.
    set_path = eeg_dir / f"{subject}_task-{task}_eeg.set"
    set_path.touch()
    tsv_path = eeg_dir / f"{subject}_task-{task}_events.tsv"
    return set_path, tsv_path, release_dir


def _write_participants_tsv(release_dir: Path, rows: list):
    """Write a minimal participants.tsv with the given list of dict rows."""
    headers = rows[0].keys()
    path = release_dir / "participants.tsv"
    with open(path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join(str(r[h]) for h in headers) + "\n")
    return path


class TestHBNParsers:
    """Empirical-data-driven tests for the new HBN parsers."""

    def test_ccd_rt_happy_path(self, tmp_path):
        from speed.annotations import parse_hbn_ccd_rt
        set_path, tsv_path, _ = _make_hbn_fake_layout(
            tmp_path, "contrastChangeDetection_run-1"
        )
        # Two valid trials + one trial without response + one with RT<0.25 s
        tsv_path.write_text(textwrap.dedent("""\
            onset\tduration\tsample\tvalue\tevent_code\tfeedback
            0.0\tn/a\t0\tbreak cnt\tbreak cnt\tn/a
            10.0\tn/a\t5000\tcontrastTrial_start\t5\tn/a
            11.0\tn/a\t5500\tright_target\t9\tn/a
            12.5\tn/a\t6250\tright_buttonPress\t13\tsmiley_face
            20.0\tn/a\t10000\tcontrastTrial_start\t5\tn/a
            21.0\tn/a\t10500\tleft_target\t8\tn/a
            21.1\tn/a\t10550\tleft_buttonPress\t12\tsmiley_face
            30.0\tn/a\t15000\tcontrastTrial_start\t5\tn/a
            31.0\tn/a\t15500\tleft_target\t8\tn/a
            40.0\tn/a\t20000\tcontrastTrial_start\t5\tn/a
            41.0\tn/a\t20500\tright_target\t9\tn/a
            42.3\tn/a\t21150\tright_buttonPress\t13\tsmiley_face
        """))
        out = parse_hbn_ccd_rt(set_path, event_tlen=2.0)
        # Trial 1: RT=1.5s; trial 2: RT=0.1s (dropped); trial 3: no response; trial 4: RT=1.3s
        assert len(out) == 2
        onsets = [o for o, _, _ in out]
        descs = [d for _, _, d in out]
        assert onsets == [11.0, 41.0]
        # Descriptions: "rt_<float>"; the regression decoder reads parts[1:]
        assert descs[0].startswith("rt_1.5")
        assert descs[1].startswith("rt_1.3")
        for _, dur, _ in out:
            assert dur == 2.0

    def test_ccd_correct_skips_non_target(self, tmp_path):
        from speed.annotations import parse_hbn_ccd_correct
        set_path, tsv_path, _ = _make_hbn_fake_layout(
            tmp_path, "contrastChangeDetection_run-1"
        )
        tsv_path.write_text(textwrap.dedent("""\
            onset\tduration\tsample\tvalue\tevent_code\tfeedback
            10.0\tn/a\t5000\tcontrastTrial_start\t5\tn/a
            11.0\tn/a\t5500\tright_target\t9\tn/a
            12.5\tn/a\t6250\tright_buttonPress\t13\tsmiley_face
            20.0\tn/a\t10000\tcontrastTrial_start\t5\tn/a
            21.0\tn/a\t10500\tleft_target\t8\tn/a
            22.8\tn/a\t11400\tleft_buttonPress\t12\tsad_face
            30.0\tn/a\t15000\tcontrastTrial_start\t5\tn/a
            31.5\tn/a\t15750\tright_buttonPress\t13\tnon_target
            32.0\tn/a\t16000\tright_target\t9\tn/a
        """))
        out = parse_hbn_ccd_correct(set_path, event_tlen=2.0)
        # Trial 1: smiley→correct, Trial 2: sad→incorrect,
        # Trial 3: response before target → non_target feedback → skipped
        labels = [d for _, _, d in out]
        assert labels == ["correct", "incorrect"]

    def test_cbcl_nan_subject_yields_empty(self, tmp_path):
        from speed.annotations import parse_hbn_cbcl, _hbn_load_participants
        _hbn_load_participants.cache_clear()
        set_path, tsv_path, release_dir = _make_hbn_fake_layout(
            tmp_path, "contrastChangeDetection_run-1",
            subject="sub-NDARNAN000NAN",
        )
        tsv_path.write_text("onset\tduration\tsample\tvalue\tevent_code\n")
        _write_participants_tsv(release_dir, [{
            "participant_id": "sub-NDARNAN000NAN",
            "p_factor": "NaN", "attention": "0.1",
            "internalizing": "0.2", "externalizing": "0.3",
        }])
        out = parse_hbn_cbcl(set_path, recording_duration=10.0, event_tlen=2.0, stride=2.0)
        assert out == []

    def test_cbcl_valid_subject_emits_fixed_windows(self, tmp_path):
        from speed.annotations import parse_hbn_cbcl, _hbn_load_participants
        _hbn_load_participants.cache_clear()
        set_path, tsv_path, release_dir = _make_hbn_fake_layout(
            tmp_path, "contrastChangeDetection_run-1",
            subject="sub-NDARGOOD123",
        )
        tsv_path.write_text("onset\tduration\tsample\tvalue\tevent_code\n")
        _write_participants_tsv(release_dir, [{
            "participant_id": "sub-NDARGOOD123",
            "p_factor": "-0.5", "attention": "0.25",
            "internalizing": "1.0", "externalizing": "-1.5",
        }])
        # Starter-kit-style 4 s windows at 2 s stride (overlapping).
        # floor((recording - tlen)/stride) + 1 = floor((12-4)/2)+1 = 5
        # Onsets: 0, 2, 4, 6, 8 (last window spans [8,12])
        out = parse_hbn_cbcl(set_path, recording_duration=12.0, event_tlen=4.0, stride=2.0)
        assert len(out) == 5
        onsets = [o for o, _, _ in out]
        assert onsets == [0.0, 2.0, 4.0, 6.0, 8.0]
        durations = [d for _, d, _ in out]
        assert all(d == 4.0 for d in durations)
        desc = out[0][2]
        assert desc.startswith("cbcl_")
        # Regression decoder splits on '_' and parses parts[1:] as floats
        parts = desc.split("_")
        vals = [float(p) for p in parts[1:]]
        assert vals == [-0.5, 0.25, 1.0, -1.5]
        # all windows carry the same 4-D label
        assert all(o[2] == desc for o in out)

    def test_rest_ec_eo_expected_counts(self, tmp_path):
        from speed.annotations import parse_hbn_rest_ec_eo
        set_path, tsv_path, _ = _make_hbn_fake_layout(tmp_path, "RestingState")
        # Build a minimally valid RestingState events.tsv with 2 EC + 2 EO
        # instructions spaced 60 s apart to stay inside a 500 s recording.
        lines = ["onset\tduration\tsample\tvalue\tevent_code"]
        for i, val in enumerate(["instructed_toOpenEyes", "instructed_toCloseEyes",
                                 "instructed_toOpenEyes", "instructed_toCloseEyes"]):
            onset = 30.0 + i * 60.0
            lines.append(f"{onset}\tn/a\t{int(onset*500)}\t{val}\t20")
        tsv_path.write_text("\n".join(lines) + "\n")
        out = parse_hbn_rest_ec_eo(set_path, recording_duration=500.0, event_tlen=2.0)
        # 2 EC × 8 offsets + 2 EO × 8 offsets = 32 windows
        assert len(out) == 32
        ec = [o for o in out if o[2] == "eyes_closed"]
        eo = [o for o in out if o[2] == "eyes_open"]
        assert len(ec) == 16
        assert len(eo) == 16

    def test_rest_ec_eo_clips_past_recording_end(self, tmp_path):
        from speed.annotations import parse_hbn_rest_ec_eo
        set_path, tsv_path, _ = _make_hbn_fake_layout(tmp_path, "RestingState")
        tsv_path.write_text(textwrap.dedent("""\
            onset\tduration\tsample\tvalue\tevent_code
            300.0\tn/a\t150000\tinstructed_toCloseEyes\t30
        """))
        # Recording ends at 325 s — EC offsets 15,17,19,21,23 fit (ends at
        # 300+{15..23}+2=317..325 s) but 25,27,29 do not.
        out = parse_hbn_rest_ec_eo(set_path, recording_duration=325.0, event_tlen=2.0)
        onsets = sorted(o[0] for o in out)
        assert onsets == [315.0, 317.0, 319.0, 321.0, 323.0]

    def test_surroundsupp_happy_and_nan(self, tmp_path):
        from speed.annotations import parse_hbn_surroundsupp
        set_path, tsv_path, _ = _make_hbn_fake_layout(
            tmp_path, "surroundSupp_run-1"
        )
        tsv_path.write_text(textwrap.dedent("""\
            onset\tduration\tsample\tvalue\tevent_code\tbackground\tforeground_contrast\tstimulus_cond
            72.0\t2.4\t36000\tstim_ON\t8\t1\t0.0\t2
            75.4\t2.4\t37700\tstim_ON\t8\t1\t0.3\t3
            78.8\t2.4\t39400\tstim_ON\t8\t0\t0.6\tn/a
            82.2\t2.4\t41100\tstim_ON\t8\t1\t1.0\t1
        """))
        # Config default is tlen=2.4 (matches stim_ON duration).
        out = parse_hbn_surroundsupp(set_path, event_tlen=2.4)
        # Row with n/a stimulus_cond must be skipped.
        labels = [d for _, _, d in out]
        assert labels == ["stimcond_2", "stimcond_3", "stimcond_1"]
        assert all(dur == 2.4 for _, dur, _ in out)

    def test_symbolsearch_correct_vs_incorrect(self, tmp_path):
        from speed.annotations import parse_hbn_symbolsearch
        set_path, tsv_path, _ = _make_hbn_fake_layout(tmp_path, "symbolSearch")
        tsv_path.write_text(textwrap.dedent("""\
            onset\tduration\tsample\tvalue\tevent_code\tuser_answer\tcorrect_answer
            38.1\tn/a\t19050\ttrialResponse\t14\t0\t0
            40.9\tn/a\t20481\ttrialResponse\t14\t1\t0
            46.5\tn/a\t23231\ttrialResponse\t14\tn/a\t1
        """))
        out = parse_hbn_symbolsearch(set_path, event_tlen=2.0)
        labels = [d for _, _, d in out]
        # row 1: match → correct; row 2: mismatch → incorrect; row 3: NaN ua → skip
        assert labels == ["correct", "incorrect"]
