"""Tests for speed.annotations error logging."""

import logging
import tempfile
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
