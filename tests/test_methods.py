"""Tests for speed.methods.PreprocessMethods."""

import mne
import numpy as np
import pytest

from speed.methods import PreprocessMethods


class TestResample:
    """Tests for the anti-aliased resampling method."""

    def test_resample_preserves_sub_nyquist_frequencies(self, synthetic_raw):
        """10 Hz and 50 Hz components should survive resampling to 128 Hz."""
        target_sfreq = 128.0
        PreprocessMethods.resample(synthetic_raw, sfreq=target_sfreq)

        assert synthetic_raw.info["sfreq"] == target_sfreq

        # Check frequency content via FFT
        data = synthetic_raw.get_data()[0]  # first channel
        freqs = np.fft.rfftfreq(len(data), d=1.0 / target_sfreq)
        fft_mag = np.abs(np.fft.rfft(data))

        # Find magnitude at 10 Hz (should be strong)
        idx_10 = np.argmin(np.abs(freqs - 10.0))
        # Find magnitude at 50 Hz (should be present)
        idx_50 = np.argmin(np.abs(freqs - 50.0))
        # Noise floor: average magnitude away from signal peaks
        noise_mask = np.ones(len(freqs), dtype=bool)
        for peak_idx in [idx_10, idx_50]:
            noise_mask[max(0, peak_idx - 3):peak_idx + 4] = False
        noise_floor = np.mean(fft_mag[noise_mask])

        assert fft_mag[idx_10] > 10 * noise_floor, "10 Hz component lost after resampling"
        assert fft_mag[idx_50] > 5 * noise_floor, "50 Hz component lost after resampling"

    def test_resample_attenuates_above_nyquist(self, synthetic_raw):
        """90 Hz component should be attenuated when resampling to 128 Hz (Nyquist = 64 Hz)."""
        # Get original 90 Hz magnitude
        orig_data = synthetic_raw.get_data()[0].copy()
        orig_freqs = np.fft.rfftfreq(len(orig_data), d=1.0 / 256.0)
        orig_fft = np.abs(np.fft.rfft(orig_data))
        idx_90_orig = np.argmin(np.abs(orig_freqs - 90.0))
        orig_90_mag = orig_fft[idx_90_orig]

        # Resample to 128 Hz
        PreprocessMethods.resample(synthetic_raw, sfreq=128.0)

        # After resampling, 90 Hz should not appear as aliased energy
        data = synthetic_raw.get_data()[0]
        freqs = np.fft.rfftfreq(len(data), d=1.0 / 128.0)
        fft_mag = np.abs(np.fft.rfft(data))

        # Check there's no spike at 90-128=38 Hz (where nearest-neighbor would alias)
        idx_38 = np.argmin(np.abs(freqs - 38.0))
        noise_mask = np.ones(len(freqs), dtype=bool)
        # Exclude signal peaks at 10 and 50 Hz
        for f in [10, 50]:
            fidx = np.argmin(np.abs(freqs - f))
            noise_mask[max(0, fidx - 3):fidx + 4] = False
        noise_floor = np.mean(fft_mag[noise_mask])

        # The aliased energy at 38 Hz should be near noise floor
        assert fft_mag[idx_38] < 5 * noise_floor, (
            f"Aliased energy at 38 Hz ({fft_mag[idx_38]:.2e}) is too high "
            f"relative to noise floor ({noise_floor:.2e})"
        )

    def test_resample_updates_sfreq(self, synthetic_raw_short):
        """raw.info['sfreq'] should be updated after resampling."""
        PreprocessMethods.resample(synthetic_raw_short, sfreq=200.0)
        assert synthetic_raw_short.info["sfreq"] == 200.0

    def test_resample_noop_when_same_sfreq(self, synthetic_raw_short):
        """No resampling when target sfreq matches current sfreq."""
        n_samples_before = synthetic_raw_short.n_times
        PreprocessMethods.resample(synthetic_raw_short, sfreq=256.0)
        assert synthetic_raw_short.n_times == n_samples_before

    def test_resample_updates_sample_count(self, synthetic_raw_short):
        """Number of samples should change proportionally to sfreq ratio."""
        original_sfreq = synthetic_raw_short.info["sfreq"]
        original_n = synthetic_raw_short.n_times
        target_sfreq = 128.0

        PreprocessMethods.resample(synthetic_raw_short, sfreq=target_sfreq)

        expected_n = int(round(original_n * target_sfreq / original_sfreq))
        assert abs(synthetic_raw_short.n_times - expected_n) <= 1


class TestNormalize:
    """Tests for the normalization method."""

    def test_zscore_normalization(self, synthetic_raw_short):
        PreprocessMethods.normalize(synthetic_raw_short, method="zscore")
        data = synthetic_raw_short.get_data()
        # Per-channel mean should be ~0, std should be ~1
        means = data.mean(axis=-1)
        stds = data.std(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-6)
        np.testing.assert_allclose(stds, 1.0, atol=0.01)

    def test_minmax_normalization(self, synthetic_raw_short):
        PreprocessMethods.normalize(synthetic_raw_short, method="minmax")
        data = synthetic_raw_short.get_data()
        assert data.min() >= -1e-6
        assert data.max() <= 1.0 + 1e-6

    def test_robust_normalization(self, synthetic_raw_short):
        PreprocessMethods.normalize(synthetic_raw_short, method="robust")
        data = synthetic_raw_short.get_data()
        # Median should be ~0
        medians = np.median(data, axis=-1)
        np.testing.assert_allclose(medians, 0.0, atol=1e-6)

    def test_invalid_method_raises(self, synthetic_raw_short):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            PreprocessMethods.normalize(synthetic_raw_short, method="invalid")


class TestBackwardCompat:
    """Test backward compatibility aliases."""

    def test_interpolate_nearest_alias(self):
        assert PreprocessMethods.interpolate_nearest is PreprocessMethods.resample
