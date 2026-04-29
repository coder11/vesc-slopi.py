import numpy as np
import pytest

from yalsa import (
    biquad_lowpass_hz,
    butter_lowpass_hz,
    ema_alpha_from_cutoff_hz,
    ema_lowpass_hz,
    fft_magnitude,
    fir_ma_decimator_block_hold,
    ma_decimation_factor_from_cutoff_hz,
    measured_sample_rate_hz,
    nominal_or_measured_sample_rate_hz,
    signal_stats,
    welch_psd,
)


def test_measured_sample_rate_hz_uses_median_positive_delta() -> None:
    timestamps = np.array([0.0, 0.01, 0.02, 0.031, 0.04], dtype=np.float64)

    sample_rate_hz = measured_sample_rate_hz(timestamps)

    assert sample_rate_hz == pytest.approx(100.0)


def test_butter_lowpass_hz_can_start_from_steady_initial_value() -> None:
    values = np.full(16, 3.0, dtype=np.float64)

    filtered = butter_lowpass_hz(
        values,
        cutoff_hz=25.0,
        sample_rate_hz=200.0,
        order=2,
        initial_value=float(values[0]),
    )

    np.testing.assert_allclose(filtered, values)


def test_butter_lowpass_hz_rejects_cutoff_at_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        butter_lowpass_hz(
            np.array([1.0], dtype=np.float64),
            cutoff_hz=50.0,
            sample_rate_hz=100.0,
        )


def test_signal_stats_returns_expected_metrics() -> None:
    stats = signal_stats(np.array([-2.0, 1.0, 4.0], dtype=np.float64))

    assert stats is not None
    assert stats.mean == pytest.approx(1.0)
    assert stats.std == pytest.approx(np.sqrt(6.0))
    assert stats.rms == pytest.approx(np.sqrt(7.0))
    assert stats.peak_to_peak == pytest.approx(6.0)


def test_welch_psd_detects_peak_frequency() -> None:
    sample_rate_hz = 200.0
    timestamps = np.arange(400, dtype=np.float64) / sample_rate_hz
    values = np.sin(2.0 * np.pi * 25.0 * timestamps)

    spectrum = welch_psd(timestamps, values)

    assert spectrum is not None
    frequencies, power = spectrum
    peak_frequency = float(frequencies[int(np.argmax(power))])
    assert peak_frequency == pytest.approx(25.0, abs=1.0)


def test_fft_magnitude_detects_peak_frequency() -> None:
    sample_rate_hz = 200.0
    timestamps = np.arange(400, dtype=np.float64) / sample_rate_hz
    values = np.sin(2.0 * np.pi * 25.0 * timestamps)

    spectrum = fft_magnitude(timestamps, values)

    assert spectrum is not None
    frequencies, magnitudes = spectrum
    peak_index = int(np.argmax(magnitudes[1:]) + 1)
    assert float(frequencies[peak_index]) == pytest.approx(25.0)
    assert float(magnitudes[peak_index]) == pytest.approx(1.0, rel=0.05)


def test_nominal_or_measured_sample_rate_hz_prefers_nominal() -> None:
    timestamps = np.array([0.0, 1.0], dtype=np.float64)
    assert nominal_or_measured_sample_rate_hz(321.0, timestamps) == pytest.approx(321.0)


def test_nominal_or_measured_sample_rate_hz_falls_back_to_timestamps() -> None:
    timestamps = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    assert nominal_or_measured_sample_rate_hz(0.0, timestamps) == pytest.approx(100.0)


def test_ma_decimation_factor_from_cutoff_hz() -> None:
    assert ma_decimation_factor_from_cutoff_hz(500.0, 6000.0) == 6


def test_fir_ma_decimator_block_hold_averages_blocks() -> None:
    raw = np.arange(12, dtype=np.float64)
    y, factor = fir_ma_decimator_block_hold(
        raw,
        cutoff_hz=500.0,
        sample_rate_hz=6000.0,
    )
    assert factor == 6
    np.testing.assert_allclose(y[:6], np.mean(raw[:6]))
    np.testing.assert_allclose(y[6:12], np.mean(raw[6:12]))


def test_ema_alpha_from_cutoff_hz_in_range() -> None:
    alpha = ema_alpha_from_cutoff_hz(10.0, 1000.0)
    assert 0.0 < alpha <= 1.0


def test_ema_lowpass_hz_smooths_step() -> None:
    x = np.zeros(200, dtype=np.float64)
    x[100:] = 1.0
    y = ema_lowpass_hz(x, cutoff_hz=5.0, sample_rate_hz=200.0)
    assert y[0] == pytest.approx(0.0)
    assert float(y[-1]) > 0.99


def test_biquad_lowpass_hz_passes_dc() -> None:
    x = np.ones(256, dtype=np.float64)
    y = biquad_lowpass_hz(
        x,
        cutoff_hz=10.0,
        sample_rate_hz=1000.0,
        q=0.707,
    )
    assert float(y[-1]) == pytest.approx(1.0, abs=0.02)
