import os
import sys

import numpy as np
import pytest

from examples.imu_signal_bench import (
    Biquad,
    DEFAULT_NOISY_DETERMINISTIC_RATE,
    biquad_config_lowpass,
    biquad_lowpass,
    biquad_lowpass_hz,
    build_parser,
    fir_lowpass_coefficients,
    fir_lowpass_hz,
    format_stats,
    make_source,
    normalized_biquad_cutoff,
    one_pole_lowpass_hz,
    prefer_qt_xcb_platform,
    signal_fft,
    signal_frequency_analysis,
    signal_psd,
    signal_stats,
    validate_args,
)


def test_biquad_config_lowpass_uses_firmware_coefficients() -> None:
    biquad = Biquad()

    biquad_config_lowpass(biquad, cutoff=0.25, shape=0.707)

    assert biquad.a0 == pytest.approx(0.2928748964374482)
    assert biquad.a1 == pytest.approx(0.5857497928748964)
    assert biquad.a2 == pytest.approx(0.2928748964374482)
    assert biquad.b1 == pytest.approx(0.0, abs=1e-12)
    assert biquad.b2 == pytest.approx(0.17149958574979282)


def test_biquad_lowpass_filters_impulse_causally() -> None:
    values = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    filtered = biquad_lowpass_hz(
        values,
        cutoff_hz=250.0,
        sample_rate_hz=1000.0,
        shape=0.707,
    )

    np.testing.assert_allclose(
        filtered,
        np.array(
            [
                0.2928748964374482,
                0.5857497928748964,
                0.24264697302191243,
                -0.10045584683107164,
            ],
            dtype=np.float64,
        ),
    )


def test_biquad_lowpass_can_start_from_steady_initial_value() -> None:
    values = np.full(8, 3.0, dtype=np.float64)

    filtered = biquad_lowpass_hz(
        values,
        cutoff_hz=50.0,
        sample_rate_hz=1000.0,
        shape=0.707,
        initial_value=float(values[0]),
    )

    np.testing.assert_allclose(filtered, values)


def test_normalized_biquad_cutoff_uses_sample_rate() -> None:
    assert normalized_biquad_cutoff(
        cutoff_hz=15.9,
        sample_rate_hz=1000.0,
    ) == pytest.approx(0.0159)


def test_biquad_lowpass_rejects_invalid_params() -> None:
    values = np.array([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Nyquist"):
        biquad_lowpass_hz(values, cutoff_hz=500.0, sample_rate_hz=1000.0, shape=0.707)

    with pytest.raises(ValueError, match="shape"):
        biquad_lowpass(values, cutoff=0.1, shape=0.0)


def test_one_pole_lowpass_filters_impulse_causally() -> None:
    values = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    filtered = one_pole_lowpass_hz(
        values,
        cutoff_hz=250.0,
        sample_rate_hz=1000.0,
    )

    alpha = 1.0 - np.exp(-2.0 * np.pi * 250.0 / 1000.0)
    np.testing.assert_allclose(
        filtered,
        np.array(
            [
                alpha,
                alpha * (1.0 - alpha),
                alpha * (1.0 - alpha) ** 2,
            ],
            dtype=np.float64,
        ),
    )


def test_one_pole_lowpass_can_start_from_steady_initial_value() -> None:
    values = np.full(8, 3.0, dtype=np.float64)

    filtered = one_pole_lowpass_hz(
        values,
        cutoff_hz=50.0,
        sample_rate_hz=1000.0,
        initial_value=float(values[0]),
    )

    np.testing.assert_allclose(filtered, values)


def test_fir_lowpass_coefficients_are_normalized() -> None:
    coefficients = fir_lowpass_coefficients(
        cutoff_hz=25.0,
        sample_rate_hz=100.0,
        taps=11,
    )

    assert coefficients.size == 11
    assert float(np.sum(coefficients)) == pytest.approx(1.0)
    np.testing.assert_allclose(coefficients, coefficients[::-1])


def test_fir_lowpass_filters_causally() -> None:
    values = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    coefficients = fir_lowpass_coefficients(
        cutoff_hz=25.0,
        sample_rate_hz=100.0,
        taps=3,
    )

    filtered = fir_lowpass_hz(
        values,
        cutoff_hz=25.0,
        sample_rate_hz=100.0,
        taps=3,
    )

    expected = np.concatenate((coefficients, np.array([0.0], dtype=np.float64)))
    np.testing.assert_allclose(filtered, expected)


def test_fir_lowpass_can_pad_initial_value() -> None:
    values = np.full(8, 3.0, dtype=np.float64)

    filtered = fir_lowpass_hz(
        values,
        cutoff_hz=50.0,
        sample_rate_hz=1000.0,
        taps=11,
        pad_initial=True,
    )

    np.testing.assert_allclose(filtered, values)


def test_signal_stats_returns_window_metrics() -> None:
    stats = signal_stats(np.array([-2.0, 1.0, 4.0], dtype=np.float64))

    assert stats is not None
    assert stats.mean == pytest.approx(1.0)
    assert stats.std == pytest.approx(np.sqrt(6.0))
    assert stats.rms == pytest.approx(np.sqrt(7.0))
    assert stats.peak_to_peak == pytest.approx(6.0)


def test_signal_stats_handles_empty_values() -> None:
    stats = signal_stats(np.empty(0, dtype=np.float64))

    assert stats is None
    assert (
        format_stats(stats, "g")
        == "mean: n/a\nstd: n/a\nRMS: n/a\npeak-to-peak: n/a"
    )


def test_signal_psd_returns_frequency_bins_and_positive_power() -> None:
    sample_rate_hz = 100.0
    timestamps = np.arange(100, dtype=np.float64) / sample_rate_hz
    values = np.sin(2.0 * np.pi * 10.0 * timestamps)

    psd = signal_psd(timestamps, values)

    assert psd is not None
    frequencies, power = psd
    assert frequencies[0] == pytest.approx(0.0)
    assert frequencies[-1] == pytest.approx(sample_rate_hz / 2.0)
    assert frequencies[int(np.argmax(power))] == pytest.approx(10.0)
    assert np.all(power >= 0.0)


def test_signal_fft_returns_frequency_bins_and_peak_magnitude() -> None:
    sample_rate_hz = 100.0
    timestamps = np.arange(100, dtype=np.float64) / sample_rate_hz
    values = np.sin(2.0 * np.pi * 10.0 * timestamps)

    spectrum = signal_fft(timestamps, values)

    assert spectrum is not None
    frequencies, magnitudes = spectrum
    peak_index = int(np.argmax(magnitudes[1:]) + 1)
    assert frequencies[peak_index] == pytest.approx(10.0)
    assert magnitudes[peak_index] == pytest.approx(1.0, rel=0.05)


def test_signal_psd_handles_missing_timing() -> None:
    psd = signal_psd(
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    )

    assert psd is None


def test_signal_frequency_analysis_only_dispatches_selected_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    timestamps = np.array([0.0, 1.0], dtype=np.float64)
    values = np.array([0.0, 1.0], dtype=np.float64)

    def fake_fft(
        _timestamps: np.ndarray,
        _values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        calls.append("fft")
        return None

    def fake_psd(
        _timestamps: np.ndarray,
        _values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        calls.append("psd")
        return None

    module = sys.modules[signal_frequency_analysis.__module__]
    monkeypatch.setattr(module, "signal_fft", fake_fft)
    monkeypatch.setattr(module, "signal_psd", fake_psd)

    signal_frequency_analysis("fft", timestamps, values)
    signal_frequency_analysis("psd", timestamps, values)

    assert calls == ["fft", "psd"]


def test_prefer_qt_xcb_platform_replaces_wayland_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland;xcb")

    prefer_qt_xcb_platform()

    assert os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_prefer_qt_xcb_platform_preserves_explicit_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")

    prefer_qt_xcb_platform()

    assert os.environ["QT_QPA_PLATFORM"] == "wayland"


def test_noisy_deterministic_source_uses_fixed_200_hz_rate() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source",
            "deterministic-noisy",
            "--axis",
            "acc_z",
            "--deterministic-rate",
            "123",
        ]
    )

    validate_args(parser, args)
    source, source_label = make_source(args)

    assert source.channel_name == "acc_z"
    assert source.unit == "g"
    assert (
        source_label
        == f"Deterministic noisy source @ {DEFAULT_NOISY_DETERMINISTIC_RATE:g} Hz"
    )


def test_deterministic_white_noise_source_uses_configured_rate() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source",
            "deterministic-white-noise",
            "--axis",
            "gyro_z",
            "--deterministic-rate",
            "321",
        ]
    )

    validate_args(parser, args)
    source, source_label = make_source(args)

    assert source.channel_name == "gyro_z"
    assert source.unit == "deg/s"
    assert source_label == "Deterministic white noise source @ 321 Hz"
