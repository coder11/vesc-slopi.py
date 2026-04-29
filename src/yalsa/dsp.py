"""DSP helpers for YALSA scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal  # type: ignore[import-untyped]

FloatArray = npt.NDArray[np.float64]


def _as_float_array(values: npt.ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def measured_sample_rate_hz(timestamps_s: npt.ArrayLike) -> float | None:
    """Return a measured sample rate from monotonically increasing timestamps."""
    timestamps = _as_float_array(timestamps_s)
    if int(timestamps.size) < 2:
        return None
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0.0]
    if int(deltas.size) == 0:
        return None
    sample_period_s = float(np.median(deltas))
    if not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        return None
    return 1.0 / sample_period_s


def nominal_or_measured_sample_rate_hz(
    nominal_hz: float | None,
    timestamps_s: npt.ArrayLike,
) -> float | None:
    """Prefer a positive nominal sample rate; otherwise estimate from timestamps."""
    if nominal_hz is not None:
        n = float(nominal_hz)
        if n > 0.0 and math.isfinite(n):
            return n
    return measured_sample_rate_hz(timestamps_s)


def butter_lowpass_hz(
    values: npt.ArrayLike,
    *,
    cutoff_hz: float,
    sample_rate_hz: float,
    order: int = 2,
    initial_value: float | None = None,
) -> FloatArray:
    """Return a causal Butterworth low-pass response using scipy.signal.sosfilt."""
    raw = _as_float_array(values)
    if int(raw.size) == 0:
        return raw.copy()
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be greater than 0")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be greater than 0")
    if order <= 0:
        raise ValueError("order must be greater than 0")

    nyquist_hz = sample_rate_hz / 2.0
    if cutoff_hz >= nyquist_hz:
        raise ValueError("cutoff_hz must be less than Nyquist")

    sos = signal.butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=sample_rate_hz,
        output="sos",
    )
    steady_state = raw[0] if initial_value is None else float(initial_value)
    zi = signal.sosfilt_zi(sos) * steady_state
    filtered, _state = signal.sosfilt(sos, raw, zi=zi)
    return np.asarray(filtered, dtype=np.float64)


def ema_alpha_from_cutoff_hz(cutoff_hz: float, sample_rate_hz: float) -> float:
    """Smoothing coefficient for y[n]=y[n-1]+α(x[n]-y[n-1]) from a cutoff (Hz)."""
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be greater than 0")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be greater than 0")
    wc = 2.0 * math.pi * cutoff_hz / sample_rate_hz
    return max(0.0, min(1.0, 1.0 - math.exp(-wc)))


def ema_lowpass_hz(
    values: npt.ArrayLike,
    *,
    cutoff_hz: float,
    sample_rate_hz: float,
) -> FloatArray:
    """First-order low-pass: y[n] = y[n-1] + α (x[n] - y[n-1])."""
    raw = _as_float_array(values)
    n = int(raw.size)
    if n == 0:
        return raw.copy()
    alpha = ema_alpha_from_cutoff_hz(cutoff_hz, sample_rate_hz)
    y = np.empty_like(raw)
    y[0] = raw[0]
    for i in range(1, n):
        yi = float(y[i - 1])
        xi = float(raw[i])
        y[i] = yi + alpha * (xi - yi)
    return np.asarray(y, dtype=np.float64)


def ma_decimation_factor_from_cutoff_hz(
    cutoff_hz: float,
    sample_rate_hz: float,
) -> int:
    """Block length N so Fs/N ≈ 2·fc (output rate ≈ Fs/N, Nyquist ≈ Fs/(2N))."""
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be greater than 0")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be greater than 0")
    return max(1, int(round(sample_rate_hz / (2.0 * cutoff_hz))))


def fir_ma_decimator_block_hold(
    values: npt.ArrayLike,
    *,
    cutoff_hz: float,
    sample_rate_hz: float,
) -> tuple[FloatArray, int]:
    """Non-overlapping N-sample moving average; hold each mean across its block.

    Same length as input so time-series plots stay aligned with raw timestamps.
    """
    raw = _as_float_array(values)
    n = int(raw.size)
    if n == 0:
        return raw.copy(), 1
    factor = ma_decimation_factor_from_cutoff_hz(cutoff_hz, sample_rate_hz)
    factor = min(factor, n)
    out = np.empty_like(raw)
    full = (n // factor) * factor
    for start in range(0, full, factor):
        block_mean = float(np.mean(raw[start : start + factor]))
        out[start : start + factor] = block_mean
    if full < n:
        tail_mean = float(np.mean(raw[full:]))
        out[full:] = tail_mean
    return np.asarray(out, dtype=np.float64), factor


def biquad_lowpass_hz(
    values: npt.ArrayLike,
    *,
    cutoff_hz: float,
    sample_rate_hz: float,
    q: float = 0.707,
) -> FloatArray:
    """Second-order low-pass (tangent prewarp), matching common VESC-style biquad."""
    raw = _as_float_array(values)
    n = int(raw.size)
    if n == 0:
        return raw.copy()
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be greater than 0")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be greater than 0")
    if q <= 0.0:
        raise ValueError("q must be greater than 0")
    nyquist_hz = sample_rate_hz / 2.0
    if cutoff_hz >= nyquist_hz:
        raise ValueError("cutoff_hz must be less than Nyquist")

    k = math.tan(math.pi * cutoff_hz / sample_rate_hz)
    norm = 1.0 / (1.0 + k / q + k * k)
    a0 = k * k * norm
    a1 = 2.0 * a0
    a2 = a0
    b1 = 2.0 * (k * k - 1.0) * norm
    b2 = (1.0 - k / q + k * k) * norm

    y = np.empty_like(raw)
    z1 = 0.0
    z2 = 0.0
    for i in range(n):
        inp = float(raw[i])
        out = inp * a0 + z1
        z1 = inp * a1 + z2 - b1 * out
        z2 = inp * a2 - b2 * out
        y[i] = out
    return np.asarray(y, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SignalStats:
    """Summary metrics for one signal window."""

    mean: float
    std: float
    rms: float
    peak_to_peak: float


def signal_stats(values: npt.ArrayLike) -> SignalStats | None:
    """Return summary metrics for the supplied signal values."""
    data = _as_float_array(values)
    if int(data.size) == 0:
        return None
    return SignalStats(
        mean=float(np.mean(data)),
        std=float(np.std(data)),
        rms=float(np.sqrt(np.mean(data * data))),
        peak_to_peak=float(np.ptp(data)),
    )


def fft_magnitude(
    timestamps_s: npt.ArrayLike,
    values: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray] | None:
    """Return one-sided FFT magnitude bins for a scalar signal window."""
    timestamps = _as_float_array(timestamps_s)
    samples = _as_float_array(values)
    if timestamps.shape != samples.shape:
        raise ValueError("timestamps_s and values must have the same shape")
    sample_rate_hz = measured_sample_rate_hz(timestamps)
    if sample_rate_hz is None or int(samples.size) < 2:
        return None

    centered = samples - float(np.mean(samples))
    magnitudes = np.abs(np.fft.rfft(centered)) / float(samples.size)
    if int(magnitudes.size) > 1:
        if int(samples.size) % 2 == 0:
            magnitudes[1:-1] *= 2.0
        else:
            magnitudes[1:] *= 2.0

    frequencies = np.fft.rfftfreq(int(samples.size), d=1.0 / sample_rate_hz)
    return (
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(magnitudes, dtype=np.float64),
    )


def welch_psd(
    timestamps_s: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    nperseg: int | None = None,
) -> tuple[FloatArray, FloatArray] | None:
    """Return a one-sided power spectral density estimate using SciPy Welch."""
    timestamps = _as_float_array(timestamps_s)
    samples = _as_float_array(values)
    if timestamps.shape != samples.shape:
        raise ValueError("timestamps_s and values must have the same shape")
    sample_rate_hz = measured_sample_rate_hz(timestamps)
    if sample_rate_hz is None or int(samples.size) < 2:
        return None

    segment_size = min(int(samples.size), 1024 if nperseg is None else nperseg)
    if segment_size < 2:
        return None

    frequencies, power = signal.welch(
        samples,
        fs=sample_rate_hz,
        nperseg=segment_size,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return (
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(power, dtype=np.float64),
    )


__all__ = [
    "FloatArray",
    "SignalStats",
    "biquad_lowpass_hz",
    "butter_lowpass_hz",
    "ema_alpha_from_cutoff_hz",
    "ema_lowpass_hz",
    "fir_ma_decimator_block_hold",
    "fft_magnitude",
    "ma_decimation_factor_from_cutoff_hz",
    "measured_sample_rate_hz",
    "nominal_or_measured_sample_rate_hz",
    "signal_stats",
    "welch_psd",
]
