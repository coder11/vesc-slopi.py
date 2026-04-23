#!/usr/bin/env python3
"""Live one-axis IMU signal bench for software filter tuning.

Examples:
    uv run examples/imu_signal_bench.py --source deterministic --axis acc_z --filter sma
    uv run examples/imu_signal_bench.py --source deterministic-white-noise --axis acc_z
    uv run examples/imu_signal_bench.py --source vesc --axis acc_z --pipeline-depth 4
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from vesc_py import list_serial_ports
from vesc_py.fast_imu_source import (
    DEFAULT_BAUDRATE,
    DEFAULT_PIPELINE_DEPTH,
    DEFAULT_TIMEOUT,
    IMU_BENCH_FIELDS,
    VescImuSignalSource,
    imu_axis_unit,
    parse_imu_axis,
)
from vesc_py.live_signal import (
    DeterministicWhiteNoiseSignalSource,
    DeterministicSignalSource,
    NoisyDeterministicSignalSource,
    SignalRingHistory,
    SignalSource,
    residual,
    trailing_sma,
)

DEFAULT_FILTER = "biquad"
DEFAULT_SMA_WINDOW = 10
DEFAULT_FIR_TAPS = 31
DEFAULT_HISTORY = 20000
DEFAULT_MAX_POINTS = 1200
DEFAULT_PLOT_RATE = 30.0
DEFAULT_PENDING_SAMPLES = 20000
DEFAULT_DETERMINISTIC_RATE = 500.0
DEFAULT_NOISY_DETERMINISTIC_RATE = 200.0
DEFAULT_BIQUAD_NORMALIZED_CUTOFF = 0.0159
DEFAULT_BIQUAD_CUTOFF_HZ = (
    DEFAULT_BIQUAD_NORMALIZED_CUTOFF * DEFAULT_DETERMINISTIC_RATE
)
DEFAULT_FILTER_CUTOFF_HZ = DEFAULT_BIQUAD_CUTOFF_HZ
DEFAULT_BIQUAD_SHAPE = 0.707
QT_XCB_RUNTIME_LIBS = ("libxcb-cursor.so.0", "libxcb-icccm.so.4")

SpectrumMode = Literal["fft", "psd"]
SPECTRUM_MODES: tuple[SpectrumMode, SpectrumMode] = ("fft", "psd")
DEFAULT_SPECTRUM_MODE: SpectrumMode = "psd"

FILTER_LABELS = {
    "biquad": "Biquad IIR LPF",
    "one_pole": "One-pole IIR LPF",
    "fir": "Windowed FIR LPF",
    "sma": "SMA",
}
FILTER_CHOICES = tuple(FILTER_LABELS)


class PlotTheme:
    """Simple color bundle for the PyQtGraph bench."""

    def __init__(
        self,
        *,
        pg_background: str,
        pg_foreground: str,
        window_background: str,
        text_color: str,
        muted_color: str,
        grid_alpha: float,
        line_colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> None:
        self.pg_background = pg_background
        self.pg_foreground = pg_foreground
        self.window_background = window_background
        self.text_color = text_color
        self.muted_color = muted_color
        self.grid_alpha = grid_alpha
        self.line_colors = line_colors


PLOT_THEMES = {
    "light": PlotTheme(
        pg_background="#ffffff",
        pg_foreground="#202124",
        window_background="#f6f7f9",
        text_color="#202124",
        muted_color="#4f5b66",
        grid_alpha=0.22,
        line_colors=((196, 57, 54), (28, 128, 75), (37, 98, 180)),
    ),
    "dark": PlotTheme(
        pg_background="#000000",
        pg_foreground="#d0d0d0",
        window_background="#000000",
        text_color="#d0d0d0",
        muted_color="#999999",
        grid_alpha=0.3,
        line_colors=((230, 88, 85), (80, 190, 120), (85, 150, 245)),
    ),
}


@dataclass(frozen=True)
class SignalStats:
    """Summary metrics for the retained signal window."""

    mean: float
    std: float
    rms: float
    peak_to_peak: float


@dataclass
class Biquad:
    """State and coefficients for one direct-form II biquad section."""

    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    z1: float = 0.0
    z2: float = 0.0


def parse_axis_arg(text: str) -> str:
    """Argparse wrapper for IMU axis parsing."""
    try:
        return parse_imu_axis(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def prefer_qt_xcb_platform() -> None:
    """Prefer Qt's XCB backend when Linux exposes a Wayland/X11 fallback chain."""
    if (
        sys.platform.startswith("linux")
        and "DISPLAY" in os.environ
        and os.environ.get("QT_QPA_PLATFORM") in (None, "", "wayland;xcb")
    ):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def import_pyqtgraph() -> tuple[Any, Any, Any]:
    """Import PyQtGraph lazily so --help does not require Qt startup."""
    prefer_qt_xcb_platform()

    try:
        import pyqtgraph as pg  # type: ignore[import-untyped]
        from pyqtgraph.Qt import QtCore  # type: ignore[import-untyped]
        from pyqtgraph.Qt import QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "PyQtGraph plotting requires pyqtgraph and a Qt binding. "
            "Install the project dependencies, or install pyqtgraph and PySide6."
        ) from exc

    return pg, QtCore, QtWidgets


def require_qt_platform_runtime() -> None:
    """Fail before QApplication aborts when XCB runtime libraries are missing."""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("QT_QPA_PLATFORM") != "xcb":
        return

    missing: list[str] = []
    for lib_name in QT_XCB_RUNTIME_LIBS:
        try:
            ctypes.CDLL(lib_name)
        except OSError:
            missing.append(lib_name)

    if missing:
        raise RuntimeError(
            "Qt's xcb platform plugin is missing runtime libraries: "
            f"{', '.join(missing)}. Run this from the python Nix dev shell "
            "(`nix develop .#python`), or install the matching system packages "
            "(for example libxcb-cursor0 and libxcb-icccm4 on Debian/Ubuntu)."
        )


def autodetect_port() -> str:
    """Return the first discovered serial port, preferring VESC-like devices."""
    ports = list_serial_ports()
    if not ports:
        raise SystemExit(
            "No serial ports found. Connect the VESC over USB or pass --port explicitly."
        )
    return ports[0].system_path


def initial_y_range(axis: str) -> tuple[float, float] | None:
    """Return a sensible initial y-range for a known IMU axis."""
    if axis.startswith("acc_"):
        return (-8.0, 8.0)
    if axis.startswith("gyro_"):
        return (-2000.0, 2000.0)
    if axis in ("roll", "pitch", "yaw"):
        return (-200.0, 200.0)
    return None


def decimate_indexes(sample_count: int, max_points: int) -> npt.NDArray[np.intp] | None:
    """Return display indexes after transform computation."""
    if sample_count <= max_points:
        return None
    return cast(
        "npt.NDArray[np.intp]",
        np.linspace(0, sample_count - 1, max_points, dtype=np.intp),
    )


def history_axis_values(sample_count: int, history: int) -> npt.NDArray[np.float64]:
    """Return x positions in fixed history sample slots."""
    if sample_count <= 0:
        return np.empty(0, dtype=np.float64)
    return cast(
        "npt.NDArray[np.float64]",
        np.arange(history - sample_count, history, dtype=np.float64),
    )


def actual_rate(timestamps: deque[float]) -> float | None:
    """Return rate from recent event timestamps."""
    if len(timestamps) < 2:
        return None
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0.0:
        return None
    return (len(timestamps) - 1) / elapsed


def format_rate(value: float | None) -> str:
    """Format an optional rate for status text."""
    return "measuring" if value is None else f"{value:.1f} Hz"


def format_value(value: float | None, unit: str) -> str:
    """Format an optional signal value for status text."""
    return "n/a" if value is None else f"{value:.6g} {unit}"


def biquad_process(biquad: Biquad, value: float) -> float:
    """Process one sample through a biquad using the VESC firmware form."""
    out = value * biquad.a0 + biquad.z1
    biquad.z1 = value * biquad.a1 + biquad.z2 - biquad.b1 * out
    biquad.z2 = value * biquad.a2 - biquad.b2 * out
    return out


def biquad_config_lowpass(biquad: Biquad, cutoff: float, shape: float) -> None:
    """Configure a low-pass biquad from normalized cutoff and Q shape."""
    if not 0.0 < cutoff < 0.5:
        raise ValueError("cutoff must be greater than 0 and less than 0.5")
    if shape <= 0.0:
        raise ValueError("shape must be greater than 0")

    k = math.tan(math.pi * cutoff)
    norm = 1.0 / (1.0 + k / shape + k * k)
    biquad.a0 = k * k * norm
    biquad.a1 = 2.0 * biquad.a0
    biquad.a2 = biquad.a0
    biquad.b1 = 2.0 * (k * k - 1.0) * norm
    biquad.b2 = (1.0 - k / shape + k * k) * norm


def normalized_biquad_cutoff(cutoff_hz: float, sample_rate_hz: float) -> float:
    """Convert a cutoff in Hz to the normalized Fc used by the biquad."""
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be greater than 0")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be greater than 0")

    cutoff = cutoff_hz / sample_rate_hz
    if cutoff >= 0.5:
        raise ValueError("cutoff_hz must be less than Nyquist")
    return cutoff


def biquad_reset(biquad: Biquad, initial_value: float = 0.0) -> None:
    """Reset a biquad's delay state for a steady initial input."""
    biquad.z1 = initial_value * (1.0 - biquad.a0)
    biquad.z2 = initial_value * (biquad.a2 - biquad.b2)


def biquad_lowpass(
    values: npt.NDArray[np.float64],
    cutoff: float,
    shape: float,
    *,
    initial_value: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Return values filtered by a causal low-pass biquad."""
    if values.size == 0:
        return np.empty(0, dtype=np.float64)

    biquad = Biquad()
    biquad_config_lowpass(biquad, cutoff, shape)
    biquad_reset(biquad, initial_value)

    raw = values.astype(np.float64, copy=False)
    filtered = np.empty(raw.shape, dtype=np.float64)
    for index, value in enumerate(raw):
        filtered[index] = biquad_process(biquad, float(value))
    return filtered


def biquad_lowpass_hz(
    values: npt.NDArray[np.float64],
    cutoff_hz: float,
    sample_rate_hz: float,
    shape: float,
    *,
    initial_value: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Return values filtered by a causal low-pass biquad using cutoff in Hz."""
    return biquad_lowpass(
        values,
        normalized_biquad_cutoff(cutoff_hz, sample_rate_hz),
        shape,
        initial_value=initial_value,
    )


def one_pole_lowpass_hz(
    values: npt.NDArray[np.float64],
    cutoff_hz: float,
    sample_rate_hz: float,
    *,
    initial_value: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Return values filtered by a causal one-pole low-pass IIR."""
    normalized_biquad_cutoff(cutoff_hz, sample_rate_hz)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)

    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / sample_rate_hz)
    raw = values.astype(np.float64, copy=False)
    filtered = np.empty(raw.shape, dtype=np.float64)
    state = initial_value
    for index, value in enumerate(raw):
        state += alpha * (float(value) - state)
        filtered[index] = state
    return filtered


def fir_lowpass_coefficients(
    cutoff_hz: float,
    sample_rate_hz: float,
    taps: int,
) -> npt.NDArray[np.float64]:
    """Return Hamming-windowed sinc low-pass FIR coefficients."""
    if taps <= 0:
        raise ValueError("taps must be greater than 0")

    normalized_cutoff = normalized_biquad_cutoff(cutoff_hz, sample_rate_hz)
    sample_indexes = np.arange(taps, dtype=np.float64)
    center = (float(taps) - 1.0) / 2.0
    coefficients = (
        2.0
        * normalized_cutoff
        * np.sinc(2.0 * normalized_cutoff * (sample_indexes - center))
    )
    coefficients *= np.hamming(taps)
    coefficient_sum = float(np.sum(coefficients))
    if coefficient_sum == 0.0:
        raise ValueError("FIR coefficient sum is zero")
    return coefficients / coefficient_sum


def fir_lowpass_hz(
    values: npt.NDArray[np.float64],
    cutoff_hz: float,
    sample_rate_hz: float,
    taps: int,
    *,
    pad_initial: bool = False,
) -> npt.NDArray[np.float64]:
    """Return values filtered by a causal windowed-sinc low-pass FIR."""
    if values.size == 0:
        return np.empty(0, dtype=np.float64)

    coefficients = fir_lowpass_coefficients(cutoff_hz, sample_rate_hz, taps)
    raw = values.astype(np.float64, copy=False)
    sample_count = int(raw.size)
    if pad_initial and coefficients.size > 1:
        raw = np.pad(raw, (coefficients.size - 1, 0), mode="edge")
        start = int(coefficients.size - 1)
    else:
        start = 0
    filtered = np.convolve(raw, coefficients, mode="full")
    return filtered[start : start + sample_count]


def signal_stats(values: npt.NDArray[np.float64]) -> SignalStats | None:
    """Return summary metrics for the supplied signal values."""
    if values.size == 0:
        return None
    return SignalStats(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        rms=float(np.sqrt(np.mean(values * values))),
        peak_to_peak=float(np.ptp(values)),
    )


def _signal_frequency_window(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float] | None:
    sample_count = int(min(timestamps.size, values.size))
    if sample_count < 2:
        return None

    window_timestamps = timestamps[-sample_count:]
    window_values = values[-sample_count:]
    sample_periods = np.diff(window_timestamps)
    sample_periods = sample_periods[sample_periods > 0.0]
    if sample_periods.size == 0:
        return None

    sample_period = float(np.median(sample_periods))
    if not np.isfinite(sample_period) or sample_period <= 0.0:
        return None

    return window_values, sample_period


def signal_fft(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Return one-sided FFT magnitude bins for a scalar signal window."""
    frequency_window = _signal_frequency_window(timestamps, values)
    if frequency_window is None:
        return None

    window_values, sample_period = frequency_window
    sample_count = int(window_values.size)
    centered = window_values.astype(np.float64, copy=False) - float(
        np.mean(window_values)
    )
    magnitudes = np.abs(np.fft.rfft(centered)) / sample_count
    if magnitudes.size > 1:
        if sample_count % 2 == 0:
            magnitudes[1:-1] *= 2.0
        else:
            magnitudes[1:] *= 2.0

    frequencies = cast(
        "npt.NDArray[np.float64]",
        np.fft.rfftfreq(sample_count, d=sample_period),
    )
    return frequencies, magnitudes


def signal_psd(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Return one-sided PSD bins for a scalar signal window."""
    frequency_window = _signal_frequency_window(timestamps, values)
    if frequency_window is None:
        return None

    window_values, sample_period = frequency_window
    sample_count = int(window_values.size)
    sample_rate_hz = 1.0 / sample_period
    window = np.hanning(sample_count)
    if not np.any(window):
        window = np.ones(sample_count, dtype=np.float64)

    centered = window_values.astype(np.float64, copy=False) - float(
        np.mean(window_values)
    )
    spectrum = np.fft.rfft(centered * window)
    scale = sample_rate_hz * float(np.sum(window * window))
    if scale <= 0.0:
        return None

    psd = (np.abs(spectrum) ** 2) / scale
    if psd.size > 1:
        if sample_count % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0

    frequencies = cast(
        "npt.NDArray[np.float64]",
        np.fft.rfftfreq(sample_count, d=sample_period),
    )
    return frequencies, psd


def signal_frequency_analysis(
    mode: SpectrumMode,
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Return scalar frequency analysis bins for the selected mode."""
    if mode == "fft":
        return signal_fft(timestamps, values)
    return signal_psd(timestamps, values)


def format_stats(stats: SignalStats | None, unit: str) -> str:
    """Format optional signal metrics for status text."""
    if stats is None:
        return "\n".join(
            (
                "mean: n/a",
                "std: n/a",
                "RMS: n/a",
                "peak-to-peak: n/a",
            )
        )
    return (
        f"mean: {stats.mean:.6g} {unit}\n"
        f"std: {stats.std:.6g} {unit}\n"
        f"RMS: {stats.rms:.6g} {unit}\n"
        f"peak-to-peak: {stats.peak_to_peak:.6g} {unit}"
    )


def run_signal_bench(
    source: SignalSource,
    *,
    source_label: str,
    initial_filter: str,
    initial_filter_cutoff_hz: float,
    initial_biquad_shape: float,
    initial_sma_window: int,
    initial_fir_taps: int,
    history: int,
    max_points: int,
    plot_rate: float,
    theme: str,
    antialias: bool,
) -> None:
    """Run the PyQtGraph live signal bench."""
    selected_theme = PLOT_THEMES[theme]
    pg, QtCore, QtWidgets = import_pyqtgraph()
    require_qt_platform_runtime()
    pg.setConfigOptions(
        antialias=antialias,
        background=selected_theme.pg_background,
        foreground=selected_theme.pg_foreground,
    )

    app = pg.mkQApp("VESC IMU Signal Bench")
    window = QtWidgets.QWidget()
    window.setWindowTitle("VESC IMU Signal Bench")
    window.resize(1280, 720)
    window.setStyleSheet(f"background-color: {selected_theme.window_background};")

    qt_alignment = getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt)
    qt_size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)

    root = QtWidgets.QVBoxLayout(window)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(8)

    controls = QtWidgets.QHBoxLayout()
    controls.setContentsMargins(0, 0, 0, 0)
    controls.setSpacing(8)
    root.addLayout(controls)

    source_text = QtWidgets.QLabel(source_label)
    source_text.setStyleSheet(f"color: {selected_theme.text_color}; font-weight: 700;")
    controls.addWidget(source_text)

    axis_text = QtWidgets.QLabel(f"{source.channel_name} ({source.unit})")
    axis_text.setStyleSheet(f"color: {selected_theme.muted_color};")
    controls.addWidget(axis_text)

    controls.addSpacing(12)
    controls.addWidget(QtWidgets.QLabel("Filter"))
    filter_combo = QtWidgets.QComboBox()
    for filter_name, label in FILTER_LABELS.items():
        filter_combo.addItem(label, filter_name)
    initial_filter_index = filter_combo.findData(initial_filter)
    filter_combo.setCurrentIndex(max(0, initial_filter_index))
    controls.addWidget(filter_combo)

    cutoff_label = QtWidgets.QLabel("Cutoff Hz")
    controls.addWidget(cutoff_label)
    cutoff_spin = QtWidgets.QDoubleSpinBox()
    cutoff_spin.setRange(0.001, 100000.0)
    cutoff_spin.setDecimals(3)
    cutoff_spin.setSingleStep(0.5)
    cutoff_spin.setValue(initial_filter_cutoff_hz)
    controls.addWidget(cutoff_spin)

    shape_label = QtWidgets.QLabel("Q")
    controls.addWidget(shape_label)
    shape_spin = QtWidgets.QDoubleSpinBox()
    shape_spin.setRange(0.05, 10.0)
    shape_spin.setDecimals(3)
    shape_spin.setSingleStep(0.05)
    shape_spin.setValue(initial_biquad_shape)
    controls.addWidget(shape_spin)

    sma_label = QtWidgets.QLabel("SMA")
    controls.addWidget(sma_label)
    sma_spin = QtWidgets.QSpinBox()
    sma_spin.setRange(1, 100000)
    sma_spin.setValue(initial_sma_window)
    controls.addWidget(sma_spin)

    fir_taps_label = QtWidgets.QLabel("FIR taps")
    controls.addWidget(fir_taps_label)
    fir_taps_spin = QtWidgets.QSpinBox()
    fir_taps_spin.setRange(1, 10001)
    fir_taps_spin.setSingleStep(2)
    fir_taps_spin.setValue(initial_fir_taps)
    controls.addWidget(fir_taps_spin)

    mode_group = QtWidgets.QButtonGroup(window)
    mode_group.setExclusive(True)
    mode_buttons: dict[str, Any] = {}
    for mode_name in ("raw", "filtered", "both", "residual"):
        button = QtWidgets.QPushButton(mode_name.title())
        button.setCheckable(True)
        if mode_name == "both":
            button.setChecked(True)
        mode_group.addButton(button)
        mode_buttons[mode_name] = button
        controls.addWidget(button)

    spectrum_label = QtWidgets.QLabel("Spectrum")
    controls.addWidget(spectrum_label)
    spectrum_mode_group = QtWidgets.QButtonGroup(window)
    spectrum_mode_group.setExclusive(True)
    spectrum_mode_buttons: dict[SpectrumMode, Any] = {}
    for spectrum_mode in SPECTRUM_MODES:
        button = QtWidgets.QPushButton(spectrum_mode.upper())
        button.setCheckable(True)
        if spectrum_mode == DEFAULT_SPECTRUM_MODE:
            button.setChecked(True)
        spectrum_mode_group.addButton(button)
        spectrum_mode_buttons[spectrum_mode] = button
        controls.addWidget(button)

    clear_button = QtWidgets.QPushButton("Clear")
    controls.addWidget(clear_button)
    controls.addStretch(1)

    metrics_text = QtWidgets.QLabel(format_stats(None, source.unit))
    metrics_text.setAlignment(qt_alignment.AlignLeft)
    metrics_text.setStyleSheet(
        f"color: {selected_theme.text_color}; font-family: monospace;"
    )
    root.addWidget(metrics_text)

    plot_row = QtWidgets.QHBoxLayout()
    plot_row.setContentsMargins(0, 0, 0, 0)
    plot_row.setSpacing(8)
    root.addLayout(plot_row, stretch=1)

    plot_widget = pg.PlotWidget(title="Time Series")
    spectrum_widget = pg.PlotWidget(title=DEFAULT_SPECTRUM_MODE.upper())
    for widget in (plot_widget, spectrum_widget):
        widget.setMinimumSize(0, 0)
        widget.setSizePolicy(qt_size_policy.Ignored, qt_size_policy.Ignored)
        plot_row.addWidget(widget, stretch=1)

    status = QtWidgets.QLabel("Waiting for signal data...")
    status.setAlignment(qt_alignment.AlignLeft)
    status.setWordWrap(True)
    status.setStyleSheet(f"color: {selected_theme.muted_color};")
    root.addWidget(status)

    plot = plot_widget.getPlotItem()
    plot.showGrid(x=True, y=True, alpha=selected_theme.grid_alpha)
    plot.setLabel("left", source.channel_name, units=source.unit)
    plot.setLabel("bottom", "history", units="samples")
    plot.setMouseEnabled(x=False, y=True)
    plot.setXRange(0.0, float(history), padding=0.0)
    plot.setLimits(
        xMin=0.0,
        xMax=float(history),
        minXRange=float(history),
        maxXRange=float(history),
    )
    y_range = initial_y_range(source.channel_name)
    if y_range is not None:
        plot.setYRange(*y_range, padding=0.0)
        plot.enableAutoRange(axis="y", enable=False)
    for method_name, args in (
        ("setClipToView", (True,)),
        ("setDownsampling", (1, True, "peak")),
    ):
        method = getattr(plot, method_name, None)
        if method is not None:
            method(*args)

    empty = np.empty(0, dtype=np.float64)
    raw_line = pg.PlotCurveItem(
        empty,
        empty,
        pen=pg.mkPen(selected_theme.line_colors[0], width=1.3),
        name="Raw",
        connect="all",
        skipFiniteCheck=True,
    )
    filtered_line = pg.PlotCurveItem(
        empty,
        empty,
        pen=pg.mkPen(selected_theme.line_colors[1], width=1.6),
        name="Filtered",
        connect="all",
        skipFiniteCheck=True,
    )
    residual_line = pg.PlotCurveItem(
        empty,
        empty,
        pen=pg.mkPen(selected_theme.line_colors[2], width=1.4),
        name="Residual",
        connect="all",
        skipFiniteCheck=True,
    )
    for line in (raw_line, filtered_line, residual_line):
        line.setSkipFiniteCheck(True)
        plot.addItem(line)
    legend = plot.addLegend(offset=(10, 10))
    zero_line = pg.InfiniteLine(
        pos=0.0,
        angle=0,
        pen=pg.mkPen(selected_theme.muted_color, width=1),
    )
    zero_line.setVisible(False)
    plot.addItem(zero_line)

    spectrum_plot = spectrum_widget.getPlotItem()
    spectrum_plot.showGrid(x=True, y=True, alpha=selected_theme.grid_alpha)
    spectrum_plot.setLabel("left", "PSD", units=f"{source.unit}^2/Hz")
    spectrum_plot.setLabel("bottom", "frequency", units="Hz")
    spectrum_plot.setMouseEnabled(x=False, y=True)
    spectrum_plot.setLimits(xMin=0.0, yMin=0.0)
    for method_name, args in (
        ("setClipToView", (True,)),
        ("setDownsampling", (1, True, "peak")),
    ):
        method = getattr(spectrum_plot, method_name, None)
        if method is not None:
            method(*args)

    raw_spectrum_line = pg.PlotCurveItem(
        empty,
        empty,
        pen=pg.mkPen(selected_theme.line_colors[0], width=1.3),
        name="Raw",
        connect="all",
        skipFiniteCheck=True,
    )
    filtered_spectrum_line = pg.PlotCurveItem(
        empty,
        empty,
        pen=pg.mkPen(selected_theme.line_colors[1], width=1.6),
        name="Filtered",
        connect="all",
        skipFiniteCheck=True,
    )
    residual_spectrum_line = pg.PlotCurveItem(
        empty,
        empty,
        pen=pg.mkPen(selected_theme.line_colors[2], width=1.4),
        name="Residual",
        connect="all",
        skipFiniteCheck=True,
    )
    for line in (raw_spectrum_line, filtered_spectrum_line, residual_spectrum_line):
        line.setSkipFiniteCheck(True)
        spectrum_plot.addItem(line)
    spectrum_legend = spectrum_plot.addLegend(offset=(10, 10))

    signal_history = SignalRingHistory(history)
    refresh_timestamps: deque[float] = deque(maxlen=120)
    dropped_pending = 0
    latest_raw: float | None = None
    latest_filtered: float | None = None
    latest_residual: float | None = None
    latest_stats: SignalStats | None = None
    initial_filter_label = FILTER_LABELS.get(
        initial_filter,
        FILTER_LABELS[DEFAULT_FILTER],
    )
    latest_filter_status = f"{initial_filter_label}: n/a"

    def selected_mode() -> str:
        for mode_name, button in mode_buttons.items():
            if button.isChecked():
                return mode_name
        return "both"

    def selected_spectrum_mode() -> SpectrumMode:
        for mode_name, button in spectrum_mode_buttons.items():
            if button.isChecked():
                return mode_name
        return DEFAULT_SPECTRUM_MODE

    def selected_filter() -> str:
        filter_name = filter_combo.currentData()
        if isinstance(filter_name, str) and filter_name in FILTER_LABELS:
            return filter_name
        return DEFAULT_FILTER

    def update_spectrum_labels(mode_name: SpectrumMode) -> None:
        if mode_name == "fft":
            spectrum_plot.setTitle("FFT")
            spectrum_plot.setLabel("left", "FFT magnitude", units=source.unit)
        else:
            spectrum_plot.setTitle("PSD")
            spectrum_plot.setLabel("left", "PSD", units=f"{source.unit}^2/Hz")

    def clear_line_data(line: Any) -> None:
        line.setData(x=empty, y=empty, connect="all", skipFiniteCheck=True)

    def clear_spectrum_lines() -> None:
        for line in (
            raw_spectrum_line,
            filtered_spectrum_line,
            residual_spectrum_line,
        ):
            clear_line_data(line)

    def update_filter_controls() -> None:
        filter_name = selected_filter()
        uses_cutoff = filter_name in ("biquad", "one_pole", "fir")
        uses_shape = filter_name == "biquad"
        uses_sma = filter_name == "sma"
        uses_fir = filter_name == "fir"
        for widget in (cutoff_label, cutoff_spin):
            widget.setVisible(uses_cutoff)
        for widget in (shape_label, shape_spin):
            widget.setVisible(uses_shape)
        for widget in (sma_label, sma_spin):
            widget.setVisible(uses_sma)
        for widget in (fir_taps_label, fir_taps_spin):
            widget.setVisible(uses_fir)

    def clear_history() -> None:
        nonlocal latest_raw, latest_filtered, latest_residual, latest_stats
        nonlocal latest_filter_status
        signal_history.clear()
        latest_raw = None
        latest_filtered = None
        latest_residual = None
        latest_stats = None
        latest_filter_status = f"{FILTER_LABELS[selected_filter()]}: n/a"
        for line in (
            raw_line,
            filtered_line,
            residual_line,
            raw_spectrum_line,
            filtered_spectrum_line,
            residual_spectrum_line,
        ):
            clear_line_data(line)

    def set_visible_lines(mode_name: str) -> None:
        raw_line.setVisible(mode_name in ("raw", "both"))
        filtered_line.setVisible(mode_name in ("filtered", "both"))
        residual_line.setVisible(mode_name == "residual")
        raw_spectrum_line.setVisible(mode_name in ("raw", "both"))
        filtered_spectrum_line.setVisible(mode_name in ("filtered", "both"))
        residual_spectrum_line.setVisible(mode_name == "residual")
        zero_line.setVisible(mode_name == "residual")
        legend.setVisible(mode_name == "both")
        spectrum_legend.setVisible(mode_name == "both")

    def set_spectrum_data(
        line: Any,
        spectrum_mode: SpectrumMode,
        timestamps: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> float | None:
        spectrum_data = signal_frequency_analysis(spectrum_mode, timestamps, values)
        if spectrum_data is None:
            clear_line_data(line)
            return None

        frequencies, spectrum_values = spectrum_data
        indexes = decimate_indexes(int(frequencies.size), max_points)
        if indexes is not None:
            frequencies = frequencies[indexes]
            spectrum_values = spectrum_values[indexes]

        line.setData(
            x=frequencies,
            y=spectrum_values,
            connect="all",
            skipFiniteCheck=True,
        )
        return float(frequencies[-1]) if frequencies.size > 0 else None

    def refresh_plot() -> None:
        nonlocal dropped_pending, latest_raw, latest_filtered, latest_residual
        nonlocal latest_stats, latest_filter_status

        timestamps, values, dropped = source.drain()
        dropped_pending += dropped
        if timestamps.size > 0:
            refresh_timestamps.append(time.monotonic())
            signal_history.append_samples(timestamps, values)

        raw_values = signal_history.valid_values()
        if raw_values.size == 0:
            set_visible_lines(selected_mode())
            return

        timestamps_values = signal_history.valid_timestamps()
        x_values = history_axis_values(int(raw_values.size), history)
        snapshot = source.snapshot()
        sample_rate_hz = signal_history.sample_hz()
        if sample_rate_hz is None and snapshot.average_rate_hz > 0.0:
            sample_rate_hz = snapshot.average_rate_hz

        filter_name = selected_filter()
        if filter_name == "sma":
            sma_window = int(sma_spin.value())
            filtered_values = trailing_sma(raw_values, sma_window)
            latest_filter_status = f"{FILTER_LABELS[filter_name]} window: {sma_window}"
        elif sample_rate_hz is None:
            filtered_values = raw_values.astype(np.float64, copy=True)
            latest_filter_status = (
                f"{FILTER_LABELS[filter_name]}: waiting for sample rate"
            )
        else:
            initial_filter_value = float(raw_values[0])
            cutoff_hz = float(cutoff_spin.value())
            effective_cutoff_hz = min(cutoff_hz, sample_rate_hz * 0.499)
            if filter_name == "biquad":
                filter_extra = f" | Q: {float(shape_spin.value()):.3g}"
                filtered_values = biquad_lowpass_hz(
                    raw_values,
                    effective_cutoff_hz,
                    sample_rate_hz,
                    float(shape_spin.value()),
                    initial_value=initial_filter_value,
                )
            elif filter_name == "one_pole":
                filter_extra = ""
                filtered_values = one_pole_lowpass_hz(
                    raw_values,
                    effective_cutoff_hz,
                    sample_rate_hz,
                    initial_value=initial_filter_value,
                )
            else:
                fir_taps = int(fir_taps_spin.value())
                filter_extra = f" | taps: {fir_taps}"
                filtered_values = fir_lowpass_hz(
                    raw_values,
                    effective_cutoff_hz,
                    sample_rate_hz,
                    fir_taps,
                    pad_initial=True,
                )
            normalized_cutoff = normalized_biquad_cutoff(
                effective_cutoff_hz,
                sample_rate_hz,
            )
            cutoff_text = f"{cutoff_hz:.3g} Hz"
            if effective_cutoff_hz < cutoff_hz:
                cutoff_text = (
                    f"{cutoff_hz:.3g} Hz "
                    f"(effective {effective_cutoff_hz:.3g} Hz)"
                )
            latest_filter_status = (
                f"{FILTER_LABELS[filter_name]} cutoff: {cutoff_text} | "
                f"Fc: {normalized_cutoff:.5g} | "
                f"sample: {format_rate(sample_rate_hz)}{filter_extra}"
            )
        residual_values = residual(raw_values, filtered_values)
        latest_raw = float(raw_values[-1])
        latest_filtered = float(filtered_values[-1])
        latest_residual = float(residual_values[-1])
        latest_stats = signal_stats(filtered_values)

        indexes = decimate_indexes(int(x_values.size), max_points)
        if indexes is not None:
            plot_x_values = x_values[indexes]
            plot_raw_values = raw_values[indexes]
            plot_filtered_values = filtered_values[indexes]
            plot_residual_values = residual_values[indexes]
        else:
            plot_x_values = x_values
            plot_raw_values = raw_values
            plot_filtered_values = filtered_values
            plot_residual_values = residual_values

        raw_line.setData(
            x=plot_x_values,
            y=plot_raw_values,
            connect="all",
            skipFiniteCheck=True,
        )
        filtered_line.setData(
            x=plot_x_values,
            y=plot_filtered_values,
            connect="all",
            skipFiniteCheck=True,
        )
        residual_line.setData(
            x=plot_x_values,
            y=plot_residual_values,
            connect="all",
            skipFiniteCheck=True,
        )
        display_mode = selected_mode()
        spectrum_mode = selected_spectrum_mode()
        raw_nyquist = (
            set_spectrum_data(
                raw_spectrum_line,
                spectrum_mode,
                timestamps_values,
                raw_values,
            )
            if display_mode in ("raw", "both")
            else None
        )
        filtered_nyquist = (
            set_spectrum_data(
                filtered_spectrum_line,
                spectrum_mode,
                timestamps_values,
                filtered_values,
            )
            if display_mode in ("filtered", "both")
            else None
        )
        residual_nyquist = (
            set_spectrum_data(
                residual_spectrum_line,
                spectrum_mode,
                timestamps_values,
                residual_values,
            )
            if display_mode == "residual"
            else None
        )
        if display_mode not in ("raw", "both"):
            clear_line_data(raw_spectrum_line)
        if display_mode not in ("filtered", "both"):
            clear_line_data(filtered_spectrum_line)
        if display_mode != "residual":
            clear_line_data(residual_spectrum_line)
        nyquist_values = (
            value
            for value in (raw_nyquist, filtered_nyquist, residual_nyquist)
            if value is not None
        )
        nyquist_hz = max(
            nyquist_values,
            default=None,
        )
        if nyquist_hz is not None and nyquist_hz > 0.0:
            spectrum_plot.setXRange(0.0, nyquist_hz, padding=0.0)
        set_visible_lines(display_mode)
        plot.setXRange(0.0, float(history), padding=0.0)

    def refresh_status() -> None:
        snapshot = source.snapshot()
        history_hz = signal_history.sample_hz()
        plot_hz = actual_rate(refresh_timestamps)
        state = "stopped" if snapshot.done else "running"
        error_text = (
            f" | error: {snapshot.last_error}"
            if snapshot.last_error is not None
            else ""
        )
        metrics_text.setText(format_stats(latest_stats, source.unit))
        status.setText(
            f"{state} | raw: {format_value(latest_raw, source.unit)} | "
            f"filtered: {format_value(latest_filtered, source.unit)} | "
            f"residual: {format_value(latest_residual, source.unit)} | "
            f"filter: {latest_filter_status} | samples: {snapshot.samples} | "
            f"source avg: {snapshot.average_rate_hz:.1f} Hz | "
            f"history: {format_rate(history_hz)} | plot: {format_rate(plot_hz)} | "
            f"spectrum: {selected_spectrum_mode().upper()} | "
            f"dropped: {snapshot.dropped + dropped_pending} | errors: {snapshot.errors}"
            f"{error_text}"
        )

    plot_timer = QtCore.QTimer()
    plot_timer.setInterval(round(1000.0 / plot_rate))
    plot_timer.timeout.connect(refresh_plot)

    status_timer = QtCore.QTimer()
    status_timer.setInterval(250)
    status_timer.timeout.connect(refresh_status)

    def stop_updates(*_args: object) -> None:
        plot_timer.stop()
        status_timer.stop()

    def apply_spectrum_mode() -> None:
        update_spectrum_labels(selected_spectrum_mode())
        clear_spectrum_lines()
        refresh_plot()
        refresh_status()

    clear_button.clicked.connect(clear_history)
    filter_combo.currentIndexChanged.connect(lambda _index=0: update_filter_controls())
    for button in mode_buttons.values():
        button.clicked.connect(lambda _checked=False: refresh_plot())
    for button in spectrum_mode_buttons.values():
        button.clicked.connect(lambda _checked=False: apply_spectrum_mode())
    window.destroyed.connect(stop_updates)

    update_filter_controls()
    update_spectrum_labels(DEFAULT_SPECTRUM_MODE)
    set_visible_lines("both")
    refresh_plot()
    refresh_status()
    window.show()
    source.start()
    plot_timer.start()
    status_timer.start()

    exec_app = getattr(app, "exec", None)
    if exec_app is None:
        exec_app = app.exec_
    try:
        exec_app()
    finally:
        source.stop()


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Live one-axis IMU signal bench for software filter tuning.",
    )
    parser.add_argument(
        "--source",
        choices=(
            "vesc",
            "deterministic",
            "deterministic-noisy",
            "deterministic-white-noise",
        ),
        default="vesc",
        help="Signal source to use (default: vesc).",
    )
    parser.add_argument(
        "--port",
        help="Serial port path for --source vesc. If omitted, autodetect is used.",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help=f"Serial baudrate for --source vesc (default: {DEFAULT_BAUDRATE}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SEC",
        help=f"Per-response timeout for --source vesc (default: {DEFAULT_TIMEOUT:g}).",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=DEFAULT_PIPELINE_DEPTH,
        metavar="N",
        help=(
            "Outstanding IMU requests for --source vesc "
            f"(default: {DEFAULT_PIPELINE_DEPTH})."
        ),
    )
    parser.add_argument(
        "--axis",
        type=parse_axis_arg,
        default="acc_x",
        metavar="AXIS",
        help=f"IMU axis to capture. Valid: {', '.join(IMU_BENCH_FIELDS)}.",
    )
    parser.add_argument(
        "--filter",
        choices=FILTER_CHOICES,
        default=DEFAULT_FILTER,
        help=f"Initial filter to apply (default: {DEFAULT_FILTER}).",
    )
    parser.add_argument(
        "--filter-cutoff-hz",
        "--biquad-cutoff-hz",
        dest="filter_cutoff_hz",
        type=float,
        default=DEFAULT_FILTER_CUTOFF_HZ,
        metavar="HZ",
        help=(
            "Initial low-pass cutoff in Hz for cutoff-based filters "
            f"(default: {DEFAULT_FILTER_CUTOFF_HZ:g}; equivalent to "
            f"Fc={DEFAULT_BIQUAD_NORMALIZED_CUTOFF:g} at "
            f"{DEFAULT_DETERMINISTIC_RATE:g} Hz). The --biquad-cutoff-hz "
            "alias is kept for existing command lines."
        ),
    )
    parser.add_argument(
        "--biquad-shape",
        type=float,
        default=DEFAULT_BIQUAD_SHAPE,
        metavar="Q",
        help=(
            "Initial low-pass biquad shape/Q "
            f"(default: {DEFAULT_BIQUAD_SHAPE:g})."
        ),
    )
    parser.add_argument(
        "--sma-window",
        type=int,
        default=DEFAULT_SMA_WINDOW,
        metavar="N",
        help=f"Initial SMA sample window (default: {DEFAULT_SMA_WINDOW}).",
    )
    parser.add_argument(
        "--fir-taps",
        type=int,
        default=DEFAULT_FIR_TAPS,
        metavar="N",
        help=f"Initial FIR tap count (default: {DEFAULT_FIR_TAPS}).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=DEFAULT_HISTORY,
        metavar="N",
        help=f"Retained raw sample history (default: {DEFAULT_HISTORY}).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        metavar="N",
        help=f"Maximum plotted points after decimation (default: {DEFAULT_MAX_POINTS}).",
    )
    parser.add_argument(
        "--plot-rate",
        type=float,
        default=DEFAULT_PLOT_RATE,
        metavar="HZ",
        help=f"Plot redraw rate (default: {DEFAULT_PLOT_RATE:g}).",
    )
    parser.add_argument(
        "--pending-samples",
        type=int,
        default=DEFAULT_PENDING_SAMPLES,
        metavar="N",
        help=(
            "Pending sample ring size between source and UI "
            f"(default: {DEFAULT_PENDING_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--theme",
        choices=tuple(PLOT_THEMES),
        default="light",
        help="Plot theme (default: light).",
    )
    parser.add_argument(
        "--antialias",
        action="store_true",
        help="Enable PyQtGraph antialiasing. Disabled by default for speed.",
    )
    parser.add_argument(
        "--no-exclusive",
        action="store_true",
        help="Do not request exclusive serial access for --source vesc.",
    )
    parser.add_argument(
        "--deterministic-rate",
        type=float,
        default=DEFAULT_DETERMINISTIC_RATE,
        metavar="HZ",
        help=(
            "Sample rate for --source deterministic and "
            "--source deterministic-white-noise "
            f"(default: {DEFAULT_DETERMINISTIC_RATE:g}). "
            "--source deterministic-noisy always uses 200 Hz."
        ),
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate parsed arguments with concrete CLI errors."""
    if args.baudrate <= 0:
        parser.error("--baudrate must be greater than 0")
    if args.timeout <= 0.0:
        parser.error("--timeout must be greater than 0")
    if args.pipeline_depth <= 0:
        parser.error("--pipeline-depth must be greater than 0")
    if args.filter_cutoff_hz <= 0.0:
        parser.error("--filter-cutoff-hz must be greater than 0")
    if args.biquad_shape <= 0.0:
        parser.error("--biquad-shape must be greater than 0")
    if args.sma_window <= 0:
        parser.error("--sma-window must be greater than 0")
    if args.fir_taps <= 0:
        parser.error("--fir-taps must be greater than 0")
    if args.history <= 0:
        parser.error("--history must be greater than 0")
    if args.max_points <= 0:
        parser.error("--max-points must be greater than 0")
    if args.plot_rate <= 0.0:
        parser.error("--plot-rate must be greater than 0")
    if args.pending_samples <= 0:
        parser.error("--pending-samples must be greater than 0")
    if args.deterministic_rate <= 0.0:
        parser.error("--deterministic-rate must be greater than 0")


def make_source(args: argparse.Namespace) -> tuple[SignalSource, str]:
    """Construct the requested signal source."""
    axis = cast(str, args.axis)
    if args.source == "deterministic":
        return (
            DeterministicSignalSource(
                channel_name=axis,
                unit=imu_axis_unit(axis),
                sample_rate_hz=cast(float, args.deterministic_rate),
                pending_samples=cast(int, args.pending_samples),
            ),
            f"Deterministic source @ {args.deterministic_rate:g} Hz",
        )
    if args.source == "deterministic-noisy":
        return (
            NoisyDeterministicSignalSource(
                channel_name=axis,
                unit=imu_axis_unit(axis),
                sample_rate_hz=DEFAULT_NOISY_DETERMINISTIC_RATE,
                pending_samples=cast(int, args.pending_samples),
            ),
            f"Deterministic noisy source @ {DEFAULT_NOISY_DETERMINISTIC_RATE:g} Hz",
        )
    if args.source == "deterministic-white-noise":
        return (
            DeterministicWhiteNoiseSignalSource(
                channel_name=axis,
                unit=imu_axis_unit(axis),
                sample_rate_hz=cast(float, args.deterministic_rate),
                pending_samples=cast(int, args.pending_samples),
            ),
            f"Deterministic white noise source @ {args.deterministic_rate:g} Hz",
        )

    port = cast(str | None, args.port) or autodetect_port()
    return (
        VescImuSignalSource(
            port=port,
            baudrate=cast(int, args.baudrate),
            axis=axis,
            timeout=cast(float, args.timeout),
            pipeline_depth=cast(int, args.pipeline_depth),
            pending_samples=cast(int, args.pending_samples),
            exclusive=not cast(bool, args.no_exclusive),
        ),
        f"VESC serial source: {port}",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    source, source_label = make_source(args)
    run_signal_bench(
        source,
        source_label=source_label,
        initial_filter=cast(str, args.filter),
        initial_filter_cutoff_hz=cast(float, args.filter_cutoff_hz),
        initial_biquad_shape=cast(float, args.biquad_shape),
        initial_sma_window=cast(int, args.sma_window),
        initial_fir_taps=cast(int, args.fir_taps),
        history=cast(int, args.history),
        max_points=cast(int, args.max_points),
        plot_rate=cast(float, args.plot_rate),
        theme=cast(str, args.theme),
        antialias=cast(bool, args.antialias),
    )


if __name__ == "__main__":
    main()
