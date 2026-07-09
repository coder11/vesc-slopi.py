#!/usr/bin/env python3
"""Plot high-rate IMU data from the IMU Streamer VESC package.

The package must already be loaded on the directly connected controller.
The Python side sends the package start command, receives timestamped
``COMM_CUSTOM_APP_DATA`` samples, and stops the package when the plot exits.

Examples:
    uv run examples/poll_imu_from_package.py --serial /dev/ttyACM0
    uv run examples/poll_imu_from_package.py --serial /dev/ttyACM0 --show-freq
    uv run examples/poll_imu_from_package.py --serial /dev/ttyACM0 --theme dark
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
from typing import Any

import numpy as np
import numpy.typing as npt

from vesc_py.connection import VescConnectionKind
from vesc_py.connection_cli import (
    add_vesc_connection_arguments,
    resolve_vesc_connection_from_args,
)
from vesc_py.imu_streamer import (
    DEFAULT_PENDING_SAMPLES,
    DEFAULT_READ_CHUNK_SIZE,
    DEFAULT_TIMEOUT,
    IMU_STREAMER_CHANNEL_COUNT,
    VescImuStreamer,
)

DEFAULT_HISTORY = 3000
DEFAULT_PLOT_REFRESH_HZ = 30.0
DEFAULT_SPECTRUM_REFRESH_HZ = 2.0
DEFAULT_STATUS_REFRESH_HZ = 4.0
DEFAULT_THEME = "light"
QT_XCB_RUNTIME_LIBS = ("libxcb-cursor.so.0", "libxcb-icccm.so.4")

ACC_X_INDEX = 0
ACC_Y_INDEX = 1
ACC_Z_INDEX = 2
GYRO_X_INDEX = 3
GYRO_Y_INDEX = 4
GYRO_Z_INDEX = 5

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PlotTheme:
    """Colors for the live plot window and PyQtGraph widgets."""

    pg_background: str
    pg_foreground: str
    window_background: str
    title_color: str
    status_color: str
    grid_alpha: float
    line_colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]


PLOT_THEMES: dict[str, PlotTheme] = {
    "light": PlotTheme(
        pg_background="#ffffff",
        pg_foreground="#202124",
        window_background="#f6f7f9",
        title_color="#202124",
        status_color="#4f5b66",
        grid_alpha=0.22,
        line_colors=((196, 57, 54), (28, 128, 75), (37, 98, 180)),
    ),
    "dark": PlotTheme(
        pg_background="#000000",
        pg_foreground="#d0d0d0",
        window_background="#000000",
        title_color="#999999",
        status_color="#999999",
        grid_alpha=0.3,
        line_colors=((230, 88, 85), (80, 190, 120), (85, 150, 245)),
    ),
}


@dataclass
class FrequencyAxisRange:
    """Track frequency plot bounds so range changes are not forced every FFT."""

    x_max: float = 0.0
    y_max: float = 0.0


class ImuStreamerHistory:
    """Fixed-size ring history for package-streamed IMU channels."""

    def __init__(self, history: int) -> None:
        if history <= 0:
            raise ValueError("history must be greater than 0")
        self._history = history
        self._count = 0
        self._write_index = 0
        self._timestamps = np.zeros(history, dtype=np.float64)
        self._values = np.zeros(
            (history, IMU_STREAMER_CHANNEL_COUNT),
            dtype=np.float64,
        )

    @property
    def count(self) -> int:
        return self._count

    @property
    def latest_timestamp(self) -> float | None:
        if self._count == 0:
            return None
        return float(self.valid_timestamps()[-1])

    def append_batch(self, timestamps: FloatArray, values: FloatArray) -> None:
        sample_count = int(timestamps.size)
        if sample_count == 0:
            return
        if values.shape != (sample_count, IMU_STREAMER_CHANNEL_COUNT):
            raise ValueError("timestamp and value counts must match")

        if sample_count >= self._history:
            self._timestamps[:] = timestamps[-self._history :]
            self._values[:, :] = values[-self._history :, :]
            self._count = self._history
            self._write_index = 0
            return

        first_count = min(sample_count, self._history - self._write_index)
        write_end = self._write_index + first_count
        self._timestamps[self._write_index : write_end] = timestamps[:first_count]
        self._values[self._write_index : write_end, :] = values[:first_count, :]

        remaining = sample_count - first_count
        if remaining > 0:
            self._timestamps[:remaining] = timestamps[first_count:]
            self._values[:remaining, :] = values[first_count:, :]

        self._write_index = (self._write_index + sample_count) % self._history
        self._count = min(self._history, self._count + sample_count)

    def valid_timestamps(self) -> FloatArray:
        if self._count == 0:
            return self._timestamps[:0]
        start = (self._write_index - self._count) % self._history
        if start + self._count <= self._history:
            return self._timestamps[start : start + self._count]
        return np.concatenate((self._timestamps[start:], self._timestamps[: self._write_index]))

    def valid_values(self) -> FloatArray:
        if self._count == 0:
            return self._values[:0, :]
        start = (self._write_index - self._count) % self._history
        if start + self._count <= self._history:
            return self._values[start : start + self._count, :]
        return np.concatenate(
            (self._values[start:, :], self._values[: self._write_index, :]),
            axis=0,
        )

    def channel(self, index: int) -> FloatArray:
        return self.valid_values()[:, index]

    def sample_hz(self) -> float | None:
        if self._count < 2:
            return None
        timestamps = self.valid_timestamps()
        elapsed = float(timestamps[-1] - timestamps[0])
        if elapsed <= 0.0:
            return None
        return float((self._count - 1) / elapsed)


def prefer_qt_xcb_platform() -> None:
    """Prefer Qt's XCB backend when Linux exposes a Wayland/X11 fallback chain."""
    if (
        sys.platform.startswith("linux")
        and "DISPLAY" in os.environ
        and os.environ.get("QT_QPA_PLATFORM") in (None, "", "wayland;xcb")
    ):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def import_pyqtgraph() -> tuple[Any, Any, Any]:
    """Import PyQtGraph lazily so parser tests do not need Qt."""
    prefer_qt_xcb_platform()

    try:
        import pyqtgraph as pg  # type: ignore[import-untyped]
        from pyqtgraph.Qt import QtCore  # type: ignore[import-untyped]
        from pyqtgraph.Qt import QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "PyQtGraph live plotting requires pyqtgraph and a Qt binding. "
            "Run `uv sync` inside `nix develop` to install project dependencies."
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


def _actual_refresh_hz(refresh_timestamp_hist: deque[float]) -> float | None:
    if len(refresh_timestamp_hist) < 2:
        return None
    elapsed = refresh_timestamp_hist[-1] - refresh_timestamp_hist[0]
    if elapsed <= 0.0:
        return None
    return (len(refresh_timestamp_hist) - 1) / elapsed


def run_live_plot(
    streamer: VescImuStreamer,
    *,
    history: int = DEFAULT_HISTORY,
    show_freq: bool = False,
    plot_refresh_hz: float = DEFAULT_PLOT_REFRESH_HZ,
    spectrum_refresh_hz: float = DEFAULT_SPECTRUM_REFRESH_HZ,
    antialias: bool = False,
    theme: str = DEFAULT_THEME,
) -> None:
    """Run a PyQtGraph live plot of package-streamed accel and gyro data."""
    selected_theme = PLOT_THEMES[theme]
    pg, QtCore, QtWidgets = import_pyqtgraph()
    pg.setConfigOptions(
        antialias=antialias,
        background=selected_theme.pg_background,
        foreground=selected_theme.pg_foreground,
    )

    imu_history = ImuStreamerHistory(history)
    refresh_timestamp_hist: deque[float] = deque(maxlen=120)
    spectrum_enabled = show_freq and spectrum_refresh_hz > 0.0
    spectrum_period = 1.0 / spectrum_refresh_hz if spectrum_enabled else math.inf
    next_spectrum_update = 0.0
    frequency_window_count = 0
    frequency_window = np.empty(0, dtype=np.float64)

    require_qt_platform_runtime()
    app = pg.mkQApp("VESC IMU Package Stream")
    window = QtWidgets.QWidget()
    window.setWindowTitle("VESC IMU Package Stream")
    window.resize(1400 if show_freq else 1000, 720)

    column_count = 2 if show_freq else 1
    qt_alignment = getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt)
    title = QtWidgets.QLabel("VESC IMU Package Stream")
    title.setAlignment(qt_alignment.AlignCenter)
    title.setStyleSheet(
        f"font-size: 14pt; font-weight: 700; color: {selected_theme.title_color};"
    )
    status = QtWidgets.QLabel("Waiting for package stream data...")
    status.setStyleSheet(f"color: {selected_theme.status_color};")

    layout = QtWidgets.QGridLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(title, 0, 0, 1, column_count)
    layout.addWidget(status, 3, 0, 1, column_count)
    for plot_row in (1, 2):
        layout.setRowStretch(plot_row, 1)
    for plot_col in range(column_count):
        layout.setColumnStretch(plot_col, 1)
    window.setStyleSheet(f"background-color: {selected_theme.window_background};")
    qt_size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)

    def _make_plot(
        row: int,
        col: int,
        title_text: str,
        y_label: str,
        *,
        x_label: str | None = None,
    ) -> Any:
        plot_widget = pg.PlotWidget(title=title_text)
        plot_widget.setMinimumSize(0, 0)
        plot_widget.setSizePolicy(qt_size_policy.Ignored, qt_size_policy.Ignored)
        layout.addWidget(plot_widget, row, col)
        plot = plot_widget.getPlotItem()
        plot.showGrid(x=True, y=True, alpha=selected_theme.grid_alpha)
        plot.addLegend(offset=(10, 10))
        plot.setLabel("left", y_label)
        plot.getAxis("left").setWidth(56)
        plot.getAxis("bottom").setHeight(36)
        if x_label is not None:
            plot.setLabel("bottom", x_label)
        return plot

    def _add_lines(
        plot: Any,
        names: tuple[str, str, str],
    ) -> tuple[Any, Any, Any]:
        lines = []
        for name, color in zip(names, selected_theme.line_colors):
            line = pg.PlotCurveItem(
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                pen=pg.mkPen(color, width=1.5),
                name=name,
                connect="all",
                skipFiniteCheck=True,
            )
            line.setSkipFiniteCheck(True)
            plot.addItem(line)
            lines.append(line)
        return tuple(lines)

    ax_acc = _make_plot(1, 0, "Raw Accel Data", "Accel", x_label="Samples")
    line_ax, line_ay, line_az = _add_lines(ax_acc, ("Acc X", "Acc Y", "Acc Z"))

    ax_gyro = _make_plot(2, 0, "Raw Gyro Data", "Gyro", x_label="Samples")
    line_gx, line_gy, line_gz = _add_lines(ax_gyro, ("Gyro X", "Gyro Y", "Gyro Z"))

    acc_freq_lines: tuple[Any, Any, Any] | None = None
    gyro_freq_lines: tuple[Any, Any, Any] | None = None
    acc_freq_range = FrequencyAxisRange()
    gyro_freq_range = FrequencyAxisRange()
    ax_acc_freq = ax_gyro_freq = None

    if show_freq:
        ax_acc_freq = _make_plot(
            1,
            1,
            "Raw Accel Frequency Analysis",
            "Magnitude",
            x_label="Frequency (Hz)",
        )
        acc_freq_lines = _add_lines(ax_acc_freq, ("Acc X", "Acc Y", "Acc Z"))

        ax_gyro_freq = _make_plot(
            2,
            1,
            "Raw Gyro Frequency Analysis",
            "Magnitude",
            x_label="Frequency (Hz)",
        )
        gyro_freq_lines = _add_lines(ax_gyro_freq, ("Gyro X", "Gyro Y", "Gyro Z"))

    def _set_curve_data(line: Any, x_values: FloatArray, y_values: FloatArray) -> None:
        line.setData(x=x_values, y=y_values, connect="all", skipFiniteCheck=True)

    def _frequency_bins() -> tuple[int, FloatArray, FloatArray] | None:
        nonlocal frequency_window, frequency_window_count

        sample_count = imu_history.count
        if sample_count < 2:
            return None

        timestamps = imu_history.valid_timestamps()
        sample_periods = np.diff(timestamps)
        sample_periods = sample_periods[sample_periods > 0.0]
        if sample_periods.size == 0:
            return None

        sample_period = float(np.median(sample_periods))
        if not np.isfinite(sample_period) or sample_period <= 0.0:
            return None

        if sample_count != frequency_window_count:
            if sample_count > 2:
                frequency_window = np.hanning(sample_count).astype(np.float64, copy=False)
            else:
                frequency_window = np.ones(sample_count, dtype=np.float64)
            frequency_window_count = sample_count

        frequencies = np.fft.rfftfreq(sample_count, d=sample_period).astype(
            np.float64,
            copy=False,
        )
        return sample_count, frequencies, frequency_window

    def _frequency_magnitudes(sample_count: int, window: FloatArray) -> FloatArray:
        samples = imu_history.valid_values()[-sample_count:, :].T
        centered = samples - np.mean(samples, axis=1, keepdims=True)
        centered = centered * window

        magnitudes = np.abs(np.fft.rfft(centered, axis=1)) / sample_count
        if magnitudes.shape[1] > 2:
            magnitudes[:, 1:-1] *= 2.0
        return magnitudes

    def _update_frequency_axis(
        axis: Any,
        lines: tuple[Any, Any, Any],
        magnitudes: FloatArray,
        channel_indexes: tuple[int, int, int],
        frequencies: FloatArray,
        axis_range: FrequencyAxisRange,
    ) -> None:
        max_magnitude = 0.0
        for line, channel_index in zip(lines, channel_indexes):
            channel_magnitudes = magnitudes[channel_index]
            if channel_magnitudes.size > 0:
                max_magnitude = max(max_magnitude, float(np.max(channel_magnitudes)))
            line.setData(
                x=frequencies,
                y=channel_magnitudes,
                connect="all",
                skipFiniteCheck=True,
            )

        next_x_max = max(float(frequencies[-1]), 1.0)
        if not math.isclose(next_x_max, axis_range.x_max, rel_tol=0.01, abs_tol=0.01):
            axis.setXRange(0.0, next_x_max, padding=0.0)
            axis_range.x_max = next_x_max

        next_y_max = max(max_magnitude * 1.1, 1e-6)
        if (
            axis_range.y_max == 0.0
            or next_y_max > axis_range.y_max
            or next_y_max < axis_range.y_max * 0.5
        ):
            axis.setYRange(0.0, next_y_max, padding=0.0)
            axis_range.y_max = next_y_max

    def update() -> None:
        nonlocal next_spectrum_update

        timestamps, values, _stats = streamer.drain()
        if timestamps.size == 0:
            return

        now = time.monotonic()
        refresh_timestamp_hist.append(now)
        imu_history.append_batch(timestamps, values)

        x_values = np.arange(-imu_history.count + 1, 1, dtype=np.float64)
        _set_curve_data(line_ax, x_values, imu_history.channel(ACC_X_INDEX))
        _set_curve_data(line_ay, x_values, imu_history.channel(ACC_Y_INDEX))
        _set_curve_data(line_az, x_values, imu_history.channel(ACC_Z_INDEX))
        _set_curve_data(line_gx, x_values, imu_history.channel(GYRO_X_INDEX))
        _set_curve_data(line_gy, x_values, imu_history.channel(GYRO_Y_INDEX))
        _set_curve_data(line_gz, x_values, imu_history.channel(GYRO_Z_INDEX))

        if spectrum_enabled and now >= next_spectrum_update:
            bins = _frequency_bins()
            if bins is not None:
                sample_count, frequencies, window = bins
                magnitudes = _frequency_magnitudes(sample_count, window)

                assert ax_acc_freq is not None
                assert ax_gyro_freq is not None
                assert acc_freq_lines is not None
                assert gyro_freq_lines is not None

                _update_frequency_axis(
                    ax_acc_freq,
                    acc_freq_lines,
                    magnitudes,
                    (ACC_X_INDEX, ACC_Y_INDEX, ACC_Z_INDEX),
                    frequencies,
                    acc_freq_range,
                )
                _update_frequency_axis(
                    ax_gyro_freq,
                    gyro_freq_lines,
                    magnitudes,
                    (GYRO_X_INDEX, GYRO_Y_INDEX, GYRO_Z_INDEX),
                    frequencies,
                    gyro_freq_range,
                )
                next_spectrum_update = now + spectrum_period

    def refresh_status() -> None:
        source_stats = streamer.source_stats()
        actual_refresh_hz = _actual_refresh_hz(refresh_timestamp_hist)
        plot_text = (
            "Plot: measuring"
            if actual_refresh_hz is None
            else f"Plot: {actual_refresh_hz:.1f} Hz actual"
        )

        latest_sample_timestamp = imu_history.latest_timestamp
        if source_stats.last_error is not None:
            sample_text = f"Package stream error: {source_stats.last_error}"
        elif latest_sample_timestamp is not None:
            actual_sample_hz = imu_history.sample_hz()
            if actual_sample_hz is None:
                sample_text = f"Sample: measuring | Stream: {latest_sample_timestamp:.3f}s"
            else:
                sample_text = (
                    f"Sample: {actual_sample_hz:.1f} Hz actual | "
                    f"Stream: {latest_sample_timestamp:.3f}s"
                )
        elif source_stats.last_print is not None:
            sample_text = f"Package: {source_stats.last_print}"
        else:
            sample_text = "Waiting for package stream data..."

        status.setText(
            f"{sample_text} | {plot_text} | "
            f"dropped={source_stats.dropped} seq_gap={source_stats.sequence_drops} "
            f"errors={source_stats.errors} prints={source_stats.lisp_prints}"
        )

    plot_timer = QtCore.QTimer()
    plot_timer.setInterval(round(1000.0 / plot_refresh_hz))
    plot_timer.timeout.connect(update)

    status_timer = QtCore.QTimer()
    status_timer.setInterval(round(1000.0 / DEFAULT_STATUS_REFRESH_HZ))
    status_timer.timeout.connect(refresh_status)

    def stop_updates(*_args: object) -> None:
        plot_timer.stop()
        status_timer.stop()

    window.destroyed.connect(stop_updates)
    update()
    refresh_status()
    window.show()
    plot_timer.start()
    status_timer.start()
    exec_app = getattr(app, "exec", None)
    if exec_app is None:
        exec_app = app.exec_
    try:
        exec_app()
    finally:
        plot_timer.stop()
        status_timer.stop()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot live accel and gyro data from the IMU Streamer package.",
    )
    add_vesc_connection_arguments(parser, include_can_id=False)
    parser.set_defaults(timeout=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--show-freq",
        action="store_true",
        help="Show frequency-analysis plots. Hidden by default for faster redraws.",
    )
    parser.add_argument(
        "--antialias",
        action="store_true",
        help="Render smoother lines at the cost of lower redraw performance.",
    )
    parser.add_argument(
        "--theme",
        choices=tuple(PLOT_THEMES),
        default=DEFAULT_THEME,
        help=f"Plot color theme (default: {DEFAULT_THEME}).",
    )
    parser.add_argument(
        "--plot-refresh-rate",
        type=float,
        default=DEFAULT_PLOT_REFRESH_HZ,
        metavar="HZ",
        help="Plot redraw rate in Hz (default: 30).",
    )
    parser.add_argument(
        "--spectrum-refresh-rate",
        type=float,
        default=DEFAULT_SPECTRUM_REFRESH_HZ,
        metavar="HZ",
        help="Frequency-analysis redraw rate in Hz when --show-freq is set (default: 2).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=DEFAULT_HISTORY,
        metavar="SAMPLES",
        help=f"Number of samples retained in the plot (default: {DEFAULT_HISTORY}).",
    )
    parser.add_argument(
        "--pending-samples",
        type=int,
        default=DEFAULT_PENDING_SAMPLES,
        metavar="SAMPLES",
        help=(
            "Receiver ring size for samples waiting on the UI thread "
            f"(default: {DEFAULT_PENDING_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--read-chunk-size",
        type=int,
        default=DEFAULT_READ_CHUNK_SIZE,
        metavar="BYTES",
        help=f"Serial read chunk size in bytes (default: {DEFAULT_READ_CHUNK_SIZE}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be greater than 0")
    if args.ble_scan_timeout <= 0.0:
        raise SystemExit("--ble-scan-timeout must be greater than 0")
    if args.ble_connect_timeout <= 0.0:
        raise SystemExit("--ble-connect-timeout must be greater than 0")
    if args.ble_chunk_size <= 0:
        raise SystemExit("--ble-chunk-size must be greater than 0")
    if args.history <= 0:
        raise SystemExit("--history must be greater than 0")
    if args.pending_samples <= 0:
        raise SystemExit("--pending-samples must be greater than 0")
    if args.read_chunk_size <= 0:
        raise SystemExit("--read-chunk-size must be greater than 0")
    if args.plot_refresh_rate <= 0.0:
        raise SystemExit("--plot-refresh-rate must be greater than 0")
    if args.spectrum_refresh_rate < 0.0:
        raise SystemExit("--spectrum-refresh-rate must be greater than or equal to 0")

    connection = resolve_vesc_connection_from_args(args)
    streamer = VescImuStreamer(
        connection=connection,
        timeout=args.timeout,
        pending_samples=args.pending_samples,
        read_chunk_size=args.read_chunk_size,
    )
    link_label = (
        f"BLE {connection.address}"
        if connection.kind is VescConnectionKind.BLE
        else f"{connection.address} at {connection.baudrate} baud"
    )

    print(
        f"Opening {link_label}; package=imu-streamer; channels=accel+gyro; "
        f"display={'freq' if args.show_freq else 'time'}",
        file=sys.stderr,
    )

    try:
        streamer.start()
        run_live_plot(
            streamer,
            history=args.history,
            show_freq=args.show_freq,
            plot_refresh_hz=args.plot_refresh_rate,
            spectrum_refresh_hz=args.spectrum_refresh_rate,
            antialias=args.antialias,
            theme=args.theme,
        )
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        streamer.stop(timeout=args.timeout + 0.2)


if __name__ == "__main__":
    main()
