#!/usr/bin/env python3
"""Live IMU plot through the VESC Tool TCP server or direct BLE.

Start the bridge first, for example:
    ./vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer 65102

Usage:
    uv run examples/imu_live_plot.py
    uv run examples/imu_live_plot.py --theme dark
    uv run examples/imu_live_plot.py --tcp 192.168.1.50:65102
    uv run examples/imu_live_plot.py --ble AA:BB:CC:DD:EE:FF
    uv run examples/imu_live_plot.py --ble AA:BB:CC:DD:EE:FF --can-id 1
    uv run examples/imu_live_plot.py --scan-udp
    uv run examples/imu_live_plot.py --scan-ble
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from vesc_py import ImuValues, VescClient, ble_scan, udp_scan

DEFAULT_TCP_ENDPOINT = ("127.0.0.1", 65102)
DEFAULT_MASK = 0x01FF  # roll/pitch/yaw + accelerometer + gyroscope.
DEFAULT_POLL_HZ = 50.0
DEFAULT_SPECTRUM_REFRESH_HZ = 2.0
DEFAULT_STATUS_REFRESH_HZ = 4.0
DEFAULT_THEME = "light"
QT_XCB_RUNTIME_LIBS = ("libxcb-cursor.so.0", "libxcb-icccm.so.4")
ROLL_INDEX = 0
PITCH_INDEX = 1
YAW_INDEX = 2
ACC_X_INDEX = 3
ACC_Y_INDEX = 4
ACC_Z_INDEX = 5
GYRO_X_INDEX = 6
GYRO_Y_INDEX = 7
GYRO_Z_INDEX = 8
IMU_PLOT_CHANNELS = 9


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


@dataclass(frozen=True)
class ImuSample:
    """One timestamped IMU response from the poller thread."""

    timestamp: float
    values: ImuValues


class ImuHistory:
    """Fixed-size numeric history for the live curves."""

    def __init__(self, history: int) -> None:
        self._history = history
        self._count = 0
        self._timestamps: npt.NDArray[np.float64] = np.zeros(history, dtype=float)
        self._values: npt.NDArray[np.float64] = np.zeros(
            (IMU_PLOT_CHANNELS, history),
            dtype=float,
        )

    @property
    def count(self) -> int:
        return self._count

    @property
    def latest_timestamp(self) -> float | None:
        if self._count == 0:
            return None
        return float(self._timestamps[-1])

    def channel(self, index: int) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self._values[index])

    def valid_channel(self, index: int) -> npt.NDArray[np.float64]:
        if self._count == 0:
            return self._values[index, :0]
        return self._values[index, -self._count:]

    def valid_timestamps(self) -> npt.NDArray[np.float64]:
        if self._count == 0:
            return self._timestamps[:0]
        return self._timestamps[-self._count:]

    def valid_values(self) -> npt.NDArray[np.float64]:
        if self._count == 0:
            return self._values[:, :0]
        return self._values[:, -self._count:]

    def sample_hz(self) -> float | None:
        if self._count < 2:
            return None

        timestamps = self.valid_timestamps()
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0.0:
            return None

        return float((self._count - 1) / elapsed)

    def append_samples(self, samples: Sequence[ImuSample], rad2deg: float) -> None:
        sample_count = len(samples)
        if sample_count == 0:
            return

        timestamps = np.empty(sample_count, dtype=float)
        values = np.empty((IMU_PLOT_CHANNELS, sample_count), dtype=float)
        for column, sample in enumerate(samples):
            imu = sample.values
            timestamps[column] = sample.timestamp
            values[:, column] = (
                imu.roll * rad2deg,
                imu.pitch * rad2deg,
                imu.yaw * rad2deg,
                imu.acc_x,
                imu.acc_y,
                imu.acc_z,
                imu.gyro_x,
                imu.gyro_y,
                imu.gyro_z,
            )

        if sample_count >= self._history:
            self._timestamps[:] = timestamps[-self._history:]
            self._values[:, :] = values[:, -self._history:]
            self._count = self._history
            return

        self._timestamps[:-sample_count] = self._timestamps[sample_count:]
        self._timestamps[-sample_count:] = timestamps
        self._values[:, :-sample_count] = self._values[:, sample_count:]
        self._values[:, -sample_count:] = values
        self._count = min(self._history, self._count + sample_count)


@dataclass
class FrequencyAxisRange:
    """Track frequency plot bounds so range changes are not forced every FFT."""

    x_max: float = 0.0
    y_max: float = 0.0


class ImuPoller:
    """Poll IMU data in the background so TCP timeouts do not block the UI."""

    def __init__(
        self,
        client: VescClient,
        *,
        mask: int,
        poll_hz: float,
        can_id: int | None = None,
        max_queue: int = 500,
    ) -> None:
        self._client = client
        self._mask = mask
        self._can_id = can_id
        self._period = 1.0 / poll_hz
        self._samples: queue.Queue[ImuSample] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vesc-imu-poller", daemon=True)
        self._callback_lock = threading.Lock()
        self._sample_callback: Callable[[], None] | None = None
        self._notification_pending = False
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def set_sample_callback(self, callback: Callable[[], None] | None) -> None:
        with self._callback_lock:
            self._sample_callback = callback

    def drain(self) -> list[ImuSample]:
        with self._callback_lock:
            self._notification_pending = False

        samples: list[ImuSample] = []
        while True:
            try:
                samples.append(self._samples.get_nowait())
            except queue.Empty:
                return samples

    def _run(self) -> None:
        next_poll = time.monotonic()
        while not self._stop.is_set():
            try:
                imu = self._client.get_imu_data(self._mask, can_id=self._can_id)
            except Exception as exc:  # Keep polling; transient TCP stalls are common enough.
                self._last_error = str(exc)
            else:
                self._last_error = None
                self._put_latest(ImuSample(timestamp=time.monotonic(), values=imu))
                if self._mark_notification_pending():
                    self._notify_sample()

            next_poll += self._period
            sleep_s = max(0.0, next_poll - time.monotonic())
            if self._stop.wait(sleep_s):
                return
            if sleep_s == 0.0:
                next_poll = time.monotonic()

    def _put_latest(self, sample: ImuSample) -> None:
        try:
            self._samples.put_nowait(sample)
        except queue.Full:
            try:
                self._samples.get_nowait()
            except queue.Empty:
                pass
            self._samples.put_nowait(sample)

    def _mark_notification_pending(self) -> bool:
        with self._callback_lock:
            if self._sample_callback is None or self._notification_pending:
                return False
            self._notification_pending = True
            return True

    def _notify_sample(self) -> None:
        with self._callback_lock:
            callback = self._sample_callback
        if callback is not None:
            try:
                callback()
            except RuntimeError:
                # The Qt receiver may already be gone during application shutdown.
                pass


def parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse HOST:PORT for argparse."""
    host, sep, port_str = endpoint.rpartition(":")
    if not sep or not host or not port_str:
        raise argparse.ArgumentTypeError("TCP endpoint must be HOST:PORT")

    try:
        port = int(port_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("TCP port must be an integer") from exc

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("TCP port must be in range 1..65535")

    return host, port


def scan_and_print_udp(timeout: float = 3.0) -> None:
    """Listen for VESC Tool TCP server broadcast announcements and print them."""
    print(f"Listening for VESC Tool TCP server broadcasts on UDP port 65109 for {timeout}s ...")
    devices = udp_scan(timeout=timeout)
    if not devices:
        print("No VESC Tool TCP servers found.")
        return

    for device in devices:
        print(f"  {device.hw_name}  {device.ip}:{device.port}")


def scan_and_print_ble(timeout: float = 5.0) -> None:
    """Scan for BLE devices advertising the Nordic UART service and print them."""
    print(f"Scanning for VESC BLE devices for {timeout}s ...")
    devices = ble_scan(timeout=timeout)
    if not devices:
        print("No VESC BLE devices found.")
        return

    for device in devices:
        name = device.name or "(unnamed)"
        rssi = "" if device.rssi is None else f"  RSSI {device.rssi} dBm"
        print(f"  {name}  {device.address}{rssi}")


def tcp_server_help(endpoint: str, port: int) -> str:
    """Return a concise hint for starting the VESC Tool TCP bridge."""
    return "\n".join(
        [
            f"Could not connect to VESC Tool TCP server at {endpoint}.",
            "",
            "Start VESC Tool with tcpServer enabled, then retry. For example:",
            f"  vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer {port}",
            "",
            "If VESC Tool is already running, check the host and port passed to --tcp.",
        ]
    )


def ble_connection_help(address: str, message: str) -> str:
    """Return a concise hint for direct BLE connection failures."""
    return "\n".join(
        [
            f"Could not connect to VESC BLE device at {address}.",
            "",
            message,
            "",
            "Use --scan-ble to list VESC BLE devices advertising Nordic UART, "
            "then pass the printed address or OS-specific UUID to --ble.",
        ]
    )


def prefer_qt_xcb_platform() -> None:
    """Prefer Qt's XCB backend when Linux exposes a Wayland/X11 fallback chain."""
    if (
        sys.platform.startswith("linux")
        and "DISPLAY" in os.environ
        and os.environ.get("QT_QPA_PLATFORM") in (None, "", "wayland;xcb")
    ):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def import_pyqtgraph() -> tuple[Any, Any, Any]:
    """Import PyQtGraph lazily so non-plot commands do not require Qt."""
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


def run_live_plot(
    poller: ImuPoller,
    *,
    history: int = 300,
    show_freq: bool = False,
    spectrum_refresh_hz: float = DEFAULT_SPECTRUM_REFRESH_HZ,
    antialias: bool = False,
    theme: str = DEFAULT_THEME,
) -> None:
    """Run a PyQtGraph live plot of IMU data."""
    selected_theme = PLOT_THEMES[theme]
    pg, QtCore, QtWidgets = import_pyqtgraph()
    pg.setConfigOptions(
        antialias=antialias,
        background=selected_theme.pg_background,
        foreground=selected_theme.pg_foreground,
    )

    rad2deg = 180.0 / math.pi
    imu_history = ImuHistory(history)
    refresh_timestamp_hist: deque[float] = deque(maxlen=120)
    spectrum_enabled = show_freq and spectrum_refresh_hz > 0.0
    spectrum_period = 1.0 / spectrum_refresh_hz if spectrum_enabled else math.inf
    next_spectrum_update = 0.0
    frequency_window_count = 0
    frequency_window = np.array([])

    x = np.arange(-history + 1, 1)
    zeros = np.zeros(history)
    require_qt_platform_runtime()
    app = pg.mkQApp("VESC IMU Live Data")
    window = QtWidgets.QWidget()
    window.setWindowTitle("VESC IMU Live Data")
    window.resize(1400 if show_freq else 1000, 900)

    column_count = 2 if show_freq else 1
    qt_alignment = getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt)
    title = QtWidgets.QLabel("VESC IMU Live Data")
    title.setAlignment(qt_alignment.AlignCenter)
    title.setStyleSheet(
        f"font-size: 14pt; font-weight: 700; color: {selected_theme.title_color};"
    )
    status = QtWidgets.QLabel("Waiting for IMU data... | Sample: measuring | Plot: measuring")
    status.setStyleSheet(f"color: {selected_theme.status_color};")

    layout = QtWidgets.QGridLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(title, 0, 0, 1, column_count)
    layout.addWidget(status, 4, 0, 1, column_count)
    for plot_row in (1, 2, 3):
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
        y_range: tuple[float, float] | None = None,
        x_range: tuple[float, float] | None = None,
    ) -> Any:
        plot_widget = pg.PlotWidget(title=title_text)
        plot_widget.setMinimumSize(0, 0)
        plot_widget.setSizePolicy(
            qt_size_policy.Ignored,
            qt_size_policy.Ignored,
        )
        layout.addWidget(plot_widget, row, col)
        plot = plot_widget.getPlotItem()
        plot.showGrid(x=True, y=True, alpha=selected_theme.grid_alpha)
        plot.addLegend(offset=(10, 10))
        plot.setLabel("left", y_label)
        plot.getAxis("left").setWidth(56)
        plot.getAxis("bottom").setHeight(36)
        if x_label is not None:
            plot.setLabel("bottom", x_label)
        if y_range is not None:
            plot.setYRange(*y_range, padding=0.0)
        if x_range is not None:
            plot.setXRange(*x_range, padding=0.0)
        return plot

    def _add_lines(
        plot: Any,
        names: tuple[str, str, str],
        *,
        initial_x: np.ndarray = x,
        initial_y: np.ndarray = zeros,
    ) -> tuple[Any, Any, Any]:
        lines = []
        for name, color in zip(names, selected_theme.line_colors):
            line = pg.PlotCurveItem(
                initial_x,
                initial_y,
                pen=pg.mkPen(color, width=1.5),
                name=name,
                connect="all",
                skipFiniteCheck=True,
            )
            line.setSkipFiniteCheck(True)
            plot.addItem(line)
            lines.append(line)
        return tuple(lines)

    ax_acc = _make_plot(
        1,
        0,
        "Accel Data",
        "g",
        y_range=(-8, 8),
        x_range=(-history + 1, 0),
    )
    line_ax, line_ay, line_az = _add_lines(ax_acc, ("Acc X", "Acc Y", "Acc Z"))

    ax_gyro = _make_plot(
        2,
        0,
        "Gyro Data",
        "Gyro",
        y_range=(-2000, 2000),
        x_range=(-history + 1, 0),
    )
    line_gx, line_gy, line_gz = _add_lines(ax_gyro, ("Gyro X", "Gyro Y", "Gyro Z"))

    ax_rpy = _make_plot(
        3,
        0,
        "RPY Data",
        "Degrees",
        x_label="Samples",
        y_range=(-200, 200),
        x_range=(-history + 1, 0),
    )
    line_r, line_p, line_y = _add_lines(ax_rpy, ("Roll", "Pitch", "Yaw"))

    acc_freq_lines: tuple[Any, Any, Any] | None = None
    gyro_freq_lines: tuple[Any, Any, Any] | None = None
    rpy_freq_lines: tuple[Any, Any, Any] | None = None
    acc_freq_range = FrequencyAxisRange()
    gyro_freq_range = FrequencyAxisRange()
    rpy_freq_range = FrequencyAxisRange()
    ax_acc_freq = ax_gyro_freq = ax_rpy_freq = None

    if show_freq:
        empty = np.array([])
        ax_acc_freq = _make_plot(1, 1, "Accel Data Frequency Analysis", "Magnitude")
        acc_freq_lines = _add_lines(
            ax_acc_freq,
            ("Acc X", "Acc Y", "Acc Z"),
            initial_x=empty,
            initial_y=empty,
        )

        ax_gyro_freq = _make_plot(2, 1, "Gyro Data Frequency Analysis", "Magnitude")
        gyro_freq_lines = _add_lines(
            ax_gyro_freq,
            ("Gyro X", "Gyro Y", "Gyro Z"),
            initial_x=empty,
            initial_y=empty,
        )

        ax_rpy_freq = _make_plot(
            3,
            1,
            "RPY Data Frequency Analysis",
            "Magnitude",
            x_label="Frequency (Hz)",
        )
        rpy_freq_lines = _add_lines(
            ax_rpy_freq,
            ("Roll", "Pitch", "Yaw"),
            initial_x=empty,
            initial_y=empty,
        )

    def _set_curve_data(line: Any, y_values: np.ndarray) -> None:
        line.setData(x=x, y=y_values, connect="all", skipFiniteCheck=True)

    def _frequency_bins() -> tuple[int, np.ndarray, np.ndarray] | None:
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
                frequency_window = np.hanning(sample_count)
            else:
                frequency_window = np.ones(sample_count)
            frequency_window_count = sample_count

        frequencies = np.fft.rfftfreq(sample_count, d=sample_period)
        return sample_count, frequencies, frequency_window

    def _frequency_magnitudes(
        sample_count: int,
        window: np.ndarray,
    ) -> np.ndarray:
        samples = imu_history.valid_values()[:, -sample_count:]
        centered = samples - np.mean(samples, axis=1, keepdims=True)
        centered = centered * window

        magnitudes = np.abs(np.fft.rfft(centered, axis=1)) / sample_count
        if magnitudes.shape[1] > 2:
            magnitudes[:, 1:-1] *= 2.0
        return magnitudes

    def _update_frequency_axis(
        axis: Any,
        lines: tuple[Any, Any, Any],
        magnitudes: np.ndarray,
        channel_indexes: tuple[int, int, int],
        frequencies: np.ndarray,
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

    def _actual_refresh_hz() -> float | None:
        if len(refresh_timestamp_hist) < 2:
            return None

        elapsed = refresh_timestamp_hist[-1] - refresh_timestamp_hist[0]
        if elapsed <= 0.0:
            return None

        return (len(refresh_timestamp_hist) - 1) / elapsed

    def _actual_sample_hz() -> float | None:
        return imu_history.sample_hz()

    def update() -> None:
        nonlocal next_spectrum_update

        now = time.monotonic()
        samples = poller.drain()
        if not samples:
            return

        refresh_timestamp_hist.append(now)
        imu_history.append_samples(samples, rad2deg)

        _set_curve_data(line_ax, imu_history.channel(ACC_X_INDEX))
        _set_curve_data(line_ay, imu_history.channel(ACC_Y_INDEX))
        _set_curve_data(line_az, imu_history.channel(ACC_Z_INDEX))
        _set_curve_data(line_gx, imu_history.channel(GYRO_X_INDEX))
        _set_curve_data(line_gy, imu_history.channel(GYRO_Y_INDEX))
        _set_curve_data(line_gz, imu_history.channel(GYRO_Z_INDEX))
        _set_curve_data(line_r, imu_history.channel(ROLL_INDEX))
        _set_curve_data(line_p, imu_history.channel(PITCH_INDEX))
        _set_curve_data(line_y, imu_history.channel(YAW_INDEX))

        if spectrum_enabled and now >= next_spectrum_update:
            bins = _frequency_bins()
            if bins is not None:
                sample_count, frequencies, window = bins
                magnitudes = _frequency_magnitudes(sample_count, window)

                assert ax_acc_freq is not None
                assert ax_gyro_freq is not None
                assert ax_rpy_freq is not None
                assert acc_freq_lines is not None
                assert gyro_freq_lines is not None
                assert rpy_freq_lines is not None

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
                _update_frequency_axis(
                    ax_rpy_freq,
                    rpy_freq_lines,
                    magnitudes,
                    (ROLL_INDEX, PITCH_INDEX, YAW_INDEX),
                    frequencies,
                    rpy_freq_range,
                )
                next_spectrum_update = now + spectrum_period

    def refresh_status() -> None:
        now = time.monotonic()
        actual_refresh_hz = _actual_refresh_hz()
        if actual_refresh_hz is None:
            plot_text = "Plot: measuring"
        else:
            plot_text = f"Plot: {actual_refresh_hz:.1f} Hz actual"

        latest_sample_timestamp = imu_history.latest_timestamp
        if poller.last_error is not None:
            sample_text = f"IMU read error: {poller.last_error}"
        elif latest_sample_timestamp is not None:
            actual_sample_hz = _actual_sample_hz()
            if actual_sample_hz is None:
                sample_text = f"Sample: measuring | Age: {now - latest_sample_timestamp:.2f}s"
            else:
                sample_text = (
                    f"Sample: {actual_sample_hz:.1f} Hz actual | "
                    f"Age: {now - latest_sample_timestamp:.2f}s"
                )
        else:
            sample_text = "Waiting for IMU data..."
        status.setText(f"{sample_text} | {plot_text}")

    signal_type: Any = getattr(QtCore, "Signal", None)
    if signal_type is None:
        signal_type = QtCore.pyqtSignal

    class SampleNotifier(QtCore.QObject):  # type: ignore[name-defined, misc]
        sample_ready = signal_type()

    sample_notifier = SampleNotifier()
    sample_notifier.sample_ready.connect(update)

    status_timer = QtCore.QTimer()
    status_timer.setInterval(round(1000.0 / DEFAULT_STATUS_REFRESH_HZ))
    status_timer.timeout.connect(refresh_status)

    def stop_updates(*_args: object) -> None:
        status_timer.stop()
        poller.set_sample_callback(None)

    window.destroyed.connect(stop_updates)
    poller.set_sample_callback(sample_notifier.sample_ready.emit)
    update()
    refresh_status()
    window.show()
    status_timer.start()
    exec_app = getattr(app, "exec", None)
    if exec_app is None:
        exec_app = app.exec_
    try:
        exec_app()
    finally:
        poller.set_sample_callback(None)
        status_timer.stop()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot live VESC IMU data through TCP or direct BLE.",
        epilog=(
            "TCP mode uses a VESC Tool bridge, for example: "
            "vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer 65102. "
            "BLE mode connects directly to a VESC BLE module advertising Nordic UART."
        ),
    )
    parser.add_argument(
        "--tcp",
        type=parse_tcp_endpoint,
        default=DEFAULT_TCP_ENDPOINT,
        metavar="HOST:PORT",
        help=(
            "VESC Tool TCP server endpoint used when --ble is not set "
            "(default: 127.0.0.1:65102)."
        ),
    )
    parser.add_argument(
        "--ble",
        metavar="ADDRESS",
        help=(
            "Connect directly to a VESC BLE module by address or OS-specific UUID "
            "instead of using --tcp."
        ),
    )
    parser.add_argument(
        "--can-id",
        type=int,
        metavar="ID",
        help=(
            "Forward IMU requests over CAN to the target VESC ID. "
            "Use this when connected to a VESC Express gateway."
        ),
    )
    parser.add_argument(
        "--scan-udp",
        action="store_true",
        help="Scan for VESC Tool TCP server UDP broadcasts and exit.",
    )
    parser.add_argument(
        "--scan-ble",
        action="store_true",
        help="Scan for VESC BLE devices advertising Nordic UART and exit.",
    )
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=3.0,
        metavar="SEC",
        help="UDP scan duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--ble-scan-timeout",
        type=float,
        default=5.0,
        metavar="SEC",
        help="BLE scan duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--ble-connect-timeout",
        type=float,
        default=10.0,
        metavar="SEC",
        help="BLE connection timeout in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--ble-chunk-size",
        type=int,
        default=20,
        metavar="BYTES",
        help="Maximum BLE Nordic UART write chunk size in bytes (default: 20).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.5,
        metavar="SEC",
        help="VESC response timeout in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_POLL_HZ,
        metavar="HZ",
        help="IMU poll rate in Hz (default: 50).",
    )
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
        "--spectrum-refresh-rate",
        type=float,
        default=DEFAULT_SPECTRUM_REFRESH_HZ,
        metavar="HZ",
        help="Frequency-analysis redraw rate in Hz when --show-freq is set (default: 2).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=300,
        metavar="SAMPLES",
        help="Number of samples visible in the plot (default: 300).",
    )
    parser.add_argument(
        "--mask",
        type=lambda value: int(value, 0),
        default=DEFAULT_MASK,
        metavar="MASK",
        help="IMU field bitmask (default: 0x01ff = RPY + accelerometer + gyroscope).",
    )
    return parser


def connect_vesc(args: argparse.Namespace) -> VescClient:
    """Open the transport selected by parsed CLI arguments."""
    if args.ble is not None:
        address = str(args.ble)
        print(f"Connecting to VESC BLE device at {address} ...")
        try:
            return VescClient.connect_ble(
                address,
                timeout=args.timeout,
                connect_timeout=args.ble_connect_timeout,
                chunk_size=args.ble_chunk_size,
            )
        except TimeoutError:
            raise SystemExit(
                ble_connection_help(
                    address,
                    "Timed out while connecting to the BLE peripheral.",
                )
            ) from None
        except ConnectionError as exc:
            raise SystemExit(
                ble_connection_help(
                    address,
                    (
                        f"{exc}\n\n"
                        "The BLE link opened, but no VESC firmware response was received. "
                        "Make sure the BLE module is connected to a controller."
                    ),
                )
            ) from None

    host, port = args.tcp
    endpoint = f"{host}:{port}"
    print(f"Connecting to VESC Tool TCP server at {endpoint} ...")
    try:
        return VescClient.connect_tcp(host, port, timeout=args.timeout)
    except ConnectionRefusedError:
        raise SystemExit(tcp_server_help(endpoint, port)) from None
    except ConnectionError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "The TCP socket opened, but no VESC firmware response was received. "
            "Make sure VESC Tool is connected to a controller before starting the plot."
        ) from None


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scan_udp and args.scan_ble:
        parser.error("use only one of --scan-udp or --scan-ble")
    if args.scan_timeout <= 0.0:
        raise SystemExit("--scan-timeout must be greater than 0")
    if args.ble_scan_timeout <= 0.0:
        raise SystemExit("--ble-scan-timeout must be greater than 0")

    if args.scan_udp:
        scan_and_print_udp(args.scan_timeout)
        return
    if args.scan_ble:
        scan_and_print_ble(args.ble_scan_timeout)
        return

    if args.rate <= 0.0:
        raise SystemExit("--rate must be greater than 0")
    if args.spectrum_refresh_rate < 0.0:
        raise SystemExit("--spectrum-refresh-rate must be greater than or equal to 0")
    if args.history <= 0:
        raise SystemExit("--history must be greater than 0")
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be greater than 0")
    if args.ble_connect_timeout <= 0.0:
        raise SystemExit("--ble-connect-timeout must be greater than 0")
    if args.ble_chunk_size <= 0:
        raise SystemExit("--ble-chunk-size must be greater than 0")
    if args.can_id is not None and not 0 <= args.can_id <= 253:
        raise SystemExit("--can-id must be in range [0, 253]")

    client = connect_vesc(args)
    poller = ImuPoller(client, mask=args.mask, poll_hz=args.rate, can_id=args.can_id)

    try:
        gateway_fw = client.fw_version
        if args.can_id is not None and gateway_fw is not None:
            print(f"Gateway firmware: {gateway_fw.major}.{gateway_fw.minor:02d}  HW: {gateway_fw.hw}")
        if args.can_id is not None:
            target_fw = client.get_fw_version(can_id=args.can_id, timeout=args.timeout)
            print(
                "Target firmware: "
                f"{target_fw.major}.{target_fw.minor:02d}  HW: {target_fw.hw}  CAN ID: {args.can_id}"
            )
        elif gateway_fw is not None:
            print(f"Firmware: {gateway_fw.major}.{gateway_fw.minor:02d}  HW: {gateway_fw.hw}")
        poller.start()
        run_live_plot(
            poller,
            history=args.history,
            show_freq=args.show_freq,
            spectrum_refresh_hz=args.spectrum_refresh_rate,
            antialias=args.antialias,
            theme=args.theme,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        poller.stop(timeout=args.timeout + 0.1)
        client.close()


if __name__ == "__main__":
    main()
