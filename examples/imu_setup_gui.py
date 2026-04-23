#!/usr/bin/env python3
"""Graphical IMU setup wizard through the VESC Tool TCP server.

Start the bridge first, for example:
    ./vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer 65102

Usage:
    uv run examples/imu_setup_gui.py
    uv run examples/imu_setup_gui.py --tcp 192.168.1.50:65102
    uv run examples/imu_setup_gui.py --scan-udp
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, SupportsInt, cast

from vesc_py import VescClient, udp_scan
from vesc_py.imu_setup import (
    IMU_SETUP_MASK,
    FilteredImuState,
    ImuBasicProfile,
    OrientationRestore,
    RollingMeanImuState,
    RollingMeanValue,
    YawOffsetEstimator,
    apply_basic_profile,
    apply_pitch_offset,
    apply_roll_offset,
    apply_yaw_offset,
    axis_current,
    axis_max,
    config_float,
    config_int,
    copy_filtered_imu_state,
    imu_type_name,
    mean_label,
    prepare_orientation_calibration,
    restore_orientation_config,
    save_accel_offset,
    save_gyro_offsets,
    save_orientation_restore,
    set_config_int,
    validate_required_fields,
)

DEFAULT_TCP_ENDPOINT = ("127.0.0.1", 65102)
DEFAULT_POLL_HZ = 50.0
DEFAULT_SEARCH_SECONDS = 2.0
DEFAULT_MEAN_SECONDS = 3.0
DEFAULT_STATUS_REFRESH_HZ = 20.0
DEG_TO_RAD = 3.141592653589793 / 180.0
RAD_TO_DEG = 180.0 / 3.141592653589793
QT_XCB_RUNTIME_LIBS = ("libxcb-cursor.so.0", "libxcb-icccm.so.4")
ProfileName = Literal["default", "logs", "balance-unicycle", "balance-skateboard"]


class WizardPage(IntEnum):
    """Stacked widget page indexes."""

    MENU = 0
    CONFIGURATOR = 1
    GYRO = 2
    ACCEL_X = 3
    ACCEL_Y = 4
    ACCEL_Z = 5
    ORIENTATION_ROLL = 6
    ORIENTATION_PITCH = 7
    ORIENTATION_YAW = 8


@dataclass(frozen=True)
class ImuSnapshot:
    """Thread-safe snapshot of live IMU setup values."""

    timestamp: float | None
    raw: FilteredImuState
    mean: FilteredImuState
    yaw_offset: float
    yaw_mean_offset: float | None
    last_error: str | None


class LockedVescClient:
    """Serialize all commands against one VESC TCP connection."""

    def __init__(self, client: VescClient) -> None:
        self._client = client
        self._lock = threading.RLock()

    @property
    def fw_label(self) -> str:
        """Return a compact firmware label for status text."""

        fw = self._client.fw_version
        if fw is None:
            return "Firmware: unknown"
        return f"Firmware: {fw.major}.{fw.minor:02d}  HW: {fw.hw}"

    def get_imu_data(self, mask: int) -> Any:
        """Read IMU data while holding the command lock."""

        with self._lock:
            return self._client.get_imu_data(mask)

    def get_appconf(self) -> dict[str, object]:
        """Read APPCONF while holding the command lock."""

        with self._lock:
            return self._client.get_appconf()

    def set_appconf(
        self,
        config: dict[str, object],
        *,
        store: bool,
        wait_ack: bool,
    ) -> None:
        """Write APPCONF while holding the command lock."""

        with self._lock:
            self._client.set_appconf(config, store=store, wait_ack=wait_ack)

    def close(self) -> None:
        """Close the underlying connection."""

        with self._lock:
            self._client.close()


class ImuSetupPoller:
    """Poll and filter IMU data in the background."""

    def __init__(
        self,
        client: LockedVescClient,
        *,
        poll_hz: float,
        mean_seconds: float,
    ) -> None:
        self._client = client
        self._period = 1.0 / poll_hz
        self._mean_seconds = mean_seconds
        self._lock = threading.Lock()
        self._state = FilteredImuState()
        self._mean_window = RollingMeanImuState(mean_seconds) if mean_seconds > 0.0 else None
        self._mean_state = FilteredImuState()
        self._yaw_estimator = YawOffsetEstimator()
        self._yaw_mean = RollingMeanValue(mean_seconds) if mean_seconds > 0.0 else None
        self._yaw_mean_offset: float | None = None
        self._last_timestamp: float | None = None
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vesc-imu-setup-poller", daemon=True)

    def start(self) -> None:
        """Start polling."""

        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Stop polling and wait briefly for the thread."""

        self._stop.set()
        self._thread.join(timeout=timeout)

    def snapshot(self) -> ImuSnapshot:
        """Return the latest filtered values."""

        with self._lock:
            return ImuSnapshot(
                timestamp=self._last_timestamp,
                raw=copy_filtered_imu_state(self._state),
                mean=copy_filtered_imu_state(self._mean_state),
                yaw_offset=self._yaw_estimator.yaw_offset,
                yaw_mean_offset=self._yaw_mean_offset,
                last_error=self._last_error,
            )

    def reset_sample_window(self) -> None:
        """Restart rolling averages without clearing the filter history."""

        with self._lock:
            self._mean_window = (
                RollingMeanImuState(self._mean_seconds) if self._mean_seconds > 0.0 else None
            )
            self._mean_state = copy_filtered_imu_state(self._state)

    def reset_accel_max(self) -> None:
        """Clear peak accelerometer values and rolling mean peak state."""

        with self._lock:
            self._state.max_acc_x = -10.0
            self._state.max_acc_y = -10.0
            self._state.max_acc_z = -10.0
            self._mean_window = (
                RollingMeanImuState(self._mean_seconds) if self._mean_seconds > 0.0 else None
            )
            self._mean_state = copy_filtered_imu_state(self._state)

    def reset_yaw(self) -> None:
        """Restart yaw-offset estimation."""

        with self._lock:
            self._yaw_estimator = YawOffsetEstimator()
            self._yaw_mean = RollingMeanValue(self._mean_seconds) if self._mean_seconds > 0.0 else None
            self._yaw_mean_offset = None

    def _run(self) -> None:
        next_poll = time.monotonic()
        while not self._stop.is_set():
            try:
                values = self._client.get_imu_data(IMU_SETUP_MASK)
            except Exception as exc:  # Keep the UI open through transient TCP stalls.
                with self._lock:
                    self._last_error = str(exc)
            else:
                now = time.monotonic()
                with self._lock:
                    self._last_error = None
                    self._state.update(values)
                    self._yaw_estimator.update(values)
                    if self._mean_window is not None:
                        self._mean_state = self._mean_window.update(now, self._state)
                    else:
                        self._mean_state = copy_filtered_imu_state(self._state)
                    if self._yaw_mean is not None:
                        self._yaw_mean_offset = self._yaw_mean.update(
                            now,
                            self._yaw_estimator.yaw_offset,
                        )
                    else:
                        self._yaw_mean_offset = None
                    self._last_timestamp = now

            next_poll += self._period
            sleep_s = max(0.0, next_poll - time.monotonic())
            if self._stop.wait(sleep_s):
                return
            if sleep_s == 0.0:
                next_poll = time.monotonic()


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


def prefer_qt_xcb_platform() -> None:
    """Prefer Qt's XCB backend when Linux exposes a Wayland/X11 fallback chain."""
    if (
        sys.platform.startswith("linux")
        and "DISPLAY" in os.environ
        and os.environ.get("QT_QPA_PLATFORM") in (None, "", "wayland;xcb")
    ):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def import_pyside6() -> tuple[Any, Any]:
    """Import PySide6 lazily so --scan-udp does not require Qt."""
    prefer_qt_xcb_platform()

    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "The IMU setup GUI requires PySide6. Install the project dependencies, "
            "or run `uv sync` inside `nix develop`."
        ) from exc

    return QtCore, QtWidgets


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


def format_signed(value: float, digits: int = 3) -> str:
    """Format a signed float with fixed precision."""

    return f"{value:+.{digits}f}"


def mode_name(mode: int) -> str:
    """Return the AHRS mode name used by the original QML wizard."""

    names = {
        0: "Madgwick",
        1: "Mahony",
        2: "Madgwick Fusion",
        3: "Unknown",
    }
    return names.get(mode, f"Unknown ({mode})")


class ImuSetupController:
    """Stateful controller for the PySide IMU setup wizard."""

    def __init__(
        self,
        qt_core: Any,
        qt_widgets: Any,
        client: LockedVescClient,
        poller: ImuSetupPoller,
        config: dict[str, object],
        *,
        wait_ack: bool,
        mean_seconds: float,
        search_seconds: float,
    ) -> None:
        self._qt_core = qt_core
        self._qt_widgets = qt_widgets
        self._client = client
        self._poller = poller
        self._config = config
        self._wait_ack = wait_ack
        self._mean_seconds = mean_seconds
        self._search_seconds = search_seconds
        self._configurator_restore: dict[str, object] | None = None
        self._orientation_restore: OrientationRestore | None = None
        self._search_previous_app: int | None = None
        self._search_candidate = 0
        self._search_active = False
        self._profile_choice: ProfileName = "default"

        self.window = qt_widgets.QWidget()
        self.window.setWindowTitle("VESC IMU Setup")
        self.window.resize(760, 620)
        self._pages = qt_widgets.QStackedWidget()
        self._status_label = qt_widgets.QLabel()
        self._prev_button = qt_widgets.QPushButton("Close")
        self._save_button = qt_widgets.QPushButton("Save")
        self._save_button.setVisible(False)
        self._search_button: Any | None = None
        self._profile_radios: dict[ProfileName, Any] = {}
        self._config_readout = qt_widgets.QLabel()
        self._gyro_readout = qt_widgets.QLabel()
        self._accel_readouts: dict[str, Any] = {}
        self._orientation_readouts: dict[str, Any] = {}

        self._build_ui()
        self._refresh_timer = qt_core.QTimer()
        self._refresh_timer.setInterval(round(1000.0 / DEFAULT_STATUS_REFRESH_HZ))
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start()
        self.window.destroyed.connect(self.stop)
        self.refresh()

    def stop(self, *_args: object) -> None:
        """Stop UI timers."""

        self._search_active = False
        self._refresh_timer.stop()

    def _build_ui(self) -> None:
        layout = self._qt_widgets.QVBoxLayout(self.window)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = self._qt_widgets.QLabel("IMU Wizard")
        title.setAlignment(self._qt_core.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        self._status_label.setTextInteractionFlags(
            self._qt_core.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._pages.addWidget(self._build_menu_page())
        self._pages.addWidget(self._build_configurator_page())
        self._pages.addWidget(self._build_gyro_page())
        self._pages.addWidget(self._build_accel_page("x"))
        self._pages.addWidget(self._build_accel_page("y"))
        self._pages.addWidget(self._build_accel_page("z"))
        self._pages.addWidget(self._build_orientation_page("roll"))
        self._pages.addWidget(self._build_orientation_page("pitch"))
        self._pages.addWidget(self._build_orientation_page("yaw"))
        layout.addWidget(self._pages, 1)

        footer = self._qt_widgets.QHBoxLayout()
        footer.addWidget(self._prev_button)
        footer.addWidget(self._save_button)
        self._prev_button.clicked.connect(self._cancel_or_close)
        self._save_button.clicked.connect(self._save_current_page)
        layout.addLayout(footer)

        self._pages.currentChanged.connect(self._update_footer)
        self._update_footer()

    def _build_menu_page(self) -> Any:
        page = self._make_page()
        layout = page.layout()
        assert layout is not None
        layout.addWidget(
            self._paragraph(
                "Welcome to the IMU Wizard. This tool is split into tasks that simplify "
                "IMU configuration and calibration. They can be run sequentially or as one-offs."
            )
        )
        for label, callback in (
            ("IMU Configurator", self._open_configurator),
            ("Gyroscope Calibration", self._open_gyro),
            ("Accelerometer Calibration", self._open_accel),
            ("Orientation Calibration", self._open_orientation),
        ):
            button = self._qt_widgets.QPushButton(label)
            button.clicked.connect(callback)
            layout.addWidget(button)
        layout.addStretch(1)
        return page

    def _build_configurator_page(self) -> Any:
        page = self._make_page()
        layout = page.layout()
        assert layout is not None
        layout.addWidget(self._heading("IMU Detector"))
        detector_text = self._paragraph(
            "Search tests each supported IMU type with temporary settings. Type 'Off' means no "
            "working IMU was detected."
        )
        layout.addWidget(detector_text)
        self._search_button = self._qt_widgets.QPushButton("Search for IMU")
        self._search_button.clicked.connect(self._start_imu_search)
        layout.addWidget(self._search_button)
        layout.addWidget(self._separator())
        layout.addWidget(self._heading("IMU Profile Selector"))

        group = self._qt_widgets.QButtonGroup(page)
        profile_layout = self._qt_widgets.QGridLayout()
        profile_layout.setColumnStretch(0, 1)
        profile_layout.setColumnStretch(1, 1)
        profile_specs: tuple[tuple[ProfileName, str], ...] = (
            ("default", "Default"),
            ("logs", "Logs"),
            ("balance-unicycle", "Balance Unicycle"),
            ("balance-skateboard", "Balance Skateboard"),
        )
        for index, (profile, label) in enumerate(profile_specs):
            radio = self._qt_widgets.QRadioButton(label)
            radio.clicked.connect(self._profile_click_handler(profile))
            group.addButton(radio)
            profile_layout.addWidget(radio, index // 2, index % 2)
            self._profile_radios[profile] = radio
        self._profile_radios["default"].setChecked(True)
        layout.addLayout(profile_layout)
        self._config_readout.setTextInteractionFlags(
            self._qt_core.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._config_readout.setStyleSheet("font-family: monospace;")
        layout.addWidget(self._config_readout)
        layout.addStretch(1)
        return page

    def _build_gyro_page(self) -> Any:
        page = self._make_page()
        layout = page.layout()
        assert layout is not None
        layout.addWidget(self._heading("Gyroscope Calibration"))
        layout.addWidget(
            self._paragraph(
                "To find gyroscope offsets, leave the IMU in a stable position without "
                "vibration. Any orientation is acceptable. Wait for the offsets to stabilize, "
                "then save."
            )
        )
        self._gyro_readout.setStyleSheet("font-family: monospace; font-size: 16px;")
        layout.addWidget(self._gyro_readout)
        layout.addStretch(1)
        return page

    def _build_accel_page(self, axis: str) -> Any:
        page = self._make_page()
        layout = page.layout()
        assert layout is not None
        axis_name = axis.upper()
        layout.addWidget(self._heading(f"Accel {axis_name} Calibration"))
        layout.addWidget(
            self._paragraph(
                f"Rotate the IMU to a stable maximum positive {axis_name} acceleration. "
                "Avoid sharp movement and bumps. Clear Max resets the tracked peak."
            )
        )
        readout = self._qt_widgets.QLabel()
        readout.setStyleSheet("font-family: monospace; font-size: 16px;")
        self._accel_readouts[axis] = readout
        layout.addWidget(readout)
        buttons = self._qt_widgets.QHBoxLayout()
        clear_button = self._qt_widgets.QPushButton("Clear Max")
        skip_button = self._qt_widgets.QPushButton("Skip")
        clear_button.clicked.connect(self._poller.reset_accel_max)
        skip_button.clicked.connect(self._skip_accel)
        buttons.addWidget(clear_button)
        buttons.addWidget(skip_button)
        layout.addLayout(buttons)
        layout.addStretch(1)
        return page

    def _build_orientation_page(self, axis: str) -> Any:
        page = self._make_page()
        layout = page.layout()
        assert layout is not None
        title = {
            "roll": "Roll Angle Calibration",
            "pitch": "Pitch Angle Calibration",
            "yaw": "Yaw Angle Calibration",
        }[axis]
        layout.addWidget(self._heading(title))
        if axis == "yaw":
            body = (
                "Place the IMU on a stable surface with roll leveled out and pitch raised "
                "to roughly 45 degrees. Wait for the offset to stabilize, then save."
            )
        else:
            body = (
                f"Place the IMU on a flat stable surface with pitch and roll leveled out. "
                f"Wait for the {axis} offset to stabilize, then save."
            )
        layout.addWidget(self._paragraph(body))
        readout = self._qt_widgets.QLabel()
        readout.setStyleSheet("font-family: monospace; font-size: 16px;")
        self._orientation_readouts[axis] = readout
        layout.addWidget(readout)
        skip_button = self._qt_widgets.QPushButton("Skip")
        skip_button.clicked.connect(self._skip_orientation)
        layout.addWidget(skip_button)
        layout.addStretch(1)
        return page

    def _make_page(self) -> Any:
        page = self._qt_widgets.QWidget()
        layout = self._qt_widgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        return page

    def _heading(self, text: str) -> Any:
        label = self._qt_widgets.QLabel(text)
        label.setAlignment(self._qt_core.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 18px; font-weight: 700; text-decoration: underline;")
        return label

    def _paragraph(self, text: str) -> Any:
        label = self._qt_widgets.QLabel(text)
        label.setWordWrap(True)
        return label

    def _separator(self) -> Any:
        line = self._qt_widgets.QFrame()
        line.setFrameShape(self._qt_widgets.QFrame.Shape.HLine)
        line.setFrameShadow(self._qt_widgets.QFrame.Shadow.Sunken)
        return line

    def _profile_click_handler(self, profile: ProfileName) -> Callable[[], None]:
        def handler() -> None:
            self._profile_choice = profile
            self._apply_selected_profile()

        return handler

    def _current_page(self) -> WizardPage:
        return WizardPage(self._pages.currentIndex())

    def _set_page(self, page: WizardPage) -> None:
        self._pages.setCurrentIndex(page)
        self._update_footer()
        self.refresh()

    def _update_footer(self) -> None:
        page = self._current_page()
        self._prev_button.setText("Close" if page == WizardPage.MENU else "Cancel")
        self._save_button.setVisible(page != WizardPage.MENU)
        self._save_button.setEnabled(page != WizardPage.MENU and not self._search_active)

    def _open_configurator(self) -> None:
        self._configurator_restore = dict(self._config)
        self._profile_choice = "default"
        self._profile_radios["default"].setChecked(True)
        self._apply_selected_profile()
        self._set_page(WizardPage.CONFIGURATOR)

    def _open_gyro(self) -> None:
        self._poller.reset_sample_window()
        self._set_page(WizardPage.GYRO)

    def _open_accel(self) -> None:
        self._poller.reset_accel_max()
        self._set_page(WizardPage.ACCEL_X)

    def _open_orientation(self) -> None:
        try:
            self._orientation_restore = save_orientation_restore(self._config)
            prepare_orientation_calibration(self._config)
            self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)
            self._poller.reset_sample_window()
        except Exception as exc:
            self._show_error("Could not start orientation calibration", exc)
            return
        self._set_page(WizardPage.ORIENTATION_ROLL)

    def _apply_selected_profile(self) -> None:
        try:
            apply_basic_profile(self._config, ImuBasicProfile(self._profile_choice))
        except Exception as exc:
            self._show_error("Could not apply IMU profile", exc)
        self.refresh()

    def _start_imu_search(self) -> None:
        if self._search_button is None:
            return
        self._search_active = True
        self._search_button.setEnabled(False)
        self._search_button.setText("Searching...")
        self._update_footer()
        previous_app_raw = self._config.get("app_to_use")
        self._search_previous_app = (
            int(cast(SupportsInt, previous_app_raw)) if previous_app_raw is not None else None
        )
        if self._search_previous_app is not None:
            try:
                set_config_int(self._config, "app_to_use", 0)
            except KeyError:
                self._search_previous_app = None
        self._search_candidate = 1
        self._try_search_candidate()

    def _try_search_candidate(self) -> None:
        if not self._search_active:
            return
        if self._search_candidate > 5:
            self._finish_imu_search(0)
            return

        try:
            set_config_int(self._config, "imu_conf.type", self._search_candidate)
            self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)
        except Exception as exc:
            self._finish_imu_search(config_int(self._config, "imu_conf.type"))
            self._show_error("Could not write temporary IMU type", exc)
            return

        self._qt_core.QTimer.singleShot(
            round(self._search_seconds * 1000.0),
            self._check_search_candidate,
        )
        self.refresh()

    def _check_search_candidate(self) -> None:
        if not self._search_active:
            return
        snapshot = self._poller.snapshot()
        if snapshot.raw.working_imu:
            self._finish_imu_search(self._search_candidate)
            return

        self._search_candidate += 1
        self._try_search_candidate()

    def _finish_imu_search(self, found_type: int) -> None:
        if not self._search_active:
            return
        try:
            set_config_int(self._config, "imu_conf.type", found_type)
            if self._search_previous_app is not None:
                set_config_int(self._config, "app_to_use", self._search_previous_app)
            self._apply_selected_profile()
            self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)
        except Exception as exc:
            self._show_error("Could not finish IMU search", exc)
        finally:
            self._search_active = False
            if self._search_button is not None:
                self._search_button.setEnabled(True)
                self._search_button.setText("Search for IMU")
            self._search_previous_app = None
            self._update_footer()
            self.refresh()

    def _cancel_or_close(self) -> None:
        page = self._current_page()
        if page == WizardPage.MENU:
            self.window.close()
            return

        try:
            if page == WizardPage.CONFIGURATOR and self._configurator_restore is not None:
                self._search_active = False
                if self._search_button is not None:
                    self._search_button.setEnabled(True)
                    self._search_button.setText("Search for IMU")
                self._config.clear()
                self._config.update(self._configurator_restore)
                self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)
            elif page in (
                WizardPage.ORIENTATION_ROLL,
                WizardPage.ORIENTATION_PITCH,
                WizardPage.ORIENTATION_YAW,
            ):
                self._restore_orientation()
        except Exception as exc:
            self._show_error("Could not cancel current step cleanly", exc)
        self._set_page(WizardPage.MENU)

    def _save_current_page(self) -> None:
        page = self._current_page()
        try:
            if page == WizardPage.CONFIGURATOR:
                self._client.set_appconf(self._config, store=True, wait_ack=self._wait_ack)
                self._set_page(WizardPage.MENU)
            elif page == WizardPage.GYRO:
                self._save_gyro()
            elif page in (WizardPage.ACCEL_X, WizardPage.ACCEL_Y, WizardPage.ACCEL_Z):
                self._save_accel()
            elif page in (
                WizardPage.ORIENTATION_ROLL,
                WizardPage.ORIENTATION_PITCH,
                WizardPage.ORIENTATION_YAW,
            ):
                self._save_orientation()
        except Exception as exc:
            self._show_error("Could not save calibration step", exc)

    def _save_gyro(self) -> None:
        state = self._display_state()
        save_gyro_offsets(self._config, state)
        self._client.set_appconf(self._config, store=True, wait_ack=self._wait_ack)
        self._set_page(WizardPage.MENU)

    def _save_accel(self) -> None:
        axis = self._accel_axis()
        save_accel_offset(self._config, axis, axis_max(self._display_state(), axis))
        self._client.set_appconf(self._config, store=True, wait_ack=self._wait_ack)
        self._advance_accel()

    def _skip_accel(self) -> None:
        self._advance_accel()

    def _advance_accel(self) -> None:
        page = self._current_page()
        if page == WizardPage.ACCEL_X:
            self._set_page(WizardPage.ACCEL_Y)
        elif page == WizardPage.ACCEL_Y:
            self._set_page(WizardPage.ACCEL_Z)
        else:
            self._set_page(WizardPage.MENU)

    def _accel_axis(self) -> str:
        page = self._current_page()
        if page == WizardPage.ACCEL_X:
            return "x"
        if page == WizardPage.ACCEL_Y:
            return "y"
        if page == WizardPage.ACCEL_Z:
            return "z"
        raise RuntimeError("Current page is not an accelerometer calibration page")

    def _save_orientation(self) -> None:
        state = self._display_state()
        page = self._current_page()
        if page == WizardPage.ORIENTATION_ROLL:
            apply_roll_offset(self._config, -state.roll)
            self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)
            self._poller.reset_sample_window()
            self._set_page(WizardPage.ORIENTATION_PITCH)
        elif page == WizardPage.ORIENTATION_PITCH:
            apply_pitch_offset(self._config, state.pitch)
            self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)
            self._poller.reset_yaw()
            self._set_page(WizardPage.ORIENTATION_YAW)
        elif page == WizardPage.ORIENTATION_YAW:
            apply_yaw_offset(self._config, -self._display_yaw_offset())
            self._client.set_appconf(self._config, store=True, wait_ack=self._wait_ack)
            self._set_page(WizardPage.MENU)

    def _skip_orientation(self) -> None:
        restore = self._orientation_restore
        if restore is None:
            self._set_page(WizardPage.MENU)
            return
        page = self._current_page()
        try:
            if page == WizardPage.ORIENTATION_ROLL:
                apply_roll_offset(self._config, restore.rot_roll * DEG_TO_RAD)
                self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)
                self._poller.reset_sample_window()
                self._set_page(WizardPage.ORIENTATION_PITCH)
            elif page == WizardPage.ORIENTATION_PITCH:
                apply_pitch_offset(self._config, restore.rot_pitch * DEG_TO_RAD)
                self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)
                self._poller.reset_yaw()
                self._set_page(WizardPage.ORIENTATION_YAW)
            elif page == WizardPage.ORIENTATION_YAW:
                apply_yaw_offset(self._config, restore.rot_yaw * DEG_TO_RAD)
                self._client.set_appconf(self._config, store=True, wait_ack=self._wait_ack)
                self._set_page(WizardPage.MENU)
        except Exception as exc:
            self._show_error("Could not skip orientation step", exc)

    def _restore_orientation(self) -> None:
        if self._orientation_restore is None:
            return
        restore_orientation_config(self._config, self._orientation_restore)
        self._client.set_appconf(self._config, store=False, wait_ack=self._wait_ack)

    def refresh(self) -> None:
        """Refresh all visible readouts from the latest live data."""

        snapshot = self._poller.snapshot()
        state = self._display_state_from_snapshot(snapshot)
        imu_type = config_int(self._config, "imu_conf.type")
        imu_status = "detected" if snapshot.raw.working_imu else "not detected"
        if snapshot.timestamp is None:
            age = "waiting for samples"
        else:
            age = f"sample age {time.monotonic() - snapshot.timestamp:.2f}s"
        error_text = f" | read error: {snapshot.last_error}" if snapshot.last_error else ""
        self._status_label.setText(
            f"{self._client.fw_label} | IMU {imu_status} | "
            f"{imu_type_name(imu_type)} ({imu_type}) | {age}{error_text}"
        )
        self._config_readout.setText(self._format_basic_config())
        self._gyro_readout.setText(
            "Gyro offsets ({label})\n"
            "X {gx}  Y {gy}  Z {gz}".format(
                label=mean_label(self._mean_seconds),
                gx=format_signed(state.gyro_x),
                gy=format_signed(state.gyro_y),
                gz=format_signed(state.gyro_z),
            )
        )
        for axis, label in self._accel_readouts.items():
            axis_name = axis.upper()
            label.setText(
                "Accel {axis} ({mean})\n"
                "{axis} {current}  Max {maximum}".format(
                    axis=axis_name,
                    mean=mean_label(self._mean_seconds),
                    current=format_signed(axis_current(state, axis)),
                    maximum=format_signed(axis_max(state, axis)),
                )
            )
        self._refresh_orientation_readouts(snapshot)

    def _refresh_orientation_readouts(self, snapshot: ImuSnapshot) -> None:
        restore = self._orientation_restore
        state = self._display_state_from_snapshot(snapshot)
        restore_roll = restore.rot_roll if restore is not None else config_float(
            self._config,
            "imu_conf.rot_roll",
        )
        restore_pitch = restore.rot_pitch if restore is not None else config_float(
            self._config,
            "imu_conf.rot_pitch",
        )
        restore_yaw = restore.rot_yaw if restore is not None else config_float(
            self._config,
            "imu_conf.rot_yaw",
        )
        self._orientation_readouts["roll"].setText(
            "Roll offset ({mean})\n{offset} deg".format(
                mean=mean_label(self._mean_seconds),
                offset=format_signed(-state.roll * RAD_TO_DEG - restore_roll),
            )
        )
        self._orientation_readouts["pitch"].setText(
            "Pitch offset ({mean})\n{offset} deg".format(
                mean=mean_label(self._mean_seconds),
                offset=format_signed(state.pitch * RAD_TO_DEG - restore_pitch),
            )
        )
        self._orientation_readouts["yaw"].setText(
            "Yaw offset ({mean})\n{offset} deg".format(
                mean=mean_label(self._mean_seconds),
                offset=format_signed(-self._display_yaw_offset_from_snapshot(snapshot) * RAD_TO_DEG - restore_yaw),
            )
        )

    def _display_state(self) -> FilteredImuState:
        return self._display_state_from_snapshot(self._poller.snapshot())

    def _display_state_from_snapshot(self, snapshot: ImuSnapshot) -> FilteredImuState:
        if self._mean_seconds > 0.0:
            return snapshot.mean
        return snapshot.raw

    def _display_yaw_offset(self) -> float:
        return self._display_yaw_offset_from_snapshot(self._poller.snapshot())

    def _display_yaw_offset_from_snapshot(self, snapshot: ImuSnapshot) -> float:
        if self._mean_seconds > 0.0 and snapshot.yaw_mean_offset is not None:
            return snapshot.yaw_mean_offset
        return snapshot.yaw_offset

    def _format_basic_config(self) -> str:
        imu_type = config_int(self._config, "imu_conf.type")
        return (
            f"Type:           {imu_type_name(imu_type)} ({imu_type})\n"
            f"Frequency:      {config_int(self._config, 'imu_conf.sample_rate_hz')} Hz\n"
            f"AHRS:           {mode_name(config_int(self._config, 'imu_conf.mode'))}\n"
            f"Acc Decay:      {config_float(self._config, 'imu_conf.accel_confidence_decay'):.3f}\n"
            f"Mahony kp:      {config_float(self._config, 'imu_conf.mahony_kp'):.3f}\n"
            f"Accel Z Filter: {config_float(self._config, 'imu_conf.accel_lowpass_filter_z'):.1f}\n"
            f"Gyro Filter:    {config_float(self._config, 'imu_conf.gyro_lowpass_filter'):.1f}"
        )

    def _show_error(self, title: str, exc: Exception) -> None:
        self._qt_widgets.QMessageBox.critical(self.window, title, str(exc))


def run_gui(
    client: LockedVescClient,
    *,
    poll_hz: float,
    mean_seconds: float,
    search_seconds: float,
    wait_ack: bool,
) -> None:
    """Run the PySide6 IMU setup GUI."""

    qt_core, qt_widgets = import_pyside6()
    require_qt_platform_runtime()
    app = qt_widgets.QApplication.instance()
    if app is None:
        app = qt_widgets.QApplication(sys.argv[:1])

    config = client.get_appconf()
    validate_required_fields(config)
    poller = ImuSetupPoller(client, poll_hz=poll_hz, mean_seconds=mean_seconds)
    controller = ImuSetupController(
        qt_core,
        qt_widgets,
        client,
        poller,
        config,
        wait_ack=wait_ack,
        mean_seconds=mean_seconds,
        search_seconds=search_seconds,
    )
    poller.start()
    controller.window.show()
    exec_app = getattr(app, "exec", None)
    if exec_app is None:
        exec_app = app.exec_
    try:
        exec_app()
    finally:
        controller.stop()
        poller.stop(timeout=1.0)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run a graphical VESC IMU setup wizard through a VESC Tool --tcpServer bridge.",
        epilog=(
            "Start VESC Tool first, for example: "
            "vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer 65102"
        ),
    )
    parser.add_argument(
        "--tcp",
        type=parse_tcp_endpoint,
        default=DEFAULT_TCP_ENDPOINT,
        metavar="HOST:PORT",
        help="VESC Tool TCP server endpoint (default: 127.0.0.1:65102).",
    )
    parser.add_argument(
        "--scan-udp",
        action="store_true",
        help="Scan for VESC Tool TCP server UDP broadcasts and exit.",
    )
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=3.0,
        metavar="SEC",
        help="UDP scan duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        metavar="SEC",
        help="VESC response timeout in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_POLL_HZ,
        metavar="HZ",
        help="IMU poll rate in Hz (default: 50).",
    )
    parser.add_argument(
        "--search-seconds",
        type=float,
        default=DEFAULT_SEARCH_SECONDS,
        metavar="SEC",
        help="Seconds to test each IMU type during search (default: 2).",
    )
    parser.add_argument(
        "--mean-seconds",
        type=float,
        default=DEFAULT_MEAN_SECONDS,
        metavar="SEC",
        help="Rolling mean window for displayed and saved calibration values (default: 3).",
    )
    parser.add_argument(
        "--wait-ack",
        action="store_true",
        help="Wait for acknowledgements after APPCONF writes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scan_udp:
        scan_and_print_udp(args.scan_timeout)
        return

    if args.rate <= 0.0:
        raise SystemExit("--rate must be greater than 0")
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be greater than 0")
    if args.search_seconds <= 0.0:
        raise SystemExit("--search-seconds must be greater than 0")
    if args.mean_seconds < 0.0:
        raise SystemExit("--mean-seconds must be greater than or equal to 0")

    host, port = args.tcp
    endpoint = f"{host}:{port}"
    print(f"Connecting to VESC Tool TCP server at {endpoint} ...")
    try:
        raw_client = VescClient.connect_tcp(host, port, timeout=args.timeout)
    except ConnectionRefusedError:
        raise SystemExit(tcp_server_help(endpoint, port)) from None
    except ConnectionError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "The TCP socket opened, but no VESC firmware response was received. "
            "Make sure VESC Tool is connected to a controller before starting the wizard."
        ) from None

    client = LockedVescClient(raw_client)
    try:
        print(client.fw_label)
        run_gui(
            client,
            poll_hz=args.rate,
            mean_seconds=args.mean_seconds,
            search_seconds=args.search_seconds,
            wait_ack=args.wait_ack,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
