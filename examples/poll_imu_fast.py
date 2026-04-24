#!/usr/bin/env python3
"""Poll VESC IMU data over USB serial or direct BLE with minimal overhead.

The regular VescClient API is easier to use, but it performs a firmware
handshake, uses generic packet dispatch, and returns Pydantic models. This
script keeps the hot path small: one pre-encoded COMM_GET_IMU_DATA request,
direct serial reads, CRC validation, and lightweight value decoding.

Timing note: the firmware IMU drivers run at the configured IMU sample rate and
update cached IMU state, but COMM_GET_IMU_DATA does not carry a sample timestamp
or sequence number. All timestamps in this example are host monotonic
receive-completion timestamps taken after a full response packet has been read.
dt, plot rates, and PSD frequency bins therefore describe host-observed packet
timing; do not treat them as direct IMU sampling jitter or use them for precise
phase estimates without firmware-side sample timestamps or sample indexes.

Examples:
    uv run examples/poll_imu_fast.py
    uv run examples/poll_imu_fast.py --serial /dev/ttyACM0 --duration 10
    uv run examples/poll_imu_fast.py --ble AA:BB:CC:DD:EE:FF --duration 10
    uv run examples/poll_imu_fast.py --can-id 1 --duration 10
    uv run examples/poll_imu_fast.py --fields rpy,acc,gyro --pipeline-depth 4
    uv run examples/poll_imu_fast.py --plot --plot-axis z --pipeline-depth 4
    uv run examples/poll_imu_fast.py --mask 0x01ff --csv > imu.csv
    uv run examples/poll_imu_fast.py --scan-ble
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import struct
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO, cast

from vesc_py.client import Transport, VescClient
from vesc_py.comm_ids import CommPacketId
from vesc_py.connection import VescConnectionKind, VescTarget, open_blocking_io
from vesc_py.connection_cli import (
    add_vesc_connection_arguments,
    discover_connection_candidates,
    resolve_vesc_target_from_args,
)
from vesc_py.config_schema import CfgType, ConfigParam, ConfigSchema
from vesc_py.crc import crc16
from vesc_py.packet import MAX_PACKET_LEN, encode_packet

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

DEFAULT_MASK = 0x01FF  # roll/pitch/yaw + accelerometer + gyroscope.
DEFAULT_STATUS_INTERVAL = 1.0
DEFAULT_UI_RATE = 10.0
DEFAULT_PLOT_RATE = 30.0
DEFAULT_FFT_RATE = 2.0
DEFAULT_FFT_WINDOW = 2.0
DEFAULT_PLOT_HISTORY = 20000
DEFAULT_PLOT_MAX_POINTS = 1200
DEFAULT_PLOT_PENDING = 20000
DEFAULT_APPCONF_TIMEOUT = 1.0
DEFAULT_APPCONF_FW_RETRIES = 25
WELCH_TARGET_AVERAGES = 8
WELCH_MIN_SEGMENT_SAMPLES = 8
APPCONF_LIVE_APPLY_DELAY_MS = 200
APPCONF_NON_LIVE_PARAMS = frozenset({"imu_conf.filter"})
PLOT_SAMPLE_BATCH = 64
PLOT_SAMPLE_BATCH_NS = 5_000_000
DEFAULT_PIPELINE_DEPTH = 1
NSEC_PER_SEC = 1_000_000_000
HOST_RX_TIMESTAMP_SOURCE = "host_rx_after_packet"
HOST_RX_TIMING_NOTICE = (
    "Firmware samples IMU data at the configured IMU sample rate, but "
    "COMM_GET_IMU_DATA returns only the latest cached values. This tool timestamps "
    "host packet receive completion, so dt, plot rates, and PSD timing describe "
    "host-observed packet timing. Use firmware-side sample timestamps or sample "
    "indexes for sampling-jitter or phase analysis."
)
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
IMU_PLOT_FIELDS: tuple[str, ...] = (
    "roll",
    "pitch",
    "yaw",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)

FIELD_NAMES: tuple[str, ...] = (
    "roll",
    "pitch",
    "yaw",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "q0",
    "q1",
    "q2",
    "q3",
)
FIELD_MASKS: dict[str, int] = {
    name: 1 << index for index, name in enumerate(FIELD_NAMES)
}
FIELD_GROUP_MASKS: dict[str, int] = {
    "default": DEFAULT_MASK,
    "rpy": 0x0007,
    "attitude": 0x0007,
    "acc": 0x0038,
    "accel": 0x0038,
    "accelerometer": 0x0038,
    "gyro": 0x01C0,
    "gyroscope": 0x01C0,
    "mag": 0x0E00,
    "magnetometer": 0x0E00,
    "quat": 0xF000,
    "quaternion": 0xF000,
    "all": 0xFFFF,
}
ACCEL_AXIS_ALIASES: dict[str, str] = {
    "x": "acc_x",
    "acc_x": "acc_x",
    "accel_x": "acc_x",
    "y": "acc_y",
    "acc_y": "acc_y",
    "accel_y": "acc_y",
    "z": "acc_z",
    "acc_z": "acc_z",
    "accel_z": "acc_z",
}


def import_numpy() -> Any:
    """Import NumPy lazily so non-plot commands keep their original dependency path."""
    try:
        import numpy as numpy_module
    except ImportError as exc:
        raise RuntimeError(
            "Plot mode requires numpy. Run `uv sync` inside `nix develop` "
            "to install project dependencies."
        ) from exc
    return numpy_module


class PollIo(Protocol):
    """Small blocking byte-stream surface used by the fast poller."""

    timeout: float | None

    def read(self, size: int = 1) -> bytes: ...

    def write(self, data: bytes) -> int | None: ...

    def reset_input_buffer(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class ReaderStats:
    """Counters for packet-level decode problems."""

    discarded_bytes: int = 0
    bad_crc: int = 0
    bad_stop: int = 0
    invalid_length: int = 0
    unexpected_packets: int = 0


@dataclass(slots=True)
class PollStats:
    """Counters for the polling loop."""

    requests: int = 0
    samples: int = 0
    timeouts: int = 0
    parse_errors: int = 0


@dataclass(frozen=True, slots=True)
class ParsedImu:
    """A decoded COMM_GET_IMU_DATA payload."""

    mask: int
    values: tuple[float, ...]
    vesc_id: int | None


@dataclass(frozen=True, slots=True)
class PlotTheme:
    """Colors for the single-axis PyQtGraph view."""

    pg_background: str
    pg_foreground: str
    window_background: str
    title_color: str
    status_color: str
    grid_alpha: float
    line_colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

    @property
    def line_color(self) -> tuple[int, int, int]:
        """Primary line color for the legacy single-axis plot helper."""
        return self.line_colors[0]


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


@dataclass(frozen=True, slots=True)
class FastPlotSnapshot:
    """Low-rate status data read by the plot UI."""

    samples: int
    requests: int
    timeouts: int
    parse_errors: int
    bad_crc: int
    bad_stop: int
    discarded_bytes: int
    unexpected_packets: int
    elapsed_s: float
    average_rate_hz: float
    latest_sample_s: float | None
    latest_value: float | None
    last_error: str | None
    done: bool


class TerminalImuDisplay:
    """Small ANSI terminal view that redraws IMU values in place."""

    def __init__(self, *, refresh_hz: float, stream: TextIO) -> None:
        self._stream = stream
        self._interval_ns = max(1, round(NSEC_PER_SEC / refresh_hz))
        self._next_render_ns = 0
        self._line_count = 0
        self._started = False

    def start(self) -> None:
        """Hide the cursor while the in-place display is active."""
        if self._started:
            return
        self._started = True
        self._next_render_ns = time.perf_counter_ns() + self._interval_ns
        self._stream.write("\x1b[?25l")
        self._stream.flush()

    def close(self) -> None:
        """Restore the cursor and leave the final display visible."""
        if not self._started:
            return
        self._stream.write("\x1b[?25h\n")
        self._stream.flush()
        self._started = False

    def due(self, now_ns: int) -> bool:
        """Return whether enough time has passed for another redraw."""
        return now_ns >= self._next_render_ns

    def render(
        self,
        *,
        now_ns: int,
        start_ns: int,
        sample_timestamp_ns: int,
        parsed: ParsedImu,
        poll_stats: PollStats,
        reader_stats: ReaderStats,
        current_rate: float,
        average_rate: float,
    ) -> None:
        """Redraw the latest sample and real poll-rate counters."""
        elapsed = (now_ns - start_ns) / NSEC_PER_SEC
        age_ms = (now_ns - sample_timestamp_ns) / 1_000_000.0
        vesc_id = "n/a" if parsed.vesc_id is None else str(parsed.vesc_id)
        names = field_names_for_mask(parsed.mask)

        lines = [
            "VESC IMU fast poller",
            (
                f"poll rate: {current_rate:9.1f} Hz   "
                f"average: {average_rate:9.1f} Hz   elapsed: {elapsed:8.2f}s"
            ),
            (
                f"samples: {poll_stats.samples}   requests: {poll_stats.requests}   "
                f"timeouts: {poll_stats.timeouts}   parse_errors: {poll_stats.parse_errors}"
            ),
            (
                f"bad_crc: {reader_stats.bad_crc}   bad_stop: {reader_stats.bad_stop}   "
                f"discarded: {reader_stats.discarded_bytes}   unexpected: "
                f"{reader_stats.unexpected_packets}"
            ),
            f"timing: {HOST_RX_TIMESTAMP_SOURCE}   sample_age: {age_ms:.2f} ms",
            f"vesc_id: {vesc_id}   rx_mask: 0x{parsed.mask:04x}",
            "",
            "IMU values",
        ]
        lines.extend(
            f"{name:<8} {value:>16.8g}"
            for name, value in zip(names, parsed.values)
        )

        if self._line_count > 0:
            self._stream.write(f"\x1b[{self._line_count}A")
        for line in lines:
            self._stream.write(f"\r\x1b[2K{line}\n")
        self._stream.flush()

        self._line_count = len(lines)
        self._next_render_ns = now_ns + self._interval_ns


def parse_mask_arg(text: str) -> int:
    """Parse a numeric 16-bit IMU mask."""
    try:
        mask = int(text, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mask must be an integer, e.g. 0x01ff") from exc

    if not 1 <= mask <= 0xFFFF:
        raise argparse.ArgumentTypeError("mask must be in range 1..0xffff")
    return mask


def parse_fields_arg(text: str) -> int:
    """Parse comma-separated field or field-group names into an IMU mask."""
    mask = 0
    tokens = text.replace("+", ",").split(",")
    for raw_token in tokens:
        token = raw_token.strip().lower().replace("-", "_")
        if not token:
            continue
        if token in FIELD_GROUP_MASKS:
            mask |= FIELD_GROUP_MASKS[token]
            continue
        if token in FIELD_MASKS:
            mask |= FIELD_MASKS[token]
            continue
        valid = ", ".join(sorted((*FIELD_GROUP_MASKS, *FIELD_MASKS)))
        raise argparse.ArgumentTypeError(
            f"unknown IMU field '{raw_token}'. Valid names: {valid}"
        )

    if mask == 0:
        raise argparse.ArgumentTypeError("at least one IMU field is required")
    return mask


def parse_accel_axis_arg(text: str) -> str:
    """Parse a single accelerometer axis name."""
    token = text.strip().lower().replace("-", "_")
    try:
        return ACCEL_AXIS_ALIASES[token]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "axis must be one of x, y, z, acc_x, acc_y, or acc_z"
        ) from exc


def field_names_for_mask(mask: int) -> tuple[str, ...]:
    """Return IMU field names in VESC wire order for *mask*."""
    return tuple(name for index, name in enumerate(FIELD_NAMES) if mask & (1 << index))


def field_value_index(mask: int, field_name: str) -> int | None:
    """Return the decoded values tuple index for *field_name* under *mask*."""
    try:
        field_index = FIELD_NAMES.index(field_name)
    except ValueError:
        return None

    field_bit = 1 << field_index
    if not mask & field_bit:
        return None
    preceding_fields = mask & (field_bit - 1)
    return preceding_fields.bit_count()


def decode_double32_auto(word: int) -> float:
    """Decode VESC's custom 32-bit auto-scaled float format."""
    exponent = (word >> 23) & 0xFF
    fraction = word & 0x7FFFFF
    negative = bool(word & (1 << 31))

    value = 0.0
    if exponent != 0 or fraction != 0:
        value = fraction / (8388608.0 * 2.0) + 0.5
        exponent -= 126

    if negative:
        value = -value
    return math.ldexp(value, exponent)


def parse_imu_payload(payload: bytes) -> ParsedImu:
    """Decode a COMM_GET_IMU_DATA response payload without Pydantic overhead."""
    if len(payload) < 3:
        raise ValueError(f"IMU payload is too short: {len(payload)} bytes")
    if payload[0] != CommPacketId.COMM_GET_IMU_DATA:
        raise ValueError(f"unexpected command id {payload[0]}")

    mask = (payload[1] << 8) | payload[2]
    offset = 3
    values: list[float] = []
    for index in range(len(FIELD_NAMES)):
        if mask & (1 << index):
            if offset + 4 > len(payload):
                raise ValueError(
                    f"IMU payload ended inside field {FIELD_NAMES[index]}"
                )
            word = struct.unpack_from(">I", payload, offset)[0]
            values.append(decode_double32_auto(word))
            offset += 4

    vesc_id = None
    if offset < len(payload):
        vesc_id = payload[offset]

    return ParsedImu(mask=mask, values=tuple(values), vesc_id=vesc_id)


def build_imu_request(mask: int, can_id: int | None = None) -> bytes:
    """Build a fully framed COMM_GET_IMU_DATA request once for reuse."""
    payload = bytes(
        (
            int(CommPacketId.COMM_GET_IMU_DATA),
            (mask >> 8) & 0xFF,
            mask & 0xFF,
        )
    )
    if can_id is not None:
        if not 0 <= can_id <= 253:
            raise ValueError(f"CAN ID {can_id} out of range [0, 253]")
        payload = bytes((int(CommPacketId.COMM_FORWARD_CAN), can_id)) + payload
    return encode_packet(payload)


def read_exact(serial_port: PollIo, size: int, deadline_ns: int) -> bytes | None:
    """Read exactly *size* bytes before *deadline_ns*."""
    chunks = bytearray()
    while len(chunks) < size:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return None

        serial_port.timeout = remaining_ns / NSEC_PER_SEC
        chunk = serial_port.read(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)

    return bytes(chunks)


def read_packet(
    serial_port: PollIo,
    deadline_ns: int,
    stats: ReaderStats,
) -> bytes | None:
    """Read one valid VESC packet payload before *deadline_ns*."""
    while True:
        start = read_exact(serial_port, 1, deadline_ns)
        if start is None:
            return None
        start_byte = start[0]
        if start_byte in (2, 3, 4):
            break
        stats.discarded_bytes += 1

    if start_byte == 2:
        length_raw = read_exact(serial_port, 1, deadline_ns)
        if length_raw is None:
            return None
        payload_len = length_raw[0]
        if payload_len < 1:
            stats.invalid_length += 1
            return None
    elif start_byte == 3:
        length_raw = read_exact(serial_port, 2, deadline_ns)
        if length_raw is None:
            return None
        payload_len = (length_raw[0] << 8) | length_raw[1]
        if payload_len < 255:
            stats.invalid_length += 1
            return None
    else:
        length_raw = read_exact(serial_port, 3, deadline_ns)
        if length_raw is None:
            return None
        payload_len = (length_raw[0] << 16) | (length_raw[1] << 8) | length_raw[2]
        if payload_len < 65535:
            stats.invalid_length += 1
            return None

    if payload_len > MAX_PACKET_LEN:
        stats.invalid_length += 1
        return None

    body = read_exact(serial_port, payload_len + 3, deadline_ns)
    if body is None:
        return None
    if body[-1] != 3:
        stats.bad_stop += 1
        return None

    payload = body[:payload_len]
    received_crc = (body[payload_len] << 8) | body[payload_len + 1]
    if crc16(payload) != received_crc:
        stats.bad_crc += 1
        return None

    return payload


def read_expected_imu_packet(
    serial_port: PollIo,
    packet_timeout: float,
    reader_stats: ReaderStats,
) -> bytes | None:
    """Read packets until an IMU response arrives or the timeout expires."""
    deadline_ns = time.perf_counter_ns() + round(packet_timeout * NSEC_PER_SEC)
    while True:
        payload = read_packet(serial_port, deadline_ns, reader_stats)
        if payload is None:
            return None
        if payload and payload[0] == CommPacketId.COMM_GET_IMU_DATA:
            return payload
        reader_stats.unexpected_packets += 1


class SharedPollTransport(Transport):
    """VescClient transport that reuses the already-open fast-poller I/O path."""

    def __init__(self, poll_io: PollIo) -> None:
        self._poll_io = poll_io

    def send(self, data: bytes) -> None:
        self._poll_io.write(data)

    def recv(self, timeout: float) -> bytes:
        self._poll_io.timeout = timeout
        return self._poll_io.read(4096)

    def close(self) -> None:
        """Leave ownership of the I/O object with the fast poller."""
        return None


class PlotAppConfigClient:
    """Small locked APPCONF client for plot-mode live IMU config edits."""

    def __init__(self, client: VescClient, command_lock: Any) -> None:
        self._client = client
        self._command_lock = command_lock

    @classmethod
    def connect(
        cls,
        poll_io: PollIo,
        *,
        command_lock: Any,
        timeout: float,
        fw_retries: int,
        config_dir: Path | None = None,
    ) -> PlotAppConfigClient:
        transport = SharedPollTransport(poll_io)
        client = VescClient(
            transport,
            timeout=timeout,
            fw_retries=fw_retries,
            config_dir=config_dir,
        )
        with command_lock:
            client._handshake()

        if client.appconf_schema is None:
            raise RuntimeError(
                "No APPCONF schema loaded for this firmware version; "
                "cannot edit IMU app config."
            )
        return cls(client, command_lock)

    @property
    def schema(self) -> ConfigSchema:
        schema = self._client.appconf_schema
        if schema is None:
            raise RuntimeError("No APPCONF schema loaded")
        return schema

    @property
    def firmware_label(self) -> str:
        fw = self._client.fw_version
        if fw is None:
            return "Firmware: unknown"
        name = f" {fw.fw_name}" if fw.fw_name else ""
        hw = f"  HW: {fw.hw}" if fw.hw else ""
        return f"Firmware: {fw.major}.{fw.minor:02d}{name}{hw}"

    def get_appconf(self) -> dict[str, object]:
        with self._command_lock:
            return self._client.get_appconf()

    def set_appconf(
        self,
        config: dict[str, object],
        *,
        store: bool,
        wait_ack: bool,
    ) -> None:
        with self._command_lock:
            self._client.set_appconf(config, store=store, wait_ack=wait_ack)

    def close(self) -> None:
        with self._command_lock:
            self._client.close()


def print_serial_ports(args: argparse.Namespace) -> None:
    """Print serial candidates found by the shared VESC discovery helper."""
    candidates = [
        candidate
        for candidate in discover_connection_candidates(args, output_stream=sys.stderr)
        if candidate.connection.kind is VescConnectionKind.SERIAL
    ]
    if not candidates:
        print("No serial ports found.")
        return

    for candidate in candidates:
        print(candidate.label)


def scan_and_print_ble(args: argparse.Namespace) -> None:
    """Print BLE candidates found by the shared VESC discovery helper."""
    print(f"Scanning for VESC BLE devices for {args.ble_scan_timeout:g}s ...")
    candidates = [
        candidate
        for candidate in discover_connection_candidates(args, output_stream=sys.stderr)
        if candidate.connection.kind is VescConnectionKind.BLE
    ]
    if not candidates:
        print("No VESC BLE devices found.")
        return

    for candidate in candidates:
        print(f"  {candidate.label}")


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


def open_poll_connection(target: VescTarget, *, timeout: float) -> tuple[PollIo, str]:
    """Open the resolved transport selected by the shared VESC CLI helper."""
    connection = target.connection
    if connection.kind is VescConnectionKind.BLE:
        print(f"Connecting to VESC BLE device at {connection.address} ...", file=sys.stderr)

    try:
        return cast(PollIo, open_blocking_io(connection, timeout=timeout)), connection.address
    except TimeoutError:
        if connection.kind is VescConnectionKind.BLE:
            raise SystemExit(
                ble_connection_help(
                    connection.address,
                    "Timed out while connecting to the BLE peripheral.",
                )
            ) from None
        raise SystemExit(f"Timed out while opening {connection.describe()}.") from None
    except (ConnectionError, OSError, ValueError) as exc:
        if connection.kind is VescConnectionKind.BLE:
            raise SystemExit(
                ble_connection_help(
                    connection.address,
                    (
                        f"{exc}\n\n"
                        "The BLE link opened, but the VESC did not respond as expected. "
                        "Make sure the BLE module is connected to a controller."
                    ),
                )
            ) from None
        raise SystemExit(f"Could not open {connection.describe()}: {exc}") from None


def format_values(mask: int, values: tuple[float, ...], max_fields: int = 9) -> str:
    """Format selected IMU values for low-rate status output."""
    names = field_names_for_mask(mask)
    parts: list[str] = []
    for name, value in zip(names[:max_fields], values[:max_fields]):
        parts.append(f"{name}={value:.6g}")
    if len(values) > max_fields:
        parts.append("...")
    return " ".join(parts)


def write_csv_header(mask: int) -> None:
    """Write the CSV header for the requested field mask."""
    fields = ",".join(field_names_for_mask(mask))
    print(f"host_rx_time_s,host_rx_dt_s,vesc_id,rx_mask,{fields}")


def write_sample(
    sample_index: int,
    timestamp_ns: int,
    start_ns: int,
    previous_ns: int | None,
    parsed: ParsedImu,
    *,
    csv: bool,
) -> None:
    """Write one sample in CSV or human-readable form."""
    host_rx_elapsed = (timestamp_ns - start_ns) / NSEC_PER_SEC
    dt = "" if previous_ns is None else f"{(timestamp_ns - previous_ns) / NSEC_PER_SEC:.9f}"
    vesc_id = "" if parsed.vesc_id is None else str(parsed.vesc_id)

    if csv:
        values = ",".join(f"{value:.9g}" for value in parsed.values)
        sys.stdout.write(
            f"{host_rx_elapsed:.9f},{dt},{vesc_id},0x{parsed.mask:04x},{values}\n"
        )
        return

    value_text = format_values(parsed.mask, parsed.values, max_fields=len(parsed.values))
    sys.stdout.write(
        f"sample={sample_index} host_rx_t={host_rx_elapsed:.6f}s "
        f"host_rx_dt={dt or 'n/a'} "
        f"id={vesc_id or 'n/a'} mask=0x{parsed.mask:04x} {value_text}\n"
    )


def should_stop(
    start_ns: int,
    samples: int,
    *,
    duration: float,
    max_samples: int,
) -> bool:
    """Return whether the requested run limit has been reached."""
    if max_samples > 0 and samples >= max_samples:
        return True
    if duration > 0.0:
        elapsed = (time.perf_counter_ns() - start_ns) / NSEC_PER_SEC
        return elapsed >= duration
    return False


class AxisSampleBuffer:
    """Thread-safe overwrite ring for samples waiting for the UI tick."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        np_module = import_numpy()
        self._capacity = capacity
        self._timestamps: npt.NDArray[np.float64] = np_module.zeros(
            capacity,
            dtype=np_module.float64,
        )
        self._values: npt.NDArray[np.float64] = np_module.zeros(
            capacity,
            dtype=np_module.float64,
        )
        self._lock = threading.Lock()
        self._read_index = 0
        self._write_index = 0
        self._count = 0
        self._dropped = 0

    def append(self, timestamp_s: float, value: float) -> None:
        """Append one sample, overwriting the oldest pending sample if full."""
        self.append_many((timestamp_s,), (value,))

    def append_many(
        self,
        timestamps: Sequence[float],
        values: Sequence[float],
    ) -> None:
        """Append multiple samples while taking the shared lock once."""
        count = len(timestamps)
        if count == 0:
            return
        if count != len(values):
            raise ValueError("timestamp and value counts must match")

        with self._lock:
            if count >= self._capacity:
                self._dropped += self._count + count - self._capacity
                self._timestamps[:] = timestamps[-self._capacity :]
                self._values[:] = values[-self._capacity :]
                self._read_index = 0
                self._write_index = 0
                self._count = self._capacity
                return

            overflow = max(0, self._count + count - self._capacity)
            if overflow > 0:
                self._read_index = (self._read_index + overflow) % self._capacity
                self._dropped += overflow
            self._count = min(self._capacity, self._count + count)

            first_count = min(count, self._capacity - self._write_index)
            self._timestamps[self._write_index : self._write_index + first_count] = (
                timestamps[:first_count]
            )
            self._values[self._write_index : self._write_index + first_count] = values[
                :first_count
            ]

            remaining = count - first_count
            if remaining > 0:
                self._timestamps[:remaining] = timestamps[first_count:]
                self._values[:remaining] = values[first_count:]

            self._write_index = (self._write_index + count) % self._capacity

    def drain(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        """Return pending samples in order and clear the pending ring."""
        np_module = import_numpy()
        with self._lock:
            count = self._count
            dropped = self._dropped
            self._dropped = 0
            if count == 0:
                return (
                    np_module.empty(0, dtype=np_module.float64),
                    np_module.empty(0, dtype=np_module.float64),
                    dropped,
                )

            read_index = self._read_index
            if read_index + count <= self._capacity:
                timestamps = self._timestamps[read_index : read_index + count].copy()
                values = self._values[read_index : read_index + count].copy()
            else:
                first_count = self._capacity - read_index
                timestamps = np_module.concatenate(
                    (
                        self._timestamps[read_index:],
                        self._timestamps[: count - first_count],
                    )
                )
                values = np_module.concatenate(
                    (
                        self._values[read_index:],
                        self._values[: count - first_count],
                    )
                )

            self._read_index = self._write_index
            self._count = 0
            return timestamps, values, dropped


class ImuSampleBuffer:
    """Thread-safe overwrite ring for full 9-channel plot samples."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        np_module = import_numpy()
        self._capacity = capacity
        self._timestamps: npt.NDArray[np.float64] = np_module.zeros(
            capacity,
            dtype=np_module.float64,
        )
        self._values: npt.NDArray[np.float64] = np_module.zeros(
            (IMU_PLOT_CHANNELS, capacity),
            dtype=np_module.float64,
        )
        self._lock = threading.Lock()
        self._read_index = 0
        self._write_index = 0
        self._count = 0
        self._dropped = 0

    def append_many(
        self,
        timestamps: Sequence[float],
        values: Sequence[Sequence[float]],
    ) -> None:
        """Append batched IMU samples while taking the shared lock once."""
        count = len(timestamps)
        if count == 0:
            return
        if count != len(values):
            raise ValueError("timestamp and value counts must match")

        np_module = import_numpy()
        value_array = np_module.asarray(values, dtype=np_module.float64)
        if value_array.shape != (count, IMU_PLOT_CHANNELS):
            raise ValueError(
                f"values must have shape ({count}, {IMU_PLOT_CHANNELS}), "
                f"got {value_array.shape}"
            )

        with self._lock:
            if count >= self._capacity:
                self._dropped += self._count + count - self._capacity
                self._timestamps[:] = timestamps[-self._capacity :]
                self._values[:, :] = value_array[-self._capacity :].T
                self._read_index = 0
                self._write_index = 0
                self._count = self._capacity
                return

            overflow = max(0, self._count + count - self._capacity)
            if overflow > 0:
                self._read_index = (self._read_index + overflow) % self._capacity
                self._dropped += overflow
            self._count = min(self._capacity, self._count + count)

            first_count = min(count, self._capacity - self._write_index)
            self._timestamps[self._write_index : self._write_index + first_count] = (
                timestamps[:first_count]
            )
            self._values[:, self._write_index : self._write_index + first_count] = (
                value_array[:first_count].T
            )

            remaining = count - first_count
            if remaining > 0:
                self._timestamps[:remaining] = timestamps[first_count:]
                self._values[:, :remaining] = value_array[first_count:].T

            self._write_index = (self._write_index + count) % self._capacity

    def drain(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        """Return pending samples in order and clear the pending ring."""
        np_module = import_numpy()
        with self._lock:
            count = self._count
            dropped = self._dropped
            self._dropped = 0
            if count == 0:
                return (
                    np_module.empty(0, dtype=np_module.float64),
                    np_module.empty((IMU_PLOT_CHANNELS, 0), dtype=np_module.float64),
                    dropped,
                )

            read_index = self._read_index
            if read_index + count <= self._capacity:
                timestamps = self._timestamps[read_index : read_index + count].copy()
                values = self._values[:, read_index : read_index + count].copy()
            else:
                first_count = self._capacity - read_index
                timestamps = np_module.concatenate(
                    (
                        self._timestamps[read_index:],
                        self._timestamps[: count - first_count],
                    )
                )
                values = np_module.concatenate(
                    (
                        self._values[:, read_index:],
                        self._values[:, : count - first_count],
                    ),
                    axis=1,
                )

            self._read_index = self._write_index
            self._count = 0
            return timestamps, values, dropped


class AxisPlotHistory:
    """Fixed-size numeric history for one plotted axis."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        np_module = import_numpy()
        self._capacity = capacity
        self._count = 0
        self._timestamps: npt.NDArray[np.float64] = np_module.zeros(
            capacity,
            dtype=np_module.float64,
        )
        self._values: npt.NDArray[np.float64] = np_module.zeros(
            capacity,
            dtype=np_module.float64,
        )
        self._write_index = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def latest_timestamp(self) -> float | None:
        if self._count == 0:
            return None
        return float(self._timestamps[(self._write_index - 1) % self._capacity])

    def valid_timestamps(self) -> npt.NDArray[np.float64]:
        if self._count == 0:
            return self._timestamps[:0]
        return self._ordered_values(self._timestamps)

    def valid_values(self) -> npt.NDArray[np.float64]:
        if self._count == 0:
            return self._values[:0]
        return self._ordered_values(self._values)

    def sample_hz(self) -> float | None:
        if self._count < 2:
            return None

        timestamps = self.valid_timestamps()
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0.0:
            return None

        return float((self._count - 1) / elapsed)

    def append_samples(
        self,
        timestamps: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> None:
        sample_count = int(timestamps.size)
        if sample_count == 0:
            return

        if sample_count >= self._capacity:
            self._timestamps[:] = timestamps[-self._capacity :]
            self._values[:] = values[-self._capacity :]
            self._count = self._capacity
            self._write_index = 0
            return

        first_count = min(sample_count, self._capacity - self._write_index)
        self._timestamps[self._write_index : self._write_index + first_count] = (
            timestamps[:first_count]
        )
        self._values[self._write_index : self._write_index + first_count] = values[
            :first_count
        ]

        remaining = sample_count - first_count
        if remaining > 0:
            self._timestamps[:remaining] = timestamps[first_count:]
            self._values[:remaining] = values[first_count:]

        self._write_index = (self._write_index + sample_count) % self._capacity
        self._count = min(self._capacity, self._count + sample_count)

    def _ordered_values(
        self,
        source: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        np_module = import_numpy()
        start = (self._write_index - self._count) % self._capacity
        if start + self._count <= self._capacity:
            return source[start : start + self._count]
        return cast(
            "npt.NDArray[np.float64]",
            np_module.concatenate((source[start:], source[: self._write_index])),
        )


class ImuPlotHistory:
    """Fixed-size ring history for the 9-channel IMU plot grid."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        np_module = import_numpy()
        self._capacity = capacity
        self._count = 0
        self._write_index = 0
        self._timestamps: npt.NDArray[np.float64] = np_module.zeros(
            capacity,
            dtype=np_module.float64,
        )
        self._values: npt.NDArray[np.float64] = np_module.zeros(
            (IMU_PLOT_CHANNELS, capacity),
            dtype=np_module.float64,
        )

    @property
    def count(self) -> int:
        return self._count

    @property
    def latest_timestamp(self) -> float | None:
        if self._count == 0:
            return None
        return float(self._timestamps[(self._write_index - 1) % self._capacity])

    def valid_timestamps(self) -> npt.NDArray[np.float64]:
        if self._count == 0:
            return self._timestamps[:0]
        return self._ordered_timestamps()

    def valid_values(self) -> npt.NDArray[np.float64]:
        if self._count == 0:
            return self._values[:, :0]
        return self._ordered_values()

    def sample_hz(self) -> float | None:
        if self._count < 2:
            return None

        timestamps = self.valid_timestamps()
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0.0:
            return None
        return float((self._count - 1) / elapsed)

    def append_samples(
        self,
        timestamps: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> None:
        sample_count = int(timestamps.size)
        if sample_count == 0:
            return
        if values.shape != (IMU_PLOT_CHANNELS, sample_count):
            raise ValueError(
                f"values must have shape ({IMU_PLOT_CHANNELS}, {sample_count}), "
                f"got {values.shape}"
            )

        if sample_count >= self._capacity:
            self._timestamps[:] = timestamps[-self._capacity :]
            self._values[:, :] = values[:, -self._capacity :]
            self._count = self._capacity
            self._write_index = 0
            return

        first_count = min(sample_count, self._capacity - self._write_index)
        self._timestamps[self._write_index : self._write_index + first_count] = (
            timestamps[:first_count]
        )
        self._values[:, self._write_index : self._write_index + first_count] = values[
            :,
            :first_count,
        ]

        remaining = sample_count - first_count
        if remaining > 0:
            self._timestamps[:remaining] = timestamps[first_count:]
            self._values[:, :remaining] = values[:, first_count:]

        self._write_index = (self._write_index + sample_count) % self._capacity
        self._count = min(self._capacity, self._count + sample_count)

    def _ordered_timestamps(self) -> npt.NDArray[np.float64]:
        np_module = import_numpy()
        start = (self._write_index - self._count) % self._capacity
        if start + self._count <= self._capacity:
            return self._timestamps[start : start + self._count]
        return cast(
            "npt.NDArray[np.float64]",
            np_module.concatenate(
                (self._timestamps[start:], self._timestamps[: self._write_index])
            ),
        )

    def _ordered_values(self) -> npt.NDArray[np.float64]:
        np_module = import_numpy()
        start = (self._write_index - self._count) % self._capacity
        if start + self._count <= self._capacity:
            return self._values[:, start : start + self._count]
        return cast(
            "npt.NDArray[np.float64]",
            np_module.concatenate(
                (self._values[:, start:], self._values[:, : self._write_index]),
                axis=1,
            ),
        )


@dataclass
class FrequencyAxisRange:
    """Track frequency plot bounds so ranges are not forced every PSD refresh."""

    x_max: float = 0.0
    y_max: float = 0.0


def _welch_segment_length(sample_count: int) -> int:
    if sample_count < WELCH_MIN_SEGMENT_SAMPLES * 2:
        return sample_count

    target_segments = min(
        WELCH_TARGET_AVERAGES,
        max(1, sample_count // WELCH_MIN_SEGMENT_SAMPLES),
    )
    if target_segments <= 1:
        return sample_count

    segment_length = int((2 * sample_count) / (target_segments + 1))
    return min(sample_count, max(WELCH_MIN_SEGMENT_SAMPLES, segment_length))


def _welch_psd(
    samples: npt.NDArray[np.float64],
    *,
    sample_period: float,
    axis: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return one-sided Welch PSD for samples along *axis*."""
    np_module = import_numpy()
    sample_count = int(samples.shape[axis])
    segment_length = _welch_segment_length(sample_count)
    step = max(1, segment_length // 2)
    window = np_module.hanning(segment_length)
    if not np_module.any(window):
        window = np_module.ones(segment_length, dtype=np_module.float64)
    window_shape = [1] * samples.ndim
    window_shape[axis] = segment_length
    shaped_window = window.reshape(tuple(window_shape))
    sample_rate_hz = 1.0 / sample_period
    scale = sample_rate_hz * float(np_module.sum(window * window))

    segment_psds: list[npt.NDArray[np.float64]] = []
    for start in range(0, sample_count - segment_length + 1, step):
        segment = np_module.take(
            samples,
            np_module.arange(start, start + segment_length),
            axis=axis,
        )
        segment = segment - np_module.mean(segment, axis=axis, keepdims=True)
        segment = segment * shaped_window
        spectrum = np_module.fft.rfft(segment, axis=axis)
        psd = (np_module.abs(spectrum) ** 2) / scale
        if psd.shape[axis] > 1:
            multiplier = np_module.ones(psd.shape[axis], dtype=np_module.float64)
            if segment_length % 2 == 0:
                multiplier[1:-1] = 2.0
            else:
                multiplier[1:] = 2.0
            multiplier_shape = [1] * psd.ndim
            multiplier_shape[axis] = psd.shape[axis]
            psd = psd * multiplier.reshape(tuple(multiplier_shape))
        segment_psds.append(cast("npt.NDArray[np.float64]", psd))

    frequencies = cast(
        "npt.NDArray[np.float64]",
        np_module.fft.rfftfreq(segment_length, d=sample_period),
    )
    psd_mean = cast("npt.NDArray[np.float64]", np_module.mean(segment_psds, axis=0))
    return frequencies, psd_mean


def axis_frequency_psd(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    *,
    window_s: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, int] | None:
    """Return Welch PSD bins using the supplied host receive timestamps."""
    if timestamps.size < 2:
        return None

    np_module = import_numpy()
    window_start_s = float(timestamps[-1]) - window_s
    start_index = int(np_module.searchsorted(timestamps, window_start_s, side="left"))
    window_timestamps = timestamps[start_index:]
    window_values = values[start_index:]
    sample_count = int(window_timestamps.size)
    if sample_count < 2:
        return None

    sample_periods = np_module.diff(window_timestamps)
    sample_periods = sample_periods[sample_periods > 0.0]
    if sample_periods.size == 0:
        return None

    sample_period = float(np_module.median(sample_periods))
    if not math.isfinite(sample_period) or sample_period <= 0.0:
        return None

    frequencies, psd = _welch_psd(window_values, sample_period=sample_period, axis=0)
    nyquist_hz = 0.5 / sample_period
    return frequencies, psd, nyquist_hz, sample_count


def imu_frequency_psd(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    *,
    window_s: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, int] | None:
    """Return Welch PSD bins using the supplied host receive timestamps."""
    if timestamps.size < 2:
        return None

    np_module = import_numpy()
    window_start_s = float(timestamps[-1]) - window_s
    start_index = int(np_module.searchsorted(timestamps, window_start_s, side="left"))
    window_timestamps = timestamps[start_index:]
    window_values = values[:, start_index:]
    sample_count = int(window_timestamps.size)
    if sample_count < 2:
        return None

    sample_periods = np_module.diff(window_timestamps)
    sample_periods = sample_periods[sample_periods > 0.0]
    if sample_periods.size == 0:
        return None

    sample_period = float(np_module.median(sample_periods))
    if not math.isfinite(sample_period) or sample_period <= 0.0:
        return None

    frequencies, psd = _welch_psd(window_values, sample_period=sample_period, axis=1)
    nyquist_hz = 0.5 / sample_period
    return frequencies, psd, nyquist_hz, sample_count


def axis_frequency_spectrum(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    *,
    window_s: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, int] | None:
    """Return full-window FFT magnitudes using host receive timestamps."""
    if timestamps.size < 2:
        return None

    np_module = import_numpy()
    window_start_s = float(timestamps[-1]) - window_s
    start_index = int(np_module.searchsorted(timestamps, window_start_s, side="left"))
    window_timestamps = timestamps[start_index:]
    window_values = values[start_index:]
    sample_count = int(window_timestamps.size)
    if sample_count < 2:
        return None

    sample_periods = np_module.diff(window_timestamps)
    sample_periods = sample_periods[sample_periods > 0.0]
    if sample_periods.size == 0:
        return None

    sample_period = float(np_module.median(sample_periods))
    if not math.isfinite(sample_period) or sample_period <= 0.0:
        return None

    centered_values = window_values - float(np_module.mean(window_values))
    magnitudes = cast(
        "npt.NDArray[np.float64]",
        np_module.abs(np_module.fft.rfft(centered_values)) / sample_count,
    )
    frequencies = cast(
        "npt.NDArray[np.float64]",
        np_module.fft.rfftfreq(sample_count, d=sample_period),
    )
    nyquist_hz = 0.5 / sample_period
    return frequencies, magnitudes, nyquist_hz, sample_count


def imu_frequency_spectrum(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    *,
    window_s: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, int] | None:
    """Return full-window FFT magnitudes using host receive timestamps."""
    if timestamps.size < 2:
        return None

    np_module = import_numpy()
    window_start_s = float(timestamps[-1]) - window_s
    start_index = int(np_module.searchsorted(timestamps, window_start_s, side="left"))
    window_timestamps = timestamps[start_index:]
    window_values = values[:, start_index:]
    sample_count = int(window_timestamps.size)
    if sample_count < 2:
        return None

    sample_periods = np_module.diff(window_timestamps)
    sample_periods = sample_periods[sample_periods > 0.0]
    if sample_periods.size == 0:
        return None

    sample_period = float(np_module.median(sample_periods))
    if not math.isfinite(sample_period) or sample_period <= 0.0:
        return None

    centered_values = window_values - np_module.mean(window_values, axis=1, keepdims=True)
    magnitudes = cast(
        "npt.NDArray[np.float64]",
        np_module.abs(np_module.fft.rfft(centered_values, axis=1)) / sample_count,
    )
    frequencies = cast(
        "npt.NDArray[np.float64]",
        np_module.fft.rfftfreq(sample_count, d=sample_period),
    )
    nyquist_hz = 0.5 / sample_period
    return frequencies, magnitudes, nyquist_hz, sample_count


class FastImuAxisPoller:
    """Poll a single decoded IMU field on a background thread."""

    def __init__(
        self,
        serial_port: PollIo,
        *,
        request: bytes,
        mask: int,
        field_name: str,
        packet_timeout: float,
        duration: float,
        max_samples: int,
        pipeline_depth: int,
        pending_samples: int,
        command_lock: Any | None = None,
    ) -> None:
        value_index = field_value_index(mask, field_name)
        if value_index is None:
            raise ValueError(f"mask 0x{mask:04x} does not include {field_name}")

        self._serial_port = serial_port
        self._request = request
        self._mask = mask
        self._field_name = field_name
        self._field_value_index = value_index
        self._packet_timeout = packet_timeout
        self._duration = duration
        self._max_samples = max_samples
        self._pipeline_depth = pipeline_depth
        self._command_lock = command_lock or threading.RLock()
        self._samples = AxisSampleBuffer(pending_samples)
        self._reader_stats = ReaderStats()
        self._poll_stats = PollStats()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="vesc-fast-imu-axis-poller",
            daemon=True,
        )
        self._start_ns = 0
        self._latest_sample_s: float | None = None
        self._latest_value: float | None = None
        self._last_error: str | None = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def drain(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        return self._samples.drain()

    def snapshot(self) -> FastPlotSnapshot:
        now_ns = time.perf_counter_ns()
        elapsed_s = (
            (now_ns - self._start_ns) / NSEC_PER_SEC
            if self._start_ns > 0
            else 0.0
        )
        average_rate = (
            self._poll_stats.samples / elapsed_s
            if elapsed_s > 0.0
            else 0.0
        )
        return FastPlotSnapshot(
            samples=self._poll_stats.samples,
            requests=self._poll_stats.requests,
            timeouts=self._poll_stats.timeouts,
            parse_errors=self._poll_stats.parse_errors,
            bad_crc=self._reader_stats.bad_crc,
            bad_stop=self._reader_stats.bad_stop,
            discarded_bytes=self._reader_stats.discarded_bytes,
            unexpected_packets=self._reader_stats.unexpected_packets,
            elapsed_s=elapsed_s,
            average_rate_hz=average_rate,
            latest_sample_s=self._latest_sample_s,
            latest_value=self._latest_value,
            last_error=self._last_error,
            done=self.done,
        )

    def _value_from_payload(self, payload: bytes) -> float:
        parsed = parse_imu_payload(payload)
        if parsed.mask == self._mask:
            return parsed.values[self._field_value_index]

        value_index = field_value_index(parsed.mask, self._field_name)
        if value_index is None:
            raise ValueError(
                f"IMU response mask 0x{parsed.mask:04x} does not include "
                f"{self._field_name}"
            )
        return parsed.values[value_index]

    def _run(self) -> None:
        self._start_ns = time.perf_counter_ns()
        outstanding = 0
        pending_timestamps: list[float] = []
        pending_values: list[float] = []
        next_batch_ns = self._start_ns + PLOT_SAMPLE_BATCH_NS

        def flush_pending() -> None:
            if not pending_timestamps:
                return
            self._samples.append_many(pending_timestamps, pending_values)
            pending_timestamps.clear()
            pending_values.clear()

        try:
            while not self._stop.is_set() and not should_stop(
                self._start_ns,
                self._poll_stats.samples,
                duration=self._duration,
                max_samples=self._max_samples,
            ):
                with self._command_lock:
                    while outstanding < self._pipeline_depth and not self._stop.is_set():
                        self._serial_port.write(self._request)
                        self._poll_stats.requests += 1
                        outstanding += 1

                    payload = read_expected_imu_packet(
                        self._serial_port,
                        self._packet_timeout,
                        self._reader_stats,
                    )
                    # This is a host receive-completion timestamp, not an IMU
                    # acquisition timestamp.
                    now_ns = time.perf_counter_ns()
                    if payload is None:
                        self._poll_stats.timeouts += 1
                        outstanding = 0
                        self._serial_port.reset_input_buffer()
                        continue

                outstanding = max(0, outstanding - 1)
                self._poll_stats.samples += 1
                try:
                    value = self._value_from_payload(payload)
                except ValueError as exc:
                    self._poll_stats.parse_errors += 1
                    self._last_error = str(exc)
                    continue

                sample_s = (now_ns - self._start_ns) / NSEC_PER_SEC
                self._latest_sample_s = sample_s
                self._latest_value = value
                self._last_error = None
                pending_timestamps.append(sample_s)
                pending_values.append(value)
                if len(pending_timestamps) >= PLOT_SAMPLE_BATCH or now_ns >= next_batch_ns:
                    flush_pending()
                    next_batch_ns = now_ns + PLOT_SAMPLE_BATCH_NS
        finally:
            flush_pending()
            self._done.set()


class FastImuPlotPoller:
    """Poll and decode the 9 channels used by the IMU plot grid."""

    def __init__(
        self,
        serial_port: PollIo,
        *,
        request: bytes,
        mask: int,
        packet_timeout: float,
        duration: float,
        max_samples: int,
        pipeline_depth: int,
        pending_samples: int,
        command_lock: Any | None = None,
    ) -> None:
        missing_fields = [
            field_name
            for field_name in IMU_PLOT_FIELDS
            if field_value_index(mask, field_name) is None
        ]
        if missing_fields:
            fields = ", ".join(missing_fields)
            raise ValueError(f"mask 0x{mask:04x} does not include plot fields: {fields}")

        self._serial_port = serial_port
        self._request = request
        self._mask = mask
        self._field_indexes = tuple(
            cast(int, field_value_index(mask, field_name))
            for field_name in IMU_PLOT_FIELDS
        )
        self._packet_timeout = packet_timeout
        self._duration = duration
        self._max_samples = max_samples
        self._pipeline_depth = pipeline_depth
        self._command_lock = command_lock or threading.RLock()
        self._samples = ImuSampleBuffer(pending_samples)
        self._reader_stats = ReaderStats()
        self._poll_stats = PollStats()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="vesc-fast-imu-grid-poller",
            daemon=True,
        )
        self._start_ns = 0
        self._latest_sample_s: float | None = None
        self._latest_values: tuple[float, ...] | None = None
        self._last_error: str | None = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def drain(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        return self._samples.drain()

    def snapshot(self) -> FastPlotSnapshot:
        now_ns = time.perf_counter_ns()
        elapsed_s = (
            (now_ns - self._start_ns) / NSEC_PER_SEC
            if self._start_ns > 0
            else 0.0
        )
        average_rate = (
            self._poll_stats.samples / elapsed_s
            if elapsed_s > 0.0
            else 0.0
        )
        latest_value = None if self._latest_values is None else self._latest_values[ACC_Z_INDEX]
        return FastPlotSnapshot(
            samples=self._poll_stats.samples,
            requests=self._poll_stats.requests,
            timeouts=self._poll_stats.timeouts,
            parse_errors=self._poll_stats.parse_errors,
            bad_crc=self._reader_stats.bad_crc,
            bad_stop=self._reader_stats.bad_stop,
            discarded_bytes=self._reader_stats.discarded_bytes,
            unexpected_packets=self._reader_stats.unexpected_packets,
            elapsed_s=elapsed_s,
            average_rate_hz=average_rate,
            latest_sample_s=self._latest_sample_s,
            latest_value=latest_value,
            last_error=self._last_error,
            done=self.done,
        )

    def _values_from_payload(self, payload: bytes) -> tuple[float, ...]:
        parsed = parse_imu_payload(payload)
        if parsed.mask != self._mask:
            field_indexes_list = []
            for field_name in IMU_PLOT_FIELDS:
                value_index = field_value_index(parsed.mask, field_name)
                if value_index is None:
                    raise ValueError(
                        f"IMU response mask 0x{parsed.mask:04x} does not include "
                        f"{field_name}"
                    )
                field_indexes_list.append(value_index)
            field_indexes = tuple(field_indexes_list)
        else:
            field_indexes = self._field_indexes

        rad2deg = 180.0 / math.pi
        raw_values = parsed.values
        return (
            raw_values[field_indexes[ROLL_INDEX]] * rad2deg,
            raw_values[field_indexes[PITCH_INDEX]] * rad2deg,
            raw_values[field_indexes[YAW_INDEX]] * rad2deg,
            raw_values[field_indexes[ACC_X_INDEX]],
            raw_values[field_indexes[ACC_Y_INDEX]],
            raw_values[field_indexes[ACC_Z_INDEX]],
            raw_values[field_indexes[GYRO_X_INDEX]],
            raw_values[field_indexes[GYRO_Y_INDEX]],
            raw_values[field_indexes[GYRO_Z_INDEX]],
        )

    def _run(self) -> None:
        self._start_ns = time.perf_counter_ns()
        outstanding = 0
        pending_timestamps: list[float] = []
        pending_values: list[tuple[float, ...]] = []
        next_batch_ns = self._start_ns + PLOT_SAMPLE_BATCH_NS

        def flush_pending() -> None:
            if not pending_timestamps:
                return
            self._samples.append_many(pending_timestamps, pending_values)
            pending_timestamps.clear()
            pending_values.clear()

        try:
            while not self._stop.is_set() and not should_stop(
                self._start_ns,
                self._poll_stats.samples,
                duration=self._duration,
                max_samples=self._max_samples,
            ):
                with self._command_lock:
                    while outstanding < self._pipeline_depth and not self._stop.is_set():
                        self._serial_port.write(self._request)
                        self._poll_stats.requests += 1
                        outstanding += 1

                    payload = read_expected_imu_packet(
                        self._serial_port,
                        self._packet_timeout,
                        self._reader_stats,
                    )
                    # This is a host receive-completion timestamp, not an IMU
                    # acquisition timestamp.
                    now_ns = time.perf_counter_ns()
                    if payload is None:
                        self._poll_stats.timeouts += 1
                        outstanding = 0
                        self._serial_port.reset_input_buffer()
                        continue

                outstanding = max(0, outstanding - 1)
                self._poll_stats.samples += 1
                try:
                    values = self._values_from_payload(payload)
                except ValueError as exc:
                    self._poll_stats.parse_errors += 1
                    self._last_error = str(exc)
                    continue

                sample_s = (now_ns - self._start_ns) / NSEC_PER_SEC
                self._latest_sample_s = sample_s
                self._latest_values = values
                self._last_error = None
                pending_timestamps.append(sample_s)
                pending_values.append(values)
                if len(pending_timestamps) >= PLOT_SAMPLE_BATCH or now_ns >= next_batch_ns:
                    flush_pending()
                    next_batch_ns = now_ns + PLOT_SAMPLE_BATCH_NS
        finally:
            flush_pending()
            self._done.set()


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
            "PyQtGraph plotting requires pyqtgraph and a Qt binding. "
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


def imu_general_param_names(schema: ConfigSchema) -> list[str]:
    """Return the same IMU/general app-config rows used by PageAppImu."""
    for group in schema.groups:
        if group.name.casefold() != "imu":
            continue
        for subgroup in group.subgroups:
            if subgroup.name.casefold() == "general":
                return list(subgroup.items)

    return [name for name in schema.ser_order if name.startswith("imu_conf.")]


def _set_item_read_only(item: Any, qt_core: Any) -> None:
    item_flag = getattr(qt_core.Qt, "ItemFlag", qt_core.Qt)
    item.setFlags(item.flags() & ~item_flag.ItemIsEditable)


def _enum_label(name: str) -> str:
    for prefix in (
        "IMU_TYPE_",
        "IMU_FILTER_",
        "IMU_MODE_",
        "AHRS_MODE_",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :].replace("_", " ").title()
    return name.replace("_", " ").title()


def _object_as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _object_as_int(value: object, default: int = 0) -> int:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _editor_scale(param: ConfigParam) -> float:
    return param.editor_scale if param.editor_scale != 0.0 else 1.0


class ImuAppConfigPanel:
    """Qt table of live IMU APPCONF editors matching the PageAppImu subgroup."""

    def __init__(
        self,
        qt_core: Any,
        qt_widgets: Any,
        client: PlotAppConfigClient,
        *,
        title_color: str,
        status_color: str,
    ) -> None:
        self._qt_core = qt_core
        self._qt_widgets = qt_widgets
        self._client = client
        self._schema = client.schema
        self._config: dict[str, object] = {}
        self._live_config: dict[str, object] = {}
        self._controls: dict[str, Any] = {}
        self._loading = False

        self.widget = qt_widgets.QGroupBox("IMU App Config")
        self.widget.setMinimumWidth(390)
        self.widget.setMaximumWidth(520)
        self.widget.setStyleSheet(f"QGroupBox {{ color: {title_color}; font-weight: 700; }}")
        layout = qt_widgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        firmware = qt_widgets.QLabel(client.firmware_label)
        firmware.setStyleSheet(f"color: {status_color}; font-weight: 400;")
        firmware.setWordWrap(True)
        layout.addWidget(firmware)

        self._table = qt_widgets.QTableWidget()
        self._setup_table()
        layout.addWidget(self._table, stretch=1)

        button_row = qt_widgets.QHBoxLayout()
        self._reload_button = qt_widgets.QPushButton("Reload")
        self._store_button = qt_widgets.QPushButton("Store")
        button_row.addWidget(self._reload_button)
        button_row.addWidget(self._store_button)
        layout.addLayout(button_row)

        self._status = qt_widgets.QLabel("Loading app config...")
        self._status.setStyleSheet(f"color: {status_color}; font-weight: 400;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._apply_timer = qt_core.QTimer()
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(APPCONF_LIVE_APPLY_DELAY_MS)
        self._apply_timer.timeout.connect(self._apply_live)
        self._reload_button.clicked.connect(self.reload)
        self._store_button.clicked.connect(self.store)

        self.reload()

    def _setup_table(self) -> None:
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(("Name", "Edit"))
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setVisible(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)

        abstract_item_view = self._qt_widgets.QAbstractItemView
        selection_behavior = getattr(
            getattr(abstract_item_view, "SelectionBehavior", abstract_item_view),
            "SelectRows",
        )
        selection_mode = getattr(
            getattr(abstract_item_view, "SelectionMode", abstract_item_view),
            "NoSelection",
        )
        self._table.setSelectionBehavior(selection_behavior)
        self._table.setSelectionMode(selection_mode)

    def reload(self) -> None:
        try:
            config = self._client.get_appconf()
        except Exception as exc:  # noqa: BLE001 - user-visible hardware error path.
            self._set_status(f"Reload failed: {exc}")
            return

        self._loading = True
        try:
            self._config = config
            self._live_config = dict(config)
            self._controls.clear()
            self._table.setRowCount(0)
            for name in imu_general_param_names(self._schema):
                if name.startswith("::sep::"):
                    self._add_separator(name[7:])
                    continue
                param = self._schema.params.get(name)
                if param is not None and param.type != CfgType.UNDEFINED:
                    self._add_param_row(name, param, config.get(name))
            self._table.resizeColumnToContents(0)
            self._table.resizeRowsToContents()
        finally:
            self._loading = False

        self._set_status("Live edits apply temporarily; Store writes to flash.")

    def store(self) -> None:
        self._apply_timer.stop()
        try:
            self._client.set_appconf(self._config, store=True, wait_ack=True)
        except Exception as exc:  # noqa: BLE001 - user-visible hardware error path.
            self._set_status(f"Store failed: {exc}")
            return
        self._live_config = dict(self._config)
        self._set_status("Stored app config.")

    def _add_separator(self, text: str) -> None:
        row = self._table.rowCount()
        self._table.setRowCount(row + 1)
        label = self._qt_widgets.QLabel(text)
        alignment = getattr(self._qt_core.Qt, "AlignmentFlag", self._qt_core.Qt)
        label.setAlignment(alignment.AlignCenter)
        label.setStyleSheet("font-weight: 700; padding: 3px;")
        self._table.setCellWidget(row, 0, label)
        self._table.setSpan(row, 0, 1, 2)

    def _add_param_row(self, name: str, param: ConfigParam, value: object) -> None:
        row = self._table.rowCount()
        self._table.setRowCount(row + 1)

        label = self._qt_widgets.QTableWidgetItem(param.long_name or name)
        if param.description_text:
            label.setToolTip(param.description_text)
        _set_item_read_only(label, self._qt_core)
        self._table.setItem(row, 0, label)

        editor = self._make_editor(name, param, value)
        if param.description_text:
            editor.setToolTip(param.description_text)
        self._controls[name] = editor
        self._table.setCellWidget(row, 1, editor)

    def _make_editor(self, name: str, param: ConfigParam, value: object) -> Any:
        if param.type == CfgType.DOUBLE:
            editor = self._qt_widgets.QDoubleSpinBox()
            editor.setKeyboardTracking(False)
            editor.setDecimals(max(0, min(param.decimals_double, 9)))
            editor.setRange(
                self._display_double(param, param.min_double),
                self._display_double(param, param.max_double),
            )
            editor.setSingleStep(param.step_double if param.step_double > 0.0 else 0.1)
            if param.suffix:
                editor.setSuffix(param.suffix)
            editor.setValue(self._display_double(param, _object_as_float(value)))
            editor.valueChanged.connect(
                lambda _value, n=name, p=param, w=editor: self._control_changed(
                    n,
                    self._double_config_value(p, float(w.value())),
                )
            )
            return editor

        if param.type in (CfgType.INT, CfgType.BITFIELD):
            editor = self._qt_widgets.QSpinBox()
            editor.setKeyboardTracking(False)
            editor.setRange(
                self._display_int(param, param.min_int),
                self._display_int(param, param.max_int),
            )
            editor.setSingleStep(param.step_int if param.step_int > 0 else 1)
            if param.suffix:
                editor.setSuffix(param.suffix)
            editor.setValue(self._display_int(param, _object_as_int(value)))
            editor.valueChanged.connect(
                lambda _value, n=name, p=param, w=editor: self._control_changed(
                    n,
                    self._int_config_value(p, int(w.value())),
                )
            )
            return editor

        if param.type == CfgType.ENUM:
            if name in APPCONF_NON_LIVE_PARAMS:
                return self._make_radio_enum_editor(name, param, value)

            editor = self._qt_widgets.QComboBox()
            editor.addItems([_enum_label(item) for item in param.enum_names])
            current = _object_as_int(value)
            if 0 <= current < len(param.enum_names):
                editor.setCurrentIndex(current)
            editor.activated.connect(
                lambda _index, n=name, w=editor: self._control_changed(
                    n,
                    int(w.currentIndex()),
                )
            )
            return editor

        if param.type == CfgType.BOOL:
            editor = self._qt_widgets.QCheckBox()
            editor.setChecked(bool(_object_as_int(value)))
            editor.stateChanged.connect(
                lambda _state, n=name, w=editor: self._control_changed(
                    n,
                    1 if w.isChecked() else 0,
                )
            )
            return editor

        editor = self._qt_widgets.QLineEdit(str(value or ""))
        if param.max_len > 0:
            editor.setMaxLength(param.max_len)
        editor.editingFinished.connect(
            lambda n=name, w=editor: self._control_changed(n, w.text())
        )
        return editor

    def _make_radio_enum_editor(
        self,
        name: str,
        param: ConfigParam,
        value: object,
    ) -> Any:
        editor = self._qt_widgets.QWidget()
        layout = self._qt_widgets.QHBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        current = _object_as_int(value)
        for index, enum_name in enumerate(param.enum_names):
            button = self._qt_widgets.QRadioButton(_enum_label(enum_name))
            button.setChecked(index == current)
            button.toggled.connect(
                lambda checked, n=name, i=index: (
                    self._control_changed(n, i) if checked else None
                )
            )
            layout.addWidget(button)

        layout.addStretch(1)
        return editor

    def _control_changed(self, name: str, value: object) -> None:
        try:
            if self._loading:
                return
            self._config[name] = value
            if name in APPCONF_NON_LIVE_PARAMS:
                self._set_status(f"{name} changed locally; live apply skipped.")
                return
            self._live_config[name] = value
            self._set_status(f"Pending live apply: {name}")
            self._apply_timer.start()
        except Exception as exc:  # noqa: BLE001 - keep Qt signal exceptions local.
            self._set_status(f"Edit failed for {name}: {exc}")

    def _apply_live(self) -> None:
        try:
            self._client.set_appconf(self._live_config, store=False, wait_ack=False)
        except Exception as exc:  # noqa: BLE001 - user-visible hardware error path.
            self._set_status(f"Live apply failed: {exc}")
            return
        self._set_status("Applied temporary app config.")

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    @staticmethod
    def _display_double(param: ConfigParam, value: float) -> float:
        if param.edit_as_percentage:
            max_value = max(abs(param.max_double), abs(param.min_double), 1e-12)
            return (value * 100.0) / max_value
        return value * _editor_scale(param)

    @staticmethod
    def _double_config_value(param: ConfigParam, display_value: float) -> float:
        if param.edit_as_percentage:
            max_value = max(abs(param.max_double), abs(param.min_double), 1e-12)
            return (display_value / 100.0) * max_value
        return display_value / _editor_scale(param)

    @staticmethod
    def _display_int(param: ConfigParam, value: int) -> int:
        if param.edit_as_percentage:
            max_value = max(abs(param.max_int), abs(param.min_int), 1)
            return round((value * 100) / max_value)
        return round(value * _editor_scale(param))

    @staticmethod
    def _int_config_value(param: ConfigParam, display_value: int) -> int:
        if param.edit_as_percentage:
            max_value = max(abs(param.max_int), abs(param.min_int), 1)
            return round((display_value * max_value) / 100)
        return round(display_value / _editor_scale(param))


def make_app_config_error_panel(
    qt_widgets: Any,
    *,
    error: str,
    title_color: str,
    status_color: str,
) -> Any:
    group = qt_widgets.QGroupBox("IMU App Config")
    group.setMinimumWidth(390)
    group.setMaximumWidth(520)
    group.setStyleSheet(f"QGroupBox {{ color: {title_color}; font-weight: 700; }}")
    layout = qt_widgets.QVBoxLayout(group)
    label = qt_widgets.QLabel(f"App config controls unavailable: {error}")
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {status_color}; font-weight: 400;")
    layout.addWidget(label)
    layout.addStretch(1)
    return group


def run_fast_imu_grid_plot(
    poller: FastImuPlotPoller,
    *,
    history: int,
    max_points: int,
    plot_rate: float,
    fft_window: float,
    fft_rate: float,
    theme: str,
    antialias: bool,
    app_config_client: PlotAppConfigClient | None = None,
    app_config_error: str | None = None,
) -> None:
    """Run the fast direct-USB 2x3 IMU plot grid."""
    selected_theme = PLOT_THEMES[theme]
    np_module = import_numpy()
    pg, QtCore, QtWidgets = import_pyqtgraph()
    pg.setConfigOptions(
        antialias=antialias,
        background=selected_theme.pg_background,
        foreground=selected_theme.pg_foreground,
    )

    require_qt_platform_runtime()
    app = pg.mkQApp("VESC Fast IMU Data")
    window = QtWidgets.QWidget()
    window.setWindowTitle("VESC Fast IMU Data")
    window.resize(1400, 900)
    window.setStyleSheet(f"background-color: {selected_theme.window_background};")

    qt_alignment = getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt)
    root_layout = QtWidgets.QHBoxLayout(window)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(8)
    if app_config_client is not None:
        config_panel = ImuAppConfigPanel(
            QtCore,
            QtWidgets,
            app_config_client,
            title_color=selected_theme.title_color,
            status_color=selected_theme.status_color,
        )
        root_layout.addWidget(config_panel.widget, stretch=0)
    elif app_config_error is not None:
        root_layout.addWidget(
            make_app_config_error_panel(
                QtWidgets,
                error=app_config_error,
                title_color=selected_theme.title_color,
                status_color=selected_theme.status_color,
            ),
            stretch=0,
        )

    plot_area = QtWidgets.QWidget()
    root_layout.addWidget(plot_area, stretch=1)

    title = QtWidgets.QLabel("VESC Fast IMU Data")
    title.setAlignment(qt_alignment.AlignCenter)
    title.setStyleSheet(
        f"font-size: 14pt; font-weight: 700; color: {selected_theme.title_color};"
    )
    status = QtWidgets.QLabel("Waiting for IMU data...")
    status.setStyleSheet(f"color: {selected_theme.status_color};")

    layout = QtWidgets.QGridLayout(plot_area)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(title, 0, 0, 1, 2)
    layout.addWidget(status, 4, 0, 1, 2)
    for row in (1, 2, 3):
        layout.setRowStretch(row, 1)
    for col in (0, 1):
        layout.setColumnStretch(col, 1)

    qt_size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)

    def make_plot(
        row: int,
        col: int,
        title_text: str,
        y_label: str,
        *,
        x_label: str | None = None,
        y_range: tuple[float, float] | None = None,
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
        if y_range is not None:
            plot.setYRange(*y_range, padding=0.0)
        for method_name, args in (
            ("setClipToView", (True,)),
            ("setDownsampling", (1, True, "peak")),
        ):
            method = getattr(plot, method_name, None)
            if method is not None:
                method(*args)
        return plot

    def add_lines(plot: Any, names: tuple[str, str, str], width: float) -> tuple[Any, Any, Any]:
        empty = np_module.empty(0, dtype=np_module.float64)
        lines = []
        for name, color in zip(names, selected_theme.line_colors):
            line = pg.PlotCurveItem(
                empty,
                empty,
                pen=pg.mkPen(color, width=width),
                name=name,
                connect="all",
                skipFiniteCheck=True,
            )
            line.setSkipFiniteCheck(True)
            plot.addItem(line)
            lines.append(line)
        return cast(tuple[Any, Any, Any], tuple(lines))

    ax_acc = make_plot(1, 0, "Accel Data", "g", y_range=(-8, 8))
    ax_gyro = make_plot(2, 0, "Gyro Data", "Gyro", y_range=(-2000, 2000))
    ax_rpy = make_plot(
        3,
        0,
        "RPY Data",
        "Degrees",
        x_label="Host RX time (s)",
        y_range=(-200, 200),
    )
    acc_lines = add_lines(ax_acc, ("Acc X", "Acc Y", "Acc Z"), 1.5)
    gyro_lines = add_lines(ax_gyro, ("Gyro X", "Gyro Y", "Gyro Z"), 1.5)
    rpy_lines = add_lines(ax_rpy, ("Roll", "Pitch", "Yaw"), 1.5)

    ax_acc_freq = make_plot(1, 1, "Accel Data PSD", "PSD (g^2/Hz)")
    ax_gyro_freq = make_plot(2, 1, "Gyro Data PSD", "PSD ((deg/s)^2/Hz)")
    ax_rpy_freq = make_plot(
        3,
        1,
        "RPY Data PSD",
        "PSD (deg^2/Hz)",
        x_label="Frequency (Hz)",
    )
    acc_freq_lines = add_lines(ax_acc_freq, ("Acc X", "Acc Y", "Acc Z"), 1.2)
    gyro_freq_lines = add_lines(ax_gyro_freq, ("Gyro X", "Gyro Y", "Gyro Z"), 1.2)
    rpy_freq_lines = add_lines(ax_rpy_freq, ("Roll", "Pitch", "Yaw"), 1.2)
    acc_freq_range = FrequencyAxisRange()
    gyro_freq_range = FrequencyAxisRange()
    rpy_freq_range = FrequencyAxisRange()

    plot_history = ImuPlotHistory(history)
    refresh_timestamps: deque[float] = deque(maxlen=120)
    dropped_pending = 0
    latest_nyquist_hz: float | None = None
    latest_psd_samples = 0

    def decimation_indexes(sample_count: int) -> npt.NDArray[np.intp] | None:
        if sample_count <= max_points:
            return None
        return cast(
            "npt.NDArray[np.intp]",
            np_module.linspace(0, sample_count - 1, max_points, dtype=np_module.intp),
        )

    def set_time_lines(
        lines: tuple[Any, Any, Any],
        channel_indexes: tuple[int, int, int],
        x_values: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> None:
        for line, channel_index in zip(lines, channel_indexes):
            line.setData(
                x=x_values,
                y=values[channel_index],
                connect="all",
                skipFiniteCheck=True,
            )

    def update_x_range(axis: Any, x_values: npt.NDArray[np.float64]) -> None:
        if x_values.size >= 2:
            axis.setXRange(float(x_values[0]), float(x_values[-1]), padding=0.0)

    def actual_plot_hz() -> float | None:
        if len(refresh_timestamps) < 2:
            return None
        elapsed = refresh_timestamps[-1] - refresh_timestamps[0]
        if elapsed <= 0.0:
            return None
        return (len(refresh_timestamps) - 1) / elapsed

    def refresh_plot() -> None:
        nonlocal dropped_pending

        timestamps, values, dropped = poller.drain()
        dropped_pending += dropped
        if timestamps.size == 0:
            return

        refresh_timestamps.append(time.monotonic())
        plot_history.append_samples(timestamps, values)
        x_values = plot_history.valid_timestamps()
        plot_values = plot_history.valid_values()
        indexes = decimation_indexes(int(x_values.size))
        if indexes is not None:
            x_values = x_values[indexes]
            plot_values = plot_values[:, indexes]

        set_time_lines(acc_lines, (ACC_X_INDEX, ACC_Y_INDEX, ACC_Z_INDEX), x_values, plot_values)
        set_time_lines(
            gyro_lines,
            (GYRO_X_INDEX, GYRO_Y_INDEX, GYRO_Z_INDEX),
            x_values,
            plot_values,
        )
        set_time_lines(rpy_lines, (ROLL_INDEX, PITCH_INDEX, YAW_INDEX), x_values, plot_values)
        update_x_range(ax_acc, x_values)
        update_x_range(ax_gyro, x_values)
        update_x_range(ax_rpy, x_values)

    def update_frequency_axis(
        axis: Any,
        lines: tuple[Any, Any, Any],
        psd_values: npt.NDArray[np.float64],
        channel_indexes: tuple[int, int, int],
        frequencies: npt.NDArray[np.float64],
        axis_range: FrequencyAxisRange,
    ) -> None:
        max_density = 0.0
        for line, channel_index in zip(lines, channel_indexes):
            channel_psd = psd_values[channel_index]
            if channel_psd.size > 0:
                max_density = max(max_density, float(np_module.max(channel_psd)))
            line.setData(
                x=frequencies,
                y=channel_psd,
                connect="all",
                skipFiniteCheck=True,
            )

        next_x_max = max(float(frequencies[-1]), 1.0)
        if not math.isclose(next_x_max, axis_range.x_max, rel_tol=0.01, abs_tol=0.01):
            axis.setXRange(0.0, next_x_max, padding=0.0)
            axis_range.x_max = next_x_max

        next_y_max = max(max_density * 1.1, 1e-12)
        if (
            axis_range.y_max == 0.0
            or next_y_max > axis_range.y_max
            or next_y_max < axis_range.y_max * 0.5
        ):
            axis.setYRange(0.0, next_y_max, padding=0.0)
            axis_range.y_max = next_y_max

    def refresh_spectrum() -> None:
        nonlocal latest_nyquist_hz, latest_psd_samples

        spectrum = imu_frequency_psd(
            plot_history.valid_timestamps(),
            plot_history.valid_values(),
            window_s=fft_window,
        )
        if spectrum is None:
            latest_nyquist_hz = None
            latest_psd_samples = 0
            return

        frequencies, psd_values, nyquist_hz, sample_count = spectrum
        latest_nyquist_hz = nyquist_hz
        latest_psd_samples = sample_count
        update_frequency_axis(
            ax_acc_freq,
            acc_freq_lines,
            psd_values,
            (ACC_X_INDEX, ACC_Y_INDEX, ACC_Z_INDEX),
            frequencies,
            acc_freq_range,
        )
        update_frequency_axis(
            ax_gyro_freq,
            gyro_freq_lines,
            psd_values,
            (GYRO_X_INDEX, GYRO_Y_INDEX, GYRO_Z_INDEX),
            frequencies,
            gyro_freq_range,
        )
        update_frequency_axis(
            ax_rpy_freq,
            rpy_freq_lines,
            psd_values,
            (ROLL_INDEX, PITCH_INDEX, YAW_INDEX),
            frequencies,
            rpy_freq_range,
        )

    def refresh_status() -> None:
        snapshot = poller.snapshot()
        sample_hz = plot_history.sample_hz()
        sample_text = "measuring" if sample_hz is None else f"{sample_hz:.1f} Hz"
        plot_hz = actual_plot_hz()
        plot_text = "measuring" if plot_hz is None else f"{plot_hz:.1f} Hz"
        psd_text = (
            "host-timed PSD: measuring"
            if latest_nyquist_hz is None
            else (
                f"host-timed PSD: {latest_psd_samples} samples/{fft_window:g}s, "
                f"Nyquist {latest_nyquist_hz:.1f} Hz"
            )
        )
        error_text = (
            f" | error: {snapshot.last_error}"
            if snapshot.last_error is not None
            else ""
        )
        state = "stopped" if snapshot.done else "running"
        status.setText(
            f"{state} | samples: {snapshot.samples} | "
            f"poll avg: {snapshot.average_rate_hz:.1f} Hz | "
            f"host-dt window: {sample_text} | "
            f"plot: {plot_text} | {psd_text} | timeouts: {snapshot.timeouts} | "
            f"parse: {snapshot.parse_errors} | bad_crc: {snapshot.bad_crc} | "
            f"dropped_ui: {dropped_pending}{error_text}"
        )

    plot_timer = QtCore.QTimer()
    plot_timer.setInterval(round(1000.0 / plot_rate))
    plot_timer.timeout.connect(refresh_plot)

    psd_timer = QtCore.QTimer()
    psd_timer.setInterval(round(1000.0 / fft_rate))
    psd_timer.timeout.connect(refresh_spectrum)

    status_timer = QtCore.QTimer()
    status_timer.setInterval(round(1000.0 / DEFAULT_STATUS_INTERVAL))
    status_timer.timeout.connect(refresh_status)

    def stop_updates(*_args: object) -> None:
        plot_timer.stop()
        psd_timer.stop()
        status_timer.stop()

    window.destroyed.connect(stop_updates)
    refresh_plot()
    refresh_spectrum()
    refresh_status()
    window.show()
    poller.start()
    plot_timer.start()
    psd_timer.start()
    status_timer.start()

    exec_app = getattr(app, "exec", None)
    if exec_app is None:
        exec_app = app.exec_
    try:
        exec_app()
    finally:
        poller.stop()


def run_accel_axis_plot(
    poller: FastImuAxisPoller,
    *,
    axis_name: str,
    history: int,
    max_points: int,
    plot_rate: float,
    fft_window: float,
    fft_rate: float,
    theme: str,
    antialias: bool,
) -> None:
    """Run a batched single-axis PyQtGraph plot."""
    selected_theme = PLOT_THEMES[theme]
    np_module = import_numpy()
    pg, QtCore, QtWidgets = import_pyqtgraph()
    pg.setConfigOptions(
        antialias=antialias,
        background=selected_theme.pg_background,
        foreground=selected_theme.pg_foreground,
    )

    require_qt_platform_runtime()
    app = pg.mkQApp("VESC Fast IMU Axis Plot")
    window = QtWidgets.QWidget()
    window.setWindowTitle("VESC Fast IMU Axis Plot")
    window.resize(1400, 620)
    window.setStyleSheet(f"background-color: {selected_theme.window_background};")

    qt_alignment = getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt)
    title = QtWidgets.QLabel(f"VESC {axis_name.upper()} Fast Plot")
    title.setAlignment(qt_alignment.AlignCenter)
    title.setStyleSheet(
        f"font-size: 14pt; font-weight: 700; color: {selected_theme.title_color};"
    )
    status = QtWidgets.QLabel("Waiting for IMU data...")
    status.setStyleSheet(f"color: {selected_theme.status_color};")

    layout = QtWidgets.QVBoxLayout(window)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    layout.addWidget(title)

    plot_row = QtWidgets.QHBoxLayout()
    plot_row.setContentsMargins(0, 0, 0, 0)
    plot_row.setSpacing(8)
    layout.addLayout(plot_row, stretch=1)
    layout.addWidget(status)

    plot_widget = pg.PlotWidget(title="Time Series")
    spectrum_widget = pg.PlotWidget(title="Frequency Analysis")
    plot_row.addWidget(plot_widget, stretch=1)
    plot_row.addWidget(spectrum_widget, stretch=1)

    plot = plot_widget.getPlotItem()
    plot.showGrid(x=True, y=True, alpha=selected_theme.grid_alpha)
    plot.setLabel("left", axis_name, units="g")
    plot.setLabel("bottom", "host RX time", units="s")
    plot.setYRange(-8.0, 8.0, padding=0.0)
    plot.enableAutoRange(axis="y", enable=False)
    for method_name, args in (
        ("setClipToView", (True,)),
        ("setDownsampling", (1, True, "peak")),
    ):
        method = getattr(plot, method_name, None)
        if method is not None:
            method(*args)

    line = pg.PlotCurveItem(
        np_module.empty(0, dtype=np_module.float64),
        np_module.empty(0, dtype=np_module.float64),
        pen=pg.mkPen(selected_theme.line_color, width=1.5),
        name=axis_name,
        connect="all",
        skipFiniteCheck=True,
    )
    line.setSkipFiniteCheck(True)
    plot.addItem(line)

    spectrum_plot = spectrum_widget.getPlotItem()
    spectrum_plot.showGrid(x=True, y=True, alpha=selected_theme.grid_alpha)
    spectrum_plot.setLabel("left", "PSD", units="g^2/Hz")
    spectrum_plot.setLabel("bottom", "frequency", units="Hz")
    spectrum_plot.setXRange(0.0, 1.0, padding=0.0)
    spectrum_plot.setYRange(0.0, 1e-12, padding=0.0)
    spectrum_line = pg.PlotCurveItem(
        np_module.empty(0, dtype=np_module.float64),
        np_module.empty(0, dtype=np_module.float64),
        pen=pg.mkPen(selected_theme.line_color, width=1.2),
        name=f"{axis_name} PSD",
        connect="all",
        skipFiniteCheck=True,
    )
    spectrum_line.setSkipFiniteCheck(True)
    spectrum_plot.addItem(spectrum_line)

    plot_history = AxisPlotHistory(history)
    refresh_timestamps: deque[float] = deque(maxlen=120)
    dropped_pending = 0
    latest_nyquist_hz: float | None = None
    latest_psd_samples = 0

    def actual_plot_hz() -> float | None:
        if len(refresh_timestamps) < 2:
            return None
        elapsed = refresh_timestamps[-1] - refresh_timestamps[0]
        if elapsed <= 0.0:
            return None
        return (len(refresh_timestamps) - 1) / elapsed

    def refresh_plot() -> None:
        nonlocal dropped_pending

        timestamps, values, dropped = poller.drain()
        dropped_pending += dropped
        if timestamps.size == 0:
            return

        refresh_timestamps.append(time.monotonic())
        plot_history.append_samples(timestamps, values)
        x_values = plot_history.valid_timestamps()
        y_values = plot_history.valid_values()
        if x_values.size > max_points:
            index_values = np_module.linspace(
                0,
                x_values.size - 1,
                max_points,
                dtype=np_module.intp,
            )
            plot_x_values = x_values[index_values]
            plot_y_values = y_values[index_values]
        else:
            plot_x_values = x_values
            plot_y_values = y_values
        line.setData(
            x=plot_x_values,
            y=plot_y_values,
            connect="all",
            skipFiniteCheck=True,
        )
        if x_values.size >= 2:
            plot.setXRange(float(x_values[0]), float(x_values[-1]), padding=0.0)

    def refresh_spectrum() -> None:
        nonlocal latest_nyquist_hz, latest_psd_samples

        spectrum = axis_frequency_psd(
            plot_history.valid_timestamps(),
            plot_history.valid_values(),
            window_s=fft_window,
        )
        if spectrum is None:
            latest_nyquist_hz = None
            latest_psd_samples = 0
            return

        frequencies, psd_values, nyquist_hz, sample_count = spectrum
        latest_nyquist_hz = nyquist_hz
        latest_psd_samples = sample_count
        spectrum_line.setData(
            x=frequencies,
            y=psd_values,
            connect="all",
            skipFiniteCheck=True,
        )

        if frequencies.size > 0:
            spectrum_plot.setXRange(0.0, float(frequencies[-1]), padding=0.0)
        max_density = float(np_module.max(psd_values)) if psd_values.size else 0.0
        spectrum_plot.setYRange(0.0, max(max_density * 1.1, 1e-12), padding=0.0)

    def refresh_status() -> None:
        snapshot = poller.snapshot()
        latest = (
            "n/a"
            if snapshot.latest_value is None
            else f"{snapshot.latest_value:.6g} g"
        )
        sample_hz = plot_history.sample_hz()
        sample_text = "measuring" if sample_hz is None else f"{sample_hz:.1f} Hz"
        plot_hz = actual_plot_hz()
        plot_text = "measuring" if plot_hz is None else f"{plot_hz:.1f} Hz"
        psd_text = (
            "host-timed PSD: measuring"
            if latest_nyquist_hz is None
            else (
                f"host-timed PSD: {latest_psd_samples} samples/{fft_window:g}s, "
                f"Nyquist {latest_nyquist_hz:.1f} Hz"
            )
        )
        error_text = (
            f" | error: {snapshot.last_error}"
            if snapshot.last_error is not None
            else ""
        )
        state = "stopped" if snapshot.done else "running"
        status.setText(
            f"{state} | latest: {latest} | samples: {snapshot.samples} | "
            f"poll avg: {snapshot.average_rate_hz:.1f} Hz | "
            f"host-dt window: {sample_text} | "
            f"plot: {plot_text} | {psd_text} | timeouts: {snapshot.timeouts} | "
            f"parse: {snapshot.parse_errors} | bad_crc: {snapshot.bad_crc} | "
            f"dropped_ui: {dropped_pending}{error_text}"
        )

    plot_timer = QtCore.QTimer()
    plot_timer.setInterval(round(1000.0 / plot_rate))
    plot_timer.timeout.connect(refresh_plot)

    psd_timer = QtCore.QTimer()
    psd_timer.setInterval(round(1000.0 / fft_rate))
    psd_timer.timeout.connect(refresh_spectrum)

    status_timer = QtCore.QTimer()
    status_timer.setInterval(round(1000.0 / DEFAULT_STATUS_INTERVAL))
    status_timer.timeout.connect(refresh_status)

    def stop_updates(*_args: object) -> None:
        plot_timer.stop()
        psd_timer.stop()
        status_timer.stop()

    window.destroyed.connect(stop_updates)
    refresh_plot()
    refresh_spectrum()
    refresh_status()
    window.show()
    poller.start()
    plot_timer.start()
    psd_timer.start()
    status_timer.start()

    exec_app = getattr(app, "exec", None)
    if exec_app is None:
        exec_app = app.exec_
    try:
        exec_app()
    finally:
        poller.stop()


def poll_imu(
    serial_port: PollIo,
    *,
    request: bytes,
    mask: int,
    packet_timeout: float,
    status_interval: float,
    duration: float,
    max_samples: int,
    pipeline_depth: int,
    print_every: int,
    csv: bool,
    no_decode: bool,
    tui: TerminalImuDisplay | None,
) -> None:
    """Run the high-rate IMU polling loop."""
    reader_stats = ReaderStats()
    poll_stats = PollStats()
    start_ns = time.perf_counter_ns()
    previous_sample_ns: int | None = None
    next_status_ns = start_ns + round(status_interval * NSEC_PER_SEC)
    status_window_ns = start_ns
    status_window_samples = 0
    outstanding = 0
    latest: ParsedImu | None = None

    if csv and not no_decode:
        write_csv_header(mask)
    if tui is not None:
        tui.start()

    try:
        while not should_stop(
            start_ns,
            poll_stats.samples,
            duration=duration,
            max_samples=max_samples,
        ):
            while outstanding < pipeline_depth:
                serial_port.write(request)
                poll_stats.requests += 1
                outstanding += 1

            payload = read_expected_imu_packet(
                serial_port,
                packet_timeout,
                reader_stats,
            )
            # This is a host receive-completion timestamp, not an IMU
            # acquisition timestamp.
            now_ns = time.perf_counter_ns()
            if payload is None:
                poll_stats.timeouts += 1
                outstanding = 0
                serial_port.reset_input_buffer()
                continue

            outstanding = max(0, outstanding - 1)
            poll_stats.samples += 1
            status_window_samples += 1

            if not no_decode:
                try:
                    latest = parse_imu_payload(payload)
                except ValueError as exc:
                    poll_stats.parse_errors += 1
                    print(f"parse error: {exc}", file=sys.stderr)
                    continue

                if print_every > 0 and poll_stats.samples % print_every == 0:
                    write_sample(
                        poll_stats.samples,
                        now_ns,
                        start_ns,
                        previous_sample_ns,
                        latest,
                        csv=csv,
                    )

            previous_sample_ns = now_ns

            if tui is not None and latest is not None and tui.due(now_ns):
                elapsed = (now_ns - start_ns) / NSEC_PER_SEC
                window_elapsed = (now_ns - status_window_ns) / NSEC_PER_SEC
                current_rate = (
                    status_window_samples / window_elapsed
                    if window_elapsed > 0.0
                    else 0.0
                )
                average_rate = poll_stats.samples / elapsed if elapsed > 0.0 else 0.0
                tui.render(
                    now_ns=now_ns,
                    start_ns=start_ns,
                    sample_timestamp_ns=previous_sample_ns,
                    parsed=latest,
                    poll_stats=poll_stats,
                    reader_stats=reader_stats,
                    current_rate=current_rate,
                    average_rate=average_rate,
                )

                status_window_ns = now_ns
                status_window_samples = 0
            elif tui is None and status_interval > 0.0 and now_ns >= next_status_ns:
                elapsed = (now_ns - start_ns) / NSEC_PER_SEC
                window_elapsed = (now_ns - status_window_ns) / NSEC_PER_SEC
                window_rate = (
                    status_window_samples / window_elapsed
                    if window_elapsed > 0.0
                    else 0.0
                )
                total_rate = poll_stats.samples / elapsed if elapsed > 0.0 else 0.0
                status = (
                    f"{elapsed:8.2f}s  rate={window_rate:8.1f} Hz  "
                    f"avg={total_rate:8.1f} Hz  samples={poll_stats.samples}  "
                    f"requests={poll_stats.requests}  timeouts={poll_stats.timeouts}  "
                    f"bad_crc={reader_stats.bad_crc}"
                )
                if latest is not None:
                    status += "  " + format_values(latest.mask, latest.values)
                print(status, file=sys.stderr)

                status_window_ns = now_ns
                status_window_samples = 0
                next_status_ns = now_ns + round(status_interval * NSEC_PER_SEC)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    finally:
        if tui is not None:
            tui.close()
        elapsed = (time.perf_counter_ns() - start_ns) / NSEC_PER_SEC
        rate = poll_stats.samples / elapsed if elapsed > 0.0 else 0.0
        print(
            f"Done. samples={poll_stats.samples} requests={poll_stats.requests} "
            f"elapsed={elapsed:.3f}s avg_rate={rate:.1f}Hz "
            f"timeouts={poll_stats.timeouts} parse_errors={poll_stats.parse_errors} "
            f"discarded={reader_stats.discarded_bytes} bad_crc={reader_stats.bad_crc} "
            f"bad_stop={reader_stats.bad_stop} invalid_length={reader_stats.invalid_length} "
            f"unexpected={reader_stats.unexpected_packets}",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Poll VESC IMU data over USB serial or direct BLE with minimal host overhead.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List discovered serial ports and exit.",
    )
    parser.add_argument(
        "--scan-ble",
        action="store_true",
        help="Scan for VESC BLE devices advertising Nordic UART and exit.",
    )
    add_vesc_connection_arguments(parser)
    parser.add_argument(
        "--mask",
        type=parse_mask_arg,
        default=None,
        metavar="MASK",
        help="IMU field bitmask, e.g. 0x01ff. Mutually exclusive with --fields.",
    )
    parser.add_argument(
        "--fields",
        type=parse_fields_arg,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated fields/groups. Groups: default, rpy, acc, gyro, "
            "mag, quat, all. Default: default."
        ),
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=DEFAULT_PIPELINE_DEPTH,
        metavar="N",
        help=(
            "Number of outstanding IMU requests to keep in flight. Higher values "
            "can improve throughput over USB; 1 gives lowest request/response latency "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Stop after this many seconds. Default 0 runs until Ctrl-C.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        metavar="N",
        help="Stop after this many decoded samples. Default 0 runs until another limit.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL,
        metavar="SEC",
        help=(
            "Plain status print interval on stderr when the TUI is disabled. "
            "Use 0 to disable (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--ui-rate",
        type=float,
        default=DEFAULT_UI_RATE,
        metavar="HZ",
        help=(
            "Terminal UI refresh rate in Hz. Poll-rate measurement still uses every "
            "sample (default: 10)."
        ),
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the in-place terminal UI and use plain periodic status lines.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Open a PyQtGraph 2x3 UI with accel, gyro, and RPY time/PSD plots. "
            "If --mask/--fields is omitted, RPY + accelerometer + gyroscope are requested."
        ),
    )
    parser.add_argument(
        "--plot-axis",
        type=parse_accel_axis_arg,
        default="acc_x",
        metavar="AXIS",
        help="Legacy single-axis plot selector; ignored by the 2x3 plot grid.",
    )
    parser.add_argument(
        "--plot-history",
        type=int,
        default=DEFAULT_PLOT_HISTORY,
        metavar="N",
        help=f"Number of samples retained by the plot (default: {DEFAULT_PLOT_HISTORY}).",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=DEFAULT_PLOT_MAX_POINTS,
        metavar="N",
        help=(
            "Maximum points sent to PyQtGraph each redraw after display decimation "
            f"(default: {DEFAULT_PLOT_MAX_POINTS})."
        ),
    )
    parser.add_argument(
        "--plot-rate",
        type=float,
        default=DEFAULT_PLOT_RATE,
        metavar="HZ",
        help=f"Plot redraw rate in Hz (default: {DEFAULT_PLOT_RATE:g}).",
    )
    parser.add_argument(
        "--fft-window",
        type=float,
        default=DEFAULT_FFT_WINDOW,
        metavar="SEC",
        help=(
            "Seconds of raw samples in each PSD analysis window "
            f"(default: {DEFAULT_FFT_WINDOW:g})."
        ),
    )
    parser.add_argument(
        "--fft-rate",
        type=float,
        default=DEFAULT_FFT_RATE,
        metavar="HZ",
        help=f"PSD plot redraw rate in Hz (default: {DEFAULT_FFT_RATE:g}).",
    )
    parser.add_argument(
        "--plot-pending",
        type=int,
        default=DEFAULT_PLOT_PENDING,
        metavar="N",
        help=(
            "Pending sample ring size between the poll thread and UI "
            f"(default: {DEFAULT_PLOT_PENDING})."
        ),
    )
    parser.add_argument(
        "--theme",
        choices=tuple(PLOT_THEMES),
        default="light",
        help="Plot theme for --plot (default: light).",
    )
    parser.add_argument(
        "--antialias",
        action="store_true",
        help="Enable PyQtGraph antialiasing in --plot mode. Disabled by default for speed.",
    )
    parser.add_argument(
        "--no-app-config-controls",
        action="store_true",
        help="Disable the live IMU app-config controls in --plot mode.",
    )
    parser.add_argument(
        "--app-config-timeout",
        type=float,
        default=DEFAULT_APPCONF_TIMEOUT,
        metavar="SEC",
        help=(
            "Timeout for plot-mode app-config reads/writes "
            f"(default: {DEFAULT_APPCONF_TIMEOUT:g})."
        ),
    )
    parser.add_argument(
        "--app-config-fw-retries",
        type=int,
        default=DEFAULT_APPCONF_FW_RETRIES,
        metavar="N",
        help=(
            "Firmware-version retries before disabling plot-mode app-config controls "
            f"(default: {DEFAULT_APPCONF_FW_RETRIES})."
        ),
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        metavar="N",
        help="Print every Nth decoded sample. Default 0 avoids sample-print overhead.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write printed samples as CSV to stdout. Implies --print-every 1.",
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Validate IMU packets but skip value decoding for a raw transport benchmark.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_ports and args.scan_ble:
        parser.error("use only one of --list-ports or --scan-ble")
    if args.list_ports:
        print_serial_ports(args)
        return
    if args.scan_ble:
        if args.ble_scan_timeout <= 0.0:
            parser.error("--ble-scan-timeout must be greater than 0")
        scan_and_print_ble(args)
        return

    if args.mask is not None and args.fields is not None:
        parser.error("--mask and --fields are mutually exclusive")
    if args.baudrate <= 0:
        parser.error("--baudrate must be greater than 0")
    if args.timeout <= 0.0:
        parser.error("--timeout must be greater than 0")
    if args.duration < 0.0:
        parser.error("--duration must be greater than or equal to 0")
    if args.max_samples < 0:
        parser.error("--max-samples must be greater than or equal to 0")
    if args.status_interval < 0.0:
        parser.error("--status-interval must be greater than or equal to 0")
    if args.ui_rate <= 0.0:
        parser.error("--ui-rate must be greater than 0")
    if args.plot_history <= 0:
        parser.error("--plot-history must be greater than 0")
    if args.plot_max_points <= 0:
        parser.error("--plot-max-points must be greater than 0")
    if args.plot_rate <= 0.0:
        parser.error("--plot-rate must be greater than 0")
    if args.fft_window <= 0.0:
        parser.error("--fft-window must be greater than 0")
    if args.fft_rate <= 0.0:
        parser.error("--fft-rate must be greater than 0")
    if args.plot_pending <= 0:
        parser.error("--plot-pending must be greater than 0")
    if args.app_config_timeout <= 0.0:
        parser.error("--app-config-timeout must be greater than 0")
    if args.app_config_fw_retries <= 0:
        parser.error("--app-config-fw-retries must be greater than 0")
    if args.print_every < 0:
        parser.error("--print-every must be greater than or equal to 0")
    if args.pipeline_depth <= 0:
        parser.error("--pipeline-depth must be greater than 0")
    if args.ble_connect_timeout <= 0.0:
        parser.error("--ble-connect-timeout must be greater than 0")
    if args.ble_chunk_size <= 0:
        parser.error("--ble-chunk-size must be greater than 0")
    if args.plot and args.csv:
        parser.error("--plot cannot be combined with --csv")
    if args.plot and args.no_decode:
        parser.error("--plot cannot be combined with --no-decode")
    if args.plot and args.print_every > 0:
        parser.error("--plot cannot be combined with --print-every")
    mask = DEFAULT_MASK
    if args.fields is not None:
        mask = args.fields
    if args.mask is not None:
        mask = args.mask
    if args.plot:
        missing_plot_fields = [
            field_name
            for field_name in IMU_PLOT_FIELDS
            if field_value_index(mask, field_name) is None
        ]
        if missing_plot_fields:
            fields = ", ".join(missing_plot_fields)
            parser.error(f"--plot requires mask 0x{DEFAULT_MASK:04x} fields; missing: {fields}")

    target = resolve_vesc_target_from_args(
        args,
        input_stream=sys.stdin,
        output_stream=sys.stderr,
    )
    if args.plot and target.can_id is not None and not args.no_app_config_controls:
        parser.error(
            "--plot with a CAN target requires --no-app-config-controls because "
            "the live app-config client only targets the directly connected controller"
        )

    print_every = args.print_every
    if args.csv and print_every == 0:
        print_every = 1

    use_tui = (
        not args.no_tui
        and not args.csv
        and not args.no_decode
        and print_every == 0
        and sys.stderr.isatty()
    )
    tui = (
        TerminalImuDisplay(refresh_hz=args.ui_rate, stream=sys.stderr)
        if use_tui
        else None
    )

    request = build_imu_request(mask, can_id=target.can_id)
    fields = ", ".join(field_names_for_mask(mask))
    target_label = "" if target.can_id is None else f"; target_can_id={target.can_id}"

    serial_port, connection_label = open_poll_connection(target, timeout=args.timeout)
    link_label = (
        f"BLE {connection_label}"
        if target.connection.kind is VescConnectionKind.BLE
        else f"{connection_label} at {target.connection.baudrate} baud"
    )
    print(
        f"Opening {link_label}; mask=0x{mask:04x} ({fields}); "
        f"pipeline_depth={args.pipeline_depth}; "
        f"display={'plot' if args.plot else 'tui' if tui is not None else 'plain'}"
        f"{target_label}",
        file=sys.stderr,
    )
    print(HOST_RX_TIMING_NOTICE, file=sys.stderr)
    command_lock = threading.RLock()
    app_config_client: PlotAppConfigClient | None = None
    app_config_error: str | None = None
    try:
        if args.plot:
            if not args.no_app_config_controls:
                try:
                    app_config_client = PlotAppConfigClient.connect(
                        serial_port,
                        command_lock=command_lock,
                        timeout=args.app_config_timeout,
                        fw_retries=args.app_config_fw_retries,
                    )
                except Exception as exc:  # noqa: BLE001 - keep plotting if config fails.
                    app_config_error = str(exc)
                    print(
                        f"IMU app-config controls disabled: {app_config_error}",
                        file=sys.stderr,
                    )

            plot_poller = FastImuPlotPoller(
                serial_port,
                request=request,
                mask=mask,
                packet_timeout=args.timeout,
                duration=args.duration,
                max_samples=args.max_samples,
                pipeline_depth=args.pipeline_depth,
                pending_samples=args.plot_pending,
                command_lock=command_lock,
            )
            run_fast_imu_grid_plot(
                plot_poller,
                history=args.plot_history,
                max_points=args.plot_max_points,
                plot_rate=args.plot_rate,
                fft_window=args.fft_window,
                fft_rate=args.fft_rate,
                theme=args.theme,
                antialias=args.antialias,
                app_config_client=app_config_client,
                app_config_error=app_config_error,
            )
        else:
            poll_imu(
                serial_port,
                request=request,
                mask=mask,
                packet_timeout=args.timeout,
                status_interval=args.status_interval,
                duration=args.duration,
                max_samples=args.max_samples,
                pipeline_depth=args.pipeline_depth,
                print_every=print_every,
                csv=args.csv,
                no_decode=args.no_decode,
                tui=tui,
            )
    finally:
        if app_config_client is not None:
            app_config_client.close()
        serial_port.close()


if __name__ == "__main__":
    main()
