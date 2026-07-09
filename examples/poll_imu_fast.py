#!/usr/bin/env python3
"""Poll VESC IMU data in a terminal TUI with the shared VESC connection CLI.

This example keeps the hot path small: one pre-encoded
``COMM_GET_IMU_DATA`` request, direct blocking reads, CRC validation,
lightweight value decoding, and in-place terminal redraws.

The script intentionally exposes only the shared connection-selection
interface from ``vesc_py.connection_cli``. All polling settings are fixed
in this file:

- mask: roll/pitch/yaw + accelerometer + gyroscope
- packet timeout: 0.1 s

Examples:
    uv run examples/poll_imu_fast.py
    uv run examples/poll_imu_fast.py --serial /dev/ttyACM0
    uv run examples/poll_imu_fast.py --ble AA:BB:CC:DD:EE:FF
    uv run examples/poll_imu_fast.py --can-id 1
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TextIO, cast

from vesc_py.buffer import VescBuffer
from vesc_py.comm_ids import CommPacketId
from vesc_py.connection import (
    VescConnectionKind,
    VescTarget,
    build_imu_request,
    open_blocking_io,
)
from vesc_py.connection_cli import (
    add_vesc_connection_arguments,
    resolve_vesc_target_from_args,
)
from vesc_py.crc import crc16
from vesc_py.imu import IMU_FIELDS
from vesc_py.packet import MAX_PACKET_LEN

DEFAULT_MASK = 0x01FF
DEFAULT_TIMEOUT = 0.1
DEFAULT_STATUS_INTERVAL = 1.0
DEFAULT_UI_RATE = 10.0
NSEC_PER_SEC = 1_000_000_000
RATE_LIMIT_SLEEP_SLACK_NS = 200_000
HOST_RX_TIMESTAMP_SOURCE = "host_rx_after_packet"
HOST_RX_TIMING_NOTICE = (
    "Firmware samples IMU data at the configured IMU sample rate, but "
    "COMM_GET_IMU_DATA returns only the latest cached values. This tool timestamps "
    "host packet receive completion, so dt, plot rates, and PSD timing describe "
    "host-observed packet timing. Use firmware-side sample timestamps or sample "
    "indexes for sampling-jitter or phase analysis."
)


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
            f"{name:<8} {value:>16.8g}" for name, value in zip(names, parsed.values)
        )

        if self._line_count > 0:
            self._stream.write(f"\x1b[{self._line_count}A")
        for line in lines:
            self._stream.write(f"\r\x1b[2K{line}\n")
        self._stream.flush()

        self._line_count = len(lines)
        self._next_render_ns = now_ns + self._interval_ns


def field_names_for_mask(mask: int) -> tuple[str, ...]:
    """Return IMU field names in VESC wire order for *mask*."""
    return tuple(name for index, name in enumerate(IMU_FIELDS) if mask & (1 << index))


def parse_imu_payload(payload: bytes) -> ParsedImu:
    """Decode a COMM_GET_IMU_DATA response payload without Pydantic overhead."""
    buffer = VescBuffer(payload)
    command = buffer.pop_uint8()
    if command != CommPacketId.COMM_GET_IMU_DATA:
        raise ValueError(f"unexpected command id {command}")

    mask = buffer.pop_uint16()
    values = [
        buffer.pop_double32_auto()
        for index in range(len(IMU_FIELDS))
        if mask & (1 << index)
    ]
    vesc_id = buffer.pop_uint8() if buffer.remaining >= 1 else None
    return ParsedImu(mask=mask, values=tuple(values), vesc_id=vesc_id)


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


def ble_connection_help(address: str, message: str) -> str:
    """Return a concise hint for direct BLE connection failures."""
    return "\n".join(
        [
            f"Could not connect to VESC BLE device at {address}.",
            "",
            message,
        ]
    )


def open_poll_connection(target: VescTarget, *, timeout: float) -> tuple[PollIo, str]:
    """Open the resolved transport selected by the shared VESC CLI helper."""
    connection = target.connection
    if connection.kind is VescConnectionKind.BLE:
        print(
            f"Connecting to VESC BLE device at {connection.address} ...",
            file=sys.stderr,
        )

    try:
        return open_blocking_io(connection, timeout=timeout), connection.address
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
    """Format selected IMU values for low-rate plain status output."""
    names = field_names_for_mask(mask)
    parts: list[str] = []
    for name, value in zip(names[:max_fields], values[:max_fields]):
        parts.append(f"{name}={value:.6g}")
    if len(values) > max_fields:
        parts.append("...")
    return " ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for shared connection flags plus fast-poll settings."""
    parser = argparse.ArgumentParser(
        description="Poll VESC IMU data in a terminal TUI.",
    )
    add_vesc_connection_arguments(parser)
    parser.add_argument(
        "--poll-rate",
        type=float,
        default=None,
        metavar="HZ",
        help=(
            "Cap request rate in Hz. "
            "When omitted, polling runs as fast as responses arrive."
        ),
    )
    return parser


def wait_until_ns(deadline_ns: int) -> None:
    """Wait until *deadline_ns* using coarse sleep plus a short busy-spin."""
    while True:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > RATE_LIMIT_SLEEP_SLACK_NS:
            sleep_s = (remaining_ns - RATE_LIMIT_SLEEP_SLACK_NS) / NSEC_PER_SEC
            time.sleep(min(sleep_s, 0.01))


def poll_imu(
    serial_port: PollIo,
    *,
    request: bytes,
    packet_timeout: float,
    poll_rate_hz: float | None = None,
    status_stream: TextIO = sys.stderr,
    status_interval: float = DEFAULT_STATUS_INTERVAL,
    tui: TerminalImuDisplay | None = None,
    max_samples: int = 0,
) -> None:
    """Run the high-rate IMU polling loop."""
    if poll_rate_hz is not None and poll_rate_hz <= 0.0:
        raise ValueError("poll_rate_hz must be greater than 0")

    reader_stats = ReaderStats()
    poll_stats = PollStats()
    start_ns = time.perf_counter_ns()
    previous_sample_ns: int | None = None
    previous_request_ns: int | None = None
    next_status_ns = start_ns + round(status_interval * NSEC_PER_SEC)
    status_window_ns = start_ns
    status_window_samples = 0
    latest: ParsedImu | None = None
    poll_interval_ns = (
        None
        if poll_rate_hz is None
        else max(1, round(NSEC_PER_SEC / poll_rate_hz))
    )

    if tui is not None:
        tui.start()

    try:
        while max_samples <= 0 or poll_stats.samples < max_samples:
            if poll_interval_ns is not None and previous_request_ns is not None:
                next_request_ns = previous_request_ns + poll_interval_ns
                wait_until_ns(next_request_ns)

            previous_request_ns = time.perf_counter_ns()
            serial_port.write(request)
            poll_stats.requests += 1

            payload = read_expected_imu_packet(
                serial_port,
                packet_timeout,
                reader_stats,
            )
            now_ns = time.perf_counter_ns()
            if payload is None:
                poll_stats.timeouts += 1
                serial_port.reset_input_buffer()
                continue

            status_window_samples += 1
            try:
                latest = parse_imu_payload(payload)
            except ValueError as exc:
                poll_stats.parse_errors += 1
                print(f"parse error: {exc}", file=status_stream)
                continue

            poll_stats.samples += 1
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
                print(status, file=status_stream)
                status_window_ns = now_ns
                status_window_samples = 0
                next_status_ns = now_ns + round(status_interval * NSEC_PER_SEC)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=status_stream)
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
            file=status_stream,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.poll_rate is not None and args.poll_rate <= 0.0:
        parser.error("--poll-rate must be greater than 0")

    target = resolve_vesc_target_from_args(args)

    request = build_imu_request(DEFAULT_MASK, can_id=target.can_id)
    fields = ", ".join(field_names_for_mask(DEFAULT_MASK))
    target_label = "" if target.can_id is None else f"; target_can_id={target.can_id}"
    use_tui = sys.stderr.isatty()
    display = "tui" if use_tui else "plain"
    tui = (
        TerminalImuDisplay(refresh_hz=DEFAULT_UI_RATE, stream=sys.stderr)
        if use_tui
        else None
    )

    serial_port, connection_label = open_poll_connection(
        target, timeout=DEFAULT_TIMEOUT
    )
    link_label = (
        f"BLE {connection_label}"
        if target.connection.kind is VescConnectionKind.BLE
        else f"{connection_label} at {target.connection.baudrate} baud"
    )

    print(
        f"Opening {link_label}; mask=0x{DEFAULT_MASK:04x} ({fields}); "
        f"display={display}; poll_rate="
        f"{'max' if args.poll_rate is None else f'<= {args.poll_rate:g} Hz'}"
        f"{target_label}",
        file=sys.stderr,
    )
    print(HOST_RX_TIMING_NOTICE, file=sys.stderr)

    try:
        poll_imu(
            serial_port,
            request=request,
            packet_timeout=DEFAULT_TIMEOUT,
            poll_rate_hz=cast(float | None, args.poll_rate),
            tui=tui,
        )
    finally:
        serial_port.close()


if __name__ == "__main__":
    main()
