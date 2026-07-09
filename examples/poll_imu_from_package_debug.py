#!/usr/bin/env python3
"""Debug the IMU Streamer package receive path without starting the GUI."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

from vesc_py.comm_ids import CommPacketId
from vesc_py.connection import BlockingIo, VescConnectionKind, open_blocking_io
from vesc_py.connection_cli import (
    add_vesc_connection_arguments,
    resolve_vesc_connection_from_args,
)
from vesc_py.imu_streamer import (
    DEFAULT_READ_CHUNK_SIZE,
    DEFAULT_TIMEOUT,
    IMU_STREAMER_ACK_OK,
    IMU_STREAMER_ACK_UNKNOWN_CMD,
    IMU_STREAMER_ACK_UNSUPPORTED,
    IMU_STREAMER_CMD_START,
    IMU_STREAMER_CMD_STATUS,
    IMU_STREAMER_CMD_STOP,
    ImuStreamerAck,
    ImuStreamerLispPrint,
    ImuStreamerSample,
    ImuStreamerStatus,
    build_imu_streamer_command,
    parse_imu_streamer_packet,
)
from vesc_py.packet import PacketDecoder

DEFAULT_DURATION = 5.0
DEFAULT_PRINT_SAMPLES = 10
DEFAULT_STATUS_INTERVAL = 1.0
_UINT32_MODULO = 1 << 32
_UINT32_HALF_RANGE = 1 << 31

_COMMAND_NAMES = {
    IMU_STREAMER_CMD_START: "start",
    IMU_STREAMER_CMD_STOP: "stop",
    IMU_STREAMER_CMD_STATUS: "status",
}
_ACK_RESULT_NAMES = {
    IMU_STREAMER_ACK_OK: "ok",
    IMU_STREAMER_ACK_UNKNOWN_CMD: "unknown-command",
    IMU_STREAMER_ACK_UNSUPPORTED: "unsupported-firmware",
}


@dataclass(slots=True)
class DebugStats:
    """Counters for a CLI debug run."""

    raw_bytes: int = 0
    packets: int = 0
    custom_packets: int = 0
    samples: int = 0
    acknowledgements: int = 0
    statuses: int = 0
    lisp_prints: int = 0
    unexpected_packets: int = 0
    parse_errors: int = 0
    empty_reads: int = 0
    sequence_drops: int = 0
    out_of_order: int = 0
    first_sequence: int | None = None
    latest_sequence: int | None = None
    latest_timestamp_s: float | None = None
    expected_sequence: int | None = None
    start_ack: ImuStreamerAck | None = None
    latest_status: ImuStreamerStatus | None = None
    latest_print: str | None = None
    last_error: str | None = None

    @property
    def packet_errors(self) -> int:
        """Return parser/protocol failures, excluding empty reads."""
        return self.unexpected_packets + self.parse_errors + self.out_of_order


def command_name(command: int) -> str:
    """Return a readable streamer command name."""
    return _COMMAND_NAMES.get(command, f"0x{command:02x}")


def ack_result_name(result: int) -> str:
    """Return a readable ACK result name."""
    return _ACK_RESULT_NAMES.get(result, f"result-{result}")


def sample_line(sample: ImuStreamerSample) -> str:
    """Format one streamer sample for debug output."""
    return (
        f"sample seq={sample.sequence} t={sample.timestamp_s:.6f}s "
        f"acc=({sample.acc_x:.6g}, {sample.acc_y:.6g}, {sample.acc_z:.6g}) "
        f"gyro=({sample.gyro_x:.6g}, {sample.gyro_y:.6g}, {sample.gyro_z:.6g}) "
        f"mag=({sample.mag_x:.6g}, {sample.mag_y:.6g}, {sample.mag_z:.6g})"
    )


def summary_line(stats: DebugStats, elapsed_s: float) -> str:
    """Format a one-line run summary."""
    rate = stats.samples / elapsed_s if elapsed_s > 0.0 else 0.0
    latest = (
        "n/a"
        if stats.latest_timestamp_s is None
        else f"{stats.latest_timestamp_s:.6f}s"
    )
    return (
        f"summary elapsed={elapsed_s:.2f}s raw={stats.raw_bytes}B "
        f"vesc_packets={stats.packets} custom={stats.custom_packets} "
        f"samples={stats.samples} rate={rate:.1f}Hz "
        f"acks={stats.acknowledgements} statuses={stats.statuses} "
        f"prints={stats.lisp_prints} "
        f"empty_reads={stats.empty_reads} packet_errors={stats.packet_errors} "
        f"seq_gap={stats.sequence_drops} out_of_order={stats.out_of_order} "
        f"latest_seq={stats.latest_sequence if stats.latest_sequence is not None else 'n/a'} "
        f"latest_stream_t={latest}"
    )


def update_sequence_stats(stats: DebugStats, sample: ImuStreamerSample) -> None:
    """Track sequence continuity for one sample."""
    if stats.first_sequence is None:
        stats.first_sequence = sample.sequence
    if stats.expected_sequence is not None and sample.sequence != stats.expected_sequence:
        gap = (sample.sequence - stats.expected_sequence) % _UINT32_MODULO
        if 0 < gap < _UINT32_HALF_RANGE:
            stats.sequence_drops += gap
        else:
            stats.out_of_order += 1

    stats.latest_sequence = sample.sequence
    stats.latest_timestamp_s = sample.timestamp_s
    stats.expected_sequence = (sample.sequence + 1) % _UINT32_MODULO


def process_debug_packet(
    payload: bytes,
    stats: DebugStats,
    *,
    print_samples: int,
    raw_packets: int,
) -> list[str]:
    """Decode one VESC payload and return debug output lines."""
    stats.packets += 1
    lines: list[str] = []
    if raw_packets < 0 or stats.packets <= raw_packets:
        lines.append(f"packet #{stats.packets} payload={payload.hex(' ')}")

    if not payload:
        stats.unexpected_packets += 1
        lines.append("unexpected empty VESC payload")
        return lines

    try:
        message = parse_imu_streamer_packet(payload)
    except ValueError as exc:
        stats.parse_errors += 1
        stats.last_error = str(exc)
        lines.append(f"parse error: {exc}; payload={payload.hex(' ')}")
        return lines

    stats.last_error = None
    if payload[0] == CommPacketId.COMM_CUSTOM_APP_DATA:
        stats.custom_packets += 1
    elif payload[0] != CommPacketId.COMM_LISP_PRINT:
        stats.unexpected_packets += 1
        lines.append(f"unexpected VESC packet command=0x{payload[0]:02x} len={len(payload)}")
        return lines

    if isinstance(message, ImuStreamerSample):
        stats.samples += 1
        update_sequence_stats(stats, message)
        if print_samples < 0 or stats.samples <= print_samples:
            lines.append(sample_line(message))
    elif isinstance(message, ImuStreamerAck):
        stats.acknowledgements += 1
        if message.command == IMU_STREAMER_CMD_START:
            stats.start_ack = message
        lines.append(
            "ack "
            f"cmd={command_name(message.command)} "
            f"version={message.protocol_version} "
            f"result={ack_result_name(message.result)}"
        )
    elif isinstance(message, ImuStreamerStatus):
        stats.statuses += 1
        stats.latest_status = message
        state = "running" if message.streaming else "stopped"
        lines.append(
            f"status version={message.protocol_version} "
            f"stream={state} next_seq={message.next_sequence}"
        )
    elif isinstance(message, ImuStreamerLispPrint):
        stats.lisp_prints += 1
        stats.latest_print = message.text
        lines.append(f"lisp_print {message.text}")

    return lines


def emit_lines(lines: Sequence[str], stream: TextIO) -> None:
    """Print debug lines and flush so long runs remain live."""
    for line in lines:
        print(line, file=stream)
    if lines:
        stream.flush()


def debug_imu_stream(
    serial_port: BlockingIo,
    *,
    duration: float,
    timeout: float,
    read_chunk_size: int,
    status_interval: float,
    print_samples: int,
    raw_packets: int,
    max_samples: int,
    stream: TextIO = sys.stdout,
) -> DebugStats:
    """Run the low-level streamer debug loop on an open byte stream."""
    decoder = PacketDecoder()
    stats = DebugStats()
    start_s = time.monotonic()
    deadline_s = math.inf if duration <= 0.0 else start_s + duration
    next_status_s = (
        math.inf if status_interval <= 0.0 else start_s + status_interval
    )

    serial_port.reset_input_buffer()
    serial_port.write(build_imu_streamer_command(IMU_STREAMER_CMD_STATUS))
    serial_port.write(build_imu_streamer_command(IMU_STREAMER_CMD_START))
    print("sent status request and start command", file=stream)
    stream.flush()

    while time.monotonic() < deadline_s:
        if max_samples > 0 and stats.samples >= max_samples:
            break

        now_s = time.monotonic()
        if now_s >= next_status_s:
            serial_port.write(build_imu_streamer_command(IMU_STREAMER_CMD_STATUS))
            emit_lines((summary_line(stats, now_s - start_s),), stream)
            next_status_s = now_s + status_interval

        read_timeout = timeout
        if math.isfinite(deadline_s):
            read_timeout = min(read_timeout, max(0.001, deadline_s - now_s))
        serial_port.timeout = read_timeout
        chunk = serial_port.read(read_chunk_size)
        if not chunk:
            stats.empty_reads += 1
            continue

        stats.raw_bytes += len(chunk)
        for payload in decoder.process(chunk):
            emit_lines(
                process_debug_packet(
                    payload,
                    stats,
                    print_samples=print_samples,
                    raw_packets=raw_packets,
                ),
                stream,
            )

    emit_lines((summary_line(stats, time.monotonic() - start_s),), stream)
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Debug raw IMU Streamer package packets without the GUI.",
    )
    add_vesc_connection_arguments(parser, include_can_id=False)
    parser.set_defaults(timeout=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        metavar="SEC",
        help="Capture duration in seconds; <= 0 runs until interrupted (default: 5).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N samples; 0 means duration decides (default: 0).",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=DEFAULT_PRINT_SAMPLES,
        metavar="N",
        help="Print first N samples; negative prints all (default: 10).",
    )
    parser.add_argument(
        "--raw-packets",
        type=int,
        default=0,
        metavar="N",
        help="Print first N decoded VESC payloads as hex; negative prints all.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL,
        metavar="SEC",
        help="Seconds between status requests and summary lines; <= 0 disables.",
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
    """CLI entrypoint."""
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
    if args.read_chunk_size <= 0:
        raise SystemExit("--read-chunk-size must be greater than 0")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be greater than or equal to 0")

    connection = resolve_vesc_connection_from_args(args)
    link_label = (
        f"BLE {connection.address}"
        if connection.kind is VescConnectionKind.BLE
        else f"{connection.address} at {connection.baudrate} baud"
    )
    print(
        f"Opening {link_label}; timeout={args.timeout:g}s; "
        f"duration={'infinite' if args.duration <= 0.0 else f'{args.duration:g}s'}",
        file=sys.stderr,
    )

    serial_port = open_blocking_io(connection, timeout=args.timeout)
    try:
        debug_imu_stream(
            serial_port,
            duration=args.duration,
            timeout=args.timeout,
            read_chunk_size=args.read_chunk_size,
            status_interval=args.status_interval,
            print_samples=args.print_samples,
            raw_packets=args.raw_packets,
            max_samples=args.max_samples,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    finally:
        try:
            serial_port.write(build_imu_streamer_command(IMU_STREAMER_CMD_STOP))
            print("sent stop command", file=sys.stderr)
        finally:
            serial_port.close()


if __name__ == "__main__":
    main()
