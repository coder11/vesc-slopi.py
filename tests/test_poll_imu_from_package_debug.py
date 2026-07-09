from __future__ import annotations

import struct
from io import StringIO

import pytest

from examples.poll_imu_from_package_debug import (
    DebugStats,
    build_parser,
    debug_imu_stream,
    process_debug_packet,
)
from vesc_py.comm_ids import CommPacketId
from vesc_py.imu_streamer import (
    IMU_STREAMER_ACK_OK,
    IMU_STREAMER_CMD_START,
    IMU_STREAMER_CMD_STATUS,
    IMU_STREAMER_MSG_ACK,
    IMU_STREAMER_MSG_SAMPLE,
    IMU_STREAMER_MSG_STATUS,
    IMU_STREAMER_PROTOCOL_VERSION,
    build_imu_streamer_command,
)
from vesc_py.packet import encode_packet


def _custom_payload(app_payload: bytes) -> bytes:
    return bytes([CommPacketId.COMM_CUSTOM_APP_DATA]) + app_payload


def _custom_packet(app_payload: bytes) -> bytes:
    return encode_packet(_custom_payload(app_payload))


def _sample_app_payload(sequence: int, timestamp_us: int) -> bytes:
    values = (
        10.0 + sequence,
        20.0 + sequence,
        30.0 + sequence,
        1.0 + sequence,
        2.0 + sequence,
        3.0 + sequence,
        100.0 + sequence,
        200.0 + sequence,
        300.0 + sequence,
    )
    return struct.pack(
        ">BIQ9f",
        IMU_STREAMER_MSG_SAMPLE,
        sequence,
        timestamp_us,
        *values,
    )


def test_process_debug_packet_prints_ack_status_and_sample() -> None:
    stats = DebugStats()

    ack_lines = process_debug_packet(
        _custom_payload(
            bytes(
                [
                    IMU_STREAMER_MSG_ACK,
                    IMU_STREAMER_PROTOCOL_VERSION,
                    IMU_STREAMER_CMD_START,
                    IMU_STREAMER_ACK_OK,
                ]
            )
        ),
        stats,
        print_samples=1,
        raw_packets=1,
    )
    status_lines = process_debug_packet(
        _custom_payload(
            struct.pack(
                ">BBBI",
                IMU_STREAMER_MSG_STATUS,
                IMU_STREAMER_PROTOCOL_VERSION,
                1,
                5,
            )
        ),
        stats,
        print_samples=1,
        raw_packets=1,
    )
    sample_lines = process_debug_packet(
        _custom_payload(_sample_app_payload(sequence=5, timestamp_us=123_000)),
        stats,
        print_samples=1,
        raw_packets=1,
    )
    print_lines = process_debug_packet(
        bytes([CommPacketId.COMM_LISP_PRINT]) + b"IMU Streamer: streaming started",
        stats,
        print_samples=1,
        raw_packets=1,
    )

    assert any("ack cmd=start" in line for line in ack_lines)
    assert any("status version=1 stream=running next_seq=5" in line for line in status_lines)
    assert any("sample seq=5 t=0.123000s" in line for line in sample_lines)
    assert any("lisp_print IMU Streamer: streaming started" in line for line in print_lines)
    assert stats.custom_packets == 3
    assert stats.acknowledgements == 1
    assert stats.statuses == 1
    assert stats.samples == 1
    assert stats.lisp_prints == 1


def test_debug_stream_sends_status_start_and_decodes_sample_batch() -> None:
    stream = (
        _custom_packet(
            struct.pack(
                ">BBBI",
                IMU_STREAMER_MSG_STATUS,
                IMU_STREAMER_PROTOCOL_VERSION,
                0,
                0,
            )
        )
        + _custom_packet(
            bytes(
                [
                    IMU_STREAMER_MSG_ACK,
                    IMU_STREAMER_PROTOCOL_VERSION,
                    IMU_STREAMER_CMD_START,
                    IMU_STREAMER_ACK_OK,
                ]
            )
        )
        + encode_packet(bytes([CommPacketId.COMM_LISP_PRINT]) + b"IMU Streamer: streaming started")
        + _custom_packet(_sample_app_payload(sequence=0, timestamp_us=1_000))
        + _custom_packet(_sample_app_payload(sequence=1, timestamp_us=2_000))
    )

    class FakeSerial:
        timeout: float | None = None

        def __init__(self, data: bytes) -> None:
            self._data = bytearray(data)
            self.writes: list[bytes] = []

        def read(self, size: int = 1) -> bytes:
            if not self._data:
                return b""
            chunk = bytes(self._data[:size])
            del self._data[:size]
            return chunk

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            return len(data)

        def reset_input_buffer(self) -> None:
            return None

        def close(self) -> None:
            return None

    fake_serial = FakeSerial(stream)
    output = StringIO()

    stats = debug_imu_stream(
        fake_serial,
        duration=1.0,
        timeout=0.1,
        read_chunk_size=4096,
        status_interval=0.0,
        print_samples=2,
        raw_packets=0,
        max_samples=2,
        stream=output,
    )

    assert fake_serial.writes[:2] == [
        build_imu_streamer_command(IMU_STREAMER_CMD_STATUS),
        build_imu_streamer_command(IMU_STREAMER_CMD_START),
    ]
    assert stats.samples == 2
    assert stats.acknowledgements == 1
    assert stats.statuses == 1
    assert stats.lisp_prints == 1
    assert stats.sequence_drops == 0
    assert stats.packet_errors == 0
    assert "sample seq=0" in output.getvalue()
    assert "lisp_print IMU Streamer: streaming started" in output.getvalue()
    assert "samples=2" in output.getvalue()


def test_debug_parser_accepts_direct_connection_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--serial",
            "/dev/ttyACM0",
            "--duration",
            "2",
            "--max-samples",
            "5",
            "--raw-packets",
            "3",
        ]
    )

    assert args.serial == "/dev/ttyACM0"
    assert not hasattr(args, "can_id")
    assert args.duration == pytest.approx(2.0)
    assert args.max_samples == 5
    assert args.raw_packets == 3
