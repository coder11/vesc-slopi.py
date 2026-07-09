from __future__ import annotations

import struct

import pytest

from vesc_py.comm_ids import CommPacketId
from vesc_py.connection import VescConnection
from vesc_py.imu_streamer import (
    IMU_STREAMER_ACK_OK,
    IMU_STREAMER_CHANNEL_COUNT,
    IMU_STREAMER_CMD_START,
    IMU_STREAMER_CMD_STOP,
    IMU_STREAMER_CMD_STATUS,
    IMU_STREAMER_MSG_ACK,
    IMU_STREAMER_MSG_SAMPLE,
    IMU_STREAMER_MSG_STATUS,
    IMU_STREAMER_PROTOCOL_VERSION,
    ImuStreamerAck,
    ImuStreamerLispPrint,
    ImuStreamerSample,
    ImuStreamerStatus,
    PendingImuStreamBuffer,
    VescImuStreamer,
    build_imu_streamer_command,
    parse_imu_streamer_packet,
)
from vesc_py.packet import PacketDecoder, encode_packet


def _app_packet(app_payload: bytes) -> bytes:
    return encode_packet(bytes([CommPacketId.COMM_CUSTOM_APP_DATA]) + app_payload)


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


def test_build_command_wraps_custom_app_data_packet() -> None:
    decoder = PacketDecoder()

    payloads = list(decoder.process(build_imu_streamer_command(IMU_STREAMER_CMD_START)))

    assert payloads == [bytes([CommPacketId.COMM_CUSTOM_APP_DATA, IMU_STREAMER_CMD_START])]


def test_parse_sample_ack_and_status_messages() -> None:
    sample = parse_imu_streamer_packet(
        bytes([CommPacketId.COMM_CUSTOM_APP_DATA])
        + _sample_app_payload(sequence=7, timestamp_us=123_456)
    )
    assert isinstance(sample, ImuStreamerSample)
    assert sample.sequence == 7
    assert sample.timestamp_s == pytest.approx(0.123456)
    assert sample.channel_values() == pytest.approx((8.0, 9.0, 10.0, 17.0, 27.0, 37.0))

    ack = parse_imu_streamer_packet(
        bytes(
            [
                CommPacketId.COMM_CUSTOM_APP_DATA,
                IMU_STREAMER_MSG_ACK,
                IMU_STREAMER_PROTOCOL_VERSION,
                IMU_STREAMER_CMD_START,
                IMU_STREAMER_ACK_OK,
            ]
        )
    )
    assert ack == ImuStreamerAck(
        protocol_version=IMU_STREAMER_PROTOCOL_VERSION,
        command=IMU_STREAMER_CMD_START,
        result=IMU_STREAMER_ACK_OK,
    )
    assert ack.ok

    status = parse_imu_streamer_packet(
        bytes([CommPacketId.COMM_CUSTOM_APP_DATA])
        + struct.pack(
            ">BBBI",
            IMU_STREAMER_MSG_STATUS,
            IMU_STREAMER_PROTOCOL_VERSION,
            1,
            42,
        )
    )
    assert status == ImuStreamerStatus(
        protocol_version=IMU_STREAMER_PROTOCOL_VERSION,
        streaming=True,
        next_sequence=42,
    )

    lisp_print = parse_imu_streamer_packet(
        bytes([CommPacketId.COMM_LISP_PRINT]) + b"IMU Streamer: streaming started"
    )
    assert lisp_print == ImuStreamerLispPrint("IMU Streamer: streaming started")


def test_pending_imu_stream_buffer_drains_latest_values_after_rollover() -> None:
    buffer = PendingImuStreamBuffer(3)
    first_values = [[float(i + channel) for channel in range(IMU_STREAMER_CHANNEL_COUNT)] for i in range(2)]
    second_values = [
        [float(10 + i + channel) for channel in range(IMU_STREAMER_CHANNEL_COUNT)]
        for i in range(2)
    ]

    buffer.append_many([1.0, 2.0], first_values)
    buffer.append_many([3.0, 4.0], second_values)

    timestamps, values, dropped, stats = buffer.drain()

    assert timestamps.tolist() == [2.0, 3.0, 4.0]
    assert values[:, 0].tolist() == [1.0, 10.0, 11.0]
    assert dropped == 1
    assert stats.cumulative_dropped == 1
    assert stats.average_rate_hz == pytest.approx(1.0)
    assert stats.latest_values is not None
    assert stats.latest_values.tolist() == second_values[-1]


def test_vesc_imu_streamer_reads_samples_in_chunks_and_tracks_sequence_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lisp_print = bytes([CommPacketId.COMM_LISP_PRINT]) + b"IMU Streamer: streaming started"
    ack = bytes(
        [
            CommPacketId.COMM_CUSTOM_APP_DATA,
            IMU_STREAMER_MSG_ACK,
            IMU_STREAMER_PROTOCOL_VERSION,
            IMU_STREAMER_CMD_START,
            IMU_STREAMER_ACK_OK,
        ]
    )
    stream = (
        encode_packet(ack)
        + encode_packet(lisp_print)
        + _app_packet(_sample_app_payload(sequence=0, timestamp_us=1_000))
        + _app_packet(_sample_app_payload(sequence=2, timestamp_us=2_000))
    )

    class FakeSerial:
        timeout: float | None = None

        def __init__(self, data: bytes) -> None:
            self._data = bytearray(data)
            self.writes: list[bytes] = []
            self.closed = False

        def read(self, size: int = 1) -> bytes:
            if not self._data:
                source._stop.set()
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
            self.closed = True

    fake_serial = FakeSerial(stream)
    source = VescImuStreamer(
        connection=VescConnection.serial("/dev/null"),
        timeout=0.1,
        pending_samples=8,
        read_chunk_size=4096,
        flush_samples=8,
    )

    def fake_open_blocking_io(connection: VescConnection, *, timeout: float) -> FakeSerial:
        assert connection == VescConnection.serial("/dev/null")
        assert timeout == pytest.approx(0.1)
        return fake_serial

    monkeypatch.setattr("vesc_py.imu_streamer.open_blocking_io", fake_open_blocking_io)

    source._run()

    timestamps, values, stats = source.drain()
    assert timestamps.tolist() == pytest.approx([0.001, 0.002])
    assert values[:, :3].tolist() == [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
    assert stats.sequence_drops == 1
    assert stats.errors == 0
    assert stats.lisp_prints == 1
    assert stats.last_print == "IMU Streamer: streaming started"
    assert fake_serial.writes[0] == build_imu_streamer_command(IMU_STREAMER_CMD_START)
    assert fake_serial.writes[-1] == build_imu_streamer_command(IMU_STREAMER_CMD_STOP)
    assert fake_serial.closed is True
