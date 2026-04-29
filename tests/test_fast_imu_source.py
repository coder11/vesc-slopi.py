import math
import time

import pytest

from vesc_py.connection import VescConnection
from vesc_py.buffer import VescBuffer
from vesc_py.comm_ids import CommPacketId
from vesc_py.fast_imu_source import (
    VescImuSignalSource,
    imu_axes_mask,
    imu_axis_display_value,
    imu_axis_mask,
    imu_axis_unit,
    imu_data_from_payload,
    imu_values_from_payload,
    parse_imu_axis,
    parse_imu_axes,
)


def test_parse_imu_axis_accepts_accel_gyro_rpy_and_aliases() -> None:
    assert parse_imu_axis("x") == "acc_x"
    assert parse_imu_axis("accel-z") == "acc_z"
    assert parse_imu_axis("gyro-y") == "gyro_y"
    assert parse_imu_axis("roll") == "roll"
    assert parse_imu_axis("pitch") == "pitch"
    assert parse_imu_axis("yaw") == "yaw"


def test_parse_imu_axis_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="unknown IMU axis"):
        parse_imu_axis("temperature")


def test_parse_imu_axes_rejects_empty_and_duplicate_axes() -> None:
    with pytest.raises(ValueError, match="at least one"):
        parse_imu_axes(())
    with pytest.raises(ValueError, match="unique"):
        parse_imu_axes(("acc_z", "accel-z"))


def test_imu_axis_mask_matches_expected_field_bits() -> None:
    assert imu_axis_mask("roll") == 1 << 0
    assert imu_axis_mask("pitch") == 1 << 1
    assert imu_axis_mask("yaw") == 1 << 2
    assert imu_axis_mask("acc_x") == 1 << 3
    assert imu_axis_mask("acc_y") == 1 << 4
    assert imu_axis_mask("acc_z") == 1 << 5
    assert imu_axis_mask("gyro_x") == 1 << 6
    assert imu_axis_mask("gyro_y") == 1 << 7
    assert imu_axis_mask("gyro_z") == 1 << 8
    assert imu_axes_mask(("acc_z", "gyro_z")) == (1 << 5) | (1 << 8)


def test_imu_values_from_payload_extracts_ordered_axis_values() -> None:
    buffer = VescBuffer()
    buffer.append_uint8(CommPacketId.COMM_GET_IMU_DATA)
    buffer.append_uint16((1 << 5) | (1 << 8))
    buffer.append_double32_auto(0.5)
    buffer.append_double32_auto(42.0)

    payload = buffer.to_bytes()
    imu_data = imu_data_from_payload(payload)

    assert imu_data.response_mask == (1 << 5) | (1 << 8)
    assert imu_data.acc_z == pytest.approx(0.5)
    assert imu_data.gyro_z == pytest.approx(42.0)
    assert imu_values_from_payload(payload, ("gyro_z", "acc_z")) == {
        "gyro_z": pytest.approx(42.0),
        "acc_z": pytest.approx(0.5),
    }


def test_imu_axis_units_match_display_values() -> None:
    assert imu_axis_unit("roll") == "deg"
    assert imu_axis_unit("acc_x") == "g"
    assert imu_axis_unit("gyro_z") == "deg/s"


def test_imu_axis_display_value_converts_rpy_to_degrees() -> None:
    assert imu_axis_display_value("roll", math.pi) == pytest.approx(180.0)
    assert imu_axis_display_value("pitch", math.pi / 2.0) == pytest.approx(90.0)
    assert imu_axis_display_value("yaw", -math.pi / 2.0) == pytest.approx(-90.0)


def test_imu_axis_display_value_leaves_accel_and_gyro_unchanged() -> None:
    assert imu_axis_display_value("acc_x", 1.25) == pytest.approx(1.25)
    assert imu_axis_display_value("gyro-y", -42.5) == pytest.approx(-42.5)


def test_vesc_source_constructor_rejects_invalid_values() -> None:
    base = {
        "connection": VescConnection.serial("/dev/null"),
        "axis": "acc_x",
        "timeout": 0.1,
        "pending_samples": 64,
    }

    with pytest.raises(ValueError, match="axis"):
        VescImuSignalSource(**{**base, "axis": "bad"})
    with pytest.raises(ValueError, match="timeout"):
        VescImuSignalSource(**{**base, "timeout": 0.0})
    with pytest.raises(ValueError, match="poll_rate_hz"):
        VescImuSignalSource(**{**base, "poll_rate_hz": 0.0})
    with pytest.raises(ValueError, match="can_id"):
        VescImuSignalSource(**{**base, "can_id": 254})
    with pytest.raises(ValueError, match="capacity"):
        VescImuSignalSource(**{**base, "pending_samples": 0})


def test_vesc_source_constructor_exposes_channel_and_unit_without_opening_serial() -> None:
    source = VescImuSignalSource(
        connection=VescConnection.serial("/dev/null"),
        axis="gyro-z",
        timeout=0.1,
        pending_samples=64,
    )

    assert source.channel_name == "gyro_z"
    assert source.unit == "deg/s"


def test_vesc_source_run_caps_poll_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_times: list[float] = []

    class FakeSerial:
        timeout = 0.0

        def write(self, _data: bytes) -> None:
            write_times.append(time.perf_counter())

        def read(self, _size: int) -> bytes:
            return b""

        def reset_input_buffer(self) -> None:
            return None

        def close(self) -> None:
            return None

    fake_serial = FakeSerial()
    source = VescImuSignalSource(
        connection=VescConnection.serial("/dev/null"),
        axis="acc_x",
        timeout=0.1,
        pending_samples=64,
        poll_rate_hz=20.0,
    )

    def fake_open_blocking_io(connection: VescConnection, *, timeout: float) -> FakeSerial:
        assert connection == VescConnection.serial("/dev/null")
        assert timeout == pytest.approx(0.1)
        return fake_serial

    def fake_read_expected_imu_packet(
        serial_port: FakeSerial,
        packet_timeout: float,
        stats: object,
    ) -> bytes:
        assert serial_port is fake_serial
        assert packet_timeout == pytest.approx(0.1)
        assert stats is not None
        if len(write_times) >= 3:
            source._stop.set()
        return b"payload"

    monkeypatch.setattr("vesc_py.fast_imu_source.open_blocking_io", fake_open_blocking_io)
    monkeypatch.setattr(
        "vesc_py.fast_imu_source._read_expected_imu_packet",
        fake_read_expected_imu_packet,
    )
    monkeypatch.setattr(source, "_value_from_payload", lambda _payload: 1.25)

    source._run()

    assert len(write_times) == 3
    latency_stats = source.response_latency_stats()
    assert latency_stats.latest_value is not None
    assert latency_stats.latest_value >= 0.0
    lateness_stats = source.request_lateness_stats()
    assert lateness_stats.latest_value is not None
    assert lateness_stats.latest_value >= 0.0
    missed_stats = source.missed_slots_stats()
    assert missed_stats.latest_value is not None
    assert missed_stats.latest_value >= 0.0
    timestamps, values, latencies, latenesses, missed_slots, stats = (
        source.drain_with_timing()
    )
    assert timestamps.size == 3
    assert values.size == 3
    assert latencies.size == 3
    assert latenesses.size == 3
    assert missed_slots.size == 3
    assert stats.latest_value == pytest.approx(1.25)
    assert all(latency >= 0.0 for latency in latencies)
    assert all(lateness >= 0.0 for lateness in latenesses)
    assert all(missed >= 0.0 for missed in missed_slots)
    intervals = [later - earlier for earlier, later in zip(write_times, write_times[1:])]
    assert intervals[0] >= 0.045
    assert intervals[1] >= 0.045


def test_vesc_source_timestamps_samples_at_request_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_ns = 1_000_000_000
    request_times_ns: list[int] = []
    response_latencies_ns = [200_000, 1_400_000, 100_000]
    wake_latenesses_ns = [0, 300_000, 50_000]

    class FakeSerial:
        timeout = 0.0

        def write(self, _data: bytes) -> None:
            request_times_ns.append(clock_ns)

        def read(self, _size: int) -> bytes:
            return b""

        def reset_input_buffer(self) -> None:
            return None

        def close(self) -> None:
            return None

    fake_serial = FakeSerial()
    source = VescImuSignalSource(
        connection=VescConnection.serial("/dev/null"),
        axis="acc_x",
        timeout=0.1,
        pending_samples=64,
        poll_rate_hz=500.0,
    )

    def fake_open_blocking_io(connection: VescConnection, *, timeout: float) -> FakeSerial:
        assert connection == VescConnection.serial("/dev/null")
        assert timeout == pytest.approx(0.1)
        return fake_serial

    def fake_wait_until_ns(deadline_ns: int, _stop: object) -> bool:
        nonlocal clock_ns
        clock_ns = max(clock_ns, deadline_ns) + wake_latenesses_ns[
            len(request_times_ns)
        ]
        return True

    def fake_read_expected_imu_packet(
        serial_port: FakeSerial,
        packet_timeout: float,
        stats: object,
    ) -> bytes:
        nonlocal clock_ns
        assert serial_port is fake_serial
        assert packet_timeout == pytest.approx(0.1)
        assert stats is not None
        clock_ns += response_latencies_ns[len(request_times_ns) - 1]
        if len(request_times_ns) >= len(response_latencies_ns):
            source._stop.set()
        return b"payload"

    monkeypatch.setattr("vesc_py.fast_imu_source.open_blocking_io", fake_open_blocking_io)
    monkeypatch.setattr("vesc_py.fast_imu_source._wait_until_ns", fake_wait_until_ns)
    monkeypatch.setattr(
        "vesc_py.fast_imu_source._read_expected_imu_packet",
        fake_read_expected_imu_packet,
    )
    monkeypatch.setattr(
        "vesc_py.fast_imu_source.time.perf_counter_ns",
        lambda: clock_ns,
    )
    monkeypatch.setattr(source, "_value_from_payload", lambda _payload: 1.25)

    source._run()

    timestamps, _values, latencies, latenesses, missed_slots, _stats = (
        source.drain_with_timing()
    )
    assert request_times_ns == [1_000_000_000, 1_002_300_000, 1_004_050_000]
    assert timestamps.tolist() == pytest.approx([0.0, 0.002, 0.004])
    assert latencies.tolist() == pytest.approx([0.0002, 0.0014, 0.0001])
    assert latenesses.tolist() == pytest.approx([0.0, 0.0003, 0.00005])
    assert missed_slots.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_vesc_source_skips_missed_poll_slots_after_long_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_ns = 1_000_000_000
    request_times_ns: list[int] = []
    response_latencies_ns = [100_000, 6_100_000, 100_000, 100_000]

    class FakeSerial:
        timeout = 0.0

        def write(self, _data: bytes) -> None:
            request_times_ns.append(clock_ns)

        def read(self, _size: int) -> bytes:
            return b""

        def reset_input_buffer(self) -> None:
            return None

        def close(self) -> None:
            return None

    fake_serial = FakeSerial()
    source = VescImuSignalSource(
        connection=VescConnection.serial("/dev/null"),
        axis="acc_x",
        timeout=0.1,
        pending_samples=64,
        poll_rate_hz=500.0,
    )

    def fake_open_blocking_io(connection: VescConnection, *, timeout: float) -> FakeSerial:
        assert connection == VescConnection.serial("/dev/null")
        assert timeout == pytest.approx(0.1)
        return fake_serial

    def fake_wait_until_ns(deadline_ns: int, _stop: object) -> bool:
        nonlocal clock_ns
        clock_ns = max(clock_ns, deadline_ns)
        return True

    def fake_read_expected_imu_packet(
        serial_port: FakeSerial,
        packet_timeout: float,
        stats: object,
    ) -> bytes:
        nonlocal clock_ns
        assert serial_port is fake_serial
        assert packet_timeout == pytest.approx(0.1)
        assert stats is not None
        clock_ns += response_latencies_ns[len(request_times_ns) - 1]
        if len(request_times_ns) >= len(response_latencies_ns):
            source._stop.set()
        return b"payload"

    monkeypatch.setattr("vesc_py.fast_imu_source.open_blocking_io", fake_open_blocking_io)
    monkeypatch.setattr("vesc_py.fast_imu_source._wait_until_ns", fake_wait_until_ns)
    monkeypatch.setattr(
        "vesc_py.fast_imu_source._read_expected_imu_packet",
        fake_read_expected_imu_packet,
    )
    monkeypatch.setattr(
        "vesc_py.fast_imu_source.time.perf_counter_ns",
        lambda: clock_ns,
    )
    monkeypatch.setattr(source, "_value_from_payload", lambda _payload: 1.25)

    source._run()

    timestamps, _values, _latencies, latenesses, missed_slots, _stats = (
        source.drain_with_timing()
    )
    assert request_times_ns == [
        1_000_000_000,
        1_002_000_000,
        1_010_000_000,
        1_012_000_000,
    ]
    assert timestamps.tolist() == pytest.approx([0.0, 0.002, 0.010, 0.012])
    assert latenesses.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert missed_slots.tolist() == pytest.approx([0.0, 0.0, 3.0, 0.0])
