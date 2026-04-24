import math
import time

import pytest

from vesc_py.connection import VescConnection
from vesc_py.fast_imu_source import (
    VescImuSignalSource,
    imu_axis_display_value,
    imu_axis_mask,
    imu_axis_unit,
    parse_imu_axis,
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
    intervals = [later - earlier for earlier, later in zip(write_times, write_times[1:])]
    assert intervals[0] >= 0.045
    assert intervals[1] >= 0.045
