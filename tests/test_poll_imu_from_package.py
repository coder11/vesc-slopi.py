from __future__ import annotations

import numpy as np
import pytest

from examples.poll_imu_from_package import (
    ACC_X_INDEX,
    GYRO_Z_INDEX,
    DEFAULT_READ_CHUNK_SIZE,
    IMU_POLL_RATE_CONFIG_KEY,
    ImuStreamerHistory,
    _normalize_imu_poll_rate_hz,
    build_parser,
    read_imu_poll_rate_hz,
    restart_streamer_with_retry,
    wait_for_streamer_startup,
    write_imu_poll_rate_hz,
)
from vesc_py.connection import VescConnection
from vesc_py.imu_streamer import ImuStreamerSourceStats


def test_streamer_history_keeps_latest_samples_in_channel_order() -> None:
    history = ImuStreamerHistory(3)
    timestamps = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    values = np.array(
        [
            [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
            [7.0, 8.0, 9.0, 70.0, 80.0, 90.0],
            [10.0, 11.0, 12.0, 100.0, 110.0, 120.0],
        ],
        dtype=np.float64,
    )

    history.append_batch(timestamps, values)

    assert history.count == 3
    assert history.latest_timestamp == 4.0
    assert history.valid_timestamps().tolist() == [2.0, 3.0, 4.0]
    assert history.channel(ACC_X_INDEX).tolist() == [4.0, 7.0, 10.0]
    assert history.channel(GYRO_Z_INDEX).tolist() == [60.0, 90.0, 120.0]
    assert history.sample_hz() == pytest.approx(1.0)


def test_parser_accepts_direct_connection_options_without_can_id() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--serial",
            "/dev/ttyACM0",
            "--timeout",
            "0.2",
            "--history",
            "1024",
            "--read-chunk-size",
            "8192",
        ]
    )

    assert args.serial == "/dev/ttyACM0"
    assert not hasattr(args, "can_id")
    assert args.timeout == pytest.approx(0.2)
    assert args.history == 1024
    assert args.read_chunk_size == 8192


def test_parser_uses_streamer_read_chunk_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["--serial", "/dev/ttyACM0"])

    assert args.read_chunk_size == DEFAULT_READ_CHUNK_SIZE


def test_read_imu_poll_rate_hz_reads_appconf(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = VescConnection.serial("/dev/ttyACM0")
    calls: dict[str, object] = {}

    class FakeClient:
        def get_appconf(self) -> dict[str, object]:
            return {IMU_POLL_RATE_CONFIG_KEY: 833}

        def close(self) -> None:
            calls["closed"] = True

    def fake_connect_client(
        actual_connection: VescConnection,
        *,
        timeout: float,
    ) -> FakeClient:
        calls["connection"] = actual_connection
        calls["timeout"] = timeout
        return FakeClient()

    monkeypatch.setattr("examples.poll_imu_from_package.connect_client", fake_connect_client)

    assert read_imu_poll_rate_hz(connection, timeout=0.25) == 833
    assert calls == {
        "connection": connection,
        "timeout": 0.25,
        "closed": True,
    }


def test_read_imu_poll_rate_hz_normalizes_legacy_416(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = VescConnection.serial("/dev/ttyACM0")

    class FakeClient:
        def get_appconf(self) -> dict[str, object]:
            return {IMU_POLL_RATE_CONFIG_KEY: 416}

        def close(self) -> None:
            return None

    def fake_connect_client(
        actual_connection: VescConnection,
        *,
        timeout: float,
    ) -> FakeClient:
        return FakeClient()

    monkeypatch.setattr("examples.poll_imu_from_package.connect_client", fake_connect_client)

    assert read_imu_poll_rate_hz(connection, timeout=0.25) == 417


def test_write_imu_poll_rate_hz_updates_and_stores_appconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = VescConnection.serial("/dev/ttyACM0")
    calls: dict[str, object] = {}

    class FakeClient:
        def get_appconf(self) -> dict[str, object]:
            return {
                IMU_POLL_RATE_CONFIG_KEY: 417,
                "app_to_use": 3,
            }

        def set_appconf(
            self,
            values: dict[str, object],
            *,
            store: bool,
            wait_ack: bool,
        ) -> None:
            calls["values"] = values
            calls["store"] = store
            calls["wait_ack"] = wait_ack

        def close(self) -> None:
            calls["closed"] = True

    def fake_connect_client(
        actual_connection: VescConnection,
        *,
        timeout: float,
    ) -> FakeClient:
        calls["connection"] = actual_connection
        calls["timeout"] = timeout
        return FakeClient()

    monkeypatch.setattr("examples.poll_imu_from_package.connect_client", fake_connect_client)

    assert write_imu_poll_rate_hz(connection, timeout=0.4, poll_rate_hz=1666) == 1666
    assert calls == {
        "connection": connection,
        "timeout": 0.4,
        "values": {
            IMU_POLL_RATE_CONFIG_KEY: 1666,
            "app_to_use": 3,
        },
        "store": True,
        "wait_ack": False,
        "closed": True,
    }


def test_write_imu_poll_rate_hz_rejects_unsupported_rate() -> None:
    connection = VescConnection.serial("/dev/ttyACM0")

    with pytest.raises(ValueError, match="unsupported IMU poll rate 1000"):
        write_imu_poll_rate_hz(connection, timeout=0.1, poll_rate_hz=1000)


def test_normalize_imu_poll_rate_hz_maps_416_to_417() -> None:
    assert _normalize_imu_poll_rate_hz(416) == 417
    assert _normalize_imu_poll_rate_hz(417) == 417
    assert _normalize_imu_poll_rate_hz(None) is None


def test_wait_for_streamer_startup_returns_when_streamer_keeps_running() -> None:
    class FakeStreamer:
        def source_stats(self) -> ImuStreamerSourceStats:
            return ImuStreamerSourceStats(
                samples=0,
                dropped=0,
                errors=0,
                average_rate_hz=0.0,
                latest_sample_s=None,
                latest_values=None,
                last_error=None,
                done=False,
                sequence_drops=0,
                timeouts=0,
                idle_reads=0,
                lisp_prints=0,
                last_print=None,
            )

    wait_for_streamer_startup(FakeStreamer(), timeout_s=0.01)


def test_wait_for_streamer_startup_raises_when_streamer_stops() -> None:
    class FakeStreamer:
        def source_stats(self) -> ImuStreamerSourceStats:
            return ImuStreamerSourceStats(
                samples=0,
                dropped=0,
                errors=1,
                average_rate_hz=0.0,
                latest_sample_s=None,
                latest_values=None,
                last_error="timed out waiting for IMU streamer start ack",
                done=True,
                sequence_drops=0,
                timeouts=1,
                idle_reads=0,
                lisp_prints=0,
                last_print=None,
            )

    with pytest.raises(RuntimeError, match="timed out waiting for IMU streamer start ack"):
        wait_for_streamer_startup(FakeStreamer(), timeout_s=0.01)


def test_restart_streamer_with_retry_retries_after_startup_failure() -> None:
    running_stats = ImuStreamerSourceStats(
        samples=0,
        dropped=0,
        errors=0,
        average_rate_hz=0.0,
        latest_sample_s=None,
        latest_values=None,
        last_error=None,
        done=False,
        sequence_drops=0,
        timeouts=0,
        idle_reads=0,
        lisp_prints=0,
        last_print=None,
    )
    failed_stats = ImuStreamerSourceStats(
        samples=0,
        dropped=0,
        errors=1,
        average_rate_hz=0.0,
        latest_sample_s=None,
        latest_values=None,
        last_error="timed out waiting for IMU streamer start ack",
        done=True,
        sequence_drops=0,
        timeouts=1,
        idle_reads=0,
        lisp_prints=0,
        last_print=None,
    )

    class FakeStreamer:
        def __init__(self) -> None:
            self.start_calls = 0
            self.stop_calls = 0

        def start(self) -> None:
            self.start_calls += 1

        def stop(self, timeout: float = 1.0) -> None:
            self.stop_calls += 1

        def source_stats(self) -> ImuStreamerSourceStats:
            return failed_stats if self.start_calls == 1 else running_stats

    streamer = FakeStreamer()

    restart_streamer_with_retry(
        streamer,
        startup_timeout_s=0.01,
        attempts=2,
        retry_delay_s=0.0,
    )

    assert streamer.start_calls == 2
    assert streamer.stop_calls == 1
