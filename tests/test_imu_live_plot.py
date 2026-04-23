import math

import pytest

from examples.imu_live_plot import (
    ACC_X_INDEX,
    GYRO_Z_INDEX,
    ROLL_INDEX,
    build_parser,
    connect_vesc,
    scan_and_print_ble,
    ImuHistory,
    ImuSample,
)
from vesc_py import BleDevice, ImuValues


def _sample(timestamp: float, value: float) -> ImuSample:
    return ImuSample(
        timestamp=timestamp,
        values=ImuValues(
            roll=value,
            pitch=value + 1.0,
            yaw=value + 2.0,
            acc_x=value + 3.0,
            acc_y=value + 4.0,
            acc_z=value + 5.0,
            gyro_x=value + 6.0,
            gyro_y=value + 7.0,
            gyro_z=value + 8.0,
        ),
    )


def test_imu_history_keeps_zero_padded_fixed_width_channels() -> None:
    history = ImuHistory(5)

    history.append_samples([_sample(1.0, 0.1), _sample(1.1, 0.2)], 180.0 / math.pi)

    assert history.count == 2
    assert history.latest_timestamp == 1.1
    assert history.channel(ACC_X_INDEX).tolist() == [0.0, 0.0, 0.0, 3.1, 3.2]
    assert history.valid_timestamps().tolist() == [1.0, 1.1]
    assert history.sample_hz() == pytest.approx(10.0)


def test_imu_history_rolls_over_to_latest_samples() -> None:
    history = ImuHistory(3)

    history.append_samples(
        [_sample(1.0, 1.0), _sample(2.0, 2.0), _sample(3.0, 3.0), _sample(4.0, 4.0)],
        1.0,
    )

    assert history.count == 3
    assert history.valid_timestamps().tolist() == [2.0, 3.0, 4.0]
    assert history.channel(ROLL_INDEX).tolist() == [2.0, 3.0, 4.0]
    assert history.valid_channel(GYRO_Z_INDEX).tolist() == [10.0, 11.0, 12.0]
    assert history.sample_hz() == pytest.approx(1.0)


def test_parser_accepts_ble_connection_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--ble",
            "AA:BB:CC:DD:EE:FF",
            "--can-id",
            "7",
            "--timeout",
            "1.5",
            "--ble-connect-timeout",
            "2.5",
            "--ble-chunk-size",
            "64",
        ]
    )

    assert args.ble == "AA:BB:CC:DD:EE:FF"
    assert args.can_id == 7
    assert args.timeout == pytest.approx(1.5)
    assert args.ble_connect_timeout == pytest.approx(2.5)
    assert args.ble_chunk_size == 64


def test_connect_vesc_uses_ble_options(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ble",
            "AA:BB:CC:DD:EE:FF",
            "--timeout",
            "1.5",
            "--ble-connect-timeout",
            "2.5",
            "--ble-chunk-size",
            "64",
        ]
    )
    fake_client = object()
    calls: dict[str, object] = {}

    def fake_connect_ble(address: str, **kwargs: object) -> object:
        calls["address"] = address
        calls.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        "examples.imu_live_plot.VescClient.connect_ble",
        staticmethod(fake_connect_ble),
    )

    assert connect_vesc(args) is fake_client
    assert calls == {
        "address": "AA:BB:CC:DD:EE:FF",
        "timeout": 1.5,
        "connect_timeout": 2.5,
        "chunk_size": 64,
    }


def test_scan_and_print_ble_lists_discovered_devices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_ble_scan(timeout: float) -> list[BleDevice]:
        assert timeout == 0.25
        return [
            BleDevice(name="VESC BLE", address="AA:BB:CC:DD:EE:FF", rssi=-51),
            BleDevice(name="", address="11:22:33:44:55:66", rssi=None),
        ]

    monkeypatch.setattr("examples.imu_live_plot.ble_scan", fake_ble_scan)

    scan_and_print_ble(timeout=0.25)

    output = capsys.readouterr().out
    assert "VESC BLE  AA:BB:CC:DD:EE:FF  RSSI -51 dBm" in output
    assert "(unnamed)  11:22:33:44:55:66" in output
