import argparse
import math

import numpy as np
import pytest

from examples.poll_imu_fast import (
    AxisPlotHistory,
    AxisSampleBuffer,
    build_imu_request,
    build_parser,
    HOST_RX_TIMING_NOTICE,
    HOST_RX_TIMESTAMP_SOURCE,
    IMU_PLOT_CHANNELS,
    ImuPlotHistory,
    ImuSampleBuffer,
    ParsedImu,
    axis_frequency_spectrum,
    field_value_index,
    imu_frequency_spectrum,
    open_poll_connection,
    parse_accel_axis_arg,
    scan_and_print_ble,
    write_csv_header,
    write_sample,
)
from vesc_py import BleDevice
from vesc_py.comm_ids import CommPacketId
from vesc_py.packet import encode_packet


def test_field_value_index_matches_compacted_response_values() -> None:
    mask = 0x0038

    assert field_value_index(mask, "acc_x") == 0
    assert field_value_index(mask, "acc_y") == 1
    assert field_value_index(mask, "acc_z") == 2
    assert field_value_index(mask, "gyro_x") is None


def test_parse_accel_axis_accepts_short_and_field_names() -> None:
    assert parse_accel_axis_arg("x") == "acc_x"
    assert parse_accel_axis_arg("accel-z") == "acc_z"

    with pytest.raises(argparse.ArgumentTypeError):
        parse_accel_axis_arg("roll")


def test_parser_accepts_ble_connection_options() -> None:
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

    assert args.ble == "AA:BB:CC:DD:EE:FF"
    assert args.timeout == pytest.approx(1.5)
    assert args.ble_connect_timeout == pytest.approx(2.5)
    assert args.ble_chunk_size == 64


def test_parser_accepts_can_id() -> None:
    parser = build_parser()

    args = parser.parse_args(["--ble", "AA:BB:CC:DD:EE:FF", "--can-id", "7"])

    assert args.can_id == 7


def test_build_imu_request_wraps_can_forwarding() -> None:
    request = build_imu_request(0x01FF, can_id=7)

    assert request == encode_packet(
        bytes(
            [
                CommPacketId.COMM_FORWARD_CAN,
                7,
                CommPacketId.COMM_GET_IMU_DATA,
                0x01,
                0xFF,
            ]
        )
    )


def test_open_poll_connection_uses_ble_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    fake_transport = object()
    calls: dict[str, object] = {}

    def fake_open_ble(address: str, **kwargs: object) -> object:
        calls["address"] = address
        calls.update(kwargs)
        return fake_transport

    monkeypatch.setattr("examples.poll_imu_fast.open_ble", fake_open_ble)

    transport, label = open_poll_connection(args)

    assert transport is fake_transport
    assert label == "AA:BB:CC:DD:EE:FF"
    assert calls == {
        "address": "AA:BB:CC:DD:EE:FF",
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

    monkeypatch.setattr("examples.poll_imu_fast.ble_scan", fake_ble_scan)

    scan_and_print_ble(timeout=0.25)

    output = capsys.readouterr().out
    assert "VESC BLE  AA:BB:CC:DD:EE:FF  RSSI -51 dBm" in output
    assert "(unnamed)  11:22:33:44:55:66" in output


def test_csv_header_labels_host_receive_timing(capsys: pytest.CaptureFixture[str]) -> None:
    write_csv_header(0x0038)

    assert capsys.readouterr().out == (
        "host_rx_time_s,host_rx_dt_s,vesc_id,rx_mask,acc_x,acc_y,acc_z\n"
    )


def test_human_sample_output_labels_host_receive_timing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = ParsedImu(mask=0x0001, values=(1.25,), vesc_id=None)

    write_sample(
        2,
        timestamp_ns=1_250_000_000,
        start_ns=1_000_000_000,
        previous_ns=1_200_000_000,
        parsed=parsed,
        csv=False,
    )

    output = capsys.readouterr().out
    assert "host_rx_t=0.250000s" in output
    assert "host_rx_dt=0.050000000" in output
    assert "sample=2 t=" not in output


def test_timestamp_source_documents_receive_completion() -> None:
    assert HOST_RX_TIMESTAMP_SOURCE == "host_rx_after_packet"
    assert "configured IMU sample rate" in HOST_RX_TIMING_NOTICE
    assert "latest cached values" in HOST_RX_TIMING_NOTICE
    assert "sample timestamps or sample indexes" in HOST_RX_TIMING_NOTICE


def test_axis_sample_buffer_drains_in_order_after_overwrite() -> None:
    buffer = AxisSampleBuffer(3)

    for index in range(5):
        buffer.append(float(index), float(index + 10))

    timestamps, values, dropped = buffer.drain()

    assert timestamps.tolist() == [2.0, 3.0, 4.0]
    assert values.tolist() == [12.0, 13.0, 14.0]
    assert dropped == 2


def test_axis_sample_buffer_appends_batch_with_one_overwrite_window() -> None:
    buffer = AxisSampleBuffer(4)

    buffer.append(0.0, 10.0)
    buffer.append_many((1.0, 2.0, 3.0, 4.0), (11.0, 12.0, 13.0, 14.0))
    timestamps, values, dropped = buffer.drain()

    assert timestamps.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert values.tolist() == [11.0, 12.0, 13.0, 14.0]
    assert dropped == 1


def test_imu_sample_buffer_drains_channel_matrix_in_order() -> None:
    buffer = ImuSampleBuffer(3)
    values = [
        tuple(float(index + channel) for channel in range(IMU_PLOT_CHANNELS))
        for index in range(5)
    ]

    buffer.append_many(tuple(float(index) for index in range(5)), values)
    timestamps, drained_values, dropped = buffer.drain()

    assert timestamps.tolist() == [2.0, 3.0, 4.0]
    assert drained_values.shape == (IMU_PLOT_CHANNELS, 3)
    assert drained_values[0].tolist() == [2.0, 3.0, 4.0]
    assert drained_values[8].tolist() == [10.0, 11.0, 12.0]
    assert dropped == 2


def test_axis_plot_history_keeps_latest_fixed_width_samples() -> None:
    history = AxisPlotHistory(3)

    history.append_samples(
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([10.0, 11.0], dtype=np.float64),
    )
    history.append_samples(
        np.array([2.0, 3.0], dtype=np.float64),
        np.array([12.0, 13.0], dtype=np.float64),
    )

    assert history.count == 3
    assert history.latest_timestamp == 3.0
    assert history.valid_timestamps().tolist() == [1.0, 2.0, 3.0]
    assert history.valid_values().tolist() == [11.0, 12.0, 13.0]
    assert history.sample_hz() == pytest.approx(1.0)


def test_imu_plot_history_keeps_latest_channel_matrix() -> None:
    history = ImuPlotHistory(3)
    first_values = np.vstack(
        [
            np.array([float(channel), float(channel + 10)], dtype=np.float64)
            for channel in range(IMU_PLOT_CHANNELS)
        ]
    )
    second_values = np.vstack(
        [
            np.array([float(channel + 20), float(channel + 30)], dtype=np.float64)
            for channel in range(IMU_PLOT_CHANNELS)
        ]
    )

    history.append_samples(np.array([0.0, 1.0], dtype=np.float64), first_values)
    history.append_samples(np.array([2.0, 3.0], dtype=np.float64), second_values)

    assert history.count == 3
    assert history.latest_timestamp == 3.0
    assert history.valid_timestamps().tolist() == [1.0, 2.0, 3.0]
    assert history.valid_values()[0].tolist() == [10.0, 20.0, 30.0]
    assert history.valid_values()[8].tolist() == [18.0, 28.0, 38.0]


def test_axis_frequency_spectrum_uses_recent_time_window() -> None:
    timestamps = np.arange(0.0, 4.0, 0.01, dtype=np.float64)
    values = np.sin(2.0 * math.pi * 5.0 * timestamps)

    spectrum = axis_frequency_spectrum(timestamps, values, window_s=2.0)

    assert spectrum is not None
    frequencies, magnitudes, nyquist_hz, sample_count = spectrum
    peak_index = int(np.argmax(magnitudes[1:]) + 1)

    assert sample_count == 200
    assert frequencies[peak_index] == pytest.approx(5.0, abs=0.1)
    assert nyquist_hz == pytest.approx(50.0)


def test_imu_frequency_spectrum_finds_channel_peak() -> None:
    timestamps = np.arange(0.0, 4.0, 0.01, dtype=np.float64)
    values = np.zeros((IMU_PLOT_CHANNELS, timestamps.size), dtype=np.float64)
    values[3] = np.sin(2.0 * math.pi * 7.0 * timestamps)

    spectrum = imu_frequency_spectrum(timestamps, values, window_s=2.0)

    assert spectrum is not None
    frequencies, magnitudes, nyquist_hz, sample_count = spectrum
    peak_index = int(np.argmax(magnitudes[3, 1:]) + 1)

    assert sample_count == 200
    assert frequencies[peak_index] == pytest.approx(7.0, abs=0.1)
    assert nyquist_hz == pytest.approx(50.0)
