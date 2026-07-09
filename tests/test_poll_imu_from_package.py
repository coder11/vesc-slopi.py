from __future__ import annotations

import numpy as np
import pytest

from examples.poll_imu_from_package import (
    ACC_X_INDEX,
    GYRO_Z_INDEX,
    DEFAULT_READ_CHUNK_SIZE,
    ImuStreamerHistory,
    build_parser,
)


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
