import math

import pytest

from examples.imu_live_plot import (
    ACC_X_INDEX,
    GYRO_Z_INDEX,
    ROLL_INDEX,
    ImuHistory,
    ImuSample,
)
from vesc_py import ImuValues


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
