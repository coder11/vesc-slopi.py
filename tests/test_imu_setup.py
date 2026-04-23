import math

import pytest

from vesc_py import ImuValues
from vesc_py.imu_setup import (
    AxisTriple,
    FilteredImuState,
    ImuBasicProfile,
    RollingMeanImuState,
    RollingMeanValue,
    YawOffsetEstimator,
    apply_basic_profile,
    apply_pitch_offset,
    apply_roll_offset,
    apply_rotation_to_calibration_config,
    apply_yaw_offset,
    prepare_orientation_calibration,
    rotate_euler_angles,
    save_accel_offset,
    save_gyro_offsets,
)


def _base_config() -> dict[str, object]:
    return {
        "app_to_use": 1,
        "imu_conf.type": 5,
        "imu_conf.sample_rate_hz": 200,
        "imu_conf.mode": 0,
        "imu_conf.accel_confidence_decay": 1.0,
        "imu_conf.mahony_kp": 0.3,
        "imu_conf.accel_lowpass_filter_z": 0.0,
        "imu_conf.gyro_lowpass_filter": 0.0,
        "imu_conf.rot_roll": 10.0,
        "imu_conf.rot_pitch": 20.0,
        "imu_conf.rot_yaw": 30.0,
        "imu_conf.gyro_offsets__0": 1.0,
        "imu_conf.gyro_offsets__1": 2.0,
        "imu_conf.gyro_offsets__2": 3.0,
        "imu_conf.accel_offsets__0": 0.1,
        "imu_conf.accel_offsets__1": 0.2,
        "imu_conf.accel_offsets__2": 0.3,
    }


def test_basic_profiles_match_qml_rates_for_lsm6ds3() -> None:
    config = _base_config()

    apply_basic_profile(config, ImuBasicProfile.BALANCE_UNICYCLE)
    assert config["imu_conf.sample_rate_hz"] == 833
    assert config["imu_conf.mode"] == 1
    assert config["imu_conf.mahony_kp"] == pytest.approx(0.2)

    apply_basic_profile(config, ImuBasicProfile.BALANCE_SKATEBOARD)
    assert config["imu_conf.sample_rate_hz"] == 832
    assert config["imu_conf.accel_confidence_decay"] == pytest.approx(0.02)
    assert config["imu_conf.mahony_kp"] == pytest.approx(2.0)
    assert config["imu_conf.accel_lowpass_filter_z"] == pytest.approx(1.0)


def test_basic_profiles_match_qml_rates_for_bmi160() -> None:
    config = _base_config()
    config["imu_conf.type"] = 4

    apply_basic_profile(config, ImuBasicProfile.BALANCE_UNICYCLE)
    assert config["imu_conf.sample_rate_hz"] == 800

    apply_basic_profile(config, ImuBasicProfile.DEFAULT)
    assert config["imu_conf.sample_rate_hz"] == 200
    assert config["imu_conf.mode"] == 0


def test_filtered_imu_state_matches_wizard_low_pass_update() -> None:
    state = FilteredImuState()
    state.update(
        ImuValues(
            roll=1.0,
            pitch=2.0,
            yaw=3.0,
            acc_x=10.0,
            acc_y=20.0,
            acc_z=30.0,
            gyro_x=100.0,
            gyro_y=200.0,
            gyro_z=300.0,
        )
    )

    assert state.roll == pytest.approx(1.0)
    assert state.acc_x == pytest.approx(0.2)
    assert state.acc_y == pytest.approx(0.4)
    assert state.acc_z == pytest.approx(0.6)
    assert state.gyro_x == pytest.approx(1.0)
    assert state.gyro_y == pytest.approx(2.0)
    assert state.gyro_z == pytest.approx(3.0)
    assert state.max_acc_x == pytest.approx(0.2)
    assert state.working_imu


def test_rolling_mean_imu_state_tracks_recent_window_and_peak_mean_accel() -> None:
    mean = RollingMeanImuState(3.0)

    first = mean.update(
        0.0,
        FilteredImuState(
            roll=1.0,
            pitch=2.0,
            yaw=3.0,
            acc_x=0.2,
            acc_y=0.3,
            acc_z=0.4,
            gyro_x=10.0,
            gyro_y=20.0,
            gyro_z=30.0,
        ),
    )
    assert first.gyro_x == pytest.approx(10.0)
    assert first.max_acc_x == pytest.approx(0.2)

    second = mean.update(
        1.0,
        FilteredImuState(
            roll=3.0,
            pitch=4.0,
            yaw=5.0,
            acc_x=0.6,
            acc_y=0.7,
            acc_z=0.8,
            gyro_x=14.0,
            gyro_y=24.0,
            gyro_z=34.0,
        ),
    )
    assert second.roll == pytest.approx(2.0)
    assert second.acc_x == pytest.approx(0.4)
    assert second.gyro_z == pytest.approx(32.0)
    assert second.max_acc_x == pytest.approx(0.4)

    third = mean.update(
        4.5,
        FilteredImuState(
            roll=5.0,
            pitch=6.0,
            yaw=7.0,
            acc_x=0.1,
            acc_y=0.2,
            acc_z=0.3,
            gyro_x=18.0,
            gyro_y=28.0,
            gyro_z=38.0,
        ),
    )
    assert mean.sample_count == 1
    assert third.roll == pytest.approx(5.0)
    assert third.acc_x == pytest.approx(0.1)
    assert third.max_acc_x == pytest.approx(0.4)


def test_rolling_mean_value_tracks_recent_window() -> None:
    mean = RollingMeanValue(3.0)

    assert mean.update(0.0, 10.0) == pytest.approx(10.0)
    assert mean.update(1.0, 14.0) == pytest.approx(12.0)
    assert mean.update(4.5, 18.0) == pytest.approx(18.0)
    assert mean.sample_count == 1


def test_save_gyro_and_accel_offsets_match_qml_formulas() -> None:
    config = _base_config()
    state = FilteredImuState(gyro_x=0.4, gyro_y=-0.5, gyro_z=0.6)

    save_gyro_offsets(config, state)
    assert config["imu_conf.gyro_offsets__0"] == pytest.approx(1.4)
    assert config["imu_conf.gyro_offsets__1"] == pytest.approx(1.5)
    assert config["imu_conf.gyro_offsets__2"] == pytest.approx(3.6)

    save_accel_offset(config, "x", 1.08)
    assert config["imu_conf.accel_offsets__0"] == pytest.approx(0.18)


def test_rotate_euler_angles_identity_and_yaw() -> None:
    angles = AxisTriple(1.0, 0.0, 0.0)

    assert rotate_euler_angles(angles, AxisTriple(0.0, 0.0, 0.0)) == angles

    rotated = rotate_euler_angles(angles, AxisTriple(0.0, 0.0, math.pi / 2.0))
    assert rotated.roll == pytest.approx(0.0)
    assert rotated.pitch == pytest.approx(1.0)
    assert rotated.yaw == pytest.approx(0.0)


def test_apply_rotation_to_calibration_config_rotates_offsets() -> None:
    config = _base_config()

    apply_rotation_to_calibration_config(config, AxisTriple(0.0, 0.0, math.pi / 2.0))

    assert config["imu_conf.gyro_offsets__0"] == pytest.approx(-2.0)
    assert config["imu_conf.gyro_offsets__1"] == pytest.approx(1.0)
    assert config["imu_conf.gyro_offsets__2"] == pytest.approx(3.0)
    assert config["imu_conf.accel_offsets__0"] == pytest.approx(-0.2)
    assert config["imu_conf.accel_offsets__1"] == pytest.approx(0.1)
    assert config["imu_conf.accel_offsets__2"] == pytest.approx(0.3)


def test_prepare_orientation_calibration_zeroes_rotation_and_derotates_offsets() -> None:
    config = _base_config()
    original_offsets = (
        config["imu_conf.gyro_offsets__0"],
        config["imu_conf.gyro_offsets__1"],
        config["imu_conf.gyro_offsets__2"],
    )

    prepare_orientation_calibration(config)

    assert config["imu_conf.rot_roll"] == pytest.approx(0.0)
    assert config["imu_conf.rot_pitch"] == pytest.approx(0.0)
    assert config["imu_conf.rot_yaw"] == pytest.approx(0.0)
    assert (
        config["imu_conf.gyro_offsets__0"],
        config["imu_conf.gyro_offsets__1"],
        config["imu_conf.gyro_offsets__2"],
    ) != original_offsets


def test_apply_orientation_offsets_store_degrees() -> None:
    config = _base_config()

    apply_roll_offset(config, math.radians(5.0))
    apply_pitch_offset(config, math.radians(-6.0))
    apply_yaw_offset(config, math.radians(7.0))

    assert config["imu_conf.rot_roll"] == pytest.approx(5.0)
    assert config["imu_conf.rot_pitch"] == pytest.approx(-6.0)
    assert config["imu_conf.rot_yaw"] == pytest.approx(7.0)


def test_yaw_offset_estimator_moves_toward_larger_rotated_pitch() -> None:
    estimator = YawOffsetEstimator()

    estimator.update(ImuValues(roll=1.0, pitch=0.0, yaw=0.0))

    assert estimator.yaw_offset == pytest.approx(math.radians(15.0))
