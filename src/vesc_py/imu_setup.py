"""Helpers for the VESC IMU setup workflow.

The functions in this module mirror the calibration math used by the VESC Tool
IMU setup wizard while keeping hardware prompting in examples.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import MutableMapping, SupportsFloat, SupportsInt, cast

from vesc_py.models import ImuValues

IMU_SETUP_MASK = 0x01FF
IMU_TYPE_OFF = 0
IMU_TYPE_INTERNAL = 1
IMU_TYPE_MPU6050 = 2
IMU_TYPE_ICM20948 = 3
IMU_TYPE_BMI160 = 4
IMU_TYPE_LSM6DS3 = 5
IMU_TYPE_NAMES: dict[int, str] = {
    IMU_TYPE_OFF: "Off",
    IMU_TYPE_INTERNAL: "Internal",
    IMU_TYPE_MPU6050: "MPU6050",
    IMU_TYPE_ICM20948: "ICM20948",
    IMU_TYPE_BMI160: "BMI160",
    IMU_TYPE_LSM6DS3: "LSM6DS3",
}

_RAD_TO_DEG = 180.0 / math.pi
_DEG_TO_RAD = math.pi / 180.0
_YAW_SEARCH_STEP = math.radians(15.0)


class ImuBasicProfile(str, Enum):
    """Basic IMU parameter presets from the VESC Tool wizard."""

    DEFAULT = "default"
    LOGS = "logs"
    BALANCE_UNICYCLE = "balance-unicycle"
    BALANCE_SKATEBOARD = "balance-skateboard"


@dataclass(frozen=True)
class AxisTriple:
    """Three values in the VESC IMU roll/x, pitch/y, yaw/z order."""

    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class OrientationRestore:
    """APPCONF fields restored when orientation calibration is cancelled."""

    rot_roll: float
    rot_pitch: float
    rot_yaw: float
    gyro_offsets_0: float
    gyro_offsets_1: float
    gyro_offsets_2: float
    accel_offsets_0: float
    accel_offsets_1: float
    accel_offsets_2: float


REQUIRED_IMU_SETUP_FIELDS = (
    "imu_conf.type",
    "imu_conf.sample_rate_hz",
    "imu_conf.mode",
    "imu_conf.accel_confidence_decay",
    "imu_conf.mahony_kp",
    "imu_conf.accel_lowpass_filter_z",
    "imu_conf.gyro_lowpass_filter",
    "imu_conf.rot_roll",
    "imu_conf.rot_pitch",
    "imu_conf.rot_yaw",
    "imu_conf.accel_offsets__0",
    "imu_conf.accel_offsets__1",
    "imu_conf.accel_offsets__2",
    "imu_conf.gyro_offsets__0",
    "imu_conf.gyro_offsets__1",
    "imu_conf.gyro_offsets__2",
)


@dataclass
# pylint: disable=too-many-instance-attributes
class FilteredImuState:
    """Low-pass filtered IMU state matching the QML setup wizard."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    gyro_x_noise: float = 0.0
    gyro_x_previous: float = 0.0
    max_acc_x: float = -10.0
    max_acc_y: float = -10.0
    max_acc_z: float = -10.0

    def update(self, values: ImuValues) -> None:
        """Apply one IMU sample with the same filters as the Qt wizard."""

        self.roll = values.roll
        self.pitch = values.pitch
        self.yaw = values.yaw
        self.acc_x = self.acc_x * 0.98 + values.acc_x * 0.02
        self.acc_y = self.acc_y * 0.98 + values.acc_y * 0.02
        self.acc_z = self.acc_z * 0.98 + values.acc_z * 0.02
        self.gyro_x = self.gyro_x * 0.99 + values.gyro_x * 0.01
        self.gyro_y = self.gyro_y * 0.99 + values.gyro_y * 0.01
        self.gyro_z = self.gyro_z * 0.99 + values.gyro_z * 0.01
        self.max_acc_x = max(self.max_acc_x, self.acc_x)
        self.max_acc_y = max(self.max_acc_y, self.acc_y)
        self.max_acc_z = max(self.max_acc_z, self.acc_z)

        noise = abs(values.gyro_x - self.gyro_x_previous)
        self.gyro_x_previous = values.gyro_x
        self.gyro_x_noise = self.gyro_x_noise * 0.9 + noise * 0.1

    @property
    def working_imu(self) -> bool:
        """Return the wizard's simple gyro-noise based IMU working check."""

        return self.gyro_x_noise >= 0.01


def _mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count else 0.0


def copy_filtered_imu_state(state: FilteredImuState) -> FilteredImuState:
    """Return a copy of a filtered IMU state."""

    return FilteredImuState(
        roll=state.roll,
        pitch=state.pitch,
        yaw=state.yaw,
        acc_x=state.acc_x,
        acc_y=state.acc_y,
        acc_z=state.acc_z,
        gyro_x=state.gyro_x,
        gyro_y=state.gyro_y,
        gyro_z=state.gyro_z,
        gyro_x_noise=state.gyro_x_noise,
        gyro_x_previous=state.gyro_x_previous,
        max_acc_x=state.max_acc_x,
        max_acc_y=state.max_acc_y,
        max_acc_z=state.max_acc_z,
    )


class RollingMeanImuState:
    """Rolling mean of filtered IMU states over a time window."""

    def __init__(self, window_seconds: float) -> None:
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be greater than 0")
        self._window_seconds = window_seconds
        self._samples: deque[tuple[float, FilteredImuState]] = deque()
        self._max_mean_acc_x = -10.0
        self._max_mean_acc_y = -10.0
        self._max_mean_acc_z = -10.0

    @property
    def sample_count(self) -> int:
        """Number of samples currently in the rolling window."""

        return len(self._samples)

    def update(self, timestamp: float, state: FilteredImuState) -> FilteredImuState:
        """Add a sample and return the current rolling mean state."""

        self._samples.append((timestamp, copy_filtered_imu_state(state)))
        self._drop_old_samples(timestamp)
        mean_state = self.mean_state()
        self._max_mean_acc_x = max(self._max_mean_acc_x, mean_state.acc_x)
        self._max_mean_acc_y = max(self._max_mean_acc_y, mean_state.acc_y)
        self._max_mean_acc_z = max(self._max_mean_acc_z, mean_state.acc_z)
        mean_state.max_acc_x = self._max_mean_acc_x
        mean_state.max_acc_y = self._max_mean_acc_y
        mean_state.max_acc_z = self._max_mean_acc_z
        return mean_state

    def mean_state(self) -> FilteredImuState:
        """Return the mean of the samples in the current window."""

        states = [state for _, state in self._samples]
        if not states:
            return FilteredImuState()

        return FilteredImuState(
            roll=_mean(state.roll for state in states),
            pitch=_mean(state.pitch for state in states),
            yaw=_mean(state.yaw for state in states),
            acc_x=_mean(state.acc_x for state in states),
            acc_y=_mean(state.acc_y for state in states),
            acc_z=_mean(state.acc_z for state in states),
            gyro_x=_mean(state.gyro_x for state in states),
            gyro_y=_mean(state.gyro_y for state in states),
            gyro_z=_mean(state.gyro_z for state in states),
            gyro_x_noise=_mean(state.gyro_x_noise for state in states),
            gyro_x_previous=states[-1].gyro_x_previous,
            max_acc_x=self._max_mean_acc_x,
            max_acc_y=self._max_mean_acc_y,
            max_acc_z=self._max_mean_acc_z,
        )

    def _drop_old_samples(self, timestamp: float) -> None:
        cutoff = timestamp - self._window_seconds
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()


class RollingMeanValue:
    """Rolling mean for a scalar value over a time window."""

    def __init__(self, window_seconds: float) -> None:
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be greater than 0")
        self._window_seconds = window_seconds
        self._samples: deque[tuple[float, float]] = deque()

    @property
    def value(self) -> float:
        """Mean value of the current samples."""

        return _mean(value for _, value in self._samples)

    @property
    def sample_count(self) -> int:
        """Number of samples currently in the rolling window."""

        return len(self._samples)

    def update(self, timestamp: float, value: float) -> float:
        """Add a sample and return the current rolling mean."""

        self._samples.append((timestamp, value))
        self._drop_old_samples(timestamp)
        return self.value

    def _drop_old_samples(self, timestamp: float) -> None:
        cutoff = timestamp - self._window_seconds
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()


@dataclass
class YawOffsetEstimator:
    """Hill-climb yaw estimator used by the VESC Tool orientation page."""

    yaw_offset: float = 0.0

    def update(self, values: ImuValues | AxisTriple) -> None:
        """Update the candidate yaw offset from the latest IMU orientation."""

        current_max_pitch = pitch_for_yaw_offset(values, self.yaw_offset)
        if pitch_for_yaw_offset(values, self.yaw_offset + _YAW_SEARCH_STEP) > current_max_pitch:
            self.yaw_offset += _YAW_SEARCH_STEP
        elif pitch_for_yaw_offset(values, self.yaw_offset - _YAW_SEARCH_STEP) > current_max_pitch:
            self.yaw_offset -= _YAW_SEARCH_STEP

        if self.yaw_offset > math.pi:
            self.yaw_offset = -math.pi
        elif self.yaw_offset < -math.pi:
            self.yaw_offset = math.pi


def config_float(config: Mapping[str, object], name: str) -> float:
    """Read a required floating-point APPCONF value."""

    try:
        return float(cast(SupportsFloat, config[name]))
    except KeyError as exc:
        raise KeyError(f"APPCONF is missing required IMU parameter {name!r}") from exc


def config_int(config: Mapping[str, object], name: str) -> int:
    """Read a required integer APPCONF value."""

    try:
        return int(cast(SupportsInt, config[name]))
    except KeyError as exc:
        raise KeyError(f"APPCONF is missing required parameter {name!r}") from exc


def set_config_float(
    config: MutableMapping[str, object],
    name: str,
    value: float,
) -> None:
    """Set a required floating-point APPCONF value."""

    if name not in config:
        raise KeyError(f"APPCONF is missing required IMU parameter {name!r}")
    config[name] = float(value)


def set_config_int(
    config: MutableMapping[str, object],
    name: str,
    value: int,
) -> None:
    """Set a required integer APPCONF value."""

    if name not in config:
        raise KeyError(f"APPCONF is missing required parameter {name!r}")
    config[name] = int(value)


def imu_type_name(imu_type: int) -> str:
    """Return a display name for a VESC ``imu_conf.type`` enum value."""

    return IMU_TYPE_NAMES.get(imu_type, f"Unknown ({imu_type})")


def mean_label(mean_seconds: float) -> str:
    """Return a short label for displayed calibration values."""

    if mean_seconds > 0.0:
        return f"rolling {mean_seconds:g}s"
    return "instant"


def axis_max(state: FilteredImuState, axis: str) -> float:
    """Return max accelerometer value for an axis."""

    if axis == "x":
        return state.max_acc_x
    if axis == "y":
        return state.max_acc_y
    if axis == "z":
        return state.max_acc_z
    raise ValueError("axis must be 'x', 'y', or 'z'")


def axis_current(state: FilteredImuState, axis: str) -> float:
    """Return current filtered accelerometer value for an axis."""

    if axis == "x":
        return state.acc_x
    if axis == "y":
        return state.acc_y
    if axis == "z":
        return state.acc_z
    raise ValueError("axis must be 'x', 'y', or 'z'")


def apply_basic_profile(
    config: MutableMapping[str, object],
    profile: ImuBasicProfile,
    *,
    imu_type: int | None = None,
) -> None:
    """Apply one of the wizard's basic IMU parameter presets."""

    actual_imu_type = config_int(config, "imu_conf.type") if imu_type is None else imu_type

    if profile in (ImuBasicProfile.DEFAULT, ImuBasicProfile.LOGS):
        sample_rate_hz = 200
    elif actual_imu_type == IMU_TYPE_BMI160:
        sample_rate_hz = 800
    elif actual_imu_type == IMU_TYPE_LSM6DS3:
        sample_rate_hz = 833 if profile == ImuBasicProfile.BALANCE_UNICYCLE else 832
    else:
        sample_rate_hz = 1000

    if profile == ImuBasicProfile.DEFAULT:
        mode = 0
        accel_confidence_decay = 1.0
        mahony_kp = 0.3
        accel_lowpass_filter_z = 0.0
        gyro_lowpass_filter = 0.0
    elif profile == ImuBasicProfile.LOGS:
        mode = 1
        accel_confidence_decay = 1.0
        mahony_kp = 0.3
        accel_lowpass_filter_z = 0.0
        gyro_lowpass_filter = 0.0
    elif profile == ImuBasicProfile.BALANCE_UNICYCLE:
        mode = 1
        accel_confidence_decay = 1.0
        mahony_kp = 0.2
        accel_lowpass_filter_z = 0.0
        gyro_lowpass_filter = 0.0
    else:
        mode = 1
        accel_confidence_decay = 0.02
        mahony_kp = 2.0
        accel_lowpass_filter_z = 1.0
        gyro_lowpass_filter = 0.0

    set_config_int(config, "imu_conf.sample_rate_hz", sample_rate_hz)
    set_config_int(config, "imu_conf.mode", mode)
    set_config_float(config, "imu_conf.accel_confidence_decay", accel_confidence_decay)
    set_config_float(config, "imu_conf.mahony_kp", mahony_kp)
    set_config_float(config, "imu_conf.accel_lowpass_filter_z", accel_lowpass_filter_z)
    set_config_float(config, "imu_conf.gyro_lowpass_filter", gyro_lowpass_filter)


def save_gyro_offsets(
    config: MutableMapping[str, object],
    filtered: FilteredImuState,
) -> None:
    """Add measured gyro offsets to ``imu_conf.gyro_offsets__*``."""

    set_config_float(
        config,
        "imu_conf.gyro_offsets__0",
        config_float(config, "imu_conf.gyro_offsets__0") + filtered.gyro_x,
    )
    set_config_float(
        config,
        "imu_conf.gyro_offsets__1",
        config_float(config, "imu_conf.gyro_offsets__1") + filtered.gyro_y,
    )
    set_config_float(
        config,
        "imu_conf.gyro_offsets__2",
        config_float(config, "imu_conf.gyro_offsets__2") + filtered.gyro_z,
    )


def save_accel_offset(
    config: MutableMapping[str, object],
    axis: str,
    max_value: float,
) -> None:
    """Add one accelerometer max reading to ``imu_conf.accel_offsets__*``."""

    axis_indexes = {"x": 0, "y": 1, "z": 2}
    try:
        index = axis_indexes[axis]
    except KeyError as exc:
        raise ValueError("axis must be 'x', 'y', or 'z'") from exc

    name = f"imu_conf.accel_offsets__{index}"
    set_config_float(config, name, config_float(config, name) + max_value - 1.0)


def _axis_from_values(values: ImuValues | AxisTriple) -> AxisTriple:
    if isinstance(values, AxisTriple):
        return values
    return AxisTriple(values.roll, values.pitch, values.yaw)


def rotate_euler_angles(angles: AxisTriple, rotation: AxisTriple) -> AxisTriple:
    """Rotate a three-axis value with the wizard's yaw/pitch/roll matrix."""
    # pylint: disable=too-many-locals

    if rotation.yaw != 0.0:
        s1 = math.sin(rotation.yaw)
        c1 = math.cos(rotation.yaw)
    else:
        s1 = 0.0
        c1 = 1.0

    if rotation.pitch != 0.0:
        s2 = math.sin(rotation.pitch)
        c2 = math.cos(rotation.pitch)
    else:
        s2 = 0.0
        c2 = 1.0

    if rotation.roll != 0.0:
        s3 = math.sin(rotation.roll)
        c3 = math.cos(rotation.roll)
    else:
        s3 = 0.0
        c3 = 1.0

    m11 = c1 * c2
    m12 = c1 * s2 * s3 - c3 * s1
    m13 = s1 * s3 + c1 * c3 * s2

    m21 = c2 * s1
    m22 = c1 * c3 + s1 * s2 * s3
    m23 = c3 * s1 * s2 - c1 * s3

    m31 = -s2
    m32 = c2 * s3
    m33 = c2 * c3

    return AxisTriple(
        roll=angles.roll * m11 + angles.pitch * m12 + angles.yaw * m13,
        pitch=angles.roll * m21 + angles.pitch * m22 + angles.yaw * m23,
        yaw=angles.roll * m31 + angles.pitch * m32 + angles.yaw * m33,
    )


def apply_rotation_to_calibration_config(
    config: MutableMapping[str, object],
    rotation: AxisTriple,
) -> None:
    """Rotate gyro and accelerometer calibration offsets in APPCONF."""

    gyro_offsets = rotate_euler_angles(
        AxisTriple(
            config_float(config, "imu_conf.gyro_offsets__0"),
            config_float(config, "imu_conf.gyro_offsets__1"),
            config_float(config, "imu_conf.gyro_offsets__2"),
        ),
        rotation,
    )
    accel_offsets = rotate_euler_angles(
        AxisTriple(
            config_float(config, "imu_conf.accel_offsets__0"),
            config_float(config, "imu_conf.accel_offsets__1"),
            config_float(config, "imu_conf.accel_offsets__2"),
        ),
        rotation,
    )

    set_config_float(config, "imu_conf.gyro_offsets__0", gyro_offsets.roll)
    set_config_float(config, "imu_conf.gyro_offsets__1", gyro_offsets.pitch)
    set_config_float(config, "imu_conf.gyro_offsets__2", gyro_offsets.yaw)
    set_config_float(config, "imu_conf.accel_offsets__0", accel_offsets.roll)
    set_config_float(config, "imu_conf.accel_offsets__1", accel_offsets.pitch)
    set_config_float(config, "imu_conf.accel_offsets__2", accel_offsets.yaw)


def prepare_orientation_calibration(config: MutableMapping[str, object]) -> None:
    """Zero orientation and derotate calibration offsets temporarily."""

    rot_roll = config_float(config, "imu_conf.rot_roll")
    rot_pitch = config_float(config, "imu_conf.rot_pitch")
    rot_yaw = config_float(config, "imu_conf.rot_yaw")

    set_config_float(config, "imu_conf.rot_roll", 0.0)
    set_config_float(config, "imu_conf.rot_pitch", 0.0)
    set_config_float(config, "imu_conf.rot_yaw", 0.0)
    apply_rotation_to_calibration_config(config, AxisTriple(0.0, 0.0, -rot_yaw * _DEG_TO_RAD))
    apply_rotation_to_calibration_config(config, AxisTriple(0.0, -rot_pitch * _DEG_TO_RAD, 0.0))
    apply_rotation_to_calibration_config(config, AxisTriple(-rot_roll * _DEG_TO_RAD, 0.0, 0.0))


def apply_roll_offset(config: MutableMapping[str, object], roll: float) -> None:
    """Set roll rotation in radians and rotate existing calibration offsets."""

    set_config_float(config, "imu_conf.rot_roll", roll * _RAD_TO_DEG)
    apply_rotation_to_calibration_config(config, AxisTriple(roll, 0.0, 0.0))


def apply_pitch_offset(config: MutableMapping[str, object], pitch: float) -> None:
    """Set pitch rotation in radians and rotate existing calibration offsets."""

    set_config_float(config, "imu_conf.rot_pitch", pitch * _RAD_TO_DEG)
    apply_rotation_to_calibration_config(config, AxisTriple(0.0, pitch, 0.0))


def apply_yaw_offset(config: MutableMapping[str, object], yaw: float) -> None:
    """Set yaw rotation in radians and rotate existing calibration offsets."""

    set_config_float(config, "imu_conf.rot_yaw", yaw * _RAD_TO_DEG)
    apply_rotation_to_calibration_config(config, AxisTriple(0.0, 0.0, yaw))


def pitch_for_yaw_offset(values: ImuValues | AxisTriple, yaw: float) -> float:
    """Return pitch after applying a candidate yaw offset."""

    return rotate_euler_angles(_axis_from_values(values), AxisTriple(0.0, 0.0, yaw)).pitch


def save_orientation_restore(config: Mapping[str, object]) -> OrientationRestore:
    """Snapshot orientation and calibration fields."""

    return OrientationRestore(
        rot_roll=config_float(config, "imu_conf.rot_roll"),
        rot_pitch=config_float(config, "imu_conf.rot_pitch"),
        rot_yaw=config_float(config, "imu_conf.rot_yaw"),
        gyro_offsets_0=config_float(config, "imu_conf.gyro_offsets__0"),
        gyro_offsets_1=config_float(config, "imu_conf.gyro_offsets__1"),
        gyro_offsets_2=config_float(config, "imu_conf.gyro_offsets__2"),
        accel_offsets_0=config_float(config, "imu_conf.accel_offsets__0"),
        accel_offsets_1=config_float(config, "imu_conf.accel_offsets__1"),
        accel_offsets_2=config_float(config, "imu_conf.accel_offsets__2"),
    )


def restore_orientation_config(
    config: MutableMapping[str, object],
    restore: OrientationRestore,
) -> None:
    """Restore an orientation snapshot to APPCONF."""

    config["imu_conf.rot_roll"] = restore.rot_roll
    config["imu_conf.rot_pitch"] = restore.rot_pitch
    config["imu_conf.rot_yaw"] = restore.rot_yaw
    config["imu_conf.gyro_offsets__0"] = restore.gyro_offsets_0
    config["imu_conf.gyro_offsets__1"] = restore.gyro_offsets_1
    config["imu_conf.gyro_offsets__2"] = restore.gyro_offsets_2
    config["imu_conf.accel_offsets__0"] = restore.accel_offsets_0
    config["imu_conf.accel_offsets__1"] = restore.accel_offsets_1
    config["imu_conf.accel_offsets__2"] = restore.accel_offsets_2


def validate_required_fields(config: Mapping[str, object]) -> None:
    """Fail early if this firmware lacks fields used by the IMU setup wizard."""

    missing = [name for name in REQUIRED_IMU_SETUP_FIELDS if name not in config]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"APPCONF is missing required IMU setup fields: {names}")
