"""COMM_GET_IMU_DATA request builder and response parser matching commands.cpp."""

from __future__ import annotations

from vesc_py.buffer import VescBuffer
from vesc_py.comm_ids import CommPacketId
from vesc_py.models import ImuValues

IMU_FIELDS: list[str] = [
    "roll", "pitch", "yaw",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "q0", "q1", "q2", "q3",
]


def build_get_imu_data(mask: int = 0xFFFF) -> bytes:
    """Build a COMM_GET_IMU_DATA request payload."""
    buf = VescBuffer()
    buf.append_uint8(CommPacketId.COMM_GET_IMU_DATA)
    buf.append_uint16(mask & 0xFFFF)
    return buf.to_bytes()


def parse_imu_data(payload: bytes) -> ImuValues:
    """Parse a COMM_GET_IMU_DATA response payload into ImuValues."""
    buf = VescBuffer(payload)
    cmd = buf.pop_uint8()
    if cmd != CommPacketId.COMM_GET_IMU_DATA:
        raise ValueError(f"Expected COMM_GET_IMU_DATA (65), got {cmd}")

    mask = buf.pop_uint16()

    values: dict[str, float] = {}
    for i, field in enumerate(IMU_FIELDS):
        if mask & (1 << i):
            values[field] = buf.pop_double32_auto()

    vesc_id = 0
    if buf.remaining >= 1:
        vesc_id = buf.pop_uint8()

    return ImuValues(vesc_id=vesc_id, **values)
