import struct

from vesc_py.buffer import VescBuffer
from vesc_py.comm_ids import CommPacketId
from vesc_py.imu import IMU_FIELDS, build_get_imu_data, parse_imu_data


def test_build_get_imu_data() -> None:
    payload = build_get_imu_data(0x1FF)
    assert payload[0] == CommPacketId.COMM_GET_IMU_DATA
    assert struct.unpack(">H", payload[1:3])[0] == 0x1FF


def test_parse_imu_minimal_mask() -> None:
    """Build a fake response with only roll (bit 0) set."""
    buf = VescBuffer()
    buf.append_uint8(CommPacketId.COMM_GET_IMU_DATA)
    buf.append_uint16(0x0001)  # only roll
    buf.append_double32_auto(1.5)  # roll value
    buf.append_uint8(42)  # vesc_id

    result = parse_imu_data(buf.to_bytes())
    assert abs(result.roll - 1.5) < 1e-5
    assert result.pitch == 0.0
    assert result.vesc_id == 42


def test_parse_imu_full_mask() -> None:
    """Build a fake response with all 16 fields set."""
    buf = VescBuffer()
    buf.append_uint8(CommPacketId.COMM_GET_IMU_DATA)
    buf.append_uint16(0xFFFF)
    for i in range(16):
        buf.append_double32_auto(float(i) * 0.1)

    result = parse_imu_data(buf.to_bytes())
    assert abs(result.roll - 0.0) < 1e-5
    assert abs(result.pitch - 0.1) < 1e-5
    assert abs(result.q3 - 1.5) < 1e-5
