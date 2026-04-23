"""COMM_FW_VERSION request builder and response parser matching commands.cpp."""

from __future__ import annotations

from vesc_py.buffer import VescBuffer
from vesc_py.comm_ids import CommPacketId
from vesc_py.models import FwVersion, HwType


def build_get_fw_version() -> bytes:
    """Build a COMM_FW_VERSION request payload (single byte)."""
    buf = VescBuffer()
    buf.append_uint8(CommPacketId.COMM_FW_VERSION)
    return buf.to_bytes()


def parse_fw_version(payload: bytes) -> FwVersion:
    """Parse a COMM_FW_VERSION response payload into FwVersion."""
    buf = VescBuffer(payload)
    cmd = buf.pop_uint8()
    if cmd != CommPacketId.COMM_FW_VERSION:
        raise ValueError(f"Expected COMM_FW_VERSION (0), got {cmd}")

    fw = FwVersion()

    if buf.remaining >= 2:
        fw.major = buf.pop_int8()
        fw.minor = buf.pop_int8()
        fw.hw = buf.pop_string()

    if buf.remaining >= 12:
        uuid_bytes = buf._pop(12)
        fw.uuid = uuid_bytes

    if buf.remaining >= 1:
        fw.is_paired = bool(buf.pop_int8())

    if buf.remaining >= 1:
        fw.is_test_fw = bool(buf.pop_int8())

    if buf.remaining >= 1:
        fw.hw_type = HwType(buf.pop_int8())

    if buf.remaining >= 1:
        fw.custom_config_num = buf.pop_int8()

    if buf.remaining >= 1:
        fw.has_phase_filters = bool(buf.pop_int8())

    if buf.remaining >= 2:
        qml_hw = buf.pop_int8()
        qml_app = buf.pop_int8()
        fw.has_qml_hw = qml_hw > 0
        fw.qml_hw_fullscreen = qml_hw == 2
        fw.has_qml_app = qml_app > 0
        fw.qml_app_fullscreen = qml_app == 2

    if buf.remaining >= 1:
        nrf_flags = buf.pop_uint8()
        fw.nrf_name_supported = bool(nrf_flags & 1)
        fw.nrf_pin_supported = bool(nrf_flags & 2)

    if buf.remaining >= 1:
        fw.fw_name = buf.pop_string()

    if buf.remaining >= 4:
        fw.hw_conf_crc = buf.pop_uint32()

    return fw
