"""Pydantic models for VESC data structures."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel


class HwType(IntEnum):
    HW_TYPE_VESC = 0
    HW_TYPE_VESC_BMS = 1
    HW_TYPE_CUSTOM_MODULE = 2


class FwVersion(BaseModel):
    major: int = -1
    minor: int = -1
    hw: str = ""
    uuid: bytes = b""
    is_paired: bool = False
    is_test_fw: bool = False
    hw_type: HwType = HwType.HW_TYPE_VESC
    custom_config_num: int = 0
    has_phase_filters: bool = False
    has_qml_hw: bool = False
    qml_hw_fullscreen: bool = False
    has_qml_app: bool = False
    qml_app_fullscreen: bool = False
    nrf_name_supported: bool = False
    nrf_pin_supported: bool = False
    fw_name: str = ""
    hw_conf_crc: int = 0


class ImuValues(BaseModel):
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    vesc_id: int = 0


class VescSerialPort(BaseModel):
    name: str
    system_path: str
    is_vesc: bool = False
    is_esp: bool = False


class UdpDevice(BaseModel):
    hw_name: str
    ip: str
    port: int


class BleDevice(BaseModel):
    name: str
    address: str
    rssi: int | None = None
