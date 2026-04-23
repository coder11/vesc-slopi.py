"""COMM_PACKET_ID subset used by this library, matching datatypes.h."""

from __future__ import annotations

from enum import IntEnum


class CommPacketId(IntEnum):
    COMM_FW_VERSION = 0
    COMM_SET_MCCONF = 13
    COMM_GET_MCCONF = 14
    COMM_GET_MCCONF_DEFAULT = 15
    COMM_SET_APPCONF = 16
    COMM_GET_APPCONF = 17
    COMM_GET_APPCONF_DEFAULT = 18
    COMM_ALIVE = 30
    COMM_FORWARD_CAN = 34
    COMM_PING_CAN = 62
    COMM_GET_IMU_DATA = 65
    COMM_SET_APPCONF_NO_STORE = 149
