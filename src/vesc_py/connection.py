"""Reusable VESC connection descriptions and open helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

import serial  # type: ignore[import-untyped]

from vesc_py.client import SerialTransport, VescClient
from vesc_py.comm_ids import CommPacketId
from vesc_py.imu import build_get_imu_data
from vesc_py.packet import encode_packet
from vesc_py.transport import Transport


class VescConnectionKind(StrEnum):
    """Supported direct connection methods."""

    SERIAL = "serial"
    BLE = "ble"


@dataclass(frozen=True, slots=True)
class VescConnection:
    """Resolved direct connection settings for one VESC transport."""

    kind: VescConnectionKind
    address: str
    baudrate: int = 115200
    exclusive: bool = True
    connect_timeout: float = 10.0
    chunk_size: int = 20

    def __post_init__(self) -> None:
        if not self.address:
            raise ValueError("connection address must not be empty")
        if self.kind is VescConnectionKind.SERIAL and self.baudrate <= 0:
            raise ValueError("baudrate must be greater than 0")
        if self.kind is VescConnectionKind.BLE and self.connect_timeout <= 0.0:
            raise ValueError("connect_timeout must be greater than 0")
        if self.kind is VescConnectionKind.BLE and self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

    @classmethod
    def serial(
        cls,
        port: str,
        *,
        baudrate: int = 115200,
        exclusive: bool = True,
    ) -> VescConnection:
        """Create a serial connection description."""
        return cls(
            kind=VescConnectionKind.SERIAL,
            address=port,
            baudrate=baudrate,
            exclusive=exclusive,
        )

    @classmethod
    def ble(
        cls,
        address: str,
        *,
        connect_timeout: float = 10.0,
        chunk_size: int = 20,
    ) -> VescConnection:
        """Create a BLE connection description."""
        return cls(
            kind=VescConnectionKind.BLE,
            address=address,
            connect_timeout=connect_timeout,
            chunk_size=chunk_size,
        )

    def describe(self) -> str:
        """Return a short human-readable transport label."""
        return f"{self.kind.value} {self.address}"


@dataclass(frozen=True, slots=True)
class VescTarget:
    """One direct connection plus an optional forwarded CAN target."""

    connection: VescConnection
    can_id: int | None = None

    def __post_init__(self) -> None:
        if self.can_id is not None and not 0 <= self.can_id <= 253:
            raise ValueError("can_id must be in range 0..253")

    def describe(self) -> str:
        """Return a short human-readable target label."""
        if self.can_id is None:
            return self.connection.describe()
        return f"{self.connection.describe()} -> can {self.can_id}"


class BlockingIo(Protocol):
    """Small blocking byte-stream surface used by the fast IMU poller."""

    timeout: float | None

    def read(self, size: int = 1) -> bytes:
        """Read up to *size* bytes."""
        ...

    def write(self, data: bytes) -> int | None:
        """Write a bytes payload."""
        ...

    def reset_input_buffer(self) -> None:
        """Drop any buffered unread bytes."""
        ...

    def close(self) -> None:
        """Release the connection resources."""
        ...


class TransportIoAdapter:
    """Adapt a chunked ``Transport`` to a blocking ``read(size)`` interface."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._buffer = bytearray()
        self.timeout: float | None = None

    def read(self, size: int = 1) -> bytes:
        """Read exactly enough chunks to satisfy a blocking poller request."""
        if size <= 0:
            return b""

        deadline: float | None = None
        if self.timeout is not None:
            deadline = time.monotonic() + self.timeout

        while len(self._buffer) < size:
            if deadline is None:
                remaining = 1.0
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break

            chunk = self._transport.recv(remaining)
            if not chunk:
                break
            self._buffer.extend(chunk)

        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def write(self, data: bytes) -> int:
        """Write bytes through the wrapped transport."""
        self._transport.send(data)
        return len(data)

    def reset_input_buffer(self) -> None:
        """Discard already buffered unread bytes."""
        self._buffer.clear()
        while True:
            chunk = self._transport.recv(0.0)
            if not chunk:
                return

    def close(self) -> None:
        """Close the wrapped transport."""
        self._transport.close()


def build_imu_request(mask: int, can_id: int | None = None) -> bytes:
    """Build a framed COMM_GET_IMU_DATA request, optionally forwarded over CAN."""
    payload = build_get_imu_data(mask)
    if can_id is not None:
        if not 0 <= can_id <= 253:
            raise ValueError(f"CAN ID {can_id} out of range [0, 253]")
        payload = bytes([CommPacketId.COMM_FORWARD_CAN, can_id]) + payload
    return encode_packet(payload)


def open_transport(connection: VescConnection) -> Transport:
    """Open a packet transport for the provided direct connection."""
    if connection.kind is VescConnectionKind.SERIAL:
        return SerialTransport(
            connection.address,
            connection.baudrate,
            exclusive=connection.exclusive,
        )

    from vesc_py.ble import BleTransport  # pylint: disable=import-outside-toplevel

    return BleTransport(
        connection.address,
        connect_timeout=connection.connect_timeout,
        chunk_size=connection.chunk_size,
    )


def connect_client(
    connection: VescConnection,
    *,
    timeout: float,
    fw_retries: int = 25,
    config_dir: Path | None = None,
) -> VescClient:
    """Open a ``VescClient`` over a previously described direct connection."""
    if connection.kind is VescConnectionKind.SERIAL:
        return VescClient.connect_serial(
            connection.address,
            connection.baudrate,
            timeout=timeout,
            fw_retries=fw_retries,
            config_dir=config_dir,
        )
    return VescClient.connect_ble(
        connection.address,
        timeout=timeout,
        fw_retries=fw_retries,
        config_dir=config_dir,
        connect_timeout=connection.connect_timeout,
        chunk_size=connection.chunk_size,
    )


def open_blocking_io(
    connection: VescConnection,
    *,
    timeout: float,
) -> BlockingIo:
    """Open a blocking byte-stream interface for fast IMU polling."""
    if timeout <= 0.0:
        raise ValueError("timeout must be greater than 0")

    if connection.kind is VescConnectionKind.SERIAL:
        kwargs: dict[str, Any] = {
            "port": connection.address,
            "baudrate": connection.baudrate,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE,
            "xonxoff": False,
            "rtscts": False,
            "dsrdtr": False,
            "timeout": timeout,
            "write_timeout": timeout,
        }
        try:
            raw_serial = serial.Serial(**kwargs, exclusive=connection.exclusive)
        except TypeError:
            raw_serial = serial.Serial(**kwargs)
        serial_port = cast(BlockingIo, raw_serial)
        serial_port.reset_input_buffer()
        return serial_port

    return TransportIoAdapter(open_transport(connection))


__all__ = [
    "BlockingIo",
    "TransportIoAdapter",
    "VescConnection",
    "VescConnectionKind",
    "VescTarget",
    "build_imu_request",
    "connect_client",
    "open_blocking_io",
    "open_transport",
]
