from __future__ import annotations

from collections.abc import Callable

import pytest

from vesc_py.ble import (
    NUS_RX_UUID,
    NUS_TX_UUID,
    BleClientProtocol,
    BleTransport,
)


class FakeBleClient:
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False
        self.notify_uuid = ""
        self.stopped_notify_uuid = ""
        self.notify_callback: Callable[[object, bytearray], None] | None = None
        self.writes: list[tuple[str, bytes, bool | None]] = []

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> bool:
        self.disconnected = True
        self.connected = False
        return True

    async def start_notify(
        self,
        char_specifier: str,
        callback: Callable[[object, bytearray], None],
    ) -> None:
        self.notify_uuid = char_specifier
        self.notify_callback = callback

    async def stop_notify(self, char_specifier: str) -> None:
        self.stopped_notify_uuid = char_specifier

    async def write_gatt_char(
        self,
        char_specifier: str,
        data: bytes | bytearray,
        response: bool | None = None,
    ) -> None:
        self.writes.append((char_specifier, bytes(data), response))

    def emit(self, data: bytes) -> None:
        assert self.notify_callback is not None
        self.notify_callback("sender", bytearray(data))


def test_ble_transport_connects_and_subscribes_to_tx_notifications() -> None:
    fake = FakeBleClient()

    def factory(
        address: str,
        disconnected_callback: Callable[[BleClientProtocol], None] | None,
        timeout: float,
    ) -> BleClientProtocol:
        del disconnected_callback
        assert address == "AA:BB:CC"
        assert timeout == 0.2
        return fake

    transport = BleTransport(
        "AA:BB:CC",
        connect_timeout=0.2,
        client_factory=factory,
    )

    try:
        assert fake.connected
        assert fake.notify_uuid == NUS_TX_UUID
    finally:
        transport.close()


def test_ble_transport_chunks_writes_to_rx_characteristic() -> None:
    fake = FakeBleClient()
    transport = BleTransport(
        "AA:BB:CC",
        connect_timeout=0.2,
        chunk_size=20,
        client_factory=lambda _address, _callback, _timeout: fake,
    )

    try:
        transport.send(bytes(range(45)))

        assert fake.writes == [
            (NUS_RX_UUID, bytes(range(20)), False),
            (NUS_RX_UUID, bytes(range(20, 40)), False),
            (NUS_RX_UUID, bytes(range(40, 45)), False),
        ]
    finally:
        transport.close()


def test_ble_transport_recv_returns_notification_bytes() -> None:
    fake = FakeBleClient()
    transport = BleTransport(
        "AA:BB:CC",
        connect_timeout=0.2,
        client_factory=lambda _address, _callback, _timeout: fake,
    )

    try:
        fake.emit(b"\x02\x01\x00")

        assert transport.recv(0.1) == b"\x02\x01\x00"
    finally:
        transport.close()


def test_ble_transport_close_stops_notifications_and_disconnects() -> None:
    fake = FakeBleClient()
    transport = BleTransport(
        "AA:BB:CC",
        connect_timeout=0.2,
        client_factory=lambda _address, _callback, _timeout: fake,
    )

    transport.close()

    assert fake.stopped_notify_uuid == NUS_TX_UUID
    assert fake.disconnected


def test_ble_transport_remote_disconnect_makes_send_fail_but_close_cleans_up() -> None:
    fake = FakeBleClient()
    captured_callback: Callable[[BleClientProtocol], None] | None = None

    def factory(
        _address: str,
        disconnected_callback: Callable[[BleClientProtocol], None] | None,
        _timeout: float,
    ) -> BleClientProtocol:
        nonlocal captured_callback
        captured_callback = disconnected_callback
        return fake

    transport = BleTransport(
        "AA:BB:CC",
        connect_timeout=0.2,
        client_factory=factory,
    )

    assert captured_callback is not None
    captured_callback(fake)

    with pytest.raises(RuntimeError, match="closed"):
        transport.send(b"\x00")

    transport.close()
    assert fake.disconnected
