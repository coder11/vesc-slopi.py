"""Synchronous VESC BLE transport backed by an internal asyncio loop."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, TypeVar, cast

from vesc_py.transport import Transport

NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
DEFAULT_BLE_CHUNK_SIZE = 20
DEFAULT_BLE_CONNECT_TIMEOUT = 10.0

_T = TypeVar("_T")


class BleClientProtocol(Protocol):
    """Small bleak client surface used by ``BleTransport``."""

    async def connect(self) -> bool | None:
        """Connect to the BLE peripheral."""

    async def disconnect(self) -> bool | None:
        """Disconnect from the BLE peripheral."""

    async def start_notify(
        self,
        char_specifier: str,
        callback: Callable[[object, bytearray], None],
    ) -> None:
        """Subscribe to characteristic notifications."""

    async def stop_notify(self, char_specifier: str) -> None:
        """Unsubscribe from characteristic notifications."""

    async def write_gatt_char(
        self,
        char_specifier: str,
        data: bytes | bytearray,
        response: bool | None = None,
    ) -> None:
        """Write a GATT characteristic."""


BleClientFactory = Callable[
    [str, Callable[[BleClientProtocol], None] | None, float],
    BleClientProtocol,
]


class _AsyncLoopThread:
    """Own an asyncio event loop on a background thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="vesc-ble-asyncio",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()

    def run(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        timeout: float | None = None,
    ) -> _T:
        """Run a coroutine on the owned event loop and wait for its result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self, timeout: float = 1.0) -> None:
        """Stop the owned event loop and wait for the thread to exit."""
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)


def _create_bleak_client(
    address: str,
    disconnected_callback: Callable[[BleClientProtocol], None] | None,
    timeout: float,
) -> BleClientProtocol:
    try:
        from bleak import BleakClient  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "BLE support requires the 'bleak' package. Run `uv sync` inside "
            "`nix develop` to install project dependencies."
        ) from exc

    return cast(
        BleClientProtocol,
        BleakClient(
            address,
            disconnected_callback=disconnected_callback,
            timeout=timeout,
        ),
    )


class BleTransport(Transport):  # pylint: disable=too-many-instance-attributes
    """Nordic UART Service transport for VESC BLE modules.

    The public transport methods are synchronous to match ``VescClient``. BLE
    GATT operations run on an internal asyncio loop because bleak is async.
    """

    def __init__(
        self,
        address: str,
        *,
        connect_timeout: float = DEFAULT_BLE_CONNECT_TIMEOUT,
        chunk_size: int = DEFAULT_BLE_CHUNK_SIZE,
        client_factory: BleClientFactory | None = None,
    ) -> None:
        if not address:
            raise ValueError("address must not be empty")
        if connect_timeout <= 0.0:
            raise ValueError("connect_timeout must be greater than 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        self._address = address
        self._connect_timeout = connect_timeout
        self._chunk_size = chunk_size
        self._client_factory = client_factory or _create_bleak_client
        self._rx_queue: queue.Queue[bytes] = queue.Queue()
        self._loop_thread = _AsyncLoopThread()
        self._client: BleClientProtocol | None = None
        self._closed = False
        self._disconnected = False

        try:
            self._loop_thread.run(
                self._connect(),
                timeout=connect_timeout + 1.0,
            )
        except Exception:
            self._closed = True
            self._loop_thread.close()
            raise

    @property
    def address(self) -> str:
        """Return the BLE address or OS-specific UUID used to connect."""
        return self._address

    def send(self, data: bytes) -> None:
        """Send raw VESC frame bytes over the NUS RX characteristic."""
        if self._closed or self._disconnected:
            raise RuntimeError("BLE transport is closed")
        if not data:
            return
        self._loop_thread.run(self._write_chunks(data))

    def recv(self, timeout: float) -> bytes:
        """Return notification bytes from the NUS TX characteristic."""
        if (self._closed or self._disconnected) and self._rx_queue.empty():
            return b""
        try:
            return self._rx_queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return b""

    def close(self) -> None:
        """Disconnect from BLE and stop the internal asyncio loop."""
        if self._closed:
            return
        self._closed = True
        try:
            self._loop_thread.run(self._disconnect(), timeout=2.0)
        finally:
            self._loop_thread.close()

    async def _connect(self) -> None:
        self._client = self._client_factory(
            self._address,
            self._handle_disconnect,
            self._connect_timeout,
        )
        await self._client.connect()
        await self._client.start_notify(NUS_TX_UUID, self._handle_notify)

    async def _disconnect(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            await client.stop_notify(NUS_TX_UUID)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        await client.disconnect()
        self._client = None
        self._disconnected = True

    async def _write_chunks(self, data: bytes) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("BLE transport is not connected")

        for offset in range(0, len(data), self._chunk_size):
            chunk = data[offset : offset + self._chunk_size]
            await client.write_gatt_char(NUS_RX_UUID, chunk, response=False)

    def _handle_notify(self, _sender: object, data: bytearray) -> None:
        self._rx_queue.put(bytes(data))

    def _handle_disconnect(self, _client: BleClientProtocol) -> None:
        self._disconnected = True


__all__ = [
    "DEFAULT_BLE_CHUNK_SIZE",
    "DEFAULT_BLE_CONNECT_TIMEOUT",
    "BleClientFactory",
    "BleClientProtocol",
    "BleTransport",
    "NUS_RX_UUID",
    "NUS_SERVICE_UUID",
    "NUS_TX_UUID",
]
