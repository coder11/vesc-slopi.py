"""Device discovery: UDP broadcast listener and serial/BLE scanners."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Coroutine
from typing import Any, TypeVar, cast

import serial.tools.list_ports  # type: ignore[import-untyped]

from vesc_py.ble import NUS_SERVICE_UUID
from vesc_py.models import BleDevice, UdpDevice, VescSerialPort

DISCOVERY_PORT = 65109

_T = TypeVar("_T")


def udp_scan(timeout: float = 3.0) -> list[UdpDevice]:
    """Listen for VESC UDP broadcast announcements on port 65109.

    Blocks for *timeout* seconds, collecting and deduplicating discovered devices.
    Broadcast format is ``name::ip::port`` (ASCII).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", DISCOVERY_PORT))
        sock.settimeout(0.5)

        seen: dict[tuple[str, int], UdpDevice] = {}
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                data, _addr = sock.recvfrom(1024)
            except OSError:
                continue

            text = data.decode("utf-8", errors="replace")
            tokens = text.split("::")
            if len(tokens) == 3:
                hw_name, ip, port_str = tokens
                try:
                    port = int(port_str)
                except ValueError:
                    continue
                key = (ip, port)
                if key not in seen:
                    seen[key] = UdpDevice(hw_name=hw_name, ip=ip, port=port)

        return list(seen.values())
    finally:
        sock.close()


def list_serial_ports() -> list[VescSerialPort]:
    """List serial ports with VESC/ESP heuristics matching vescinterface.cpp.

    VESC ports (STMicroelectronics manufacturer or PID 0x5740) and ESP ports
    (PID 0x1001) are sorted to the front.
    """
    vesc_ports: list[VescSerialPort] = []
    other_ports: list[VescSerialPort] = []

    for port_info in serial.tools.list_ports.comports():
        name: str = port_info.device
        system_path: str = port_info.device
        is_vesc = False
        is_esp = False

        manufacturer = port_info.manufacturer or ""
        pid = port_info.pid

        if manufacturer.startswith("STMicroelectronics") or pid == 0x5740:
            name = f"VESC - {port_info.device}"
            is_vesc = True

        if pid == 0x1001:
            name = f"ESP32 - {port_info.device}"
            is_esp = True

        entry = VescSerialPort(
            name=name,
            system_path=system_path,
            is_vesc=is_vesc,
            is_esp=is_esp,
        )
        if is_vesc or is_esp:
            vesc_ports.append(entry)
        else:
            other_ports.append(entry)

    return vesc_ports + other_ports


def _run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: _T | None = None
    error: BaseException | None = None

    def runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = exc

    thread = threading.Thread(target=runner, name="vesc-ble-scan", daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return cast(_T, result)


async def _ble_scan_async(timeout: float) -> list[BleDevice]:
    try:
        from bleak import BleakScanner  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "BLE scanning requires the 'bleak' package. Run `uv sync` inside "
            "`nix develop` to install project dependencies."
        ) from exc

    devices = await BleakScanner.discover(
        timeout=timeout,
        service_uuids=[NUS_SERVICE_UUID],
    )

    result: list[BleDevice] = []
    seen: set[str] = set()
    for device in devices:
        address = str(getattr(device, "address", ""))
        if not address or address in seen:
            continue
        seen.add(address)
        name_obj = getattr(device, "name", None)
        rssi_obj = getattr(device, "rssi", None)
        result.append(
            BleDevice(
                name="" if name_obj is None else str(name_obj),
                address=address,
                rssi=rssi_obj if isinstance(rssi_obj, int) else None,
            )
        )

    return result


def ble_scan(timeout: float = 5.0) -> list[BleDevice]:
    """Scan for VESC BLE devices advertising the Nordic UART service."""
    if timeout <= 0.0:
        raise ValueError("timeout must be greater than 0")
    return _run_coro_sync(_ble_scan_async(timeout))
