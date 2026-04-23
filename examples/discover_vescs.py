#!/usr/bin/env python3
"""Discover serial/BLE VESCs and their CAN-connected devices.

Usage:
    python discover_vescs.py
    python discover_vescs.py --ble-only
    python discover_vescs.py --can-timeout 8
    python discover_vescs.py --udp-only
"""

from __future__ import annotations

import argparse
import sys

import serial  # type: ignore[import-untyped]

from vesc_py import VescClient, ble_scan, list_serial_ports, udp_scan
from vesc_py.models import BleDevice, FwVersion, HwType, VescSerialPort


def _device_type(fw: FwVersion) -> str:
    """Return the broad VESC device type from firmware metadata."""
    if fw.hw.strip().lower().startswith("vesc express"):
        return "VESC Express"
    if fw.hw_type == HwType.HW_TYPE_VESC_BMS:
        return "VESC BMS"
    if fw.hw_type == HwType.HW_TYPE_CUSTOM_MODULE:
        return "Custom Module"
    return "VESC"


def _fw_label(fw: FwVersion) -> str:
    """Return a compact human-readable firmware/device label."""
    parts = [
        f"Type: {_device_type(fw)}",
        f"HW: {fw.hw or 'Unknown'}",
        f"FW {fw.major}.{fw.minor}",
    ]
    if fw.fw_name:
        parts.append(fw.fw_name)
    if fw.uuid:
        parts.append(f"UUID {fw.uuid.hex()}")
    return " | ".join(parts)


def _print_can_tree(client: VescClient, *, timeout: float, verbose: bool) -> None:
    """Scan CAN IDs and print each node as soon as it is available."""
    print("    CAN:", flush=True)
    try:
        can_ids = client.scan_can(timeout=timeout)
    except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
        if verbose:
            print(f"  CAN scan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"      scan failed: {exc}", flush=True)
        return

    if not can_ids:
        print("      none", flush=True)
        return

    for can_id in can_ids:
        try:
            fw = client.get_fw_version(can_id=can_id, timeout=timeout)
            print(f"      {can_id}: {_fw_label(fw)}", flush=True)
        except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
            print(f"      {can_id}: no firmware response ({exc})", flush=True)


def _ble_device_label(device: BleDevice) -> str:
    """Return a compact BLE device label for console output."""
    name = device.name or "Unknown BLE device"
    rssi = "" if device.rssi is None else f", RSSI {device.rssi} dBm"
    return f"{name} ({device.address}{rssi})"


def _probe_serial_port(
    port: VescSerialPort,
    *,
    timeout: float,
    can_timeout: float,
    fw_retries: int,
    verbose: bool,
) -> bool:
    """Open *port*, print VESC info, and stream its CAN scan results."""
    client: VescClient | None = None
    if verbose:
        print(f"Probing {port.system_path} ({port.name}) ...", file=sys.stderr)
    try:
        client = VescClient.connect_serial(
            port.system_path,
            timeout=timeout,
            fw_retries=fw_retries,
        )
        fw = client.fw_version
        if fw is None:
            fw = client.get_fw_version()
        print(f"{port.system_path}:", flush=True)
        print(f"    {_fw_label(fw)}", flush=True)
        _print_can_tree(client, timeout=can_timeout, verbose=verbose)
        return True
    except (ConnectionError, OSError, serial.SerialException, TimeoutError, ValueError) as exc:
        if verbose:
            print(f"  skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    finally:
        if client is not None:
            client.close()


def _scan_serial_tree(
    timeout: float,
    can_timeout: float,
    fw_retries: int,
    *,
    verbose: bool,
) -> bool:
    """Probe available serial ports and stream all VESC/CAN results."""
    found = False
    ports = list_serial_ports()
    if verbose and not ports:
        print("No serial ports reported by pyserial.", file=sys.stderr)

    # VESC-like ports are listed first by list_serial_ports(); still probe the
    # rest because USB metadata is not always populated on every platform.
    for port in ports:
        found = _probe_serial_port(
            port,
            timeout=timeout,
            can_timeout=can_timeout,
            fw_retries=fw_retries,
            verbose=verbose,
        ) or found

    return found


def _probe_ble_device(
    device: BleDevice,
    *,
    timeout: float,
    can_timeout: float,
    fw_retries: int,
    connect_timeout: float,
    verbose: bool,
) -> bool:
    """Connect to *device*, print VESC info, and stream its CAN scan results."""
    client: VescClient | None = None
    label = _ble_device_label(device)
    if verbose:
        print(f"Probing BLE {label} ...", file=sys.stderr)
    try:
        client = VescClient.connect_ble(
            device.address,
            timeout=timeout,
            fw_retries=fw_retries,
            connect_timeout=connect_timeout,
        )
        fw = client.fw_version
        if fw is None:
            fw = client.get_fw_version()
        print(f"BLE {label}:", flush=True)
        print(f"    {_fw_label(fw)}", flush=True)
        _print_can_tree(client, timeout=can_timeout, verbose=verbose)
        return True
    except TimeoutError:
        if verbose:
            print(f"  skipped: timed out probing BLE {label}", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001 - keep scanning after one failed device.
        if verbose:
            print(f"  skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except TimeoutError:
                print(
                    f"  BLE disconnect timed out for {label}; continuing.",
                    file=sys.stderr,
                    flush=True,
                )


def _scan_ble_tree(
    timeout: float,
    connect_timeout: float,
    response_timeout: float,
    can_timeout: float,
    fw_retries: int,
    *,
    verbose: bool,
) -> bool:
    """Scan BLE advertisements and stream all VESC/CAN results."""
    print(f"BLE (scanning for {timeout}s) ...", flush=True)
    try:
        devices = ble_scan(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - surface scanner setup/backend errors.
        if verbose:
            print(f"  scan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"BLE scan failed: {exc}", flush=True)
        return False

    if not devices:
        print("No BLE devices found.", flush=True)
        return False

    found = False
    for device in devices:
        found = _probe_ble_device(
            device,
            timeout=response_timeout,
            can_timeout=can_timeout,
            fw_retries=fw_retries,
            connect_timeout=connect_timeout,
            verbose=verbose,
        ) or found

    return found


def _print_udp(timeout: float) -> None:
    """Print UDP-discovered VESC Tool bridge announcements."""
    print(f"UDP (listening on port 65109 for {timeout}s) ...")
    devices = udp_scan(timeout=timeout)
    if not devices:
        print("No devices found via UDP.")
        return
    print("UDP devices:")
    for device in devices:
        print(f"  {device.hw_name}  {device.ip}:{device.port}")


def main() -> None:
    """Parse CLI flags and print VESC/CAN discovery results."""
    parser = argparse.ArgumentParser(
        description="Discover serial/BLE VESCs and print their CAN device tree.",
    )
    parser.add_argument(
        "--serial-only",
        action="store_true",
        help="Only scan serial ports.",
    )
    parser.add_argument(
        "--ble-only",
        action="store_true",
        help="Only scan BLE devices advertising the Nordic UART service.",
    )
    parser.add_argument(
        "--udp-only",
        action="store_true",
        help="Only listen for VESC Tool UDP announcements.",
    )
    parser.add_argument(
        "--udp-timeout",
        type=float,
        default=3.0,
        metavar="SEC",
        help="Seconds to listen for UDP broadcasts (default: 3).",
    )
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Response timeout for each controller request (default: 1.0).",
    )
    parser.add_argument(
        "--ble-timeout",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Seconds to scan for BLE advertisements (default: 5.0).",
    )
    parser.add_argument(
        "--ble-connect-timeout",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Seconds to wait for each BLE connection (default: 10.0).",
    )
    parser.add_argument(
        "--can-timeout",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Timeout for CAN scan and CAN firmware queries (default: 5.0).",
    )
    parser.add_argument(
        "--fw-retries",
        type=int,
        default=5,
        metavar="N",
        help="Firmware-version handshake retries per connection (default: 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each probed endpoint and the reason it was skipped.",
    )
    args = parser.parse_args()

    selected_modes = [args.serial_only, args.ble_only, args.udp_only]
    if sum(selected_modes) > 1:
        parser.error("use only one of --serial-only, --ble-only, or --udp-only")

    if args.udp_only:
        _print_udp(args.udp_timeout)
        return

    found = False
    if not args.ble_only:
        found = _scan_serial_tree(
            args.serial_timeout,
            args.can_timeout,
            args.fw_retries,
            verbose=args.verbose,
        )

    if not args.serial_only:
        found = _scan_ble_tree(
            args.ble_timeout,
            args.ble_connect_timeout,
            args.serial_timeout,
            args.can_timeout,
            args.fw_retries,
            verbose=args.verbose,
        ) or found

    if not found:
        print("No VESCs found.")


if __name__ == "__main__":
    main()
