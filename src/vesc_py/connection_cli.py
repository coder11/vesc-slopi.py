"""CLI helpers for selecting a direct VESC connection and optional CAN target."""

from __future__ import annotations

import argparse
import curses
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

from vesc_py import ble_scan, list_serial_ports
from vesc_py.connection import VescConnection, VescTarget, connect_client
from vesc_py.client import VescClient
from vesc_py.models import BleDevice, FwVersion, VescSerialPort


def parse_can_id_arg(value: str) -> int:
    """Parse a CAN target ID from decimal or ``0x`` prefixed text."""
    try:
        can_id = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("CAN ID must be an integer") from exc
    if not 0 <= can_id <= 253:
        raise argparse.ArgumentTypeError("CAN ID must be in range 0..253")
    return can_id


def add_vesc_connection_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_can_id: bool = True,
) -> None:
    """Add reusable VESC direct-connection CLI flags to *parser*."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--serial",
        metavar="PORT",
        help="Connect directly over serial, for example /dev/ttyACM0.",
    )
    group.add_argument(
        "--ble",
        metavar="ADDRESS",
        help="Connect directly over BLE using the device address or OS UUID.",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial baudrate used by --serial (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.1,
        help="VESC response timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--no-exclusive",
        action="store_true",
        help="Disable exclusive pyserial port access.",
    )
    parser.add_argument(
        "--ble-scan-timeout",
        type=float,
        default=5.0,
        metavar="SEC",
        help="BLE scan duration in seconds when interactive discovery is needed.",
    )
    parser.add_argument(
        "--ble-connect-timeout",
        type=float,
        default=10.0,
        metavar="SEC",
        help="BLE connection timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--ble-chunk-size",
        type=int,
        default=20,
        metavar="BYTES",
        help="Maximum BLE Nordic UART write chunk size in bytes (default: %(default)s).",
    )
    parser.add_argument(
        "--force-discovery",
        action="store_true",
        help="Skip cached connection and force interactive discovery.",
    )
    if include_can_id:
        parser.add_argument(
            "--can-id",
            type=parse_can_id_arg,
            metavar="ID",
            help="Forward requests over CAN to the target controller ID.",
        )


@dataclass(frozen=True, slots=True)
class _ConnectionCandidate:
    connection: VescConnection
    label: str


@dataclass(frozen=True, slots=True)
class _CanTargetCandidate:
    can_id: int | None
    label: str


class _HasConnectionArgs(Protocol):
    serial: str | None
    ble: str | None
    force_discovery: bool
    baudrate: int
    timeout: float
    no_exclusive: bool
    ble_scan_timeout: float
    ble_connect_timeout: float
    ble_chunk_size: int
    can_id: int | None


def resolve_vesc_target_from_args(
    args: _HasConnectionArgs,
    *,
    fw_retries: int = 25,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stderr,
) -> VescTarget:
    """Resolve a direct connection and optional CAN target from parsed args."""
    connection = _resolve_connection_from_args(
        args,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    can_id = _resolve_can_target(
        connection,
        args,
        fw_retries=fw_retries,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    _save_cached_connection(connection)
    return VescTarget(connection=connection, can_id=can_id)


def run_vesc_connection_cli(
    argv: Sequence[str] | None = None,
    *,
    include_can_id: bool = True,
    fw_retries: int = 25,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stderr,
) -> VescTarget:
    """Parse VESC connection CLI arguments and return the selected target."""
    parser = argparse.ArgumentParser(
        description="Select a direct VESC connection and optional CAN target.",
    )
    add_vesc_connection_arguments(parser, include_can_id=include_can_id)
    args = parser.parse_args(argv)
    return resolve_vesc_target_from_args(
        args,
        fw_retries=fw_retries,
        input_stream=input_stream,
        output_stream=output_stream,
    )


def _resolve_connection_from_args(
    args: _HasConnectionArgs,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
) -> VescConnection:
    if args.serial is not None:
        connection = VescConnection.serial(
            args.serial,
            baudrate=args.baudrate,
            exclusive=not args.no_exclusive,
        )
        _save_cached_connection(connection)
        return connection
    if args.ble is not None:
        connection = VescConnection.ble(
            args.ble,
            connect_timeout=args.ble_connect_timeout,
            chunk_size=args.ble_chunk_size,
        )
        _save_cached_connection(connection)
        return connection

    if not args.force_discovery:
        cached_connection = _load_cached_connection(args)
        if cached_connection is not None and _can_connect(
            cached_connection,
            timeout=args.timeout,
            output_stream=output_stream,
        ):
            print(
                f"Using cached connection: {cached_connection.describe()}",
                file=output_stream,
            )
            return cached_connection

    candidates = discover_connection_candidates(args, output_stream=output_stream)
    if not candidates:
        raise SystemExit(
            "No VESC serial ports or BLE devices were discovered. "
            "Connect hardware or pass --serial/--ble explicitly."
        )
    if len(candidates) == 1:
        print(f"Using only discovered connection: {candidates[0].label}", file=output_stream)
        return candidates[0].connection
    if not _is_tty(input_stream, output_stream):
        raise SystemExit(
            "Multiple VESC connections were discovered. "
            "Rerun with --serial or --ble to select one explicitly."
        )

    index = _choose_menu_entry(
        "Select VESC connection",
        tuple(candidate.label for candidate in candidates),
    )
    return candidates[index].connection


def discover_connection_candidates(
    args: _HasConnectionArgs,
    *,
    output_stream: TextIO,
) -> list[_ConnectionCandidate]:
    """Discover direct serial and BLE connections using current CLI timeouts."""
    candidates: list[_ConnectionCandidate] = []

    for port in list_serial_ports():
        candidates.append(
            _ConnectionCandidate(
                connection=VescConnection.serial(
                    port.system_path,
                    baudrate=args.baudrate,
                    exclusive=not args.no_exclusive,
                ),
                label=_serial_candidate_label(port),
            )
        )

    try:
        devices = ble_scan(timeout=args.ble_scan_timeout)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"BLE scan skipped: {exc}", file=output_stream)
        devices = []

    for device in devices:
        candidates.append(
            _ConnectionCandidate(
                connection=VescConnection.ble(
                    device.address,
                    connect_timeout=args.ble_connect_timeout,
                    chunk_size=args.ble_chunk_size,
                ),
                label=_ble_candidate_label(device),
            )
        )

    return candidates


def _resolve_can_target(
    connection: VescConnection,
    args: _HasConnectionArgs,
    *,
    fw_retries: int,
    input_stream: TextIO,
    output_stream: TextIO,
) -> int | None:
    if args.can_id is not None:
        return args.can_id

    client: VescClient | None = None
    try:
        client = connect_client(
            connection,
            timeout=args.timeout,
            fw_retries=fw_retries,
        )
        can_ids = client.scan_can(timeout=args.timeout)
        if not can_ids:
            return None

        candidates = [_direct_can_candidate(client.fw_version)]
        candidates.extend(
            _remote_can_candidate(client, can_id, timeout=args.timeout)
            for can_id in can_ids
        )

        if not _is_tty(input_stream, output_stream):
            ids = ", ".join(str(can_id) for can_id in can_ids)
            raise SystemExit(
                f"Connection {connection.describe()} exposes CAN devices {ids}. "
                "Rerun with --can-id to choose one explicitly."
            )

        index = _choose_menu_entry(
            "Select VESC target",
            tuple(candidate.label for candidate in candidates),
        )
        return candidates[index].can_id
    except TimeoutError:
        print(
            (
                f"CAN scan timed out on {connection.describe()}; "
                "using the directly connected controller."
            ),
            file=output_stream,
        )
        return None
    except (ConnectionError, OSError, ValueError) as exc:
        print(
            f"CAN scan failed on {connection.describe()}: {exc}. "
            "Using the directly connected controller.",
            file=output_stream,
        )
        return None
    finally:
        if client is not None:
            client.close()


def _cache_file_path() -> Path:
    state_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_dir / "vescpy" / "connection.json"


def _load_cached_connection(args: _HasConnectionArgs) -> VescConnection | None:
    cache_file = _cache_file_path()
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    kind = payload.get("kind")
    address = payload.get("address")
    if not isinstance(kind, str) or not isinstance(address, str):
        return None

    if kind == "serial":
        return VescConnection.serial(
            address,
            baudrate=args.baudrate,
            exclusive=not args.no_exclusive,
        )
    if kind == "ble":
        return VescConnection.ble(
            address,
            connect_timeout=args.ble_connect_timeout,
            chunk_size=args.ble_chunk_size,
        )
    return None


def _save_cached_connection(connection: VescConnection) -> None:
    if connection.kind.value not in {"serial", "ble"}:
        return
    payload = {"kind": connection.kind.value, "address": connection.address}
    cache_file = _cache_file_path()
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        return


def _can_connect(connection: VescConnection, *, timeout: float, output_stream: TextIO) -> bool:
    client: VescClient | None = None
    try:
        client = connect_client(connection, timeout=timeout)
    except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
        print(
            f"Cached connection failed ({connection.describe()}): {exc}. Falling back to discovery.",
            file=output_stream,
        )
        return False
    finally:
        if client is not None:
            client.close()
    return True


def _direct_can_candidate(fw: FwVersion | None) -> _CanTargetCandidate:
    label = "Directly connected controller"
    if fw is not None:
        label = f"{label} | {_fw_label(fw)}"
    return _CanTargetCandidate(can_id=None, label=label)


def _remote_can_candidate(
    client: VescClient,
    can_id: int,
    *,
    timeout: float,
) -> _CanTargetCandidate:
    label = f"CAN ID {can_id}"
    try:
        fw = client.get_fw_version(can_id=can_id, timeout=timeout)
    except (ConnectionError, OSError, TimeoutError, ValueError):
        return _CanTargetCandidate(can_id=can_id, label=f"{label} | no firmware response")
    return _CanTargetCandidate(can_id=can_id, label=f"{label} | {_fw_label(fw)}")


def _serial_candidate_label(port: VescSerialPort) -> str:
    tags: list[str] = []
    if port.is_vesc:
        tags.append("vesc")
    if port.is_esp:
        tags.append("esp")
    tag_text = f" [{', '.join(tags)}]" if tags else ""
    return f"Serial{tag_text}: {port.system_path}"


def _ble_candidate_label(device: BleDevice) -> str:
    name = device.name or "(unnamed)"
    rssi = "" if device.rssi is None else f" | RSSI {device.rssi} dBm"
    return f"BLE: {name} | {device.address}{rssi}"


def _fw_label(fw: FwVersion) -> str:
    parts = [f"HW {fw.hw or 'Unknown'}", f"FW {fw.major}.{fw.minor}"]
    if fw.fw_name:
        parts.append(fw.fw_name)
    return " | ".join(parts)


def _is_tty(input_stream: TextIO, output_stream: TextIO) -> bool:
    return bool(getattr(input_stream, "isatty", lambda: False)()) and bool(
        getattr(output_stream, "isatty", lambda: False)()
    )


def _choose_menu_entry(title: str, options: tuple[str, ...]) -> int:
    def _run(stdscr: Any) -> int:
        screen = stdscr
        curses.curs_set(0)
        screen.keypad(True)
        selected = 0
        while True:
            screen.erase()
            screen.addstr(0, 0, title)
            screen.addstr(1, 0, "Use up/down arrows and Enter.")
            for index, option in enumerate(options):
                attr = curses.A_REVERSE if index == selected else curses.A_NORMAL
                screen.addstr(index + 3, 0, option, attr)
            screen.refresh()
            key = screen.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = (selected - 1) % len(options)
                continue
            if key in (curses.KEY_DOWN, ord("j")):
                selected = (selected + 1) % len(options)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                return selected
            if key in (27, ord("q")):
                raise SystemExit("Selection cancelled.")

    return curses.wrapper(_run)


def _choose_menu_entry_fallback(title: str, options: tuple[str, ...]) -> int:
    print(title, file=sys.stderr)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}", file=sys.stderr)
    while True:
        raw = input("Select item number: ").strip()
        try:
            selected = int(raw)
        except ValueError:
            continue
        if 1 <= selected <= len(options):
            return selected - 1


__all__ = [
    "add_vesc_connection_arguments",
    "discover_connection_candidates",
    "parse_can_id_arg",
    "run_vesc_connection_cli",
    "resolve_vesc_target_from_args",
]
