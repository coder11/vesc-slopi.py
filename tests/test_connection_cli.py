from __future__ import annotations

import argparse
import io

import pytest

from vesc_py.connection import VescConnectionKind
from vesc_py.connection_cli import (
    add_vesc_connection_arguments,
    parse_can_id_arg,
    run_vesc_connection_cli,
    resolve_vesc_target_from_args,
)
from vesc_py.models import BleDevice, FwVersion, VescSerialPort


class _TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakeClient:
    def __init__(self) -> None:
        self.fw_version = FwVersion(major=6, minor=5, hw="VESC Express")
        self.closed = False

    def scan_can(self, timeout: float | None = None) -> list[int]:
        assert timeout == pytest.approx(0.5)
        return [7]

    def get_fw_version(
        self,
        can_id: int | None = None,
        timeout: float | None = None,
    ) -> FwVersion:
        assert can_id == 7
        assert timeout == pytest.approx(0.5)
        return FwVersion(major=6, minor=6, hw="ENNOID")

    def close(self) -> None:
        self.closed = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_vesc_connection_arguments(parser)
    return parser


def test_parse_can_id_accepts_decimal_and_hex() -> None:
    assert parse_can_id_arg("12") == 12
    assert parse_can_id_arg("0x0c") == 12


def test_parse_can_id_rejects_out_of_range() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_can_id_arg("254")


def test_resolve_target_uses_explicit_serial_and_can_id() -> None:
    parser = _parser()
    args = parser.parse_args(["--serial", "/dev/ttyACM0", "--can-id", "9"])

    target = resolve_vesc_target_from_args(args, input_stream=io.StringIO(), output_stream=io.StringIO())

    assert target.connection.kind is VescConnectionKind.SERIAL
    assert target.connection.address == "/dev/ttyACM0"
    assert target.can_id == 9


def test_run_connection_cli_parses_args_and_returns_target() -> None:
    target = run_vesc_connection_cli(
        ["--serial", "/dev/ttyACM0", "--can-id", "9"],
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )

    assert target.connection.kind is VescConnectionKind.SERIAL
    assert target.connection.address == "/dev/ttyACM0"
    assert target.can_id == 9


def test_resolve_target_interactively_selects_ble_then_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _parser()
    args = parser.parse_args(["--timeout", "0.5"])
    fake_client = _FakeClient()
    selections = iter([1, 1])

    monkeypatch.setattr(
        "vesc_py.connection_cli.list_serial_ports",
        lambda: [VescSerialPort(name="VESC - /dev/ttyACM0", system_path="/dev/ttyACM0", is_vesc=True)],
    )
    monkeypatch.setattr(
        "vesc_py.connection_cli.ble_scan",
        lambda timeout: [BleDevice(name="VESC BLE", address="AA:BB", rssi=-40)],
    )
    monkeypatch.setattr(
        "vesc_py.connection_cli.connect_client",
        lambda connection, **kwargs: fake_client,
    )
    monkeypatch.setattr(
        "vesc_py.connection_cli._choose_menu_entry",
        lambda title, options: next(selections),
    )

    target = resolve_vesc_target_from_args(
        args,
        input_stream=_TtyStringIO(),
        output_stream=_TtyStringIO(),
    )

    assert target.connection.kind is VescConnectionKind.BLE
    assert target.connection.address == "AA:BB"
    assert target.can_id == 7
    assert fake_client.closed


def test_resolve_target_rejects_multiple_connections_without_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _parser()
    args = parser.parse_args([])

    monkeypatch.setattr(
        "vesc_py.connection_cli.list_serial_ports",
        lambda: [
            VescSerialPort(name="VESC - /dev/ttyACM0", system_path="/dev/ttyACM0", is_vesc=True),
            VescSerialPort(name="VESC - /dev/ttyACM1", system_path="/dev/ttyACM1", is_vesc=True),
        ],
    )
    monkeypatch.setattr("vesc_py.connection_cli.ble_scan", lambda timeout: [])

    with pytest.raises(SystemExit, match="Multiple VESC connections"):
        resolve_vesc_target_from_args(args, input_stream=io.StringIO(), output_stream=io.StringIO())
