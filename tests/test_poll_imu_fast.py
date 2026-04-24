from __future__ import annotations

from io import StringIO
from unittest.mock import ANY

import pytest

from examples.poll_imu_fast import (
    DEFAULT_MASK,
    DEFAULT_TIMEOUT,
    HOST_RX_TIMING_NOTICE,
    ParsedImu,
    TerminalImuDisplay,
    field_names_for_mask,
    main,
    open_poll_connection,
    parse_imu_payload,
    poll_imu,
)
from vesc_py.buffer import VescBuffer
from vesc_py.comm_ids import CommPacketId
from vesc_py.connection import VescConnection, VescTarget
from vesc_py.packet import encode_packet


def _imu_payload(mask: int, values: tuple[float, ...], *, vesc_id: int | None = None) -> bytes:
    buffer = VescBuffer()
    buffer.append_uint8(CommPacketId.COMM_GET_IMU_DATA)
    buffer.append_uint16(mask)
    for value in values:
        buffer.append_double32_auto(value)
    if vesc_id is not None:
        buffer.append_uint8(vesc_id)
    return buffer.to_bytes()


class _FakePollIo:
    def __init__(self, packet: bytes) -> None:
        self.timeout: float | None = None
        self._buffer = bytearray(packet)
        self.requests: list[bytes] = []
        self.closed = False

    def read(self, size: int = 1) -> bytes:
        if not self._buffer:
            return b""
        chunk = bytes(self._buffer[:size])
        del self._buffer[:size]
        return chunk

    def write(self, data: bytes) -> int:
        self.requests.append(data)
        return len(data)

    def reset_input_buffer(self) -> None:
        self._buffer.clear()

    def close(self) -> None:
        self.closed = True


class _FakeTui:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.render_calls: list[dict[str, object]] = []

    def start(self) -> None:
        self.started = True

    def due(self, now_ns: int) -> bool:
        return True

    def render(self, **kwargs: object) -> None:
        self.render_calls.append(kwargs)

    def close(self) -> None:
        self.closed = True


class _TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


def test_field_names_for_mask_returns_wire_order() -> None:
    assert field_names_for_mask(0x0038) == ("acc_x", "acc_y", "acc_z")


def test_parse_imu_payload_decodes_selected_fields_and_vesc_id() -> None:
    payload = _imu_payload(0x0009, (1.25, -0.5), vesc_id=7)

    parsed = parse_imu_payload(payload)

    assert parsed.mask == 0x0009
    assert parsed.values == pytest.approx((1.25, -0.5))
    assert parsed.vesc_id == 7


def test_open_poll_connection_uses_ble_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = VescTarget(
        connection=VescConnection.ble(
            "AA:BB:CC:DD:EE:FF",
            connect_timeout=2.5,
            chunk_size=64,
        )
    )
    fake_transport = object()
    calls: dict[str, object] = {}

    def fake_open_blocking_io(connection: VescConnection, *, timeout: float) -> object:
        calls["connection"] = connection
        calls["timeout"] = timeout
        return fake_transport

    monkeypatch.setattr("examples.poll_imu_fast.open_blocking_io", fake_open_blocking_io)

    transport, label = open_poll_connection(target, timeout=1.5)

    assert transport is fake_transport
    assert label == "AA:BB:CC:DD:EE:FF"
    assert calls == {
        "connection": target.connection,
        "timeout": 1.5,
    }


def test_terminal_imu_display_renders_status_block() -> None:
    stream = StringIO()
    display = TerminalImuDisplay(refresh_hz=10.0, stream=stream)
    parsed = ParsedImu(mask=0x0001, values=(1.25,), vesc_id=69)

    display.start()
    display.render(
        now_ns=2_000_000_000,
        start_ns=1_000_000_000,
        sample_timestamp_ns=2_000_000_000,
        parsed=parsed,
        poll_stats=type("PollStatsObj", (), {"samples": 10, "requests": 10, "timeouts": 0, "parse_errors": 0})(),
        reader_stats=type(
            "ReaderStatsObj",
            (),
            {
                "bad_crc": 0,
                "bad_stop": 0,
                "discarded_bytes": 0,
                "unexpected_packets": 0,
            },
        )(),
        current_rate=1234.5,
        average_rate=1200.0,
    )

    rendered = stream.getvalue()
    assert "VESC IMU fast poller" in rendered
    assert "timing: host_rx_after_packet   sample_age: 0.00 ms" in rendered
    assert "vesc_id: 69   rx_mask: 0x0001" in rendered
    assert "IMU values" in rendered
    assert "roll                 1.25" in rendered


def test_poll_imu_renders_via_tui() -> None:
    payload = _imu_payload(0x0001, (1.25,), vesc_id=3)
    packet = encode_packet(payload)
    serial_port = _FakePollIo(packet)
    status_stream = StringIO()
    tui = _FakeTui()

    poll_imu(
        serial_port,
        request=b"imu-request",
        packet_timeout=1.0,
        status_stream=status_stream,
        tui=tui,
        max_samples=1,
    )

    assert serial_port.requests == [b"imu-request"]
    assert tui.started is True
    assert len(tui.render_calls) == 1
    assert tui.render_calls[0]["parsed"] == ParsedImu(mask=0x0001, values=(1.25,), vesc_id=3)
    assert tui.closed is True
    assert "Done. samples=1 requests=1" in status_stream.getvalue()


def test_main_uses_shared_connection_cli_and_fixed_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    target = VescTarget(VescConnection.serial("/dev/ttyACM0"))
    serial_port = _FakePollIo(b"")
    stderr_stream = _TtyStringIO()

    def fake_run_vesc_connection_cli(argv: object) -> VescTarget:
        calls["argv"] = tuple(argv) if argv is not None else None
        return target

    def fake_build_imu_request(mask: int, can_id: int | None = None) -> bytes:
        calls["request_args"] = (mask, can_id)
        return b"imu-request"

    def fake_open_poll_connection(
        resolved_target: VescTarget,
        *,
        timeout: float,
    ) -> tuple[_FakePollIo, str]:
        calls["open_target"] = resolved_target
        calls["open_timeout"] = timeout
        return serial_port, "/dev/ttyACM0"

    def fake_poll_imu(
        resolved_serial_port: _FakePollIo,
        *,
        request: bytes,
        packet_timeout: float,
        tui: object | None = None,
        **_: object,
    ) -> None:
        calls["poll_args"] = (
            resolved_serial_port,
            request,
            packet_timeout,
            tui,
        )

    monkeypatch.setattr("examples.poll_imu_fast.run_vesc_connection_cli", fake_run_vesc_connection_cli)
    monkeypatch.setattr("examples.poll_imu_fast.build_imu_request", fake_build_imu_request)
    monkeypatch.setattr("examples.poll_imu_fast.open_poll_connection", fake_open_poll_connection)
    monkeypatch.setattr("examples.poll_imu_fast.poll_imu", fake_poll_imu)
    monkeypatch.setattr("examples.poll_imu_fast.sys.stderr", stderr_stream)

    main(["--serial", "/dev/ttyACM0"])

    assert calls["argv"] == ("--serial", "/dev/ttyACM0")
    assert calls["request_args"] == (DEFAULT_MASK, None)
    assert calls["open_target"] == target
    assert calls["open_timeout"] == DEFAULT_TIMEOUT
    assert calls["poll_args"] == (
        serial_port,
        b"imu-request",
        DEFAULT_TIMEOUT,
        ANY,
    )
    assert serial_port.closed is True
    assert "display=tui" in stderr_stream.getvalue()
    assert HOST_RX_TIMING_NOTICE in stderr_stream.getvalue()
