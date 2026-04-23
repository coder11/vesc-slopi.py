"""High-level VESC client with serial, TCP, and BLE transports."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any, Mapping

import serial  # type: ignore[import-untyped]

from vesc_py.buffer import VescBuffer
from vesc_py.comm_ids import CommPacketId
from vesc_py.config_paths import find_appconf_xml, find_mcconf_xml
from vesc_py.config_schema import ConfigSchema, deserialize_config, serialize_config
from vesc_py.fw_version import build_get_fw_version, parse_fw_version
from vesc_py.imu import build_get_imu_data, parse_imu_data
from vesc_py.appconf import load_appconf_xml
from vesc_py.mcconf import load_mcconf_xml
from vesc_py.models import FwVersion, ImuValues
from vesc_py.packet import PacketDecoder, encode_packet
from vesc_py.transport import Transport


class SerialTransport(Transport):
    """pyserial transport at 115200 8N1, no flow control."""

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            timeout=0.1,
        )

    def send(self, data: bytes) -> None:
        self._ser.write(data)

    def recv(self, timeout: float) -> bytes:
        self._ser.timeout = timeout
        data: bytes = self._ser.read(4096)
        return data

    def close(self) -> None:
        self._ser.close()


class TcpTransport(Transport):
    """Plain TCP socket transport."""

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.connect((host, port))

    def send(self, data: bytes) -> None:
        self._sock.sendall(data)

    def recv(self, timeout: float) -> bytes:
        self._sock.settimeout(timeout)
        try:
            return self._sock.recv(4096)
        except OSError:
            return b""

    def close(self) -> None:
        self._sock.close()


class VescClient:
    """High-level client for communicating with a VESC controller.

    Manages packet framing, firmware version handshake, and commands.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        timeout: float = 1.0,
        fw_retries: int = 25,
        config_dir: Path | None = None,
    ) -> None:
        self._transport = transport
        self._decoder = PacketDecoder()
        self._timeout = timeout
        self._fw: FwVersion | None = None
        self._appconf_schema: ConfigSchema | None = None
        self._mcconf_schema: ConfigSchema | None = None
        self._fw_retries = fw_retries
        self._config_dir = config_dir

    @classmethod
    def connect_serial(
        cls,
        port: str,
        baudrate: int = 115200,
        **kwargs: Any,
    ) -> VescClient:
        """Open a serial connection and perform the FW handshake."""
        transport = SerialTransport(port, baudrate)
        client = cls(transport, **kwargs)
        client._handshake()
        return client

    @classmethod
    def connect_tcp(
        cls,
        host: str,
        port: int,
        **kwargs: Any,
    ) -> VescClient:
        """Open a TCP connection and perform the FW handshake."""
        transport = TcpTransport(host, port)
        client = cls(transport, **kwargs)
        client._handshake()
        return client

    @classmethod
    def connect_ble(
        cls,
        address: str,
        *,
        connect_timeout: float = 10.0,
        chunk_size: int = 20,
        client_factory: Any = None,
        **kwargs: Any,
    ) -> VescClient:
        """Open a BLE Nordic UART connection and perform the FW handshake."""
        from vesc_py.ble import BleTransport  # pylint: disable=import-outside-toplevel

        transport = BleTransport(
            address,
            connect_timeout=connect_timeout,
            chunk_size=chunk_size,
            client_factory=client_factory,
        )
        client = cls(transport, **kwargs)
        client._handshake()
        return client

    def close(self) -> None:
        self._transport.close()

    @property
    def fw_version(self) -> FwVersion | None:
        return self._fw

    @property
    def appconf_schema(self) -> ConfigSchema | None:
        return self._appconf_schema

    @property
    def mcconf_schema(self) -> ConfigSchema | None:
        return self._mcconf_schema

    def config_schemas_for_fw(
        self,
        fw: FwVersion,
    ) -> tuple[ConfigSchema | None, ConfigSchema | None]:
        """Load APPCONF/MCCONF schemas matching a firmware version."""
        app_schema: ConfigSchema | None = None
        mc_schema: ConfigSchema | None = None
        try:
            app_xml_path = find_appconf_xml(fw.major, fw.minor, self._config_dir)
            app_schema = load_appconf_xml(app_xml_path)
        except FileNotFoundError:
            pass
        try:
            mc_xml_path = find_mcconf_xml(fw.major, fw.minor, self._config_dir)
            mc_schema = load_mcconf_xml(mc_xml_path)
        except FileNotFoundError:
            pass
        return app_schema, mc_schema

    @staticmethod
    def _forward_can_payload(payload: bytes, can_id: int) -> bytes:
        """Wrap a command payload in COMM_FORWARD_CAN for a target CAN ID."""
        if not 0 <= can_id <= 253:
            raise ValueError(f"CAN ID {can_id} out of range [0, 253]")
        return bytes([CommPacketId.COMM_FORWARD_CAN, can_id]) + payload

    def _send_command(self, payload: bytes) -> None:
        """Encode and send a command payload."""
        self._transport.send(encode_packet(payload))

    def _recv_response(
        self,
        timeout: float | None = None,
        expected_cmds: set[int] | None = None,
    ) -> bytes:
        """Receive and decode exactly one response payload."""
        t = timeout if timeout is not None else self._timeout
        deadline = time.monotonic() + t

        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            raw = self._transport.recv(remaining)
            if raw:
                for payload in self._decoder.process(raw):
                    if expected_cmds is None or (payload and payload[0] in expected_cmds):
                        return payload

        raise TimeoutError("No response from VESC")

    def _handshake(self) -> None:
        """Request firmware version and load matching config XML."""
        for attempt in range(self._fw_retries):
            self._send_command(build_get_fw_version())
            try:
                payload = self._recv_response(
                    timeout=0.2,
                    expected_cmds={int(CommPacketId.COMM_FW_VERSION)},
                )
            except TimeoutError:
                continue

            if len(payload) >= 1 and payload[0] == CommPacketId.COMM_FW_VERSION:
                self._fw = parse_fw_version(payload)
                break
        else:
            raise ConnectionError(
                f"No firmware version response after {self._fw_retries} retries"
            )

        if self._fw.major >= 0 and self._fw.minor >= 0:
            self._appconf_schema, self._mcconf_schema = self.config_schemas_for_fw(self._fw)

    def get_fw_version(
        self,
        can_id: int | None = None,
        timeout: float | None = None,
    ) -> FwVersion:
        """Request firmware version from the VESC."""
        request = build_get_fw_version()
        if can_id is not None:
            request = self._forward_can_payload(request, can_id)
        self._send_command(request)
        payload = self._recv_response(
            timeout=timeout,
            expected_cmds={int(CommPacketId.COMM_FW_VERSION)},
        )
        fw = parse_fw_version(payload)
        if can_id is None:
            self._fw = fw
        return fw

    def scan_can(self, timeout: float | None = None) -> list[int]:
        """Return CAN IDs reported by COMM_PING_CAN."""
        buf = VescBuffer()
        buf.append_uint8(CommPacketId.COMM_PING_CAN)
        self._send_command(buf.to_bytes())
        payload = self._recv_response(
            timeout=timeout,
            expected_cmds={int(CommPacketId.COMM_PING_CAN)},
        )

        if len(payload) < 1 or payload[0] != CommPacketId.COMM_PING_CAN:
            raise ValueError(f"Unexpected response command: {payload[0] if payload else 'empty'}")

        return list(payload[1:])

    def get_imu_data(
        self,
        mask: int = 0xFFFF,
        *,
        can_id: int | None = None,
    ) -> ImuValues:
        """Request IMU data with the given field mask."""
        request = build_get_imu_data(mask)
        if can_id is not None:
            request = self._forward_can_payload(request, can_id)
        self._send_command(request)
        payload = self._recv_response(
            expected_cmds={int(CommPacketId.COMM_GET_IMU_DATA)},
        )
        return parse_imu_data(payload)

    def _get_config(
        self,
        request_cmd: CommPacketId,
        response_cmds: set[CommPacketId],
        schema: ConfigSchema,
        *,
        can_id: int | None = None,
    ) -> dict[str, object]:
        buf = VescBuffer()
        buf.append_uint8(request_cmd)
        request = buf.to_bytes()
        if can_id is not None:
            request = self._forward_can_payload(request, can_id)
        self._send_command(request)
        payload = self._recv_response(expected_cmds={int(cmd) for cmd in response_cmds})

        if len(payload) < 1 or payload[0] not in response_cmds:
            raise ValueError(f"Unexpected response command: {payload[0] if payload else 'empty'}")

        return deserialize_config(schema, payload[1:])

    # pylint: disable-next=too-many-arguments
    def _set_config(
        self,
        request_cmd: CommPacketId,
        schema: ConfigSchema,
        values: Mapping[str, object],
        *,
        can_id: int | None = None,
        wait_ack: bool = False,
    ) -> None:
        blob = serialize_config(schema, values)
        buf = VescBuffer()
        buf.append_uint8(request_cmd)
        buf._buf.extend(blob)
        request = buf.to_bytes()
        if can_id is not None:
            request = self._forward_can_payload(request, can_id)
        self._send_command(request)

        if wait_ack:
            self._recv_response(expected_cmds={int(request_cmd)})

    def get_appconf(
        self,
        *,
        can_id: int | None = None,
        schema: ConfigSchema | None = None,
    ) -> dict[str, object]:
        """Read the current app configuration from the VESC."""
        app_schema = schema or self._appconf_schema
        if app_schema is None:
            raise RuntimeError("No appconf schema loaded (unknown firmware version?)")

        return self._get_config(
            CommPacketId.COMM_GET_APPCONF,
            {CommPacketId.COMM_GET_APPCONF, CommPacketId.COMM_GET_APPCONF_DEFAULT},
            app_schema,
            can_id=can_id,
        )

    # pylint: disable-next=too-many-arguments
    def set_appconf(
        self,
        values: dict[str, object],
        *,
        can_id: int | None = None,
        schema: ConfigSchema | None = None,
        store: bool = True,
        wait_ack: bool = False,
    ) -> None:
        """Write app configuration to the VESC.

        If *store* is False, uses COMM_SET_APPCONF_NO_STORE (temporary).
        """
        app_schema = schema or self._appconf_schema
        if app_schema is None:
            raise RuntimeError("No appconf schema loaded (unknown firmware version?)")

        cmd = (
            CommPacketId.COMM_SET_APPCONF
            if store
            else CommPacketId.COMM_SET_APPCONF_NO_STORE
        )
        self._set_config(
            cmd,
            app_schema,
            values,
            can_id=can_id,
            wait_ack=wait_ack,
        )

    def get_mcconf(
        self,
        *,
        can_id: int | None = None,
        schema: ConfigSchema | None = None,
    ) -> dict[str, object]:
        """Read the current motor configuration from the VESC."""
        mc_schema = schema or self._mcconf_schema
        if mc_schema is None:
            raise RuntimeError("No mcconf schema loaded (unknown firmware version?)")

        return self._get_config(
            CommPacketId.COMM_GET_MCCONF,
            {CommPacketId.COMM_GET_MCCONF, CommPacketId.COMM_GET_MCCONF_DEFAULT},
            mc_schema,
            can_id=can_id,
        )

    def set_mcconf(
        self,
        values: dict[str, object],
        *,
        can_id: int | None = None,
        schema: ConfigSchema | None = None,
        wait_ack: bool = False,
    ) -> None:
        """Write motor configuration to the VESC."""
        mc_schema = schema or self._mcconf_schema
        if mc_schema is None:
            raise RuntimeError("No mcconf schema loaded (unknown firmware version?)")

        self._set_config(
            CommPacketId.COMM_SET_MCCONF,
            mc_schema,
            values,
            can_id=can_id,
            wait_ack=wait_ack,
        )

    def send_alive(self) -> None:
        """Send a COMM_ALIVE keepalive."""
        buf = VescBuffer()
        buf.append_uint8(CommPacketId.COMM_ALIVE)
        self._send_command(buf.to_bytes())


__all__ = [
    "SerialTransport",
    "TcpTransport",
    "Transport",
    "VescClient",
]
