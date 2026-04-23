from __future__ import annotations

from vesc_py.buffer import VescBuffer
from vesc_py.client import Transport, VescClient
from vesc_py.comm_ids import CommPacketId
from vesc_py.config_schema import CfgType, ConfigParam, ConfigSchema, VescTx, serialize_config
from vesc_py.packet import PacketDecoder, encode_packet


class FakeTransport(Transport):
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = responses
        self.sent: list[bytes] = []

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, timeout: float) -> bytes:
        del timeout
        if not self.responses:
            return b""
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def _decode_sent_payload(frame: bytes) -> bytes:
    payloads = list(PacketDecoder().process(frame))
    assert len(payloads) == 1
    return payloads[0]


def test_scan_can_parses_returned_ids() -> None:
    response = encode_packet(bytes([CommPacketId.COMM_PING_CAN, 2, 7, 42]))
    transport = FakeTransport([response])
    client = VescClient(transport, timeout=0.1)

    assert client.scan_can() == [2, 7, 42]
    assert _decode_sent_payload(transport.sent[0]) == bytes([CommPacketId.COMM_PING_CAN])


def test_get_fw_version_can_wraps_request() -> None:
    response = VescBuffer()
    response.append_uint8(CommPacketId.COMM_FW_VERSION)
    response.append_int8(6)
    response.append_int8(6)
    response.append_string("VESC Express")
    transport = FakeTransport([encode_packet(response.to_bytes())])
    client = VescClient(transport, timeout=0.1)

    fw = client.get_fw_version(can_id=12)

    assert fw.hw == "VESC Express"
    assert _decode_sent_payload(transport.sent[0]) == bytes(
        [
            CommPacketId.COMM_FORWARD_CAN,
            12,
            CommPacketId.COMM_FW_VERSION,
        ]
    )


def _tiny_schema(name: str = "mcconf") -> ConfigSchema:
    return ConfigSchema(
        name=name,
        params={
            "value": ConfigParam(
                name="value",
                type=CfgType.INT,
                vTx=VescTx.INT32,
                val_int=1,
            )
        },
        ser_order=["value"],
    )


def test_get_mcconf_sends_command_and_deserializes() -> None:
    schema = _tiny_schema("mcconf")
    payload = bytes([CommPacketId.COMM_GET_MCCONF]) + serialize_config(schema, {"value": 42})
    transport = FakeTransport([encode_packet(payload)])
    client = VescClient(transport, timeout=0.1)
    client._mcconf_schema = schema

    result = client.get_mcconf()

    assert result["value"] == 42
    assert _decode_sent_payload(transport.sent[0]) == bytes([CommPacketId.COMM_GET_MCCONF])


def test_set_mcconf_waits_for_ack() -> None:
    schema = _tiny_schema("mcconf")
    transport = FakeTransport([encode_packet(bytes([CommPacketId.COMM_SET_MCCONF]))])
    client = VescClient(transport, timeout=0.1)
    client._mcconf_schema = schema

    client.set_mcconf({"value": 7}, wait_ack=True)

    payload = _decode_sent_payload(transport.sent[0])
    assert payload[0] == CommPacketId.COMM_SET_MCCONF


def test_set_appconf_waits_for_ack() -> None:
    schema = _tiny_schema("appconf")
    transport = FakeTransport([encode_packet(bytes([CommPacketId.COMM_SET_APPCONF]))])
    client = VescClient(transport, timeout=0.1)
    client._appconf_schema = schema

    client.set_appconf({"value": 8}, wait_ack=True)

    payload = _decode_sent_payload(transport.sent[0])
    assert payload[0] == CommPacketId.COMM_SET_APPCONF


def test_get_imu_data_skips_queued_appconf_ack() -> None:
    imu_response = VescBuffer()
    imu_response.append_uint8(CommPacketId.COMM_GET_IMU_DATA)
    imu_response.append_uint16(0x0001)
    imu_response.append_double32_auto(1.25)

    transport = FakeTransport(
        [
            encode_packet(bytes([CommPacketId.COMM_SET_APPCONF_NO_STORE])),
            encode_packet(imu_response.to_bytes()),
        ]
    )
    client = VescClient(transport, timeout=0.1)

    imu = client.get_imu_data(0x0001)

    assert imu.roll == 1.25
    assert _decode_sent_payload(transport.sent[0]) == bytes(
        [CommPacketId.COMM_GET_IMU_DATA, 0x00, 0x01]
    )
