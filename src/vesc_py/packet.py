"""VESC packet framing: encode and streaming decode matching packet.cpp."""

from __future__ import annotations

from collections.abc import Iterator

from vesc_py.crc import crc16

MAX_PACKET_LEN = 10000


def encode_packet(payload: bytes | bytearray) -> bytes:
    """Frame *payload* into a VESC wire packet (length prefix + CRC16 + stop)."""
    length = len(payload)
    if length == 0 or length > MAX_PACKET_LEN:
        raise ValueError(f"Payload length {length} out of range [1, {MAX_PACKET_LEN}]")

    header = bytearray()
    if length <= 255:
        header.append(2)
        header.append(length)
    elif length <= 65535:
        header.append(3)
        header.append((length >> 8) & 0xFF)
        header.append(length & 0xFF)
    else:
        header.append(4)
        header.append((length >> 16) & 0xFF)
        header.append((length >> 8) & 0xFF)
        header.append(length & 0xFF)

    crc = crc16(payload)
    trailer = bytes([(crc >> 8) & 0xFF, crc & 0xFF, 3])

    return bytes(header) + bytes(payload) + trailer


class PacketDecoder:
    """Stateful streaming decoder that yields complete payloads from raw bytes."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def process(self, data: bytes | bytearray) -> Iterator[bytes]:
        """Feed raw wire bytes and yield decoded payloads."""
        self._buf.extend(data)

        while True:
            result = self._try_decode()
            if result is None:
                break
            yield result

    def _try_decode(self) -> bytes | None:
        """Try to decode one packet from the front of the buffer.

        Returns the payload on success, or None if more data is needed.
        Discards invalid leading bytes automatically.
        """
        while len(self._buf) > 0:
            start_byte = self._buf[0]

            if start_byte not in (2, 3, 4):
                del self._buf[0]
                continue

            data_start = start_byte  # 2, 3, or 4

            if len(self._buf) < data_start:
                return None

            if start_byte == 2:
                length = self._buf[1]
                if length < 1:
                    del self._buf[0]
                    continue
            elif start_byte == 3:
                length = (self._buf[1] << 8) | self._buf[2]
                if length < 255:
                    del self._buf[0]
                    continue
            else:
                length = (self._buf[1] << 16) | (self._buf[2] << 8) | self._buf[3]
                if length < 65535:
                    del self._buf[0]
                    continue

            if length > MAX_PACKET_LEN:
                del self._buf[0]
                continue

            total = data_start + length + 3  # header + payload + crc(2) + stop(1)
            if len(self._buf) < total:
                return None

            if self._buf[data_start + length + 2] != 3:
                del self._buf[0]
                continue

            payload = bytes(self._buf[data_start : data_start + length])
            crc_calc = crc16(payload)
            crc_rx = (self._buf[data_start + length] << 8) | self._buf[data_start + length + 1]

            if crc_calc != crc_rx:
                del self._buf[0]
                continue

            del self._buf[:total]
            return payload

        return None
