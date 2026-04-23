"""Binary serialization buffer matching VByteArray from vbytearray.cpp."""

from __future__ import annotations

import math
import struct


def _round_double(x: float) -> int:
    """Round-away-from-zero, matching the C++ roundDouble helper."""
    if x < 0.0:
        return int(math.ceil(x - 0.5))
    return int(math.floor(x + 0.5))


class VescBuffer:
    """Mutable byte buffer with big-endian VESC serialization methods."""

    __slots__ = ("_buf", "_pos")

    def __init__(self, data: bytes | bytearray = b"") -> None:
        self._buf = bytearray(data)
        self._pos = 0

    # -- raw access ----------------------------------------------------------

    @property
    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def to_bytes(self) -> bytes:
        return bytes(self._buf)

    # -- append (write) ------------------------------------------------------

    def append_int8(self, v: int) -> None:
        self._buf.extend(struct.pack(">b", v))

    def append_uint8(self, v: int) -> None:
        self._buf.extend(struct.pack(">B", v))

    def append_int16(self, v: int) -> None:
        self._buf.extend(struct.pack(">h", v))

    def append_uint16(self, v: int) -> None:
        self._buf.extend(struct.pack(">H", v))

    def append_int32(self, v: int) -> None:
        self._buf.extend(struct.pack(">i", v))

    def append_uint32(self, v: int) -> None:
        self._buf.extend(struct.pack(">I", v))

    def append_double16(self, number: float, scale: float) -> None:
        self.append_int16(_round_double(number * scale))

    def append_double32(self, number: float, scale: float) -> None:
        self.append_int32(_round_double(number * scale))

    def append_double32_auto(self, number: float) -> None:
        if abs(number) < 1.5e-38:
            number = 0.0

        fr, e = math.frexp(number)
        fr_f = float(fr)  # already float in Python, but mirrors C float semantics
        fr_abs = abs(fr_f)
        fr_s = 0

        if fr_abs >= 0.5:
            fr_s = int((fr_abs - 0.5) * 2.0 * 8388608.0) & 0x7FFFFF
            e += 126

        res = ((e & 0xFF) << 23) | (fr_s & 0x7FFFFF)
        if fr_f < 0:
            res |= 1 << 31

        self.append_uint32(res)

    def append_double64_auto(self, number: float) -> None:
        n = struct.unpack("f", struct.pack("f", number))[0]
        err = struct.unpack("f", struct.pack("f", number - n))[0]
        self.append_double32_auto(n)
        self.append_double32_auto(err)

    def append_string(self, s: str) -> None:
        self._buf.extend(s.encode("utf-8"))
        self._buf.append(0)

    # -- pop (read) ----------------------------------------------------------

    def _pop(self, n: int) -> bytes:
        if self.remaining < n:
            raise ValueError(f"Need {n} bytes but only {self.remaining} remain")
        data = bytes(self._buf[self._pos : self._pos + n])
        self._pos += n
        return data

    def pop_int8(self) -> int:
        return int(struct.unpack(">b", self._pop(1))[0])

    def pop_uint8(self) -> int:
        return int(struct.unpack(">B", self._pop(1))[0])

    def pop_int16(self) -> int:
        return int(struct.unpack(">h", self._pop(2))[0])

    def pop_uint16(self) -> int:
        return int(struct.unpack(">H", self._pop(2))[0])

    def pop_int32(self) -> int:
        return int(struct.unpack(">i", self._pop(4))[0])

    def pop_uint32(self) -> int:
        return int(struct.unpack(">I", self._pop(4))[0])

    def pop_double16(self, scale: float) -> float:
        return float(self.pop_int16()) / scale

    def pop_double32(self, scale: float) -> float:
        return float(self.pop_int32()) / scale

    def pop_double32_auto(self) -> float:
        res = self.pop_uint32()

        e = (res >> 23) & 0xFF
        fr = res & 0x7FFFFF
        negative = bool(res & (1 << 31))

        f = 0.0
        if e != 0 or fr != 0:
            f = fr / (8388608.0 * 2.0) + 0.5
            e -= 126

        if negative:
            f = -f

        return math.ldexp(f, e)

    def pop_double64_auto(self) -> float:
        n = self.pop_double32_auto()
        err = self.pop_double32_auto()
        return n + err

    def pop_string(self) -> str:
        try:
            nul = self._buf.index(0, self._pos)
        except ValueError:
            nul = len(self._buf)
        raw = bytes(self._buf[self._pos : nul])
        self._pos = nul + 1
        return raw.decode("utf-8")
