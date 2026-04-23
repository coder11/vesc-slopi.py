import math
import struct

from vesc_py.buffer import VescBuffer


def test_integers_big_endian() -> None:
    buf = VescBuffer()
    buf.append_uint8(0xAB)
    buf.append_int16(-1)
    buf.append_uint32(0xDEADBEEF)

    reader = VescBuffer(buf.to_bytes())
    assert reader.pop_uint8() == 0xAB
    assert reader.pop_int16() == -1
    assert reader.pop_uint32() == 0xDEADBEEF


def test_double32_auto_roundtrip_pi() -> None:
    buf = VescBuffer()
    buf.append_double32_auto(math.pi)
    reader = VescBuffer(buf.to_bytes())
    val = reader.pop_double32_auto()
    assert abs(val - math.pi) < 1e-6


def test_double32_auto_roundtrip_zero() -> None:
    buf = VescBuffer()
    buf.append_double32_auto(0.0)
    reader = VescBuffer(buf.to_bytes())
    assert reader.pop_double32_auto() == 0.0


def test_double32_auto_roundtrip_negative() -> None:
    buf = VescBuffer()
    buf.append_double32_auto(-42.5)
    reader = VescBuffer(buf.to_bytes())
    val = reader.pop_double32_auto()
    assert abs(val - (-42.5)) < 1e-4


def test_pop_string() -> None:
    buf = VescBuffer()
    buf.append_string("hello")
    buf.append_uint8(0xFF)

    reader = VescBuffer(buf.to_bytes())
    assert reader.pop_string() == "hello"
    assert reader.pop_uint8() == 0xFF


def test_double64_auto_roundtrip() -> None:
    buf = VescBuffer()
    buf.append_double64_auto(math.e)
    reader = VescBuffer(buf.to_bytes())
    val = reader.pop_double64_auto()
    assert abs(val - math.e) < 1e-6


def test_scaled_double32() -> None:
    buf = VescBuffer()
    buf.append_double32(1.5, 1000.0)
    reader = VescBuffer(buf.to_bytes())
    val = reader.pop_double32(1000.0)
    assert abs(val - 1.5) < 0.001
