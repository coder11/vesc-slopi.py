from vesc_py.crc import crc16, crc32c


def test_crc16_empty() -> None:
    assert crc16(b"") == 0


def test_crc16_single_byte() -> None:
    # CRC-16/XMODEM of b"\x01": table[0 ^ 1] ^ 0 = table[1] = 0x1021
    assert crc16(b"\x01") == 0x1021


def test_crc32c_known_vector() -> None:
    # Standard CRC-32C test vector: crc32c(b"123456789") == 0xE3069283
    assert crc32c(b"123456789") == 0xE3069283
