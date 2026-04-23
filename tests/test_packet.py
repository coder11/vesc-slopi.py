from vesc_py.packet import PacketDecoder, encode_packet


def test_encode_decode_short_roundtrip() -> None:
    payload = b"\x00\x01\x02\x03"
    frame = encode_packet(payload)
    decoder = PacketDecoder()
    results = list(decoder.process(frame))
    assert results == [payload]


def test_decoder_incremental() -> None:
    payload = b"\xAA" * 10
    frame = encode_packet(payload)
    decoder = PacketDecoder()

    results: list[bytes] = []
    for byte in frame:
        results.extend(decoder.process(bytes([byte])))
    assert results == [payload]


def test_decoder_skips_garbage() -> None:
    payload = b"\x42"
    frame = encode_packet(payload)
    garbage = b"\xFF\xFE\xFD"
    decoder = PacketDecoder()
    results = list(decoder.process(garbage + frame))
    assert results == [payload]


def test_encode_medium_packet() -> None:
    payload = bytes(range(256)) * 2  # 512 bytes -> uses 3-byte header
    frame = encode_packet(payload)
    assert frame[0] == 3
    decoder = PacketDecoder()
    results = list(decoder.process(frame))
    assert results == [payload]
