from vesc_py.appconf import (
    AppConfSchema,
    CfgType,
    ConfigParam,
    VescTx,
    deserialize_appconf,
    serialize_appconf,
    signature,
)


def _make_schema() -> AppConfSchema:
    """Build a small schema for testing."""
    return AppConfSchema(
        params={
            "controller_id": ConfigParam(
                name="controller_id",
                type=CfgType.INT,
                vTx=VescTx.UINT8,
                val_int=74,
            ),
            "timeout_msec": ConfigParam(
                name="timeout_msec",
                type=CfgType.INT,
                vTx=VescTx.UINT32,
                val_int=1000,
            ),
            "app_to_use": ConfigParam(
                name="app_to_use",
                type=CfgType.ENUM,
                val_int=0,
                enum_names=["APP_NONE", "APP_PPM", "APP_ADC"],
            ),
            "max_speed": ConfigParam(
                name="max_speed",
                type=CfgType.DOUBLE,
                vTx=VescTx.DOUBLE32_AUTO,
                val_double=50.0,
            ),
        },
        ser_order=["controller_id", "timeout_msec", "app_to_use", "max_speed"],
    )


def test_appconf_signature_and_roundtrip() -> None:
    schema = _make_schema()
    sig = signature(schema)
    assert isinstance(sig, int)
    assert sig != 0

    values: dict[str, object] = {
        "controller_id": 42,
        "timeout_msec": 2000,
        "app_to_use": 1,
        "max_speed": 25.5,
    }
    blob = serialize_appconf(schema, values)
    result = deserialize_appconf(schema, blob)

    assert result["controller_id"] == 42
    assert result["timeout_msec"] == 2000
    assert result["app_to_use"] == 1
    assert abs(float(result["max_speed"]) - 25.5) < 1e-4  # type: ignore[arg-type]


def test_appconf_rejects_bad_ser_order() -> None:
    schema = _make_schema()
    values: dict[str, object] = {
        "controller_id": 42,
        "timeout_msec": 2000,
        "app_to_use": 1,
        "max_speed": 25.5,
    }
    blob = serialize_appconf(schema, values)

    # Corrupt the signature
    corrupted = b"\x00\x00\x00\x00" + blob[4:]
    try:
        deserialize_appconf(schema, corrupted)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Signature mismatch" in str(e)


def test_appconf_uses_defaults_for_missing_values() -> None:
    schema = _make_schema()
    blob = serialize_appconf(schema, {})
    result = deserialize_appconf(schema, blob)
    assert result["controller_id"] == 74
    assert result["timeout_msec"] == 1000
