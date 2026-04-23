from __future__ import annotations

import pytest

from vesc_py.config_schema import (
    CfgType,
    ConfigParam,
    ConfigSchema,
    VescTx,
    default_values,
    deserialize_config,
    description_html_to_text,
    load_config_xml,
    serialize_config,
)
from vesc_py.config_paths import find_appconf_xml, find_mcconf_xml


def test_load_appconf_schema_with_grouping() -> None:
    schema = load_config_xml(
        find_appconf_xml(6, 6),
        "appconf",
    )

    assert "controller_id" in schema.params
    assert schema.params["controller_id"].long_name == "VESC ID"
    assert schema.params["controller_id"].type == CfgType.INT
    assert schema.ser_order
    assert any(group.name == "General" for group in schema.groups)


def test_load_mcconf_schema_with_grouping_and_separators() -> None:
    schema = load_config_xml(
        find_mcconf_xml(6, 6),
        "mcconf",
    )

    assert "l_current_max" in schema.params
    assert "FOC" in schema.params["motor_type"].enum_names
    assert any(group.name == "FOC" for group in schema.groups)
    assert any(
        item == "::sep::Encoder"
        for group in schema.groups
        for subgroup in group.subgroups
        for item in subgroup.items
    )


def test_description_html_to_text_strips_qt_markup() -> None:
    html = (
        '<!DOCTYPE HTML><html><body><p style="x">'
        '<span style="font-weight:600;">Hello</span> world</p></body></html>'
    )

    text = description_html_to_text(html)

    assert "Hello world" in text
    assert "<p" not in text
    assert "<!DOCTYPE" not in text
    assert "font-weight" not in text


def test_mcconf_roundtrip_defaults() -> None:
    schema = load_config_xml(
        find_mcconf_xml(6, 6),
        "mcconf",
    )

    values = default_values(schema)
    blob = serialize_config(schema, values)
    decoded = deserialize_config(schema, blob)

    assert decoded["motor_type"] == values["motor_type"]
    assert decoded["l_current_max"] == pytest.approx(values["l_current_max"])
    assert decoded["si_motor_poles"] == values["si_motor_poles"]


def test_bad_signature_rejected() -> None:
    schema = ConfigSchema(
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

    blob = serialize_config(schema, {"value": 2})
    with pytest.raises(ValueError, match="Signature mismatch"):
        deserialize_config(schema, b"\x00\x00\x00\x00" + blob[4:])
