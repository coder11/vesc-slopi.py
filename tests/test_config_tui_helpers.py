from __future__ import annotations

import argparse

import pytest
from textual.widgets import Select

from examples.config_tui import (
    _connect_and_load,
    _parse_can_id,
    _select_value_is_empty,
    format_value,
    step_numeric_value,
    validate_numeric_text,
)
from vesc_py.config_schema import CfgType, ConfigParam, ConfigSchema
from vesc_py.models import FwVersion


def _tiny_schema(name: str) -> ConfigSchema:
    return ConfigSchema(
        name=name,
        params={
            "value": ConfigParam(
                name="value",
                type=CfgType.INT,
                min_int=0,
                max_int=100,
            )
        },
        ser_order=["value"],
    )


class _FakeConfigClient:
    def __init__(self) -> None:
        self.fw_version = FwVersion(major=6, minor=5, hw="VESC Express T")
        self.appconf_schema = _tiny_schema("express_appconf")
        self.mcconf_schema = None
        self.closed = False
        self.appconf_calls: list[int | None] = []
        self.mcconf_calls: list[int | None] = []

    def get_appconf(
        self,
        *,
        can_id: int | None = None,
        schema: ConfigSchema | None = None,
    ) -> dict[str, object]:
        del schema
        self.appconf_calls.append(can_id)
        if can_id is None:
            raise TimeoutError("direct express appconf timed out")
        return {"value": 2}

    def get_mcconf(
        self,
        *,
        can_id: int | None = None,
        schema: ConfigSchema | None = None,
    ) -> dict[str, object]:
        del schema
        self.mcconf_calls.append(can_id)
        return {"value": 1}

    def get_fw_version(
        self,
        can_id: int | None = None,
        timeout: float | None = None,
    ) -> FwVersion:
        del timeout
        assert can_id == 69
        return FwVersion(major=6, minor=6, hw="ENNOID_150V_modded_by_coder11")

    def config_schemas_for_fw(
        self,
        fw: FwVersion,
    ) -> tuple[ConfigSchema | None, ConfigSchema | None]:
        assert fw.minor == 6
        return _tiny_schema("appconf"), _tiny_schema("mcconf")

    def close(self) -> None:
        self.closed = True


def test_valid_int_text_commits_as_int() -> None:
    param = ConfigParam(type=CfgType.INT, min_int=0, max_int=10)
    result = validate_numeric_text(param, "7")
    assert result.ok
    assert result.value == 7


def test_decimal_text_for_int_is_rejected() -> None:
    param = ConfigParam(type=CfgType.INT, min_int=0, max_int=10)
    result = validate_numeric_text(param, "7.1")
    assert not result.ok


def test_valid_float_text_commits_as_float() -> None:
    param = ConfigParam(type=CfgType.DOUBLE, min_double=-1.0, max_double=10.0)
    result = validate_numeric_text(param, "1.5e1")
    assert not result.ok

    result = validate_numeric_text(param, "1.5e0")
    assert result.ok
    assert result.value == 1.5


def test_non_finite_and_empty_float_rejected() -> None:
    param = ConfigParam(type=CfgType.DOUBLE, min_double=-1.0, max_double=10.0)
    for text in ("", "nan", "inf", "abc"):
        assert not validate_numeric_text(param, text).ok


def test_manual_values_outside_range_rejected() -> None:
    param = ConfigParam(type=CfgType.INT, min_int=2, max_int=4)
    assert not validate_numeric_text(param, "1").ok
    assert not validate_numeric_text(param, "5").ok


def test_keyboard_step_clamps_to_range() -> None:
    param = ConfigParam(type=CfgType.INT, min_int=0, max_int=10, step_int=3)
    value, clamped = step_numeric_value(param, 9, 1)
    assert value == 10
    assert clamped


def test_double_display_respects_decimals() -> None:
    param = ConfigParam(type=CfgType.DOUBLE, decimals_double=3)
    assert format_value(param, 1.23456) == "1.235"


def test_empty_select_sentinels_are_ignored() -> None:
    assert _select_value_is_empty(Select.BLANK)
    assert _select_value_is_empty(Select.NULL)
    assert not _select_value_is_empty(0)


def test_parse_can_id_accepts_decimal_and_hex() -> None:
    assert _parse_can_id("12") == 12
    assert _parse_can_id("0x0c") == 12


def test_parse_can_id_rejects_out_of_range() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_can_id("254")


def test_connect_and_load_skips_optional_express_tab_timeout_for_can_target() -> None:
    client = _FakeConfigClient()

    connected_client, entries = _connect_and_load(
        endpoint="ble AA:BB:CC",
        connect=lambda: client,
        timeout=2.0,
        can_id=69,
    )

    assert connected_client is client
    assert [entry.kind for entry in entries] == ["mcconf", "appconf"]
    assert client.appconf_calls == [None, 69]
    assert client.mcconf_calls == [69]
    assert not client.closed
