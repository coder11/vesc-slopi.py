from __future__ import annotations

from textual.widgets import Select

from examples.config_tui import (
    _select_value_is_empty,
    format_value,
    step_numeric_value,
    validate_numeric_text,
)
from vesc_py.config_schema import CfgType, ConfigParam


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
