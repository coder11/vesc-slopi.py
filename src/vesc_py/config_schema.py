"""Generic VESC configuration XML loading and binary serialization."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import IntEnum
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Mapping, SupportsFloat, SupportsInt, Union, cast

from vesc_py.buffer import VescBuffer
from vesc_py.crc import crc32c


class CfgType(IntEnum):
    """ConfigParam type values matching configparam.h."""

    UNDEFINED = 0
    DOUBLE = 1
    INT = 2
    QSTRING = 3
    ENUM = 4
    BOOL = 5
    BITFIELD = 6


class VescTx(IntEnum):
    """VESC binary wire type values matching configparam.h."""

    UNDEFINED = 0
    UINT8 = 1
    INT8 = 2
    UINT16 = 3
    INT16 = 4
    UINT32 = 5
    INT32 = 6
    DOUBLE16 = 7
    DOUBLE32 = 8
    DOUBLE32_AUTO = 9


@dataclass
class ConfigParam:
    """One VESC config parameter definition.

    The legacy camel-case attributes are kept for compatibility with the
    previous APPCONF-only module. Snake-case aliases are provided as properties.
    """

    name: str = ""
    long_name: str = ""
    description_html: str = ""
    description_text: str = ""
    c_define: str = ""
    type: CfgType = CfgType.UNDEFINED
    vTx: VescTx = VescTx.UNDEFINED
    vTx_double_scale: float = 0.0
    val_double: float = 0.0
    val_int: int = 0
    val_string: str = ""
    enum_names: list[str] = field(default_factory=list)
    max_double: float = 99.0
    min_double: float = 0.0
    step_double: float = 1.0
    decimals_double: int = 2
    max_int: int = 99
    min_int: int = 0
    step_int: int = 1
    max_len: int = 0
    suffix: str = ""
    editor_scale: float = 1.0
    edit_as_percentage: bool = False
    show_display: bool = False
    transmittable: bool = True

    @property
    def v_tx(self) -> VescTx:
        return self.vTx

    @v_tx.setter
    def v_tx(self, value: VescTx) -> None:
        self.vTx = value

    @property
    def v_tx_double_scale(self) -> float:
        return self.vTx_double_scale

    @v_tx_double_scale.setter
    def v_tx_double_scale(self, value: float) -> None:
        self.vTx_double_scale = value


@dataclass
class ConfigSubgroup:
    name: str
    items: list[str] = field(default_factory=list)


@dataclass
class ConfigGroup:
    name: str
    subgroups: list[ConfigSubgroup] = field(default_factory=list)


@dataclass
class ConfigSchema:
    name: Literal["appconf", "mcconf"] | str = "appconf"
    xml_path: Path = Path()
    params: dict[str, ConfigParam] = field(default_factory=dict)
    ser_order: list[str] = field(default_factory=list)
    groups: list[ConfigGroup] = field(default_factory=list)


class _DescriptionParser(HTMLParser):
    """Small HTML-to-text parser for the Qt rich text stored in config XML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"p", "br", "li"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def text(self) -> str:
        raw = unescape("".join(self._parts))
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r" *\n *", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def description_html_to_text(html: str) -> str:
    """Convert Qt rich text from config XML to readable plain text."""

    parser = _DescriptionParser()
    parser.feed(html)
    parser.close()
    return parser.text()


def _text(el: ET.Element) -> str:
    return el.text or ""


def _to_bool(text: str) -> bool:
    return bool(int(text))


def load_config_xml(path: Path, config_name: str) -> ConfigSchema:
    """Parse a VESC ``parameters_*conf.xml`` file."""

    tree = ET.parse(path)
    root = tree.getroot()
    schema = ConfigSchema(name=config_name, xml_path=path)

    params_el = root.find("Params")
    if params_el is not None:
        for param_el in params_el:
            p = ConfigParam(name=param_el.tag)

            for child in param_el:
                tag = child.tag
                text = _text(child)

                if tag == "longName":
                    p.long_name = text
                elif tag == "description":
                    p.description_html = text
                    p.description_text = description_html_to_text(text)
                elif tag == "cDefine":
                    p.c_define = text
                elif tag == "type":
                    p.type = CfgType(int(text))
                elif tag == "vTx":
                    p.vTx = VescTx(int(text))
                elif tag == "vTxDoubleScale":
                    p.vTx_double_scale = float(text)
                elif tag == "valDouble":
                    p.val_double = float(text)
                elif tag == "valInt":
                    p.val_int = int(text)
                elif tag == "valString":
                    p.val_string = text
                elif tag == "enumNames":
                    p.enum_names.append(text)
                elif tag == "maxDouble":
                    p.max_double = float(text)
                elif tag == "minDouble":
                    p.min_double = float(text)
                elif tag == "stepDouble":
                    p.step_double = float(text)
                elif tag == "editorDecimalsDouble":
                    p.decimals_double = int(text)
                elif tag == "maxInt":
                    p.max_int = int(text)
                elif tag == "minInt":
                    p.min_int = int(text)
                elif tag == "stepInt":
                    p.step_int = int(text)
                elif tag == "maxLen":
                    p.max_len = int(text)
                elif tag == "suffix":
                    p.suffix = text
                elif tag == "editorScale":
                    p.editor_scale = float(text)
                elif tag == "editAsPercentage":
                    p.edit_as_percentage = _to_bool(text)
                elif tag == "showDisplay":
                    p.show_display = _to_bool(text)
                elif tag == "transmittable":
                    p.transmittable = _to_bool(text)

            schema.params[p.name] = p

    ser_order_el = root.find("SerOrder")
    if ser_order_el is not None:
        for ser_el in ser_order_el:
            if ser_el.tag == "ser" and ser_el.text:
                schema.ser_order.append(ser_el.text)

    grouping_el = root.find("Grouping")
    if grouping_el is not None:
        for group_el in grouping_el.findall("group"):
            group_name_el = group_el.find("groupName")
            group = ConfigGroup(name=_text(group_name_el) if group_name_el is not None else "")

            for subgroup_el in group_el.findall("subgroup"):
                subgroup_name_el = subgroup_el.find("subgroupName")
                subgroup = ConfigSubgroup(
                    name=_text(subgroup_name_el) if subgroup_name_el is not None else ""
                )
                params_parent = subgroup_el.find("subgroupParams")
                if params_parent is not None:
                    for param_name_el in params_parent.findall("param"):
                        if param_name_el.text:
                            subgroup.items.append(param_name_el.text)
                group.subgroups.append(subgroup)

            schema.groups.append(group)

    return schema


def signature(schema: ConfigSchema) -> int:
    """Compute the config CRC32C signature matching ConfigParams::getSignature."""

    sig_str = ""
    for name in schema.ser_order:
        sig_str += name
        p = schema.params.get(name)
        if p is not None:
            sig_str += str(int(p.type))
            sig_str += str(int(p.vTx))
            for en in p.enum_names:
                sig_str += en

    return crc32c(sig_str.encode("utf-8"))


def _as_float(val: object) -> float:
    return float(cast(SupportsFloat, val))


def _as_int(val: object) -> int:
    return int(cast(SupportsInt, val))


def default_values(schema: ConfigSchema) -> dict[str, object]:
    """Return default values from XML for every serialized parameter."""

    values: dict[str, object] = {}
    for name in schema.ser_order:
        p = schema.params[name]
        if p.type == CfgType.DOUBLE:
            values[name] = p.val_double
        elif p.type == CfgType.QSTRING:
            values[name] = p.val_string
        else:
            values[name] = p.val_int
    return values


def serialize_config(schema: ConfigSchema, values: Mapping[str, object]) -> bytes:
    """Serialize config values into a VESC binary blob."""

    buf = VescBuffer()
    buf.append_uint32(signature(schema))

    for name in schema.ser_order:
        p = schema.params.get(name)
        if p is None:
            raise KeyError(f"Parameter {name!r} in SerOrder not found in schema")

        val = values.get(name)

        if p.type == CfgType.DOUBLE:
            fval = _as_float(val) if val is not None else p.val_double
            if not math.isfinite(fval):
                raise ValueError(f"Parameter {name!r}: value must be finite")
            if p.vTx == VescTx.DOUBLE16:
                buf.append_double16(fval, p.vTx_double_scale)
            elif p.vTx == VescTx.DOUBLE32:
                buf.append_double32(fval, p.vTx_double_scale)
            elif p.vTx == VescTx.DOUBLE32_AUTO:
                buf.append_double32_auto(fval)
            else:
                raise ValueError(f"Parameter {name!r}: unsupported vTx {p.vTx} for DOUBLE")

        elif p.type == CfgType.INT:
            ival = _as_int(val) if val is not None else p.val_int
            if p.vTx == VescTx.UINT8:
                buf.append_uint8(ival)
            elif p.vTx == VescTx.INT8:
                buf.append_int8(ival)
            elif p.vTx == VescTx.UINT16:
                buf.append_uint16(ival)
            elif p.vTx == VescTx.INT16:
                buf.append_int16(ival)
            elif p.vTx == VescTx.UINT32:
                buf.append_uint32(ival)
            elif p.vTx == VescTx.INT32:
                buf.append_int32(ival)
            else:
                raise ValueError(f"Parameter {name!r}: unsupported vTx {p.vTx} for INT")

        elif p.type == CfgType.QSTRING:
            sval = str(val) if val is not None else p.val_string
            buf.append_string(sval)

        elif p.type in (CfgType.ENUM, CfgType.BOOL, CfgType.BITFIELD):
            ival = _as_int(val) if val is not None else p.val_int
            buf.append_int8(ival)

        elif p.type == CfgType.UNDEFINED:
            raise ValueError(f"Parameter {name!r}: type is UNDEFINED")

    return buf.to_bytes()


def deserialize_config(schema: ConfigSchema, data: bytes) -> dict[str, object]:
    """Deserialize a VESC binary config blob and verify its signature."""

    buf = VescBuffer(data)
    sig = buf.pop_uint32()
    expected = signature(schema)
    if sig != expected:
        raise ValueError(
            f"Signature mismatch: got 0x{sig:08X}, expected 0x{expected:08X}"
        )

    values: dict[str, object] = {}
    for name in schema.ser_order:
        p = schema.params.get(name)
        if p is None:
            raise KeyError(f"Parameter {name!r} in SerOrder not found in schema")

        if p.type == CfgType.DOUBLE:
            if p.vTx == VescTx.DOUBLE16:
                values[name] = buf.pop_double16(p.vTx_double_scale)
            elif p.vTx == VescTx.DOUBLE32:
                values[name] = buf.pop_double32(p.vTx_double_scale)
            elif p.vTx == VescTx.DOUBLE32_AUTO:
                values[name] = buf.pop_double32_auto()
            else:
                raise ValueError(f"Parameter {name!r}: unsupported vTx {p.vTx} for DOUBLE")

        elif p.type in (CfgType.INT, CfgType.BITFIELD):
            if p.vTx == VescTx.UINT8 or p.type == CfgType.BITFIELD:
                values[name] = buf.pop_uint8()
            elif p.vTx == VescTx.INT8:
                values[name] = buf.pop_int8()
            elif p.vTx == VescTx.UINT16:
                values[name] = buf.pop_uint16()
            elif p.vTx == VescTx.INT16:
                values[name] = buf.pop_int16()
            elif p.vTx == VescTx.UINT32:
                values[name] = buf.pop_uint32()
            elif p.vTx == VescTx.INT32:
                values[name] = buf.pop_int32()
            else:
                raise ValueError(f"Parameter {name!r}: unsupported vTx {p.vTx} for INT")

        elif p.type == CfgType.QSTRING:
            values[name] = buf.pop_string()

        elif p.type in (CfgType.ENUM, CfgType.BOOL):
            values[name] = buf.pop_int8()

        elif p.type == CfgType.UNDEFINED:
            raise ValueError(f"Parameter {name!r}: type is UNDEFINED")

    return values
