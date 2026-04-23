"""App configuration compatibility wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from vesc_py.config_schema import (
    CfgType,
    ConfigParam,
    ConfigSchema,
    VescTx,
    deserialize_config,
    load_config_xml,
    serialize_config,
    signature,
)

AppConfSchema = ConfigSchema


def load_appconf_xml(path: Path) -> ConfigSchema:
    """Parse a parameters_appconf.xml file into a ConfigSchema."""

    return load_config_xml(path, "appconf")


def serialize_appconf(schema: ConfigSchema, values: Mapping[str, object]) -> bytes:
    """Serialize app config values into a binary blob."""

    return serialize_config(schema, values)


def deserialize_appconf(schema: ConfigSchema, data: bytes) -> dict[str, object]:
    """Deserialize app config values from a binary blob."""

    return deserialize_config(schema, data)


__all__ = [
    "AppConfSchema",
    "CfgType",
    "ConfigParam",
    "VescTx",
    "deserialize_appconf",
    "load_appconf_xml",
    "serialize_appconf",
    "signature",
]
