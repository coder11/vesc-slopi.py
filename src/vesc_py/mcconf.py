"""Motor configuration wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from vesc_py.config_schema import (
    ConfigSchema,
    deserialize_config,
    load_config_xml,
    serialize_config,
)

McConfSchema = ConfigSchema


def load_mcconf_xml(path: Path) -> ConfigSchema:
    """Parse a parameters_mcconf.xml file into a ConfigSchema."""

    return load_config_xml(path, "mcconf")


def serialize_mcconf(schema: ConfigSchema, values: Mapping[str, object]) -> bytes:
    """Serialize motor config values into a binary blob."""

    return serialize_config(schema, values)


def deserialize_mcconf(schema: ConfigSchema, data: bytes) -> dict[str, object]:
    """Deserialize motor config values from a binary blob."""

    return deserialize_config(schema, data)
