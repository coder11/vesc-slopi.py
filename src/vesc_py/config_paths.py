"""Resolve firmware-version-specific config directories matching Utility::configLoad."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "vesc_tool" / "res" / "config"
_CONFIG_DIR_ENV = "VESC_CONFIG_DIR"
_VESC_TOOL_DIR_ENV = "VESC_TOOL_DIR"


def default_config_dir() -> Path:
    """Return the default firmware XML config directory."""

    config_dir = os.environ.get(_CONFIG_DIR_ENV)
    if config_dir:
        return Path(config_dir).expanduser()

    vesc_tool_dir = os.environ.get(_VESC_TOOL_DIR_ENV)
    if vesc_tool_dir:
        return Path(vesc_tool_dir).expanduser() / "res" / "config"

    return _CONFIG_DIR


def _available_versions(config_dir: Path | None = None) -> list[tuple[int, int, Path]]:
    """Return (major, minor, dir_path) for all FW config directories."""
    base = config_dir or default_config_dir()
    versions: list[tuple[int, int, Path]] = []
    if not base.is_dir():
        return versions

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        # Directory name may contain _o_ separating multiple version aliases
        for segment in entry.name.split("_o_"):
            parts = segment.split(".")
            if len(parts) == 2:
                try:
                    major = int(parts[0])
                    minor = int(parts[1])
                    versions.append((major, minor, entry))
                except ValueError:
                    continue

    return versions


def _version_miss_message(
    fw_major: int,
    fw_minor: int,
    config_dir: Path | None = None,
) -> str:
    avail = _available_versions(config_dir)
    version_strs = sorted(f"{m}.{n:02d}" for m, n, _ in avail)
    searched = config_dir or default_config_dir()
    if not version_strs:
        return (
            f"No config for firmware {fw_major}.{fw_minor:02d}. "
            f"No config XML directories found under {searched}. "
            "Initialize the vesc_tool submodule or set VESC_CONFIG_DIR."
        )

    return (
        f"No config for firmware {fw_major}.{fw_minor:02d}. "
        f"Available versions: {', '.join(version_strs)}"
    )


def find_config_dir(
    fw_major: int,
    fw_minor: int,
    config_dir: Path | None = None,
) -> Path:
    """Find the local config directory for a given firmware version."""

    for major, minor, dir_path in _available_versions(config_dir):
        if major == fw_major and minor == fw_minor:
            return dir_path

    raise FileNotFoundError(_version_miss_message(fw_major, fw_minor, config_dir))


def find_config_xml(
    fw_major: int,
    fw_minor: int,
    kind: str,
    config_dir: Path | None = None,
) -> Path:
    """Find a parameters_<kind>.xml file for a given firmware version."""

    if kind not in {"appconf", "mcconf"}:
        raise ValueError(f"Unsupported config kind: {kind}")

    dir_path = find_config_dir(fw_major, fw_minor, config_dir)
    path = dir_path / f"parameters_{kind}.xml"
    if path.exists():
        return path

    raise FileNotFoundError(f"Config directory {dir_path} exists but {path.name} is missing")


def find_appconf_xml(
    fw_major: int,
    fw_minor: int,
    config_dir: Path | None = None,
) -> Path:
    """Find the parameters_appconf.xml for a given firmware version.

    Raises FileNotFoundError with a list of available versions on miss.
    """

    return find_config_xml(fw_major, fw_minor, "appconf", config_dir)


def find_mcconf_xml(
    fw_major: int,
    fw_minor: int,
    config_dir: Path | None = None,
) -> Path:
    """Find the parameters_mcconf.xml for a given firmware version."""

    return find_config_xml(fw_major, fw_minor, "mcconf", config_dir)
