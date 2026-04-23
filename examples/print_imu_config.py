#!/usr/bin/env python3
"""Print IMU-related APPCONF fields through a VESC Tool TCP server.

Usage:
    uv run examples/print_imu_config.py
    uv run examples/print_imu_config.py --tcp 192.168.1.50:65102

Start the bridge first, for example:
    ./vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer 65102
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vesc_py import VescClient
from vesc_py.appconf import AppConfSchema, CfgType, ConfigParam


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    host, sep, port_str = endpoint.rpartition(":")
    if not sep or not host or not port_str:
        raise argparse.ArgumentTypeError("TCP endpoint must be HOST:PORT")

    try:
        port = int(port_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("TCP port must be an integer") from exc

    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError("TCP port must be in range 1..65535")

    return host, port


def _schema_ordered_imu_names(schema: AppConfSchema) -> list[str]:
    return [name for name in schema.ser_order if name.startswith("imu_conf.")]


def _format_value(param: ConfigParam, value: object) -> str:
    if param.type == CfgType.ENUM and isinstance(value, int):
        if 0 <= value < len(param.enum_names):
            return f"{param.enum_names[value]} ({value})"
        return str(value)

    if param.type == CfgType.BOOL and isinstance(value, int):
        return f"{bool(value)} ({value})"

    if isinstance(value, float):
        return f"{value:.9g}"

    return str(value)


def print_imu_config(client: VescClient) -> None:
    """Read APPCONF and print every imu_conf.* field."""
    schema = client.appconf_schema
    if schema is None:
        raise RuntimeError(
            "No APPCONF schema loaded for this firmware version; "
            "cannot decode IMU config."
        )

    config = client.get_appconf()
    names = _schema_ordered_imu_names(schema)
    if not names:
        raise RuntimeError("This APPCONF schema has no imu_conf.* fields.")

    fw = client.fw_version
    if fw is not None:
        fw_name = f" ({fw.fw_name})" if fw.fw_name else ""
        print(f"Firmware: {fw.major}.{fw.minor:02d}{fw_name}")
        if fw.hw:
            print(f"Hardware: {fw.hw}")
        print()

    width = max(len(name) for name in names)
    print("IMU config:")
    for name in names:
        param = schema.params[name]
        value = config[name]
        print(f"  {name:<{width}}  {_format_value(param, value)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read and print imu_conf.* APPCONF fields via VESC Tool TCP server.",
    )
    parser.add_argument(
        "--tcp",
        type=_parse_tcp_endpoint,
        default=("127.0.0.1", 65102),
        metavar="HOST:PORT",
        help="VESC Tool TCP server endpoint (default: 127.0.0.1:65102).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Response timeout in seconds (default: 2.0).",
    )
    args = parser.parse_args(argv)

    host, port = args.tcp
    print(f"Connecting to VESC Tool TCP server at {host}:{port} ...")
    client = VescClient.connect_tcp(host, port, timeout=args.timeout)
    try:
        print_imu_config(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
