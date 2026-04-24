"""CLI parsing helpers for the live signal analysis example."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from examples.yalsa.live_signal_analysis import (
    DEFAULT_DETERMINISTIC_RATE,
    DEFAULT_SOURCE,
    DEFAULT_TIMEOUT,
    LiveSignalAnalysisConfig,
    SOURCE_CHOICES,
)
from vesc_py.fast_imu_source import DEFAULT_PIPELINE_DEPTH, parse_imu_axis


VESC_ARGUMENT_HELP = (
    "When --source vesc is selected, additional VESC connection flags are "
    "forwarded to the connection CLI: --serial, --ble, --baudrate, --no-exclusive, "
    "--ble-scan-timeout, --ble-connect-timeout, --ble-chunk-size, and --can-id."
)


@dataclass(frozen=True, slots=True)
class ParsedLiveSignalAnalysisCli:
    """Parsed example config plus forwarded VESC connection arguments."""

    config: LiveSignalAnalysisConfig
    vesc_argv: tuple[str, ...]


def _parse_axis_arg(text: str) -> str:
    """Normalize an IMU axis name for argparse."""
    try:
        return parse_imu_axis(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_live_signal_analysis_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser for the example."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a modular live signal analysis GUI with tunable controls and plots. "
            "The VESC IMU axis analysis is the default proof-of-concept pipeline."
        ),
        epilog=VESC_ARGUMENT_HELP,
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default=DEFAULT_SOURCE,
        help="Signal source to use (default: %(default)s).",
    )
    parser.add_argument(
        "--axis",
        type=_parse_axis_arg,
        default="acc_z",
        help="IMU axis to analyze (default: %(default)s).",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=DEFAULT_PIPELINE_DEPTH,
        help="Outstanding COMM_GET_IMU_DATA requests for --source vesc (default: %(default)s).",
    )
    parser.add_argument(
        "--deterministic-rate",
        type=float,
        default=DEFAULT_DETERMINISTIC_RATE,
        help="Sample rate for synthetic sources in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=(
            "VESC response timeout in seconds for CAN discovery and source polling "
            "(default: %(default)s)."
        ),
    )
    return parser


def parse_live_signal_analysis_cli(
    argv: Sequence[str] | None = None,
) -> ParsedLiveSignalAnalysisCli:
    """Parse example-specific CLI flags and preserve remaining VESC arguments."""
    parser = build_live_signal_analysis_parser()
    args, vesc_argv = parser.parse_known_args(argv)
    if args.source != DEFAULT_SOURCE and vesc_argv:
        parser.error(
            "VESC connection flags are only accepted when --source vesc is selected."
        )

    try:
        config = LiveSignalAnalysisConfig(
            source=args.source,
            axis=args.axis,
            pipeline_depth=args.pipeline_depth,
            deterministic_rate=args.deterministic_rate,
            timeout=args.timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))

    return ParsedLiveSignalAnalysisCli(config=config, vesc_argv=tuple(vesc_argv))
