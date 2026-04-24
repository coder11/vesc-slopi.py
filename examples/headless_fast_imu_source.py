#!/usr/bin/env python3
"""Run ``VescImuSignalSource`` headless for transport-path comparison.

This isolates the fast IMU source from the Qt/YALSA runtime. The source still
uses its own worker thread internally, but the main process only drains and
prints status periodically.

Examples:
    uv run examples/headless_fast_imu_source.py
    uv run examples/headless_fast_imu_source.py --serial /dev/ttyACM0
    uv run examples/headless_fast_imu_source.py --serial /dev/ttyACM0 --axis acc_z
    uv run examples/headless_fast_imu_source.py --serial /dev/ttyACM0 --status-interval 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from typing import TextIO

from vesc_py.connection_cli import (
    add_vesc_connection_arguments,
    resolve_vesc_target_from_args,
)
from vesc_py.fast_imu_source import VescImuSignalSource, imu_axis_unit, parse_imu_axis

DEFAULT_AXIS = "acc_z"
DEFAULT_STATUS_INTERVAL = 1.0
DEFAULT_PENDING_SAMPLES = 20_000


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the headless source bench."""
    parser = argparse.ArgumentParser(
        description="Run VescImuSignalSource without the Qt/YALSA runtime.",
    )
    add_vesc_connection_arguments(parser)
    parser.add_argument(
        "--axis",
        default=DEFAULT_AXIS,
        help="IMU axis to decode from the fast source (default: %(default)s).",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL,
        metavar="SEC",
        help="Seconds between status prints (default: %(default)s).",
    )
    parser.add_argument(
        "--pending-samples",
        type=int,
        default=DEFAULT_PENDING_SAMPLES,
        metavar="COUNT",
        help="Pending overwrite-buffer capacity (default: %(default)s).",
    )
    return parser


def drained_rate_hz(timestamps_s: Sequence[float]) -> float | None:
    """Estimate the sample rate from one drained batch of timestamps."""
    if len(timestamps_s) < 2:
        return None
    elapsed_s = float(timestamps_s[-1] - timestamps_s[0])
    if elapsed_s <= 0.0:
        return None
    return (len(timestamps_s) - 1) / elapsed_s


def expand_debug_text(text: str) -> list[str]:
    """Split one compact debug line into one line per metric."""
    tokens = text.split()
    if not tokens:
        return []
    prefix = tokens[0]
    return [f"{prefix}: {token}" for token in tokens[1:]]


def print_snapshot(
    *,
    snapshot: object,
    batch_samples: int,
    batch_dropped: int,
    batch_rate_hz: float | None,
    axis: str,
    unit: str,
    output_stream: TextIO,
) -> None:
    """Print one multiline status block for the headless source bench."""
    source_rate_hz = getattr(snapshot, "average_rate_hz")
    latest_value = getattr(snapshot, "latest_value")
    latest_text = "n/a"
    if latest_value is not None:
        latest_text = f"{axis}={latest_value:.6g} {unit}"

    print(
        f"source: {'measuring' if source_rate_hz is None else f'{source_rate_hz:.1f} Hz'}",
        file=output_stream,
    )
    print(
        f"drain: {'measuring' if batch_rate_hz is None else f'{batch_rate_hz:.1f} Hz'}",
        file=output_stream,
    )
    print(f"samples: {getattr(snapshot, 'samples')}", file=output_stream)
    print(f"dropped: {getattr(snapshot, 'dropped')}", file=output_stream)
    print(f"errors: {getattr(snapshot, 'errors')}", file=output_stream)
    print(f"batch_samples: {batch_samples}", file=output_stream)
    print(f"batch_dropped: {batch_dropped}", file=output_stream)
    print(f"latest: {latest_text}", file=output_stream)
    last_error = getattr(snapshot, "last_error")
    if last_error:
        print(f"source error: {last_error}", file=output_stream)
    debug_text = getattr(snapshot, "debug_text")
    if debug_text:
        for line in expand_debug_text(debug_text):
            print(line, file=output_stream)
    print("", file=output_stream)
    output_stream.flush()


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.status_interval <= 0.0:
        parser.error("--status-interval must be greater than 0")
    if args.pending_samples <= 0:
        parser.error("--pending-samples must be greater than 0")

    target = resolve_vesc_target_from_args(args)
    axis = parse_imu_axis(args.axis)
    unit = imu_axis_unit(axis)
    source = VescImuSignalSource(
        connection=target.connection,
        axis=axis,
        timeout=args.timeout,
        pending_samples=args.pending_samples,
        can_id=target.can_id,
    )

    print(
        (
            f"Starting headless VescImuSignalSource on {target.describe()}; "
            f"axis={axis}; status_interval={args.status_interval:g}s; "
            f"pending_samples={args.pending_samples}"
        ),
        file=sys.stderr,
    )

    source.start()
    try:
        while True:
            time.sleep(args.status_interval)
            timestamps_s, _values, batch_dropped = source.drain()
            snapshot = source.snapshot()
            print_snapshot(
                snapshot=snapshot,
                batch_samples=len(timestamps_s),
                batch_dropped=batch_dropped,
                batch_rate_hz=drained_rate_hz(timestamps_s),
                axis=axis,
                unit=unit,
                output_stream=sys.stderr,
            )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
    finally:
        source.stop(timeout=args.timeout + 0.2)


if __name__ == "__main__":
    main()
