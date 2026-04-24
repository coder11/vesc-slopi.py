#!/usr/bin/env python3
"""Run the YALSA analysis loop without Qt to isolate non-GUI overhead.

This benchmark keeps the same VESC source and analysis callback used by
``examples/yalsa/live_signal_analysis.py`` while replacing the Qt event loop
and PyQtGraph rendering with a simple timer loop.

Examples:
    uv run examples/headless_yalsa_analysis.py --serial /dev/ttyACM0
    uv run examples/headless_yalsa_analysis.py --serial /dev/ttyACM0 --mode noop
    uv run examples/headless_yalsa_analysis.py --serial /dev/ttyACM0 --duration 10
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

# When this file is executed directly, Python adds `examples/` to sys.path.
# Ensure both the repository root and `src/` precede it so imports resolve to
# the real packages instead of the `examples/yalsa` directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for bootstrap_path in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if bootstrap_path in sys.path:
        sys.path.remove(bootstrap_path)
sys.path[:0] = [str(_SRC_ROOT), str(_REPO_ROOT)]

from examples.yalsa.live_signal_analysis import (
    LiveSignalAnalysisConfig,
    build_analysis,
    make_source,
)
from vesc_py.connection_cli import (
    add_vesc_connection_arguments,
    resolve_vesc_target_from_args,
)
from yalsa import (
    AnalysisInput,
    AnalysisResult,
    SignalBatchHistory,
    SignalBatchSourceSnapshot,
    default_parameter_values,
)

NSEC_PER_SEC = 1_000_000_000
_SLEEP_SLACK_NS = 200_000

DEFAULT_AXIS = "acc_z"
DEFAULT_STATUS_INTERVAL = 1.0
DEFAULT_MODE = "full"


@dataclass(slots=True)
class DurationStat:
    count: int = 0
    total_ns: int = 0
    max_ns: int = 0

    def add(self, duration_ns: int) -> None:
        self.count += 1
        self.total_ns += duration_ns
        self.max_ns = max(self.max_ns, duration_ns)

    @property
    def average_ns(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ns / self.count


@dataclass(slots=True)
class LoopProfile:
    interval_ns: DurationStat
    total_ns: DurationStat
    drain_ns: DurationStat
    history_ns: DurationStat
    snapshot_ns: DurationStat
    process_ns: DurationStat
    empty_refreshes: int = 0
    overruns: int = 0


def new_loop_profile() -> LoopProfile:
    return LoopProfile(
        interval_ns=DurationStat(),
        total_ns=DurationStat(),
        drain_ns=DurationStat(),
        history_ns=DurationStat(),
        snapshot_ns=DurationStat(),
        process_ns=DurationStat(),
    )


def average_ms(stat: DurationStat) -> float:
    return stat.average_ns / 1_000_000.0


def wait_until_ns(deadline_ns: int) -> None:
    """Wait until *deadline_ns* using coarse sleep plus a short busy-spin."""
    while True:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > _SLEEP_SLACK_NS:
            sleep_s = (remaining_ns - _SLEEP_SLACK_NS) / NSEC_PER_SEC
            time.sleep(min(sleep_s, 0.01))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the headless YALSA analysis bench."""
    parser = argparse.ArgumentParser(
        description="Run the YALSA analysis loop without Qt or PyQtGraph.",
    )
    add_vesc_connection_arguments(parser)
    parser.add_argument(
        "--axis",
        default=DEFAULT_AXIS,
        help="IMU axis to analyse (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "noop"),
        default=DEFAULT_MODE,
        help=(
            "Analysis workload: 'full' runs the same YALSA DSP callback as the GUI; "
            "'noop' skips the DSP callback but keeps the same timer/history path."
        ),
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL,
        metavar="SEC",
        help="Seconds between status prints (default: %(default)s).",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=None,
        metavar="HZ",
        help="Refresh rate in Hz; defaults to the GUI analysis rate.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Optional run duration; 0 means run until interrupted.",
    )
    return parser


def format_latest_values(
    snapshot: SignalBatchSourceSnapshot,
    channels: Mapping[str, str],
) -> str:
    """Format latest batch-source values for status output."""
    if not snapshot.latest_values:
        return "n/a"
    parts = []
    for channel_name, value in snapshot.latest_values.items():
        unit = channels.get(channel_name, "")
        parts.append(f"{channel_name}={value:.6g}{(' ' + unit) if unit else ''}")
    return ", ".join(parts)


def expand_debug_text(text: str) -> list[str]:
    """Split one compact debug line into one line per metric."""
    tokens = text.split()
    if not tokens:
        return []
    prefix = tokens[0]
    return [f"{prefix}: {token}" for token in tokens[1:]]


def print_status(
    *,
    snapshot: SignalBatchSourceSnapshot,
    channels: Mapping[str, str],
    history_rate_hz: float | None,
    mode: str,
    refresh_rate_hz: float,
    profile: LoopProfile,
    process_result: AnalysisResult | None,
    process_error: str | None,
    output_stream: TextIO,
) -> None:
    """Print one multiline status block for the headless analysis bench."""
    print(
        f"source: {snapshot.average_rate_hz:.1f} Hz",
        file=output_stream,
    )
    if history_rate_hz is None:
        print("history: measuring", file=output_stream)
    else:
        print(f"history: {history_rate_hz:.1f} Hz", file=output_stream)
    print(f"samples: {snapshot.samples}", file=output_stream)
    print(f"dropped: {snapshot.dropped}", file=output_stream)
    print(f"errors: {snapshot.errors}", file=output_stream)
    print(f"latest: {format_latest_values(snapshot, channels)}", file=output_stream)
    print(f"mode: {mode}", file=output_stream)
    if snapshot.debug_text:
        for line in expand_debug_text(snapshot.debug_text):
            print(line, file=output_stream)
    print(
        (
            f"uidbg: int={average_ms(profile.interval_ns):.2f}/"
            f"{1000.0 / refresh_rate_hz:.2f}ms"
        ),
        file=output_stream,
    )
    print(f"uidbg: tot={average_ms(profile.total_ns):.3f}", file=output_stream)
    print(f"uidbg: drain={average_ms(profile.drain_ns):.3f}", file=output_stream)
    print(f"uidbg: hist={average_ms(profile.history_ns):.3f}", file=output_stream)
    print(f"uidbg: snap={average_ms(profile.snapshot_ns):.3f}", file=output_stream)
    print(f"uidbg: proc={average_ms(profile.process_ns):.3f}", file=output_stream)
    print(f"uidbg: empty={profile.empty_refreshes}", file=output_stream)
    print(f"uidbg: over={profile.overruns}", file=output_stream)
    if process_error:
        print(f"analysis error: {process_error}", file=output_stream)
    elif process_result is not None and process_result.status_text:
        print(process_result.status_text, file=output_stream)
    print("", file=output_stream)
    output_stream.flush()


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.status_interval <= 0.0:
        parser.error("--status-interval must be greater than 0")
    if args.refresh_rate is not None and args.refresh_rate <= 0.0:
        parser.error("--refresh-rate must be greater than 0")
    if args.duration < 0.0:
        parser.error("--duration must be greater than or equal to 0")

    config = LiveSignalAnalysisConfig(
        source="vesc",
        axis=args.axis,
        timeout=args.timeout,
    )
    target = resolve_vesc_target_from_args(args)
    source, source_label = make_source(config, vesc_target=target)
    app = build_analysis(
        source=source,
        source_label=source_label,
        axis=config.axis,
        unit=source.channels[config.axis],
    )
    parameter_values = default_parameter_values(app.parameters)
    refresh_rate_hz = app.plot_rate_hz if args.refresh_rate is None else args.refresh_rate
    refresh_interval_ns = max(1, round(NSEC_PER_SEC / refresh_rate_hz))
    status_interval_ns = round(args.status_interval * NSEC_PER_SEC)
    duration_ns = round(args.duration * NSEC_PER_SEC)

    retained = SignalBatchHistory(app.history, app.source.channels)
    profile = new_loop_profile()
    last_result: AnalysisResult | None = None
    last_process_error: str | None = None
    last_snapshot: SignalBatchSourceSnapshot | None = None
    last_history_rate_hz: float | None = None

    start_ns = time.perf_counter_ns()
    next_refresh_ns = start_ns
    next_status_ns = start_ns + status_interval_ns
    previous_refresh_start_ns: int | None = None

    print(
        (
            f"Starting headless YALSA analysis on {target.describe()}; "
            f"axis={config.axis}; mode={args.mode}; refresh_rate={refresh_rate_hz:g} Hz; "
            f"status_interval={args.status_interval:g}s"
        ),
        file=sys.stderr,
    )

    app.source.start()
    try:
        while True:
            if duration_ns > 0 and time.perf_counter_ns() - start_ns >= duration_ns:
                break

            wait_until_ns(next_refresh_ns)
            refresh_start_ns = time.perf_counter_ns()
            if previous_refresh_start_ns is not None:
                profile.interval_ns.add(refresh_start_ns - previous_refresh_start_ns)
            previous_refresh_start_ns = refresh_start_ns
            if refresh_start_ns > next_refresh_ns + refresh_interval_ns:
                profile.overruns += 1
            next_refresh_ns += refresh_interval_ns

            drain_start_ns = time.perf_counter_ns()
            batch, _batch_dropped = app.source.drain()
            profile.drain_ns.add(time.perf_counter_ns() - drain_start_ns)
            if batch.sample_count == 0:
                profile.empty_refreshes += 1

            history_start_ns = time.perf_counter_ns()
            retained.append_batch(batch)
            profile.history_ns.add(time.perf_counter_ns() - history_start_ns)

            snapshot_start_ns = time.perf_counter_ns()
            last_history_rate_hz = retained.sample_hz()
            analysis_input = AnalysisInput(
                batch=retained.snapshot(),
                sample_rate_hz=last_history_rate_hz,
                snapshot=app.source.snapshot(),
            )
            profile.snapshot_ns.add(time.perf_counter_ns() - snapshot_start_ns)
            last_snapshot = analysis_input.snapshot

            process_start_ns = time.perf_counter_ns()
            if args.mode == "full":
                try:
                    last_result = app.process(analysis_input, parameter_values)
                except Exception as exc:  # noqa: BLE001 - surfaced in status output.
                    last_result = None
                    last_process_error = str(exc)
                else:
                    last_process_error = None
            else:
                last_result = None
                last_process_error = None
            profile.process_ns.add(time.perf_counter_ns() - process_start_ns)
            profile.total_ns.add(time.perf_counter_ns() - refresh_start_ns)

            now_ns = time.perf_counter_ns()
            if last_snapshot is not None and now_ns >= next_status_ns:
                print_status(
                    snapshot=last_snapshot,
                    channels=app.source.channels,
                    history_rate_hz=last_history_rate_hz,
                    mode=args.mode,
                    refresh_rate_hz=refresh_rate_hz,
                    profile=profile,
                    process_result=last_result,
                    process_error=last_process_error,
                    output_stream=sys.stderr,
                )
                next_status_ns = now_ns + status_interval_ns
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
    finally:
        app.source.stop(timeout=args.timeout + 0.2)


if __name__ == "__main__":
    main()
