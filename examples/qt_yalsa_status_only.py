#!/usr/bin/env python3
"""Run the YALSA analysis under Qt without creating PyQtGraph plots.

This isolates the Qt event loop and QLabel/status updates from the PyQtGraph
plot widgets used by ``examples/yalsa/live_signal_analysis.py``.

Examples:
    uv run examples/qt_yalsa_status_only.py --serial /dev/ttyACM0
    uv run examples/qt_yalsa_status_only.py --serial /dev/ttyACM0 --mode noop
    uv run examples/qt_yalsa_status_only.py --serial /dev/ttyACM0 --duration 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project packages resolve ahead of the local `examples/` tree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for bootstrap_path in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if bootstrap_path in sys.path:
        sys.path.remove(bootstrap_path)
sys.path[:0] = [str(_SRC_ROOT), str(_REPO_ROOT)]

from examples.headless_yalsa_analysis import (
    DEFAULT_AXIS,
    DEFAULT_MODE,
    DEFAULT_STATUS_INTERVAL,
    LoopProfile,
    NSEC_PER_SEC,
    average_ms,
    expand_debug_text,
    format_latest_values,
    new_loop_profile,
)
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
    import_pyqtgraph,
    require_qt_platform_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Qt status-only bench."""
    parser = argparse.ArgumentParser(
        description="Run the YALSA analysis under Qt without plot widgets.",
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
            "'noop' skips the DSP callback but keeps the same Qt timer/history path."
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
        "--duration",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Optional run duration; 0 means run until interrupted.",
    )
    return parser


def _status_text(
    *,
    snapshot: SignalBatchSourceSnapshot,
    channels: dict[str, str],
    history_rate_hz: float | None,
    mode: str,
    refresh_interval_ms: float,
    profile: LoopProfile,
    process_result: AnalysisResult | None,
    process_error: str | None,
) -> str:
    lines = [
        f"source: {snapshot.average_rate_hz:.1f} Hz",
        (
            "history: measuring"
            if history_rate_hz is None
            else f"history: {history_rate_hz:.1f} Hz"
        ),
        f"samples: {snapshot.samples}",
        f"dropped: {snapshot.dropped}",
        f"errors: {snapshot.errors}",
        f"latest: {format_latest_values(snapshot, channels)}",
        f"mode: {mode}",
    ]
    if snapshot.debug_text:
        lines.extend(expand_debug_text(snapshot.debug_text))
    lines.extend(
        (
            f"uidbg: int={average_ms(profile.interval_ns):.2f}/{refresh_interval_ms:.2f}ms",
            f"uidbg: tot={average_ms(profile.total_ns):.3f}",
            f"uidbg: drain={average_ms(profile.drain_ns):.3f}",
            f"uidbg: hist={average_ms(profile.history_ns):.3f}",
            f"uidbg: snap={average_ms(profile.snapshot_ns):.3f}",
            f"uidbg: proc={average_ms(profile.process_ns):.3f}",
            f"uidbg: empty={profile.empty_refreshes}",
            f"uidbg: over={profile.overruns}",
        )
    )
    if process_error:
        lines.append(f"analysis error: {process_error}")
    elif process_result is not None and process_result.status_text:
        lines.append(process_result.status_text)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    if args.status_interval <= 0.0:
        parser.error("--status-interval must be greater than 0")
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
    refresh_interval_ms = max(1, round(1000.0 / app.plot_rate_hz))
    status_interval_ns = round(args.status_interval * NSEC_PER_SEC)
    duration_ms = 0 if args.duration <= 0.0 else max(1, round(args.duration * 1000.0))

    pg, QtCore, QtWidgets = import_pyqtgraph()
    require_qt_platform_runtime()
    qt_app = pg.mkQApp("Qt YALSA Status Only")

    window = QtWidgets.QWidget()
    window.setWindowTitle("Qt YALSA Status Only")
    window.resize(720, 420)
    layout = QtWidgets.QVBoxLayout(window)
    source_text = QtWidgets.QLabel(source_label)
    source_text.setWordWrap(True)
    layout.addWidget(source_text)
    status = QtWidgets.QLabel("Waiting for signal data...")
    status.setWordWrap(True)
    layout.addWidget(status, stretch=1)

    retained = SignalBatchHistory(app.history, app.source.channels)
    profile = new_loop_profile()
    last_result: AnalysisResult | None = None
    last_process_error: str | None = None
    last_snapshot: SignalBatchSourceSnapshot | None = None
    last_history_rate_hz: float | None = None
    previous_refresh_start_ns: int | None = None
    next_status_ns = time.perf_counter_ns() + status_interval_ns
    source_started = False

    def refresh() -> None:
        nonlocal last_result
        nonlocal last_process_error
        nonlocal last_snapshot
        nonlocal last_history_rate_hz
        nonlocal previous_refresh_start_ns
        nonlocal next_status_ns
        refresh_start_ns = time.perf_counter_ns()
        if previous_refresh_start_ns is not None:
            profile.interval_ns.add(refresh_start_ns - previous_refresh_start_ns)
        previous_refresh_start_ns = refresh_start_ns

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
            except Exception as exc:  # noqa: BLE001 - surfaced in label and stderr.
                last_result = None
                last_process_error = str(exc)
            else:
                last_process_error = None
        else:
            last_result = None
            last_process_error = None
        profile.process_ns.add(time.perf_counter_ns() - process_start_ns)
        total_ns = time.perf_counter_ns() - refresh_start_ns
        profile.total_ns.add(total_ns)

        if total_ns > round(refresh_interval_ms * 1_000_000):
            profile.overruns += 1

        channels = dict(app.source.channels)
        text = _status_text(
            snapshot=analysis_input.snapshot,
            channels=channels,
            history_rate_hz=last_history_rate_hz,
            mode=args.mode,
            refresh_interval_ms=float(refresh_interval_ms),
            profile=profile,
            process_result=last_result,
            process_error=last_process_error,
        )
        status.setText(text)

        now_ns = time.perf_counter_ns()
        if now_ns >= next_status_ns:
            print(text, file=sys.stderr)
            print("", file=sys.stderr)
            next_status_ns = now_ns + status_interval_ns

    def stop_source() -> None:
        nonlocal source_started
        if source_started:
            app.source.stop(timeout=args.timeout + 0.2)
            source_started = False

    timer = QtCore.QTimer(window)
    timer.timeout.connect(refresh)
    qt_app.aboutToQuit.connect(stop_source)

    app.source.start()
    source_started = True
    timer.start(refresh_interval_ms)
    refresh()
    window.show()

    if duration_ms > 0:
        QtCore.QTimer.singleShot(duration_ms, qt_app.quit)

    try:
        exec_fn = getattr(pg, "exec", None)
        if callable(exec_fn):
            exec_fn()
        else:
            qt_app.exec()
    finally:
        timer.stop()
        stop_source()


if __name__ == "__main__":
    main()
