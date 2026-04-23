#!/usr/bin/env python3
"""YALSA app with a VESC IMU axis proof of concept.

Examples:
    uv run examples/yalsa/live_signal_analysis.py --source deterministic --axis acc_z
    uv run examples/yalsa/live_signal_analysis.py --source deterministic-white-noise --axis gyro_z
    uv run examples/yalsa/live_signal_analysis.py --source vesc --axis acc_z --pipeline-depth 4
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Literal, cast

import numpy as np

from vesc_py import list_serial_ports
from vesc_py.fast_imu_source import (
    DEFAULT_BAUDRATE,
    DEFAULT_PIPELINE_DEPTH,
    DEFAULT_TIMEOUT,
    VescImuSignalSource,
    imu_axis_unit,
    parse_imu_axis,
)
from yalsa import (
    AnalysisInput,
    AnalysisResult,
    ChoiceOption,
    LiveAnalysisApp,
    ParamValue,
    ProcessCallback,
    PlotSpec,
    PlotTrace,
    ScalarSignalSourceAdapter,
    SignalBatchSource,
    choice_parameter,
    float_parameter,
    int_parameter,
    butter_lowpass_hz,
    fft_magnitude,
    run_live_analysis,
    signal_stats,
    welch_psd,
    xy_series,
)
from vesc_py.live_signal import (
    DeterministicSignalSource,
    DeterministicWhiteNoiseSignalSource,
    NoisyDeterministicSignalSource,
    SignalSource,
)

DEFAULT_HISTORY = 20_000
DEFAULT_MAX_POINTS = 1_200
DEFAULT_PLOT_RATE = 30.0
DEFAULT_PENDING_SAMPLES = 20_000
DEFAULT_DETERMINISTIC_RATE = 500.0
DEFAULT_CUTOFF_HZ = 15.0
DEFAULT_FILTER_ORDER = 2
DEFAULT_THEME = "dark"

SPECTRUM_OPTIONS = (
    ChoiceOption(value="psd", label="PSD"),
    ChoiceOption(value="fft", label="FFT"),
)


def parse_axis_arg(text: str) -> str:
    """Argparse wrapper for IMU axis parsing."""
    try:
        return parse_imu_axis(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def autodetect_port() -> str:
    """Return the first discovered serial port, preferring VESC-like devices."""
    ports = list_serial_ports()
    if not ports:
        raise SystemExit(
            "No serial ports found. Connect the VESC over USB or pass --port explicitly."
        )
    return ports[0].system_path


def clamp_cutoff_hz(cutoff_hz: float, sample_rate_hz: float | None) -> float | None:
    """Clamp the requested cutoff away from Nyquist for live tuning."""
    if sample_rate_hz is None or sample_rate_hz <= 0.0:
        return None
    nyquist_margin_hz = sample_rate_hz * 0.49
    return min(cutoff_hz, nyquist_margin_hz)


def empty_series() -> tuple[np.ndarray, np.ndarray]:
    """Return a shared empty x/y pair."""
    empty = np.empty(0, dtype=np.float64)
    return empty, empty


def build_axis_analysis_processor(axis: str, unit: str) -> ProcessCallback:
    """Return a processing callback for one selected IMU axis."""

    def process(
        data: AnalysisInput,
        params: Mapping[str, ParamValue],
    ) -> AnalysisResult:
        timestamps = data.timestamps_s
        raw = data.channel(axis)
        if int(raw.size) == 0:
            empty_x, empty_y = empty_series()
            return AnalysisResult(
                series={
                    "raw": xy_series(empty_x, empty_y),
                    "filtered": xy_series(empty_x, empty_y),
                    "raw_spectrum": xy_series(empty_x, empty_y),
                    "filtered_spectrum": xy_series(empty_x, empty_y),
                },
                status_text="waiting for samples",
            )

        sample_rate_hz = data.sample_rate_hz
        requested_cutoff_hz = float(cast(float, params["cutoff_hz"]))
        cutoff_hz = clamp_cutoff_hz(requested_cutoff_hz, sample_rate_hz)
        filter_order = int(cast(int, params["filter_order"]))
        spectrum_mode = cast(str, params["spectrum_mode"])

        filtered = raw.copy()
        if cutoff_hz is not None and int(raw.size) >= 2:
            filtered = butter_lowpass_hz(
                raw,
                cutoff_hz=cutoff_hz,
                sample_rate_hz=cast(float, sample_rate_hz),
                order=filter_order,
                initial_value=float(raw[0]),
            )

        if spectrum_mode == "psd":
            raw_spectrum = welch_psd(timestamps, raw)
            filtered_spectrum = welch_psd(timestamps, filtered)
        else:
            raw_spectrum = fft_magnitude(timestamps, raw)
            filtered_spectrum = fft_magnitude(timestamps, filtered)

        if raw_spectrum is None:
            raw_spectrum_series = xy_series(*empty_series())
        else:
            raw_spectrum_series = xy_series(raw_spectrum[0], raw_spectrum[1])

        if filtered_spectrum is None:
            filtered_spectrum_series = xy_series(*empty_series())
        else:
            filtered_spectrum_series = xy_series(
                filtered_spectrum[0],
                filtered_spectrum[1],
            )

        raw_stats = signal_stats(raw)
        filtered_stats = signal_stats(filtered)
        status_parts = [
            f"mode: {spectrum_mode.upper()}",
            (
                "cutoff: measuring"
                if cutoff_hz is None
                else f"cutoff: {cutoff_hz:.2f} Hz"
            ),
            f"order: {filter_order}",
        ]
        if cutoff_hz is not None and cutoff_hz != requested_cutoff_hz:
            status_parts.append(f"requested cutoff clamped from {requested_cutoff_hz:.2f} Hz")
        if raw_stats is not None:
            status_parts.append(f"raw RMS: {raw_stats.rms:.6g} {unit}")
        if filtered_stats is not None:
            status_parts.append(f"filtered RMS: {filtered_stats.rms:.6g} {unit}")

        return AnalysisResult(
            series={
                "raw": xy_series(timestamps, raw),
                "filtered": xy_series(timestamps, filtered),
                "raw_spectrum": raw_spectrum_series,
                "filtered_spectrum": filtered_spectrum_series,
            },
            status_text=" | ".join(status_parts),
        )

    return process


def build_analysis(
    *,
    source: SignalBatchSource,
    source_label: str,
    axis: str,
    unit: str,
    history: int,
    max_points: int,
    plot_rate_hz: float,
    theme: Literal["light", "dark"],
    antialias: bool,
) -> LiveAnalysisApp:
    """Build the declarative analysis app consumed by the generic runtime."""
    return LiveAnalysisApp(
        title=f"Live Signal Analysis: {axis}",
        source=source,
        source_label=source_label,
        history=history,
        plot_rate_hz=plot_rate_hz,
        theme=theme,
        antialias=antialias,
        parameters=(
            float_parameter(
                "cutoff_hz",
                label="Cutoff Hz",
                default=DEFAULT_CUTOFF_HZ,
                minimum=0.1,
                maximum=500.0,
                step=0.5,
                decimals=2,
            ),
            int_parameter(
                "filter_order",
                label="Order",
                default=DEFAULT_FILTER_ORDER,
                minimum=1,
                maximum=8,
                step=1,
            ),
            choice_parameter(
                "spectrum_mode",
                label="Spectrum",
                default="psd",
                choices=SPECTRUM_OPTIONS,
            ),
        ),
        plots=(
            PlotSpec(
                title="Time Domain",
                traces=(
                    PlotTrace(series="raw", label="Raw"),
                    PlotTrace(series="filtered", label="Filtered"),
                ),
                x_label="time",
                x_unit="s",
                y_label=axis,
                y_unit=unit,
                max_points=max_points,
            ),
            PlotSpec(
                title="Frequency Domain",
                traces=(
                    PlotTrace(series="raw_spectrum", label="Raw"),
                    PlotTrace(series="filtered_spectrum", label="Filtered"),
                ),
                x_label="frequency",
                x_unit="Hz",
                y_label="spectrum",
                max_points=max_points,
            ),
        ),
        process=build_axis_analysis_processor(axis, unit),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the analysis example."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a modular live signal analysis GUI with tunable controls and plots. "
            "The VESC IMU axis analysis is the default proof-of-concept pipeline."
        )
    )
    parser.add_argument(
        "--source",
        choices=("vesc", "deterministic", "deterministic-noisy", "deterministic-white-noise"),
        default="vesc",
        help="Signal source to use (default: %(default)s).",
    )
    parser.add_argument(
        "--axis",
        type=parse_axis_arg,
        default="acc_z",
        help="IMU axis to analyze (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        help="Serial port for --source vesc (default: autodetect first serial port).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help="Serial baudrate for --source vesc (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Serial read timeout in seconds for --source vesc (default: %(default)s).",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=DEFAULT_PIPELINE_DEPTH,
        help="Outstanding COMM_GET_IMU_DATA requests for --source vesc (default: %(default)s).",
    )
    parser.add_argument(
        "--no-exclusive",
        action="store_true",
        help="Disable exclusive pyserial port access.",
    )
    parser.add_argument(
        "--deterministic-rate",
        type=float,
        default=DEFAULT_DETERMINISTIC_RATE,
        help="Sample rate for synthetic sources in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=DEFAULT_HISTORY,
        help="Retained history length in samples (default: %(default)s).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum points drawn per trace after decimation (default: %(default)s).",
    )
    parser.add_argument(
        "--plot-rate",
        type=float,
        default=DEFAULT_PLOT_RATE,
        help="GUI update rate in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default=DEFAULT_THEME,
        help="GUI theme (default: %(default)s).",
    )
    parser.add_argument(
        "--antialias",
        action="store_true",
        help="Enable PyQtGraph antialiasing.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject obviously invalid values before opening the GUI."""
    if args.baudrate <= 0:
        parser.error("--baudrate must be greater than 0")
    if args.timeout <= 0.0:
        parser.error("--timeout must be greater than 0")
    if args.pipeline_depth <= 0:
        parser.error("--pipeline-depth must be greater than 0")
    if args.deterministic_rate <= 0.0:
        parser.error("--deterministic-rate must be greater than 0")
    if args.history <= 0:
        parser.error("--history must be greater than 0")
    if args.max_points <= 0:
        parser.error("--max-points must be greater than 0")
    if args.plot_rate <= 0.0:
        parser.error("--plot-rate must be greater than 0")


def make_source(args: argparse.Namespace) -> tuple[SignalBatchSource, str]:
    """Create the selected source and a UI label for it."""
    axis = cast(str, args.axis)
    scalar_source: SignalSource
    if args.source == "vesc":
        port = cast("str | None", args.port) or autodetect_port()
        scalar_source = VescImuSignalSource(
            port=port,
            baudrate=cast(int, args.baudrate),
            axis=axis,
            timeout=cast(float, args.timeout),
            pipeline_depth=cast(int, args.pipeline_depth),
            pending_samples=DEFAULT_PENDING_SAMPLES,
            exclusive=not cast(bool, args.no_exclusive),
        )
        return (
            ScalarSignalSourceAdapter(scalar_source),
            f"VESC IMU axis source: {axis} via {port}",
        )

    if args.source == "deterministic":
        scalar_source = DeterministicSignalSource(
            channel_name=axis,
            unit=imu_axis_unit(axis),
            sample_rate_hz=cast(float, args.deterministic_rate),
            pending_samples=DEFAULT_PENDING_SAMPLES,
        )
        return (
            ScalarSignalSourceAdapter(scalar_source),
            f"Deterministic source @ {cast(float, args.deterministic_rate):g} Hz",
        )

    if args.source == "deterministic-noisy":
        scalar_source = NoisyDeterministicSignalSource(
            channel_name=axis,
            unit=imu_axis_unit(axis),
            sample_rate_hz=cast(float, args.deterministic_rate),
            pending_samples=DEFAULT_PENDING_SAMPLES,
        )
        return (
            ScalarSignalSourceAdapter(scalar_source),
            f"Deterministic noisy source @ {cast(float, args.deterministic_rate):g} Hz",
        )

    scalar_source = DeterministicWhiteNoiseSignalSource(
        channel_name=axis,
        unit=imu_axis_unit(axis),
        sample_rate_hz=cast(float, args.deterministic_rate),
        pending_samples=DEFAULT_PENDING_SAMPLES,
    )
    return (
        ScalarSignalSourceAdapter(scalar_source),
        f"Deterministic white-noise source @ {cast(float, args.deterministic_rate):g} Hz",
    )


def main() -> None:
    """Parse CLI args and run the proof-of-concept analysis app."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    source, source_label = make_source(args)
    axis = cast(str, args.axis)
    unit = imu_axis_unit(axis)
    app = build_analysis(
        source=source,
        source_label=source_label,
        axis=axis,
        unit=unit,
        history=cast(int, args.history),
        max_points=cast(int, args.max_points),
        plot_rate_hz=cast(float, args.plot_rate),
        theme=cast(Literal["light", "dark"], args.theme),
        antialias=cast(bool, args.antialias),
    )
    run_live_analysis(app)


if __name__ == "__main__":
    main()
