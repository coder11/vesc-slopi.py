#!/usr/bin/env python3
"""Plot VESC IMU sample timestamps over retained sample index.

Edit the ``RUN_*`` module variables below to choose the IMU axis and runtime
parameters.

Examples:
    uv run examples/yalsa/vesc_imu_timestamp_index.py
    uv run examples/yalsa/vesc_imu_timestamp_index.py --serial /dev/ttyACM0
    uv run examples/yalsa/vesc_imu_timestamp_index.py --ble AA:BB:CC:DD:EE:FF
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vesc_py.connection import VescTarget
from vesc_py.connection_cli import run_vesc_connection_cli
from vesc_py.fast_imu_source import VescImuSignalSource, parse_imu_axis
from vesc_py.live_signal import SignalSourceStats
from yalsa import (
    AnalysisInput,
    AnalysisResult,
    LiveAnalysisApp,
    ParamValue,
    PlotSpec,
    PlotTrace,
    ProcessCallback,
    SignalBatch,
    SignalBatchSource,
    SignalBatchSourceStats,
    run_live_analysis,
    xy_series,
)

DEFAULT_HISTORY = 5_000
DEFAULT_MAX_POINTS = 2_000
DEFAULT_PLOT_RATE = 60.0
DEFAULT_WORKER_DRAIN_STRIDE = 1
DEFAULT_PENDING_SAMPLES = 20_000
DEFAULT_VESC_POLL_RATE = 500.0
# DEFAULT_VESC_POLL_RATE = None
DEFAULT_TIMEOUT = 0.1
DEFAULT_AXIS = "acc_z"
RESPONSE_LATENCY_CHANNEL = "response_latency"
REQUEST_LATENESS_CHANNEL = "request_lateness"
MISSED_SLOTS_CHANNEL = "missed_slots"

# Edit these values directly instead of passing example-specific CLI flags.
RUN_AXIS = DEFAULT_AXIS
RUN_VESC_POLL_RATE: float | None = DEFAULT_VESC_POLL_RATE
RUN_TIMEOUT = DEFAULT_TIMEOUT


def _empty_array() -> npt.NDArray[np.float64]:
    return np.empty(0, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VescImuTimestampIndexConfig:
    """Validated runtime options for the timestamp/index example."""

    axis: str = DEFAULT_AXIS
    vesc_poll_rate: float | None = DEFAULT_VESC_POLL_RATE
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        if self.vesc_poll_rate is not None and self.vesc_poll_rate <= 0.0:
            raise ValueError("vesc_poll_rate must be greater than 0")
        if self.timeout <= 0.0:
            raise ValueError("timeout must be greater than 0")
        object.__setattr__(self, "axis", parse_imu_axis(self.axis))


def build_runtime_config() -> VescImuTimestampIndexConfig:
    """Build the example configuration from module-level settings."""
    return VescImuTimestampIndexConfig(
        axis=RUN_AXIS,
        vesc_poll_rate=RUN_VESC_POLL_RATE,
        timeout=RUN_TIMEOUT,
    )


def build_timestamp_index_processor() -> ProcessCallback:
    """Return a processing callback that plots timestamp and sample lag."""

    def process(
        data: AnalysisInput,
        _params: Mapping[str, ParamValue],
    ) -> AnalysisResult:
        timestamps = data.timestamps_s
        response_latency_s = data.channel(RESPONSE_LATENCY_CHANNEL)
        request_lateness_s = data.channel(REQUEST_LATENESS_CHANNEL)
        missed_slots = data.channel(MISSED_SLOTS_CHANNEL)
        sample_indexes = np.arange(timestamps.size, dtype=np.float64)
        lag_s = np.diff(timestamps)
        lag_sample_indexes = np.arange(1, timestamps.size, dtype=np.float64)
        if int(timestamps.size) == 0:
            status_text = "waiting for samples"
        else:
            lag_text = "sample lag measuring"
            if int(lag_s.size) > 0:
                lag_text = f"latest lag: {lag_s[-1]:.6f} s"
            latency_text = "latency measuring"
            if int(response_latency_s.size) > 0:
                latency_text = f"latest latency: {response_latency_s[-1]:.6f} s"
            lateness_text = "lateness measuring"
            if int(request_lateness_s.size) > 0:
                lateness_text = f"latest lateness: {request_lateness_s[-1]:.6f} s"
            missed_text = "missed slots measuring"
            if int(missed_slots.size) > 0:
                missed_text = f"latest missed slots: {missed_slots[-1]:.0f}"
            status_text = (
                f"samples: {timestamps.size} | "
                f"latest timestamp: {timestamps[-1]:.6f} s | "
                f"{lag_text} | "
                f"{latency_text} | "
                f"{lateness_text} | "
                f"{missed_text}"
            )

        return AnalysisResult(
            series={
                "timestamp": xy_series(sample_indexes, timestamps),
                "sample_lag": xy_series(lag_sample_indexes, lag_s),
                "response_latency": xy_series(sample_indexes, response_latency_s),
                "request_lateness": xy_series(sample_indexes, request_lateness_s),
                "missed_slots": xy_series(sample_indexes, missed_slots),
            },
            status_text=status_text,
        )

    return process


def build_analysis(
    *,
    source: SignalBatchSource,
    source_label: str,
) -> LiveAnalysisApp:
    """Build the declarative timestamp/index app consumed by YALSA."""
    return LiveAnalysisApp(
        title="VESC IMU Timestamp Over Sample Index",
        source=source,
        source_label=source_label,
        history=DEFAULT_HISTORY,
        plot_rate_hz=DEFAULT_PLOT_RATE,
        drain_stride=DEFAULT_WORKER_DRAIN_STRIDE,
        plots=(
            PlotSpec(
                title="Lag Between Samples",
                traces=(
                    PlotTrace(
                        series="sample_lag",
                        label="sample lag",
                        color="#d62728",
                    ),
                ),
                x_label="sample index",
                y_label="lag",
                y_unit="s",
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=True,
                allow_mouse_x=True,
                allow_mouse_y=True,
                mouse_mode="rect",
            ),
            PlotSpec(
                title="VESC Response Latency",
                traces=(
                    PlotTrace(
                        series="response_latency",
                        label="response latency",
                        color="#2ca02c",
                    ),
                ),
                x_label="sample index",
                y_label="latency",
                y_unit="s",
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=True,
                allow_mouse_x=True,
                allow_mouse_y=True,
                mouse_mode="rect",
            ),
            PlotSpec(
                title="Request Lateness",
                traces=(
                    PlotTrace(
                        series="request_lateness",
                        label="request lateness",
                        color="#9467bd",
                    ),
                ),
                x_label="sample index",
                y_label="lateness",
                y_unit="s",
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=True,
                allow_mouse_x=True,
                allow_mouse_y=True,
                mouse_mode="rect",
            ),
            PlotSpec(
                title="Missed Poll Slots",
                traces=(
                    PlotTrace(
                        series="missed_slots",
                        label="missed slots",
                        color="#ff7f0e",
                    ),
                ),
                x_label="sample index",
                y_label="slots",
                y_unit="",
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=True,
                allow_mouse_x=True,
                allow_mouse_y=True,
                mouse_mode="rect",
            ),
            PlotSpec(
                title="Timestamp Over Sample Index",
                traces=(
                    PlotTrace(
                        series="timestamp",
                        label="timestamp",
                        color="#1f77b4",
                    ),
                ),
                x_label="sample index",
                y_label="timestamp",
                y_unit="s",
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=True,
                allow_mouse_x=True,
                allow_mouse_y=True,
                mouse_mode="rect",
            ),
        ),
        process=build_timestamp_index_processor(),
    )


class VescImuTimestampBatchSource:
    """Expose IMU values and transport latency as one timestamp-aligned batch."""

    def __init__(self, source: VescImuSignalSource) -> None:
        self._source = source
        self._channels = {
            source.channel_name: source.unit,
            source.response_latency_channel_name: source.response_latency_unit,
            source.request_lateness_channel_name: source.request_lateness_unit,
            source.missed_slots_channel_name: source.missed_slots_unit,
        }

    @property
    def channels(self) -> Mapping[str, str]:
        return self._channels

    def start(self) -> None:
        self._source.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._source.stop(timeout=timeout)

    def drain(self) -> tuple[SignalBatch, SignalBatchSourceStats]:
        timestamps, values, latencies, latenesses, missed_slots, stats = (
            self._source.drain_with_timing()
        )
        timestamps, values, latencies, latenesses, missed_slots = _align_drained_arrays(
            timestamps,
            values,
            latencies,
            latenesses,
            missed_slots,
        )
        batch = SignalBatch(
            timestamps_s=timestamps,
            values={
                self._source.channel_name: values,
                self._source.response_latency_channel_name: latencies,
                self._source.request_lateness_channel_name: latenesses,
                self._source.missed_slots_channel_name: missed_slots,
            },
            units=self._channels,
        )
        return batch, self._batch_stats(stats, latencies, latenesses, missed_slots)

    def source_stats(self) -> SignalBatchSourceStats:
        return self._batch_stats(
            self._source.source_stats(),
            _latest_latency_array(self._source),
            _latest_lateness_array(self._source),
            _latest_missed_slots_array(self._source),
        )

    def _batch_stats(
        self,
        stats: SignalSourceStats,
        latencies: npt.NDArray[np.float64],
        latenesses: npt.NDArray[np.float64],
        missed_slots: npt.NDArray[np.float64],
    ) -> SignalBatchSourceStats:
        latest_values: dict[str, float] = {}
        if stats.latest_value is not None:
            latest_values[self._source.channel_name] = stats.latest_value
        if int(latencies.size) > 0:
            latest_values[self._source.response_latency_channel_name] = float(
                latencies[-1]
            )
        if int(latenesses.size) > 0:
            latest_values[self._source.request_lateness_channel_name] = float(
                latenesses[-1]
            )
        if int(missed_slots.size) > 0:
            latest_values[self._source.missed_slots_channel_name] = float(
                missed_slots[-1]
            )
        return SignalBatchSourceStats(
            samples=stats.samples,
            dropped=stats.dropped,
            errors=stats.errors,
            average_rate_hz=stats.average_rate_hz,
            latest_sample_s=stats.latest_sample_s,
            latest_values=latest_values,
            last_error=stats.last_error,
            done=stats.done,
            rate_label="VESC poll rate",
        )


def _align_drained_arrays(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    latencies: npt.NDArray[np.float64],
    latenesses: npt.NDArray[np.float64],
    missed_slots: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    count = min(
        int(timestamps.size),
        int(values.size),
        int(latencies.size),
        int(latenesses.size),
        int(missed_slots.size),
    )
    if (
        count
        == int(timestamps.size)
        == int(values.size)
        == int(latencies.size)
        == int(latenesses.size)
        == int(missed_slots.size)
    ):
        return timestamps, values, latencies, latenesses, missed_slots
    if count == 0:
        return (
            _empty_array(),
            _empty_array(),
            _empty_array(),
            _empty_array(),
            _empty_array(),
        )
    return (
        timestamps[-count:],
        values[-count:],
        latencies[-count:],
        latenesses[-count:],
        missed_slots[-count:],
    )


def _latest_latency_array(source: VescImuSignalSource) -> npt.NDArray[np.float64]:
    stats = source.response_latency_stats()
    if stats.latest_value is None:
        return _empty_array()
    return np.asarray((stats.latest_value,), dtype=np.float64)


def _latest_lateness_array(source: VescImuSignalSource) -> npt.NDArray[np.float64]:
    stats = source.request_lateness_stats()
    if stats.latest_value is None:
        return _empty_array()
    return np.asarray((stats.latest_value,), dtype=np.float64)


def _latest_missed_slots_array(source: VescImuSignalSource) -> npt.NDArray[np.float64]:
    stats = source.missed_slots_stats()
    if stats.latest_value is None:
        return _empty_array()
    return np.asarray((stats.latest_value,), dtype=np.float64)


def make_source(
    config: VescImuTimestampIndexConfig,
    *,
    vesc_target: VescTarget,
) -> tuple[SignalBatchSource, str]:
    """Create the VESC IMU source and a UI label for it."""
    source = VescImuSignalSource(
        connection=vesc_target.connection,
        axis=config.axis,
        timeout=config.timeout,
        pending_samples=DEFAULT_PENDING_SAMPLES,
        poll_rate_hz=config.vesc_poll_rate,
        can_id=vesc_target.can_id,
    )
    target_label = "direct controller"
    if vesc_target.can_id is not None:
        target_label = f"CAN {vesc_target.can_id}"
    rate_label = "uncapped"
    if config.vesc_poll_rate is not None:
        rate_label = f"<= {config.vesc_poll_rate:g} Hz"
    source_label = (
        f"VESC IMU timestamp source via {vesc_target.connection.describe()} "
        f"({target_label}; axis {source.channel_name}; {rate_label})"
    )
    return VescImuTimestampBatchSource(source), source_label


def main(argv: Sequence[str] | None = None) -> None:
    """Run the timestamp/index example."""
    config = build_runtime_config()
    vesc_argv = () if argv is None else tuple(argv)
    vesc_target = run_vesc_connection_cli(
        (*vesc_argv, "--timeout", str(config.timeout))
    )
    source, source_label = make_source(config, vesc_target=vesc_target)
    run_app = build_analysis(source=source, source_label=source_label)
    run_live_analysis(run_app)


if __name__ == "__main__":
    main()
