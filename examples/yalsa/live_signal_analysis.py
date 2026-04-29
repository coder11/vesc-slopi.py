#!/usr/bin/env python3
"""YALSA app with a VESC IMU axis proof of concept.

Edit the ``RUN_*`` module variables below to choose the signal source and
runtime parameters.

Examples:
    uv run examples/yalsa/live_signal_analysis.py
    uv run examples/yalsa/live_signal_analysis.py --serial /dev/ttyACM0
    uv run examples/yalsa/live_signal_analysis.py --ble AA:BB:CC:DD:EE:FF
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from vesc_py.connection import (
    BlockingIo,
    VescConnection,
    VescTarget,
    build_imu_request,
    open_blocking_io,
)
from vesc_py.connection_cli import run_vesc_connection_cli
from vesc_py.fast_imu_source import (
    ImuSourceStats,
    _read_expected_imu_packet,
    _wait_until_ns,
    imu_axes_mask,
    imu_axis_unit,
    imu_data_from_payload,
    parse_imu_axes,
    parse_imu_axis,
)
from vesc_py.live_signal import (
    NSEC_PER_SEC,
    deterministic_noisy_signal_value,
    deterministic_signal_value,
    deterministic_white_noise,
)
from yalsa import (
    AnalysisInput,
    AnalysisResult,
    ChoiceOption,
    LiveAnalysisApp,
    ParamValue,
    PlotSpec,
    PlotTrace,
    ProcessCallback,
    SeriesData,
    SignalBatch,
    SignalBatchSource,
    SignalBatchSourceStats,
    butter_lowpass_hz,
    choice_parameter,
    fft_magnitude,
    float_parameter,
    int_parameter,
    merge_signal_batch_source_stats,
    pending_batch_ring_stats,
    run_live_analysis,
    signal_stats,
    welch_psd,
    xy_series,
)

DEFAULT_HISTORY = 5_000
DEFAULT_MAX_POINTS = 1_200
DEFAULT_PLOT_RATE = 30.0
DEFAULT_PENDING_SAMPLES = 20_000
DEFAULT_DETERMINISTIC_RATE = 500.0
# DEFAULT_VESC_POLL_RATE = 1000
DEFAULT_VESC_POLL_RATE = None
DEFAULT_CUTOFF_HZ = 15.0
DEFAULT_FILTER_ORDER = 2
DEFAULT_THEME: Literal["light", "dark"] = "light"
DEFAULT_SOURCE = "vesc"
RAW_TRACE_COLOR = "#afafaf"
FILTERED_TRACE_COLOR = "#1f77b4"
SOURCE_CHOICES = (
    DEFAULT_SOURCE,
    "deterministic",
    "deterministic-noisy",
    "deterministic-white-noise",
)
DEFAULT_TIMEOUT = 0.1

# Edit these values directly instead of passing example-specific CLI flags.
RUN_SOURCE = DEFAULT_SOURCE
RUN_AXIS = "acc_z"
RUN_DETERMINISTIC_RATE = DEFAULT_DETERMINISTIC_RATE
RUN_VESC_POLL_RATE = DEFAULT_VESC_POLL_RATE
RUN_TIMEOUT = DEFAULT_TIMEOUT
RUN_GYRO_AXIS = "gyro_z"

SPECTRUM_OPTIONS = (
    ChoiceOption(value="psd", label="PSD"),
    ChoiceOption(value="fft", label="FFT"),
)


@dataclass(frozen=True, slots=True)
class LiveSignalAnalysisConfig:
    """Validated runtime options for the live signal analysis example."""

    source: str = DEFAULT_SOURCE
    axis: str = "acc_z"
    gyro_axis: str = "gyro_z"
    deterministic_rate: float = DEFAULT_DETERMINISTIC_RATE
    vesc_poll_rate: float | None = DEFAULT_VESC_POLL_RATE
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        if self.source not in SOURCE_CHOICES:
            raise ValueError(f"unsupported source {self.source!r}")
        if self.deterministic_rate <= 0.0:
            raise ValueError("deterministic_rate must be greater than 0")
        if self.vesc_poll_rate is not None and self.vesc_poll_rate <= 0.0:
            raise ValueError("vesc_poll_rate must be greater than 0")
        if self.timeout <= 0.0:
            raise ValueError("timeout must be greater than 0")
        object.__setattr__(self, "axis", parse_imu_axis(self.axis))
        object.__setattr__(self, "gyro_axis", parse_imu_axis(self.gyro_axis))
        parse_imu_axes((self.axis, self.gyro_axis))


def build_runtime_config() -> LiveSignalAnalysisConfig:
    """Build the example configuration from module-level settings."""
    return LiveSignalAnalysisConfig(
        source=RUN_SOURCE,
        axis=RUN_AXIS,
        gyro_axis=RUN_GYRO_AXIS,
        deterministic_rate=RUN_DETERMINISTIC_RATE,
        vesc_poll_rate=RUN_VESC_POLL_RATE,
        timeout=RUN_TIMEOUT,
    )


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


class PendingSignalBatchBuffer:
    """Thread-safe overwrite ring for pending multi-channel samples."""

    def __init__(self, capacity: int, channels: Mapping[str, str]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        if not channels:
            raise ValueError("channels must not be empty")
        self._capacity = capacity
        self._channels = dict(channels)
        self._channel_names = tuple(self._channels)
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._values = {
            channel_name: np.zeros(capacity, dtype=np.float64)
            for channel_name in self._channel_names
        }
        self._lock = threading.Lock()
        self._read_index = 0
        self._write_index = 0
        self._count = 0
        self._dropped = 0
        self._cumulative_dropped = 0

    def append_many(
        self,
        timestamps: Sequence[float],
        values: Mapping[str, Sequence[float]],
    ) -> None:
        """Append timestamp-aligned samples, overwriting oldest pending data."""
        count = len(timestamps)
        if count == 0:
            return

        received = set(values)
        expected = set(self._channel_names)
        if received != expected:
            raise ValueError("values channels must match source channels")

        timestamp_values = np.asarray(timestamps, dtype=np.float64)
        channel_values = {
            channel_name: np.asarray(values[channel_name], dtype=np.float64)
            for channel_name in self._channel_names
        }
        for channel_name, channel_array in channel_values.items():
            if int(channel_array.size) != count:
                raise ValueError(
                    f"channel {channel_name!r} length does not match timestamps"
                )

        with self._lock:
            if count >= self._capacity:
                drop = self._count + count - self._capacity
                self._dropped += drop
                self._cumulative_dropped += drop
                self._timestamps[:] = timestamp_values[-self._capacity :]
                for channel_name in self._channel_names:
                    self._values[channel_name][:] = channel_values[channel_name][
                        -self._capacity :
                    ]
                self._read_index = 0
                self._write_index = 0
                self._count = self._capacity
                return

            overflow = max(0, self._count + count - self._capacity)
            if overflow > 0:
                self._read_index = (self._read_index + overflow) % self._capacity
                self._dropped += overflow
                self._cumulative_dropped += overflow
            self._count = min(self._capacity, self._count + count)

            first_count = min(count, self._capacity - self._write_index)
            self._timestamps[self._write_index : self._write_index + first_count] = (
                timestamp_values[:first_count]
            )
            for channel_name in self._channel_names:
                self._values[channel_name][
                    self._write_index : self._write_index + first_count
                ] = channel_values[channel_name][:first_count]

            remaining = count - first_count
            if remaining > 0:
                self._timestamps[:remaining] = timestamp_values[first_count:]
                for channel_name in self._channel_names:
                    self._values[channel_name][:remaining] = channel_values[
                        channel_name
                    ][first_count:]

            self._write_index = (self._write_index + count) % self._capacity

    def pending_source_stats(self, *, rate_label: str) -> SignalBatchSourceStats:
        """Stats from the current pending ring without consuming samples."""
        with self._lock:
            batch = self._pending_batch_locked()
            cumulative = self._cumulative_dropped
        return pending_batch_ring_stats(
            batch, cumulative_dropped=cumulative, rate_label=rate_label
        )

    def drain(
        self,
        *,
        rate_label: str,
    ) -> tuple[SignalBatch, SignalBatchSourceStats]:
        """Return pending samples in order, clear the ring, and ring-aligned stats."""
        with self._lock:
            self._dropped = 0
            count = self._count
            if count == 0:
                empty = np.empty(0, dtype=np.float64)
                batch = SignalBatch(
                    timestamps_s=empty,
                    values={
                        channel_name: empty for channel_name in self._channel_names
                    },
                    units=self._channels,
                )
                stats = pending_batch_ring_stats(
                    batch,
                    cumulative_dropped=self._cumulative_dropped,
                    rate_label=rate_label,
                )
                return batch, stats

            read_index = self._read_index
            if read_index + count <= self._capacity:
                timestamps = self._timestamps[read_index : read_index + count].copy()
                values_map = {
                    channel_name: self._values[channel_name][
                        read_index : read_index + count
                    ].copy()
                    for channel_name in self._channel_names
                }
            else:
                first_count = self._capacity - read_index
                timestamps = np.concatenate(
                    (
                        self._timestamps[read_index:],
                        self._timestamps[: count - first_count],
                    )
                )
                values_map = {
                    channel_name: np.concatenate(
                        (
                            self._values[channel_name][read_index:],
                            self._values[channel_name][: count - first_count],
                        )
                    )
                    for channel_name in self._channel_names
                }

            self._read_index = self._write_index
            self._count = 0
            batch = SignalBatch(
                timestamps_s=timestamps,
                values=values_map,
                units=self._channels,
            )
            stats = pending_batch_ring_stats(
                batch,
                cumulative_dropped=self._cumulative_dropped,
                rate_label=rate_label,
            )
            return batch, stats

    def _pending_batch_locked(self) -> SignalBatch:
        count = self._count
        if count == 0:
            empty = np.empty(0, dtype=np.float64)
            return SignalBatch(
                timestamps_s=empty,
                values={
                    channel_name: empty for channel_name in self._channel_names
                },
                units=self._channels,
            )
        read_index = self._read_index
        if read_index + count <= self._capacity:
            timestamps = self._timestamps[read_index : read_index + count].copy()
            values_map = {
                channel_name: self._values[channel_name][
                    read_index : read_index + count
                ].copy()
                for channel_name in self._channel_names
            }
        else:
            first_count = self._capacity - read_index
            timestamps = np.concatenate(
                (
                    self._timestamps[read_index:],
                    self._timestamps[: count - first_count],
                )
            )
            values_map = {
                channel_name: np.concatenate(
                    (
                        self._values[channel_name][read_index:],
                        self._values[channel_name][: count - first_count],
                    )
                )
                for channel_name in self._channel_names
            }
        return SignalBatch(
            timestamps_s=timestamps,
            values=values_map,
            units=self._channels,
        )


class DeterministicImuBatchSource:
    """Two-channel deterministic IMU source for offline example runs."""

    def __init__(
        self,
        *,
        source: str,
        axes: Sequence[str],
        sample_rate_hz: float,
        pending_samples: int,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be greater than 0")
        self._source = source
        self._axes = parse_imu_axes(axes)
        self._sample_rate_hz = sample_rate_hz
        self._channels = {axis: imu_axis_unit(axis) for axis in self._axes}
        self._samples = PendingSignalBatchBuffer(pending_samples, self._channels)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._start_ns = 0
        self._sample_index = 0

    @property
    def channels(self) -> Mapping[str, str]:
        return self._channels

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._done.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="deterministic-imu-batch-source",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def drain(self) -> tuple[SignalBatch, SignalBatchSourceStats]:
        batch, ring = self._samples.drain(rate_label="Data acquisition rate")
        with self._lock:
            done = self._done.is_set()
        return batch, merge_signal_batch_source_stats(
            ring, errors=0, last_error=None, done=done
        )

    def source_stats(self) -> SignalBatchSourceStats:
        ring = self._samples.pending_source_stats(rate_label="Data acquisition rate")
        with self._lock:
            done = self._done.is_set()
        return merge_signal_batch_source_stats(
            ring, errors=0, last_error=None, done=done
        )

    def _value_for_sample(self, sample_index: int) -> float:
        if self._source == "deterministic-noisy":
            return deterministic_noisy_signal_value(sample_index, self._sample_rate_hz)
        if self._source == "deterministic-white-noise":
            return deterministic_white_noise(sample_index)
        return deterministic_signal_value(sample_index, self._sample_rate_hz)

    def _run(self) -> None:
        self._start_ns = time.perf_counter_ns()
        next_sample_ns = self._start_ns
        sample_period_ns = max(1, round(NSEC_PER_SEC / self._sample_rate_hz))
        batch_size = 32
        try:
            while not self._stop.is_set():
                now_ns = time.perf_counter_ns()
                if now_ns < next_sample_ns:
                    sleep_s = min((next_sample_ns - now_ns) / NSEC_PER_SEC, 0.01)
                    time.sleep(max(0.0, sleep_s))
                    continue

                due_count = min(
                    batch_size,
                    max(1, ((now_ns - next_sample_ns) // sample_period_ns) + 1),
                )
                indexes = np.arange(
                    self._sample_index,
                    self._sample_index + due_count,
                    dtype=np.int64,
                )
                timestamps = (
                    indexes.astype(np.float64) / self._sample_rate_hz
                ).tolist()
                values = {
                    axis: [self._value_for_sample(int(index)) for index in indexes]
                    for axis in self._axes
                }
                self._samples.append_many(timestamps, values)
                self._sample_index += due_count
                next_sample_ns += due_count * sample_period_ns
        finally:
            self._done.set()


class VescImuBatchSignalSource:
    """Batch source backed by one direct VESC IMU polling loop."""

    def __init__(
        self,
        *,
        connection: VescConnection,
        axes: Sequence[str],
        timeout: float,
        pending_samples: int,
        poll_rate_hz: float | None = None,
        can_id: int | None = None,
    ) -> None:
        if timeout <= 0.0:
            raise ValueError("timeout must be greater than 0")
        if poll_rate_hz is not None and poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be greater than 0")
        if can_id is not None and not 0 <= can_id <= 253:
            raise ValueError("can_id must be in range 0..253")

        self._connection = connection
        self._axes = parse_imu_axes(axes)
        self._timeout = timeout
        self._poll_rate_hz = poll_rate_hz
        self._poll_interval_ns = (
            None if poll_rate_hz is None else max(1, round(NSEC_PER_SEC / poll_rate_hz))
        )
        self._can_id = can_id
        self._channels = {axis: imu_axis_unit(axis) for axis in self._axes}
        self._request = build_imu_request(imu_axes_mask(self._axes), can_id=can_id)
        self._samples = PendingSignalBatchBuffer(pending_samples, self._channels)
        self._stats = ImuSourceStats()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial_port: BlockingIo | None = None
        self._start_ns = 0
        self._last_error: str | None = None

    @property
    def channels(self) -> Mapping[str, str]:
        return self._channels

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._done.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="vesc-imu-batch-signal-source",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._serial_port is not None:
            self._serial_port.close()
            self._serial_port = None

    def drain(self) -> tuple[SignalBatch, SignalBatchSourceStats]:
        batch, ring = self._samples.drain(rate_label="VESC poll rate")
        with self._lock:
            done = self._done.is_set()
            last_error = self._last_error
            errors = self._stats.errors
        return batch, merge_signal_batch_source_stats(
            ring,
            errors=errors,
            last_error=last_error,
            done=done,
            rate_label="VESC poll rate",
        )

    def source_stats(self) -> SignalBatchSourceStats:
        ring = self._samples.pending_source_stats(rate_label="VESC poll rate")
        with self._lock:
            done = self._done.is_set()
            last_error = self._last_error
            errors = self._stats.errors
        return merge_signal_batch_source_stats(
            ring,
            errors=errors,
            last_error=last_error,
            done=done,
            rate_label="VESC poll rate",
        )

    def _run(self) -> None:
        self._start_ns = time.perf_counter_ns()
        pending_timestamps: list[float] = []
        pending_values: dict[str, list[float]] = {axis: [] for axis in self._axes}
        next_flush_ns = self._start_ns + 5_000_000
        previous_request_ns: int | None = None
        try:
            self._serial_port = open_blocking_io(
                self._connection,
                timeout=self._timeout,
            )
            while not self._stop.is_set():
                serial_port = self._serial_port
                if serial_port is None:
                    break

                if (
                    self._poll_interval_ns is not None
                    and previous_request_ns is not None
                ):
                    next_request_ns = previous_request_ns + self._poll_interval_ns
                    if not _wait_until_ns(next_request_ns, self._stop):
                        break

                previous_request_ns = time.perf_counter_ns()
                serial_port.write(self._request)

                payload = _read_expected_imu_packet(
                    serial_port,
                    self._timeout,
                    self._stats,
                )
                now_ns = time.perf_counter_ns()
                if payload is None:
                    self._stats.timeouts += 1
                    serial_port.reset_input_buffer()
                    continue

                try:
                    values = imu_data_from_payload(payload).axis_values(self._axes)
                except ValueError as exc:
                    self._stats.parse_errors += 1
                    with self._lock:
                        self._last_error = str(exc)
                    continue

                sample_s = (now_ns - self._start_ns) / NSEC_PER_SEC
                pending_timestamps.append(sample_s)
                for axis, value in values.items():
                    pending_values[axis].append(value)
                with self._lock:
                    self._last_error = None

                if len(pending_timestamps) >= 64 or now_ns >= next_flush_ns:
                    self._samples.append_many(pending_timestamps, pending_values)
                    pending_timestamps.clear()
                    for axis in self._axes:
                        pending_values[axis].clear()
                    next_flush_ns = now_ns + 5_000_000
        except Exception as exc:  # noqa: BLE001 - source errors are surfaced in status.
            with self._lock:
                self._stats.parse_errors += 1
                self._last_error = str(exc)
        finally:
            if pending_timestamps:
                self._samples.append_many(pending_timestamps, pending_values)
            if self._serial_port is not None:
                self._serial_port.close()
                self._serial_port = None
            self._done.set()


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
            status_parts.append(
                f"requested cutoff clamped from {requested_cutoff_hz:.2f} Hz"
            )
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


def build_dual_axis_analysis_processor(
    axis: str,
    unit: str,
    gyro_axis: str,
    gyro_unit: str,
) -> ProcessCallback:
    """Return a processing callback for accel plus gyro IMU axes."""

    def process(
        data: AnalysisInput,
        params: Mapping[str, ParamValue],
    ) -> AnalysisResult:
        timestamps = data.timestamps_s
        sample_rate_hz = data.sample_rate_hz
        requested_cutoff_hz = float(cast(float, params["cutoff_hz"]))
        cutoff_hz = clamp_cutoff_hz(requested_cutoff_hz, sample_rate_hz)
        filter_order = int(cast(int, params["filter_order"]))
        spectrum_mode = cast(str, params["spectrum_mode"])

        series: dict[str, SeriesData] = {}
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
            status_parts.append(
                f"requested cutoff clamped from {requested_cutoff_hz:.2f} Hz"
            )

        has_samples = False
        for prefix, selected_axis, selected_unit, label in (
            ("accel", axis, unit, "accel"),
            ("gyro", gyro_axis, gyro_unit, "gyro"),
        ):
            raw = data.channel(selected_axis)
            if int(raw.size) == 0:
                empty_x, empty_y = empty_series()
                series[f"{prefix}_raw"] = xy_series(empty_x, empty_y)
                series[f"{prefix}_filtered"] = xy_series(empty_x, empty_y)
                series[f"{prefix}_raw_spectrum"] = xy_series(empty_x, empty_y)
                series[f"{prefix}_filtered_spectrum"] = xy_series(empty_x, empty_y)
                continue

            has_samples = True
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

            series[f"{prefix}_raw"] = xy_series(timestamps, raw)
            series[f"{prefix}_filtered"] = xy_series(timestamps, filtered)
            series[f"{prefix}_raw_spectrum"] = raw_spectrum_series
            series[f"{prefix}_filtered_spectrum"] = filtered_spectrum_series

            raw_stats = signal_stats(raw)
            filtered_stats = signal_stats(filtered)
            if raw_stats is not None:
                status_parts.append(
                    f"{label} raw RMS: {raw_stats.rms:.6g} {selected_unit}"
                )
            if filtered_stats is not None:
                status_parts.append(
                    f"{label} filtered RMS: {filtered_stats.rms:.6g} {selected_unit}"
                )

        if not has_samples:
            return AnalysisResult(
                series=series,
                status_text="waiting for samples",
            )

        return AnalysisResult(
            series=series,
            status_text=" | ".join(status_parts),
        )

    return process


def build_analysis(
    *,
    source: SignalBatchSource,
    source_label: str,
    axis: str,
    unit: str,
    gyro_axis: str | None = None,
    gyro_unit: str | None = None,
) -> LiveAnalysisApp:
    """Build the declarative analysis app consumed by the generic runtime."""
    plots: tuple[PlotSpec, ...]
    process: ProcessCallback
    if gyro_axis is None or gyro_unit is None:
        plots = (
            PlotSpec(
                title="Time Domain",
                traces=(
                    PlotTrace(series="raw", label="Raw", color=RAW_TRACE_COLOR),
                    PlotTrace(
                        series="filtered",
                        label="Filtered",
                        color=FILTERED_TRACE_COLOR,
                    ),
                ),
                x_label="time",
                x_unit="s",
                y_label=axis,
                y_unit=unit,
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=False,
                allow_mouse_x=False,
                allow_mouse_y=True,
                mouse_mode="rect",
                x_axis_mode="follow_latest",
                allow_left_drag=False,
            ),
            PlotSpec(
                title="Frequency Domain",
                traces=(
                    PlotTrace(
                        series="raw_spectrum",
                        label="Raw",
                        color=RAW_TRACE_COLOR,
                    ),
                    PlotTrace(
                        series="filtered_spectrum",
                        label="Filtered",
                        color=FILTERED_TRACE_COLOR,
                    ),
                ),
                x_label="frequency",
                x_unit="Hz",
                y_label="spectrum",
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=False,
                allow_mouse_x=False,
                allow_mouse_y=True,
                mouse_mode="rect",
                allow_left_drag=False,
            ),
        )
        process = build_axis_analysis_processor(axis, unit)
        title = f"Live Signal Analysis: {axis}"
    else:
        plots = (
            PlotSpec(
                title=f"Accel Time Domain: {axis}",
                traces=(
                    PlotTrace(series="accel_raw", label="Raw", color=RAW_TRACE_COLOR),
                    PlotTrace(
                        series="accel_filtered",
                        label="Filtered",
                        color=FILTERED_TRACE_COLOR,
                    ),
                ),
                x_label="time",
                x_unit="s",
                y_label=axis,
                y_unit=unit,
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=False,
                allow_mouse_x=False,
                allow_mouse_y=True,
                mouse_mode="rect",
                x_axis_mode="follow_latest",
                allow_left_drag=False,
            ),
            PlotSpec(
                title=f"Accel Frequency Domain: {axis}",
                traces=(
                    PlotTrace(
                        series="accel_raw_spectrum",
                        label="Raw",
                        color=RAW_TRACE_COLOR,
                    ),
                    PlotTrace(
                        series="accel_filtered_spectrum",
                        label="Filtered",
                        color=FILTERED_TRACE_COLOR,
                    ),
                ),
                x_label="frequency",
                x_unit="Hz",
                y_label="spectrum",
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=False,
                allow_mouse_x=False,
                allow_mouse_y=True,
                mouse_mode="rect",
                allow_left_drag=False,
            ),
            PlotSpec(
                title=f"Gyro Time Domain: {gyro_axis}",
                traces=(
                    PlotTrace(series="gyro_raw", label="Raw", color=RAW_TRACE_COLOR),
                    PlotTrace(
                        series="gyro_filtered",
                        label="Filtered",
                        color=FILTERED_TRACE_COLOR,
                    ),
                ),
                x_label="time",
                x_unit="s",
                y_label=gyro_axis,
                y_unit=gyro_unit,
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=False,
                allow_mouse_x=False,
                allow_mouse_y=True,
                mouse_mode="rect",
                x_axis_mode="follow_latest",
                allow_left_drag=False,
            ),
            PlotSpec(
                title=f"Gyro Frequency Domain: {gyro_axis}",
                traces=(
                    PlotTrace(
                        series="gyro_raw_spectrum",
                        label="Raw",
                        color=RAW_TRACE_COLOR,
                    ),
                    PlotTrace(
                        series="gyro_filtered_spectrum",
                        label="Filtered",
                        color=FILTERED_TRACE_COLOR,
                    ),
                ),
                x_label="frequency",
                x_unit="Hz",
                y_label="spectrum",
                max_points=DEFAULT_MAX_POINTS,
                auto_range_x=True,
                auto_range_y=False,
                allow_mouse_x=False,
                allow_mouse_y=True,
                mouse_mode="rect",
                allow_left_drag=False,
            ),
        )
        process = build_dual_axis_analysis_processor(axis, unit, gyro_axis, gyro_unit)
        title = f"Live Signal Analysis: {axis} + {gyro_axis}"

    return LiveAnalysisApp(
        title=title,
        source=source,
        source_label=source_label,
        history=DEFAULT_HISTORY,
        plot_rate_hz=DEFAULT_PLOT_RATE,
        theme=DEFAULT_THEME,
        antialias=False,
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
        plots=plots,
        process=process,
    )


def make_source(
    config: LiveSignalAnalysisConfig,
    *,
    vesc_target: VescTarget | None = None,
) -> tuple[SignalBatchSource, str]:
    """Create the selected source and a UI label for it."""
    axes = (config.axis, config.gyro_axis)
    if config.source == DEFAULT_SOURCE:
        if vesc_target is None:
            raise ValueError("vesc_target is required when source='vesc'")
        vesc_source = VescImuBatchSignalSource(
            connection=vesc_target.connection,
            axes=axes,
            timeout=config.timeout,
            pending_samples=DEFAULT_PENDING_SAMPLES,
            poll_rate_hz=config.vesc_poll_rate,
            can_id=vesc_target.can_id,
        )
        target_label = (
            "direct controller"
            if vesc_target.can_id is None
            else f"CAN {vesc_target.can_id}"
        )
        rate_label = (
            "uncapped"
            if config.vesc_poll_rate is None
            else f"<= {config.vesc_poll_rate:g} Hz"
        )
        return (
            vesc_source,
            (
                f"VESC IMU axes source: {', '.join(axes)} "
                f"via {vesc_target.connection.describe()} "
                f"({target_label}; {rate_label})"
            ),
        )

    deterministic_source = DeterministicImuBatchSource(
        source=config.source,
        axes=axes,
        sample_rate_hz=config.deterministic_rate,
        pending_samples=DEFAULT_PENDING_SAMPLES,
    )
    source_label = {
        "deterministic": "Deterministic source",
        "deterministic-noisy": "Deterministic noisy source",
        "deterministic-white-noise": "Deterministic white-noise source",
    }[config.source]
    return deterministic_source, f"{source_label} @ {config.deterministic_rate:g} Hz"


def main(argv: Sequence[str] | None = None) -> None:
    """Run the proof-of-concept analysis app using module-level settings."""
    config = build_runtime_config()
    vesc_target = None
    if config.source == DEFAULT_SOURCE:
        vesc_argv = () if argv is None else tuple(argv)
        vesc_target = run_vesc_connection_cli(
            (*vesc_argv, "--timeout", str(config.timeout))
        )

    source, source_label = make_source(config, vesc_target=vesc_target)
    unit = imu_axis_unit(config.axis)
    gyro_unit = imu_axis_unit(config.gyro_axis)
    app = build_analysis(
        source=source,
        source_label=source_label,
        axis=config.axis,
        unit=unit,
        gyro_axis=config.gyro_axis,
        gyro_unit=gyro_unit,
    )
    run_live_analysis(app)


if __name__ == "__main__":
    main()
