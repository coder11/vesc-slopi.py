"""YALSA application runtime with pluggable live signal sources."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import pickle
import struct
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import cycle
from multiprocessing import shared_memory as mp_shared_memory
from multiprocessing.context import BaseContext
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from vesc_py.live_signal import SignalSource, SignalSourceStats

FloatArray = npt.NDArray[np.float64]
ParamValue: TypeAlias = int | float | bool | str
PlotColor: TypeAlias = str | tuple[int, int, int]

QT_XCB_RUNTIME_LIBS = ("libxcb-cursor.so.0", "libxcb-icccm.so.4")
NSEC_PER_SEC = 1_000_000_000
_ANALYSIS_IDLE_SLEEP_S = 0.001
_WORKER_STATUS_INTERVAL_NS = 50_000_000
_SHARED_CONFIG_BYTES = 1 * 1024 * 1024
_SHARED_CONTROL_BYTES = 1 * 1024 * 1024
_SHARED_STATE_BYTES = 32 * 1024 * 1024
_SUPERVISOR_POLL_S = 0.2
# Rolling window for smoothing measured VESC poll rate in the GUI (not the poll loop).
_VESC_POLL_RATE_DISPLAY_SMA_WINDOW = 16
_PROCESS_STOP_TIMEOUT_S = 2.0
_SHARED_SLOT_MAGIC = b"YALSA001"
_SHARED_SLOT_HEADER = struct.Struct("<8sQQQQ")


def _empty_array() -> FloatArray:
    return np.empty(0, dtype=np.float64)


def _as_float_array(values: npt.ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _is_int_value(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float_value(value: object) -> bool:
    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(
        value, bool
    )


@dataclass(frozen=True, slots=True)
class SignalBatch:
    """Timestamp-aligned signal channels from one source drain."""

    timestamps_s: FloatArray
    values: Mapping[str, FloatArray]
    units: Mapping[str, str]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s)
        if timestamps.ndim != 1:
            raise ValueError("timestamps_s must be one-dimensional")
        sample_count = int(timestamps.size)
        for channel_name, channel_values in self.values.items():
            if channel_name not in self.units:
                raise ValueError(f"missing unit for channel {channel_name!r}")
            array = np.asarray(channel_values)
            if array.ndim != 1:
                raise ValueError(f"channel {channel_name!r} must be one-dimensional")
            if int(array.size) != sample_count:
                raise ValueError(
                    f"channel {channel_name!r} length does not match timestamps"
                )

    @property
    def sample_count(self) -> int:
        return int(self.timestamps_s.size)

    def channel(self, name: str) -> FloatArray:
        return self.values[name]


def empty_signal_batch(channels: Mapping[str, str]) -> SignalBatch:
    """Return an empty batch for the supplied channel map."""
    return SignalBatch(
        timestamps_s=_empty_array(),
        values={name: _empty_array() for name in channels},
        units=dict(channels),
    )


@dataclass(frozen=True, slots=True)
class SignalBatchSourceStats:
    """Low-rate status data exposed by a batch signal source."""

    samples: int
    dropped: int
    errors: int
    average_rate_hz: float
    latest_sample_s: float | None
    latest_values: Mapping[str, float]
    last_error: str | None
    done: bool
    rate_label: str = "Data acquisition rate"


def pending_batch_ring_stats(
    batch: SignalBatch,
    *,
    cumulative_dropped: int,
    rate_label: str = "Data acquisition rate",
) -> SignalBatchSourceStats:
    """Stats aligned with the supplied batch timestamps and channel values."""
    timestamps = batch.timestamps_s
    n = int(timestamps.size)
    if n >= 2:
        elapsed = float(timestamps[-1] - timestamps[0])
        rate = (n - 1) / elapsed if elapsed > 0.0 else 0.0
    else:
        rate = 0.0
    latest_sample_s = float(timestamps[-1]) if n else None
    latest_values = (
        {name: float(batch.values[name][-1]) for name in batch.values} if n else {}
    )
    return SignalBatchSourceStats(
        samples=n,
        dropped=cumulative_dropped,
        errors=0,
        average_rate_hz=rate,
        latest_sample_s=latest_sample_s,
        latest_values=latest_values,
        last_error=None,
        done=False,
        rate_label=rate_label,
    )


def merge_signal_batch_source_stats(
    ring: SignalBatchSourceStats,
    *,
    errors: int,
    last_error: str | None,
    done: bool,
    rate_label: str | None = None,
) -> SignalBatchSourceStats:
    """Overlay transport or lifecycle fields onto ring-aligned stats."""
    return SignalBatchSourceStats(
        samples=ring.samples,
        dropped=ring.dropped,
        errors=errors,
        average_rate_hz=ring.average_rate_hz,
        latest_sample_s=ring.latest_sample_s,
        latest_values=dict(ring.latest_values),
        last_error=last_error,
        done=done,
        rate_label=ring.rate_label if rate_label is None else rate_label,
    )


class SignalBatchSource(Protocol):
    """Common interface for live signal sources exposed to the analysis app."""

    @property
    def channels(self) -> Mapping[str, str]: ...

    def start(self) -> None: ...

    def stop(self, timeout: float = 1.0) -> None: ...

    def drain(self) -> tuple[SignalBatch, SignalBatchSourceStats]: ...

    def source_stats(self) -> SignalBatchSourceStats: ...


class ScalarSignalSourceAdapter:
    """Adapt the existing scalar SignalSource protocol to SignalBatchSource."""

    def __init__(self, source: SignalSource) -> None:
        self._source = source
        self._channels = {source.channel_name: source.unit}

    @property
    def channels(self) -> Mapping[str, str]:
        return self._channels

    def start(self) -> None:
        self._source.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._source.stop(timeout=timeout)

    def drain(self) -> tuple[SignalBatch, SignalBatchSourceStats]:
        timestamps, values, stats = self._source.drain()
        batch = SignalBatch(
            timestamps_s=timestamps,
            values={self._source.channel_name: values},
            units=self._channels,
        )
        latest_values: dict[str, float] = {}
        if stats.latest_value is not None:
            latest_values[self._source.channel_name] = stats.latest_value
        return batch, _scalar_stats_to_batch(stats, latest_values)

    def source_stats(self) -> SignalBatchSourceStats:
        stats = self._source.source_stats()
        latest_values: dict[str, float] = {}
        if stats.latest_value is not None:
            latest_values[self._source.channel_name] = stats.latest_value
        return _scalar_stats_to_batch(stats, latest_values)


def _scalar_stats_to_batch(
    stats: SignalSourceStats,
    latest_values: Mapping[str, float],
) -> SignalBatchSourceStats:
    return SignalBatchSourceStats(
        samples=stats.samples,
        dropped=stats.dropped,
        errors=stats.errors,
        average_rate_hz=stats.average_rate_hz,
        latest_sample_s=stats.latest_sample_s,
        latest_values=dict(latest_values),
        last_error=stats.last_error,
        done=stats.done,
        rate_label="Data acquisition rate",
    )


class SignalBatchHistory:
    """Fixed-size ring history for timestamp-aligned channels."""

    def __init__(self, capacity: int, channels: Mapping[str, str]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        if not channels:
            raise ValueError("channels must not be empty")
        self._capacity = capacity
        self._units = dict(channels)
        self._channel_names = tuple(self._units)
        self._count = 0
        self._write_index = 0
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._values = {
            name: np.zeros(capacity, dtype=np.float64) for name in self._channel_names
        }

    @property
    def count(self) -> int:
        return self._count

    @property
    def channels(self) -> Mapping[str, str]:
        return self._units

    def clear(self) -> None:
        self._count = 0
        self._write_index = 0

    def append_batch(self, batch: SignalBatch) -> None:
        self._validate_batch_channels(batch)
        sample_count = batch.sample_count
        if sample_count == 0:
            return

        timestamps = batch.timestamps_s.astype(np.float64, copy=False)
        if sample_count >= self._capacity:
            self._timestamps[:] = timestamps[-self._capacity :]
            for channel_name in self._channel_names:
                channel_values = batch.values[channel_name].astype(
                    np.float64, copy=False
                )
                self._values[channel_name][:] = channel_values[-self._capacity :]
            self._count = self._capacity
            self._write_index = 0
            return

        first_count = min(sample_count, self._capacity - self._write_index)
        self._timestamps[self._write_index : self._write_index + first_count] = (
            timestamps[:first_count]
        )
        for channel_name in self._channel_names:
            channel_values = batch.values[channel_name].astype(np.float64, copy=False)
            self._values[channel_name][
                self._write_index : self._write_index + first_count
            ] = channel_values[:first_count]

        remaining = sample_count - first_count
        if remaining > 0:
            self._timestamps[:remaining] = timestamps[first_count:]
            for channel_name in self._channel_names:
                channel_values = batch.values[channel_name].astype(
                    np.float64, copy=False
                )
                self._values[channel_name][:remaining] = channel_values[first_count:]

        self._write_index = (self._write_index + sample_count) % self._capacity
        self._count = min(self._capacity, self._count + sample_count)

    def snapshot(self) -> SignalBatch:
        return SignalBatch(
            timestamps_s=self.valid_timestamps().copy(),
            values={
                channel_name: self.channel(channel_name).copy()
                for channel_name in self._channel_names
            },
            units=self._units,
        )

    def valid_timestamps(self) -> FloatArray:
        if self._count == 0:
            return self._timestamps[:0]
        return self._ordered_values(self._timestamps)

    def channel(self, name: str) -> FloatArray:
        try:
            source = self._values[name]
        except KeyError as exc:
            raise KeyError(f"unknown channel {name!r}") from exc
        if self._count == 0:
            return source[:0]
        return self._ordered_values(source)

    def sample_hz(self) -> float | None:
        if self._count < 2:
            return None
        timestamps = self.valid_timestamps()
        elapsed = float(timestamps[-1] - timestamps[0])
        if elapsed <= 0.0:
            return None
        return float((self._count - 1) / elapsed)

    def _ordered_values(self, source: FloatArray) -> FloatArray:
        start = (self._write_index - self._count) % self._capacity
        if start + self._count <= self._capacity:
            return source[start : start + self._count]
        return np.concatenate((source[start:], source[: self._write_index]))

    def _validate_batch_channels(self, batch: SignalBatch) -> None:
        expected = set(self._channel_names)
        received = set(batch.values)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            problems: list[str] = []
            if missing:
                problems.append(f"missing channels: {', '.join(missing)}")
            if extra:
                problems.append(f"unexpected channels: {', '.join(extra)}")
            raise ValueError("; ".join(problems))


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    """One discrete value exposed as a GUI combo-box choice."""

    value: str
    label: str


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Description of one live-tunable parameter."""

    name: str
    label: str
    kind: Literal["int", "float", "bool", "choice"]
    default: ParamValue
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    decimals: int | None = None
    choices: tuple[ChoiceOption, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must not be empty")
        if not self.label:
            raise ValueError("parameter label must not be empty")
        if self.kind == "int":
            if not _is_int_value(self.default):
                raise TypeError("int parameter default must be an int")
            if self.decimals is not None:
                raise ValueError("int parameters do not support decimals")
        elif self.kind == "float":
            if not _is_float_value(self.default):
                raise TypeError("float parameter default must be numeric")
        elif self.kind == "bool":
            if not isinstance(self.default, bool):
                raise TypeError("bool parameter default must be a bool")
            if (
                self.minimum is not None
                or self.maximum is not None
                or self.step is not None
            ):
                raise ValueError("bool parameters do not support min/max/step")
            if self.decimals is not None:
                raise ValueError("bool parameters do not support decimals")
        else:
            if not isinstance(self.default, str):
                raise TypeError("choice parameter default must be a string")
            if not self.choices:
                raise ValueError("choice parameters require at least one option")
            values = {choice.value for choice in self.choices}
            if self.default not in values:
                raise ValueError("choice parameter default must match one option")
            if (
                self.minimum is not None
                or self.maximum is not None
                or self.step is not None
            ):
                raise ValueError("choice parameters do not support min/max/step")
            if self.decimals is not None:
                raise ValueError("choice parameters do not support decimals")


def int_parameter(
    name: str,
    *,
    default: int,
    label: str | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
    step: int | None = None,
) -> ParameterSpec:
    """Build an integer parameter spec."""
    return ParameterSpec(
        name=name,
        label=label or name,
        kind="int",
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
    )


def float_parameter(
    name: str,
    *,
    default: float,
    label: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    decimals: int = 3,
) -> ParameterSpec:
    """Build a floating-point parameter spec."""
    return ParameterSpec(
        name=name,
        label=label or name,
        kind="float",
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        decimals=decimals,
    )


def bool_parameter(
    name: str,
    *,
    default: bool,
    label: str | None = None,
) -> ParameterSpec:
    """Build a boolean parameter spec."""
    return ParameterSpec(
        name=name,
        label=label or name,
        kind="bool",
        default=default,
    )


def choice_parameter(
    name: str,
    *,
    default: str,
    choices: tuple[ChoiceOption, ...],
    label: str | None = None,
) -> ParameterSpec:
    """Build a discrete-choice parameter spec."""
    return ParameterSpec(
        name=name,
        label=label or name,
        kind="choice",
        default=default,
        choices=choices,
    )


def default_parameter_values(
    parameters: tuple[ParameterSpec, ...],
) -> dict[str, ParamValue]:
    """Return the default values keyed by parameter name."""
    defaults: dict[str, ParamValue] = {}
    for parameter in parameters:
        if parameter.name in defaults:
            raise ValueError(f"duplicate parameter name {parameter.name!r}")
        defaults[parameter.name] = parameter.default
    return defaults


@dataclass(frozen=True, slots=True)
class SeriesData:
    """One x/y series ready to draw in PyQtGraph."""

    x: FloatArray
    y: FloatArray

    def __post_init__(self) -> None:
        x_values = np.asarray(self.x)
        y_values = np.asarray(self.y)
        if x_values.ndim != 1 or y_values.ndim != 1:
            raise ValueError("series x and y must be one-dimensional")
        if int(x_values.size) != int(y_values.size):
            raise ValueError("series x and y lengths must match")


def xy_series(x: npt.ArrayLike, y: npt.ArrayLike) -> SeriesData:
    """Build a SeriesData value from arbitrary array-like inputs."""
    return SeriesData(x=_as_float_array(x), y=_as_float_array(y))


@dataclass(frozen=True, slots=True)
class PlotTrace:
    """One rendered trace inside a plot widget."""

    series: str
    label: str
    color: PlotColor | None = None
    width: float = 1.5


@dataclass(frozen=True, slots=True)
class PlotSpec:
    """Plot widget description for the generic analysis UI."""

    title: str
    traces: tuple[PlotTrace, ...]
    x_label: str
    y_label: str
    x_unit: str = ""
    y_unit: str = ""
    max_points: int | None = None
    log_x: bool = False
    log_y: bool = False
    auto_range_x: bool = True
    auto_range_y: bool = True
    allow_mouse_x: bool = True
    allow_mouse_y: bool = True
    mouse_mode: Literal["pan", "rect"] = "pan"
    x_axis_mode: Literal["auto", "follow_latest"] = "auto"
    allow_left_drag: bool = True

    def __post_init__(self) -> None:
        if not self.traces:
            raise ValueError("plots must contain at least one trace")
        if self.max_points is not None and self.max_points <= 0:
            raise ValueError("max_points must be greater than 0")
        if self.mouse_mode not in ("pan", "rect"):
            raise ValueError("mouse_mode must be 'pan' or 'rect'")
        if self.x_axis_mode not in ("auto", "follow_latest"):
            raise ValueError("x_axis_mode must be 'auto' or 'follow_latest'")


@dataclass(frozen=True, slots=True)
class AnalysisInput:
    """Current retained source data passed into the processing callback."""

    batch: SignalBatch
    sample_rate_hz: float | None
    source_stats: SignalBatchSourceStats

    @property
    def timestamps_s(self) -> FloatArray:
        return self.batch.timestamps_s

    @property
    def channels(self) -> Mapping[str, FloatArray]:
        return self.batch.values

    @property
    def units(self) -> Mapping[str, str]:
        return self.batch.units

    def channel(self, name: str) -> FloatArray:
        return self.batch.channel(name)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Rendered series and optional status text returned by the process callback."""

    series: Mapping[str, SeriesData]
    status_text: str | None = None


ProcessCallback: TypeAlias = Callable[
    [AnalysisInput, Mapping[str, ParamValue]], AnalysisResult
]


@dataclass(frozen=True, slots=True)
class LiveAnalysisApp:
    """Complete analysis definition consumed by the generic UI runtime."""

    title: str
    source: SignalBatchSource
    plots: tuple[PlotSpec, ...]
    process: ProcessCallback
    parameters: tuple[ParameterSpec, ...] = ()
    history: int = 20_000
    plot_rate_hz: float = 30.0
    drain_stride: int = 1
    source_label: str | None = None
    theme: Literal["light", "dark"] = "light"
    antialias: bool = False

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("title must not be empty")
        if not self.plots:
            raise ValueError("at least one plot is required")
        if self.history <= 0:
            raise ValueError("history must be greater than 0")
        if self.plot_rate_hz <= 0.0:
            raise ValueError("plot_rate_hz must be greater than 0")
        if self.drain_stride < 1:
            raise ValueError("drain_stride must be at least 1")
        default_parameter_values(self.parameters)


@dataclass(frozen=True, slots=True)
class LiveAnalysisUiConfig:
    """Qt-free declaration consumed by the GUI process."""

    title: str
    channels: Mapping[str, str]
    plots: tuple[PlotSpec, ...]
    parameters: tuple[ParameterSpec, ...] = ()
    plot_rate_hz: float = 30.0
    source_label: str | None = None
    theme: Literal["light", "dark"] = "light"
    antialias: bool = False

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("title must not be empty")
        if not self.channels:
            raise ValueError("channels must not be empty")
        if not self.plots:
            raise ValueError("at least one plot is required")
        if self.plot_rate_hz <= 0.0:
            raise ValueError("plot_rate_hz must be greater than 0")
        default_parameter_values(self.parameters)
        object.__setattr__(self, "channels", dict(self.channels))


def live_analysis_ui_config(config: LiveAnalysisApp) -> LiveAnalysisUiConfig:
    """Extract the GUI-only declaration from a full analysis app."""
    return LiveAnalysisUiConfig(
        title=config.title,
        channels=config.source.channels,
        plots=config.plots,
        parameters=config.parameters,
        plot_rate_hz=config.plot_rate_hz,
        source_label=config.source_label,
        theme=config.theme,
        antialias=config.antialias,
    )


@dataclass(frozen=True, slots=True)
class PlotTheme:
    """Simple color bundle for the PyQtGraph UI."""

    pg_background: str
    pg_foreground: str
    window_background: str
    text_color: str
    muted_color: str
    control_background: str
    control_border: str
    control_hover: str
    control_pressed: str
    grid_alpha: float
    line_colors: tuple[tuple[int, int, int], ...]


PLOT_THEMES = {
    "light": PlotTheme(
        pg_background="#ffffff",
        pg_foreground="#202124",
        window_background="#f6f7f9",
        text_color="#202124",
        muted_color="#4f5b66",
        control_background="#ffffff",
        control_border="#c7cdd4",
        control_hover="#eef2f6",
        control_pressed="#e2e8ef",
        grid_alpha=0.22,
        line_colors=(
            (196, 57, 54),
            (28, 128, 75),
            (37, 98, 180),
            (184, 118, 0),
            (109, 76, 155),
            (0, 138, 148),
        ),
    ),
    "dark": PlotTheme(
        pg_background="#000000",
        pg_foreground="#d0d0d0",
        window_background="#000000",
        text_color="#d0d0d0",
        muted_color="#999999",
        control_background="#151515",
        control_border="#3a3a3a",
        control_hover="#202020",
        control_pressed="#2a2a2a",
        grid_alpha=0.3,
        line_colors=(
            (230, 88, 85),
            (80, 190, 120),
            (85, 150, 245),
            (241, 180, 74),
            (194, 138, 255),
            (75, 204, 216),
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class _AnalysisWorkerSnapshot:
    result: AnalysisResult | None
    process_error: str | None
    source_stats: SignalBatchSourceStats
    history_rate_hz: float | None


@dataclass(frozen=True, slots=True)
class _LiveAnalysisControl:
    parameter_values: Mapping[str, ParamValue]
    parameter_revision: int = 0
    clear_revision: int = 0
    stop_requested: bool = False
    shutdown_requested: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_values", dict(self.parameter_values))


@dataclass(frozen=True, slots=True)
class _SharedMemorySlotSpec:
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class _SharedAnalysisMemorySpec:
    config: _SharedMemorySlotSpec
    state: _SharedMemorySlotSpec
    control: _SharedMemorySlotSpec


class _PickleSharedMemorySlot:
    """Double-buffered pickle payload stored in multiprocessing shared memory."""

    def __init__(self, shared_memory: mp_shared_memory.SharedMemory) -> None:
        self._shared_memory = shared_memory
        self._payload_capacity = (shared_memory.size - _SHARED_SLOT_HEADER.size) // 2
        if self._payload_capacity <= 0:
            raise ValueError("shared-memory slot is too small")

    @classmethod
    def create(cls, size: int) -> "_PickleSharedMemorySlot":
        slot = cls(mp_shared_memory.SharedMemory(create=True, size=size))
        slot._write_header(version=0, active_slot=0, length_0=0, length_1=0)
        return slot

    @classmethod
    def attach(cls, spec: _SharedMemorySlotSpec) -> "_PickleSharedMemorySlot":
        return cls(mp_shared_memory.SharedMemory(name=spec.name))

    @property
    def spec(self) -> _SharedMemorySlotSpec:
        return _SharedMemorySlotSpec(
            name=self._shared_memory.name,
            size=self._shared_memory.size,
        )

    def write(self, value: object) -> int:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > self._payload_capacity:
            raise ValueError(
                "shared-memory payload is too large "
                f"({len(payload)} bytes > {self._payload_capacity} bytes)"
            )

        _magic, version, active_slot, length_0, length_1 = self._read_header()
        next_slot = 1 - int(active_slot)
        offset = self._payload_offset(next_slot)
        buffer = cast(Any, self._shared_memory.buf)
        buffer[offset : offset + len(payload)] = payload
        if next_slot == 0:
            length_0 = len(payload)
        else:
            length_1 = len(payload)
        next_version = int(version) + 1
        self._write_header(
            version=next_version,
            active_slot=next_slot,
            length_0=int(length_0),
            length_1=int(length_1),
        )
        return next_version

    def read(self) -> object:
        value, _version = self.read_with_version()
        return value

    def read_with_version(self) -> tuple[object, int]:
        for _attempt in range(5):
            header = self._read_header()
            _magic, version, active_slot, length_0, length_1 = header
            if active_slot not in (0, 1):
                time.sleep(0.0)
                continue
            length = int(length_0 if active_slot == 0 else length_1)
            if length < 0 or length > self._payload_capacity:
                time.sleep(0.0)
                continue
            offset = self._payload_offset(int(active_slot))
            buffer = cast(Any, self._shared_memory.buf)
            payload = bytes(buffer[offset : offset + length])
            if self._read_header() == header:
                if not payload:
                    raise RuntimeError("shared-memory slot has not been initialised")
                return pickle.loads(payload), int(version)
        raise RuntimeError("shared-memory slot changed while being read")

    def close(self) -> None:
        self._shared_memory.close()

    def unlink(self) -> None:
        try:
            self._shared_memory.unlink()
        except FileNotFoundError:
            return

    def _payload_offset(self, slot: int) -> int:
        return _SHARED_SLOT_HEADER.size + slot * self._payload_capacity

    def _read_header(self) -> tuple[bytes, int, int, int, int]:
        header = _SHARED_SLOT_HEADER.unpack_from(cast(Any, self._shared_memory.buf), 0)
        magic, version, active_slot, length_0, length_1 = header
        if magic != _SHARED_SLOT_MAGIC:
            raise RuntimeError("invalid YALSA shared-memory slot")
        return (
            cast(bytes, magic),
            int(version),
            int(active_slot),
            int(length_0),
            int(length_1),
        )

    def _write_header(
        self,
        *,
        version: int,
        active_slot: int,
        length_0: int,
        length_1: int,
    ) -> None:
        _SHARED_SLOT_HEADER.pack_into(
            cast(Any, self._shared_memory.buf),
            0,
            _SHARED_SLOT_MAGIC,
            version,
            active_slot,
            length_0,
            length_1,
        )


class _SharedAnalysisMemory:
    """Shared-memory channels used by the YALSA supervisor and child processes."""

    def __init__(
        self,
        *,
        config: _PickleSharedMemorySlot,
        state: _PickleSharedMemorySlot,
        control: _PickleSharedMemorySlot,
    ) -> None:
        self.config = config
        self.state = state
        self.control = control

    @classmethod
    def create(cls, ui_config: LiveAnalysisUiConfig) -> "_SharedAnalysisMemory":
        memory = cls(
            config=_PickleSharedMemorySlot.create(_SHARED_CONFIG_BYTES),
            state=_PickleSharedMemorySlot.create(_SHARED_STATE_BYTES),
            control=_PickleSharedMemorySlot.create(_SHARED_CONTROL_BYTES),
        )
        memory.config.write(ui_config)
        memory.control.write(
            _LiveAnalysisControl(
                parameter_values=default_parameter_values(ui_config.parameters),
            )
        )
        memory.state.write(_initial_analysis_snapshot())
        return memory

    @classmethod
    def attach(cls, spec: _SharedAnalysisMemorySpec) -> "_SharedAnalysisMemory":
        return cls(
            config=_PickleSharedMemorySlot.attach(spec.config),
            state=_PickleSharedMemorySlot.attach(spec.state),
            control=_PickleSharedMemorySlot.attach(spec.control),
        )

    @property
    def spec(self) -> _SharedAnalysisMemorySpec:
        return _SharedAnalysisMemorySpec(
            config=self.config.spec,
            state=self.state.spec,
            control=self.control.spec,
        )

    def close(self) -> None:
        self.config.close()
        self.state.close()
        self.control.close()

    def unlink(self) -> None:
        self.config.unlink()
        self.state.unlink()
        self.control.unlink()


def _initial_analysis_snapshot() -> _AnalysisWorkerSnapshot:
    return _AnalysisWorkerSnapshot(
        result=None,
        process_error=None,
        source_stats=SignalBatchSourceStats(
            samples=0,
            dropped=0,
            errors=0,
            average_rate_hz=0.0,
            latest_sample_s=None,
            latest_values={},
            last_error=None,
            done=False,
        ),
        history_rate_hz=None,
    )


class _LiveAnalysisWorker:
    """Drain, retain, and process source samples outside the Qt GUI thread."""

    def __init__(
        self,
        config: LiveAnalysisApp,
        parameter_values: Mapping[str, ParamValue],
    ) -> None:
        self._config = config
        self._history = SignalBatchHistory(config.history, config.source.channels)
        self._parameter_values = dict(parameter_values)
        self._parameter_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._clear_requested = threading.Event()
        self._process_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._source_started = False
        self._last_result: AnalysisResult | None = None
        self._last_process_error: str | None = None
        self._source_stats = config.source.source_stats()
        self._history_rate_hz: float | None = None
        self._last_status_update_ns = 0
        self._drain_phase = 0

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._clear_requested.clear()
            self._process_requested.clear()
            with self._state_lock:
                self._last_result = None
                self._last_process_error = None
                self._source_stats = self._config.source.source_stats()
                self._history_rate_hz = None
            self._config.source.start()
            self._source_started = True
            self._thread = threading.Thread(
                target=self._run,
                name="yalsa-analysis-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            if self._source_started:
                self._config.source.stop(timeout=timeout)
                self._source_started = False
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def set_parameter(self, name: str, value: ParamValue) -> None:
        with self._parameter_lock:
            if name not in self._parameter_values:
                raise KeyError(f"unknown parameter {name!r}")
            self._parameter_values[name] = value
        self._process_requested.set()

    def clear_history(self) -> None:
        self._clear_requested.set()
        self._process_requested.set()
        with self._state_lock:
            self._last_result = None
            self._last_process_error = None
            self._history_rate_hz = None

    def snapshot(self) -> _AnalysisWorkerSnapshot:
        with self._state_lock:
            return _AnalysisWorkerSnapshot(
                result=self._last_result,
                process_error=self._last_process_error,
                source_stats=self._source_stats,
                history_rate_hz=self._history_rate_hz,
            )

    def _parameter_snapshot(self) -> dict[str, ParamValue]:
        with self._parameter_lock:
            return dict(self._parameter_values)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._clear_requested.is_set():
                    self._clear_requested.clear()
                    self._history.clear()
                    self._drain_phase = 0

                stride = max(1, self._config.drain_stride)
                forced_process = self._process_requested.is_set()
                if (
                    stride > 1
                    and not forced_process
                    and (self._drain_phase % stride) != 0
                ):
                    self._drain_phase += 1
                    self._update_source_status_if_due()
                    self._stop.wait(_ANALYSIS_IDLE_SLEEP_S)
                    continue

                batch, drain_stats = self._config.source.drain()
                has_new_samples = batch.sample_count > 0
                if has_new_samples:
                    self._history.append_batch(batch)

                forced_process = self._process_requested.is_set()
                if forced_process:
                    self._process_requested.clear()

                if has_new_samples or forced_process:
                    source_stats = (
                        drain_stats
                        if has_new_samples
                        else self._config.source.source_stats()
                    )
                    self._process_latest(source_stats)
                else:
                    self._update_source_status_if_due()
                    self._stop.wait(_ANALYSIS_IDLE_SLEEP_S)

                self._drain_phase += 1
        except Exception as exc:  # noqa: BLE001 - worker errors are surfaced in UI.
            with self._state_lock:
                self._last_process_error = f"analysis worker error: {exc}"

    def _process_latest(self, source_stats: SignalBatchSourceStats) -> None:
        history_rate_hz = self._history.sample_hz()
        analysis_input = AnalysisInput(
            batch=self._history.snapshot(),
            sample_rate_hz=history_rate_hz,
            source_stats=source_stats,
        )
        try:
            result = self._config.process(analysis_input, self._parameter_snapshot())
        except Exception as exc:  # noqa: BLE001 - analysis errors are surfaced in UI.
            result = None
            process_error: str | None = str(exc)
        else:
            process_error = None

        with self._state_lock:
            if process_error is None:
                self._last_result = result
            self._last_process_error = process_error
            self._source_stats = source_stats
            self._history_rate_hz = history_rate_hz

    def _update_source_status_if_due(self) -> None:
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_status_update_ns < _WORKER_STATUS_INTERVAL_NS:
            return
        self._last_status_update_ns = now_ns
        with self._state_lock:
            self._source_stats = self._config.source.source_stats()
            self._history_rate_hz = self._history.sample_hz()


def prefer_qt_xcb_platform() -> None:
    """Prefer Qt's XCB backend when Linux exposes a Wayland/X11 fallback chain."""
    if (
        sys.platform.startswith("linux")
        and "DISPLAY" in os.environ
        and os.environ.get("QT_QPA_PLATFORM") in (None, "", "wayland;xcb")
    ):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def import_pyqtgraph() -> tuple[Any, Any, Any]:
    """Import PyQtGraph lazily so module import does not require Qt startup."""
    prefer_qt_xcb_platform()

    try:
        import pyqtgraph as pg  # type: ignore[import-untyped]
        from pyqtgraph.Qt import QtCore  # type: ignore[import-untyped]
        from pyqtgraph.Qt import QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "PyQtGraph plotting requires pyqtgraph and a Qt binding. "
            "Install the project dependencies with uv sync."
        ) from exc

    return pg, QtCore, QtWidgets


def require_qt_platform_runtime() -> None:
    """Fail before QApplication aborts when XCB runtime libraries are missing."""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("QT_QPA_PLATFORM") != "xcb":
        return

    missing: list[str] = []
    for lib_name in QT_XCB_RUNTIME_LIBS:
        try:
            ctypes.CDLL(lib_name)
        except OSError:
            missing.append(lib_name)

    if missing:
        raise RuntimeError(
            "Qt's xcb platform plugin is missing runtime libraries: "
            f"{', '.join(missing)}. Run this from the Nix dev shell."
        )


def _decimate_series(series: SeriesData, max_points: int | None) -> SeriesData:
    if max_points is None or int(series.x.size) <= max_points:
        return series
    indexes = np.linspace(0, int(series.x.size) - 1, max_points, dtype=np.intp)
    return SeriesData(x=series.x[indexes], y=series.y[indexes])


def _format_rate(value: float | None) -> str:
    return "measuring" if value is None else f"{value:.1f} Hz"


class _VescPollRateDisplaySma:
    """Rolling SMA for the measured rate shown in the GUI when ``rate_label`` is VESC poll rate."""

    __slots__ = ("_buf",)

    def __init__(self, *, window: int = _VESC_POLL_RATE_DISPLAY_SMA_WINDOW) -> None:
        self._buf: deque[float] = deque(maxlen=max(1, window))

    def reset(self) -> None:
        self._buf.clear()

    def smooth(self, raw_hz: float, *, rate_label: str) -> float:
        if rate_label != "VESC poll rate":
            self._buf.clear()
            return raw_hz
        self._buf.append(raw_hz)
        return sum(self._buf) / len(self._buf)


def _configure_plot_interaction(plot_item: Any, plot_spec: PlotSpec) -> None:
    view_box = plot_item.getViewBox()
    mouse_mode = view_box.PanMode
    if plot_spec.mouse_mode == "rect":
        mouse_mode = view_box.RectMode
    view_box.setMouseMode(mouse_mode)
    plot_item.setMouseEnabled(
        x=plot_spec.allow_mouse_x,
        y=plot_spec.allow_mouse_y,
    )
    view_box.enableAutoRange(
        x=plot_spec.auto_range_x and plot_spec.x_axis_mode == "auto",
        y=plot_spec.auto_range_y,
    )


def _series_x_range(series: SeriesData) -> tuple[float, float] | None:
    if int(series.x.size) == 0:
        return None
    minimum = float(np.min(series.x))
    maximum = float(np.max(series.x))
    return minimum, maximum


def _follow_latest_x_range(
    x_min: float,
    x_max: float,
) -> tuple[float, float]:
    if x_max <= x_min:
        padding = 0.5 if x_min == 0.0 else abs(x_min) * 0.05
        if padding == 0.0:
            padding = 0.5
        return x_min - padding, x_max + padding
    return x_min, x_max


def _format_latest_values(
    stats: SignalBatchSourceStats,
    channels: Mapping[str, str],
) -> str:
    if not stats.latest_values:
        return "n/a"
    parts = []
    for channel_name, value in stats.latest_values.items():
        unit = channels.get(channel_name, "")
        parts.append(f"{channel_name}={value:.6g}{(' ' + unit) if unit else ''}")
    return ", ".join(parts)


def _qt_theme_stylesheet(theme: PlotTheme) -> str:
    return f"""
QWidget {{
    background-color: {theme.window_background};
    color: {theme.text_color};
}}
QLabel, QCheckBox, QToolButton {{
    color: {theme.text_color};
}}
QPushButton, QToolButton, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {theme.control_background};
    border: 1px solid {theme.control_border};
    border-radius: 3px;
    color: {theme.text_color};
    padding: 3px 6px;
}}
QPushButton:hover, QToolButton:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    background-color: {theme.control_hover};
}}
QPushButton:pressed, QToolButton:pressed {{
    background-color: {theme.control_pressed};
}}
QComboBox QAbstractItemView {{
    background-color: {theme.control_background};
    color: {theme.text_color};
    selection-background-color: {theme.control_hover};
}}
"""


def _format_latest_value_lines(
    stats: SignalBatchSourceStats,
    channels: Mapping[str, str],
) -> list[str]:
    if not stats.latest_values:
        return ["current: n/a"]
    lines: list[str] = []
    for channel_name, value in stats.latest_values.items():
        unit = channels.get(channel_name, "")
        suffix = f" {unit}" if unit else ""
        lines.append(f"current {channel_name}: {value:.6g}{suffix}")
    return lines


def _split_status_lines(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _signal_status_lines(
    worker_snapshot: _AnalysisWorkerSnapshot,
    channels: Mapping[str, str],
) -> list[str]:
    src_stats = worker_snapshot.source_stats
    lines = _format_latest_value_lines(src_stats, channels)
    if src_stats.last_error:
        lines.append(f"source error: {src_stats.last_error}")
    if worker_snapshot.process_error:
        lines.append(f"analysis error: {worker_snapshot.process_error}")
    elif worker_snapshot.result is not None:
        lines.extend(_split_status_lines(worker_snapshot.result.status_text))
    return lines


def _debug_status_lines(
    worker_snapshot: _AnalysisWorkerSnapshot,
    *,
    display_rate_hz: float | None = None,
) -> list[str]:
    src_stats = worker_snapshot.source_stats
    rate_hz = (
        src_stats.average_rate_hz
        if display_rate_hz is None
        else display_rate_hz
    )
    rate_label = src_stats.rate_label[:1].lower() + src_stats.rate_label[1:]
    lines = [
        f"{rate_label}: {_format_rate(rate_hz)}",
        f"history: {_format_rate(worker_snapshot.history_rate_hz)}",
        f"samples: {src_stats.samples}",
        f"dropped: {src_stats.dropped}",
        f"errors: {src_stats.errors}",
    ]
    if src_stats.last_error:
        lines.append(f"source error: {src_stats.last_error}")
    if worker_snapshot.process_error:
        lines.append(f"analysis error: {worker_snapshot.process_error}")
    return lines


def _debug_toggle_text(
    worker_snapshot: _AnalysisWorkerSnapshot | None,
    *,
    display_rate_hz: float | None = None,
) -> str:
    if worker_snapshot is None:
        return "Data acquisition rate: measuring"
    src_stats = worker_snapshot.source_stats
    rate_hz = (
        src_stats.average_rate_hz
        if display_rate_hz is None
        else display_rate_hz
    )
    source_rate = _format_rate(rate_hz)
    return f"{src_stats.rate_label}: {source_rate}"


def _read_live_analysis_control(slot: _PickleSharedMemorySlot) -> _LiveAnalysisControl:
    value = slot.read()
    if not isinstance(value, _LiveAnalysisControl):
        raise RuntimeError("shared-memory control slot contains an unexpected payload")
    return value


def _read_live_analysis_ui_config(
    slot: _PickleSharedMemorySlot,
) -> LiveAnalysisUiConfig:
    value = slot.read()
    if not isinstance(value, LiveAnalysisUiConfig):
        raise RuntimeError("shared-memory config slot contains an unexpected payload")
    return value


def _read_analysis_snapshot(
    slot: _PickleSharedMemorySlot,
) -> tuple[_AnalysisWorkerSnapshot, int]:
    value, version = slot.read_with_version()
    if not isinstance(value, _AnalysisWorkerSnapshot):
        raise RuntimeError("shared-memory state slot contains an unexpected payload")
    return value, version


def _update_live_analysis_control(
    slot: _PickleSharedMemorySlot,
    update: Callable[[_LiveAnalysisControl], _LiveAnalysisControl],
) -> None:
    slot.write(update(_read_live_analysis_control(slot)))


def _snapshot_without_result(
    snapshot: _AnalysisWorkerSnapshot,
    error: str,
) -> _AnalysisWorkerSnapshot:
    process_error = error
    if snapshot.process_error:
        process_error = f"{snapshot.process_error}; {error}"
    return _AnalysisWorkerSnapshot(
        result=None,
        process_error=process_error,
        source_stats=snapshot.source_stats,
        history_rate_hz=snapshot.history_rate_hz,
    )


def _write_analysis_snapshot(
    slot: _PickleSharedMemorySlot,
    snapshot: _AnalysisWorkerSnapshot,
) -> None:
    try:
        slot.write(snapshot)
    except ValueError as exc:
        slot.write(_snapshot_without_result(snapshot, str(exc)))


def _analysis_process_error_snapshot(error: str) -> _AnalysisWorkerSnapshot:
    snapshot = _initial_analysis_snapshot()
    return _AnalysisWorkerSnapshot(
        result=None,
        process_error=error,
        source_stats=snapshot.source_stats,
        history_rate_hz=snapshot.history_rate_hz,
    )


def _run_analysis_worker_process(
    config: LiveAnalysisApp,
    memory_spec: _SharedAnalysisMemorySpec,
) -> None:
    """Run source acquisition and analysis in the script child process."""
    memory = _SharedAnalysisMemory.attach(memory_spec)
    worker: _LiveAnalysisWorker | None = None
    try:
        control = _read_live_analysis_control(memory.control)
        worker = _LiveAnalysisWorker(config, control.parameter_values)
        last_parameter_revision = control.parameter_revision
        last_clear_revision = control.clear_revision
        publish_interval_ns = max(1, round(NSEC_PER_SEC / config.plot_rate_hz))
        next_publish_ns = time.perf_counter_ns()

        worker.start()
        while True:
            control = _read_live_analysis_control(memory.control)
            if control.stop_requested or control.shutdown_requested:
                break

            if control.parameter_revision != last_parameter_revision:
                for name, value in control.parameter_values.items():
                    worker.set_parameter(name, value)
                last_parameter_revision = control.parameter_revision

            if control.clear_revision != last_clear_revision:
                worker.clear_history()
                last_clear_revision = control.clear_revision

            now_ns = time.perf_counter_ns()
            if now_ns >= next_publish_ns:
                _write_analysis_snapshot(memory.state, worker.snapshot())
                next_publish_ns = now_ns + publish_interval_ns

            sleep_ns = max(0, next_publish_ns - time.perf_counter_ns())
            time.sleep(min(_ANALYSIS_IDLE_SLEEP_S, sleep_ns / NSEC_PER_SEC))
    except Exception as exc:  # noqa: BLE001 - supervisor restarts this process.
        try:
            memory.state.write(_analysis_process_error_snapshot(str(exc)))
        finally:
            raise
    finally:
        if worker is not None:
            worker.stop()
        memory.close()


def _run_live_analysis_gui_process(memory_spec: _SharedAnalysisMemorySpec) -> None:
    """Run the Qt/PyQtGraph child process."""
    memory = _SharedAnalysisMemory.attach(memory_spec)
    try:
        _run_live_analysis_gui(_read_live_analysis_ui_config(memory.config), memory)
    finally:
        memory.close()


def _run_live_analysis_gui(
    config: LiveAnalysisUiConfig,
    memory: _SharedAnalysisMemory,
) -> None:
    """Run the generic live-analysis GUI from shared-memory state."""
    selected_theme = PLOT_THEMES[config.theme]
    pg, QtCore, QtWidgets = import_pyqtgraph()
    require_qt_platform_runtime()
    pg.setConfigOptions(
        antialias=config.antialias,
        background=selected_theme.pg_background,
        foreground=selected_theme.pg_foreground,
    )

    qt_app = pg.mkQApp(config.title)
    window = QtWidgets.QWidget()
    window.setWindowTitle(config.title)
    window.resize(1440, 900)
    window.setStyleSheet(_qt_theme_stylesheet(selected_theme))

    qt_alignment = getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt)
    qt_arrow = getattr(QtCore.Qt, "ArrowType", QtCore.Qt)
    qt_size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
    qt_mouse_button = getattr(QtCore.Qt, "MouseButton", QtCore.Qt)

    class _PlotViewBox(pg.ViewBox):
        def __init__(self, *, allow_left_drag: bool) -> None:
            super().__init__()
            self._allow_left_drag = allow_left_drag

        def mouseDragEvent(self, ev: Any, axis: int | None = None) -> None:
            if (
                not self._allow_left_drag
                and axis is None
                and ev.button()
                in (
                    qt_mouse_button.LeftButton,
                    qt_mouse_button.MiddleButton,
                )
            ):
                ev.ignore()
                return
            super().mouseDragEvent(ev, axis=axis)

    root = QtWidgets.QVBoxLayout(window)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(8)

    controls = QtWidgets.QHBoxLayout()
    controls.setContentsMargins(0, 0, 0, 0)
    controls.setSpacing(8)
    root.addLayout(controls)

    source_text = QtWidgets.QLabel(config.source_label or config.title)
    source_text.setStyleSheet(f"color: {selected_theme.text_color}; font-weight: 700;")
    controls.addWidget(source_text)

    channel_summary = ", ".join(
        f"{name} ({unit})" if unit else name for name, unit in config.channels.items()
    )
    channel_text = QtWidgets.QLabel(channel_summary)
    channel_text.setStyleSheet(f"color: {selected_theme.muted_color};")
    controls.addWidget(channel_text)

    controls.addSpacing(12)
    try:
        parameter_values = dict(
            _read_live_analysis_control(memory.control).parameter_values
        )
    except RuntimeError:
        parameter_values = default_parameter_values(config.parameters)
    parameter_widgets: dict[str, Any] = {}

    def set_parameter(name: str, value: ParamValue) -> None:
        def update(control: _LiveAnalysisControl) -> _LiveAnalysisControl:
            values = dict(control.parameter_values)
            values[name] = value
            return _LiveAnalysisControl(
                parameter_values=values,
                parameter_revision=control.parameter_revision + 1,
                clear_revision=control.clear_revision,
                stop_requested=control.stop_requested,
                shutdown_requested=control.shutdown_requested,
            )

        _update_live_analysis_control(memory.control, update)

    for parameter in config.parameters:
        controls.addWidget(QtWidgets.QLabel(parameter.label))
        widget = _build_parameter_widget(
            parameter,
            parameter_values,
            QtWidgets=QtWidgets,
            on_change=set_parameter,
        )
        parameter_widgets[parameter.name] = widget
        controls.addWidget(widget)

    clear_button = QtWidgets.QPushButton("Clear")
    controls.addWidget(clear_button)
    controls.addStretch(1)

    signal_status = QtWidgets.QLabel("Waiting for signal data...")
    signal_status.setAlignment(qt_alignment.AlignLeft)
    signal_status.setWordWrap(True)
    signal_status.setStyleSheet(f"color: {selected_theme.muted_color};")
    root.addWidget(signal_status)

    plot_grid = QtWidgets.QGridLayout()
    plot_grid.setContentsMargins(0, 0, 0, 0)
    plot_grid.setHorizontalSpacing(8)
    plot_grid.setVerticalSpacing(8)
    root.addLayout(plot_grid, stretch=1)

    debug_header = QtWidgets.QHBoxLayout()
    debug_header.setContentsMargins(0, 0, 0, 0)
    debug_header.setSpacing(6)
    root.addLayout(debug_header)

    debug_toggle = QtWidgets.QToolButton()
    debug_toggle.setCheckable(True)
    debug_toggle.setChecked(False)
    debug_toggle.setArrowType(qt_arrow.RightArrow)
    debug_toggle.setText("Runtime")
    debug_header.addWidget(debug_toggle)

    debug_summary = QtWidgets.QLabel(_debug_toggle_text(None))
    debug_summary.setAlignment(qt_alignment.AlignLeft)
    debug_summary.setStyleSheet(f"color: {selected_theme.muted_color};")
    debug_header.addWidget(debug_summary, stretch=1)

    debug_status = QtWidgets.QLabel("")
    debug_status.setAlignment(qt_alignment.AlignLeft)
    debug_status.setWordWrap(True)
    debug_status.setVisible(False)
    debug_status.setStyleSheet(f"color: {selected_theme.muted_color};")
    root.addWidget(debug_status)

    plot_items: list[Any] = []
    plot_curves: list[dict[str, Any]] = []
    plot_has_seen_data = [False] * len(config.plots)
    color_cycle = cycle(selected_theme.line_colors)
    for plot_index, plot_spec in enumerate(config.plots):
        widget = pg.PlotWidget(
            title=plot_spec.title,
            viewBox=_PlotViewBox(allow_left_drag=plot_spec.allow_left_drag),
        )
        widget.setMinimumSize(0, 0)
        widget.setSizePolicy(qt_size_policy.Ignored, qt_size_policy.Ignored)
        plot_grid.addWidget(widget, plot_index // 2, plot_index % 2)
        plot_item = widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=selected_theme.grid_alpha)
        plot_item.setLabel("bottom", plot_spec.x_label, units=plot_spec.x_unit)
        plot_item.setLabel("left", plot_spec.y_label, units=plot_spec.y_unit)
        plot_item.setLogMode(x=plot_spec.log_x, y=plot_spec.log_y)
        plot_item.addLegend(offset=(10, 10))
        _configure_plot_interaction(plot_item, plot_spec)

        curves: dict[str, Any] = {}
        for trace in plot_spec.traces:
            color = trace.color
            if color is None:
                color = next(color_cycle)
            curve = pg.PlotCurveItem(
                _empty_array(),
                _empty_array(),
                pen=pg.mkPen(color, width=trace.width),
                name=trace.label,
                connect="all",
                skipFiniteCheck=True,
            )
            curve.setSkipFiniteCheck(True)
            plot_item.addItem(curve)
            curves[trace.series] = curve

        plot_items.append(plot_item)
        plot_curves.append(curves)

    poll_rate_display_sma = _VescPollRateDisplaySma()

    rendered_state_version: int | None = None

    def refresh() -> None:
        nonlocal rendered_state_version

        try:
            worker_snapshot, state_version = _read_analysis_snapshot(memory.state)
        except RuntimeError as exc:
            signal_status.setText(f"shared-memory error: {exc}")
            return

        if state_version != rendered_state_version:
            rendered_state_version = state_version
            for plot_index, (plot_spec, plot_item, curves) in enumerate(
                zip(
                    config.plots,
                    plot_items,
                    plot_curves,
                    strict=True,
                )
            ):
                plot_has_data = False
                plot_x_min: float | None = None
                plot_x_max: float | None = None
                for trace in plot_spec.traces:
                    series = (
                        None
                        if worker_snapshot.result is None
                        else worker_snapshot.result.series.get(trace.series)
                    )
                    curve = curves[trace.series]
                    if series is None:
                        curve.setData(_empty_array(), _empty_array())
                        continue
                    decimated = _decimate_series(series, plot_spec.max_points)
                    curve.setData(decimated.x, decimated.y)
                    if int(decimated.x.size) > 0 and int(decimated.y.size) > 0:
                        plot_has_data = True
                    x_range = _series_x_range(series)
                    if x_range is not None:
                        x_min, x_max = x_range
                        plot_x_min = x_min if plot_x_min is None else min(plot_x_min, x_min)
                        plot_x_max = x_max if plot_x_max is None else max(plot_x_max, x_max)
                if plot_has_data and not plot_has_seen_data[plot_index]:
                    plot_has_seen_data[plot_index] = True
                    if not plot_spec.auto_range_x or not plot_spec.auto_range_y:
                        plot_item.autoRange()
                if (
                    plot_spec.x_axis_mode == "follow_latest"
                    and plot_x_min is not None
                    and plot_x_max is not None
                ):
                    x_min, x_max = _follow_latest_x_range(plot_x_min, plot_x_max)
                    plot_item.setXRange(x_min, x_max, padding=0.0)

        signal_status.setText(
            "\n".join(_signal_status_lines(worker_snapshot, config.channels))
        )
        src_stats = worker_snapshot.source_stats
        display_rate_hz = poll_rate_display_sma.smooth(
            src_stats.average_rate_hz,
            rate_label=src_stats.rate_label,
        )
        debug_summary.setText(
            _debug_toggle_text(worker_snapshot, display_rate_hz=display_rate_hz)
        )
        debug_status.setText(
            "\n".join(
                _debug_status_lines(worker_snapshot, display_rate_hz=display_rate_hz)
            )
        )

    def clear_history() -> None:
        nonlocal rendered_state_version
        rendered_state_version = None
        poll_rate_display_sma.reset()

        def update(control: _LiveAnalysisControl) -> _LiveAnalysisControl:
            return _LiveAnalysisControl(
                parameter_values=control.parameter_values,
                parameter_revision=control.parameter_revision,
                clear_revision=control.clear_revision + 1,
                stop_requested=control.stop_requested,
                shutdown_requested=control.shutdown_requested,
            )

        _update_live_analysis_control(memory.control, update)
        refresh()

    def request_shutdown() -> None:
        def update(control: _LiveAnalysisControl) -> _LiveAnalysisControl:
            return _LiveAnalysisControl(
                parameter_values=control.parameter_values,
                parameter_revision=control.parameter_revision,
                clear_revision=control.clear_revision,
                stop_requested=True,
                shutdown_requested=True,
            )

        try:
            _update_live_analysis_control(memory.control, update)
        except RuntimeError:
            return

    clear_button.clicked.connect(clear_history)

    def set_debug_visible(visible: bool) -> None:
        debug_status.setVisible(visible)
        debug_toggle.setArrowType(
            qt_arrow.DownArrow if visible else qt_arrow.RightArrow
        )

    debug_toggle.toggled.connect(set_debug_visible)

    timer = QtCore.QTimer(window)
    interval_ms = max(1, round(1000.0 / config.plot_rate_hz))
    timer.timeout.connect(refresh)

    qt_app.aboutToQuit.connect(request_shutdown)

    timer.start(interval_ms)
    refresh()
    window.show()

    try:
        exec_fn = getattr(pg, "exec", None)
        if exec_fn is not None:
            exec_fn()
        else:
            qt_app.exec()
    finally:
        timer.stop()
        request_shutdown()


def _multiprocessing_context() -> BaseContext:
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("YALSA live analysis multiprocessing requires fork support")
    return mp.get_context("fork")


def _start_analysis_worker_process(
    context: BaseContext,
    config: LiveAnalysisApp,
    memory_spec: _SharedAnalysisMemorySpec,
) -> mp.Process:
    process = cast(
        mp.Process,
        cast(Any, context).Process(
            target=_run_analysis_worker_process,
            args=(config, memory_spec),
            name="yalsa-script-process",
        ),
    )
    process.start()
    return process


def _start_live_analysis_gui_process(
    context: BaseContext,
    memory_spec: _SharedAnalysisMemorySpec,
) -> mp.Process:
    process = cast(
        mp.Process,
        cast(Any, context).Process(
            target=_run_live_analysis_gui_process,
            args=(memory_spec,),
            name="yalsa-gui-process",
        ),
    )
    process.start()
    return process


def _request_shared_shutdown(memory: _SharedAnalysisMemory) -> None:
    def update(control: _LiveAnalysisControl) -> _LiveAnalysisControl:
        return _LiveAnalysisControl(
            parameter_values=control.parameter_values,
            parameter_revision=control.parameter_revision,
            clear_revision=control.clear_revision,
            stop_requested=True,
            shutdown_requested=True,
        )

    try:
        _update_live_analysis_control(memory.control, update)
    except RuntimeError:
        return


def _stop_child_process(process: mp.Process) -> None:
    if process.is_alive():
        process.join(timeout=_PROCESS_STOP_TIMEOUT_S)
    if process.is_alive():
        process.terminate()
        process.join(timeout=_PROCESS_STOP_TIMEOUT_S)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=_PROCESS_STOP_TIMEOUT_S)
    process.join(timeout=0.0)


def run_live_analysis(config: LiveAnalysisApp) -> None:
    """Run YALSA as a supervised script process plus a Qt GUI process."""
    context = _multiprocessing_context()
    memory = _SharedAnalysisMemory.create(live_analysis_ui_config(config))
    memory_spec = memory.spec
    worker_process = _start_analysis_worker_process(context, config, memory_spec)
    gui_process = _start_live_analysis_gui_process(context, memory_spec)
    try:
        while True:
            time.sleep(_SUPERVISOR_POLL_S)
            control = _read_live_analysis_control(memory.control)
            if control.shutdown_requested or control.stop_requested:
                break

            if gui_process.exitcode is not None:
                if gui_process.exitcode == 0:
                    _request_shared_shutdown(memory)
                    break
                gui_process.join(timeout=0.0)
                gui_process = _start_live_analysis_gui_process(context, memory_spec)

            if worker_process.exitcode is not None:
                worker_process.join(timeout=0.0)
                worker_process = _start_analysis_worker_process(
                    context,
                    config,
                    memory_spec,
                )
    except KeyboardInterrupt:
        _request_shared_shutdown(memory)
    finally:
        _request_shared_shutdown(memory)
        _stop_child_process(gui_process)
        _stop_child_process(worker_process)
        memory.close()
        memory.unlink()


def _build_parameter_widget(
    parameter: ParameterSpec,
    values: dict[str, ParamValue],
    *,
    QtWidgets: Any,
    on_change: Callable[[str, ParamValue], None] | None = None,
) -> Any:
    if parameter.kind == "int":
        widget = QtWidgets.QSpinBox()
        value = values.get(parameter.name, parameter.default)
        widget.setRange(
            -2_147_483_648 if parameter.minimum is None else int(parameter.minimum),
            2_147_483_647 if parameter.maximum is None else int(parameter.maximum),
        )
        widget.setSingleStep(1 if parameter.step is None else int(parameter.step))
        widget.setValue(int(value))
        widget.valueChanged.connect(
            lambda value, name=parameter.name: _set_parameter_value(
                values,
                on_change,
                name,
                int(value),
            )
        )
        return widget

    if parameter.kind == "float":
        widget = QtWidgets.QDoubleSpinBox()
        value = values.get(parameter.name, parameter.default)
        widget.setRange(
            -1e12 if parameter.minimum is None else float(parameter.minimum),
            1e12 if parameter.maximum is None else float(parameter.maximum),
        )
        widget.setDecimals(parameter.decimals if parameter.decimals is not None else 3)
        widget.setSingleStep(0.1 if parameter.step is None else float(parameter.step))
        widget.setValue(float(value))
        widget.valueChanged.connect(
            lambda value, name=parameter.name: _set_parameter_value(
                values,
                on_change,
                name,
                float(value),
            )
        )
        return widget

    if parameter.kind == "bool":
        widget = QtWidgets.QCheckBox()
        value = values.get(parameter.name, parameter.default)
        widget.setChecked(bool(value))
        widget.toggled.connect(
            lambda checked, name=parameter.name: _set_parameter_value(
                values,
                on_change,
                name,
                bool(checked),
            )
        )
        return widget

    widget = QtWidgets.QComboBox()
    for choice in parameter.choices:
        widget.addItem(choice.label, choice.value)
    value = values.get(parameter.name, parameter.default)
    current_index = widget.findData(value)
    widget.setCurrentIndex(max(0, current_index))
    widget.currentIndexChanged.connect(
        lambda _index, name=parameter.name, combo=widget: _set_parameter_value(
            values,
            on_change,
            name,
            cast(str, combo.currentData()),
        )
    )
    return widget


def _set_parameter_value(
    values: dict[str, ParamValue],
    on_change: Callable[[str, ParamValue], None] | None,
    name: str,
    value: ParamValue,
) -> None:
    values[name] = value
    if on_change is not None:
        on_change(name, value)


__all__ = [
    "AnalysisInput",
    "AnalysisResult",
    "ChoiceOption",
    "FloatArray",
    "LiveAnalysisApp",
    "LiveAnalysisUiConfig",
    "ParamValue",
    "ParameterSpec",
    "PlotSpec",
    "PlotTrace",
    "ProcessCallback",
    "ScalarSignalSourceAdapter",
    "SeriesData",
    "SignalBatch",
    "SignalBatchHistory",
    "SignalBatchSource",
    "SignalBatchSourceStats",
    "bool_parameter",
    "choice_parameter",
    "default_parameter_values",
    "empty_signal_batch",
    "float_parameter",
    "import_pyqtgraph",
    "int_parameter",
    "live_analysis_ui_config",
    "merge_signal_batch_source_stats",
    "pending_batch_ring_stats",
    "prefer_qt_xcb_platform",
    "require_qt_platform_runtime",
    "run_live_analysis",
    "xy_series",
]
