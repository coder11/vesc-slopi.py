"""YALSA application runtime with pluggable live signal sources."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from vesc_py.live_signal import SignalSource, SignalSourceSnapshot

FloatArray = npt.NDArray[np.float64]
ParamValue: TypeAlias = int | float | bool | str
PlotColor: TypeAlias = str | tuple[int, int, int]

QT_XCB_RUNTIME_LIBS = ("libxcb-cursor.so.0", "libxcb-icccm.so.4")


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
class SignalBatchSourceSnapshot:
    """Low-rate status data exposed by a batch signal source."""

    samples: int
    dropped: int
    errors: int
    average_rate_hz: float
    latest_sample_s: float | None
    latest_values: Mapping[str, float]
    last_error: str | None
    done: bool


class SignalBatchSource(Protocol):
    """Common interface for live signal sources exposed to the analysis app."""

    @property
    def channels(self) -> Mapping[str, str]: ...

    def start(self) -> None: ...

    def stop(self, timeout: float = 1.0) -> None: ...

    def drain(self) -> tuple[SignalBatch, int]: ...

    def snapshot(self) -> SignalBatchSourceSnapshot: ...


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

    def drain(self) -> tuple[SignalBatch, int]:
        timestamps, values, dropped = self._source.drain()
        batch = SignalBatch(
            timestamps_s=timestamps,
            values={self._source.channel_name: values},
            units=self._channels,
        )
        return batch, dropped

    def snapshot(self) -> SignalBatchSourceSnapshot:
        snapshot = self._source.snapshot()
        latest_values: dict[str, float] = {}
        if snapshot.latest_value is not None:
            latest_values[self._source.channel_name] = snapshot.latest_value
        return _scalar_snapshot_to_batch(snapshot, latest_values)


def _scalar_snapshot_to_batch(
    snapshot: SignalSourceSnapshot,
    latest_values: Mapping[str, float],
) -> SignalBatchSourceSnapshot:
    return SignalBatchSourceSnapshot(
        samples=snapshot.samples,
        dropped=snapshot.dropped,
        errors=snapshot.errors,
        average_rate_hz=snapshot.average_rate_hz,
        latest_sample_s=snapshot.latest_sample_s,
        latest_values=dict(latest_values),
        last_error=snapshot.last_error,
        done=snapshot.done,
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
                channel_values = batch.values[channel_name].astype(np.float64, copy=False)
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
            if self.minimum is not None or self.maximum is not None or self.step is not None:
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
            if self.minimum is not None or self.maximum is not None or self.step is not None:
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

    def __post_init__(self) -> None:
        if not self.traces:
            raise ValueError("plots must contain at least one trace")
        if self.max_points is not None and self.max_points <= 0:
            raise ValueError("max_points must be greater than 0")


@dataclass(frozen=True, slots=True)
class AnalysisInput:
    """Current retained source data passed into the processing callback."""

    batch: SignalBatch
    sample_rate_hz: float | None
    snapshot: SignalBatchSourceSnapshot

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


ProcessCallback: TypeAlias = Callable[[AnalysisInput, Mapping[str, ParamValue]], AnalysisResult]


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
    source_label: str | None = None
    theme: Literal["light", "dark"] = "dark"
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
        default_parameter_values(self.parameters)


@dataclass(frozen=True, slots=True)
class PlotTheme:
    """Simple color bundle for the PyQtGraph UI."""

    pg_background: str
    pg_foreground: str
    window_background: str
    text_color: str
    muted_color: str
    grid_alpha: float
    line_colors: tuple[tuple[int, int, int], ...]


PLOT_THEMES = {
    "light": PlotTheme(
        pg_background="#ffffff",
        pg_foreground="#202124",
        window_background="#f6f7f9",
        text_color="#202124",
        muted_color="#4f5b66",
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


def _format_latest_values(
    snapshot: SignalBatchSourceSnapshot,
    channels: Mapping[str, str],
) -> str:
    if not snapshot.latest_values:
        return "n/a"
    parts = []
    for channel_name, value in snapshot.latest_values.items():
        unit = channels.get(channel_name, "")
        parts.append(f"{channel_name}={value:.6g}{(' ' + unit) if unit else ''}")
    return ", ".join(parts)


def run_live_analysis(config: LiveAnalysisApp) -> None:
    """Run the generic live-analysis GUI until the Qt app exits."""
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
    window.setStyleSheet(f"background-color: {selected_theme.window_background};")

    qt_alignment = getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt)
    qt_size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)

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
        f"{name} ({unit})" if unit else name for name, unit in config.source.channels.items()
    )
    channel_text = QtWidgets.QLabel(channel_summary)
    channel_text.setStyleSheet(f"color: {selected_theme.muted_color};")
    controls.addWidget(channel_text)

    controls.addSpacing(12)
    parameter_values = default_parameter_values(config.parameters)
    parameter_widgets: dict[str, Any] = {}
    for parameter in config.parameters:
        controls.addWidget(QtWidgets.QLabel(parameter.label))
        widget = _build_parameter_widget(
            parameter,
            parameter_values,
            QtWidgets=QtWidgets,
        )
        parameter_widgets[parameter.name] = widget
        controls.addWidget(widget)

    clear_button = QtWidgets.QPushButton("Clear")
    controls.addWidget(clear_button)
    controls.addStretch(1)

    status = QtWidgets.QLabel("Waiting for signal data...")
    status.setAlignment(qt_alignment.AlignLeft)
    status.setWordWrap(True)
    status.setStyleSheet(f"color: {selected_theme.muted_color};")
    root.addWidget(status)

    plot_grid = QtWidgets.QGridLayout()
    plot_grid.setContentsMargins(0, 0, 0, 0)
    plot_grid.setHorizontalSpacing(8)
    plot_grid.setVerticalSpacing(8)
    root.addLayout(plot_grid, stretch=1)

    plot_items: list[Any] = []
    plot_curves: list[dict[str, Any]] = []
    color_cycle = cycle(selected_theme.line_colors)
    for plot_index, plot_spec in enumerate(config.plots):
        widget = pg.PlotWidget(title=plot_spec.title)
        widget.setMinimumSize(0, 0)
        widget.setSizePolicy(qt_size_policy.Ignored, qt_size_policy.Ignored)
        plot_grid.addWidget(widget, plot_index // 2, plot_index % 2)
        plot_item = widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=selected_theme.grid_alpha)
        plot_item.setLabel("bottom", plot_spec.x_label, units=plot_spec.x_unit)
        plot_item.setLabel("left", plot_spec.y_label, units=plot_spec.y_unit)
        plot_item.setLogMode(x=plot_spec.log_x, y=plot_spec.log_y)
        plot_item.addLegend(offset=(10, 10))
        plot_item.setMouseEnabled(x=True, y=True)

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

    history = SignalBatchHistory(config.history, config.source.channels)
    last_result: AnalysisResult | None = None
    last_process_error: str | None = None
    source_started = False

    def refresh() -> None:
        nonlocal last_result, last_process_error
        batch, _dropped = config.source.drain()
        history.append_batch(batch)
        analysis_input = AnalysisInput(
            batch=history.snapshot(),
            sample_rate_hz=history.sample_hz(),
            snapshot=config.source.snapshot(),
        )
        try:
            result = config.process(analysis_input, parameter_values)
        except Exception as exc:  # noqa: BLE001 - analysis errors are surfaced in UI.
            last_process_error = str(exc)
        else:
            last_result = result
            last_process_error = None

        for plot_spec, plot_item, curves in zip(
            config.plots,
            plot_items,
            plot_curves,
            strict=True,
        ):
            for trace in plot_spec.traces:
                series = None if last_result is None else last_result.series.get(trace.series)
                curve = curves[trace.series]
                if series is None:
                    curve.setData(_empty_array(), _empty_array())
                    continue
                decimated = _decimate_series(series, plot_spec.max_points)
                curve.setData(decimated.x, decimated.y)
            plot_item.autoRange()

        snapshot = analysis_input.snapshot
        parts = [
            f"source: {_format_rate(snapshot.average_rate_hz)}",
            f"history: {_format_rate(analysis_input.sample_rate_hz)}",
            f"samples: {snapshot.samples}",
            f"dropped: {snapshot.dropped}",
            f"errors: {snapshot.errors}",
            f"latest: {_format_latest_values(snapshot, config.source.channels)}",
        ]
        if snapshot.last_error:
            parts.append(f"source error: {snapshot.last_error}")
        if last_process_error:
            parts.append(f"analysis error: {last_process_error}")
        elif last_result is not None and last_result.status_text:
            parts.append(last_result.status_text)
        status.setText(" | ".join(parts))

    def clear_history() -> None:
        nonlocal last_result, last_process_error
        history.clear()
        last_result = None
        last_process_error = None
        refresh()

    def stop_source() -> None:
        nonlocal source_started
        if source_started:
            config.source.stop()
            source_started = False

    clear_button.clicked.connect(clear_history)

    timer = QtCore.QTimer(window)
    interval_ms = max(1, round(1000.0 / config.plot_rate_hz))
    timer.timeout.connect(refresh)

    qt_app.aboutToQuit.connect(stop_source)

    config.source.start()
    source_started = True
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
        stop_source()


def _build_parameter_widget(
    parameter: ParameterSpec,
    values: dict[str, ParamValue],
    *,
    QtWidgets: Any,
) -> Any:
    if parameter.kind == "int":
        widget = QtWidgets.QSpinBox()
        widget.setRange(
            -2_147_483_648 if parameter.minimum is None else int(parameter.minimum),
            2_147_483_647 if parameter.maximum is None else int(parameter.maximum),
        )
        widget.setSingleStep(1 if parameter.step is None else int(parameter.step))
        widget.setValue(int(parameter.default))
        widget.valueChanged.connect(
            lambda value, name=parameter.name: values.__setitem__(name, int(value))
        )
        return widget

    if parameter.kind == "float":
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(
            -1e12 if parameter.minimum is None else float(parameter.minimum),
            1e12 if parameter.maximum is None else float(parameter.maximum),
        )
        widget.setDecimals(parameter.decimals if parameter.decimals is not None else 3)
        widget.setSingleStep(0.1 if parameter.step is None else float(parameter.step))
        widget.setValue(float(parameter.default))
        widget.valueChanged.connect(
            lambda value, name=parameter.name: values.__setitem__(name, float(value))
        )
        return widget

    if parameter.kind == "bool":
        widget = QtWidgets.QCheckBox()
        widget.setChecked(bool(parameter.default))
        widget.toggled.connect(
            lambda checked, name=parameter.name: values.__setitem__(name, bool(checked))
        )
        return widget

    widget = QtWidgets.QComboBox()
    for choice in parameter.choices:
        widget.addItem(choice.label, choice.value)
    current_index = widget.findData(parameter.default)
    widget.setCurrentIndex(max(0, current_index))
    widget.currentIndexChanged.connect(
        lambda _index, name=parameter.name, combo=widget: values.__setitem__(
            name,
            cast(str, combo.currentData()),
        )
    )
    return widget


__all__ = [
    "AnalysisInput",
    "AnalysisResult",
    "ChoiceOption",
    "FloatArray",
    "LiveAnalysisApp",
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
    "SignalBatchSourceSnapshot",
    "bool_parameter",
    "choice_parameter",
    "default_parameter_values",
    "empty_signal_batch",
    "float_parameter",
    "import_pyqtgraph",
    "int_parameter",
    "prefer_qt_xcb_platform",
    "require_qt_platform_runtime",
    "run_live_analysis",
    "xy_series",
]
