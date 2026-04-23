# Live Signal Bench Architecture TODO

## Goal

Move the current one-axis IMU filter bench toward a general live signal-processing
test bench:

1. Capture one or more live signals from interchangeable sources.
2. Apply arbitrary Python transformations that can be changed while the app runs.
3. Plot one or more derived signals, including multiple signals in one plot.
4. Keep source capture, transform execution, buffering, and plotting independent.

The current starting point is:

- `src/vesc_py/live_signal.py`: generic single-signal source protocol, history,
  deterministic source, SMA/residual helpers.
- `src/vesc_py/fast_imu_source.py`: one-axis direct VESC IMU source.
- `examples/imu_signal_bench.py`: PyQtGraph one-axis SMA tuning UI.

`examples/poll_imu_fast.py` should remain a separate high-rate reference/tool.

## Target Shape

The long-term pipeline should look like this:

```text
Source(s) -> SampleBus/History -> TransformRuntime -> PlotModel -> PlotView
```

Each boundary should be explicit:

- **Source**: produces timestamped raw channels.
- **SampleBus/History**: stores recent samples, handles overwrite/drop behavior,
  exposes snapshots to transforms without blocking capture.
- **TransformRuntime**: loads user Python code, executes it against snapshots, and
  returns named output series.
- **PlotModel**: maps named series into plots, modes, colors, units, and viewport
  policies.
- **PlotView**: only renders model state and dispatches UI changes.

## Step 1: Generalize From Scalar Signal To Channel Frames

Current `SignalSource` emits one scalar channel. The next architectural step is to
support frames with multiple named channels while keeping the one-axis path easy.

Add:

```python
@dataclass(frozen=True, slots=True)
class SignalFrame:
    timestamp_s: float
    values: Mapping[str, float]
```

or, for batch efficiency:

```python
@dataclass(frozen=True, slots=True)
class SignalBatch:
    timestamps_s: np.ndarray
    values: Mapping[str, np.ndarray]
    units: Mapping[str, str]
```

Update source interface:

```python
class SignalSource(Protocol):
    @property
    def channels(self) -> Mapping[str, str]: ...
    def start(self) -> None: ...
    def stop(self, timeout: float = 1.0) -> None: ...
    def drain(self) -> tuple[SignalBatch, int]: ...
    def snapshot(self) -> SignalSourceSnapshot: ...
```

Acceptance criteria:

- Existing one-axis bench still works through an adapter or by selecting one
  channel from a multi-channel batch.
- Deterministic source can emit at least two channels, for example `raw` and
  `reference`.
- VESC IMU source can request and emit all nine bench IMU fields in one packet
  when needed.
- Tests cover mismatched channel lengths and dropped pending frames.

## Step 2: Introduce A Transform Protocol

Move SMA/residual out of UI callbacks and behind a transform interface.

Add:

```python
@dataclass(frozen=True, slots=True)
class TransformInput:
    timestamps_s: np.ndarray
    channels: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    params: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class TransformOutput:
    channels: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    metadata: Mapping[str, object]

class SignalTransform(Protocol):
    name: str
    def apply(self, data: TransformInput) -> TransformOutput: ...
```

Implement built-in transforms:

- `IdentityTransform`: returns selected raw channels.
- `SmaTransform`: returns `filtered` and `residual` for one selected channel.
- Later: `BiquadTransform`.

Acceptance criteria:

- The UI no longer calls `trailing_sma()` directly.
- Changing the SMA window updates transform params and recomputes from history.
- Transform code is pure and testable without Qt.
- Plot modes consume named transform outputs, not hardcoded local variables.

## Step 3: Add A Plot Model

Separate "what should be plotted" from PyQtGraph widgets.

Add:

```python
@dataclass(frozen=True, slots=True)
class PlotSeries:
    name: str
    channel: str
    unit: str
    color_role: str
    visible: bool = True

@dataclass(frozen=True, slots=True)
class PlotSpec:
    title: str
    x_mode: Literal["history_samples", "time_s"]
    history_samples: int
    series: tuple[PlotSeries, ...]
```

Initial plot specs:

- Raw only.
- Filtered only.
- Raw + filtered.
- Residual.

Acceptance criteria:

- Display modes are implemented by swapping/updating `PlotSpec`.
- Multiple signals inside one plot are represented as multiple `PlotSeries`.
- The fixed history-sample viewport policy lives in the plot model, not ad hoc in
  the redraw callback.

## Step 4: Add Runtime Transform Loading

Add an optional watched Python transform file.

Proposed user transform API:

```python
def transform(data, params):
    raw = data.channels["acc_z"]
    filtered = my_filter(raw, params["cutoff_hz"])
    return {
        "channels": {
            "raw": raw,
            "filtered": filtered,
            "residual": raw - filtered,
        },
        "units": {
            "raw": data.units["acc_z"],
            "filtered": data.units["acc_z"],
            "residual": data.units["acc_z"],
        },
    }
```

Implementation notes:

- Watch file modification time on a timer.
- Load with `importlib.util.spec_from_file_location` under a unique module name.
- Keep the last good transform active if reload fails.
- Surface syntax/runtime errors in the UI status row.
- Do not execute transform code in the capture thread.

Acceptance criteria:

- Editing and saving the transform file changes plotted output without restarting
  the app.
- Syntax errors do not crash the app.
- Runtime errors are shown and the last successful output remains visible.
- A deterministic-source test can exercise reload behavior.

## Step 5: Add Parameter Controls For Transforms

The current SMA spinbox is hardcoded. Replace it with transform-declared params.

Proposed declaration:

```python
PARAMS = {
    "window": {"type": "int", "default": 10, "min": 1, "max": 100000},
    "cutoff_hz": {"type": "float", "default": 20.0, "min": 0.1, "max": 500.0},
    "enabled": {"type": "bool", "default": True},
}
```

Supported controls:

- `int`: `QSpinBox`
- `float`: `QDoubleSpinBox`
- `bool`: `QCheckBox`
- `choice`: `QComboBox`

Acceptance criteria:

- Built-in SMA exposes `window` through the same param mechanism as user
  transforms.
- UI rebuilds controls when the transform file reloads with a new param schema.
- Invalid param schema is reported and ignored.

## Step 6: Make Sources Pluggable

Add a source registry instead of hardcoded CLI conditionals.

Proposed shape:

```python
class SourceFactory(Protocol):
    name: str
    def add_cli_args(self, parser: argparse.ArgumentParser) -> None: ...
    def create(self, args: argparse.Namespace) -> SignalSource: ...
```

Initial factories:

- `deterministic`
- `vesc-imu`
- later: `csv-replay`, `socket`, `serial-lines`

Acceptance criteria:

- Adding a new source does not require editing the UI code.
- Source-specific CLI args are grouped and validated by the factory.
- Deterministic source remains the default test/smoke source.

## Step 7: Support Multiple Plots

Move from one plot widget to a layout driven by plot specs.

Example:

```python
plots = (
    PlotSpec(title="Time", series=(raw, filtered), x_mode="history_samples"),
    PlotSpec(title="Residual", series=(residual,), x_mode="history_samples"),
)
```

Acceptance criteria:

- One plot with multiple series still works.
- Two or more plots can share the same transformed output.
- Plot specs can be generated by built-in modes or by transform metadata.
- UI performance remains acceptable with decimation after transform execution.

## Step 8: Add Replay And Recording

Add a lightweight recording format for reproducible filter tuning.

Start with CSV or NPZ:

- timestamps
- channel arrays
- units metadata
- source metadata

Acceptance criteria:

- Any live source can be recorded.
- `csv-replay` or `npz-replay` source can feed the same transform/plot path.
- A recorded VESC IMU session can be replayed without hardware.

## Step 9: Add Biquad Filter Support

After the transform interface exists, add a built-in biquad transform.

Controls:

- type: low-pass, high-pass, band-pass, notch
- cutoff frequency
- Q
- gain for filter types that need it

Implementation notes:

- Prefer a small, tested implementation or a dependency with clear behavior.
- Use measured sample rate from the retained history.
- Recompute coefficients when params or sample rate change materially.

Acceptance criteria:

- Biquad transform works with deterministic source.
- Frequency response can be validated with generated sine inputs.
- UI can compare raw, filtered, and residual.

## Near-Term Recommended Order

1. Generalize data structures to `SignalBatch`.
2. Move SMA into `SmaTransform`.
3. Introduce `PlotSpec` and keep the current UI behavior unchanged.
4. Add watched transform-file loading.
5. Add transform-declared parameter controls.
6. Generalize VESC source from one axis to selectable channel sets.
7. Add multi-plot layout.
8. Add replay/recording.
9. Add biquad and frequency-domain helpers.

This order keeps the current bench usable at every step while progressively
removing hardcoded assumptions from the UI.
