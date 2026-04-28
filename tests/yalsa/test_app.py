import time
from collections.abc import Callable, Mapping

import numpy as np
import pytest

import yalsa.app as yalsa_app
from yalsa import (
    AnalysisResult,
    ChoiceOption,
    LiveAnalysisApp,
    PlotSpec,
    PlotTrace,
    ScalarSignalSourceAdapter,
    SignalBatch,
    SignalBatchHistory,
    SignalBatchSourceSnapshot,
    choice_parameter,
    default_parameter_values,
    float_parameter,
    int_parameter,
    xy_series,
)
from vesc_py.live_signal import DeterministicSignalSource, SignalSourceSnapshot


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


class _FiniteBatchSource:
    def __init__(self, batches: list[SignalBatch]) -> None:
        self._batches = batches
        self._index = 0
        self._samples = 0
        self._latest_sample_s: float | None = None
        self._latest_value: float | None = None
        self.started = False
        self.stopped = False

    @property
    def channels(self) -> dict[str, str]:
        return {"acc_z": "g"}

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 1.0) -> None:
        self.stopped = True

    def drain(self) -> tuple[SignalBatch, int]:
        if self._index >= len(self._batches):
            return yalsa_app.empty_signal_batch(self.channels), 0
        batch = self._batches[self._index]
        self._index += 1
        self._samples += batch.sample_count
        if batch.sample_count > 0:
            self._latest_sample_s = float(batch.timestamps_s[-1])
            self._latest_value = float(batch.channel("acc_z")[-1])
        return batch, 0

    def snapshot(self) -> SignalBatchSourceSnapshot:
        latest_values = (
            {} if self._latest_value is None else {"acc_z": self._latest_value}
        )
        return SignalBatchSourceSnapshot(
            samples=self._samples,
            dropped=0,
            errors=0,
            average_rate_hz=float(self._samples),
            latest_sample_s=self._latest_sample_s,
            latest_values=latest_values,
            last_error=None,
            done=self.stopped,
        )


def _single_plot_app(
    source: _FiniteBatchSource,
    process: yalsa_app.ProcessCallback,
) -> LiveAnalysisApp:
    return LiveAnalysisApp(
        title="test",
        source=source,
        plots=(
            PlotSpec(
                title="time",
                traces=(PlotTrace(series="raw", label="Raw"),),
                x_label="time",
                y_label="acc_z",
            ),
        ),
        process=process,
        history=8,
    )


def test_signal_batch_rejects_mismatched_channel_lengths() -> None:
    with pytest.raises(ValueError, match="does not match timestamps"):
        SignalBatch(
            timestamps_s=np.array([0.0, 1.0], dtype=np.float64),
            values={"acc_z": np.array([1.0], dtype=np.float64)},
            units={"acc_z": "g"},
        )


def test_signal_batch_history_keeps_latest_samples_for_all_channels() -> None:
    history = SignalBatchHistory(3, {"acc_z": "g", "gyro_z": "deg/s"})

    history.append_batch(
        SignalBatch(
            timestamps_s=np.array([0.0, 1.0], dtype=np.float64),
            values={
                "acc_z": np.array([10.0, 11.0], dtype=np.float64),
                "gyro_z": np.array([20.0, 21.0], dtype=np.float64),
            },
            units={"acc_z": "g", "gyro_z": "deg/s"},
        )
    )
    history.append_batch(
        SignalBatch(
            timestamps_s=np.array([2.0, 3.0], dtype=np.float64),
            values={
                "acc_z": np.array([12.0, 13.0], dtype=np.float64),
                "gyro_z": np.array([22.0, 23.0], dtype=np.float64),
            },
            units={"acc_z": "g", "gyro_z": "deg/s"},
        )
    )

    snapshot = history.snapshot()

    assert history.count == 3
    assert snapshot.timestamps_s.tolist() == [1.0, 2.0, 3.0]
    assert snapshot.channel("acc_z").tolist() == [11.0, 12.0, 13.0]
    assert snapshot.channel("gyro_z").tolist() == [21.0, 22.0, 23.0]
    assert history.sample_hz() == pytest.approx(1.0)


def test_signal_batch_history_clear_resets_retained_samples() -> None:
    history = SignalBatchHistory(4, {"acc_z": "g"})
    history.append_batch(
        SignalBatch(
            timestamps_s=np.array([0.0, 1.0], dtype=np.float64),
            values={"acc_z": np.array([1.0, 2.0], dtype=np.float64)},
            units={"acc_z": "g"},
        )
    )

    history.clear()

    assert history.count == 0
    assert history.snapshot().sample_count == 0
    assert history.sample_hz() is None


def test_default_parameter_values_reject_duplicate_names() -> None:
    parameters = (
        int_parameter("window", default=4),
        float_parameter("window", default=2.5),
    )

    with pytest.raises(ValueError, match="duplicate parameter name"):
        default_parameter_values(parameters)


def test_choice_parameter_rejects_default_outside_choices() -> None:
    with pytest.raises(ValueError, match="default must match one option"):
        choice_parameter(
            "mode",
            default="fir",
            choices=(ChoiceOption(value="psd", label="PSD"),),
        )


def test_xy_series_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        xy_series(np.array([0.0], dtype=np.float64), np.array([], dtype=np.float64))


def test_finite_bounds_pads_flat_series() -> None:
    bounds = yalsa_app._finite_bounds(np.array([3.0, 3.0], dtype=np.float64))

    assert bounds == pytest.approx((2.85, 3.15))


def test_update_plot_ranges_skips_redundant_updates() -> None:
    tracker = yalsa_app._PlotRangeTracker()
    calls: list[tuple[str, float, float, float]] = []

    class FakePlotItem:
        def setXRange(self, lower: float, upper: float, *, padding: float) -> None:
            calls.append(("x", lower, upper, padding))

        def setYRange(self, lower: float, upper: float, *, padding: float) -> None:
            calls.append(("y", lower, upper, padding))

    plot_item = FakePlotItem()

    yalsa_app._update_plot_ranges(
        plot_item,
        tracker,
        x_bounds=(1.0, 2.0),
        y_bounds=(-1.0, 1.0),
    )
    yalsa_app._update_plot_ranges(
        plot_item,
        tracker,
        x_bounds=(1.01, 2.01),
        y_bounds=(-1.0, 1.0),
    )
    yalsa_app._update_plot_ranges(
        plot_item,
        tracker,
        x_bounds=(0.5, 1.5),
        y_bounds=(-2.0, 2.0),
    )

    assert calls == [
        ("x", 1.0, 2.0, 0.0),
        ("y", -1.0, 1.0, 0.0),
        ("x", 0.5, 1.5, 0.0),
        ("y", -2.0, 2.0, 0.0),
    ]


def test_scalar_signal_source_adapter_exposes_batch_protocol() -> None:
    scalar_source = DeterministicSignalSource(
        channel_name="acc_z",
        unit="g",
        sample_rate_hz=100.0,
        pending_samples=64,
    )
    source = ScalarSignalSourceAdapter(scalar_source)

    source.start()
    time.sleep(0.03)
    source.stop()
    batch, dropped = source.drain()
    snapshot = source.snapshot()

    assert source.channels == {"acc_z": "g"}
    assert dropped >= 0
    assert batch.sample_count >= 1
    assert batch.channel("acc_z").size == batch.timestamps_s.size
    assert snapshot.samples >= 1
    assert snapshot.done


def test_scalar_signal_source_adapter_preserves_debug_text() -> None:
    class FakeSource:
        channel_name = "acc_z"
        unit = "g"

        def start(self) -> None:
            return None

        def stop(self, timeout: float = 1.0) -> None:
            return None

        def drain(self) -> tuple[np.ndarray, np.ndarray, int]:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                0,
            )

        def snapshot(self) -> SignalSourceSnapshot:
            return SignalSourceSnapshot(
                samples=12,
                dropped=0,
                errors=0,
                average_rate_hz=345.0,
                latest_sample_s=1.23,
                latest_value=0.98,
                last_error=None,
                done=False,
                debug_text="srcdbg loop=2.666ms rd=1.802",
            )

    snapshot = ScalarSignalSourceAdapter(FakeSource()).snapshot()

    assert snapshot.debug_text == "srcdbg loop=2.666ms rd=1.802"


def test_live_analysis_worker_processes_source_batches_off_gui_thread() -> None:
    source = _FiniteBatchSource(
        [
            SignalBatch(
                timestamps_s=np.array([0.0, 0.001], dtype=np.float64),
                values={"acc_z": np.array([1.0, 2.0], dtype=np.float64)},
                units={"acc_z": "g"},
            )
        ]
    )
    sample_counts: list[int] = []

    def process(
        data: yalsa_app.AnalysisInput,
        params: Mapping[str, yalsa_app.ParamValue],
    ) -> AnalysisResult:
        del params
        sample_counts.append(data.batch.sample_count)
        return AnalysisResult(
            series={"raw": xy_series(data.timestamps_s, data.channel("acc_z"))},
            status_text=f"samples={data.batch.sample_count}",
        )

    worker = yalsa_app._LiveAnalysisWorker(_single_plot_app(source, process), {})
    worker.start()
    try:
        _wait_until(lambda: worker.snapshot().result is not None)
    finally:
        worker.stop()

    snapshot = worker.snapshot()
    assert source.started
    assert source.stopped
    assert sample_counts[-1] == 2
    assert snapshot.result is not None
    assert snapshot.result.status_text == "samples=2"
    assert snapshot.processing_rate_hz is not None
    assert snapshot.last_process_ms is not None


def test_live_analysis_worker_reprocesses_when_parameter_changes() -> None:
    source = _FiniteBatchSource(
        [
            SignalBatch(
                timestamps_s=np.array([0.0, 0.001], dtype=np.float64),
                values={"acc_z": np.array([1.0, 2.0], dtype=np.float64)},
                units={"acc_z": "g"},
            )
        ]
    )
    scales: list[float] = []

    def process(
        data: yalsa_app.AnalysisInput,
        params: Mapping[str, yalsa_app.ParamValue],
    ) -> AnalysisResult:
        scale = float(params["scale"])
        scales.append(scale)
        return AnalysisResult(
            series={"raw": xy_series(data.timestamps_s, data.channel("acc_z") * scale)},
            status_text=f"scale={scale:g}",
        )

    app = _single_plot_app(source, process)
    app = LiveAnalysisApp(
        title=app.title,
        source=app.source,
        plots=app.plots,
        process=app.process,
        parameters=(float_parameter("scale", default=1.0),),
        history=app.history,
    )
    worker = yalsa_app._LiveAnalysisWorker(
        app,
        default_parameter_values(app.parameters),
    )
    worker.start()
    try:
        _wait_until(lambda: worker.snapshot().result is not None)
        worker.set_parameter("scale", 3.0)
        _wait_until(
            lambda: (
                worker.snapshot().result is not None
                and worker.snapshot().result.status_text == "scale=3"
            )
        )
    finally:
        worker.stop()

    assert 1.0 in scales
    assert scales[-1] == 3.0
