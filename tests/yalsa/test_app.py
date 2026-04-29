import time
from collections.abc import Callable, Mapping

import numpy as np
import pytest

import yalsa.app as yalsa_app
from vesc_py.live_signal import DeterministicSignalSource
from yalsa import (
    AnalysisResult,
    ChoiceOption,
    LiveAnalysisApp,
    PlotSpec,
    PlotTrace,
    ScalarSignalSourceAdapter,
    SignalBatch,
    SignalBatchHistory,
    SignalBatchSourceStats,
    choice_parameter,
    default_parameter_values,
    float_parameter,
    int_parameter,
    xy_series,
)


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

    def drain(self) -> tuple[SignalBatch, SignalBatchSourceStats]:
        if self._index >= len(self._batches):
            empty = yalsa_app.empty_signal_batch(self.channels)
            ring = yalsa_app.pending_batch_ring_stats(empty, cumulative_dropped=0)
            stats = yalsa_app.merge_signal_batch_source_stats(
                ring,
                errors=0,
                last_error=None,
                done=self.stopped,
            )
            return empty, stats
        batch = self._batches[self._index]
        self._index += 1
        self._samples += batch.sample_count
        if batch.sample_count > 0:
            self._latest_sample_s = float(batch.timestamps_s[-1])
            self._latest_value = float(batch.channel("acc_z")[-1])
        ring = yalsa_app.pending_batch_ring_stats(batch, cumulative_dropped=0)
        return batch, yalsa_app.merge_signal_batch_source_stats(
            ring,
            errors=0,
            last_error=None,
            done=self.stopped,
        )

    def source_stats(self) -> SignalBatchSourceStats:
        latest_values = (
            {} if self._latest_value is None else {"acc_z": self._latest_value}
        )
        return SignalBatchSourceStats(
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


def test_plot_spec_rejects_invalid_y_range() -> None:
    with pytest.raises(ValueError, match="y_range"):
        PlotSpec(
            title="t",
            traces=(PlotTrace(series="raw", label="Raw"),),
            x_label="x",
            y_label="y",
            y_range=(1.0, -1.0),
        )


def test_plot_spec_rejects_invalid_mouse_mode() -> None:
    with pytest.raises(ValueError, match="mouse_mode must be 'pan' or 'rect'"):
        PlotSpec(
            title="time",
            traces=(PlotTrace(series="raw", label="Raw"),),
            x_label="time",
            y_label="acc_z",
            mouse_mode="scale-box",  # type: ignore[arg-type]
        )


def test_plot_spec_rejects_invalid_x_axis_mode() -> None:
    with pytest.raises(ValueError, match="x_axis_mode must be 'auto' or 'follow_latest'"):
        PlotSpec(
            title="time",
            traces=(PlotTrace(series="raw", label="Raw"),),
            x_label="time",
            y_label="acc_z",
            x_axis_mode="fixed",  # type: ignore[arg-type]
        )


def test_follow_latest_x_range_expands_single_point() -> None:
    assert yalsa_app._follow_latest_x_range(2.0, 2.0) == pytest.approx((1.9, 2.1))


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
    batch, stats = source.drain()

    assert source.channels == {"acc_z": "g"}
    assert batch.sample_count >= 1
    assert batch.channel("acc_z").size == batch.timestamps_s.size
    assert stats.samples >= 1
    assert stats.done

    peek = source.source_stats()
    assert peek.samples >= 0


def test_live_analysis_status_lines_include_signal_and_source_data() -> None:
    worker_snapshot = yalsa_app._AnalysisWorkerSnapshot(
        result=AnalysisResult(
            series={"raw": xy_series([0.0], [1.0])},
            status_text="mode: PSD | raw RMS: 0.5 g | filtered RMS: 0.4 g",
        ),
        process_error=None,
        source_stats=SignalBatchSourceStats(
            samples=10,
            dropped=2,
            errors=1,
            average_rate_hz=123.0,
            latest_sample_s=0.1,
            latest_values={"acc_z": 0.25},
            last_error=None,
            done=False,
        ),
        history_rate_hz=120.0,
    )

    assert yalsa_app._signal_status_lines(worker_snapshot, {"acc_z": "g"}) == [
        "current acc_z: 0.25 g",
        "mode: PSD",
        "raw RMS: 0.5 g",
        "filtered RMS: 0.4 g",
    ]
    assert yalsa_app._debug_status_lines(worker_snapshot) == [
        "data acquisition rate: 123.0 Hz",
        "history: 120.0 Hz",
        "samples: 10",
        "dropped: 2",
        "errors: 1",
    ]
    assert (
        yalsa_app._debug_toggle_text(worker_snapshot)
        == "Data acquisition rate: 123.0 Hz"
    )
    assert yalsa_app._debug_toggle_text(None) == "Data acquisition rate: measuring"


def test_vesc_poll_rate_display_sma_smooths_only_vesc_label() -> None:
    sma = yalsa_app._VescPollRateDisplaySma(window=4)
    assert sma.smooth(100.0, rate_label="Data acquisition rate") == pytest.approx(100.0)
    assert sma.smooth(200.0, rate_label="VESC poll rate") == pytest.approx(200.0)
    assert sma.smooth(400.0, rate_label="VESC poll rate") == pytest.approx(300.0)
    assert sma.smooth(400.0, rate_label="VESC poll rate") == pytest.approx(
        (200.0 + 400.0 + 400.0) / 3.0
    )
    assert sma.smooth(80.0, rate_label="Data acquisition rate") == pytest.approx(80.0)
    assert sma.smooth(120.0, rate_label="VESC poll rate") == pytest.approx(120.0)
    sma.reset()
    assert sma.smooth(50.0, rate_label="VESC poll rate") == pytest.approx(50.0)


def test_qt_theme_stylesheet_forces_light_widget_palette() -> None:
    stylesheet = yalsa_app._qt_theme_stylesheet(yalsa_app.PLOT_THEMES["light"])

    assert "background-color: #f6f7f9" in stylesheet
    assert "color: #202124" in stylesheet
    assert "QPushButton, QToolButton, QSpinBox, QDoubleSpinBox, QComboBox" in stylesheet
    assert "border: 1px solid #c7cdd4" in stylesheet


def test_live_analysis_app_rejects_invalid_drain_stride() -> None:
    source = _FiniteBatchSource([])
    base = _single_plot_app(
        source,
        lambda d, p: AnalysisResult(
            series={"raw": xy_series(d.timestamps_s, d.channel("acc_z"))},
        ),
    )
    with pytest.raises(ValueError, match="drain_stride"):
        LiveAnalysisApp(
            title=base.title,
            source=base.source,
            plots=base.plots,
            process=base.process,
            history=base.history,
            drain_stride=0,
        )


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


def test_live_analysis_ui_config_excludes_script_runtime_objects() -> None:
    source = _FiniteBatchSource([])

    def process(
        data: yalsa_app.AnalysisInput,
        params: Mapping[str, yalsa_app.ParamValue],
    ) -> AnalysisResult:
        del params
        return AnalysisResult(
            series={"raw": xy_series(data.timestamps_s, data.channel("acc_z"))},
        )

    ui_config = yalsa_app.live_analysis_ui_config(_single_plot_app(source, process))

    assert ui_config.title == "test"
    assert ui_config.channels == {"acc_z": "g"}
    assert ui_config.plots[0].title == "time"
    assert ui_config.plots[0].auto_range_x is True
    assert ui_config.plots[0].auto_range_y is True
    assert ui_config.plots[0].allow_mouse_x is True
    assert ui_config.plots[0].allow_mouse_y is True
    assert ui_config.plots[0].mouse_mode == "pan"
    assert ui_config.plots[0].x_axis_mode == "auto"
    assert ui_config.plots[0].allow_left_drag is True
    assert not hasattr(ui_config, "source")
    assert not hasattr(ui_config, "process")


def test_pickle_shared_memory_slot_keeps_latest_payload() -> None:
    slot = yalsa_app._PickleSharedMemorySlot.create(4096)
    attached: yalsa_app._PickleSharedMemorySlot | None = None
    try:
        first_version = slot.write({"value": 1})
        second_version = slot.write({"value": 2})
        attached = yalsa_app._PickleSharedMemorySlot.attach(slot.spec)

        payload, version = attached.read_with_version()

        assert first_version == 1
        assert second_version == 2
        assert payload == {"value": 2}
        assert version == second_version
    finally:
        if attached is not None:
            attached.close()
        slot.close()
        slot.unlink()


def test_analysis_worker_process_publishes_results_via_shared_memory() -> None:
    source = _FiniteBatchSource(
        [
            SignalBatch(
                timestamps_s=np.array([0.0, 0.001], dtype=np.float64),
                values={"acc_z": np.array([1.0, 2.0], dtype=np.float64)},
                units={"acc_z": "g"},
            )
        ]
    )

    def process(
        data: yalsa_app.AnalysisInput,
        params: Mapping[str, yalsa_app.ParamValue],
    ) -> AnalysisResult:
        scale = float(params["scale"])
        return AnalysisResult(
            series={"raw": xy_series(data.timestamps_s, data.channel("acc_z") * scale)},
            status_text=f"scale={scale:g}",
        )

    base_app = _single_plot_app(source, process)
    app = LiveAnalysisApp(
        title=base_app.title,
        source=base_app.source,
        plots=base_app.plots,
        process=base_app.process,
        parameters=(float_parameter("scale", default=2.0),),
        history=base_app.history,
        plot_rate_hz=30.0,
    )
    memory = yalsa_app._SharedAnalysisMemory.create(
        yalsa_app.live_analysis_ui_config(app)
    )
    context = yalsa_app._multiprocessing_context()
    worker = context.Process(
        target=yalsa_app._run_analysis_worker_process,
        args=(app, memory.spec),
    )
    worker.start()

    def latest_status_text() -> str | None:
        snapshot, _version = yalsa_app._read_analysis_snapshot(memory.state)
        return None if snapshot.result is None else snapshot.result.status_text

    try:
        _wait_until(lambda: latest_status_text() == "scale=2")
        control = yalsa_app._read_live_analysis_control(memory.control)
        memory.control.write(
            yalsa_app._LiveAnalysisControl(
                parameter_values={"scale": 5.0},
                parameter_revision=control.parameter_revision + 1,
                clear_revision=control.clear_revision,
                stop_requested=control.stop_requested,
                shutdown_requested=control.shutdown_requested,
            )
        )

        _wait_until(lambda: latest_status_text() == "scale=5")
    finally:
        control = yalsa_app._read_live_analysis_control(memory.control)
        memory.control.write(
            yalsa_app._LiveAnalysisControl(
                parameter_values=control.parameter_values,
                parameter_revision=control.parameter_revision,
                clear_revision=control.clear_revision,
                stop_requested=True,
                shutdown_requested=True,
            )
        )
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)
        memory.close()
        memory.unlink()

    assert worker.exitcode == 0
