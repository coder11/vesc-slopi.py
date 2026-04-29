import numpy as np
import pytest

import examples.yalsa.live_signal_analysis as live_signal_analysis
from examples.yalsa.live_signal_analysis import (
    LiveSignalAnalysisConfig,
    build_analysis,
    build_axis_analysis_processor,
    build_runtime_config,
    clamp_cutoff_hz,
    make_source,
)
from vesc_py.connection import VescConnection, VescTarget
from yalsa import AnalysisInput, SignalBatch, SignalBatchSourceStats


def test_clamp_cutoff_hz_limits_requested_frequency_to_nyquist_margin() -> None:
    assert clamp_cutoff_hz(80.0, 100.0) == pytest.approx(49.0)
    assert clamp_cutoff_hz(20.0, 100.0) == pytest.approx(20.0)
    assert clamp_cutoff_hz(20.0, None) is None


def test_make_source_wraps_deterministic_source() -> None:
    config = LiveSignalAnalysisConfig(
        source="deterministic",
        axis="acc_z",
        gyro_axis="gyro_z",
        deterministic_rate=321.0,
    )

    source, source_label = make_source(config)

    assert source.channels == {"acc_z": "g", "gyro_z": "deg/s"}
    assert source_label == "Deterministic source @ 321 Hz"


def test_make_source_labels_vesc_snapshot_rate_as_poll_rate() -> None:
    config = LiveSignalAnalysisConfig(
        source="vesc",
        axis="acc_z",
        gyro_axis="gyro_z",
        vesc_poll_rate=500.0,
    )
    target = VescTarget(VescConnection.serial("/dev/ttyACM0"), can_id=7)

    source, _source_label = make_source(config, vesc_target=target)
    stats = source.source_stats()

    assert stats.rate_label == "VESC poll rate"
    assert stats.average_rate_hz == pytest.approx(0.0)


def test_vesc_source_stats_average_rate_matches_pending_ring_timestamps() -> None:
    source = live_signal_analysis.VescImuBatchSignalSource(
        connection=VescConnection.serial("/dev/ttyACM0"),
        axes=("acc_z", "gyro_z"),
        timeout=live_signal_analysis.DEFAULT_TIMEOUT,
        pending_samples=live_signal_analysis.DEFAULT_PENDING_SAMPLES,
        poll_rate_hz=500.0,
    )
    source._samples.append_many(
        [0.0, 0.01, 0.02],
        {"acc_z": [1.0, 2.0, 3.0], "gyro_z": [0.1, 0.2, 0.3]},
    )

    stats = source.source_stats()

    assert stats.average_rate_hz == pytest.approx(100.0)


def test_make_source_passes_vesc_poll_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeVescImuBatchSignalSource:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)
            self.channels = {"acc_z": "g", "gyro_z": "deg/s"}

        def start(self) -> None:
            return None

        def stop(self, timeout: float = 1.0) -> None:
            return None

        def drain(self) -> tuple[object, object]:
            raise AssertionError("drain should not be called in this test")

        def source_stats(self) -> object:
            raise AssertionError("source_stats should not be called in this test")

    monkeypatch.setattr(
        live_signal_analysis,
        "VescImuBatchSignalSource",
        FakeVescImuBatchSignalSource,
    )
    config = LiveSignalAnalysisConfig(
        source="vesc",
        axis="acc_z",
        gyro_axis="gyro_z",
        vesc_poll_rate=321.0,
    )
    target = VescTarget(VescConnection.serial("/dev/ttyACM0"), can_id=7)

    _source, source_label = make_source(config, vesc_target=target)

    assert calls == {
        "connection": target.connection,
        "axes": ("acc_z", "gyro_z"),
        "timeout": pytest.approx(live_signal_analysis.DEFAULT_TIMEOUT),
        "pending_samples": live_signal_analysis.DEFAULT_PENDING_SAMPLES,
        "poll_rate_hz": 321.0,
        "can_id": 7,
    }
    assert "<= 321 Hz" in source_label


def test_build_axis_analysis_processor_returns_expected_series() -> None:
    processor = build_axis_analysis_processor("acc_z", "g")
    timestamps = np.arange(400, dtype=np.float64) / 200.0
    values = np.sin(2.0 * np.pi * 5.0 * timestamps)
    analysis_input = AnalysisInput(
        batch=SignalBatch(
            timestamps_s=timestamps,
            values={"acc_z": values},
            units={"acc_z": "g"},
        ),
        sample_rate_hz=200.0,
        source_stats=SignalBatchSourceStats(
            samples=400,
            dropped=0,
            errors=0,
            average_rate_hz=200.0,
            latest_sample_s=float(timestamps[-1]),
            latest_values={"acc_z": float(values[-1])},
            last_error=None,
            done=False,
        ),
    )

    result = processor(
        analysis_input,
        {
            "cutoff_hz": 12.0,
            "filter_order": 2,
            "spectrum_mode": "psd",
        },
    )

    assert set(result.series) == {
        "raw",
        "filtered",
        "raw_spectrum",
        "filtered_spectrum",
    }
    assert result.series["raw"].x.size == values.size
    assert result.series["filtered"].y.size == values.size
    assert result.series["raw_spectrum"].x.size >= 1
    assert result.status_text is not None
    assert "cutoff: 12.00 Hz" in result.status_text


def test_build_axis_analysis_processor_reports_cutoff_clamp() -> None:
    processor = build_axis_analysis_processor("acc_z", "g")
    timestamps = np.arange(200, dtype=np.float64) / 100.0
    values = np.sin(2.0 * np.pi * 5.0 * timestamps)
    analysis_input = AnalysisInput(
        batch=SignalBatch(
            timestamps_s=timestamps,
            values={"acc_z": values},
            units={"acc_z": "g"},
        ),
        sample_rate_hz=100.0,
        source_stats=SignalBatchSourceStats(
            samples=200,
            dropped=0,
            errors=0,
            average_rate_hz=100.0,
            latest_sample_s=float(timestamps[-1]),
            latest_values={"acc_z": float(values[-1])},
            last_error=None,
            done=False,
        ),
    )

    result = processor(
        analysis_input,
        {
            "cutoff_hz": 80.0,
            "filter_order": 2,
            "spectrum_mode": "fft",
        },
    )

    assert result.status_text is not None
    assert "requested cutoff clamped" in result.status_text


def test_build_analysis_exposes_live_tunable_parameters() -> None:
    config = LiveSignalAnalysisConfig(
        source="deterministic",
        axis="acc_z",
        gyro_axis="gyro_z",
    )
    source, source_label = make_source(config)
    app = build_analysis(
        source=source,
        source_label=source_label,
        axis="acc_z",
        unit="g",
        gyro_axis="gyro_z",
        gyro_unit="deg/s",
    )

    assert [parameter.name for parameter in app.parameters] == [
        "cutoff_hz",
        "filter_order",
        "spectrum_mode",
    ]
    assert app.theme == "light"
    assert len(app.plots) == 4
    assert [plot.title for plot in app.plots] == [
        "Accel Time Domain: acc_z",
        "Accel Frequency Domain: acc_z",
        "Gyro Time Domain: gyro_z",
        "Gyro Frequency Domain: gyro_z",
    ]
    assert [trace.color for trace in app.plots[0].traces] == [
        live_signal_analysis.RAW_TRACE_COLOR,
        live_signal_analysis.FILTERED_TRACE_COLOR,
    ]


def test_build_runtime_config_reads_module_level_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_signal_analysis, "RUN_SOURCE", "deterministic-noisy")
    monkeypatch.setattr(live_signal_analysis, "RUN_AXIS", "acc_y")
    monkeypatch.setattr(live_signal_analysis, "RUN_GYRO_AXIS", "gyro_y")
    monkeypatch.setattr(live_signal_analysis, "RUN_DETERMINISTIC_RATE", 321.0)
    monkeypatch.setattr(live_signal_analysis, "RUN_VESC_POLL_RATE", 123.0)
    monkeypatch.setattr(live_signal_analysis, "RUN_TIMEOUT", 0.25)

    assert build_runtime_config() == LiveSignalAnalysisConfig(
        source="deterministic-noisy",
        axis="acc_y",
        gyro_axis="gyro_y",
        timeout=0.25,
        deterministic_rate=321.0,
        vesc_poll_rate=123.0,
    )


def test_main_skips_vesc_connection_cli_for_non_vesc_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    fake_app = object()

    def fake_run_vesc_connection_cli(argv: tuple[str, ...]) -> VescTarget:
        raise AssertionError(f"unexpected VESC CLI call with {argv!r}")

    def fake_make_source(
        config: LiveSignalAnalysisConfig,
        *,
        vesc_target: VescTarget | None = None,
    ) -> tuple[object, str]:
        calls["config"] = config
        calls["vesc_target"] = vesc_target
        return object(), "source label"

    def fake_build_analysis(**kwargs: object) -> object:
        calls["build_analysis_kwargs"] = kwargs
        return fake_app

    def fake_run_live_analysis(app: object) -> None:
        calls["app"] = app

    monkeypatch.setattr(live_signal_analysis, "RUN_SOURCE", "deterministic")
    monkeypatch.setattr(live_signal_analysis, "RUN_AXIS", "acc_z")
    monkeypatch.setattr(live_signal_analysis, "RUN_GYRO_AXIS", "gyro_z")
    monkeypatch.setattr(live_signal_analysis, "RUN_DETERMINISTIC_RATE", 321.0)
    monkeypatch.setattr(
        live_signal_analysis,
        "run_vesc_connection_cli",
        fake_run_vesc_connection_cli,
    )
    monkeypatch.setattr(live_signal_analysis, "make_source", fake_make_source)
    monkeypatch.setattr(live_signal_analysis, "build_analysis", fake_build_analysis)
    monkeypatch.setattr(live_signal_analysis, "run_live_analysis", fake_run_live_analysis)

    live_signal_analysis.main(["--serial", "/dev/ttyACM0"])

    assert calls["config"] == LiveSignalAnalysisConfig(
        source="deterministic",
        axis="acc_z",
        gyro_axis="gyro_z",
        deterministic_rate=321.0,
        vesc_poll_rate=live_signal_analysis.DEFAULT_VESC_POLL_RATE,
    )
    assert calls["vesc_target"] is None
    assert calls["app"] is fake_app


def test_main_runs_vesc_connection_cli_for_vesc_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    fake_target = VescTarget(VescConnection.serial("/dev/ttyACM0"))
    fake_app = object()

    def fake_run_vesc_connection_cli(argv: tuple[str, ...]) -> VescTarget:
        calls["vesc_argv"] = argv
        return fake_target

    def fake_make_source(
        config: LiveSignalAnalysisConfig,
        *,
        vesc_target: VescTarget | None = None,
    ) -> tuple[object, str]:
        calls["config"] = config
        calls["vesc_target"] = vesc_target
        return object(), "source label"

    def fake_build_analysis(**kwargs: object) -> object:
        calls["build_analysis_kwargs"] = kwargs
        return fake_app

    def fake_run_live_analysis(app: object) -> None:
        calls["app"] = app

    monkeypatch.setattr(
        live_signal_analysis,
        "run_vesc_connection_cli",
        fake_run_vesc_connection_cli,
    )
    monkeypatch.setattr(live_signal_analysis, "make_source", fake_make_source)
    monkeypatch.setattr(live_signal_analysis, "build_analysis", fake_build_analysis)
    monkeypatch.setattr(live_signal_analysis, "run_live_analysis", fake_run_live_analysis)
    monkeypatch.setattr(live_signal_analysis, "RUN_SOURCE", "vesc")
    monkeypatch.setattr(live_signal_analysis, "RUN_AXIS", "acc_z")
    monkeypatch.setattr(live_signal_analysis, "RUN_GYRO_AXIS", "gyro_z")
    monkeypatch.setattr(live_signal_analysis, "RUN_VESC_POLL_RATE", 321.0)
    monkeypatch.setattr(live_signal_analysis, "RUN_TIMEOUT", 0.25)

    live_signal_analysis.main(
        [
            "--serial",
            "/dev/ttyACM0",
        ]
    )

    assert calls["vesc_argv"] == ("--serial", "/dev/ttyACM0", "--timeout", "0.25")
    assert calls["config"] == LiveSignalAnalysisConfig(
        source="vesc",
        axis="acc_z",
        gyro_axis="gyro_z",
        vesc_poll_rate=321.0,
        timeout=0.25,
    )
    assert calls["vesc_target"] == fake_target
    assert calls["app"] is fake_app
