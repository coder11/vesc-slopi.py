import numpy as np
import pytest

import examples.yalsa.live_signal_analysis as live_signal_analysis
from examples.yalsa.live_signal_analysis import (
    LiveSignalAnalysisConfig,
    build_analysis,
    build_axis_analysis_processor,
    build_multi_axis_analysis_processor,
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
        deterministic_rate=321.0,
    )

    source, source_label = make_source(config)

    assert source.channels == {
        "acc_x": "g",
        "acc_y": "g",
        "acc_z": "g",
        "gyro_x": "deg/s",
        "gyro_y": "deg/s",
        "gyro_z": "deg/s",
    }
    assert source_label == "Deterministic source @ 321 Hz"


def test_make_source_labels_vesc_snapshot_rate_as_poll_rate() -> None:
    config = LiveSignalAnalysisConfig(
        source="vesc",
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
        axes=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        timeout=live_signal_analysis.DEFAULT_TIMEOUT,
        pending_samples=live_signal_analysis.DEFAULT_PENDING_SAMPLES,
        poll_rate_hz=500.0,
    )
    source._samples.append_many(
        [0.0, 0.01, 0.02],
        {
            "acc_x": [1.0, 2.0, 3.0],
            "acc_y": [1.0, 2.0, 3.0],
            "acc_z": [1.0, 2.0, 3.0],
            "gyro_x": [0.1, 0.2, 0.3],
            "gyro_y": [0.1, 0.2, 0.3],
            "gyro_z": [0.1, 0.2, 0.3],
        },
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
            self.channels = {
                "acc_x": "g",
                "acc_y": "g",
                "acc_z": "g",
                "gyro_x": "deg/s",
                "gyro_y": "deg/s",
                "gyro_z": "deg/s",
            }

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
        vesc_poll_rate=321.0,
    )
    target = VescTarget(VescConnection.serial("/dev/ttyACM0"), can_id=7)

    _source, source_label = make_source(config, vesc_target=target)

    assert calls == {
        "connection": target.connection,
        "axes": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
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


def test_build_multi_axis_analysis_processor_uses_independent_filters() -> None:
    processor = build_multi_axis_analysis_processor(
        ("acc_x", "gyro_z"),
        {"acc_x": "g", "gyro_z": "deg/s"},
    )
    timestamps = np.arange(400, dtype=np.float64) / 200.0
    acc_values = np.sin(2.0 * np.pi * 5.0 * timestamps)
    gyro_values = np.sin(2.0 * np.pi * 25.0 * timestamps)
    analysis_input = AnalysisInput(
        batch=SignalBatch(
            timestamps_s=timestamps,
            values={"acc_x": acc_values, "gyro_z": gyro_values},
            units={"acc_x": "g", "gyro_z": "deg/s"},
        ),
        sample_rate_hz=200.0,
        source_stats=SignalBatchSourceStats(
            samples=400,
            dropped=0,
            errors=0,
            average_rate_hz=200.0,
            latest_sample_s=float(timestamps[-1]),
            latest_values={
                "acc_x": float(acc_values[-1]),
                "gyro_z": float(gyro_values[-1]),
            },
            last_error=None,
            done=False,
        ),
    )

    result = processor(
        analysis_input,
        {
            "acc_x_filter_type": "lowpass",
            "acc_x_cutoff_hz": 10.0,
            "acc_x_filter_order": 2,
            "gyro_z_filter_type": "lowpass",
            "gyro_z_cutoff_hz": 40.0,
            "gyro_z_filter_order": 4,
            "spectrum_mode": "psd",
        },
    )

    assert {
        "acc_x_raw",
        "acc_x_filtered",
        "acc_x_raw_spectrum",
        "acc_x_filtered_spectrum",
        "gyro_z_raw",
        "gyro_z_filtered",
        "gyro_z_raw_spectrum",
        "gyro_z_filtered_spectrum",
    } <= set(result.series)
    assert not np.array_equal(result.series["acc_x_filtered"].y, acc_values)
    assert not np.array_equal(result.series["gyro_z_filtered"].y, gyro_values)
    assert result.metrics == {
        "acc_x_rms": "0.707107 g",
        "gyro_z_rms": "0.707107 deg/s",
    }
    assert result.status_text is not None
    assert "acc_x: lowpass, cutoff 10.00 Hz, order 2" in result.status_text
    assert "gyro_z: lowpass, cutoff 40.00 Hz, order 4" in result.status_text


def test_build_multi_axis_analysis_processor_adds_mahony_rpy_series() -> None:
    processor = build_multi_axis_analysis_processor(
        ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        {
            "acc_x": "g",
            "acc_y": "g",
            "acc_z": "g",
            "gyro_x": "deg/s",
            "gyro_y": "deg/s",
            "gyro_z": "deg/s",
        },
    )
    timestamps = np.arange(100, dtype=np.float64) / 100.0
    zeros = np.zeros_like(timestamps)
    ones = np.ones_like(timestamps)
    analysis_input = AnalysisInput(
        batch=SignalBatch(
            timestamps_s=timestamps,
            values={
                "acc_x": zeros,
                "acc_y": zeros,
                "acc_z": ones,
                "gyro_x": zeros,
                "gyro_y": zeros,
                "gyro_z": zeros,
            },
            units={
                "acc_x": "g",
                "acc_y": "g",
                "acc_z": "g",
                "gyro_x": "deg/s",
                "gyro_y": "deg/s",
                "gyro_z": "deg/s",
            },
        ),
        sample_rate_hz=100.0,
        source_stats=SignalBatchSourceStats(
            samples=100,
            dropped=0,
            errors=0,
            average_rate_hz=100.0,
            latest_sample_s=float(timestamps[-1]),
            latest_values={},
            last_error=None,
            done=False,
        ),
    )
    params = {
        f"{axis}_{suffix}": value
        for axis in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
        for suffix, value in (
            ("filter_type", "none"),
            ("cutoff_hz", 10.0),
            ("filter_order", 2),
        )
    }
    params["spectrum_mode"] = "psd"

    result = processor(analysis_input, params)

    assert result.series["mahony_roll"].x.size == timestamps.size
    assert result.series["mahony_pitch"].x.size == timestamps.size
    assert result.series["mahony_yaw"].x.size == timestamps.size
    np.testing.assert_allclose(result.series["mahony_roll"].y, 0.0)
    np.testing.assert_allclose(result.series["mahony_pitch"].y, 0.0)
    np.testing.assert_allclose(result.series["mahony_yaw"].y, 0.0)


def test_build_multi_axis_analysis_processor_feeds_filtered_data_to_mahony(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = build_multi_axis_analysis_processor(
        ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        {
            "acc_x": "g",
            "acc_y": "g",
            "acc_z": "g",
            "gyro_x": "deg/s",
            "gyro_y": "deg/s",
            "gyro_z": "deg/s",
        },
    )
    timestamps = np.arange(400, dtype=np.float64) / 200.0
    acc_x = np.sin(2.0 * np.pi * 25.0 * timestamps)
    zeros = np.zeros_like(timestamps)
    ones = np.ones_like(timestamps)
    captured: dict[str, np.ndarray] = {}

    def fake_mahony_roll_pitch_yaw_deg(
        timestamps: np.ndarray,
        *,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        gyro_x: np.ndarray,
        gyro_y: np.ndarray,
        gyro_z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del acc_y, acc_z, gyro_x, gyro_y, gyro_z
        captured["acc_x"] = acc_x
        empty = np.zeros_like(timestamps)
        return empty, empty, empty

    monkeypatch.setattr(
        live_signal_analysis,
        "_mahony_roll_pitch_yaw_deg",
        fake_mahony_roll_pitch_yaw_deg,
    )
    analysis_input = AnalysisInput(
        batch=SignalBatch(
            timestamps_s=timestamps,
            values={
                "acc_x": acc_x,
                "acc_y": zeros,
                "acc_z": ones,
                "gyro_x": zeros,
                "gyro_y": zeros,
                "gyro_z": zeros,
            },
            units={
                "acc_x": "g",
                "acc_y": "g",
                "acc_z": "g",
                "gyro_x": "deg/s",
                "gyro_y": "deg/s",
                "gyro_z": "deg/s",
            },
        ),
        sample_rate_hz=200.0,
        source_stats=SignalBatchSourceStats(
            samples=400,
            dropped=0,
            errors=0,
            average_rate_hz=200.0,
            latest_sample_s=float(timestamps[-1]),
            latest_values={},
            last_error=None,
            done=False,
        ),
    )
    params = {
        f"{axis}_{suffix}": value
        for axis in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
        for suffix, value in (
            ("filter_type", "lowpass"),
            ("cutoff_hz", 5.0),
            ("filter_order", 2),
        )
    }
    params["spectrum_mode"] = "psd"

    result = processor(analysis_input, params)

    assert captured["acc_x"] is result.series["acc_x_filtered"].y
    assert captured["acc_x"] is not acc_x


def test_build_multi_axis_analysis_processor_skips_mahony_when_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_signal_analysis, "DEFAULT_SHOW_MAHONY", False)

    def fake_mahony_roll_pitch_yaw_deg(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Mahony algorithm should be disabled")

    monkeypatch.setattr(
        live_signal_analysis,
        "_mahony_roll_pitch_yaw_deg",
        fake_mahony_roll_pitch_yaw_deg,
    )
    processor = build_multi_axis_analysis_processor(
        ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        {
            "acc_x": "g",
            "acc_y": "g",
            "acc_z": "g",
            "gyro_x": "deg/s",
            "gyro_y": "deg/s",
            "gyro_z": "deg/s",
        },
    )
    timestamps = np.arange(10, dtype=np.float64) / 100.0
    zeros = np.zeros_like(timestamps)
    params = {
        f"{axis}_{suffix}": value
        for axis in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
        for suffix, value in (
            ("filter_type", "none"),
            ("cutoff_hz", 10.0),
            ("filter_order", 2),
        )
    }
    params["spectrum_mode"] = "psd"

    result = processor(
        AnalysisInput(
            batch=SignalBatch(
                timestamps_s=timestamps,
                values={
                    "acc_x": zeros,
                    "acc_y": zeros,
                    "acc_z": np.ones_like(timestamps),
                    "gyro_x": zeros,
                    "gyro_y": zeros,
                    "gyro_z": zeros,
                },
                units={
                    "acc_x": "g",
                    "acc_y": "g",
                    "acc_z": "g",
                    "gyro_x": "deg/s",
                    "gyro_y": "deg/s",
                    "gyro_z": "deg/s",
                },
            ),
            sample_rate_hz=100.0,
            source_stats=SignalBatchSourceStats(
                samples=10,
                dropped=0,
                errors=0,
                average_rate_hz=100.0,
                latest_sample_s=float(timestamps[-1]),
                latest_values={},
                last_error=None,
                done=False,
            ),
        ),
        params,
    )

    assert "mahony_roll" not in result.series
    assert "mahony_pitch" not in result.series
    assert "mahony_yaw" not in result.series


def test_build_analysis_exposes_live_tunable_parameters() -> None:
    config = LiveSignalAnalysisConfig(
        source="deterministic",
    )
    source, source_label = make_source(config)
    app = build_analysis(
        source=source,
        source_label=source_label,
    )

    assert [parameter.name for parameter in app.parameters] == [
        "acc_x_filter_type",
        "acc_x_cutoff_hz",
        "acc_x_filter_order",
        "acc_y_filter_type",
        "acc_y_cutoff_hz",
        "acc_y_filter_order",
        "acc_z_filter_type",
        "acc_z_cutoff_hz",
        "acc_z_filter_order",
        "gyro_x_filter_type",
        "gyro_x_cutoff_hz",
        "gyro_x_filter_order",
        "gyro_y_filter_type",
        "gyro_y_cutoff_hz",
        "gyro_y_filter_order",
        "gyro_z_filter_type",
        "gyro_z_cutoff_hz",
        "gyro_z_filter_order",
        "spectrum_mode",
    ]
    assert app.theme == "light"
    assert len(app.plots) == 18
    assert [plot.section for plot in app.plots] == [
        "accel",
        "accel",
        "accel",
        "accel",
        "accel",
        "accel",
        "accel",
        "accel",
        "accel",
        "gyro",
        "gyro",
        "gyro",
        "gyro",
        "gyro",
        "gyro",
        "gyro",
        "gyro",
        "gyro",
    ]
    assert [plot.group for plot in app.plots] == [
        "X",
        "X",
        "X",
        "Y",
        "Y",
        "Y",
        "Z",
        "Z",
        "Z",
        "X",
        "X",
        "X",
        "Y",
        "Y",
        "Y",
        "Z",
        "Z",
        "Z",
    ]
    assert [plot.title for plot in app.plots[:4]] == [
        "Accel X Time Series",
        "Accel X Frequency",
        "Mahony RPY",
        "Accel Y Time Series",
    ]
    assert app.plots[11].title == "Mahony RPY"
    assert app.plots[11].widget == "empty"
    assert [metric.name for metric in app.metrics] == [
        "acc_x_rms",
        "acc_y_rms",
        "acc_z_rms",
        "gyro_x_rms",
        "gyro_y_rms",
        "gyro_z_rms",
    ]
    assert [metric.group for metric in app.metrics] == ["X", "Y", "Z", "X", "Y", "Z"]
    assert [parameter.group for parameter in app.parameters[:9]] == [
        "X",
        "X",
        "X",
        "Y",
        "Y",
        "Y",
        "Z",
        "Z",
        "Z",
    ]
    assert len(app.plots[0].traces) == 2


def test_build_analysis_can_show_3d_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_signal_analysis, "DEFAULT_SHOW_3D_OBJECT", True)
    config = LiveSignalAnalysisConfig(source="deterministic")
    source, source_label = make_source(config)

    app = build_analysis(source=source, source_label=source_label)

    assert app.plots[11].title == "VESC 3D View"
    assert app.plots[11].widget == "orientation_3d"


def test_build_analysis_can_hide_mahony(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_signal_analysis, "DEFAULT_SHOW_MAHONY", False)
    config = LiveSignalAnalysisConfig(source="deterministic")
    source, source_label = make_source(config)

    app = build_analysis(source=source, source_label=source_label)

    assert app.plots[2].widget == "empty"
    assert app.plots[11].widget == "empty"


def test_build_runtime_config_reads_module_level_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_signal_analysis, "RUN_SOURCE", "deterministic-noisy")
    monkeypatch.setattr(live_signal_analysis, "RUN_DETERMINISTIC_RATE", 321.0)
    monkeypatch.setattr(live_signal_analysis, "RUN_VESC_POLL_RATE", 123.0)
    monkeypatch.setattr(live_signal_analysis, "RUN_TIMEOUT", 0.25)

    assert build_runtime_config() == LiveSignalAnalysisConfig(
        source="deterministic-noisy",
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
        vesc_poll_rate=321.0,
        timeout=0.25,
    )
    assert calls["vesc_target"] == fake_target
    assert calls["app"] is fake_app
