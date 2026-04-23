import argparse

import numpy as np
import pytest

from examples.live_signal_analysis import (
    build_analysis,
    build_axis_analysis_processor,
    build_parser,
    clamp_cutoff_hz,
    make_source,
    validate_args,
)
from vesc_py.live_analysis import AnalysisInput, SignalBatch, SignalBatchSourceSnapshot


def test_clamp_cutoff_hz_limits_requested_frequency_to_nyquist_margin() -> None:
    assert clamp_cutoff_hz(80.0, 100.0) == pytest.approx(49.0)
    assert clamp_cutoff_hz(20.0, 100.0) == pytest.approx(20.0)
    assert clamp_cutoff_hz(20.0, None) is None


def test_make_source_wraps_deterministic_source() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source",
            "deterministic",
            "--axis",
            "gyro_z",
            "--deterministic-rate",
            "321",
        ]
    )

    validate_args(parser, args)
    source, source_label = make_source(args)

    assert source.channels == {"gyro_z": "deg/s"}
    assert source_label == "Deterministic source @ 321 Hz"


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
        snapshot=SignalBatchSourceSnapshot(
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
        snapshot=SignalBatchSourceSnapshot(
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
    parser = build_parser()
    args = parser.parse_args(["--source", "deterministic", "--axis", "acc_z"])
    validate_args(parser, args)
    source, source_label = make_source(args)
    app = build_analysis(
        source=source,
        source_label=source_label,
        axis="acc_z",
        unit="g",
        history=2_000,
        max_points=500,
        plot_rate_hz=25.0,
        theme="dark",
        antialias=False,
    )

    assert [parameter.name for parameter in app.parameters] == [
        "cutoff_hz",
        "filter_order",
        "spectrum_mode",
    ]
    assert len(app.plots) == 2
