import time

import numpy as np
import pytest

from yalsa import (
    ChoiceOption,
    ScalarSignalSourceAdapter,
    SignalBatch,
    SignalBatchHistory,
    choice_parameter,
    default_parameter_values,
    float_parameter,
    int_parameter,
    xy_series,
)
from vesc_py.live_signal import DeterministicSignalSource


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
