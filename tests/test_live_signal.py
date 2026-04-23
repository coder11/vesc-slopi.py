import time

import numpy as np
import pytest

from vesc_py.live_signal import (
    DeterministicWhiteNoiseSignalSource,
    DeterministicSignalSource,
    NoisyDeterministicSignalSource,
    PendingSignalBuffer,
    SignalRingHistory,
    deterministic_noisy_signal_value,
    deterministic_signal_value,
    deterministic_white_noise,
    residual,
    trailing_sma,
)


def test_pending_signal_buffer_drains_samples_in_order_after_overwrite() -> None:
    buffer = PendingSignalBuffer(3)

    for index in range(5):
        buffer.append(float(index), float(index + 10))

    timestamps, values, dropped = buffer.drain()

    assert timestamps.tolist() == [2.0, 3.0, 4.0]
    assert values.tolist() == [12.0, 13.0, 14.0]
    assert dropped == 2


def test_pending_signal_buffer_reports_batch_overwrite_count() -> None:
    buffer = PendingSignalBuffer(4)

    buffer.append(0.0, 10.0)
    buffer.append_many(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float64),
    )

    timestamps, values, dropped = buffer.drain()

    assert timestamps.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert values.tolist() == [11.0, 12.0, 13.0, 14.0]
    assert dropped == 1


def test_signal_ring_history_keeps_latest_fixed_width_samples() -> None:
    history = SignalRingHistory(3)

    history.append_samples(
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([10.0, 11.0], dtype=np.float64),
    )
    history.append_samples(
        np.array([2.0, 3.0], dtype=np.float64),
        np.array([12.0, 13.0], dtype=np.float64),
    )

    assert history.count == 3
    assert history.valid_timestamps().tolist() == [1.0, 2.0, 3.0]
    assert history.valid_values().tolist() == [11.0, 12.0, 13.0]
    assert history.sample_hz() == pytest.approx(1.0)


def test_signal_ring_history_clears_and_accepts_new_samples() -> None:
    history = SignalRingHistory(3)
    history.append_samples(
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([10.0, 11.0], dtype=np.float64),
    )

    history.clear()
    history.append_samples(
        np.array([5.0], dtype=np.float64),
        np.array([15.0], dtype=np.float64),
    )

    assert history.count == 1
    assert history.valid_timestamps().tolist() == [5.0]
    assert history.valid_values().tolist() == [15.0]
    assert history.sample_hz() is None


def test_trailing_sma_uses_causal_warmup_window() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)

    filtered = trailing_sma(values, 3)

    np.testing.assert_allclose(filtered, np.array([1.0, 1.5, 7.0 / 3.0, 14.0 / 3.0]))


def test_trailing_sma_window_one_matches_raw() -> None:
    values = np.array([1.0, -2.0, 4.0], dtype=np.float64)

    filtered = trailing_sma(values, 1)

    np.testing.assert_allclose(filtered, values)
    assert filtered is not values


def test_trailing_sma_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        trailing_sma(np.array([1.0], dtype=np.float64), 0)


def test_residual_returns_raw_minus_filtered() -> None:
    raw = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    filtered = np.array([0.5, 1.5, 3.0], dtype=np.float64)

    np.testing.assert_allclose(residual(raw, filtered), np.array([0.5, 0.5, 1.0]))


def test_residual_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        residual(
            np.array([1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_deterministic_source_produces_repeatable_interface_samples() -> None:
    sample_rate_hz = 100.0
    source = DeterministicSignalSource(
        channel_name="acc_z",
        unit="g",
        sample_rate_hz=sample_rate_hz,
        pending_samples=64,
    )

    source.start()
    time.sleep(0.03)
    source.stop()
    timestamps, values, _dropped = source.drain()
    snapshot = source.snapshot()

    assert source.channel_name == "acc_z"
    assert source.unit == "g"
    assert snapshot.samples >= 1
    assert snapshot.done
    assert timestamps.size == values.size
    assert timestamps.size >= 1
    for timestamp, value in zip(timestamps, values):
        sample_index = round(float(timestamp) * sample_rate_hz)
        assert value == pytest.approx(
            deterministic_signal_value(sample_index, sample_rate_hz)
        )


def test_deterministic_white_noise_is_repeatable_and_bounded() -> None:
    samples = np.array([deterministic_white_noise(index) for index in range(1024)])
    repeated = np.array([deterministic_white_noise(index) for index in range(1024)])

    np.testing.assert_allclose(samples, repeated)
    assert np.all(samples >= -1.0)
    assert np.all(samples < 1.0)
    assert float(np.std(samples)) > 0.5


def test_deterministic_noisy_signal_is_repeatable_at_fixed_rate() -> None:
    sample_rate_hz = 200.0
    values = np.array(
        [
            deterministic_noisy_signal_value(index, sample_rate_hz)
            for index in range(400)
        ],
        dtype=np.float64,
    )
    repeated = np.array(
        [
            deterministic_noisy_signal_value(index, sample_rate_hz)
            for index in range(400)
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(values, repeated)
    assert float(np.std(values)) > 0.6


def test_noisy_deterministic_source_uses_noisy_signal_function() -> None:
    sample_rate_hz = 200.0
    source = NoisyDeterministicSignalSource(
        channel_name="acc_z",
        unit="g",
        sample_rate_hz=sample_rate_hz,
        pending_samples=64,
    )

    source.start()
    time.sleep(0.03)
    source.stop()
    timestamps, values, _dropped = source.drain()

    assert timestamps.size == values.size
    assert timestamps.size >= 1
    for timestamp, value in zip(timestamps, values):
        sample_index = round(float(timestamp) * sample_rate_hz)
        assert value == pytest.approx(
            deterministic_noisy_signal_value(sample_index, sample_rate_hz)
        )


def test_deterministic_white_noise_source_uses_white_noise_function() -> None:
    sample_rate_hz = 200.0
    source = DeterministicWhiteNoiseSignalSource(
        channel_name="acc_z",
        unit="g",
        sample_rate_hz=sample_rate_hz,
        pending_samples=64,
    )

    source.start()
    time.sleep(0.03)
    source.stop()
    timestamps, values, _dropped = source.drain()

    assert timestamps.size == values.size
    assert timestamps.size >= 1
    for timestamp, value in zip(timestamps, values):
        sample_index = round(float(timestamp) * sample_rate_hz)
        assert value == pytest.approx(deterministic_white_noise(sample_index))
