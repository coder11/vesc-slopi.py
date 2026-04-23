"""Generic live signal primitives for capture, filtering, and plotting."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

NSEC_PER_SEC = 1_000_000_000

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SignalSourceSnapshot:
    """Low-rate status data exposed by a live signal source."""

    samples: int
    dropped: int
    errors: int
    average_rate_hz: float
    latest_sample_s: float | None
    latest_value: float | None
    last_error: str | None
    done: bool


class SignalSource(Protocol):
    """Common interface for live scalar signal sources."""

    @property
    def channel_name(self) -> str: ...

    @property
    def unit(self) -> str: ...

    def start(self) -> None: ...

    def stop(self, timeout: float = 1.0) -> None: ...

    def drain(self) -> tuple[FloatArray, FloatArray, int]: ...

    def snapshot(self) -> SignalSourceSnapshot: ...


class PendingSignalBuffer:
    """Thread-safe overwrite ring for samples waiting for a consumer."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self._capacity = capacity
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._values = np.zeros(capacity, dtype=np.float64)
        self._lock = threading.Lock()
        self._read_index = 0
        self._write_index = 0
        self._count = 0
        self._dropped = 0

    def append(self, timestamp_s: float, value: float) -> None:
        """Append one sample, overwriting the oldest pending sample if full."""
        self.append_many((timestamp_s,), (value,))

    def append_many(
        self,
        timestamps: Sequence[float] | FloatArray,
        values: Sequence[float] | FloatArray,
    ) -> None:
        """Append multiple samples while taking the shared lock once."""
        count = len(timestamps)
        if count == 0:
            return
        if count != len(values):
            raise ValueError("timestamp and value counts must match")

        timestamp_values = np.asarray(timestamps, dtype=np.float64)
        sample_values = np.asarray(values, dtype=np.float64)

        with self._lock:
            if count >= self._capacity:
                self._dropped += self._count + count - self._capacity
                self._timestamps[:] = timestamp_values[-self._capacity :]
                self._values[:] = sample_values[-self._capacity :]
                self._read_index = 0
                self._write_index = 0
                self._count = self._capacity
                return

            overflow = max(0, self._count + count - self._capacity)
            if overflow > 0:
                self._read_index = (self._read_index + overflow) % self._capacity
                self._dropped += overflow
            self._count = min(self._capacity, self._count + count)

            first_count = min(count, self._capacity - self._write_index)
            self._timestamps[self._write_index : self._write_index + first_count] = (
                timestamp_values[:first_count]
            )
            self._values[self._write_index : self._write_index + first_count] = (
                sample_values[:first_count]
            )

            remaining = count - first_count
            if remaining > 0:
                self._timestamps[:remaining] = timestamp_values[first_count:]
                self._values[:remaining] = sample_values[first_count:]

            self._write_index = (self._write_index + count) % self._capacity

    def drain(self) -> tuple[FloatArray, FloatArray, int]:
        """Return pending samples in order and clear the pending ring."""
        with self._lock:
            count = self._count
            dropped = self._dropped
            self._dropped = 0
            if count == 0:
                return (
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                    dropped,
                )

            read_index = self._read_index
            if read_index + count <= self._capacity:
                timestamps = self._timestamps[read_index : read_index + count].copy()
                values = self._values[read_index : read_index + count].copy()
            else:
                first_count = self._capacity - read_index
                timestamps = np.concatenate(
                    (
                        self._timestamps[read_index:],
                        self._timestamps[: count - first_count],
                    )
                )
                values = np.concatenate(
                    (
                        self._values[read_index:],
                        self._values[: count - first_count],
                    )
                )

            self._read_index = self._write_index
            self._count = 0
            return timestamps, values, dropped


class SignalRingHistory:
    """Fixed-size ring history for one scalar signal."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self._capacity = capacity
        self._count = 0
        self._write_index = 0
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._values = np.zeros(capacity, dtype=np.float64)

    @property
    def count(self) -> int:
        return self._count

    def clear(self) -> None:
        """Remove retained samples without changing capacity."""
        self._count = 0
        self._write_index = 0

    def append_samples(self, timestamps: FloatArray, values: FloatArray) -> None:
        """Append samples, keeping only the latest capacity samples."""
        sample_count = int(timestamps.size)
        if sample_count == 0:
            return
        if values.size != sample_count:
            raise ValueError("timestamp and value counts must match")

        if sample_count >= self._capacity:
            self._timestamps[:] = timestamps[-self._capacity :]
            self._values[:] = values[-self._capacity :]
            self._count = self._capacity
            self._write_index = 0
            return

        first_count = min(sample_count, self._capacity - self._write_index)
        self._timestamps[self._write_index : self._write_index + first_count] = (
            timestamps[:first_count]
        )
        self._values[self._write_index : self._write_index + first_count] = values[
            :first_count
        ]

        remaining = sample_count - first_count
        if remaining > 0:
            self._timestamps[:remaining] = timestamps[first_count:]
            self._values[:remaining] = values[first_count:]

        self._write_index = (self._write_index + sample_count) % self._capacity
        self._count = min(self._capacity, self._count + sample_count)

    def valid_timestamps(self) -> FloatArray:
        """Return retained timestamps in chronological order."""
        if self._count == 0:
            return self._timestamps[:0]
        return self._ordered_values(self._timestamps)

    def valid_values(self) -> FloatArray:
        """Return retained values in chronological order."""
        if self._count == 0:
            return self._values[:0]
        return self._ordered_values(self._values)

    def sample_hz(self) -> float | None:
        """Return measured sample rate from retained timestamps."""
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


def trailing_sma(values: FloatArray, window: int) -> FloatArray:
    """Return a causal trailing simple moving average."""
    if window <= 0:
        raise ValueError("window must be greater than 0")
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    if window == 1:
        return values.astype(np.float64, copy=True)

    raw = values.astype(np.float64, copy=False)
    cumulative = np.cumsum(raw, dtype=np.float64)
    result = cumulative.copy()
    result[window:] = cumulative[window:] - cumulative[:-window]
    counts = np.minimum(np.arange(raw.size, dtype=np.float64) + 1.0, float(window))
    return result / counts


def residual(raw: FloatArray, filtered: FloatArray) -> FloatArray:
    """Return raw minus filtered samples."""
    if raw.shape != filtered.shape:
        raise ValueError("raw and filtered arrays must have the same shape")
    return raw - filtered


def deterministic_signal_value(sample_index: int, sample_rate_hz: float) -> float:
    """Return a repeatable mixed signal value for a deterministic sample index."""
    t = sample_index / sample_rate_hz
    base = math.sin(2.0 * math.pi * 1.7 * t)
    ripple = 0.25 * math.sin(2.0 * math.pi * 23.0 * t)
    deterministic_noise = 0.08 * math.sin(2.0 * math.pi * 71.0 * t + 0.3)
    return base + ripple + deterministic_noise


def deterministic_white_noise(sample_index: int) -> float:
    """Return a repeatable uniform white-noise sample in the range [-1, 1)."""
    mask = (1 << 64) - 1
    state = (sample_index + 0x9E3779B97F4A7C15) & mask
    state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
    state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & mask
    state ^= state >> 31
    unit = (state >> 11) * (1.0 / (1 << 53))
    return unit * 2.0 - 1.0


def deterministic_noisy_signal_value(
    sample_index: int,
    sample_rate_hz: float,
) -> float:
    """Return a repeatable mixed signal with pseudo-white noise."""
    t = sample_index / sample_rate_hz
    base = math.sin(2.0 * math.pi * 2.0 * t)
    mid_tone = 0.35 * math.sin(2.0 * math.pi * 12.0 * t + 0.2)
    high_tone = 0.2 * math.sin(2.0 * math.pi * 43.0 * t + 0.7)
    white_noise = 0.18 * deterministic_white_noise(sample_index)
    return base + mid_tone + high_tone + white_noise


class DeterministicSignalSource:
    """Repeatable synthetic signal source implementing the SignalSource protocol."""

    def __init__(
        self,
        *,
        channel_name: str,
        unit: str,
        sample_rate_hz: float,
        pending_samples: int,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be greater than 0")
        self._channel_name = channel_name
        self._unit = unit
        self._sample_rate_hz = sample_rate_hz
        self._samples = PendingSignalBuffer(pending_samples)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._start_ns = 0
        self._sample_index = 0
        self._sample_count = 0
        self._dropped = 0
        self._latest_sample_s: float | None = None
        self._latest_value: float | None = None

    @property
    def channel_name(self) -> str:
        return self._channel_name

    @property
    def unit(self) -> str:
        return self._unit

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._done.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="deterministic-signal-source",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def drain(self) -> tuple[FloatArray, FloatArray, int]:
        timestamps, values, dropped = self._samples.drain()
        if dropped:
            with self._lock:
                self._dropped += dropped
        return timestamps, values, dropped

    def snapshot(self) -> SignalSourceSnapshot:
        with self._lock:
            elapsed_s = (
                (time.perf_counter_ns() - self._start_ns) / NSEC_PER_SEC
                if self._start_ns > 0
                else 0.0
            )
            average_rate = self._sample_count / elapsed_s if elapsed_s > 0.0 else 0.0
            return SignalSourceSnapshot(
                samples=self._sample_count,
                dropped=self._dropped,
                errors=0,
                average_rate_hz=average_rate,
                latest_sample_s=self._latest_sample_s,
                latest_value=self._latest_value,
                last_error=None,
                done=self._done.is_set(),
            )

    def _value_for_sample(self, sample_index: int) -> float:
        return deterministic_signal_value(sample_index, self._sample_rate_hz)

    def _run(self) -> None:
        self._start_ns = time.perf_counter_ns()
        next_sample_ns = self._start_ns
        sample_period_ns = max(1, round(NSEC_PER_SEC / self._sample_rate_hz))
        batch_size = 32
        try:
            while not self._stop.is_set():
                now_ns = time.perf_counter_ns()
                if now_ns < next_sample_ns:
                    sleep_s = min((next_sample_ns - now_ns) / NSEC_PER_SEC, 0.01)
                    time.sleep(max(0.0, sleep_s))
                    continue

                due_count = min(
                    batch_size,
                    max(1, ((now_ns - next_sample_ns) // sample_period_ns) + 1),
                )
                indexes = np.arange(
                    self._sample_index,
                    self._sample_index + due_count,
                    dtype=np.int64,
                )
                timestamps = indexes.astype(np.float64) / self._sample_rate_hz
                values = np.array(
                    [self._value_for_sample(int(index)) for index in indexes],
                    dtype=np.float64,
                )
                self._samples.append_many(timestamps, values)
                self._sample_index += due_count
                next_sample_ns += due_count * sample_period_ns

                with self._lock:
                    self._sample_count += due_count
                    self._latest_sample_s = float(timestamps[-1])
                    self._latest_value = float(values[-1])
        finally:
            self._done.set()


class NoisyDeterministicSignalSource(DeterministicSignalSource):
    """Repeatable synthetic signal source with pseudo-white noise."""

    def _value_for_sample(self, sample_index: int) -> float:
        return deterministic_noisy_signal_value(sample_index, self._sample_rate_hz)


class DeterministicWhiteNoiseSignalSource(DeterministicSignalSource):
    """Repeatable synthetic signal source containing only pseudo-white noise."""

    def _value_for_sample(self, sample_index: int) -> float:
        return deterministic_white_noise(sample_index)


__all__ = [
    "DeterministicWhiteNoiseSignalSource",
    "DeterministicSignalSource",
    "FloatArray",
    "NoisyDeterministicSignalSource",
    "PendingSignalBuffer",
    "SignalRingHistory",
    "SignalSource",
    "SignalSourceSnapshot",
    "deterministic_noisy_signal_value",
    "deterministic_signal_value",
    "deterministic_white_noise",
    "residual",
    "trailing_sma",
]
