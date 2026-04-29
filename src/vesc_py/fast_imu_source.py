"""Fast direct IMU signal source for the live signal bench."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vesc_py.buffer import VescBuffer
from vesc_py.comm_ids import CommPacketId
from vesc_py.connection import (
    BlockingIo,
    VescConnection,
    build_imu_request,
    open_blocking_io,
)
from vesc_py.crc import crc16
from vesc_py.imu import IMU_FIELDS
from vesc_py.live_signal import (
    NSEC_PER_SEC,
    PendingSignalBuffer,
    SignalSourceStats,
)
from vesc_py.packet import MAX_PACKET_LEN

DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 0.1

IMU_BENCH_FIELDS = (
    "roll",
    "pitch",
    "yaw",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)

_IMU_AXIS_ALIASES = {
    "x": "acc_x",
    "y": "acc_y",
    "z": "acc_z",
    "accel_x": "acc_x",
    "accel_y": "acc_y",
    "accel_z": "acc_z",
    **{field_name: field_name for field_name in IMU_BENCH_FIELDS},
}
_IMU_AXIS_UNITS = {
    "roll": "deg",
    "pitch": "deg",
    "yaw": "deg",
    "acc_x": "g",
    "acc_y": "g",
    "acc_z": "g",
    "gyro_x": "deg/s",
    "gyro_y": "deg/s",
    "gyro_z": "deg/s",
}
_RATE_LIMIT_SLEEP_SLACK_NS = 1_000_000
_SAMPLE_PUBLISH_INTERVAL_NS = 20_000_000
_SAMPLE_PUBLISH_BATCH_SIZE = 64


@dataclass(slots=True)
class ImuSourceStats:
    """Internal counters for VESC source failures."""

    timeouts: int = 0
    parse_errors: int = 0
    discarded_bytes: int = 0
    bad_crc: int = 0
    bad_stop: int = 0
    invalid_length: int = 0
    unexpected_packets: int = 0

    @property
    def errors(self) -> int:
        return (
            self.timeouts
            + self.parse_errors
            + self.bad_crc
            + self.bad_stop
            + self.invalid_length
            + self.unexpected_packets
        )


@dataclass(frozen=True, slots=True)
class ImuData:
    """Display-unit IMU data parsed from one COMM_GET_IMU_DATA response."""

    response_mask: int = 0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    vesc_id: int = 0

    def includes_axis(self, axis: str) -> bool:
        """Return whether the response mask includes one bench axis."""
        return _field_value_index(self.response_mask, parse_imu_axis(axis)) is not None

    def axis_value(self, axis: str) -> float:
        """Return one bench axis value by parsed axis name or alias."""
        return float(getattr(self, parse_imu_axis(axis)))

    def axis_values(self, axes: Sequence[str]) -> dict[str, float]:
        """Return selected bench axis values keyed by parsed axis name."""
        parsed_axes = parse_imu_axes(axes)
        missing_axes = [axis for axis in parsed_axes if not self.includes_axis(axis)]
        if missing_axes:
            joined = ", ".join(missing_axes)
            raise ValueError(
                f"IMU response mask 0x{self.response_mask:04x} does not include "
                f"{joined}"
            )
        return {axis: self.axis_value(axis) for axis in parsed_axes}


@dataclass(frozen=True, slots=True)
class _QueuedImuSample:
    timestamp_s: float
    value: float
    response_latency_s: float
    request_lateness_s: float
    missed_slots: float


def parse_imu_axis(text: str) -> str:
    """Parse an IMU bench axis name or alias."""
    token = text.strip().lower().replace("-", "_")
    try:
        return _IMU_AXIS_ALIASES[token]
    except KeyError as exc:
        valid = ", ".join(IMU_BENCH_FIELDS)
        raise ValueError(f"unknown IMU axis {text!r}; valid axes: {valid}") from exc


def imu_axis_mask(axis: str) -> int:
    """Return the COMM_GET_IMU_DATA mask bit for one bench axis."""
    parsed_axis = parse_imu_axis(axis)
    return 1 << IMU_FIELDS.index(parsed_axis)


def parse_imu_axes(axes: Sequence[str]) -> tuple[str, ...]:
    """Parse and validate an ordered set of IMU bench axes."""
    parsed_axes = tuple(parse_imu_axis(axis) for axis in axes)
    if not parsed_axes:
        raise ValueError("at least one IMU axis is required")
    if len(set(parsed_axes)) != len(parsed_axes):
        raise ValueError("IMU axes must be unique")
    return parsed_axes


def imu_axes_mask(axes: Sequence[str]) -> int:
    """Return the COMM_GET_IMU_DATA mask bits for ordered bench axes."""
    mask = 0
    for axis in parse_imu_axes(axes):
        mask |= 1 << IMU_FIELDS.index(axis)
    return mask


def imu_axis_unit(axis: str) -> str:
    """Return the display unit for one bench axis."""
    parsed_axis = parse_imu_axis(axis)
    return _IMU_AXIS_UNITS[parsed_axis]


def imu_axis_display_value(axis: str, wire_value: float) -> float:
    """Convert one wire value into the bench display unit."""
    parsed_axis = parse_imu_axis(axis)
    if parsed_axis in ("roll", "pitch", "yaw"):
        return wire_value * 180.0 / math.pi
    return wire_value


def _field_value_index(mask: int, field_name: str) -> int | None:
    try:
        field_index = IMU_FIELDS.index(field_name)
    except ValueError:
        return None

    field_bit = 1 << field_index
    if not mask & field_bit:
        return None
    preceding_fields = mask & (field_bit - 1)
    return preceding_fields.bit_count()


def imu_values_from_payload(payload: bytes, axes: Sequence[str]) -> dict[str, float]:
    """Parse selected IMU axis values from a COMM_GET_IMU_DATA response payload."""
    return imu_data_from_payload(payload).axis_values(axes)


def imu_data_from_payload(payload: bytes) -> ImuData:
    """Parse a COMM_GET_IMU_DATA response payload into a full IMU dataclass."""
    values: dict[str, float | int] = {}

    buf = VescBuffer(payload)
    cmd = buf.pop_uint8()
    if cmd != CommPacketId.COMM_GET_IMU_DATA:
        raise ValueError(f"unexpected command id {cmd}")

    response_mask = buf.pop_uint16()
    values["response_mask"] = response_mask

    for index, field_name in enumerate(IMU_FIELDS):
        if not response_mask & (1 << index):
            continue
        wire_value = buf.pop_double32_auto()
        values[field_name] = (
            imu_axis_display_value(field_name, wire_value)
            if field_name in IMU_BENCH_FIELDS
            else wire_value
        )

    if buf.remaining >= 1:
        values["vesc_id"] = buf.pop_uint8()

    return ImuData(
        response_mask=int(values.get("response_mask", 0)),
        roll=float(values.get("roll", 0.0)),
        pitch=float(values.get("pitch", 0.0)),
        yaw=float(values.get("yaw", 0.0)),
        acc_x=float(values.get("acc_x", 0.0)),
        acc_y=float(values.get("acc_y", 0.0)),
        acc_z=float(values.get("acc_z", 0.0)),
        gyro_x=float(values.get("gyro_x", 0.0)),
        gyro_y=float(values.get("gyro_y", 0.0)),
        gyro_z=float(values.get("gyro_z", 0.0)),
        mag_x=float(values.get("mag_x", 0.0)),
        mag_y=float(values.get("mag_y", 0.0)),
        mag_z=float(values.get("mag_z", 0.0)),
        q0=float(values.get("q0", 1.0)),
        q1=float(values.get("q1", 0.0)),
        q2=float(values.get("q2", 0.0)),
        q3=float(values.get("q3", 0.0)),
        vesc_id=int(values.get("vesc_id", 0)),
    )


def _read_exact(serial_port: BlockingIo, size: int, deadline_ns: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return None

        serial_port.timeout = remaining_ns / NSEC_PER_SEC
        chunk = serial_port.read(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)

    return bytes(chunks)


def _read_packet(
    serial_port: BlockingIo,
    deadline_ns: int,
    stats: ImuSourceStats,
) -> bytes | None:
    while True:
        start = _read_exact(serial_port, 1, deadline_ns)
        if start is None:
            return None
        start_byte = start[0]
        if start_byte in (2, 3, 4):
            break
        stats.discarded_bytes += 1

    if start_byte == 2:
        length_raw = _read_exact(serial_port, 1, deadline_ns)
        if length_raw is None:
            return None
        payload_len = length_raw[0]
        if payload_len < 1:
            stats.invalid_length += 1
            return None
    elif start_byte == 3:
        length_raw = _read_exact(serial_port, 2, deadline_ns)
        if length_raw is None:
            return None
        payload_len = (length_raw[0] << 8) | length_raw[1]
        if payload_len < 255:
            stats.invalid_length += 1
            return None
    else:
        length_raw = _read_exact(serial_port, 3, deadline_ns)
        if length_raw is None:
            return None
        payload_len = (length_raw[0] << 16) | (length_raw[1] << 8) | length_raw[2]
        if payload_len < 65535:
            stats.invalid_length += 1
            return None

    if payload_len > MAX_PACKET_LEN:
        stats.invalid_length += 1
        return None

    body = _read_exact(serial_port, payload_len + 3, deadline_ns)
    if body is None:
        return None
    if body[-1] != 3:
        stats.bad_stop += 1
        return None

    payload = body[:payload_len]
    received_crc = (body[payload_len] << 8) | body[payload_len + 1]
    if crc16(payload) != received_crc:
        stats.bad_crc += 1
        return None

    return payload


def _read_expected_imu_packet(
    serial_port: BlockingIo,
    packet_timeout: float,
    stats: ImuSourceStats,
) -> bytes | None:
    deadline_ns = time.perf_counter_ns() + round(packet_timeout * NSEC_PER_SEC)
    while True:
        payload = _read_packet(serial_port, deadline_ns, stats)
        if payload is None:
            return None
        if payload and payload[0] == CommPacketId.COMM_GET_IMU_DATA:
            return payload
        stats.unexpected_packets += 1


def _wait_until_ns(deadline_ns: int, stop: threading.Event) -> bool:
    while not stop.is_set():
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return True
        if remaining_ns > _RATE_LIMIT_SLEEP_SLACK_NS:
            sleep_s = (remaining_ns - _RATE_LIMIT_SLEEP_SLACK_NS) / NSEC_PER_SEC
            time.sleep(min(sleep_s, 0.01))
    return False


def _next_scheduled_request(
    previous_scheduled_request_ns: int | None,
    *,
    start_ns: int,
    poll_interval_ns: int,
    now_ns: int,
) -> tuple[int, int]:
    if previous_scheduled_request_ns is None:
        return start_ns, 0

    scheduled_request_ns = previous_scheduled_request_ns + poll_interval_ns
    missed_slots = 0
    if scheduled_request_ns < now_ns:
        missed_slots = (now_ns - scheduled_request_ns) // poll_interval_ns + 1
        scheduled_request_ns += missed_slots * poll_interval_ns
    return scheduled_request_ns, missed_slots


class VescImuSignalSource:
    """SignalSource implementation backed by direct VESC transport IMU polling."""

    def __init__(
        self,
        *,
        connection: VescConnection,
        axis: str,
        timeout: float,
        pending_samples: int,
        poll_rate_hz: float | None = None,
        can_id: int | None = None,
    ) -> None:
        if timeout <= 0.0:
            raise ValueError("timeout must be greater than 0")
        if poll_rate_hz is not None and poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be greater than 0")
        if can_id is not None and not 0 <= can_id <= 253:
            raise ValueError("can_id must be in range 0..253")

        self._connection = connection
        self._axis = parse_imu_axis(axis)
        self._timeout = timeout
        self._poll_rate_hz = poll_rate_hz
        self._poll_interval_ns = (
            None if poll_rate_hz is None else max(1, round(NSEC_PER_SEC / poll_rate_hz))
        )
        self._can_id = can_id
        self._mask = imu_axes_mask((self._axis,))
        self._request = build_imu_request(self._mask, can_id=can_id)
        self._samples = PendingSignalBuffer(pending_samples)
        self._response_latencies = PendingSignalBuffer(pending_samples)
        self._request_latenesses = PendingSignalBuffer(pending_samples)
        self._missed_slots = PendingSignalBuffer(pending_samples)
        self._sample_queue: queue.SimpleQueue[_QueuedImuSample] = queue.SimpleQueue()
        self._stats = ImuSourceStats()
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._publish_thread: threading.Thread | None = None
        self._serial_port: BlockingIo | None = None
        self._start_ns = 0
        self._last_error: str | None = None

    @property
    def channel_name(self) -> str:
        return self._axis

    @property
    def unit(self) -> str:
        return imu_axis_unit(self._axis)

    @property
    def response_latency_channel_name(self) -> str:
        return "response_latency"

    @property
    def response_latency_unit(self) -> str:
        return "s"

    @property
    def request_lateness_channel_name(self) -> str:
        return "request_lateness"

    @property
    def request_lateness_unit(self) -> str:
        return "s"

    @property
    def missed_slots_channel_name(self) -> str:
        return "missed_slots"

    @property
    def missed_slots_unit(self) -> str:
        return "slots"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._done.clear()
        self._publish_thread = threading.Thread(
            target=self._publish_samples,
            name="vesc-imu-signal-publisher",
            daemon=True,
        )
        self._publish_thread.start()
        self._thread = threading.Thread(
            target=self._run,
            name="vesc-imu-signal-source",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=timeout)
        if self._serial_port is not None:
            self._serial_port.close()
            self._serial_port = None

    def drain(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], SignalSourceStats]:
        timestamps, values, _latencies, _latenesses, _missed, stats = (
            self.drain_with_timing()
        )
        return timestamps, values, stats

    def drain_with_response_latency(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        SignalSourceStats,
    ]:
        """Drain IMU samples with request-to-response latency for each sample."""
        timestamps, values, latencies, _latenesses, _missed, stats = (
            self.drain_with_timing()
        )
        return timestamps, values, latencies, stats

    def drain_with_timing(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        SignalSourceStats,
    ]:
        """Drain IMU samples with response latency, request lateness, and skips."""
        with self._publish_lock:
            timestamps, values, _dropped_interval, buf_stats = self._samples.drain()
            _latency_timestamps, latencies, _latency_dropped, _latency_stats = (
                self._response_latencies.drain()
            )
            _lateness_timestamps, latenesses, _lateness_dropped, _lateness_stats = (
                self._request_latenesses.drain()
            )
            _missed_timestamps, missed_slots, _missed_dropped, _missed_stats = (
                self._missed_slots.drain()
            )
        with self._lock:
            stats = SignalSourceStats(
                samples=buf_stats.sample_count,
                dropped=buf_stats.cumulative_dropped,
                errors=self._stats.errors,
                average_rate_hz=buf_stats.average_rate_hz,
                latest_sample_s=buf_stats.latest_sample_s,
                latest_value=buf_stats.latest_value,
                last_error=self._last_error,
                done=self._done.is_set(),
            )
        return timestamps, values, latencies, latenesses, missed_slots, stats

    def source_stats(self) -> SignalSourceStats:
        buf_stats = self._samples.pending_stats()
        with self._lock:
            return SignalSourceStats(
                samples=buf_stats.sample_count,
                dropped=buf_stats.cumulative_dropped,
                errors=self._stats.errors,
                average_rate_hz=buf_stats.average_rate_hz,
                latest_sample_s=buf_stats.latest_sample_s,
                latest_value=buf_stats.latest_value,
                last_error=self._last_error,
                done=self._done.is_set(),
            )

    def response_latency_stats(self) -> SignalSourceStats:
        buf_stats = self._response_latencies.pending_stats()
        with self._lock:
            return SignalSourceStats(
                samples=buf_stats.sample_count,
                dropped=buf_stats.cumulative_dropped,
                errors=self._stats.errors,
                average_rate_hz=buf_stats.average_rate_hz,
                latest_sample_s=buf_stats.latest_sample_s,
                latest_value=buf_stats.latest_value,
                last_error=self._last_error,
                done=self._done.is_set(),
            )

    def request_lateness_stats(self) -> SignalSourceStats:
        buf_stats = self._request_latenesses.pending_stats()
        with self._lock:
            return SignalSourceStats(
                samples=buf_stats.sample_count,
                dropped=buf_stats.cumulative_dropped,
                errors=self._stats.errors,
                average_rate_hz=buf_stats.average_rate_hz,
                latest_sample_s=buf_stats.latest_sample_s,
                latest_value=buf_stats.latest_value,
                last_error=self._last_error,
                done=self._done.is_set(),
            )

    def missed_slots_stats(self) -> SignalSourceStats:
        buf_stats = self._missed_slots.pending_stats()
        with self._lock:
            return SignalSourceStats(
                samples=buf_stats.sample_count,
                dropped=buf_stats.cumulative_dropped,
                errors=self._stats.errors,
                average_rate_hz=buf_stats.average_rate_hz,
                latest_sample_s=buf_stats.latest_sample_s,
                latest_value=buf_stats.latest_value,
                last_error=self._last_error,
                done=self._done.is_set(),
            )

    def _value_from_payload(self, payload: bytes) -> float:
        return imu_values_from_payload(payload, (self._axis,))[self._axis]

    def _enqueue_sample(
        self,
        *,
        timestamp_s: float,
        value: float,
        response_latency_s: float,
        request_lateness_s: float,
        missed_slots: int,
    ) -> None:
        self._sample_queue.put(
            _QueuedImuSample(
                timestamp_s=timestamp_s,
                value=value,
                response_latency_s=response_latency_s,
                request_lateness_s=request_lateness_s,
                missed_slots=float(missed_slots),
            )
        )

    def _flush_sample_queue(self, max_samples: int | None = None) -> int:
        timestamps: list[float] = []
        values: list[float] = []
        response_latencies: list[float] = []
        request_latenesses: list[float] = []
        missed_slots: list[float] = []
        while max_samples is None or len(timestamps) < max_samples:
            try:
                sample = self._sample_queue.get_nowait()
            except queue.Empty:
                break
            timestamps.append(sample.timestamp_s)
            values.append(sample.value)
            response_latencies.append(sample.response_latency_s)
            request_latenesses.append(sample.request_lateness_s)
            missed_slots.append(sample.missed_slots)

        if not timestamps:
            return 0

        with self._publish_lock:
            self._samples.append_many(timestamps, values)
            self._response_latencies.append_many(timestamps, response_latencies)
            self._request_latenesses.append_many(timestamps, request_latenesses)
            self._missed_slots.append_many(timestamps, missed_slots)
        return len(timestamps)

    def _publish_samples(self) -> None:
        while not self._done.is_set() or not self._sample_queue.empty():
            flushed = self._flush_sample_queue(_SAMPLE_PUBLISH_BATCH_SIZE)
            if flushed >= _SAMPLE_PUBLISH_BATCH_SIZE:
                continue
            if self._done.wait(_SAMPLE_PUBLISH_INTERVAL_NS / NSEC_PER_SEC):
                continue
            if self._stop.is_set() and self._sample_queue.empty():
                break
        self._flush_sample_queue()

    def _run(self) -> None:
        scheduled_request_ns: int | None = None
        missed_slots = 0
        try:
            self._serial_port = open_blocking_io(
                self._connection,
                timeout=self._timeout,
            )
            self._start_ns = time.perf_counter_ns()
            while not self._stop.is_set():
                serial_port = self._serial_port
                if serial_port is None:
                    break

                if self._poll_interval_ns is not None:
                    scheduled_request_ns, missed_slots = _next_scheduled_request(
                        scheduled_request_ns,
                        start_ns=self._start_ns,
                        poll_interval_ns=self._poll_interval_ns,
                        now_ns=time.perf_counter_ns(),
                    )
                    if not _wait_until_ns(scheduled_request_ns, self._stop):
                        break

                request_ns = time.perf_counter_ns()
                if scheduled_request_ns is None:
                    scheduled_request_ns = request_ns
                    missed_slots = 0
                request_lateness_s = (
                    request_ns - scheduled_request_ns
                ) / NSEC_PER_SEC
                serial_port.write(self._request)

                payload = _read_expected_imu_packet(
                    serial_port,
                    self._timeout,
                    self._stats,
                )
                now_ns = time.perf_counter_ns()
                response_latency_s = (now_ns - request_ns) / NSEC_PER_SEC
                if payload is None:
                    self._stats.timeouts += 1
                    serial_port.reset_input_buffer()
                    continue

                try:
                    value = self._value_from_payload(payload)
                except ValueError as exc:
                    self._stats.parse_errors += 1
                    with self._lock:
                        self._last_error = str(exc)
                    continue

                sample_s = (scheduled_request_ns - self._start_ns) / NSEC_PER_SEC
                self._enqueue_sample(
                    timestamp_s=sample_s,
                    value=value,
                    response_latency_s=response_latency_s,
                    request_lateness_s=request_lateness_s,
                    missed_slots=missed_slots,
                )
                with self._lock:
                    self._last_error = None
        except Exception as exc:  # noqa: BLE001 - source errors are surfaced in status.
            with self._lock:
                self._stats.parse_errors += 1
                self._last_error = str(exc)
        finally:
            if self._serial_port is not None:
                self._serial_port.close()
                self._serial_port = None
            self._done.set()
            if self._publish_thread is None or not self._publish_thread.is_alive():
                self._flush_sample_queue()


__all__ = [
    "DEFAULT_BAUDRATE",
    "DEFAULT_TIMEOUT",
    "IMU_BENCH_FIELDS",
    "ImuData",
    "VescImuSignalSource",
    "imu_data_from_payload",
    "imu_axes_mask",
    "imu_axis_display_value",
    "imu_axis_mask",
    "imu_axis_unit",
    "imu_values_from_payload",
    "parse_imu_axis",
    "parse_imu_axes",
]
