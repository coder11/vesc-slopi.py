"""Fast direct IMU signal source for the live signal bench."""

from __future__ import annotations

import math
import threading
import time
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
    SignalSourceSnapshot,
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
_RATE_LIMIT_SLEEP_SLACK_NS = 200_000


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


@dataclass(slots=True)
class _DurationStat:
    count: int = 0
    total_ns: int = 0
    max_ns: int = 0
    last_ns: int = 0

    def add(self, duration_ns: int) -> None:
        self.count += 1
        self.total_ns += duration_ns
        self.max_ns = max(self.max_ns, duration_ns)
        self.last_ns = duration_ns

    @property
    def average_ns(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ns / self.count


@dataclass(slots=True)
class _SourceProfile:
    write_ns: _DurationStat
    read_ns: _DurationStat
    parse_ns: _DurationStat
    publish_ns: _DurationStat
    flush_ns: _DurationStat
    loop_ns: _DurationStat
    state_lock_wait_ns: _DurationStat
    state_lock_hold_ns: _DurationStat


def _new_source_profile() -> _SourceProfile:
    return _SourceProfile(
        write_ns=_DurationStat(),
        read_ns=_DurationStat(),
        parse_ns=_DurationStat(),
        publish_ns=_DurationStat(),
        flush_ns=_DurationStat(),
        loop_ns=_DurationStat(),
        state_lock_wait_ns=_DurationStat(),
        state_lock_hold_ns=_DurationStat(),
    )


def _ns_to_ms(duration_ns: float) -> float:
    return duration_ns / 1_000_000.0


def _avg_ms(stat: _DurationStat) -> float:
    return _ns_to_ms(stat.average_ns)


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
            None
            if poll_rate_hz is None
            else max(1, round(NSEC_PER_SEC / poll_rate_hz))
        )
        self._can_id = can_id
        self._mask = imu_axis_mask(self._axis)
        self._request = build_imu_request(self._mask, can_id=can_id)
        self._samples = PendingSignalBuffer(pending_samples)
        self._stats = ImuSourceStats()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial_port: BlockingIo | None = None
        self._start_ns = 0
        self._sample_count = 0
        self._dropped = 0
        self._latest_sample_s: float | None = None
        self._latest_value: float | None = None
        self._last_error: str | None = None
        self._profile = _new_source_profile()

    @property
    def channel_name(self) -> str:
        return self._axis

    @property
    def unit(self) -> str:
        return imu_axis_unit(self._axis)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._done.clear()
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
        if self._serial_port is not None:
            self._serial_port.close()
            self._serial_port = None

    def drain(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
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
            debug_text = self._debug_text_locked()
            return SignalSourceSnapshot(
                samples=self._sample_count,
                dropped=self._dropped,
                errors=self._stats.errors,
                average_rate_hz=average_rate,
                latest_sample_s=self._latest_sample_s,
                latest_value=self._latest_value,
                last_error=self._last_error,
                done=self._done.is_set(),
                debug_text=debug_text,
            )

    def _debug_text_locked(self) -> str:
        buffer_stats = self._samples.stats_snapshot()
        append_wait_ms = (
            _ns_to_ms(buffer_stats.append_lock_wait_ns_total / buffer_stats.append_calls)
            if buffer_stats.append_calls
            else 0.0
        )
        append_hold_ms = (
            _ns_to_ms(buffer_stats.append_lock_hold_ns_total / buffer_stats.append_calls)
            if buffer_stats.append_calls
            else 0.0
        )
        drain_wait_ms = (
            _ns_to_ms(buffer_stats.drain_lock_wait_ns_total / buffer_stats.drain_calls)
            if buffer_stats.drain_calls
            else 0.0
        )
        drain_hold_ms = (
            _ns_to_ms(buffer_stats.drain_lock_hold_ns_total / buffer_stats.drain_calls)
            if buffer_stats.drain_calls
            else 0.0
        )
        return (
            "srcdbg "
            f"loop={_avg_ms(self._profile.loop_ns):.3f}ms "
            f"rd={_avg_ms(self._profile.read_ns):.3f} "
            f"wr={_avg_ms(self._profile.write_ns):.3f} "
            f"parse={_avg_ms(self._profile.parse_ns):.3f} "
            f"pub={_avg_ms(self._profile.publish_ns):.3f} "
            f"flush={_avg_ms(self._profile.flush_ns):.3f} "
            f"state_lock={_avg_ms(self._profile.state_lock_wait_ns):.3f}/{_avg_ms(self._profile.state_lock_hold_ns):.3f} "
            f"buf_wait a/d={append_wait_ms:.3f}/{drain_wait_ms:.3f} "
            f"buf_hold a/d={append_hold_ms:.3f}/{drain_hold_ms:.3f} "
            f"buf_hi={buffer_stats.high_watermark}"
        )

    def _value_from_payload(self, payload: bytes) -> float:
        buf = VescBuffer(payload)
        cmd = buf.pop_uint8()
        if cmd != CommPacketId.COMM_GET_IMU_DATA:
            raise ValueError(f"unexpected command id {cmd}")

        response_mask = buf.pop_uint16()
        value_index = _field_value_index(response_mask, self._axis)
        if value_index is None:
            raise ValueError(
                f"IMU response mask 0x{response_mask:04x} does not include {self._axis}"
            )

        wire_value = 0.0
        for index in range(len(IMU_FIELDS)):
            if response_mask & (1 << index):
                value = buf.pop_double32_auto()
                if value_index == 0:
                    wire_value = value
                    break
                value_index -= 1
        return imu_axis_display_value(self._axis, wire_value)

    def _run(self) -> None:
        self._start_ns = time.perf_counter_ns()
        pending_timestamps: list[float] = []
        pending_values: list[float] = []
        next_flush_ns = self._start_ns + 5_000_000
        previous_request_ns: int | None = None
        try:
            self._serial_port = open_blocking_io(
                self._connection,
                timeout=self._timeout,
            )
            while not self._stop.is_set():
                loop_start_ns = time.perf_counter_ns()
                serial_port = self._serial_port
                if serial_port is None:
                    break

                if (
                    self._poll_interval_ns is not None
                    and previous_request_ns is not None
                ):
                    next_request_ns = previous_request_ns + self._poll_interval_ns
                    if not _wait_until_ns(next_request_ns, self._stop):
                        break

                previous_request_ns = time.perf_counter_ns()
                write_start_ns = previous_request_ns
                serial_port.write(self._request)
                after_write_ns = time.perf_counter_ns()
                self._profile.write_ns.add(after_write_ns - write_start_ns)

                read_start_ns = after_write_ns
                payload = _read_expected_imu_packet(
                    serial_port,
                    self._timeout,
                    self._stats,
                )
                now_ns = time.perf_counter_ns()
                self._profile.read_ns.add(now_ns - read_start_ns)
                if payload is None:
                    self._stats.timeouts += 1
                    serial_port.reset_input_buffer()
                    self._profile.loop_ns.add(time.perf_counter_ns() - loop_start_ns)
                    continue

                try:
                    parse_start_ns = time.perf_counter_ns()
                    value = self._value_from_payload(payload)
                    self._profile.parse_ns.add(time.perf_counter_ns() - parse_start_ns)
                except ValueError as exc:
                    self._stats.parse_errors += 1
                    with self._lock:
                        self._last_error = str(exc)
                    self._profile.loop_ns.add(time.perf_counter_ns() - loop_start_ns)
                    continue

                sample_s = (now_ns - self._start_ns) / NSEC_PER_SEC
                pending_timestamps.append(sample_s)
                pending_values.append(value)
                state_lock_wait_start_ns = time.perf_counter_ns()
                with self._lock:
                    state_lock_acquired_ns = time.perf_counter_ns()
                    self._profile.state_lock_wait_ns.add(
                        state_lock_acquired_ns - state_lock_wait_start_ns
                    )
                    self._sample_count += 1
                    self._latest_sample_s = sample_s
                    self._latest_value = value
                    self._last_error = None
                    state_lock_hold_ns = time.perf_counter_ns() - state_lock_acquired_ns
                    self._profile.state_lock_hold_ns.add(state_lock_hold_ns)

                publish_end_ns = time.perf_counter_ns()
                self._profile.publish_ns.add(publish_end_ns - now_ns)

                if len(pending_timestamps) >= 64 or now_ns >= next_flush_ns:
                    flush_start_ns = time.perf_counter_ns()
                    self._samples.append_many(pending_timestamps, pending_values)
                    flush_end_ns = time.perf_counter_ns()
                    self._profile.flush_ns.add(flush_end_ns - flush_start_ns)
                    pending_timestamps.clear()
                    pending_values.clear()
                    next_flush_ns = now_ns + 5_000_000
                self._profile.loop_ns.add(time.perf_counter_ns() - loop_start_ns)
        except Exception as exc:  # noqa: BLE001 - source errors are surfaced in status.
            with self._lock:
                self._stats.parse_errors += 1
                self._last_error = str(exc)
        finally:
            if pending_timestamps:
                self._samples.append_many(pending_timestamps, pending_values)
            if self._serial_port is not None:
                self._serial_port.close()
                self._serial_port = None
            self._done.set()


__all__ = [
    "DEFAULT_BAUDRATE",
    "DEFAULT_TIMEOUT",
    "IMU_BENCH_FIELDS",
    "VescImuSignalSource",
    "imu_axis_display_value",
    "imu_axis_mask",
    "imu_axis_unit",
    "parse_imu_axis",
]
