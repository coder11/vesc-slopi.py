"""Receiver for the IMU Streamer VESC package."""

from __future__ import annotations

# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=broad-exception-caught,duplicate-code

import struct
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from vesc_py.comm_ids import CommPacketId
from vesc_py.connection import BlockingIo, VescConnection, open_blocking_io
from vesc_py.live_signal import NSEC_PER_SEC
from vesc_py.packet import PacketDecoder, encode_packet

DEFAULT_TIMEOUT = 0.1
DEFAULT_START_TIMEOUT = 0.5
DEFAULT_PENDING_SAMPLES = 8192
DEFAULT_READ_CHUNK_SIZE = 4096
DEFAULT_FLUSH_SAMPLES = 128
DEFAULT_FLUSH_INTERVAL_S = 0.005

IMU_STREAMER_PROTOCOL_VERSION = 1

IMU_STREAMER_CMD_START = 0x01
IMU_STREAMER_CMD_STOP = 0x02
IMU_STREAMER_CMD_STATUS = 0x03

IMU_STREAMER_MSG_SAMPLE = 0x10
IMU_STREAMER_MSG_ACK = 0x11
IMU_STREAMER_MSG_STATUS = 0x12

IMU_STREAMER_ACK_OK = 0
IMU_STREAMER_ACK_UNKNOWN_CMD = 1
IMU_STREAMER_ACK_UNSUPPORTED = 2

IMU_STREAMER_CHANNELS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
IMU_STREAMER_CHANNEL_COUNT = len(IMU_STREAMER_CHANNELS)

FloatArray: TypeAlias = npt.NDArray[np.float64]

_SAMPLE_STRUCT = struct.Struct(">BIQ9f")
_ACK_STRUCT = struct.Struct(">BBBB")
_STATUS_STRUCT = struct.Struct(">BBBI")
_UINT32_MODULO = 1 << 32
_UINT32_HALF_RANGE = 1 << 31


@dataclass(frozen=True, slots=True)
class ImuStreamerSample:
    """One raw sample from the IMU Streamer package."""

    sequence: int
    timestamp_us: int
    gyro_x: float
    gyro_y: float
    gyro_z: float
    acc_x: float
    acc_y: float
    acc_z: float
    mag_x: float
    mag_y: float
    mag_z: float

    @property
    def timestamp_s(self) -> float:
        """Return the firmware-side sample timestamp in seconds."""
        return self.timestamp_us / 1_000_000.0

    def channel_values(self) -> tuple[float, float, float, float, float, float]:
        """Return values in ``IMU_STREAMER_CHANNELS`` order."""
        return (
            self.acc_x,
            self.acc_y,
            self.acc_z,
            self.gyro_x,
            self.gyro_y,
            self.gyro_z,
        )


@dataclass(frozen=True, slots=True)
class ImuStreamerAck:
    """Command acknowledgement from the IMU Streamer package."""

    protocol_version: int
    command: int
    result: int

    @property
    def ok(self) -> bool:
        """Return whether the command succeeded."""
        return (
            self.protocol_version == IMU_STREAMER_PROTOCOL_VERSION
            and self.result == IMU_STREAMER_ACK_OK
        )


@dataclass(frozen=True, slots=True)
class ImuStreamerStatus:
    """Status response from the IMU Streamer package."""

    protocol_version: int
    streaming: bool
    next_sequence: int


@dataclass(frozen=True, slots=True)
class ImuStreamerLispPrint:
    """One ``COMM_LISP_PRINT`` message emitted by the package."""

    text: str


ImuStreamerMessage: TypeAlias = (
    ImuStreamerSample | ImuStreamerAck | ImuStreamerStatus | ImuStreamerLispPrint
)


@dataclass(slots=True)
class ImuStreamerReaderStats:
    """Counters maintained by the package-stream receiver."""

    samples: int = 0
    timeouts: int = 0
    parse_errors: int = 0
    unexpected_packets: int = 0
    acknowledgements: int = 0
    statuses: int = 0
    lisp_prints: int = 0
    idle_reads: int = 0
    sequence_drops: int = 0
    out_of_order: int = 0

    @property
    def errors(self) -> int:
        """Return the aggregate error count used in UI status."""
        return (
            self.timeouts + self.parse_errors + self.unexpected_packets + self.out_of_order
        )


@dataclass(frozen=True, slots=True)
class ImuStreamBufferStats:
    """Stats derived from pending sample timestamps."""

    cumulative_dropped: int
    sample_count: int
    average_rate_hz: float
    latest_sample_s: float | None
    latest_values: FloatArray | None


@dataclass(frozen=True, slots=True)
class ImuStreamerSourceStats:
    """Low-rate status for a running IMU streamer receiver."""

    samples: int
    dropped: int
    errors: int
    average_rate_hz: float
    latest_sample_s: float | None
    latest_values: FloatArray | None
    last_error: str | None
    done: bool
    sequence_drops: int
    timeouts: int
    idle_reads: int
    lisp_prints: int
    last_print: str | None


def build_imu_streamer_command(command: int) -> bytes:
    """Build a framed command for the IMU Streamer package."""
    if not 0 <= command <= 0xFF:
        raise ValueError("command must fit in one byte")
    payload = bytes([CommPacketId.COMM_CUSTOM_APP_DATA, command])
    return encode_packet(payload)


def parse_imu_streamer_payload(payload: bytes) -> ImuStreamerMessage:
    """Parse the payload following ``COMM_CUSTOM_APP_DATA``."""
    if not payload:
        raise ValueError("empty IMU streamer payload")

    message_id = payload[0]
    if message_id == IMU_STREAMER_MSG_SAMPLE:
        if len(payload) != _SAMPLE_STRUCT.size:
            raise ValueError(
                "invalid IMU streamer sample length "
                f"{len(payload)} != {_SAMPLE_STRUCT.size}"
            )
        unpacked = _SAMPLE_STRUCT.unpack(payload)
        floats = tuple(float(value) for value in unpacked[3:])
        return ImuStreamerSample(
            sequence=int(unpacked[1]),
            timestamp_us=int(unpacked[2]),
            gyro_x=floats[0],
            gyro_y=floats[1],
            gyro_z=floats[2],
            acc_x=floats[3],
            acc_y=floats[4],
            acc_z=floats[5],
            mag_x=floats[6],
            mag_y=floats[7],
            mag_z=floats[8],
        )

    if message_id == IMU_STREAMER_MSG_ACK:
        if len(payload) != _ACK_STRUCT.size:
            raise ValueError(
                f"invalid IMU streamer ack length {len(payload)} != {_ACK_STRUCT.size}"
            )
        _msg, version, command, result = _ACK_STRUCT.unpack(payload)
        return ImuStreamerAck(
            protocol_version=int(version),
            command=int(command),
            result=int(result),
        )

    if message_id == IMU_STREAMER_MSG_STATUS:
        if len(payload) != _STATUS_STRUCT.size:
            raise ValueError(
                "invalid IMU streamer status length "
                f"{len(payload)} != {_STATUS_STRUCT.size}"
            )
        _msg, version, streaming, next_sequence = _STATUS_STRUCT.unpack(payload)
        return ImuStreamerStatus(
            protocol_version=int(version),
            streaming=bool(streaming),
            next_sequence=int(next_sequence),
        )

    raise ValueError(f"unknown IMU streamer message id 0x{message_id:02x}")


def parse_imu_streamer_packet(payload: bytes) -> ImuStreamerMessage:
    """Parse a full VESC packet payload carrying streamer custom app data."""
    if not payload:
        raise ValueError("empty VESC packet payload")
    if payload[0] == CommPacketId.COMM_LISP_PRINT:
        return ImuStreamerLispPrint(text=payload[1:].decode("latin1"))
    if payload[0] != CommPacketId.COMM_CUSTOM_APP_DATA:
        raise ValueError(f"unexpected command id {payload[0]}")
    return parse_imu_streamer_payload(payload[1:])


def _buffer_stats_from_arrays(
    timestamps: FloatArray,
    values: FloatArray,
    *,
    cumulative_dropped: int,
) -> ImuStreamBufferStats:
    sample_count = int(timestamps.size)
    if sample_count == 0:
        return ImuStreamBufferStats(cumulative_dropped, 0, 0.0, None, None)
    if values.shape != (sample_count, IMU_STREAMER_CHANNEL_COUNT):
        raise ValueError("timestamp and value counts must match")
    if sample_count >= 2:
        elapsed = float(timestamps[-1] - timestamps[0])
        rate = (sample_count - 1) / elapsed if elapsed > 0.0 else 0.0
    else:
        rate = 0.0
    return ImuStreamBufferStats(
        cumulative_dropped=cumulative_dropped,
        sample_count=sample_count,
        average_rate_hz=rate,
        latest_sample_s=float(timestamps[-1]),
        latest_values=values[-1].copy(),
    )


def _source_stats_from_pending(
    pending: ImuStreamBufferStats,
    *,
    reader_stats: ImuStreamerReaderStats,
    last_error: str | None,
    last_print: str | None,
    done: bool,
) -> ImuStreamerSourceStats:
    return ImuStreamerSourceStats(
        samples=pending.sample_count,
        dropped=pending.cumulative_dropped,
        errors=reader_stats.errors,
        average_rate_hz=pending.average_rate_hz,
        latest_sample_s=pending.latest_sample_s,
        latest_values=pending.latest_values,
        last_error=last_error,
        done=done,
        sequence_drops=reader_stats.sequence_drops,
        timeouts=reader_stats.timeouts,
        idle_reads=reader_stats.idle_reads,
        lisp_prints=reader_stats.lisp_prints,
        last_print=last_print,
    )


class PendingImuStreamBuffer:
    """Thread-safe overwrite ring for pending multi-channel IMU samples."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self._capacity = capacity
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._values = np.zeros(
            (capacity, IMU_STREAMER_CHANNEL_COUNT),
            dtype=np.float64,
        )
        self._lock = threading.Lock()
        self._read_index = 0
        self._write_index = 0
        self._count = 0
        self._dropped = 0
        self._cumulative_dropped = 0

    def append_many(
        self,
        timestamps: Sequence[float] | FloatArray,
        values: Sequence[Sequence[float]] | FloatArray,
    ) -> None:
        """Append a batch while taking the shared lock once."""
        timestamp_values = np.asarray(timestamps, dtype=np.float64)
        sample_values = np.asarray(values, dtype=np.float64)
        count = int(timestamp_values.size)
        if count == 0:
            return
        if sample_values.shape != (count, IMU_STREAMER_CHANNEL_COUNT):
            raise ValueError(
                "values must have shape "
                f"({count}, {IMU_STREAMER_CHANNEL_COUNT})"
            )

        with self._lock:
            if count >= self._capacity:
                drop = self._count + count - self._capacity
                self._dropped += drop
                self._cumulative_dropped += drop
                self._timestamps[:] = timestamp_values[-self._capacity :]
                self._values[:, :] = sample_values[-self._capacity :, :]
                self._read_index = 0
                self._write_index = 0
                self._count = self._capacity
                return

            overflow = max(0, self._count + count - self._capacity)
            if overflow > 0:
                self._read_index = (self._read_index + overflow) % self._capacity
                self._dropped += overflow
                self._cumulative_dropped += overflow
            self._count = min(self._capacity, self._count + count)

            first_count = min(count, self._capacity - self._write_index)
            write_end = self._write_index + first_count
            self._timestamps[self._write_index : write_end] = (
                timestamp_values[:first_count]
            )
            self._values[self._write_index : write_end, :] = (
                sample_values[:first_count, :]
            )

            remaining = count - first_count
            if remaining > 0:
                self._timestamps[:remaining] = timestamp_values[first_count:]
                self._values[:remaining, :] = sample_values[first_count:, :]

            self._write_index = (self._write_index + count) % self._capacity

    def pending_stats(self) -> ImuStreamBufferStats:
        """Return stats for currently pending samples without consuming them."""
        with self._lock:
            timestamps, values = self._pending_arrays_locked()
            return _buffer_stats_from_arrays(
                timestamps,
                values,
                cumulative_dropped=self._cumulative_dropped,
            )

    def drain(self) -> tuple[FloatArray, FloatArray, int, ImuStreamBufferStats]:
        """Return pending timestamps and ``(sample, channel)`` values in order."""
        with self._lock:
            count = self._count
            dropped_interval = self._dropped
            self._dropped = 0
            if count == 0:
                timestamps = np.empty(0, dtype=np.float64)
                values = np.empty(
                    (0, IMU_STREAMER_CHANNEL_COUNT),
                    dtype=np.float64,
                )
                stats = _buffer_stats_from_arrays(
                    timestamps,
                    values,
                    cumulative_dropped=self._cumulative_dropped,
                )
                return timestamps, values, dropped_interval, stats

            timestamps, values = self._pending_arrays_locked()
            self._read_index = self._write_index
            self._count = 0
            stats = _buffer_stats_from_arrays(
                timestamps,
                values,
                cumulative_dropped=self._cumulative_dropped,
            )
            return timestamps, values, dropped_interval, stats

    def _pending_arrays_locked(self) -> tuple[FloatArray, FloatArray]:
        count = self._count
        if count == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, IMU_STREAMER_CHANNEL_COUNT), dtype=np.float64),
            )
        read_index = self._read_index
        if read_index + count <= self._capacity:
            return (
                self._timestamps[read_index : read_index + count].copy(),
                self._values[read_index : read_index + count, :].copy(),
            )
        first_count = self._capacity - read_index
        return (
            np.concatenate(
                (
                    self._timestamps[read_index:],
                    self._timestamps[: count - first_count],
                )
            ),
            np.concatenate(
                (
                    self._values[read_index:, :],
                    self._values[: count - first_count, :],
                ),
                axis=0,
            ),
        )


class VescImuStreamer:
    """Threaded receiver for the IMU Streamer package."""

    def __init__(
        self,
        *,
        connection: VescConnection,
        timeout: float = DEFAULT_TIMEOUT,
        start_timeout: float | None = None,
        pending_samples: int = DEFAULT_PENDING_SAMPLES,
        read_chunk_size: int = DEFAULT_READ_CHUNK_SIZE,
        flush_samples: int = DEFAULT_FLUSH_SAMPLES,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
    ) -> None:
        if timeout <= 0.0:
            raise ValueError("timeout must be greater than 0")
        if start_timeout is not None and start_timeout <= 0.0:
            raise ValueError("start_timeout must be greater than 0")
        if read_chunk_size <= 0:
            raise ValueError("read_chunk_size must be greater than 0")
        if flush_samples <= 0:
            raise ValueError("flush_samples must be greater than 0")
        if flush_interval_s <= 0.0:
            raise ValueError("flush_interval_s must be greater than 0")

        self._connection = connection
        self._timeout = timeout
        self._start_timeout = timeout if start_timeout is None else start_timeout
        self._read_chunk_size = read_chunk_size
        self._flush_samples = flush_samples
        self._flush_interval_ns = round(flush_interval_s * NSEC_PER_SEC)
        self._samples = PendingImuStreamBuffer(pending_samples)
        self._stats = ImuStreamerReaderStats()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial_port: BlockingIo | None = None
        self._last_error: str | None = None
        self._last_print: str | None = None

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return value channel names in drained batch order."""
        return IMU_STREAMER_CHANNELS

    @property
    def last_error(self) -> str | None:
        """Return the latest reader error, if any."""
        with self._lock:
            return self._last_error

    @property
    def last_print(self) -> str | None:
        """Return the latest ``COMM_LISP_PRINT`` text, if any."""
        with self._lock:
            return self._last_print

    def start(self) -> None:
        """Open the VESC connection and start package streaming."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._done.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="vesc-imu-streamer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Stop package streaming and close the transport."""
        self._stop.set()
        serial_port = self._serial_port
        if serial_port is not None:
            try:
                serial_port.write(build_imu_streamer_command(IMU_STREAMER_CMD_STOP))
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._serial_port is not None:
            self._serial_port.close()
            self._serial_port = None

    def drain(self) -> tuple[FloatArray, FloatArray, ImuStreamerSourceStats]:
        """Drain pending samples as timestamps and ``(sample, channel)`` values."""
        timestamps, values, _dropped_interval, pending = self._samples.drain()
        with self._lock:
            return timestamps, values, _source_stats_from_pending(
                pending,
                reader_stats=self._stats,
                last_error=self._last_error,
                last_print=self._last_print,
                done=self._done.is_set(),
            )

    def source_stats(self) -> ImuStreamerSourceStats:
        """Return low-rate status without consuming pending samples."""
        pending = self._samples.pending_stats()
        with self._lock:
            return _source_stats_from_pending(
                pending,
                reader_stats=self._stats,
                last_error=self._last_error,
                last_print=self._last_print,
                done=self._done.is_set(),
            )

    def _set_last_error(self, message: str | None) -> None:
        with self._lock:
            self._last_error = message

    def _set_last_print(self, message: str | None) -> None:
        with self._lock:
            self._last_print = message

    def _decode_packet(self, payload: bytes) -> ImuStreamerMessage | None:
        if not payload:
            self._stats.unexpected_packets += 1
            return None
        if payload[0] == CommPacketId.COMM_LISP_PRINT:
            lisp_print = ImuStreamerLispPrint(text=payload[1:].decode("latin1"))
            self._stats.lisp_prints += 1
            self._set_last_print(lisp_print.text)
            return lisp_print
        if not payload or payload[0] != CommPacketId.COMM_CUSTOM_APP_DATA:
            self._stats.unexpected_packets += 1
            return None
        try:
            message = parse_imu_streamer_payload(payload[1:])
        except ValueError as exc:
            self._stats.parse_errors += 1
            self._set_last_error(str(exc))
            return None

        if isinstance(message, ImuStreamerAck):
            self._stats.acknowledgements += 1
        elif isinstance(message, ImuStreamerStatus):
            self._stats.statuses += 1
        self._set_last_error(None)
        return message

    def _read_ack(
        self,
        serial_port: BlockingIo,
        decoder: PacketDecoder,
        command: int,
        queued_messages: list[ImuStreamerMessage],
        *,
        timeout_s: float,
    ) -> ImuStreamerAck | None:
        deadline_ns = time.perf_counter_ns() + round(timeout_s * NSEC_PER_SEC)
        while not self._stop.is_set():
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns <= 0:
                return None
            serial_port.timeout = min(self._timeout, remaining_ns / NSEC_PER_SEC)
            chunk = serial_port.read(self._read_chunk_size)
            if not chunk:
                continue
            ack: ImuStreamerAck | None = None
            for payload in decoder.process(chunk):
                message = self._decode_packet(payload)
                if (
                    isinstance(message, ImuStreamerAck)
                    and message.command == command
                ):
                    ack = message
                elif message is not None:
                    queued_messages.append(message)
            if ack is not None:
                return ack
        return None

    def _update_sequence_stats(
        self,
        sample: ImuStreamerSample,
        expected_sequence: int | None,
    ) -> int:
        next_expected = (sample.sequence + 1) % _UINT32_MODULO
        if expected_sequence is None or sample.sequence == expected_sequence:
            return next_expected

        gap = (sample.sequence - expected_sequence) % _UINT32_MODULO
        if 0 < gap < _UINT32_HALF_RANGE:
            self._stats.sequence_drops += gap
        else:
            self._stats.out_of_order += 1
        return next_expected

    def _flush_pending(
        self,
        timestamps: list[float],
        values: list[tuple[float, float, float, float, float, float]],
    ) -> None:
        if not timestamps:
            return
        self._samples.append_many(timestamps, values)
        timestamps.clear()
        values.clear()

    def _append_sample(
        self,
        sample: ImuStreamerSample,
        expected_sequence: int | None,
        timestamps: list[float],
        values: list[tuple[float, float, float, float, float, float]],
    ) -> int:
        next_expected = self._update_sequence_stats(sample, expected_sequence)
        timestamps.append(sample.timestamp_s)
        values.append(sample.channel_values())
        self._stats.samples += 1
        return next_expected

    def _run(self) -> None:
        decoder = PacketDecoder()
        queued_messages: list[ImuStreamerMessage] = []
        pending_timestamps: list[float] = []
        pending_values: list[tuple[float, float, float, float, float, float]] = []
        next_flush_ns = time.perf_counter_ns() + self._flush_interval_ns
        expected_sequence: int | None = None

        try:
            serial_port = open_blocking_io(
                self._connection,
                timeout=self._timeout,
            )
            self._serial_port = serial_port
            serial_port.reset_input_buffer()
            serial_port.write(build_imu_streamer_command(IMU_STREAMER_CMD_START))

            ack = self._read_ack(
                serial_port,
                decoder,
                IMU_STREAMER_CMD_START,
                queued_messages,
                timeout_s=self._start_timeout,
            )
            if ack is None:
                self._stats.timeouts += 1
                self._set_last_error("timed out waiting for IMU streamer start ack")
                return
            if not ack.ok:
                self._set_last_error(_ack_result_message(ack))
                return
            self._set_last_error(None)

            for queued_message in queued_messages:
                if isinstance(queued_message, ImuStreamerSample):
                    expected_sequence = self._append_sample(
                        queued_message,
                        expected_sequence,
                        pending_timestamps,
                        pending_values,
                    )
            queued_messages.clear()

            while not self._stop.is_set():
                serial_port.timeout = min(self._timeout, DEFAULT_FLUSH_INTERVAL_S)
                chunk = serial_port.read(self._read_chunk_size)
                now_ns = time.perf_counter_ns()
                if not chunk:
                    self._stats.idle_reads += 1
                    if now_ns >= next_flush_ns:
                        self._flush_pending(pending_timestamps, pending_values)
                        next_flush_ns = now_ns + self._flush_interval_ns
                    continue

                for payload in decoder.process(chunk):
                    decoded = self._decode_packet(payload)
                    if not isinstance(decoded, ImuStreamerSample):
                        continue

                    expected_sequence = self._append_sample(
                        decoded,
                        expected_sequence,
                        pending_timestamps,
                        pending_values,
                    )

                if (
                    len(pending_timestamps) >= self._flush_samples
                    or now_ns >= next_flush_ns
                ):
                    self._flush_pending(pending_timestamps, pending_values)
                    next_flush_ns = now_ns + self._flush_interval_ns
        except Exception as exc:  # noqa: BLE001 - surfaced through source status.
            self._stats.parse_errors += 1
            self._set_last_error(str(exc))
        finally:
            self._flush_pending(pending_timestamps, pending_values)
            open_serial_port = self._serial_port
            if open_serial_port is not None:
                try:
                    open_serial_port.write(
                        build_imu_streamer_command(IMU_STREAMER_CMD_STOP)
                    )
                except OSError:
                    pass
                open_serial_port.close()
                self._serial_port = None
            self._done.set()


def _ack_result_message(ack: ImuStreamerAck) -> str:
    if ack.protocol_version != IMU_STREAMER_PROTOCOL_VERSION:
        return (
            "unsupported IMU streamer protocol version "
            f"{ack.protocol_version}; expected {IMU_STREAMER_PROTOCOL_VERSION}"
        )
    if ack.result == IMU_STREAMER_ACK_UNKNOWN_CMD:
        return "IMU streamer rejected command as unknown"
    if ack.result == IMU_STREAMER_ACK_UNSUPPORTED:
        return "IMU streamer package is not supported by this firmware"
    return f"IMU streamer command failed with result {ack.result}"


__all__ = [
    "DEFAULT_FLUSH_INTERVAL_S",
    "DEFAULT_FLUSH_SAMPLES",
    "DEFAULT_PENDING_SAMPLES",
    "DEFAULT_READ_CHUNK_SIZE",
    "DEFAULT_TIMEOUT",
    "IMU_STREAMER_ACK_OK",
    "IMU_STREAMER_ACK_UNKNOWN_CMD",
    "IMU_STREAMER_ACK_UNSUPPORTED",
    "IMU_STREAMER_CHANNELS",
    "IMU_STREAMER_CHANNEL_COUNT",
    "IMU_STREAMER_CMD_START",
    "IMU_STREAMER_CMD_STATUS",
    "IMU_STREAMER_CMD_STOP",
    "IMU_STREAMER_MSG_ACK",
    "IMU_STREAMER_MSG_SAMPLE",
    "IMU_STREAMER_MSG_STATUS",
    "IMU_STREAMER_PROTOCOL_VERSION",
    "ImuStreamBufferStats",
    "ImuStreamerAck",
    "ImuStreamerLispPrint",
    "ImuStreamerMessage",
    "ImuStreamerReaderStats",
    "ImuStreamerSample",
    "ImuStreamerSourceStats",
    "ImuStreamerStatus",
    "PendingImuStreamBuffer",
    "VescImuStreamer",
    "build_imu_streamer_command",
    "parse_imu_streamer_packet",
    "parse_imu_streamer_payload",
]
