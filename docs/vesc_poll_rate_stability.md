# VESC IMU Poll Rate Stability

This note documents the changes made while improving VESC IMU request/response
poll stability at high host-side poll rates such as 500 Hz.

## Initial Observation

The timestamp/index example originally plotted sample lag by taking the
difference between stored sample timestamps. At 500 Hz the expected interval is
2 ms, but the plot showed frequent spikes.

The first important distinction was that the stored timestamp was taken when the
VESC response was received, not when the request was sent. That made the sample
lag include response-latency jitter:

```text
sample_lag ~= request_interval + current_response_latency - previous_response_latency
```

So response latency changes were being interpreted as poll-rate instability.

## Timestamping Requests

The first fix was to timestamp samples at request dispatch time. This made the
sample interval reflect the host poll loop more directly, while keeping
request-to-response latency as a separate diagnostic channel.

After this change, response latency spikes no longer stretched every adjacent
sample lag. The remaining lag spikes represented real scheduling misses or
request/response cycles that took longer than the configured period.

## Moving Buffer Work Off The Poll Loop

At 500 Hz the poll period is only 2 ms. The original loop periodically flushed
pending Python lists into `PendingSignalBuffer`, which converts data through
NumPy arrays and takes locks. That work is not large, but it was happening in
the same thread responsible for waking and sending serial requests.

The stronger fix split responsibilities:

- The poll thread now handles only timing-sensitive serial request/read work.
- Each accepted sample is pushed into a `queue.SimpleQueue`.
- A publisher thread drains that queue into the pending buffers in batches.

This reduces avoidable work in the 500 Hz critical path.

## Request Lateness

A `request_lateness` channel was added to measure:

```text
actual_request_time - scheduled_request_time
```

This channel answers whether a spike came from Python/Linux waking late. Small
values near zero mean the host sent the request on time. Spikes mean the host
thread missed its scheduled wakeup.

## Skipping Missed Slots

After long responses or scheduler stalls, the poll loop could fall behind its
nominal schedule. Catching up by issuing immediate back-to-back requests is bad
for a stable poll stream, because it creates bursts.

The scheduler now skips stale slots. If the next nominal 500 Hz slots were
4 ms, 6 ms, and 8 ms, but the host is already at 8.1 ms, the next request is
scheduled at 10 ms instead of sending catch-up requests immediately.

The lag plot then shows honest gaps:

- 2 ms means no slot was missed at 500 Hz.
- 4 ms means one poll slot was missed.
- 6 ms means two poll slots were missed.

## Missed Slots

A `missed_slots` channel was added to quantify skipped nominal poll slots per
sample. At 500 Hz:

```text
missed_slots = 0  -> interval should stay near 2 ms
missed_slots = 1  -> next interval should be near 4 ms
missed_slots = 2  -> next interval should be near 6 ms
```

Use this channel to measure whether a target poll rate is realistic. A high
poll rate is only useful if the missed-slot rate is acceptable for the analysis.

## Reading The Plots

Use the timestamp/index example as follows:

- `Lag Between Samples`: effective spacing of accepted samples. At 500 Hz, the
  baseline should be 2 ms. Spikes in multiples of 2 ms are skipped slots.
- `VESC Response Latency`: serial/BLE/CAN and firmware response time. If this
  exceeds the poll period, misses are unavoidable.
- `Request Lateness`: host wakeup/scheduler delay. If this exceeds the poll
  period, misses are host-side.
- `Missed Poll Slots`: direct count of skipped nominal slots before each sample.
- `Timestamp Over Sample Index`: should remain mostly linear, with slope changes
  when slots are skipped.

## Practical Rate Selection

For request/response polling, choose a poll period that is comfortably above the
tail latency, not just the average latency.

Examples:

- 500 Hz has a 2 ms period. Any response or wakeup delay above 2 ms can force a
  skipped slot.
- 333 Hz has a 3 ms period. It tolerates moderate latency spikes better.
- 250 Hz has a 4 ms period. It is a better fit if occasional response latency is
  around 3-4 ms.

The best rate is the highest rate where `missed_slots` is almost always zero
during the workload that matters.

## Remaining Limits

Host-side Python request/response polling cannot guarantee hard real-time
sampling. USB, BLE, CAN forwarding, firmware response time, the Python runtime,
and Linux scheduling all contribute to tail latency.

For truly stable high-rate sampling, prefer device-side streaming or
device-side timestamped samples if the VESC firmware path can provide them.
