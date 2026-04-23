# TODO: Decouple Fast IMU Acquisition From Plotting


`python/examples/poll_imu_fast.py` can poll IMU data faster in text mode than in
GUI mode. A representative comparison is roughly:

- Text/status mode: about 6 kHz
- PyQtGraph GUI mode: about 4 kHz

The serial/device side is probably not the only limiter. The GUI path currently
uses Python threads: one thread polls serial data while the Qt main thread drains
samples and redraws PyQtGraph. Those threads still share the Python GIL, and the
plot path adds periodic Python work for batching, `setData(...)`, axis updates,
and repaint scheduling. That UI work can delay the serial polling thread even
when USB bandwidth is still available.

## Goal

Keep the high-rate serial acquisition loop near text-mode throughput while still
showing a live plot. The plot should consume a decimated or windowed stream
without being able to stall the raw acquisition path.

## Proposed Direction

Split acquisition and plotting into separate processes:

- Acquisition process owns the serial port and runs the tight
  `COMM_GET_IMU_DATA` request/response loop.
- GUI process owns Qt/PyQtGraph and renders only a display-rate stream.
- Communication between processes should be bounded and overwrite/drop old plot
  samples rather than blocking acquisition.

This avoids GIL contention between the serial hot path and PyQtGraph. It also
makes backpressure explicit: if the GUI cannot keep up, it loses plot points,
not raw polling throughput.

## Implementation Sketch

1. Extract the fast single-axis polling loop into a reusable acquisition worker.
   It should report:
   - elapsed sample timestamp
   - selected axis value
   - counters: samples, requests, timeouts, CRC errors, parse errors

2. Add a multiprocessing transport for GUI mode.
   Candidate approaches:
   - `multiprocessing.Queue(maxsize=N)` with non-blocking put and drop-oldest
     behavior
   - shared-memory ring buffer with atomic-ish write/read indexes
   - `multiprocessing.Pipe` for low-rate status plus shared memory for samples

3. Keep acquisition non-blocking.
   The acquisition process must never block on GUI delivery. If the GUI buffer is
   full, it should overwrite or drop old display samples and increment a dropped
   counter.

4. Send plot data at a controlled rate.
   The acquisition process can still poll every packet, but only forward:
   - every Nth sample, or
   - min/max/last buckets per UI frame interval, or
   - a bounded rolling window sampled down to `--plot-max-points`

5. Keep status counters high fidelity.
   The GUI should display the acquisition process's true poll average, not the
   display sample rate. The status line should distinguish:
   - raw poll average
   - samples forwarded to GUI
   - UI redraw rate
   - dropped GUI samples

6. Preserve current CLI behavior.
   Existing text, CSV, and TUI modes should remain in-process. Multiprocessing
   should be used only for `--plot`, or hidden behind a flag such as
   `--plot-process`.

7. Add tests around the transport policy.
   Unit tests should cover:
   - full-buffer drop/overwrite behavior
   - monotonic ordering of drained samples
   - status counter snapshots
   - acquisition continuing when the consumer is slow

## Open Questions

- Should plot mode show decimated samples only, or min/max envelopes so spikes
  remain visible when downsampling?
- Should CSV logging be added to the acquisition process so raw-rate capture can
  run alongside a decimated live plot?
- Which IPC mechanism is fastest and simplest enough to maintain in this example
  script?
- Should the default GUI path always use multiprocessing once stable, or should
  threaded mode remain available for easier debugging?

## Acceptance Criteria

- GUI mode raw poll average is close to text-mode raw poll average using the same
  `--fields`/`--mask` and `--pipeline-depth`.
- Closing the GUI reliably stops the acquisition process and closes the serial
  port.
- A slow or busy GUI does not block acquisition.
- The UI clearly reports raw poll rate, rendered plot rate, and dropped display
  samples.
