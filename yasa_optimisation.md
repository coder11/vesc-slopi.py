`# YALSA Optimisation Notes

This file captures the current findings from the IMU polling investigation.

## Goal

`examples/yalsa/live_signal_analysis.py` was only reaching about `300-400 Hz`
while `examples/poll_imu_fast.py` was reaching about `1300 Hz` on the same
controller and serial link.

## Instrumentation Added

- `src/vesc_py/live_signal.py`
  - pending-buffer lock wait/hold timing
- `src/vesc_py/fast_imu_source.py`
  - source loop timing buckets and `debug_text`
- `src/yalsa/app.py`
  - GUI refresh timing breakdown in the status label
- `examples/poll_imu_fast.py`
  - matching `srcdbg` timing buckets
  - `--worker-thread` mode to run the same loop from a background thread

## Measured Results

### YALSA live source

Observed from the instrumented status view:

- `source: 374.6 Hz`
- `srcdbg: loop=2.666ms`
- `srcdbg: rd=1.802`
- `srcdbg: wr=0.802`
- `srcdbg: parse=0.021`
- `srcdbg: pub=0.028`
- `srcdbg: flush=0.031`
- `srcdbg: state_lock=0.002/0.001`
- `srcdbg: buf_wait a/d=0.001/0.003`
- `srcdbg: buf_hold a/d=0.010/0.016`
- `srcdbg: buf_hi=106`
- `uidbg: int=80.27/74.84ms`
- `uidbg: tot=6.286/6.000`
- `uidbg: drain=0.053`
- `uidbg: hist=0.022`
- `uidbg: snap=0.151`
- `uidbg: proc=4.520`
- `uidbg: plot=1.457`
- `uidbg: status=0.073`
- `uidbg: empty=1`
- `uidbg: over=0`

### Fast poller on main thread

Observed from `examples/poll_imu_fast.py`:

- about `1314 Hz`
- `srcdbg: loop=0.760ms`
- `srcdbg: rd=0.627`
- `srcdbg: wr=0.094`
- `srcdbg: parse=0.034`

### Fast poller on worker thread

Observed from `examples/poll_imu_fast.py --worker-thread`:

- about `1340 Hz`
- `srcdbg: loop=0.746ms`
- `srcdbg: rd=0.647`
- `srcdbg: wr=0.070`
- `srcdbg: parse=0.025`

## Conclusions So Far

### Locks are not the bottleneck

The shared pending-buffer and source-state locks are all in the microsecond
range. They are negligible relative to the `2.666 ms` YALSA source loop.

### Qt refresh is not directly stalling the source

The GUI refresh path is doing real work (`proc` and `plot` are not free), but
it is still completing well inside the timer interval (`over=0`). That means
the source slowdown is not explained by the GUI thread missing its own schedule.

### Parsing is not the bottleneck

`parse` is small in both paths.

### The gap is in transport time inside the YALSA source path

The large difference is almost entirely:

- `wr`
- `rd`

The strongest signal is `wr`:

- fast poller main-thread: `wr=0.094`
- fast poller worker-thread: `wr=0.070`
- YALSA source: `wr=0.802`

That points away from:

- plotting more axes
- response payload size
- parser cost
- generic use of `threading.Thread`

## Ruled Out

- pending-buffer lock contention
- source-state lock contention
- Qt timer overruns
- parser overhead
- "background thread by itself" as the cause

## Current Hypothesis

The remaining issue is specific to the `VescImuSignalSource` path or to
something that exists only when running that source inside the live-analysis
stack, but not when running the `poll_imu_fast.py` loop directly.

## Next Step

Run `VescImuSignalSource` headless, without Qt or YALSA UI, and compare its
`srcdbg` values against:

- YALSA live analysis
- `poll_imu_fast.py`
- `poll_imu_fast.py --worker-thread`

## headless

results:

(vesc-slopi.py) pavel@pavel-asus ~/c/v/vesc-slopi.py (main)> nix develop -c uv run examples/headless_fast_imu_source.py --serial /dev/ttyACM0 --axis acc_z

warning: Git tree '/home/pavel/code/vesc/vesc-slopi.py' is dirty
CAN scan timed out on serial /dev/ttyACM0; using the directly connected controller.
Starting headless VescImuSignalSource on serial /dev/ttyACM0; axis=acc_z; status_interval=1s; pending_samples=20000

source: 1354.9 Hz
drain: 1345.3 Hz
samples: 5422
dropped: 0
errors: 1
batch_samples: 1346
batch_dropped: 0
latest: acc_z=0.953435 g
srcdbg: loop=0.737ms
srcdbg: rd=0.616
srcdbg: wr=0.095
srcdbg: parse=0.015
srcdbg: pub=0.021
srcdbg: flush=0.030
srcdbg: state_lock=0.001/0.001
srcdbg: buf_wait
srcdbg: a/d=0.001/0.004
srcdbg: buf_hold
srcdbg: a/d=0.012/0.062
srcdbg: buf_hi=1387

## Additional Code Finding

After the headless run confirmed that `VescImuSignalSource` itself is not the
bottleneck, the remaining live-analysis path was reviewed again.

`src/yalsa/app.py` already had the start of a plot-range optimisation:

- `_PlotRangeTracker`
- `_finite_bounds`
- `_merge_bounds`
- `_update_plot_ranges`
- `disableAutoRange()`

But `refresh()` was still calling `plot_item.autoRange()` on every update, and
the computed `x_bounds` / `y_bounds` values were never applied.

That means the intended range-tracking optimisation was effectively inactive in
the live Qt/YALSA path.

## Current Hypothesis

The most concrete remaining suspect is now repeated PyQtGraph/Qt plot
auto-ranging and the repaint work it triggers. That work would exist only in
the live GUI path, not in the headless source bench, and some of it can happen
outside the measured `refresh()` callback timing.

## Change Made

`src/yalsa/app.py` now:

- stops calling `plot_item.autoRange()` every refresh
- computes explicit x/y bounds from the visible decimated series
- applies range updates only when the bounds materially change

`src/yalsa/app.py` and `ScalarSignalSourceAdapter` also now carry scalar
`debug_text` through the batch-source path again so live YALSA runs can expose
`srcdbg` metrics in the status label.

## Result After Change

The first live rerun still reported roughly the same source rate:

- `source: 342.3 Hz`
- `history: 342.6 Hz`
- `samples: 2574`
- `dropped: 0`
- `errors: 1`

So disabling per-refresh plot auto-range was not enough to recover the missing
throughput by itself.

## Headless YALSA Timer Result

Running the YALSA timer/history path without Qt and without the DSP callback:

- `nix develop -c uv run examples/headless_yalsa_analysis.py --serial /dev/ttyACM0 --mode noop`

produced:

- `source: 1424.1 Hz`
- `history: 1417.5 Hz`
- `srcdbg: loop=0.701ms`
- `srcdbg: rd=0.614`
- `srcdbg: wr=0.071`
- `uidbg: tot=0.151`
- `uidbg: drain=0.038`
- `uidbg: hist=0.015`
- `uidbg: snap=0.090`
- `uidbg: proc=0.001`

That rules out:

- the source worker thread
- the 15 Hz refresh schedule by itself
- the pending/history buffer path
- basic non-Qt status bookkeeping

So the remaining suspects are now:

- the actual YALSA DSP callback
- the Qt/PyQtGraph runtime

Synthetic local measurement of the full YALSA processor on a 4096-sample window
was only a few milliseconds per refresh, which makes the GUI/render side the
stronger suspect.

## Protocol Result

For the protocol runs, all three non-plot variants stayed fast on hardware:

- `examples/headless_yalsa_analysis.py --mode noop`
- `examples/headless_yalsa_analysis.py --mode full`
- `examples/qt_yalsa_status_only.py --mode full`

That means the following paths are all fast enough:

- the source worker thread
- the non-Qt timer/history path
- the full YALSA DSP callback
- the Qt event loop plus simple status-label updates

## Current Conclusion

The remaining bottleneck is therefore in the full live GUI path, specifically
the Qt/PyQtGraph plot/widget update and render path used by
`examples/yalsa/live_signal_analysis.py`.

## Changes Added During This Investigation

- `examples/headless_fast_imu_source.py`
  - isolates `VescImuSignalSource` from the GUI stack
- `examples/headless_yalsa_analysis.py`
  - runs the YALSA timer/history/process loop without Qt/PyQtGraph
- `examples/qt_yalsa_status_only.py`
  - runs the YALSA timer/history/process loop under Qt without plot widgets
- `src/yalsa/app.py`
  - restored scalar `debug_text` propagation into the batch-source status path
  - fixed the unfinished plot-range optimisation so per-refresh `autoRange()`
    is no longer called

## Next Step

The next isolation steps are:

1. Run the same YALSA source and DSP callback without Qt/PyQtGraph:

- `uv run examples/headless_yalsa_analysis.py --serial /dev/ttyACM0 --mode noop`
- `uv run examples/headless_yalsa_analysis.py --serial /dev/ttyACM0 --mode full`

Interpretation:

- if `noop` stays fast and `full` drops, the bottleneck is in the YALSA
  analysis/history path or GIL pressure from DSP work
- if both stay fast, the remaining bottleneck is in Qt/PyQtGraph rendering or
  event-loop side effects

2. Run the same YALSA source and DSP callback under a real Qt event loop but
   without PyQtGraph plots:

- `uv run examples/qt_yalsa_status_only.py --serial /dev/ttyACM0 --mode noop`
- `uv run examples/qt_yalsa_status_only.py --serial /dev/ttyACM0 --mode full`

Interpretation:

- if Qt status-only stays fast, the remaining culprit is specifically the
  PyQtGraph plot/widget update path
- if Qt status-only drops while headless full stays fast, the culprit is Qt
  event-loop / QWidget / label-update side effects rather than the DSP callback

3. Re-run `examples/yalsa/live_signal_analysis.py` and compare:

- `source`
- `srcdbg: loop`
- `srcdbg: rd`
- `srcdbg: wr`

against the earlier `374.6 Hz` / `2.666 ms` live-analysis result.

## GUI Decoupling Fix

The live GUI runtime has now been changed so Qt/PyQtGraph no longer owns the
source drain and DSP cadence.

`src/yalsa/app.py` now runs a `_LiveAnalysisWorker` background thread that:

- drains the source as samples arrive
- updates `SignalBatchHistory`
- runs the configured analysis callback, including filter and FFT/PSD work
- publishes the latest full-resolution `AnalysisResult` to the GUI

The Qt timer now only:

- reads the latest already-processed result
- decimates series for display according to each `PlotSpec.max_points`
- updates PyQtGraph curves at the capped plot cadence
- updates the status label

The GUI plot cadence is also capped at `60 Hz` even if a higher
`plot_rate_hz` is configured. The worker status now exposes a separate
`processing` rate and last `proc` duration in the live status label, so future
hardware runs can distinguish:

- source acquisition rate
- retained-history sample rate
- analysis processing rate
- GUI plot update rate

The next hardware validation is to rerun:

- `uv run examples/yalsa/live_signal_analysis.py --serial /dev/ttyACM0`

and compare the live status values against the known headless-good values of
roughly `1.3-1.4 kHz`.
