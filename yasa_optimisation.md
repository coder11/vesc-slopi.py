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
