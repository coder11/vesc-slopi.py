# AI slop warning

This repository was 100% vibecoded with llm for research and prototyping purposes.
Be **extra** cautions and when using it with the real hardware.

VESC is a registered trademark of Benjamin Vedder. Read the [original trademark policies](https://vesc-project.com/trademark_policies) for more information.
There is no intention to replace the original [vesc_tool](https://github.com/vedderb/vesc_tool)

# vesc.(slopi)py

Pure-Python programmatic API for talking to VESC hardware over serial, TCP, or
BLE.

The VESC binary protocol is implemented in Python. Firmware configuration XMLs
are read from the upstream `vesc_tool` git submodule at `vesc_tool/res/config`.

## Setup

Use the repository through Nix:

```bash
git submodule update --init --recursive
nix develop
uv sync --extra dev
```

`nix develop` provides `uv`, the selected Python interpreter, and native
libraries needed by PyPI wheels. `uv` owns the Python environment and packages
from `pyproject.toml` / `uv.lock`; do not use system `python`, `pip`, or a
separate virtualenv for this repo.

## Run

Run scripts and tools with `uv run` from the repository root:

```bash
uv run examples/discover_vescs.py
uv run examples/poll_imu_fast.py --port /dev/ttyACM0
uv run examples/imu_live_plot.py --tcp 127.0.0.1:65102
uv run examples/imu_live_plot.py --scan-ble
uv run examples/imu_live_plot.py --ble AA:BB:CC:DD:EE:FF
uv run examples/yalsa/live_signal_analysis.py --source deterministic --axis acc_z
uv run examples/yalsa/live_signal_analysis.py --source vesc --axis acc_z --pipeline-depth 4
uv run examples/config_tui.py --tcp 127.0.0.1:65102
```

For examples that use the VESC Tool TCP bridge, start VESC Tool separately with
`--tcpServer 65102`.

## Development

```bash
uv sync --extra dev
uv run poe test
uv run poe typecheck
```

Tests are intended to be unit tests and do not require VESC hardware.

## Live Analysis Scripts

`examples/yalsa/live_signal_analysis.py` is a YALSA-based PyQtGraph runtime for
live signal sources, tunable parameters, and arbitrary Python processing. The
proof-of-concept pipeline analyzes one VESC IMU axis with a SciPy Butterworth
low-pass and time/frequency plots.

New analysis scripts are meant to stay small: define a source, declare the
live-tunable parameters, write a `process(data, params)` function, and pass the
resulting `LiveAnalysisApp` into `run_live_analysis(...)`.

## Firmware XML Paths

By default, config XML lookup uses:

```text
vesc_tool/res/config/<major>.<minor>/parameters_appconf.xml
vesc_tool/res/config/<major>.<minor>/parameters_mcconf.xml
```

The `vesc_tool` directory is the upstream submodule. If you need a different
checkout, set `VESC_TOOL_DIR=/path/to/vesc_tool` or point directly at config XMLs
with `VESC_CONFIG_DIR=/path/to/res/config`.

## Usage

```python
from vesc_py import VescClient

client = VescClient.connect_serial("/dev/ttyACM0")
print(client.fw_version)

imu = client.get_imu_data()
print(f"Roll: {imu.roll:.2f}  Pitch: {imu.pitch:.2f}  Yaw: {imu.yaw:.2f}")

client.close()
```

```python
conf = client.get_appconf()
conf["controller_id"] = 42
client.set_appconf(conf)
```

## Project Layout

```text
pyproject.toml
flake.nix
src/vesc_py/          # library
tests/                # unit tests
examples/             # runnable scripts
vesc_tool/            # upstream VESC Tool submodule for firmware XMLs
```

## License

Same as upstream `vesc_tool` (GPL-3.0).
