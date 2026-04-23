#!/usr/bin/env python3
"""Interactive IMU setup wizard through the VESC Tool TCP server.

Start the bridge first, for example:
    ./vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer 65102

Usage:
    uv run examples/imu_setup.py
    uv run examples/imu_setup.py --tcp 192.168.1.50:65102
    uv run examples/imu_setup.py --scan-udp
"""

# pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-lines

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from collections.abc import Callable, Sequence
from typing import Literal, SupportsInt, cast

from vesc_py import VescClient, udp_scan
from vesc_py.imu_setup import (
    IMU_SETUP_MASK,
    FilteredImuState,
    ImuBasicProfile,
    OrientationRestore,
    RollingMeanImuState,
    RollingMeanValue,
    YawOffsetEstimator,
    apply_basic_profile,
    apply_pitch_offset,
    apply_roll_offset,
    apply_yaw_offset,
    axis_current,
    axis_max,
    config_float,
    config_int,
    imu_type_name,
    mean_label,
    prepare_orientation_calibration,
    restore_orientation_config,
    save_accel_offset,
    save_gyro_offsets,
    save_orientation_restore,
    set_config_int,
    validate_required_fields,
)

DEFAULT_TCP_ENDPOINT = ("127.0.0.1", 65102)
DEFAULT_POLL_HZ = 50.0
DEFAULT_SAMPLE_SECONDS = 5.0
DEFAULT_SEARCH_SECONDS = 2.0
DEFAULT_MEAN_SECONDS = 3.0
RAD_TO_DEG = 180.0 / 3.141592653589793
DEG_TO_RAD = 3.141592653589793 / 180.0
ProfileChoice = Literal["current", "default", "logs", "balance-unicycle", "balance-skateboard"]
StepChoice = Literal["save", "retry", "skip", "cancel"]
STEP_PROMPT = "s/Enter=save  r=retry  k=skip  c=cancel"
PostUpdateCallback = Callable[[float, FilteredImuState], None]
StatusCallback = Callable[[FilteredImuState, FilteredImuState], str]


def parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse HOST:PORT for argparse."""

    host, sep, port_str = endpoint.rpartition(":")
    if not sep or not host or not port_str:
        raise argparse.ArgumentTypeError("TCP endpoint must be HOST:PORT")

    try:
        port = int(port_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("TCP port must be an integer") from exc

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("TCP port must be in range 1..65535")

    return host, port


def scan_and_print_udp(timeout: float = 3.0) -> None:
    """Listen for VESC Tool TCP server broadcast announcements and print them."""

    print(f"Listening for VESC Tool TCP server broadcasts on UDP port 65109 for {timeout}s ...")
    devices = udp_scan(timeout=timeout)
    if not devices:
        print("No VESC Tool TCP servers found.")
        return

    for device in devices:
        print(f"  {device.hw_name}  {device.ip}:{device.port}")


def tcp_server_help(endpoint: str, port: int) -> str:
    """Return a concise hint for starting the VESC Tool TCP bridge."""

    return "\n".join(
        [
            f"Could not connect to VESC Tool TCP server at {endpoint}.",
            "",
            "Start VESC Tool with tcpServer enabled, then retry. For example:",
            f"  vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer {port}",
            "",
            "If VESC Tool is already running, check the host and port passed to --tcp.",
        ]
    )


def prompt_bool(prompt: str, *, default: bool) -> bool:
    """Prompt for a yes/no answer."""

    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{suffix}] ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_profile(default: ProfileChoice) -> ProfileChoice:
    """Prompt for the basic IMU parameter profile."""

    choices: dict[str, ProfileChoice] = {
        "c": "current",
        "d": "default",
        "l": "logs",
        "u": "balance-unicycle",
        "s": "balance-skateboard",
    }
    labels = {
        "current": "keep current values",
        "default": "Default",
        "logs": "Logs",
        "balance-unicycle": "Balance Unicycle",
        "balance-skateboard": "Balance Skateboard",
    }
    print("Choose basic IMU profile:")
    print("  c  keep current values")
    print("  d  Default")
    print("  l  Logs")
    print("  u  Balance Unicycle")
    print("  s  Balance Skateboard")
    while True:
        answer = input(f"Profile [{default[0]} = {labels[default]}] ").strip().lower()
        if not answer:
            return default
        if answer in choices:
            return choices[answer]
        for value in labels:
            if answer == value:
                return cast(ProfileChoice, value)
        print("Please choose c, d, l, u, or s.")


def prompt_step(default: StepChoice = "save") -> StepChoice:
    """Prompt for a calibration-step action."""

    while True:
        answer = input(f"Action: save, retry, skip, cancel [{default}] ").strip().lower()
        choice = parse_step_choice(answer, default=default)
        if choice is not None:
            return choice
        print("Please choose save, retry, skip, or cancel.")


def parse_step_choice(answer: str, *, default: StepChoice) -> StepChoice | None:
    """Parse a calibration-step action."""

    choices: dict[str, StepChoice] = {
        "": default,
        "\n": default,
        "\r": default,
        "s": "save",
        "r": "retry",
        "k": "skip",
        "c": "cancel",
    }
    normalized = answer.strip().lower()
    if normalized in choices:
        return choices[normalized]
    if normalized in {"save", "retry", "skip", "cancel"}:
        return cast(StepChoice, normalized)
    return None


def wait_for_enter(message: str, *, assume_yes: bool) -> None:
    """Wait for operator confirmation unless --yes is active."""

    if assume_yes:
        print(message)
        return
    input(f"{message}\nPress Enter when ready...")


def write_appconf(
    client: VescClient,
    config: dict[str, object],
    *,
    store: bool,
    wait_ack: bool,
) -> None:
    """Write APPCONF, matching the QML wizard's store/no-store distinction."""

    client.set_appconf(config, store=store, wait_ack=wait_ack)


class StatusBlockRenderer:
    """Refresh a terminal status block without leaving stale lines behind."""

    def __init__(self, *, include_prompt: bool) -> None:
        self._include_prompt = include_prompt
        self._line_count = 0
        self._use_ansi = sys.stdout.isatty()

    def render(self, block: str) -> None:
        """Render a status block."""

        text = f"{block}\n{STEP_PROMPT}" if self._include_prompt else block
        if self._use_ansi:
            if self._line_count > 1:
                print(f"\033[{self._line_count - 1}F\r\033[J", end="")
            elif self._line_count == 1:
                print("\r\033[J", end="")
            print(text, end="", flush=True)
            self._line_count = text.count("\n") + 1
        else:
            print(text, flush=True)

    def finish(self) -> None:
        """Move to the next line after a terminal status block."""

        if self._use_ansi and self._line_count:
            print()


def sample_filtered_imu(
    client: VescClient,
    *,
    seconds: float,
    poll_hz: float,
    status: StatusCallback | None = None,
    yaw_estimator: YawOffsetEstimator | None = None,
    mean_seconds: float = 0.0,
    post_update: PostUpdateCallback | None = None,
) -> FilteredImuState:
    """Poll and filter IMU data for a fixed duration."""
    # pylint: disable=too-many-locals

    state = FilteredImuState()
    mean_window = RollingMeanImuState(mean_seconds) if mean_seconds > 0.0 else None
    display_state = state
    period = 1.0 / poll_hz
    end_time = time.monotonic() + seconds
    next_poll = time.monotonic()
    next_status = 0.0
    status_renderer = StatusBlockRenderer(include_prompt=False) if status is not None else None

    while time.monotonic() < end_time:
        values = client.get_imu_data(IMU_SETUP_MASK)
        state.update(values)
        if yaw_estimator is not None:
            yaw_estimator.update(values)

        now = time.monotonic()
        if post_update is not None:
            post_update(now, state)
        display_state = mean_window.update(now, state) if mean_window is not None else state
        if status is not None and now >= next_status:
            assert status_renderer is not None
            status_renderer.render(status(state, display_state))
            next_status = now + 0.25

        next_poll += period
        sleep_s = max(0.0, next_poll - time.monotonic())
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_poll = time.monotonic()

    if status_renderer is not None:
        status_renderer.finish()
    return display_state


def sample_filtered_imu_until_step_choice(
    client: VescClient,
    *,
    poll_hz: float,
    status: StatusCallback,
    yaw_estimator: YawOffsetEstimator | None = None,
    mean_seconds: float = 0.0,
    post_update: PostUpdateCallback | None = None,
) -> tuple[FilteredImuState, StepChoice]:
    """Poll IMU data until the user chooses a calibration-step action."""
    # pylint: disable=too-many-locals

    if not sys.stdin.isatty():
        state = sample_filtered_imu(
            client,
            seconds=DEFAULT_SAMPLE_SECONDS,
            poll_hz=poll_hz,
            status=status,
            yaw_estimator=yaw_estimator,
            mean_seconds=mean_seconds,
            post_update=post_update,
        )
        return state, prompt_step()

    state = FilteredImuState()
    mean_window = RollingMeanImuState(mean_seconds) if mean_seconds > 0.0 else None
    display_state = state
    period = 1.0 / poll_hz
    next_poll = time.monotonic()
    next_status = 0.0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    status_renderer = StatusBlockRenderer(include_prompt=True)

    mean_note = f"{mean_seconds:g}s rolling mean" if mean_seconds > 0.0 else "instant value"
    print(f"Sampling {mean_note} until action {STEP_PROMPT}")
    try:
        tty.setcbreak(fd)
        while True:
            values = client.get_imu_data(IMU_SETUP_MASK)
            state.update(values)
            if yaw_estimator is not None:
                yaw_estimator.update(values)

            now = time.monotonic()
            if post_update is not None:
                post_update(now, state)
            display_state = mean_window.update(now, state) if mean_window is not None else state
            if now >= next_status:
                status_renderer.render(status(state, display_state))
                next_status = now + 0.25

            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                char = sys.stdin.read(1)
                choice = parse_step_choice(char, default="save")
                if choice is not None:
                    status_renderer.finish()
                    return display_state, choice
                status_renderer.finish()
                print(f"Unknown action {char!r}. Use {STEP_PROMPT}.")
                status_renderer = StatusBlockRenderer(include_prompt=True)
                status_renderer.render(status(state, display_state))

            next_poll += period
            sleep_s = max(0.0, next_poll - time.monotonic())
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_poll = time.monotonic()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def print_basic_config(config: dict[str, object]) -> None:
    """Print the fields controlled by the basic setup step."""

    imu_type = config_int(config, "imu_conf.type")
    print(
        "Current IMU config:\n"
        f"  Type:             {imu_type_name(imu_type)} ({imu_type})\n"
        f"  Sample rate:      {config_int(config, 'imu_conf.sample_rate_hz')} Hz\n"
        f"  AHRS mode:        {config_int(config, 'imu_conf.mode')}\n"
        f"  Acc decay:        {config_float(config, 'imu_conf.accel_confidence_decay'):.3f}\n"
        f"  Mahony kp:        {config_float(config, 'imu_conf.mahony_kp'):.3f}\n"
        f"  Accel Z filter:   {config_float(config, 'imu_conf.accel_lowpass_filter_z'):.1f}\n"
        f"  Gyro filter:      {config_float(config, 'imu_conf.gyro_lowpass_filter'):.1f}"
    )


def search_imu_type(
    client: VescClient,
    config: dict[str, object],
    *,
    poll_hz: float,
    sample_seconds: float,
    wait_ack: bool,
) -> int:
    """Try IMU type enum values 1..5 with temporary APPCONF writes."""

    previous_app_raw = config.get("app_to_use")
    previous_app = (
        int(cast(SupportsInt, previous_app_raw)) if previous_app_raw is not None else None
    )
    if previous_app is not None:
        set_config_int(config, "app_to_use", 0)

    found_type = 0
    print("Searching IMU types with temporary config writes...")
    for imu_type in range(1, 6):
        set_config_int(config, "imu_conf.type", imu_type)
        write_appconf(client, config, store=False, wait_ack=wait_ack)

        def status(
            state: FilteredImuState,
            _: FilteredImuState,
            name: str = imu_type_name(imu_type),
        ) -> str:
            return f"Testing {name}: gyro noise {state.gyro_x_noise:.4f}"

        state = sample_filtered_imu(
            client,
            seconds=sample_seconds,
            poll_hz=poll_hz,
            status=status,
        )
        if state.working_imu:
            found_type = imu_type
            break

    set_config_int(config, "imu_conf.type", found_type)
    if previous_app is not None:
        set_config_int(config, "app_to_use", previous_app)
    write_appconf(client, config, store=False, wait_ack=wait_ack)
    print(f"IMU search result: {imu_type_name(found_type)} ({found_type})")
    return found_type


def run_basic_step(
    client: VescClient,
    config: dict[str, object],
    *,
    profile_arg: str | None,
    search_arg: bool | None,
    assume_yes: bool,
    poll_hz: float,
    sample_seconds: float,
    wait_ack: bool,
) -> None:
    """Run the basic parameter selection step."""

    print("\nStep 1/4: Basic IMU parameters")
    original = dict(config)
    print_basic_config(config)

    do_search = (
        search_arg
        if search_arg is not None
        else prompt_bool("Search for a working IMU type?", default=False)
    )
    if do_search:
        search_imu_type(
            client,
            config,
            poll_hz=poll_hz,
            sample_seconds=sample_seconds,
            wait_ack=wait_ack,
        )

    if profile_arg is not None:
        profile_choice = profile_arg
    elif assume_yes:
        profile_choice = ImuBasicProfile.DEFAULT.value
    else:
        profile_choice = prompt_profile("default")

    if profile_choice != "current":
        profile = ImuBasicProfile(profile_choice)
        apply_basic_profile(config, profile)
        print(f"Selected profile: {profile.value}")
    else:
        print("Keeping current basic parameter values.")

    print_basic_config(config)
    should_save = assume_yes or prompt_bool("Store these basic IMU parameters?", default=True)
    if should_save:
        write_appconf(client, config, store=True, wait_ack=wait_ack)
        print("Stored basic IMU parameters.")
    else:
        config.clear()
        config.update(original)
        write_appconf(client, config, store=False, wait_ack=wait_ack)
        print("Basic parameter changes discarded.")


def run_gyro_step(
    client: VescClient,
    config: dict[str, object],
    *,
    assume_yes: bool,
    poll_hz: float,
    sample_seconds: float,
    mean_seconds: float,
    wait_ack: bool,
) -> None:
    """Run the gyro-offset calibration step."""

    print("\nStep 2/4: Gyroscope calibration")
    while True:
        wait_for_enter(
            "Leave the IMU stable in any orientation without vibration.",
            assume_yes=assume_yes,
        )
        def gyro_status(raw: FilteredImuState, mean: FilteredImuState) -> str:
            return (
                "Gyro offsets (instant):\n"
                f"X {raw.gyro_x:+.3f}  Y {raw.gyro_y:+.3f}  Z {raw.gyro_z:+.3f}\n"
                f"Gyro offsets ({mean_label(mean_seconds)}):\n"
                f"X {mean.gyro_x:+.3f}  Y {mean.gyro_y:+.3f}  Z {mean.gyro_z:+.3f}"
            )
        if assume_yes:
            state = sample_filtered_imu(
                client,
                seconds=sample_seconds,
                poll_hz=poll_hz,
                status=gyro_status,
                mean_seconds=mean_seconds,
            )
            choice: StepChoice = "save"
        else:
            state, choice = sample_filtered_imu_until_step_choice(
                client,
                poll_hz=poll_hz,
                status=gyro_status,
                mean_seconds=mean_seconds,
            )
        print(
            f"Measured gyro offsets: X {state.gyro_x:+.6f}, "
            f"Y {state.gyro_y:+.6f}, Z {state.gyro_z:+.6f}"
        )
        if choice == "save":
            save_gyro_offsets(config, state)
            write_appconf(client, config, store=True, wait_ack=wait_ack)
            print("Stored gyroscope offsets.")
            print()
            return
        if choice == "skip":
            print("Skipped gyroscope calibration.")
            return
        if choice == "cancel":
            print("Cancelled gyroscope calibration.")
            return


def run_accel_step(
    client: VescClient,
    config: dict[str, object],
    *,
    assume_yes: bool,
    poll_hz: float,
    sample_seconds: float,
    mean_seconds: float,
    wait_ack: bool,
) -> None:
    """Run the accelerometer max-axis calibration step."""

    print("\nStep 3/4: Accelerometer calibration")
    for axis in ("x", "y", "z"):
        while True:
            wait_for_enter(
                f"Rotate the IMU to a stable maximum positive {axis.upper()} acceleration.",
                assume_yes=assume_yes,
            )

            def status(
                raw: FilteredImuState,
                mean: FilteredImuState,
                current_axis: str = axis,
            ) -> str:
                return (
                    f"Accel {current_axis.upper()} (instant):\n"
                    f"{current_axis.upper()} {axis_current(raw, current_axis):+.3f}  "
                    f"Max {axis_max(raw, current_axis):+.3f}\n"
                    f"Accel {current_axis.upper()} ({mean_label(mean_seconds)}):\n"
                    f"{current_axis.upper()} {axis_current(mean, current_axis):+.3f}  "
                    f"Max {axis_max(mean, current_axis):+.3f}"
                )

            if assume_yes:
                state = sample_filtered_imu(
                    client,
                    seconds=sample_seconds,
                    poll_hz=poll_hz,
                    status=status,
                    mean_seconds=mean_seconds,
                )
                choice: StepChoice = "save"
            else:
                state, choice = sample_filtered_imu_until_step_choice(
                    client,
                    poll_hz=poll_hz,
                    status=status,
                    mean_seconds=mean_seconds,
                )
            max_value = axis_max(state, axis)
            print(f"Measured max {axis.upper()}: {max_value:+.6f}")
            if choice == "save":
                save_accel_offset(config, axis, max_value)
                write_appconf(client, config, store=True, wait_ack=wait_ack)
                print(f"Stored accelerometer {axis.upper()} offset.")
                print()
                break
            if choice == "skip":
                print(f"Skipped accelerometer {axis.upper()} calibration.")
                break
            if choice == "cancel":
                print("Cancelled accelerometer calibration.")
                return


def run_roll_orientation(
    client: VescClient,
    config: dict[str, object],
    restore: OrientationRestore,
    *,
    assume_yes: bool,
    poll_hz: float,
    sample_seconds: float,
    mean_seconds: float,
    wait_ack: bool,
) -> StepChoice:
    """Run the roll orientation sub-step."""

    while True:
        wait_for_enter(
            "Place the IMU level on a flat stable surface for roll calibration.",
            assume_yes=assume_yes,
        )
        def roll_status(raw: FilteredImuState, mean: FilteredImuState) -> str:
            raw_offset = -raw.roll * RAD_TO_DEG - restore.rot_roll
            mean_offset = -mean.roll * RAD_TO_DEG - restore.rot_roll
            return (
                "Roll offset (instant):\n"
                f"{raw_offset:+.3f} deg\n"
                f"Roll offset ({mean_label(mean_seconds)}):\n"
                f"{mean_offset:+.3f} deg"
            )
        if assume_yes:
            state = sample_filtered_imu(
                client,
                seconds=sample_seconds,
                poll_hz=poll_hz,
                status=roll_status,
                mean_seconds=mean_seconds,
            )
            choice: StepChoice = "save"
        else:
            state, choice = sample_filtered_imu_until_step_choice(
                client,
                poll_hz=poll_hz,
                status=roll_status,
                mean_seconds=mean_seconds,
            )
        print(f"Measured roll offset: {(-state.roll * RAD_TO_DEG - restore.rot_roll):+.6f} deg")
        if choice == "save":
            apply_roll_offset(config, -state.roll)
            write_appconf(client, config, store=False, wait_ack=wait_ack)
            return "save"
        if choice == "skip":
            apply_roll_offset(config, restore.rot_roll * DEG_TO_RAD)
            write_appconf(client, config, store=False, wait_ack=wait_ack)
            return "skip"
        if choice == "cancel":
            return "cancel"


def run_pitch_orientation(
    client: VescClient,
    config: dict[str, object],
    restore: OrientationRestore,
    *,
    assume_yes: bool,
    poll_hz: float,
    sample_seconds: float,
    mean_seconds: float,
    wait_ack: bool,
) -> StepChoice:
    """Run the pitch orientation sub-step."""

    while True:
        wait_for_enter(
            "Keep the IMU level on a flat stable surface for pitch calibration.",
            assume_yes=assume_yes,
        )
        def pitch_status(raw: FilteredImuState, mean: FilteredImuState) -> str:
            raw_offset = raw.pitch * RAD_TO_DEG - restore.rot_pitch
            mean_offset = mean.pitch * RAD_TO_DEG - restore.rot_pitch
            return (
                "Pitch offset (instant):\n"
                f"{raw_offset:+.3f} deg\n"
                f"Pitch offset ({mean_label(mean_seconds)}):\n"
                f"{mean_offset:+.3f} deg"
            )
        if assume_yes:
            state = sample_filtered_imu(
                client,
                seconds=sample_seconds,
                poll_hz=poll_hz,
                status=pitch_status,
                mean_seconds=mean_seconds,
            )
            choice: StepChoice = "save"
        else:
            state, choice = sample_filtered_imu_until_step_choice(
                client,
                poll_hz=poll_hz,
                status=pitch_status,
                mean_seconds=mean_seconds,
            )
        print(
            f"Measured pitch offset: {(state.pitch * RAD_TO_DEG - restore.rot_pitch):+.6f} deg"
        )
        if choice == "save":
            apply_pitch_offset(config, state.pitch)
            write_appconf(client, config, store=False, wait_ack=wait_ack)
            return "save"
        if choice == "skip":
            apply_pitch_offset(config, restore.rot_pitch * DEG_TO_RAD)
            write_appconf(client, config, store=False, wait_ack=wait_ack)
            return "skip"
        if choice == "cancel":
            return "cancel"


def run_yaw_orientation(
    client: VescClient,
    config: dict[str, object],
    restore: OrientationRestore,
    *,
    assume_yes: bool,
    poll_hz: float,
    sample_seconds: float,
    mean_seconds: float,
    wait_ack: bool,
) -> StepChoice:
    """Run the yaw orientation sub-step."""
    # pylint: disable=too-many-locals

    while True:
        wait_for_enter(
            "Raise pitch to roughly 45 degrees while keeping roll level for yaw calibration.",
            assume_yes=assume_yes,
        )
        estimator = YawOffsetEstimator()
        yaw_mean = RollingMeanValue(mean_seconds) if mean_seconds > 0.0 else None

        def record_yaw_offset(timestamp: float, _: FilteredImuState) -> None:
            if yaw_mean is not None:
                yaw_mean.update(timestamp, estimator.yaw_offset)

        def current_yaw_offset() -> float:
            if yaw_mean is not None and yaw_mean.sample_count > 0:
                return yaw_mean.value
            return estimator.yaw_offset

        def status(_: FilteredImuState, __: FilteredImuState) -> str:
            raw_offset = -estimator.yaw_offset * RAD_TO_DEG - restore.rot_yaw
            mean_offset = -current_yaw_offset() * RAD_TO_DEG - restore.rot_yaw
            return (
                "Yaw offset (instant):\n"
                f"{raw_offset:+.3f} deg\n"
                f"Yaw offset ({mean_label(mean_seconds)}):\n"
                f"{mean_offset:+.3f} deg"
            )

        if assume_yes:
            sample_filtered_imu(
                client,
                seconds=sample_seconds,
                poll_hz=poll_hz,
                status=status,
                yaw_estimator=estimator,
                post_update=record_yaw_offset,
            )
            choice: StepChoice = "save"
        else:
            _, choice = sample_filtered_imu_until_step_choice(
                client,
                poll_hz=poll_hz,
                status=status,
                yaw_estimator=estimator,
                post_update=record_yaw_offset,
            )
        yaw_offset_raw = current_yaw_offset()
        yaw_offset = -yaw_offset_raw * RAD_TO_DEG - restore.rot_yaw
        print(f"Measured yaw offset: {yaw_offset:+.6f} deg")
        if choice == "save":
            apply_yaw_offset(config, -yaw_offset_raw)
            write_appconf(client, config, store=True, wait_ack=wait_ack)
            return "save"
        if choice == "skip":
            apply_yaw_offset(config, restore.rot_yaw * DEG_TO_RAD)
            write_appconf(client, config, store=True, wait_ack=wait_ack)
            return "skip"
        if choice == "cancel":
            return "cancel"


def run_orientation_step(
    client: VescClient,
    config: dict[str, object],
    *,
    assume_yes: bool,
    poll_hz: float,
    sample_seconds: float,
    mean_seconds: float,
    wait_ack: bool,
) -> None:
    """Run the orientation calibration step."""

    print("\nStep 4/4: Orientation calibration")
    restore = save_orientation_restore(config)
    prepare_orientation_calibration(config)
    write_appconf(client, config, store=False, wait_ack=wait_ack)

    try:
        for runner in (run_roll_orientation, run_pitch_orientation, run_yaw_orientation):
            choice = runner(
                client,
                config,
                restore,
                assume_yes=assume_yes,
                poll_hz=poll_hz,
                sample_seconds=sample_seconds,
                mean_seconds=mean_seconds,
                wait_ack=wait_ack,
            )
            if choice == "cancel":
                restore_orientation_config(config, restore)
                write_appconf(client, config, store=False, wait_ack=wait_ack)
                print("Orientation calibration cancelled; previous values restored temporarily.")
                return
    except KeyboardInterrupt:
        restore_orientation_config(config, restore)
        write_appconf(client, config, store=False, wait_ack=wait_ack)
        raise

    print(
        "Stored orientation calibration:\n"
        f"  Roll:  {config_float(config, 'imu_conf.rot_roll'):+.6f} deg\n"
        f"  Pitch: {config_float(config, 'imu_conf.rot_pitch'):+.6f} deg\n"
        f"  Yaw:   {config_float(config, 'imu_conf.rot_yaw'):+.6f} deg"
    )


def run_wizard(client: VescClient, args: argparse.Namespace) -> None:
    """Run all requested IMU setup steps."""

    config = client.get_appconf()
    validate_required_fields(config)
    if args.mean_seconds > 0.0:
        print(
            f"Calibration readouts and saved values use a "
            f"{args.mean_seconds:g} second rolling mean."
        )
    else:
        print("Calibration readouts and saved values use instantaneous filtered values.")

    if not args.skip_basic:
        run_basic_step(
            client,
            config,
            profile_arg=args.profile,
            search_arg=args.search_imu,
            assume_yes=args.yes,
            poll_hz=args.rate,
            sample_seconds=args.search_seconds,
            wait_ack=args.wait_ack,
        )

    if not args.skip_gyro:
        run_gyro_step(
            client,
            config,
            assume_yes=args.yes,
            poll_hz=args.rate,
            sample_seconds=args.sample_seconds,
            mean_seconds=args.mean_seconds,
            wait_ack=args.wait_ack,
        )

    if not args.skip_accel:
        run_accel_step(
            client,
            config,
            assume_yes=args.yes,
            poll_hz=args.rate,
            sample_seconds=args.sample_seconds,
            mean_seconds=args.mean_seconds,
            wait_ack=args.wait_ack,
        )

    if not args.skip_orientation:
        run_orientation_step(
            client,
            config,
            assume_yes=args.yes,
            poll_hz=args.rate,
            sample_seconds=args.sample_seconds,
            mean_seconds=args.mean_seconds,
            wait_ack=args.wait_ack,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run the VESC IMU setup wizard through a VESC Tool --tcpServer bridge.",
        epilog=(
            "Start VESC Tool first, for example: "
            "vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer 65102"
        ),
    )
    parser.add_argument(
        "--tcp",
        type=parse_tcp_endpoint,
        default=DEFAULT_TCP_ENDPOINT,
        metavar="HOST:PORT",
        help="VESC Tool TCP server endpoint (default: 127.0.0.1:65102).",
    )
    parser.add_argument(
        "--scan-udp",
        action="store_true",
        help="Scan for VESC Tool TCP server UDP broadcasts and exit.",
    )
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=3.0,
        metavar="SEC",
        help="UDP scan duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        metavar="SEC",
        help="VESC response timeout in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_POLL_HZ,
        metavar="HZ",
        help="IMU poll rate during calibration in Hz (default: 50).",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=DEFAULT_SAMPLE_SECONDS,
        metavar="SEC",
        help=(
            "Sample duration for gyro/accel/orientation steps in --yes mode "
            "(default: 5.0). Interactive mode samples until you respond."
        ),
    )
    parser.add_argument(
        "--search-seconds",
        type=float,
        default=DEFAULT_SEARCH_SECONDS,
        metavar="SEC",
        help="Sample duration for each IMU type during search (default: 2.0).",
    )
    parser.add_argument(
        "--mean-seconds",
        type=float,
        default=DEFAULT_MEAN_SECONDS,
        metavar="SEC",
        help=(
            "Rolling mean window for displayed and saved calibration values "
            "(default: 3.0, use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in ImuBasicProfile] + ["current"],
        metavar="PROFILE",
        help="Basic profile to apply without prompting.",
    )
    parser.add_argument(
        "--search-imu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable IMU type search without prompting.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Use defaults and save each measured step without prompts.",
    )
    parser.add_argument(
        "--wait-ack",
        action="store_true",
        help="Wait for APPCONF write acknowledgements.",
    )
    parser.add_argument("--skip-basic", action="store_true", help="Skip basic parameter setup.")
    parser.add_argument("--skip-gyro", action="store_true", help="Skip gyro calibration.")
    parser.add_argument("--skip-accel", action="store_true", help="Skip accelerometer calibration.")
    parser.add_argument(
        "--skip-orientation",
        action="store_true",
        help="Skip orientation calibration.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI arguments."""

    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be greater than 0")
    if args.rate <= 0.0:
        raise SystemExit("--rate must be greater than 0")
    if args.sample_seconds <= 0.0:
        raise SystemExit("--sample-seconds must be greater than 0")
    if args.search_seconds <= 0.0:
        raise SystemExit("--search-seconds must be greater than 0")
    if args.mean_seconds < 0.0:
        raise SystemExit("--mean-seconds must be greater than or equal to 0")


def main(argv: Sequence[str] | None = None) -> None:
    """Program entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    if args.scan_udp:
        scan_and_print_udp(args.scan_timeout)
        return

    host, port = args.tcp
    endpoint = f"{host}:{port}"
    print(f"Connecting to VESC Tool TCP server at {endpoint} ...")
    try:
        client = VescClient.connect_tcp(host, port, timeout=args.timeout)
    except ConnectionRefusedError:
        raise SystemExit(tcp_server_help(endpoint, port)) from None
    except ConnectionError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "The TCP socket opened, but no VESC firmware response was received. "
            "Make sure VESC Tool is connected to a controller before starting setup."
        ) from None

    try:
        fw = client.fw_version
        if fw is not None:
            print(f"Firmware: {fw.major}.{fw.minor:02d}  HW: {fw.hw}")
        run_wizard(client, args)
        print("\nIMU setup finished.")
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        client.close()


if __name__ == "__main__":
    main()
