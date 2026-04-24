import pytest

from examples.headless_yalsa_analysis import build_parser, expand_debug_text, format_latest_values
from yalsa import SignalBatchSourceSnapshot


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args(["--serial", "/dev/ttyACM0"])

    assert args.axis == "acc_z"
    assert args.mode == "full"
    assert args.status_interval == pytest.approx(1.0)
    assert args.refresh_rate is None
    assert args.duration == pytest.approx(0.0)


def test_expand_debug_text_splits_metrics() -> None:
    assert expand_debug_text("srcdbg loop=2.666ms rd=1.802") == [
        "srcdbg: loop=2.666ms",
        "srcdbg: rd=1.802",
    ]


def test_format_latest_values_includes_units() -> None:
    snapshot = SignalBatchSourceSnapshot(
        samples=1,
        dropped=0,
        errors=0,
        average_rate_hz=100.0,
        latest_sample_s=0.1,
        latest_values={"acc_z": 0.998},
        last_error=None,
        done=False,
    )

    assert format_latest_values(snapshot, {"acc_z": "g"}) == "acc_z=0.998 g"
