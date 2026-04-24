import pytest

from examples.qt_yalsa_status_only import build_parser


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args(["--serial", "/dev/ttyACM0"])

    assert args.axis == "acc_z"
    assert args.mode == "full"
    assert args.status_interval == pytest.approx(1.0)
    assert args.duration == pytest.approx(0.0)
