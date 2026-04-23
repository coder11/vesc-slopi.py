import pytest

from vesc_py.config_paths import find_appconf_xml, find_config_dir, find_mcconf_xml


def test_find_parameters_appconf_lists_available_on_miss() -> None:
    with pytest.raises(FileNotFoundError, match="Available versions"):
        find_appconf_xml(99, 99)


def test_find_known_version() -> None:
    path = find_appconf_xml(6, 6)
    assert path.name == "parameters_appconf.xml"
    assert path.exists()


def test_find_known_mcconf_version() -> None:
    path = find_mcconf_xml(6, 6)
    assert path.name == "parameters_mcconf.xml"
    assert path.exists()


def test_find_known_config_dir() -> None:
    path = find_config_dir(6, 6)
    assert path.name == "6.06"
    assert path.exists()
