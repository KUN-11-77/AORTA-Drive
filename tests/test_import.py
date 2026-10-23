import aorta_drive


def test_version_is_exposed() -> None:
    assert aorta_drive.__version__ == "0.1.0"
