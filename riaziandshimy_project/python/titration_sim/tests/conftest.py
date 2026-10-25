"""pytest configuration: slow paper-scenario test only with --runslow."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests (~50 s)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="need --runslow")
    for item in items:
        fn = getattr(item, "function", None)  # DoctestItem has no .function
        if fn is not None and getattr(fn, "slow", False):
            item.add_marker(skip)
