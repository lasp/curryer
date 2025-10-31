"""Pytest configuration for curryer tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--run-extra",
        action="store_true",
        default=False,
        help="run extra tests (e.g., slow or integration tests)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "extra: mark test as extra to run (deselected by default)")


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with 'extra' unless --run-extra is passed."""
    if config.getoption("--run-extra"):
        # --run-extra given in cli: do not skip extra tests
        return

    skip_extra = pytest.mark.skip(reason="need --run-extra option to run")
    for item in items:
        if "extra" in item.keywords:
            item.add_marker(skip_extra)
