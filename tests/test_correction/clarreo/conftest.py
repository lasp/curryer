"""Pytest configuration for CLARREO integration tests.

Exposes session-scoped path fixtures for the CLARREO test data directories.
The ``clarreo/`` directory is already on ``sys.path`` via the parent
``test_correction/conftest.py``, so test files here can import
``clarreo_config`` and ``clarreo_data_loaders`` without any additional
``sys.path`` manipulation.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def clarreo_root():
    return Path(__file__).parents[3]


@pytest.fixture(scope="session")
def clarreo_gcs_data_dir(clarreo_root):
    return clarreo_root / "tests" / "data" / "clarreo" / "gcs"


@pytest.fixture(scope="session")
def clarreo_image_match_data_dir(clarreo_root):
    return clarreo_root / "tests" / "data" / "clarreo" / "image_match"


@pytest.fixture(scope="session")
def clarreo_generic_dir(clarreo_root):
    return clarreo_root / "data" / "generic"
