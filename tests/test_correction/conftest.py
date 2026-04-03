"""Generic pytest configuration for test_correction.

- Adds ``clarreo/`` sub-directory to ``sys.path`` so test files in this package
  can import ``clarreo_config``, ``clarreo_data_loaders``, etc. without
  repeating ``sys.path`` manipulation in every file.
- Adds this directory itself to ``sys.path`` so ``_synthetic_helpers`` is
  importable by the ``clarreo/`` sub-package helpers.
"""

import sys
from pathlib import Path

import pytest

_here = str(Path(__file__).parent)
_clarreo_dir = str(Path(__file__).parent / "clarreo")

for _p in (_here, _clarreo_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def root_dir():
    """Repository root directory (two levels above ``tests/test_correction/``)."""
    return Path(__file__).parents[2]


@pytest.fixture
def temp_work_dir(tmp_path):
    """Clean temporary working directory for each test."""
    work = tmp_path / "correction_work"
    work.mkdir(parents=True, exist_ok=True)
    return work


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_gcs: marks tests requiring GCS credentials")
