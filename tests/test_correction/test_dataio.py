"""Tests for ``curryer.correction.dataio`` (generic, no AWS credentials needed).

CLARREO-specific S3 integration tests live in ``clarreo/test_clarreo_dataio.py``.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import pytest

from curryer.correction.dataio import (
    S3Configuration,
    download_netcdf_objects,
    find_netcdf_objects,
    validate_science_output,
    validate_telemetry_output,
)

logger = logging.getLogger(__name__)


# ── FakeS3Client ──────────────────────────────────────────────────────────────


class FakeS3Client:
    """Minimal in-memory S3 mock."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.list_calls: list = []
        self.download_calls: list = []

    def list_objects_v2(self, **kwargs):
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix", "")
        self.list_calls.append((bucket, prefix))
        contents = [{"Key": k} for k in sorted(self.objects) if k.startswith(prefix)]
        return {"Contents": contents, "IsTruncated": False}

    def download_file(self, bucket, key, filename):
        self.download_calls.append((bucket, key, filename))
        Path(filename).write_bytes(self.objects[key])


# ── find / download ───────────────────────────────────────────────────────────


def test_find_netcdf_objects_filters_and_matches_prefix():
    config = S3Configuration("test-bucket", "L1a/nadir")
    objects = {
        "L1a/nadir/20181225/file1.nc": b"data1",
        "L1a/nadir/20181225/file2.txt": b"ignored",
        "L1a/nadir/20181226/file3.nc": b"data3",
        "L1a/nadir/20181227/file4.nc": b"out_of_range",
    }
    client = FakeS3Client(objects)
    keys = find_netcdf_objects(
        config, start_date=dt.date(2018, 12, 25), end_date=dt.date(2018, 12, 26), s3_client=client
    )
    assert keys == ["L1a/nadir/20181225/file1.nc", "L1a/nadir/20181226/file3.nc"]
    assert client.list_calls == [
        ("test-bucket", "L1a/nadir/20181225/"),
        ("test-bucket", "L1a/nadir/20181226/"),
    ]


def test_download_netcdf_objects_writes_files(tmp_path):
    config = S3Configuration("test-bucket", "L1a/nadir")
    objects = {
        "L1a/nadir/20181225/file1.nc": b"data1",
        "L1a/nadir/20181225/file2.nc": b"data2",
    }
    client = FakeS3Client(objects)
    output_paths = download_netcdf_objects(config, objects.keys(), tmp_path, s3_client=client)
    assert {p.name for p in output_paths} == {"file1.nc", "file2.nc"}
    for p in output_paths:
        assert p.read_bytes() == objects[f"L1a/nadir/20181225/{p.name}"]


# ── validation ────────────────────────────────────────────────────────────────


class MockConfig:
    class MockGeo:
        time_field = "corrected_timestamp"

    def __init__(self):
        self.geo = self.MockGeo()


@pytest.fixture
def mock_config():
    return MockConfig()


def test_validate_telemetry_valid(mock_config):
    df = pd.DataFrame({"time": [1.0, 2.0], "position_x": [100.0, 200.0]})
    validate_telemetry_output(df, mock_config)  # must not raise


def test_validate_telemetry_not_dataframe(mock_config):
    with pytest.raises(TypeError, match="must return pd.DataFrame"):
        validate_telemetry_output({"not": "a dataframe"}, mock_config)


def test_validate_telemetry_empty(mock_config):
    with pytest.raises(ValueError, match="empty DataFrame"):
        validate_telemetry_output(pd.DataFrame(), mock_config)


def test_validate_science_valid(mock_config):
    df = pd.DataFrame({"corrected_timestamp": [1e6, 2e6], "frame_id": [1, 2]})
    validate_science_output(df, mock_config)  # must not raise


def test_validate_science_not_dataframe(mock_config):
    with pytest.raises(TypeError, match="must return pd.DataFrame"):
        validate_science_output([1, 2, 3], mock_config)


def test_validate_science_empty(mock_config):
    with pytest.raises(ValueError, match="empty DataFrame"):
        validate_science_output(pd.DataFrame(), mock_config)


def test_validate_science_missing_time_field(mock_config):
    df = pd.DataFrame({"frame_id": [1, 2], "other": [100, 200]})
    with pytest.raises(ValueError, match="must include time field 'corrected_timestamp'"):
        validate_science_output(df, mock_config)


def test_validate_science_custom_time_field(mock_config):
    mock_config.geo.time_field = "custom_time"
    df_ok = pd.DataFrame({"custom_time": [1.0, 2.0], "data": [1, 2]})
    validate_science_output(df_ok, mock_config)  # must not raise
    df_bad = pd.DataFrame({"corrected_timestamp": [1.0, 2.0], "data": [1, 2]})
    with pytest.raises(ValueError, match="must include time field 'custom_time'"):
        validate_science_output(df_bad, mock_config)
