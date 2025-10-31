"""
Tests for dataio.py module

This module tests data I/O functionality:
- S3 object discovery and download
- NetCDF file handling
- Configuration management
- File path operations

Running Tests:
-------------
# Via pytest (recommended)
pytest tests/test_correction/test_dataio.py -v

# Run specific test
pytest tests/test_correction/test_dataio.py::DataIOTestCase::test_find_objects -v

# Standalone execution
python tests/test_correction/test_dataio.py

Notes:
-----
These tests use mock S3 clients to avoid requiring AWS credentials
or network access during testing.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
import unittest
from pathlib import Path

from curryer.correction.dataio import (
    S3Configuration,
    download_netcdf_objects,
    find_netcdf_objects,
)


class FakeS3Client:
    def __init__(self, objects):
        self.objects = objects  # dict: key -> bytes
        self.list_calls = []
        self.download_calls = []

    def list_objects_v2(self, **kwargs):
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix", "")
        self.list_calls.append((bucket, prefix))
        contents = []
        for key in sorted(self.objects):
            if key.startswith(prefix):
                contents.append({"Key": key})
        return {"Contents": contents, "IsTruncated": False}

    def download_file(self, bucket, key, filename):
        self.download_calls.append((bucket, key, filename))
        Path(filename).write_bytes(self.objects[key])


class DataIOTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

    def test_find_netcdf_objects_filters_and_matches_prefix(self):
        config = S3Configuration("test-bucket", "L1a/nadir")
        objects = {
            "L1a/nadir/20181225/file1.nc": b"data1",
            "L1a/nadir/20181225/file2.txt": b"ignored",
            "L1a/nadir/20181226/file3.nc": b"data3",
            "L1a/nadir/20181227/file4.nc": b"out_of_range",
        }
        client = FakeS3Client(objects)

        keys = find_netcdf_objects(
            config,
            start_date=dt.date(2018, 12, 25),
            end_date=dt.date(2018, 12, 26),
            s3_client=client,
        )
        self.assertListEqual(
            keys,
            [
                "L1a/nadir/20181225/file1.nc",
                "L1a/nadir/20181226/file3.nc",
            ],
        )
        self.assertListEqual(
            client.list_calls,
            [
                ("test-bucket", "L1a/nadir/20181225/"),
                ("test-bucket", "L1a/nadir/20181226/"),
            ],
        )

    def test_download_netcdf_objects_writes_files(self):
        config = S3Configuration("test-bucket", "L1a/nadir")
        objects = {
            "L1a/nadir/20181225/file1.nc": b"data1",
            "L1a/nadir/20181225/file2.nc": b"data2",
        }
        client = FakeS3Client(objects)

        output_paths = download_netcdf_objects(
            config,
            objects.keys(),
            self.tmp_dir,
            s3_client=client,
        )

        self.assertSetEqual({p.name for p in output_paths}, {"file1.nc", "file2.nc"})
        for path in output_paths:
            self.assertEqual(path.read_bytes(), objects[f"L1a/nadir/20181225/{path.name}"])


@unittest.skipUnless(
    os.getenv("AWS_ACCESS_KEY_ID", "")
    and os.getenv("AWS_SECRET_ACCESS_KEY", "")
    and os.getenv("AWS_SESSION_TOKEN", ""),
    "Requires tester to set AWS access key environment variables.",
)
class ClarreoDataIOTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

    def test_l0(self):
        config = S3Configuration("clarreo", "L0/telemetry/hps_navigation/")
        keys = find_netcdf_objects(
            config,
            start_date=dt.date(2017, 1, 15),
            end_date=dt.date(2017, 1, 15),
        )
        self.assertListEqual(
            keys, ["L0/telemetry/hps_navigation/20170115/CPF_TLM_L0.V00-000.hps_navigation-20170115-0.0.0.nc"]
        )

    def test_l1a(self):
        config = S3Configuration("clarreo", "L1a/nadir/")
        keys = find_netcdf_objects(
            config,
            start_date=dt.date(2022, 6, 3),
            end_date=dt.date(2022, 6, 3),
        )
        self.assertEqual(34, len(keys))
        self.assertIn("L1a/nadir/20220603/nadir-20220603T235952-step22-geolocation_creation-0.0.0.nc", keys)


if __name__ == "__main__":
    unittest.main()
