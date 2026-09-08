"""kernels.cache - Unit test

@author: Matthew Maclay
"""

import datetime
import logging
import os
import tempfile
import time
import unittest
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch

import boto3
import requests
import responses
from moto import mock_aws

from curryer import utils
from curryer.kernels import cache

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])

HTTP_URL = "https://naif.example.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
S3_URI = "s3://test-bucket/kernels/naif0012.tls"


def _age_entry(entry: Path, age: datetime.timedelta) -> None:
    """Backdate a cache entry's mtime so it exceeds a max age."""
    old = time.time() - age.total_seconds()
    os.utime(entry, (old, old))


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        self.__tmp_dir = tempfile.TemporaryDirectory(prefix="/tmp/")
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)
        self.cache_dir = self.tmp_dir / "cache"

        self.source_dir = self.tmp_dir / "source"
        self.source_dir.mkdir()
        self.local_source = self.source_dir / "local_kernel.tls"
        self.local_source.write_bytes(b"LOCAL KERNEL DATA")

    def _moto_s3(self, payload: bytes):
        """Create the mocked bucket/object behind S3_URI (inside @mock_aws)."""
        env = patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": "us-east-1"},
        )
        env.start()
        self.addCleanup(env.stop)
        client = boto3.client("s3")
        client.create_bucket(Bucket="test-bucket")
        client.put_object(Bucket="test-bucket", Key="kernels/naif0012.tls", Body=payload)
        return client

    def test_local_copy(self):
        entry = cache.fetch(self.local_source, cache_dir=self.cache_dir)
        self.assertEqual(self.cache_dir / "local_kernel.tls", entry)
        self.assertEqual(b"LOCAL KERNEL DATA", entry.read_bytes())

    def test_local_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            cache.fetch(self.source_dir / "no_such_kernel.tls", cache_dir=self.cache_dir)

    @responses.activate
    def test_http_download(self):
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")
        entry = cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
        self.assertEqual(self.cache_dir / "naif0012.tls", entry)
        self.assertEqual(b"HTTP KERNEL DATA", entry.read_bytes())

    @responses.activate
    def test_http_download_retries_with_backoff(self):
        responses.add(responses.GET, HTTP_URL, body=requests.exceptions.ConnectionError("first attempt"))
        responses.add(responses.GET, HTTP_URL, body=requests.exceptions.ConnectionError("second attempt"))
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")
        with patch("time.sleep") as mock_sleep:
            entry = cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
        self.assertEqual(b"HTTP KERNEL DATA", entry.read_bytes())
        self.assertEqual([1, 5], [call.args[0] for call in mock_sleep.call_args_list])

    @mock_aws
    def test_s3_download(self):
        self._moto_s3(b"S3 KERNEL DATA")
        entry = cache.fetch(S3_URI, cache_dir=self.cache_dir)
        self.assertEqual(self.cache_dir / "naif0012.tls", entry)
        self.assertEqual(b"S3 KERNEL DATA", entry.read_bytes())

    @mock_aws
    def test_s3_stale_size_match_revalidates_without_download(self):
        client = self._moto_s3(b"S3 KERNEL DATA")
        entry = cache.fetch(S3_URI, cache_dir=self.cache_dir)
        _age_entry(entry, datetime.timedelta(days=2))

        # Same size, different bytes: a size-based revalidation keeps the
        # cached copy, so seeing the old bytes proves no re-download ran.
        client.put_object(Bucket="test-bucket", Key="kernels/naif0012.tls", Body=b"S3 ALTERED DATA!"[:14])
        again = cache.fetch(S3_URI, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)

        self.assertEqual(entry, again)
        self.assertEqual(b"S3 KERNEL DATA", again.read_bytes())
        # The mtime refresh makes the entry warm again.
        self.assertLess(time.time() - again.stat().st_mtime, 60)

    @mock_aws
    def test_s3_stale_size_mismatch_redownloads(self):
        client = self._moto_s3(b"S3 KERNEL DATA")
        entry = cache.fetch(S3_URI, cache_dir=self.cache_dir)
        _age_entry(entry, datetime.timedelta(days=2))

        client.put_object(Bucket="test-bucket", Key="kernels/naif0012.tls", Body=b"S3 NEW ROLLING KERNEL DATA")
        again = cache.fetch(S3_URI, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)

        self.assertEqual(entry, again)
        self.assertEqual(b"S3 NEW ROLLING KERNEL DATA", again.read_bytes())

    def test_source_without_basename_raises(self):
        for source in ("https://naif.example.gov/pub/naif/generic_kernels/lsk/", "s3://test-bucket/"):
            with self.assertRaises(ValueError) as context:
                cache.fetch(source, cache_dir=self.cache_dir)
            self.assertIn("no file basename", str(context.exception))

    @responses.activate
    def test_warm_hit_no_network(self):
        # Nothing registered with responses: any network call errors out, and
        # a fallback to a stale copy would emit a warning.
        entry = cache.fetch(self.local_source, cache_dir=self.cache_dir)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            again = cache.fetch(HTTP_URL.replace("naif0012.tls", "local_kernel.tls"), cache_dir=self.cache_dir)
        self.assertEqual([], [warning for warning in caught if issubclass(warning.category, UserWarning)])
        self.assertEqual(entry, again)
        self.assertEqual(b"LOCAL KERNEL DATA", again.read_bytes())

    @responses.activate
    def test_stale_size_match_revalidates_without_download(self):
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")
        entry = cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
        _age_entry(entry, datetime.timedelta(days=2))

        responses.reset()
        responses.add(responses.HEAD, HTTP_URL, headers={"Content-Length": str(entry.stat().st_size)})
        again = cache.fetch(HTTP_URL, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)

        self.assertEqual(entry, again)
        self.assertEqual(b"HTTP KERNEL DATA", again.read_bytes())
        # The mtime refresh makes the entry warm again.
        self.assertLess(time.time() - again.stat().st_mtime, 60)

    @responses.activate
    def test_stale_size_mismatch_redownloads(self):
        responses.add(responses.GET, HTTP_URL, body=b"OLD DATA")
        entry = cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
        _age_entry(entry, datetime.timedelta(days=2))

        responses.reset()
        new_payload = b"NEW ROLLING KERNEL DATA"
        responses.add(responses.HEAD, HTTP_URL, headers={"Content-Length": str(len(new_payload))})
        responses.add(responses.GET, HTTP_URL, body=new_payload)
        again = cache.fetch(HTTP_URL, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)

        self.assertEqual(entry, again)
        self.assertEqual(new_payload, again.read_bytes())

    def test_per_entry_max_age(self):
        entry = cache.fetch(self.local_source, cache_dir=self.cache_dir)
        _age_entry(entry, datetime.timedelta(hours=12))
        self.local_source.write_bytes(b"CHANGED SOURCE DATA")

        # A long max age keeps the stale-but-younger entry warm...
        warm = cache.fetch(self.local_source, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)
        self.assertEqual(b"LOCAL KERNEL DATA", warm.read_bytes())

        # ...while a short one on the same entry triggers revalidation.
        fresh = cache.fetch(self.local_source, max_age=datetime.timedelta(hours=1), cache_dir=self.cache_dir)
        self.assertEqual(b"CHANGED SOURCE DATA", fresh.read_bytes())

    @responses.activate
    def test_interrupted_download_leaves_no_entry(self):
        responses.add(responses.GET, HTTP_URL, body=requests.exceptions.ConnectionError("mid-transfer failure"))
        with patch("time.sleep"):
            with self.assertRaises(requests.exceptions.RequestException):
                cache.fetch(HTTP_URL, cache_dir=self.cache_dir)

        self.assertFalse((self.cache_dir / "naif0012.tls").exists())
        self.assertEqual([], [f for f in self.cache_dir.iterdir() if f.is_file()])

    @responses.activate
    def test_network_failure_returns_stale_with_warning(self):
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")
        entry = cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
        _age_entry(entry, datetime.timedelta(days=2))

        responses.reset()
        responses.add(responses.HEAD, HTTP_URL, body=requests.exceptions.ConnectionError("no route to NAIF"))
        with self.assertWarns(UserWarning) as caught:
            again = cache.fetch(HTTP_URL, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)

        self.assertEqual(entry, again)
        self.assertEqual(b"HTTP KERNEL DATA", again.read_bytes())
        self.assertIn("stale cached copy", str(caught.warning))

    @responses.activate
    def test_failed_redownload_returns_stale_with_warning(self):
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")
        entry = cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
        _age_entry(entry, datetime.timedelta(days=2))

        responses.reset()
        responses.add(responses.HEAD, HTTP_URL, headers={"Content-Length": "9999"})
        responses.add(responses.GET, HTTP_URL, body=requests.exceptions.ConnectionError("dropped mid-download"))
        with patch("time.sleep"):
            with self.assertWarns(UserWarning) as caught:
                again = cache.fetch(HTTP_URL, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)

        self.assertEqual(b"HTTP KERNEL DATA", again.read_bytes())
        self.assertIn("stale cached copy", str(caught.warning))

    @responses.activate
    def test_normal_flow_logs_debug_never_warns(self):
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs(cache.logger, level="DEBUG") as miss_logs:
                cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
            with self.assertLogs(cache.logger, level="DEBUG") as hit_logs:
                cache.fetch(HTTP_URL, cache_dir=self.cache_dir)

        self.assertEqual([], [warning for warning in caught if issubclass(warning.category, UserWarning)])
        self.assertTrue(any("Cached" in message for message in miss_logs.output))
        self.assertTrue(any("Cache hit" in message for message in hit_logs.output))

    def test_clear_cache(self):
        entry = cache.fetch(self.local_source, cache_dir=self.cache_dir)
        # A tmp cache dir is only clearable when it is the package's own:
        # point the cache root at it.
        with patch("curryer.kernels.cache.get_local_cache_dir", return_value=self.cache_dir):
            removed = cache.clear_cache(cache_dir=self.cache_dir)
        self.assertEqual([entry], removed)
        self.assertFalse(entry.exists())

    def test_clear_cache_refuses_foreign_directory(self):
        important = self.tmp_dir / "important"
        important.mkdir()
        keeper = important / "data.txt"
        keeper.write_text("do not delete")

        with self.assertRaises(ValueError) as context:
            cache.clear_cache(cache_dir=important)
        self.assertIn("cache root", str(context.exception))
        self.assertTrue(keeper.exists())

    def test_get_local_cache_dir_version_keyed(self):
        cache_dir = cache.get_local_cache_dir()
        self.assertEqual(version(utils.DISTRIBUTION_NAME), cache_dir.name)
        self.assertEqual("curryer", cache_dir.parent.name)

    def test_package_version_never_raises(self):
        # The version keys the cache directory, so source checkouts without
        # an installed distribution must still resolve to something.
        with patch("curryer.utils.version", side_effect=PackageNotFoundError):
            self.assertEqual("unknown", utils.package_version())


if __name__ == "__main__":
    unittest.main()
