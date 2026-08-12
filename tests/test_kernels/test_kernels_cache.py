"""kernels.cache - Unit test

@author: Matthew Maclay
"""

import datetime
import io
import logging
import os
import tempfile
import time
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import boto3
import requests
import responses
from botocore.response import StreamingBody
from botocore.stub import Stubber

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

    def _stubbed_s3(self, payload: bytes, expect_head: bool = False):
        """Create a stubbed S3 client and patch boto3 to return it."""
        client = boto3.client("s3", region_name="us-east-1")
        stubber = Stubber(client)
        params = {"Bucket": "test-bucket", "Key": "kernels/naif0012.tls"}
        if expect_head:
            stubber.add_response("head_object", {"ContentLength": len(payload)}, params)
        else:
            stubber.add_response(
                "get_object",
                {"Body": StreamingBody(io.BytesIO(payload), len(payload)), "ContentLength": len(payload)},
                params,
            )
        stubber.activate()
        self.addCleanup(stubber.deactivate)
        patcher = patch("boto3.client", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        return stubber

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
    def test_http_download_retries(self):
        responses.add(responses.GET, HTTP_URL, body=requests.exceptions.ConnectionError("first attempt"))
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")
        with patch("time.sleep"):
            entry = cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
        self.assertEqual(b"HTTP KERNEL DATA", entry.read_bytes())

    def test_s3_download(self):
        self._stubbed_s3(b"S3 KERNEL DATA")
        entry = cache.fetch(S3_URI, cache_dir=self.cache_dir)
        self.assertEqual(self.cache_dir / "naif0012.tls", entry)
        self.assertEqual(b"S3 KERNEL DATA", entry.read_bytes())

    @responses.activate
    def test_warm_hit_no_network(self):
        # Nothing registered with responses: any network call errors out, and
        # a fallback to a stale copy would emit a warning.
        entry = cache.fetch(self.local_source, cache_dir=self.cache_dir)

        with self.assertNoLogs(cache.logger, level="WARNING"):
            again = cache.fetch(HTTP_URL.replace("naif0012.tls", "local_kernel.tls"), cache_dir=self.cache_dir)
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
        with self.assertLogs(cache.logger, level="WARNING") as caught:
            again = cache.fetch(HTTP_URL, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)

        self.assertEqual(entry, again)
        self.assertEqual(b"HTTP KERNEL DATA", again.read_bytes())
        self.assertTrue(any("stale cached copy" in message for message in caught.output))

    @responses.activate
    def test_failed_redownload_returns_stale_with_warning(self):
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")
        entry = cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
        _age_entry(entry, datetime.timedelta(days=2))

        responses.reset()
        responses.add(responses.HEAD, HTTP_URL, headers={"Content-Length": "9999"})
        responses.add(responses.GET, HTTP_URL, body=requests.exceptions.ConnectionError("dropped mid-download"))
        with patch("time.sleep"):
            with self.assertLogs(cache.logger, level="WARNING") as caught:
                again = cache.fetch(HTTP_URL, max_age=datetime.timedelta(days=1), cache_dir=self.cache_dir)

        self.assertEqual(b"HTTP KERNEL DATA", again.read_bytes())
        self.assertTrue(any("stale cached copy" in message for message in caught.output))

    @responses.activate
    def test_normal_flow_logs_debug_never_warns(self):
        responses.add(responses.GET, HTTP_URL, body=b"HTTP KERNEL DATA")

        with self.assertNoLogs(cache.logger, level="WARNING"):
            with self.assertLogs(cache.logger, level="DEBUG") as miss_logs:
                cache.fetch(HTTP_URL, cache_dir=self.cache_dir)
            with self.assertLogs(cache.logger, level="DEBUG") as hit_logs:
                cache.fetch(HTTP_URL, cache_dir=self.cache_dir)

        self.assertTrue(any("Cached" in message for message in miss_logs.output))
        self.assertTrue(any("Cache hit" in message for message in hit_logs.output))

    def test_clear_cache(self):
        entry = cache.fetch(self.local_source, cache_dir=self.cache_dir)
        removed = cache.clear_cache(cache_dir=self.cache_dir)
        self.assertEqual([entry], removed)
        self.assertFalse(entry.exists())

    def test_get_local_cache_dir_version_keyed(self):
        cache_dir = cache.get_local_cache_dir()
        self.assertEqual(version("curryer"), cache_dir.name)
        self.assertEqual("curryer", cache_dir.parent.name)


if __name__ == "__main__":
    unittest.main()
