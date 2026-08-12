"""kernels.discovery - Unit test

@author: Matthew Maclay
"""

import logging
import unittest
from unittest.mock import patch

import requests
import responses

from curryer import utils
from curryer.kernels import discovery

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])

LSK_INDEX_URL = discovery.NAIF_GENERIC_KERNELS_URL + "lsk/"
PCK_INDEX_URL = discovery.NAIF_GENERIC_KERNELS_URL + "pck/"

# Abridged captures of NAIF's Apache index listings.
LSK_INDEX_HTML = """<html>
 <head><title>Index of /pub/naif/generic_kernels/lsk</title></head>
 <body>
<h1>Index of /pub/naif/generic_kernels/lsk</h1>
<pre><img src="/icons/blank.gif" alt="Icon "> <a href="?C=N;O=D">Name</a> <a href="?C=M;O=A">Last modified</a>
<img src="/icons/back.gif" alt="[PARENTDIR]"> <a href="/pub/naif/generic_kernels/">Parent Directory</a>
<img src="/icons/text.gif" alt="[TXT]"> <a href="aareadme.txt">aareadme.txt</a>            2019-05-28 09:26  1.1K
<img src="/icons/unknown.gif" alt="[   ]"> <a href="latest_leapseconds.tls">latest_leapseconds.tls</a> 2016-07-14
<img src="/icons/unknown.gif" alt="[   ]"> <a href="naif0011.tls">naif0011.tls</a>          2015-01-06 09:35  4.9K
<img src="/icons/unknown.gif" alt="[   ]"> <a href="naif0012.tls">naif0012.tls</a>          2016-07-14 10:20  5.1K
</pre></body></html>"""

PCK_INDEX_HTML = """<html>
 <head><title>Index of /pub/naif/generic_kernels/pck</title></head>
 <body>
<h1>Index of /pub/naif/generic_kernels/pck</h1>
<pre><img src="/icons/unknown.gif" alt="[   ]"> <a href="earth_000101_250316_241220.bpc">earth_000101_250316_241220.bpc</a>
<img src="/icons/unknown.gif" alt="[   ]"> <a href="earth_000101_250607_250314.bpc">earth_000101_250607_250314.bpc</a>
<img src="/icons/unknown.gif" alt="[   ]"> <a href="earth_1962_240827_2124_combined.bpc">earth_1962_240827_2124_combined.bpc</a>
<img src="/icons/unknown.gif" alt="[   ]"> <a href="earth_2000_200628_2100_predict.bpc">earth_2000_200628_2100_predict.bpc</a>
<img src="/icons/unknown.gif" alt="[   ]"> <a href="earth_latest_high_prec.bpc">earth_latest_high_prec.bpc</a>
<img src="/icons/unknown.gif" alt="[   ]"> <a href="pck00010.tpc">pck00010.tpc</a>
<img src="/icons/unknown.gif" alt="[   ]"> <a href="pck00011.tpc">pck00011.tpc</a>
</pre></body></html>"""


class DiscoveryTestCase(unittest.TestCase):
    @responses.activate
    def test_most_recent_selection(self):
        responses.add(responses.GET, LSK_INDEX_URL, body=LSK_INDEX_HTML)
        url = discovery.find_most_recent_naif_kernel(LSK_INDEX_URL, discovery.NAIF_LSK_REGEX)
        self.assertEqual(LSK_INDEX_URL + "naif0012.tls", url)

    @responses.activate
    def test_pck_patterns_select_within_shared_index(self):
        responses.add(responses.GET, PCK_INDEX_URL, body=PCK_INDEX_HTML)

        url = discovery.find_most_recent_naif_kernel(PCK_INDEX_URL, discovery.NAIF_PCK_REGEX)
        self.assertEqual(PCK_INDEX_URL + "pck00011.tpc", url)

        url = discovery.find_most_recent_naif_kernel(PCK_INDEX_URL, discovery.NAIF_HIGH_PREC_PCK_REGEX)
        self.assertEqual(PCK_INDEX_URL + "earth_000101_250607_250314.bpc", url)

        url = discovery.find_most_recent_naif_kernel(PCK_INDEX_URL, discovery.NAIF_EARTH_EXTENDED_PCK_REGEX)
        self.assertEqual(PCK_INDEX_URL + "earth_2000_200628_2100_predict.bpc", url)

    @responses.activate
    def test_no_match_raises(self):
        responses.add(responses.GET, LSK_INDEX_URL, body=LSK_INDEX_HTML)
        with self.assertRaises(ValueError) as context:
            discovery.find_most_recent_naif_kernel(LSK_INDEX_URL, discovery.NAIF_DE_REGEX)
        self.assertIn("No files matching", str(context.exception))

    @responses.activate
    def test_index_fetch_retries(self):
        responses.add(responses.GET, LSK_INDEX_URL, body=requests.exceptions.ConnectionError("first attempt"))
        responses.add(responses.GET, LSK_INDEX_URL, body=LSK_INDEX_HTML)
        with patch("time.sleep"):
            url = discovery.find_most_recent_naif_kernel(LSK_INDEX_URL, discovery.NAIF_LSK_REGEX)
        self.assertEqual(LSK_INDEX_URL + "naif0012.tls", url)

    @responses.activate
    def test_index_fetch_exhausted_raises(self):
        for _ in range(3):
            responses.add(responses.GET, LSK_INDEX_URL, body=requests.exceptions.ConnectionError("down"))
        with patch("time.sleep"):
            with self.assertRaises(requests.exceptions.RequestException):
                discovery.find_most_recent_naif_kernel(LSK_INDEX_URL, discovery.NAIF_LSK_REGEX)

    def test_subdir_routing(self):
        base = discovery.NAIF_GENERIC_KERNELS_URL
        self.assertEqual(base + "lsk/", discovery.naif_index_url(discovery.NAIF_LSK_REGEX))
        self.assertEqual(base + "pck/", discovery.naif_index_url(discovery.NAIF_PCK_REGEX))
        self.assertEqual(base + "pck/", discovery.naif_index_url(discovery.NAIF_HIGH_PREC_PCK_REGEX))
        self.assertEqual(base + "pck/", discovery.naif_index_url(discovery.NAIF_EARTH_EXTENDED_PCK_REGEX))
        self.assertEqual(base + "spk/planets/", discovery.naif_index_url(discovery.NAIF_DE_REGEX))
        self.assertEqual(base + "fk/planets/", discovery.naif_index_url(r"earth_assoc_itrf93\.tf"))

    def test_subdir_routing_unknown_extension_raises(self):
        with self.assertRaises(ValueError) as context:
            discovery.naif_index_url(r"kernel[0-9]{2}\.xyz")
        self.assertIn("xyz", str(context.exception))

    def test_alternate_base_url(self):
        self.assertEqual(
            "https://mirror.example.edu/naif/lsk/",
            discovery.naif_index_url(discovery.NAIF_LSK_REGEX, base_url="https://mirror.example.edu/naif/"),
        )

    def test_flat_base_url_skips_routing(self):
        self.assertEqual(
            "https://test.example.edu/kernels/",
            discovery.naif_index_url(discovery.NAIF_LSK_REGEX, base_url="https://test.example.edu/kernels/", flat=True),
        )

    @responses.activate
    def test_find_latest_kernel_url_end_to_end(self):
        responses.add(responses.GET, LSK_INDEX_URL, body=LSK_INDEX_HTML)
        url = discovery.find_latest_naif_kernel_url(discovery.NAIF_LSK_REGEX)
        self.assertEqual(LSK_INDEX_URL + "naif0012.tls", url)

    @responses.activate
    def test_find_latest_kernel_url_flat_mirror(self):
        mirror = "https://test.example.edu/kernels/"
        responses.add(responses.GET, mirror, body=LSK_INDEX_HTML)
        url = discovery.find_latest_naif_kernel_url(discovery.NAIF_LSK_REGEX, base_url=mirror, flat=True)
        self.assertEqual(mirror + "naif0012.tls", url)


if __name__ == "__main__":
    unittest.main()
