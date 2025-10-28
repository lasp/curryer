"""leapsecond - Unit test

@author: Brandon Stone
"""

import logging
import os
import tempfile
import unittest
from importlib import reload
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import requests
from requests import exceptions

from curryer import utils
from curryer.spicetime import leapsecond
from curryer.spicierpy import kclear, unitim

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


def is_url_dependency_available(url):
    """Check if a URL can be accessed."""
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception("Exception raised when accessing the url:")
        logger.error("The required URL is inaccessible, this test can not continue!")
        logger.error("URL dependency: %r", url)
        return False


class LeapsecondTestCase(unittest.TestCase):
    """Test the leapsecond module under real conditions.

    NOTE: Two tests use the kernel pool (but don't modify it), and two tests
    make real URL requests to the NAIF website (but don't modify real files).
    The URL tests will be skipped if the site is unavailable.

    """

    def setUp(self):
        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

    def test_find_default_file_for_leapsecond_kernel(self):
        fn = leapsecond.find_default_file()
        self.assertIsInstance(fn, Path)
        self.assertTrue(fn.is_file())

    def test_import_does_not_load_the_leapsecond_kernel(self):
        # Unload any previously loaded kernels.
        kclear()
        # Reload the module, just in case a bad test cleared the kernel pool.
        reload(leapsecond)
        self.assertFalse(leapsecond.are_loaded())

    def test_load_actually_works(self):
        leapsecond.load()
        self.assertTrue(leapsecond.are_loaded())

        tai = unitim(0, "et", "tai")
        self.assertAlmostEqual(-32.183927, tai, places=6)

    @unittest.skipUnless(
        is_url_dependency_available(leapsecond.LEAPSECOND_BASE_URL), "URL dependency is not available!"
    )
    def test_check_for_update_and_included_kernel_is_uptodate(self):
        # WARNING: This test depends on the following URL:
        #   https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/
        r = leapsecond.check_for_update()
        self.assertIsNone(r, "The modules leapsecond kernel is out of date!")

    @patch.object(leapsecond.sp, "furnsh", autospec=True)
    @unittest.skipUnless(
        is_url_dependency_available(leapsecond.LEAPSECOND_BASE_URL), "URL dependency is not available!"
    )
    def test_update_an_outdated_kernel_file(self, mock_furnsh):
        # WARNING: This test depends on the following URL:
        #   https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/
        # NOTE: The `fursh` routine is mocked to prevent changes to the kernel
        #   pool.

        # Make the default look like it's very old and needs updating.
        outdated_fn = Path(self.tmp_dir, "naif0001.tls")
        with patch.object(leapsecond, "find_default_file", return_value=outdated_fn):
            fn = leapsecond.update_file()
            self.assertIsInstance(fn, Path)
            self.assertTrue(fn.is_file())

        with patch.object(leapsecond, "_LEAPSECOND_FILE_PATH", self.tmp_dir):
            default_file = leapsecond.find_default_file()
            self.assertTrue(default_file.samefile(fn), "Module did not find the new file!")

        # Check that it tried to load the new kernel.
        mock_furnsh.assert_called_once_with(str(fn))


class LeapsecondFakeTestCase(unittest.TestCase):
    """Test the leapsecond module under fake (mocked) conditions."""

    def setUp(self):
        # The "furnsh" routine controls loading kernels into the pool.
        #   Mock it so we know how it's called.
        patcher = patch.object(leapsecond.sp, "furnsh", autospec=True)
        self.addCleanup(patcher.stop)
        self.mock_furnsh = patcher.start()

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

    def test_find_default_file_warns_user_if_kernel_is_older_than_2_yrs(self, *_):
        # Create a fake, very old, file (mtime).
        fake_file = Path(self.tmp_dir, "naif0001.tls")
        fake_file.touch()
        os.utime(str(fake_file), (0, 0))

        with patch.object(leapsecond, "_LEAPSECOND_FILE_PATH", fake_file.parent):
            with self.assertLogs(leapsecond.__name__, level="DEBUG") as logs:
                leapsecond.find_default_file()
            self.assertIn("The leapsecond kernel is older than two years", "\n".join(logs.output))

    @patch.object(leapsecond.sp, "ktotal", return_value=0, autospec=True)
    def test_are_loaded_empty(self, *_):
        r = leapsecond.are_loaded()

        self.assertIsInstance(r, list)
        self.assertEqual(0, len(r))

    @patch.object(leapsecond.sp, "ktotal", return_value=1, autospec=True)
    @patch.object(leapsecond.sp, "kdata", return_value=["fake.tls"], autospec=True)
    @patch.object(leapsecond.sp, "getfat", return_value=(None, "LSK"), autospec=True)
    def test_are_loaded_one(self, *_):
        r = leapsecond.are_loaded()

        self.assertIsInstance(r, list)
        self.assertEqual(1, len(r))
        self.assertEqual("fake.tls", r[0])

    @patch.object(leapsecond.sp, "ktotal", return_value=2, autospec=True)
    @patch.object(leapsecond.sp, "kdata", return_value=["fake.tls"], autospec=True)
    @patch.object(leapsecond.sp, "getfat", return_value=(None, "LSK"), autospec=True)
    def test_are_loaded_two(self, *_):
        r = leapsecond.are_loaded()

        self.assertIsInstance(r, list)
        self.assertEqual(2, len(r))
        self.assertEqual("fake.tls", r[0])
        self.assertEqual("fake.tls", r[1])

    def test_load_given_file_once(self):
        self.assertFalse(self.mock_furnsh.called)
        leapsecond.load("fake.tls")
        self.mock_furnsh.assert_called_once_with("fake.tls")

    def test_load_given_file_twice_allowing_reloading(self):
        self.assertFalse(self.mock_furnsh.called)
        leapsecond.load("fake.tls")
        leapsecond.load("fake.tls")
        self.mock_furnsh.assert_has_calls([call("fake.tls"), call("fake.tls")])

    @patch.object(leapsecond, "are_loaded", autospec=True)
    def test_load_default_file_if_no_lsk_are_loaded(self, mock_are_loaded):
        self.assertFalse(self.mock_furnsh.called)

        # Won't load the default leapsecond kernel because one is loaded.
        mock_are_loaded.return_value = ["fake.tls"]
        leapsecond.load()
        self.assertFalse(self.mock_furnsh.called)

        # Will load the default leapsecond kernel because none are loaded.
        mock_are_loaded.return_value = []
        leapsecond.load()
        default_file = leapsecond.find_default_file()
        self.mock_furnsh.assert_called_once_with(str(default_file))

    def test_read_leapseconds_default(self):
        leaps = leapsecond.read_leapseconds()
        self.assertIsInstance(leaps, pd.DataFrame)
        self.assertGreaterEqual(28, leaps.shape[0])
        self.assertEqual(4, leaps.shape[1])

        # Only check values prior to when the test is written.
        last_date = pd.to_datetime("2018-05-08")
        known_leaps = leaps.loc[:last_date, :]
        self.assertEqual(28, known_leaps.shape[0])

        self.assertEqual(pd.to_datetime("1972-01-01"), known_leaps.index[0])
        self.assertEqual(pd.to_datetime("2017-01-01"), known_leaps.index[-1])

        self.assertListEqual(["nsec", "unix", "ugps", "offset"], leaps.columns.tolist())
        self.assertListEqual([10, 63072000000000, -252892809000000, 315964809000000], known_leaps.iloc[0].tolist())
        self.assertListEqual([37, 1483228800000000, 1167264018000000, 315964782000000], known_leaps.iloc[-1].tolist())

    def test_read_leapseconds_specify_filename(self):
        fake_fn = str(self.tmp_dir / "test1")
        with open(fake_fn, "w") as f:
            f.write("""Header ignored...
            \\begindata
            Other data ignored...

            The real data...
            DELTET/DELTA_AT = ( 10,   @1972-JAN-1
                                11,   @1972-JUL-1 )

            \\begintext
            Foot ignored...
            """)
        leaps = leapsecond.read_leapseconds(fake_fn)
        self.assertIsInstance(leaps, pd.DataFrame)
        self.assertTupleEqual((2, 4), leaps.shape)

        self.assertListEqual([pd.to_datetime("1972-01-01"), pd.to_datetime("1972-07-01")], leaps.index.tolist())
        self.assertListEqual(["nsec", "unix", "ugps", "offset"], leaps.columns.tolist())
        self.assertListEqual([10, 63072000000000, -252892809000000, 315964809000000], leaps.iloc[0].tolist())
        self.assertListEqual([11, 78796800000000, -237168008000000, 315964808000000], leaps.iloc[1].tolist())


class UpdateLeapsecondTestCase(unittest.TestCase):
    """Test updating the leapsecond kernel under fake (mocked) conditions."""

    def setUp(self):
        # Mock the requests response.
        self.mock_resp = MagicMock(
            spec=requests.models.Response,
            text='href="naif0001.tls"\nhref="naif0002.tls"\nhref="readme.txt"',
        )

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

    @patch.object(leapsecond, "find_default_file", return_value=Path("naif0002.tls"))
    def test_check_for_update_but_no_update_needed(self, mock_find_default_file):
        # The return will only be None if `leapsecond.LEAPSECOND_FILE` matches
        #   the latest name (highest number), found on the page/text.
        with patch.object(leapsecond.requests, "get", autospec=True, return_value=self.mock_resp) as mock_get:
            r = leapsecond.check_for_update()

        self.assertIsNone(r, "Failed to match default file with latest file on the fake page.")
        mock_get.assert_called_once_with(leapsecond.LEAPSECOND_BASE_URL, timeout=10)
        self.mock_resp.raise_for_status.assert_called_once_with()

    @patch.object(leapsecond, "find_default_file", return_value=Path("naif0001.tls"))
    def test_check_for_update_can_update(self, mock_find_default_file):
        with patch.object(leapsecond.requests, "get", autospec=True, return_value=self.mock_resp) as mock_get:
            r = leapsecond.check_for_update()

        self.assertIsInstance(r, str)
        self.assertEqual(r, "naif0002.tls")
        mock_get.assert_called_once_with(leapsecond.LEAPSECOND_BASE_URL, timeout=10)
        self.mock_resp.raise_for_status.assert_called_once_with()

    def test_check_for_update_fails_with_bad_url(self):
        # Create a response with a bad status code.
        resp = requests.models.Response()
        resp.status_code = 404

        with patch.object(leapsecond.requests, "get", autospec=True, return_value=resp):
            with self.assertRaises(exceptions.HTTPError):
                leapsecond.check_for_update()

    def test_check_for_update_fails_with_no_files_on_page(self):
        # Change the fake page content to have no valid file names.
        self.mock_resp.text = 'href="readme.txt"'

        with patch.object(leapsecond.requests, "get", autospec=True, return_value=self.mock_resp):
            with self.assertRaises(ValueError):
                leapsecond.check_for_update()

    @patch.object(leapsecond, "check_for_update", return_value=None)
    def test_update_file_no_update_needed(self, mock_check_for_update):
        # Mock check_for_updates to indicate no updates are needed.
        #   Because there are no updates, it should not query the url for data.
        with patch.object(leapsecond.requests, "get", side_effect=AssertionError):
            r = leapsecond.update_file()

        self.assertIsNone(r)
        mock_check_for_update.assert_called_once_with()

    @patch.object(leapsecond.sp, "furnsh", autospec=True)
    def test_update_file_is_outdated(self, mock_furnsh):
        new_kernel = Path(self.tmp_dir, "naif0002.tls")

        # Use the real default kernel file to fake the text that is downloaded.
        self.mock_resp.text = leapsecond.find_default_file().read_text()

        # Get fake data instead of requesting it from the website.
        with patch.object(leapsecond.requests, "get", autospec=True, return_value=self.mock_resp):
            # Make it appear like we need to update.
            with patch.object(leapsecond, "check_for_update", return_value=new_kernel.name):
                # Make the default look like the temp file so it uses the same path.
                #   It's not an issue since it doesn't exist (yet).
                with patch.object(leapsecond, "find_default_file", return_value=new_kernel):
                    fn = leapsecond.update_file()
                    self.assertIsInstance(fn, Path)
                    default_file = leapsecond.find_default_file()
                    self.assertEqual(default_file, fn, "Module didn't find the new file!")

        self.assertTrue(fn.is_file())
        self.assertTrue(new_kernel.samefile(fn))

        # Check that it tried to load the new kernel.
        mock_furnsh.assert_called_once_with(str(new_kernel))

        new_txt = fn.read_text()
        self.assertEqual(self.mock_resp.text, new_txt, "Kernel text doesn't match fake web data!")

    @patch.object(leapsecond.sp, "furnsh", autospec=True)
    def test_update_file_fails_web_data_not_an_lsk_kernel(self, mock_furnsh):
        new_kernel = Path(self.tmp_dir, "naif0002.tls")

        # Get fake, but INCORRECT, data instead of requesting it from the website.
        with patch.object(leapsecond.requests, "get", autospec=True, return_value=self.mock_resp):
            # Make it appear like we need to update.
            with patch.object(leapsecond, "check_for_update", return_value=new_kernel.name):
                # Make the default look like the temp file so it uses the same path.
                #   It's not an issues since it doesn't exist (yet).
                with patch.object(leapsecond, "find_default_file", return_value=new_kernel):
                    # Fails because it wasn't a valid kernel.
                    with self.assertRaises(AssertionError):
                        leapsecond.update_file()

        self.assertFalse(mock_furnsh.called, "Tried to load the invalid kernel.")

    @patch.object(leapsecond.sp, "furnsh", autospec=True)
    def test_update_file_fails_if_new_file_exists(self, mock_furnsh):
        # Note that this creates the file, which we want to fail the test.
        new_kernel = Path(self.tmp_dir, "naif0002.tls")
        new_kernel.touch()

        # Fail if we make it far enough to make a request.
        with patch.object(leapsecond.requests, "get", autospec=True, side_effect=AssertionError):
            # Make it appear like we need to update. Use the filename from the
            #   temp file (i.e., random chars + 'naif0002.tls').
            with patch.object(leapsecond, "check_for_update", return_value=new_kernel.name):
                # Make the default look like the temp file so it uses the same path.
                #   It WILL fail because we didn't delete the temp file.
                # with patch.object(leapsecond, 'LEAPSECOND_FILE', new_kernel):
                with patch.object(leapsecond, "find_default_file", return_value=new_kernel):
                    # Fails because it wasn't a valid kernel.
                    with self.assertRaises(FileExistsError):
                        leapsecond.update_file()

        self.assertFalse(mock_furnsh.called, "Tried to load the existing kernel.")


if __name__ == "__main__":
    unittest.main()
