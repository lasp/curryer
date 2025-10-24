"""ext - Unit test

@author: Brandon Stone
"""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from curryer import meta, utils
from curryer.spicierpy import ext, obj

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class ExtTestCase(unittest.TestCase):
    def setUp(self):
        root_dir = Path(__file__).parents[2]
        self.generic_dir = root_dir / "data" / "generic"
        self.data_dir = root_dir / "data" / "tsis1"
        self.test_dir = root_dir / "tests" / "data" / "tsis1"
        self.assertTrue(self.generic_dir.is_dir())
        self.assertTrue(self.data_dir.is_dir())
        self.assertTrue(self.test_dir.is_dir())

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

        self.mkrn = meta.MetaKernel.from_json(
            self.test_dir / "tsis_v01.kernels.tm.json",
            relative=True,
            sds_dir=self.generic_dir,
        )
        self.kernels_all = self.mkrn.sds_kernels + self.mkrn.mission_kernels
        self.kernels = {
            "frame": [fn for fn in self.kernels_all if ".tf" == fn.suffix and "v01" in fn.name],
            "clock": [fn for fn in self.kernels_all if ".tsc" == fn.suffix and "v01" in fn.name],
            "ephemeris": [fn for fn in self.kernels_all if ".bsp" == fn.suffix and "v01" in fn.name],
            "attitude": [fn for fn in self.kernels_all if ".bc" == fn.suffix and "v01" in fn.name],
            "instrument": [fn for fn in self.kernels_all if ".ti" == fn.suffix and "v01" in fn.name],
        }

        # Test values.
        self.ugps_range = (1307354426293690, 1307361616166280)
        self.spice_name = "iss_sc"
        self.spice_id = -125544
        self.frame_name = "iss_issacs"
        self.frame_id = self.spice_id * 1000

    def test_load_kernel_arguments(self):
        # Note: Mock the kernel loading/unloading methods (furnsh, unload), to
        #   make testing easier.

        # Test context manager load a single kernel (str).
        with patch.object(ext, "spiceypy") as mock_spice:
            with ext.load_kernel("fake.k") as kern:
                self.assertListEqual(["fake.k"], kern.loaded)
                mock_spice.furnsh.assert_called_once_with("fake.k")
                mock_spice.unload.assert_not_called()
            mock_spice.unload.assert_called_once_with("fake.k")

        # Load list of kernels.
        with patch.object(ext, "spiceypy") as mock_spice:
            with ext.load_kernel(["fake1", "fake2"]) as kern:
                self.assertListEqual(["fake1", "fake2"], kern.loaded)
                mock_spice.furnsh.assert_has_calls([call("fake1"), call("fake2")])
                mock_spice.unload.assert_not_called()
            mock_spice.unload.assert_has_calls([call("fake2"), call("fake1")])  # Note reverse order!

        # Load dict of kernels (str or list).
        #   Note: Using ordered dict for easy of testing.
        fake_krn = dict([("grp1", "fake1"), ("grp2", ["fake2", "fake3"])])
        with patch.object(ext, "spiceypy") as mock_spice:
            with ext.load_kernel(fake_krn) as kern:
                self.assertListEqual(["fake1", "fake2", "fake3"], kern.loaded)
                mock_spice.furnsh.assert_has_calls([call("fake1"), call("fake2"), call("fake3")])
                mock_spice.unload.assert_not_called()
            mock_spice.unload.assert_has_calls([call("fake3"), call("fake2"), call("fake1")])  # Note reverse order!

        # Load dict of kernels (str or list), but load the key "meta" first.
        #   Note: Using ordered dict for easy of testing.
        fake_krn = dict(
            [
                ("aaa", "fake1"),
                ("meta", "fake2"),  # Should be loaded first, and unloaded last.
                ("zzz", ["fake3"]),
            ]
        )
        with patch.object(ext, "spiceypy") as mock_spice:
            with ext.load_kernel(fake_krn) as kern:
                self.assertListEqual(["fake2", "fake1", "fake3"], kern.loaded)
                mock_spice.furnsh.assert_has_calls([call("fake2"), call("fake1"), call("fake3")])
                mock_spice.unload.assert_not_called()
            mock_spice.unload.assert_has_calls([call("fake3"), call("fake1"), call("fake2")])

        # Alternatively, deleting the obj will unload kernels.
        with patch.object(ext, "spiceypy") as mock_spice:
            kern = ext.load_kernel(["fake1", "fake2"])
            self.assertListEqual(["fake1", "fake2"], kern.loaded)
            mock_spice.furnsh.assert_has_calls([call("fake1"), call("fake2")])
            mock_spice.unload.assert_not_called()
            del kern
            mock_spice.unload.assert_has_calls([call("fake2"), call("fake1")])

        # Value error if not a iter of 1+ str.
        with patch.object(ext, "spiceypy"):
            with self.assertRaises(ValueError) as raised:
                with ext.load_kernel(123):
                    pass
        self.assertIn("Invalid `kernels`: 123", raised.exception.args[0])

        # Value error if unloading kernel that is a kernel.
        with patch.object(ext, "spiceypy"):
            with self.assertRaises(ValueError) as raised:
                kern = ext.load_kernel("fake1")
                kern.unload(kernel=123)
        self.assertIn("Must specify `kernel` (str) or `all`=True.", raised.exception.args[0])

    def test_object_frame(self):
        # Load the kernels to activate name <--> code.
        #   Load frame kernel for name <--> code.
        with ext.load_kernel(self.kernels["frame"]):
            # Body name -> Frame name.
            frame_name = ext.object_frame(self.spice_name)
            self.assertEqual(self.frame_name, frame_name.lower())

            # Body name -> Frame id.
            frame_id = ext.object_frame(self.spice_name, as_id=True)
            self.assertEqual(self.frame_id, frame_id)

            # Body id -> Frame name.
            frame_name = ext.object_frame(self.spice_id)
            self.assertEqual(self.frame_name, frame_name.lower())

            # Body id -> Frame id.
            frame_id = ext.object_frame(self.spice_id, as_id=True)
            self.assertEqual(self.frame_id, frame_id)

    def test_kernel_coverage(self):
        # Load the kernels to activate name <--> code.
        #   Load clock kernel for CK coverage.
        #   Load frame kernel for name <--> code.
        with ext.load_kernel([self.kernels["clock"], self.kernels["frame"]]):
            # SPK kernel.
            ugps = ext.kernel_coverage(self.kernels["ephemeris"][-1], self.spice_name)
            self.assertTupleEqual(self.ugps_range, ugps)

            ugps = ext.kernel_coverage(self.kernels["ephemeris"][-1], self.spice_id)
            self.assertTupleEqual(self.ugps_range, ugps)

            # CK kernel.
            ugps = ext.kernel_coverage(self.kernels["attitude"][-2], self.frame_name)
            self.assertTupleEqual(self.ugps_range, ugps)

            ugps = ext.kernel_coverage(self.kernels["attitude"][-2], self.frame_id)
            self.assertTupleEqual(self.ugps_range, ugps)

            # No data in kernel or bad body code.
            with self.assertRaises(ValueError) as raised:
                ext.kernel_coverage(self.kernels["attitude"][-1], 0)
            self.assertIn("No data for body [Frame(0)] was found in the kernel", raised.exception.args[0])

        # PCK is valid, but not implemented.
        tmp_file = str(self.tmp_dir / "test1")
        with open(tmp_file, "w") as fobj:
            fobj.write("KPL/PCK\n\n")

        with self.assertRaises(NotImplementedError) as raised:
            ext.kernel_coverage(tmp_file, 0)
        self.assertIn("For kernel type: 'PCK'", raised.exception.args[0])

        # Invalid kernel for coverage data.
        tmp_file = str(self.tmp_dir / "test2")
        with open(tmp_file, "w") as fobj:
            fobj.write("KPL/SCLK\n\n")

        with self.assertRaises(ValueError) as raised:
            ext.kernel_coverage(tmp_file, 0)
        self.assertIn("Unknown or unexpected kernel type: 'SCLK' from", raised.exception.args[0])

    def test_kernel_objects(self):
        # Load the kernels to activate name <--> code.
        #   Load clock kernel for CK coverage.
        #   Load frame kernel for name <--> code.
        with ext.load_kernel([self.kernels["clock"], self.kernels["frame"]]):
            # SPK kernel.
            obj_names = ext.kernel_objects(self.kernels["ephemeris"][-1])
            self.assertIsInstance(obj_names, tuple)
            self.assertEqual(1, len(obj_names))
            self.assertIsInstance(obj_names[0], obj.Body)
            self.assertEqual(self.spice_name, obj_names[0].name.lower())

            obj_ids = ext.kernel_objects(self.kernels["ephemeris"][-1], as_id=True)
            self.assertIsInstance(obj_ids, tuple)
            self.assertEqual(1, len(obj_ids))
            self.assertEqual(self.spice_id, obj_ids[0])

            # CK kernel.
            obj_names = ext.kernel_objects(self.kernels["attitude"][-2])
            self.assertIsInstance(obj_names, tuple)
            self.assertEqual(1, len(obj_names))
            self.assertIsInstance(obj_names[0], obj.Frame)
            self.assertEqual(self.frame_name, obj_names[0].name.lower())

            obj_ids = ext.kernel_objects(self.kernels["attitude"][-2], as_id=True)
            self.assertIsInstance(obj_ids, tuple)
            self.assertEqual(1, len(obj_ids))
            self.assertEqual(self.frame_id, obj_ids[0])

        # DSK is valid, but not implemented.
        tmp_file = str(self.tmp_dir / "test1")
        with open(tmp_file, "w") as fobj:
            fobj.write("KPL/DSK\n\n")

        with self.assertRaises(NotImplementedError) as raised:
            ext.kernel_objects(tmp_file)
        self.assertIn("For kernel type: 'DSK'", raised.exception.args[0])

        # Invalid kernel for coverage data.
        tmp_file = str(self.tmp_dir / "test2")
        with open(tmp_file, "w") as fobj:
            fobj.write("KPL/SCLK\n\n")

        with self.assertRaises(ValueError) as raised:
            ext.kernel_objects(tmp_file)
        self.assertIn("Unknown or unexpected kernel type: 'SCLK' from", raised.exception.args[0])

    def test_infer_kernel_ids_basic(self):
        ids = ext.infer_ids("iss_sc", -125544, instruments=["tim"])
        self.assertDictEqual(
            dict(
                [
                    ("mission", "iss_sc"),
                    ("spacecraft", -125544),
                    ("clock", -125544),
                    ("ephemeris", -125544),
                    ("attitude", -125544000),
                    ("instruments", dict([("tim", -125544001)])),
                ]
            ),
            ids,
        )
        logger.debug("Inferred: %s", ids)

    def test_infer_kernel_ids_from_dsn(self):
        ids = ext.infer_ids("CASSINI", 82, from_dsn=True)
        self.assertDictEqual(
            dict(
                [
                    ("mission", "CASSINI"),
                    ("spacecraft", -82),
                    ("clock", -82),
                    ("ephemeris", -82),
                    ("attitude", -82000),
                    ("instruments", dict()),
                ]
            ),
            ids,
        )

    def test_infer_kernel_ids_from_norad(self):
        ids = ext.infer_ids("NOAA9", 15427, instruments="fake", from_norad=True)
        self.assertDictEqual(
            dict(
                [
                    ("mission", "NOAA9"),
                    ("spacecraft", -115427),
                    ("clock", -115427),
                    ("ephemeris", -115427),
                    ("attitude", -115427000),
                    ("instruments", dict([("fake", -115427001)])),
                ]
            ),
            ids,
        )

    def test_instrument_boresight(self):
        with ext.load_kernel([self.kernels["frame"], self.kernels["instrument"]]):
            arr = ext.instrument_boresight("tsis_tim_glint")
            self.assertListEqual([0.0, 0.0, 1.0], list(arr))


if __name__ == "__main__":
    unittest.main()
