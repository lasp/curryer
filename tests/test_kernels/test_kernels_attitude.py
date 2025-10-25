import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from curryer import spicierpy, utils
from curryer.kernels import attitude

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class AttitudeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / "data" / "generic"
        self.assertTrue(generic_dir.is_dir())

        self.bin_dir = None  # Rely on system SPICE installation.

        self.data_dir = root_dir / "tests" / "data"
        self.assertTrue(self.data_dir.is_dir())

        self.tsis_quat_prop = {
            "input_body": "ISS_SC",
            "input_frame": "J2000",
            "input_data_type": "SPICE QUATERNIONS",
            "input_time_type": "utc",  # New due to CSV.
            "input_time_columns": ["UTC"],  # New due to CSV.
            "input_angular_rate": "YES",
            "input_gap_threshold": "30s",
            "ck_type": "LINEAR_QUAT",  # Was 3 instead of string!
            "frame_kernel": str(self.data_dir / "tsis1" / "tsis_v01.frames.fk.tf"),
            "clock_kernel": str(self.data_dir / "tsis1" / "iss_v01.fakeclock.sclk.tsc"),
        }
        self.tsis_euler_prop = {
            "input_body": "TSIS_AZEL",
            "input_frame": "TSIS_TADS_COORD",
            "input_data_type": "EULER ANGLES",
            "input_time_type": "utc",  # New due to CSV.
            "input_time_columns": ["UTC"],  # New due to CSV.
            "input_angle_units": "RADIANS",
            "input_angular_rate": "MAKE UP",
            "input_gap_threshold": "30s",
            "ck_type": "LINEAR_QUAT",  # Was 3 instead of string!
            "frame_kernel": str(self.data_dir / "tsis1" / "tsis_v01.frames.fk.tf"),
            "clock_kernel": str(self.data_dir / "tsis1" / "iss_v01.fakeclock.sclk.tsc"),
        }
        self.ctim_quat_prop = {
            "input_body": "CTIM",
            "input_frame": "J2000",
            "input_data_type": "FLIP SPICE QUATERNIONS",
            "input_time_type": "utc",  # New due to CSV.
            "input_time_columns": ["UTC"],  # New due to CSV.
            "input_angular_rate": "MAKE UP/NO AVERAGING",
            "input_gap_threshold": "5min",
            "ck_type": "LINEAR_QUAT",  # Was 3 instead of string!
            "frame_kernel": str(self.data_dir / "ctim" / "ctim_v01.frames.fk.tf"),
            "clock_kernel": str(self.data_dir / "ctim" / "ctim_v01.fakeclock.sclk.tsc"),
        }
        spicierpy.boddef("ISS_SC", -125544)
        spicierpy.boddef("TSIS_AZEL", -125544108)
        spicierpy.boddef("CTIM", -152950)

    def test_tsis_quat_kernel(self):
        krn_input_data = pd.read_csv(self.data_dir / "tsis1" / "iss_sc_v01.attitude.ck.20210610.csv")

        krn_prop = attitude.AttitudeQuaternionProperties(**self.tsis_quat_prop)
        krn_writer = attitude.AttitudeWriter(properties=krn_prop, bin_dir=self.bin_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            kfile = Path(tmpdir) / "test_quat.ck"
            self.assertFalse(kfile.is_file())
            krn_writer(krn_input_data, kfile)

            self.assertTrue(kfile.is_file())

            # Check the kernel time span.
            with spicierpy.ext.load_kernel([self.tsis_quat_prop["frame_kernel"], self.tsis_quat_prop["clock_kernel"]]):
                span = spicierpy.ext.kernel_coverage(kfile, "ISS_ISSACS", to_fmt="utc")
                self.assertTupleEqual(span, ("2021-06-10 10:00:08.293690", "2021-06-10 11:59:58.166280"))

                span = spicierpy.ext.kernel_coverage(kfile, "ISS_ISSACS", to_fmt="dt64")
                input_times = pd.to_datetime(krn_input_data["UTC"])
                self.assertEqual(input_times.min(), span[0])
                self.assertEqual(input_times.max(), span[1])

    def test_tsis_euler_kernel(self):
        krn_input_data = pd.read_csv(self.data_dir / "tsis1" / "tsis_azel_v01.attitude.ck.20210610.csv")

        krn_prop = attitude.AttitudeEulerProperties(**self.tsis_euler_prop)
        krn_writer = attitude.AttitudeWriter(properties=krn_prop, bin_dir=self.bin_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            kfile = Path(tmpdir) / "test_euler.ck"
            self.assertFalse(kfile.is_file())
            krn_writer(krn_input_data, kfile)

            self.assertTrue(kfile.is_file())

            # Check the kernel time span.
            with spicierpy.ext.load_kernel([self.tsis_quat_prop["frame_kernel"], self.tsis_quat_prop["clock_kernel"]]):
                span = spicierpy.ext.kernel_coverage(kfile, "TSIS_AZEL_COORD", to_fmt="utc")
                self.assertTupleEqual(span, ("2021-06-10 10:00:06.113434", "2021-06-10 11:59:55.985852"))

                span = spicierpy.ext.kernel_coverage(kfile, "TSIS_AZEL_COORD", to_fmt="dt64")
                input_times = pd.to_datetime(krn_input_data["UTC"])
                self.assertEqual(input_times.min(), span[0])
                self.assertEqual(input_times.max(), span[1])

    def test_ctim_quat_kernel(self):
        krn_input_data = pd.read_csv(self.data_dir / "ctim" / "ctim_v01.attitude_apid1.ck.20230420.csv")

        krn_prop = attitude.AttitudeQuaternionProperties(**self.ctim_quat_prop)
        krn_writer = attitude.AttitudeWriter(properties=krn_prop, bin_dir=self.bin_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            kfile = Path(tmpdir) / "test_quat.ck"
            self.assertFalse(kfile.is_file())
            krn_writer(krn_input_data, kfile)

            self.assertTrue(kfile.is_file())

            # Check the kernel time span.
            with spicierpy.ext.load_kernel([self.ctim_quat_prop["frame_kernel"], self.ctim_quat_prop["clock_kernel"]]):
                span = spicierpy.ext.kernel_coverage(kfile, "CTIM_COORD", to_fmt="utc")
                self.assertTupleEqual(span, ("2023-04-20 05:00:09.001000", "2023-04-20 06:59:59.001000"))

                span = spicierpy.ext.kernel_coverage(kfile, "CTIM_COORD", to_fmt="dt64")
                input_times = pd.to_datetime(krn_input_data["UTC"])
                self.assertEqual(input_times.min(), span[0])
                self.assertEqual(input_times.max(), span[1])


if __name__ == "__main__":
    unittest.main()
