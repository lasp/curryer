import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from curryer import spicetime, spicierpy, utils
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

    def _gap_writer(self, time_type, gap_threshold="30s"):
        krn_prop = attitude.AttitudeQuaternionProperties(
            **{
                **self.tsis_quat_prop,
                "input_time_type": time_type,
                "input_time_columns": [time_type],
                "input_angular_rate": "MAKE UP",
                "input_gap_threshold": gap_threshold,
            }
        )
        return attitude.AttitudeWriter(properties=krn_prop, bin_dir=self.bin_dir)

    @staticmethod
    def _ugps_with_steps(steps_us, cadence_us=1_000_000):
        """uGPS times at `cadence_us`, with each step in `steps_us` inserted after every 10 samples."""
        deltas = []
        for step in steps_us:
            deltas.extend([cadence_us] * 9 + [step])
        deltas.extend([cadence_us] * 9)
        return 1_400_000_000_000_000 + np.concatenate([[0], np.cumsum(deltas)]).astype(np.int64)

    @staticmethod
    def _identity_quats(times, time_col):
        return pd.DataFrame(
            {time_col: times, "quaternion_c": 1.0, "quaternion_x": 0.0, "quaternion_y": 0.0, "quaternion_z": 0.0}
        )

    def test_prepare_input_data_splits_at_gap_threshold(self):
        ugps = self._ugps_with_steps([40_000_000, 30_000_000, 29_999_999])
        for time_type in ("ugps", "et", "utc"):
            with self.subTest(time_type=time_type):
                times = spicetime.adapt(ugps, "ugps", time_type)
                chunks = self._gap_writer(time_type).prepare_input_data(self._identity_quats(times, time_type))
                self.assertListEqual([len(chunk) for chunk in chunks], [10, 10, 20])

    def test_prepare_input_data_gap_threshold_uses_total_duration(self):
        ugps = self._ugps_with_steps([40_000_000, 40_000_000])
        chunks = self._gap_writer("ugps", "1D").prepare_input_data(self._identity_quats(ugps, "ugps"))
        self.assertListEqual([len(chunk) for chunk in chunks], [30])

        ugps = self._ugps_with_steps([600_000, 500_000, 499_999], cadence_us=100_000)
        chunks = self._gap_writer("ugps", "500ms").prepare_input_data(self._identity_quats(ugps, "ugps"))
        self.assertListEqual([len(chunk) for chunk in chunks], [10, 10, 20])

    def test_prepare_input_data_fractional_microsecond_threshold_rounds_up(self):
        ugps = self._ugps_with_steps([2, 1], cadence_us=1)
        chunks = self._gap_writer("ugps", "1500ns").prepare_input_data(self._identity_quats(ugps, "ugps"))
        self.assertListEqual([len(chunk) for chunk in chunks], [10, 20])

    def test_properties_reject_gap_threshold_under_1us(self):
        for gap_threshold in ("0s", "-30s", "500ns"):
            with self.subTest(gap_threshold=gap_threshold):
                with self.assertRaisesRegex(ValueError, "input_gap_threshold must be a duration of at least 1"):
                    self._gap_writer("ugps", gap_threshold)

    def test_prepare_input_data_gaps_after_invalid_quaternion_removal(self):
        writer = self._gap_writer("ugps")

        # Invalid quaternions on both sides of a gap widen it; the split stays between the same valid rows.
        table = self._identity_quats(self._ugps_with_steps([40_000_000]), "ugps")
        table.loc[[9, 10], "quaternion_c"] = 0.0
        chunks = writer.prepare_input_data(table)
        self.assertListEqual([len(chunk) for chunk in chunks], [9, 9])

        # A run of invalid quaternions longer than the threshold creates a gap.
        table = self._identity_quats(1_400_000_000_000_000 + np.arange(60, dtype=np.int64) * 1_000_000, "ugps")
        table.loc[10:54, "quaternion_c"] = 0.0
        chunks = writer.prepare_input_data(table)
        self.assertListEqual([len(chunk) for chunk in chunks], [10, 5])

    def test_chunk_table_by_gaps_rejects_misaligned_times(self):
        ugps = self._ugps_with_steps([40_000_000])
        table = self._identity_quats(ugps, "ugps").set_index("ugps")
        with self.assertRaisesRegex(ValueError, "times must have shape"):
            self._gap_writer("ugps")._chunk_table_by_gaps(table, ugps[:-1], "ugps")


if __name__ == "__main__":
    unittest.main()
