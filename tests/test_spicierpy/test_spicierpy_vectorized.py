"""vectorized - Unit test

@author: Brandon Stone
"""

import logging
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import spiceypy  # Third-party wrapper library (w/o modifications).
from spiceypy.utils.exceptions import SpiceyError

from curryer import meta, spicetime, utils
from curryer.spicierpy import ext, vectorized

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class VectorizedTestCase(unittest.TestCase):
    def setUp(self):
        root_dir = Path(__file__).parents[2]
        self.generic_dir = root_dir / "data" / "generic"
        self.test_dir = root_dir / "tests" / "data" / "tsis1"
        self.assertTrue(self.generic_dir.is_dir())
        self.assertTrue(self.test_dir.is_dir())

        self.mkrn = meta.MetaKernel.from_json(
            self.test_dir / "tsis_v01.kernels.tm.json",
            relative=True,
            sds_dir=self.generic_dir,
        )
        self.kernels = [self.mkrn.sds_kernels, self.mkrn.mission_kernels]

        # Test time range (UTC):
        self.ugps_range = spicetime.adapt(("2021-06-10T10:10", "2021-06-10T11:50"), "iso")
        self.ugps_times = np.arange(4) * 1000000 + self.ugps_range[0]
        self.et_times = spicetime.adapt(self.ugps_times, to="et")

        # Test values.
        self.target_name = "ISS_SC"
        self.target_id = -125544
        self.observer_name = "earth"
        self.observer_id = spiceypy.bodn2c(self.observer_name)
        self.reference_frame = "J2000"
        self.correction = "NONE"
        self.target_frame_name = "ISS_ISSACS"
        self.target_frame_id = self.target_id * 1000

    def test_spkezr_original_behavior(self):
        # NOTE: If this tests starts failing, it might mean that the original
        #   function was improved, and the vectorized variant is no longer
        #   necessary.
        with ext.load_kernel(self.kernels):
            # Correct call - Scalar time value, target string, observer string.
            arr, _ = spiceypy.spkezr(
                self.target_name,
                self.et_times[0],
                obs=self.observer_name,
                abcorr=self.correction,
                ref=self.reference_frame,
            )
            self.assertTupleEqual((6,), arr.shape)

            # It _now_ supports time array (>1 value)!
            out = spiceypy.spkezr(
                self.target_name,
                self.et_times,
                obs=self.observer_name,
                abcorr=self.correction,
                ref=self.reference_frame,
            )
            npt.assert_allclose(arr, out[0][0])
            self.assertEqual(len(self.et_times), len(out[1]))

            # Doesn't support target id (int).
            with self.assertRaises(SpiceyError) as raised:
                spiceypy.spkezr(
                    self.target_id,
                    self.et_times[0],
                    obs=self.observer_name,
                    abcorr=self.correction,
                    ref=self.reference_frame,
                )
            self.assertEqual(raised.exception.short, "SPICE(EMPTYSTRING)")

            # Doesn't support observer id (int).
            with self.assertRaises(SpiceyError) as raised:
                spiceypy.spkezr(
                    self.target_name,
                    self.et_times[0],
                    obs=self.observer_id,
                    abcorr=self.correction,
                    ref=self.reference_frame,
                )
            self.assertEqual(raised.exception.short, "SPICE(IDCODENOTFOUND)")

    def test_spkezr_vectorized_variant_improved_api(self):
        with ext.load_kernel(self.kernels):
            # Correct call - Scalar time value, target string, observer string.
            arr, _ = vectorized.spkezr(
                self.target_name,
                self.et_times[0],
                obs=self.observer_name,
                abcorr=self.correction,
                ref=self.reference_frame,
            )
            self.assertTupleEqual((6,), arr.shape)

            # Support arrays!
            arr, _ = vectorized.spkezr(
                self.target_name,
                self.et_times,
                obs=self.observer_name,
                abcorr=self.correction,
                ref=self.reference_frame,
            )
            self.assertTupleEqual((4, 6), arr.shape)

            # Supports ID values (ints)!
            arr, _ = vectorized.spkezr(
                self.target_id, self.et_times[0], obs=self.observer_id, abcorr=self.correction, ref=self.reference_frame
            )
            self.assertTupleEqual((6,), arr.shape)

    def test_spkezp_original_behavior(self):
        # NOTE: If this tests starts failing, it might mean that the original
        #   function was improved, and the vectorized variant is no longer
        #   necessary.
        with ext.load_kernel(self.kernels):
            # Correct call - Scalar time value, target id (int), observer id (int).
            arr, _ = spiceypy.spkezp(
                self.target_id, self.et_times[0], obs=self.observer_id, abcorr=self.correction, ref=self.reference_frame
            )
            self.assertTupleEqual((3,), arr.shape)

            # Doesn't support time array (>1 value).
            with self.assertRaises(TypeError) as raised:
                spiceypy.spkezp(
                    self.target_id,
                    self.et_times,
                    obs=self.observer_id,
                    abcorr=self.correction,
                    ref=self.reference_frame,
                )
            self.assertIn("only length-1 arrays can be converted", raised.exception.args[0])

            # Doesn't support target name (string).
            with self.assertRaises(TypeError) as raised:
                spiceypy.spkezp(
                    self.target_name,
                    self.et_times[0],
                    obs=self.observer_id,
                    abcorr=self.correction,
                    ref=self.reference_frame,
                )
            self.assertIn("integer", raised.exception.args[0])

            # Doesn't support observer name (string).
            with self.assertRaises(TypeError) as raised:
                spiceypy.spkezp(
                    self.target_id,
                    self.et_times[0],
                    obs=self.observer_name,
                    abcorr=self.correction,
                    ref=self.reference_frame,
                )
            self.assertIn("integer", raised.exception.args[0])

    def test_spkezp_vectorized_variant_improved_api(self):
        with ext.load_kernel(self.kernels):
            # Correct call - Scalar time value, target id, observer id.
            arr, _ = vectorized.spkezp(
                self.target_id, self.et_times[0], obs=self.observer_id, abcorr=self.correction, ref=self.reference_frame
            )
            self.assertTupleEqual((3,), arr.shape)

            # Supports arrays!
            arr, _ = vectorized.spkezp(
                self.target_id, self.et_times, obs=self.observer_id, abcorr=self.correction, ref=self.reference_frame
            )
            self.assertTupleEqual((4, 3), arr.shape)

            # Supports name values (str)!
            arr, _ = vectorized.spkezp(
                self.target_name,
                self.et_times[0],
                obs=self.observer_name,
                abcorr=self.correction,
                ref=self.reference_frame,
            )
            self.assertTupleEqual((3,), arr.shape)

    def test_ckgp_vectorized(self):
        with ext.load_kernel(self.kernels):
            sc_times = np.array([spiceypy.sce2c(self.target_id, et) for et in self.et_times])

            # Supports scalar time.
            arr, sc_times_found = vectorized.ckgp(self.target_frame_id, sc_times[0], tol=1, ref=self.reference_frame)
            self.assertTupleEqual((3, 3), arr.shape)

            # Supports arrays!
            arr, sc_times_found = vectorized.ckgp(self.target_frame_id, sc_times, tol=1, ref=self.reference_frame)
            self.assertTupleEqual((4, 3, 3), arr.shape)

            # Supports names/ids!
            arr, sc_times_found = vectorized.ckgp(self.target_frame_name, sc_times[0], tol=1, ref=1)
            self.assertTupleEqual((3, 3), arr.shape)

    def test_sce2c_vectorized(self):
        with ext.load_kernel(self.kernels):
            expected_times = np.array([spiceypy.sce2c(self.target_id, et) for et in self.et_times])

            # Supports scalar time.
            sc_times = vectorized.sce2c(self.target_id, self.et_times[0])
            self.assertEqual(expected_times[0], sc_times)

            # Supports arrays!
            sc_times = vectorized.sce2c(self.target_id, self.et_times)
            for i, sc_time in enumerate(sc_times):
                self.assertEqual(expected_times[i], sc_time)

            # Supports SC name!
            sc_times = vectorized.sce2c(self.target_name, self.et_times[0])
            self.assertEqual(expected_times[0], sc_times)

    def test_sct2e_vectorized(self):
        with ext.load_kernel(self.kernels):
            sc_times = np.array([spiceypy.sce2c(self.target_id, et) for et in self.et_times])

            # Supports scalar time.
            et_times = vectorized.sct2e(self.target_id, sc_times[0])
            self.assertEqual(self.et_times[0], et_times)  # , places=6)

            # Supports arrays!
            et_times = vectorized.sct2e(self.target_id, sc_times)
            for i, et_time in enumerate(et_times):
                self.assertEqual(self.et_times[i], et_time)

            # Supports SC name!
            et_times = vectorized.sct2e(self.target_name, sc_times[0])
            self.assertEqual(self.et_times[0], et_times)

    def test_recgeo_vectorized(self):
        sphere_radius = 1.0
        sphere_flattening = 0.0

        # Test being at the center of the planet.
        arr = vectorized.recgeo(np.array([0.0, 0.0, 0.0]), sphere_radius, sphere_flattening)
        self.assertIsInstance(arr, np.ndarray)
        self.assertTupleEqual((3,), arr.shape, "Not an array of (lon, lat, alt)")
        self.assertListEqual([0.0, 0.0, -1.0], arr.tolist())

        # Test being on surface and above the north pole.
        c45 = np.cos(np.deg2rad(45))
        geo_data = np.array([[c45**2, c45**2, c45], [0.0, 0.0, 3.0]])
        arr = vectorized.recgeo(geo_data, sphere_radius, sphere_flattening, as_deg=True)
        self.assertIsInstance(arr, np.ndarray)
        self.assertTupleEqual((2, 3), arr.shape, "Not an array of 2x(lon, lat, alt)")
        self.assertAlmostEqual(0.0, arr[0, 2], places=15, msg="Not on the surface of the planet!")
        self.assertListEqual([45.0, 45.0], arr[0, 0:2].tolist(), 'Not on the "corner" of the planet!')
        self.assertListEqual([0.0, 90.0, 2.0], arr[1].tolist(), "Not above the north pole!")

        # Support input of a list.
        arr = vectorized.recgeo([0.0, 0.0, 1.0], sphere_radius, sphere_flattening, as_deg=True)
        self.assertIsInstance(arr, np.ndarray)
        self.assertTupleEqual((3,), arr.shape, "Not an array of (lon, lat, alt)")
        self.assertListEqual([0.0, 90.0, 0.0], arr.tolist(), "Santa was a lie!")

    def test_str2et_vectorized(self):
        test_utc_values = np.array(["2000-01-01T12:00:00", "2000-01-01T12:00:01"])

        et = vectorized.str2et(test_utc_values[0])
        self.assertIsInstance(et, float)
        self.assertAlmostEqual(64.18, et, places=2)

        et = vectorized.str2et(test_utc_values)
        self.assertIsInstance(et, np.ndarray)
        self.assertAlmostEqual(64.18, et[0], places=2)
        self.assertAlmostEqual(65.18, et[1], places=2)

    def test_timout_vectorized(self):
        test_et_values = np.array([64.18, 65.18])  # 2000-01-01T12:00:00, +1 sec
        fmt = "YYYY-MM-DDTHR:MN:SC ::UTC ::RND"
        n_char = 20

        utc = vectorized.timout(test_et_values[0], fmt, n_char)
        self.assertIsInstance(utc, str)
        self.assertEqual("2000-01-01T12:00:00", utc)

        utc = vectorized.timout(test_et_values, fmt, n_char)
        self.assertIsInstance(utc, np.ndarray)
        self.assertEqual("2000-01-01T12:00:00", utc[0])
        self.assertEqual("2000-01-01T12:00:01", utc[1])

        # Supports an array of 0-dim (scalar).
        utc = vectorized.timout(np.array(test_et_values[0]), fmt, n_char)
        self.assertIsInstance(utc, str)
        self.assertEqual("2000-01-01T12:00:00", utc)

        # Supports multi-dim arrays.
        utc = vectorized.timout(np.stack([test_et_values, test_et_values, test_et_values]), fmt, n_char)
        self.assertTupleEqual((3, 2), utc.shape)
        self.assertListEqual(["2000-01-01T12:00:00", "2000-01-01T12:00:00", "2000-01-01T12:00:00"], list(utc[:, 0]))
        self.assertListEqual(["2000-01-01T12:00:01", "2000-01-01T12:00:01", "2000-01-01T12:00:01"], list(utc[:, 1]))

        # # Proof that the original library does not support them.
        with self.assertRaisesRegex(TypeError, "only length-1 arrays"):
            spiceypy.timout(np.stack([test_et_values, test_et_values]), fmt, n_char)

    def test_unitim_vectorized(self):
        # Note: TAI in this context is not the typical "1958" epoch.
        test_et_values = np.array([64.18, 65.18])  # 2000-01-01T12:00:00, +1 sec

        tai = vectorized.unitim(test_et_values[0], "ET", "TAI")
        self.assertIsInstance(tai, float)
        self.assertAlmostEqual(32, tai, places=2)

        tai = vectorized.unitim(test_et_values, "ET", "TAI")
        self.assertIsInstance(tai, np.ndarray)
        self.assertAlmostEqual(32, tai[0], places=2)
        self.assertAlmostEqual(33, tai[1], places=2)


if __name__ == "__main__":
    unittest.main()
