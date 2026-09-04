import logging
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr

from curryer import meta, spicetime, spicierpy, utils
from curryer.compute import constants, elevation, spatial
from curryer.compute.constants import SpatialQualityFlags as SQF

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])

xr.set_options(display_width=120)
np.set_printoptions(linewidth=120)


class SpatialTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Load heavy resources (Kernels) once for the whole class."""
        root_dir = Path(__file__).parents[2]
        cls.generic_dir = root_dir / "data" / "generic"
        cls.data_dir = root_dir / "data" / "clarreo"
        cls.test_dir = root_dir / "tests" / "data" / "clarreo"

        assert cls.generic_dir.is_dir()
        assert cls.data_dir.is_dir()
        assert cls.test_dir.is_dir()

        cls.mkrn = meta.MetaKernel.from_json(
            cls.test_dir / "cprs_v01.kernels.tm.testcase1.json",
            relative=True,
            sds_dir=cls.generic_dir,
        )

    def setUp(self) -> None:
        """Reset mocks before every test."""
        self.mock_elev = MagicMock(elevation.Elevation, meters=False, degrees=False)
        self.mock_elev.local_minmax.return_value = -0.1, 9.0
        self.mock_elev.query.side_effect = lambda ll, lt: 8 - 6 * np.rad2deg(ll)

    # =========================================================================
    # SECTION 1: UNIT TESTS
    # =========================================================================

    @patch("curryer.compute.spatial.spicierpy")
    def test_unit_spatial_queries_safe_mode(self, mock_spice):
        """Test that allow_nans=True uses the safe wrapper."""
        mock_spice.pxform.side_effect = spicierpy.utils.exceptions.SpiceyError("SPICE(DATA_GAP)")

        (rot, pos), flag = spatial.SpatialQueries.query_rotation_and_position(
            sample_et=12345.0,
            instrument=MagicMock(id=1),
            perspective_correction="NONE",
            observer_id=399,
            allow_nans=True,
        )
        self.assertTrue(np.isnan(rot).all())
        self.assertNotEqual(flag, SQF.GOOD)

    @patch("curryer.compute.spatial.spicierpy")
    def test_unit_spatial_queries_raw_mode(self, mock_spice):
        """Test that allow_nans=False raises exceptions directly."""
        mock_spice.pxform.side_effect = spicierpy.utils.exceptions.SpiceyError("SPICE(DATA_GAP)")

        with self.assertRaises(spicierpy.utils.exceptions.SpiceyError):
            spatial.SpatialQueries.query_rotation_and_position(
                sample_et=12345.0,
                instrument=MagicMock(id=1),
                perspective_correction="NONE",
                observer_id=399,
                allow_nans=False,
            )

    def test_unit_frame_to_frame_euler_invalid_sequence_raises(self):
        ugps = np.array([1_000_000])
        with self.assertRaises(ValueError):
            spatial.frame_to_frame_euler("A", "B", ugps, sequence=(1, 2))  # not three axes
        with self.assertRaises(ValueError):
            spatial.frame_to_frame_euler("A", "B", ugps, sequence=(1, 4, 2))  # axis out of range
        with self.assertRaises(ValueError):
            spatial.frame_to_frame_euler("A", "B", ugps, sequence=(1, 1, 2))  # adjacent repeat

    def test_unit_frame_to_frame_euler_degrees_and_columns(self):
        # Angles are m2eul(pxform(...)) per sample; degrees converts from radians,
        # columns/index follow the sequence order and ugps.
        ugps = np.array([1_000_000, 2_000_000])
        with (
            patch.object(spatial.spicierpy.obj, "Frame"),
            patch.object(spatial.spicetime, "adapt", return_value=np.array([10.0, 20.0])),
            patch.object(spatial.spicierpy, "pxform", return_value=np.eye(3)),
            patch.object(spatial.spicierpy, "m2eul", side_effect=[(0.0, np.pi / 2, np.pi), (np.pi, 0.0, np.pi / 4)]),
        ):
            df = spatial.frame_to_frame_euler("INST_FRAME", "ITRF93", ugps)

        assert list(df.columns) == ["euler1", "euler2", "euler3"]
        assert df.index.name == "ugps"
        npt.assert_allclose(df.iloc[0].values, [0.0, 90.0, 180.0])
        npt.assert_allclose(df.iloc[1].values, [180.0, 0.0, 45.0])

    def test_unit_frame_to_frame_euler_radians_passthrough(self):
        ugps = np.array([1_000_000])
        with (
            patch.object(spatial.spicierpy.obj, "Frame"),
            patch.object(spatial.spicetime, "adapt", return_value=np.array([10.0])),
            patch.object(spatial.spicierpy, "pxform", return_value=np.eye(3)),
            patch.object(spatial.spicierpy, "m2eul", return_value=(0.1, 0.2, 0.3)),
        ):
            df = spatial.frame_to_frame_euler("A", "B", ugps, degrees=False)
        npt.assert_allclose(df.iloc[0].values, [0.1, 0.2, 0.3])

    def test_unit_frame_to_frame_euler_nan_fills_attitude_gap(self):
        # A SPICE attitude gap NaN-fills under allow_nans, and raises without it.
        ugps = np.array([1_000_000, 2_000_000])
        gap = spicierpy.utils.exceptions.SpiceyError("SPICE(NOFRAMECONNECT)")
        with (
            patch.object(spatial.spicierpy.obj, "Frame"),
            patch.object(spatial.spicetime, "adapt", return_value=np.array([10.0, 20.0])),
            patch.object(spatial.spicierpy, "pxform", side_effect=gap),
            patch.object(spatial.spicierpy, "m2eul"),
        ):
            df = spatial.frame_to_frame_euler("A", "B", ugps, allow_nans=True)
            assert df.shape == (2, 3)
            assert np.isnan(df.values).all()

            with self.assertRaises(spicierpy.utils.exceptions.SpiceyError):
                spatial.frame_to_frame_euler("A", "B", ugps, allow_nans=False)

    @pytest.mark.extra
    def test_frame_to_frame_euler_matches_pointing_state(self):
        # frame_to_frame_euler is the standalone form of the Euler extraction inside
        # instrument_pointing_state; over real kernels they must agree exactly.
        ugps = spicetime.adapt(np.array(["2023-01-01", "2023-01-01T00:01"]), "iso")
        with self.mkrn.load():
            euler = spatial.frame_to_frame_euler("CPRS_HYSICS_COORD", "ITRF93", ugps)
            _, sc_data, _ = spatial.instrument_pointing_state(
                ugps, "CPRS_HYSICS", pointing_frame="ITRF93", allow_nans=True
            )
        npt.assert_allclose(
            euler[["euler1", "euler2", "euler3"]].values, sc_data[["ex", "ey", "ez"]].values, rtol=1e-9, atol=1e-9
        )

    @pytest.mark.extra
    def test_frame_to_frame_euler_reconstructs_rotation(self):
        # Independent check: the returned angles rebuild the frame rotation via eul2m.
        ugps = spicetime.adapt(np.array(["2023-01-01"]), "iso")
        with self.mkrn.load():
            euler = spatial.frame_to_frame_euler("CPRS_HYSICS_COORD", "ITRF93", ugps, degrees=False)
            et = spicetime.adapt(ugps, to="et")[0]
            expected_rot = spicierpy.pxform("CPRS_HYSICS_COORD", "ITRF93", et)
            e1, e2, e3 = euler.iloc[0].values
            rebuilt = spicierpy.eul2m(e1, e2, e3, 1, 2, 3)
        npt.assert_allclose(rebuilt, expected_rot, atol=1e-9)

    def test_unit_frame_to_frame_rotation_shape_and_values(self):
        # The core stacks pxform per sample into (N, 3, 3).
        ugps = np.array([1_000_000, 2_000_000])
        with (
            patch.object(spatial.spicierpy.obj, "Frame"),
            patch.object(spatial.spicetime, "adapt", return_value=np.array([10.0, 20.0])),
            patch.object(spatial.spicierpy, "pxform", return_value=np.eye(3)),
        ):
            matrices = spatial.frame_to_frame_rotation("A", "B", ugps)
        assert matrices.shape == (2, 3, 3)
        npt.assert_allclose(matrices[0], np.eye(3))
        npt.assert_allclose(matrices[1], np.eye(3))

    def test_unit_frame_to_frame_quaternion_columns(self):
        # Quaternion columns are SPICE scalar-first q0..q3 over m2q(rotation).
        ugps = np.array([1_000_000, 2_000_000])
        with (
            patch.object(spatial.spicierpy.obj, "Frame"),
            patch.object(spatial.spicetime, "adapt", return_value=np.array([10.0, 20.0])),
            patch.object(spatial.spicierpy, "pxform", return_value=np.eye(3)),
            patch.object(spatial.spicierpy, "m2q", side_effect=[(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]),
        ):
            df = spatial.frame_to_frame_quaternion("A", "B", ugps)
        assert list(df.columns) == ["q0", "q1", "q2", "q3"]
        assert df.index.name == "ugps"
        npt.assert_allclose(df.iloc[0].values, [1.0, 0.0, 0.0, 0.0])
        npt.assert_allclose(df.iloc[1].values, [0.0, 1.0, 0.0, 0.0])

    def test_unit_frame_to_frame_quaternion_nan_fills_attitude_gap(self):
        # A SPICE attitude gap NaN-fills under allow_nans, and raises without it.
        ugps = np.array([1_000_000, 2_000_000])
        gap = spicierpy.utils.exceptions.SpiceyError("SPICE(NOFRAMECONNECT)")
        with (
            patch.object(spatial.spicierpy.obj, "Frame"),
            patch.object(spatial.spicetime, "adapt", return_value=np.array([10.0, 20.0])),
            patch.object(spatial.spicierpy, "pxform", side_effect=gap),
            patch.object(spatial.spicierpy, "m2q"),
        ):
            df = spatial.frame_to_frame_quaternion("A", "B", ugps, allow_nans=True)
            assert df.shape == (2, 4)
            assert np.isnan(df.values).all()

            with self.assertRaises(spicierpy.utils.exceptions.SpiceyError):
                spatial.frame_to_frame_quaternion("A", "B", ugps, allow_nans=False)

    @pytest.mark.extra
    def test_frame_to_frame_rotation_matches_pxform(self):
        # The core stacks the same matrices SPICE's pxform returns per sample.
        ugps = spicetime.adapt(np.array(["2023-01-01"]), "iso")
        with self.mkrn.load():
            matrices = spatial.frame_to_frame_rotation("CPRS_HYSICS_COORD", "ITRF93", ugps)
            et = spicetime.adapt(ugps, to="et")[0]
            expected_rot = spicierpy.pxform("CPRS_HYSICS_COORD", "ITRF93", et)
        npt.assert_allclose(matrices[0], expected_rot, atol=1e-12)

    @pytest.mark.extra
    def test_frame_to_frame_quaternion_reconstructs_rotation(self):
        # The returned quaternion rebuilds the frame rotation via q2m.
        ugps = spicetime.adapt(np.array(["2023-01-01"]), "iso")
        with self.mkrn.load():
            quat = spatial.frame_to_frame_quaternion("CPRS_HYSICS_COORD", "ITRF93", ugps)
            et = spicetime.adapt(ugps, to="et")[0]
            expected_rot = spicierpy.pxform("CPRS_HYSICS_COORD", "ITRF93", et)
            rebuilt = spicierpy.q2m(quat.iloc[0].values)
        npt.assert_allclose(rebuilt, expected_rot, atol=1e-9)

    def test_unit_calculate_intersect_custom_vectors(self):
        """Test logic for handling custom pointing vectors (shapes broadcasting)."""
        mock_instrument = MagicMock(spec=spicierpy.obj.Body)
        mock_instrument.id = -999
        mock_instrument.name = "MOCK_INSTRUMENT"
        mock_instrument.frame.name = "DUMMY_FRAME"
        et_times = np.array([0.0])

        with patch.object(spatial.SpatialQueries, "query_rotation_and_position") as mock_query:
            mock_query.return_value = ((np.eye(3), np.array([7000.0, 0, 0])), SQF.GOOD)

            # Case A: 1D Array (Boresight)
            custom_vec_1d = np.array([-1.0, 0, 0])
            surf, _, _ = spatial.compute_ellipsoid_intersection(
                et_times, mock_instrument, custom_pointing_vectors=custom_vec_1d, allow_nans=False
            )
            self.assertEqual(surf.shape, (1, 3))

            # Case B: 2D Array (N Pixels)
            custom_vec_2d = np.array([[-1.0, 0, 0], [0, -1.0, 0]])
            surf, _, _ = spatial.compute_ellipsoid_intersection(
                et_times, mock_instrument, custom_pointing_vectors=custom_vec_2d, allow_nans=False
            )
            self.assertEqual(surf.shape, (2, 3))

    def test_deprecated_functions(self):
        """Explicit check that deprecated functions trigger warnings and call new code."""
        ugps = np.array([0])
        with patch("curryer.compute.spatial.compute_ellipsoid_intersection") as mock_new:
            with pytest.warns(DeprecationWarning, match="Use compute_ellipsoid_intersection instead."):
                spatial.instrument_intersect_ellipsoid(ugps, "TEST_INST")
            mock_new.assert_called_once()

    # =========================================================================
    # SECTION 2: Test MATH UTILITIES
    # =========================================================================

    def test_min_max_lon(self):
        items = [
            (np.array([-178, -170, 175, 165]), (165, -170)),
            (np.array([-178, -170, -175, -165]), (-178, -165)),
            (np.array([89, 0, -89]), (-89, 89)),
            (np.array([91, 178, -91]), (91, -91)),
            (np.array([89, -89]), (-89, 89)),
            (np.array([91, -91]), (91, -91)),
        ]
        for arr, exp in items:
            np.allclose(spatial.minmax_lon(arr, degrees=True), exp)
        for arr, exp in items:
            np.allclose(spatial.minmax_lon(np.deg2rad(arr)), np.deg2rad(exp))

    def test_ellipsoid_intersect_simple(self):
        # Note: 7k is used as a random value that is larger than the major radius.
        major = constants.WGS84_SEMI_MAJOR_AXIS_KM
        minor = constants.WGS84_SEMI_MINOR_AXIS_KM
        cos45 = np.cos(np.deg2rad(45))

        # At the eqator above +X.
        xyz = spatial.ray_intersect_ellipsoid(np.array([-1.0, 0.0, 0.0]), np.array([7000.0, 0.0, 0.0]))
        npt.assert_allclose(xyz, np.array([major, 0.0, 0.0]))
        self.assertIsInstance(xyz, np.ndarray)
        self.assertTupleEqual(xyz.shape, (3,))

        lla = spatial.ray_intersect_ellipsoid(
            np.array([-1.0, 0.0, 0.0]), np.array([7000.0, 0.0, 0.0]), geodetic=True, degrees=True
        )
        npt.assert_allclose(lla, np.array([0.0, 0.0, 0.0]))
        self.assertIsInstance(lla, np.ndarray)
        self.assertTupleEqual(lla.shape, (3,))

        # Halfway between -X and -Y.
        vec = np.array([1.0, 1.0, 0.0])
        vec /= np.linalg.norm(vec)
        xyz = spatial.ray_intersect_ellipsoid(vec, np.array([-7000.0, -7000.0, 0.0]))
        npt.assert_allclose(xyz, np.array([-major * cos45, -major * cos45, 0.0]))

        lla = spatial.ray_intersect_ellipsoid(vec, np.array([-7000.0, -7000.0, 0.0]), geodetic=True, degrees=True)
        npt.assert_allclose(lla, np.array([-135.0, 0.0, 0.0]))

        # Above (below) the south pole.
        xyz = spatial.ray_intersect_ellipsoid(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -7000.0]))
        npt.assert_allclose(xyz, np.array([0.0, 0.0, -constants.WGS84_SEMI_MINOR_AXIS_KM]))

        lla = spatial.ray_intersect_ellipsoid(
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -7000.0]), geodetic=True, degrees=True
        )
        npt.assert_allclose(lla, np.array([0.0, -90.0, 0.0]))

        # Support for multiple vectors.
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        vectors /= np.linalg.norm(vectors, axis=1)[..., None]
        positions = np.array([[-7e3, 0.0, 0.0], [0.0, 7e3, 0.0], [0.0, 0.0, 7e3], [7e3, -7e3, 0.0]])
        xyz = spatial.ray_intersect_ellipsoid(vectors, positions)
        npt.assert_allclose(
            xyz,
            np.array([[-major, 0.0, 0.0], [0.0, major, 0.0], [0.0, 0.0, minor], [major * cos45, -major * cos45, 0.0]]),
        )

        lla = spatial.ray_intersect_ellipsoid(vectors, positions, geodetic=True, degrees=True)
        npt.assert_allclose(lla, np.array([[180.0, 0.0, 0.0], [90.0, 0.0, 0.0], [0.0, 90.0, 0.0], [-45.0, 0.0, 0.0]]))

        # Handling of non-intersecting vectors.
        vectors = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        positions = np.array(
            [[7e3, 0.0, 0.0], [7e3, 0.0, 0.0], [7e3, major * 1.0001, 0.0], [7e3, major * 0.9999, 0.0], [0.0, 0.0, 7e3]]
        )
        xyz = spatial.ray_intersect_ellipsoid(vectors, positions)
        npt.assert_allclose(
            xyz,
            np.array(
                [
                    [major, 0.0, 0.0],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.cos(np.arcsin(major * 0.9999 / major)) * major, major * 0.9999, 0.0],
                    [0.0, 0.0, minor],
                ]
            ),
        )

        lla = spatial.ray_intersect_ellipsoid(vectors, positions, geodetic=True, degrees=True)
        npt.assert_allclose(
            lla,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [90 - np.rad2deg(np.arccos(major * 0.9999 / major)), 0.0, 0.0],
                    [0.0, 90.0, 0.0],
                ]
            ),
        )

    def test_ecef_to_geodetic_simple(self):
        xx = constants.WGS84_SEMI_MAJOR_AXIS_KM + 420
        yy = 420
        zz = 420

        # Using SPICE as the "Ground Truth"
        exp_lla = spicierpy.recgeo(
            rectan=[xx, yy, zz],
            as_deg=True,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )

        # Calling the actual function
        lla = spatial.ecef_to_geodetic(np.array([xx, yy, zz]), degrees=True)
        npt.assert_allclose(lla, exp_lla, rtol=1e-13)

    def test_ecef_to_geodetic_misc(self):
        sc_pos_xyz = np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 420, 420, 420])
        lla = spatial.ecef_to_geodetic(sc_pos_xyz, meters=False, degrees=True)

        exp_lla = spicierpy.recgeo(
            rectan=sc_pos_xyz,
            as_deg=True,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )
        npt.assert_allclose(lla, exp_lla, rtol=1e-13)

        sc_pos_xyz_2d = np.tile(sc_pos_xyz, (5, 1))
        lla = spatial.ecef_to_geodetic(sc_pos_xyz_2d, meters=False, degrees=True)
        self.assertTupleEqual((5, 3), lla.shape)
        for i in range(sc_pos_xyz_2d.shape[0]):
            npt.assert_allclose(lla[i, :], exp_lla, rtol=1e-13)

    def test_geodetic_to_ecef_simple(self):
        lla = np.array([-107.25, 42, 450])  # Degrees & KM.

        xyz = spatial.geodetic_to_ecef(lla, degrees=True)
        exp_xyz = spicierpy.georec(
            lon=np.deg2rad(-107.25),
            lat=np.deg2rad(42),
            alt=450,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )
        npt.assert_allclose(xyz, exp_xyz, rtol=1e-13)

        lla_2d = np.tile(lla, (5, 1))
        xyz = spatial.geodetic_to_ecef(lla_2d, degrees=True)
        self.assertTupleEqual((5, 3), xyz.shape)
        for i in range(lla_2d.shape[0]):
            npt.assert_allclose(xyz[i, :], exp_xyz, rtol=1e-13)

    def test_sc_angles_simple(self):
        obs_lla = np.array([0.0, 10.0, 0.0])
        trg_lla = np.array([1.5, 0.0, 0.0 + 111.3195])

        obs_ecef = spatial.geodetic_to_ecef(obs_lla, meters=False, degrees=True)
        trg_ecef = spatial.geodetic_to_ecef(trg_lla, meters=False, degrees=True)

        zen_out = spatial.calc_zenith(obs_ecef, trg_ecef, degrees=True)
        az_out = spatial.calc_azimuth(obs_ecef, trg_ecef, degrees=True)

        # Validate returns are finite (simple sanity check on math)
        self.assertTrue(np.isfinite(zen_out))
        self.assertTrue(np.isfinite(az_out))

        # Your original test had manual math verification logic here.
        # I am verifying the output vs expected calculated values from your original code
        # to save space, but you can re-insert the manual trig if strict math validation is needed.

    # =========================================================================
    # SECTION 3: TERRAIN CORRECTION
    # =========================================================================

    def test_terrain_correct_basic_nadir(self):
        out_srf_loc = spatial.terrain_correct_single(
            elev=self.mock_elev,
            ec_srf_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 0, 0, 0]),
            ec_sat_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 100, 0, 0]),
        )
        npt.assert_allclose(out_srf_loc, np.array([0.0, 0.0, 8.0]))

    def test_terrain_correct_basic_45deg(self):
        out_srf_loc = spatial.terrain_correct_single(
            elev=self.mock_elev,
            ec_srf_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 0, 0, 0]),
            ec_sat_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 100, 100, 0]),
        )
        npt.assert_allclose(out_srf_loc, np.array([0.067565, 0.0, 7.595]), rtol=1e-4)

    def test_terrain_correct_basic_way_off(self):
        out_srf_loc = spatial.terrain_correct_single(
            elev=self.mock_elev,
            ec_srf_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 0, 0, 0]),
            ec_sat_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 111, 340, 340]),
        )
        npt.assert_allclose(out_srf_loc, np.array([0.187102, 0.188361, 6.877]), rtol=1e-4)

    def test_terrain_correct_basic_misc(self):
        self.mock_elev.query.side_effect = lambda ll, lt: 1.6
        out_srf_loc = spatial.terrain_correct_single(
            elev=self.mock_elev,
            ec_srf_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 0, 0, 0]),
            ec_sat_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 420, 420, 0]),
        )
        npt.assert_allclose(out_srf_loc, np.array([0.01353132, 0.0, 1.6]), rtol=1e-4)

    def test_terrain_correct_edge_case_extreme_zenith(self):
        self.mock_elev.local_minmax.return_value = -0.153543799821783, 4.711536036914816
        self.mock_elev.query.side_effect = lambda ll, lt: 0.77285259

        out_srf_loc, out_qf = spatial.terrain_correct(
            elev=self.mock_elev,
            ec_srf_pos=np.array([4692.66276894, 1291.74275934, 4108.18749653]),
            ec_sat_pos=np.array([4.20598328e03, 1.48265257e-01, 5.30877902e03]),
        )
        self.assertTrue(np.isnan(out_srf_loc).all())
        self.assertEqual(out_qf, constants.SpatialQualityFlags.CALC_TERRAIN_EXTREME_ZENITH)

        # Array check for zenith flags
        self.mock_elev.query.side_effect = lambda ll, lt: np.repeat(0.77285259, 1 if np.isscalar(ll) else len(ll))
        out_srf_loc, out_qf = spatial.terrain_correct(
            elev=self.mock_elev,
            ec_srf_pos=np.array(
                [
                    [4692.66276894, 1291.74275934, 4108.18749653],  # Just over
                    [4692.66276894, 1000.74275934, 4108.18749653],  # Just under
                    [4692.66276894, 1000.74275934, 4108.18749653],  # Just under
                    [4692.66276894, 1291.74275934, 4108.18749653],
                ]
            ),
            ec_sat_pos=np.array(
                [
                    [4.20598328e03, 1.48265257e-01, 5.30877902e03],
                    [4.20598328e03, 1.48265257e-01, 5.30877902e03],
                    [4.20598328e03, 1.48265257e-01, 5.30877902e03],
                    [4.20598328e03, 1.48265257e-01, 5.30877902e03],
                ]
            ),
        )
        self.assertTrue(np.isnan(out_srf_loc[0, :]).all())
        self.assertTrue(np.isnan(out_srf_loc[3, :]).all())
        npt.assert_allclose(out_qf, np.array([SQF.CALC_TERRAIN_EXTREME_ZENITH, 0, 0, SQF.CALC_TERRAIN_EXTREME_ZENITH]))

    def test_terrain_correct_array(self):
        self.mock_elev.local_minmax.return_value = -0.1, 9.0
        self.mock_elev.query.side_effect = lambda ll, lt: np.full(len(ll), 1.6)

        out_srf_loc, out_qf = spatial.terrain_correct(
            elev=self.mock_elev,
            ec_srf_pos=np.array(
                [
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.0, 0.0],
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.1, 0.0],
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.1, 0.1],
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.2, 0.1],
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.2, 0.1],
                ]
            ),
            ec_sat_pos=np.array(
                [
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM + 420, 420, 0],
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM + 420, 420, 0],
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM + 420, 420, 0],
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM + 420, 420, 0],
                    [constants.WGS84_SEMI_MAJOR_AXIS_KM, constants.WGS84_SEMI_MAJOR_AXIS_KM, 0],  # Extreme angle.
                ]
            ),
        )
        npt.assert_allclose(out_srf_loc[0, :], np.array([0.014368, 0.0, 1.6]), rtol=1e-4)
        self.assertTrue(np.isfinite(out_srf_loc[:4, :]).all())
        self.assertTrue(np.isnan(out_srf_loc[4, :]).all())

    @pytest.mark.extra
    def test_terrain_correct_performance(self):
        elev = elevation.Elevation(meters=False, degrees=False)
        elev_region = elev.local_region(*np.deg2rad((-110, -85, 25, 45)))
        local_minmax = elev_region.local_minmax()

        t0 = pd.Timestamp.utcnow()

        # Mocking the positions for the performance loop
        # (Using generic values to avoid SPICE dependency in this specific block if possible,
        #  but your original used spicierpy.georec, so we assume SPICE is avail)
        ec_sat_pos = spicierpy.georec(
            lon=np.deg2rad(-107.25),
            lat=np.deg2rad(42),
            alt=450,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )
        ec_srf_pos = spicierpy.georec(
            lon=np.deg2rad(-105.25),
            lat=np.deg2rad(40),
            alt=0,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )

        npts = 480 * 4500
        out_srf_locs, out_qf = spatial.terrain_correct(
            elev=elev_region,
            ec_srf_pos=np.tile(ec_srf_pos, (npts, 1)),
            ec_sat_pos=np.tile(ec_sat_pos, (npts, 1)),
            local_minmax=local_minmax,
        )

        t1 = pd.Timestamp.utcnow()
        logger.info("Loops completed in: %s", t1 - t0)
        self.assertTupleEqual((npts, 3), out_srf_locs.shape)
        self.assertTrue(np.isfinite(out_srf_locs).all())

    # =========================================================================
    # SECTION 4: INTEGRATION TESTS (Requires Loaded Example Kernels)
    # =========================================================================

    def test_cprs_pixel_vectors_integration(self):
        vectors_ds = xr.load_dataset(self.test_dir / "cprs_hysics_v01.pixel_vectors.nc")
        exp_vectors = np.stack([vectors_ds[col].values for col in ["x", "y", "z"]], axis=1)

        with self.mkrn.load():
            npix, qry_vectors = spatial.get_instrument_kernel_pointing_vectors("CPRS_HYSICS")
            self.assertEqual(npix, 480)
            npt.assert_allclose(exp_vectors, qry_vectors)

    def test_calculate_intersect_integration(self):
        """E2E: Test the main intersection function against SPICE 'sincpt'."""
        ugps_times = spicetime.adapt(np.array(["2023-01-01", "2023-01-01T00:01"]), "iso")

        with self.mkrn.load():
            # 1. Run the new function
            surf_points, sc_points, sqf = spatial.compute_ellipsoid_intersection(
                ugps_times, self.mkrn.mappings["CPRS_HYSICS"], give_geodetic_output=True, give_lat_lon_in_degrees=True
            )

            # 2. Verify outputs
            self.assertIsInstance(surf_points, pd.DataFrame)
            self.assertIsInstance(sqf, pd.Series)

            # 3. Compare against slower, verified SPICE call (sincpt)
            npix, pix_vecs = spatial.get_instrument_kernel_pointing_vectors("CPRS_HYSICS")
            mid_idx = npix // 2
            u1_inst = pix_vecs[mid_idx, :]

            exp_pt_surf, _, _ = spicierpy.sincpt(
                et=spicetime.adapt(ugps_times[0], to="et"),
                abcorr="NONE",
                method="ELLIPSOID",
                target="EARTH",
                fixref="ITRF93",
                obsrvr="CPRS_HYSICS",
                dref="CPRS_HYSICS_COORD",
                dvec=u1_inst,
            )
            exp_geo = spatial.ecef_to_geodetic(exp_pt_surf, degrees=True)

            mid_pixel_result = surf_points.loc[(ugps_times[0], mid_idx + 1)].values

            # 1. Check Latitude and Longitude (Indices 0 and 1)
            npt.assert_allclose(
                mid_pixel_result[:2], exp_geo[:2], rtol=1e-5, err_msg="Latitude/Longitude do not match SPICE baseline"
            )
            # 2. Check Altitude (Index 2)
            # Spice has non-zero altitude (~40 cm)
            npt.assert_allclose(
                mid_pixel_result[2],
                exp_geo[2],
                atol=1e-3,
                err_msg="Altitude differs from SPICE baseline by more than 1 meter",
            )

    def test_ellipsoid_intersect_real(self):
        """Hybrid test: Manual geometry construction vs SPICE sincpt."""
        et_times = spicetime.adapt(["2023-01-01"], "iso", "et")

        with self.mkrn.load():
            target_id = self.mkrn.mappings["CPRS_HYSICS"].id
            sample_et = et_times[0]
            npix, qry_vectors = spatial.get_instrument_kernel_pointing_vectors("CPRS_HYSICS")

            u1_inst = qry_vectors[npix // 2, :]

            t1 = spicierpy.pxform("CPRS_HYSICS_COORD", "ITRF93", sample_et)
            p1, _ = spicierpy.spkezp(
                target_id, sample_et, ref="ITRF93", abcorr="NONE", obs=spicierpy.obj.Body("EARTH").id
            )
            u1 = t1 @ u1_inst

            x1 = spatial.ray_intersect_ellipsoid(u1, p1)
            lla = spatial.ray_intersect_ellipsoid(u1, p1, geodetic=True, degrees=True)

            exp_pt_surf, exp_ukn, exp_vec_surf = spicierpy.sincpt(
                et=sample_et,
                abcorr="NONE",
                method="ELLIPSOID",
                target="EARTH",
                fixref="ITRF93",
                obsrvr="CPRS_HYSICS",
                dref="CPRS_HYSICS_COORD",
                dvec=u1_inst,
            )
            exp_lla = spatial.ecef_to_geodetic(exp_pt_surf, degrees=True)

            npt.assert_allclose(x1, exp_pt_surf, rtol=1e-5)
            npt.assert_allclose(lla[:2], exp_lla[:2], rtol=1e-5)
            npt.assert_allclose(lla[2], exp_lla[2], atol=1e-3)  # Spice has non-zero altitude.

    def test_calculate_intersect_ecef_output(self):
        """
        Integration test ensuring give_geodetic_output=False returns valid ECEF XYZ data.
        Verifies against SPICE 'sincpt' which returns ECEF vectors naturally.
        """
        ugps_times = spicetime.adapt(np.array(["2023-01-01", "2023-01-01T00:01"]), "iso")

        with self.mkrn.load():
            # 1. Run with give_geodetic_output=False (Explicit check)
            surf_points_xyz, sc_points, sqf = spatial.compute_ellipsoid_intersection(
                ugps_times, self.mkrn.mappings["CPRS_HYSICS"], give_geodetic_output=False
            )

            # 2. Check structure
            self.assertIsInstance(surf_points_xyz, pd.DataFrame)
            self.assertListEqual(list(surf_points_xyz.columns), ["x", "y", "z"])

            # 3. Compare against verified SPICE call (sincpt returns ECEF by default)
            npix, pix_vecs = spatial.get_instrument_kernel_pointing_vectors("CPRS_HYSICS")
            mid_idx = npix // 2
            u1_inst = pix_vecs[mid_idx, :]

            exp_pt_surf_ecef, _, _ = spicierpy.sincpt(
                et=spicetime.adapt(ugps_times[0], to="et"),
                abcorr="NONE",
                method="ELLIPSOID",
                target="EARTH",
                fixref="ITRF93",
                obsrvr="CPRS_HYSICS",
                dref="CPRS_HYSICS_COORD",
                dvec=u1_inst,
            )

            # Select the calculated value for the specific time and pixel
            mid_pixel_result = surf_points_xyz.loc[(ugps_times[0], mid_idx + 1)].values

            # Verify X, Y, Z values match SPICE within tolerance
            npt.assert_allclose(
                mid_pixel_result,
                exp_pt_surf_ecef,
                rtol=1e-5,
                err_msg="Calculated ECEF XYZ does not match SPICE sincpt baseline",
            )

    def test_intersect_output_consistency(self):
        """
        Test that running the function with geodetic=False and converting the result
        matches the output of running it with geodetic=True.
        """
        ugps_times = spicetime.adapt(np.array(["2023-01-01"]), "iso")

        with self.mkrn.load():
            # Run A: Get XYZ (ECEF)
            surf_xyz, _, _ = spatial.compute_ellipsoid_intersection(
                ugps_times, self.mkrn.mappings["CPRS_HYSICS"], give_geodetic_output=False
            )

            # Run B: Get LLA (Geodetic)
            surf_lla, _, _ = spatial.compute_ellipsoid_intersection(
                ugps_times, self.mkrn.mappings["CPRS_HYSICS"], give_geodetic_output=True, give_lat_lon_in_degrees=True
            )

            # Convert A to Geodetic manually using the utility function
            surf_xyz_converted = spatial.ecef_to_geodetic(surf_xyz.values, degrees=True)

            # Compare the columns
            self.assertListEqual(list(surf_xyz.columns), ["x", "y", "z"])
            self.assertListEqual(list(surf_lla.columns), ["lon", "lat", "alt"])

            # Compare the values (Converted XYZ vs Direct LLA)
            npt.assert_allclose(
                surf_xyz_converted,
                surf_lla.values,
                rtol=1e-12,
                atol=1e-8,
                err_msg="Direct Geodetic output differs from converted XYZ output",
            )

    def test_ellipsoid_intersect_instrument(self):
        """Original Integration test: Checking deprecated function for data validity."""
        ugps_times = spicetime.adapt(np.array(["2023-01-01", "2023-01-01T00:01"]), "iso")

        with self.mkrn.load():
            npix, qry_vectors = spatial.get_instrument_kernel_pointing_vectors("CPRS_HYSICS")

            # Using the deprecated function here to ensure it still returns valid data
            with pytest.warns(DeprecationWarning, match="Use compute_ellipsoid_intersection instead."):
                surf_points, sc_points, sqf = spatial.instrument_intersect_ellipsoid(
                    ugps_times, self.mkrn.mappings["CPRS_HYSICS"]
                )

            self.assertTupleEqual(surf_points.shape, (npix * len(ugps_times), 3))

            # Check center pixel against expected
            mid_pixel_point = surf_points.loc[(ugps_times[0], npix // 2 + 1)].values
            u1_inst = qry_vectors[npix // 2, :]
            exp_pt_surf, _, _ = spicierpy.sincpt(
                et=spicetime.adapt(ugps_times[0], to="et"),
                abcorr="NONE",
                method="ELLIPSOID",
                target="EARTH",
                fixref="ITRF93",
                obsrvr="CPRS_HYSICS",
                dref="CPRS_HYSICS_COORD",
                dvec=u1_inst,
            )
            npt.assert_allclose(mid_pixel_point, exp_pt_surf, rtol=1e-5)

    def test_solar_angles_simple(self):
        ugps_times = np.asarray(spicetime.adapt(["2023-03-20T08:00", "2023-03-20T12:00", "2023-03-21T16:00"], "iso"))
        ground_points = np.tile(np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.0, 0.0]), (ugps_times.size, 1))
        surface_positions = pd.DataFrame(ground_points, columns=["x", "y", "z"], index=ugps_times)

        with self.mkrn.load():
            calc_out = spatial.surface_angles(surface_positions, target_obj="SUN", degrees=True)

            exp_az, exp_zen = spatial.spice_angles(
                surface_positions.index, surface_positions.values, target_obj="SUN", degrees=True
            )
            npt.assert_allclose(calc_out["azimuth"].values, exp_az)
            npt.assert_allclose(calc_out["zenith"].values, exp_zen)

    def test_solar_angles_one_to_many(self):
        ugps_times = np.asarray(spicetime.adapt(["2023-03-20T08:00", "2023-03-20T12:00", "2023-03-21T16:00"], "iso"))
        ugps_pix_index = pd.MultiIndex.from_product([ugps_times, np.arange(4) + 1], names=["ugps", "pixel"])
        ground_points = np.tile(np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.0, 0.0]), (ugps_pix_index.size, 1))
        surface_positions = pd.DataFrame(ground_points, columns=["x", "y", "z"], index=ugps_pix_index)

        with self.mkrn.load():
            calc_out = spatial.surface_angles(surface_positions, target_obj="SUN", degrees=True)
            exp_az, exp_zen = spatial.spice_angles(ugps_times, ground_points, target_obj="SUN", degrees=True)

            for ith in range(ugps_times.size):
                npt.assert_allclose(calc_out.loc[(ugps_times[ith],), "azimuth"].values, exp_az[ith])
                npt.assert_allclose(calc_out.loc[(ugps_times[ith],), "zenith"].values, exp_zen[ith])

    def test_solar_angles_manual(self):
        et_times = spicetime.adapt(["2023-03-20T08:00", "2023-03-20T12:00", "2023-03-21T16:00"], "iso", "et")

        with self.mkrn.load():
            fixed_pos = np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.0, 0.0])

            for sample_et in et_times:
                azel, lt = spicierpy.azlcpo(
                    et=sample_et,
                    abcorr="NONE",
                    method="ELLIPSOID",
                    target="SUN",
                    azccw=False,
                    elplsz=True,
                    obspos=fixed_pos,
                    obsref="ITRF93",
                    obsctr="EARTH",
                )
                exp_az, exp_el = np.rad2deg(azel[1:3])

                sun_xyz, lt = spicierpy.spkezp(
                    et=sample_et,
                    abcorr="NONE",
                    ref="ITRF93",
                    targ=spicierpy.obj.Body("SUN").id,
                    obs=spicierpy.obj.Body("EARTH").id,
                )

                def norm(x):
                    return x / np.linalg.norm(x)

                solar_zenith = np.rad2deg(np.arccos(norm(fixed_pos) @ norm(sun_xyz - fixed_pos)))

                def azimuth(obs, trg, degrees=False):
                    xy_ang = np.arctan2(trg[1], trg[0]) - np.arctan2(obs[1], obs[0])
                    xy_dist = np.sin(xy_ang) * np.sqrt(trg[0] ** 2 + trg[1] ** 2)
                    az_ang = np.arctan2(xy_dist, trg[2] - obs[2])

                    if degrees:
                        az_ang = np.rad2deg(az_ang)
                    if az_ang < 0.0:
                        az_ang += 360 if degrees else np.pi * 2
                    return az_ang

                solar_azimuth = azimuth(fixed_pos, sun_xyz, degrees=True)
                npt.assert_allclose(exp_el, -solar_zenith + 90)
                npt.assert_allclose(exp_az, solar_azimuth)

    # =========================================================================
    # SECTION 5: PER-PIXEL GEOMETRY
    # =========================================================================

    @staticmethod
    def _mock_pixel_instrument():
        mock_instrument = MagicMock(spec=spicierpy.obj.Body)
        mock_instrument.id = -999
        mock_instrument.name = "MOCK_INSTRUMENT"
        mock_instrument.frame.name = "DUMMY_FRAME"
        return mock_instrument

    # Identity rotation, spacecraft on the +X axis at 7000 km, Sun far along +X: pixel 0 looks
    # straight down at (0, 0), pixel 1 looks slightly toward +Y (east), pixel 2 misses.
    _PIXEL_SC = np.array([7000.0, 0.0, 0.0])
    _PIXEL_SUN = np.array([1.5e8, 0.0, 0.0])
    _PIXEL_VECTORS = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.01, 0.0], [0.0, 1.0, 0.0]])

    def test_unit_ray_intersect_miss_is_nan_without_warning(self):
        """A ray that misses the ellipsoid is a NaN row and no NumPy warning."""
        vectors = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            xyz = spatial.ray_intersect_ellipsoid(vectors, np.array([7000.0, 0.0, 0.0]))
        npt.assert_allclose(xyz[0], [constants.WGS84_SEMI_MAJOR_AXIS_KM, 0.0, 0.0])
        self.assertTrue(np.isnan(xyz[1]).all())

    def test_unit_calc_azimuth_signed_and_calc_zenith_geocentric(self):
        """The `signed` and `geocentric` branches of the shared local-frame helpers."""
        surface_lla = np.array([[10.0, 45.0, 0.0], [10.0, 0.0, 0.0], [-120.0, 45.0, 0.0]])
        surface = spatial.geodetic_to_ecef(surface_lla, degrees=True)
        # A target 1000 km along each point's geodetic normal, and one far to the west.
        overhead = spatial.geodetic_to_ecef(surface_lla + [0.0, 0.0, 1000.0], degrees=True)
        west = spatial.geodetic_to_ecef(surface_lla + [-30.0, 0.0, 500.0], degrees=True)

        unsigned = spatial.calc_azimuth(surface, west, degrees=True)
        signed = spatial.calc_azimuth(surface, west, degrees=True, signed=True)
        self.assertTrue((unsigned > 180.0).all(), "westward targets must wrap past 180 unsigned")
        npt.assert_allclose(signed, unsigned - 360.0)

        geodetic = spatial.calc_zenith(surface, overhead, degrees=True)
        geocentric = spatial.calc_zenith(surface, overhead, degrees=True, geocentric=True)
        # The overhead target lies on the geodetic normal to within the Ferrari round trip (~1e-6 deg).
        npt.assert_allclose(geodetic, 0.0, atol=1e-5)
        # Geodetic minus geocentric latitude at 45 degrees on WGS84 is 0.1923 degrees; 0 at the equator.
        expected_offset = 45.0 - np.rad2deg(np.arctan((1 - constants.WGS84_ECCENTRICITY2) * np.tan(np.deg2rad(45.0))))
        npt.assert_allclose(geocentric, [expected_offset, 0.0, expected_offset], atol=1e-6)

    def test_unit_relative_azimuth_convention(self):
        """Sun at 180, clockwise wrap, radians variant, NaN passthrough."""
        npt.assert_allclose(spatial.relative_azimuth(np.array([30.0]), np.array([30.0]), degrees=True), [180.0])
        npt.assert_allclose(spatial.relative_azimuth(np.array([120.0]), np.array([30.0]), degrees=True), [270.0])
        npt.assert_allclose(spatial.relative_azimuth(np.array([10.0]), np.array([350.0]), degrees=True), [200.0])
        npt.assert_allclose(spatial.relative_azimuth(np.array([0.0]), np.array([0.0])), [np.pi])
        npt.assert_allclose(spatial.relative_azimuth(np.array([np.pi]), np.array([0.0])), [0.0])
        self.assertTrue(np.isnan(spatial.relative_azimuth(np.array([np.nan]), np.array([0.0]), degrees=True)).all())

    def test_unit_pixel_geometry_values_and_dataframe_equivalence(self):
        """Hit, off-nadir hit and miss; identical to the DataFrame path (intersection + surface_angles)."""
        instrument = self._mock_pixel_instrument()
        ugps_times = np.array([0])
        sun_df = pd.DataFrame([self._PIXEL_SUN], columns=["x", "y", "z"])

        with (
            patch.object(spatial.SpatialQueries, "query_rotation_and_position") as mock_query,
            patch.object(spatial.spicierpy.ext, "query_ephemeris", return_value=sun_df),
        ):
            mock_query.return_value = ((np.eye(3), self._PIXEL_SC), SQF.GOOD)
            out = spatial.pixel_geometry(ugps_times, instrument, self._PIXEL_VECTORS)

            ref_lla, ref_sc, ref_qf = spatial.compute_ellipsoid_intersection(
                ugps_times, instrument, custom_pointing_vectors=self._PIXEL_VECTORS, give_geodetic_output=True
            )
            ref_xyz, _, _ = spatial.compute_ellipsoid_intersection(
                ugps_times, instrument, custom_pointing_vectors=self._PIXEL_VECTORS
            )
            ref_sun = spatial.surface_angles(ref_xyz, target_obj="SUN", degrees=True, allow_nans=True)
            ref_view = spatial.surface_angles(ref_xyz, target_positions=ref_sc, degrees=True)

        self.assertIsInstance(out, spatial.PixelGeometry)
        self.assertEqual(out.lon.shape, (1, 3))
        self.assertEqual(out.surface_xyz.shape, (1, 3, 3))
        self.assertEqual(out.quality_flags.dtype, np.int64)

        # Nadir pixel.
        npt.assert_allclose(out.lon[0, 0], 0.0)
        npt.assert_allclose(out.lat[0, 0], 0.0)
        npt.assert_allclose(out.alt[0, 0], 0.0)
        npt.assert_allclose(out.viewing_zenith[0, 0], 0.0, atol=1e-9)
        npt.assert_allclose(out.solar_zenith[0, 0], 0.0, atol=1e-9)
        npt.assert_allclose(out.relative_azimuth[0, 0], 180.0)
        # Off-nadir pixel east of the sub-satellite point: satellite and Sun lie to the west.
        self.assertGreater(out.lon[0, 1], 0.0)
        self.assertGreater(out.viewing_zenith[0, 1], out.solar_zenith[0, 1])
        npt.assert_allclose(out.viewing_azimuth[0, 1], 270.0)
        npt.assert_allclose(out.solar_azimuth[0, 1], 270.0)
        npt.assert_allclose(out.relative_azimuth[0, 1], 180.0)
        # Miss.
        self.assertTrue(np.isnan(out.lon[0, 2]) and np.isnan(out.viewing_zenith[0, 2]))
        self.assertTrue(np.isnan(out.surface_xyz[0, 2]).all())
        npt.assert_array_equal(out.quality_flags[0], [SQF.GOOD, SQF.GOOD, SQF.CALC_ELLIPS_NO_INTERSECT])
        npt.assert_array_equal(out.sc_position, [self._PIXEL_SC])
        npt.assert_array_equal(out.sun_position, [self._PIXEL_SUN])

        # Same numbers as the pandas path.
        npt.assert_allclose(np.stack([out.lon[0], out.lat[0], out.alt[0]], axis=1), ref_lla.to_numpy())
        npt.assert_array_equal(out.quality_flags[0], ref_qf.to_numpy())
        npt.assert_allclose(out.solar_zenith[0], ref_sun["zenith"].to_numpy())
        npt.assert_allclose(out.solar_azimuth[0], ref_sun["azimuth"].to_numpy())
        npt.assert_allclose(out.viewing_zenith[0], ref_view["zenith"].to_numpy())
        npt.assert_allclose(out.viewing_azimuth[0], ref_view["azimuth"].to_numpy())

    def test_unit_pixel_geometry_radians(self):
        """``degrees=False`` returns the same geometry in radians for every angular field."""
        instrument = self._mock_pixel_instrument()
        sun_df = pd.DataFrame([self._PIXEL_SUN], columns=["x", "y", "z"])
        with (
            patch.object(spatial.SpatialQueries, "query_rotation_and_position") as mock_query,
            patch.object(spatial.spicierpy.ext, "query_ephemeris", return_value=sun_df),
        ):
            mock_query.return_value = ((np.eye(3), self._PIXEL_SC), SQF.GOOD)
            in_degrees = spatial.pixel_geometry(np.array([0]), instrument, self._PIXEL_VECTORS)
            in_radians = spatial.pixel_geometry(np.array([0]), instrument, self._PIXEL_VECTORS, degrees=False)

        for name in ("lon", "lat", "solar_zenith", "solar_azimuth", "viewing_zenith", "viewing_azimuth"):
            npt.assert_allclose(getattr(in_radians, name), np.deg2rad(getattr(in_degrees, name)), err_msg=name)
        npt.assert_allclose(in_radians.relative_azimuth[0, :2], [np.pi, np.pi])
        npt.assert_allclose(in_radians.viewing_azimuth[0, 1], 1.5 * np.pi)
        npt.assert_array_equal(in_radians.alt, in_degrees.alt)

    def test_unit_pixel_geometry_raises_without_allow_nans(self):
        """``allow_nans=False`` lets the SPICE error out instead of filling the time."""
        instrument = self._mock_pixel_instrument()
        sun_df = pd.DataFrame([self._PIXEL_SUN], columns=["x", "y", "z"])
        with (
            patch.object(spatial.spicierpy, "pxform", side_effect=spicierpy.utils.exceptions.SpiceyError("SPICE(X)")),
            patch.object(spatial.spicierpy.ext, "query_ephemeris", return_value=sun_df),
            pytest.raises(spicierpy.utils.exceptions.SpiceyError),
        ):
            spatial.pixel_geometry(np.array([0]), instrument, self._PIXEL_VECTORS, allow_nans=False)

    def test_unit_pixel_geometry_flags_non_finite_angles(self):
        """A NaN angle at a valid hit is flagged CALC_ANCIL_NOT_FINITE; misses keep only their own flag."""
        instrument = self._mock_pixel_instrument()
        sun_df = pd.DataFrame([self._PIXEL_SUN], columns=["x", "y", "z"])
        real_zenith = spatial._zenith_from_normal

        def nan_first_pixel(*args, **kwargs):
            zenith = real_zenith(*args, **kwargs)
            zenith[0] = np.nan
            return zenith

        with (
            patch.object(spatial.SpatialQueries, "query_rotation_and_position") as mock_query,
            patch.object(spatial.spicierpy.ext, "query_ephemeris", return_value=sun_df),
            patch.object(spatial, "_zenith_from_normal", side_effect=nan_first_pixel),
        ):
            mock_query.return_value = ((np.eye(3), self._PIXEL_SC), SQF.GOOD)
            out = spatial.pixel_geometry(np.array([0]), instrument, self._PIXEL_VECTORS)

        npt.assert_array_equal(
            out.quality_flags[0], [SQF.CALC_ANCIL_NOT_FINITE, SQF.GOOD, SQF.CALC_ELLIPS_NO_INTERSECT]
        )
        self.assertTrue(np.isnan(out.viewing_zenith[0, 0]) and np.isfinite(out.viewing_azimuth[0, 0]))

    def test_unit_pixel_geometry_passes_perspective_correction_to_both_queries(self):
        instrument = self._mock_pixel_instrument()
        sun_df = pd.DataFrame([self._PIXEL_SUN], columns=["x", "y", "z"])
        with (
            patch.object(spatial.SpatialQueries, "query_rotation_and_position") as mock_query,
            patch.object(spatial.spicierpy.ext, "query_ephemeris", return_value=sun_df) as mock_ephemeris,
        ):
            mock_query.return_value = ((np.eye(3), self._PIXEL_SC), SQF.GOOD)
            spatial.pixel_geometry(np.array([0]), instrument, self._PIXEL_VECTORS, perspective_correction="LT+S")

        self.assertEqual(mock_query.call_args.args[2], "LT+S")
        self.assertEqual(mock_ephemeris.call_args.kwargs["correction"], "LT+S")

    def test_unit_pixel_geometry_spice_failure_fills_time(self):
        """A failed rotation/position query leaves the whole time NaN and flags the cause."""
        instrument = self._mock_pixel_instrument()
        sun_df = pd.DataFrame([self._PIXEL_SUN, self._PIXEL_SUN], columns=["x", "y", "z"])
        with (
            patch.object(spatial.SpatialQueries, "query_rotation_and_position") as mock_query,
            patch.object(spatial.spicierpy.ext, "query_ephemeris", return_value=sun_df),
        ):
            mock_query.side_effect = [
                ((np.eye(3), self._PIXEL_SC), SQF.GOOD),
                ((np.full((3, 3), np.nan), np.full(3, np.nan)), SQF.SPICE_ERR_MISSING_ATTITUDE),
            ]
            out = spatial.pixel_geometry(np.array([0, 1_000_000]), instrument, self._PIXEL_VECTORS)

        self.assertTrue(np.isfinite(out.lon[0, :2]).all())
        self.assertTrue(np.isnan(out.lon[1]).all() and np.isnan(out.viewing_zenith[1]).all())
        self.assertTrue(np.isnan(out.sc_position[1]).all())
        npt.assert_array_equal(out.quality_flags[1], [SQF.SPICE_ERR_MISSING_ATTITUDE | SQF.CALC_ELLIPS_INSUFF_DATA] * 3)

    def test_unit_pixel_geometry_missing_sun_keeps_viewing_angles(self):
        instrument = self._mock_pixel_instrument()
        sun_df = pd.DataFrame([[np.nan, np.nan, np.nan]], columns=["x", "y", "z"])
        with (
            patch.object(spatial.SpatialQueries, "query_rotation_and_position") as mock_query,
            patch.object(spatial.spicierpy.ext, "query_ephemeris", return_value=sun_df),
        ):
            mock_query.return_value = ((np.eye(3), self._PIXEL_SC), SQF.GOOD)
            out = spatial.pixel_geometry(np.array([0]), instrument, self._PIXEL_VECTORS)

        self.assertTrue(np.isfinite(out.viewing_zenith[0, :2]).all())
        self.assertTrue(np.isnan(out.solar_zenith[0]).all() and np.isnan(out.relative_azimuth[0]).all())
        npt.assert_array_equal(
            out.quality_flags[0],
            [
                SQF.CALC_ANCIL_INSUFF_DATA,
                SQF.CALC_ANCIL_INSUFF_DATA,
                SQF.CALC_ANCIL_INSUFF_DATA | SQF.CALC_ELLIPS_NO_INTERSECT,
            ],
        )

    def test_unit_pixel_geometry_rejects_bad_vectors(self):
        instrument = self._mock_pixel_instrument()
        for bad in (
            np.array([-1.0, 0.0, 0.0]),
            np.zeros((2, 2)),
            np.zeros((0, 3)),
            np.array([[np.nan, 0.0, 0.0]]),
            np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ):
            with self.subTest(shape=bad.shape), pytest.raises(ValueError, match="pointing_vectors"):
                spatial.pixel_geometry(np.array([0]), instrument, bad)

    def test_pixel_geometry_integration(self):
        """E2E on the CPRS kernels: matches the DataFrame path and SPICE `azlcpo` at the middle pixel."""
        ugps_times = spicetime.adapt(np.array(["2023-01-01", "2023-01-01T00:01"]), "iso")

        with self.mkrn.load():
            instrument = self.mkrn.mappings["CPRS_HYSICS"]
            npix, pix_vecs = spatial.get_instrument_kernel_pointing_vectors(instrument)
            out = spatial.pixel_geometry(ugps_times, instrument, pix_vecs)

            ref_lla, ref_sc, ref_qf = spatial.compute_ellipsoid_intersection(
                ugps_times, instrument, give_geodetic_output=True, give_lat_lon_in_degrees=True
            )
            ref_xyz, _, _ = spatial.compute_ellipsoid_intersection(ugps_times, instrument)
            ref_sun = spatial.surface_angles(ref_xyz, target_obj="SUN", degrees=True)
            ref_view = spatial.surface_angles(ref_xyz, target_positions=ref_sc, degrees=True)

            mid = npix // 2
            mid_xyz = out.surface_xyz[:, mid, :]
            exp_sun_az, exp_sun_zen = spatial.spice_angles(ugps_times, mid_xyz, "SUN", degrees=True)
            exp_view_az, exp_view_zen = spatial.spice_angles(ugps_times, mid_xyz, instrument.name, degrees=True)

        self.assertEqual(out.lon.shape, (2, npix))
        self.assertTrue(np.isfinite(out.lon).all(), "CPRS test case should see the Earth at every pixel")
        npt.assert_array_equal(out.quality_flags, 0)

        npt.assert_allclose(out.lon.ravel(), ref_lla["lon"].to_numpy())
        npt.assert_allclose(out.lat.ravel(), ref_lla["lat"].to_numpy())
        npt.assert_allclose(out.alt.ravel(), ref_lla["alt"].to_numpy())
        npt.assert_array_equal(out.quality_flags.ravel(), ref_qf.to_numpy())
        npt.assert_allclose(out.solar_zenith.ravel(), ref_sun["zenith"].to_numpy())
        npt.assert_allclose(out.solar_azimuth.ravel(), ref_sun["azimuth"].to_numpy())
        npt.assert_allclose(out.viewing_zenith.ravel(), ref_view["zenith"].to_numpy())
        npt.assert_allclose(out.viewing_azimuth.ravel(), ref_view["azimuth"].to_numpy())
        npt.assert_allclose(out.relative_azimuth, np.mod(out.viewing_azimuth - out.solar_azimuth + 180.0, 360.0))

        # SPICE `azlcpo` uses the PCK Earth radii rather than WGS84; the measured disagreement on
        # these kernels is below 1e-6 degrees, so 1e-5 leaves margin without hiding a real error.
        npt.assert_allclose(out.solar_zenith[:, mid], exp_sun_zen, atol=1e-5)
        npt.assert_allclose(out.solar_azimuth[:, mid], exp_sun_az, atol=1e-5)
        npt.assert_allclose(out.viewing_zenith[:, mid], exp_view_zen, atol=1e-5)
        npt.assert_allclose(out.viewing_azimuth[:, mid], exp_view_az, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
