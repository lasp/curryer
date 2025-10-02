import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pandas as pd
import xarray as xr

from curryer import meta, spicetime, spicierpy, utils
from curryer.compute import constants, elevation, spatial

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])

xr.set_options(display_width=120)
np.set_printoptions(linewidth=120)


class SpatialTestCase(unittest.TestCase):
    # Note that many of the routines in `spatial` were originally located in
    # the `pointing` module, where their original tests might remain.

    def setUp(self) -> None:
        root_dir = Path(__file__).parents[2]
        self.generic_dir = root_dir / "data" / "generic"
        self.data_dir = root_dir / "data" / "clarreo"
        self.test_dir = root_dir / "tests" / "data" / "clarreo"
        self.assertTrue(self.generic_dir.is_dir())
        self.assertTrue(self.data_dir.is_dir())
        self.assertTrue(self.test_dir.is_dir())

        self.mkrn = meta.MetaKernel.from_json(
            self.test_dir / "cprs_v01.kernels.tm.testcase1.json",
            relative=True,
            sds_dir=self.generic_dir,
        )
        self.mock_elev = MagicMock(elevation.Elevation, meters=False, degrees=False)
        self.mock_elev.local_minmax.return_value = -0.1, 9.0
        self.mock_elev.query.side_effect = lambda ll, lt: 8 - 6 * np.rad2deg(ll)

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

    def test_pixel_vectors(self):
        vectors_ds = xr.load_dataset(self.test_dir / "cprs_hysics_v01.pixel_vectors.nc")
        exp_vectors = np.stack([vectors_ds[col].values for col in ["x", "y", "z"]], axis=1)
        self.assertTupleEqual(exp_vectors.shape, (480, 3))

        with self.mkrn.load():
            npix, qry_vectors = spatial.pixel_vectors("CPRS_HYSICS")
            self.assertEqual(npix, 480)
            npt.assert_allclose(exp_vectors, qry_vectors)

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

        # Handling of non-intersecting vectors:
        #   * 1st vector is simple nadir.
        #   * 2nd vector is facing zenith (away from Earth).
        #   * 3rd vector barely misses the surface.
        #   * 4th vector barely hits the surface.
        #   * 5th vector is above the north pole (edge case of lon/lat div zero).
        vectors = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        positions = np.array(
            [[7e3, 0.0, 0.0], [7e3, 0.0, 0.0], [7e3, major * 1.0001, 0.0], [7e3, major * 0.9999, 0.0], [0.0, 0.0, 7e3]]
        )
        xyz = spatial.ray_intersect_ellipsoid(vectors, positions)
        self.assertIsInstance(xyz, np.ndarray)
        self.assertTupleEqual(xyz.shape, vectors.shape)
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
        self.assertIsInstance(lla, np.ndarray)
        self.assertTupleEqual(lla.shape, vectors.shape)
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

    def test_ellipsoid_intersect_real(self):
        et_times = spicetime.adapt(["2023-01-01"], "iso", "et")

        with self.mkrn.load():
            target_id = self.mkrn.mappings["CPRS_HYSICS"].id
            sample_et = et_times[0]

            npix, qry_vectors = spatial.pixel_vectors("CPRS_HYSICS")

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

    def test_sc_angles_simple(self):
        # TODO: Clean up.
        # # Directly nadir (geodetic).
        # obs_lla = np.array([-69.6365451, -31.2084545, 2.4191])
        # trg_lla = np.array([-69.6365451, -31.2084545, 202.4191])
        #
        # # North-east.
        # obs_lla = np.array([-69.6365451, -31.2084545, 2.4191])
        # trg_lla = np.array([-68.6365451, -30.2084545, 2.4191 + 111.3195 * 1])
        #
        # # East.
        # obs_lla = np.array([-69.6365451, -31.2084545, 2.4191])
        # trg_lla = np.array([-68.6365451, -31.2084545, 2.4191 + 111.3195 * 1])
        #
        # # Simple equator - east 1-deg.
        # obs_lla = np.array([90.0, 0.0, 0.0])
        # trg_lla = np.array([91.0, 0.0, 0.0 + 111.3195])
        #
        # # Simple equator - north 1-deg.
        # obs_lla = np.array([90.0, 0.0, 0.0])
        # trg_lla = np.array([90.0, 1.0, 0.0 + 111.3195])
        #
        # # Simple equator - south & west 1-deg.
        # obs_lla = np.array([90.0, 0.0, 0.0])
        # trg_lla = np.array([89.0, -1.0, 0.0 + 111.3195])

        obs_lla = np.array([0.0, 10.0, 0.0])
        trg_lla = np.array([1.5, 0.0, 0.0 + 111.3195])

        obs_ecef = spatial.geodetic_to_ecef(obs_lla, meters=False, degrees=True)
        trg_ecef = spatial.geodetic_to_ecef(trg_lla, meters=False, degrees=True)

        zen_out = spatial.calc_zenith(obs_ecef, trg_ecef, degrees=True)
        az_out = spatial.calc_azimuth(obs_ecef, trg_ecef, degrees=True)

        obs_lla_rad = np.deg2rad(obs_lla)
        obs_uvec = np.array(
            [
                np.cos(obs_lla_rad[1]) * np.cos(obs_lla_rad[0]),
                np.cos(obs_lla_rad[1]) * np.sin(obs_lla_rad[0]),
                np.sin(obs_lla_rad[1]),
            ]
        )

        sc_vec = trg_ecef - obs_ecef
        rng_vec = np.linalg.norm(sc_vec)
        sc_uvec = sc_vec / rng_vec
        # sc_uvec = sc_vec / np.linalg.norm(sc_vec)

        zen_rad = np.arccos(obs_uvec @ sc_uvec)
        zen_deg = np.rad2deg(zen_rad)

        east_uvec = np.array([-np.sin(obs_lla_rad[0]), np.cos(obs_lla_rad[0]), 0.0])
        north_uvec = np.cross(obs_uvec, east_uvec)

        az_l_cos = sc_vec @ east_uvec
        az_m_cos = sc_vec @ north_uvec
        az_rad = np.arctan2(az_l_cos, az_m_cos)  # TODO: Reg arctan?
        if az_rad < 0:
            az_rad += np.pi * 2
        # az_rad[az_rad < 0] += np.pi * 2
        az_deg = np.rad2deg(az_rad)

        # print(zen_out, zen_deg)
        # print(az_out, az_deg)
        npt.assert_allclose(zen_out, zen_deg)
        npt.assert_allclose(az_out, az_deg)
        # with self.mkrn.load():
        #     sun_positions = spicierpy.ext.query_ephemeris(
        #         ugps_times, target='SUN', observer='EARTH', ref_frame='ITRF93', allow_nans=True
        #     )
        # trg_ecef = sun_positions.values[0, :]

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
                # ( r, az, el, dr/dt, daz/dt, del/dt )
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

                solar_elevation = np.rad2deg(np.arcsin(norm(fixed_pos) @ norm(sun_xyz - fixed_pos)))

                # Direct method instead of: solar_zenith = 90 - solar_elevation
                solar_zenith = np.rad2deg(np.arccos(norm(fixed_pos) @ norm(sun_xyz - fixed_pos)))

                def azimuth(obs, trg, degrees=False):
                    # xy_ang = np.arccos(norm(obs[:2]) @ norm(trg[:2]))
                    xy_ang = np.arctan2(trg[1], trg[0]) - np.arctan2(obs[1], obs[0])
                    xy_dist = np.sin(xy_ang) * np.sqrt(trg[0] ** 2 + trg[1] ** 2)
                    az_ang = np.arctan2(xy_dist, trg[2] - obs[2])

                    if degrees:
                        az_ang = np.rad2deg(az_ang)
                    if az_ang < 0.0:
                        az_ang += 360 if degrees else np.pi * 2
                    return az_ang

                solar_azimuth = azimuth(fixed_pos, sun_xyz, degrees=True)

                npt.assert_allclose(solar_zenith, 90 - solar_elevation)
                npt.assert_allclose(exp_el, -solar_zenith + 90)
                npt.assert_allclose(exp_az, solar_azimuth)

    def test_ellipsoid_intersect_instrument(self):
        ugps_times = spicetime.adapt(np.array(["2023-01-01", "2023-01-01T00:01"]), "iso")

        with self.mkrn.load():
            npix, qry_vectors = spatial.pixel_vectors("CPRS_HYSICS")

            surf_points, sc_points, sqf = spatial.instrument_intersect_ellipsoid(
                ugps_times, self.mkrn.mappings["CPRS_HYSICS"]
            )
            self.assertIsInstance(surf_points, pd.DataFrame)
            self.assertIsInstance(sc_points, pd.DataFrame)
            self.assertIsInstance(sqf, pd.Series)
            self.assertTupleEqual(surf_points.shape, (npix * len(ugps_times), 3))
            mid_pixel_point = surf_points.loc[(ugps_times[0], npix // 2 + 1)].values

            # Compare results to the slower previous method.
            u1_inst = qry_vectors[npix // 2, :]
            exp_pt_surf, _, exp_vec_surf = spicierpy.sincpt(
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

    def test_terrain_correct_basic_45deg(self):
        out_srf_loc = spatial.terrain_correct_single(
            elev=self.mock_elev,
            ec_srf_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 0, 0, 0]),
            ec_sat_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 100, 100, 0]),
        )
        self.assertIsInstance(out_srf_loc, np.ndarray)
        npt.assert_allclose(out_srf_loc, np.array([0.067565, 0.0, 7.595]), rtol=1e-4)

    def test_terrain_correct_basic_nadir(self):
        out_srf_loc = spatial.terrain_correct_single(
            elev=self.mock_elev,
            ec_srf_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 0, 0, 0]),
            ec_sat_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 100, 0, 0]),
        )
        self.assertIsInstance(out_srf_loc, np.ndarray)
        npt.assert_allclose(out_srf_loc, np.array([0.0, 0.0, 8.0]))

    def test_terrain_correct_basic_way_off(self):
        out_srf_loc = spatial.terrain_correct_single(
            elev=self.mock_elev,
            ec_srf_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 0, 0, 0]),
            ec_sat_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 111, 340, 340]),
        )
        self.assertIsInstance(out_srf_loc, np.ndarray)
        npt.assert_allclose(out_srf_loc, np.array([0.187102, 0.188361, 6.877]), rtol=1e-4)

    def test_terrain_correct_basic_misc(self):
        self.mock_elev.query.side_effect = lambda ll, lt: 1.6
        out_srf_loc = spatial.terrain_correct_single(
            elev=self.mock_elev,
            ec_srf_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 0, 0, 0]),
            ec_sat_pos=np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 420, 420, 0]),
        )
        self.assertIsInstance(out_srf_loc, np.ndarray)
        npt.assert_allclose(out_srf_loc, np.array([0.01353132, 0.0, 1.6]), rtol=1e-4)

    def test_terrain_correct_edge_case_extreme_zenith(self):
        # Results in a zenith angle just above 85, the threshold.
        self.mock_elev.local_minmax.return_value = -0.153543799821783, 4.711536036914816
        self.mock_elev.query.side_effect = lambda ll, lt: 0.77285259

        out_srf_loc, out_qf = spatial.terrain_correct(
            elev=self.mock_elev,
            ec_srf_pos=np.array([4692.66276894, 1291.74275934, 4108.18749653]),
            ec_sat_pos=np.array([4.20598328e03, 1.48265257e-01, 5.30877902e03]),
        )
        self.assertIsInstance(out_srf_loc, np.ndarray)
        self.assertTrue(np.isnan(out_srf_loc).all())
        self.assertEqual(out_qf, constants.SpatialQualityFlags.CALC_TERRAIN_EXTREME_ZENITH)

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
            ),  # Just over
            ec_sat_pos=np.array(
                [
                    [4.20598328e03, 1.48265257e-01, 5.30877902e03],
                    [4.20598328e03, 1.48265257e-01, 5.30877902e03],
                    [4.20598328e03, 1.48265257e-01, 5.30877902e03],
                    [4.20598328e03, 1.48265257e-01, 5.30877902e03],
                ]
            ),
        )
        self.assertIsInstance(out_srf_loc, np.ndarray)
        self.assertTupleEqual(out_srf_loc.shape, (4, 3))
        self.assertTrue(np.isnan(out_srf_loc[0, :]).all())
        self.assertTrue(np.isnan(out_srf_loc[3, :]).all())
        npt.assert_allclose(
            out_qf,
            np.array(
                [
                    constants.SpatialQualityFlags.CALC_TERRAIN_EXTREME_ZENITH,
                    0,
                    0,
                    constants.SpatialQualityFlags.CALC_TERRAIN_EXTREME_ZENITH,
                ]
            ),
        )

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
        self.assertIsInstance(out_qf, np.ndarray)
        self.assertIsInstance(out_srf_loc, np.ndarray)
        self.assertTupleEqual((5, 3), out_srf_loc.shape)
        npt.assert_allclose(out_srf_loc[0, :], np.array([0.014368, 0.0, 1.6]), rtol=1e-4)
        self.assertTrue(np.isfinite(out_srf_loc[:4, :]).all())
        self.assertTrue(np.isnan(out_srf_loc[4, :]).all())

    def test_terrain_correct_performance(self):
        elev = elevation.Elevation(meters=False, degrees=False)
        self.assertIsNotNone(elev)

        elev_region = elev.local_region(*np.deg2rad((-110, -85, 25, 45)))
        self.assertIsNotNone(elev_region)

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
        local_minmax = elev_region.local_minmax()

        t0 = pd.Timestamp.utcnow()

        npts = 480 * 4500
        out_srf_locs, out_qf = spatial.terrain_correct(
            elev=elev_region,
            ec_srf_pos=np.tile(ec_srf_pos, (npts, 1)),
            ec_sat_pos=np.tile(ec_sat_pos, (npts, 1)),
            local_minmax=local_minmax,
        )

        t1 = pd.Timestamp.utcnow()
        logger.info("Loops completed in: %s", t1 - t0)
        logger.info(utils.format_performance(elev, p=3, ascending=True))
        logger.info(utils.format_performance(elev_region, p=3, ascending=True))

        self.assertTupleEqual((npts, 3), out_srf_locs.shape)
        self.assertTrue(np.isfinite(out_srf_locs).all())

    def test_ecef_to_geodetic_simple(self):
        xx = constants.WGS84_SEMI_MAJOR_AXIS_KM + 420
        yy = 420
        zz = 420

        exp_lla = spicierpy.recgeo(
            rectan=[xx, yy, zz],
            as_deg=True,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )

        a = constants.WGS84_SEMI_MAJOR_AXIS_KM
        b = constants.WGS84_SEMI_MINOR_AXIS_KM

        e2 = (a**2 - b**2) / a**2
        ep2 = (a**2 - b**2) / b**2

        p = np.sqrt(xx**2 + yy**2)
        ff = 54 * b**2 * zz**2
        gg = p**2 + (1 - e2) * zz**2 - e2 * (a**2 - b**2)
        c = (e2 * e2) * ff * p**2 / gg**3
        s = (1 + c + np.sqrt(c**2 + 2 * c)) ** (1 / 3)
        k = s + 1 + 1 / s
        pp = ff / (3 * k**2 * gg**2)
        qq = np.sqrt(1 + 2 * (e2 * e2) * pp)

        r0 = (-1 * pp * e2 * p) / (1 + qq) + np.sqrt(
            (1 / 2) * a**2 * (1 + 1 / qq) - (pp * (1 - e2) * zz**2) / (qq * (1 + qq)) - (1 / 2) * pp * p**2
        )
        uu = np.sqrt((p - e2 * r0) ** 2 + zz**2)
        vv = np.sqrt((p - e2 * r0) ** 2 + (1 - e2) * zz**2)
        z0 = b**2 * zz / (a * vv)

        h = uu * (1 - b**2 / (a * vv))
        phi = np.arctan((zz + ep2 * z0) / p)
        lam = np.arctan2(yy, xx)

        lla = np.array([np.rad2deg(lam), np.rad2deg(phi), h])

        self.assertIsInstance(lla, np.ndarray)
        self.assertTupleEqual((3,), lla.shape)
        self.assertTrue(np.isfinite(lla).all())
        npt.assert_allclose(lla, exp_lla, rtol=1e-13)

    def test_ecef_to_geodetic_misc(self):
        sc_pos_xyz = np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 420, 420, 420])
        lla = spatial.ecef_to_geodetic(sc_pos_xyz, meters=False, degrees=True)
        self.assertIsInstance(lla, np.ndarray)
        self.assertTupleEqual((3,), lla.shape)
        self.assertTrue(np.isfinite(lla).all())

        exp_lla = spicierpy.recgeo(
            rectan=sc_pos_xyz,
            as_deg=True,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )
        npt.assert_allclose(lla, exp_lla, rtol=1e-13)

        sc_pos_xyz_2d = np.tile(sc_pos_xyz, (5, 1))
        self.assertTupleEqual((5, 3), sc_pos_xyz_2d.shape)

        lla = spatial.ecef_to_geodetic(sc_pos_xyz_2d, meters=False, degrees=True)
        self.assertIsInstance(lla, np.ndarray)
        self.assertTupleEqual((5, 3), lla.shape)
        self.assertTrue(np.isfinite(lla).all())
        for i in range(sc_pos_xyz_2d.shape[0]):
            npt.assert_allclose(lla[i, :], exp_lla, rtol=1e-13)

    def test_geodetic_to_ecef_simple(self):
        lla = np.array([-107.25, 42, 450])  # Degrees & KM.

        xyz = spatial.geodetic_to_ecef(lla, degrees=True)
        self.assertIsInstance(xyz, np.ndarray)
        self.assertTupleEqual((3,), xyz.shape)
        self.assertTrue(np.isfinite(xyz).all())

        exp_xyz = spicierpy.georec(
            lon=np.deg2rad(-107.25),
            lat=np.deg2rad(42),
            alt=450,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )
        npt.assert_allclose(xyz, exp_xyz, rtol=1e-13)

        lla_2d = np.tile(lla, (5, 1))
        self.assertTupleEqual((5, 3), lla_2d.shape)

        xyz = spatial.geodetic_to_ecef(lla_2d, degrees=True)
        self.assertIsInstance(xyz, np.ndarray)
        self.assertTupleEqual((5, 3), xyz.shape)
        self.assertTrue(np.isfinite(xyz).all())
        for i in range(lla_2d.shape[0]):
            npt.assert_allclose(xyz[i, :], exp_xyz, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
