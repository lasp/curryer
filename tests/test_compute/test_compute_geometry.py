"""Tests for the geometry data-field framework.

The pure-math leaves (in ``geometry``) are tested against known values and the
``GeometryData`` orchestration with fake providers, so the bulk of the suite
needs no SPICE kernels. ``GeometryIntegrationTestCase`` exercises the real
provider paths against the committed CLARREO testcase kernels.
"""

import logging
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from curryer import meta, spicetime, spicierpy, utils
from curryer.compute import constants, geometry, spatial

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class TestGeometryLeaves:
    """Pure-math leaf functions in ``geometry`` (no SPICE)."""

    def test_sc_radius_known_values(self):
        major = constants.WGS84_SEMI_MAJOR_AXIS_KM
        position = np.array([[major, 0.0, 0.0], [0.0, 3.0, 4.0]])
        npt.assert_allclose(geometry.sc_radius(position), [major, 5.0])

    def test_sc_radius_single_point(self):
        radius = geometry.sc_radius(np.array([0.0, 0.0, 7000.0]))
        assert np.ndim(radius) == 0
        npt.assert_allclose(radius, 7000.0)

    def test_sc_radius_bad_shape(self):
        with pytest.raises(ValueError, match="3 values per point"):
            geometry.sc_radius(np.array([1.0, 2.0]))

    def test_colatitude_degrees(self):
        npt.assert_allclose(geometry.colatitude(np.array([90.0, 0.0, -90.0])), [0.0, 90.0, 180.0])

    def test_colatitude_radians(self):
        npt.assert_allclose(geometry.colatitude(np.array([0.0]), degrees=False), [np.pi / 2])

    def test_subobserver_point_equator(self):
        # A point on the +X equator: lat 0, lon 0, colat 90.
        position = np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 500.0, 0.0, 0.0])
        lat, lon, colat = geometry.subobserver_point(position)
        npt.assert_allclose([lat, lon, colat], [0.0, 0.0, 90.0], atol=1e-9)

    def test_subobserver_point_north_pole(self):
        position = np.array([0.0, 0.0, constants.WGS84_SEMI_MINOR_AXIS_KM + 500.0])
        lat, lon, colat = geometry.subobserver_point(position)
        npt.assert_allclose(lat, 90.0, atol=1e-9)
        npt.assert_allclose(colat, 0.0, atol=1e-9)

    def test_subobserver_point_colat_matches_latitude(self):
        rng = np.random.default_rng(0)
        position = rng.uniform(-7000, 7000, size=(5, 3))
        out = geometry.subobserver_point(position)
        npt.assert_allclose(out[:, 2], 90.0 - out[:, 0])

    def test_earth_sun_distance_au(self):
        position = np.array([[constants.KM_PER_ASTRONOMICAL_UNIT, 0.0, 0.0]])
        npt.assert_allclose(geometry.earth_sun_distance(position), [1.0])
        npt.assert_allclose(geometry.earth_sun_distance(position, au=False), [constants.KM_PER_ASTRONOMICAL_UNIT])


def _fake_providers(call_counter, **values):
    """Build a patch dict of counting fake providers from ``{key: value}``."""

    def _make(key, value):
        def _provider(ugps_times, ctx):
            call_counter[key] = call_counter.get(key, 0) + 1
            return value

        return _provider

    return {key: _make(key, value) for key, value in values.items()}


class TestGeometryOrchestration:
    """``GeometryData`` field selection, minimal querying, and assembly."""

    UGPS = np.array([1_000_000, 2_000_000])

    def _build(self):
        # __init__ stores raw names and touches no SPICE; safe without kernels.
        return geometry.GeometryData("SOME_INSTRUMENT")

    def test_get_geometry_single_field_queries_only_its_provider(self):
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, _fake_providers(counter, sc_position=sc_pos)):
            df = geo.get_geometry(self.UGPS, fields=["sc_radius"])

        assert list(df.columns) == ["scradius"]
        npt.assert_allclose(df["scradius"].values, [7000.0, 7000.0])
        assert df.index.name == "ugps"
        # Only sc_position was queried; sun_position untouched.
        assert counter == {"sc_position": 1}

    def test_get_geometry_multiple_fields_union_of_providers(self):
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
        sun_pos = np.array([[1.5e8, 0.0, 0.0], [1.5e8, 1.0e6, 0.0]])
        geo = self._build()
        fakes = _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos)
        with patch.dict(geometry._PROVIDERS, fakes):
            df = geo.get_geometry(self.UGPS, fields=["sc_position", "subsolar"])

        assert list(df.columns) == ["scx", "scy", "scz", "subsollat", "subsollon", "subsolcolat"]
        npt.assert_allclose(df[["scx", "scy", "scz"]].values, sc_pos)
        # Each needed provider queried once.
        assert counter == {"sc_position": 1, "sun_position": 1}

    def test_get_vectors_returns_n3_arrays(self):
        counter = {}
        sc_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, _fake_providers(counter, sc_position=sc_pos)):
            vecs = geo.get_vectors(self.UGPS, fields=["sc_position"])

        assert vecs["sc_position"].shape == (2, 3)
        npt.assert_allclose(vecs["sc_position"], sc_pos)

    def _full_fakes(self, counter):
        """Fakes for every registered provider, sized to UGPS (N=2)."""
        sc_pos = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
        sun_pos = np.array([[1.5e8, 0.0, 0.0], [1.5e8, 1.0e6, 0.0]])
        # Boresights point back toward Earth center so the footprint ray-cast hits.
        boresight = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        return _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos, boresight=boresight)

    def test_default_fields_are_ephemeris_only(self):
        # fields=None defaults to the ephemeris-only set (valid for any observer);
        # attitude/instrument fields are opt-in, not in the default output, and
        # the attitude provider is never queried by default.
        counter = {}
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, self._full_fakes(counter)):
            df = geo.get_geometry(self.UGPS)

        ephemeris_fields = list(geometry._DEFAULT_FIELDS)
        attitude_fields = [f for f in geometry._FIELDS if f not in ephemeris_fields]
        assert attitude_fields  # guard: this test is meaningful only with some
        for field in ephemeris_fields:
            for column in geometry._FIELDS[field].columns:
                assert column in df.columns
        for field in attitude_fields:
            for column in geometry._FIELDS[field].columns:
                assert column not in df.columns
        # No attitude (non-ephemeris) provider was queried for the default set.
        attitude_providers = set(geometry._PROVIDERS) - geometry._EPHEMERIS_PROVIDERS
        assert attitude_providers.isdisjoint(counter)

    def test_subset_queries_only_needed_providers(self):
        counter = {}
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, self._full_fakes(counter)):
            geo.get_geometry(self.UGPS, fields=["subsatellite", "earth_sun_distance"])
        # subsatellite -> sc_position, earth_sun_distance -> sun_position only.
        assert counter == {"sc_position": 1, "sun_position": 1}

    def test_unknown_field_raises(self):
        geo = self._build()
        with pytest.raises(KeyError, match="Unknown geometry field"):
            geo.get_geometry(self.UGPS, fields=["not_a_field"])

    def test_shared_provider_queried_once(self):
        # Requesting sc_radius + subsatellite + sc_position must hit sc_position
        # exactly once (shared provider, not re-queried per field).
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, _fake_providers(counter, sc_position=sc_pos)):
            df = geo.get_geometry(self.UGPS, fields=["sc_radius", "subsatellite", "sc_position"])

        assert counter == {"sc_position": 1}
        assert "scradius" in df.columns
        assert "subsatlat" in df.columns
        assert "scx" in df.columns

    def test_nan_provider_row_propagates(self):
        # A data gap (NaN provider row) must surface as NaN, not crash, so
        # callers can detect it; finite rows stay finite.
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, _fake_providers(counter, sc_position=sc_pos)):
            df = geo.get_geometry(self.UGPS, fields=["sc_radius", "subsatellite"])
        assert np.isfinite(df.iloc[0]).all()
        assert df.iloc[1].isna().all()

    @pytest.mark.parametrize("field", list(geometry._FIELDS))
    def test_nan_propagates_per_field(self, field):
        # Fill contract: every field must NaN exactly the rows its inputs are
        # missing and stay finite elsewhere, whichever provider feeds it. Both
        # providers carry a NaN second row so the test is provider-agnostic.
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        sun_pos = np.array([[1.5e8, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        boresight = np.array([[-1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        geo = self._build()
        fakes = _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos, boresight=boresight)
        with patch.dict(geometry._PROVIDERS, fakes):
            df = geo.get_geometry(self.UGPS, fields=[field])

        columns = list(geometry._FIELDS[field].columns)
        assert np.isfinite(df.iloc[0][columns]).all()
        assert df.iloc[1][columns].isna().all()

    def test_boresight_and_surface_colatitude(self):
        # boresight is a passthrough of its provider; surface_colatitude is the
        # colatitude (90 - lat) of where the boresight, cast from the S/C
        # position, meets the ellipsoid. Both rows are equatorial intercepts
        # (lat 0 -> colat 90); the differing longitudes guard against the field
        # reading the longitude column instead of latitude.
        counter = {}
        boresight = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        sc_pos = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
        geo = self._build()
        fakes = _fake_providers(counter, boresight=boresight, sc_position=sc_pos)
        with patch.dict(geometry._PROVIDERS, fakes):
            df = geo.get_geometry(self.UGPS, fields=["boresight", "surface_colatitude"])

        npt.assert_allclose(df[["boresightx", "boresighty", "boresightz"]].values, boresight)
        npt.assert_allclose(df["surfcolat"].values, [90.0, 90.0], atol=1e-9)
        assert counter == {"boresight": 1, "sc_position": 1}

    def test_surface_angles_physical_nadir(self):
        # Boresight cast straight down from an equatorial sub-satellite point: the
        # satellite sits at the footprint's local zenith, so the viewing zenith is
        # ~0. The Sun on +Y is on the local horizon of the +X footprint (zenith ~90,
        # with a slight excess from finite-Sun parallax as the footprint is offset
        # from the geocenter) and at the local zenith of the +Y footprint (zenith 0).
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
        boresight = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        sun_pos = np.array([[0.0, 1.5e8, 0.0], [0.0, 1.5e8, 0.0]])
        geo = self._build()
        fakes = _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos, boresight=boresight)
        with patch.dict(geometry._PROVIDERS, fakes):
            df = geo.get_geometry(self.UGPS, fields=["viewing_zenith", "solar_zenith"])

        npt.assert_allclose(df["viewzen"].values, [0.0, 0.0], atol=1e-6)
        npt.assert_allclose(df["solzen"].values, [90.0, 0.0], atol=0.05)

    def test_surface_angles_compose_spatial_leaves(self):
        # The fields must wire the documented spatial leaves over the boresight
        # footprint. An off-nadir boresight makes the azimuths non-degenerate; the
        # registry output must equal a direct leaf composition, and each provider
        # is queried once across the four fields.
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [0.0, 7200.0, 0.0]])
        sun_pos = np.array([[1.0e8, 1.0e8, 5.0e7], [-1.2e8, 0.4e8, 2.0e7]])
        boresight = np.array([[-0.9, 0.3, 0.3], [0.2, -0.9, 0.3]])
        boresight /= np.linalg.norm(boresight, axis=1)[:, None]
        geo = self._build()
        fakes = _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos, boresight=boresight)
        fields = ["viewing_zenith", "solar_zenith", "viewing_azimuth", "relative_azimuth"]
        with patch.dict(geometry._PROVIDERS, fakes):
            df = geo.get_geometry(self.UGPS, fields=fields)

        footprint = spatial.ray_intersect_ellipsoid(boresight, sc_pos)
        exp_vaz = spatial.calc_azimuth(footprint, sc_pos, degrees=True)
        exp_saz = spatial.calc_azimuth(footprint, sun_pos, degrees=True)
        npt.assert_allclose(df["viewzen"].values, spatial.calc_zenith(footprint, sc_pos, degrees=True))
        npt.assert_allclose(df["solzen"].values, spatial.calc_zenith(footprint, sun_pos, degrees=True))
        npt.assert_allclose(df["viewaz"].values, exp_vaz)
        npt.assert_allclose(df["relaz"].values, np.mod(exp_vaz - exp_saz, 360.0))
        assert counter == {"boresight": 1, "sc_position": 1, "sun_position": 1}

    def test_footprint_memoized_across_surface_angle_fields(self):
        # Every surface-angle field shares the boresight footprint; the memo on the
        # per-request providers dict casts the ray once, not once per field.
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [0.0, 7200.0, 0.0]])
        sun_pos = np.array([[1.0e8, 1.0e8, 5.0e7], [-1.2e8, 0.4e8, 2.0e7]])
        boresight = np.array([[-0.9, 0.3, 0.3], [0.2, -0.9, 0.3]])
        boresight /= np.linalg.norm(boresight, axis=1)[:, None]
        geo = self._build()
        fakes = _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos, boresight=boresight)
        fields = ["viewing_zenith", "solar_zenith", "viewing_azimuth", "relative_azimuth"]
        with (
            patch.dict(geometry._PROVIDERS, fakes),
            patch.object(geometry.spatial, "ray_intersect_ellipsoid", wraps=spatial.ray_intersect_ellipsoid) as m_ray,
        ):
            df = geo.get_geometry(self.UGPS, fields=fields)

        assert m_ray.call_count == 1
        assert np.isfinite(df.values).all()

    def test_relative_azimuth_is_lossless_not_folded(self):
        # relative_azimuth keeps the full [0, 360) range; a [0, 180] fold (e.g.
        # CERES) is the caller's to apply. Mirroring the Sun across the footprint's
        # principal plane (sign-flipped East component) maps the relative azimuth to
        # 360 - raa: a folded field would collapse the pair to one value, and one of
        # the pair lands above 180 -- impossible under a fold.
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [7000.0, 0.0, 0.0]])
        boresight = np.array([[-0.95, 0.0, 0.31], [-0.95, 0.0, 0.31]])
        boresight /= np.linalg.norm(boresight, axis=1)[:, None]
        sun_pos = np.array([[1.0e8, 1.0e8, 8.0e7], [1.0e8, -1.0e8, 8.0e7]])
        geo = self._build()
        fakes = _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos, boresight=boresight)
        with patch.dict(geometry._PROVIDERS, fakes):
            df = geo.get_geometry(self.UGPS, fields=["relative_azimuth"])

        relaz = df["relaz"].values
        assert ((relaz >= 0.0) & (relaz < 360.0)).all()
        # The mirrored pair sums to 360 (distinct values), one above 180 -- a fold
        # to [0, 180] could produce neither.
        npt.assert_allclose(relaz[0] + relaz[1], 360.0, atol=1e-6)
        assert relaz.max() > 180.0

    def test_boresight_provider_is_pure_attitude_transform(self):
        # The boresight provider must rotate the IK boresight with pxform only --
        # no ephemeris query -- so it never duplicates the sc_position pass and a
        # position gap cannot null an otherwise-available attitude. Patch the SPICE
        # pieces and fail loudly if it reaches for spkezr.
        ctx = geometry.GeometryData("INST")
        ik = np.array([0.0, 0.0, 1.0])
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about Z
        with (
            patch.object(geometry.spicetime, "adapt", return_value=np.array([10.0, 20.0])),
            patch.object(geometry.spicierpy.obj, "Body") as m_body,
            patch.object(geometry.spicierpy.obj, "Frame") as m_frame,
            patch.object(geometry.spicierpy.ext, "instrument_boresight", return_value=ik) as m_bore,
            patch.object(geometry.spicierpy, "pxform", return_value=rot) as m_pxform,
            patch.object(geometry.spicierpy, "spkezr", side_effect=AssertionError("queried ephemeris")),
        ):
            m_body.return_value.frame.name = "INST_FRAME"
            m_frame.return_value.name = "ITRF93"
            out = geometry._provider_boresight(self.UGPS, ctx)

        npt.assert_allclose(out, np.array([rot @ ik, rot @ ik]))
        m_bore.assert_called_once_with("INST", norm=True)
        m_pxform.assert_called_with("INST_FRAME", "ITRF93", 20.0)
        assert m_pxform.call_count == 2  # one rotation per sample, no ephemeris pass

    def test_boresight_provider_nan_fills_missing_fov(self):
        # A body with no defined FOV makes instrument_boresight raise during the
        # one-time pointing lookup. Under allow_nans the provider must NaN-fill
        # (fill contract); without it, the SPICE error propagates.
        geo = geometry.GeometryData("SPACECRAFT")  # no instrument FOV/IK
        boom = geometry.spicierpy.SpiceyError("SPICE(NOFRAMECONNECT)")
        with (
            patch.object(geometry.spicetime, "adapt", return_value=np.array([10.0, 20.0])),
            patch.object(geometry.spicierpy.obj, "Body"),
            patch.object(geometry.spicierpy.obj, "Frame"),
            patch.object(geometry.spicierpy.ext, "instrument_boresight", side_effect=boom),
        ):
            out = geometry._provider_boresight(self.UGPS, geo)
            assert out.shape == (2, 3)
            assert np.isnan(out).all()

            geo.allow_nans = False
            with pytest.raises(geometry.spicierpy.SpiceyError):
                geometry._provider_boresight(self.UGPS, geo)


class GeometryIntegrationTestCase(unittest.TestCase):
    """End-to-end tests exercising the real SPICE provider paths.

    Uses the committed CLARREO testcase kernels (same fixture as the spatial
    integration tests): instrument ``CPRS_HYSICS`` over 2023-01-01.
    """

    @classmethod
    def setUpClass(cls):
        root_dir = Path(__file__).parents[2]
        cls.generic_dir = root_dir / "data" / "generic"
        cls.test_dir = root_dir / "tests" / "data" / "clarreo"
        assert cls.generic_dir.is_dir()
        assert cls.test_dir.is_dir()

        cls.mkrn = meta.MetaKernel.from_json(
            cls.test_dir / "cprs_v01.kernels.tm.testcase1.json",
            relative=True,
            sds_dir=cls.generic_dir,
        )
        cls.instrument = "CPRS_HYSICS"
        cls.ugps = spicetime.adapt(np.array(["2023-01-01", "2023-01-01T00:01"]), "iso")

    def test_get_geometry_all_fields_real_kernels(self):
        geo = geometry.GeometryData(self.instrument)
        # Request every field explicitly: the attitude fields are opt-in (the
        # default set is ephemeris-only), and CPRS_HYSICS has an instrument FOV.
        all_fields = geometry.GeometryData.available_fields()
        with self.mkrn.load():
            df = geo.get_geometry(self.ugps, fields=all_fields)

        for field in all_fields:
            for column in geometry._FIELDS[field].columns:
                self.assertIn(column, df.columns)

        # Position-derived fields are gap-free over covered times (unlike the
        # attitude-derived boresight/surfcolat, which may be NaN in attitude gaps).
        position_columns = [
            "subsatlat",
            "subsatlon",
            "subsatcolat",
            "subsollat",
            "subsollon",
            "subsolcolat",
            "scradius",
            "earthsundist",
            "scx",
            "scy",
            "scz",
        ]
        self.assertFalse(
            df[position_columns].isna().any().any(),
            msg=f"unexpected NaNs:\n{df[position_columns].isna().sum()}",
        )

        # Physical sanity ranges.
        self.assertTrue(df["subsatcolat"].between(0, 180).all())
        self.assertTrue(df["subsatlon"].between(-180, 180).all())
        self.assertTrue(df["scradius"].between(6500, 7500).all())  # ISS-class orbit.
        self.assertTrue(df["earthsundist"].between(0.97, 1.03).all())

        # Boresight is a unit direction in ECEF where the attitude is available.
        boresight = df[["boresightx", "boresighty", "boresightz"]].values
        finite = np.isfinite(boresight).all(axis=1)
        self.assertTrue(finite.any(), msg="boresight all-NaN over covered times")
        npt.assert_allclose(np.linalg.norm(boresight[finite], axis=1), 1.0, atol=1e-6)

        # Surface colatitude is in [0, 180] where the boresight hits the ellipsoid.
        surfcolat = df["surfcolat"]
        self.assertTrue(surfcolat.notna().any(), msg="surface colatitude all-NaN over covered times")
        self.assertTrue(surfcolat.dropna().between(0, 180).all())

        # Surface angles resolve where the boresight does, in their documented
        # ranges: zeniths in [0, 180], azimuths in [0, 360).
        self.assertTrue(df["viewzen"].notna().any(), msg="viewing zenith all-NaN over covered times")
        for col in ("viewzen", "solzen"):
            self.assertTrue(df[col].dropna().between(0, 180).all())
        for col in ("viewaz", "relaz"):
            vals = df[col].dropna()
            self.assertTrue(((vals >= 0.0) & (vals < 360.0)).all())

        # The viewing angles must match the existing surface_angles primitive (what
        # downstream calls today) over the covered rows -- same answer, computed
        # without its redundant ephemeris re-query or geodetic round-trip.
        sc = df.loc[finite, ["scx", "scy", "scz"]].values
        footprint = spatial.ray_intersect_ellipsoid(boresight[finite], sc)
        idx = df.index[finite]
        surf_df = pd.DataFrame(footprint, columns=["x", "y", "z"], index=idx)
        tgt_df = pd.DataFrame(sc, columns=["x", "y", "z"], index=idx)
        ref = spatial.surface_angles(surf_df, target_positions=tgt_df, degrees=True)
        npt.assert_allclose(df.loc[finite, "viewzen"].values, ref["zenith"].values, atol=1e-9)
        npt.assert_allclose(df.loc[finite, "viewaz"].values, ref["azimuth"].values, atol=1e-9)

    def test_get_vectors_itrf93_matches_direct_query(self):
        geo = geometry.GeometryData(self.instrument)
        with self.mkrn.load():
            vectors = geo.get_vectors(self.ugps, fields=["sc_position"])
            reference = spicierpy.ext.query_ephemeris(
                self.ugps, target=self.instrument, observer="EARTH", ref_frame="ITRF93", velocity=False
            )

        self.assertEqual(vectors["sc_position"].shape, (2, 3))
        # sc_position is exactly the ITRF93 ephemeris position.
        npt.assert_allclose(vectors["sc_position"], reference[["x", "y", "z"]].values, rtol=1e-9)
