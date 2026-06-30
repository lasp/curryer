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
import pytest

from curryer import meta, spicetime, spicierpy, utils
from curryer.compute import constants, geometry

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

    def test_satellite_altitude_known_values(self):
        # 500 km above the +X equator and above the north pole.
        equator = np.array([constants.WGS84_SEMI_MAJOR_AXIS_KM + 500.0, 0.0, 0.0])
        pole = np.array([0.0, 0.0, constants.WGS84_SEMI_MINOR_AXIS_KM + 500.0])
        npt.assert_allclose(geometry.satellite_altitude(np.stack([equator, pole])), [500.0, 500.0], atol=1e-6)


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

        assert list(df.columns) == ["spacecraft_radius"]
        npt.assert_allclose(df["spacecraft_radius"].values, [7000.0, 7000.0])
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

        assert list(df.columns) == [
            "spacecraft_position_x",
            "spacecraft_position_y",
            "spacecraft_position_z",
            "subsolar_latitude",
            "subsolar_longitude",
            "subsolar_colatitude",
        ]
        npt.assert_allclose(
            df[["spacecraft_position_x", "spacecraft_position_y", "spacecraft_position_z"]].values, sc_pos
        )
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
        return _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos)

    def test_default_fields_are_ephemeris_only(self):
        counter = {}
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, self._full_fakes(counter)):
            df = geo.get_geometry(self.UGPS)

        # Default is exactly the ephemeris-only field set...
        expected = [col for field in geometry._DEFAULT_FIELDS for col in geometry._FIELDS[field].columns]
        assert list(df.columns) == expected
        # ...and that set needs only ephemeris providers (no attitude/FOV), so the
        # default never implicitly requires attitude kernels as fields are added.
        for field in geometry._DEFAULT_FIELDS:
            assert geometry._FIELDS[field].providers <= geometry._EPHEMERIS_PROVIDERS

    def test_subset_queries_only_needed_providers(self):
        counter = {}
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, self._full_fakes(counter)):
            geo.get_geometry(self.UGPS, fields=["subsatellite", "earth_sun_distance"])
        # subsatellite -> sc_position, earth_sun_distance -> sun_position only.
        assert counter == {"sc_position": 1, "sun_position": 1}

    def test_sc_altitude_field(self):
        counter = {}
        # 500 km and 800 km above the +X equator.
        sc_pos = np.array(
            [
                [constants.WGS84_SEMI_MAJOR_AXIS_KM + 500.0, 0.0, 0.0],
                [constants.WGS84_SEMI_MAJOR_AXIS_KM + 800.0, 0.0, 0.0],
            ]
        )
        geo = self._build()
        with patch.dict(geometry._PROVIDERS, _fake_providers(counter, sc_position=sc_pos)):
            df = geo.get_geometry(self.UGPS, fields=["sc_altitude"])
        assert list(df.columns) == ["spacecraft_altitude"]
        npt.assert_allclose(df["spacecraft_altitude"].values, [500.0, 800.0], atol=1e-6)
        assert counter == {"sc_position": 1}

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
        assert "spacecraft_radius" in df.columns
        assert "subsatellite_latitude" in df.columns
        assert "spacecraft_position_x" in df.columns

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

    def test_all_nan_provider_warns_only_when_fully_empty(self, caplog):
        # An all-NaN provider is the documented missing-kernel signal and must
        # warn; a partial gap (some finite rows) is normal coverage and must not.
        geo = self._build()

        partial = np.array([[7000.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        with patch.dict(geometry._PROVIDERS, _fake_providers({}, sc_position=partial)):
            with caplog.at_level(logging.WARNING, logger="curryer.compute.geometry"):
                geo.get_geometry(self.UGPS, fields=["sc_radius"])
        assert not [r for r in caplog.records if "all-NaN" in r.getMessage()]

        caplog.clear()
        empty = np.full((2, 3), np.nan)
        with patch.dict(geometry._PROVIDERS, _fake_providers({}, sc_position=empty)):
            with caplog.at_level(logging.WARNING, logger="curryer.compute.geometry"):
                geo.get_geometry(self.UGPS, fields=["sc_radius"])
        warnings = [r.getMessage() for r in caplog.records if "all-NaN" in r.getMessage()]
        assert warnings
        assert "sc_position" in warnings[0]

    @pytest.mark.parametrize("field", list(geometry._FIELDS))
    def test_nan_propagates_per_field(self, field):
        # Fill contract: every field must NaN exactly the rows its inputs are
        # missing and stay finite elsewhere, whichever provider feeds it. Both
        # providers carry a NaN second row so the test is provider-agnostic.
        counter = {}
        sc_pos = np.array([[7000.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        sun_pos = np.array([[1.5e8, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        geo = self._build()
        fakes = _fake_providers(counter, sc_position=sc_pos, sun_position=sun_pos)
        with patch.dict(geometry._PROVIDERS, fakes):
            df = geo.get_geometry(self.UGPS, fields=[field])

        columns = list(geometry._FIELDS[field].columns)
        assert np.isfinite(df.iloc[0][columns]).all()
        assert df.iloc[1][columns].isna().all()


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
        with self.mkrn.load():
            df = geo.get_geometry(self.ugps)

        for field in geometry.GeometryData.available_fields():
            for column in geometry._FIELDS[field].columns:
                self.assertIn(column, df.columns)

        # These fields are all position-derived -> gap-free over covered times.
        self.assertFalse(df.isna().any().any(), msg=f"unexpected NaNs:\n{df.isna().sum()}")

        # Physical sanity ranges.
        self.assertTrue(df["subsatellite_colatitude"].between(0, 180).all())
        self.assertTrue(df["subsatellite_longitude"].between(-180, 180).all())
        self.assertTrue(df["spacecraft_radius"].between(6500, 7500).all())  # ISS-class orbit.
        self.assertTrue(df["earth_sun_distance"].between(0.97, 1.03).all())

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
