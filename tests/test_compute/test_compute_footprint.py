import logging

import numpy as np
import pytest

from curryer.compute import footprint
from curryer.compute.footprint import (
    EARTH_MEAN_RADIUS_KM,
    FootprintGeometryError,
    convex_footprint,
    footprint_margin_m,
    footprint_polygon,
    swath_footprint,
)

TOL_KM = 0.25
PIXEL_KM = 0.46


def record_simplify_tolerances(monkeypatch):
    """Record the tolerance, in radians, of every simplification the module runs.

    Footprints from equal tolerances can keep different near-collinear vertices on different platforms,
    so the default tolerance is checked where it is used rather than by comparing polygons.
    """
    tolerances = []
    simplify = footprint._simplify

    def recording(points, tol):
        tolerances.append(tol)
        return simplify(points, tol)

    monkeypatch.setattr(footprint, "_simplify", recording)
    return tolerances


def unit_vectors(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def make_swath(
    lat0, lon0, azimuth_deg, n_frames=400, n_pixels=60, drift_deg_per_frame=0.0, along_km=PIXEL_KM, cross_km=PIXEL_KM
):
    """Swath along a great-circle ground track, pixels spaced ``along_km`` and ``cross_km``.

    ``drift_deg_per_frame`` rotates each frame about the polar axis, as Earth rotation does to an orbit
    ground track, which curves the swath.
    """
    start = unit_vectors(lat0, lon0)
    east = np.cross([0.0, 0.0, 1.0], start)
    east /= np.linalg.norm(east)
    north = np.cross(start, east)
    az = np.radians(azimuth_deg)
    heading = np.cos(az) * north + np.sin(az) * east
    cross_axis = np.cross(start, heading)

    s = np.arange(n_frames)[:, None, None] * along_km / EARTH_MEAN_RADIUS_KM
    t = (np.arange(n_pixels) - (n_pixels - 1) / 2)[None, :, None] * cross_km / EARTH_MEAN_RADIUS_KM
    center = np.cos(s) * start + np.sin(s) * heading
    xyz = np.cos(t) * center + np.sin(t) * cross_axis

    rot = np.radians(drift_deg_per_frame) * np.arange(n_frames)[:, None]
    x = xyz[..., 0] * np.cos(rot) - xyz[..., 1] * np.sin(rot)
    y = xyz[..., 0] * np.sin(rot) + xyz[..., 1] * np.cos(rot)
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def gnomonic(lat, lon, center):
    """Project onto the plane tangent at ``center``; great circles become straight lines."""
    east = np.cross([0.0, 0.0, 1.0], center)
    east /= np.linalg.norm(east)
    north = np.cross(center, east)
    xyz = unit_vectors(lat, lon)
    d = xyz @ center
    return np.stack([(xyz @ east) / d, (xyz @ north) / d], axis=-1)


def winding_inside(points, ring):
    """Winding-number test of ``points`` (N, 2) against the closed ``ring`` (M + 1, 2), independent of the module."""
    winding = np.zeros(len(points), dtype=int)
    px, py = points[:, 0], points[:, 1]
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        side = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
        winding += ((y1 <= py) & (y2 > py) & (side > 0)).astype(int)
        winding -= ((y1 > py) & (y2 <= py) & (side < 0)).astype(int)
    return winding != 0


def outside_count(vertices, lat, lon):
    """Number of valid pixels not inside the polygon, tested independently of the module."""
    valid = ~np.isnan(lat)
    center = unit_vectors(lat[valid], lon[valid]).mean(axis=0)
    center /= np.linalg.norm(center)
    ring = gnomonic(vertices[:, 1], vertices[:, 0], center)
    return int((~winding_inside(gnomonic(lat[valid], lon[valid], center), ring)).sum())


def signed_area(vertices, center):
    xy = gnomonic(vertices[:-1, 1], vertices[:-1, 0], center)
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def swath_center(lat, lon):
    c = unit_vectors(lat[~np.isnan(lat)], lon[~np.isnan(lon)]).mean(axis=0)
    return c / np.linalg.norm(c)


class TestSwathFootprint:
    def test_straight_swath(self):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert vertices.shape[1] == 2
        assert vertices.dtype == np.float64
        assert np.array_equal(vertices[0], vertices[-1])
        assert signed_area(vertices, swath_center(lat, lon)) > 0
        assert outside_count(vertices, lat, lon) == 0
        assert len(vertices) - 1 <= 12

        # Every vertex sits just outside the swath: within (half pixel + tolerance) of an outer pixel
        # along each grid direction, so at most sqrt(2) times that at a corner.
        perimeter = np.concatenate(
            [
                unit_vectors(lat[0], lon[0]),
                unit_vectors(lat[-1], lon[-1]),
                unit_vectors(lat[:, 0], lon[:, 0]),
                unit_vectors(lat[:, -1], lon[:, -1]),
            ]
        )
        v = unit_vectors(vertices[:-1, 1], vertices[:-1, 0])
        nearest_km = np.arccos(np.clip(v @ perimeter.T, -1, 1)).min(axis=1) * EARTH_MEAN_RADIUS_KM
        margin = PIXEL_KM / 2 + TOL_KM
        assert nearest_km.min() >= margin * 0.99
        assert np.sqrt(2) * margin * 0.97 <= nearest_km.max() <= np.sqrt(2) * margin * 1.01

    def test_clockwise_grid_returns_counter_clockwise(self):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        forward = swath_footprint(lat, lon, tolerance_km=TOL_KM)
        flipped = swath_footprint(lat[:, ::-1], lon[:, ::-1], tolerance_km=TOL_KM)

        assert signed_area(flipped, swath_center(lat, lon)) > 0
        assert outside_count(flipped, lat, lon) == 0
        assert len(flipped) == len(forward)

    def test_curved_swath_uses_more_vertices(self):
        lat, lon = make_swath(30.0, -100.0, 50.0, n_frames=2000)
        straight = swath_footprint(lat, lon, tolerance_km=TOL_KM)
        lat_c, lon_c = make_swath(30.0, -100.0, 50.0, n_frames=2000, drift_deg_per_frame=0.002)
        curved = swath_footprint(lat_c, lon_c, tolerance_km=TOL_KM)

        assert outside_count(curved, lat_c, lon_c) == 0
        assert len(curved) > len(straight)

    def test_antimeridian_crossing_is_one_polygon(self):
        lat, lon = make_swath(10.0, 179.5, 90.0)
        assert (lon > 179).any()
        assert (lon < -179).any()

        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert (vertices[:, 0] >= -180).all()

        assert (vertices[:, 0] < 180).all()
        assert (vertices[:, 0] > 170).any()
        assert (vertices[:, 0] < -170).any()
        assert signed_area(vertices, swath_center(lat, lon)) > 0
        assert outside_count(vertices, lat, lon) == 0

    def test_perimeter_fill_is_dropped(self):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        lat[100:120, 0] = np.nan
        lon[100:120, 0] = np.nan
        lat[0, -1] = lon[0, -1] = np.nan
        lat[-1, 30] = lon[-1, 30] = np.nan

        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert outside_count(vertices, lat, lon) == 0

    @pytest.mark.parametrize(("fill_index"), [(1, 0), (0, 1), (1, -1), (-2, -1), (-1, -2), (1, 30), (30, 1)])
    def test_fill_beside_outer_pixel_keeps_it(self, fill_index):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        lat[fill_index] = np.nan
        lon[fill_index] = np.nan

        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert outside_count(vertices, lat, lon) == 0

    def test_mostly_fill_first_row(self):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        lat[0, 1:-1] = np.nan
        lon[0, 1:-1] = np.nan

        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert outside_count(vertices, lat, lon) == 0

    def test_all_fill_rows_and_columns_are_cropped(self):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        expected = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        padded_lat = np.pad(lat, ((3, 2), (1, 4)), constant_values=np.nan)
        padded_lon = np.pad(lon, ((3, 2), (1, 4)), constant_values=np.nan)

        assert np.array_equal(swath_footprint(padded_lat, padded_lon, tolerance_km=TOL_KM), expected)

    def test_inputs_are_not_modified(self):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        lat[5, 0] = lon[5, 0] = np.nan
        lat_copy, lon_copy = lat.copy(), lon.copy()

        swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert np.array_equal(lat, lat_copy, equal_nan=True)
        assert np.array_equal(lon, lon_copy, equal_nan=True)

    @pytest.mark.parametrize(
        ("latitude", "longitude", "match"),
        [
            (np.zeros(10), np.zeros(10), "2-D numpy array"),
            ([[0.0, 1.0], [0.0, 1.0]], np.zeros((2, 2)), "2-D numpy array"),
            (np.zeros((4, 4), dtype=int), np.zeros((4, 4)), "floating dtype"),
            (np.zeros((4, 4)), np.zeros((4, 5)), "does not match"),
            (np.full((4, 4), 91.0), np.zeros((4, 4)), r"within \[-90, 90\]"),
            (np.zeros((4, 4)), np.full((4, 4), 181.0), r"within \[-180, 180\]"),
            (np.full((4, 4), np.inf), np.zeros((4, 4)), "infinite"),
            (np.full((4, 4), np.nan), np.full((4, 4), np.nan), "no valid pixels"),
            (np.where(np.eye(4) > 0, np.nan, 0.0), np.zeros((4, 4)), "NaN fill at the same pixels"),
        ],
    )
    def test_invalid_arrays_raise(self, latitude, longitude, match):
        with pytest.raises(ValueError, match=match) as err:
            swath_footprint(latitude, longitude, tolerance_km=TOL_KM)
        assert not isinstance(err.value, FootprintGeometryError)

    def test_single_valid_row_raises(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=5)
        lat[1:] = np.nan
        lon[1:] = np.nan
        with pytest.raises(FootprintGeometryError, match="at least 2 rows"):
            swath_footprint(lat, lon, tolerance_km=TOL_KM)

    @pytest.mark.parametrize(("tolerance_km"), [0.0, -0.25, np.nan, np.inf, True, "0.25"])
    def test_invalid_tolerance_raises(self, tolerance_km):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=10)
        with pytest.raises(ValueError, match="tolerance_km"):
            swath_footprint(lat, lon, tolerance_km=tolerance_km)

    def test_swath_too_large_raises(self):
        # About 150 degrees of ground track.
        lat, lon = make_swath(0.0, 0.0, 90.0, n_frames=36500, n_pixels=4)
        with pytest.raises(ValueError, match="more than 60.0 degrees") as err:
            swath_footprint(lat, lon, tolerance_km=TOL_KM)
        assert not isinstance(err.value, FootprintGeometryError)

    def test_polygon_around_pole_raises(self):
        # Grid centered on the north pole, 60 x 60 km.
        x = np.linspace(-30, 30, 50) / EARTH_MEAN_RADIUS_KM
        gx, gy = np.meshgrid(x, x, indexing="ij")
        z = 1.0 / np.sqrt(1.0 + gx**2 + gy**2)
        lat = np.degrees(np.arcsin(z))
        lon = np.degrees(np.arctan2(gy, gx))
        with pytest.raises(ValueError, match="contains a pole") as err:
            swath_footprint(lat, lon, tolerance_km=TOL_KM)
        assert not isinstance(err.value, FootprintGeometryError)

    def test_folded_grid_raises(self):
        # The last frames double back across the swath, so the traced edge crosses itself.
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=200)
        lat[-50:] = lat[-50:, ::-1]
        lon[-50:] = lon[-50:, ::-1]
        with pytest.raises(FootprintGeometryError, match="crosses itself"):
            swath_footprint(lat, lon, tolerance_km=TOL_KM)

    def test_polygon_missing_pixels_raises(self):
        # A triangle well inside the swath cannot contain all of its pixels.
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=50)
        xyz = unit_vectors(lat, lon).reshape(-1, 3)
        triangle = unit_vectors(lat[20:23, 30], lon[20:23, 30] + np.array([0.0, 0.001, -0.001]))
        with pytest.raises(FootprintGeometryError, match="valid pixel centers fall outside"):
            footprint._verify(triangle, xyz, xyz[:4])

    def test_too_few_vertices_raises(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=50)
        xyz = unit_vectors(lat, lon).reshape(-1, 3)
        with pytest.raises(FootprintGeometryError, match="fewer than 3 distinct vertices"):
            footprint._verify(xyz[[0, 1]], xyz, xyz[:4])

    def test_edge_ring_outside_polygon_raises(self):
        # Polygon through the pixels one step beyond the tested pixels contains every center, but not
        # edge ring points two steps out.
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=50)
        xyz = unit_vectors(lat, lon)
        corners = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
        vertices = np.array([xyz[1:-1, 1:-1][c] for c in corners])
        edge_ring = np.array([xyz[c] for c in corners])
        with pytest.raises(FootprintGeometryError, match="half-pixel edge ring points fall outside"):
            footprint._verify(vertices, xyz[2:-2, 2:-2].reshape(-1, 3), edge_ring)

    def test_repeated_first_frame_borrows_length(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=20)
        lat[0], lon[0] = lat[1], lon[1]

        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert outside_count(vertices, lat, lon) == 0

    def test_fill_frame_next_to_end_uses_median_spacing(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, along_km=0.3, cross_km=0.2)
        lat[1] = np.nan
        lon[1] = np.nan

        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert outside_count(vertices, lat, lon) == 0
        # The first-row corners are pushed along track by half the median along-track spacing plus the
        # tolerance. Row 1 is fill, so the along-track direction comes from row 2.
        v = unit_vectors(vertices[:-1, 1], vertices[:-1, 0])
        xyz = unit_vectors(lat, lon)
        for c, a in [((0, 0), (2, 0)), ((0, -1), (2, -1))]:
            p = xyz[c]
            offset = v[np.argmin(np.linalg.norm(v - p, axis=1))] - p
            out_along = (p - xyz[a]) / np.linalg.norm(p - xyz[a])
            assert np.isclose(offset @ out_along * EARTH_MEAN_RADIUS_KM, 0.3 / 2 + TOL_KM, rtol=0.02)

    def test_side_with_one_valid_pixel_raises(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=20)
        lat[0, 1:] = np.nan
        lon[0, 1:] = np.nan
        with pytest.raises(FootprintGeometryError, match="first row of the grid has fewer than 2 valid pixels"):
            swath_footprint(lat, lon, tolerance_km=TOL_KM)

    def test_no_adjacent_valid_pixels_raises(self):
        lat = np.array([[40.0, np.nan], [np.nan, 40.1]])
        lon = np.array([[-100.0, np.nan], [np.nan, -100.1]])
        with pytest.raises(FootprintGeometryError, match="no two adjacent valid pixels along grid axis 0"):
            swath_footprint(lat, lon, tolerance_km=TOL_KM)

    def test_side_with_all_pixels_at_one_position_raises(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=20, n_pixels=2)
        lat[0, 1], lon[0, 1] = lat[0, 0], lon[0, 0]
        match = "all valid pixels on the first row of the grid are at the same position"
        with pytest.raises(FootprintGeometryError, match=match):
            swath_footprint(lat, lon, tolerance_km=TOL_KM)

    def test_curved_swath_with_dead_inner_column_covers_true_edge(self):
        lat, lon = make_swath(-50.0, -100.0, 120.0, n_frames=4000, drift_deg_per_frame=0.03)
        # True half-pixel edge points, from the complete grid.
        xyz = unit_vectors(lat, lon)
        edge = np.concatenate([xyz[:, 0] + 0.5 * (xyz[:, 0] - xyz[:, 1]), xyz[:, -1] + 0.5 * (xyz[:, -1] - xyz[:, -2])])
        edge /= np.linalg.norm(edge, axis=1, keepdims=True)
        edge_lat = np.degrees(np.arcsin(edge[:, 2]))
        edge_lon = np.degrees(np.arctan2(edge[:, 1], edge[:, 0]))

        lat[1:, 1] = np.nan
        lon[1:, 1] = np.nan
        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)

        assert outside_count(vertices, lat, lon) == 0
        assert outside_count(vertices, edge_lat[:, None], edge_lon[:, None]) == 0

    def test_every_corner_is_pushed_along_both_sides(self):
        along, cross = 0.3, 0.2
        lat, lon = make_swath(40.0, -100.0, 60.0, along_km=along, cross_km=cross)
        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)
        v = unit_vectors(vertices[:-1, 1], vertices[:-1, 0])
        xyz = unit_vectors(lat, lon)
        # (corner, its along-track neighbor, its cross-track neighbor)
        for c, a, b in [
            ((0, 0), (1, 0), (0, 1)),
            ((0, -1), (1, -1), (0, -2)),
            ((-1, -1), (-2, -1), (-1, -2)),
            ((-1, 0), (-2, 0), (-1, 1)),
        ]:
            p = xyz[c]
            offset = v[np.argmin(np.linalg.norm(v - p, axis=1))] - p
            out_along = (p - xyz[a]) / np.linalg.norm(p - xyz[a])
            out_cross = (p - xyz[b]) / np.linalg.norm(p - xyz[b])
            assert np.isclose(offset @ out_along * EARTH_MEAN_RADIUS_KM, along / 2 + TOL_KM, rtol=0.02)
            assert np.isclose(offset @ out_cross * EARTH_MEAN_RADIUS_KM, cross / 2 + TOL_KM, rtol=0.02)

    def test_default_tolerance_is_half_the_smaller_pixel_spacing(self, monkeypatch):
        along, cross = 0.3, 0.2
        lat, lon = make_swath(40.0, -100.0, 60.0, along_km=along, cross_km=cross)
        tolerances = record_simplify_tolerances(monkeypatch)

        vertices = swath_footprint(lat, lon)

        assert outside_count(vertices, lat, lon) == 0
        assert tolerances
        np.testing.assert_allclose(np.array(tolerances) * EARTH_MEAN_RADIUS_KM, cross / 2, rtol=1e-3)

    def test_default_tolerance_with_zero_pixel_spacing_raises(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=10, n_pixels=10)
        lat, lon = np.repeat(lat, 3, axis=0), np.repeat(lon, 3, axis=0)
        with pytest.raises(ValueError, match="median pixel spacing along a grid axis is zero") as err:
            swath_footprint(lat, lon)
        assert not isinstance(err.value, FootprintGeometryError)
        assert len(swath_footprint(lat, lon, tolerance_km=TOL_KM)) >= 4

    def test_larger_tolerance_keeps_fewer_vertices(self):
        lat, lon = make_swath(30.0, -100.0, 50.0, n_frames=2000, drift_deg_per_frame=0.002)
        assert len(swath_footprint(lat, lon, tolerance_km=1.0)) < len(swath_footprint(lat, lon, tolerance_km=0.05))

    def test_collinear_grid_raises(self):
        lat = np.zeros((3, 3))
        lon = 0.01 * np.arange(3)[:, None] + 0.03 * np.arange(3)[None, :]
        with pytest.raises(FootprintGeometryError, match="enclose no area"):
            swath_footprint(lat, lon, tolerance_km=TOL_KM)

    def test_straight_corner_raises(self):
        # The first row leaves corner (0, 0) in the same direction the first column arrives, so the
        # two sides meet without an angle.
        lat = np.array([[40.00, 40.02, 40.04], [40.01, 40.03, 40.05], [40.02, 40.04, 40.06]])
        lon = np.array([[-100.00, -99.99, -99.98], [-100.02, -100.01, -100.00], [-100.04, -100.03, -100.02]])
        lat[0, 1] = 2 * lat[0, 0] - lat[1, 0]
        lon[0, 1] = 2 * lon[0, 0] - lon[1, 0]
        with pytest.raises(FootprintGeometryError, match="meet at a corner without an angle"):
            swath_footprint(lat, lon, tolerance_km=TOL_KM)


def x_normals(n):
    return np.tile([1.0, 0.0, 0.0], (n, 1))


class TestHalfLengths:
    def test_reach_is_measured_along_the_normal(self):
        # Step of length 0.5 at 60 degrees to the normal reaches 0.25 along it.
        outer = np.zeros((1, 3))
        inner = -0.5 * np.array([[np.cos(np.radians(60)), np.sin(np.radians(60)), 0.0]])
        half = footprint._half_lengths(outer, inner, x_normals(1), np.array([True]), fallback=99.0)
        assert np.isclose(half[0], 0.125)

    def test_borrows_from_nearest_complete_pair(self):
        # Chord length at position k is (k + 1) / 10: outer at the origin offset, inner displaced along x.
        n = 8
        outer = np.zeros((n, 3))
        inner = np.zeros((n, 3))
        inner[:, 0] = -(np.arange(n) + 1) / 10
        pair = np.array([False, False, True, False, False, True, False, False])
        pair_valid = pair.copy()

        half = footprint._half_lengths(outer, inner, x_normals(n), pair_valid, fallback=99.0)

        # Before the first pair -> 2; inside the gap -> nearest (3 -> 2, 4 -> 5); tie at none here;
        # after the last pair -> 5.
        expected_source = np.array([2, 2, 2, 2, 5, 5, 5, 5])
        assert np.allclose(half, 0.5 * (expected_source + 1) / 10)

    def test_tie_takes_earlier_pair(self):
        n = 5
        outer = np.zeros((n, 3))
        inner = np.zeros((n, 3))
        inner[:, 0] = -(np.arange(n) + 1) / 10
        pair_valid = np.array([True, False, False, False, True])
        half = footprint._half_lengths(outer, inner, x_normals(n), pair_valid, fallback=99.0)
        assert np.isclose(half[2], 0.5 * 0.1)

    def test_zero_length_pair_is_skipped(self):
        outer = np.zeros((3, 3))
        inner = np.zeros((3, 3))
        inner[2, 0] = -0.4
        half = footprint._half_lengths(outer, inner, x_normals(3), np.array([True, True, True]), fallback=99.0)
        assert np.allclose(half, 0.2)

    def test_fallback_when_no_pair(self):
        half = footprint._half_lengths(
            np.zeros((3, 3)), np.ones((3, 3)), x_normals(3), np.zeros(3, dtype=bool), fallback=0.6
        )
        assert np.allclose(half, 0.3)


class TestSelfIntersects:
    def test_collinear_disjoint_edges_are_not_a_crossing(self):
        # U shape: the two bottom edges lie on one line but do not touch.
        ring = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [2, 0], [3, 0], [3, 2], [0, 2]], dtype=float)
        assert not footprint._self_intersects(ring)

    def test_collinear_overlapping_edges_are_a_crossing(self):
        # Degenerate ring whose first and third edges overlap along one line.
        ring = np.array([[0, 0], [2, 0], [3, 0], [1, 0]], dtype=float)
        assert footprint._self_intersects(ring)

    def test_crossing_edges(self):
        ring = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
        assert footprint._self_intersects(ring)


class TestSimplify:
    def test_distance_is_to_the_arc_not_the_great_circle(self):
        points = unit_vectors(np.zeros(3), np.array([0.0, 1.0, 3.0]))
        d = footprint._arc_distance(points, points[0], points[1])
        assert np.allclose(d, [0.0, 0.0, np.radians(2.0)])

    def test_keeps_a_point_beyond_the_segment_on_its_great_circle(self):
        # The side doubles back: the middle point lies on the great circle of the endpoints but past the end.
        points = unit_vectors(np.zeros(3), np.array([0.0, 2.0, 1.0]))
        kept = footprint._simplify(points, TOL_KM / EARTH_MEAN_RADIUS_KM)
        assert len(kept) == 3


def min_edge_distance_km(vertices, lat, lon):
    """Smallest distance from any valid pixel center to the polygon's edges, in km."""
    ring = unit_vectors(vertices[:, 1], vertices[:, 0])
    pixels = unit_vectors(lat, lon)[~np.isnan(lat)]
    distances = [footprint._arc_distance(pixels, a, b).min() for a, b in zip(ring[:-1], ring[1:])]
    return min(distances) * EARTH_MEAN_RADIUS_KM


class TestConvexFootprint:
    def check(self, lat, lon):
        vertices = convex_footprint(lat, lon, tolerance_km=TOL_KM)
        assert np.array_equal(vertices[0], vertices[-1])
        assert signed_area(vertices, swath_center(lat, lon)) > 0
        assert outside_count(vertices, lat, lon) == 0
        return vertices

    def test_folded_grid(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=200)
        lat[-50:] = lat[-50:, ::-1]
        lon[-50:] = lon[-50:, ::-1]
        with pytest.raises(FootprintGeometryError):
            swath_footprint(lat, lon, tolerance_km=TOL_KM)
        vertices = self.check(lat, lon)
        assert min_edge_distance_km(vertices, lat, lon) >= 0.5 * np.hypot(PIXEL_KM, PIXEL_KM)

    def test_margin_is_at_least_half_the_pixel_diagonal(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=200, along_km=0.3, cross_km=0.5)
        vertices = self.check(lat, lon)
        assert min_edge_distance_km(vertices, lat, lon) >= 0.5 * np.hypot(0.3, 0.5)

    def test_antimeridian_crossing(self):
        lat, lon = make_swath(10.0, 179.5, 90.0, n_frames=400)
        vertices = self.check(lat, lon)
        assert (vertices[:, 0] >= -180).all()
        assert (vertices[:, 0] < 180).all()
        assert (vertices[:, 0] > 179).any()
        assert (vertices[:, 0] < -178).any()

    def test_curved_swath(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=2000, drift_deg_per_frame=0.002)
        vertices = self.check(lat, lon)
        assert min_edge_distance_km(vertices, lat, lon) >= 0.5 * np.hypot(PIXEL_KM, PIXEL_KM)

    @pytest.mark.parametrize(("n_frames", "n_pixels", "axis"), [(1, 60, 0), (2, 1, 1)])
    def test_pixel_size_unknown_along_an_axis_raises(self, n_frames, n_pixels, axis):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=n_frames, n_pixels=n_pixels)
        with pytest.raises(ValueError, match=f"no two adjacent valid pixels along grid axis {axis}") as err:
            convex_footprint(lat, lon, tolerance_km=TOL_KM)
        assert not isinstance(err.value, FootprintGeometryError)

    def test_invalid_geolocation_raises(self):
        with pytest.raises(ValueError, match="no valid pixels") as err:
            convex_footprint(np.full((4, 4), np.nan), np.full((4, 4), np.nan), tolerance_km=TOL_KM)
        assert not isinstance(err.value, FootprintGeometryError)

    def test_no_adjacent_valid_pixels_raises(self):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=4, n_pixels=4)
        lat[[0, 0, 1, 1, 2, 2, 3, 3], [1, 3, 0, 2, 1, 3, 0, 2]] = np.nan
        lon[np.isnan(lat)] = np.nan
        with pytest.raises(ValueError, match="no two adjacent valid pixels along grid axis 0") as err:
            convex_footprint(lat, lon, tolerance_km=TOL_KM)
        assert not isinstance(err.value, FootprintGeometryError)

    def test_swath_too_large_raises(self):
        lat, lon = make_swath(0.0, 0.0, 90.0, n_frames=36500, n_pixels=4)
        with pytest.raises(ValueError, match="more than 60.0 degrees") as err:
            convex_footprint(lat, lon, tolerance_km=TOL_KM)
        assert not isinstance(err.value, FootprintGeometryError)

    def test_default_tolerance(self, monkeypatch):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=200, along_km=0.3, cross_km=0.5)
        tolerances = record_simplify_tolerances(monkeypatch)

        vertices = convex_footprint(lat, lon)

        assert outside_count(vertices, lat, lon) == 0
        assert min_edge_distance_km(vertices, lat, lon) >= 0.5 * np.hypot(0.3, 0.5)
        assert tolerances
        np.testing.assert_allclose(np.array(tolerances) * EARTH_MEAN_RADIUS_KM, 0.15, rtol=1e-3)

    def test_default_tolerance_with_zero_pixel_spacing_raises(self):
        lat = np.full((3, 3), 40.0)
        lon = np.full((3, 3), -100.0)
        with pytest.raises(ValueError, match="median pixel spacing along a grid axis is zero"):
            convex_footprint(lat, lon)


class TestFootprintPolygon:
    def test_traceable_swath_returns_the_traced_edge(self, caplog):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        with caplog.at_level(logging.WARNING, logger="curryer.compute.footprint"):
            vertices = footprint_polygon(lat, lon, tolerance_km=TOL_KM)
        assert np.array_equal(vertices, swath_footprint(lat, lon, tolerance_km=TOL_KM))
        assert not caplog.records

    def test_folded_swath_falls_back_to_the_convex_hull(self, caplog):
        lat, lon = make_swath(40.0, -100.0, 60.0, n_frames=200)
        lat[-50:] = lat[-50:, ::-1]
        lon[-50:] = lon[-50:, ::-1]
        with caplog.at_level(logging.WARNING, logger="curryer.compute.footprint"):
            vertices = footprint_polygon(lat, lon, tolerance_km=TOL_KM)
        assert np.array_equal(vertices, convex_footprint(lat, lon, tolerance_km=TOL_KM))
        assert "convex hull" in caplog.text

    def test_single_valid_pixel_raises_value_error(self):
        lat = np.array([[40.0, np.nan], [np.nan, np.nan]])
        lon = np.array([[-100.0, np.nan], [np.nan, np.nan]])
        with pytest.raises(ValueError, match="no two adjacent valid pixels along grid axis 0") as err:
            footprint_polygon(lat, lon)
        assert not isinstance(err.value, FootprintGeometryError)

    def test_invalid_geolocation_raises_without_fallback(self):
        with pytest.raises(ValueError, match="no valid pixels") as err:
            footprint_polygon(np.full((4, 4), np.nan), np.full((4, 4), np.nan))
        assert not isinstance(err.value, FootprintGeometryError)


def beyond(lat, lon, inner, outer, distance_km):
    """Point ``distance_km`` past pixel ``outer``, continuing the great circle from pixel ``inner``."""
    p0, p1 = unit_vectors(lat[outer], lon[outer]), unit_vectors(lat[inner], lon[inner])
    direction = p0 - p1 - ((p0 - p1) @ p0) * p0
    direction /= np.linalg.norm(direction)
    d = distance_km / EARTH_MEAN_RADIUS_KM
    q = np.cos(d) * p0 + np.sin(d) * direction
    return np.degrees(np.arcsin(q[2])), np.degrees(np.arctan2(q[1], q[0]))


class TestFootprintMargin:
    # Square of side 20 km centred on (0, 0): edges at +-10 km east and north.
    HALF_DEG = np.degrees(10.0 / EARTH_MEAN_RADIUS_KM)
    SQUARE = np.array(
        [
            [-HALF_DEG, -HALF_DEG],
            [HALF_DEG, -HALF_DEG],
            [HALF_DEG, HALF_DEG],
            [-HALF_DEG, HALF_DEG],
            [-HALF_DEG, -HALF_DEG],
        ]
    )

    def test_inside_is_positive_outside_is_negative(self):
        # East (7, 12 km), north (13 km) and west (-7, -12 km: the closing edge is the nearest) of centre.
        km = np.degrees(1.0 / EARTH_MEAN_RADIUS_KM)
        lat = np.array([0.0, 0.0, 0.0, 13.0, 0.0, 0.0]) * km
        lon = np.array([0.0, 7.0, 12.0, 0.0, -7.0, -12.0]) * km
        margin = footprint_margin_m(self.SQUARE, lat, lon)
        np.testing.assert_allclose(margin, [10e3, 3e3, -2e3, -3e3, 3e3, -2e3], rtol=1e-3)

    def test_keeps_the_shape_of_the_points(self):
        lat = np.zeros((2, 3))
        assert footprint_margin_m(self.SQUARE, lat, lat.copy()).shape == (2, 3)

    def test_antipode_is_outside(self):
        # The antipode projects onto the square's centre without the hemisphere check. Its distance to
        # the nearest edge is half a great circle less the centre-to-corner distance, about 14.14 km.
        margin = footprint_margin_m(self.SQUARE, np.array([0.0]), np.array([180.0]))
        corner_m = np.arccos(np.cos(10.0 / EARTH_MEAN_RADIUS_KM) ** 2) * EARTH_MEAN_RADIUS_KM * 1e3
        np.testing.assert_allclose(margin[0], -(np.pi * EARTH_MEAN_RADIUS_KM * 1e3 - corner_m), rtol=1e-9)

    def test_swath_footprint_margin_of_points_beside_the_swath(self):
        lat, lon = make_swath(40.0, -100.0, 60.0)
        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)
        # 5 km beyond the first column is outside, by 5 km less the polygon's offset from the pixel
        # (half a pixel, plus 0 to 2 tolerances); the middle pixel of the middle frame is inside by
        # about half the swath width plus that offset.
        out_lat, out_lon = beyond(lat, lon, (200, 1), (200, 0), 5.0)
        margin = footprint_margin_m(vertices, np.array([out_lat, lat[200, 30]]), np.array([out_lon, lon[200, 30]]))
        assert -5e3 < margin[0] < -5e3 + (PIXEL_KM / 2 + 2 * TOL_KM) * 1e3 + 50
        half_swath_m = 30 * PIXEL_KM * 1e3
        assert half_swath_m < margin[1] < half_swath_m + (PIXEL_KM / 2 + 2 * TOL_KM) * 1e3 + 50

    def test_antimeridian_footprint(self):
        lat, lon = make_swath(10.0, 179.5, 90.0)
        vertices = swath_footprint(lat, lon, tolerance_km=TOL_KM)
        margin = footprint_margin_m(vertices, lat[::50, ::10].copy(), lon[::50, ::10].copy())
        assert (margin > 0).all()

    @pytest.mark.parametrize(
        ("polygon", "match"),
        [
            (np.zeros((4, 3)), r"shape \(N \+ 1, 2\)"),
            (np.zeros((4, 2), dtype=int), "floating dtype"),
            (np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0], [0.0, 0.0]]), "finite"),
            (np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 91.0], [0.0, 0.0]]), "within"),
            (np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), "closed"),
            (np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]), "fewer than 3 distinct"),
            (np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]), "crosses itself"),
            (np.array([[-80.0, 0.0], [0.0, 80.0], [80.0, 0.0], [0.0, -80.0], [-80.0, 0.0]]), "more than 60.0 degrees"),
            (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]), "counter-clockwise"),
        ],
    )
    def test_invalid_footprint_raises(self, polygon, match):
        with pytest.raises(ValueError, match=match):
            footprint_margin_m(polygon, np.zeros(1), np.zeros(1))

    @pytest.mark.parametrize(
        ("lat", "lon", "match"),
        [
            ([0.0], np.zeros(1), "floating dtype"),
            (np.zeros(1, dtype=int), np.zeros(1), "floating dtype"),
            (np.array([np.nan]), np.zeros(1), "finite"),
            (np.zeros(2), np.zeros(3), "does not match"),
            (np.array([91.0]), np.zeros(1), "within"),
            (np.zeros(1), np.array([-181.0]), "within"),
        ],
    )
    def test_invalid_points_raise(self, lat, lon, match):
        with pytest.raises(ValueError, match=match):
            footprint_margin_m(self.SQUARE, lat, lon)
