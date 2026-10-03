"""Tests for ``curryer.correction.psf.resolve_spacecraft_ecef`` (viewing geometry)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from curryer.compute.spatial import geodetic_to_ecef
from curryer.correction.error_stats import ErrorStatsConfig, ErrorStatsProcessor
from curryer.correction.grid_types import ImageGrid
from curryer.correction.psf import resolve_spacecraft_ecef

CENTER_LON, CENTER_LAT = -102.33, 26.15
ALTITUDE_M = 410_000.0


def _grid(lon: float = CENTER_LON, lat: float = CENTER_LAT, h: np.ndarray | None = None) -> ImageGrid:
    lats = lat + np.array([[0.01, 0.01, 0.01], [0.0, 0.0, 0.0], [-0.01, -0.01, -0.01]])
    lons = lon + np.array([[-0.01, 0.0, 0.01], [-0.01, 0.0, 0.01], [-0.01, 0.0, 0.01]])
    return ImageGrid(data=np.ones((3, 3)), lat=lats, lon=lons, h=h)


def _local_up_east(lon: float, lat: float) -> tuple[np.ndarray, np.ndarray]:
    lon_r, lat_r = np.deg2rad(lon), np.deg2rad(lat)
    up = np.array([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    east = np.array([-np.sin(lon_r), np.cos(lon_r), 0.0])
    return up, east


def _center_ecef(h_m: float = 0.0) -> np.ndarray:
    return geodetic_to_ecef(np.array([CENTER_LON, CENTER_LAT, h_m]), meters=True, degrees=True)


def _spacecraft_at_view_zenith(view_zenith_deg: float, slant_range_m: float) -> np.ndarray:
    up, east = _local_up_east(CENTER_LON, CENTER_LAT)
    zeta = np.deg2rad(view_zenith_deg)
    return _center_ecef() + slant_range_m * (np.cos(zeta) * up + np.sin(zeta) * east)


def test_nadir_spacecraft_boresight_is_local_down():
    r = geodetic_to_ecef(np.array([CENTER_LON, CENTER_LAT, ALTITUDE_M]), meters=True, degrees=True)
    r_out, boresight, t_matrix = resolve_spacecraft_ecef(_grid(), r, CENTER_LAT, CENTER_LON)
    up, _ = _local_up_east(CENTER_LON, CENTER_LAT)
    npt_angle = np.rad2deg(np.arccos(np.clip(np.dot(boresight, -up), -1.0, 1.0)))
    assert npt_angle < 1e-6
    np.testing.assert_array_equal(r_out, r)
    np.testing.assert_array_equal(t_matrix, np.eye(3))


def test_off_nadir_boresight_is_line_of_sight_to_grid_center():
    r = _spacecraft_at_view_zenith(60.0, 820_000.0)
    _, boresight, _ = resolve_spacecraft_ecef(_grid(), r, CENTER_LAT, CENTER_LON)
    expected = (_center_ecef() - r) / np.linalg.norm(_center_ecef() - r)
    np.testing.assert_allclose(boresight, expected, atol=1e-12)
    assert np.linalg.norm(boresight) == pytest.approx(1.0)


def test_boresight_uses_grid_height():
    r = _spacecraft_at_view_zenith(60.0, 820_000.0)
    h = np.full((3, 3), 2_000.0)
    _, boresight, _ = resolve_spacecraft_ecef(_grid(h=h), r, CENTER_LAT, CENTER_LON)
    expected = (_center_ecef(2_000.0) - r) / np.linalg.norm(_center_ecef(2_000.0) - r)
    np.testing.assert_allclose(boresight, expected, atol=1e-12)


def test_off_nadir_angle_reaches_nadir_equivalent_scaling():
    """A 60° view-zenith observation reports its true off-nadir angle, not 0."""
    view_zenith_deg = 60.0
    r = _spacecraft_at_view_zenith(view_zenith_deg, 820_000.0)
    r_out, boresight, t_matrix = resolve_spacecraft_ecef(_grid(), r, CENTER_LAT, CENTER_LON)

    data = xr.Dataset(
        {
            "lat_error_deg": (["measurement"], [0.001]),
            "lon_error_deg": (["measurement"], [0.001]),
            "gcp_lat_deg": (["measurement"], [CENTER_LAT]),
            "gcp_lon_deg": (["measurement"], [CENTER_LON]),
            "gcp_alt": (["measurement"], [0.0]),
            "sc_position": (["measurement", "xyz"], [r_out]),
            "boresight": (["measurement", "xyz"], [boresight]),
            "t_inst2ref": (["measurement", "xyz_from", "xyz_to"], t_matrix[np.newaxis]),
        },
        coords={"measurement": [0]},
    )
    processor = ErrorStatsProcessor(
        ErrorStatsConfig(
            variable_names={
                "spacecraft_position": "sc_position",
                "boresight": "boresight",
                "transformation_matrix": "t_inst2ref",
            }
        )
    )
    out = processor.compute_nadir_equivalent_errors(data)

    # Spherical-Earth law of sines: sin(off_nadir) = |p| sin(view_zenith) / |r|.
    expected_off_nadir_deg = np.rad2deg(
        np.arcsin(np.linalg.norm(_center_ecef()) * np.sin(np.deg2rad(view_zenith_deg)) / np.linalg.norm(r))
    )
    assert float(out["off_nadir_angle_deg"][0]) == pytest.approx(expected_off_nadir_deg, abs=0.01)
    assert float(out["vp_scaling_factor"][0]) < 0.5


@pytest.mark.parametrize(
    ("position", "match"),
    [
        (None, "Spacecraft ECEF position is required"),
        (np.zeros((3, 1)), r"shape \(3,\)"),
        (np.array([np.nan, 0.0, 7.0e6]), "finite"),
        (np.array([-1.0e6, -5.0e6, 2.9e6]) * 1e-3, "inside the Earth"),
        (
            geodetic_to_ecef(np.array([CENTER_LON, CENTER_LAT, -1_000.0]), meters=True, degrees=True),
            "relative to the WGS-84",
        ),
    ],
    ids=["missing", "wrong-shape", "non-finite", "kilometers-not-meters", "below-ellipsoid"],
)
def test_invalid_spacecraft_position_raises(position, match):
    with pytest.raises(ValueError, match=match):
        resolve_spacecraft_ecef(_grid(), position, CENTER_LAT, CENTER_LON)


def test_boresight_targets_gcp_column_in_mid_frame_row():
    """The line of sight is to the mid-frame pixel in the GCP's cross-track column."""
    r = _spacecraft_at_view_zenith(60.0, 820_000.0)
    h = np.arange(9, dtype=float).reshape(3, 3) * 100.0
    grid = _grid(h=h)
    # Nearest pixel (0, 2): column 2, imaged one frame before the mid-frame row 1.
    _, boresight, _ = resolve_spacecraft_ecef(grid, r, CENTER_LAT + 0.009, CENTER_LON + 0.008)
    p = geodetic_to_ecef(np.array([grid.lon[1, 2], grid.lat[1, 2], h[1, 2]]), meters=True, degrees=True)
    np.testing.assert_allclose(boresight, (p - r) / np.linalg.norm(p - r), atol=1e-12)
    _, center_boresight, _ = resolve_spacecraft_ecef(grid, r, CENTER_LAT, CENTER_LON)
    assert np.rad2deg(np.arccos(np.clip(np.dot(boresight, center_boresight), -1.0, 1.0))) > 0.01


@pytest.mark.parametrize(
    ("target", "match"),
    [
        ((CENTER_LAT + 0.05, CENTER_LON), "outside the observation grid"),
        ((CENTER_LAT, CENTER_LON - 0.03), "outside the observation grid"),
        ((np.nan, CENTER_LON), "Target lat/lon must be finite"),
    ],
    ids=["north-of-grid", "west-of-grid", "non-finite"],
)
def test_invalid_target_raises(target, match):
    r = _spacecraft_at_view_zenith(30.0, 500_000.0)
    with pytest.raises(ValueError, match=match):
        resolve_spacecraft_ecef(_grid(), r, *target)


def test_non_finite_mid_frame_pixel_raises():
    h = np.zeros((3, 3))
    h[1, 2] = np.nan
    r = _spacecraft_at_view_zenith(30.0, 500_000.0)
    with pytest.raises(ValueError, match=r"Mid-frame pixel \(1, 2\)"):
        resolve_spacecraft_ecef(_grid(h=h), r, CENTER_LAT + 0.01, CENTER_LON + 0.01)
