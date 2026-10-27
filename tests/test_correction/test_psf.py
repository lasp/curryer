"""Unit tests for ``curryer.correction.psf``: viewing and ground-track geometry, the PSF crop and the motion blur."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from curryer.compute.spatial import geodetic_to_ecef
from curryer.correction.config import PSFSamplingConfig
from curryer.correction.error_stats import ErrorStatsConfig, ErrorStatsProcessor
from curryer.correction.grid_types import ImageGrid, ProjectedPSF, PSFGrid
from curryer.correction.psf import (
    convolve_psf_with_spacecraft_motion,
    ground_track_azimuth_deg,
    resample_psf_to_gcp_resolution,
    resolve_spacecraft_ecef,
)

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


def _track_grid(azimuth_deg: float, rows: int = 7, cols: int = 5, step_deg: float = 0.004) -> ImageGrid:
    """Grid whose rows advance along *azimuth_deg* and columns 90 degrees clockwise from it."""
    az = np.deg2rad(azimuth_deg)
    r = np.arange(rows)[:, None] - rows // 2
    c = np.arange(cols)[None, :] - cols // 2
    lat = CENTER_LAT + step_deg * (r * np.cos(az) - c * np.sin(az))
    lon = CENTER_LON + step_deg * (r * np.sin(az) + c * np.cos(az)) / np.cos(np.deg2rad(CENTER_LAT))
    return ImageGrid(data=np.ones((rows, cols)), lat=lat, lon=lon)


@pytest.mark.parametrize("azimuth_deg", [12.0, 168.0, 191.0, 348.0])
def test_ground_track_azimuth_ascending_and_descending(azimuth_deg):
    grid = _track_grid(azimuth_deg)
    assert ground_track_azimuth_deg(grid, CENTER_LAT, CENTER_LON) == pytest.approx(azimuth_deg, abs=0.2)


@pytest.mark.parametrize("row", [0, -1])
def test_ground_track_azimuth_at_first_and_last_row(row):
    grid = _track_grid(191.0)
    lat, lon = float(grid.lat[row, 2]), float(grid.lon[row, 2])
    assert ground_track_azimuth_deg(grid, lat, lon) == pytest.approx(191.0, abs=0.2)


def test_ground_track_azimuth_target_outside_grid_raises():
    with pytest.raises(ValueError, match="outside the observation grid"):
        ground_track_azimuth_deg(_track_grid(191.0), CENTER_LAT + 1.0, CENTER_LON)


def test_ground_track_azimuth_single_row_raises():
    grid = _track_grid(191.0, rows=1)
    with pytest.raises(ValueError, match="at least two"):
        ground_track_azimuth_deg(grid, CENTER_LAT, CENTER_LON)


def test_ground_track_azimuth_non_finite_neighbour_raises():
    grid = _track_grid(191.0)
    grid.lat[4, 2] = np.nan
    with pytest.raises(ValueError, match="finite lat/lon"):
        ground_track_azimuth_deg(grid, CENTER_LAT, CENTER_LON)


@pytest.mark.parametrize(("target", "match"), [((np.nan, CENTER_LON), "must be finite")])
def test_ground_track_azimuth_non_finite_target_raises(target, match):
    with pytest.raises(ValueError, match=match):
        ground_track_azimuth_deg(_track_grid(191.0), *target)


@pytest.mark.parametrize("kernel_shape", [(5, 8), (4, 7), (1, 1)])
def test_crop_to_full_psf_overlap_keeps_the_valid_convolution(kernel_shape):
    from scipy.signal import fftconvolve

    from curryer.correction.grid_types import PSFGrid
    from curryer.correction.psf import convolve_gcp_with_psf, crop_to_full_psf_overlap

    rng = np.random.default_rng(0)
    lon, lat = np.meshgrid(np.arange(50) * 1e-3, 30.0 - np.arange(40) * 1e-3)
    chip = ImageGrid(data=rng.normal(size=(40, 50)), lat=lat, lon=lon)
    psf = PSFGrid(data=rng.random(kernel_shape), lat=np.arange(kernel_shape[0]), lon=np.arange(kernel_shape[1]))

    cropped = crop_to_full_psf_overlap(convolve_gcp_with_psf(chip, psf), psf)

    valid = fftconvolve(chip.data, np.flipud(psf.data), mode="valid")
    np.testing.assert_allclose(cropped.data, valid, atol=1e-10)
    rows, cols = kernel_shape[0] // 2, kernel_shape[1] // 2
    np.testing.assert_array_equal(cropped.lat, lat[rows : rows + valid.shape[0], cols : cols + valid.shape[1]])
    np.testing.assert_array_equal(cropped.lon, lon[rows : rows + valid.shape[0], cols : cols + valid.shape[1]])


def test_crop_with_psf_not_smaller_than_chip_raises():
    from curryer.correction.grid_types import PSFGrid
    from curryer.correction.psf import crop_to_full_psf_overlap

    lon, lat = np.meshgrid(np.arange(6) * 1e-3, np.arange(4) * 1e-3)
    chip = ImageGrid(data=np.ones((4, 6)), lat=lat, lon=lon)
    psf = PSFGrid(data=np.ones((4, 3)), lat=np.arange(4), lon=np.arange(3))
    with pytest.raises(ValueError, match="not smaller than the GCP chip"):
        crop_to_full_psf_overlap(chip, psf)


SIGMA_DEG = 0.002
FRAME_STEP_DEG = 0.006
BLUR_SAMPLING = PSFSamplingConfig(psf_lat_sample_dist_deg=1e-4, psf_lon_sample_dist_deg=1e-4)
UNEQUAL_SAMPLING = PSFSamplingConfig(psf_lat_sample_dist_deg=0.85e-4, psf_lon_sample_dist_deg=1e-4)


def _rotated(heading_deg: float, along: np.ndarray, across: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat of offsets *along* / *across* a heading measured counterclockwise from east."""
    h = np.deg2rad(heading_deg)
    return CENTER_LON + along * np.cos(h) - across * np.sin(h), CENTER_LAT + along * np.sin(h) + across * np.cos(h)


def _gaussian_psf(heading_deg: float, floor: float = 0.0) -> ProjectedPSF:
    """Gaussian PSF twice as wide across the heading as along it, sampled on a grid aligned with it."""
    along, across = np.meshgrid(
        np.linspace(-4, 4, 41) * SIGMA_DEG, np.linspace(-4, 4, 41) * 2 * SIGMA_DEG, indexing="ij"
    )
    lon, lat = _rotated(heading_deg, along, across)
    data = np.exp(-0.5 * ((along / SIGMA_DEG) ** 2 + (across / (2 * SIGMA_DEG)) ** 2)) + floor
    return ProjectedPSF(data=data, lat=lat, lon=lon)


def _frames(heading_deg: float, n_frames: int = 5) -> ImageGrid:
    """Frames advancing FRAME_STEP_DEG along the heading, three pixels across it."""
    along, across = np.meshgrid(np.arange(n_frames) * FRAME_STEP_DEG, np.array([-1.0, 0.0, 1.0]) * 0.005, indexing="ij")
    lon, lat = _rotated(heading_deg, along, across)
    return ImageGrid(data=np.ones(lat.shape), lat=lat, lon=lon)


def _rms_widths(psf: PSFGrid, heading_deg: float) -> tuple[float, float]:
    lat, lon = np.meshgrid(psf.lat, psf.lon, indexing="ij")
    w = psf.data / psf.data.sum()
    d_lon, d_lat = lon - np.sum(w * lon), lat - np.sum(w * lat)
    h = np.deg2rad(heading_deg)
    along = d_lon * np.cos(h) + d_lat * np.sin(h)
    across = -d_lon * np.sin(h) + d_lat * np.cos(h)
    return float(np.sqrt(np.sum(w * along**2))), float(np.sqrt(np.sum(w * across**2)))


@pytest.mark.parametrize("sampling", [BLUR_SAMPLING, UNEQUAL_SAMPLING], ids=["equal-sampling", "unequal-sampling"])
@pytest.mark.parametrize("heading_deg", [0.0, 41.0, 90.0, 139.0, 225.0])
def test_motion_blur_widens_the_psf_by_one_frame_step_along_track(heading_deg, sampling):
    blurred = convolve_psf_with_spacecraft_motion(_gaussian_psf(heading_deg), _frames(heading_deg), sampling)

    along, across = _rms_widths(blurred, heading_deg)
    assert along == pytest.approx(np.sqrt(SIGMA_DEG**2 + FRAME_STEP_DEG**2 / 12), rel=0.03)
    assert across == pytest.approx(2 * SIGMA_DEG, rel=0.03)


def test_motion_blur_is_zero_outside_the_sampled_psf():
    # The PSF is sampled on a grid aligned with east, the frames advance at 45 degrees: the
    # samples do not fill the grid the blur runs on, and nothing may be filled in beyond them.
    blurred = convolve_psf_with_spacecraft_motion(_gaussian_psf(0.0, floor=0.5), _frames(45.0), BLUR_SAMPLING)

    lat, lon = np.meshgrid(blurred.lat, blurred.lon, indexing="ij")
    reach = FRAME_STEP_DEG / 2 + 2e-4
    outside = (np.abs(lon - CENTER_LON) > 4 * SIGMA_DEG + reach) | (np.abs(lat - CENTER_LAT) > 8 * SIGMA_DEG + reach)
    assert outside.any()
    assert blurred.data[~outside].max() > 0
    np.testing.assert_array_equal(blurred.data[outside], 0.0)


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        (_frames(0.0, n_frames=1), "at least two frames"),
        (
            ImageGrid(data=np.ones((2, 3)), lat=np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]), lon=np.zeros((2, 3))),
            "finite and nonzero",
        ),
        (
            ImageGrid(data=np.ones((2, 3)), lat=np.zeros((2, 3)), lon=np.zeros((2, 3))),
            "finite and nonzero",
        ),
    ],
    ids=["one-frame", "non-finite-step", "zero-step"],
)
def test_motion_blur_without_a_frame_step_raises(frames, match):
    with pytest.raises(ValueError, match=match):
        convolve_psf_with_spacecraft_motion(_gaussian_psf(0.0), frames, BLUR_SAMPLING)


def _chip_grid(step_deg: float = 1e-3, n: int = 41) -> ImageGrid:
    lon, lat = np.meshgrid(CENTER_LON + np.arange(n) * step_deg, CENTER_LAT - np.arange(n) * step_deg)
    return ImageGrid(data=np.zeros((n, n)), lat=lat, lon=lon)


def test_resample_onto_gcp_pixels_wider_than_the_psf_raises():
    # Lobes of 0.001 deg sampled every 0.003 deg would put the samples' weighted centre 0.1 pixel off.
    lat = CENTER_LAT + np.linspace(-0.004, 0.006, 101)
    lon = CENTER_LON + np.linspace(-0.005, 0.003, 81)
    d_lat, d_lon = np.meshgrid(lat - CENTER_LAT, lon - CENTER_LON, indexing="ij")
    data = np.exp(-0.5 * (d_lat**2 + d_lon**2) / 0.001**2)
    with pytest.raises(ValueError, match="wider than the PSF's RMS width"):
        resample_psf_to_gcp_resolution(PSFGrid(data=data, lat=lat, lon=lon), _chip_grid(step_deg=0.003))


def test_resampled_psf_is_centred_on_its_centroid():
    # Two lobes of unequal weight: the centroid is off the middle of the PSF's own grid.
    lat = CENTER_LAT + np.linspace(-0.004, 0.006, 101)
    lon = CENTER_LON + np.linspace(-0.005, 0.003, 81)
    d_lat, d_lon = np.meshgrid(lat - CENTER_LAT, lon - CENTER_LON, indexing="ij")
    data = np.exp(-0.5 * (d_lat**2 + d_lon**2) / 0.001**2) + 0.4 * np.exp(
        -0.5 * ((d_lat - 0.003) ** 2 + (d_lon + 0.002) ** 2) / 0.001**2
    )
    psf = PSFGrid(data=data, lat=lat, lon=lon)
    centroid_lat = np.sum(data.sum(axis=1) * lat) / data.sum()
    centroid_lon = np.sum(data.sum(axis=0) * lon) / data.sum()

    resampled = resample_psf_to_gcp_resolution(psf, _chip_grid())

    n_lat, n_lon = resampled.data.shape
    assert n_lat % 2 == 1
    assert n_lon % 2 == 1
    assert resampled.lat[n_lat // 2] == pytest.approx(centroid_lat, abs=1e-12)
    assert resampled.lon[n_lon // 2] == pytest.approx(centroid_lon, abs=1e-12)
    weights = resampled.data / resampled.data.sum()
    assert np.sum(weights.sum(axis=1) * np.arange(n_lat)) == pytest.approx(n_lat // 2, abs=0.01)
    assert np.sum(weights.sum(axis=0) * np.arange(n_lon)) == pytest.approx(n_lon // 2, abs=0.01)
    assert resampled.lat.min() <= lat.min()
    assert resampled.lat.max() >= lat.max()
    assert resampled.lon.min() <= lon.min()
    assert resampled.lon.max() >= lon.max()


@pytest.mark.parametrize("data", [np.zeros((5, 5)), -np.ones((5, 5))], ids=["zero", "negative"])
def test_resample_without_positive_psf_weights_raises(data):
    psf = PSFGrid(data=data, lat=CENTER_LAT + np.arange(5) * 1e-3, lon=CENTER_LON + np.arange(5) * 1e-3)
    with pytest.raises(ValueError, match="centroid is undefined"):
        resample_psf_to_gcp_resolution(psf, _chip_grid())
