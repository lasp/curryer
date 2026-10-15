"""Integration test of ``curryer.correction.image_match`` on a synthetic scene.

A pushbroom observation of an analytic scene is simulated by integrating the scene
over the projected PSF and the ground motion during each frame, directly and without
curryer's PSF convolution.  ``integrated_image_match`` must recover the geolocation
error imposed on the observation, from the GCP chip sampled from the same scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from curryer.compute.spatial import geodetic_to_ecef
from curryer.correction.config import PSFSamplingConfig, SearchConfig
from curryer.correction.grid_types import ImageGrid, OpticalPSFEntry
from curryer.correction.image_match import integrated_image_match
from curryer.correction.psf import project_psf

CENTER_LAT, CENTER_LON = 26.15, -102.33
ALTITUDE_M = 410_000.0
KM_PER_DEG = 111.32
PIXEL_KM = 0.46
N_FRAMES = N_PIXELS = 25
SAMPLING = PSFSamplingConfig(psf_lat_sample_dist_deg=2e-4, psf_lon_sample_dist_deg=2e-4)
SEARCH = SearchConfig(grid_size=21, grid_span_km=4.0, reduction_factor=0.5, spacing_limit_m=20.0, peak_exclusion_km=1.0)


def _offset(lat_km, lon_km):
    """Lat/lon (deg) of offsets north and east (km) of the scene centre."""
    return CENTER_LAT + lat_km / KM_PER_DEG, CENTER_LON + lon_km / (KM_PER_DEG * np.cos(np.deg2rad(CENTER_LAT)))


def _scene(lat, lon):
    """Gaussian blobs 0.1-0.6 km wide, near the pixel size, scattered over +-20 km of the centre."""
    rng = np.random.default_rng(7)
    n = 80
    blob_lat, blob_lon = _offset(rng.uniform(-20, 20, n), rng.uniform(-20, 20, n))
    width_km, amplitude = rng.uniform(0.1, 0.6, n), rng.uniform(-1.0, 1.0, n)
    north = (lat[..., None] - blob_lat) * KM_PER_DEG
    east = (lon[..., None] - blob_lon) * KM_PER_DEG * np.cos(np.deg2rad(CENTER_LAT))
    return np.sum(amplitude * np.exp(-0.5 * (north**2 + east**2) / width_km**2), axis=-1)


def _chip(half_km=16.0, step_km=0.1):
    north = np.arange(half_km, -half_km - step_km / 2, -step_km)
    east = np.arange(-half_km, half_km + step_km / 2, step_km)
    lat, lon = _offset(*np.meshgrid(north, east, indexing="ij"))
    return ImageGrid(data=_scene(lat, lon), lat=lat, lon=lon)


def _ground(track_az_deg):
    """Pixel centres, rows = frames advancing along *track_az_deg* (clockwise from north)."""
    az = np.deg2rad(track_az_deg)
    along, across = np.meshgrid(
        (np.arange(N_FRAMES) - N_FRAMES // 2) * PIXEL_KM,
        (np.arange(N_PIXELS) - N_PIXELS // 2) * PIXEL_KM,
        indexing="ij",
    )
    return _offset(along * np.cos(az) - across * np.sin(az), along * np.sin(az) + across * np.cos(az))


def _geometry(lat, lon):
    """Spacecraft ECEF position over the centre and the instrument-frame line of sight per column."""
    r = geodetic_to_ecef(np.array([CENTER_LON, CENTER_LAT, ALTITUDE_M]), meters=True, degrees=True)
    mid = N_FRAMES // 2
    ground = geodetic_to_ecef(np.column_stack([lon[mid], lat[mid], np.zeros(N_PIXELS)]), meters=True, degrees=True)
    los = (ground - r) / np.linalg.norm(ground - r, axis=1, keepdims=True)
    z = los[N_PIXELS // 2]
    y = los[-1] - los[0] - np.dot(los[-1] - los[0], z) * z
    y /= np.linalg.norm(y)
    return r, los @ np.vstack([np.cross(y, z), y, z]).T


def _optical_psf():
    """Gaussian optical PSF, 0.012 deg RMS along track and 0.03 deg along the slit."""
    x, field = np.linspace(-0.036, 0.036, 15), np.linspace(-0.09, 0.09, 15)
    data = np.exp(-0.5 * ((x[None, :] / 0.012) ** 2 + (field[:, None] / 0.03) ** 2))
    return [OpticalPSFEntry(data=data, x=x, field_angle=field)]


def _observe(lat, lon, r, los_hs, psfs):
    """Each pixel: the scene averaged over the projected PSF and over one frame step of motion."""
    grid = ImageGrid(data=np.zeros(lat.shape), lat=lat, lon=lon)
    projected = project_psf(r, psfs, grid, los_hs)
    w = projected.data / projected.data.sum()
    d_lat = projected.lat - np.sum(w * projected.lat)
    d_lon = projected.lon - np.sum(w * projected.lon)
    step_lat, step_lon = np.mean(np.diff(lat, axis=0)), np.mean(np.diff(lon, axis=0))
    motion = np.linspace(-0.5, 0.5, 9)
    sample_lat = lat[..., None, None] + d_lat.ravel()[:, None] + motion * step_lat
    sample_lon = lon[..., None, None] + d_lon.ravel()[:, None] + motion * step_lon
    return np.einsum("ijkm,k->ij", _scene(sample_lat, sample_lon), w.ravel()) / motion.size


@pytest.mark.parametrize(
    ("track_az_deg", "error_km"),
    [(0.0, (1.5, -0.8)), (41.0, (-1.2, 1.0)), (139.0, (2.0, 0.3)), (230.0, (-0.6, -1.7))],
    ids=["north", "northeast", "southeast", "southwest"],
)
def test_integrated_image_match_recovers_an_imposed_error(track_az_deg, error_km):
    lat, lon = _ground(track_az_deg)
    r, los_hs = _geometry(lat, lon)
    psfs = _optical_psf()
    observed = _observe(lat, lon, r, los_hs, psfs)
    geolocated_lat, geolocated_lon = _offset(
        (lat - CENTER_LAT) * KM_PER_DEG + error_km[0],
        (lon - CENTER_LON) * KM_PER_DEG * np.cos(np.deg2rad(CENTER_LAT)) + error_km[1],
    )
    subimage = ImageGrid(data=observed, lat=geolocated_lat, lon=geolocated_lon)

    result = integrated_image_match(subimage, _chip(), r, los_hs, psfs, SAMPLING, SEARCH)

    assert result.lat_error_km == pytest.approx(error_km[0], abs=0.015)
    assert result.lon_error_km == pytest.approx(error_km[1], abs=0.015)
    assert result.ccv_final > 0.999
