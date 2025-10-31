from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..compute import constants
from .data_structures import ImageGrid, SearchConfig

logger = logging.getLogger(__name__)


def emulate_image(test_lon: np.ndarray, test_lat: np.ndarray, gcp: ImageGrid) -> np.ndarray:
    """Interpolate the GCP chip at test latitude/longitude coordinates."""

    lon_axis = gcp.lon[0, :]
    lat_axis = gcp.lat[:, 0]
    data = gcp.data

    lat_axis_work = lat_axis
    data_work = data
    if np.any(np.diff(lat_axis) <= 0):
        lat_axis_work = lat_axis[::-1]
        data_work = data_work[::-1, :]
        if not np.all(np.diff(lat_axis_work) > 0):
            raise ValueError("GCP latitude axis must be monotonic.")

    lon_axis_work = lon_axis
    if np.any(np.diff(lon_axis) <= 0):
        lon_axis_work = lon_axis[::-1]
        data_work = data_work[:, ::-1]
        if not np.all(np.diff(lon_axis_work) > 0):
            raise ValueError("GCP longitude axis must be monotonic.")

    interpolant = RegularGridInterpolator((lat_axis_work, lon_axis_work), data_work, bounds_error=False, fill_value=0.0)
    points = np.stack((test_lat.ravel(), test_lon.ravel()), axis=-1)
    values = interpolant(points)
    return values.reshape(test_lat.shape)


def ccv2d(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute the correlation coefficient between two images."""

    img1 = np.asarray(image1, dtype=float)
    img2 = np.asarray(image2, dtype=float)
    diff1 = img1 - img1.mean()
    diff2 = img2 - img2.mean()
    numerator = np.sum(diff1 * diff2)
    denominator = np.sqrt(np.sum(diff1**2) * np.sum(diff2**2))
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def im_search(
    gcp: ImageGrid,
    subimage: ImageGrid,
    config: SearchConfig,
) -> tuple[float, float, float, int, int, float]:
    """Perform the iterative grid search used by the MATLAB implementation."""

    nframes, nrows = subimage.data.shape
    midframe = nframes // 2
    midrow = nrows // 2

    new_image_lat = subimage.lat.copy()
    new_image_lon = subimage.lon.copy()

    grid_dim_lat = (config.grid_span_km / constants.WGS84_SEMI_MAJOR_AXIS_KM) * 180.0 / np.pi
    lat_spacing = grid_dim_lat / (config.grid_size - 1)
    mid_index = config.grid_size // 2

    lat_spacing_min = (config.spacing_limit_m / (constants.WGS84_SEMI_MAJOR_AXIS_KM * 1000.0)) * 180.0 / np.pi

    best_grid = (0, 0)
    ccv_max = -np.inf

    while lat_spacing > lat_spacing_min:
        ccv_max = -np.inf
        for k in range(config.grid_size):
            for kk in range(config.grid_size):
                lat_shift = (mid_index - k) * lat_spacing
                lon_shift = (kk - mid_index) * lat_spacing
                test_lat = new_image_lat + lat_shift
                test_lon = new_image_lon + lon_shift
                test_image = emulate_image(test_lon, test_lat, gcp)
                ccv_value = ccv2d(test_image, subimage.data)
                if ccv_value > ccv_max:
                    ccv_max = ccv_value
                    best_grid = (k, kk)

        lon_shift = (best_grid[1] - mid_index) * lat_spacing
        lat_shift = (mid_index - best_grid[0]) * lat_spacing
        new_image_lat = new_image_lat + lat_shift
        new_image_lon = new_image_lon + lon_shift
        lat_spacing *= config.reduction_factor

        logger.debug("Best point in grid = %s", best_grid)
        logger.debug("CCVmax= %s", ccv_max)
        logger.debug("Set dgrid to [m]= %s", lat_spacing * np.pi / 180 * (constants.WGS84_SEMI_MAJOR_AXIS_KM * 1000.0))

    lat_error_km = (
        (subimage.lat[midframe, midrow] - new_image_lat[midframe, midrow])
        * np.pi
        / 180.0
        * constants.WGS84_SEMI_MAJOR_AXIS_KM
    )
    lon_error_km = (
        (subimage.lon[midframe, midrow] - new_image_lon[midframe, midrow])
        * np.pi
        / 180.0
        * constants.WGS84_SEMI_MAJOR_AXIS_KM
        * np.cos(np.deg2rad(subimage.lat[midframe, midrow]))
    )

    final_grid_step_m = lat_spacing * np.pi / 180.0 * (constants.WGS84_SEMI_MAJOR_AXIS_KM * 1000.0)

    return (
        float(lat_error_km),
        float(lon_error_km),
        float(ccv_max),
        int(best_grid[0]),
        int(best_grid[1]),
        float(final_grid_step_m),
    )
