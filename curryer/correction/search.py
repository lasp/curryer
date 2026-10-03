from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..compute import constants
from .config import SearchConfig
from .grid_types import ImageGrid

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
) -> tuple[float, float, float, float, int, int, float]:
    """Perform the iterative grid search used by the MATLAB implementation.

    Every shifted subimage must lie within the GCP chip: the correlation is only
    defined where the chip has data.

    Returns
    -------
    lat_error_km, lon_error_km : float
        Geolocation error of the subimage centre (subimage minus matched), km.
    ccv_max : float
        Correlation coefficient at the final match.
    ccv_secondary : float
        Strongest correlation in the first (coarsest) search grid farther than
        ``config.peak_exclusion_km`` from that grid's best point; ``-inf`` when the
        grid has no such point.  A value close to *ccv_max* means the match is not
        distinct (e.g. cloud or featureless terrain).
    final_index_row, final_index_col : int
        Best grid indices of the last iteration.
    final_grid_step_m : float
        Grid spacing after the last reduction, metres.

    Raises
    ------
    ValueError
        If a search shift samples the subimage outside the GCP chip's lat/lon
        extent; crop the subimage further inside the chip or reduce
        ``grid_span_km`` / ``reduction_factor``.
    """

    nframes, nrows = subimage.data.shape
    midframe = nframes // 2
    midrow = nrows // 2

    gcp_lat_min, gcp_lat_max = float(np.min(gcp.lat)), float(np.max(gcp.lat))
    gcp_lon_min, gcp_lon_max = float(np.min(gcp.lon)), float(np.max(gcp.lon))
    km_per_deg = constants.WGS84_SEMI_MAJOR_AXIS_KM * np.pi / 180.0
    cos_lat = np.cos(np.deg2rad(subimage.lat[midframe, midrow]))
    ccv_secondary = -np.inf
    first_pass = True

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
        ccv_grid = np.full((config.grid_size, config.grid_size), -np.inf)
        for k in range(config.grid_size):
            for kk in range(config.grid_size):
                lat_shift = (mid_index - k) * lat_spacing
                lon_shift = (kk - mid_index) * lat_spacing
                test_lat = new_image_lat + lat_shift
                test_lon = new_image_lon + lon_shift
                if (
                    test_lat.min() < gcp_lat_min
                    or test_lat.max() > gcp_lat_max
                    or test_lon.min() < gcp_lon_min
                    or test_lon.max() > gcp_lon_max
                ):
                    raise ValueError(
                        f"Search shift ({lat_shift * km_per_deg:+.2f} km N, "
                        f"{lon_shift * km_per_deg * cos_lat:+.2f} km E) samples the subimage outside the GCP chip "
                        f"(lat {gcp_lat_min:.4f}..{gcp_lat_max:.4f}, lon {gcp_lon_min:.4f}..{gcp_lon_max:.4f}); "
                        "crop the subimage further inside the chip or reduce grid_span_km / reduction_factor."
                    )
                test_image = emulate_image(test_lon, test_lat, gcp)
                ccv_value = ccv2d(test_image, subimage.data)
                ccv_grid[k, kk] = ccv_value
                if ccv_value > ccv_max:
                    ccv_max = ccv_value
                    best_grid = (k, kk)

        if first_pass:
            rows, cols = np.indices(ccv_grid.shape)
            dist_km = np.hypot(
                (rows - best_grid[0]) * lat_spacing * km_per_deg,
                (cols - best_grid[1]) * lat_spacing * km_per_deg * cos_lat,
            )
            far = dist_km > config.peak_exclusion_km
            if far.any():
                ccv_secondary = float(ccv_grid[far].max())
            first_pass = False

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
        ccv_secondary,
        int(best_grid[0]),
        int(best_grid[1]),
        float(final_grid_step_m),
    )
