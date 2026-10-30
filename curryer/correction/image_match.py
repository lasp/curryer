from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..compute import constants
from .config import PSFSamplingConfig, SearchConfig
from .grid_types import (
    ImageGrid,
    OpticalPSFEntry,
    ProjectedPSF,
    PSFGrid,
)
from .psf import (
    convolve_gcp_with_psf,
    convolve_psf_with_spacecraft_motion,
    crop_to_full_psf_overlap,
    normalize_psf,
    project_psf,
    resample_psf_to_gcp_resolution,
    zero_pad_psf,
)
from .search import emulate_image, im_search

logger = logging.getLogger(__name__)


# ============================================================================
# Output Validation
# ============================================================================


def validate_image_matching_output(output: xr.Dataset) -> None:
    """
    Validate image matching output conforms to expected format.

    Parameters
    ----------
    output : xr.Dataset
        Dataset returned by image matching function.

    Raises
    ------
    TypeError
        If output is not an xarray Dataset.
    ValueError
        If required fields are missing or malformed.

    Examples
    --------
    >>> result = image_matching_func(...)
    >>> validate_image_matching_output(result)
    """
    import xarray as xr

    if not isinstance(output, xr.Dataset):
        raise TypeError(f"Image matching must return xr.Dataset, got {type(output)}")

    required_vars = [
        "lat_error_deg",
        "lon_error_deg",
        "gcp_lat_deg",
        "gcp_lon_deg",
        "gcp_alt",
    ]

    missing = [v for v in required_vars if v not in output.data_vars]
    if missing:
        available = list(output.data_vars)
        raise ValueError(f"Image matching output missing required variables: {missing}. Available: {available}")

    if "measurement" not in output.coords:
        raise ValueError(
            f"Image matching output missing 'measurement' coordinate. Available coords: {list(output.coords)}"
        )

    # Check that error arrays have the measurement dimension
    for var in ["lat_error_deg", "lon_error_deg"]:
        if "measurement" not in output[var].dims:
            raise ValueError(f"Variable '{var}' must have 'measurement' dimension, got dimensions: {output[var].dims}")


# ============================================================================
# Integrated Image Matching Implementation
# ============================================================================


@dataclass
class IntegratedImageMatchResult:
    lat_error_km: float
    lon_error_km: float
    ccv_final: float
    ccv_secondary: float
    final_index_row: int
    final_index_col: int
    final_grid_step_m: float
    dynamic_psf: PSFGrid
    projected_psf: ProjectedPSF
    convolved_gcp: ImageGrid


def integrated_image_match(
    subimage: ImageGrid,
    gcp: ImageGrid,
    r_iss_midframe_m: np.ndarray,
    los_vectors_hs: np.ndarray,
    optical_psfs: Iterable[OpticalPSFEntry],
    geolocation_config: PSFSamplingConfig | None = None,
    search_config: SearchConfig | None = None,
) -> IntegratedImageMatchResult:
    """
    Perform complete image matching workflow with PSF modeling.

    Replicates MATLAB IntegratedImageMatch: projects PSF, applies spacecraft
    motion blur, convolves with GCP, and performs correlation-based search.

    Parameters
    ----------
    subimage : ImageGrid
        Observed image data with geolocation.
    gcp : ImageGrid
        Ground control point reference image.
    r_iss_midframe_m : np.ndarray
        Spacecraft position at mid-frame, shape (3,), units: meters.
    los_vectors_hs : np.ndarray
        Line-of-sight vectors in instrument frame, shape (n_pixels, 3).
    optical_psfs : Iterable[OpticalPSFEntry]
        Optical PSF samples at different field angles.
    geolocation_config : PSFSamplingConfig, optional
        PSF geolocation parameters. Defaults to standard config.
    search_config : SearchConfig, optional
        Image search parameters. Defaults to standard config.

    Returns
    -------
    IntegratedImageMatchResult
        Geolocation errors, correlation value, and intermediate products.
    """

    geo_config = geolocation_config or PSFSamplingConfig()
    search_cfg = search_config or SearchConfig()

    logger.debug("Projecting the PSF...")
    projected_psf = project_psf(r_iss_midframe_m, optical_psfs, subimage, los_vectors_hs)

    logger.debug("Convolving the PSF with the Spacecraft...")
    dynamic_psf = convolve_psf_with_spacecraft_motion(projected_psf, subimage, geo_config)

    logger.debug("Zero padding the PSF...")
    dynamic_psf = zero_pad_psf(dynamic_psf)

    logger.debug("Resampling the PSG to the GCP resolution...")
    dynamic_psf = resample_psf_to_gcp_resolution(dynamic_psf, gcp)

    logger.debug("Normalizing the PSF...")
    dynamic_psf = normalize_psf(dynamic_psf)

    logger.debug("Convolving the GCP with the PSF...")
    gcp_convolved = convolve_gcp_with_psf(gcp, dynamic_psf)

    logger.debug("Performing image search...")
    (
        lat_error_est,
        lon_error_est,
        ccv_final,
        ccv_secondary,
        final_idx_row,
        final_idx_col,
        final_grid_step_m,
    ) = im_search(crop_to_full_psf_overlap(gcp_convolved, dynamic_psf), subimage, search_cfg)

    return IntegratedImageMatchResult(
        lat_error_km=lat_error_est,
        lon_error_km=lon_error_est,
        ccv_final=ccv_final,
        ccv_secondary=ccv_secondary,
        final_index_row=final_idx_row,
        final_index_col=final_idx_col,
        final_grid_step_m=final_grid_step_m,
        dynamic_psf=dynamic_psf,
        projected_psf=projected_psf,
        convolved_gcp=gcp_convolved,
    )


def chip_image_dataset(
    subimage: ImageGrid, gcp: ImageGrid, result: IntegratedImageMatchResult, search_config: SearchConfig
) -> xr.Dataset:
    """Collect the images behind one GCP match for review.

    The emulated image is the PSF-convolved GCP chip sampled at the observation
    pixels shifted by the matched error, i.e. the image whose correlation with
    the observation is ``result.ccv_final``.  The reference images are cropped
    to the chip rows and columns the search could reach, the subimage's
    lat/lon extent widened by ``search_config.max_shift_km()`` and one chip
    pixel each way, and stored as float32.

    Parameters
    ----------
    subimage : ImageGrid
        Observation subimage that was matched (rows = frames, columns =
        cross-track pixels), lat/lon in degrees.
    gcp : ImageGrid
        GCP reference chip as loaded, before PSF convolution.
    result : IntegratedImageMatchResult
        Output of :func:`integrated_image_match` for *subimage* and *gcp*.
    search_config : SearchConfig
        The search configuration of the match, which bounds the shift.

    Returns
    -------
    xr.Dataset
        On dims ``(row, col)``: ``observed``, ``emulated``, ``latitude`` and
        ``longitude`` of the observation.  On dims ``(gcp_row, gcp_col)``:
        ``reference``, ``reference_convolved``, ``reference_latitude`` and
        ``reference_longitude``, cropped as above.  Attributes
        ``lat_error_km``, ``lon_error_km``, ``correlation`` and
        ``correlation_secondary``.

    Raises
    ------
    ValueError
        If no chip row or column lies within the reachable region.
    """
    n_rows, n_cols = subimage.data.shape
    mid_lat = float(subimage.lat[n_rows // 2, n_cols // 2])
    radius_km = constants.WGS84_SEMI_MAJOR_AXIS_KM
    lat_shift_deg = np.rad2deg(result.lat_error_km / radius_km)
    lon_shift_deg = np.rad2deg(result.lon_error_km / (radius_km * np.cos(np.deg2rad(mid_lat))))
    emulated = emulate_image(subimage.lon - lon_shift_deg, subimage.lat - lat_shift_deg, result.convolved_gcp)

    reach_lat_deg = np.rad2deg(search_config.max_shift_km() / radius_km)
    reach_lon_deg = reach_lat_deg / np.cos(np.deg2rad(mid_lat))
    rows = _reachable(gcp.lat[:, 0], subimage.lat, reach_lat_deg)
    cols = _reachable(gcp.lon[0, :], subimage.lon, reach_lon_deg)
    if rows.stop <= rows.start or cols.stop <= cols.start:
        raise ValueError("No GCP chip row or column lies within the region the search could reach.")
    crop = (rows, cols)

    pixel = ("row", "col")
    chip = ("gcp_row", "gcp_col")
    return xr.Dataset(
        {
            "observed": (pixel, np.asarray(subimage.data, dtype=float)),
            "emulated": (pixel, emulated, {"long_name": "PSF-convolved GCP sampled at the matched positions"}),
            "latitude": (pixel, subimage.lat, {"units": "degrees_north"}),
            "longitude": (pixel, subimage.lon, {"units": "degrees_east"}),
            "reference": (chip, np.asarray(gcp.data[crop], dtype=np.float32)),
            "reference_convolved": (chip, np.asarray(result.convolved_gcp.data[crop], dtype=np.float32)),
            "reference_latitude": (chip, gcp.lat[crop], {"units": "degrees_north"}),
            "reference_longitude": (chip, gcp.lon[crop], {"units": "degrees_east"}),
        },
        attrs={
            "lat_error_km": result.lat_error_km,
            "lon_error_km": result.lon_error_km,
            "correlation": result.ccv_final,
            "correlation_secondary": result.ccv_secondary,
        },
    )


def _reachable(axis: np.ndarray, coords: np.ndarray, reach_deg: float) -> slice:
    """Slice of a monotonic chip *axis* covering *coords* +/- *reach_deg*, plus one sample each way."""
    inside = np.flatnonzero((axis >= np.min(coords) - reach_deg) & (axis <= np.max(coords) + reach_deg))
    if inside.size == 0:
        return slice(0, 0)
    return slice(max(int(inside[0]) - 1, 0), min(int(inside[-1]) + 2, axis.size))
