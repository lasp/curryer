from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .data_structures import (
    GeolocationConfig,
    ImageGrid,
    NamedImageGrid,
    OpticalPSFEntry,
    ProjectedPSF,
    PSFGrid,
    SearchConfig,
)
from .image_io import (
    load_image_grid_from_mat as _load_image_grid_from_mat,
)
from .image_io import (
    load_los_vectors_from_mat as _load_los_vectors_from_mat,
)
from .image_io import (
    load_optical_psf_from_mat as _load_optical_psf_from_mat,
)
from .psf import (
    convolve_gcp_with_psf,
    convolve_psf_with_spacecraft_motion,
    normalize_psf,
    project_psf,
    resample_psf_to_gcp_resolution,
    zero_pad_psf,
)
from .search import im_search

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

logger = logging.getLogger(__name__)


# ============================================================================
# Image Matching Interface Protocol
# ============================================================================


class ImageMatchingFunc(Protocol):
    """
    Protocol for image matching functions in Correction pipeline.

    Image matching functions perform spatial correlation between geolocated
    observations and GCP reference imagery to measure geolocation errors.

    Standard Signature:
        def image_matching(
            geolocated_data: xr.Dataset,
            gcp_reference_file: Path,
            telemetry: pd.DataFrame,
            calibration_dir: Path,
            params_info: list,
            config,
            los_vectors_cached: Optional[np.ndarray] = None,
            optical_psfs_cached: Optional[list] = None,
        ) -> xr.Dataset

    Required Output Fields:
        - lat_error_deg: (measurement,) Latitude errors in degrees
        - lon_error_deg: (measurement,) Longitude errors in degrees
        - gcp_lat_deg, gcp_lon_deg, gcp_alt: GCP location
        - Spacecraft state: position, boresight, transformation matrix

    See correction.image_matching() for reference implementation.
    """

    def __call__(
        self,
        geolocated_data: xr.Dataset,
        gcp_reference_file: Path,
        telemetry: pd.DataFrame,
        calibration_dir: Path,
        params_info: list,
        config,
        los_vectors_cached: np.ndarray | None = None,
        optical_psfs_cached: list | None = None,
    ) -> xr.Dataset:
        """Perform image matching and return error measurements."""
        ...


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
    geolocation_config: GeolocationConfig | None = None,
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
    geolocation_config : GeolocationConfig, optional
        PSF geolocation parameters. Defaults to standard config.
    search_config : SearchConfig, optional
        Image search parameters. Defaults to standard config.

    Returns
    -------
    IntegratedImageMatchResult
        Geolocation errors, correlation value, and intermediate products.
    """

    geo_config = geolocation_config or GeolocationConfig()
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
        final_idx_row,
        final_idx_col,
        final_grid_step_m,
    ) = im_search(gcp_convolved, subimage, search_cfg)

    return IntegratedImageMatchResult(
        lat_error_km=lat_error_est,
        lon_error_km=lon_error_est,
        ccv_final=ccv_final,
        final_index_row=final_idx_row,
        final_index_col=final_idx_col,
        final_grid_step_m=final_grid_step_m,
        dynamic_psf=dynamic_psf,
        projected_psf=projected_psf,
        convolved_gcp=gcp_convolved,
    )


# ============================================================================
# MATLAB File Loading Utilities
# ============================================================================


def load_image_grid_from_mat(
    mat_file: Path, key: str = "subimage", name: str | None = None, as_named: bool = False
) -> ImageGrid | NamedImageGrid:
    """
    Load ImageGrid from MATLAB .mat file.

    .. deprecated::
        Use :func:`curryer.correction.image_io.load_image_grid_from_mat` instead.
        This function will be removed in a future version.

    Parameters
    ----------
    mat_file : Path
        Path to .mat file.
    key : str, default="subimage"
        MATLAB struct key (e.g., "subimage" for L1A, "GCP" for reference).
    name : str, optional
        Name for NamedImageGrid. Defaults to file path.
    as_named : bool, default=False
        If True, return NamedImageGrid; otherwise return ImageGrid.

    Returns
    -------
    ImageGrid or NamedImageGrid
        Loaded image grid with data, lat, lon, h fields.
    """
    import warnings

    warnings.warn(
        "load_image_grid_from_mat from image_match is deprecated. "
        "Use: from curryer.correction.image_io import load_image_grid_from_mat",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_image_grid_from_mat(mat_file, key=key, name=name, as_named=as_named)


def load_optical_psf_from_mat(mat_file: Path, key: str = "PSF_struct_675nm") -> list[OpticalPSFEntry]:
    """
    Load optical PSF entries from MATLAB .mat file.

    .. deprecated::
        Use :func:`curryer.correction.image_io.load_optical_psf_from_mat` instead.
        This function will be removed in a future version.

    Parameters
    ----------
    mat_file : Path
        Path to MATLAB file with PSF data.
    key : str, default="PSF_struct_675nm"
        Primary key to try for PSF data.

    Returns
    -------
    list[OpticalPSFEntry]
        Optical PSF samples with data, x, and field_angle arrays.
    """
    import warnings

    warnings.warn(
        "load_optical_psf_from_mat from image_match is deprecated. "
        "Use: from curryer.correction.image_io import load_optical_psf_from_mat",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_optical_psf_from_mat(mat_file, key=key)


def load_los_vectors_from_mat(mat_file: Path, key: str = "b_HS") -> np.ndarray:
    """
    Load line-of-sight vectors from MATLAB .mat file.

    .. deprecated::
        Use :func:`curryer.correction.image_io.load_los_vectors_from_mat` instead.
        This function will be removed in a future version.

    Parameters
    ----------
    mat_file : Path
        Path to MATLAB file with LOS vectors.
    key : str, default="b_HS"
        Primary key to try for LOS data.

    Returns
    -------
    np.ndarray
        LOS unit vectors in instrument frame, shape (n_pixels, 3).
    """
    import warnings

    warnings.warn(
        "load_los_vectors_from_mat from image_match is deprecated. "
        "Use: from curryer.correction.image_io import load_los_vectors_from_mat",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_los_vectors_from_mat(mat_file, key=key)
