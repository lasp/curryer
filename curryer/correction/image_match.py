from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .data_structures import (
    GeolocationConfig,
    ImageGrid,
    OpticalPSFEntry,
    ProjectedPSF,
    PSFGrid,
    SearchConfig,
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

logger = logging.getLogger(__name__)


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
    geolocation_config: Optional[GeolocationConfig] = None,
    search_config: Optional[SearchConfig] = None,
) -> IntegratedImageMatchResult:
    """Replicate the MATLAB IntegratedImageMatch workflow in Python."""

    geo_config = geolocation_config or GeolocationConfig()
    search_cfg = search_config or SearchConfig()

    logger.debug('Projecting the PSF...')
    projected_psf = project_psf(r_iss_midframe_m, optical_psfs, subimage, los_vectors_hs)
    
    logger.debug('Convolving the PSF with the Spacecraft...')
    dynamic_psf = convolve_psf_with_spacecraft_motion(projected_psf, subimage, geo_config)

    logger.debug('Zero padding the PSF...')
    dynamic_psf = zero_pad_psf(dynamic_psf)

    logger.debug('Resampling the PSG to the GCP resolution...')
    dynamic_psf = resample_psf_to_gcp_resolution(dynamic_psf, gcp)

    logger.debug('Normalizing the PSF...')
    dynamic_psf = normalize_psf(dynamic_psf)

    logger.debug('Convolving the GCP with the PSF...')
    gcp_convolved = convolve_gcp_with_psf(gcp, dynamic_psf)

    logger.debug('Performing image search...')
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

