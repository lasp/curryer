"""
Image-matching helpers for CLARREO integration tests.

Provides utilities for discovering test cases, applying artificial
geolocation errors, and running image matching with those errors.
These are *test infrastructure helpers*, not pytest tests themselves.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr

from curryer.correction.config import PSFSamplingConfig, SearchConfig
from curryer.correction.grid_types import ImageGrid
from curryer.correction.image_io import (
    load_image_grid,
    load_los_vectors,
    load_optical_psf,
)
from curryer.correction.image_match import integrated_image_match

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test-case metadata
# ---------------------------------------------------------------------------

_TEST_CASE_METADATA: dict[str, dict] = {
    "1": {
        "name": "Dili",
        "gcp_file": "GCP12055Dili_resampled.mat",
        "ancil_file": "R_ISS_midframe_TestCase1.mat",
        "expected_error_km": (3.0, -3.0),
        "cases": [
            {"subimage": "TestCase1a_subimage.mat", "binned": False},
            {"subimage": "TestCase1b_subimage.mat", "binned": False},
            {"subimage": "TestCase1c_subimage_binned.mat", "binned": True},
            {"subimage": "TestCase1d_subimage_binned.mat", "binned": True},
        ],
    },
    "2": {
        "name": "Maracaibo",
        "gcp_file": "GCP10121Maracaibo_resampled.mat",
        "ancil_file": "R_ISS_midframe_TestCase2.mat",
        "expected_error_km": (-3.0, 2.0),
        "cases": [
            {"subimage": "TestCase2a_subimage.mat", "binned": False},
            {"subimage": "TestCase2b_subimage.mat", "binned": False},
            {"subimage": "TestCase2c_subimage_binned.mat", "binned": True},
        ],
    },
    "3": {
        "name": "Algeria3",
        "gcp_file": "GCP10181Algeria3_resampled.mat",
        "ancil_file": "R_ISS_midframe_TestCase3.mat",
        "expected_error_km": (2.0, 3.0),
        "cases": [
            {"subimage": "TestCase3a_subimage.mat", "binned": False},
            {"subimage": "TestCase3b_subimage_binned.mat", "binned": True},
        ],
    },
    "4": {
        "name": "Dunhuang",
        "gcp_file": "GCP10142Dunhuang_resampled.mat",
        "ancil_file": "R_ISS_midframe_TestCase4.mat",
        "expected_error_km": (-2.0, -3.0),
        "cases": [
            {"subimage": "TestCase4a_subimage.mat", "binned": False},
            {"subimage": "TestCase4b_subimage_binned.mat", "binned": True},
        ],
    },
    "5": {
        "name": "Algeria5",
        "gcp_file": "GCP10071Algeria5_resampled.mat",
        "ancil_file": "R_ISS_midframe_TestCase5.mat",
        "expected_error_km": (1.0, -1.0),
        "cases": [
            {"subimage": "TestCase5a_subimage.mat", "binned": False},
        ],
    },
}


def discover_test_image_match_cases(
    test_data_dir: Path,
    test_cases: list[str] | None = None,
) -> list[dict]:
    """Scan *test_data_dir* for validated image-matching test cases.

    Parameters
    ----------
    test_data_dir:
        Root directory (e.g. ``tests/data/clarreo/image_match/``).
    test_cases:
        Specific case IDs to include (e.g. ``['1', '2']``).  ``None`` means
        all available cases.

    Returns
    -------
    list[dict]
        One dict per sub-case variant with keys: ``case_id``, ``case_name``,
        ``subcase_name``, ``subimage_file``, ``gcp_file``, ``ancil_file``,
        ``los_file``, ``psf_file``, ``expected_lat_error_km``,
        ``expected_lon_error_km``, ``binned``.
    """
    logger.info("Discovering image-matching test cases in: %s", test_data_dir)

    los_file = test_data_dir / "b_HS.mat"
    psf_file_unbinned = test_data_dir / "optical_PSF_675nm_upsampled.mat"
    psf_file_binned = test_data_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat"

    if not los_file.exists():
        raise FileNotFoundError(f"LOS vectors file not found: {los_file}")
    if not psf_file_unbinned.exists():
        raise FileNotFoundError(f"PSF file not found: {psf_file_unbinned}")

    if test_cases is None:
        test_cases = sorted(_TEST_CASE_METADATA.keys())

    discovered: list[dict] = []
    for case_id in test_cases:
        if case_id not in _TEST_CASE_METADATA:
            logger.warning("Test case '%s' not in metadata, skipping", case_id)
            continue
        meta = _TEST_CASE_METADATA[case_id]
        case_dir = test_data_dir / case_id
        if not case_dir.is_dir():
            logger.warning("Test case directory not found: %s, skipping", case_dir)
            continue

        for subcase in meta["cases"]:
            subimage_file = case_dir / subcase["subimage"]
            gcp_file = case_dir / meta["gcp_file"]
            ancil_file = case_dir / meta["ancil_file"]
            psf_file = psf_file_binned if subcase["binned"] else psf_file_unbinned

            if not subimage_file.exists():
                logger.warning("Subimage not found: %s, skipping", subimage_file)
                continue
            if not gcp_file.exists():
                logger.warning("GCP file not found: %s, skipping", gcp_file)
                continue
            if not ancil_file.exists():
                logger.warning("Ancil file not found: %s, skipping", ancil_file)
                continue

            discovered.append(
                {
                    "case_id": case_id,
                    "case_name": meta["name"],
                    "subcase_name": subcase["subimage"],
                    "subimage_file": subimage_file,
                    "gcp_file": gcp_file,
                    "ancil_file": ancil_file,
                    "los_file": los_file,
                    "psf_file": psf_file,
                    "expected_lat_error_km": meta["expected_error_km"][0],
                    "expected_lon_error_km": meta["expected_error_km"][1],
                    "binned": subcase["binned"],
                }
            )

    logger.info("Discovered %d test-case variants", len(discovered))
    return discovered


# ---------------------------------------------------------------------------
# Error-variation helpers
# ---------------------------------------------------------------------------


def apply_error_variation_for_testing(
    base_result: xr.Dataset,
    param_idx: int,
    error_variation_percent: float = 3.0,
) -> xr.Dataset:
    """Apply random variation to image-matching results to simulate parameter effects.

    Uses *param_idx* as a reproducible random seed so each parameter set gets
    a distinct but deterministic perturbation.
    """
    output = base_result.copy(deep=True)
    rng = np.random.default_rng(param_idx)

    vf = error_variation_percent / 100.0
    lat_factor = 1.0 + rng.normal(0, vf)
    lon_factor = 1.0 + rng.normal(0, vf)
    ccv_factor = 1.0 + rng.normal(0, vf / 10.0)

    orig_lat = base_result.attrs["lat_error_km"]
    orig_lon = base_result.attrs["lon_error_km"]
    orig_ccv = base_result.attrs["correlation_ccv"]

    varied_lat = orig_lat * lat_factor
    varied_lon = orig_lon * lon_factor
    varied_ccv = float(np.clip(orig_ccv * ccv_factor, 0.0, 1.0))

    if "gcp_center_lat" in base_result.attrs:
        gcp_center_lat = base_result.attrs["gcp_center_lat"]
    elif "gcp_lat_deg" in base_result:
        gcp_center_lat = float(base_result["gcp_lat_deg"].values[0])
    else:
        gcp_center_lat = 45.0

    lat_error_deg = varied_lat / 111.0
    lon_radius_km = 6378.0 * np.cos(np.deg2rad(gcp_center_lat))
    lon_error_deg = varied_lon / (lon_radius_km * np.pi / 180.0)

    output["lat_error_deg"].values[0] = lat_error_deg
    output["lon_error_deg"].values[0] = lon_error_deg
    output.attrs.update(
        {
            "lat_error_km": varied_lat,
            "lon_error_km": varied_lon,
            "correlation_ccv": varied_ccv,
            "param_idx": param_idx,
            "variation_applied": True,
        }
    )
    return output


def apply_geolocation_error_to_subimage(
    subimage: ImageGrid,
    gcp: ImageGrid,
    lat_error_km: float,
    lon_error_km: float,
) -> ImageGrid:
    """Shift *subimage* coordinates by (lat_error_km, lon_error_km) for testing."""
    from curryer.compute import constants

    mid_lat = float(gcp.lat[gcp.lat.shape[0] // 2, gcp.lat.shape[1] // 2])
    earth_radius_km = constants.WGS84_SEMI_MAJOR_AXIS_KM
    lat_offset_deg = lat_error_km / earth_radius_km * (180.0 / np.pi)
    lon_radius_km = earth_radius_km * np.cos(np.deg2rad(mid_lat))
    lon_offset_deg = lon_error_km / lon_radius_km * (180.0 / np.pi)

    return ImageGrid(
        data=subimage.data.copy(),
        lat=subimage.lat + lat_offset_deg,
        lon=subimage.lon + lon_offset_deg,
        h=subimage.h.copy() if subimage.h is not None else None,
    )


def run_image_matching_with_applied_errors(
    test_case: dict,
    param_idx: int,
    randomize_errors: bool = True,
    error_variation_percent: float = 3.0,
    cache_results: bool = True,
    cached_result: xr.Dataset | None = None,
) -> xr.Dataset:
    """Run image matching with artificial errors; return result Dataset.

    Uses *cached_result* + random variation for param_idx > 0 when
    *cache_results* is True.
    """
    if cached_result is not None and cache_results and param_idx > 0:
        if randomize_errors:
            return apply_error_variation_for_testing(cached_result, param_idx, error_variation_percent)
        return cached_result.copy()

    logger.info("Running image matching: %s", test_case["case_name"])
    start = time.time()

    subimage = load_image_grid(test_case["subimage_file"], mat_key="subimage")
    gcp = load_image_grid(test_case["gcp_file"], mat_key="GCP")
    mid_i, mid_j = gcp.mid_indices
    gcp_center_lat = float(gcp.lat[mid_i, mid_j])
    gcp_center_lon = float(gcp.lon[mid_i, mid_j])

    subimage_with_error = apply_geolocation_error_to_subimage(
        subimage, gcp, test_case["expected_lat_error_km"], test_case["expected_lon_error_km"]
    )

    los_vectors = load_los_vectors(test_case["los_file"])
    optical_psfs = load_optical_psf(test_case["psf_file"])
    from scipy.io import loadmat  # noqa: PLC0415

    ancil_data = loadmat(test_case["ancil_file"], squeeze_me=True)
    r_iss_midframe = ancil_data["R_ISS_midframe"].ravel()

    result = integrated_image_match(
        subimage=subimage_with_error,
        gcp=gcp,
        r_iss_midframe_m=r_iss_midframe,
        los_vectors_hs=los_vectors,
        optical_psfs=optical_psfs,
        geolocation_config=PSFSamplingConfig(),
        search_config=SearchConfig(),
    )
    processing_time = time.time() - start

    lat_error_deg = result.lat_error_km / 111.0
    lon_radius_km = 6378.0 * np.cos(np.deg2rad(gcp_center_lat))
    lon_error_deg = result.lon_error_km / (lon_radius_km * np.pi / 180.0)

    t_matrix = np.array(
        [
            [-0.418977524967338, 0.748005379751721, 0.514728846515064],
            [-0.421890284446342, 0.341604851993858, -0.839830169131854],
            [-0.804031356019172, -0.569029065124742, 0.172451447025628],
        ]
    )
    boresight = np.array([0.0, 0.0625969755450201, 0.99803888634292])

    output = xr.Dataset(
        {
            "lat_error_deg": (["measurement"], [lat_error_deg]),
            "lon_error_deg": (["measurement"], [lon_error_deg]),
            "riss_ctrs": (["measurement", "xyz"], [r_iss_midframe]),
            "bhat_hs": (["measurement", "xyz"], [boresight]),
            "t_hs2ctrs": (["measurement", "xyz_from", "xyz_to"], t_matrix[np.newaxis, :, :]),
            "gcp_lat_deg": (["measurement"], [gcp_center_lat]),
            "gcp_lon_deg": (["measurement"], [gcp_center_lon]),
            "gcp_alt": (["measurement"], [0.0]),
        },
        coords={"measurement": [0], "xyz": ["x", "y", "z"], "xyz_from": ["x", "y", "z"], "xyz_to": ["x", "y", "z"]},
    )
    output.attrs.update(
        {
            "lat_error_km": result.lat_error_km,
            "lon_error_km": result.lon_error_km,
            "correlation_ccv": result.ccv_final,
            "final_grid_step_m": result.final_grid_step_m,
            "processing_time_s": processing_time,
            "test_mode": True,
            "param_idx": param_idx,
            "gcp_center_lat": gcp_center_lat,
            "gcp_center_lon": gcp_center_lon,
        }
    )
    return output
