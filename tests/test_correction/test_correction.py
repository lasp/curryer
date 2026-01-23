#!/usr/bin/env python3
"""
Unified Correction Test Suite

This module consolidates two complementary Correction test approaches:

1. UPSTREAM Testing (run_upstream_pipeline):
   - Tests kernel creation and geolocation with parameter variations
   - Uses real telemetry data
   - Validates parameter modification and kernel generation
   - Stops before pairing (no valid GCP pairs available)

2. DOWNSTREAM Testing (run_downstream_pipeline):
   - Tests GCP pairing, image matching, and error statistics
   - Uses pre-geolocated test images with known GCP pairs
   - Validates spatial pairing, image matching algorithms, and error metrics
   - Skips kernel/geolocation (uses pre-computed test data)

Both tests share the same CLARREO configuration base but configure differently
for their specific testing needs.

Running Tests:
-------------
# Via pytest (recommended)
pytest tests/test_correction/test_correction.py -v

# Run specific test
pytest tests/test_correction/test_correction.py::test_generate_clarreo_config_json -v

# Standalone execution with arguments (for pipeline runs)
python tests/test_correction/test_correction.py --mode downstream --quick

Requirements:
-----------------
These tests validate the complete Correction geolocation pipeline,
demonstrating parameter sensitivity analysis and error statistics
computation for mission requirements validation.

"""

import argparse
import atexit
import json
import logging
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.io import loadmat

from curryer import meta, utils
from curryer import spicierpy as sp
from curryer.compute import constants
from curryer.correction import correction
from curryer.correction.data_structures import (
    GeolocationConfig as ImageMatchGeolocationConfig,
)
from curryer.correction.data_structures import (
    ImageGrid,
    SearchConfig,
)
from curryer.correction.image_match import (
    integrated_image_match,
    load_image_grid_from_mat,
    load_los_vectors_from_mat,
    load_optical_psf_from_mat,
)
from curryer.correction.pairing import find_l1a_gcp_pairs
from curryer.kernels import create

# Import CLARREO config and data loaders
sys.path.insert(0, str(Path(__file__).parent))
from clarreo_config import create_clarreo_correction_config
from clarreo_data_loaders import load_clarreo_gcp, load_clarreo_science, load_clarreo_telemetry

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])


# =============================================================================
# TEST PLACEHOLDER FUNCTIONS (For Synthetic Data Generation)
# =============================================================================
# These functions generate SYNTHETIC test data for testing the Correction pipeline


class _PlaceholderConfig:
    """Configuration for placeholder test data generation."""

    base_error_m: float = 50.0
    param_error_scale: float = 10.0
    max_measurements: int = 100
    min_measurements: int = 10
    orbit_radius_mean_m: float = 6.78e6
    orbit_radius_std_m: float = 4e3
    latitude_range: tuple[float, float] = (-60.0, 60.0)
    longitude_range: tuple[float, float] = (-180.0, 180.0)
    altitude_range: tuple[float, float] = (0.0, 1000.0)
    max_off_nadir_rad: float = 0.1


def synthetic_gcp_pairing(science_data_files):
    """Generate SYNTHETIC GCP pairs for testing (TEST ONLY - not a test itself)."""
    logger.warning("=" * 80)
    logger.warning("!!!!️  USING SYNTHETIC GCP PAIRING - FAKE DATA!  !!!!️")
    logger.warning("=" * 80)
    synthetic_pairs = [(f"{sci_file}", f"landsat_gcp_{i:03d}.tif") for i, sci_file in enumerate(science_data_files)]
    return synthetic_pairs


def synthetic_image_matching(
    geolocated_data,
    gcp_reference_file,
    telemetry,
    calibration_dir,
    params_info,
    config,
    los_vectors_cached=None,
    optical_psfs_cached=None,
):
    """
    Generate SYNTHETIC image matching error data (TEST ONLY - not a test itself).

    This function matches the signature of the real image_matching() function
    but only uses a subset of parameters for upstream testing of the Correction.

    Used parameters:
        geolocated_data: For generating realistic synthetic errors
        params_info: To scale errors based on parameter variations
        config: For coordinate names and placeholder configuration

    Ignored parameters (accepted for compatibility):
        gcp_reference_file: Not needed for synthetic data
        telemetry: Not needed for synthetic data
        calibration_dir: Not needed for synthetic data
        los_vectors_cached: Not needed for synthetic data
        optical_psfs_cached: Not needed for synthetic data
    """
    logger.warning("=" * 80)
    logger.warning("!!!!️  USING SYNTHETIC IMAGE MATCHING - FAKE DATA!  !!!!️")
    logger.warning("=" * 80)

    placeholder_cfg = (
        config.placeholder if hasattr(config, "placeholder") and config.placeholder else _PlaceholderConfig()
    )
    sc_pos_name = getattr(config, "spacecraft_position_name", "sc_position")
    boresight_name = getattr(config, "boresight_name", "boresight")
    transform_name = getattr(config, "transformation_matrix_name", "t_inst2ref")

    valid_mask = ~np.isnan(geolocated_data["latitude"].values).any(axis=1)
    n_valid = valid_mask.sum()
    n_measurements = (
        placeholder_cfg.min_measurements if n_valid == 0 else min(n_valid, placeholder_cfg.max_measurements)
    )

    # Generate realistic synthetic data
    riss_ctrs = _generate_spherical_positions(
        n_measurements, placeholder_cfg.orbit_radius_mean_m, placeholder_cfg.orbit_radius_std_m
    )
    boresights = _generate_synthetic_boresights(n_measurements, placeholder_cfg.max_off_nadir_rad)
    t_matrices = _generate_nadir_aligned_transforms(n_measurements, riss_ctrs, boresights)

    # Generate synthetic errors
    base_error = placeholder_cfg.base_error_m
    param_contribution = (
        sum(abs(p) if isinstance(p, int | float) else np.linalg.norm(p) for _, p in params_info)
        * placeholder_cfg.param_error_scale
    )
    error_magnitude = base_error + param_contribution
    lat_errors = np.random.normal(0, error_magnitude / 111000, n_measurements)
    lon_errors = np.random.normal(0, error_magnitude / 111000, n_measurements)

    if n_valid > 0:
        valid_indices = np.where(valid_mask)[0][:n_measurements]
        gcp_lat = geolocated_data["latitude"].values[valid_indices, 0]
        gcp_lon = geolocated_data["longitude"].values[valid_indices, 0]
    else:
        gcp_lat = np.random.uniform(*placeholder_cfg.latitude_range, n_measurements)
        gcp_lon = np.random.uniform(*placeholder_cfg.longitude_range, n_measurements)

    gcp_alt = np.random.uniform(*placeholder_cfg.altitude_range, n_measurements)

    return xr.Dataset(
        {
            "lat_error_deg": (["measurement"], lat_errors),
            "lon_error_deg": (["measurement"], lon_errors),
            sc_pos_name: (["measurement", "xyz"], riss_ctrs),
            boresight_name: (["measurement", "xyz"], boresights),
            transform_name: (["measurement", "xyz_from", "xyz_to"], t_matrices),
            "gcp_lat_deg": (["measurement"], gcp_lat),
            "gcp_lon_deg": (["measurement"], gcp_lon),
            "gcp_alt": (["measurement"], gcp_alt),
        },
        coords={
            "measurement": range(n_measurements),
            "xyz": ["x", "y", "z"],
            "xyz_from": ["x", "y", "z"],
            "xyz_to": ["x", "y", "z"],
        },
    )


def _generate_synthetic_boresights(n_measurements, max_off_nadir_rad=0.07):
    """Generate synthetic boresight vectors (test helper)."""
    boresights = np.zeros((n_measurements, 3))
    for i in range(n_measurements):
        theta = np.random.uniform(-max_off_nadir_rad, max_off_nadir_rad)
        boresights[i] = [0.0, np.sin(theta), np.cos(theta)]
    return boresights


def _generate_spherical_positions(n_measurements, radius_mean_m, radius_std_m):
    """Generate synthetic spacecraft positions on sphere (test helper)."""
    positions = np.zeros((n_measurements, 3))
    for i in range(n_measurements):
        radius = np.random.normal(radius_mean_m, radius_std_m)
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        positions[i] = [radius * sin_theta * np.cos(phi), radius * sin_theta * np.sin(phi), radius * cos_theta]
    return positions


def _generate_nadir_aligned_transforms(n_measurements, riss_ctrs, boresights_hs):
    """Generate transformation matrices aligning boresights with nadir (test helper)."""
    t_matrices = np.zeros((n_measurements, 3, 3))
    for i in range(n_measurements):
        nadir_ctrs = -riss_ctrs[i] / np.linalg.norm(riss_ctrs[i])
        bhat_hs_norm = boresights_hs[i] / np.linalg.norm(boresights_hs[i])
        rotation_axis = np.cross(bhat_hs_norm, nadir_ctrs)
        axis_norm = np.linalg.norm(rotation_axis)

        if axis_norm < 1e-6:
            if np.dot(bhat_hs_norm, nadir_ctrs) > 0:
                t_matrices[i] = np.eye(3)
            else:
                perp = np.array([1, 0, 0]) if abs(bhat_hs_norm[0]) < 0.9 else np.array([0, 1, 0])
                rotation_axis = np.cross(bhat_hs_norm, perp) / np.linalg.norm(np.cross(bhat_hs_norm, perp))
                K = np.array(
                    [
                        [0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0],
                    ]
                )
                t_matrices[i] = np.eye(3) + 2 * K @ K
        else:
            rotation_axis = rotation_axis / axis_norm
            angle = np.arccos(np.clip(np.dot(bhat_hs_norm, nadir_ctrs), -1.0, 1.0))
            K = np.array(
                [
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0],
                ]
            )
            t_matrices[i] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return t_matrices


# =============================================================================
# TEST MODE FUNCTIONS (Extracted from correction.py)
# =============================================================================
# These functions were moved from the core correction module to keep test-specific
# code separate from mission-agnostic core functionality.


def discover_test_image_match_cases(test_data_dir: Path, test_cases: list[str] | None = None) -> list[dict]:
    """
    Discover available image matching test cases.

    This function scans the test data directory for validated image matching
    test cases and returns metadata about available test files.

    Args:
        test_data_dir: Root directory for test data (tests/data/clarreo/image_match/)
        test_cases: Specific test cases to use (e.g., ['1', '2']) or None for all

    Returns:
        List of test case dictionaries with file paths and metadata
    """
    logger.info(f"Discovering image matching test cases in: {test_data_dir}")

    # Shared calibration files (same for all test cases)
    los_file = test_data_dir / "b_HS.mat"
    psf_file_unbinned = test_data_dir / "optical_PSF_675nm_upsampled.mat"
    psf_file_binned = test_data_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat"

    if not los_file.exists():
        raise FileNotFoundError(f"LOS vectors file not found: {los_file}")
    if not psf_file_unbinned.exists():
        raise FileNotFoundError(f"Optical PSF file not found: {psf_file_unbinned}")

    # Test case metadata (from test_image_match.py)
    test_case_metadata = {
        "1": {
            "name": "Dili",
            "gcp_file": "GCP12055Dili_resampled.mat",
            "ancil_file": "R_ISS_midframe_TestCase1.mat",
            "expected_error_km": (3.0, -3.0),  # (lat, lon)
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

    # Filter to requested test cases
    if test_cases is None:
        test_cases = sorted(test_case_metadata.keys())

    discovered_cases = []

    for case_id in test_cases:
        if case_id not in test_case_metadata:
            logger.warning(f"Test case '{case_id}' not found in metadata, skipping")
            continue

        metadata = test_case_metadata[case_id]
        case_dir = test_data_dir / case_id

        if not case_dir.is_dir():
            logger.warning(f"Test case directory not found: {case_dir}, skipping")
            continue

        # Add each subcase variant (a, b, c, d)
        for subcase in metadata["cases"]:
            subimage_file = case_dir / subcase["subimage"]
            gcp_file = case_dir / metadata["gcp_file"]
            ancil_file = case_dir / metadata["ancil_file"]
            psf_file = psf_file_binned if subcase["binned"] else psf_file_unbinned

            # Validate all files exist
            if not subimage_file.exists():
                logger.warning(f"Subimage file not found: {subimage_file}, skipping")
                continue
            if not gcp_file.exists():
                logger.warning(f"GCP file not found: {gcp_file}, skipping")
                continue
            if not ancil_file.exists():
                logger.warning(f"Ancillary file not found: {ancil_file}, skipping")
                continue

            discovered_cases.append(
                {
                    "case_id": case_id,
                    "case_name": metadata["name"],
                    "subcase_name": subcase["subimage"],
                    "subimage_file": subimage_file,
                    "gcp_file": gcp_file,
                    "ancil_file": ancil_file,
                    "los_file": los_file,
                    "psf_file": psf_file,
                    "expected_lat_error_km": metadata["expected_error_km"][0],
                    "expected_lon_error_km": metadata["expected_error_km"][1],
                    "binned": subcase["binned"],
                }
            )

    logger.info(f"Discovered {len(discovered_cases)} test case variants from {len(test_cases)} test case groups")
    for case in discovered_cases:
        logger.info(
            f"  - {case['case_id']}/{case['subcase_name']}: {case['case_name']}, "
            f"expected error=({case['expected_lat_error_km']:.1f}, {case['expected_lon_error_km']:.1f}) km"
        )

    return discovered_cases


def apply_error_variation_for_testing(
    base_result: xr.Dataset, param_idx: int, error_variation_percent: float = 3.0
) -> xr.Dataset:
    """
    Apply random variation to image matching results to simulate parameter effects.

    This is used in test mode to simulate how different parameter values would
    affect geolocation errors, without actually re-running image matching.

    Args:
        base_result: Original image matching result
        param_idx: Parameter set index (used as random seed)
        error_variation_percent: Percentage variation to apply (e.g., 3.0 = ±3%)

    Returns:
        New Dataset with varied error values
    """
    # Create copy
    output = base_result.copy(deep=True)

    # Set reproducible random seed based on param_idx
    np.random.seed(param_idx)

    # Generate variation factors (centered at 1.0, with specified percentage variation)
    variation_fraction = error_variation_percent / 100.0
    lat_factor = 1.0 + np.random.normal(0, variation_fraction)
    lon_factor = 1.0 + np.random.normal(0, variation_fraction)
    ccv_factor = 1.0 + np.random.normal(0, variation_fraction / 10.0)  # Smaller variation for correlation

    # Apply variations to error values
    original_lat_km = base_result.attrs["lat_error_km"]
    original_lon_km = base_result.attrs["lon_error_km"]
    original_ccv = base_result.attrs["correlation_ccv"]

    varied_lat_km = original_lat_km * lat_factor
    varied_lon_km = original_lon_km * lon_factor
    varied_ccv = np.clip(original_ccv * ccv_factor, 0.0, 1.0)  # Keep correlation in valid range

    # Update dataset values
    # Get GCP center latitude from multiple possible sources
    if "gcp_center_lat" in base_result.attrs:
        gcp_center_lat = base_result.attrs["gcp_center_lat"]
    elif "gcp_lat_deg" in base_result:
        gcp_center_lat = float(base_result["gcp_lat_deg"].values[0])
    else:
        # Fallback to a reasonable default (mid-latitude)
        logger.warning("GCP center latitude not found in dataset, using default 45.0°")
        gcp_center_lat = 45.0

    lat_error_deg = varied_lat_km / 111.0
    lon_radius_km = 6378.0 * np.cos(np.deg2rad(gcp_center_lat))
    lon_error_deg = varied_lon_km / (lon_radius_km * np.pi / 180.0)

    output["lat_error_deg"].values[0] = lat_error_deg
    output["lon_error_deg"].values[0] = lon_error_deg

    # Update attributes
    output.attrs["lat_error_km"] = varied_lat_km
    output.attrs["lon_error_km"] = varied_lon_km
    output.attrs["correlation_ccv"] = varied_ccv
    output.attrs["param_idx"] = param_idx
    output.attrs["variation_applied"] = True
    output.attrs["variation_lat_factor"] = lat_factor
    output.attrs["variation_lon_factor"] = lon_factor

    logger.info(
        f"  Applied variation: lat {original_lat_km:.3f} → {varied_lat_km:.3f} km ({(lat_factor - 1) * 100:+.1f}%), "
        f"lon {original_lon_km:.3f} → {varied_lon_km:.3f} km ({(lon_factor - 1) * 100:+.1f}%)"
    )

    return output


# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================


def apply_geolocation_error_to_subimage(
    subimage: ImageGrid, gcp: ImageGrid, lat_error_km: float, lon_error_km: float
) -> ImageGrid:
    """
    Apply artificial geolocation error to a subimage for testing.

    This creates a misaligned subimage that the image matching algorithm
    should detect and measure.
    """
    mid_lat = float(gcp.lat[gcp.lat.shape[0] // 2, gcp.lat.shape[1] // 2])

    # WGS84 Earth radius - use semi-major axis for latitude/longitude conversions
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
    """
    Run image matching with artificial errors applied to test data.

    This loads test data, applies known geolocation errors, then runs
    image matching to verify it can detect those errors.

    Args:
        test_case: Test case dictionary with file paths and expected errors
        param_idx: Parameter set index (for variation seed)
        randomize_errors: Whether to apply random variations
        error_variation_percent: Percentage variation (default 3.0%)
        cache_results: Whether to use cached results with variation
        cached_result: Previously cached result to vary
    """
    # Use cached result with variation if available
    if cached_result is not None and cache_results and param_idx > 0:
        if randomize_errors:
            logger.info(f"  Applying ±{error_variation_percent}% variation to cached result")
            return apply_error_variation_for_testing(cached_result, param_idx, error_variation_percent)
        else:
            return cached_result.copy()

    logger.info(f"  Running image matching with applied errors: {test_case['case_name']}")
    start_time = time.time()

    # Load subimage
    subimage_struct = loadmat(test_case["subimage_file"], squeeze_me=True, struct_as_record=False)["subimage"]
    subimage = ImageGrid(
        data=np.asarray(subimage_struct.data),
        lat=np.asarray(subimage_struct.lat),
        lon=np.asarray(subimage_struct.lon),
        h=np.asarray(subimage_struct.h) if hasattr(subimage_struct, "h") else None,
    )

    # Load GCP
    gcp = load_image_grid_from_mat(test_case["gcp_file"], key="GCP")
    gcp_center_lat = float(gcp.lat[gcp.lat.shape[0] // 2, gcp.lat.shape[1] // 2])
    gcp_center_lon = float(gcp.lon[gcp.lon.shape[0] // 2, gcp.lon.shape[1] // 2])

    # Apply expected error
    expected_lat_error = test_case["expected_lat_error_km"]
    expected_lon_error = test_case["expected_lon_error_km"]

    subimage_with_error = apply_geolocation_error_to_subimage(subimage, gcp, expected_lat_error, expected_lon_error)

    # Load calibration data
    los_vectors = load_los_vectors_from_mat(test_case["los_file"])
    optical_psfs = load_optical_psf_from_mat(test_case["psf_file"])
    ancil_data = loadmat(test_case["ancil_file"], squeeze_me=True)
    r_iss_midframe = ancil_data["R_ISS_midframe"].ravel()

    # Run image matching
    result = integrated_image_match(
        subimage=subimage_with_error,
        gcp=gcp,
        r_iss_midframe_m=r_iss_midframe,
        los_vectors_hs=los_vectors,
        optical_psfs=optical_psfs,
        geolocation_config=ImageMatchGeolocationConfig(),
        search_config=SearchConfig(),
    )

    processing_time = time.time() - start_time

    # Convert to dataset format
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


# =============================================================================
# CONFIGURATION GENERATION TEST
# =============================================================================


def test_generate_clarreo_config_json(tmp_path):
    """Generate CLARREO config JSON and validate structure.

    This test generates the canonical CLARREO configuration JSON file
    that is used by all other CLARREO tests. The generated JSON is
    saved to configs/ and can be committed for version control.

    This ensures:
    - Single source of truth for CLARREO configuration
    - Programmatic config matches JSON config
    - JSON structure is valid and complete
    """

    logger.info("=" * 80)
    logger.info("TEST: Generate CLARREO Configuration JSON")
    logger.info("=" * 80)

    # Define paths
    data_dir = Path(__file__).parent.parent / "data/clarreo/gcs"
    generic_dir = Path("data/generic")
    output_path = tmp_path / "configs/clarreo_correction_config.json"

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Generic kernels: {generic_dir}")
    logger.info(f"Output path: {output_path}")

    # Generate config programmatically
    logger.info("\n1. Generating config programmatically...")
    config = create_clarreo_correction_config(data_dir, generic_dir, config_output_path=output_path)

    logger.info(f"✓ Config created: {len(config.parameters)} parameter groups, {config.n_iterations} iterations")

    # Validate the generated JSON exists
    logger.info("\n2. Validating generated JSON file...")
    assert output_path.exists(), f"Config JSON not created: {output_path}"
    logger.info(f"✓ JSON file exists: {output_path}")

    # Reload and verify structure
    logger.info("\n3. Reloading and validating JSON structure...")
    with open(output_path) as f:
        config_data = json.load(f)

    # Validate top-level sections
    assert "mission_config" in config_data, "Missing 'mission_config' section"
    assert "correction" in config_data, "Missing 'correction' section"
    assert "geolocation" in config_data, "Missing 'geolocation' section"
    logger.info("✓ All required top-level sections present")

    # Validate mission config
    mission_cfg = config_data["mission_config"]
    assert mission_cfg["mission_name"] == "CLARREO_Pathfinder"
    assert "kernel_mappings" in mission_cfg
    logger.info(f"✓ Mission: {mission_cfg['mission_name']}")

    # Validate correction config
    corr_cfg = config_data["correction"]
    assert "parameters" in corr_cfg
    assert isinstance(corr_cfg["parameters"], list)
    assert len(corr_cfg["parameters"]) > 0
    assert "seed" in corr_cfg
    assert "n_iterations" in corr_cfg

    # NEW: Validate required fields are present
    assert "earth_radius_m" in corr_cfg, "Missing 'earth_radius_m' in correction config"
    assert "performance_threshold_m" in corr_cfg, "Missing 'performance_threshold_m'"
    assert "performance_spec_percent" in corr_cfg, "Missing 'performance_spec_percent'"

    assert corr_cfg["earth_radius_m"] == 6378140.0
    assert corr_cfg["performance_threshold_m"] == 250.0
    assert corr_cfg["performance_spec_percent"] == 39.0

    logger.info(f"✓ Correction config: {len(corr_cfg['parameters'])} parameters, {corr_cfg['n_iterations']} iterations")
    logger.info(
        f"✓ Required fields: earth_radius={corr_cfg['earth_radius_m']}, "
        f"threshold={corr_cfg['performance_threshold_m']}m, "
        f"spec={corr_cfg['performance_spec_percent']}%"
    )

    # Validate geolocation config
    geo_cfg = config_data["geolocation"]
    assert "meta_kernel_file" in geo_cfg
    assert "instrument_name" in geo_cfg
    assert geo_cfg["instrument_name"] == "CPRS_HYSICS"
    logger.info(f"✓ Geolocation config: instrument={geo_cfg['instrument_name']}")

    # Test that JSON can be loaded back into CorrectionConfig
    logger.info("\n4. Testing JSON → CorrectionConfig loading...")
    reloaded_config = correction.load_config_from_json(output_path)
    assert reloaded_config.n_iterations == config.n_iterations
    assert len(reloaded_config.parameters) == len(config.parameters)
    assert reloaded_config.earth_radius_m == 6378140.0
    assert reloaded_config.performance_threshold_m == 250.0
    assert reloaded_config.performance_spec_percent == 39.0
    logger.info("✓ JSON successfully loads into CorrectionConfig")

    # Validate reloaded config
    logger.info("\n5. Validating reloaded config...")
    reloaded_config.validate()
    logger.info("✓ Reloaded config passes validation")

    logger.info("\n" + "=" * 80)
    logger.info("✓ CONFIG GENERATION TEST PASSED")
    logger.info(f"✓ Canonical config saved: {output_path}")
    logger.info(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info("=" * 80)

    # Note: Test functions should not return values per pytest best practices
    # All validation is performed via assert statements above


# =============================================================================
# UPSTREAM TESTING (Kernel Creation + Geolocation)
# =============================================================================


def run_upstream_pipeline(n_iterations: int = 5, work_dir: Path | None = None) -> tuple[list, dict, Path]:
    """
    Test UPSTREAM segment of Correction pipeline.

    This tests:
    - Parameter set generation
    - Kernel creation from parameters
    - Geolocation with varied parameters

    This does NOT test:
    - GCP pairing (no valid pairs available)
    - Image matching (no valid data)
    - Error statistics (no matched data)

    This is focused on testing the upstream kernel creation
    and geolocation part of the pipeline.

    Args:
        n_iterations: Number of Correction iterations
        work_dir: Working directory for outputs. If None, uses a temporary
                  directory that will be cleaned up when the process exits.

    Returns:
        Tuple of (results, netcdf_data, output_path)
    """
    logger.info("=" * 80)
    logger.info("UPSTREAM PIPELINE TEST")
    logger.info("Tests: Kernel Creation + Geolocation")
    logger.info("=" * 80)

    root_dir = Path(__file__).parents[2]
    generic_dir = root_dir / "data" / "generic"
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"

    # Use temporary directory if work_dir not specified
    if work_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="curryer_upstream_")
        work_dir = Path(_tmp_dir)

        # Register cleanup to run on process exit
        def cleanup_temp_dir():
            if work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    logger.debug(f"Cleaned up temporary directory: {work_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {work_dir}: {e}")

        atexit.register(cleanup_temp_dir)

        logger.info(f"Using temporary directory: {work_dir}")
        logger.info("(will be cleaned up on process exit)")
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Iterations: {n_iterations}")

    # Create configuration using CLARREO config
    config = create_clarreo_correction_config(data_dir, generic_dir)
    config.n_iterations = n_iterations
    # Set output filename for test (consistent name for version control)
    config.output_filename = "upstream_results.nc"

    # Add loaders and processing functions to config (Config-Centric Design)
    config.telemetry_loader = load_clarreo_telemetry
    config.science_loader = load_clarreo_science
    config.gcp_loader = load_clarreo_gcp
    config.gcp_pairing_func = synthetic_gcp_pairing  # Test helper from this file
    config.image_matching_func = synthetic_image_matching  # Test helper from this file

    logger.info(f"Configuration loaded:")
    logger.info(f"  Mission: CLARREO Pathfinder")
    logger.info(f"  Instrument: {config.geo.instrument_name}")
    logger.info(f"  Parameters: {len(config.parameters)}")
    logger.info(f"  Iterations: {n_iterations}")
    logger.info(f"  Data loaders: telemetry, science, gcp")
    logger.info(f"  Processing: synthetic pairing and image matching")

    # Prepare data sets (synthetic GCP pairs since we don't have real data)
    # For upstream testing, we just need telemetry and science keys
    tlm_sci_gcp_sets = [
        ("telemetry_5a", "science_5a", "synthetic_gcp_1"),
    ]

    logger.info(f"Data sets: {len(tlm_sci_gcp_sets)} (synthetic for upstream testing)")

    # Execute the Correction loop - all config comes from config object!
    # This will test parameter generation, kernel creation, and geolocation
    logger.info("=" * 80)
    logger.info("EXECUTING CORRECTION UPSTREAM WORKFLOW")
    logger.info("=" * 80)

    results, netcdf_data = correction.loop(config, work_dir, tlm_sci_gcp_sets)

    logger.info("=" * 80)
    logger.info("UPSTREAM PIPELINE COMPLETE")
    logger.info(f"Processed {len(results)} total iterations")
    logger.info(f"Generated results for {len(netcdf_data['parameter_set_id'])} parameter sets")
    logger.info("=" * 80)

    # Output file is determined by config and saved by loop()
    output_file = work_dir / config.get_output_filename()
    logger.info(f"Results saved to: {output_file}")

    # Create results summary dict for consistency with downstream
    results_summary = {
        "mode": "upstream",
        "iterations": n_iterations,
        "parameter_sets": len(netcdf_data["parameter_set_id"]),
        "status": "complete",
    }

    return results, results_summary, output_file


# =============================================================================
# DOWNSTREAM TESTING (Pairing + Image Matching + Error Statistics)
# =============================================================================


def run_downstream_pipeline(
    n_iterations: int = 5, test_cases: list[str] | None = None, work_dir: Path | None = None
) -> tuple[list, dict, Path]:
    """
    Test DOWNSTREAM segment of Correction pipeline.

    IMPORTANT: This test uses a CUSTOM LOOP (not correction.loop()) because it works with
    pre-geolocated test data that doesn't have the telemetry/parameters needed for
    the normal upstream pipeline.

    Pipeline Comparison:
        Normal correction.loop():  Parameters → Kernels → Geolocation → Matching → Stats
        This test:         Pre-geolocated Test Data → Pairing → Matching → Stats

    Parameter effects are simulated by varying the geolocation errors directly
    (bumping lat/lon values), rather than varying parameters and re-running SPICE.
    This is the correct approach for testing with pre-geolocated imagery!

    Tests (Real):
        - GCP spatial pairing algorithms
        - Image matching with real correlation
        - Error statistics computation

    Does NOT Test (No Data Available):
        - Kernel creation (no telemetry)
        - SPICE geolocation (test data is pre-geolocated)
        - True parameter sensitivity (simulated via error variation)

    Args:
        n_iterations: Number of Correction iterations
        test_cases: Specific test cases to use (e.g., ['1', '2'])
        work_dir: Working directory for outputs. If None, uses a temporary
                  directory that will be cleaned up when the process exits.

    Returns:
        Tuple of (results, netcdf_data, output_path)
    """
    logger.info("=" * 80)
    logger.info("DOWNSTREAM PIPELINE TEST")
    logger.info("Tests: GCP Pairing + Image Matching + Error Statistics")
    logger.info("=" * 80)

    root_dir = Path(__file__).parents[2]
    generic_dir = root_dir / "data" / "generic"
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
    test_data_dir = root_dir / "tests" / "data" / "clarreo" / "image_match"

    # Use temporary directory if work_dir not specified
    if work_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="curryer_downstream_")
        work_dir = Path(_tmp_dir)

        # Register cleanup to run on process exit
        def cleanup_temp_dir():
            if work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    logger.debug(f"Cleaned up temporary directory: {work_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {work_dir}: {e}")

        atexit.register(cleanup_temp_dir)

        logger.info(f"Using temporary directory: {work_dir}")
        logger.info("(will be cleaned up on process exit)")
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Test data directory: {test_data_dir}")
    logger.info(f"Iterations: {n_iterations}")
    logger.info(f"Test cases: {test_cases or 'all'}")

    # Test configuration (simple parameters - no config object needed)
    randomize_errors = True
    error_variation_percent = 3.0
    cache_results = True

    # Discover test cases
    discovered_cases = discover_test_image_match_cases(test_data_dir, test_cases)
    logger.info(f"Discovered {len(discovered_cases)} test case variants")

    # ==========================================================================
    # STEP 1: GCP PAIRING
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: GCP SPATIAL PAIRING")
    logger.info("=" * 80)

    # Load L1A images
    l1a_images = []
    l1a_to_testcase = {}

    for test_case in discovered_cases:
        l1a_img = load_image_grid_from_mat(
            test_case["subimage_file"],
            key="subimage",
            as_named=True,
            name=str(test_case["subimage_file"].relative_to(test_data_dir)),
        )
        l1a_images.append(l1a_img)
        l1a_to_testcase[l1a_img.name] = test_case

    # Load unique GCP references
    gcp_files_seen = set()
    gcp_images = []

    for test_case in discovered_cases:
        gcp_file = test_case["gcp_file"]
        if gcp_file not in gcp_files_seen:
            gcp_img = load_image_grid_from_mat(
                gcp_file, key="GCP", as_named=True, name=str(gcp_file.relative_to(test_data_dir))
            )
            gcp_images.append(gcp_img)
            gcp_files_seen.add(gcp_file)

    logger.info(f"Loaded {len(l1a_images)} L1A images")
    logger.info(f"Loaded {len(gcp_images)} unique GCP references")

    # Run spatial pairing
    pairing_result = find_l1a_gcp_pairs(l1a_images, gcp_images, max_distance_m=0.0)
    logger.info(f"Pairing complete: Found {len(pairing_result.matches)} valid pairs")

    for match in pairing_result.matches:
        l1a_name = pairing_result.l1a_images[match.l1a_index].name
        gcp_name = pairing_result.gcp_images[match.gcp_index].name
        logger.info(f"  {l1a_name} ↔ {gcp_name} (distance: {match.distance_m:.1f}m)")

    # Convert to paired test cases
    paired_test_cases = []
    for match in pairing_result.matches:
        l1a_img = pairing_result.l1a_images[match.l1a_index]
        test_case = l1a_to_testcase[l1a_img.name]
        paired_test_cases.append(test_case)

    logger.info(f"Created {len(paired_test_cases)} paired test cases")

    # ==========================================================================
    # STEP 2: CONFIGURATION
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CONFIGURATION")
    logger.info("=" * 80)

    # Create base CLARREO config to get standard settings
    base_config = create_clarreo_correction_config(data_dir, generic_dir)

    # Create minimal test config (CorrectionConfig = THE one config)
    # For downstream testing, we use minimal parameters (sigma=0) because
    # variations come from test_mode_config randomization, not parameter tweaking
    config = correction.CorrectionConfig(
        # Core settings
        seed=42,
        n_iterations=n_iterations,
        # Minimal parameter (no real variation - sigma=0)
        parameters=[
            correction.ParameterConfig(
                ptype=correction.ParameterType.CONSTANT_KERNEL,
                config_file=data_dir / "cprs_hysics_v01.attitude.ck.json",
                data={
                    "current_value": [0.0, 0.0, 0.0],
                    "sigma": 0.0,  # No parameter variation (test variations applied differently)
                    "units": "arcseconds",
                    "transformation_type": "dcm_rotation",
                    "coordinate_frames": ["HYSICS_SLIT", "CRADLE_ELEVATION"],
                },
            )
        ],
        # Copy required fields from base_config
        geo=base_config.geo,
        performance_threshold_m=base_config.performance_threshold_m,
        performance_spec_percent=base_config.performance_spec_percent,
        earth_radius_m=base_config.earth_radius_m,
        # Copy optional fields from base_config
        netcdf=base_config.netcdf,
        calibration_file_names=base_config.calibration_file_names,
        spacecraft_position_name=base_config.spacecraft_position_name,
        boresight_name=base_config.boresight_name,
        transformation_matrix_name=base_config.transformation_matrix_name,
    )

    # Add loaders to config (Config-Centric Design)
    config.telemetry_loader = load_clarreo_telemetry
    config.science_loader = load_clarreo_science
    config.gcp_loader = load_clarreo_gcp

    # Validate complete config
    config.validate(check_loaders=True)

    logger.info(f"Configuration created:")
    logger.info(f"  Mission: CLARREO (from clarreo_config)")
    logger.info(f"  Instrument: {config.geo.instrument_name}")
    logger.info(f"  Iterations: {config.n_iterations}")
    logger.info(f"  Parameters: {len(config.parameters)} (minimal for test mode)")
    logger.info(f"  Sigma: 0.0 (variations from randomization, not parameters)")
    logger.info(f"  Performance threshold: {config.performance_threshold_m}m")

    # ==========================================================================
    # STEP 3: CORRECTION ITERATIONS
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CORRECTION ITERATIONS")
    logger.info("=" * 80)

    n_param_sets = n_iterations
    n_gcp_pairs = len(paired_test_cases)

    # Use dynamic NetCDF structure builder instead of hardcoded
    netcdf_data = correction._build_netcdf_structure(config, n_param_sets, n_gcp_pairs)
    logger.info(f"NetCDF structure built dynamically with {len(netcdf_data)} variables")

    # Get threshold metric name dynamically
    threshold_metric = config.netcdf.get_threshold_metric_name()
    logger.info(f"Using threshold metric: {threshold_metric}")

    image_match_cache = {}

    for param_idx in range(n_iterations):
        logger.info(f"\n=== Iteration {param_idx + 1}/{n_iterations} ===")

        image_matching_results = []
        pair_errors = []

        for pair_idx, test_case in enumerate(paired_test_cases):
            logger.info(f"Processing pair {pair_idx + 1}/{len(paired_test_cases)}: {test_case['case_name']}")

            cache_key = f"{test_case['case_id']}_{test_case.get('subcase_name', '')}"
            cached_result = image_match_cache.get(cache_key)

            # Run image matching
            image_matching_output = run_image_matching_with_applied_errors(
                test_case,
                param_idx,
                randomize_errors=randomize_errors,
                error_variation_percent=error_variation_percent,
                cache_results=cache_results,
                cached_result=cached_result,
            )

            # Cache first result
            if cache_key not in image_match_cache:
                image_match_cache[cache_key] = image_matching_output

            image_matching_output.attrs["gcp_pair_index"] = pair_idx
            image_matching_results.append(image_matching_output)

            # Extract and store metrics
            lat_error_m = abs(image_matching_output.attrs["lat_error_km"] * 1000)
            lon_error_m = abs(image_matching_output.attrs["lon_error_km"] * 1000)
            rms_error_m = np.sqrt(lat_error_m**2 + lon_error_m**2)
            pair_errors.append(rms_error_m)

            netcdf_data["rms_error_m"][param_idx, pair_idx] = rms_error_m
            netcdf_data["mean_error_m"][param_idx, pair_idx] = rms_error_m
            netcdf_data["max_error_m"][param_idx, pair_idx] = rms_error_m
            netcdf_data["n_measurements"][param_idx, pair_idx] = 1
            netcdf_data["im_lat_error_km"][param_idx, pair_idx] = image_matching_output.attrs["lat_error_km"]
            netcdf_data["im_lon_error_km"][param_idx, pair_idx] = image_matching_output.attrs["lon_error_km"]
            netcdf_data["im_ccv"][param_idx, pair_idx] = image_matching_output.attrs["correlation_ccv"]
            netcdf_data["im_grid_step_m"][param_idx, pair_idx] = image_matching_output.attrs["final_grid_step_m"]

        # Compute aggregate metrics (use dynamic threshold)
        pair_errors = np.array(pair_errors)
        valid_errors = pair_errors[~np.isnan(pair_errors)]

        if len(valid_errors) > 0:
            threshold_value = config.performance_threshold_m
            percent_under_threshold = (valid_errors < threshold_value).sum() / len(valid_errors) * 100
            netcdf_data[threshold_metric][param_idx] = percent_under_threshold
            netcdf_data["mean_rms_all_pairs"][param_idx] = np.mean(valid_errors)
            netcdf_data["best_pair_rms"][param_idx] = np.min(valid_errors)
            netcdf_data["worst_pair_rms"][param_idx] = np.max(valid_errors)

            logger.info(f"Iteration {param_idx + 1} complete:")
            logger.info(f"  {percent_under_threshold:.1f}% under {threshold_value}m threshold")
            logger.info(f"  Mean RMS: {np.mean(valid_errors):.2f}m")

    # ==========================================================================
    # STEP 4: ERROR STATISTICS
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: ERROR STATISTICS")
    logger.info("=" * 80)

    error_stats = correction.call_error_stats_module(image_matching_results, correction_config=config)
    logger.info(f"Error statistics computed: {len(error_stats)} metrics")

    # ==========================================================================
    # STEP 5: SAVE RESULTS
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: SAVE RESULTS")
    logger.info("=" * 80)

    output_file = work_dir / "downstream_results.nc"
    correction._save_netcdf_results(netcdf_data, output_file, config)

    logger.info("=" * 80)
    logger.info("DOWNSTREAM TEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output: {output_file}")
    logger.info(f"Iterations: {n_iterations}")
    logger.info(f"Test pairs: {n_gcp_pairs}")

    results = {"mode": "downstream", "iterations": n_iterations, "test_pairs": n_gcp_pairs, "status": "complete"}

    return [], results, output_file


# =============================================================================
# UNITTEST TEST CASES
# =============================================================================


class CorrectionUnifiedTests(unittest.TestCase):
    """Unified test cases for both upstream and downstream pipelines.

    Note: Only test_upstream_quick requires GMTED elevation data and is marked
    with @pytest.mark.extra. All other tests use either config-only validation
    or pre-geolocated test data and will run in CI without GMTED files.
    """

    def setUp(self):
        """Set up test environment."""
        self.root_dir = Path(__file__).parents[2]
        self.test_data_dir = self.root_dir / "tests" / "data" / "clarreo" / "image_match"

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.work_dir = Path(self.__tmp_dir.name)

    def test_upstream_configuration(self):
        """Test that upstream configuration loads correctly."""
        logger.info("Testing upstream configuration...")

        data_dir = self.root_dir / "tests" / "data" / "clarreo" / "gcs"
        generic_dir = self.root_dir / "data" / "generic"

        config = create_clarreo_correction_config(data_dir, generic_dir)

        # Add required loaders (Config-Centric Design)
        config.telemetry_loader = load_clarreo_telemetry
        config.science_loader = load_clarreo_science

        # Validate config is complete
        config.validate(check_loaders=True)

        self.assertEqual(config.geo.instrument_name, "CPRS_HYSICS")
        self.assertGreater(len(config.parameters), 0)
        self.assertEqual(config.seed, 42)
        self.assertIsNotNone(config.telemetry_loader)
        self.assertIsNotNone(config.science_loader)

        logger.info(f"✓ Configuration valid: {len(config.parameters)} parameters")

    def test_downstream_test_case_discovery(self):
        """Test that downstream test cases can be discovered."""
        logger.info("Testing test case discovery...")

        test_cases = discover_test_image_match_cases(self.test_data_dir)

        self.assertGreater(len(test_cases), 0, "No test cases discovered")

        for tc in test_cases:
            self.assertIn("case_id", tc)
            self.assertIn("subimage_file", tc)
            self.assertIn("gcp_file", tc)
            self.assertTrue(tc["subimage_file"].exists())
            self.assertTrue(tc["gcp_file"].exists())

        logger.info(f"✓ Discovered {len(test_cases)} test cases")

    def test_downstream_image_matching(self):
        """Test that downstream image matching works."""
        logger.info("Testing image matching...")

        test_cases = discover_test_image_match_cases(self.test_data_dir, test_cases=["1"])
        self.assertGreater(len(test_cases), 0)

        test_case = test_cases[0]

        result = run_image_matching_with_applied_errors(
            test_case, param_idx=0, randomize_errors=False, cache_results=True
        )

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("lat_error_km", result.attrs)
        self.assertIn("lon_error_km", result.attrs)

        logger.info(f"✓ Image matching successful")

    @pytest.mark.extra
    def test_upstream_quick(self):
        """Run quick upstream test.

        This test requires GMTED elevation data which is not available in CI.
        Run with: pytest --run-extra
        """
        logger.info("Running quick upstream test...")

        results_list, results_dict, output_file = run_upstream_pipeline(n_iterations=2, work_dir=self.work_dir)

        self.assertEqual(results_dict["status"], "complete")
        self.assertEqual(results_dict["iterations"], 2)
        self.assertEqual(results_dict["mode"], "upstream")
        self.assertGreater(results_dict["parameter_sets"], 0)
        self.assertTrue(output_file.exists())

        logger.info(f"✓ Quick upstream test complete: {output_file}")

    def test_downstream_quick(self):
        """Run quick downstream test."""
        logger.info("Running quick downstream test...")

        results_list, results_dict, output_file = run_downstream_pipeline(
            n_iterations=2, test_cases=["1"], work_dir=self.work_dir
        )

        self.assertEqual(results_dict["status"], "complete")
        self.assertEqual(results_dict["iterations"], 2)
        self.assertTrue(output_file.exists())

        logger.info(f"✓ Quick downstream test complete: {output_file}")

    def test_synthetic_helpers_basic(self):
        """Test that synthetic helper functions work correctly (for coverage)."""
        logger.info("Testing synthetic helper functions...")

        # Test synthetic GCP pairing
        science_files = ["science_1.nc", "science_2.nc"]
        pairs = synthetic_gcp_pairing(science_files)
        self.assertEqual(len(pairs), 2)
        self.assertIsInstance(pairs, list)
        logger.info("✓ synthetic_gcp_pairing works")

        # Test synthetic boresights generation
        boresights = _generate_synthetic_boresights(5, max_off_nadir_rad=0.07)
        self.assertEqual(boresights.shape, (5, 3))
        self.assertTrue(np.all(np.abs(boresights[:, 0]) < 0.01))  # Small x component
        logger.info("✓ _generate_synthetic_boresights works")

        # Test synthetic positions generation
        positions = _generate_spherical_positions(5, 6.78e6, 4e3)
        self.assertEqual(positions.shape, (5, 3))
        radii = np.linalg.norm(positions, axis=1)
        self.assertTrue(np.all(radii > 6.7e6))  # Reasonable orbit altitude
        logger.info("✓ _generate_spherical_positions works")

        # Test transform generation
        transforms = _generate_nadir_aligned_transforms(5, positions, boresights)
        self.assertEqual(transforms.shape, (5, 3, 3))
        # Check it's a valid rotation matrix (det should be close to 1)
        det = np.linalg.det(transforms[0])
        self.assertAlmostEqual(abs(det), 1.0, places=1)
        logger.info("✓ _generate_nadir_aligned_transforms works")

        logger.info("✓ All synthetic helpers validated")

    def test_downstream_helpers_basic(self):
        """Test downstream helper functions (for coverage)."""
        logger.info("Testing downstream helper functions...")

        # Test test case discovery
        test_cases = discover_test_image_match_cases(self.test_data_dir, test_cases=["1"])
        self.assertGreater(len(test_cases), 0)
        self.assertIn("case_id", test_cases[0])
        self.assertIn("subimage_file", test_cases[0])
        logger.info(f"✓ discover_test_image_match_cases found {len(test_cases)} cases")

        # Test error variation (create a simple test dataset)
        base_result = xr.Dataset(
            {
                "lat_error_deg": (["measurement"], [0.001]),
                "lon_error_deg": (["measurement"], [0.002]),
            },
            attrs={"lat_error_km": 0.1, "lon_error_km": 0.2, "correlation_ccv": 0.95},
        )

        varied_result = apply_error_variation_for_testing(base_result, param_idx=1, error_variation_percent=3.0)
        self.assertIsInstance(varied_result, xr.Dataset)
        self.assertIn("lat_error_deg", varied_result)
        # Check that variation was applied (should be different from base)
        self.assertNotEqual(varied_result.attrs["lat_error_km"], base_result.attrs["lat_error_km"])
        logger.info("✓ apply_error_variation_for_testing works")

        logger.info("✓ All downstream helpers validated")

    @pytest.mark.extra
    def test_loop_optimized(self):
        """
        Test loop() function (optimized pair-outer implementation).

        This test validates the main loop() function which is now the optimized
        default implementation. It covers the core Correction workflow.
        """
        logger.info("=" * 80)
        logger.info("TEST: loop() (OPTIMIZED IMPLEMENTATION)")
        logger.info("=" * 80)

        # Setup configuration
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / "data" / "generic"
        data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"

        config = create_clarreo_correction_config(data_dir, generic_dir)
        config.n_iterations = 2  # Small for fast testing
        config.output_filename = "test_loop_optimized.nc"

        # Add loaders and processing functions
        config.telemetry_loader = load_clarreo_telemetry
        config.science_loader = load_clarreo_science
        config.gcp_loader = load_clarreo_gcp
        config.gcp_pairing_func = synthetic_gcp_pairing
        config.image_matching_func = synthetic_image_matching

        # Prepare data sets
        tlm_sci_gcp_sets = [
            ("telemetry_5a", "science_5a", "synthetic_gcp_1"),
        ]

        work_dir = self.work_dir / "test_loop_optimized"
        work_dir.mkdir(exist_ok=True)

        # Run loop()
        logger.info("Running loop()...")
        np.random.seed(42)
        results, netcdf_data = correction.loop(config, work_dir, tlm_sci_gcp_sets, resume_from_checkpoint=False)

        # Validate results structure
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        expected_count = config.n_iterations * len(tlm_sci_gcp_sets)
        self.assertEqual(len(results), expected_count)
        logger.info(f"✓ loop() returned {len(results)} results")

        # Validate NetCDF structure
        self.assertIsInstance(netcdf_data, dict)
        self.assertIn("rms_error_m", netcdf_data)
        self.assertIn("parameter_set_id", netcdf_data)
        self.assertEqual(netcdf_data["rms_error_m"].shape, (config.n_iterations, len(tlm_sci_gcp_sets)))
        logger.info(f"✓ NetCDF structure valid")

        # Validate result contents
        for result in results:
            self.assertIn("param_index", result)
            self.assertIn("pair_index", result)
            self.assertIn("rms_error_m", result)
            self.assertIn("aggregate_rms_error_m", result)
            # Verify aggregate_rms_error_m is populated (not None)
            self.assertIsNotNone(
                result["aggregate_rms_error_m"],
                f"aggregate_rms_error_m should be populated for result {result['iteration']}",
            )
            # Verify it's a valid numeric value
            self.assertIsInstance(result["aggregate_rms_error_m"], (int, float, np.number))
        logger.info(f"✓ All result entries have required fields")

        logger.info("=" * 80)
        logger.info("✓ loop() TEST PASSED")
        logger.info("=" * 80)

    def test_helper_extract_parameter_values(self):
        """Test _extract_parameter_values helper function."""
        logger.info("Testing _extract_parameter_values...")

        # Create sample params with proper structure for CONSTANT_KERNEL type
        param_config = correction.ParameterConfig(
            ptype=correction.ParameterType.CONSTANT_KERNEL, config_file=Path("test_kernel.json"), data=None
        )
        # Create DataFrame with angle data as expected by the function
        param_data = pd.DataFrame(
            {
                "angle_x": [np.radians(1.0 / 3600)],  # 1 arcsec in radians
                "angle_y": [np.radians(2.0 / 3600)],  # 2 arcsec in radians
                "angle_z": [np.radians(3.0 / 3600)],  # 3 arcsec in radians
            }
        )
        params = [(param_config, param_data)]

        result = correction._extract_parameter_values(params)

        self.assertIsInstance(result, dict)
        # Should extract 3 values: roll, pitch, yaw
        self.assertEqual(len(result), 3)
        self.assertIn("test_kernel_roll", result)
        self.assertIn("test_kernel_pitch", result)
        self.assertIn("test_kernel_yaw", result)
        logger.info(f"✓ _extract_parameter_values works correctly")

    def test_helper_build_netcdf_structure(self):
        """Test _build_netcdf_structure helper function."""
        logger.info("Testing _build_netcdf_structure...")

        # Create minimal config
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / "data" / "generic"
        data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
        config = create_clarreo_correction_config(data_dir, generic_dir)

        n_params = 3
        n_pairs = 2

        netcdf_data = correction._build_netcdf_structure(config, n_params, n_pairs)

        # Validate structure
        self.assertIsInstance(netcdf_data, dict)
        self.assertIn("rms_error_m", netcdf_data)
        self.assertIn("parameter_set_id", netcdf_data)
        self.assertEqual(netcdf_data["rms_error_m"].shape, (n_params, n_pairs))
        self.assertEqual(len(netcdf_data["parameter_set_id"]), n_params)
        logger.info(f"✓ _build_netcdf_structure creates correct structure")

    def test_helper_extract_error_metrics(self):
        """Test _extract_error_metrics helper function."""
        logger.info("Testing _extract_error_metrics...")

        # Create sample error stats dataset with correct attribute name
        stats_dataset = xr.Dataset(
            {
                "measurement": (["point"], [0, 1, 2]),
                "lat_error_deg": (["point"], [0.001, 0.002, 0.001]),
                "lon_error_deg": (["point"], [0.001, 0.002, 0.001]),
            }
        )
        stats_dataset.attrs["rms_error_m"] = 150.0
        stats_dataset.attrs["mean_error_m"] = 140.0
        stats_dataset.attrs["max_error_m"] = 200.0
        stats_dataset.attrs["std_error_m"] = 10.0
        stats_dataset.attrs["total_measurements"] = 3  # Correct attribute name

        metrics = correction._extract_error_metrics(stats_dataset)

        # Validate metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn("rms_error_m", metrics)
        self.assertIn("mean_error_m", metrics)
        self.assertIn("n_measurements", metrics)
        self.assertEqual(metrics["n_measurements"], 3)
        self.assertEqual(metrics["rms_error_m"], 150.0)
        logger.info(f"✓ _extract_error_metrics extracts metrics correctly")

    def test_helper_store_parameter_values(self):
        """Test _store_parameter_values helper function."""
        logger.info("Testing _store_parameter_values...")

        # Create netcdf structure with parameter arrays pre-created
        # (as _build_netcdf_structure would do)
        netcdf_data = {
            "parameter_set_id": np.zeros(3, dtype=int),
            "param_test_param": np.zeros(3),  # Must match naming convention
        }

        param_values = {"test_param": 1.5}
        param_idx = 1

        correction._store_parameter_values(netcdf_data, param_idx, param_values)

        # Validate storage
        self.assertEqual(netcdf_data["param_test_param"][param_idx], 1.5)
        logger.info(f"✓ _store_parameter_values stores correctly")

    def test_helper_store_gcp_pair_results(self):
        """Test _store_gcp_pair_results helper function."""
        logger.info("Testing _store_gcp_pair_results...")

        # Create netcdf structure with all required fields
        netcdf_data = {
            "rms_error_m": np.zeros((2, 2)),
            "mean_error_m": np.zeros((2, 2)),
            "max_error_m": np.zeros((2, 2)),
            "std_error_m": np.zeros((2, 2)),  # Must include this
            "n_measurements": np.zeros((2, 2), dtype=int),
        }

        error_metrics = {
            "rms_error_m": 150.0,
            "mean_error_m": 140.0,
            "max_error_m": 200.0,
            "std_error_m": 10.0,  # Must include this
            "n_measurements": 10,
        }

        param_idx = 0
        pair_idx = 1

        correction._store_gcp_pair_results(netcdf_data, param_idx, pair_idx, error_metrics)

        # Validate storage
        self.assertEqual(netcdf_data["rms_error_m"][param_idx, pair_idx], 150.0)
        self.assertEqual(netcdf_data["std_error_m"][param_idx, pair_idx], 10.0)
        self.assertEqual(netcdf_data["n_measurements"][param_idx, pair_idx], 10)
        logger.info(f"✓ _store_gcp_pair_results stores correctly")

    def test_helper_compute_parameter_set_metrics(self):
        """Test _compute_parameter_set_metrics helper function."""
        logger.info("Testing _compute_parameter_set_metrics...")

        # Create netcdf structure
        netcdf_data = {
            "percent_under_250m": np.zeros(2),
            "mean_rms_all_pairs": np.zeros(2),
            "best_pair_rms": np.zeros(2),
            "worst_pair_rms": np.zeros(2),
        }

        pair_errors = [100.0, 200.0, 300.0]
        param_idx = 0
        threshold_m = 250.0

        correction._compute_parameter_set_metrics(netcdf_data, param_idx, pair_errors, threshold_m)

        # Validate computed metrics
        self.assertGreater(netcdf_data["percent_under_250m"][param_idx], 0)
        self.assertGreater(netcdf_data["mean_rms_all_pairs"][param_idx], 0)
        self.assertEqual(netcdf_data["best_pair_rms"][param_idx], 100.0)
        self.assertEqual(netcdf_data["worst_pair_rms"][param_idx], 300.0)
        logger.info(f"✓ _compute_parameter_set_metrics computes correctly")

    def test_helper_load_image_pair_data(self):
        """Test _load_image_pair_data helper function."""
        logger.info("Testing _load_image_pair_data...")

        # Setup configuration
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / "data" / "generic"
        data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
        config = create_clarreo_correction_config(data_dir, generic_dir)

        tlm_dataset, sci_dataset, ugps_times = correction._load_image_pair_data(
            "telemetry_5a", "science_5a", config, load_clarreo_telemetry, load_clarreo_science
        )

        # Validate return types
        self.assertIsInstance(tlm_dataset, pd.DataFrame)
        self.assertIsInstance(sci_dataset, pd.DataFrame)
        self.assertIsNotNone(ugps_times)
        logger.info(f"✓ _load_image_pair_data loads data correctly")

    def test_helper_create_dynamic_kernels(self):
        """Test _create_dynamic_kernels helper function."""
        logger.info("Testing _create_dynamic_kernels...")

        # Setup
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / "data" / "generic"
        data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
        config = create_clarreo_correction_config(data_dir, generic_dir)

        work_dir = self.work_dir / "test_dynamic_kernels"
        work_dir.mkdir(exist_ok=True)

        # Load data
        tlm_dataset = load_clarreo_telemetry("telemetry_5a", config)
        creator = create.KernelCreator(overwrite=True, append=False)

        # Load SPICE kernels needed for kernel creation
        # (frame kernel defines ISS_SC body which is needed by ephemeris writer)
        mkrn = meta.MetaKernel.from_json(
            config.geo.meta_kernel_file,
            relative=True,
            sds_dir=config.geo.generic_kernel_dir,
        )
        with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels]):
            dynamic_kernels = correction._create_dynamic_kernels(config, work_dir, tlm_dataset, creator)

        # Validate
        self.assertIsInstance(dynamic_kernels, list)
        logger.info(f"✓ _create_dynamic_kernels creates {len(dynamic_kernels)} kernels")

    def test_helper_load_calibration_data(self):
        """Test _load_calibration_data helper function."""
        logger.info("Testing _load_calibration_data...")

        # Setup minimal config without calibration dir
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / "data" / "generic"
        data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
        config = create_clarreo_correction_config(data_dir, generic_dir)
        config.calibration_dir = None

        # Load LOS vectors and PSF data into calibration data.
        calibration_data = correction._load_calibration_data(config)

        # Should return CalibrationData with None values when no calibration dir
        self.assertIsNone(calibration_data.los_vectors)
        self.assertIsNone(calibration_data.optical_psfs)
        logger.info(f"✓ _load_calibration_data handles None calibration_dir")

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load functionality."""
        logger.info("=" * 80)
        logger.info("TEST: Checkpoint Save/Load")
        logger.info("=" * 80)

        # Setup configuration
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / "data" / "generic"
        data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"

        config = create_clarreo_correction_config(data_dir, generic_dir)
        config.n_iterations = 2
        config.output_filename = "test_checkpoint.nc"

        # Add loaders
        config.telemetry_loader = load_clarreo_telemetry
        config.science_loader = load_clarreo_science
        config.gcp_loader = load_clarreo_gcp
        config.gcp_pairing_func = synthetic_gcp_pairing
        config.image_matching_func = synthetic_image_matching

        work_dir = self.work_dir / "test_checkpoint"
        work_dir.mkdir(exist_ok=True)

        output_file = work_dir / config.output_filename

        # Build simple netcdf structure for testing
        netcdf_data = correction._build_netcdf_structure(config, 2, 2)
        netcdf_data["rms_error_m"][0, 0] = 100.0
        netcdf_data["rms_error_m"][1, 0] = 150.0

        # Save checkpoint
        logger.info("Saving checkpoint...")
        correction._save_netcdf_checkpoint(netcdf_data, output_file, config, pair_idx_completed=0)

        checkpoint_file = output_file.parent / f"{output_file.stem}_checkpoint.nc"
        self.assertTrue(checkpoint_file.exists())
        logger.info(f"✓ Checkpoint file created: {checkpoint_file}")

        # Load checkpoint
        logger.info("Loading checkpoint...")
        loaded_data, completed_pairs = correction._load_checkpoint(output_file, config)

        self.assertIsNotNone(loaded_data)
        self.assertEqual(completed_pairs, 1)  # 0-indexed, so pair 0 completed = 1
        self.assertEqual(loaded_data["rms_error_m"][0, 0], 100.0)
        self.assertEqual(loaded_data["rms_error_m"][1, 0], 150.0)
        logger.info(f"✓ Checkpoint loaded correctly, completed pairs = {completed_pairs}")

        # Cleanup
        correction._cleanup_checkpoint(output_file)
        self.assertFalse(checkpoint_file.exists())
        logger.info(f"✓ Checkpoint cleanup successful")

        logger.info("=" * 80)
        logger.info("✓ CHECKPOINT SAVE/LOAD TEST PASSED")
        logger.info("=" * 80)

    def test_apply_offset_function(self):
        """Test apply_offset function for all parameter types."""
        logger.info("=" * 80)
        logger.info("TEST: apply_offset() Function")
        logger.info("=" * 80)

        # ========== Test 1: OFFSET_KERNEL with arcseconds ==========
        logger.info("\nTest 1: OFFSET_KERNEL with arcseconds unit conversion")

        # Create realistic telemetry data matching CLARREO structure
        telemetry_data = pd.DataFrame(
            {
                "frame": range(5),
                "hps.az_ang_nonlin": [1.14252] * 5,
                "hps.el_ang_nonlin": [-0.55009] * 5,
                "hps.resolver_tms": [1168477154.0 + i for i in range(5)],
                "ert": [1431903180.58 + i for i in range(5)],
            }
        )

        # Create parameter config for azimuth angle offset
        az_param_config = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=Path("cprs_az_v01.attitude.ck.json"),
            data=dict(
                field="hps.az_ang_nonlin",
                units="arcseconds",
            ),
        )

        # Apply offset of 100 arcseconds
        offset_arcsec = 100.0
        original_mean = telemetry_data["hps.az_ang_nonlin"].mean()
        modified_data = correction.apply_offset(az_param_config, offset_arcsec, telemetry_data)

        # Verify offset was applied correctly
        expected_offset_rad = np.deg2rad(offset_arcsec / 3600.0)
        actual_delta = modified_data["hps.az_ang_nonlin"].mean() - original_mean
        self.assertAlmostEqual(actual_delta, expected_offset_rad, places=9)
        self.assertIsInstance(modified_data, pd.DataFrame)
        logger.info(f"✓ OFFSET_KERNEL (arcseconds): {offset_arcsec} arcsec = {expected_offset_rad:.9f} rad")

        # ========== Test 2: OFFSET_KERNEL with elevation angle ==========
        logger.info("\nTest 2: OFFSET_KERNEL with elevation angle (negative offset)")

        el_param_config = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=Path("cprs_el_v01.attitude.ck.json"),
            data=dict(
                field="hps.el_ang_nonlin",
                units="arcseconds",
            ),
        )

        # Apply negative offset
        offset_arcsec = -50.0
        original_mean = telemetry_data["hps.el_ang_nonlin"].mean()
        modified_data = correction.apply_offset(el_param_config, offset_arcsec, telemetry_data)

        # Verify
        expected_offset_rad = np.deg2rad(offset_arcsec / 3600.0)
        actual_delta = modified_data["hps.el_ang_nonlin"].mean() - original_mean
        self.assertAlmostEqual(actual_delta, expected_offset_rad, places=9)
        logger.info(f"✓ OFFSET_KERNEL (negative): {offset_arcsec} arcsec = {expected_offset_rad:.9f} rad")

        # ========== Test 3: OFFSET_KERNEL with non-existent field ==========
        logger.info("\nTest 3: OFFSET_KERNEL with non-existent field (should warn)")

        bad_param_config = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=Path("dummy.json"),
            data=dict(
                field="nonexistent_field",
                units="arcseconds",
            ),
        )

        # Should return unmodified data when field not found
        modified_data = correction.apply_offset(bad_param_config, 10.0, telemetry_data)
        self.assertIsInstance(modified_data, pd.DataFrame)
        # Data should be unchanged
        pd.testing.assert_frame_equal(modified_data, telemetry_data)
        logger.info("✓ OFFSET_KERNEL correctly handles missing field")

        # ========== Test 4: OFFSET_TIME with milliseconds ==========
        logger.info("\nTest 4: OFFSET_TIME with milliseconds unit conversion")

        science_data = pd.DataFrame(
            {
                "corrected_timestamp": [1000000.0, 2000000.0, 3000000.0, 4000000.0, 5000000.0],
                "measurement": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        time_param_config = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_TIME,
            config_file=None,
            data=dict(
                field="corrected_timestamp",
                units="milliseconds",
            ),
        )

        # Apply time offset - value must be in seconds
        # This simulates the production flow where load_param_sets converts to seconds
        offset_ms = 10.0
        offset_seconds = offset_ms / 1000.0  # Convert to internal units (seconds)
        original_mean = science_data["corrected_timestamp"].mean()
        modified_data = correction.apply_offset(time_param_config, offset_seconds, science_data)

        # Verify offset was applied correctly (seconds -> microseconds)
        expected_offset_us = offset_ms * 1000.0  # 10 ms = 10000 µs
        actual_delta = modified_data["corrected_timestamp"].mean() - original_mean
        self.assertAlmostEqual(actual_delta, expected_offset_us, places=6)
        logger.info(f"✓ OFFSET_TIME: {offset_ms} ms = {expected_offset_us:.6f} µs")

        # ========== Test 5: OFFSET_TIME with negative offset ==========
        logger.info("\nTest 5: OFFSET_TIME with negative offset")

        offset_ms = -5.5
        offset_seconds = offset_ms / 1000.0  # Convert to internal units (seconds)
        original_mean = science_data["corrected_timestamp"].mean()
        modified_data = correction.apply_offset(time_param_config, offset_seconds, science_data)

        # Verify
        expected_offset_us = offset_ms * 1000.0
        actual_delta = modified_data["corrected_timestamp"].mean() - original_mean
        self.assertAlmostEqual(actual_delta, expected_offset_us, places=6)
        logger.info(f"✓ OFFSET_TIME (negative): {offset_ms} ms = {expected_offset_us:.6f} µs")

        # ========== Test 6: CONSTANT_KERNEL (pass-through) ==========
        logger.info("\nTest 6: CONSTANT_KERNEL (pass-through, no modification)")

        constant_kernel_data = pd.DataFrame(
            {
                "ugps": [1000000, 2000000],
                "angle_x": [0.001, 0.001],
                "angle_y": [0.002, 0.002],
                "angle_z": [0.003, 0.003],
            }
        )

        constant_param_config = correction.ParameterConfig(
            ptype=correction.ParameterType.CONSTANT_KERNEL,
            config_file=Path("cprs_base_v01.attitude.ck.json"),
            data=dict(
                field="cprs_base",
            ),
        )

        # For CONSTANT_KERNEL, param_data is already the kernel data
        modified_data = correction.apply_offset(constant_param_config, constant_kernel_data, pd.DataFrame())

        # Should return the constant kernel data unchanged
        self.assertIsInstance(modified_data, pd.DataFrame)
        pd.testing.assert_frame_equal(modified_data, constant_kernel_data)
        logger.info("✓ CONSTANT_KERNEL returns data unchanged")

        # ========== Test 7: OFFSET_KERNEL without units ==========
        logger.info("\nTest 7: OFFSET_KERNEL without unit conversion")

        param_no_units = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=Path("test.json"),
            data=dict(
                field="hps.az_ang_nonlin",
                # No units specified
            ),
        )

        # Apply offset directly (no conversion)
        offset_value = 0.001  # radians
        original_mean = telemetry_data["hps.az_ang_nonlin"].mean()
        modified_data = correction.apply_offset(param_no_units, offset_value, telemetry_data)

        # Verify offset applied directly without conversion
        actual_delta = modified_data["hps.az_ang_nonlin"].mean() - original_mean
        self.assertAlmostEqual(actual_delta, offset_value, places=9)
        logger.info(f"✓ OFFSET_KERNEL (no units): {offset_value} rad applied directly")

        # ========== Test 8: Data is not modified in place ==========
        logger.info("\nTest 8: Original data is not modified in place")

        original_telemetry = telemetry_data.copy()
        modified_data = correction.apply_offset(az_param_config, 100.0, telemetry_data)

        # Verify original data unchanged
        pd.testing.assert_frame_equal(telemetry_data, original_telemetry)
        # Verify modified data is different
        self.assertFalse(modified_data["hps.az_ang_nonlin"].equals(original_telemetry["hps.az_ang_nonlin"]))
        logger.info("✓ Original data not modified (proper copy made)")

        # ========== Test 9: Multiple columns preserved ==========
        logger.info("\nTest 9: All DataFrame columns preserved after offset")

        modified_data = correction.apply_offset(az_param_config, 50.0, telemetry_data)

        # Verify all columns still present
        self.assertEqual(set(modified_data.columns), set(telemetry_data.columns))
        # Verify only target column modified
        self.assertTrue(modified_data["frame"].equals(telemetry_data["frame"]))
        self.assertTrue(modified_data["ert"].equals(telemetry_data["ert"]))
        self.assertFalse(modified_data["hps.az_ang_nonlin"].equals(telemetry_data["hps.az_ang_nonlin"]))
        logger.info("✓ All DataFrame columns preserved, only target modified")

        logger.info("\n" + "=" * 80)
        logger.info("✓ apply_offset() TEST PASSED")
        logger.info("  - OFFSET_KERNEL with arcseconds conversion ✓")
        logger.info("  - OFFSET_KERNEL with negative values ✓")
        logger.info("  - OFFSET_KERNEL missing field handling ✓")
        logger.info("  - OFFSET_KERNEL without units ✓")
        logger.info("  - OFFSET_TIME with milliseconds conversion ✓")
        logger.info("  - OFFSET_TIME with negative values ✓")
        logger.info("  - CONSTANT_KERNEL pass-through ✓")
        logger.info("  - Data not modified in place ✓")
        logger.info("  - All columns preserved ✓")
        logger.info("=" * 80)

    def test_helper_load_param_sets(self):
        """Test load_param_sets function for parameter set generation."""
        logger.info("=" * 80)
        logger.info("TEST: load_param_sets() Function")
        logger.info("=" * 80)

        # Create minimal config with different parameter types
        data_dir = self.root_dir / "tests" / "data" / "clarreo" / "gcs"
        generic_dir = self.root_dir / "data" / "generic"

        # ========== Test 1: OFFSET_KERNEL parameter with arcseconds ==========
        logger.info("\nTest 1: OFFSET_KERNEL parameter generation with arcseconds")

        offset_kernel_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=Path("cprs_az_v01.attitude.ck.json"),
            data=dict(
                field="hps.az_ang_nonlin",
                units="arcseconds",
                current_value=0.0,  # Starting at zero
                sigma=100.0,  # ±100 arcseconds standard deviation
                bounds=[-200.0, 200.0],  # ±200 arcseconds limits
            ),
        )

        config = correction.CorrectionConfig(
            seed=42,  # For reproducibility
            n_iterations=3,
            parameters=[offset_kernel_param],
            geo=correction.GeolocationConfig(
                meta_kernel_file=data_dir / "meta_kernel.tm",
                generic_kernel_dir=generic_dir,
                dynamic_kernels=[],
                instrument_name="CPRS_HYSICS",
                time_field="corrected_timestamp",
            ),
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
            earth_radius_m=6378137.0,
        )

        param_sets = correction.load_param_sets(config)

        # Validate structure
        self.assertEqual(len(param_sets), 3, "Should generate 3 parameter sets")
        self.assertEqual(len(param_sets[0]), 1, "Each set should have 1 parameter")

        # Check each parameter set
        for i, param_set in enumerate(param_sets):
            param_config, param_value = param_set[0]
            self.assertEqual(param_config.ptype, correction.ParameterType.OFFSET_KERNEL)
            self.assertIsInstance(param_value, float, "OFFSET_KERNEL should produce float value")
            # Value should be in radians (converted from arcseconds)
            self.assertLess(abs(param_value), np.deg2rad(200.0 / 3600.0), "Value should be within bounds")
            logger.info(f"  Set {i}: {param_value:.9f} rad ({np.rad2deg(param_value) * 3600.0:.3f} arcsec)")

        logger.info("✓ OFFSET_KERNEL parameter generation works correctly")

        # ========== Test 2: OFFSET_TIME parameter with milliseconds ==========
        logger.info("\nTest 2: OFFSET_TIME parameter generation with milliseconds")

        offset_time_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_TIME,
            config_file=None,
            data=dict(
                field="corrected_timestamp",
                units="milliseconds",
                current_value=0.0,
                sigma=10.0,  # ±10 ms standard deviation
                bounds=[-50.0, 50.0],  # ±50 ms limits
            ),
        )

        config.parameters = [offset_time_param]
        config.n_iterations = 3

        param_sets = correction.load_param_sets(config)

        self.assertEqual(len(param_sets), 3)
        for i, param_set in enumerate(param_sets):
            param_config, param_value = param_set[0]
            self.assertEqual(param_config.ptype, correction.ParameterType.OFFSET_TIME)
            self.assertIsInstance(param_value, float, "OFFSET_TIME should produce float value")
            # Value should be in seconds (converted from milliseconds)
            self.assertLess(abs(param_value), 0.050, "Value should be within bounds (50 ms = 0.050 s)")
            logger.info(f"  Set {i}: {param_value:.6f} s ({param_value * 1000.0:.3f} ms)")

        logger.info("✓ OFFSET_TIME parameter generation works correctly")

        # ========== Test 3: CONSTANT_KERNEL parameter with 3D angles ==========
        logger.info("\nTest 3: CONSTANT_KERNEL parameter generation with 3D angles")

        constant_kernel_param = correction.ParameterConfig(
            ptype=correction.ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_base_v01.attitude.ck.json",
            data=dict(
                field="cprs_base",
                units="arcseconds",
                current_value=[0.0, 0.0, 0.0],  # [roll, pitch, yaw]
                sigma=50.0,  # ±50 arcseconds for each axis
                bounds=[-100.0, 100.0],  # ±100 arcseconds limits
            ),
        )

        config.parameters = [constant_kernel_param]
        config.n_iterations = 2

        param_sets = correction.load_param_sets(config)

        self.assertEqual(len(param_sets), 2)
        for i, param_set in enumerate(param_sets):
            param_config, param_value = param_set[0]
            self.assertEqual(param_config.ptype, correction.ParameterType.CONSTANT_KERNEL)
            self.assertIsInstance(param_value, pd.DataFrame, "CONSTANT_KERNEL should produce DataFrame")
            self.assertIn("angle_x", param_value.columns)
            self.assertIn("angle_y", param_value.columns)
            self.assertIn("angle_z", param_value.columns)
            self.assertIn("ugps", param_value.columns)

            # Check each angle is within bounds (in radians)
            max_bound_rad = np.deg2rad(100.0 / 3600.0)
            for angle_col in ["angle_x", "angle_y", "angle_z"]:
                angle_val = param_value[angle_col].iloc[0]
                self.assertLess(abs(angle_val), max_bound_rad, f"{angle_col} should be within bounds")

            logger.info(
                f"  Set {i}: roll={param_value['angle_x'].iloc[0]:.9f}, "
                f"pitch={param_value['angle_y'].iloc[0]:.9f}, "
                f"yaw={param_value['angle_z'].iloc[0]:.9f} rad"
            )

        logger.info("✓ CONSTANT_KERNEL parameter generation works correctly")

        # ========== Test 4: Multiple parameters together ==========
        logger.info("\nTest 4: Multiple parameters in single config")

        config.parameters = [offset_kernel_param, offset_time_param, constant_kernel_param]
        config.n_iterations = 2

        param_sets = correction.load_param_sets(config)

        self.assertEqual(len(param_sets), 2, "Should generate 2 parameter sets")
        self.assertEqual(len(param_sets[0]), 3, "Each set should have 3 parameters")

        # Verify each parameter type is present
        for i, param_set in enumerate(param_sets):
            types_found = [p[0].ptype for p in param_set]
            self.assertIn(correction.ParameterType.OFFSET_KERNEL, types_found)
            self.assertIn(correction.ParameterType.OFFSET_TIME, types_found)
            self.assertIn(correction.ParameterType.CONSTANT_KERNEL, types_found)
            logger.info(f"  Set {i}: Contains all 3 parameter types ✓")

        logger.info("✓ Multiple parameters handled correctly")

        # ========== Test 5: Fixed parameter (sigma=0) ==========
        logger.info("\nTest 5: Fixed parameter with sigma=0")

        fixed_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=Path("fixed.json"),
            data=dict(
                field="fixed_field",
                units="arcseconds",
                current_value=25.0,  # Fixed at 25 arcseconds
                sigma=0.0,  # No variation
                bounds=[-100.0, 100.0],
            ),
        )

        config.parameters = [fixed_param]
        config.n_iterations = 3

        param_sets = correction.load_param_sets(config)

        expected_value_rad = np.deg2rad(25.0 / 3600.0)
        for i, param_set in enumerate(param_sets):
            param_config, param_value = param_set[0]
            self.assertAlmostEqual(param_value, expected_value_rad, places=12, msg="Fixed parameter should not vary")
            logger.info(f"  Set {i}: {param_value:.9f} rad (constant)")

        logger.info("✓ Fixed parameter (sigma=0) works correctly")

        # ========== Test 6: Seed reproducibility ==========
        logger.info("\nTest 6: Random seed reproducibility")

        config.parameters = [offset_kernel_param]
        config.n_iterations = 3
        config.seed = 123

        param_sets_1 = correction.load_param_sets(config)

        # Reset and generate again with same seed
        config.seed = 123
        param_sets_2 = correction.load_param_sets(config)

        # Should produce identical values
        for i in range(len(param_sets_1)):
            val_1 = param_sets_1[i][0][1]
            val_2 = param_sets_2[i][0][1]
            self.assertAlmostEqual(val_1, val_2, places=12, msg=f"Set {i} should be identical with same seed")
            logger.info(f"  Set {i}: {val_1:.9f} rad (reproducible)")

        logger.info("✓ Random seed reproducibility verified")

        # ========== Test 7: Parameter without sigma (should use current_value) ==========
        logger.info("\nTest 7: Parameter without sigma field")

        no_sigma_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=Path("no_sigma.json"),
            data=dict(
                field="test_field",
                units="arcseconds",
                current_value=15.0,
                # No sigma specified
                bounds=[-100.0, 100.0],
            ),
        )

        config.parameters = [no_sigma_param]
        config.n_iterations = 3

        param_sets = correction.load_param_sets(config)

        expected_value_rad = np.deg2rad(15.0 / 3600.0)
        for i, param_set in enumerate(param_sets):
            param_config, param_value = param_set[0]
            self.assertAlmostEqual(
                param_value, expected_value_rad, places=12, msg="Parameter without sigma should use current_value"
            )

        logger.info("✓ Parameter without sigma uses current_value correctly")

        logger.info("\n" + "=" * 80)
        logger.info("✓ load_param_sets() TEST PASSED")
        logger.info("  - OFFSET_KERNEL generation ✓")
        logger.info("  - OFFSET_TIME generation ✓")
        logger.info("  - CONSTANT_KERNEL generation ✓")
        logger.info("  - Multiple parameters ✓")
        logger.info("  - Fixed parameters (sigma=0) ✓")
        logger.info("  - Seed reproducibility ✓")
        logger.info("  - Parameters without sigma ✓")
        logger.info("=" * 80)

    def test_offset_time_unit_conversion_integration(self):
        """Test the full integration of load_param_sets -> apply_offset for OFFSET_TIME with all unit types.

        This test verifies that:
        1. load_param_sets correctly converts milliseconds/microseconds -> seconds
        2. apply_offset correctly converts seconds -> microseconds for the timestamp field
        3. The end-to-end pipeline produces correct results
        4. All unit conversion paths are exercised (including uncovered lines)
        """
        logger.info("=" * 80)
        logger.info("TEST: OFFSET_TIME Unit Conversion Integration (load_param_sets -> apply_offset)")
        logger.info("=" * 80)

        data_dir = self.root_dir / "tests" / "data" / "clarreo" / "gcs"
        generic_dir = self.root_dir / "data" / "generic"

        # Test data with timestamps in microseconds (typical format)
        science_data = pd.DataFrame(
            {
                "corrected_timestamp": [1000000.0, 2000000.0, 3000000.0, 4000000.0, 5000000.0],
                "measurement": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        # ========== Test 1: Milliseconds with sigma ==========
        logger.info("\nTest 1: OFFSET_TIME with milliseconds unit (with sigma)")

        offset_time_ms_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_TIME,
            config_file=None,
            data=dict(
                field="corrected_timestamp",
                units="milliseconds",
                current_value=10.0,  # 10 milliseconds
                sigma=2.0,  # ±2 ms variation
                bounds=[-50.0, 50.0],  # ±50 ms limits
            ),
        )

        config = correction.CorrectionConfig(
            seed=42,
            n_iterations=1,
            parameters=[offset_time_ms_param],
            geo=correction.GeolocationConfig(
                meta_kernel_file=data_dir / "meta_kernel.tm",
                generic_kernel_dir=generic_dir,
                dynamic_kernels=[],
                instrument_name="CPRS_HYSICS",
                time_field="corrected_timestamp",
            ),
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
            earth_radius_m=6378137.0,
        )

        # Step 1: load_param_sets converts milliseconds -> seconds
        param_sets = correction.load_param_sets(config)
        self.assertEqual(len(param_sets), 1)
        param_config, param_value_seconds = param_sets[0][0]

        # Verify conversion to seconds
        self.assertLess(abs(param_value_seconds), 0.050, "Value should be in seconds, within 50ms bound")
        logger.info(f"  load_param_sets output: {param_value_seconds:.6f} s = {param_value_seconds * 1000.0:.3f} ms")

        # Step 2: apply_offset converts seconds -> microseconds
        original_mean = science_data["corrected_timestamp"].mean()
        modified_data = correction.apply_offset(param_config, param_value_seconds, science_data)

        # Verify the offset was applied correctly
        actual_delta_us = modified_data["corrected_timestamp"].mean() - original_mean
        expected_delta_us = param_value_seconds * 1000000.0  # seconds -> microseconds
        self.assertAlmostEqual(actual_delta_us, expected_delta_us, places=3)
        logger.info(f"  Expected delta: {expected_delta_us:.3f} µs")
        logger.info(f"  Actual delta:   {actual_delta_us:.3f} µs")
        logger.info("✓ Milliseconds path works correctly (load_param_sets -> apply_offset)")

        # ========== Test 2: Microseconds with sigma (covers lines 1442-1446) ==========
        logger.info("\nTest 2: OFFSET_TIME with microseconds unit (with sigma)")

        offset_time_us_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_TIME,
            config_file=None,
            data=dict(
                field="corrected_timestamp",
                units="microseconds",
                current_value=5000.0,  # 5000 microseconds = 5 ms
                sigma=1000.0,  # ±1000 µs variation
                bounds=[-10000.0, 10000.0],  # ±10000 µs limits
            ),
        )

        config.parameters = [offset_time_us_param]

        # Step 1: load_param_sets converts microseconds -> seconds
        param_sets = correction.load_param_sets(config)
        param_config, param_value_seconds = param_sets[0][0]

        # Verify conversion to seconds
        self.assertLess(abs(param_value_seconds), 0.010, "Value should be in seconds, within 10ms bound")
        logger.info(f"  load_param_sets output: {param_value_seconds:.6f} s = {param_value_seconds * 1000000.0:.1f} µs")

        # Step 2: apply_offset converts seconds -> microseconds
        original_mean = science_data["corrected_timestamp"].mean()
        modified_data = correction.apply_offset(param_config, param_value_seconds, science_data)

        # Verify the offset was applied correctly
        actual_delta_us = modified_data["corrected_timestamp"].mean() - original_mean
        expected_delta_us = param_value_seconds * 1000000.0
        self.assertAlmostEqual(actual_delta_us, expected_delta_us, places=3)
        logger.info(f"  Expected delta: {expected_delta_us:.3f} µs")
        logger.info(f"  Actual delta:   {actual_delta_us:.3f} µs")
        logger.info("✓ Microseconds path works correctly (load_param_sets -> apply_offset)")

        # ========== Test 3: Seconds unit (baseline) ==========
        logger.info("\nTest 3: OFFSET_TIME with seconds unit (baseline)")

        offset_time_s_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_TIME,
            config_file=None,
            data=dict(
                field="corrected_timestamp",
                units="seconds",
                current_value=0.008,  # 8 milliseconds
                sigma=0.002,  # ±2 ms variation
                bounds=[-0.050, 0.050],  # ±50 ms limits
            ),
        )

        config.parameters = [offset_time_s_param]

        # Step 1: load_param_sets (no conversion needed, already in seconds)
        param_sets = correction.load_param_sets(config)
        param_config, param_value_seconds = param_sets[0][0]

        self.assertLess(abs(param_value_seconds), 0.050, "Value should be in seconds")
        logger.info(f"  load_param_sets output: {param_value_seconds:.6f} s")

        # Step 2: apply_offset converts seconds -> microseconds
        original_mean = science_data["corrected_timestamp"].mean()
        modified_data = correction.apply_offset(param_config, param_value_seconds, science_data)

        # Verify the offset was applied correctly
        actual_delta_us = modified_data["corrected_timestamp"].mean() - original_mean
        expected_delta_us = param_value_seconds * 1000000.0
        self.assertAlmostEqual(actual_delta_us, expected_delta_us, places=3)
        logger.info(f"  Expected delta: {expected_delta_us:.3f} µs")
        logger.info(f"  Actual delta:   {actual_delta_us:.3f} µs")
        logger.info("✓ Seconds path works correctly (load_param_sets -> apply_offset)")

        # ========== Test 4: Fixed offset (sigma=0) with milliseconds ==========
        logger.info("\nTest 4: OFFSET_TIME fixed offset (sigma=0) with milliseconds")

        fixed_time_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_TIME,
            config_file=None,
            data=dict(
                field="corrected_timestamp",
                units="milliseconds",
                current_value=15.0,  # Fixed 15 ms
                sigma=0.0,  # No variation
                bounds=[-50.0, 50.0],
            ),
        )

        config.parameters = [fixed_time_param]
        config.n_iterations = 3

        # Generate multiple sets - all should be identical
        param_sets = correction.load_param_sets(config)
        self.assertEqual(len(param_sets), 3)

        expected_seconds = 15.0 / 1000.0  # 15 ms = 0.015 s
        for i, param_set in enumerate(param_sets):
            param_config, param_value_seconds = param_set[0]
            self.assertAlmostEqual(param_value_seconds, expected_seconds, places=9)
            logger.info(f"  Set {i}: {param_value_seconds:.6f} s (constant)")

        # Apply to data and verify
        original_mean = science_data["corrected_timestamp"].mean()
        modified_data = correction.apply_offset(param_config, expected_seconds, science_data)

        actual_delta_us = modified_data["corrected_timestamp"].mean() - original_mean
        expected_delta_us = 15000.0  # 15 ms = 15000 µs
        self.assertAlmostEqual(actual_delta_us, expected_delta_us, places=3)
        logger.info(f"  Applied offset: {actual_delta_us:.3f} µs (expected {expected_delta_us:.3f} µs)")
        logger.info("✓ Fixed offset with milliseconds works correctly")

        # ========== Test 5: Fixed offset (sigma=0) with microseconds ==========
        logger.info("\nTest 5: OFFSET_TIME fixed offset (sigma=0) with microseconds")

        fixed_time_us_param = correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_TIME,
            config_file=None,
            data=dict(
                field="corrected_timestamp",
                units="microseconds",
                current_value=7500.0,  # Fixed 7500 µs = 7.5 ms
                sigma=0.0,  # No variation
                bounds=[-50000.0, 50000.0],
            ),
        )

        config.parameters = [fixed_time_us_param]
        config.n_iterations = 2

        param_sets = correction.load_param_sets(config)
        expected_seconds = 7500.0 / 1000000.0  # 7500 µs = 0.0075 s

        for i, param_set in enumerate(param_sets):
            param_config, param_value_seconds = param_set[0]
            self.assertAlmostEqual(param_value_seconds, expected_seconds, places=9)
            logger.info(f"  Set {i}: {param_value_seconds:.6f} s (constant)")

        # Apply and verify
        original_mean = science_data["corrected_timestamp"].mean()
        modified_data = correction.apply_offset(param_config, expected_seconds, science_data)

        actual_delta_us = modified_data["corrected_timestamp"].mean() - original_mean
        expected_delta_us = 7500.0  # 7500 µs
        self.assertAlmostEqual(actual_delta_us, expected_delta_us, places=3)
        logger.info(f"  Applied offset: {actual_delta_us:.3f} µs (expected {expected_delta_us:.3f} µs)")
        logger.info("✓ Fixed offset with microseconds works correctly")

        # ========== Summary ==========
        logger.info("\n" + "=" * 80)
        logger.info("✓ OFFSET_TIME UNIT CONVERSION INTEGRATION TEST PASSED")
        logger.info("  - Integration: load_param_sets -> apply_offset ✓")
        logger.info("=" * 80)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Unified Correction Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
-----------
upstream     - Test kernel creation + geolocation
downstream   - Test pairing + matching + error statistics
unittest     - Run all unit tests

Examples:
---------
# Run downstream test
python test_correction.py --mode downstream --quick

# Run with specific test cases
python test_correction.py --mode downstream --test-cases 1 2 --iterations 10

# Run unit tests
python test_correction.py --mode unittest
pytest test_correction.py -v

# Run upstream test (when implemented)
python test_correction.py --mode upstream --iterations 5
        """,
    )

    parser.add_argument(
        "--mode", type=str, choices=["upstream", "downstream", "unittest"], required=True, help="Test mode to run"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test (2 iterations, test case 1 only)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of correction iterations (default: 5)")
    parser.add_argument(
        "--test-cases", nargs="+", default=None, help="Specific test cases for downstream mode (e.g., 1 2 3)"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    # Setup logging
    utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])

    if args.mode == "unittest":
        # Run unit tests
        unittest.main(argv=[""], exit=True)

    elif args.mode == "upstream":
        # Run upstream test
        if args.quick:
            n_iterations = 2
        else:
            n_iterations = args.iterations

        work_dir = Path(args.output_dir) if args.output_dir else None

        results_list, results_dict, output_file = run_upstream_pipeline(n_iterations=n_iterations, work_dir=work_dir)

        logger.info(f"\n✅ Upstream test complete!")
        logger.info(f"Status: {results_dict['status']}")
        logger.info(f"Iterations: {results_dict['iterations']}")
        logger.info(f"Parameter sets: {results_dict['parameter_sets']}")
        logger.info(f"Output file: {output_file}")

    elif args.mode == "downstream":
        # Run downstream test
        if args.quick:
            n_iterations = 2
            test_cases = ["1"]
        else:
            n_iterations = args.iterations
            test_cases = args.test_cases

        work_dir = Path(args.output_dir) if args.output_dir else None

        results_list, results_dict, output_file = run_downstream_pipeline(
            n_iterations=n_iterations, test_cases=test_cases, work_dir=work_dir
        )

        logger.info(f"\n✅ Downstream test complete!")
        logger.info(f"Output: {output_file}")

        # Validate results
        ds = xr.open_dataset(output_file)
        logger.info(f"NetCDF dimensions: {dict(ds.sizes)}")
        assert not np.all(np.isnan(ds["im_lat_error_km"].values)), "No data stored"
        logger.info("✅ Validation passed!")


if __name__ == "__main__":
    main()
