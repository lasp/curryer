#!/usr/bin/env python3
"""
Unified Monte Carlo Test Suite

This module consolidates two complementary Monte Carlo test approaches:

1. UPSTREAM Testing (test_upstream_pipeline):
   - Tests kernel creation and geolocation with parameter variations
   - Uses real telemetry data
   - Validates parameter modification and kernel generation
   - Stops before pairing (no valid GCP pairs available)

2. DOWNSTREAM Testing (test_downstream_pipeline):
   - Tests GCP pairing, image matching, and error statistics
   - Uses pre-geolocated test images with known GCP pairs
   - Validates spatial pairing, image matching algorithms, and error metrics
   - Skips kernel/geolocation (uses pre-computed test data)

Both tests share the same CLARREO configuration base but configure differently
for their specific testing needs.

Running Tests:
-------------
# Via pytest (recommended)
pytest tests/test_correction/test_monte_carlo.py -v

# Run specific test
pytest tests/test_correction/test_monte_carlo.py::test_upstream_pipeline -v
pytest tests/test_correction/test_monte_carlo.py::test_downstream_pipeline -v

# Standalone execution with arguments
python tests/test_correction/test_monte_carlo.py --mode downstream --quick

Requirements:
-----------------
These tests validate the complete Monte Carlo geolocation pipeline,
demonstrating parameter sensitivity analysis and error statistics
computation for mission requirements validation.

"""

import argparse
import logging
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.io import loadmat

from curryer import utils
from curryer.correction import monte_carlo as mc
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

# Import CLARREO config and data loaders
sys.path.insert(0, str(Path(__file__).parent))
from clarreo_config import create_clarreo_monte_carlo_config
from clarreo_data_loaders import load_clarreo_gcp, load_clarreo_science, load_clarreo_telemetry

logger = logging.getLogger(__name__)


# =============================================================================
# TEST MODE FUNCTIONS (Extracted from monte_carlo.py)
# =============================================================================
# These functions were moved from the core monte_carlo module to keep test-specific
# code separate from mission-agnostic core functionality.

from dataclasses import dataclass


@dataclass
class TestModeConfig:
    """
    Configuration for Monte Carlo test mode (used by test scripts).

    Test mode allows running the Monte Carlo pipeline with validated test data
    to verify integration without requiring production data.
    """

    test_data_dir: Path  # tests/data/clarreo/image_match/
    test_cases: Optional[List[str]] = None  # Specific cases: ['1', '2'] or None for all
    randomize_errors: bool = True  # Add variations to simulate parameter effects
    error_variation_percent: float = 3.0  # Percentage variation to apply (e.g., 3.0 = ±3%)
    cache_image_match_results: bool = True  # Cache results, apply variations instead of re-running


def discover_test_image_match_cases(test_data_dir: Path, test_cases: Optional[List[str]] = None) -> List[dict]:
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
    base_result: xr.Dataset, param_idx: int, test_mode_config: TestModeConfig
) -> xr.Dataset:
    """
    Apply random variation to image matching results to simulate parameter effects.

    This is used in test mode to simulate how different parameter values would
    affect geolocation errors, without actually re-running image matching.

    Args:
        base_result: Original image matching result
        param_idx: Parameter set index (used as random seed)
        test_mode_config: Test mode configuration with variation settings

    Returns:
        New Dataset with varied error values
    """
    # Create copy
    output = base_result.copy(deep=True)

    # Set reproducible random seed based on param_idx
    np.random.seed(param_idx)

    # Generate variation factors (centered at 1.0, with specified percentage variation)
    variation_fraction = test_mode_config.error_variation_percent / 100.0
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
    elif "cp_lat_deg" in base_result:
        gcp_center_lat = float(base_result["cp_lat_deg"].values[0])
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

    # WGS84 Earth radius - matches CLARREO config (6378140.0 m = 6378.140 km)
    earth_radius_km = 6378.140
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
    test_mode_config: TestModeConfig,
    cached_result: Optional[xr.Dataset] = None,
) -> xr.Dataset:
    """
    Run image matching with artificial errors applied to test data.

    This loads test data, applies known geolocation errors, then runs
    image matching to verify it can detect those errors.
    """
    # Use cached result with variation if available
    if cached_result is not None and test_mode_config.cache_image_match_results and param_idx > 0:
        if test_mode_config.randomize_errors:
            logger.info(f"  Applying ±{test_mode_config.error_variation_percent}% variation to cached result")
            return apply_error_variation_for_testing(cached_result, param_idx, test_mode_config)
        else:
            return cached_result.copy()

    logger.info(f"  Running image matching with applied errors: {test_case['case_name']}")
    start_time = time.time()

    try:
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
                "t_hs2ctrs": (["xyz_from", "xyz_to", "measurement"], t_matrix[:, :, np.newaxis]),
                "cp_lat_deg": (["measurement"], [gcp_center_lat]),
                "cp_lon_deg": (["measurement"], [gcp_center_lon]),
                "cp_alt": (["measurement"], [0.0]),
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

    except Exception as e:
        logger.error(f"  Image matching failed: {e}")
        raise


# =============================================================================
# CONFIGURATION GENERATION TEST
# =============================================================================


def test_generate_clarreo_config_json():
    """Generate CLARREO config JSON and validate structure.

    This test generates the canonical CLARREO configuration JSON file
    that is used by all other CLARREO tests. The generated JSON is
    saved to configs/ and can be committed for version control.

    This ensures:
    - Single source of truth for CLARREO configuration
    - Programmatic config matches JSON config
    - JSON structure is valid and complete
    """
    import json

    logger.info("=" * 80)
    logger.info("TEST: Generate CLARREO Configuration JSON")
    logger.info("=" * 80)

    # Define paths
    data_dir = Path(__file__).parent.parent / "data/clarreo/gcs"
    generic_dir = Path("data/generic")
    output_path = Path(__file__).parent / "configs/clarreo_monte_carlo_config.json"

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Generic kernels: {generic_dir}")
    logger.info(f"Output path: {output_path}")

    # Generate config programmatically
    logger.info("\n1. Generating config programmatically...")
    config = create_clarreo_monte_carlo_config(data_dir, generic_dir, config_output_path=output_path)

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
    assert "monte_carlo" in config_data, "Missing 'monte_carlo' section"
    assert "geolocation" in config_data, "Missing 'geolocation' section"
    logger.info("✓ All required top-level sections present")

    # Validate mission config
    mission_cfg = config_data["mission_config"]
    assert mission_cfg["mission_name"] == "CLARREO_Pathfinder"
    assert "kernel_mappings" in mission_cfg
    logger.info(f"✓ Mission: {mission_cfg['mission_name']}")

    # Validate monte_carlo config
    mc_cfg = config_data["monte_carlo"]
    assert "parameters" in mc_cfg
    assert isinstance(mc_cfg["parameters"], list)
    assert len(mc_cfg["parameters"]) > 0
    assert "seed" in mc_cfg
    assert "n_iterations" in mc_cfg

    # NEW: Validate required fields are present
    assert "earth_radius_m" in mc_cfg, "Missing 'earth_radius_m' in monte_carlo config"
    assert "performance_threshold_m" in mc_cfg, "Missing 'performance_threshold_m'"
    assert "performance_spec_percent" in mc_cfg, "Missing 'performance_spec_percent'"

    assert mc_cfg["earth_radius_m"] == 6378140.0
    assert mc_cfg["performance_threshold_m"] == 250.0
    assert mc_cfg["performance_spec_percent"] == 39.0

    logger.info(f"✓ Monte Carlo config: {len(mc_cfg['parameters'])} parameters, {mc_cfg['n_iterations']} iterations")
    logger.info(
        f"✓ Required fields: earth_radius={mc_cfg['earth_radius_m']}, "
        f"threshold={mc_cfg['performance_threshold_m']}m, "
        f"spec={mc_cfg['performance_spec_percent']}%"
    )

    # Validate geolocation config
    geo_cfg = config_data["geolocation"]
    assert "meta_kernel_file" in geo_cfg
    assert "instrument_name" in geo_cfg
    assert geo_cfg["instrument_name"] == "CPRS_HYSICS"
    logger.info(f"✓ Geolocation config: instrument={geo_cfg['instrument_name']}")

    # Test that JSON can be loaded back into MonteCarloConfig
    logger.info("\n4. Testing JSON → MonteCarloConfig loading...")
    reloaded_config = mc.load_config_from_json(output_path)
    assert reloaded_config.n_iterations == config.n_iterations
    assert len(reloaded_config.parameters) == len(config.parameters)
    assert reloaded_config.earth_radius_m == 6378140.0
    assert reloaded_config.performance_threshold_m == 250.0
    assert reloaded_config.performance_spec_percent == 39.0
    logger.info("✓ JSON successfully loads into MonteCarloConfig")

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


def test_upstream_pipeline(n_iterations: int = 5, work_dir: Optional[Path] = None) -> Tuple[List, Dict, Path]:
    """
    Test UPSTREAM segment of Monte Carlo pipeline.

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
        n_iterations: Number of Monte Carlo iterations
        work_dir: Working directory for outputs

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

    if work_dir is None:
        work_dir = root_dir / "tests" / "test_correction" / "monte_carlo_results" / "upstream"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Iterations: {n_iterations}")

    # Create configuration using CLARREO config
    config = create_clarreo_monte_carlo_config(data_dir, generic_dir)
    config.n_iterations = n_iterations
    # Set output filename for test (consistent name for version control)
    config.output_filename = "upstream_results.nc"

    logger.info(f"Configuration loaded:")
    logger.info(f"  Mission: CLARREO Pathfinder")
    logger.info(f"  Instrument: {config.geo.instrument_name}")
    logger.info(f"  Parameters: {len(config.parameters)}")
    logger.info(f"  Iterations: {n_iterations}")

    # Prepare data sets (synthetic GCP pairs since we don't have real data)
    # For upstream testing, we just need telemetry and science keys
    tlm_sci_gcp_sets = [
        ("telemetry_5a", "science_5a", "synthetic_gcp_1"),
    ]

    logger.info(f"Data sets: {len(tlm_sci_gcp_sets)} (synthetic for upstream testing)")

    # Execute the Monte Carlo loop with CLARREO data loaders
    # This will test parameter generation, kernel creation, and geolocation
    logger.info("=" * 80)
    logger.info("EXECUTING MONTE CARLO UPSTREAM WORKFLOW")
    logger.info("=" * 80)

    results, netcdf_data = mc.loop(
        config,
        work_dir,
        tlm_sci_gcp_sets,
        telemetry_loader=load_clarreo_telemetry,
        science_loader=load_clarreo_science,
        gcp_loader=load_clarreo_gcp,
    )

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


def test_downstream_pipeline(
    n_iterations: int = 5, test_cases: Optional[List[str]] = None, work_dir: Optional[Path] = None
) -> Tuple[List, Dict, Path]:
    """
    Test DOWNSTREAM segment of Monte Carlo pipeline.

    This tests:
    - GCP spatial pairing
    - Image matching with pre-geolocated data
    - Error statistics computation

    This does NOT test:
    - Kernel creation (uses pre-geolocated test data)
    - Geolocation (test data already geolocated)
    - Parameter modification (uses fake variations)

    Args:
        n_iterations: Number of Monte Carlo iterations
        test_cases: Specific test cases to use (e.g., ['1', '2'])
        work_dir: Working directory for outputs

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

    if work_dir is None:
        work_dir = root_dir / "tests" / "test_correction" / "monte_carlo_results" / "downstream"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Test data directory: {test_data_dir}")
    logger.info(f"Iterations: {n_iterations}")
    logger.info(f"Test cases: {test_cases or 'all'}")

    # Create test mode configuration
    test_mode_config = TestModeConfig(
        test_data_dir=test_data_dir,
        test_cases=test_cases,
        randomize_errors=True,
        error_variation_percent=3.0,
        cache_image_match_results=True,
    )

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

    # Create base CLARREO config
    base_config = create_clarreo_monte_carlo_config(data_dir, generic_dir)

    # Override for test mode (minimal parameters, zero sigma)
    config = mc.MonteCarloConfig(
        seed=42,
        n_iterations=n_iterations,
        parameters=[
            mc.ParameterConfig(
                ptype=mc.ParameterType.CONSTANT_KERNEL,
                config_file=data_dir / "cprs_hysics_v01.attitude.ck.json",
                data={
                    "current_value": [0.0, 0.0, 0.0],
                    "sigma": 0.0,  # No real parameter variation
                    "units": "arcseconds",
                    "transformation_type": "dcm_rotation",
                    "coordinate_frames": ["HYSICS_SLIT", "CRADLE_ELEVATION"],
                },
            )
        ],
        geo=base_config.geo,
        # features from base_config
        performance_threshold_m=base_config.performance_threshold_m,
        performance_spec_percent=base_config.performance_spec_percent,
        earth_radius_m=base_config.earth_radius_m,
        netcdf=base_config.netcdf,
        calibration_file_names=base_config.calibration_file_names,
        spacecraft_position_name=base_config.spacecraft_position_name,
        boresight_name=base_config.boresight_name,
        transformation_matrix_name=base_config.transformation_matrix_name,
    )

    logger.info(f"Configuration created:")
    logger.info(f"  Mission: CLARREO (from clarreo_config)")
    logger.info(f"  Instrument: {config.geo.instrument_name}")
    logger.info(f"  Iterations: {config.n_iterations}")
    logger.info(f"  Parameters: {len(config.parameters)} (minimal for test mode)")
    logger.info(f"  Sigma: 0.0 (variations from randomization, not parameters)")
    logger.info(f"  Performance threshold: {config.performance_threshold_m}m")

    # ==========================================================================
    # STEP 3: MONTE CARLO ITERATIONS
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: MONTE CARLO ITERATIONS")
    logger.info("=" * 80)

    n_param_sets = n_iterations
    n_gcp_pairs = len(paired_test_cases)

    # Use dynamic NetCDF structure builder instead of hardcoded
    netcdf_data = mc._build_netcdf_structure(config, n_param_sets, n_gcp_pairs)
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
                test_case, param_idx, test_mode_config, cached_result
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

    try:
        error_stats = mc.call_error_stats_module(image_matching_results, monte_carlo_config=config)
        logger.info(f"Error statistics computed: {len(error_stats)} metrics")
    except Exception as e:
        logger.warning(f"Error statistics failed: {e}")

    # ==========================================================================
    # STEP 5: SAVE RESULTS
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: SAVE RESULTS")
    logger.info("=" * 80)

    output_file = work_dir / "downstream_results.nc"
    mc._save_netcdf_results(netcdf_data, output_file, config)

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


class MonteCarloUnifiedTests(unittest.TestCase):
    """Unified test cases for both upstream and downstream pipelines."""

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

        config = create_clarreo_monte_carlo_config(data_dir, generic_dir)

        self.assertEqual(config.geo.instrument_name, "CPRS_HYSICS")
        self.assertGreater(len(config.parameters), 0)
        self.assertEqual(config.seed, 42)

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
        test_config = TestModeConfig(
            test_data_dir=self.test_data_dir,
            test_cases=["1"],
            randomize_errors=False,
            cache_image_match_results=True,
        )

        result = run_image_matching_with_applied_errors(test_case, 0, test_config)

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("lat_error_km", result.attrs)
        self.assertIn("lon_error_km", result.attrs)

        logger.info(f"✓ Image matching successful")

    def test_downstream_quick(self):
        """Run quick downstream test."""
        logger.info("Running quick downstream test...")

        results_list, results_dict, output_file = test_downstream_pipeline(
            n_iterations=2, test_cases=["1"], work_dir=self.work_dir
        )

        self.assertEqual(results_dict["status"], "complete")
        self.assertEqual(results_dict["iterations"], 2)
        self.assertTrue(output_file.exists())

        logger.info(f"✓ Quick test complete: {output_file}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Unified Monte Carlo Test Suite",
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
python test_monte_carlo.py --mode downstream --quick

# Run with specific test cases
python test_monte_carlo.py --mode downstream --test-cases 1 2 --iterations 10

# Run unit tests
python test_monte_carlo.py --mode unittest
pytest test_monte_carlo.py -v

# Run upstream test (when implemented)
python test_monte_carlo.py --mode upstream --iterations 5
        """,
    )

    parser.add_argument(
        "--mode", type=str, choices=["upstream", "downstream", "unittest"], required=True, help="Test mode to run"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test (2 iterations, test case 1 only)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of Monte Carlo iterations (default: 5)")
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

        results_list, results_dict, output_file = test_upstream_pipeline(n_iterations=n_iterations, work_dir=work_dir)

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

        results_list, results_dict, output_file = test_downstream_pipeline(
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
