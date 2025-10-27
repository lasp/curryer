"""
Monte Carlo Pipeline Integration Test

This test script validates the complete Monte Carlo GCS pipeline using
validated test data from tests/data/clarreo/image_match/.

TEST SCOPE vs run_monte_carlo.py:
--------------------------------
This test focuses on:
- Real image matching with pre-geolocated test data
- Real GCP pairing with spatial matching
- Error statistics processing
- NetCDF output generation
- Parameter variation simulation (fake variations via random offsets)

SKIPPED STEPS (not needed for this test):
- Kernel creation (test data already has geometry)
- Geolocation (test data is pre-geolocated)
- Parameter modification (uses random error injection instead)

run_monte_carlo.py focuses on:
- Real kernel creation
- Real geolocation with parameter variations
- Full end-to-end Monte Carlo workflow

CONFIGURATION:
-------------
Both tests now use the clarreo_config module for consistency, but this test:
- Overrides n_iterations based on test arguments
- Sets sigma=0 for parameters (no real parameter variation)
- Uses minimal parameter set (only 1 for structure/metadata)
- Variations come from randomized image matching errors, not parameters

Usage:
    # Run all tests
    pytest tests/test_correction/test_monte_carlo_pipeline.py -v

    # Run standalone (full integration test)
    python tests/test_correction/test_monte_carlo_pipeline.py

    # Run quick test (fewer iterations)
    python tests/test_correction/test_monte_carlo_pipeline.py --quick
"""

import argparse
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from scipy.io import loadmat

from curryer import utils
from curryer.correction import monte_carlo as mc
from curryer.correction.data_structures import ImageGrid, NamedImageGrid
from curryer.correction.image_match import (
    integrated_image_match,
    load_image_grid_from_mat,
    load_los_vectors_from_mat,
    load_optical_psf_from_mat,
)
from curryer.correction.data_structures import (
    GeolocationConfig as ImageMatchGeolocationConfig,
    SearchConfig,
)
from curryer.correction.pairing import find_l1a_gcp_pairs

# Add tests directory to path for config module import
sys.path.insert(0, str(Path(__file__).parent))
from clarreo_config import create_clarreo_monte_carlo_config

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])


def apply_geolocation_error_to_subimage(
    subimage: ImageGrid,
    gcp: ImageGrid,
    lat_error_km: float,
    lon_error_km: float
) -> ImageGrid:
    """
    Apply artificial geolocation error to a subimage for testing.

    This function is used in tests to inject known errors into well-aligned
    test data, allowing validation that the image matching algorithm can
    correctly detect and measure those errors.

    Args:
        subimage: The subimage to apply error to
        gcp: Ground control point grid (used to get reference latitude)
        lat_error_km: Latitude error to apply in kilometers (north positive)
        lon_error_km: Longitude error to apply in kilometers (east positive)

    Returns:
        New ImageGrid with shifted lat/lon coordinates
    """
    # Get reference latitude for accurate km-to-degree conversion
    mid_lat = float(gcp.lat[gcp.lat.shape[0] // 2, gcp.lat.shape[1] // 2])

    # Convert km to degrees using Earth radius and local curvature
    earth_radius_km = 6378.137  # WGS84 semi-major axis
    lat_offset_deg = lat_error_km / earth_radius_km * (180.0 / np.pi)
    lon_radius_km = earth_radius_km * np.cos(np.deg2rad(mid_lat))
    lon_offset_deg = lon_error_km / lon_radius_km * (180.0 / np.pi)

    return ImageGrid(
        data=subimage.data.copy(),
        lat=subimage.lat + lat_offset_deg,
        lon=subimage.lon + lon_offset_deg,
        h=subimage.h.copy() if subimage.h is not None else None,
    )


def run_test_mode_image_matching_with_applied_errors(
    test_case: dict,
    param_idx: int,
    test_mode_config: mc.TestModeConfig,
    cached_result: Optional[xr.Dataset] = None,
) -> xr.Dataset:
    """
    Test-specific wrapper that applies artificial errors before running image matching.

    This function loads test data, applies the expected geolocation errors to create
    a misaligned subimage, then runs the image matching algorithm to verify it can
    correctly detect and measure those errors.

    This is ONLY for testing - production Monte Carlo code should not inject errors.

    Args:
        test_case: Test case dictionary from discover_test_image_match_cases()
        param_idx: Current parameter set index
        test_mode_config: Test mode configuration
        cached_result: Previously computed result (for variation testing)

    Returns:
        xarray.Dataset with error measurements
    """
    # If we have a cached result and should apply variations
    if cached_result is not None and test_mode_config.cache_image_match_results and param_idx > 0:
        if test_mode_config.randomize_errors:
            logger.info(f"Image Matching: Applying ±{test_mode_config.error_variation_percent}% variation to cached result")
            # Use the Monte Carlo module's variation function
            return mc._apply_error_variation(cached_result, param_idx, test_mode_config)
        else:
            logger.info(f"Image Matching: Using cached result without variation")
            return cached_result.copy()

    # Run real image matching with applied errors
    logger.info(f"Image Matching: TEST MODE (with applied errors) - {test_case['case_name']} ({test_case['subcase_name']})")

    import time
    start_time = time.time()

    try:
        # 1. Load subimage (pre-geolocated test data)
        subimage_struct = loadmat(test_case['subimage_file'], squeeze_me=True, struct_as_record=False)["subimage"]
        subimage = ImageGrid(
            data=np.asarray(subimage_struct.data),
            lat=np.asarray(subimage_struct.lat),
            lon=np.asarray(subimage_struct.lon),
            h=np.asarray(subimage_struct.h) if hasattr(subimage_struct, "h") else None,
        )

        # 2. Load GCP reference
        gcp = load_image_grid_from_mat(test_case['gcp_file'], key="GCP")
        # Get GCP center location (center pixel)
        gcp_center_lat = float(gcp.lat[gcp.lat.shape[0] // 2, gcp.lat.shape[1] // 2])
        gcp_center_lon = float(gcp.lon[gcp.lon.shape[0] // 2, gcp.lon.shape[1] // 2])

        # 3. **APPLY EXPECTED GEOLOCATION ERROR** (test-specific step)
        # This creates a misaligned subimage that the algorithm should detect
        expected_lat_error = test_case['expected_lat_error_km']
        expected_lon_error = test_case['expected_lon_error_km']

        logger.info(f"  Applying artificial error: lat={expected_lat_error:.3f} km, lon={expected_lon_error:.3f} km")
        subimage_with_error = apply_geolocation_error_to_subimage(
            subimage, gcp, expected_lat_error, expected_lon_error
        )

        # 4. Load calibration data
        los_vectors = load_los_vectors_from_mat(test_case['los_file'])
        optical_psfs = load_optical_psf_from_mat(test_case['psf_file'])

        # 5. Load spacecraft position
        ancil_data = loadmat(test_case['ancil_file'], squeeze_me=True)
        r_iss_midframe = ancil_data["R_ISS_midframe"].ravel()

        # 6. Run real image matching with the error-injected subimage
        result = integrated_image_match(
            subimage=subimage_with_error,  # ← Using error-injected data
            gcp=gcp,
            r_iss_midframe_m=r_iss_midframe,
            los_vectors_hs=los_vectors,
            optical_psfs=optical_psfs,
            geolocation_config=ImageMatchGeolocationConfig(),
            search_config=SearchConfig(),
        )

        # 7. Convert to expected output format
        lat_error_deg = result.lat_error_km / 111.0
        lon_radius_km = 6378.0 * np.cos(np.deg2rad(gcp_center_lat))
        lon_error_deg = result.lon_error_km / (lon_radius_km * np.pi / 180.0)

        processing_time = time.time() - start_time

        # Log validation against expected errors
        lat_diff = abs(result.lat_error_km - expected_lat_error)
        lon_diff = abs(result.lon_error_km - expected_lon_error)

        logger.info(f"  Image matching complete in {processing_time:.2f}s:")
        logger.info(f"    Lat error: {result.lat_error_km:.3f} km (expected: {expected_lat_error:.3f}, diff: {lat_diff:.3f})")
        logger.info(f"    Lon error: {result.lon_error_km:.3f} km (expected: {expected_lon_error:.3f}, diff: {lon_diff:.3f})")
        logger.info(f"    Correlation: {result.ccv_final:.4f}")

        # 8. Create output dataset (matching Monte Carlo format)
        t_matrix = np.array([
            [-0.418977524967338, 0.748005379751721, 0.514728846515064],
            [-0.421890284446342, 0.341604851993858, -0.839830169131854],
            [-0.804031356019172, -0.569029065124742, 0.172451447025628]
        ])
        boresight = np.array([0.0, 0.0625969755450201, 0.99803888634292])

        output = xr.Dataset({
            'lat_error_deg': (['measurement'], [lat_error_deg]),
            'lon_error_deg': (['measurement'], [lon_error_deg]),
            'riss_ctrs': (['measurement', 'xyz'], [r_iss_midframe]),
            'bhat_hs': (['measurement', 'xyz'], [boresight]),
            't_hs2ctrs': (['xyz_from', 'xyz_to', 'measurement'], t_matrix[:, :, np.newaxis]),
            'cp_lat_deg': (['measurement'], [gcp_center_lat]),
            'cp_lon_deg': (['measurement'], [gcp_center_lon]),
            'cp_alt': (['measurement'], [0.0]),
        }, coords={
            'measurement': [0],
            'xyz': ['x', 'y', 'z'],
            'xyz_from': ['x', 'y', 'z'],
            'xyz_to': ['x', 'y', 'z']
        })

        # Add metadata
        output.attrs.update({
            'lat_error_km': result.lat_error_km,
            'lon_error_km': result.lon_error_km,
            'correlation_ccv': result.ccv_final,
            'final_grid_step_m': result.final_grid_step_m,
            'final_index_row': result.final_index_row,
            'final_index_col': result.final_index_col,
            'processing_time_s': processing_time,
            'gcp_file': str(test_case['gcp_file'].name),
            'gcp_center_lat': gcp_center_lat,
            'gcp_center_lon': gcp_center_lon,
            'test_mode': True,
            'test_case_id': test_case['case_id'],
            'test_case_name': test_case['case_name'],
            'expected_lat_error_km': test_case['expected_lat_error_km'],
            'expected_lon_error_km': test_case['expected_lon_error_km'],
            'param_idx': param_idx,
        })

        return output

    except Exception as e:
        logger.error(f"  Test mode image matching failed: {e}")
        raise


class MonteCarloTestModeTestCase(unittest.TestCase):
    """Unit tests for Monte Carlo test mode components."""

    def setUp(self):
        self.root_dir = Path(__file__).parents[2]
        self.test_data_dir = self.root_dir / 'tests' / 'data' / 'clarreo' / 'image_match'

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.work_dir = Path(self.__tmp_dir.name)

    def test_discover_test_cases(self):
        """Test that test case discovery works correctly."""
        logger.info("Testing test case discovery...")

        test_cases = mc.discover_test_image_match_cases(self.test_data_dir)

        # Should find multiple test cases
        self.assertGreater(len(test_cases), 0, "No test cases discovered")

        logger.info(f"Discovered {len(test_cases)} test case variants")

        # Each test case should have required fields
        for tc in test_cases:
            self.assertIn('case_id', tc)
            self.assertIn('subimage_file', tc)
            self.assertIn('gcp_file', tc)
            self.assertIn('expected_lat_error_km', tc)
            self.assertIn('expected_lon_error_km', tc)
            self.assertTrue(tc['subimage_file'].exists(),
                          f"Subimage file missing: {tc['subimage_file']}")
            self.assertTrue(tc['gcp_file'].exists(),
                          f"GCP file missing: {tc['gcp_file']}")

    def test_discover_specific_cases(self):
        """Test discovery with specific test case selection."""
        logger.info("Testing specific case discovery...")

        test_cases = mc.discover_test_image_match_cases(
            self.test_data_dir,
            test_cases=['1', '2']
        )

        # Should only find cases from groups 1 and 2
        case_ids = set(tc['case_id'] for tc in test_cases)
        self.assertTrue(case_ids.issubset({'1', '2'}),
                       f"Unexpected case IDs: {case_ids}")

        logger.info(f"Discovered {len(test_cases)} cases from groups 1 and 2")

    def test_image_matching_with_test_data(self):
        """Test that image matching works with test data."""
        logger.info("Testing image matching with test data...")

        # Get first test case
        test_cases = mc.discover_test_image_match_cases(
            self.test_data_dir,
            test_cases=['1']
        )
        self.assertGreater(len(test_cases), 0, "No test cases found")

        test_case = test_cases[0]
        test_config = mc.TestModeConfig(
            test_data_dir=self.test_data_dir,
            test_cases=['1'],
            randomize_errors=False,  # No randomization for validation
            cache_image_match_results=True,
        )

        # Run image matching using test-specific wrapper that applies errors
        result = run_test_mode_image_matching_with_applied_errors(
            test_case,
            param_idx=0,
            test_mode_config=test_config
        )

        # Validate output format
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn('lat_error_deg', result.data_vars)
        self.assertIn('lon_error_deg', result.data_vars)

        # Validate metadata
        self.assertIn('lat_error_km', result.attrs)
        self.assertIn('lon_error_km', result.attrs)
        self.assertIn('correlation_ccv', result.attrs)
        self.assertTrue(result.attrs['test_mode'])

        # Validate against expected errors (should be close)
        lat_error_km = result.attrs['lat_error_km']
        lon_error_km = result.attrs['lon_error_km']
        expected_lat = test_case['expected_lat_error_km']
        expected_lon = test_case['expected_lon_error_km']

        self.assertAlmostEqual(lat_error_km, expected_lat, delta=0.1,
                              msg=f"Lat error {lat_error_km} not close to expected {expected_lat}")
        self.assertAlmostEqual(lon_error_km, expected_lon, delta=0.1,
                              msg=f"Lon error {lon_error_km} not close to expected {expected_lon}")

        logger.info(f"Image matching validated: "
                   f"lat={lat_error_km:.3f}km (exp={expected_lat}), "
                   f"lon={lon_error_km:.3f}km (exp={expected_lon})")

    def test_error_variation(self):
        """Test that error variation works correctly."""
        logger.info("Testing error variation...")

        # Get test case and run once
        test_cases = mc.discover_test_image_match_cases(
            self.test_data_dir,
            test_cases=['1']
        )
        test_case = test_cases[0]
        test_config = mc.TestModeConfig(
            test_data_dir=self.test_data_dir,
            test_cases=['1'],
            randomize_errors=True,
            error_variation_percent=5.0,  # 5% variation
            cache_image_match_results=True,
        )

        # Run first time (no cache)
        result1 = mc.run_test_mode_image_matching(test_case, 0, test_config)

        # Run with variation (using cache)
        result2 = mc.run_test_mode_image_matching(test_case, 1, test_config,
                                                  cached_result=result1)

        # Results should be different
        lat1 = result1.attrs['lat_error_km']
        lat2 = result2.attrs['lat_error_km']

        self.assertNotEqual(lat1, lat2, "Variation not applied")

        # Variation should be within expected range (roughly ±5%)
        variation_pct = abs(lat2 - lat1) / abs(lat1) * 100
        self.assertLess(variation_pct, 15.0,  # Allow 3x sigma
                       f"Variation {variation_pct:.1f}% exceeds reasonable bounds")

        logger.info(f"Variation validated: {lat1:.3f} → {lat2:.3f} km ({variation_pct:.1f}%)")


def run_full_pipeline_test(n_iterations=5, test_cases=None, work_dir=None):
    """
    Run full Monte Carlo pipeline with test data.

    This simulates the complete GCS workflow without requiring production data.

    Args:
        n_iterations: Number of Monte Carlo iterations to run
        test_cases: Specific test cases to use (e.g., ['1', '2']) or None for all
        work_dir: Working directory for outputs (or creates temp dir)

    Returns:
        Tuple of (results, netcdf_data, netcdf_file_path, temp_dir_obj)
        Note: temp_dir_obj will be None if work_dir was provided, otherwise it's
        the TemporaryDirectory object that must be kept alive until you're done
        with the NetCDF file.
    """
    logger.info("=" * 80)
    logger.info("RUNNING FULL MONTE CARLO PIPELINE TEST")
    logger.info("=" * 80)

    root_dir = Path(__file__).parents[1]
    test_data_dir = root_dir / 'data' / 'clarreo' / 'image_match'

    # Create or use provided work directory
    if work_dir is None:
        # Default to a real directory instead of temp
        work_dir = root_dir / 'test_correction/monte_carlo_results'
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir_obj = None
        cleanup_work_dir = False
        logger.info(f"No work_dir specified, using default: {work_dir}")
    else:
        tmp_dir_obj = None
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = False

    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Test data directory: {test_data_dir}")
    logger.info(f"Iterations: {n_iterations}")
    logger.info(f"Test cases: {test_cases or 'all'}")

    # Create test mode configuration
    test_config = mc.TestModeConfig(
        test_data_dir=test_data_dir,
        test_cases=test_cases,
        randomize_errors=True,
        error_variation_percent=3.0,
        cache_image_match_results=True,
    )

    # Discover test cases
    discovered_cases = mc.discover_test_image_match_cases(
        test_data_dir,
        test_cases
    )

    logger.info(f"Discovered {len(discovered_cases)} test case variants")

    # ============================================================================
    # USE REAL GCP PAIRING MODULE
    # ============================================================================
    logger.info("=" * 80)
    logger.info("PERFORMING GCP PAIRING WITH SPATIAL MATCHING")
    logger.info("=" * 80)

    # Load all L1A subimages as NamedImageGrid objects
    l1a_images = []
    l1a_to_testcase = {}  # Map L1A name to test case metadata

    for test_case in discovered_cases:
        l1a_img = load_image_grid_from_mat(
            test_case['subimage_file'],
            key="subimage",
            as_named=True,
            name=str(test_case['subimage_file'].relative_to(test_data_dir))
        )
        l1a_images.append(l1a_img)
        l1a_to_testcase[l1a_img.name] = test_case
        logger.info(f"Loaded L1A: {l1a_img.name}")

    # Load all unique GCP references as NamedImageGrid objects
    gcp_files_seen = set()
    gcp_images = []

    for test_case in discovered_cases:
        gcp_file = test_case['gcp_file']
        if gcp_file not in gcp_files_seen:
            gcp_img = load_image_grid_from_mat(
                gcp_file,
                key="GCP",
                as_named=True,
                name=str(gcp_file.relative_to(test_data_dir))
            )
            gcp_images.append(gcp_img)
            gcp_files_seen.add(gcp_file)
            logger.info(f"Loaded GCP: {gcp_img.name}")

    # Perform spatial pairing
    logger.info(f"\nRunning find_l1a_gcp_pairs with {len(l1a_images)} L1A images and {len(gcp_images)} GCP references...")
    pairing_result = find_l1a_gcp_pairs(
        l1a_images,
        gcp_images,
        max_distance_m=0.0  # Require strict overlap
    )

    logger.info(f"Pairing complete: Found {len(pairing_result.matches)} valid pairs")
    for match in pairing_result.matches:
        l1a_name = pairing_result.l1a_images[match.l1a_index].name
        gcp_name = pairing_result.gcp_images[match.gcp_index].name
        logger.info(f"  {l1a_name} ↔ {gcp_name} (distance: {match.distance_m:.1f}m)")

    # Convert pairing results to test cases for processing
    paired_test_cases = []
    for match in pairing_result.matches:
        l1a_img = pairing_result.l1a_images[match.l1a_index]
        gcp_img = pairing_result.gcp_images[match.gcp_index]

        # Get original test case metadata
        test_case = l1a_to_testcase[l1a_img.name]

        # Create paired test case with both L1A and matched GCP
        paired_test_case = {
            **test_case,  # Include all original metadata
            'l1a_image': l1a_img,
            'gcp_image': gcp_img,
            'pairing_distance_m': match.distance_m,
        }
        paired_test_cases.append(paired_test_case)

    logger.info(f"\nCreated {len(paired_test_cases)} paired test cases for Monte Carlo processing")
    logger.info("=" * 80)

    # Use paired test cases instead of discovered cases
    n_gcp_pairs = len(paired_test_cases)

    # ============================================================================
    # CREATE CONFIGURATION USING CLARREO CONFIG MODULE
    # ============================================================================
    logger.info("=" * 80)
    logger.info("CREATING TEST CONFIGURATION (based on CLARREO config)")
    logger.info("=" * 80)

    # Use the CLARREO config module to get base configuration
    # Note: We're in test mode so we skip actual geolocation/kernel creation
    root_dir = Path(__file__).parents[2]
    generic_dir = root_dir / 'data' / 'generic'
    data_dir = root_dir / 'tests' / 'data' / 'clarreo' / 'gcs'

    # Create base CLARREO config (this gives us proper structure)
    base_config = create_clarreo_monte_carlo_config(data_dir, generic_dir)

    # Override settings for test mode:
    # - Use specified number of iterations
    # - Set sigma=0 for parameters (no real parameter variation in test mode)
    # - Keep minimal parameter set (only need 1 for metadata/structure)
    config = mc.MonteCarloConfig(
        seed=42,
        n_iterations=n_iterations,
        parameters=[
            # Use minimal parameter set for test mode - only need one for structure
            # Set sigma=0 since variations come from image matching randomization, not parameters
            mc.ParameterConfig(
                ptype=mc.ParameterType.CONSTANT_KERNEL,
                config_file=data_dir / 'cprs_hysics_v01.attitude.ck.json',
                data={
                    'current_value': [0.0, 0.0, 0.0],
                    'sigma': 0.0,  # No parameter variation in test mode
                    'units': 'arcseconds',
                    'transformation_type': 'dcm_rotation',
                    'coordinate_frames': ['HYSICS_SLIT', 'CRADLE_ELEVATION']
                }
            )
        ],
        geo=base_config.geo,  # Use the full geolocation config from CLARREO config
    )

    logger.info(f"Configuration created:")
    logger.info(f"  Mission: CLARREO (from clarreo_config module)")
    logger.info(f"  Instrument: {config.geo.instrument_name}")
    logger.info(f"  Iterations: {config.n_iterations}")
    logger.info(f"  Parameters: {len(config.parameters)} (minimal for test mode)")
    logger.info(f"  Test mode: Variations from image matching, not parameter changes")
    logger.info("=" * 80)

    # Cache image matching results for efficiency
    image_match_cache = {}

    # Initialize results storage
    results = []

    # Prepare NetCDF data structure
    n_param_sets = n_iterations
    n_gcp_pairs = len(discovered_cases)

    netcdf_data = {
        'parameter_set_id': np.arange(n_param_sets),
        'gcp_pair_id': np.arange(n_gcp_pairs),
        'param_hysics_roll': np.full(n_param_sets, 0.0),
        'param_hysics_pitch': np.full(n_param_sets, 0.0),
        'param_hysics_yaw': np.full(n_param_sets, 0.0),
        'rms_error_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'mean_error_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'max_error_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'std_error_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'n_measurements': np.full((n_param_sets, n_gcp_pairs), 0, dtype=int),
        'im_lat_error_km': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'im_lon_error_km': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'im_ccv': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'im_grid_step_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'percent_under_250m': np.full(n_param_sets, np.nan),
        'mean_rms_all_pairs': np.full(n_param_sets, np.nan),
        'worst_pair_rms': np.full(n_param_sets, np.nan),
        'best_pair_rms': np.full(n_param_sets, np.nan),
    }

    logger.info(f"Initialized NetCDF structure: {n_param_sets} iterations × {n_gcp_pairs} test pairs")

    # Run Monte Carlo iterations
    for param_idx in range(n_iterations):
        logger.info(f"=== Iteration {param_idx + 1}/{n_iterations} ===")

        image_matching_results = []
        pair_errors = []

        # Process each test case
        for pair_idx, test_case in enumerate(discovered_cases):
            logger.info(f"  Processing test pair {pair_idx + 1}/{len(discovered_cases)}: "
                       f"{test_case['case_name']} - {test_case['subcase_name']}")

            # Get cached result if available, otherwise None
            cache_key = f"{test_case['case_id']}_{test_case['subcase_name']}"
            cached_result = image_match_cache.get(cache_key)

            # Run image matching (with caching and variation)
            image_matching_output = mc.run_test_mode_image_matching(
                test_case=test_case,
                param_idx=param_idx,
                test_mode_config=test_config,
                cached_result=cached_result,
            )

            # Cache first result
            if cache_key not in image_match_cache:
                image_match_cache[cache_key] = image_matching_output
                logger.info(f"    Cached image matching result for {cache_key}")

            # Store for aggregate processing
            image_matching_output.attrs['gcp_pair_index'] = pair_idx
            image_matching_output.attrs['gcp_pair_id'] = f"test_{test_case['case_id']}_pair_{pair_idx}"
            image_matching_results.append(image_matching_output)

            # Extract error metrics (convert to meters)
            lat_error_m = abs(image_matching_output.attrs['lat_error_km'] * 1000)
            lon_error_m = abs(image_matching_output.attrs['lon_error_km'] * 1000)
            rms_error_m = np.sqrt(lat_error_m**2 + lon_error_m**2)

            pair_errors.append(rms_error_m)

            # Store in NetCDF structure
            netcdf_data['rms_error_m'][param_idx, pair_idx] = rms_error_m
            netcdf_data['mean_error_m'][param_idx, pair_idx] = rms_error_m  # Single point
            netcdf_data['max_error_m'][param_idx, pair_idx] = rms_error_m
            netcdf_data['std_error_m'][param_idx, pair_idx] = 0.0  # Single point
            netcdf_data['n_measurements'][param_idx, pair_idx] = 1
            netcdf_data['im_lat_error_km'][param_idx, pair_idx] = image_matching_output.attrs['lat_error_km']
            netcdf_data['im_lon_error_km'][param_idx, pair_idx] = image_matching_output.attrs['lon_error_km']
            netcdf_data['im_ccv'][param_idx, pair_idx] = image_matching_output.attrs['correlation_ccv']
            netcdf_data['im_grid_step_m'][param_idx, pair_idx] = image_matching_output.attrs['final_grid_step_m']

            logger.info(f"    RMS error: {rms_error_m:.2f}m, "
                       f"Lat: {image_matching_output.attrs['lat_error_km']:.3f}km, "
                       f"Lon: {image_matching_output.attrs['lon_error_km']:.3f}km")

        # Compute aggregate metrics for this iteration
        pair_errors = np.array(pair_errors)
        valid_errors = pair_errors[~np.isnan(pair_errors)]

        if len(valid_errors) > 0:
            error_stats = mc.call_error_stats_module(image_matching_results)
            print("ERROR_STATS CALLED \n")
            print(error_stats)
            percent_under_250 = (valid_errors < 250.0).sum() / len(valid_errors) * 100
            netcdf_data['percent_under_250m'][param_idx] = percent_under_250
            netcdf_data['mean_rms_all_pairs'][param_idx] = np.mean(valid_errors)
            netcdf_data['best_pair_rms'][param_idx] = np.min(valid_errors)
            netcdf_data['worst_pair_rms'][param_idx] = np.max(valid_errors)

            logger.info(f"Iteration {param_idx + 1} complete:")
            logger.info(f"  {percent_under_250:.1f}% under 250m threshold")
            logger.info(f"  Mean RMS: {np.mean(valid_errors):.2f}m")
            logger.info(f"  Range: [{np.min(valid_errors):.2f}, {np.max(valid_errors):.2f}]m")

    # Save NetCDF results
    logger.info("Saving NetCDF results...")
    netcdf_file = work_dir / "test_monte_carlo_pipeline_results.nc"
    mc._save_netcdf_results(netcdf_data, netcdf_file, config)

    logger.info("=" * 80)
    logger.info("PIPELINE TEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"NetCDF output: {netcdf_file}")
    logger.info(f"Total iterations: {n_iterations}")
    logger.info(f"Total test pairs: {len(discovered_cases)}")
    logger.info(f"Results can be loaded with: xr.open_dataset('{netcdf_file}')")

    # Cleanup temp dir if we created it
    if cleanup_work_dir:
        logger.info("(Temporary work directory will be cleaned up on exit)")

    return results, netcdf_data, netcdf_file, tmp_dir_obj


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description='Monte Carlo Pipeline Integration Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run unit tests
  pytest tests/test_monte_carlo_pipeline.py -v
  
  # Run full pipeline test (standalone)
  python tests/test_monte_carlo_pipeline.py
  
  # Quick test with fewer iterations
  python tests/test_monte_carlo_pipeline.py --quick
  
  # Use specific test cases
  python tests/test_monte_carlo_pipeline.py --test-cases 1 2
  
  # Save results to specific directory
  python tests/test_monte_carlo_pipeline.py --output-dir ./test_results
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with only 2 iterations and 1 test case'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of Monte Carlo iterations (default: 5)'
    )
    parser.add_argument(
        '--test-cases',
        nargs='+',
        default=None,
        help='Specific test cases to use (e.g., 1 2 3)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: temp directory)'
    )
    parser.add_argument(
        '--unit-tests',
        action='store_true',
        help='Run unit tests instead of full pipeline'
    )

    args = parser.parse_args()

    if args.unit_tests:
        # Run unit tests
        unittest.main(argv=[''], exit=True)
    else:
        # Run full pipeline test
        if args.quick:
            n_iterations = 2
            test_cases = ['1']
            logger.info("Quick mode: 2 iterations, test case 1 only")
        else:
            n_iterations = args.iterations
            test_cases = args.test_cases

        results, netcdf_data, netcdf_file, tmp_dir_obj = run_full_pipeline_test(
            n_iterations=n_iterations,
            test_cases=test_cases,
            work_dir=args.output_dir,
        )

        # Validate results
        logger.info("\nValidating results...")
        ds = xr.open_dataset(netcdf_file)
        logger.info(f"NetCDF dimensions: {dict(ds.sizes)}")
        logger.info(f"Variables: {list(ds.data_vars.keys())}")

        # Check that we have real data
        assert not np.all(np.isnan(ds['im_lat_error_km'].values)), "No latitude errors stored"
        assert not np.all(np.isnan(ds['im_ccv'].values)), "No correlation values stored"

        logger.info("\n✅ All validations passed!")
        logger.info(f"\nResults saved to: {netcdf_file}")

if __name__ == '__main__':
    main()
