"""
Monte Carlo Pipeline Integration Test

This test script validates the complete Monte Carlo GCS pipeline using
validated test data from tests/data/clarreo/image_match/.

It enables testing the full pipeline without requiring production data:
- Real image matching with test files
- Error statistics processing
- NetCDF output generation
- Parameter variation simulation

Usage:
    # Run all tests
    pytest tests/test_monte_carlo_pipeline.py -v

    # Run standalone (full integration test)
    python tests/test_monte_carlo_pipeline.py

    # Run quick test (fewer iterations)
    python tests/test_monte_carlo_pipeline.py --quick
"""

import argparse
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from curryer import utils
from curryer.correction import monte_carlo as mc

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])


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

        # Run image matching
        result = mc.run_test_mode_image_matching(
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
        work_dir = root_dir / 'tests/test_corrections/monte_carlo_results'
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

    # Create minimal Monte Carlo configuration
    # Note: We're not actually using the geo config since we skip geolocation
    generic_dir = root_dir / 'data' / 'generic'

    geo_config = mc.GeolocationConfig(
        meta_kernel_file=Path('meta_kernel.json'),  # Not used in test mode
        generic_kernel_dir=generic_dir,
        dynamic_kernels=[],
        instrument_name='CPRS_HYSICS',
        time_field='corrected_timestamp',
    )

    # Create minimal parameter set for testing
    # Use zero sigma so all iterations use the same "parameters"
    # (variations come from image matching randomization)
    params = [
        mc.ParameterConfig(
            ptype=mc.ParameterType.CONSTANT_KERNEL,
            config_file=Path('cprs_hysics_v01.attitude.ck.json'),
            data={
                'current_value': [0.0, 0.0, 0.0],  # Changed from 'center' to 'current_value'
                'sigma': 0.0,  # No parameter variation in test mode
                'units': 'arcseconds',
            }
        )
    ]

    config = mc.MonteCarloConfig(
        seed=42,
        n_iterations=n_iterations,
        parameters=params,
        geo=geo_config,
    )

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
    netcdf_file = work_dir / "monte_carlo_test_results.nc"
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