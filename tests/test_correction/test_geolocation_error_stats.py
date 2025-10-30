"""
Unit tests for geolocation_error_stats.py

This module contains comprehensive unit tests for the ErrorStatsProcessor class
and related functionality, including edge cases, validation, and numerical accuracy.
"""

import logging
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import xarray as xr

from curryer.correction.geolocation_error_stats import (
    ErrorStatsProcessor,
    GeolocationConfig,
)


logger = logging.getLogger(__name__)

# Configure display options for better test output
xr.set_options(display_width=120)
np.set_printoptions(linewidth=120)


# ============================================================================
# TEST HELPER FUNCTIONS
# ============================================================================

def _create_test_config(**overrides):
    """
    Create GeolocationConfig for testing with CLARREO defaults.

    These values come from the CLARREO mission configuration and are appropriate
    for testing CLARREO-specific scenarios. Tests can override individual values
    to test edge cases or alternative configurations.

    Args:
        **overrides: Override any default values

    Returns:
        GeolocationConfig with CLARREO test values
    """
    # CLARREO mission defaults (from clarreo_monte_carlo_config.json)
    # These values should match the canonical CLARREO configuration
    defaults = {
        'earth_radius_m': 6378140.0,  # WGS84 Earth radius (CLARREO standard)
        'performance_threshold_m': 250.0,  # CLARREO accuracy requirement
        'performance_spec_percent': 39.0,  # CLARREO performance spec
        'variable_names': {
            'spacecraft_position': 'riss_ctrs',  # CLARREO/ISS variable name
            'boresight': 'bhat_hs',  # HySICS boresight variable name
            'transformation_matrix': 't_hs2ctrs'  # HySICS to CTRS transform
        }
    }
    defaults.update(overrides)
    return GeolocationConfig(**defaults)


def create_test_dataset_13_cases() -> xr.Dataset:
    """
    Create test dataset with the original 13 hardcoded test cases.

    This is the reference test data moved from production code.
    These cases validate the processor against known MATLAB results.
    """
    # Test case data (from original MATLAB script)
    test_data = {
        'lat_error_deg': [0.026980, -0.0269188, 0.009040, -0.008925, 0.022515,
                         0.001, -0.0015, -0.002, 0.002, 0.001, -0.001, 0.0011, 0.0025],
        'lon_error_deg': [-0.027266, 0.018384, 0.010851, -0.026241, 0.000992,
                         0.0005, 0.0015, 0.0, -0.0005, 0.001, -0.0015, 0.0011, 0.0],
        'cp_lat_deg': [-8.57802802047, 10.9913499301, 33.9986324792, 31.1629017783, -69.6971234613,
                      34.2, -19.7, -49.5, -8.2, 46.0, 32.5, -20.33, -47.6],
        'cp_lon_deg': [125.482222317, -71.8457829833, -120.248435967, -8.7815788192, -15.2311156066,
                      42.5, -51.1, -178.3, 70.0, -29.0, -170.4, 95.4, -30.87],
        'cp_alt': [44, 4, 0, 894, 3925, 0, 1000, 500, 500, 50, 100, 100, 100]
    }

    # RISS_CTRS positions (satellite positions)
    riss_ctrs_data = np.array([
        [-3888220.86746399, 5466997.0490439, -1000356.92985575],
        [2138128.91767507, -6313660.02871594, 1241996.71916521],
        [-2836930.06048711, -4869372.01247407, 3765186.91739563],
        [5764626.80186185, -843462.027662883, 3457275.08087601],
        [2210828.23546441, -6156903.77352567, -1818743.37976767],
        [4160733.71254889, 3708441.12891715, 3850046.48797648],
        [4060487.97522754, -4920200.36807653, -2308736.58835498],
        [-4274543.69565126, -116765.394831108, -5276242.10262264],
        [2520101.83352962, 6230331.37805726, -961492.530214298],
        [4248835.7920035, -2447631.1800248, 4676942.35070364],
        [-5515282.275281, -925908.369707886, 3822512.18293707],
        [-824875.850002718, 6312906.79811629, -2344264.22196647],
        [3675746.11507236, -2198122.65541618, -5270960.3157354]
    ])

    # Boresight vectors in HS coordinate system
    bhat_hs_data = np.array([
        [0, 0.0625969755450201, 0.99803888634292],
        [0, 0.0313138440396569, 0.999509601340307],
        [0, 0.000368458389306164, 0.999999932119205],
        [0, -0.0368375032472, 0.999321268839262],
        [0, -0.0699499255109834, 0.997550503945042],
        [0, -0.0368375032472, 0.999321268839262],
        [0, -0.0257892283106606, 0.999667402541035],
        [0, 0.0257892283106606, 0.999667402541035],
        [0, 0.0625969755450201, 0.99803888634292],
        [0, 0.0552406262884485, 0.998473070847311],
        [0, -0.0147378023382108, 0.999891392693346],
        [0, -0.0221057030926655, 0.999755639089262],
        [0, -0.0221057030926655, 0.999755639089262]
    ])

    # Transformation matrices from HS to CTRS (3x3x13)
    t_hs2ctrs_data = np.zeros((3, 3, 13))

    # Test case 1
    t_hs2ctrs_data[:, :, 0] = [
        [-0.418977524967338, 0.748005379751721, 0.514728846515064],
        [-0.421890284446342, 0.341604851993858, -0.839830169131854],
        [-0.804031356019172, -0.569029065124742, 0.172451447025628]
    ]

    # Test case 2
    t_hs2ctrs_data[:, :, 1] = [
        [0.509557370616697, 0.714990103896663, -0.478686157497828],
        [0.336198439435013, 0.346660121582392, 0.875669669125261],
        [0.792036549265032, -0.607137473258174, -0.0637353370903461]
    ]

    # Test case 3
    t_hs2ctrs_data[:, :, 2] = [
        [0.436608377090994, -0.795688667243495, 0.419824570355571],
        [-0.682818757213707, 0.0107593091164333, 0.730508577680278],
        [-0.585774418911493, -0.605610255930006, -0.53861354240429]
    ]

    # Test case 4
    t_hs2ctrs_data[:, :, 3] = [
        [-0.275228112982228, 0.368161232084539, -0.888091658002842],
        [0.740939532874243, 0.669849578957866, 0.0480640218257623],
        [0.612583132683508, -0.644793648200697, -0.457146646921637]
    ]

    # Test case 5
    t_hs2ctrs_data[:, :, 4] = [
        [0.497596843733441, -0.8343127195548, -0.237317650198193],
        [0.404893735025568, -0.0185495841473054, 0.914175571903453],
        [-0.767110451267327, -0.550979308973609, 0.328577778675617]
    ]

    # Test case 6
    t_hs2ctrs_data[:, :, 5] = [
        [-0.765506977045252, 0.0328563789337692, -0.642588135250651],
        [0.295444324153605, 0.905137001368494, -0.305678298647175],
        [0.571587018239969, -0.423847628778339, -0.702596052277786]
    ]

    # Test case 7
    t_hs2ctrs_data[:, :, 6] = [
        [0.629603159973548, 0.368109063956699, -0.684174861189846],
        [0.215915166022854, 0.763032030046674, 0.609230735287836],
        [0.746311263503805, -0.531296841841519, 0.400927218783918]
    ]

    # Test case 8
    t_hs2ctrs_data[:, :, 7] = [
        [0.194530273749036, 0.949748975936332, 0.245223854916003],
        [-0.978512013106359, 0.205316430746179, -0.0189577388931897],
        [-0.0683535866042618, -0.236266544212139, 0.969281089206121]
    ]

    # Test case 9
    t_hs2ctrs_data[:, :, 8] = [
        [-0.446421529583839, 0.413968410219497, -0.793307812225063],
        [0.384674732015686, -0.711668847399292, -0.587837230750712],
        [-0.807919035840032, -0.56758835193185, 0.158460875505101]
    ]

    # Test case 10
    t_hs2ctrs_data[:, :, 9] = [
        [0.632159685228781, -0.204512480192889, -0.747361135669863],
        [0.598189792213041, -0.48424547358001, 0.638493680968301],
        [-0.492486654503011, -0.850693598485042, -0.183784021560657]
    ]

    # Test case 11
    t_hs2ctrs_data[:, :, 10] = [
        [0.753428906287479, -0.49153961754033, 0.436730605865421],
        [-0.565149851875981, -0.823589060852856, 0.0480236900131712],
        [0.336081133202996, -0.283000352923251, -0.898309519920648]
    ]

    # Test case 12
    t_hs2ctrs_data[:, :, 11] = [
        [-0.585265557251293, -0.595045400433036, 0.550803349451662],
        [-0.109341614649782, -0.615175192945938, -0.780771245706364],
        [0.803435522491452, -0.517183787785386, 0.294977548217097]
    ]

    # Test case 13
    t_hs2ctrs_data[:, :, 12] = [
        [0.292122841971449, -0.95622050459562, 0.017506859615382],
        [0.95633436246494, 0.291879296125504, -0.0152004494429911],
        [0.00942509242927969, 0.0211828985704245, 0.999731155222533]
    ]

    # Create coordinates
    measurements = np.arange(13)

    # Create dataset
    dataset = xr.Dataset(
        {
            'lat_error_deg': (['measurement'], test_data['lat_error_deg']),
            'lon_error_deg': (['measurement'], test_data['lon_error_deg']),
            'riss_ctrs': (['measurement', 'xyz'], riss_ctrs_data),
            'bhat_hs': (['measurement', 'xyz'], bhat_hs_data),
            't_hs2ctrs': (['xyz_from', 'xyz_to', 'measurement'], t_hs2ctrs_data),
            'cp_lat_deg': (['measurement'], test_data['cp_lat_deg']),
            'cp_lon_deg': (['measurement'], test_data['cp_lon_deg']),
            'cp_alt': (['measurement'], test_data['cp_alt'])
        },
        coords={
            'measurement': measurements,
            'xyz': ['x', 'y', 'z'],
            'xyz_from': ['x', 'y', 'z'],
            'xyz_to': ['x', 'y', 'z']
        },
        attrs={
            'title': 'Test Dataset for Geolocation Error Statistics',
            'description': 'Original 13 hardcoded test cases from MATLAB implementation',
            'n_measurements': 13
        }
    )

    return dataset


def process_test_data(display_results: bool = True) -> xr.Dataset:
    """
    Process the original 13 test cases using the processor.

    Helper function for regression testing against known values.
    """
    config = _create_test_config()
    processor = ErrorStatsProcessor(config=config)
    test_data = create_test_dataset_13_cases()

    results = processor.process_geolocation_errors(test_data)

    if display_results:
        print(f"Processing Results Summary:")
        print(f"=" * 50)
        print(f"Total measurements: {results.attrs['total_measurements']}")
        print(f"Mean error distance: {results.attrs['mean_error_distance_m']:.2f} m")
        print(f"Std error distance: {results.attrs['std_error_distance_m']:.2f} m")
        print(f"Min/Max error: {results.attrs['min_error_distance_m']:.2f} / {results.attrs['max_error_distance_m']:.2f} m")
        print(f"Errors < 250m: {results.attrs['num_below_250m']} ({results.attrs['percent_below_250m']:.1f}%)")

        spec_status = "✓ PASS" if results.attrs['performance_spec_met'] else "✗ FAIL"
        print(f"Performance spec (>39% < 250m): {spec_status}")

    return results


# ============================================================================
# TEST CASES
# ============================================================================


class GeolocationErrorStatsTestCase(unittest.TestCase):
    """Test case for geolocation error statistics processing."""

    def setUp(self) -> None:
        """Set up test fixtures and data paths."""
        root_dir = Path(__file__).parents[2]
        print(root_dir)
        self.error_stats_dir = root_dir / 'curryer' / 'correction'
        self.data_dir = root_dir / 'data'
        self.test_dir = root_dir / 'tests' / 'data'

        # Verify directories exist
        self.assertTrue(self.error_stats_dir.is_dir())

        # Create default processor with test config
        self.config = _create_test_config()
        self.processor = ErrorStatsProcessor(config=self.config)

        # Create minimal test dataset
        self.minimal_test_data = self._create_minimal_test_data()

    def _create_minimal_test_data(self) -> xr.Dataset:
        """Create minimal test dataset for testing."""
        # Create proper transformation matrices
        t_matrices = np.zeros((3, 3, 3))
        t_matrices[:, :, 0] = np.eye(3)  # Identity matrix
        t_matrices[:, :, 1] = [[0.9, 0.1, 0], [-0.1, 0.9, 0], [0, 0, 1]]  # Simple rotation
        t_matrices[:, :, 2] = [[0.8, 0, 0.2], [0, 1, 0], [-0.2, 0, 0.8]]  # Another rotation

        return xr.Dataset({
            'lat_error_deg': (['measurement'], [0.001, -0.002, 0.0015]),
            'lon_error_deg': (['measurement'], [0.0005, 0.001, -0.001]),
            'riss_ctrs': (['measurement', 'xyz'], [
                [4000000.0, 3000000.0, 2000000.0],
                [5000000.0, -2000000.0, 1000000.0],
                [-3000000.0, 4000000.0, 3000000.0]
            ]),
            'bhat_hs': (['measurement', 'xyz'], [
                [0, 0.05, 0.9987],
                [0, -0.03, 0.9995],
                [0, 0.02, 0.9998]
            ]),
            't_hs2ctrs': (['xyz_from', 'xyz_to', 'measurement'], t_matrices),
            'cp_lat_deg': (['measurement'], [30.0, -20.0, 45.0]),
            'cp_lon_deg': (['measurement'], [120.0, -80.0, 0.0]),
            'cp_alt': (['measurement'], [100, 500, 0])
        }, coords={'measurement': [0, 1, 2], 'xyz': ['x', 'y', 'z'],
                   'xyz_from': ['x', 'y', 'z'], 'xyz_to': ['x', 'y', 'z']})

    def test_geolocation_config_default(self):
        """Test default configuration values.

        Validates standard Earth radius (WGS84) and performance specs:
        - 250m threshold: nadir equivalent accuracy requirement
        - 39%: project performance requirement (>39% of measurements must be <250m)
        """
        config = _create_test_config()
        self.assertEqual(config.earth_radius_m, 6378140.0)
        self.assertEqual(config.performance_threshold_m, 250.0)
        self.assertEqual(config.performance_spec_percent, 39.0)

    def test_geolocation_config_custom(self):
        """Test custom configuration values."""
        config = _create_test_config(
            earth_radius_m=6371000.0,
            performance_threshold_m=200.0,
            performance_spec_percent=40.0
        )
        self.assertEqual(config.earth_radius_m, 6371000.0)
        self.assertEqual(config.performance_threshold_m, 200.0)
        self.assertEqual(config.performance_spec_percent, 40.0)

    def test_processor_initialization_default(self):
        """Test processor initialization with default config."""
        config = _create_test_config()
        processor = ErrorStatsProcessor(config=config)
        self.assertIsInstance(processor.config, GeolocationConfig)
        self.assertEqual(processor.config.earth_radius_m, 6378140.0)

    def test_processor_initialization_custom(self):
        """Test processor initialization with custom config."""
        config = _create_test_config(
            earth_radius_m=6371000.0,
            performance_threshold_m=200.0
        )
        processor = ErrorStatsProcessor(config=config)
        self.assertEqual(processor.config.earth_radius_m, 6371000.0)
        self.assertEqual(processor.config.performance_threshold_m, 200.0)

    def test_validate_input_data_success(self):
        """Test successful input validation."""
        # Should not raise any exception
        self.processor._validate_input_data(self.minimal_test_data)

    def test_validate_input_data_missing_variables(self):
        """Test validation with missing required variables."""
        incomplete_data = xr.Dataset({
            'lat_error_deg': (['measurement'], [0.001]),
            'lon_error_deg': (['measurement'], [0.001])
        }, coords={'measurement': [0]})

        with self.assertRaises(ValueError) as context:
            self.processor._validate_input_data(incomplete_data)
        self.assertIn("Missing required input variables", str(context.exception))

    def test_validate_input_data_missing_dimension(self):
        """Test validation with missing measurement dimension."""
        # Create data with all required variables but wrong dimension
        t_matrix_single = np.eye(3).reshape(3, 3, 1)

        complete_data = xr.Dataset({
            'lat_error_deg': (['time'], [0.001]),
            'lon_error_deg': (['time'], [0.001]),
            'riss_ctrs': (['time', 'xyz'], [[4000000.0, 3000000.0, 2000000.0]]),
            'bhat_hs': (['time', 'xyz'], [[0, 0.05, 0.9987]]),
            't_hs2ctrs': (['xyz_from', 'xyz_to', 'time'], t_matrix_single),
            'cp_lat_deg': (['time'], [30.0]),
            'cp_lon_deg': (['time'], [120.0]),
            'cp_alt': (['time'], [100])
        }, coords={'time': [0], 'xyz': ['x', 'y', 'z'],
                   'xyz_from': ['x', 'y', 'z'], 'xyz_to': ['x', 'y', 'z']})

        with self.assertRaises(ValueError) as context:
            self.processor._validate_input_data(complete_data)
        self.assertIn("Input data must have 'measurement' dimension", str(context.exception))

    def test_transform_boresight_vectors(self):
        """Test boresight vector transformation."""
        bhat_hs = np.array([[0, 0.1, 0.995], [0, -0.05, 0.9987]])
        t_matrices = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity
            [[0.9, 0.1, 0], [-0.1, 0.9, 0], [0, 0, 1]]  # Simple rotation
        ]).transpose(1, 2, 0)

        result = self.processor._transform_boresight_vectors(bhat_hs, t_matrices)

        self.assertEqual(result.shape, (2, 3))
        # First transformation (identity) should leave vector unchanged
        npt.assert_allclose(result[0], bhat_hs[0], rtol=1e-10)

    def test_create_ctrs_to_uen_transform(self):
        """Test CTRS to UEN transformation matrix creation."""
        lat_rad = np.deg2rad(30.0)
        lon_rad = np.deg2rad(45.0)

        t_matrix = self.processor._create_ctrs_to_uen_transform(lat_rad, lon_rad)

        self.assertEqual(t_matrix.shape, (3, 3))
        # Verify orthogonality (transformation matrix should be orthogonal)
        identity = np.dot(t_matrix, t_matrix.T)
        npt.assert_allclose(identity, np.eye(3), rtol=1e-12, atol=1e-15)

    def test_calculate_view_plane_vectors(self):
        """Test view plane vector calculations."""
        bhat_uen = np.array([0.5, 0.6, 0.8])  # Example boresight in UEN

        v_uen, x_uen = self.processor._calculate_view_plane_vectors(bhat_uen)

        self.assertEqual(v_uen.shape, (3,))
        self.assertEqual(x_uen.shape, (3,))

        # Check that vectors are unit vectors
        self.assertLess(abs(np.linalg.norm(v_uen) - 1.0), 1e-10)
        self.assertLess(abs(np.linalg.norm(x_uen) - 1.0), 1e-10)

        # Check orthogonality
        self.assertLess(abs(np.dot(v_uen, x_uen)), 1e-10)

    def test_calculate_scaling_factors_nadir(self):
        """Test scaling factors for nadir viewing (theta=0)."""
        riss_ctrs = np.array([0, 0, 7000000.0])  # Satellite directly above
        theta = 0.0  # Nadir viewing

        vp_factor, xvp_factor = self.processor._calculate_scaling_factors(riss_ctrs, theta)

        # For nadir viewing, scaling factors should be close to 1
        self.assertLess(abs(vp_factor - 1.0), 0.1)  # Allow some tolerance
        self.assertLess(abs(xvp_factor - 1.0), 0.1)

    def test_calculate_scaling_factors_off_nadir(self):
        """Test scaling factors for off-nadir viewing."""
        riss_ctrs = np.array([1000000.0, 0, 6800000.0])  # Off-nadir satellite position
        theta = np.deg2rad(10.0)  # 10 degree off-nadir

        vp_factor, xvp_factor = self.processor._calculate_scaling_factors(riss_ctrs, theta)

        # Off-nadir factors should be different from 1
        self.assertIsInstance(vp_factor, float)
        self.assertIsInstance(xvp_factor, float)
        self.assertFalse(np.isnan(vp_factor))
        self.assertFalse(np.isnan(xvp_factor))

    def test_calculate_statistics_basic(self):
        """Test basic statistics calculation."""
        errors = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        stats = self.processor._calculate_statistics(errors)

        self.assertEqual(stats['mean_error_distance_m'], 300.0)
        self.assertEqual(stats['min_error_distance_m'], 100.0)
        self.assertEqual(stats['max_error_distance_m'], 500.0)
        self.assertEqual(stats['total_measurements'], 5)
        self.assertEqual(stats['num_below_250m'], 2)
        self.assertEqual(stats['percent_below_250m'], 40.0)
        self.assertTrue(stats['performance_spec_met'])  # 40% > 39%

    def test_calculate_statistics_edge_cases(self):
        """Test statistics with edge cases."""
        # All errors below threshold
        errors_low = np.array([50.0, 100.0, 150.0])
        stats_low = self.processor._calculate_statistics(errors_low)
        self.assertEqual(stats_low['percent_below_250m'], 100.0)
        self.assertTrue(stats_low['performance_spec_met'])

        # All errors above threshold
        errors_high = np.array([300.0, 400.0, 500.0])
        stats_high = self.processor._calculate_statistics(errors_high)
        self.assertEqual(stats_high['percent_below_250m'], 0.0)
        self.assertFalse(stats_high['performance_spec_met'])

    def test_create_output_dataset(self):
        """Test output dataset creation."""
        # Create sample processing results
        sample_results = {
            'nadir_equiv_total_error_m': np.array([100.0, 200.0, 300.0]),
            'nadir_equiv_vp_error_m': np.array([80.0, 160.0, 240.0]),
            'nadir_equiv_xvp_error_m': np.array([60.0, 120.0, 180.0]),
            'vp_error_m': np.array([75.0, 150.0, 225.0]),
            'xvp_error_m': np.array([55.0, 110.0, 165.0]),
            'off_nadir_angle_rad': np.array([0.1, 0.2, 0.3]),
            'vp_scaling_factor': np.array([1.1, 1.2, 1.3]),
            'xvp_scaling_factor': np.array([1.05, 1.15, 1.25])
        }

        output_ds = self.processor._create_output_dataset(self.minimal_test_data, sample_results)

        # Check that all expected variables are present
        expected_vars = [
            'nadir_equiv_total_error_m', 'nadir_equiv_vp_error_m', 'nadir_equiv_xvp_error_m',
            'vp_error_m', 'xvp_error_m', 'off_nadir_angle_deg',
            'vp_scaling_factor', 'xvp_scaling_factor'
        ]

        for var in expected_vars:
            self.assertIn(var, output_ds.data_vars)

        # Check that original input variables are preserved
        self.assertIn('lat_error_deg', output_ds.data_vars)
        self.assertIn('lon_error_deg', output_ds.data_vars)

        # Check attributes
        self.assertIn('title', output_ds.attrs)
        self.assertIn('earth_radius_m', output_ds.attrs)

    def test_end_to_end_processing(self):
        """Test complete processing pipeline with test data."""
        test_data = create_test_dataset_13_cases()

        results = self.processor.process_geolocation_errors(test_data)

        # Check output structure
        self.assertIsInstance(results, xr.Dataset)
        self.assertIn('nadir_equiv_total_error_m', results.data_vars)
        self.assertEqual(len(results.measurement), 13)

        # Check statistics are computed
        self.assertIn('mean_error_distance_m', results.attrs)
        self.assertIn('percent_below_250m', results.attrs)
        self.assertIn('performance_spec_met', results.attrs)

    def test_regression_against_known_values(self):
        """Test against known good values from original implementation."""
        results = process_test_data(display_results=False)

        # These are the expected values from the original implementation
        expected_mean = 1203.26  # meters
        expected_percent_below_250 = 61.5  # percent
        expected_num_below_250 = 8

        # Allow small numerical differences
        self.assertLess(abs(results.attrs['mean_error_distance_m'] - expected_mean), 0.1)
        self.assertLess(abs(results.attrs['percent_below_250m'] - expected_percent_below_250), 0.1)
        self.assertEqual(results.attrs['num_below_250m'], expected_num_below_250)
        self.assertTrue(results.attrs['performance_spec_met'])

    def test_custom_config_processing(self):
        """Test processing with custom configuration."""
        custom_config = _create_test_config(performance_threshold_m=300.0)
        processor = ErrorStatsProcessor(custom_config)
        test_data = create_test_dataset_13_cases()

        results = processor.process_geolocation_errors(test_data)

        # With higher threshold, more errors should be below threshold
        self.assertLessEqual(results.attrs['num_below_250m'], results.attrs['total_measurements'])
        self.assertEqual(results.attrs['performance_threshold_m'], 300.0)

    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        with self.assertRaises(AttributeError):
            self.processor.process_geolocation_errors("not a dataset")

    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        empty_data = xr.Dataset({}, coords={})

        with self.assertRaises(ValueError):
            self.processor.process_geolocation_errors(empty_data)

    def test_large_dataset_processing(self):
        """Test processing with larger datasets."""
        # Create a larger synthetic dataset with more realistic data
        n_measurements = 50

        # Generate realistic transformation matrices
        transform_matrices = np.zeros((3, 3, n_measurements))
        for i in range(n_measurements):
            # Create orthogonal transformation matrices
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            c, s = np.cos(angle), np.sin(angle)
            transform_matrices[:, :, i] = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])

        large_dataset = xr.Dataset({
            'lat_error_deg': (['measurement'], np.random.normal(0, 0.005, n_measurements)),
            'lon_error_deg': (['measurement'], np.random.normal(0, 0.005, n_measurements)),
            'riss_ctrs': (['measurement', 'xyz'],
                         np.random.normal([0, 0, 7000000], [500000, 500000, 200000], (n_measurements, 3))),
            'bhat_hs': (['measurement', 'xyz'],
                       np.column_stack([np.zeros(n_measurements),
                                       np.random.normal(0, 0.03, n_measurements),
                                       np.sqrt(1 - np.random.normal(0, 0.03, n_measurements)**2)])),
            't_hs2ctrs': (['xyz_from', 'xyz_to', 'measurement'], transform_matrices),
            'cp_lat_deg': (['measurement'], np.random.uniform(-60, 60, n_measurements)),
            'cp_lon_deg': (['measurement'], np.random.uniform(-180, 180, n_measurements)),
            'cp_alt': (['measurement'], np.random.uniform(0, 3000, n_measurements))
        }, coords={'measurement': np.arange(n_measurements),
                   'xyz': ['x', 'y', 'z'], 'xyz_from': ['x', 'y', 'z'], 'xyz_to': ['x', 'y', 'z']})

        results = self.processor.process_geolocation_errors(large_dataset)

        self.assertEqual(len(results.measurement), n_measurements)
        self.assertIn('nadir_equiv_total_error_m', results.data_vars)

    def test_coordinate_transformation_accuracy(self):
        """Test coordinate transformation accuracy with known values."""
        lat_rad = np.deg2rad(30.0)
        lon_rad = np.deg2rad(45.0)

        t_matrix = self.processor._create_ctrs_to_uen_transform(lat_rad, lon_rad)

        self.assertEqual(t_matrix.shape, (3, 3))
        # Verify orthogonality (transformation matrix should be orthogonal)
        identity = np.dot(t_matrix, t_matrix.T)
        npt.assert_allclose(identity, np.eye(3), rtol=1e-12, atol=1e-15)

    def test_view_plane_vector_calculation(self):
        """Test view plane vector calculations with unit vector validation."""
        bhat_uen = np.array([0.5, 0.6, 0.8])  # Example boresight in UEN

        v_uen, x_uen = self.processor._calculate_view_plane_vectors(bhat_uen)

        self.assertEqual(v_uen.shape, (3,))
        self.assertEqual(x_uen.shape, (3,))

        # Check that vectors are unit vectors
        self.assertLess(abs(np.linalg.norm(v_uen) - 1.0), 1e-10)
        self.assertLess(abs(np.linalg.norm(x_uen) - 1.0), 1e-10)

        # Check orthogonality
        self.assertLess(abs(np.dot(v_uen, x_uen)), 1e-10)


class TestCorrelationFiltering(unittest.TestCase):
    """Test correlation-based filtering functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _create_test_config()
        self.processor = ErrorStatsProcessor(config=self.config)

    def _create_test_data_with_correlation(self, n_measurements=10):
        """Create test dataset with correlation values."""
        # Create proper transformation matrices
        t_matrices = np.zeros((3, 3, n_measurements))
        for i in range(n_measurements):
            t_matrices[:, :, i] = np.eye(3)  # Use identity for simplicity

        return xr.Dataset({
            'lat_error_deg': (['measurement'], np.random.uniform(-0.01, 0.01, n_measurements)),
            'lon_error_deg': (['measurement'], np.random.uniform(-0.01, 0.01, n_measurements)),
            'riss_ctrs': (['measurement', 'xyz'], np.random.uniform(6e6, 7e6, (n_measurements, 3))),
            'bhat_hs': (['measurement', 'xyz'], np.tile([0, 0.05, 0.998], (n_measurements, 1))),
            't_hs2ctrs': (['xyz_from', 'xyz_to', 'measurement'], t_matrices),
            'cp_lat_deg': (['measurement'], np.random.uniform(-60, 60, n_measurements)),
            'cp_lon_deg': (['measurement'], np.random.uniform(-180, 180, n_measurements)),
            'cp_alt': (['measurement'], np.random.uniform(0, 1000, n_measurements)),
            'correlation': (['measurement'], np.linspace(0.2, 1.0, n_measurements))
        }, coords={
            'measurement': np.arange(n_measurements),
            'xyz': ['x', 'y', 'z'],
            'xyz_from': ['x', 'y', 'z'],
            'xyz_to': ['x', 'y', 'z']
        })

    def test_correlation_config_default(self):
        """Test that minimum_correlation defaults to None."""
        config = _create_test_config()
        self.assertIsNone(config.minimum_correlation)

    def test_correlation_config_custom(self):
        """Test setting custom correlation threshold."""
        config = _create_test_config(minimum_correlation=0.6)
        self.assertEqual(config.minimum_correlation, 0.6)

    def test_no_filtering_when_threshold_is_none(self):
        """Test that no filtering occurs when minimum_correlation is None."""
        test_data = self._create_test_data_with_correlation(n_measurements=10)

        config = _create_test_config(minimum_correlation=None)
        processor = ErrorStatsProcessor(config=config)

        results = processor.process_geolocation_errors(test_data)

        # All measurements should remain
        self.assertEqual(len(results.measurement), 10)
        self.assertNotIn('correlation_filtering_applied', results.attrs)

    def test_filtering_with_threshold_05(self):
        """Test filtering with correlation threshold of 0.5."""
        test_data = self._create_test_data_with_correlation(n_measurements=10)
        # Correlation values: [0.2, 0.289, 0.378, 0.467, 0.556, 0.644, 0.733, 0.822, 0.911, 1.0]

        config = _create_test_config(minimum_correlation=0.5)
        processor = ErrorStatsProcessor(config=config)

        results = processor.process_geolocation_errors(test_data)

        # Should have 6 measurements with correlation >= 0.5
        self.assertEqual(len(results.measurement), 6)
        self.assertIn('correlation_filtering_applied', results.attrs)
        self.assertTrue(results.attrs['correlation_filtering_applied'])
        self.assertEqual(results.attrs['minimum_correlation_threshold'], 0.5)

    def test_filtering_with_threshold_08(self):
        """Test filtering with stricter correlation threshold of 0.8."""
        test_data = self._create_test_data_with_correlation(n_measurements=10)

        config = _create_test_config(minimum_correlation=0.8)
        processor = ErrorStatsProcessor(config=config)

        results = processor.process_geolocation_errors(test_data)

        # Should have 3 measurements with correlation >= 0.8
        self.assertEqual(len(results.measurement), 3)
        self.assertTrue(results.attrs['correlation_filtering_applied'])

    def test_filtering_with_alternative_variable_names(self):
        """Test that filtering works with alternative correlation variable names."""
        for var_name in ['ccv', 'im_ccv']:
            test_data = self._create_test_data_with_correlation(n_measurements=10)
            # Rename correlation variable
            test_data = test_data.rename({'correlation': var_name})

            config = _create_test_config(minimum_correlation=0.5)
            processor = ErrorStatsProcessor(config=config)

            results = processor.process_geolocation_errors(test_data)

            # Should still filter correctly
            self.assertEqual(len(results.measurement), 6)

    def test_missing_correlation_variable_logs_warning(self):
        """Test graceful handling when correlation variable is missing."""
        test_data = self._create_test_data_with_correlation(n_measurements=10)
        # Remove correlation variable
        test_data = test_data.drop_vars('correlation')

        config = _create_test_config(minimum_correlation=0.5)
        processor = ErrorStatsProcessor(config=config)

        # Should process without filtering (and log warning)
        results = processor.process_geolocation_errors(test_data)

        # All measurements should remain
        self.assertEqual(len(results.measurement), 10)

    def test_all_measurements_filtered_raises_error(self):
        """Test that filtering all measurements raises an error."""
        test_data = self._create_test_data_with_correlation(n_measurements=10)

        # Set impossibly high threshold
        config = _create_test_config(minimum_correlation=1.1)
        processor = ErrorStatsProcessor(config=config)

        with self.assertRaises(ValueError) as context:
            processor.process_geolocation_errors(test_data)
        self.assertIn("No measurements remaining", str(context.exception))

    def test_correlation_values_preserved_in_output(self):
        """Test that correlation values are preserved in output dataset."""
        test_data = self._create_test_data_with_correlation(n_measurements=10)

        config = _create_test_config(minimum_correlation=0.5)
        processor = ErrorStatsProcessor(config=config)

        results = processor.process_geolocation_errors(test_data)

        # Correlation should be in output
        self.assertIn('correlation', results.data_vars)
        # All remaining correlations should be >= 0.5
        self.assertTrue((results.correlation >= 0.5).all())


class TestNetCDFReprocessing(unittest.TestCase):
    """Test NetCDF reprocessing capabilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _create_test_config()
        self.processor = ErrorStatsProcessor(config=self.config)
        self.test_dir = Path(__file__).parent.parent / "test_outputs"
        self.test_dir.mkdir(exist_ok=True, parents=True)

    def tearDown(self):
        """Clean up test files."""
        # Optional: remove test files after tests
        pass

    def _create_test_netcdf(self, filepath, include_correlation=True, n_measurements=10):
        """Create a test NetCDF file."""
        t_matrices = np.zeros((3, 3, n_measurements))
        for i in range(n_measurements):
            t_matrices[:, :, i] = np.eye(3)

        data_dict = {
            'lat_error_deg': (['measurement'], np.random.uniform(-0.01, 0.01, n_measurements)),
            'lon_error_deg': (['measurement'], np.random.uniform(-0.01, 0.01, n_measurements)),
            'riss_ctrs': (['measurement', 'xyz'], np.random.uniform(6e6, 7e6, (n_measurements, 3))),
            'bhat_hs': (['measurement', 'xyz'], np.tile([0, 0.05, 0.998], (n_measurements, 1))),
            't_hs2ctrs': (['xyz_from', 'xyz_to', 'measurement'], t_matrices),
            'cp_lat_deg': (['measurement'], np.random.uniform(-60, 60, n_measurements)),
            'cp_lon_deg': (['measurement'], np.random.uniform(-180, 180, n_measurements)),
            'cp_alt': (['measurement'], np.random.uniform(0, 1000, n_measurements))
        }

        if include_correlation:
            data_dict['correlation'] = (['measurement'], np.linspace(0.2, 1.0, n_measurements))

        test_data = xr.Dataset(
            data_dict,
            coords={
                'measurement': np.arange(n_measurements),
                'xyz': ['x', 'y', 'z'],
                'xyz_from': ['x', 'y', 'z'],
                'xyz_to': ['x', 'y', 'z']
            }
        )

        test_data.to_netcdf(filepath)
        return filepath

    def test_process_from_netcdf_basic(self):
        """Test basic NetCDF loading and reprocessing."""
        netcdf_path = self.test_dir / "test_basic.nc"
        self._create_test_netcdf(netcdf_path, n_measurements=10)

        results = self.processor.process_from_netcdf(netcdf_path)

        self.assertIsInstance(results, xr.Dataset)
        self.assertIn('nadir_equiv_total_error_m', results.data_vars)
        self.assertIn('reprocessed_from', results.attrs)
        self.assertIn('reprocessing_date', results.attrs)
        self.assertEqual(str(netcdf_path), results.attrs['reprocessed_from'])

    def test_process_from_netcdf_with_correlation_override(self):
        """Test NetCDF reprocessing with correlation threshold override."""
        netcdf_path = self.test_dir / "test_correlation_override.nc"
        self._create_test_netcdf(netcdf_path, include_correlation=True, n_measurements=10)

        # Reprocess with different thresholds
        results_50 = self.processor.process_from_netcdf(netcdf_path, minimum_correlation=0.5)
        results_80 = self.processor.process_from_netcdf(netcdf_path, minimum_correlation=0.8)

        # Different thresholds should yield different numbers of measurements
        self.assertEqual(len(results_50.measurement), 6)
        self.assertEqual(len(results_80.measurement), 3)

        # Check metadata
        self.assertEqual(results_50.attrs['correlation_threshold_override'], 0.5)
        self.assertEqual(results_80.attrs['correlation_threshold_override'], 0.8)

    def test_process_from_netcdf_file_not_found(self):
        """Test error handling for non-existent NetCDF file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_from_netcdf("nonexistent_file.nc")

    def test_process_from_netcdf_missing_required_variables(self):
        """Test error handling for NetCDF missing required variables."""
        netcdf_path = self.test_dir / "test_incomplete.nc"

        # Create incomplete dataset
        incomplete_data = xr.Dataset({
            'lat_error_deg': (['measurement'], [0.001, 0.002]),
            'lon_error_deg': (['measurement'], [0.001, 0.002])
        }, coords={'measurement': [0, 1]})
        incomplete_data.to_netcdf(netcdf_path)

        with self.assertRaises(ValueError) as context:
            self.processor.process_from_netcdf(netcdf_path)
        self.assertIn("missing required variables", str(context.exception))

    def test_process_from_netcdf_preserves_original_config(self):
        """Test that reprocessing with override preserves original config."""
        netcdf_path = self.test_dir / "test_config_preservation.nc"
        self._create_test_netcdf(netcdf_path, include_correlation=True, n_measurements=10)

        # Create processor with initial threshold
        config = _create_test_config(minimum_correlation=0.3)
        processor = ErrorStatsProcessor(config=config)

        # Reprocess with override
        results = processor.process_from_netcdf(netcdf_path, minimum_correlation=0.7)

        # Original config should be restored
        self.assertEqual(processor.config.minimum_correlation, 0.3)

    def test_process_from_netcdf_without_correlation(self):
        """Test reprocessing NetCDF file without correlation data."""
        netcdf_path = self.test_dir / "test_no_correlation.nc"
        self._create_test_netcdf(netcdf_path, include_correlation=False, n_measurements=10)

        # Should process without filtering (log warning)
        config = _create_test_config(minimum_correlation=0.5)
        processor = ErrorStatsProcessor(config=config)

        results = processor.process_from_netcdf(netcdf_path)

        # All measurements should remain
        self.assertEqual(len(results.measurement), 10)

    def test_iterative_reprocessing_workflow(self):
        """Test realistic workflow of iterative threshold testing."""
        netcdf_path = self.test_dir / "test_iterative.nc"
        self._create_test_netcdf(netcdf_path, include_correlation=True, n_measurements=20)

        # Test multiple thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = {}

        for threshold in thresholds:
            result = self.processor.process_from_netcdf(netcdf_path, minimum_correlation=threshold)
            results[threshold] = {
                'n_measurements': len(result.measurement),
                'pass_rate': result.attrs['percent_below_250m']
            }

        # Higher thresholds should have fewer measurements
        n_measurements = [results[t]['n_measurements'] for t in thresholds]
        self.assertEqual(n_measurements, sorted(n_measurements, reverse=True))


if __name__ == '__main__':
    unittest.main()
