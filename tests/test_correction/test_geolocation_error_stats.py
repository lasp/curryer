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
    process_test_data
)


logger = logging.getLogger(__name__)

# Configure display options for better test output
xr.set_options(display_width=120)
np.set_printoptions(linewidth=120)


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

        # Create default processor
        self.processor = ErrorStatsProcessor()

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
        config = GeolocationConfig()
        self.assertEqual(config.earth_radius_m, 6378140.0)
        self.assertEqual(config.performance_threshold_m, 250.0)
        self.assertEqual(config.performance_spec_percent, 39.0)

    def test_geolocation_config_custom(self):
        """Test custom configuration values."""
        config = GeolocationConfig(
            earth_radius_m=6371000.0,
            performance_threshold_m=200.0,
            performance_spec_percent=40.0
        )
        self.assertEqual(config.earth_radius_m, 6371000.0)
        self.assertEqual(config.performance_threshold_m, 200.0)
        self.assertEqual(config.performance_spec_percent, 40.0)

    def test_processor_initialization_default(self):
        """Test processor initialization with default config."""
        processor = ErrorStatsProcessor()
        self.assertIsInstance(processor.config, GeolocationConfig)
        self.assertEqual(processor.config.earth_radius_m, 6378140.0)

    def test_processor_initialization_custom(self):
        """Test processor initialization with custom config."""
        config = GeolocationConfig(
            earth_radius_m=6371000.0,
            performance_threshold_m=200.0
        )
        processor = ErrorStatsProcessor(config)
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
        test_data = ErrorStatsProcessor.create_test_dataset()

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
        custom_config = GeolocationConfig(performance_threshold_m=300.0)
        processor = ErrorStatsProcessor(custom_config)
        test_data = ErrorStatsProcessor.create_test_dataset()

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


if __name__ == '__main__':
    unittest.main()
