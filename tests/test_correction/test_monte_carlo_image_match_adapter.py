"""Tests for Monte Carlo image matching adapter functions.

Tests the minimal data conversion utilities that connect Monte Carlo
pipeline data to the image matching module.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from curryer.correction.monte_carlo_image_match_adapter import (
    geolocated_to_image_grid,
    extract_spacecraft_position_midframe,
    load_los_vectors_from_mat,
    load_optical_psf_from_mat,
    load_gcp_from_mat,
    get_gcp_center_location,
)
from curryer.correction.data_structures import ImageGrid, OpticalPSFEntry


class TestGeolocationToImageGrid(unittest.TestCase):
    """Test conversion from xarray geolocation dataset to ImageGrid."""

    def test_basic_conversion(self):
        """Test basic conversion with all required fields."""
        geo_ds = xr.Dataset({
            'latitude': (['frame', 'pixel'], np.random.randn(10, 50)),
            'longitude': (['frame', 'pixel'], np.random.randn(10, 50)),
            'altitude': (['frame', 'pixel'], np.random.randn(10, 50) * 1000),
        })

        img_grid = geolocated_to_image_grid(geo_ds)

        self.assertIsInstance(img_grid, ImageGrid)
        self.assertEqual(img_grid.lat.shape, (10, 50))
        self.assertEqual(img_grid.lon.shape, (10, 50))
        self.assertEqual(img_grid.h.shape, (10, 50))
        self.assertEqual(img_grid.data.shape, (10, 50))

    def test_missing_altitude(self):
        """Test conversion when altitude field is missing (should default to zeros)."""
        geo_ds = xr.Dataset({
            'latitude': (['frame', 'pixel'], np.random.randn(10, 50)),
            'longitude': (['frame', 'pixel'], np.random.randn(10, 50)),
        })

        img_grid = geolocated_to_image_grid(geo_ds)

        self.assertIsInstance(img_grid, ImageGrid)
        # Should default to zeros
        np.testing.assert_array_equal(img_grid.h, np.zeros((10, 50)))

    def test_with_radiance_data(self):
        """Test conversion with radiance data field."""
        radiance = np.random.rand(10, 50)
        geo_ds = xr.Dataset({
            'latitude': (['frame', 'pixel'], np.random.randn(10, 50)),
            'longitude': (['frame', 'pixel'], np.random.randn(10, 50)),
            'radiance': (['frame', 'pixel'], radiance),
        })

        img_grid = geolocated_to_image_grid(geo_ds)

        np.testing.assert_array_equal(img_grid.data, radiance)


class TestExtractSpacecraftPosition(unittest.TestCase):
    """Test extraction of spacecraft position from telemetry."""

    def test_standard_column_names(self):
        """Test with standard sc_pos_x/y/z column names."""
        tlm = pd.DataFrame({
            'sc_pos_x': [1000000.0, 1000100.0, 1000200.0],
            'sc_pos_y': [2000000.0, 2000100.0, 2000200.0],
            'sc_pos_z': [3000000.0, 3000100.0, 3000200.0],
            'time': [0.0, 1.0, 2.0],
        })

        pos = extract_spacecraft_position_midframe(tlm)

        self.assertEqual(pos.shape, (3,))
        # Should extract mid-frame (index 1)
        np.testing.assert_array_almost_equal(pos, [1000100.0, 2000100.0, 3000100.0])

    def test_alternative_column_names(self):
        """Test with alternative position_x/y/z column names."""
        tlm = pd.DataFrame({
            'position_x': [1.0e6, 2.0e6, 3.0e6],
            'position_y': [4.0e6, 5.0e6, 6.0e6],
            'position_z': [7.0e6, 8.0e6, 9.0e6],
        })

        pos = extract_spacecraft_position_midframe(tlm)

        self.assertEqual(pos.shape, (3,))
        np.testing.assert_array_almost_equal(pos, [2.0e6, 5.0e6, 8.0e6])

    def test_missing_columns_raises(self):
        """Test that missing position columns raises ValueError."""
        tlm = pd.DataFrame({
            'velocity_x': [1.0, 2.0, 3.0],
            'velocity_y': [4.0, 5.0, 6.0],
            'time': [0.0, 1.0, 2.0],
        })

        with self.assertRaises(ValueError) as context:
            extract_spacecraft_position_midframe(tlm)

        self.assertIn("Cannot find position columns", str(context.exception))


class TestLoadCalibrationData(unittest.TestCase):
    """Test loading calibration data from MATLAB files."""

    def setUp(self):
        """Set up paths to test data."""
        root_dir = Path(__file__).parent.parent.parent
        self.test_dir = root_dir / 'tests' / 'data' / 'clarreo' / 'image_match'

        # Check if test data exists
        if not self.test_dir.exists():
            self.skipTest(f"Test data directory not found: {self.test_dir}")

    def test_load_los_vectors(self):
        """Test loading LOS vectors from MATLAB file."""
        los_file = self.test_dir / 'b_HS.mat'

        if not los_file.exists():
            self.skipTest(f"LOS vector file not found: {los_file}")

        los_vectors = load_los_vectors_from_mat(los_file)

        # Should be (n_pixels, 3)
        self.assertEqual(los_vectors.ndim, 2)
        self.assertEqual(los_vectors.shape[1], 3)
        self.assertGreater(los_vectors.shape[0], 0)

    def test_load_optical_psf(self):
        """Test loading optical PSF from MATLAB file."""
        psf_file = self.test_dir / 'optical_PSF_675nm_upsampled.mat'

        if not psf_file.exists():
            self.skipTest(f"Optical PSF file not found: {psf_file}")

        psf_entries = load_optical_psf_from_mat(psf_file)

        self.assertIsInstance(psf_entries, list)
        self.assertGreater(len(psf_entries), 0)

        for entry in psf_entries:
            self.assertIsInstance(entry, OpticalPSFEntry)
            self.assertIsInstance(entry.data, np.ndarray)
            self.assertIsInstance(entry.x, np.ndarray)
            self.assertIsInstance(entry.field_angle, np.ndarray)

    def test_load_gcp(self):
        """Test loading GCP from MATLAB file."""
        gcp_file = self.test_dir / 'Dili_GCP.mat'

        if not gcp_file.exists():
            self.skipTest(f"GCP file not found: {gcp_file}")

        gcp = load_gcp_from_mat(gcp_file)

        self.assertIsInstance(gcp, ImageGrid)
        self.assertEqual(gcp.data.ndim, 2)
        self.assertEqual(gcp.lat.shape, gcp.data.shape)
        self.assertEqual(gcp.lon.shape, gcp.data.shape)

    def test_get_gcp_center_location(self):
        """Test extracting center location from GCP."""
        # Create a simple GCP with known center
        lat_grid = np.array([[10.0, 10.5, 11.0],
                             [10.0, 10.5, 11.0],
                             [10.0, 10.5, 11.0]])
        lon_grid = np.array([[120.0, 120.0, 120.0],
                             [121.0, 121.0, 121.0],
                             [122.0, 122.0, 122.0]])

        gcp = ImageGrid(
            data=np.ones_like(lat_grid),
            lat=lat_grid,
            lon=lon_grid,
        )

        center_lat, center_lon = get_gcp_center_location(gcp)

        # Center should be middle element (index 1, 1)
        self.assertAlmostEqual(center_lat, 10.5)
        self.assertAlmostEqual(center_lon, 121.0)


if __name__ == '__main__':
    unittest.main()
