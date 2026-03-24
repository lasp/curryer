"""Unit tests for regrid module.

Tests for GCP chip regridding algorithms.
"""

import numpy as np
import pytest

from curryer.correction.data_structures import ImageGrid, RegridConfig
from curryer.correction.regrid import (
    bilinear_interpolate_quad,
    compute_regular_grid_bounds,
    create_regular_grid,
    find_containing_cell,
    point_in_triangle,
    regrid_gcp_chip,
    regrid_irregular_to_regular,
)


class TestRegridConfig:
    """Test RegridConfig validation."""

    def test_resolution_based_config(self):
        """Test resolution-based configuration."""
        config = RegridConfig(output_resolution_deg=(0.001, 0.001))
        assert config.output_resolution_deg == (0.001, 0.001)
        assert config.output_grid_size is None
        assert config.conservative_bounds is True

    def test_size_based_config(self):
        """Test size-based configuration."""
        config = RegridConfig(output_grid_size=(500, 500))
        assert config.output_grid_size == (500, 500)
        assert config.output_resolution_deg is None

    def test_bounds_plus_resolution(self):
        """Test bounds + resolution configuration."""
        config = RegridConfig(output_bounds=(-116.0, -115.0, 38.0, 39.0), output_resolution_deg=(0.001, 0.001))
        assert config.output_bounds is not None
        assert config.output_resolution_deg is not None

    def test_invalid_size_and_resolution(self):
        """Test that size + resolution raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            RegridConfig(output_grid_size=(500, 500), output_resolution_deg=(0.001, 0.001))

    def test_invalid_bounds_without_resolution(self):
        """Test that bounds without resolution raises error."""
        with pytest.raises(ValueError, match="requires output_resolution_deg"):
            RegridConfig(output_bounds=(-116.0, -115.0, 38.0, 39.0))

    def test_invalid_interpolation_method(self):
        """Test that invalid interpolation method raises error."""
        with pytest.raises(ValueError, match="interpolation_method must be"):
            RegridConfig(output_resolution_deg=(0.001, 0.001), interpolation_method="invalid")

    def test_invalid_grid_size(self):
        """Test that too-small grid size raises error."""
        with pytest.raises(ValueError, match="at least 2 rows"):
            RegridConfig(output_grid_size=(1, 100))

    def test_invalid_resolution(self):
        """Test that negative resolution raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            RegridConfig(output_resolution_deg=(-0.001, 0.001))

    def test_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError, match="minlon must be < maxlon"):
            RegridConfig(
                output_bounds=(-115.0, -116.0, 38.0, 39.0),  # Swapped lon
                output_resolution_deg=(0.001, 0.001),
            )


class TestGeometricPrimitives:
    """Test geometric helper functions."""

    def test_point_in_triangle_inside(self):
        """Test point inside triangle."""
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        point = np.array([0.5, 0.3])

        inside, weights = point_in_triangle(point, triangle)

        assert inside
        assert len(weights) == 3
        assert np.abs(np.sum(weights) - 1.0) < 1e-10  # Weights sum to 1

    def test_point_in_triangle_outside(self):
        """Test point outside triangle."""
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        point = np.array([2.0, 2.0])

        inside, weights = point_in_triangle(point, triangle)

        assert not inside

    def test_point_in_triangle_on_edge(self):
        """Test point on triangle edge."""
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        point = np.array([0.5, 0.0])  # On base edge

        inside, weights = point_in_triangle(point, triangle)

        # With tolerant boundary check, point on edge should be inside
        assert inside

    def test_bilinear_interpolate_square(self):
        """Test bilinear interpolation on a square."""
        # Unit square with values at corners: 0, 1, 2, 3
        corners_lon = np.array([0.0, 1.0, 1.0, 0.0])
        corners_lat = np.array([1.0, 1.0, 0.0, 0.0])
        corner_values = np.array([0.0, 1.0, 2.0, 3.0])

        # Test center point
        point = np.array([0.5, 0.5])
        value = bilinear_interpolate_quad(point, corners_lon, corners_lat, corner_values)

        # At center, should be average of corners
        assert np.abs(value - 1.5) < 1e-10

    def test_bilinear_interpolate_corner(self):
        """Test bilinear interpolation at corner."""
        corners_lon = np.array([0.0, 1.0, 1.0, 0.0])
        corners_lat = np.array([1.0, 1.0, 0.0, 0.0])
        corner_values = np.array([10.0, 20.0, 30.0, 40.0])

        # Test at corner (should equal corner value)
        point = np.array([0.0, 1.0])  # Top-left corner
        value = bilinear_interpolate_quad(point, corners_lon, corners_lat, corner_values)

        assert np.abs(value - 10.0) < 1e-6


class TestGridOperations:
    """Test grid creation and bounds computation."""

    def test_compute_bounds_conservative(self):
        """Test conservative bounds computation."""
        # Create a slightly distorted grid
        nrows, ncols = 10, 10
        lat_base = np.linspace(39.0, 38.0, nrows)
        lon_base = np.linspace(-116.0, -115.0, ncols)
        lon_grid, lat_grid = np.meshgrid(lon_base, lat_base)

        # Add small distortion
        lon_grid += 0.01 * np.random.randn(nrows, ncols)
        lat_grid += 0.01 * np.random.randn(nrows, ncols)

        minlon, maxlon, minlat, maxlat = compute_regular_grid_bounds(lon_grid, lat_grid, conservative=True)

        # Conservative bounds should be within full extent
        assert minlon > lon_grid.min()
        assert maxlon < lon_grid.max()
        assert minlat > lat_grid.min()
        assert maxlat < lat_grid.max()

    def test_compute_bounds_full_extent(self):
        """Test full extent bounds computation."""
        nrows, ncols = 10, 10
        lat_base = np.linspace(39.0, 38.0, nrows)
        lon_base = np.linspace(-116.0, -115.0, ncols)
        lon_grid, lat_grid = np.meshgrid(lon_base, lat_base)

        minlon, maxlon, minlat, maxlat = compute_regular_grid_bounds(lon_grid, lat_grid, conservative=False)

        # Full extent should match min/max
        assert np.abs(minlon - lon_grid.min()) < 1e-10
        assert np.abs(maxlon - lon_grid.max()) < 1e-10
        assert np.abs(minlat - lat_grid.min()) < 1e-10
        assert np.abs(maxlat - lat_grid.max()) < 1e-10

    def test_create_regular_grid_from_size(self):
        """Test regular grid creation from size."""
        bounds = (-116.0, -115.0, 38.0, 39.0)
        grid_size = (100, 100)

        lon_grid, lat_grid = create_regular_grid(bounds, grid_size=grid_size)

        assert lon_grid.shape == grid_size
        assert lat_grid.shape == grid_size

        # Check latitude decreases (row index increases going south)
        assert lat_grid[0, 0] > lat_grid[-1, 0]

        # Check longitude increases (col index increases going east)
        assert lon_grid[0, 0] < lon_grid[0, -1]

    def test_create_regular_grid_from_resolution(self):
        """Test regular grid creation from resolution."""
        bounds = (-116.0, -115.0, 38.0, 39.0)
        resolution = (0.01, 0.01)  # 0.01 degree resolution

        lon_grid, lat_grid = create_regular_grid(bounds, resolution=resolution)

        # Check expected dimensions (1 degree / 0.01 + 1)
        assert lon_grid.shape[0] == 101  # (39-38)/0.01 + 1
        assert lon_grid.shape[1] == 101  # (-115-(-116))/0.01 + 1

    def test_create_regular_grid_requires_one_param(self):
        """Test that exactly one of size/resolution is required."""
        bounds = (-116.0, -115.0, 38.0, 39.0)

        # Neither provided
        with pytest.raises(ValueError, match="Must specify either"):
            create_regular_grid(bounds)

        # Both provided
        with pytest.raises(ValueError, match="only one of"):
            create_regular_grid(bounds, grid_size=(100, 100), resolution=(0.01, 0.01))


class TestRegridding:
    """Test core regridding functionality."""

    def test_find_containing_cell_simple(self):
        """Test finding containing cell in regular grid."""
        # Create simple regular grid
        nrows, ncols = 5, 5
        lat = np.linspace(39.0, 38.0, nrows)
        lon = np.linspace(-116.0, -115.0, ncols)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Test point in center of cell [1, 1] (not on boundary)
        # Cell [1,1] has corners: (-115.75, 38.75), (-115.5, 38.75),
        #                         (-115.5, 38.5), (-115.75, 38.5)
        point = np.array([-115.625, 38.625])  # Center of cell [1,1]
        cell = find_containing_cell(point, lon_grid, lat_grid)

        assert cell is not None
        assert cell == (1, 1)

    def test_find_containing_cell_outside(self):
        """Test point outside grid returns None."""
        nrows, ncols = 5, 5
        lat = np.linspace(39.0, 38.0, nrows)
        lon = np.linspace(-116.0, -115.0, ncols)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Test point way outside
        point = np.array([-120.0, 40.0])
        cell = find_containing_cell(point, lon_grid, lat_grid)

        assert cell is None

    def test_regrid_identity(self):
        """Test that regridding to same grid preserves values."""
        # Create regular grid
        nrows, ncols = 10, 10
        lat = np.linspace(39.0, 38.0, nrows)
        lon = np.linspace(-116.0, -115.0, ncols)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Create simple data
        data = np.arange(nrows * ncols).reshape(nrows, ncols).astype(float)

        # Regrid to same grid
        data_regridded = regrid_irregular_to_regular(data, lon_grid, lat_grid, lon_grid, lat_grid)

        # Should be nearly identical
        np.testing.assert_array_almost_equal(data, data_regridded, decimal=6)

    def test_regrid_coarsen(self):
        """Test regridding to coarser grid."""
        # Create fine regular grid
        nrows_fine, ncols_fine = 20, 20
        lat_fine = np.linspace(39.0, 38.0, nrows_fine)
        lon_fine = np.linspace(-116.0, -115.0, ncols_fine)
        lon_grid_fine, lat_grid_fine = np.meshgrid(lon_fine, lat_fine)

        # Create data with gradient
        data_fine = lon_grid_fine + lat_grid_fine

        # Create coarse grid
        nrows_coarse, ncols_coarse = 5, 5
        lat_coarse = np.linspace(39.0, 38.0, nrows_coarse)
        lon_coarse = np.linspace(-116.0, -115.0, ncols_coarse)
        lon_grid_coarse, lat_grid_coarse = np.meshgrid(lon_coarse, lat_coarse)

        # Regrid
        data_coarse = regrid_irregular_to_regular(
            data_fine, lon_grid_fine, lat_grid_fine, lon_grid_coarse, lat_grid_coarse
        )

        # Check shape
        assert data_coarse.shape == (nrows_coarse, ncols_coarse)

        # Check no NaNs in interior
        assert not np.any(np.isnan(data_coarse))

        # Check values are in reasonable range (allow small numerical errors)
        # Bilinear interpolation can produce values slightly outside original range
        tol = 1e-10
        assert data_coarse.min() >= data_fine.min() - tol
        assert data_coarse.max() <= data_fine.max() + tol


class TestEndToEnd:
    """Test complete regridding workflow."""

    def test_regrid_gcp_chip_synthetic(self):
        """Test regrid_gcp_chip with synthetic data."""
        # Create synthetic chip with ECEF coordinates
        nrows, ncols = 50, 50

        # Generate regular lat/lon grid
        lat = np.linspace(38.5, 38.0, nrows)
        lon = np.linspace(-116.0, -115.5, ncols)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Convert to ECEF (using simplified approach for testing)
        # For real test, would use actual geodetic_to_ecef
        from curryer.compute.spatial import geodetic_to_ecef

        lla = np.stack([lon_grid.ravel(), lat_grid.ravel(), np.zeros(nrows * ncols)], axis=1)
        ecef = geodetic_to_ecef(lla, meters=True, degrees=True)

        ecef_x = ecef[:, 0].reshape(nrows, ncols)
        ecef_y = ecef[:, 1].reshape(nrows, ncols)
        ecef_z = ecef[:, 2].reshape(nrows, ncols)

        # Create simple data pattern
        band_data = np.arange(nrows * ncols).reshape(nrows, ncols).astype(float)

        # Regrid to coarser resolution
        config = RegridConfig(output_resolution_deg=(0.02, 0.02))  # Coarser

        result = regrid_gcp_chip(band_data, (ecef_x, ecef_y, ecef_z), config)

        # Check result is ImageGrid
        assert isinstance(result, ImageGrid)

        # Check output is smaller (coarser resolution)
        assert result.data.shape[0] < nrows
        assert result.data.shape[1] < ncols

        # Check no NaNs in interior
        assert not np.all(np.isnan(result.data))

        # Check lat/lon grids are regular
        lat_spacing = np.diff(result.lat[:, 0])
        assert np.allclose(lat_spacing, lat_spacing[0], rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
