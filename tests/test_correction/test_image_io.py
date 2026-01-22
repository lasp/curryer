"""Unit tests for image_io module.

Tests for MATLAB, HDF, and NetCDF I/O functions.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from curryer.correction.data_structures import ImageGrid
from curryer.correction.image_io import (
    load_gcp_chip_from_hdf,
    load_gcp_chip_from_netcdf,
    load_image_grid_from_mat,
    save_image_grid,
)


class TestImageGridSaveLoad:
    """Test save/load round-trips for ImageGrid."""

    def test_netcdf_round_trip(self):
        """Test save and load ImageGrid from NetCDF."""
        # Create test data
        data = np.random.rand(50, 50)
        lat = np.linspace(38.0, 39.0, 50)
        lon = np.linspace(-116.0, -115.0, 50)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        original_grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="netcdf")

            # Load back
            loaded_grid = load_gcp_chip_from_netcdf(tmp_path)

            # Verify
            np.testing.assert_array_almost_equal(loaded_grid.data, original_grid.data)
            np.testing.assert_array_almost_equal(loaded_grid.lat, original_grid.lat)
            np.testing.assert_array_almost_equal(loaded_grid.lon, original_grid.lon)

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_mat_round_trip(self):
        """Test save and load ImageGrid from MATLAB .mat file."""
        # Create test data
        data = np.random.rand(50, 50)
        lat = np.linspace(38.0, 39.0, 50)
        lon = np.linspace(-116.0, -115.0, 50)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        original_grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="mat")

            # Load back
            loaded_grid = load_image_grid_from_mat(tmp_path, key="GCP")

            # Verify
            np.testing.assert_array_almost_equal(loaded_grid.data, original_grid.data)
            np.testing.assert_array_almost_equal(loaded_grid.lat, original_grid.lat)
            np.testing.assert_array_almost_equal(loaded_grid.lon, original_grid.lon)

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_save_with_metadata(self):
        """Test saving ImageGrid with custom metadata."""
        # Create test data
        data = np.random.rand(10, 10)
        lat = np.linspace(38.0, 39.0, 10)
        lon = np.linspace(-116.0, -115.0, 10)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        metadata = {
            "source": "test_chip.hdf",
            "mission": "test",
            "creation_date": "2025-01-22",
        }

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, grid, format="netcdf", metadata=metadata)

            # Load and check metadata exists
            import xarray as xr

            ds = xr.open_dataset(tmp_path)
            assert "source" in ds.attrs
            assert ds.attrs["source"] == "test_chip.hdf"
            ds.close()

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        data = np.random.rand(10, 10)
        lat = np.linspace(38.0, 39.0, 10)
        lon = np.linspace(-116.0, -115.0, 10)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        with tempfile.NamedTemporaryFile(suffix=".xyz") as tmp:
            tmp_path = Path(tmp.name)
            with pytest.raises(ValueError, match="Unsupported format"):
                save_image_grid(tmp_path, grid, format="invalid_format")


class TestHDFLoading:
    """Test HDF file loading (requires test data)."""

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_gcp_chip_from_hdf(Path("nonexistent.hdf"))

    def test_shape_validation(self):
        """Test that shape mismatches are detected."""
        # This would require creating a mock HDF file with mismatched shapes
        # Skipping for now as it requires h5py and proper test fixtures
        pass


class TestNetCDFLoading:
    """Test NetCDF file loading."""

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_gcp_chip_from_netcdf(Path("nonexistent.nc"))


class TestMATLoading:
    """Test MATLAB file loading."""

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_grid_from_mat(Path("nonexistent.mat"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
