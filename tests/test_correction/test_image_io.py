"""Unit tests for image_io module.

Tests for MATLAB, HDF, and NetCDF I/O functions.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from curryer.correction.data_structures import ImageGrid, NamedImageGrid
from curryer.correction.image_io import (
    load_gcp_chip_from_hdf,
    load_image_grid_from_mat,
    load_image_grid_from_netcdf,
    load_named_image_grid,
    save_image_grid,
    save_image_grid_to_netcdf,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(rows: int = 10, cols: int = 10, with_height: bool = False) -> ImageGrid:
    """Return a small deterministic ImageGrid for use in tests."""
    rng = np.random.default_rng(0)
    data = rng.random((rows, cols))
    lat = np.linspace(38.0, 39.0, rows)
    lon = np.linspace(-116.0, -115.0, cols)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    h = np.zeros((rows, cols)) if with_height else None
    return ImageGrid(data=data, lat=lat_grid, lon=lon_grid, h=h)


# ---------------------------------------------------------------------------
# save_image_grid / load round-trips
# ---------------------------------------------------------------------------


class TestImageGridSaveLoad:
    """Test save/load round-trips for ImageGrid."""

    def test_netcdf_round_trip(self):
        """Test save and load ImageGrid from NetCDF."""
        rng = np.random.default_rng(0)
        data = rng.random((50, 50))
        lat = np.linspace(38.0, 39.0, 50)
        lon = np.linspace(-116.0, -115.0, 50)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        original_grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="netcdf")

            loaded_grid = load_image_grid_from_netcdf(tmp_path)

            np.testing.assert_array_almost_equal(loaded_grid.data, original_grid.data)
            np.testing.assert_array_almost_equal(loaded_grid.lat, original_grid.lat)
            np.testing.assert_array_almost_equal(loaded_grid.lon, original_grid.lon)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_netcdf_round_trip_with_height(self):
        """Height field is preserved through the xarray-based NetCDF round-trip."""
        original_grid = _make_grid(with_height=True)

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="netcdf")

            loaded_grid = load_image_grid_from_netcdf(tmp_path)

            assert loaded_grid.h is not None
            np.testing.assert_array_almost_equal(loaded_grid.h, original_grid.h)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_mat_round_trip(self):
        """Test save and load ImageGrid from MATLAB .mat file."""
        rng = np.random.default_rng(0)
        data = rng.random((50, 50))
        lat = np.linspace(38.0, 39.0, 50)
        lon = np.linspace(-116.0, -115.0, 50)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        original_grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="mat")

            loaded_grid = load_image_grid_from_mat(tmp_path, key="GCP")

            np.testing.assert_array_almost_equal(loaded_grid.data, original_grid.data)
            np.testing.assert_array_almost_equal(loaded_grid.lat, original_grid.lat)
            np.testing.assert_array_almost_equal(loaded_grid.lon, original_grid.lon)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_mat_round_trip_with_height(self):
        """Height field is preserved through the MATLAB .mat round-trip."""
        original_grid = _make_grid(with_height=True)

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="mat")

            loaded_grid = load_image_grid_from_mat(tmp_path, key="GCP")

            assert loaded_grid.h is not None
            np.testing.assert_array_almost_equal(loaded_grid.h, original_grid.h)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_save_with_metadata(self):
        """Test saving ImageGrid with custom metadata."""
        rng = np.random.default_rng(0)
        data = rng.random((10, 10))
        lat = np.linspace(38.0, 39.0, 10)
        lon = np.linspace(-116.0, -115.0, 10)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        metadata = {
            "source": "test_chip.hdf",
            "mission": "test",
            "creation_date": "2025-01-22",
        }

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, grid, format="netcdf", metadata=metadata)

            import xarray as xr

            ds = xr.open_dataset(tmp_path)
            assert "source" in ds.attrs
            assert ds.attrs["source"] == "test_chip.hdf"
            ds.close()
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        rng = np.random.default_rng(0)
        data = rng.random((10, 10))
        lat = np.linspace(38.0, 39.0, 10)
        lon = np.linspace(-116.0, -115.0, 10)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        with tempfile.NamedTemporaryFile(suffix=".xyz") as tmp:
            tmp_path = Path(tmp.name)
            with pytest.raises(ValueError, match="Unsupported format"):
                save_image_grid(tmp_path, grid, format="invalid_format")


# ---------------------------------------------------------------------------
# CF-1.8 save_image_grid_to_netcdf / load_image_grid_from_netcdf pair
# ---------------------------------------------------------------------------


class TestNetCDF4DirectIO:
    """Test the lower-level CF-1.8 netCDF4 save/load pair."""

    def test_round_trip_regular_grid(self):
        """save_image_grid_to_netcdf + load_image_grid_from_netcdf: regular grid."""
        original_grid = _make_grid(rows=20, cols=25)

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid_to_netcdf(tmp_path, original_grid)

            loaded_grid = load_image_grid_from_netcdf(tmp_path)

            assert loaded_grid.data.shape == original_grid.data.shape
            np.testing.assert_array_almost_equal(loaded_grid.data, original_grid.data)
            # Coordinates reconstructed from 1-D arrays via meshgrid — values must match
            np.testing.assert_array_almost_equal(loaded_grid.lat, original_grid.lat)
            np.testing.assert_array_almost_equal(loaded_grid.lon, original_grid.lon)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_round_trip_with_height(self):
        """Height is preserved through the netCDF4 round-trip."""
        original_grid = _make_grid(rows=10, cols=10, with_height=True)

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid_to_netcdf(tmp_path, original_grid)

            loaded_grid = load_image_grid_from_netcdf(tmp_path)

            assert loaded_grid.h is not None
            np.testing.assert_array_almost_equal(loaded_grid.h, original_grid.h)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_round_trip_with_metadata(self):
        """Metadata attributes are written to the file."""
        grid = _make_grid()
        metadata = {"mission": "CLARREO", "band": "red"}

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid_to_netcdf(tmp_path, grid, metadata=metadata)

            import xarray as xr

            ds = xr.open_dataset(tmp_path)
            assert ds.attrs.get("mission") == "CLARREO"
            assert ds.attrs.get("band") == "red"
            ds.close()
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_round_trip_irregular_grid(self):
        """2-D (irregular) coordinate arrays are stored and recovered correctly."""
        rng = np.random.default_rng(1)
        nrows, ncols = 8, 10
        data = rng.random((nrows, ncols))
        # Distorted grid — coordinates are NOT separable
        lat_base = np.linspace(38.0, 39.0, nrows)
        lon_base = np.linspace(-116.0, -115.0, ncols)
        lon_grid, lat_grid = np.meshgrid(lon_base, lat_base)
        lat_grid += 0.05 * rng.standard_normal((nrows, ncols))
        lon_grid += 0.05 * rng.standard_normal((nrows, ncols))

        original_grid = ImageGrid(data=data, lat=lat_grid, lon=lon_grid)

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid_to_netcdf(tmp_path, original_grid)

            loaded_grid = load_image_grid_from_netcdf(tmp_path)

            assert loaded_grid.data.shape == original_grid.data.shape
            np.testing.assert_array_almost_equal(loaded_grid.data, original_grid.data)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_file_raises(self):
        """load_image_grid_from_netcdf raises FileNotFoundError for absent files."""
        with pytest.raises(FileNotFoundError):
            load_image_grid_from_netcdf(Path("does_not_exist.nc"))


# ---------------------------------------------------------------------------
# HDF file loading
# ---------------------------------------------------------------------------


class TestHDFLoading:
    """Test HDF file loading."""

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_gcp_chip_from_hdf(Path("nonexistent.hdf"))

    def test_hdf5_round_trip(self):
        """Load a synthetic HDF5 file via the h5py fallback path."""
        h5py = pytest.importorskip("h5py")

        rng = np.random.default_rng(0)
        nrows, ncols = 12, 15
        band = rng.random((nrows, ncols))
        ecef_x = rng.random((nrows, ncols)) * 1e6
        ecef_y = rng.random((nrows, ncols)) * 1e6
        ecef_z = rng.random((nrows, ncols)) * 1e6

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as hdf:
                hdf.create_dataset("Band_1", data=band)
                hdf.create_dataset("ECR_x_coordinate_array", data=ecef_x)
                hdf.create_dataset("ECR_y_coordinate_array", data=ecef_y)
                hdf.create_dataset("ECR_z_coordinate_array", data=ecef_z)

            loaded_band, loaded_x, loaded_y, loaded_z = load_gcp_chip_from_hdf(tmp_path)

            assert loaded_band.shape == (nrows, ncols)
            np.testing.assert_array_almost_equal(loaded_band, band)
            np.testing.assert_array_almost_equal(loaded_x, ecef_x)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_shape_validation(self):
        """Shape mismatch across HDF5 datasets raises ValueError."""
        h5py = pytest.importorskip("h5py")

        rng = np.random.default_rng(0)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as hdf:
                hdf.create_dataset("Band_1", data=rng.random((10, 10)))
                hdf.create_dataset("ECR_x_coordinate_array", data=rng.random((10, 10)))
                hdf.create_dataset("ECR_y_coordinate_array", data=rng.random((10, 10)))
                # Deliberately wrong shape for Z
                hdf.create_dataset("ECR_z_coordinate_array", data=rng.random((5, 10)))

            with pytest.raises(ValueError, match="shape mismatch"):
                load_gcp_chip_from_hdf(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load_image_grid_from_netcdf (xarray-based)
# ---------------------------------------------------------------------------


class TestNetCDFLoading:
    """Test NetCDF file loading."""

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_grid_from_netcdf(Path("nonexistent.nc"))

    def test_round_trip_with_height(self):
        """Height variable is loaded when present in file."""
        original_grid = _make_grid(with_height=True)

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="netcdf")

            loaded_grid = load_image_grid_from_netcdf(tmp_path)

            assert loaded_grid.h is not None
            np.testing.assert_array_almost_equal(loaded_grid.h, original_grid.h)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_variable_raises(self):
        """Missing band_data variable in NetCDF raises OSError."""
        import xarray as xr

        nrows, ncols = 5, 5
        lat = np.linspace(38.0, 39.0, nrows)
        lon = np.linspace(-116.0, -115.0, ncols)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        # Build a file that is valid NetCDF but lacks "band_data"
        ds = xr.Dataset(
            {"lat": (["y", "x"], lat_grid), "lon": (["y", "x"], lon_grid)},
            coords={"y": np.arange(nrows), "x": np.arange(ncols)},
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ds.to_netcdf(tmp_path)

            with pytest.raises(KeyError, match="band_data"):
                load_image_grid_from_netcdf(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load_image_grid_from_mat
# ---------------------------------------------------------------------------


class TestMATLoading:
    """Test MATLAB file loading."""

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_grid_from_mat(Path("nonexistent.mat"))

    def test_missing_key_raises(self):
        """KeyError is raised when the requested struct key is absent."""
        from scipy.io import savemat

        rng = np.random.default_rng(0)
        data = rng.random((5, 5))

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save under a different key than will be requested
            savemat(str(tmp_path), {"other_key": {"data": data}})

            with pytest.raises(KeyError, match="subimage"):
                load_image_grid_from_mat(tmp_path, key="subimage")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_as_named_returns_named_image_grid(self):
        """as_named=True returns a NamedImageGrid with the correct name."""
        original_grid = _make_grid()

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="mat")

            loaded = load_image_grid_from_mat(tmp_path, key="GCP", as_named=True)

            assert isinstance(loaded, NamedImageGrid)
            assert loaded.name is not None
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load_named_image_grid
# ---------------------------------------------------------------------------


class TestLoadNamedImageGrid:
    """Tests for the format-agnostic load_named_image_grid dispatcher."""

    def test_mat_returns_named_image_grid(self):
        """Loading a .mat file returns a NamedImageGrid with name set to the file path."""
        original_grid = _make_grid()

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="mat")

            result = load_named_image_grid(tmp_path, mat_key="GCP")

            assert isinstance(result, NamedImageGrid)
            assert result.name == str(tmp_path)
            np.testing.assert_array_almost_equal(result.data, original_grid.data)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_netcdf_returns_named_image_grid(self):
        """Loading a .nc file returns a NamedImageGrid with name set to the file path."""
        original_grid = _make_grid()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_image_grid(tmp_path, original_grid, format="netcdf")

            result = load_named_image_grid(tmp_path)

            assert isinstance(result, NamedImageGrid)
            assert result.name == str(tmp_path)
            np.testing.assert_array_almost_equal(result.data, original_grid.data)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_unknown_extension_raises_value_error(self):
        """An unrecognised file extension raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognised file extension"):
            load_named_image_grid(Path("some_file.tif"))

    def test_missing_file_raises(self):
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_named_image_grid(Path("nonexistent_file.nc"))


# ---------------------------------------------------------------------------
# load_image_grid_from_netcdf — legacy variable names and 1-D broadcast edge cases
# ---------------------------------------------------------------------------


class TestNetCDFLoadingEdgeCases:
    """Tests for legacy variable-name fallbacks and partial 1-D coordinate broadcasting."""

    def test_legacy_data_variable_name(self):
        """Files using 'data' instead of 'band_data' are loaded via the legacy fallback."""
        import xarray as xr

        rng = np.random.default_rng(1)
        nrows, ncols = 5, 6
        data = rng.random((nrows, ncols))
        lat = np.linspace(38.0, 39.0, nrows)
        lon = np.linspace(-116.0, -115.0, ncols)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        ds = xr.Dataset(
            {
                "data": (["y", "x"], data),
                "lat": (["y", "x"], lat_grid),
                "lon": (["y", "x"], lon_grid),
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ds.to_netcdf(tmp_path)
            grid = load_image_grid_from_netcdf(tmp_path)
            np.testing.assert_array_almost_equal(grid.data, data)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_legacy_latitude_longitude_variable_names(self):
        """Files using 'latitude'/'longitude' instead of 'lat'/'lon' are handled."""
        import xarray as xr

        rng = np.random.default_rng(2)
        nrows, ncols = 5, 6
        data = rng.random((nrows, ncols))
        lat = np.linspace(38.0, 39.0, nrows)
        lon = np.linspace(-116.0, -115.0, ncols)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        ds = xr.Dataset(
            {
                "band_data": (["y", "x"], data),
                "latitude": (["y", "x"], lat_grid),
                "longitude": (["y", "x"], lon_grid),
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ds.to_netcdf(tmp_path)
            grid = load_image_grid_from_netcdf(tmp_path)
            np.testing.assert_array_almost_equal(grid.data, data)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_1d_lat_only_is_broadcast(self):
        """A file with a 2-D lon but 1-D lat broadcasts lat to the full grid shape."""
        import xarray as xr

        rng = np.random.default_rng(3)
        nrows, ncols = 5, 6
        data = rng.random((nrows, ncols))
        lat_1d = np.linspace(38.0, 39.0, nrows)
        lon_2d = np.tile(np.linspace(-116.0, -115.0, ncols), (nrows, 1))

        ds = xr.Dataset(
            {
                "band_data": (["y", "x"], data),
                "lat": (["y"], lat_1d),
                "lon": (["y", "x"], lon_2d),
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ds.to_netcdf(tmp_path)
            grid = load_image_grid_from_netcdf(tmp_path)
            assert grid.lat.shape == (nrows, ncols)
            assert grid.lon.shape == (nrows, ncols)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_1d_lon_only_is_broadcast(self):
        """A file with a 2-D lat but 1-D lon broadcasts lon to the full grid shape."""
        import xarray as xr

        rng = np.random.default_rng(4)
        nrows, ncols = 5, 6
        data = rng.random((nrows, ncols))
        lat_2d = np.repeat(np.linspace(38.0, 39.0, nrows)[:, np.newaxis], ncols, axis=1)
        lon_1d = np.linspace(-116.0, -115.0, ncols)

        ds = xr.Dataset(
            {
                "band_data": (["y", "x"], data),
                "lat": (["y", "x"], lat_2d),
                "lon": (["x"], lon_1d),
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ds.to_netcdf(tmp_path)
            grid = load_image_grid_from_netcdf(tmp_path)
            assert grid.lat.shape == (nrows, ncols)
            assert grid.lon.shape == (nrows, ncols)
        finally:
            tmp_path.unlink(missing_ok=True)
