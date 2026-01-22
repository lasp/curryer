"""Image I/O utilities for correction pipeline.

This module provides format-agnostic loading and saving of image data
used in the correction pipeline, including:
- MATLAB .mat files (legacy test data)
- HDF files (raw GCP chips)
- NetCDF files (regridded GCPs, outputs)

All functions work with ImageGrid and related data structures.
No dependencies on image matching or other correction algorithms.

@author: Brandon Stone, NASA Langley Research Center
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .data_structures import ImageGrid, NamedImageGrid, OpticalPSFEntry

logger = logging.getLogger(__name__)


# ============================================================================
# MATLAB File I/O (migrated from image_match.py)
# ============================================================================


def load_image_grid_from_mat(
    mat_file: Path, key: str = "subimage", name: str | None = None, as_named: bool = False
) -> ImageGrid | NamedImageGrid:
    """
    Load ImageGrid from MATLAB .mat file.

    Parameters
    ----------
    mat_file : Path
        Path to .mat file.
    key : str, default="subimage"
        MATLAB struct key (e.g., "subimage" for L1A, "GCP" for reference).
    name : str, optional
        Name for NamedImageGrid. Defaults to file path.
    as_named : bool, default=False
        If True, return NamedImageGrid; otherwise return ImageGrid.

    Returns
    -------
    ImageGrid or NamedImageGrid
        Loaded image grid with data, lat, lon, h fields.

    Raises
    ------
    FileNotFoundError
        If mat_file doesn't exist.
    KeyError
        If key not found in MATLAB file.

    Examples
    --------
    >>> # Load L1A subimage
    >>> l1a = load_image_grid_from_mat(Path("subimage.mat"), key="subimage")
    >>> # Load GCP reference
    >>> gcp = load_image_grid_from_mat(Path("gcp.mat"), key="GCP")
    """
    from scipy.io import loadmat

    if not mat_file.exists():
        raise FileNotFoundError(f"MATLAB file not found: {mat_file}")

    mat_data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

    if key not in mat_data:
        available_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {mat_file.name}. Available keys: {available_keys}")

    struct = mat_data[key]
    h = getattr(struct, "h", None)

    # Optimize: loadmat already returns numpy arrays, avoid redundant asarray() calls
    # ImageGrid.__post_init__ will handle final type conversion
    grid_kwargs = {
        "data": struct.data,
        "lat": struct.lat,
        "lon": struct.lon,
        "h": h,
    }

    if as_named:
        grid_kwargs["name"] = name or str(mat_file)
        return NamedImageGrid(**grid_kwargs)
    else:
        return ImageGrid(**grid_kwargs)


def load_optical_psf_from_mat(mat_file: Path, key: str = "PSF_struct_675nm") -> list[OpticalPSFEntry]:
    """
    Load optical PSF entries from MATLAB .mat file.

    Parameters
    ----------
    mat_file : Path
        Path to MATLAB file with PSF data.
    key : str, default="PSF_struct_675nm"
        Primary key to try for PSF data.

    Returns
    -------
    list[OpticalPSFEntry]
        Optical PSF samples with data, x, and field_angle arrays.

    Raises
    ------
    FileNotFoundError
        If mat_file doesn't exist.
    KeyError
        If no PSF data found with common key names.
    ValueError
        If PSF entries missing field angle attribute.
    """
    from scipy.io import loadmat

    if not mat_file.exists():
        raise FileNotFoundError(f"Optical PSF file not found: {mat_file}")

    mat_data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

    # Try common keys in order of preference
    for try_key in [key, "PSF_struct_675nm", "optical_PSF", "PSF"]:
        if try_key in mat_data:
            psf_struct = mat_data[try_key]
            psf_entries_raw = np.atleast_1d(psf_struct)

            psf_entries = []
            for entry in psf_entries_raw:
                # Handle both 'FA' and 'field_angle' attribute names
                # Check if attribute exists first to avoid NumPy array boolean ambiguity
                field_angle = getattr(entry, "FA", None)
                if field_angle is None or (
                    isinstance(field_angle, list | tuple | np.ndarray) and len(field_angle) == 0
                ):
                    # Fallback if FA is missing, None, or empty
                    field_angle = getattr(entry, "field_angle", None)

                if field_angle is None:
                    raise ValueError(f"PSF entry missing field angle attribute (tried 'FA' and 'field_angle')")

                # Optimize: loadmat already returns numpy arrays, OpticalPSFEntry.__post_init__ handles conversion
                # Use np.atleast_1d to ensure 1D arrays efficiently
                psf_entries.append(
                    OpticalPSFEntry(
                        data=entry.data,
                        x=np.atleast_1d(entry.x).ravel(),
                        field_angle=np.atleast_1d(field_angle).ravel(),
                    )
                )

            logger.info(f"Loaded {len(psf_entries)} optical PSF entries from {mat_file.name}")
            return psf_entries

    available_keys = [k for k in mat_data.keys() if not k.startswith("__")]
    raise KeyError(f"No PSF data found in {mat_file.name}. Available keys: {available_keys}")


def load_los_vectors_from_mat(mat_file: Path, key: str = "b_HS") -> np.ndarray:
    """
    Load line-of-sight vectors from MATLAB .mat file.

    Parameters
    ----------
    mat_file : Path
        Path to MATLAB file with LOS vectors.
    key : str, default="b_HS"
        Primary key to try for LOS data.

    Returns
    -------
    np.ndarray
        LOS unit vectors in instrument frame, shape (n_pixels, 3).

    Raises
    ------
    FileNotFoundError
        If mat_file doesn't exist.
    KeyError
        If no LOS vectors found with common key names.
    """
    from scipy.io import loadmat

    if not mat_file.exists():
        raise FileNotFoundError(f"LOS vector file not found: {mat_file}")

    mat_data = loadmat(str(mat_file))

    # Try common keys in order of preference
    for try_key in [key, "b_HS", "los_vectors", "pixel_vectors"]:
        if try_key in mat_data:
            los = mat_data[try_key]

            # Ensure shape is (n_pixels, 3) not (3, n_pixels)
            if los.shape[0] == 3 and los.shape[1] > 3:
                los = los.T

            logger.info(f"Loaded LOS vectors from {mat_file.name}: shape {los.shape}")
            return los

    available_keys = [k for k in mat_data.keys() if not k.startswith("__")]
    raise KeyError(f"No LOS vectors found in {mat_file.name}. Available keys: {available_keys}")


# ============================================================================
# HDF File I/O (new for regridding)
# ============================================================================


def load_gcp_chip_from_hdf(
    filepath: Path,
    band_name: str = "Band_1",
    coord_names: tuple[str, str, str] = (
        "ECR_x_coordinate_array",
        "ECR_y_coordinate_array",
        "ECR_z_coordinate_array",
    ),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw GCP chip data from HDF file (Landsat format).

    Parameters
    ----------
    filepath : Path
        Path to HDF file containing GCP chip data.
    band_name : str, default="Band_1"
        Name of the dataset containing band/radiometric data.
    coord_names : tuple[str, str, str], default=("ECR_x_coordinate_array", ...)
        Names of X, Y, Z coordinate datasets (ECEF coordinates in meters).

    Returns
    -------
    band_data : np.ndarray
        2D array of radiometric values, shape (nrows, ncols).
    ecef_x, ecef_y, ecef_z : np.ndarray
        2D arrays of ECEF coordinates in meters, each shape (nrows, ncols).

    Raises
    ------
    FileNotFoundError
        If filepath doesn't exist.
    KeyError
        If required datasets not found in HDF file.
    ValueError
        If array shapes are inconsistent.

    Examples
    --------
    >>> band, x, y, z = load_gcp_chip_from_hdf("LT08CHP.20140803.hdf")
    >>> band.shape
    (1400, 1400)
    """
    import h5py

    if not filepath.exists():
        raise FileNotFoundError(f"HDF file not found: {filepath}")

    try:
        with h5py.File(filepath, "r") as hdf:
            # Load band data
            if band_name not in hdf:
                available = list(hdf.keys())
                raise KeyError(f"Band '{band_name}' not found. Available datasets: {available}")

            band_data = np.array(hdf[band_name], dtype=np.float64)

            # Load ECEF coordinates
            coord_x_name, coord_y_name, coord_z_name = coord_names

            if coord_x_name not in hdf:
                raise KeyError(f"X coordinate '{coord_x_name}' not found in {filepath.name}")
            if coord_y_name not in hdf:
                raise KeyError(f"Y coordinate '{coord_y_name}' not found in {filepath.name}")
            if coord_z_name not in hdf:
                raise KeyError(f"Z coordinate '{coord_z_name}' not found in {filepath.name}")

            ecef_x = np.array(hdf[coord_x_name], dtype=np.float64)
            ecef_y = np.array(hdf[coord_y_name], dtype=np.float64)
            ecef_z = np.array(hdf[coord_z_name], dtype=np.float64)

    except OSError as e:
        raise OSError(f"Error reading HDF file {filepath}: {e}") from e

    # Validate shapes
    if not (band_data.shape == ecef_x.shape == ecef_y.shape == ecef_z.shape):
        raise ValueError(
            f"Array shape mismatch in {filepath.name}: "
            f"band={band_data.shape}, x={ecef_x.shape}, y={ecef_y.shape}, z={ecef_z.shape}"
        )

    logger.info(f"Loaded GCP chip from {filepath.name}: shape {band_data.shape}")

    return band_data, ecef_x, ecef_y, ecef_z


def load_gcp_chip_from_netcdf(
    filepath: Path,
    band_var: str = "band_data",
    lat_var: str = "lat",
    lon_var: str = "lon",
    height_var: str = "h",
) -> ImageGrid:
    """
    Load regridded GCP chip from NetCDF file.

    Parameters
    ----------
    filepath : Path
        Path to NetCDF file.
    band_var : str, default="band_data"
        Name of the band data variable.
    lat_var : str, default="lat"
        Name of the latitude variable.
    lon_var : str, default="lon"
        Name of the longitude variable.
    height_var : str, default="h"
        Name of the height variable (optional).

    Returns
    -------
    ImageGrid
        Loaded image grid with data, lat, lon, h fields.

    Raises
    ------
    FileNotFoundError
        If filepath doesn't exist.
    KeyError
        If required variables not found.

    Examples
    --------
    >>> gcp = load_gcp_chip_from_netcdf("regridded_chip.nc")
    >>> gcp.data.shape
    (420, 420)
    """
    import xarray as xr

    if not filepath.exists():
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    try:
        ds = xr.open_dataset(filepath)

        # Check required variables
        if band_var not in ds:
            raise KeyError(f"Band variable '{band_var}' not found in {filepath.name}")
        if lat_var not in ds:
            raise KeyError(f"Latitude variable '{lat_var}' not found in {filepath.name}")
        if lon_var not in ds:
            raise KeyError(f"Longitude variable '{lon_var}' not found in {filepath.name}")

        # Load data
        data = ds[band_var].values
        lat = ds[lat_var].values
        lon = ds[lon_var].values
        h = ds[height_var].values if height_var in ds else None

        ds.close()

    except Exception as e:
        raise OSError(f"Error reading NetCDF file {filepath}: {e}") from e

    logger.info(f"Loaded GCP chip from {filepath.name}: shape {data.shape}")

    return ImageGrid(data=data, lat=lat, lon=lon, h=h)


# ============================================================================
# Generic Image Savers (new for regridding + general use)
# ============================================================================


def save_image_grid(
    filepath: Path,
    image_grid: ImageGrid,
    format: str = "netcdf",
    metadata: dict | None = None,
) -> None:
    """
    Save ImageGrid to file (netcdf, mat, or geotiff).

    This function works with any ImageGrid, not just regridded GCPs.
    It can be used throughout the correction pipeline.

    Parameters
    ----------
    filepath : Path
        Output file path.
    image_grid : ImageGrid
        Image data with lat/lon coordinates.
    format : str, default="netcdf"
        Output format: 'netcdf', 'mat', or 'geotiff'.
    metadata : dict, optional
        Additional metadata to include in output.

    Raises
    ------
    ValueError
        If format is not supported.
    IOError
        If file cannot be written.

    Examples
    --------
    >>> save_image_grid("output.nc", regridded_gcp, format="netcdf")
    >>> save_image_grid("output.mat", regridded_gcp, format="mat")
    """
    format = format.lower()

    if format == "netcdf":
        _save_image_grid_netcdf(filepath, image_grid, metadata)
    elif format == "mat":
        _save_image_grid_mat(filepath, image_grid, metadata)
    elif format == "geotiff":
        _save_image_grid_geotiff(filepath, image_grid, metadata)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: 'netcdf', 'mat', 'geotiff'")

    logger.info(f"Saved ImageGrid to {filepath} (format: {format})")


def _save_image_grid_netcdf(filepath: Path, image_grid: ImageGrid, metadata: dict | None) -> None:
    """Save ImageGrid to NetCDF file (internal helper)."""
    import xarray as xr

    # Create xarray Dataset
    nrows, ncols = image_grid.data.shape

    ds = xr.Dataset(
        {
            "band_data": (["y", "x"], image_grid.data),
            "lat": (["y", "x"], image_grid.lat),
            "lon": (["y", "x"], image_grid.lon),
        },
        coords={
            "y": np.arange(nrows),
            "x": np.arange(ncols),
        },
    )

    # Add height if present
    if image_grid.h is not None:
        ds["h"] = (["y", "x"], image_grid.h)

    # Add metadata
    if metadata:
        ds.attrs.update(metadata)

    # Add standard attributes
    ds.attrs["title"] = "Regridded GCP Chip"
    ds.attrs["Conventions"] = "CF-1.8"

    # Add variable attributes
    ds["band_data"].attrs["long_name"] = "Band radiometric data"
    ds["band_data"].attrs["units"] = "digital_number"
    ds["lat"].attrs["long_name"] = "Latitude"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["long_name"] = "Longitude"
    ds["lon"].attrs["units"] = "degrees_east"

    if "h" in ds:
        ds["h"].attrs["long_name"] = "Height above ellipsoid"
        ds["h"].attrs["units"] = "meters"

    # Write to file
    try:
        ds.to_netcdf(filepath, engine="netcdf4")
    except Exception as e:
        raise OSError(f"Error writing NetCDF file {filepath}: {e}") from e


def _save_image_grid_mat(filepath: Path, image_grid: ImageGrid, metadata: dict | None) -> None:
    """Save ImageGrid to MATLAB .mat file (internal helper)."""
    from scipy.io import savemat

    # Prepare data structure
    mat_dict = {
        "data": image_grid.data,
        "lat": image_grid.lat,
        "lon": image_grid.lon,
    }

    if image_grid.h is not None:
        mat_dict["h"] = image_grid.h

    if metadata:
        mat_dict["metadata"] = metadata

    # Write to file
    try:
        savemat(filepath, {"GCP": mat_dict}, do_compression=True)
    except Exception as e:
        raise OSError(f"Error writing MAT file {filepath}: {e}") from e


def _save_image_grid_geotiff(filepath: Path, image_grid: ImageGrid, metadata: dict | None) -> None:
    """Save ImageGrid to GeoTIFF file (internal helper)."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as e:
        raise ImportError("GeoTIFF export requires 'rasterio'. Install with: pip install rasterio") from e

    nrows, ncols = image_grid.data.shape

    # Compute geographic bounds
    minlon = image_grid.lon.min()
    maxlon = image_grid.lon.max()
    minlat = image_grid.lat.min()
    maxlat = image_grid.lat.max()

    # Create affine transform
    transform = from_bounds(minlon, minlat, maxlon, maxlat, ncols, nrows)

    # Write to file
    try:
        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=nrows,
            width=ncols,
            count=1,
            dtype=image_grid.data.dtype,
            crs="EPSG:4326",  # WGS84
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(image_grid.data, 1)

            # Add metadata as tags
            if metadata:
                dst.update_tags(**metadata)

    except Exception as e:
        raise OSError(f"Error writing GeoTIFF file {filepath}: {e}") from e
