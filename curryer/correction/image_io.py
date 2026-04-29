"""Image I/O utilities for correction pipeline.

This module provides format-agnostic loading and saving of image data
used in the correction pipeline, including:
- MATLAB .mat files (legacy test data)
- HDF files (raw GCP chips)
- NetCDF files (regridded GCPs, outputs)

All functions work with ImageGrid and related data structures.
No dependencies on image matching or other correction algorithms.
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
    mat_file: str | Path, key: str = "subimage", name: str | None = None, as_named: bool = False
) -> ImageGrid | NamedImageGrid:
    """
    Load ImageGrid from MATLAB .mat file.

    Parameters
    ----------
    mat_file : str or Path
        Path to .mat file (local path or ``s3://`` URI).
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
        If mat_file is a local path and doesn't exist.
    ImportError
        If mat_file is an S3 URI and boto3 is not installed.
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

    from curryer.correction.io import resolve_path

    mat_file = resolve_path(mat_file)
    # resolve_path already validated existence / downloaded from S3.

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


def load_optical_psf_from_mat(mat_file: str | Path, key: str = "PSF_struct_675nm") -> list[OpticalPSFEntry]:
    """
    Load optical PSF entries from MATLAB .mat file.

    Parameters
    ----------
    mat_file : str or Path
        Path to MATLAB file with PSF data (local path or ``s3://`` URI).
    key : str, default="PSF_struct_675nm"
        Primary key to try for PSF data.

    Returns
    -------
    list[OpticalPSFEntry]
        Optical PSF samples with data, x, and field_angle arrays.

    Raises
    ------
    FileNotFoundError
        If mat_file is a local path and doesn't exist.
    ImportError
        If mat_file is an S3 URI and boto3 is not installed.
    KeyError
        If no PSF data found with common key names.
    ValueError
        If PSF entries missing field angle attribute.
    """
    from scipy.io import loadmat

    from curryer.correction.io import resolve_path

    mat_file = resolve_path(mat_file)
    # resolve_path already validated existence / downloaded from S3.

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


def load_los_vectors_from_mat(mat_file: str | Path, key: str = "b_HS") -> np.ndarray:
    """
    Load line-of-sight vectors from MATLAB .mat file.

    Parameters
    ----------
    mat_file : str or Path
        Path to MATLAB file with LOS vectors (local path or ``s3://`` URI).
    key : str, default="b_HS"
        Primary key to try for LOS data.

    Returns
    -------
    np.ndarray
        LOS unit vectors in instrument frame, shape (n_pixels, 3).

    Raises
    ------
    FileNotFoundError
        If mat_file is a local path and doesn't exist.
    ImportError
        If mat_file is an S3 URI and boto3 is not installed.
    KeyError
        If no LOS vectors found with common key names.
    """
    from scipy.io import loadmat

    from curryer.correction.io import resolve_path

    mat_file = resolve_path(mat_file)
    # resolve_path already validated existence / downloaded from S3.

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
# NetCDF File I/O (for regridded chips and general image grids)
# ============================================================================


def load_image_grid_from_netcdf(
    filepath: Path,
    band_var: str = "band_data",
    lat_var: str = "lat",
    lon_var: str = "lon",
    height_var: str = "h",
) -> ImageGrid:
    """Load an :class:`ImageGrid` from a NetCDF file.

    Handles both regular (1-D lat/lon) and irregular (2-D lat/lon) grids.
    1-D coordinate arrays are broadcast to a full 2-D grid so that the
    returned :class:`ImageGrid` always has matching ``(y, x)`` shaped arrays.

    Parameters
    ----------
    filepath : Path
        Path to a NetCDF file written by :func:`save_image_grid_to_netcdf` or
        the regridding pipeline.
    band_var : str, optional
        Name of the band/radiance variable.  Falls back to ``"data"`` when
        the specified name is not present (legacy files).
    lat_var : str, optional
        Name of the latitude variable.  Falls back to ``"latitude"``.
    lon_var : str, optional
        Name of the longitude variable.  Falls back to ``"longitude"``.
    height_var : str, optional
        Name of the optional height variable.  Not required; ``h`` is set to
        ``None`` when absent.

    Returns
    -------
    ImageGrid
        Loaded image grid with ``data``, ``lat``, ``lon``, and ``h`` fields.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    KeyError
        If a required variable (band, lat, or lon) is not found.

    Examples
    --------
    >>> grid = load_image_grid_from_netcdf(Path("regridded.nc"))
    >>> grid.data.shape
    (421, 433)
    """
    import xarray as xr

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    logger.info("Loading ImageGrid from NetCDF: %s", filepath)

    with xr.open_dataset(filepath) as ds:
        # Resolve band variable (with legacy fallback)
        if band_var not in ds:
            legacy = "data"
            if legacy in ds:
                band_var = legacy
            else:
                raise KeyError(
                    f"Band variable '{band_var}' not found in {filepath.name}. "
                    f"Available variables: {list(ds.data_vars)}"
                )

        # Resolve lat/lon variables (with legacy fallbacks)
        if lat_var not in ds:
            lat_var = next((v for v in ("latitude",) if v in ds), None)
            if lat_var is None:
                raise KeyError(f"Latitude variable not found in {filepath.name}")
        if lon_var not in ds:
            lon_var = next((v for v in ("longitude",) if v in ds), None)
            if lon_var is None:
                raise KeyError(f"Longitude variable not found in {filepath.name}")

        data = np.asarray(ds[band_var].values, dtype=float)
        lat = np.asarray(ds[lat_var].values, dtype=float)
        lon = np.asarray(ds[lon_var].values, dtype=float)
        h = np.asarray(ds[height_var].values, dtype=float) if height_var in ds else None

    # Broadcast 1-D coordinates to 2-D so ImageGrid shape invariants hold
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    elif lat.ndim == 1:
        lat = np.repeat(lat[:, np.newaxis], data.shape[1], axis=1)
    elif lon.ndim == 1:
        lon = np.tile(lon[np.newaxis, :], (data.shape[0], 1))

    logger.info("Loaded ImageGrid from NetCDF: shape %s", data.shape)
    return ImageGrid(data=data, lat=lat, lon=lon, h=h)


def save_image_grid_to_netcdf(
    filepath: Path,
    image_grid: ImageGrid,
    metadata: dict[str, str] | None = None,
    compression: bool = True,
) -> None:
    """
    Save ImageGrid to NetCDF file (CF-1.8 compliant).

    This function can be used for any ImageGrid, including regridded GCP chips,
    L1A subimages, or other gridded data.

    Parameters
    ----------
    filepath : Path
        Output NetCDF file path.
    image_grid : ImageGrid
        Image data with lat/lon coordinates to save.
    metadata : dict[str, str], optional
        Additional global attributes to include in the NetCDF file.
        Common keys: 'source_file', 'mission', 'sensor', 'processing_date', 'band'.
    compression : bool, default=True
        Enable zlib compression for data variables (~50% size reduction).

    Raises
    ------
    ImportError
        If netCDF4 is not installed.
    OSError
        If file cannot be written.

    Notes
    -----
    The NetCDF file follows CF-1.8 conventions and contains:
    - Variables: 'band_data', 'lat', 'lon', 'h' (if available)
    - Dimensions: 'y' (rows), 'x' (columns)
    - Coordinates: lat(y, x) or (y,), lon(y, x) or (x,)
    - Attributes: grid info, CRS metadata, processing information

    For regular grids (1D coordinates), variables are stored efficiently as:
    - lat(y): Single column
    - lon(x): Single row

    For irregular grids (2D coordinates), full arrays are stored:
    - lat(y, x): Full grid
    - lon(y, x): Full grid

    Examples
    --------
    Save regridded GCP chip:

    >>> save_image_grid_to_netcdf(
    ...     Path("regridded.nc"),
    ...     regridded_chip,
    ...     metadata={
    ...         'source_file': 'LT08CHP.20140803.p002r071.c01.v001.hdf',
    ...         'mission': 'CLARREO Pathfinder',
    ...         'sensor': 'Landsat-8',
    ...         'band': 'red',
    ...         'processing_date': '2026-02-02'
    ...     }
    ... )

    Save L1A subimage:

    >>> save_image_grid_to_netcdf(
    ...     Path("l1a_subimage.nc"),
    ...     l1a_grid,
    ...     metadata={'mission': 'CLARREO', 'level': 'L1A'}
    ... )
    """
    try:
        import datetime

        from netCDF4 import Dataset
    except ImportError as e:
        raise ImportError("netCDF4 is required to save NetCDF files. Install with: pip install netCDF4") from e

    filepath = Path(filepath)
    logger.info(f"Saving ImageGrid to NetCDF: {filepath}")

    # Create NetCDF file
    with Dataset(filepath, "w", format="NETCDF4") as nc:
        # Global attributes
        nc.setncattr("title", "Regridded GCP Chip")
        nc.setncattr("institution", "NASA Langley Research Center")
        nc.setncattr("source", "Curryer GCP Regridding Module")
        nc.setncattr(
            "history", f"Created {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        nc.setncattr("Conventions", "CF-1.8")
        nc.setncattr("grid_type", "regular_lat_lon")

        # Add user-provided metadata
        if metadata:
            for key, value in metadata.items():
                nc.setncattr(key, str(value))

        # Create dimensions
        nrows, ncols = image_grid.data.shape
        nc.createDimension("y", nrows)
        nc.createDimension("x", ncols)

        # Compression settings
        comp_kwargs = {"zlib": True, "complevel": 4} if compression else {}

        # Determine if coordinates are 1D or 2D
        lat_is_1d = image_grid.lat.ndim == 1 or (
            image_grid.lat.ndim == 2 and np.allclose(image_grid.lat, image_grid.lat[:, 0:1])
        )
        lon_is_1d = image_grid.lon.ndim == 1 or (
            image_grid.lon.ndim == 2 and np.allclose(image_grid.lon, image_grid.lon[0:1, :])
        )

        # Create coordinate variables
        # Variable names match load_image_grid_from_netcdf defaults: lat, lon, band_data, h
        if lat_is_1d:
            # 1D latitude (varies with y only)
            lat_var = nc.createVariable("lat", "f8", ("y",), **comp_kwargs)
            lat_var[:] = image_grid.lat[:, 0] if image_grid.lat.ndim == 2 else image_grid.lat
        else:
            # 2D latitude
            lat_var = nc.createVariable("lat", "f8", ("y", "x"), **comp_kwargs)
            lat_var[:] = image_grid.lat

        lat_var.units = "degrees_north"
        lat_var.long_name = "latitude"
        lat_var.standard_name = "latitude"

        if lon_is_1d:
            # 1D longitude (varies with x only)
            lon_var = nc.createVariable("lon", "f8", ("x",), **comp_kwargs)
            lon_var[:] = image_grid.lon[0, :] if image_grid.lon.ndim == 2 else image_grid.lon
        else:
            # 2D longitude
            lon_var = nc.createVariable("lon", "f8", ("y", "x"), **comp_kwargs)
            lon_var[:] = image_grid.lon

        lon_var.units = "degrees_east"
        lon_var.long_name = "longitude"
        lon_var.standard_name = "longitude"

        # Create data variable
        data_var = nc.createVariable("band_data", "f8", ("y", "x"), fill_value=np.nan, **comp_kwargs)
        data_var[:] = image_grid.data
        data_var.long_name = "regridded_radiance"
        data_var.units = "DN"
        data_var.coordinates = "lat lon"
        data_var.grid_mapping = "crs"

        # Add grid statistics as attributes
        valid_mask = ~np.isnan(image_grid.data)
        valid_pixels = int(np.sum(valid_mask))
        data_var.setncattr("valid_pixels", valid_pixels)
        if valid_pixels > 0:
            data_var.setncattr("valid_min", float(np.nanmin(image_grid.data)))
            data_var.setncattr("valid_max", float(np.nanmax(image_grid.data)))

        # Add height if available
        if image_grid.h is not None:
            h_var = nc.createVariable("h", "f8", ("y", "x"), fill_value=np.nan, **comp_kwargs)
            h_var[:] = image_grid.h
            h_var.units = "meters"
            h_var.long_name = "height_above_reference_ellipsoid"
            h_var.standard_name = "height_above_reference_ellipsoid"
            h_var.coordinates = "lat lon"

        # Add CRS information (WGS84)
        crs = nc.createVariable("crs", "i4")
        crs.grid_mapping_name = "latitude_longitude"
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.long_name = "WGS84"
        crs.crs_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'

    logger.info(f"NetCDF file saved successfully: {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")


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

    Supports both HDF4 and HDF5 formats. Tries HDF4 first (Landsat standard),
    then falls back to HDF5 if needed.

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
    if not filepath.exists():
        raise FileNotFoundError(f"HDF file not found: {filepath}")

    coord_x_name, coord_y_name, coord_z_name = coord_names

    # Try HDF4 first (Landsat standard format)
    hdf4_error = None
    try:
        from pyhdf.SD import SD, SDC

        hdf = SD(str(filepath), SDC.READ)
        datasets = hdf.datasets()

        # Check for required datasets
        if band_name not in datasets:
            available = list(datasets.keys())
            hdf.end()
            raise KeyError(f"Band '{band_name}' not found. Available datasets: {available}")

        if coord_x_name not in datasets:
            hdf.end()
            raise KeyError(f"X coordinate '{coord_x_name}' not found in {filepath.name}")
        if coord_y_name not in datasets:
            hdf.end()
            raise KeyError(f"Y coordinate '{coord_y_name}' not found in {filepath.name}")
        if coord_z_name not in datasets:
            hdf.end()
            raise KeyError(f"Z coordinate '{coord_z_name}' not found in {filepath.name}")

        # Load datasets
        band_data = np.array(hdf.select(band_name).get(), dtype=np.float64)
        ecef_x = np.array(hdf.select(coord_x_name).get(), dtype=np.float64)
        ecef_y = np.array(hdf.select(coord_y_name).get(), dtype=np.float64)
        ecef_z = np.array(hdf.select(coord_z_name).get(), dtype=np.float64)

        hdf.end()

        logger.info(f"Loaded GCP chip from HDF4 file {filepath.name}: shape {band_data.shape}")
        return band_data, ecef_x, ecef_y, ecef_z

    except ImportError:
        # pyhdf not available, will try h5py below
        pass
    except Exception as e:
        # pyhdf raised an error (possibly HDF4Error because file is HDF5), store and try h5py
        hdf4_error = e

    # Try HDF5 if HDF4 failed or pyhdf not available
    try:
        import h5py

        with h5py.File(filepath, "r") as hdf:
            # Load band data
            if band_name not in hdf:
                available = list(hdf.keys())
                raise KeyError(f"Band '{band_name}' not found. Available datasets: {available}")

            band_data = np.array(hdf[band_name], dtype=np.float64)

            # Load ECEF coordinates
            if coord_x_name not in hdf:
                raise KeyError(f"X coordinate '{coord_x_name}' not found in {filepath.name}")
            if coord_y_name not in hdf:
                raise KeyError(f"Y coordinate '{coord_y_name}' not found in {filepath.name}")
            if coord_z_name not in hdf:
                raise KeyError(f"Z coordinate '{coord_z_name}' not found in {filepath.name}")

            ecef_x = np.array(hdf[coord_x_name], dtype=np.float64)
            ecef_y = np.array(hdf[coord_y_name], dtype=np.float64)
            ecef_z = np.array(hdf[coord_z_name], dtype=np.float64)

        logger.info(f"Loaded GCP chip from HDF5 file {filepath.name}: shape {band_data.shape}")

    except ImportError as e:
        # Neither library available
        error_msg = f"Cannot read HDF file {filepath}. Neither pyhdf (for HDF4) nor h5py (for HDF5) is available."
        if hdf4_error:
            error_msg += f" HDF4 error: {hdf4_error}"
        raise ImportError(error_msg) from e
    except (KeyError, OSError, ValueError):
        # Re-raise validation/IO errors from h5py
        raise

    # Validate shapes
    if not (band_data.shape == ecef_x.shape == ecef_y.shape == ecef_z.shape):
        raise ValueError(
            f"Array shape mismatch in {filepath.name}: "
            f"band={band_data.shape}, x={ecef_x.shape}, y={ecef_y.shape}, z={ecef_z.shape}"
        )

    return band_data, ecef_x, ecef_y, ecef_z


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
        save_image_grid_to_netcdf(filepath, image_grid, metadata)
    elif format == "mat":
        _save_image_grid_mat(filepath, image_grid, metadata)
    elif format == "geotiff":
        _save_image_grid_geotiff(filepath, image_grid, metadata)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: 'netcdf', 'mat', 'geotiff'")

    logger.info(f"Saved ImageGrid to {filepath} (format: {format})")


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


# ============================================================================
# Format-agnostic loader
# ============================================================================


def load_named_image_grid(filepath: Path | str, mat_key: str = "subimage") -> NamedImageGrid:
    """Load any supported image file as a :class:`NamedImageGrid`.

    Dispatches on file extension so callers do not need to know the underlying
    format.  The returned grid always carries the file path as its ``name``.

    Supported formats
    -----------------
    ``.mat``
        MATLAB struct file.  The struct accessed via *mat_key* must have
        ``data``, ``lat``, and ``lon`` attributes (and optionally ``h``).
    ``.nc`` / ``.netcdf`` / ``.nc4``
        NetCDF file written by :func:`save_image_grid_to_netcdf` or the
        regridding pipeline (expects ``band_data``, ``lat``, ``lon``).

    Parameters
    ----------
    filepath : path-like
        Path to the image file.
    mat_key : str, optional
        MATLAB struct key to read.  Defaults to ``"subimage"``.  Ignored for
        NetCDF files.

    Returns
    -------
    NamedImageGrid
        Loaded image grid with ``name`` set to the string representation of
        *filepath*.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    FileNotFoundError
        If *filepath* does not exist.

    Examples
    --------
    >>> obs = load_named_image_grid(Path("TestCase1a_subimage.mat"), mat_key="subimage")
    >>> gcp = load_named_image_grid(Path("GCP12055Dili_regridded.nc"))
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    name = str(filepath)

    if suffix == ".mat":
        result = load_image_grid_from_mat(filepath, key=mat_key, name=name, as_named=True)
        if not isinstance(result, NamedImageGrid):
            raise TypeError(f"Expected NamedImageGrid from .mat file, got {type(result).__name__!r}")
        return result

    if suffix in (".nc", ".netcdf", ".nc4"):
        grid = load_image_grid_from_netcdf(filepath)
        return NamedImageGrid(data=grid.data, lat=grid.lat, lon=grid.lon, h=grid.h, name=name)

    raise ValueError(
        f"Unrecognised file extension '{suffix}' for {filepath}. Supported formats: .mat, .nc, .netcdf, .nc4"
    )
