"""Image I/O utilities for correction pipeline.

Provides format-agnostic loading and saving of image data used in the
correction pipeline.  All format-specific logic is in private helpers;
the public API dispatches on file extension so callers never need to
know the underlying format.

Supported formats
-----------------
``.mat``   — MATLAB struct files (calibration and legacy test data).
``.nc`` / ``.netcdf`` / ``.nc4`` — NetCDF (regridded GCPs, outputs).
``.hdf`` / ``.h5`` — HDF4/5 raw GCP chips; use :func:`load_gcp_chip_from_hdf`
    then ``curryer.correction.regrid`` to convert ECEF to an ImageGrid.

Public API (8 functions)
------------------------
:func:`load_image_grid`       — any image file → :class:`ImageGrid`
:func:`load_named_image_grid` — any image file → :class:`NamedImageGrid`
:func:`load_observation_file` — observation + spacecraft position
:func:`load_los_vectors`      — LOS unit vectors from calibration file
:func:`load_optical_psf`      — PSF entries from calibration file
:func:`load_gcp_chip_from_hdf`— raw HDF chip (band + ECEF arrays)
:func:`save_image_grid`       — write ImageGrid; format from extension
:func:`infer_spacecraft_state`— derive boresight/t_matrix from position
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .data_structures import ImageGrid, NamedImageGrid, OpticalPSFEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private format-specific loaders
# ---------------------------------------------------------------------------

_MAT_EXTS = frozenset({".mat"})
_NC_EXTS = frozenset({".nc", ".netcdf", ".nc4"})
_TIFF_EXTS = frozenset({".tif", ".tiff"})


def _load_mat_image_grid(filepath: Path, key: str) -> ImageGrid:
    """Load an :class:`ImageGrid` from a MATLAB ``.mat`` struct.

    Parameters
    ----------
    filepath : Path
        Resolved local path to the ``.mat`` file.
    key : str
        MATLAB struct key (e.g. ``"subimage"``, ``"GCP"``).

    Returns
    -------
    ImageGrid

    Raises
    ------
    KeyError
        If *key* is not present in the file.
    """
    from scipy.io import loadmat

    from curryer.correction.io import resolve_path

    filepath = resolve_path(filepath)
    mat_data = loadmat(str(filepath), squeeze_me=True, struct_as_record=False)

    if key not in mat_data:
        available_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {filepath.name}. Available keys: {available_keys}")

    struct = mat_data[key]
    return ImageGrid(
        data=struct.data,
        lat=struct.lat,
        lon=struct.lon,
        h=getattr(struct, "h", None),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_image_grid(filepath: Path | str, mat_key: str = "subimage") -> ImageGrid:
    """Load any supported image file as an :class:`ImageGrid`.

    Dispatches on file extension — callers do not need to know the
    underlying format.

    Parameters
    ----------
    filepath : path-like
        Path to the image file.  Local paths and ``s3://`` URIs supported
        (S3 requires ``boto3``).
    mat_key : str, optional
        MATLAB struct key.  Defaults to ``"subimage"``.  Ignored for NetCDF.

    Returns
    -------
    ImageGrid

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    FileNotFoundError
        If *filepath* does not exist.

    Examples
    --------
    >>> obs = load_image_grid(Path("subimage.mat"), mat_key="subimage")
    >>> gcp = load_image_grid(Path("GCP12055_regridded.nc"))
    """
    from curryer.correction.io import resolve_path

    # Extract the suffix from the original path *before* resolving so that S3
    # URIs (which resolve to temp files with mangled names) dispatch correctly.
    suffix = Path(str(filepath)).suffix.lower()
    filepath = resolve_path(filepath)
    if suffix in _MAT_EXTS:
        return _load_mat_image_grid(filepath, key=mat_key)
    if suffix in _NC_EXTS:
        return _load_netcdf_image_grid(filepath)
    raise ValueError(
        f"Unrecognised file extension '{suffix}' for {filepath}. Supported: {sorted(_MAT_EXTS | _NC_EXTS)}"
    )


def load_optical_psf(mat_file: str | Path, key: str = "PSF_struct_675nm") -> list[OpticalPSFEntry]:
    """Load optical PSF entries from a ``.mat`` calibration file.

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


def load_los_vectors(mat_file: str | Path, key: str = "b_HS") -> np.ndarray:
    """Load line-of-sight unit vectors from a ``.mat`` calibration file.

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
# NetCDF File I/O
# ============================================================================


def _load_netcdf_image_grid(
    filepath: Path,
    band_var: str = "band_data",
    lat_var: str = "lat",
    lon_var: str = "lon",
    height_var: str = "h",
) -> ImageGrid:
    """Load an :class:`ImageGrid` from a NetCDF file (private helper).

    Called by :func:`load_image_grid` and :func:`load_named_image_grid`.
    Handles both regular (1-D lat/lon) and irregular (2-D lat/lon) grids;
    1-D coordinate arrays are broadcast to full 2-D.

    Parameters
    ----------
    filepath : Path
        Path to a NetCDF file.
    band_var : str, optional
        Band/radiance variable name.  Falls back to ``"data"`` for legacy files.
    lat_var : str, optional
        Latitude variable name.  Falls back to ``"latitude"``.
    lon_var : str, optional
        Longitude variable name.  Falls back to ``"longitude"``.
    height_var : str, optional
        Height variable name.  Optional — ``h`` is ``None`` when absent.

    Returns
    -------
    ImageGrid

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    KeyError
        If a required variable (band, lat, or lon) is not found.
    """
    import xarray as xr

    from curryer.correction.io import resolve_path

    # resolve_path validates local existence and downloads S3 URIs.
    filepath = resolve_path(filepath)

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


def _save_to_netcdf(
    filepath: Path,
    image_grid: ImageGrid,
    metadata: dict[str, str] | None = None,
    compression: bool = True,
) -> None:
    """Save *image_grid* to a CF-1.8 NetCDF file (private — called from :func:`save_image_grid`).

    Parameters
    ----------
    filepath : Path
        Output NetCDF file path.
    image_grid : ImageGrid
    metadata : dict, optional
        Additional global attributes.
    compression : bool, default=True
        Enable zlib compression.
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
        # Variable names match _load_netcdf_image_grid defaults: lat, lon, band_data, h
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

    logger.info(f"NetCDF saved: {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")


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
# Generic Image Save/Load — public dispatcher
# ============================================================================


def save_image_grid(
    filepath: Path,
    image_grid: ImageGrid,
    metadata: dict | None = None,
    compression: bool = True,
) -> None:
    """Save an :class:`ImageGrid` to file.

    The output format is determined by the file extension:

    * ``.nc`` / ``.netcdf`` / ``.nc4`` → CF-1.8 NetCDF (via ``netCDF4``)
    * ``.mat`` → MATLAB struct (key ``"GCP"``)
    * ``.tif`` / ``.tiff`` → GeoTIFF (requires ``rasterio``)

    Parameters
    ----------
    filepath : Path
        Output file path.  Extension controls the format.
    image_grid : ImageGrid
        Image data with lat/lon coordinates.
    metadata : dict, optional
        Additional metadata written as file-level attributes or tags.
    compression : bool, optional
        Enable compression for NetCDF output (default ``True``).
        Ignored for other formats.

    Raises
    ------
    ValueError
        If the file extension is not supported.

    Examples
    --------
    >>> save_image_grid(Path("chip.nc"), regridded, metadata={"band": "red"})
    >>> save_image_grid(Path("chip.mat"), regridded)
    """
    suffix = Path(filepath).suffix.lower()
    if suffix in _NC_EXTS:
        _save_to_netcdf(filepath, image_grid, metadata, compression)
    elif suffix in _MAT_EXTS:
        _save_to_mat(filepath, image_grid, metadata)
    elif suffix in _TIFF_EXTS:
        _save_to_geotiff(filepath, image_grid, metadata)
    else:
        raise ValueError(
            f"Unsupported format: extension '{suffix}' for {filepath}. "
            f"Supported: {sorted(_NC_EXTS | _MAT_EXTS | _TIFF_EXTS)}"
        )
    logger.info(f"Saved ImageGrid to {filepath}")


def _save_to_mat(filepath: Path, image_grid: ImageGrid, metadata: dict | None) -> None:
    """Save ImageGrid to MATLAB .mat file (private helper)."""
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


def _save_to_geotiff(filepath: Path, image_grid: ImageGrid, metadata: dict | None) -> None:
    """Save ImageGrid to GeoTIFF file (private helper)."""
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
        NetCDF file written by :func:`save_image_grid` or the
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
        result = _load_mat_image_grid(filepath, key=mat_key)
        return NamedImageGrid(data=result.data, lat=result.lat, lon=result.lon, h=result.h, name=name)

    if suffix in (".nc", ".netcdf", ".nc4"):
        grid = _load_netcdf_image_grid(filepath)
        return NamedImageGrid(data=grid.data, lat=grid.lat, lon=grid.lon, h=grid.h, name=name)

    raise ValueError(
        f"Unrecognised file extension '{suffix}' for {filepath}. Supported formats: .mat, .nc, .netcdf, .nc4"
    )


def load_observation_file(
    filepath: str | Path,
    mat_key: str = "subimage",
) -> tuple[ImageGrid, np.ndarray | None]:
    """Load one observation file and return ``(ImageGrid, spacecraft_position_m)``.

    Supports ``.mat`` and NetCDF (``.nc``, ``.nc4``, ``.netcdf``) formats.
    The spacecraft ECEF position is extracted when available in the file; when
    absent, ``None`` is returned and callers should fall back to
    :func:`infer_spacecraft_state`.

    Parameters
    ----------
    filepath : str or Path
        Local path or ``s3://`` URI to a ``.mat`` or ``.nc`` observation file.
    mat_key : str, optional
        MATLAB struct key containing the image data.  Defaults to
        ``"subimage"``.  Ignored for NetCDF files.

    Returns
    -------
    grid : ImageGrid
        Radiance data on a lat/lon grid.
    r_spacecraft_m : ndarray of shape (3,) or None
        Spacecraft ECEF position in metres at the mid-frame, extracted from the
        file when available.  ``None`` when not present — caller should use
        :func:`infer_spacecraft_state` to approximate.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    FileNotFoundError
        If *filepath* is a local path that does not exist.
    ImportError
        If *filepath* is an S3 URI and ``boto3`` is not installed.
    """
    import xarray as xr

    from curryer.correction.io import resolve_path

    # Extract the suffix from the original path *before* resolving so that S3
    # URIs (whose temp-file names may differ) dispatch correctly.
    suffix = Path(str(filepath)).suffix.lower()
    # Resolve S3 URIs to a local path; validate local existence.
    local_filepath = resolve_path(filepath)

    if suffix == ".mat":
        from scipy.io import loadmat  # noqa: PLC0415

        grid = load_image_grid(local_filepath, mat_key=mat_key)
        mat_raw = loadmat(str(local_filepath), squeeze_me=True)
        r_sc_m = np.asarray(mat_raw["R_ISS_midframe"]).ravel() if "R_ISS_midframe" in mat_raw else None
        return grid, r_sc_m

    if suffix in (".nc", ".netcdf", ".nc4"):
        grid = load_image_grid(local_filepath)
        r_sc_m = None
        try:
            with xr.open_dataset(local_filepath) as ds:
                if "position" in ds:
                    r_sc_m = np.asarray(ds["position"].values).ravel()
        except Exception:
            logger.debug("Could not read spacecraft position from %s", local_filepath, exc_info=True)
        return grid, r_sc_m

    raise ValueError(
        f"Unsupported observation file format '{suffix}' for {filepath}. Supported formats: .mat, .nc, .netcdf, .nc4"
    )


def infer_spacecraft_state(
    grid: ImageGrid,
    r_spacecraft_m: np.ndarray | None,
    default_altitude_m: float = 400_000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(r_spacecraft_m, boresight, t_matrix)`` for image-matching.

    When *r_spacecraft_m* is provided it is used directly; the boresight is
    the unit nadir vector ``-r / |r|``.  When *r_spacecraft_m* is ``None`` the
    grid centre lat/lon is used to build an approximate nadir position at
    *default_altitude_m* above the WGS-84 ellipsoid surface.

    The rotation matrix is always the ``3×3`` identity — the boresight is
    already expressed in the CTRS frame via the nadir-approximation, so no
    additional rotation is needed.

    Parameters
    ----------
    grid : ImageGrid
        Observation grid, used for the centre lat/lon when *r_spacecraft_m*
        is ``None``.
    r_spacecraft_m : ndarray of shape (3,) or None
        Spacecraft ECEF position in metres.  Pass ``None`` to fall back to
        the nadir approximation.
    default_altitude_m : float, optional
        Spacecraft altitude above the WGS-84 surface (metres) used when
        *r_spacecraft_m* is ``None``.  Default 400 000 m (ISS nominal orbit).
        Override for other spacecraft (e.g. 505 000 m for CTIM).

    Returns
    -------
    r_spacecraft_m : ndarray, shape (3,)
        ECEF spacecraft position in metres.
    boresight : ndarray, shape (3,)
        Nadir unit vector from spacecraft toward Earth centre.
    t_matrix : ndarray, shape (3, 3)
        Identity rotation matrix.
    """
    from curryer.compute.constants import WGS84_SEMI_MAJOR_AXIS_KM  # noqa: PLC0415

    if r_spacecraft_m is not None:
        r = np.asarray(r_spacecraft_m, dtype=float).ravel()
        boresight = -r / np.linalg.norm(r)
        return r, boresight, np.eye(3)

    # Approximate nadir from grid centre lat/lon
    mid_i, mid_j = grid.mid_indices
    lat = float(grid.lat[mid_i, mid_j])
    lon = float(grid.lon[mid_i, mid_j])
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    nadir_hat = np.array(
        [
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ]
    )
    r_approx = (WGS84_SEMI_MAJOR_AXIS_KM * 1_000.0 + default_altitude_m) * nadir_hat
    logger.debug(
        "No spacecraft position in observation file — approximating nadir "
        "from grid centre (lat=%.2f, lon=%.2f, alt=%.0f m)",
        lat,
        lon,
        default_altitude_m,
    )
    return r_approx, -nadir_hat, np.eye(3)
