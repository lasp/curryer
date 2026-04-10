"""Main correction pipeline orchestration.

This module contains the public-facing :func:`loop` function that drives
the Monte Carlo parameter sensitivity analysis, plus all of the helper
functions it calls:

- Adapter functions that bridge between the geolocation/image-matching
  sub-modules and the correction loop.
- :func:`load_config_from_json` -- build a :class:`CorrectionConfig` from
  a JSON file.
- :func:`_load_file` -- internal helper that reads CSV/NetCDF/HDF5 files
  into DataFrames, replacing the old mission-specific loader callables.
- :func:`_load_image_pair_data`, :func:`_load_calibration_data`,
  :func:`_geolocate_and_match` -- per-iteration computation helpers.
- :func:`loop` -- outer GCP-pair loop, inner parameter-set loop.
"""

import logging
import time
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from curryer.correction.results import CorrectionResult

from curryer import meta
from curryer import spicierpy as sp
from curryer.compute import constants, spatial
from curryer.correction.config import (
    CalibrationData,
    CorrectionConfig,
    CorrectionInput,
    ImageMatchingContext,
    KernelContext,
    ParameterType,
)
from curryer.correction.data_structures import (
    ImageGrid,
    PSFSamplingConfig,
    SearchConfig,
)
from curryer.correction.dataio import (
    validate_science_output,
    validate_telemetry_output,
)
from curryer.correction.image_io import (
    load_image_grid_from_mat,
    load_los_vectors_from_mat,
    load_optical_psf_from_mat,
)
from curryer.correction.image_match import (
    integrated_image_match,
    validate_image_matching_output,
)
from curryer.correction.kernel_ops import (
    _create_dynamic_kernels,
    _create_parameter_kernels,
)
from curryer.correction.parameters import load_param_sets
from curryer.correction.results_io import (
    _build_netcdf_structure,
    _cleanup_checkpoint,
    _load_checkpoint,
    _save_netcdf_checkpoint,
    _save_netcdf_results,
)
from curryer.kernels import create

logger = logging.getLogger(__name__)


def _geolocated_to_image_grid(geo_dataset: xr.Dataset):
    """
    Convert Correction geolocation output to ImageGrid for image matching.

    Internal adapter function: converts xarray.Dataset from geolocation step
    to ImageGrid format expected by image_match module.

    Args:
        geo_dataset: xarray.Dataset with latitude, longitude, altitude/height

    Returns:
        ImageGrid suitable for integrated_image_match()
    """

    lat = geo_dataset["latitude"].values
    lon = geo_dataset["longitude"].values

    # Try different field names for altitude/height
    if "altitude" in geo_dataset:
        h = geo_dataset["altitude"].values
    elif "height" in geo_dataset:
        h = geo_dataset["height"].values
    else:
        h = np.zeros_like(lat)

    # Get actual radiance/reflectance data when available
    if "radiance" in geo_dataset:
        data = geo_dataset["radiance"].values
    elif "reflectance" in geo_dataset:
        data = geo_dataset["reflectance"].values
    else:
        data = np.ones_like(lat)

    return ImageGrid(data=data, lat=lat, lon=lon, h=h)


def _extract_spacecraft_position_midframe(
    telemetry: pd.DataFrame,
    config: "CorrectionConfig | None" = None,
) -> np.ndarray:
    """Extract spacecraft position at mid-frame from telemetry.

    Parameters
    ----------
    telemetry : pd.DataFrame
        Telemetry DataFrame with spacecraft position columns.
    config : CorrectionConfig or None, optional
        If provided and ``config.data.position_columns`` is set, those
        column names are used directly. Otherwise falls back to
        pattern-guessing (with a deprecation warning).

    Returns
    -------
    np.ndarray
        Shape ``(3,)`` — ``[x, y, z]`` position in metres (J2000 frame).

    Raises
    ------
    ValueError
        If ``position_columns`` has wrong length, or specified columns are
        not found, or pattern-guessing fails.
    """
    mid_idx = len(telemetry) // 2

    # Prefer explicit column names from config
    if config is not None and config.data is not None and config.data.position_columns is not None:
        cols = config.data.position_columns
        if len(cols) != 3:
            raise ValueError(f"position_columns must have exactly 3 entries, got {len(cols)}: {cols}")
        missing = [c for c in cols if c not in telemetry.columns]
        if missing:
            raise ValueError(
                f"position_columns {missing} not found in telemetry. Available: {telemetry.columns.tolist()}"
            )
        position = telemetry[cols].iloc[mid_idx].values.astype(np.float64)
        logger.debug(
            "Extracted spacecraft position from config.data.position_columns %s: %s",
            cols,
            position,
        )
        return position

    # Legacy fallback: pattern guessing (log deprecation warning)
    logger.warning(
        "position_columns not configured — falling back to column name pattern-guessing. "
        "Set config.data.position_columns = ['col_x', 'col_y', 'col_z'] to silence this warning."
    )

    # Try common column name patterns
    position_patterns = [
        ["sc_pos_x", "sc_pos_y", "sc_pos_z"],
        ["position_x", "position_y", "position_z"],
        ["r_x", "r_y", "r_z"],
        ["pos_x", "pos_y", "pos_z"],
    ]

    for cols in position_patterns:
        if all(c in telemetry.columns for c in cols):
            position = telemetry[cols].iloc[mid_idx].values.astype(np.float64)
            logger.debug(f"Extracted spacecraft position from columns {cols}: {position}")
            return position

    # If patterns don't match, try to find any column containing 'pos' or 'r_'
    pos_cols = [c for c in telemetry.columns if "pos" in c.lower() or c.startswith("r_")]
    if len(pos_cols) >= 3:
        logger.warning(f"Using first 3 position-like columns: {pos_cols[:3]}")
        return telemetry[pos_cols[:3]].iloc[mid_idx].values.astype(np.float64)

    raise ValueError(f"Cannot find position columns in telemetry. Available columns: {telemetry.columns.tolist()}")


# ============================================================================
# ADAPTER FUNCTIONS
# ============================================================================


def image_matching(
    geolocated_data: xr.Dataset,
    gcp_reference_file: Path,
    telemetry: pd.DataFrame,
    calibration_dir: Path,
    params_info: list,
    config: "CorrectionConfig",
    los_vectors_cached: np.ndarray | None = None,
    optical_psfs_cached: list | None = None,
) -> xr.Dataset:
    """
    Image matching using integrated_image_match() module.

    This function performs actual image correlation between geolocated
    pixels and Landsat GCP reference imagery.

    Args:
        geolocated_data: xarray.Dataset with latitude, longitude from geolocation
        gcp_reference_file: Path to GCP reference image (MATLAB .mat file)
        telemetry: Telemetry DataFrame with spacecraft state
        calibration_dir: Directory containing calibration files (LOS vectors, PSF)
        params_info: Current parameter values for error tracking
        config: CorrectionConfig with coordinate name mappings
        los_vectors_cached: Pre-loaded LOS vectors (optional, for performance)
        optical_psfs_cached: Pre-loaded optical PSF entries (optional, for performance)

    Returns:
        xarray.Dataset with error measurements in format expected by error_stats:
            - lat_error_deg, lon_error_deg: Spatial errors in degrees
            - Additional metadata for error statistics processing

    Raises:
        FileNotFoundError: If calibration files are missing
        ValueError: If geolocation data is invalid
    """
    logger.info(f"Image Matching: correlation with {gcp_reference_file.name}")
    start_time = time.time()

    # Convert geolocation output to ImageGrid
    logger.info("  Converting geolocation data to ImageGrid format...")
    subimage = _geolocated_to_image_grid(geolocated_data)
    logger.info(f"    Subimage shape: {subimage.data.shape}")

    # Load GCP reference image
    logger.info(f"  Loading GCP reference from {gcp_reference_file}...")
    gcp = load_image_grid_from_mat(gcp_reference_file, key="GCP")
    # Get GCP center location (center pixel)
    gcp_center_lat = float(gcp.lat[gcp.lat.shape[0] // 2, gcp.lat.shape[1] // 2])
    gcp_center_lon = float(gcp.lon[gcp.lon.shape[0] // 2, gcp.lon.shape[1] // 2])
    logger.info(f"    GCP shape: {gcp.data.shape}, center: ({gcp_center_lat:.4f}, {gcp_center_lon:.4f})")

    # Use cached calibration data if available, otherwise load
    logger.info("  Loading calibration data...")

    if los_vectors_cached is not None and optical_psfs_cached is not None:
        # Use cached data (fast path)
        los_vectors = los_vectors_cached
        optical_psfs = optical_psfs_cached
        logger.info("    Using cached calibration data")
    else:
        # Prefer direct file paths from config; fall back to calibration_dir parameter.
        if config.los_vectors_file is not None:
            los_file = Path(config.los_vectors_file)
        elif calibration_dir is not None:
            los_filename = config.get_calibration_file("los_vectors", default="b_HS.mat")
            los_file = calibration_dir / los_filename
        else:
            raise ValueError("No LOS vectors source configured. Set config.los_vectors_file or config.calibration_dir.")
        los_vectors = load_los_vectors_from_mat(los_file)
        logger.info(f"    LOS vectors: {los_vectors.shape}")

        if config.psf_file is not None:
            psf_file = Path(config.psf_file)
        elif calibration_dir is not None:
            psf_filename = config.get_calibration_file("optical_psf", default="optical_PSF_675nm_upsampled.mat")
            psf_file = calibration_dir / psf_filename
        else:
            raise ValueError("No PSF source configured. Set config.psf_file or config.calibration_dir.")
        optical_psfs = load_optical_psf_from_mat(psf_file)
        logger.info(f"    Optical PSF: {len(optical_psfs)} entries")

    # Extract spacecraft position from telemetry
    r_iss_midframe = _extract_spacecraft_position_midframe(telemetry, config=config)
    logger.info(f"    Spacecraft position: {r_iss_midframe}")

    # Run real image matching
    logger.info("  Running integrated_image_match()...")
    geolocation_config = PSFSamplingConfig()
    search_config = SearchConfig()

    result = integrated_image_match(
        subimage=subimage,
        gcp=gcp,
        r_iss_midframe_m=r_iss_midframe,
        los_vectors_hs=los_vectors,
        optical_psfs=optical_psfs,
        geolocation_config=geolocation_config,
        search_config=search_config,
    )

    # Convert IntegratedImageMatchResult to xarray.Dataset format
    logger.info("  Converting results to error_stats format...")

    # Create single measurement result (image matching produces one correlation per GCP)

    # NOTE: Boresight and transformation matrix for error_stats module
    # ----------------------------------------------------------------
    # These values are NOT used by image_matching() itself - the image correlation
    # is complete and accurate without them. They are needed by call_error_stats_module()
    # for converting off-nadir errors to nadir-equivalent errors.
    #
    # Currently using simplified nadir assumptions which are acceptable for:
    # - Near-nadir observations (< ~5 degrees off-nadir)
    # - Testing image matching correlation accuracy (doesn't affect matching)
    #
    # For accurate nadir-equivalent error conversion with off-nadir pointing, these
    # should be extracted from SPICE/geolocation data:
    # - boresight: Extract from spicierpy.getfov(instrument) and transform via geo_dataset['attitude']
    # - t_matrix: Extract from geo_dataset['attitude'] (transformation from instrument to CTRS)
    #
    # See: error_stats.py _transform_boresight_vectors() for usage
    # See: BORESIGHT_TRANSFORM_ANALYSIS.md for detailed analysis and future enhancement plan

    t_matrix = np.eye(3)  # Simplified: Identity matrix (no rotation)
    boresight = np.array([0.0, 0.0, 1.0])  # Simplified: Nadir pointing assumption

    # Convert errors from km to degrees
    lat_error_deg = result.lat_error_km / 111.0  # ~111 km per degree latitude
    lon_radius_km = constants.WGS84_SEMI_MAJOR_AXIS_KM * np.cos(np.deg2rad(gcp_center_lat))
    lon_error_deg = result.lon_error_km / (lon_radius_km * np.pi / 180.0)

    processing_time = time.time() - start_time

    logger.info(f"  Image matching complete in {processing_time:.2f}s:")
    logger.info(f"    Lat error: {result.lat_error_km:.3f} km ({lat_error_deg:.6f}°)")
    logger.info(f"    Lon error: {result.lon_error_km:.3f} km ({lon_error_deg:.6f}°)")
    logger.info(f"    Correlation: {result.ccv_final:.4f}")
    logger.info(f"    Grid step: {result.final_grid_step_m:.1f} m")

    # Get coordinate names from config
    sc_pos_name = config.spacecraft_position_name
    boresight_name = config.boresight_name
    transform_name = config.transformation_matrix_name

    # Create output dataset in error_stats format (use config names)
    output = xr.Dataset(
        {
            "lat_error_deg": (["measurement"], [lat_error_deg]),
            "lon_error_deg": (["measurement"], [lon_error_deg]),
            sc_pos_name: (["measurement", "xyz"], [r_iss_midframe]),
            boresight_name: (["measurement", "xyz"], [boresight]),
            transform_name: (["measurement", "xyz_from", "xyz_to"], t_matrix[np.newaxis, :, :]),
            "gcp_lat_deg": (["measurement"], [gcp_center_lat]),
            "gcp_lon_deg": (["measurement"], [gcp_center_lon]),
            "gcp_alt": (["measurement"], [0.0]),  # GCP at ground level
        },
        coords={"measurement": [0], "xyz": ["x", "y", "z"], "xyz_from": ["x", "y", "z"], "xyz_to": ["x", "y", "z"]},
    )

    # Add detailed metadata (Fix #3 Part B: Add km errors to attrs)
    output.attrs.update(
        {
            "lat_error_km": result.lat_error_km,
            "lon_error_km": result.lon_error_km,
            "correlation_ccv": result.ccv_final,
            "final_grid_step_m": result.final_grid_step_m,
            "final_index_row": result.final_index_row,
            "final_index_col": result.final_index_col,
            "processing_time_s": processing_time,
            "gcp_file": str(gcp_reference_file.name),
            "gcp_center_lat": gcp_center_lat,
            "gcp_center_lon": gcp_center_lon,
        }
    )

    return output


def call_error_stats_module(image_matching_results, correction_config: "CorrectionConfig"):
    """
    Call the error_stats module with image matching output.

    Args:
        image_matching_results: Either a single image matching result (xarray.Dataset)
                              or a list of image matching results from multiple GCP pairs
        correction_config: CorrectionConfig with all configuration (REQUIRED)

    Returns:
        Aggregate error statistics dataset
    """
    # Handle both single result and list of results
    if not isinstance(image_matching_results, list):
        image_matching_results = [image_matching_results]

    try:
        from curryer.correction.error_stats import ErrorStatsConfig, ErrorStatsProcessor

        logger.info(f"Error Statistics: Processing geolocation errors from {len(image_matching_results)} GCP pairs")

        # Create error stats config directly from Correction config (single source of truth)
        error_config = ErrorStatsConfig.from_correction_config(correction_config)

        processor = ErrorStatsProcessor(config=error_config)

        if len(image_matching_results) == 1:
            # Single GCP pair case
            error_results = processor.process_geolocation_errors(image_matching_results[0])
        else:
            # Multiple GCP pairs - aggregate the data first
            aggregated_data = _aggregate_image_matching_results(image_matching_results, correction_config)
            error_results = processor.process_geolocation_errors(aggregated_data)

        return error_results

    except ImportError as e:
        logger.warning(f"Error stats module not available: {e}")
        logger.info(f"Error Statistics: Using placeholder calculations for {len(image_matching_results)} GCP pairs")

        # Fallback: compute basic statistics across all GCP pairs
        all_lat_errors = []
        all_lon_errors = []
        total_measurements = 0

        for result in image_matching_results:
            lat_errors = result["lat_error_deg"].values
            lon_errors = result["lon_error_deg"].values
            all_lat_errors.extend(lat_errors)
            all_lon_errors.extend(lon_errors)
            total_measurements += len(lat_errors)

        all_lat_errors = np.array(all_lat_errors)
        all_lon_errors = np.array(all_lon_errors)

        # Convert to meters (approximate)
        lat_error_m = all_lat_errors * 111000
        lon_error_m = all_lon_errors * 111000
        total_error_m = np.sqrt(lat_error_m**2 + lon_error_m**2)

        mean_error = float(np.mean(total_error_m))
        rms_error = float(np.sqrt(np.mean(total_error_m**2)))
        std_error = float(np.std(total_error_m))

        return xr.Dataset(
            {
                "mean_error": mean_error,
                "rms_error": rms_error,
                "std_error": std_error,
                "max_error": float(np.max(total_error_m)),
                "min_error": float(np.min(total_error_m)),
            }
        )


def _aggregate_image_matching_results(image_matching_results, config: "CorrectionConfig"):
    """
    Aggregate multiple image matching results into a single dataset for error stats processing.

    Args:
        image_matching_results: List of xarray.Dataset objects from image matching
        config: CorrectionConfig with coordinate name mappings

    Returns:
        Single aggregated xarray.Dataset with all measurements combined
    """
    logger.info(f"Aggregating {len(image_matching_results)} image matching results")

    # Get coordinate names from config
    sc_pos_name = config.spacecraft_position_name
    boresight_name = config.boresight_name
    transform_name = config.transformation_matrix_name

    # Combine all measurements into single arrays
    all_lat_errors = []
    all_lon_errors = []
    all_sc_positions = []
    all_boresights = []
    all_transforms = []
    all_gcp_lats = []
    all_gcp_lons = []
    all_gcp_alts = []

    for i, result in enumerate(image_matching_results):
        # Add GCP pair identifier to track source
        n_measurements = len(result["lat_error_deg"])

        all_lat_errors.extend(result["lat_error_deg"].values)
        all_lon_errors.extend(result["lon_error_deg"].values)

        # Handle coordinate transformation data (use config names)
        # NOTE: Individual results have shape (1, 3) for vectors and (1, 3, 3) for matrices
        if sc_pos_name in result:
            # Shape: (1, 3) -> extract as (3,) for each measurement
            for j in range(n_measurements):
                all_sc_positions.append(result[sc_pos_name].values[j])
        if boresight_name in result:
            # Shape: (1, 3) -> extract as (3,) for each measurement
            for j in range(n_measurements):
                all_boresights.append(result[boresight_name].values[j])
        if transform_name in result:
            # Shape: (1, 3, 3) -> extract as (3, 3) for each measurement
            for j in range(n_measurements):
                all_transforms.append(result[transform_name].values[j, :, :])
        if "gcp_lat_deg" in result:
            all_gcp_lats.extend(result["gcp_lat_deg"].values)
        if "gcp_lon_deg" in result:
            all_gcp_lons.extend(result["gcp_lon_deg"].values)
        if "gcp_alt" in result:
            all_gcp_alts.extend(result["gcp_alt"].values)

    n_total = len(all_lat_errors)

    # Create aggregated dataset with correct dimension names for error_stats
    aggregated = xr.Dataset(
        {
            "lat_error_deg": (["measurement"], np.array(all_lat_errors)),
            "lon_error_deg": (["measurement"], np.array(all_lon_errors)),
        },
        coords={"measurement": np.arange(n_total)},
    )

    # Add optional coordinate transformation data if available (use config names)
    # Use dimension names that match error_stats expectations
    if all_sc_positions:
        # Stack into (n_measurements, 3)
        aggregated[sc_pos_name] = (["measurement", "xyz"], np.array(all_sc_positions))
        aggregated = aggregated.assign_coords({"xyz": ["x", "y", "z"]})

    if all_boresights:
        # Stack into (n_measurements, 3)
        aggregated[boresight_name] = (["measurement", "xyz"], np.array(all_boresights))

    if all_transforms:
        # Stack into (n_measurements, 3, 3) to match error_stats format
        t_stacked = np.stack(all_transforms, axis=0)
        aggregated[transform_name] = (["measurement", "xyz_from", "xyz_to"], t_stacked)
        aggregated = aggregated.assign_coords({"xyz_from": ["x", "y", "z"], "xyz_to": ["x", "y", "z"]})

    if all_gcp_lats:
        aggregated["gcp_lat_deg"] = (["measurement"], np.array(all_gcp_lats))
    if all_gcp_lons:
        aggregated["gcp_lon_deg"] = (["measurement"], np.array(all_gcp_lons))
    if all_gcp_alts:
        aggregated["gcp_alt"] = (["measurement"], np.array(all_gcp_alts))

    aggregated.attrs["source_gcp_pairs"] = len(image_matching_results)
    aggregated.attrs["total_measurements"] = n_total

    logger.info(f"  Aggregated dataset: {n_total} measurements from {len(image_matching_results)} GCP pairs")
    logger.info(f"  Dimensions: {dict(aggregated.sizes)}")

    return aggregated


def _resolve_gcp_pairs(
    sci_key: str,
    gcp_key: str,
    config: "CorrectionConfig",
) -> list[tuple[str, str]]:
    """Return the ``[(sci_key, gcp_key)]`` pair, validating that ``gcp_key`` is set.

    Parameters
    ----------
    sci_key : str
        Science file path for this outer-loop iteration.
    gcp_key : str
        GCP ``.mat`` file path supplied as the third element of the
        ``tlm_sci_gcp_sets`` tuple.  Must be non-empty.
    config : CorrectionConfig
        Unused directly; reserved for future extension.

    Returns
    -------
    list of (sci_key, gcp_path) tuples — always length 1.

    Raises
    ------
    ValueError
        If ``gcp_key`` is empty or whitespace-only.

    Notes
    -----
    For spatial-overlap-based pairing (many L1A images × many GCP chips) call
    :func:`~curryer.correction.pairing.pair_files` *before* :func:`loop` to
    build ``tlm_sci_gcp_sets`` from the ``(l1a_file, gcp_file)`` results.
    """
    if not gcp_key or not gcp_key.strip():
        raise ValueError(
            "gcp_key must be a non-empty file path to a GCP .mat file.\n"
            "Pass the GCP file path as the third element of each tlm_sci_gcp_sets tuple:\n"
            "    tlm_sci_gcp_sets = [(tlm_path, sci_path, gcp_path), ...]\n"
            "\n"
            "To compute which GCP chips overlap a given L1A footprint use:\n"
            "    from curryer.correction.pairing import pair_files\n"
            "    pairs = pair_files(l1a_files, gcp_directory, max_distance_m=0.0)"
        )
    return [(sci_key, gcp_key)]


def _load_file(file_path: str | Path, file_format: str = "csv") -> pd.DataFrame:
    """Load a telemetry or science data file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Local path or S3 URI (``s3://bucket/key``).
    file_format : str
        One of ``"csv"``, ``"netcdf"``, or ``"hdf5"``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *file_path* is local and does not exist.
    ImportError
        If *file_path* is an S3 URI and boto3 is not installed.
    ValueError
        If ``file_format`` is not recognised.
    """
    from curryer.correction.io import resolve_path

    file_path = resolve_path(file_path)
    # NOTE: resolve_path already validated existence / downloaded from S3.
    # The old manual exists() check is removed.

    if file_format == "csv":
        return pd.read_csv(file_path, index_col=0)
    elif file_format == "netcdf":
        return xr.load_dataset(file_path).to_dataframe().reset_index()
    elif file_format == "hdf5":
        return pd.read_hdf(file_path)
    else:
        raise ValueError(f"Unsupported file_format '{file_format}'. Must be 'csv', 'netcdf', or 'hdf5'.")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# These functions extract reusable logic from the main loop to simplify the structure


def _load_calibration_data(config: "CorrectionConfig") -> CalibrationData:
    """Load LOS vectors and optical PSF if calibration_dir is configured.

    This function centralizes calibration data loading, which is now called once
    per GCP pair in the optimized implementation (previously called once per parameter set).

    Parameters
    ----------
    config : CorrectionConfig
        Configuration with calibration_dir and calibration settings

    Returns
    -------
    CalibrationData
        NamedTuple containing (los_vectors, optical_psfs), or (None, None) if
        no calibration directory configured

    Raises
    ------
    FileNotFoundError
        If calibration directory is configured but files don't exist
    ValueError
        If calibration files exist but fail to load properly

    Note
    ----
    Supports three resolution strategies (in priority order):

    1. ``config.los_vectors_file`` / ``config.psf_file`` — direct file paths
       (set in PR 1 via ``CorrectionConfig``).
    2. ``config.calibration_dir`` + ``config.calibration_file_names`` — legacy
       directory-based lookup.
    3. Neither configured — returns ``CalibrationData(None, None)`` so that
       missions without calibration files still work.

    Examples
    --------
    >>> calib_data = _load_calibration_data(config)
    >>> if calib_data.los_vectors is not None:
    ...     # Use calibration data in image matching
    ...     pass
    """
    has_direct = config.los_vectors_file is not None or config.psf_file is not None
    has_dir = bool(config.calibration_dir)

    if not has_direct and not has_dir:
        return CalibrationData(los_vectors=None, optical_psfs=None)

    logger.info("Loading calibration data...")

    # ---- LOS vectors ----
    if config.los_vectors_file is not None:
        los_file = Path(config.los_vectors_file)
    elif config.calibration_dir is not None:
        los_filename = config.get_calibration_file("los_vectors", default="b_HS.mat")
        los_file = config.calibration_dir / los_filename
    else:
        raise ValueError("No LOS vectors source configured. Set config.los_vectors_file or config.calibration_dir.")

    if not los_file.exists():
        raise FileNotFoundError(
            f"LOS vectors calibration file not found: {los_file}\nSet config.los_vectors_file to the correct path."
        )

    los_vectors_cached = load_los_vectors_from_mat(los_file)

    if los_vectors_cached is None:
        raise ValueError(
            f"Failed to load LOS vectors from {los_file}. File exists but load_los_vectors_from_mat() returned None."
        )

    # ---- Optical PSF ----
    if config.psf_file is not None:
        psf_file = Path(config.psf_file)
    elif config.calibration_dir is not None:
        psf_filename = config.get_calibration_file("optical_psf", default="optical_PSF_675nm_upsampled.mat")
        psf_file = config.calibration_dir / psf_filename
    else:
        raise ValueError("No PSF source configured. Set config.psf_file or config.calibration_dir.")

    if not psf_file.exists():
        raise FileNotFoundError(
            f"Optical PSF calibration file not found: {psf_file}\nSet config.psf_file to the correct path."
        )

    optical_psfs_cached = load_optical_psf_from_mat(psf_file)

    if optical_psfs_cached is None:
        raise ValueError(
            f"Failed to load optical PSF from {psf_file}. File exists but load_optical_psf_from_mat() returned None."
        )

    logger.info(f"  Cached LOS vectors: {los_vectors_cached.shape}")
    logger.info(f"  Cached optical PSF: {len(optical_psfs_cached)} entries")

    return CalibrationData(los_vectors=los_vectors_cached, optical_psfs=optical_psfs_cached)


def _load_image_pair_data(
    tlm_key: str,
    sci_key: str,
    config: "CorrectionConfig",
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Load telemetry and science data for an image pair from files.

    Parameters
    ----------
    tlm_key : str
        Path to the telemetry data file.
    sci_key : str
        Path to the science frame timing file.
    config : CorrectionConfig
        Configuration containing geolocation settings, file format, and
        time-scaling options (via ``config.data``).

    Returns
    -------
    tlm_dataset : pandas.DataFrame
        DataFrame containing spacecraft state / telemetry records.
    sci_dataset : pandas.DataFrame
        DataFrame containing science frame timing information.
    ugps_times : array_like
        Time array extracted from the science dataset (uGPS values).

    Raises
    ------
    FileNotFoundError
        If the telemetry or science file does not exist.
    ValueError
        If the science file is missing the required time field.
    """
    file_format = "csv"
    time_scale_factor = 1.0

    if config.data is not None:
        file_format = config.data.file_format
        time_scale_factor = config.data.time_scale_factor

    # Load telemetry from file
    tlm_dataset = _load_file(tlm_key, file_format)
    validate_telemetry_output(tlm_dataset, config)

    # Load science from file
    sci_dataset = _load_file(sci_key, file_format)

    # Apply time scale factor to convert to uGPS if needed
    time_field = config.geo.time_field
    if time_field in sci_dataset.columns and time_scale_factor != 1.0:
        sci_dataset = sci_dataset.copy()
        sci_dataset[time_field] = sci_dataset[time_field] * time_scale_factor

    validate_science_output(sci_dataset, config)
    ugps_times = sci_dataset[time_field]

    return tlm_dataset, sci_dataset, ugps_times


def _geolocate_and_match(
    config: "CorrectionConfig",
    kernel_ctx: KernelContext,
    ugps_times_modified: Any,
    tlm_dataset: pd.DataFrame,
    calibration: CalibrationData,
    image_matching_func: Any,
    match_ctx: ImageMatchingContext,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Perform geolocation and image matching for a parameter set.

    This function loads SPICE kernels, performs geolocation, and runs image
    matching against GCP reference data. It's the core computation step that
    combines all previous setup (kernels, data loading) into results.

    Parameters
    ----------
    config : CorrectionConfig
        Configuration with geo and image matching settings
    kernel_ctx : KernelContext
        NamedTuple containing:
        - mkrn: MetaKernel instance with SDS and mission kernels
        - dynamic_kernels: List of dynamic kernel file paths
        - param_kernels: List of parameter-specific kernel file paths
    ugps_times_modified : array-like
        Time array (possibly modified by OFFSET_TIME parameter)
    tlm_dataset : pd.DataFrame
        Spacecraft state telemetry data
    calibration : CalibrationData
        NamedTuple containing:
        - los_vectors: Pre-loaded LOS vectors (or None)
        - optical_psfs: Pre-loaded optical PSF (or None)
    image_matching_func : callable
        Function to perform image matching; defaults to the built-in
        ``image_matching`` function when not overridden in config.
    match_ctx : ImageMatchingContext
        NamedTuple containing:
        - gcp_pairs: List of GCP pairing tuples
        - params: List of (ParameterConfig, parameter_value) tuples
        - pair_idx: Index of current GCP pair
        - sci_key: Science dataset identifier for this pair

    Returns
    -------
    geo_dataset : xr.Dataset
        Geolocated points with latitude, longitude, altitude
    image_matching_output : xr.Dataset
        Matching results with error measurements and metadata

    Examples
    --------
    >>> kernel_ctx = KernelContext(mkrn, dynamic_kernels, param_kernels)
    >>> calibration = CalibrationData(los_vectors, optical_psfs)
    >>> match_ctx = ImageMatchingContext(gcp_pairs, params, 0, "sci_001")
    >>> geo, matching = _geolocate_and_match(
    ...     config, kernel_ctx, times, tlm_dataset,
    ...     calibration, integrated_image_match, match_ctx
    ... )
    """
    logger.info("    Performing geolocation...")
    with sp.ext.load_kernel(
        [
            kernel_ctx.mkrn.sds_kernels,
            kernel_ctx.mkrn.mission_kernels,
            kernel_ctx.dynamic_kernels,
            kernel_ctx.param_kernels,
        ]
    ):
        geoloc_inst = spatial.Geolocate(config.geo.instrument_name)
        geo_dataset = geoloc_inst(ugps_times_modified)

        # === IMAGE MATCHING MODULE ===
        logger.info("    === IMAGE MATCHING MODULE ===")

        # Use injected image matching function
        gcp_file = Path(match_ctx.gcp_pairs[0][1])

        # All image matching functions use the same signature
        image_matching_output = image_matching_func(
            geolocated_data=geo_dataset,
            gcp_reference_file=gcp_file,
            telemetry=tlm_dataset,
            calibration_dir=config.calibration_dir,
            params_info=match_ctx.params,
            config=config,
            los_vectors_cached=calibration.los_vectors,
            optical_psfs_cached=calibration.optical_psfs,
        )
        validate_image_matching_output(image_matching_output)
        logger.info("    Image matching complete")

        logger.info(f"    Generated error measurements for {len(image_matching_output.measurement)} points")

        # Store metadata for tracking
        image_matching_output.attrs["gcp_pair_index"] = match_ctx.pair_idx
        image_matching_output.attrs["gcp_pair_id"] = f"{match_ctx.sci_key}_pair_{match_ctx.pair_idx}"

    return geo_dataset, image_matching_output


def loop(
    config: CorrectionConfig,
    work_dir: Path,
    tlm_sci_gcp_sets: list[tuple[str, str, str]],
    resume_from_checkpoint: bool = False,
):
    """
    Correction loop for parameter sensitivity analysis.

    Parameters
    ----------
    config : CorrectionConfig
        The single configuration containing all settings:
        - Required: parameters, iterations, thresholds, geo config
        - Data loading: ``data`` (:class:`~curryer.correction.config.DataConfig`)
          specifying file format and time scaling
        - Optional: ``_image_matching_override`` override on ``config``
          (test injection only)
        - Calibration: `calibration_dir` (if the image-matching override uses calibration)
        - Output: netcdf, output_filename
    work_dir : Path
        Working directory for temporary files.
    tlm_sci_gcp_sets : list of (str, str, str)
        List of (`telemetry_key`, `science_key`, `gcp_key`) tuples.
    resume_from_checkpoint : bool, optional
        If True, resume from an existing checkpoint.

    Returns
    -------
    results : list
        List of iteration results (order: `pair_idx * N + param_idx`).
    netcdf_data : dict
        Dictionary of NetCDF variables indexed as `[param_idx, pair_idx]`.

    Notes
    -----
    This implementation uses a pair-outer, parameter-inner loop order:
    - Outer loop: GCP pairs (load data once per image)
    - Inner loop: Parameter sets (reuse loaded data)
    This reduces file I/O and centralizes mission-specific behavior through the
    `config` object.

    Examples
    --------
    Correction mode (parameter optimization)::

        from curryer.correction.config import CorrectionConfig, DataConfig

        config = CorrectionConfig(
            seed=42,
            n_iterations=100,
            parameters=parameters,
            geo=geo_config,
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
            data=DataConfig(file_format="csv", time_scale_factor=1e6),
        )
        results, netcdf_data = loop(config, work_dir, tlm_sci_gcp_sets)

    Where each element of ``tlm_sci_gcp_sets`` is a tuple of file paths::

        tlm_sci_gcp_sets = [
            ("telemetry.csv", "science.csv", "landsat_chip_001.mat"),
        ]
    """
    logger.info("=== CORRECTION PIPELINE ===")
    logger.info(f"  GCP pairs: {len(tlm_sci_gcp_sets)} (outer loop - load data once)")

    # Use injected image matching function override, or fall back to built-in implementation
    image_matching_func = getattr(config, "_image_matching_override", None) or image_matching

    # Initialize parameter sets
    params_set = load_param_sets(config)
    logger.info(f"  Parameter sets: {len(params_set)} (inner loop)")

    # Build NetCDF data structure
    n_param_sets = len(params_set)
    n_gcp_pairs = len(tlm_sci_gcp_sets)

    # Try to load checkpoint if resuming
    output_file = work_dir / config.get_output_filename()
    start_pair_idx = 0
    # Currently, checkpoint is bugged, since the nadir equivalent stats are not calculated until the end.
    # TODO [CURRYER-100]: Fix checkpoint resume for Monte Carlo GCS
    if resume_from_checkpoint:
        checkpoint_data, completed_pairs = _load_checkpoint(output_file, config)
        if checkpoint_data is not None:
            netcdf_data = checkpoint_data
            start_pair_idx = completed_pairs
            logger.info(f"Resuming from checkpoint: starting at GCP pair {start_pair_idx + 1}/{n_gcp_pairs}")
        else:
            netcdf_data = _build_netcdf_structure(config, n_param_sets, n_gcp_pairs)
            logger.info("No valid checkpoint found, starting from beginning")
    else:
        netcdf_data = _build_netcdf_structure(config, n_param_sets, n_gcp_pairs)

    # Initialize results dict with (param_idx, pair_idx) keys
    # This avoids nested search complexity when aggregating statistics
    results_dict = {}

    # Prepare SPICE environment
    mkrn = meta.MetaKernel.from_json(
        config.geo.meta_kernel_file,
        relative=True,
        sds_dir=config.geo.generic_kernel_dir,
    )
    creator = create.KernelCreator(overwrite=True, append=False)

    # Load calibration data once (LOS vectors and optical PSF are static instrument calibration)
    calibration_data = _load_calibration_data(config)

    # Store parameter values once (before loops)
    for param_idx, params in enumerate(params_set):
        param_values = _extract_parameter_values(params)
        _store_parameter_values(netcdf_data, param_idx, param_values)

    # OUTER LOOP: Iterate through GCP pairs
    for pair_idx, (tlm_key, sci_key, gcp_key) in enumerate(tlm_sci_gcp_sets):
        # Skip already-completed pairs if resuming
        if pair_idx < start_pair_idx:
            logger.info(f"=== GCP Pair {pair_idx + 1}/{n_gcp_pairs}: {sci_key} === (SKIPPED - already completed)")
            continue

        logger.info(f"=== GCP Pair {pair_idx + 1}/{n_gcp_pairs}: {sci_key} ===")

        # Load image pair data once (internal file-based loading)
        tlm_dataset, sci_dataset, ugps_times = _load_image_pair_data(tlm_key, sci_key, config)

        # Create dynamic kernels once (these don't change with parameters)
        dynamic_kernels = _create_dynamic_kernels(config, work_dir, tlm_dataset, creator)

        # Use gcp_key directly as the GCP file path — no pairing function needed.
        # Users specify exactly which GCP file pairs with each science file in
        # tlm_sci_gcp_sets.  An empty string disables image matching for that pair.
        gcp_pairs = [(sci_key, gcp_key)]
        logger.info(f"  GCP file: {gcp_key or '(none)'}")

        # INNER LOOP: Iterate through parameter sets
        for param_idx, params in enumerate(params_set):
            logger.info(f"  Parameter Set {param_idx + 1}/{n_param_sets}")

            # Create parameter-specific kernels (these change with parameters)
            param_kernels, ugps_times_modified = _create_parameter_kernels(
                params, work_dir, tlm_dataset, sci_dataset, ugps_times, config, creator
            )

            # Prepare context objects for cleaner function call
            kernel_ctx = KernelContext(mkrn=mkrn, dynamic_kernels=dynamic_kernels, param_kernels=param_kernels)
            match_ctx = ImageMatchingContext(gcp_pairs=gcp_pairs, params=params, pair_idx=pair_idx, sci_key=sci_key)

            # Geolocate and perform image matching
            geo_dataset, image_matching_output = _geolocate_and_match(
                config,
                kernel_ctx,
                ugps_times_modified,
                tlm_dataset,
                calibration_data,
                image_matching_func,
                match_ctx,
            )

            # Compute nadir-equivalent errors for this GCP pair.
            # compute_nadir_equivalent_errors() skips aggregate statistics —
            # computing mean/std/percentiles on a single GCP pair is
            # mathematically uninformative and wastes time in a tight loop.
            from curryer.correction.error_stats import ErrorStatsConfig, ErrorStatsProcessor

            error_config = ErrorStatsConfig.from_correction_config(config)
            processor = ErrorStatsProcessor(config=error_config)
            individual_nadir = processor.compute_nadir_equivalent_errors(image_matching_output)

            nadir_errors = individual_nadir["nadir_equiv_total_error_m"].values
            if len(nadir_errors) == 1:
                nadir_error = float(nadir_errors[0])
                individual_metrics = {
                    "rms_error_m": nadir_error,
                    "mean_error_m": nadir_error,
                    "max_error_m": nadir_error,
                    "std_error_m": 0.0,
                    "n_measurements": 1,
                }
            else:
                individual_metrics = {
                    "rms_error_m": float(np.sqrt(np.mean(nadir_errors**2))),
                    "mean_error_m": float(np.mean(nadir_errors)),
                    "max_error_m": float(np.max(nadir_errors)),
                    "std_error_m": float(np.std(nadir_errors)),
                    "n_measurements": len(nadir_errors),
                }
            individual_stats = individual_nadir

            # Store results in NetCDF (maintain [param_idx, pair_idx] ordering)
            _store_gcp_pair_results(netcdf_data, param_idx, pair_idx, individual_metrics)
            netcdf_data["im_lat_error_km"][param_idx, pair_idx] = image_matching_output.attrs.get(
                "lat_error_km", np.nan
            )
            netcdf_data["im_lon_error_km"][param_idx, pair_idx] = image_matching_output.attrs.get(
                "lon_error_km", np.nan
            )
            netcdf_data["im_ccv"][param_idx, pair_idx] = image_matching_output.attrs.get("correlation_ccv", np.nan)
            netcdf_data["im_grid_step_m"][param_idx, pair_idx] = image_matching_output.attrs.get(
                "final_grid_step_m", np.nan
            )

            # Store results in dict with (param_idx, pair_idx) key
            # Note: iteration index reflects reversed order (pair_idx * n_params + param_idx)
            param_values = _extract_parameter_values(params)
            iteration_result = {
                "iteration": pair_idx * n_param_sets + param_idx,
                "pair_index": pair_idx,
                "param_index": param_idx,
                "parameters": param_values,
                "geolocation": geo_dataset,
                "gcp_pairs": gcp_pairs,
                "image_matching": image_matching_output,
                "error_stats": individual_stats,
                "rms_error_m": individual_metrics["rms_error_m"],
                "aggregate_rms_error_m": None,
            }
            results_dict[(param_idx, pair_idx)] = iteration_result

            logger.info(
                f"    RMS error: {individual_metrics['rms_error_m']:.2f}m "
                f"({individual_metrics['n_measurements']} measurements)"
            )

        logger.info(f"  GCP pair {pair_idx + 1} complete (processed {n_param_sets} parameter sets)")

        # Save checkpoint after each pair completes
        if resume_from_checkpoint:
            _save_netcdf_checkpoint(netcdf_data, output_file, config, pair_idx)

    # Compute aggregate statistics for each parameter set (after all pairs complete)
    logger.info("=== Computing aggregate statistics for all parameter sets ===")
    for param_idx in range(n_param_sets):
        # Collect all image matching results for this parameter set
        param_image_matching_results = []
        for pair_idx in range(n_gcp_pairs):
            result = results_dict.get((param_idx, pair_idx))
            if result:
                param_image_matching_results.append(result["image_matching"])

        # Compute aggregate statistics
        aggregate_stats = call_error_stats_module(param_image_matching_results, correction_config=config)
        aggregate_error_metrics = _extract_error_metrics(aggregate_stats)

        # Extract pair errors for threshold calculation
        pair_errors = [netcdf_data["rms_error_m"][param_idx, pair_idx] for pair_idx in range(n_gcp_pairs)]
        _compute_parameter_set_metrics(netcdf_data, param_idx, pair_errors, threshold_m=config.performance_threshold_m)

        logger.info(f"  Parameter set {param_idx + 1}: Aggregate RMS = {aggregate_error_metrics['rms_error_m']:.2f}m")

        # Add aggregate stats to all results for this parameter set
        for pair_idx in range(n_gcp_pairs):
            key = (param_idx, pair_idx)
            if key in results_dict:
                results_dict[key]["aggregate_error_stats"] = aggregate_stats
                results_dict[key]["aggregate_rms_error_m"] = aggregate_error_metrics["rms_error_m"]
    # Convert results_dict back to list for backward compatibility
    # Sort by iteration index to maintain consistent ordering
    results = [results_dict[key] for key in sorted(results_dict.keys(), key=lambda k: results_dict[k]["iteration"])]

    # Save final NetCDF results
    _save_netcdf_results(netcdf_data, output_file, config)

    # Clean up checkpoint file after successful completion
    if resume_from_checkpoint:
        _cleanup_checkpoint(output_file)

    logger.info(f"=== Loop Complete: Processed {n_gcp_pairs} GCP pairs × {n_param_sets} parameter sets ===")
    logger.info(f"  Total iterations: {len(results)}")
    logger.info(f"  NetCDF output: {output_file}")

    return results, netcdf_data


def _extract_parameter_values(params):
    """Extract parameter values from a parameter set into a dictionary."""
    param_values = {}

    for param_config, param_data in params:
        if param_config.config_file:
            param_name = param_config.config_file.stem

            if param_config.ptype == ParameterType.CONSTANT_KERNEL:
                # Extract roll, pitch, yaw from DataFrame
                if isinstance(param_data, pd.DataFrame) and "angle_x" in param_data.columns:
                    # Convert back to arcseconds for storage
                    param_values[f"{param_name}_roll"] = np.degrees(param_data["angle_x"].iloc[0]) * 3600
                    param_values[f"{param_name}_pitch"] = np.degrees(param_data["angle_y"].iloc[0]) * 3600
                    param_values[f"{param_name}_yaw"] = np.degrees(param_data["angle_z"].iloc[0]) * 3600

            elif param_config.ptype == ParameterType.OFFSET_KERNEL:
                # Single bias value (keep in original units)
                param_values[param_name] = param_data

            elif param_config.ptype == ParameterType.OFFSET_TIME:
                # Time correction (keep in original units)
                param_values[param_name] = param_data

    return param_values


def _store_parameter_values(netcdf_data, param_idx, param_values):
    """Store parameter values in the NetCDF data structure.

    This function maps parameter names to NetCDF variable names for storage.
    It handles the naming convention used by _build_netcdf_structure.
    """

    for param_name, value in param_values.items():
        # Generate NetCDF variable name using same logic as _build_netcdf_structure
        # Replace dots and dashes with underscores, ensure param_ prefix
        netcdf_var = param_name.replace(".", "_").replace("-", "_")
        if not netcdf_var.startswith("param_"):
            netcdf_var = f"param_{netcdf_var}"

        if netcdf_var in netcdf_data:
            netcdf_data[netcdf_var][param_idx] = value
            logger.debug(f"  Stored {netcdf_var}[{param_idx}] = {value}")
        else:
            # Try to find a matching variable with debug info
            logger.warning(
                f"  Parameter variable '{netcdf_var}' not found in netcdf_data. Available keys: {[k for k in netcdf_data.keys() if k.startswith('param_')]}"
            )


def _extract_error_metrics(stats_dataset):
    """Extract error metrics from error statistics dataset."""
    if hasattr(stats_dataset, "attrs"):
        # Real error stats module
        return {
            "rms_error_m": stats_dataset.attrs.get("rms_error_m", np.nan),
            "mean_error_m": stats_dataset.attrs.get("mean_error_m", np.nan),
            "max_error_m": stats_dataset.attrs.get("max_error_m", np.nan),
            "std_error_m": stats_dataset.attrs.get("std_error_m", np.nan),
            "n_measurements": stats_dataset.attrs.get("total_measurements", 0),
        }
    else:
        # Fallback for placeholder
        return {
            "rms_error_m": float(stats_dataset.get("rms_error", np.nan)),
            "mean_error_m": float(stats_dataset.get("mean_error", np.nan)),
            "max_error_m": float(stats_dataset.get("max_error", np.nan)),
            "std_error_m": float(stats_dataset.get("std_error", np.nan)),
            "n_measurements": int(stats_dataset.get("n_measurements", 0)),
        }


def _store_gcp_pair_results(netcdf_data, param_idx, pair_idx, error_metrics):
    """Store GCP pair results in the NetCDF data structure."""
    netcdf_data["rms_error_m"][param_idx, pair_idx] = error_metrics["rms_error_m"]
    netcdf_data["mean_error_m"][param_idx, pair_idx] = error_metrics["mean_error_m"]
    netcdf_data["max_error_m"][param_idx, pair_idx] = error_metrics["max_error_m"]
    netcdf_data["std_error_m"][param_idx, pair_idx] = error_metrics["std_error_m"]
    netcdf_data["n_measurements"][param_idx, pair_idx] = error_metrics["n_measurements"]


def _compute_parameter_set_metrics(netcdf_data, param_idx, pair_errors, threshold_m=250.0):
    """
    Compute overall performance metrics for a parameter set.

    Args:
        netcdf_data: NetCDF data dictionary
        param_idx: Parameter set index
        pair_errors: Array of RMS errors for each GCP pair
        threshold_m: Performance threshold in meters
    """
    pair_errors = np.array(pair_errors)
    valid_errors = pair_errors[~np.isnan(pair_errors)]

    if len(valid_errors) > 0:
        # Percentage of pairs with error < threshold
        # Find the threshold metric key dynamically
        threshold_metric = None
        for key in netcdf_data.keys():
            if key.startswith("percent_under_") and key.endswith("m"):
                threshold_metric = key
                break

        if threshold_metric:
            percent_under_threshold = (valid_errors < threshold_m).sum() / len(valid_errors) * 100
            netcdf_data[threshold_metric][param_idx] = percent_under_threshold

        # Mean RMS across all pairs
        netcdf_data["mean_rms_all_pairs"][param_idx] = np.mean(valid_errors)

        # Best and worst pair performance
        netcdf_data["best_pair_rms"][param_idx] = np.min(valid_errors)
        netcdf_data["worst_pair_rms"][param_idx] = np.max(valid_errors)


# =============================================================================
# Incremental NetCDF Saving (Checkpoint/Resume)
# =============================================================================

# =============================================================================
# Preferred-name aliases  (backward-compat originals kept above)
# =============================================================================


def run_correction(
    config: CorrectionConfig,
    work_dir: Path,
    inputs: Sequence[CorrectionInput | tuple[str, str, str]],
    resume_from_checkpoint: bool = False,
) -> "CorrectionResult":
    """Run the correction parameter sweep.

    This is the preferred user-facing entry point (compared to :func:`loop`).
    Returns a structured :class:`~curryer.correction.results.CorrectionResult`
    with the best parameter set, pass/fail verdict, recommendation, and a
    human-readable summary table.  The raw ``results`` list and ``netcdf_data``
    dict from :func:`loop` are available as ``result.results`` and
    ``result.netcdf_data`` for advanced use.

    Parameters
    ----------
    config : CorrectionConfig
        Full correction configuration.
    work_dir : Path
        Working directory for temporary files.
    inputs : list of CorrectionInput or list of (str, str, str)
        Each element is either a :class:`~curryer.correction.config.CorrectionInput`
        (named fields) or a legacy ``(telemetry_key, science_key, gcp_key)`` tuple.
        Both forms may be mixed in the same list.
    resume_from_checkpoint : bool, optional
        If True, resume from an existing checkpoint.

    Returns
    -------
    CorrectionResult
        Structured result with best parameters, pass/fail verdict,
        recommendation, summary table, and raw NetCDF/intermediate data
        available on the returned object (for example,
        ``result.netcdf_data``).
    """
    from curryer.correction.results import build_correction_result

    run_start = time.time()

    normalized: list[tuple[str, str, str]] = []
    for inp in inputs:
        if isinstance(inp, CorrectionInput):
            normalized.append((str(inp.telemetry_file), str(inp.science_file), str(inp.gcp_file)))
        else:
            normalized.append(inp)

    results, netcdf_data = loop(config, work_dir, normalized, resume_from_checkpoint)
    elapsed = time.time() - run_start
    netcdf_path = work_dir / config.get_output_filename()

    correction_result = build_correction_result(
        config=config,
        results=results,
        netcdf_data=netcdf_data,
        netcdf_path=netcdf_path,
        elapsed_time_s=elapsed,
    )

    logger.info("\n%s", correction_result.summary_table)
    logger.info(correction_result.recommendation)

    return correction_result


def compute_error_stats(image_matching_results, correction_config: "CorrectionConfig"):
    """Compute error statistics from image matching results.

    This is the preferred name for :func:`call_error_stats_module`.
    See :func:`call_error_stats_module` for full documentation.

    Parameters
    ----------
    image_matching_results : xr.Dataset or list of xr.Dataset
        Output from image matching, either a single dataset or a list.
    correction_config : CorrectionConfig
        Correction configuration used to initialise the error stats processor.

    Returns
    -------
    xr.Dataset
        Aggregate error statistics dataset.
    """
    return call_error_stats_module(image_matching_results, correction_config)


def run_image_matching(
    geolocated_data: "xr.Dataset",
    gcp_reference_file: Path,
    telemetry: "pd.DataFrame",
    calibration_dir: Path,
    params_info: list,
    config: "CorrectionConfig",
    los_vectors_cached: "np.ndarray | None" = None,
    optical_psfs_cached: "list | None" = None,
) -> "xr.Dataset":
    """Run image matching against GCP reference.

    This is the preferred name for :func:`image_matching`.
    See :func:`image_matching` for full documentation.

    Parameters
    ----------
    geolocated_data : xr.Dataset
        Geolocated scene data with latitude/longitude.
    gcp_reference_file : Path
        Path to the GCP reference image (.mat file).
    telemetry : pd.DataFrame
        Telemetry DataFrame with spacecraft state.
    calibration_dir : Path
        Directory containing calibration files.
    params_info : list
        Parameter information for the current iteration.
    config : CorrectionConfig
        Full correction configuration.
    los_vectors_cached : np.ndarray or None, optional
        Pre-loaded LOS vectors; loaded from disk if None.
    optical_psfs_cached : list or None, optional
        Pre-loaded optical PSFs; loaded from disk if None.

    Returns
    -------
    xr.Dataset
        Image matching results dataset.
    """
    return image_matching(
        geolocated_data,
        gcp_reference_file,
        telemetry,
        calibration_dir,
        params_info,
        config,
        los_vectors_cached,
        optical_psfs_cached,
    )
