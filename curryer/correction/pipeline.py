"""Main correction pipeline orchestration.

This module contains the public-facing :func:`loop` function that drives
the Monte Carlo parameter sensitivity analysis, plus all of the helper
functions it calls:

- Adapter functions that bridge between the geolocation/image-matching
  sub-modules and the correction loop.
- :func:`_load_file` -- internal helper that reads CSV/NetCDF/HDF5 files
  into DataFrames, replacing the old mission-specific loader callables.
- :func:`load_loop_observation`, :func:`_load_calibration_data`,
  :func:`_geolocate_observation` -- per-pair and per-iteration helpers.
- :func:`loop` -- outer GCP-pair loop, inner parameter-set loop.

Architecture note
-----------------
The correction pipeline follows the **pipeline → verification** dependency
direction.  The core image-matching and aggregation logic lives in
``verification.py``; ``pipeline.py`` imports it from there.  This means
*verification* is a standalone module (GCP pairing + image matching +
error stats) that the correction loop reuses for its own last three steps.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from curryer.correction.results import CorrectionResult

from curryer import meta
from curryer import spicierpy as sp
from curryer.compute import elevation, spatial
from curryer.compute.constants import SpatialQualityFlags as SQF
from curryer.correction.config import (
    _CONSTANT_KERNEL_AXES,
    CalibrationData,
    CorrectionInput,
    GeolocationSetup,
    NetCDFConfig,
    OutputConfig,
    ParameterType,
    Sweep,
)
from curryer.correction.dataio import validate_telemetry_output
from curryer.correction.error_stats import ErrorStatsConfig, ErrorStatsProcessor
from curryer.correction.grid_types import ImageGrid
from curryer.correction.image_io import (
    load_image_grid,
    load_los_vectors,
    load_optical_psf,
    observation_los_vectors,
)
from curryer.correction.image_match import (
    validate_image_matching_output,
)
from curryer.correction.io import resolve_path
from curryer.correction.kernel_ops import (
    _create_dynamic_kernels,
    _create_parameter_kernels,
)
from curryer.correction.parameters import _rad_to_val, _seconds_to_val, load_param_sets
from curryer.correction.results_io import (
    _build_netcdf_structure,
    _cleanup_checkpoint,
    _load_checkpoint,
    _save_netcdf_checkpoint,
    _save_netcdf_results,
)

# Import from verification — the correct dependency direction.
# verification.py owns the "GCP pairing → image matching → error stats" pipeline.
from curryer.correction.verification import (
    _aggregate_image_matching_results,
    image_matching,
    match_observation,
)
from curryer.kernels import create

logger = logging.getLogger(__name__)

# ============================================================================
# ADAPTER FUNCTIONS
# ============================================================================
# image_matching() and _aggregate_image_matching_results() live in
# verification.py and are imported above (pipeline → verification direction).
# ============================================================================


def call_error_stats_module(image_matching_results, setup: "GeolocationSetup"):
    """
    Call the error_stats module with image matching output.

    Args:
        image_matching_results: Either a single image matching result (xarray.Dataset)
                              or a list of image matching results from multiple GCP pairs
        setup: GeolocationSetup with variable names and thresholds (REQUIRED)

    Returns:
        Aggregate error statistics dataset
    """
    # Handle both single result and list of results
    if not isinstance(image_matching_results, list):
        image_matching_results = [image_matching_results]

    logger.info(f"Error Statistics: Processing geolocation errors from {len(image_matching_results)} GCP pairs")

    # Create error stats config directly from the geolocation setup (single source of truth)
    processor = ErrorStatsProcessor(config=ErrorStatsConfig.from_setup(setup))

    if len(image_matching_results) == 1:
        return processor.process_geolocation_errors(image_matching_results[0])
    aggregated_data = _aggregate_image_matching_results(image_matching_results, setup)
    return processor.process_geolocation_errors(aggregated_data)


# _aggregate_image_matching_results and image_matching live in verification.py
# and are imported above for use within this module (pipeline → verification).


def _load_file(file_path: str | Path, file_format: str = "csv") -> pd.DataFrame:
    """Load a telemetry file into a pandas DataFrame.

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


def _load_calibration_data(setup: "GeolocationSetup") -> CalibrationData:
    """Load LOS vectors and optical PSF when ``setup.calibration`` is configured.

    This function centralizes calibration data loading, which is now called once
    per GCP pair in the optimized implementation (previously called once per parameter set).

    Parameters
    ----------
    setup : GeolocationSetup
        Setup carrying optional :class:`~curryer.correction.config.CalibrationFiles`
        with direct ``los_vectors_file`` / ``psf_file`` paths.

    Returns
    -------
    CalibrationData
        NamedTuple containing (los_vectors, optical_psfs), or (None, None) when
        no calibration files are configured.

    Raises
    ------
    FileNotFoundError
        If a calibration file is configured but does not exist.
    ValueError
        If a calibration file exists but fails to load properly.

    Note
    ----
    Calibration files are optional and interim: real LOS/spacecraft geometry will
    be SPICE-derived.  When ``setup.calibration`` is ``None`` (or both file fields
    are ``None``), returns ``CalibrationData(None, None)`` so that missions without
    calibration files still work.

    Examples
    --------
    >>> calib_data = _load_calibration_data(setup)
    >>> if calib_data.los_vectors is not None:
    ...     # Use calibration data in image matching
    ...     pass
    """
    calibration = setup.calibration
    if calibration is None or (calibration.los_vectors_file is None and calibration.psf_file is None):
        return CalibrationData(los_vectors=None, optical_psfs=None)

    logger.info("Loading calibration data...")

    # ---- LOS vectors ----
    if calibration.los_vectors_file is None:
        raise ValueError("No LOS vectors source configured. Set setup.calibration.los_vectors_file.")
    los_file = Path(calibration.los_vectors_file)

    if not los_file.exists():
        raise FileNotFoundError(
            f"LOS vectors calibration file not found: {los_file}\n"
            f"Set setup.calibration.los_vectors_file to the correct path."
        )

    los_vectors_cached = load_los_vectors(los_file)

    if los_vectors_cached is None:
        raise ValueError(
            f"Failed to load LOS vectors from {los_file}. File exists but load_los_vectors() returned None."
        )

    # ---- Optical PSF ----
    if calibration.psf_file is None:
        raise ValueError("No PSF source configured. Set setup.calibration.psf_file.")
    psf_file = Path(calibration.psf_file)

    if not psf_file.exists():
        raise FileNotFoundError(
            f"Optical PSF calibration file not found: {psf_file}\nSet setup.calibration.psf_file to the correct path."
        )

    optical_psfs_cached = load_optical_psf(psf_file)

    if optical_psfs_cached is None:
        raise ValueError(
            f"Failed to load optical PSF from {psf_file}. File exists but load_optical_psf() returned None."
        )

    logger.info(f"  Cached LOS vectors: {los_vectors_cached.shape}")
    logger.info(f"  Cached optical PSF: {len(optical_psfs_cached)} entries")

    return CalibrationData(los_vectors=los_vectors_cached, optical_psfs=optical_psfs_cached)


def _require_image_matching_inputs(setup: "GeolocationSetup", calibration_data: CalibrationData) -> None:
    """Fail fast when the loop has no calibration to run with.

    The loop re-geolocates each observation with the LOS table, so LOS vectors
    are always required; the built-in matching (no
    ``setup.observation_matching_func``) also needs the optical PSF.  Raising
    here — before kernel creation — gives a clear, early error.

    Raises
    ------
    ValueError
        If no LOS vectors are configured, or no override is set and no PSF is
        configured.
    """
    if calibration_data.los_vectors is None:
        raise ValueError(
            "The correction loop geolocates each observation with the LOS table, but no calibration data is "
            "configured. Set setup.calibration (los_vectors_file and psf_file)."
        )
    if setup.observation_matching_func is None and calibration_data.optical_psfs is None:
        raise ValueError(
            "No setup.observation_matching_func override is set, so the built-in matching is used, but no "
            "optical PSF is configured. Set setup.calibration.psf_file."
        )


def _load_telemetry(tlm_key: str, setup: "GeolocationSetup") -> pd.DataFrame:
    """Load the telemetry file the dynamic kernels are built from.

    Parameters
    ----------
    tlm_key : str
        Path to the telemetry file, read with ``setup.data_config.file_format``
        (CSV when ``setup.data_config`` is ``None``).
    setup : GeolocationSetup
        Setup supplying the file format.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the telemetry is empty.
    """
    file_format = setup.data_config.file_format if setup.data_config is not None else "csv"
    tlm_dataset = _load_file(tlm_key, file_format)
    validate_telemetry_output(tlm_dataset, setup)
    return tlm_dataset


class LoopObservation(NamedTuple):
    """An observation subimage as the correction loop uses it."""

    radiance: np.ndarray
    """``(n_frames, n_columns)`` radiance; rows are frames along track."""
    frame_ugps: np.ndarray
    """``(n_frames,)`` int64 frame times, microseconds since the GPS epoch."""
    los_vectors: np.ndarray
    """``(n_columns, 3)`` instrument-frame line of sight of each column."""


def load_loop_observation(path: str | Path, los_vectors: np.ndarray) -> LoopObservation:
    """Read an observation subimage for the correction loop.

    The file is NetCDF with:

    - ``band_data`` (frame, pixel): radiance, finite;
    - ``ugps`` (frame): integer frame times, microseconds since the GPS epoch
      (1980-01-06), strictly increasing;
    - ``detector_pixel`` (pixel): the row of the LOS table for each column
      (:func:`~curryer.correction.image_io.observation_los_vectors`); without it
      the columns must be the table's rows in order.

    Other variables (e.g. the product's own ``lat``/``lon``) are not read: each
    parameter set re-geolocates the frames and columns from SPICE.

    Parameters
    ----------
    path : str or Path
        Observation file (local path or ``s3://`` URI).
    los_vectors : ndarray, shape (n_pixels, 3)
        Instrument-frame LOS table (:func:`~curryer.correction.image_io.load_los_vectors`).

    Returns
    -------
    LoopObservation

    Raises
    ------
    ValueError
        If ``band_data`` is not 2-D and finite, ``ugps`` is missing, not 1-D
        integers of one per frame or not strictly increasing, or the columns do
        not map onto the LOS table.
    """
    name = Path(str(path)).name
    with xr.open_dataset(resolve_path(path)) as ds:
        if "band_data" not in ds or "ugps" not in ds:
            raise ValueError(f"{name} must carry 'band_data' and 'ugps'; it has {list(ds.data_vars)}.")
        radiance = np.asarray(ds["band_data"].values, dtype=float)
        frame_ugps = np.asarray(ds["ugps"].values)

    if radiance.ndim != 2 or not np.all(np.isfinite(radiance)):
        raise ValueError(
            f"'band_data' in {name} must be a finite 2-D (frame, pixel) array, got shape {radiance.shape}."
        )
    if (
        frame_ugps.ndim != 1
        or frame_ugps.dtype.kind not in "iu"
        or len(frame_ugps) != radiance.shape[0]
        or np.any(np.diff(frame_ugps.astype(np.int64)) <= 0)
    ):
        raise ValueError(
            f"'ugps' in {name} must be {radiance.shape[0]} strictly increasing integers (one per frame); "
            f"got dtype {frame_ugps.dtype}, shape {frame_ugps.shape}."
        )
    columns_los = observation_los_vectors(path, los_vectors, radiance.shape[1])
    return LoopObservation(radiance=radiance, frame_ugps=frame_ugps.astype(np.int64), los_vectors=columns_los)


def _geolocate_observation(
    instrument_name: str,
    frame_ugps: np.ndarray,
    observation: LoopObservation,
    elev: elevation.Elevation,
    pad_degrees: float = 1.0,
) -> tuple[ImageGrid, np.ndarray]:
    """Geolocate an observation's frames and columns with the loaded SPICE kernels.

    Each column's LOS vector is intersected with the WGS-84 ellipsoid at each
    frame time and terrain-corrected with *elev*.

    Parameters
    ----------
    instrument_name : str
        SPICE instrument whose frame the LOS vectors are in.
    frame_ugps : ndarray, shape (n_frames,)
        Frame times, uGPS (after any OFFSET_TIME).
    observation : LoopObservation
        Radiance and LOS vectors.
    elev : Elevation
        Elevation source, kilometers and radians.
    pad_degrees : float, optional
        Padding of the DEM region around the ellipsoid intersections, degrees.

    Returns
    -------
    grid : ImageGrid
        The radiance on the geolocated grid: ``lat``/``lon`` degrees, ``h``
        meters above the WGS-84 ellipsoid.
    r_spacecraft_m : ndarray, shape (3,)
        Spacecraft ITRF93 position in meters at the middle frame.

    Raises
    ------
    ValueError
        If any pixel cannot be geolocated (no kernel coverage at a frame time,
        LOS misses the ellipsoid) or terrain-corrected.
    """
    n_frames, n_columns = observation.radiance.shape
    surface_km, sc_km, qf = spatial.compute_ellipsoid_intersection(
        frame_ugps, instrument_name, custom_pointing_vectors=observation.los_vectors
    )
    failed = qf.values != SQF.GOOD
    if failed.any():
        raise ValueError(
            f"{int(failed.sum())} of {failed.size} subimage pixels could not be intersected with the ellipsoid "
            f"(quality flags {sorted({int(v) for v in qf.values[failed]})}); check the kernels cover the frame times."
        )

    ellipsoid_lla = spatial.ecef_to_geodetic(surface_km.values, degrees=True)
    lon_min, lon_max = spatial.minmax_lon(ellipsoid_lla[:, 0], degrees=True)
    region = elev.local_region(
        *np.deg2rad(
            [
                lon_min - pad_degrees,
                lon_max + pad_degrees,
                ellipsoid_lla[:, 1].min() - pad_degrees,
                ellipsoid_lla[:, 1].max() + pad_degrees,
            ]
        )
    )
    terrain_lla, terrain_qf = spatial.terrain_correct(region, surface_km.values, sc_km.values)
    failed = terrain_qf != SQF.GOOD
    if failed.any():
        raise ValueError(
            f"{int(failed.sum())} of {failed.size} subimage pixels could not be terrain-corrected "
            f"(quality flags {sorted({int(v) for v in terrain_qf[failed]})})."
        )

    grid = ImageGrid(
        data=observation.radiance,
        lat=terrain_lla[:, 1].reshape(n_frames, n_columns),
        lon=terrain_lla[:, 0].reshape(n_frames, n_columns),
        h=terrain_lla[:, 2].reshape(n_frames, n_columns) * 1e3,
    )
    r_spacecraft_m = sc_km.values.reshape(n_frames, n_columns, 3)[n_frames // 2, 0] * 1e3
    return grid, r_spacecraft_m


def _resolve_netcdf_config(setup: "GeolocationSetup", output: "OutputConfig") -> "NetCDFConfig":
    """Return the output's NetCDFConfig with the threshold pinned to the requirement.

    ``setup.requirements.performance_threshold_m`` is the single source of truth for
    the threshold: it drives the computed pass-rate metric *and* the output variable's
    name and metadata.  Pinning it here keeps the written NetCDF accurate to what is
    actually computed, even when a caller supplies an ``OutputConfig.netcdf`` — whose
    other fields (title, description, parameter metadata) are preserved.
    """
    threshold_m = setup.requirements.performance_threshold_m
    if output.netcdf is None:
        return NetCDFConfig(performance_threshold_m=threshold_m)
    return output.netcdf.model_copy(update={"performance_threshold_m": threshold_m})


def _per_pair_error_processor(setup: GeolocationSetup) -> ErrorStatsProcessor:
    """Return the error-stats processor for per-pair errors inside :func:`loop`.

    Per-pair errors are not gated by ``setup.geo.minimum_correlation`` or
    ``minimum_peak_margin``: a weak match under one parameter set is an
    ordinary sweep outcome, not a reason to stop the sweep.  The thresholds
    apply to the aggregate pass
    (:func:`call_error_stats_module`) and to
    :func:`~curryer.correction.verification.verify`.

    Parameters
    ----------
    setup : GeolocationSetup
        Supplies the spacecraft-state variable names.

    Returns
    -------
    ErrorStatsProcessor
        Processor with the setup's variable names and no correlation threshold.
    """
    return ErrorStatsProcessor(
        config=replace(ErrorStatsConfig.from_setup(setup), minimum_correlation=None, minimum_peak_margin=None)
    )


def loop(
    setup: GeolocationSetup,
    sweep: Sweep,
    work_dir: Path,
    tlm_sci_gcp_sets: list[tuple[str, str, str]],
    output: OutputConfig | None = None,
    resume_from_checkpoint: bool = False,
):
    """
    Correction loop for parameter sensitivity analysis.

    Parameters
    ----------
    setup : GeolocationSetup
        Durable mission setup: SPICE kernels/instrument and DEM (``geo``),
        pass/fail ``requirements``, ``data_config`` (telemetry file format),
        ``calibration`` files (LOS table and PSF), mission variable names, and an
        optional ``observation_matching_func`` override.
    sweep : Sweep
        The parameter-variation experiment: ``parameters``, ``search_strategy``,
        ``n_iterations``, ``seed``, and grid settings.
    work_dir : Path
        Working directory for the kernels and output; created if missing.
    tlm_sci_gcp_sets : list of (str, str, str)
        List of (`telemetry_key`, `science_key`, `gcp_key`) tuples: the
        telemetry the dynamic kernels are built from, an observation subimage
        cropped inside the GCP chip (:func:`load_loop_observation`), and the
        chip.  File paths are expected to be local. S3 URIs (``s3://…``) are also
        accepted as a convenience when ``boto3`` is installed; see
        :func:`~curryer.correction.io.resolve_path`.
    output : OutputConfig or None, optional
        Output settings (NetCDF metadata + filename).  ``None`` uses defaults
        derived from ``setup.requirements``.
    resume_from_checkpoint : bool, optional
        If True, resume from an existing checkpoint.

    Returns
    -------
    results : list
        List of iteration results (order: `pair_idx * N + param_idx`).
    netcdf_data : dict
        Dictionary of NetCDF variables indexed as `[param_idx, pair_idx]`.

    Raises
    ------
    ValueError
        If the calibration is incomplete (:func:`_require_image_matching_inputs`),
        an observation fails :func:`load_loop_observation`, a subimage pixel
        cannot be geolocated (:func:`_geolocate_observation`), a parameter set
        moves the subimage so the correlation search samples outside the chip,
        or no parameter set can be compared (:func:`_usable_pairs_and_valid_sets`).
    KeyError
        If an OFFSET_KERNEL field is not a telemetry column.

    Notes
    -----
    This implementation uses a pair-outer, parameter-inner loop order:
    - Outer loop: GCP pairs (load data once per image)
    - Inner loop: Parameter sets (reuse loaded data)

    For each parameter set the observation's frames and columns are
    re-geolocated (OFFSET_TIME shifts the frame times; kernel parameters change
    the pointing), terrain-corrected with ``setup.geo.dem_data_dir``, and the
    observation's radiance on that grid is matched to the chip
    (``setup.observation_matching_func``, default
    :func:`~curryer.correction.verification.match_observation`).

    Every pair's error is kept per parameter set (``rms_error_m``) with whether
    it passes ``setup.geo.minimum_correlation`` / ``minimum_peak_margin``
    (``accepted``).  Parameter sets are then compared on the same pairs
    (:func:`_usable_pairs_and_valid_sets`): a pair is used when some parameter
    set matches it, and a parameter set is ``valid`` when it matches every used
    pair.  Each valid set's errors on the used pairs give
    ``mean_rms_all_pairs`` (the selection metric), the percent under threshold
    and the aggregate statistics; an invalid set's are NaN and it is not
    selected.

    Examples
    --------
    ::

        results, netcdf_data = loop(setup, sweep, work_dir, tlm_sci_gcp_sets)

    Where each element of ``tlm_sci_gcp_sets`` is a tuple of file paths::

        tlm_sci_gcp_sets = [
            ("telemetry.csv", "granule__chip_001.nc", "chip_001.nc"),
        ]
    """
    output = output or OutputConfig()
    netcdf_config = _resolve_netcdf_config(setup, output)

    logger.info("=== CORRECTION PIPELINE ===")
    logger.info(f"  GCP pairs: {len(tlm_sci_gcp_sets)} (outer loop - load data once)")

    matching_func = setup.observation_matching_func or match_observation

    # Initialize parameter sets
    params_set = load_param_sets(sweep)
    logger.info(f"  Parameter sets: {len(params_set)} (inner loop)")

    # Build NetCDF data structure
    n_param_sets = len(params_set)
    n_gcp_pairs = len(tlm_sci_gcp_sets)

    # Kernel writers take a missing directory for a file name, so create it.
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Try to load checkpoint if resuming
    output_file = work_dir / output.get_output_filename()
    start_pair_idx = 0
    # Currently, checkpoint is bugged, since the nadir equivalent stats are not calculated until the end.
    # TODO [CURRYER-100]: Fix checkpoint resume for Monte Carlo GCS
    if resume_from_checkpoint:
        checkpoint_data, completed_pairs = _load_checkpoint(output_file)
        if checkpoint_data is not None and completed_pairs > 0:
            # TODO [CURRYER-100]: the results of the completed pairs are not in the checkpoint.
            raise NotImplementedError(
                f"{output_file} holds {completed_pairs} completed GCP pair(s), but resuming cannot restore their "
                "image-matching results for the aggregate statistics (CURRYER-100); remove it to start again."
            )
        if checkpoint_data is not None:
            netcdf_data = checkpoint_data
            start_pair_idx = completed_pairs
            logger.info(f"Resuming from checkpoint: starting at GCP pair {start_pair_idx + 1}/{n_gcp_pairs}")
        else:
            netcdf_data = _build_netcdf_structure(setup, sweep, netcdf_config, n_param_sets, n_gcp_pairs)
            logger.info("No valid checkpoint found, starting from beginning")
    else:
        netcdf_data = _build_netcdf_structure(setup, sweep, netcdf_config, n_param_sets, n_gcp_pairs)

    # Initialize results dict with (param_idx, pair_idx) keys
    # This avoids nested search complexity when aggregating statistics
    results_dict = {}

    # Prepare SPICE environment
    mkrn = meta.MetaKernel.from_json(
        setup.geo.meta_kernel_file,
        relative=True,
        sds_dir=setup.geo.generic_kernel_dir,
    )
    creator = create.KernelCreator(overwrite=True, append=False)

    # Load calibration data once (LOS vectors and optical PSF are static instrument calibration)
    calibration_data = _load_calibration_data(setup)
    _require_image_matching_inputs(setup, calibration_data)
    elev = elevation.Elevation(setup.geo.dem_data_dir, meters=False, degrees=False)

    # Create error stats processors once (setup is constant; processors are stateless)
    error_processor = _per_pair_error_processor(setup)
    gate_processor = ErrorStatsProcessor(config=ErrorStatsConfig.from_setup(setup))

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

        # Load the pair's inputs once
        tlm_dataset = _load_telemetry(tlm_key, setup)
        observation = load_loop_observation(sci_key, calibration_data.los_vectors)
        gcp = load_image_grid(gcp_key, mat_key="GCP")
        logger.info(f"  GCP file: {gcp_key}, observation {observation.radiance.shape}")

        # Create dynamic kernels once (these don't change with parameters)
        dynamic_kernels = _create_dynamic_kernels(setup, work_dir, tlm_dataset, creator)

        # INNER LOOP: Iterate through parameter sets
        for param_idx, params in enumerate(params_set):
            logger.info(f"  Parameter Set {param_idx + 1}/{n_param_sets}")

            # Create parameter-specific kernels (these change with parameters)
            param_kernels, frame_ugps = _create_parameter_kernels(
                params, work_dir, tlm_dataset, observation.frame_ugps, creator
            )

            with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, dynamic_kernels, param_kernels]):
                grid, r_spacecraft_m = _geolocate_observation(setup.geo.instrument_name, frame_ugps, observation, elev)

            image_matching_output = matching_func(
                grid, gcp, r_spacecraft_m, observation.los_vectors, calibration_data.optical_psfs, setup
            )
            validate_image_matching_output(image_matching_output)
            image_matching_output.attrs["gcp_pair_index"] = pair_idx
            image_matching_output.attrs["gcp_pair_id"] = f"{sci_key}_pair_{pair_idx}"
            netcdf_data["accepted"][param_idx, pair_idx] = not any(
                gate_processor.rejection_reasons(image_matching_output)
            )
            geo_dataset = xr.Dataset(
                {
                    "latitude": (["frame", "pixel"], grid.lat),
                    "longitude": (["frame", "pixel"], grid.lon),
                    "altitude": (["frame", "pixel"], grid.h),
                },
                coords={"ugps": ("frame", frame_ugps)},
            )

            # Compute nadir-equivalent errors for this GCP pair.
            # compute_nadir_equivalent_errors() skips aggregate statistics —
            # computing mean/std/percentiles on a single GCP pair is
            # mathematically uninformative and wastes time in a tight loop.
            individual_nadir = error_processor.compute_nadir_equivalent_errors(image_matching_output)

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
                "gcp_pairs": [(sci_key, gcp_key)],
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
            _save_netcdf_checkpoint(netcdf_data, output_file, setup, sweep, netcdf_config, pair_idx)

    usable_pairs, valid_sets = _usable_pairs_and_valid_sets(netcdf_data["accepted"])
    netcdf_data["valid"][:] = valid_sets
    logger.info(
        f"=== Comparing {int(valid_sets.sum())} of {n_param_sets} parameter sets on {len(usable_pairs)} of "
        f"{n_gcp_pairs} GCP pairs ==="
    )
    for param_idx in np.flatnonzero(valid_sets):
        param_image_matching_results = [
            results_dict[(param_idx, pair_idx)]["image_matching"] for pair_idx in usable_pairs
        ]
        aggregate_stats = call_error_stats_module(param_image_matching_results, setup)
        aggregate_error_metrics = _extract_error_metrics(aggregate_stats)

        pair_errors = netcdf_data["rms_error_m"][param_idx, usable_pairs]
        _compute_parameter_set_metrics(
            netcdf_data, param_idx, pair_errors, threshold_m=setup.requirements.performance_threshold_m
        )

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
    _save_netcdf_results(netcdf_data, output_file, setup, sweep, netcdf_config)

    # Clean up checkpoint file after successful completion
    if resume_from_checkpoint:
        _cleanup_checkpoint(output_file)

    logger.info(f"=== Loop Complete: Processed {n_gcp_pairs} GCP pairs × {n_param_sets} parameter sets ===")
    logger.info(f"  Total iterations: {len(results)}")
    logger.info(f"  NetCDF output: {output_file}")

    return results, netcdf_data


def _usable_pairs_and_valid_sets(accepted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the GCP pairs and parameter sets the correction compares.

    A pair is usable when it passes the match-quality gates under at least one
    parameter set: its scene can be matched.  A parameter set is valid when it
    passes the gates on every usable pair, so all valid sets are judged on the
    same pairs and none is rewarded for degrading a hard pair's match.  (A
    parameter set that moves the subimage so far that the search leaves the chip
    does not reach this point: :func:`loop` raises, because the crop margin must
    cover the parameter bounds.)

    Parameters
    ----------
    accepted : ndarray of bool, shape (n_param_sets, n_gcp_pairs)
        Whether each pair passes the gates under each parameter set.

    Returns
    -------
    usable_pairs : ndarray of int
        Indices of the usable pairs.
    valid_sets : ndarray of bool, shape (n_param_sets,)
        Which parameter sets are valid.

    Raises
    ------
    ValueError
        If no pair passes the gates under any parameter set, or no parameter
        set passes them on every usable pair.
    """
    usable = accepted.any(axis=0)
    if not usable.any():
        raise ValueError(
            "No GCP pair passes the match-quality gates (minimum_correlation / minimum_peak_margin) under any "
            "parameter set."
        )
    valid_sets = accepted[:, usable].all(axis=1)
    if not valid_sets.any():
        raise ValueError(
            "No parameter set passes the match-quality gates on every usable GCP pair; parameter sets accepting "
            f"each usable pair: {accepted[:, usable].sum(axis=0).tolist()} of {accepted.shape[0]}."
        )
    return np.flatnonzero(usable), valid_sets


def _extract_parameter_values(params):
    """Extract parameter values from a parameter set into a dictionary."""
    param_values = {}

    for param_config, param_data in params:
        # Parameters without a kernel file (e.g. OFFSET_TIME) still need a stable name
        # so their sampled values are stored in results/NetCDF output.
        if param_config.config_file:
            param_name = param_config.config_file.stem
        else:
            param_name = (
                param_config.spec.metadata.get("name") or param_config.spec.field or param_config.ptype.name.lower()
            )

        if param_config.ptype == ParameterType.CONSTANT_KERNEL:
            # Back from radians to the configured units, one variable per axis
            axis_name = _CONSTANT_KERNEL_AXES[param_config.spec.field]
            param_values[f"{param_name}_{axis_name}"] = _rad_to_val(param_data, param_config.spec.units)

        elif param_config.ptype == ParameterType.OFFSET_KERNEL:
            # Back from radians to the configured units
            param_values[param_name] = _rad_to_val(param_data, param_config.spec.units)

        elif param_config.ptype == ParameterType.OFFSET_TIME:
            # Back from seconds to the configured units
            param_values[param_name] = _seconds_to_val(param_data, param_config.spec.units)

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

        if netcdf_var not in netcdf_data:
            raise KeyError(
                f"Parameter variable '{netcdf_var}' not found in netcdf_data. Available keys: "
                f"{[k for k in netcdf_data.keys() if k.startswith('param_')]}"
            )
        netcdf_data[netcdf_var][param_idx] = value
        logger.debug(f"  Stored {netcdf_var}[{param_idx}] = {value}")


def _extract_error_metrics(stats_dataset):
    """Extract error metrics from error statistics dataset."""
    return {
        "rms_error_m": stats_dataset.attrs.get("rms_error_m", np.nan),
        "mean_error_m": stats_dataset.attrs.get("mean_error_m", np.nan),
        "max_error_m": stats_dataset.attrs.get("max_error_m", np.nan),
        "std_error_m": stats_dataset.attrs.get("std_error_m", np.nan),
        "n_measurements": stats_dataset.attrs.get("total_measurements", 0),
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
    setup: GeolocationSetup,
    sweep: Sweep,
    inputs: Sequence[CorrectionInput | tuple[str, str, str]],
    work_dir: Path,
    output: OutputConfig | None = None,
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
    setup : GeolocationSetup
        Durable mission setup (kernels, requirements, calibration, names).
    sweep : Sweep
        The parameter-variation experiment to run against *setup*.
    inputs : list of CorrectionInput or list of (str, str, str)
        Each element is either a :class:`~curryer.correction.config.CorrectionInput`
        (named fields) or a legacy ``(telemetry_key, science_key, gcp_key)`` tuple.
        Both forms may be mixed in the same list.
        File paths are expected to be local. S3 URIs (``s3://…``) are also
        accepted as a convenience when ``boto3`` is installed; see
        :func:`~curryer.correction.io.resolve_path`.
    work_dir : Path
        Working directory for temporary files.
    output : OutputConfig or None, optional
        Output settings (NetCDF metadata + filename).  ``None`` uses defaults
        derived from ``setup.requirements``.
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
    output = output or OutputConfig()
    netcdf_config = _resolve_netcdf_config(setup, output)

    normalized: list[tuple[str, str, str]] = []
    for inp in inputs:
        if isinstance(inp, CorrectionInput):
            normalized.append((str(inp.telemetry_file), str(inp.science_file), str(inp.gcp_file)))
        else:
            normalized.append(inp)

    results, netcdf_data = loop(setup, sweep, work_dir, normalized, output, resume_from_checkpoint)
    elapsed = time.time() - run_start
    netcdf_path = work_dir / output.get_output_filename()

    correction_result = build_correction_result(
        setup=setup,
        sweep=sweep,
        netcdf_config=netcdf_config,
        results=results,
        netcdf_data=netcdf_data,
        netcdf_path=netcdf_path,
        elapsed_time_s=elapsed,
    )

    logger.info("\n%s", correction_result.summary_table)
    logger.info(correction_result.recommendation)

    return correction_result


def compute_error_stats(image_matching_results, setup: "GeolocationSetup"):
    """Compute error statistics from image matching results.

    This is the preferred name for :func:`call_error_stats_module`.
    See :func:`call_error_stats_module` for full documentation.

    Parameters
    ----------
    image_matching_results : xr.Dataset or list of xr.Dataset
        Output from image matching, either a single dataset or a list.
    setup : GeolocationSetup
        Geolocation setup used to initialise the error stats processor.

    Returns
    -------
    xr.Dataset
        Aggregate error statistics dataset.
    """
    return call_error_stats_module(image_matching_results, setup)


def run_image_matching(
    geolocated_data: "xr.Dataset",
    gcp_reference_file: Path,
    telemetry: "pd.DataFrame",
    params_info: list,
    setup: "GeolocationSetup",
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
    params_info : list
        Parameter information for the current iteration.
    setup : GeolocationSetup
        Geolocation setup (calibration paths, variable names, instrument name).
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
        params_info,
        setup,
        los_vectors_cached,
        optical_psfs_cached,
    )
