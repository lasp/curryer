"""Verification module for geolocation requirements compliance.

Provides :func:`verify`, a standalone entry point that evaluates the current
set of SPICE kernels and alignment parameters against mission geolocation
requirements — without running the iterative correction loop.

Typical use-cases
-----------------
Weekly automated check (CLARREO)
    Pass pre-computed ``image_matching_results`` (the most common path):

    >>> result = verify(setup, work_dir, image_matching_results=weekly_datasets)
    >>> if not result.passed:
    ...     send_alert(result.summary_table)

Post-correction validation
    After a full GCS run, verify the optimised parameter set:

    >>> result = verify(setup, work_dir, image_matching_results=post_correction_datasets)

One-off compliance check with in-memory geolocated data
    Supply an already-geolocated dataset together with a GCP chip directory and
    calibration files; verification auto-pairs and image-matches without any
    additional setup:

    >>> result = verify(
    ...     setup,
    ...     geolocated_data=raw_dataset,
    ...     gcp_directory="data/gcps/",
    ...     los_file="cal/b_HS.mat",
    ...     psf_file="cal/optical_PSF_675nm.mat",
    ... )

Models
------
:class:`RequirementsConfig`
    Verification thresholds (performance limit and pass-rate).
:class:`GCPError`
    Per-measurement/GCP error detail.
:class:`VerificationResult`
    Structured pass/fail result; serialisable via Pydantic JSON methods.

"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from curryer import spicetime
from curryer import spicierpy as sp
from curryer.compute import constants
from curryer.correction.config import GeolocationSetup, PSFSamplingConfig, RequirementsConfig, SearchConfig
from curryer.correction.error_stats import ErrorStatsConfig, ErrorStatsProcessor
from curryer.correction.image_io import (
    geolocated_to_image_grid,
    load_image_grid,
    load_los_vectors,
    load_optical_psf,
)
from curryer.correction.image_match import integrated_image_match

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic models
# ============================================================================


class GCPError(BaseModel):
    """Per-measurement/GCP error detail.

    Each instance corresponds to one row in the aggregated image-matching
    output — typically one measurement from a single GCP pair.

    Attributes
    ----------
    gcp_index : int
        Zero-based measurement index in the aggregated dataset.
    science_key : str
        Identifier for the science data segment (dataset label or index).
    gcp_key : str
        Identifier for the ground-control-point source.
    lat_error_deg : float
        Latitude error in degrees (positive = northward shift).
    lon_error_deg : float
        Longitude error in degrees (positive = eastward shift).
    nadir_equiv_error_m : float or None
        Nadir-equivalent total geolocation error in metres, or ``None`` when
        error-stats processing was not performed.
    correlation : float or None
        Image-matching correlation score, or ``None`` when not available.
    passed : bool
        Whether this measurement satisfies the per-measurement threshold.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gcp_index: int
    science_key: str
    gcp_key: str
    lat_error_deg: float
    lon_error_deg: float
    nadir_equiv_error_m: float | None = None
    correlation: float | None = None
    passed: bool


class VerificationResult(BaseModel):
    """Structured result from a :func:`verify` call.

    Most fields are JSON-serialisable via Pydantic's ``model_dump()`` /
    ``model_dump_json()``.  The :attr:`aggregate_stats` field (an
    ``xr.Dataset``) must be excluded when serialising to JSON — persist it
    separately (e.g. via ``aggregate_stats.to_netcdf(path)``)::

        json_str = result.model_dump_json(exclude={"aggregate_stats"})
        result.aggregate_stats.to_netcdf("verification_stats.nc")

    Attributes
    ----------
    passed : bool
        ``True`` when :attr:`percent_within_threshold` ≥
        :attr:`requirements.performance_spec_percent`.
    per_gcp_errors : list[GCPError]
        One entry per measurement in the aggregated dataset.
    aggregate_stats : xr.Dataset
        Full output from
        :meth:`~curryer.correction.error_stats.ErrorStatsProcessor.process_geolocation_errors`.
    requirements : RequirementsConfig
        The thresholds used for this verification run.
    summary_table : str
        Human-readable ASCII table suitable for logging or reports.
    percent_within_threshold : float
        Percentage of measurements with nadir-equivalent error below
        :attr:`requirements.performance_threshold_m`.
    warnings : list[str]
        Non-empty when :attr:`passed` is ``False``.
    timestamp : datetime
        UTC wall-clock time when :func:`verify` was called.
    files_processed : list[str]
        Science/GCP key pairs that were processed, as ``"<sci_key>+<gcp_key>"``
        strings.  Empty when the source mapping is unavailable.
    elapsed_time_s : float or None
        Wall-clock time for the verify call in seconds, or ``None`` when not
        measured.
    config_snapshot : dict or None
        Key config fields used for this run (threshold, spec percent,
        instrument name), for reproducibility records.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    passed: bool
    per_gcp_errors: list[GCPError]
    aggregate_stats: xr.Dataset
    requirements: RequirementsConfig
    summary_table: str
    percent_within_threshold: float
    warnings: list[str]
    timestamp: datetime

    # Provenance fields — all optional so existing callers are unaffected.
    files_processed: list[str] = Field(default_factory=list)
    elapsed_time_s: float | None = None
    config_snapshot: dict | None = None


# ============================================================================
# Internal helpers
# ============================================================================


def _aggregate_results(
    image_matching_results: list[xr.Dataset],
    setup: GeolocationSetup,
) -> xr.Dataset:
    """Aggregate a list of per-GCP-pair image-matching datasets into one.

    For multiple input datasets, this delegates to the same aggregation logic
    used by the correction pipeline
    (:func:`~curryer.correction.pipeline._aggregate_image_matching_results`).
    For a single input dataset, it is returned directly after ensuring that
    the ``measurement`` coordinate is present and consists of sequential
    integer indices.

    Parameters
    ----------
    image_matching_results : list[xr.Dataset]
        One element per GCP pair.  Each dataset must have a ``measurement``
        dimension and at minimum ``lat_error_deg`` / ``lon_error_deg`` variables.
    setup : GeolocationSetup
        Used for mission-specific variable names
        (``spacecraft_position_name``, ``boresight_name``,
        ``transformation_matrix_name``).

    Returns
    -------
    xr.Dataset
        Combined dataset with a single ``measurement`` dimension.
    """
    if len(image_matching_results) == 1:
        ds = image_matching_results[0]
        # Always normalize the measurement coordinate to sequential integers
        # so that downstream gcp_index values are predictable.
        n = ds.sizes.get("measurement", len(ds["lat_error_deg"]))
        ds = ds.assign_coords(measurement=np.arange(n))
        return ds

    return _aggregate_image_matching_results(image_matching_results, setup)


def _run_error_stats(
    aggregated: xr.Dataset,
    setup: GeolocationSetup,
) -> xr.Dataset:
    """Run :class:`~curryer.correction.error_stats.ErrorStatsProcessor` on *aggregated*.

    Parameters
    ----------
    aggregated : xr.Dataset
        Combined image-matching result with a ``measurement`` dimension.
    setup : GeolocationSetup
        Used to build :class:`~curryer.correction.error_stats.ErrorStatsConfig`.

    Returns
    -------
    xr.Dataset
        Processed dataset with ``nadir_equiv_total_error_m`` and related
        intermediate variables.
    """
    error_config = ErrorStatsConfig.from_setup(setup)
    processor = ErrorStatsProcessor(config=error_config)
    return processor.process_geolocation_errors(aggregated)


def _check_threshold(
    aggregate_stats: xr.Dataset,
    requirements: RequirementsConfig,
) -> tuple[bool, float]:
    """Evaluate whether performance meets the threshold requirement.

    Uses ``nadir_equiv_total_error_m`` from the
    :class:`~curryer.correction.error_stats.ErrorStatsProcessor` output.

    Parameters
    ----------
    aggregate_stats : xr.Dataset
        Output of ``ErrorStatsProcessor.process_geolocation_errors()``.
    requirements : RequirementsConfig
        Performance limits to evaluate against.

    Returns
    -------
    tuple[bool, float]
        ``(passed, percent_within_threshold)``
    """
    nadir_errors = aggregate_stats["nadir_equiv_total_error_m"].values
    if len(nadir_errors) == 0:
        return False, 0.0
    count_below = int(np.sum(nadir_errors < requirements.performance_threshold_m))
    percent_below = float(count_below / len(nadir_errors) * 100.0)
    passed = percent_below >= requirements.performance_spec_percent
    return passed, percent_below


def _generate_warnings(
    passed: bool,
    percent_below: float,
    requirements: RequirementsConfig,
) -> list[str]:
    """Generate user-facing warning messages when verification fails.

    Parameters
    ----------
    passed : bool
        Overall pass/fail result.
    percent_below : float
        Percentage of measurements within the threshold.
    requirements : RequirementsConfig
        Performance limits used for the check.

    Returns
    -------
    list[str]
        Empty list when *passed* is ``True``; otherwise one warning string.
    """
    if not passed:
        return [
            f"⚠️  VERIFICATION FAILED: Only {percent_below:.1f}% of observations "
            f"meet the {requirements.performance_threshold_m}m nadir-equivalent error threshold "
            f"(required: {requirements.performance_spec_percent}%). "
            f"Recommend running the correction module to optimise calibration parameters."
        ]
    return []


def _build_per_gcp_errors(
    aggregate_stats: xr.Dataset,
    source_mapping: list[tuple[str, str]],
    requirements: RequirementsConfig,
) -> list[GCPError]:
    """Build a :class:`GCPError` for every measurement in *aggregate_stats*.

    Parameters
    ----------
    aggregate_stats : xr.Dataset
        Processed dataset from
        :func:`~curryer.correction.error_stats.ErrorStatsProcessor.process_geolocation_errors`.
    source_mapping : list[tuple[str, str]]
        ``[(science_key, gcp_key), ...]`` parallel to the measurement dimension.
        If shorter than the number of measurements the remainder fall back to
        ``("sci_{i}", "gcp_{i}")``.
    requirements : RequirementsConfig
        Used for per-measurement pass/fail evaluation.

    Returns
    -------
    list[GCPError]
    """
    n = aggregate_stats.sizes.get("measurement", 0)
    if n == 0:
        return []

    nadir_errors = aggregate_stats["nadir_equiv_total_error_m"].values
    lat_errors = aggregate_stats["lat_error_deg"].values
    lon_errors = aggregate_stats["lon_error_deg"].values

    # Optional correlation variable (several possible names)
    correlation_values: np.ndarray | None = None
    for corr_var in ("correlation", "ccv", "im_ccv"):
        if corr_var in aggregate_stats.data_vars:
            correlation_values = aggregate_stats[corr_var].values
            break

    errors: list[GCPError] = []
    for i in range(n):
        if i < len(source_mapping):
            sci_key, gcp_key = source_mapping[i]
        else:
            sci_key, gcp_key = f"sci_{i}", f"gcp_{i}"

        corr: float | None = None
        if correlation_values is not None:
            raw = float(correlation_values[i])
            if np.isfinite(raw):  # type: ignore[arg-type]
                corr = raw

        errors.append(
            GCPError(
                gcp_index=i,
                science_key=sci_key,
                gcp_key=gcp_key,
                lat_error_deg=float(lat_errors[i]),
                lon_error_deg=float(lon_errors[i]),
                nadir_equiv_error_m=float(nadir_errors[i]),
                correlation=corr,
                passed=float(nadir_errors[i]) < requirements.performance_threshold_m,
            )
        )
    return errors


def _build_source_mapping(
    image_matching_results: list[xr.Dataset],
) -> list[tuple[str, str]]:
    """Map every measurement back to its (science_key, gcp_key) pair.

    The mapping is derived from dataset attributes (``sci_key`` / ``gcp_key``)
    when present, otherwise falls back to ``"result_{i}"`` labels.

    Parameters
    ----------
    image_matching_results : list[xr.Dataset]
        Raw per-GCP-pair datasets before aggregation.

    Returns
    -------
    list[tuple[str, str]]
        Parallel to the ``measurement`` dimension of the aggregated dataset.
    """
    mapping: list[tuple[str, str]] = []
    for pair_idx, ds in enumerate(image_matching_results):
        # Prefer explicit / richer identifiers when available, with stable fallbacks.
        sci_key_attr = (
            ds.attrs.get("sci_key")
            or ds.attrs.get("science_key")
            or ds.attrs.get("science_file")
            or f"result_{pair_idx}"
        )
        gcp_key_attr = (
            ds.attrs.get("gcp_pair_id")
            or ds.attrs.get("gcp_file")
            or ds.attrs.get("gcp_key")
            or ds.attrs.get("gcp_pair_index")
            or f"gcp_{pair_idx}"
        )
        sci_key = str(sci_key_attr)
        gcp_key = str(gcp_key_attr)
        n_meas = ds.sizes.get("measurement", len(ds["lat_error_deg"]))
        for _ in range(n_meas):
            mapping.append((sci_key, gcp_key))
    return mapping


def _format_summary_table(
    per_gcp_errors: list[GCPError],
    requirements: RequirementsConfig,
    percent_within: float,
    passed: bool,
) -> str:
    """Generate a human-readable summary table.

    Example output::

        ┌──────────────────────────────────────────────────────┐
        │ Verification Summary                                 │
        ├──────┬────────────┬────────────┬──────────┬──────────┤
        │  GCP │ Lat err(°) │ Lon err(°) │ Nadir(m) │  Status  │
        ├──────┼────────────┼────────────┼──────────┼──────────┤
        │    0 │    0.00123 │  -0.00045  │   145.2  │    ✓    │
        │    1 │    0.00567 │   0.00234  │   312.8  │    ✗    │
        ├──────┴────────────┴────────────┴──────────┴──────────┤
        │ Result: PASSED — 60.0% within 250.0m (req: 39.0%)   │
        └──────────────────────────────────────────────────────┘

    Parameters
    ----------
    per_gcp_errors : list[GCPError]
        Per-measurement detail.
    requirements : RequirementsConfig
        Thresholds used for evaluation.
    percent_within : float
        Percentage of measurements within the threshold.
    passed : bool
        Overall pass/fail result.

    Returns
    -------
    str
        Multi-line formatted table.
    """
    # Column widths
    w_gcp = 6
    w_lat = 12
    w_lon = 12
    w_nadir = 10
    w_status = 8
    col_inner = w_gcp + w_lat + w_lon + w_nadir + w_status + 4  # 4 column separators

    title = " Verification Summary"
    verdict = "PASSED" if passed else "FAILED"
    footer_text = (
        f" Result: {verdict} — {percent_within:.1f}% within "
        f"{requirements.performance_threshold_m}m "
        f"(req: {requirements.performance_spec_percent}%)"
    )

    # inner_width must accommodate columns, title, AND footer
    inner_width = max(col_inner, len(title) + 2, len(footer_text))

    def _h_sep(left, mid, right, fill="─"):
        """Build a column-width separator, then pad to inner_width."""
        core = (
            left
            + fill * w_gcp
            + mid
            + fill * w_lat
            + mid
            + fill * w_lon
            + mid
            + fill * w_nadir
            + mid
            + fill * w_status
            + right
        )
        # Extend to full inner_width if footer/title made the table wider
        return core + fill * (inner_width - len(core))

    lines: list[str] = []

    lines.append("┌" + "─" * inner_width + "┐")
    lines.append("│" + title.ljust(inner_width) + "│")
    lines.append("├" + _h_sep("", "┬", "", "─") + "┤")
    # Header row
    h_gcp = " GCP".center(w_gcp)
    h_lat = "Lat err(°)".center(w_lat)
    h_lon = "Lon err(°)".center(w_lon)
    h_nadir = "Nadir(m)".center(w_nadir)
    h_status = "Status".center(w_status)
    lines.append(f"│{h_gcp}│{h_lat}│{h_lon}│{h_nadir}│{h_status}│")
    lines.append("├" + _h_sep("", "┼", "", "─") + "┤")

    for err in per_gcp_errors:
        c_gcp = str(err.gcp_index).rjust(w_gcp - 1).ljust(w_gcp)
        c_lat = f"{err.lat_error_deg:+.5f}".center(w_lat)
        c_lon = f"{err.lon_error_deg:+.5f}".center(w_lon)
        if err.nadir_equiv_error_m is not None:
            c_nadir = f"{err.nadir_equiv_error_m:.1f}".center(w_nadir)
        else:
            c_nadir = "N/A".center(w_nadir)
        c_status = ("  ✓  " if err.passed else "  ✗  ").center(w_status)
        lines.append(f"│{c_gcp}│{c_lat}│{c_lon}│{c_nadir}│{c_status}│")

    # Footer
    lines.append("├" + "─" * inner_width + "┤")
    lines.append("│" + footer_text.ljust(inner_width) + "│")
    lines.append("└" + "─" * inner_width + "┘")

    return "\n".join(lines)


def _log_pairing_summary(pairs: list[tuple[Path, Path]], unpaired: list[Path] | None = None) -> None:
    """Log a human-readable GCP pairing summary.

    Parameters
    ----------
    pairs : list of (Path, Path)
        Successfully paired (observation, gcp) paths.
    unpaired : list of Path or None, optional
        Observation paths for which no matching GCP was found.
    """
    lines = ["GCP Pairing Summary:"]
    for obs, gcp in pairs:
        lines.append(f"  ✓ {obs.name} → {gcp.name}")
    if unpaired:
        for obs in unpaired:
            lines.append(f"  ✗ {obs.name} → No matching GCP found")
    lines.append(f"Proceeding with {len(pairs)} observation(s).")
    logger.info("\n".join(lines))


# ============================================================================
# Image matching + aggregation (core of the verification pipeline)
# ============================================================================
# These functions were previously in pipeline.py, but belong here because
# verification owns the "GCP pairing → image matching → error stats" pipeline.
# pipeline.py now imports them from here, achieving the correct dependency
# direction: pipeline → verification → [pairing, image_io, image_match, error_stats]
# ============================================================================


def _get_spice_boresight_and_rotation(
    instrument_name: str,
    et_midframe: float,
    ref_frame: str = "ITRF93",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the instrument boresight and HS→CTRS rotation matrix from SPICE.

    Parameters
    ----------
    instrument_name : str
        SPICE instrument name (e.g. ``"CPRS_HYSICS"``).
    et_midframe : float
        Ephemeris time (ET seconds past J2000) at the mid-frame epoch.
    ref_frame : str, optional
        Target reference frame.  Default ``"ITRF93"`` (ECEF).

    Returns
    -------
    boresight : np.ndarray, shape (3,)
        Unit boresight vector in instrument (HS) frame.
    t_hs2ctrs : np.ndarray, shape (3, 3)
        Rotation matrix ``v_ctrs = R @ v_hs``.

    Raises
    ------
    SpiceyError
        If required kernels are not loaded or do not cover *et_midframe*.
    """
    boresight = sp.ext.instrument_boresight(instrument_name, norm=True)
    instr = sp.obj.Instrument(instrument_name)
    _, hs_frame_name, _, _, _ = sp.getfov(instr.id, 1, 80, 80)
    t_hs2ctrs = np.asarray(sp.pxform(hs_frame_name, ref_frame, et_midframe))
    return boresight, t_hs2ctrs


def _extract_spacecraft_position_midframe(
    telemetry: pd.DataFrame,
    setup: GeolocationSetup | None = None,
) -> np.ndarray:
    """Extract spacecraft position at mid-frame from telemetry.

    Parameters
    ----------
    telemetry : pd.DataFrame
        Telemetry DataFrame with spacecraft position columns.
    setup : GeolocationSetup or None, optional
        If provided and ``setup.data_config.position_columns`` is set, those
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

    if setup is not None and setup.data_config is not None and setup.data_config.position_columns is not None:
        cols = setup.data_config.position_columns
        if len(cols) != 3:
            raise ValueError(f"position_columns must have exactly 3 entries, got {len(cols)}: {cols}")
        missing = [c for c in cols if c not in telemetry.columns]
        if missing:
            raise ValueError(
                f"position_columns {missing} not found in telemetry. Available: {telemetry.columns.tolist()}"
            )
        position = telemetry[cols].iloc[mid_idx].values.astype(np.float64)
        logger.debug("Extracted spacecraft position from setup.data_config.position_columns %s: %s", cols, position)
        return position

    # Legacy fallback: pattern guessing
    logger.warning(
        "position_columns not configured — falling back to column name pattern-guessing. "
        "Set setup.data_config.position_columns = ['col_x', 'col_y', 'col_z'] to silence this warning."
    )

    for cols in [
        ["sc_pos_x", "sc_pos_y", "sc_pos_z"],
        ["position_x", "position_y", "position_z"],
        ["r_x", "r_y", "r_z"],
        ["pos_x", "pos_y", "pos_z"],
    ]:
        if all(c in telemetry.columns for c in cols):
            return telemetry[cols].iloc[mid_idx].values.astype(np.float64)

    pos_cols = [c for c in telemetry.columns if "pos" in c.lower() or c.startswith("r_")]
    if len(pos_cols) >= 3:
        logger.warning("Using first 3 position-like columns: %s", pos_cols[:3])
        return telemetry[pos_cols[:3]].iloc[mid_idx].values.astype(np.float64)

    raise ValueError(f"Cannot find position columns in telemetry. Available columns: {telemetry.columns.tolist()}")


def image_matching(
    geolocated_data: xr.Dataset,
    gcp_reference_file: Path,
    telemetry: pd.DataFrame | None = None,
    params_info: list | None = None,
    setup: GeolocationSetup | None = None,
    los_vectors_cached: np.ndarray | None = None,
    optical_psfs_cached: list | None = None,
    r_iss_midframe: np.ndarray | None = None,
) -> xr.Dataset:
    """Image matching using :func:`~curryer.correction.image_match.integrated_image_match`.

    Performs image correlation between geolocated pixels and a Landsat GCP
    reference image to measure geolocation error.

    This function is the single implementation used by both the correction loop
    (:func:`~curryer.correction.pipeline.loop`) and standalone verification
    (:func:`verify`).  ``pipeline.py`` imports it from here.

    Parameters
    ----------
    geolocated_data : xr.Dataset
        Geolocation output with ``latitude``, ``longitude``, and a ``frame``
        coordinate (GPS seconds = ``ugps_times / 1e6``).
    gcp_reference_file : Path
        Path to GCP reference image (``.mat`` or ``.nc``).
    telemetry : pd.DataFrame or None, optional
        Telemetry DataFrame with spacecraft state.  Required when
        *r_iss_midframe* is not supplied.
    params_info : list or None, optional
        Current parameter values for error tracking.  Defaults to ``[]``.
    setup : GeolocationSetup or None, optional
        Setup for coordinate names, calibration paths, and instrument
        metadata.
    los_vectors_cached : np.ndarray or None, optional
        Pre-loaded LOS vectors.
    optical_psfs_cached : list or None, optional
        Pre-loaded optical PSF entries.
    r_iss_midframe : np.ndarray of shape (3,) or None, optional
        Spacecraft ECEF position in metres at mid-frame.  When provided,
        *telemetry* is not consulted for position.

    Returns
    -------
    xr.Dataset
        Error measurements: ``lat_error_deg``, ``lon_error_deg``, and metadata.

    Raises
    ------
    ValueError
        If neither *telemetry* nor *r_iss_midframe* is supplied, or if
        calibration files are missing.
    """
    if params_info is None:
        params_info = []

    logger.info("Image Matching: correlation with %s", Path(gcp_reference_file).name)
    start_time = time.time()

    # Convert geolocation output to ImageGrid
    subimage = geolocated_to_image_grid(geolocated_data)
    logger.info("  Subimage shape: %s", subimage.data.shape)

    # Load GCP reference
    gcp = load_image_grid(gcp_reference_file, mat_key="GCP")
    gcp_center_lat = float(gcp.lat[gcp.lat.shape[0] // 2, gcp.lat.shape[1] // 2])
    gcp_center_lon = float(gcp.lon[gcp.lon.shape[0] // 2, gcp.lon.shape[1] // 2])
    logger.info("  GCP shape: %s, centre: (%.4f, %.4f)", gcp.data.shape, gcp_center_lat, gcp_center_lon)

    # Calibration data
    if los_vectors_cached is not None and optical_psfs_cached is not None:
        los_vectors = los_vectors_cached
        optical_psfs = optical_psfs_cached
        logger.info("  Using cached calibration data")
    else:
        calibration = setup.calibration if setup is not None else None
        if calibration is None or calibration.los_vectors_file is None:
            raise ValueError("No LOS vectors source configured. Set setup.calibration.los_vectors_file.")
        los_vectors = load_los_vectors(Path(calibration.los_vectors_file))

        if calibration.psf_file is None:
            raise ValueError("No PSF source configured. Set setup.calibration.psf_file.")
        optical_psfs = load_optical_psf(Path(calibration.psf_file))

    # Spacecraft position
    if r_iss_midframe is None:
        if telemetry is None:
            raise ValueError(
                "image_matching() requires either 'telemetry' (correction loop) or "
                "'r_iss_midframe' (standalone / verification use)."
            )
        r_iss_midframe = _extract_spacecraft_position_midframe(telemetry, setup=setup)
    logger.info("  Spacecraft position: %s", r_iss_midframe)

    # Run image matching
    result = integrated_image_match(
        subimage=subimage,
        gcp=gcp,
        r_iss_midframe_m=r_iss_midframe,
        los_vectors_hs=los_vectors,
        optical_psfs=optical_psfs,
        geolocation_config=PSFSamplingConfig(),
        search_config=SearchConfig(),
    )

    # Derive mid-frame epoch from geolocated_data["frame"] (GPS seconds = ugps/1e6)
    if "frame" in geolocated_data.coords:
        frame_vals = geolocated_data.coords["frame"].values
        ugps_midframe = int(float(frame_vals[len(frame_vals) // 2]) * 1e6)
    else:
        _time_field = getattr(setup.geo, "time_field", None) if setup and setup.geo else None
        if (
            _time_field
            and telemetry is not None
            and _time_field in (telemetry.columns if telemetry is not None else [])
        ):
            ugps_midframe = int(telemetry[_time_field].iloc[len(telemetry) // 2])
            logger.warning("geolocated_data has no 'frame' coord; using telemetry column '%s'.", _time_field)
        else:
            logger.warning("Cannot determine mid-frame uGPS; SPICE boresight query may fall back to nadir.")
            ugps_midframe = 0
    et_midframe = float(spicetime.adapt(ugps_midframe, from_="ugps", to="et"))

    instrument_name = setup.geo.instrument_name if setup and setup.geo else None
    try:
        if instrument_name is None:
            raise ValueError("No instrument_name in setup.geo")
        boresight, t_matrix = _get_spice_boresight_and_rotation(instrument_name, et_midframe)
        logger.info("  Boresight from SPICE IK (HS frame): %s", boresight)
    except Exception as exc:
        logger.warning("  SPICE boresight/rotation unavailable (%s); using nadir approximation.", exc)
        boresight = -r_iss_midframe / np.linalg.norm(r_iss_midframe)
        t_matrix = np.eye(3)

    # Convert errors km → degrees
    lat_error_deg = result.lat_error_km / 111.0
    lon_radius_km = constants.WGS84_SEMI_MAJOR_AXIS_KM * np.cos(np.deg2rad(gcp_center_lat))
    lon_error_deg = result.lon_error_km / (lon_radius_km * np.pi / 180.0)

    processing_time = time.time() - start_time
    logger.info(
        "  Image matching complete in %.2fs: lat=%.3f km, lon=%.3f km, ccv=%.4f",
        processing_time,
        result.lat_error_km,
        result.lon_error_km,
        result.ccv_final,
    )

    sc_pos_name = setup.spacecraft_position_name if setup else "sc_position"
    boresight_name = setup.boresight_name if setup else "boresight"
    transform_name = setup.transformation_matrix_name if setup else "t_inst2ref"

    output = xr.Dataset(
        {
            "lat_error_deg": (["measurement"], [lat_error_deg]),
            "lon_error_deg": (["measurement"], [lon_error_deg]),
            sc_pos_name: (["measurement", "xyz"], [r_iss_midframe]),
            boresight_name: (["measurement", "xyz"], [boresight]),
            transform_name: (["measurement", "xyz_from", "xyz_to"], t_matrix[np.newaxis, :, :]),
            "gcp_lat_deg": (["measurement"], [gcp_center_lat]),
            "gcp_lon_deg": (["measurement"], [gcp_center_lon]),
            "gcp_alt": (["measurement"], [0.0]),
        },
        coords={"measurement": [0], "xyz": ["x", "y", "z"], "xyz_from": ["x", "y", "z"], "xyz_to": ["x", "y", "z"]},
    )
    output.attrs.update(
        {
            "lat_error_km": result.lat_error_km,
            "lon_error_km": result.lon_error_km,
            "correlation_ccv": result.ccv_final,
            "final_grid_step_m": result.final_grid_step_m,
            "final_index_row": result.final_index_row,
            "final_index_col": result.final_index_col,
            "processing_time_s": processing_time,
            "gcp_file": str(Path(gcp_reference_file).name),
            "gcp_center_lat": gcp_center_lat,
            "gcp_center_lon": gcp_center_lon,
        }
    )
    return output


def _aggregate_image_matching_results(
    image_matching_results: list[xr.Dataset],
    setup: GeolocationSetup,
) -> xr.Dataset:
    """Aggregate multiple image matching results into one dataset.

    Parameters
    ----------
    image_matching_results : list[xr.Dataset]
        Per-GCP-pair datasets from :func:`image_matching`.
    setup : GeolocationSetup
        Used for variable name mappings.

    Returns
    -------
    xr.Dataset
        Combined dataset with a single ``measurement`` dimension.
    """
    logger.info("Aggregating %d image matching results", len(image_matching_results))

    sc_pos_name = setup.spacecraft_position_name
    boresight_name = setup.boresight_name
    transform_name = setup.transformation_matrix_name

    all_lat_errors: list[float] = []
    all_lon_errors: list[float] = []
    all_sc_positions: list[np.ndarray] = []
    all_boresights: list[np.ndarray] = []
    all_transforms: list[np.ndarray] = []
    all_gcp_lats: list[float] = []
    all_gcp_lons: list[float] = []
    all_gcp_alts: list[float] = []

    for result in image_matching_results:
        n = len(result["lat_error_deg"])
        all_lat_errors.extend(result["lat_error_deg"].values)
        all_lon_errors.extend(result["lon_error_deg"].values)
        if sc_pos_name in result:
            all_sc_positions.extend(result[sc_pos_name].values[j] for j in range(n))
        if boresight_name in result:
            all_boresights.extend(result[boresight_name].values[j] for j in range(n))
        if transform_name in result:
            all_transforms.extend(result[transform_name].values[j, :, :] for j in range(n))
        if "gcp_lat_deg" in result:
            all_gcp_lats.extend(result["gcp_lat_deg"].values)
        if "gcp_lon_deg" in result:
            all_gcp_lons.extend(result["gcp_lon_deg"].values)
        if "gcp_alt" in result:
            all_gcp_alts.extend(result["gcp_alt"].values)

    n_total = len(all_lat_errors)
    aggregated = xr.Dataset(
        {
            "lat_error_deg": (["measurement"], np.array(all_lat_errors)),
            "lon_error_deg": (["measurement"], np.array(all_lon_errors)),
        },
        coords={"measurement": np.arange(n_total)},
    )

    if all_sc_positions:
        aggregated[sc_pos_name] = (["measurement", "xyz"], np.array(all_sc_positions))
        aggregated = aggregated.assign_coords({"xyz": ["x", "y", "z"]})
    if all_boresights:
        aggregated[boresight_name] = (["measurement", "xyz"], np.array(all_boresights))
    if all_transforms:
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
    logger.info("  Aggregated: %d measurements from %d GCP pairs", n_total, len(image_matching_results))
    return aggregated


def match_geolocated_to_gcp_files(
    geolocated_data: xr.Dataset,
    gcp_files: list[Path],
    setup: GeolocationSetup,
    los_vectors_cached: np.ndarray | None = None,
    optical_psfs_cached: list | None = None,
) -> list[xr.Dataset]:
    """Run image matching between already-geolocated data and GCP reference files.

    This is the reusable *pipeline tail*: both the correction loop (after
    kernel tweaking and geolocation) and standalone :func:`verify` call this
    function.  Given geolocated data and a list of GCP reference files it
    performs image matching against each file and returns the per-GCP error
    datasets ready for aggregation and error-stats processing.

    Parameters
    ----------
    geolocated_data : xr.Dataset
        Geolocated observation dataset with ``latitude``, ``longitude``, and
        a ``frame`` coordinate (GPS seconds).
    gcp_files : list of Path
        GCP reference files to match against.
    setup : GeolocationSetup
        Mission setup (calibration paths, variable names, instrument name).
    los_vectors_cached : np.ndarray or None, optional
        Pre-loaded LOS vectors.
    optical_psfs_cached : list or None, optional
        Pre-loaded optical PSF entries.

    Returns
    -------
    list of xr.Dataset
        One error dataset per successfully matched GCP file.
        Failures are logged as warnings and skipped.
    """
    sc_pos_name = setup.spacecraft_position_name
    r_iss_midframe: np.ndarray | None = None
    if sc_pos_name and sc_pos_name in geolocated_data:
        arr = np.asarray(geolocated_data[sc_pos_name].values, dtype=float)
        if arr.ndim == 2:
            arr = arr[arr.shape[0] // 2]
        if arr.size == 3:
            r_iss_midframe = arr.ravel()

    matched: list[xr.Dataset] = []
    for gcp_file in gcp_files:
        try:
            result = image_matching(
                geolocated_data=geolocated_data,
                gcp_reference_file=Path(gcp_file),
                telemetry=None,
                params_info=[],
                setup=setup,
                los_vectors_cached=los_vectors_cached,
                optical_psfs_cached=optical_psfs_cached,
                r_iss_midframe=r_iss_midframe,
            )
            matched.append(result)
        except Exception as exc:
            logger.warning("Image match failed for GCP %s: %s", Path(gcp_file).name, exc)

    return matched


# ============================================================================
# Public API
# ============================================================================


def _run_image_matching_for_pairs(
    pairs: list[tuple[str | Path, str | Path]],
    los_file: str | Path,
    psf_file: str | Path,
    setup: GeolocationSetup,
    default_altitude_m: float = 400_000.0,
) -> list[xr.Dataset]:
    """Run image matching for a list of (observation, gcp) file-path pairs.

    Loads each observation and GCP file, infers spacecraft state, runs
    :func:`~curryer.correction.image_match.integrated_image_match`, and
    packages the result as an ``xr.Dataset`` compatible with
    :func:`verify`.

    Parameters
    ----------
    pairs : list of (Path, Path)
        ``(observation_path, gcp_path)`` tuples.
    los_file : Path
        Instrument line-of-sight vectors (``.mat`` file).
    psf_file : Path
        Optical PSF ``.mat`` file.
    setup : GeolocationSetup
        Used for spacecraft-state variable names.
    default_altitude_m : float, optional
        Fallback spacecraft altitude in metres when the observation file does
        not contain position data.  Default 400 000 m (ISS nominal orbit).

    Returns
    -------
    list[xr.Dataset]
        One dataset per successfully matched pair.  Failures are logged as
        warnings and skipped.
    """
    from curryer.compute.constants import WGS84_SEMI_MAJOR_AXIS_KM  # noqa: PLC0415

    from .config import PSFSamplingConfig, SearchConfig
    from .image_io import (
        load_image_grid,
        load_los_vectors,
        load_observation_file,
        load_optical_psf,
    )
    from .image_match import integrated_image_match
    from .psf import resolve_spacecraft_ecef

    sc_pos_name = setup.spacecraft_position_name
    boresight_name = setup.boresight_name
    t_matrix_name = setup.transformation_matrix_name

    los_vectors = load_los_vectors(los_file)
    optical_psfs = load_optical_psf(psf_file)

    datasets: list[xr.Dataset] = []
    for obs_path, gcp_path in pairs:
        try:
            obs_grid, r_sc_file = load_observation_file(obs_path)
            gcp_grid = load_image_grid(gcp_path, mat_key="GCP")

            mid_i, mid_j = gcp_grid.mid_indices
            gcp_lat = float(gcp_grid.lat[mid_i, mid_j])
            gcp_lon = float(gcp_grid.lon[mid_i, mid_j])

            r_iss_m, boresight, t_matrix = resolve_spacecraft_ecef(
                obs_grid, r_sc_file, default_altitude_m=default_altitude_m
            )

            result = integrated_image_match(
                subimage=obs_grid,
                gcp=gcp_grid,
                r_iss_midframe_m=r_iss_m,
                los_vectors_hs=los_vectors,
                optical_psfs=optical_psfs,
                geolocation_config=PSFSamplingConfig(),
                search_config=SearchConfig(),
            )

            # Convert km errors to degrees
            lat_error_deg = result.lat_error_km / 111.0
            lon_radius_km = WGS84_SEMI_MAJOR_AXIS_KM * np.cos(np.deg2rad(gcp_lat))
            lon_error_deg = result.lon_error_km / (lon_radius_km * np.pi / 180.0)

            ds = xr.Dataset(
                {
                    "lat_error_deg": (["measurement"], [lat_error_deg]),
                    "lon_error_deg": (["measurement"], [lon_error_deg]),
                    "gcp_lat_deg": (["measurement"], [gcp_lat]),
                    "gcp_lon_deg": (["measurement"], [gcp_lon]),
                    "gcp_alt": (["measurement"], [0.0]),
                    sc_pos_name: (["measurement", "xyz"], [r_iss_m]),
                    boresight_name: (["measurement", "xyz"], [boresight]),
                    t_matrix_name: (["measurement", "xyz_from", "xyz_to"], t_matrix[np.newaxis]),
                },
                coords={
                    "measurement": [0],
                    "xyz": ["x", "y", "z"],
                    "xyz_from": ["x", "y", "z"],
                    "xyz_to": ["x", "y", "z"],
                },
                attrs={
                    "lat_error_km": result.lat_error_km,
                    "lon_error_km": result.lon_error_km,
                    "correlation_ccv": result.ccv_final,
                    "obs_file": Path(obs_path).name,
                    "gcp_file": Path(gcp_path).name,
                    "sci_key": Path(obs_path).name,
                    "gcp_key": Path(gcp_path).name,
                },
            )
            datasets.append(ds)
            logger.info(
                "  Matched %s → %s: lat_err=%.3f km  lon_err=%.3f km  ccv=%.3f",
                Path(obs_path).name,
                Path(gcp_path).name,
                result.lat_error_km,
                result.lon_error_km,
                result.ccv_final,
            )
        except Exception as exc:
            logger.warning(
                "Image match failed for %s → %s: %s",
                Path(obs_path).name,
                Path(gcp_path).name,
                exc,
            )

    return datasets


def verify(
    setup: GeolocationSetup,
    # File-path-based input modes
    gcp_pairs: list[tuple[str | Path, str | Path]] | None = None,
    observation_paths: list[str | Path] | None = None,
    gcp_directory: str | Path | None = None,
    los_file: str | Path | None = None,
    psf_file: str | Path | None = None,
    max_distance_m: float = 0.0,
    gcp_pattern: str = "*_regridded.nc",
    default_altitude_m: float = 400_000.0,
    # Pre-computed input modes (backward-compatible)
    image_matching_results: list[xr.Dataset] | None = None,
    geolocated_data: xr.Dataset | None = None,
    work_dir: Path | None = None,
) -> VerificationResult:
    """Evaluate current alignment against mission requirements.

    No parameter variation, no kernel creation, no iteration loop.  This
    function checks whether a **given** set of alignment parameters meets
    geolocation requirements.

    Input priority (first match wins)
    ----------------------------------
    1. *image_matching_results* — pre-computed outputs from image matching;
       the most common entry point for weekly automated checks.
    2. *geolocated_data* — raw geolocated data; requires
       ``setup.image_matching_func`` to be set.
    3. *gcp_pairs* — explicit ``(observation_path, gcp_path)`` file-path pairs.
       Requires *los_file* and *psf_file*.
    4. *observation_paths* + *gcp_directory* — auto-paired via spatial overlap.
       Requires *los_file* and *psf_file*.
    5. None of the above provided — raises :class:`ValueError`.

    Parameters
    ----------
    setup : GeolocationSetup
        Mission setup with all geolocation/calibration settings.
    gcp_pairs : list of (path, path) or None
        Explicit ``(observation_path, gcp_path)`` pairs.  Each path may be a
        local path or an ``s3://`` URI (requires ``boto3``).
    observation_paths : list of path or None
        Observation file paths for automatic GCP pairing.
        Requires *gcp_directory*, *los_file*, and *psf_file*.
    gcp_directory : path or None
        Directory of GCP reference images for automatic pairing with
        *observation_paths*.
    los_file : path or None
        Instrument line-of-sight vectors (``.mat`` file).  Required when
        *gcp_pairs* or *observation_paths* is provided.
    psf_file : path or None
        Optical PSF ``.mat`` file.  Required when *gcp_pairs* or
        *observation_paths* is provided.
    max_distance_m : float, optional
        Spatial pairing margin for the auto-pair mode (default ``0.0`` —
        GCP centre must be inside the observation footprint).
    gcp_pattern : str, optional
        Glob pattern used to discover GCP chips when *gcp_directory* is
        provided.  Defaults to ``"*_regridded.nc"``.
    default_altitude_m : float, optional
        Fallback spacecraft altitude (metres) used when observation files do
        not contain position data.  Default ``400_000.0`` (ISS nominal orbit).
        Override for other platforms (e.g. ``505_000.0`` for CTIM).
    image_matching_results : list[xr.Dataset] or None
        Pre-computed image-matching datasets, one per GCP pair.
    geolocated_data : xr.Dataset or None
        Already-geolocated data; requires ``setup.image_matching_func``.
    work_dir : Path or None, optional
        Working directory for outputs.  Created if absent.

    Returns
    -------
    VerificationResult
        Structured pass/fail result with per-GCP detail and a
        human-readable :attr:`~VerificationResult.summary_table`.

    Raises
    ------
    ValueError
        When none of the input modes is provided; when *geolocated_data* is
        supplied without *gcp_directory* / *los_file* / *psf_file* and
        ``setup.image_matching_func`` is not set; when *los_file* or
        *psf_file* is ``None`` for a file-path mode (*gcp_pairs* or
        *observation_paths* + *gcp_directory*); when *observation_paths* and
        *gcp_directory* are not both supplied; or when image matching
        produces no results.
    FileNotFoundError
        If any of the supplied file paths do not exist.
    """
    # Handle optional work_dir with sensible default
    if work_dir is None:
        work_dir = Path("verification_output")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    verify_start = time.time()
    timestamp = datetime.now(tz=timezone.utc)
    requirements = setup.requirements

    logger.info(
        "Starting verification: threshold=%.1fm, spec=%.1f%%",
        requirements.performance_threshold_m,
        requirements.performance_spec_percent,
    )

    # ------------------------------------------------------------------
    # Step 1: Obtain image-matching results
    # ------------------------------------------------------------------
    source_mapping: list[tuple[str, str]] = []

    if image_matching_results is not None:
        if not image_matching_results:
            raise ValueError("image_matching_results must not be empty.")
        logger.info("Using %d pre-computed image-matching result(s)", len(image_matching_results))
        source_mapping = _build_source_mapping(image_matching_results)
        aggregated = _aggregate_results(image_matching_results, setup)

    elif geolocated_data is not None:
        # Primary path: the caller provides already-geolocated data.
        # GCP chips are spatial-paired using pairing.py's core algorithm,
        # then image matching runs via match_geolocated_to_gcp_files() in
        # this same module — no duplicate implementation.
        im_override = setup.image_matching_func
        if im_override is not None:
            # Backward-compat / test injection.
            logger.info("Running image matching on provided geolocated_data via override")
            matched = im_override(geolocated_data)
            if not isinstance(matched, list):
                matched = [matched]
        elif gcp_directory is not None and los_file is not None and psf_file is not None:
            from curryer.correction import pairing as _pairing  # noqa: PLC0415

            gcp_dir = Path(str(gcp_directory))
            gcp_files_all = sorted(gcp_dir.glob(gcp_pattern))
            if not gcp_files_all:
                raise ValueError(f"No GCP chips matching '{gcp_pattern}' found in '{gcp_dir}'.")

            # Spatial pairing: use the single canonical pairing algorithm in pairing.py.
            matched_gcp_files = _pairing.pair_geolocated_dataset_with_gcp_files(
                geolocated_data,
                gcp_files_all,
                max_distance_m=max_distance_m,
            )
            logger.info(
                "GCP pairing: %d chip(s) matched, %d outside footprint",
                len(matched_gcp_files),
                len(gcp_files_all) - len(matched_gcp_files),
            )
            if not matched_gcp_files:
                raise ValueError(
                    f"No GCP chips in '{gcp_dir}' (pattern: '{gcp_pattern}') overlap "
                    f"with the geolocated_data footprint."
                )

            # Pre-load calibration once.
            los_vectors = load_los_vectors(Path(str(los_file)))
            optical_psfs = load_optical_psf(Path(str(psf_file)))

            # Image matching — same code path as the correction loop.
            matched = match_geolocated_to_gcp_files(
                geolocated_data,
                matched_gcp_files,
                setup,
                los_vectors_cached=los_vectors,
                optical_psfs_cached=optical_psfs,
            )
        else:
            missing = [
                name
                for name, val in (
                    ("gcp_directory", gcp_directory),
                    ("los_file", los_file),
                    ("psf_file", psf_file),
                )
                if val is None
            ]
            raise ValueError(
                f"geolocated_data was provided but the following required arguments are missing: "
                f"{missing}. Supply gcp_directory, los_file, and psf_file to enable automatic "
                f"GCP pairing and image matching, or set setup.image_matching_func for a "
                f"custom matching function."
            )
        if not matched:
            raise ValueError(
                "Image matching produced no results for the provided geolocated_data. "
                "Check that GCP chips in gcp_directory spatially overlap the dataset footprint."
            )
        source_mapping = _build_source_mapping(matched)
        aggregated = _aggregate_results(matched, setup)

    elif gcp_pairs is not None:
        if not gcp_pairs:
            raise ValueError("gcp_pairs must not be empty.")
        if los_file is None or psf_file is None:
            raise ValueError(
                "los_file and psf_file are required when gcp_pairs is provided. "
                "Supply the instrument LOS-vector and PSF calibration .mat files."
            )

        pairs: list[tuple[str | Path, str | Path]] = []
        for pair in gcp_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each entry in gcp_pairs must be a 2-item (observation_path, gcp_path) pair.")
            obs_p, gcp_p = pair
            pairs.append((str(obs_p), str(gcp_p)))

        logger.info("Running image matching on %d explicit observation/GCP pair(s)", len(pairs))
        matched = _run_image_matching_for_pairs(
            pairs,
            str(los_file),
            str(psf_file),
            setup,
            default_altitude_m=default_altitude_m,
        )
        if not matched:
            raise ValueError("Image matching produced no results for the supplied gcp_pairs.")
        source_mapping = _build_source_mapping(matched)
        aggregated = _aggregate_results(matched, setup)

    elif observation_paths is not None or gcp_directory is not None:
        if observation_paths is None or gcp_directory is None:
            raise ValueError("observation_paths and gcp_directory must be provided together.")
        if not observation_paths:
            raise ValueError("observation_paths must not be empty.")
        if los_file is None or psf_file is None:
            raise ValueError(
                "los_file and psf_file are required when observation_paths / gcp_directory is provided. "
                "Supply the instrument LOS-vector and PSF calibration .mat files."
            )

        gcp_dir = Path(str(gcp_directory))
        obs_path_list = [Path(str(p)) for p in observation_paths]

        logger.info(
            "Auto-pairing %d observation(s) with GCP chips from '%s' (pattern: %s)",
            len(obs_path_list),
            gcp_dir,
            gcp_pattern,
        )
        # Use the single canonical pairing algorithm from pairing.py
        from curryer.correction import pairing as _pairing  # noqa: PLC0415

        raw_pairs = _pairing.pair_files(
            obs_path_list,
            gcp_dir,
            max_distance_m=max_distance_m,
            gcp_pattern=gcp_pattern,
        )
        # Derive unpaired for logging
        paired_obs = {p for p, _ in raw_pairs}
        unpaired = [p for p in obs_path_list if p not in paired_obs]
        _log_pairing_summary(raw_pairs, unpaired or None)
        if not raw_pairs:
            raise ValueError(
                f"No observations could be paired with GCP chips in '{gcp_dir}' (pattern: '{gcp_pattern}')."
            )

        matched = _run_image_matching_for_pairs(
            raw_pairs,
            str(los_file),
            str(psf_file),
            setup,
            default_altitude_m=default_altitude_m,
        )
        if not matched:
            raise ValueError("Image matching produced no results for the observation/GCP pairs.")
        source_mapping = _build_source_mapping(matched)
        aggregated = _aggregate_results(matched, setup)

    else:
        raise ValueError(
            "Neither image_matching_results nor geolocated_data was provided. "
            "Supply one of: image_matching_results, geolocated_data, "
            "gcp_pairs, or observation_paths + gcp_directory."
        )

    # ------------------------------------------------------------------
    # Step 2: Compute nadir-equivalent error statistics
    # ------------------------------------------------------------------
    logger.info("Computing nadir-equivalent error statistics")
    aggregate_stats = _run_error_stats(aggregated, setup)

    # ------------------------------------------------------------------
    # Step 3: Threshold check
    # ------------------------------------------------------------------
    passed, percent_within = _check_threshold(aggregate_stats, requirements)

    # ------------------------------------------------------------------
    # Step 4: Per-GCP detail
    # ------------------------------------------------------------------
    per_gcp_errors = _build_per_gcp_errors(aggregate_stats, source_mapping, requirements)

    # ------------------------------------------------------------------
    # Step 5: Warnings + summary table
    # ------------------------------------------------------------------
    warnings = _generate_warnings(passed, percent_within, requirements)
    summary_table = _format_summary_table(per_gcp_errors, requirements, percent_within, passed)

    if warnings:
        for w in warnings:
            logger.warning(w)

    logger.info(
        "Verification %s — %.1f%% within %.1fm threshold (requirement: %.1f%%)",
        "PASSED" if passed else "FAILED",
        percent_within,
        requirements.performance_threshold_m,
        requirements.performance_spec_percent,
    )
    logger.info("\n%s", summary_table)

    # Build provenance fields
    files_processed = [f"{sci}+{gcp}" for sci, gcp in source_mapping]
    config_snapshot = {
        "performance_threshold_m": requirements.performance_threshold_m,
        "performance_spec_percent": requirements.performance_spec_percent,
        "instrument_name": getattr(setup.geo, "instrument_name", None),
    }
    elapsed_time_s = time.time() - verify_start

    return VerificationResult(
        passed=passed,
        per_gcp_errors=per_gcp_errors,
        aggregate_stats=aggregate_stats,
        requirements=requirements,
        summary_table=summary_table,
        percent_within_threshold=percent_within,
        warnings=warnings,
        timestamp=timestamp,
        files_processed=files_processed,
        elapsed_time_s=elapsed_time_s,
        config_snapshot=config_snapshot,
    )


def compare_results(before: VerificationResult, after: VerificationResult) -> str:
    """Generate a side-by-side comparison of two verification results.

    Useful for evaluating whether a correction run improved geolocation
    accuracy relative to a baseline.

    Parameters
    ----------
    before : VerificationResult
        Baseline verification result (e.g., pre-correction).
    after : VerificationResult
        Updated verification result (e.g., post-correction).

    Returns
    -------
    str
        Human-readable side-by-side comparison table.
    """
    lines = [
        "Verification Comparison",
        "=" * 55,
        f"{'Metric':<30} {'Before':>12} {'After':>12}",
        "-" * 55,
    ]

    b_stats = dict(before.aggregate_stats.attrs) if before.aggregate_stats is not None else {}
    a_stats = dict(after.aggregate_stats.attrs) if after.aggregate_stats is not None else {}

    stat_keys = [
        "mean_error_m",
        "median_error_m",
        "rms_error_m",
        "max_error_m",
        "percent_below_250m",
        "percent_below_500m",
    ]
    for key in stat_keys:
        b_val = b_stats.get(key)
        a_val = a_stats.get(key)
        b_str = f"{b_val:.1f}" if isinstance(b_val, (int, float)) else "N/A"
        a_str = f"{a_val:.1f}" if isinstance(a_val, (int, float)) else "N/A"
        lines.append(f"{key:<30} {b_str:>12} {a_str:>12}")

    lines.append("-" * 55)
    lines.append(
        f"{'percent_within_threshold':<30} "
        f"{before.percent_within_threshold:>11.1f}% "
        f"{after.percent_within_threshold:>11.1f}%"
    )
    lines.append("-" * 55)
    b_verdict = "PASS" if before.passed else "FAIL"
    a_verdict = "PASS" if after.passed else "FAIL"
    lines.append(f"{'Overall':<30} {b_verdict:>12} {a_verdict:>12}")

    return "\n".join(lines)
