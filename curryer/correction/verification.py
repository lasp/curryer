"""Verification module for geolocation requirements compliance.

Provides :func:`verify`, a standalone entry point that evaluates the current
set of SPICE kernels and alignment parameters against mission geolocation
requirements — without running the iterative correction loop.

Typical use-cases
-----------------
Weekly automated check (CLARREO)
    Pass pre-computed ``image_matching_results`` (the most common path):

    >>> result = verify(setup, image_matching_results=weekly_datasets, work_dir=work_dir)
    >>> if not result.passed:
    ...     send_alert(result.summary_table)

Post-correction validation
    After a full GCS run, verify the optimised parameter set:

    >>> result = verify(setup, image_matching_results=post_correction_datasets, work_dir=work_dir)

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

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from curryer import spicetime
from curryer import spicierpy as sp
from curryer.compute import constants
from curryer.correction.config import GeolocationSetup, RequirementsConfig
from curryer.correction.error_stats import (
    ErrorStatsConfig,
    ErrorStatsProcessor,
    quality_weights,
    weighted_percent_below,
    weighted_statistics,
)
from curryer.correction.image_io import (
    geolocated_to_image_grid,
    load_image_grid,
    load_los_vectors,
    load_optical_psf,
)
from curryer.correction.image_match import integrated_image_match
from curryer.correction.psf import ground_track_azimuth_deg, validate_spacecraft_ecef_m

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
        Nadir-equivalent total geolocation error in meters, or ``None`` when
        error-stats processing was not performed.  Computed for rejected
        measurements too, for review; they do not enter the statistics.
    correlation : float or None
        Image-matching correlation score, or ``None`` when not available.
    passed : bool
        ``True`` when :attr:`status` is ``"pass"``; a contradiction raises.
    status : {"pass", "fail", "rejected"}
        ``"rejected"`` when the match failed a match-quality gate
        (``minimum_correlation`` / ``minimum_peak_margin``); otherwise
        ``"pass"`` or ``"fail"`` against the per-measurement threshold.
    rejection_reason : str or None
        The gates failed, with values and thresholds; given exactly when
        :attr:`status` is ``"rejected"``.
    correlation_secondary : float or None
        Strongest competing correlation away from the peak, or ``None`` when
        not available or when no competing point exists.
    off_nadir_angle_deg : float or None
        Off-nadir viewing angle of the measurement, degrees.
    along_track_error_m, cross_track_error_m : float or None
        Ground error along the ground-track direction and 90° clockwise from
        it, meters, or ``None`` when the matching path did not record the
        ground-track azimuth.
    review : {"accept", "reject"} or None
        A reviewer's decision applied with :func:`apply_review`, which sets
        :attr:`status` accordingly; ``None`` when not reviewed.
    quality_weight : float or None
        Match signal-to-noise weight
        (:func:`~curryer.correction.error_stats.match_snr_weight`); 0 when
        rejected, ``None`` without a correlation.
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
    status: Literal["pass", "fail", "rejected"]
    rejection_reason: str | None = None
    correlation_secondary: float | None = None
    off_nadir_angle_deg: float | None = None
    along_track_error_m: float | None = None
    cross_track_error_m: float | None = None
    review: Literal["accept", "reject"] | None = None
    quality_weight: float | None = None

    @model_validator(mode="after")
    def _check_status(self) -> GCPError:
        if self.passed != (self.status == "pass"):
            raise ValueError(f"passed={self.passed} contradicts status={self.status!r}.")
        if (self.status == "rejected") != (self.rejection_reason is not None):
            raise ValueError("rejection_reason must be given exactly when status is 'rejected'.")
        if self.review is not None and (self.review == "reject") != (self.status == "rejected"):
            raise ValueError(f"review={self.review!r} contradicts status={self.status!r}.")
        return self


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
        Percentage of accepted measurements with nadir-equivalent error below
        :attr:`requirements.performance_threshold_m`; decides :attr:`passed`.
    weighted_percent_within_threshold : float or None
        The same percentage with each accepted measurement weighted by its
        ``quality_weight``, reported alongside, never deciding :attr:`passed`;
        ``None`` without correlations, or when the accepted weights sum to 0
        (e.g. every measurement rejected).
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
    chip_images : list[xr.Dataset] or None
        With ``verify(..., keep_images=True)``, the images behind each match
        (:func:`~curryer.correction.image_match.chip_image_dataset`), one per
        entry of :attr:`per_gcp_errors` and in the same order, each with
        ``gcp_index``, ``science_key`` and ``gcp_key`` attributes; ``None``
        otherwise.  Never included in ``model_dump`` / ``model_dump_json``.
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
    chip_images: list[xr.Dataset] | None = Field(default=None, exclude=True)
    weighted_percent_within_threshold: float | None = None


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
        Every measurement with ``nadir_equiv_total_error_m``, ``accepted``,
        ``rejection_reason`` and related variables; statistics over the
        accepted measurements as attributes, or none when every measurement
        was rejected.
    """
    error_config = ErrorStatsConfig.from_setup(setup)
    processor = ErrorStatsProcessor(config=error_config)
    return _with_statistics(processor, processor.compute_nadir_equivalent_errors(aggregated))


def _with_statistics(processor: ErrorStatsProcessor, per_measurement: xr.Dataset) -> xr.Dataset:
    """Add statistics over the accepted measurements, or none when every one is rejected."""
    if not per_measurement["accepted"].values.any():
        logger.warning(
            "All %d matched measurements were rejected by the match-quality gates.",
            per_measurement.sizes["measurement"],
        )
        return per_measurement
    return processor.add_statistics(per_measurement)


def _check_threshold(
    aggregate_stats: xr.Dataset,
    requirements: RequirementsConfig,
) -> tuple[bool, float]:
    """Evaluate whether performance meets the threshold requirement.

    Uses ``nadir_equiv_total_error_m`` of the accepted measurements in the
    :class:`~curryer.correction.error_stats.ErrorStatsProcessor` output; with
    none accepted the check fails at 0 %.

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
    nadir_errors = aggregate_stats["nadir_equiv_total_error_m"].values[aggregate_stats["accepted"].values]
    if len(nadir_errors) == 0:
        return False, 0.0
    count_below = int(np.sum(nadir_errors < requirements.performance_threshold_m))
    percent_below = float(count_below / len(nadir_errors) * 100.0)
    passed = percent_below >= requirements.performance_spec_percent
    return passed, percent_below


def _weighted_percent_within(aggregate_stats: xr.Dataset, requirements: RequirementsConfig) -> float | None:
    """Quality-weighted percentage of accepted measurements below the threshold, or ``None``."""
    accepted = aggregate_stats["accepted"].values
    if (
        "quality_weight" not in aggregate_stats.data_vars
        or not aggregate_stats["quality_weight"].values[accepted].sum()
    ):
        return None
    return weighted_percent_below(
        aggregate_stats["nadir_equiv_total_error_m"].values[accepted],
        aggregate_stats["quality_weight"].values[accepted],
        requirements.performance_threshold_m,
    )


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
        Used for per-measurement pass/fail evaluation of accepted measurements.

    Returns
    -------
    list[GCPError]
        Accepted and rejected measurements alike, in measurement order.
    """
    n = aggregate_stats.sizes.get("measurement", 0)
    if n == 0:
        return []

    nadir_errors = aggregate_stats["nadir_equiv_total_error_m"].values
    lat_errors = aggregate_stats["lat_error_deg"].values
    lon_errors = aggregate_stats["lon_error_deg"].values
    accepted = aggregate_stats["accepted"].values
    reasons = aggregate_stats["rejection_reason"].values
    reviews = aggregate_stats["review"].values if "review" in aggregate_stats.data_vars else np.full(n, "")

    def _finite_or_none(name: str, i: int) -> float | None:
        if name not in aggregate_stats.data_vars:
            return None
        value = float(aggregate_stats[name].values[i])
        return value if np.isfinite(value) else None

    corr_name = next((name for name in ("correlation", "ccv", "im_ccv") if name in aggregate_stats.data_vars), "")
    labels = aggregate_stats["measurement"].values

    errors: list[GCPError] = []
    for i in range(n):
        label = int(labels[i])
        if label < len(source_mapping):
            sci_key, gcp_key = source_mapping[label]
        else:
            sci_key, gcp_key = f"sci_{label}", f"gcp_{label}"

        if not accepted[i]:
            status = "rejected"
        elif float(nadir_errors[i]) < requirements.performance_threshold_m:
            status = "pass"
        else:
            status = "fail"

        errors.append(
            GCPError(
                gcp_index=label,
                science_key=sci_key,
                gcp_key=gcp_key,
                lat_error_deg=float(lat_errors[i]),
                lon_error_deg=float(lon_errors[i]),
                nadir_equiv_error_m=float(nadir_errors[i]),
                correlation=_finite_or_none(corr_name, i),
                passed=status == "pass",
                status=status,
                rejection_reason=str(reasons[i]) if status == "rejected" else None,
                correlation_secondary=_finite_or_none("correlation_secondary", i),
                off_nadir_angle_deg=_finite_or_none("off_nadir_angle_deg", i),
                along_track_error_m=_finite_or_none("along_track_error_m", i),
                cross_track_error_m=_finite_or_none("cross_track_error_m", i),
                review=str(reviews[i]) or None,
                quality_weight=_finite_or_none("quality_weight", i),
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
    weighted_percent: float | None = None,
) -> str:
    """Generate a human-readable summary table.

    Example output::

        ┌──────────────────────────────────────────────────────┐
        │ Verification Summary                                 │
        ├──────┬────────────┬────────────┬──────────┬──────────┤
        │  GCP │ Lat err(°) │ Lon err(°) │ Nadir(m) │  Status  │
        ├──────┼────────────┼────────────┼──────────┼──────────┤
        │    0 │    0.00123 │  -0.00045  │   145.2  │   PASS   │
        │    1 │    0.00567 │   0.00234  │   312.8  │   FAIL   │
        │    2 │    0.01020 │   0.00810  │  1240.6  │ REJECTED │
        ├──────┴────────────┴────────────┴──────────┴──────────┤
        │ Result: FAILED — 50.0% within 250.0m (req: 60.0%)    │
        │ 2 accepted, 1 rejected                               │
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
    weighted_percent : float or None, optional
        Quality-weighted percentage within the threshold, shown on its own
        line when given.

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
    w_status = 10
    col_inner = w_gcp + w_lat + w_lon + w_nadir + w_status + 4  # 4 column separators

    title = " Verification Summary"
    verdict = "PASSED" if passed else "FAILED"
    footer_text = (
        f" Result: {verdict} — {percent_within:.1f}% within "
        f"{requirements.performance_threshold_m}m "
        f"(req: {requirements.performance_spec_percent}%)"
    )

    n_rejected = sum(err.status == "rejected" for err in per_gcp_errors)
    counts_text = f" {len(per_gcp_errors) - n_rejected} accepted, {n_rejected} rejected"
    if weighted_percent is not None:
        weights = [err.quality_weight for err in per_gcp_errors if err.status != "rejected"]
        effective = sum(weights) ** 2 / sum(w**2 for w in weights)
        counts_text += f"; weighted {weighted_percent:.1f}% within (effective n = {effective:.1f})"

    # inner_width must accommodate columns, title, AND footer
    inner_width = max(col_inner, len(title) + 2, len(footer_text), len(counts_text))

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
        c_status = err.status.upper().center(w_status)
        lines.append(f"│{c_gcp}│{c_lat}│{c_lon}│{c_nadir}│{c_status}│")

    # Footer
    lines.append("├" + "─" * inner_width + "┤")
    lines.append("│" + footer_text.ljust(inner_width) + "│")
    lines.append("│" + counts_text.ljust(inner_width) + "│")
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
        Shape ``(3,)`` — ``[x, y, z]`` position in meters (J2000 frame).

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
        Spacecraft ECEF position in meters at mid-frame.  When provided,
        *telemetry* is not consulted for position.

    Returns
    -------
    xr.Dataset
        Error measurements: ``lat_error_deg``, ``lon_error_deg``, ``correlation``
        (final normalized cross-correlation coefficient, dimensionless, at most
        1), ``correlation_secondary``, ``track_azimuth_deg`` (ground-track
        azimuth at the GCP centre, :func:`~curryer.correction.psf.ground_track_azimuth_deg`),
        geometry, and metadata.

    Raises
    ------
    ValueError
        If neither *telemetry* nor *r_iss_midframe* is supplied; if the
        spacecraft position fails
        :func:`~curryer.correction.psf.validate_spacecraft_ecef_m`; if
        calibration files are missing; if the mid-frame time cannot be
        determined (no ``frame`` coordinate and no ``setup.geo.time_field``
        telemetry column); or if ``setup.geo.instrument_name`` is not set.
        All of these are checked before the image match runs.  After it, if
        the GCP centre lies outside the geolocated grid (no ground-track
        azimuth).
    spiceypy.utils.exceptions.SpiceyError
        If the SPICE boresight or HS→CTRS rotation query fails (e.g. kernels
        not furnished or no coverage at the mid-frame time).

    Notes
    -----
    The boresight and rotation recorded for error statistics are the SPICE
    instrument boresight at the dataset's mid-frame, and the spacecraft
    position is likewise a single mid-frame value.  Every GCP matched against
    the same dataset therefore receives the mid-frame off-nadir angle, wherever
    in the swath it was imaged.  For strongly off-nadir data this differs from
    the per-GCP line of sight used by the file-pair path
    (:func:`~curryer.correction.psf.resolve_spacecraft_ecef`).
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
    logger.info("  GCP shape: %s, center: (%.4f, %.4f)", gcp.data.shape, gcp_center_lat, gcp_center_lon)

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
    r_iss_midframe = validate_spacecraft_ecef_m(r_iss_midframe)
    logger.info("  Spacecraft position: %s", r_iss_midframe)

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
            raise ValueError(
                "Cannot determine the mid-frame time for the SPICE boresight query: geolocated_data has no "
                "'frame' coordinate and no telemetry column named by setup.geo.time_field was supplied."
            )
    et_midframe = float(spicetime.adapt(ugps_midframe, from_="ugps", to="et"))

    instrument_name = setup.geo.instrument_name if setup and setup.geo else None
    if instrument_name is None:
        raise ValueError("setup.geo.instrument_name is required for the SPICE boresight query.")
    boresight, t_matrix = _get_spice_boresight_and_rotation(instrument_name, et_midframe)
    logger.info("  Boresight from SPICE IK (HS frame): %s", boresight)

    # Run image matching
    result = integrated_image_match(
        subimage=subimage,
        gcp=gcp,
        r_iss_midframe_m=r_iss_midframe,
        los_vectors_hs=los_vectors,
        optical_psfs=optical_psfs,
        geolocation_config=setup.psf_sampling,
        search_config=setup.search,
    )

    # Convert errors km → degrees on the WGS-84 equatorial radius, the same radius
    # ErrorStatsProcessor uses to convert them back to meters.
    lat_error_deg = np.rad2deg(result.lat_error_km / constants.WGS84_SEMI_MAJOR_AXIS_KM)
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
            "correlation": (["measurement"], [result.ccv_final]),
            "correlation_secondary": (["measurement"], [result.ccv_secondary]),
            "track_azimuth_deg": (
                ["measurement"],
                [ground_track_azimuth_deg(subimage, gcp_center_lat, gcp_center_lon)],
            ),
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
        Combined dataset with a single ``measurement`` dimension. Per-result
        correlation scores, named ``correlation``, ``ccv`` or ``im_ccv`` (first
        present, in that order), are combined into ``correlation``, and
        ``correlation_secondary`` and ``track_azimuth_deg`` are carried through.

    Raises
    ------
    ValueError
        If a correlation variable, ``correlation_secondary`` or
        ``track_azimuth_deg`` is present in some results but not all.
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
    all_correlations: list[float] = []
    all_secondary: list[float] = []
    all_azimuths: list[float] = []

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
        corr_name = next((name for name in ("correlation", "ccv", "im_ccv") if name in result), None)
        if corr_name is not None:
            all_correlations.extend(result[corr_name].values)
        if "correlation_secondary" in result:
            all_secondary.extend(result["correlation_secondary"].values)
        if "track_azimuth_deg" in result:
            all_azimuths.extend(result["track_azimuth_deg"].values)

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
    if all_correlations:
        if len(all_correlations) != n_total:
            raise ValueError(
                f"A correlation variable ('correlation', 'ccv' or 'im_ccv') is present in only some "
                f"image-matching results "
                f"({len(all_correlations)} of {n_total} measurements); it must be in all or none."
            )
        aggregated["correlation"] = (["measurement"], np.array(all_correlations))
    if all_secondary:
        if len(all_secondary) != n_total:
            raise ValueError(
                f"'correlation_secondary' is present in only some image-matching results "
                f"({len(all_secondary)} of {n_total} measurements); it must be in all or none."
            )
        aggregated["correlation_secondary"] = (["measurement"], np.array(all_secondary))
    if all_azimuths:
        if len(all_azimuths) != n_total:
            raise ValueError(
                f"'track_azimuth_deg' is present in only some image-matching results "
                f"({len(all_azimuths)} of {n_total} measurements); it must be in all or none."
            )
        aggregated["track_azimuth_deg"] = (["measurement"], np.array(all_azimuths))

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
        One error dataset per GCP file, in *gcp_files* order.

    Raises
    ------
    ValueError, spiceypy.utils.exceptions.SpiceyError
        Propagated from :func:`image_matching` for the first GCP file that
        fails; no file is skipped.
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

    return matched


# ============================================================================
# Public API
# ============================================================================


def _run_image_matching_for_pairs(
    pairs: list[tuple[str | Path, str | Path]],
    los_file: str | Path,
    psf_file: str | Path,
    setup: GeolocationSetup,
    keep_images: bool = False,
) -> tuple[list[xr.Dataset], list[xr.Dataset]]:
    """Run image matching for a list of (observation, gcp) file-path pairs.

    Loads each observation and GCP file, resolves the viewing geometry from
    the observation's spacecraft position, runs
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
    keep_images : bool, optional
        Also return the images behind each match
        (:func:`~curryer.correction.image_match.chip_image_dataset`).

    Returns
    -------
    datasets : list[xr.Dataset]
        One dataset per pair, in *pairs* order, with ``track_azimuth_deg``
        (:func:`~curryer.correction.psf.ground_track_azimuth_deg` at the GCP
        centre) alongside the errors, correlation and geometry.
    chip_images : list[xr.Dataset]
        One image dataset per pair, in *pairs* order, with ``science_key``
        and ``gcp_key`` attributes; empty unless *keep_images*.

    Raises
    ------
    ValueError
        If an observation file carries no valid spacecraft ECEF position, or
        the GCP chip centre lies outside the observation grid (see
        :func:`~curryer.correction.psf.resolve_spacecraft_ecef`), or a file
        cannot be read.  No pair is skipped.
    """
    from curryer.compute.constants import WGS84_SEMI_MAJOR_AXIS_KM  # noqa: PLC0415

    from .image_io import (
        load_image_grid,
        load_los_vectors,
        load_observation_file,
        load_optical_psf,
        observation_los_vectors,
    )
    from .image_match import chip_image_dataset, integrated_image_match
    from .psf import ground_track_azimuth_deg, resolve_spacecraft_ecef

    sc_pos_name = setup.spacecraft_position_name
    boresight_name = setup.boresight_name
    t_matrix_name = setup.transformation_matrix_name

    los_vectors = load_los_vectors(los_file)
    optical_psfs = load_optical_psf(psf_file)

    datasets: list[xr.Dataset] = []
    chip_images: list[xr.Dataset] = []
    for obs_path, gcp_path in pairs:
        obs_grid, r_sc_file = load_observation_file(obs_path)
        obs_los = observation_los_vectors(obs_path, los_vectors, obs_grid.data.shape[1])
        gcp_grid = load_image_grid(gcp_path, mat_key="GCP")

        mid_i, mid_j = gcp_grid.mid_indices
        gcp_lat = float(gcp_grid.lat[mid_i, mid_j])
        gcp_lon = float(gcp_grid.lon[mid_i, mid_j])

        r_iss_m, boresight, t_matrix = resolve_spacecraft_ecef(obs_grid, r_sc_file, gcp_lat, gcp_lon)
        track_azimuth = ground_track_azimuth_deg(obs_grid, gcp_lat, gcp_lon)

        result = integrated_image_match(
            subimage=obs_grid,
            gcp=gcp_grid,
            r_iss_midframe_m=r_iss_m,
            los_vectors_hs=obs_los,
            optical_psfs=optical_psfs,
            geolocation_config=setup.psf_sampling,
            search_config=setup.search,
        )

        # Convert km errors to degrees on the WGS-84 equatorial radius, the same radius
        # ErrorStatsProcessor uses to convert them back to meters.
        lat_error_deg = np.rad2deg(result.lat_error_km / WGS84_SEMI_MAJOR_AXIS_KM)
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
                "correlation": (["measurement"], [result.ccv_final]),
                "correlation_secondary": (["measurement"], [result.ccv_secondary]),
                "track_azimuth_deg": (["measurement"], [track_azimuth]),
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
        if keep_images:
            chip = chip_image_dataset(obs_grid, gcp_grid, result)
            chip.attrs.update({"science_key": Path(obs_path).name, "gcp_key": Path(gcp_path).name})
            chip_images.append(chip)
        logger.info(
            "  Matched %s → %s: lat_err=%.3f km  lon_err=%.3f km  ccv=%.3f",
            Path(obs_path).name,
            Path(gcp_path).name,
            result.lat_error_km,
            result.lon_error_km,
            result.ccv_final,
        )

    return datasets, chip_images


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
    # Pre-computed input modes (backward-compatible)
    image_matching_results: list[xr.Dataset] | None = None,
    geolocated_data: xr.Dataset | None = None,
    work_dir: Path | None = None,
    keep_images: bool = False,
) -> VerificationResult:
    """Evaluate current alignment against mission requirements.

    No parameter variation, no kernel creation, no iteration loop.  This
    function checks whether a **given** set of alignment parameters meets
    geolocation requirements.

    Input priority (first match wins)
    ----------------------------------
    1. *image_matching_results* — pre-computed outputs from image matching;
       the most common entry point for weekly automated checks.
    2. *geolocated_data* — raw geolocated data.  Either set
       ``setup.image_matching_func`` for a custom matcher, or supply
       *gcp_directory*, *los_file*, and *psf_file* to run built-in spatial
       pairing + image matching.
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
        local path or an ``s3://`` URI (requires ``boto3``).  Each observation
        file must carry the mid-frame spacecraft ECEF position in meters
        (``position`` in a NetCDF root group, ``R_ISS_midframe`` in ``.mat``),
        and its grid rows must be frames (the middle row being the mid-frame)
        and its columns cross-track pixels.  A NetCDF observation cropped to
        some detector columns carries ``detector_pixel``, their rows in the
        LOS table (see :func:`~curryer.correction.image_io.observation_los_vectors`).
    observation_paths : list of path or None
        Observation file paths for automatic GCP pairing.
        Requires *gcp_directory*, *los_file*, and *psf_file*.  Same
        spacecraft-position requirement as *gcp_pairs*.
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
        GCP center must be inside the observation footprint).
    gcp_pattern : str, optional
        Glob pattern used to discover GCP chips when *gcp_directory* is
        provided.  Defaults to ``"*_regridded.nc"``.
    image_matching_results : list[xr.Dataset] or None
        Pre-computed image-matching datasets, one per GCP pair.
    geolocated_data : xr.Dataset or None
        Already-geolocated data.  Matched either via ``setup.image_matching_func``
        (custom override) or, when *gcp_directory*, *los_file*, and *psf_file* are
        supplied, via built-in spatial pairing + image matching.  The built-in
        path queries SPICE for the instrument boresight, so the caller must have
        the instrument, frame, attitude and leapsecond kernels loaded and
        covering the dataset's mid-frame time; see :func:`image_matching` for
        the mid-frame geometry this records.
    work_dir : Path or None, optional
        Working directory for outputs.  Created if absent.
    keep_images : bool, optional
        Keep the images behind each match in
        :attr:`VerificationResult.chip_images` (observed, emulated and
        reference images; see
        :func:`~curryer.correction.image_match.chip_image_dataset`).  Only
        for the *gcp_pairs* and *observation_paths* modes, which load the
        images.  A GCP chip can be tens of MB.

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
        *gcp_directory* are not both supplied; when an observation or GCP
        file fails to load during *observation_paths* / *geolocated_data*
        pairing (including a missing file; the original exception is chained
        as ``__cause__``); when an observation file carries no valid
        spacecraft position; when a GCP chip centre lies outside its
        observation grid; when image matching produces no results; or when
        *keep_images* is set with *image_matching_results* or
        *geolocated_data*.
    FileNotFoundError
        If *los_file* or *psf_file* does not exist, if *gcp_directory* does
        not exist in *observation_paths* mode, or if a file listed in
        *gcp_pairs* does not exist. Missing observation or GCP files found
        during pairing raise ``ValueError`` as above.
    spiceypy.utils.exceptions.SpiceyError
        In the *geolocated_data* mode, if the SPICE boresight query fails.
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
    chip_images: list[xr.Dataset] | None = None
    if keep_images and (image_matching_results is not None or geolocated_data is not None):
        raise ValueError(
            "keep_images requires the gcp_pairs or observation_paths mode; "
            "image_matching_results and geolocated_data do not carry the matched images."
        )

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
        matched, chips = _run_image_matching_for_pairs(pairs, str(los_file), str(psf_file), setup, keep_images)
        chip_images = chips if keep_images else None
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

        matched, chips = _run_image_matching_for_pairs(raw_pairs, str(los_file), str(psf_file), setup, keep_images)
        chip_images = chips if keep_images else None
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
    weighted_percent = _weighted_percent_within(aggregate_stats, requirements)

    # ------------------------------------------------------------------
    # Step 4: Per-GCP detail
    # ------------------------------------------------------------------
    per_gcp_errors = _build_per_gcp_errors(aggregate_stats, source_mapping, requirements)
    for index, chip in enumerate(chip_images or []):
        chip.attrs["gcp_index"] = index

    # ------------------------------------------------------------------
    # Step 5: Warnings + summary table
    # ------------------------------------------------------------------
    warnings = _generate_warnings(passed, percent_within, requirements)
    summary_table = _format_summary_table(per_gcp_errors, requirements, percent_within, passed, weighted_percent)

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
        chip_images=chip_images,
        weighted_percent_within_threshold=weighted_percent,
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


# ============================================================================
# Saving, loading and review
# ============================================================================

_SUMMARY_COLUMNS = (
    "gcp_index",
    "science_key",
    "gcp_key",
    "status",
    "rejection_reason",
    "review",
    "nadir_equiv_error_m",
    "along_track_error_m",
    "cross_track_error_m",
    "lat_error_deg",
    "lon_error_deg",
    "correlation",
    "correlation_secondary",
    "off_nadir_angle_deg",
    "quality_weight",
)
_REVIEW_DECISIONS = ("accept", "reject")


def save_verification(result: VerificationResult, out_dir: Path) -> None:
    """Write *result* to *out_dir* for review and for :func:`load_verification`.

    Files written:

    - ``result.json``: the result without ``aggregate_stats`` and ``chip_images``.
    - ``aggregate_stats.nc``: :attr:`VerificationResult.aggregate_stats`.
    - ``summary.csv``: one row per GCP (columns ``gcp_index``, ``science_key``,
      ``gcp_key``, ``status``, ``rejection_reason``, ``review``,
      ``nadir_equiv_error_m``, ``along_track_error_m``,
      ``cross_track_error_m``, ``lat_error_deg``, ``lon_error_deg``,
      ``correlation``, ``correlation_secondary``, ``off_nadir_angle_deg``,
      ``quality_weight``; empty cells for ``None``).  A reviewer fills ``review`` with ``accept``
      or ``reject``; see :func:`read_review_decisions`.
    - ``chips/gcp_<index>.nc``: one file per entry of
      :attr:`VerificationResult.chip_images`, when present.

    Parameters
    ----------
    result : VerificationResult
        Result to save.
    out_dir : Path
        Directory; created if absent.

    Raises
    ------
    FileExistsError
        If *out_dir* already holds a ``result.json``.
    """
    out_dir = Path(out_dir)
    if (out_dir / "result.json").exists():
        raise FileExistsError(f"{out_dir} already holds a saved verification result.")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "result.json").write_text(result.model_dump_json(indent=2, exclude={"aggregate_stats"}))
    result.aggregate_stats.to_netcdf(out_dir / "aggregate_stats.nc")
    rows = [err.model_dump(include=set(_SUMMARY_COLUMNS)) for err in result.per_gcp_errors]
    pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS)).to_csv(out_dir / "summary.csv", index=False)
    if result.chip_images is not None:
        (out_dir / "chips").mkdir()
        for chip in result.chip_images:
            chip.to_netcdf(out_dir / "chips" / f"gcp_{chip.attrs['gcp_index']:04d}.nc")


def load_verification(out_dir: Path) -> VerificationResult:
    """Load a result written by :func:`save_verification`.

    Parameters
    ----------
    out_dir : Path
        Directory written by :func:`save_verification`.

    Returns
    -------
    VerificationResult
        With ``aggregate_stats`` and, when ``chips/`` exists, ``chip_images``
        in ``gcp_index`` order.

    Raises
    ------
    FileNotFoundError
        If ``result.json`` or ``aggregate_stats.nc`` is missing.
    """
    out_dir = Path(out_dir)
    data = json.loads((out_dir / "result.json").read_text())
    data["aggregate_stats"] = xr.load_dataset(out_dir / "aggregate_stats.nc")
    chip_dir = out_dir / "chips"
    if chip_dir.is_dir():
        data["chip_images"] = sorted(
            (xr.load_dataset(path) for path in chip_dir.glob("gcp_*.nc")), key=lambda chip: chip.attrs["gcp_index"]
        )
    return VerificationResult.model_validate(data)


def read_review_decisions(summary_csv: Path) -> dict[int, Literal["accept", "reject"]]:
    """Read the reviewer's decisions from a ``summary.csv`` written by :func:`save_verification`.

    Parameters
    ----------
    summary_csv : Path
        CSV with ``gcp_index`` and ``review`` columns.  An empty ``review``
        cell means no decision.

    Returns
    -------
    dict of int to {"accept", "reject"}
        ``gcp_index`` to decision, for the rows with one.

    Raises
    ------
    ValueError
        If the CSV lacks either column, or a ``review`` cell holds anything
        but ``accept``, ``reject`` or nothing.
    """
    table = pd.read_csv(summary_csv, dtype={"review": str}, keep_default_na=False)
    missing = {"gcp_index", "review"} - set(table.columns)
    if missing:
        raise ValueError(f"{summary_csv} has no {sorted(missing)} column(s).")
    decisions: dict[int, Literal["accept", "reject"]] = {}
    for index, decision in zip(table["gcp_index"], table["review"].str.strip(), strict=True):
        if decision == "":
            continue
        if decision not in _REVIEW_DECISIONS:
            raise ValueError(
                f"{summary_csv}: gcp_index {index} has review {decision!r}; expected 'accept', 'reject' or empty."
            )
        decisions[int(index)] = decision
    return decisions


def apply_review(result: VerificationResult, decisions: dict[int, Literal["accept", "reject"]]) -> VerificationResult:
    """Apply reviewer decisions and recompute the statistics without re-matching.

    ``"accept"`` brings a GCP into the statistics (status ``pass`` / ``fail``
    by its error); ``"reject"`` takes it out (status ``rejected``, reason
    ``"rejected in review"`` unless the match-quality gates already rejected
    it).  GCPs without a decision keep their current state.

    Parameters
    ----------
    result : VerificationResult
        Result from :func:`verify` or :func:`load_verification`.
    decisions : dict of int to {"accept", "reject"}
        ``gcp_index`` to decision, e.g. from :func:`read_review_decisions`.

    Returns
    -------
    VerificationResult
        A new result: ``aggregate_stats`` gains a ``review`` variable, its
        ``quality_weight`` follows the new acceptance, and its statistics
        cover the accepted GCPs (none when every one is rejected);
        ``per_gcp_errors``, ``passed``, ``percent_within_threshold``,
        ``weighted_percent_within_threshold``, ``warnings`` and
        ``summary_table`` are recomputed.  *result* is not
        modified.

    Raises
    ------
    ValueError
        If a decision names a ``gcp_index`` not in *result*, or is not
        ``"accept"`` or ``"reject"``.
    """
    stats = result.aggregate_stats.copy(deep=True)
    labels = [int(label) for label in stats["measurement"].values]
    unknown = sorted(set(decisions) - set(labels))
    if unknown:
        raise ValueError(f"Review decisions for gcp_index {unknown}, which are not in the result.")
    invalid = {index: decision for index, decision in decisions.items() if decision not in _REVIEW_DECISIONS}
    if invalid:
        raise ValueError(f"Review decisions must be 'accept' or 'reject', got {invalid}.")

    accepted = stats["accepted"].values.copy()
    reasons = stats["rejection_reason"].values.astype(object)
    reviews = stats["review"].values.astype(object) if "review" in stats.data_vars else np.full(len(labels), "", object)
    for i, label in enumerate(labels):
        decision = decisions.get(label)
        if decision == "accept":
            accepted[i], reasons[i] = True, ""
        elif decision == "reject" and accepted[i]:
            accepted[i], reasons[i] = False, "rejected in review"
        if decision is not None:
            reviews[i] = decision
    stats["accepted"] = (["measurement"], accepted)
    stats["rejection_reason"] = (["measurement"], reasons.astype(str))
    stats["review"] = (["measurement"], reviews.astype(str))
    if "quality_weight" in stats.data_vars:
        corr_name = next(name for name in ("correlation", "ccv", "im_ccv") if name in stats.data_vars)
        stats["quality_weight"] = quality_weights(stats[corr_name].values, accepted)

    processor = ErrorStatsProcessor(config=ErrorStatsConfig())
    for key in [*processor._calculate_statistics(np.zeros(1)), *weighted_statistics(np.zeros(1), np.ones(1))]:
        stats.attrs.pop(key, None)
    stats.attrs.update(n_accepted=int(accepted.sum()), n_rejected=int((~accepted).sum()))
    stats = _with_statistics(processor, stats)

    requirements = result.requirements
    passed, percent_within = _check_threshold(stats, requirements)
    weighted_percent = _weighted_percent_within(stats, requirements)
    source_mapping = [
        (err.science_key, err.gcp_key) for err in sorted(result.per_gcp_errors, key=lambda e: e.gcp_index)
    ]
    per_gcp_errors = _build_per_gcp_errors(stats, source_mapping, requirements)
    return result.model_copy(
        update={
            "passed": passed,
            "percent_within_threshold": percent_within,
            "weighted_percent_within_threshold": weighted_percent,
            "aggregate_stats": stats,
            "per_gcp_errors": per_gcp_errors,
            "warnings": _generate_warnings(passed, percent_within, requirements),
            "summary_table": _format_summary_table(
                per_gcp_errors, requirements, percent_within, passed, weighted_percent
            ),
        }
    )
