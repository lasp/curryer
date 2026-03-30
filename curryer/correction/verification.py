"""Verification module for geolocation requirements compliance.

Provides :func:`verify`, a standalone entry point that evaluates the current
set of SPICE kernels and alignment parameters against mission geolocation
requirements — without running the iterative correction loop.

Typical use-cases
-----------------
Weekly automated check (CLARREO)
    Pass pre-computed ``image_matching_results`` (the most common path):

    >>> result = verify(config, work_dir, image_matching_results=weekly_datasets)
    >>> if not result.passed:
    ...     send_alert(result.summary_table)

Post-correction validation
    After a full GCS run, verify the optimised parameter set:

    >>> result = verify(config, work_dir, image_matching_results=post_correction_datasets)

One-off compliance check
    Provide already-geolocated data and let verification run image matching:

    >>> result = verify(config, work_dir, geolocated_data=raw_dataset)

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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from curryer.correction.config import CorrectionConfig
from curryer.correction.error_stats import ErrorStatsConfig, ErrorStatsProcessor

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic models
# ============================================================================


class RequirementsConfig(BaseModel):
    """Verification requirements / thresholds.

    Can be attached as an optional ``verification`` field on
    :class:`~curryer.correction.config.CorrectionConfig`, or passed directly
    to :func:`verify`.  When neither is supplied :func:`verify` falls back to
    :attr:`~curryer.correction.config.CorrectionConfig.performance_threshold_m`
    and :attr:`~curryer.correction.config.CorrectionConfig.performance_spec_percent`.

    Attributes
    ----------
    performance_threshold_m : float
        Per-measurement nadir-equivalent error limit in metres.
        A measurement *passes* when its error is **below** this value.
    performance_spec_percent : float
        Minimum fraction of measurements (0–100) that must pass for the
        overall verification to be considered successful.
    """

    performance_threshold_m: float
    performance_spec_percent: float


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


# ============================================================================
# Internal helpers
# ============================================================================


def _build_requirements(config: CorrectionConfig) -> RequirementsConfig:
    """Extract or construct :class:`RequirementsConfig` from *config*.

    If *config* carries a ``verification`` attribute (a
    :class:`RequirementsConfig` instance), that object is returned directly.
    Otherwise the top-level :attr:`~CorrectionConfig.performance_threshold_m`
    and :attr:`~CorrectionConfig.performance_spec_percent` fields are used.

    Parameters
    ----------
    config : CorrectionConfig
        Correction configuration from which to extract thresholds.

    Returns
    -------
    RequirementsConfig
    """
    # TODO(#131): Add `verification: RequirementsConfig | None` as an optional
    # field on CorrectionConfig so this override path works without __setattr__.
    existing = getattr(config, "verification", None)
    if isinstance(existing, RequirementsConfig):
        logger.debug("Using RequirementsConfig from config.verification")
        return existing
    return RequirementsConfig(
        performance_threshold_m=config.performance_threshold_m,
        performance_spec_percent=config.performance_spec_percent,
    )


def _aggregate_results(
    image_matching_results: list[xr.Dataset],
    config: CorrectionConfig,
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
    config : CorrectionConfig
        Used for mission-specific variable names
        (``spacecraft_position_name``, ``boresight_name``,
        ``transformation_matrix_name``).

    Returns
    -------
    xr.Dataset
        Combined dataset with a single ``measurement`` dimension.
    """
    from curryer.correction.pipeline import _aggregate_image_matching_results

    if len(image_matching_results) == 1:
        ds = image_matching_results[0]
        # Always normalize the measurement coordinate to sequential integers
        # so that downstream gcp_index values are predictable.
        n = ds.sizes.get("measurement", len(ds["lat_error_deg"]))
        ds = ds.assign_coords(measurement=np.arange(n))
        return ds

    return _aggregate_image_matching_results(image_matching_results, config)


def _run_error_stats(
    aggregated: xr.Dataset,
    config: CorrectionConfig,
) -> xr.Dataset:
    """Run :class:`~curryer.correction.error_stats.ErrorStatsProcessor` on *aggregated*.

    Parameters
    ----------
    aggregated : xr.Dataset
        Combined image-matching result with a ``measurement`` dimension.
    config : CorrectionConfig
        Used to build :class:`~curryer.correction.error_stats.ErrorStatsConfig`.

    Returns
    -------
    xr.Dataset
        Processed dataset with ``nadir_equiv_total_error_m`` and related
        intermediate variables.
    """
    error_config = ErrorStatsConfig.from_correction_config(config)
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


# ============================================================================
# Public API
# ============================================================================


def verify(
    config: CorrectionConfig,
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
       ``config.image_matching_func`` to be set.
    3. Neither provided — raises :class:`ValueError`.

    Parameters
    ----------
    config : CorrectionConfig
        Configuration with all mission-specific settings:
        - Performance thresholds (``performance_threshold_m``, ``performance_spec_percent``)
        - Spacecraft variable names (``spacecraft_position_name``, ``boresight_name``, etc.)
        - Geolocation settings (SPICE kernels, instrument configuration)
        - Image matching function (``image_matching_func``)
        - Optional ``verification`` override (:class:`RequirementsConfig`)
    image_matching_results : list[xr.Dataset] or None
        Pre-computed image-matching datasets, one per GCP pair.  Each must
        have a ``measurement`` dimension and ``lat_error_deg`` /
        ``lon_error_deg`` variables plus the spacecraft-state variables
        expected by
        :class:`~curryer.correction.error_stats.ErrorStatsProcessor`.
    geolocated_data : xr.Dataset or None
        Already-geolocated data on which image matching will be run using
        ``config.image_matching_func``.  Ignored when
        *image_matching_results* is provided.
    work_dir : Path or None, optional
        Working directory for outputs.  Created if absent.
        If None (default), uses ``./verification_output``.

    Returns
    -------
    VerificationResult
        Structured pass/fail result with per-GCP detail and a
        human-readable :attr:`~VerificationResult.summary_table`.

    Raises
    ------
    ValueError
        When neither *image_matching_results* nor *geolocated_data* is
        provided, or when *geolocated_data* is supplied but
        ``config.image_matching_func`` is not set.
    """
    # Handle optional work_dir with sensible default
    if work_dir is None:
        work_dir = Path("verification_output")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc)
    requirements = _build_requirements(config)

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
        aggregated = _aggregate_results(image_matching_results, config)

    elif geolocated_data is not None:
        if config.image_matching_func is None:
            raise ValueError(
                "geolocated_data was provided but config.image_matching_func is not set. "
                "Either supply pre-computed image_matching_results or attach an "
                "image_matching_func to the config."
            )
        logger.info("Running image matching on provided geolocated_data")
        matched = config.image_matching_func(geolocated_data)
        if not isinstance(matched, list):
            matched = [matched]
        source_mapping = _build_source_mapping(matched)
        aggregated = _aggregate_results(matched, config)

    else:
        raise ValueError(
            "Neither image_matching_results nor geolocated_data was provided. Supply at least one of them to verify()."
        )

    # ------------------------------------------------------------------
    # Step 2: Compute nadir-equivalent error statistics
    # ------------------------------------------------------------------
    logger.info("Computing nadir-equivalent error statistics")
    aggregate_stats = _run_error_stats(aggregated, config)

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

    return VerificationResult(
        passed=passed,
        per_gcp_errors=per_gcp_errors,
        aggregate_stats=aggregate_stats,
        requirements=requirements,
        summary_table=summary_table,
        percent_within_threshold=percent_within,
        warnings=warnings,
        timestamp=timestamp,
    )
