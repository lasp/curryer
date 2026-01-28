"""
Weekly Geolocation Verification Module.

This module provides a verification workflow that checks operational geolocation
performance against mission thresholds WITHOUT running parameter optimization or
creating/modifying SPICE kernels.

The verification process:
1. Takes ALREADY GEOLOCATED data (no geolocation performed)
2. Performs GCP pairing (find which GCPs correspond to the data)
3. Runs image matching (measure offsets between geolocated data and GCPs)
4. Calculates error statistics (nadir-equivalent errors)
5. Checks if performance meets threshold (e.g., 39% within 250m)
6. Generates warnings if performance degrades

IMPORTANT: This module does NOT:
- Perform geolocation
- Create or modify SPICE kernels
- Load meta kernels
- Create dynamic or parameter-specific kernels

It only measures the accuracy of existing geolocated data by comparing to GCPs.

Configuration Strategy:
----------------------
The config needs:
- gcp_pairing_func: Function to find GCP pairs
- image_matching_func: Function to measure offsets
- performance_threshold_m: Error threshold (e.g., 250.0)
- performance_spec_percent: Required percentage (e.g., 39.0)

The n_iterations and parameters fields are NOT used (no kernel creation).

Quick Start:
-----------
    from curryer.correction.verification import run_verification
    from curryer.correction.correction import CorrectionConfig

    # Configure for verification
    config = CorrectionConfig(
        seed=42,
        performance_threshold_m=250.0,
        performance_spec_percent=39.0,
        earth_radius_m=6378137.0,
        geo=geo_config,
        n_iterations=1,      # Not used
        parameters=[],       # Not used
        gcp_pairing_func=your_pairing_function,
        image_matching_func=your_matching_function,
    )

    # Provide already-geolocated data
    data_sets = [
        (telemetry_key1, science_key1, gcp_key1),
        # ... more data
    ]

    # Run verification (no kernel creation!)
    result = run_verification(config, Path("/tmp/work"), data_sets)

    # Check results
    if result.passed:
        print(f"✅ {result.percent_within_threshold:.1f}% within {result.threshold_m}m")
    else:
        for warning in result.warnings:
            print(f"  {warning}")

See Also:
--------
- curryer.correction.correction.call_error_stats_module() - Error statistics
- curryer.correction.pairing - GCP pairing functions
- curryer.correction.image_match - Image matching functions
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from curryer.correction.correction import CorrectionConfig, call_error_stats_module

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """
    Results from a weekly verification run.

    Attributes
    ----------
    passed : bool
        True if performance meets the threshold requirement
    aggregate_stats : xr.Dataset
        Combined error statistics from all measurements
        Contains 'nadir_equivalent_error_m' and other error metrics
    per_pair_metrics : list[dict[str, Any]]
        Individual metrics for each GCP pair processed
        Each dict contains: pair_index, science_key, gcp_key, error metrics, etc.
    warnings : list[str]
        User-facing warning messages
        Empty list if verification passed
    timestamp : datetime
        When the verification was run
    config_summary : dict[str, Any]
        Key configuration values used for this verification
    percent_within_threshold : float
        Percentage of measurements within the error threshold
    threshold_m : float
        Error threshold in meters (from config)
    required_percent : float
        Required percentage for passing (from config)

    Examples
    --------
    >>> result = run_verification(config, work_dir, data_sets)
    >>> if result.passed:
    ...     print(f"✅ {result.percent_within_threshold:.1f}% within {result.threshold_m}m")
    ... else:
    ...     print(f"❌ Only {result.percent_within_threshold:.1f}% within threshold")
    ...     for warning in result.warnings:
    ...         print(f"  {warning}")
    """

    passed: bool
    aggregate_stats: xr.Dataset
    per_pair_metrics: list[dict[str, Any]]
    warnings: list[str]
    timestamp: datetime
    config_summary: dict[str, Any]
    percent_within_threshold: float
    threshold_m: float
    required_percent: float


def run_verification(
    config: CorrectionConfig, work_dir: Path, data_sets: list[tuple[str, str, str]]
) -> VerificationResult:
    """
    Run weekly verification on already-geolocated data.

    This function measures geolocation accuracy WITHOUT performing geolocation
    or creating/modifying SPICE kernels. It assumes input data is already
    geolocated and measures accuracy by comparing to GCPs.

    Workflow (NO kernel creation):
    1. GCP Pairing - Find which GCPs correspond to each data set
    2. Image Matching - Measure offsets between geolocated data and GCPs
    3. Error Statistics - Calculate nadir-equivalent errors
    4. Threshold Check - Determine if performance meets requirements

    Parameters
    ----------
    config : CorrectionConfig
        Configuration with verification settings:
        - gcp_pairing_func: Required - function to find GCP pairs
        - image_matching_func: Required - function to measure offsets
        - performance_threshold_m: Error threshold (e.g., 250.0)
        - performance_spec_percent: Required percentage (e.g., 39.0)
        - earth_radius_m: Earth radius for error calculations
        Note: n_iterations and parameters are NOT used
    work_dir : Path
        Working directory for temporary files (if needed by functions)
    data_sets : list[tuple[str, str, str]]
        List of (telemetry_key, science_key, gcp_key) tuples
        Each tuple identifies one already-geolocated observation
        Keys are passed to pairing and matching functions

    Returns
    -------
    VerificationResult
        Structured results with pass/fail status, statistics, and warnings

    Raises
    ------
    ValueError
        If required functions not configured or no valid results

    Examples
    --------
    >>> config = CorrectionConfig(
    ...     seed=42,
    ...     performance_threshold_m=250.0,
    ...     performance_spec_percent=39.0,
    ...     earth_radius_m=6378137.0,
    ...     geo=geo_config,
    ...     n_iterations=1,
    ...     parameters=[],
    ...     gcp_pairing_func=spatial_pairing,
    ...     image_matching_func=image_matching,
    ... )
    >>> data_sets = [
    ...     ("/data/geo_001.nc", "/data/sci_001.nc", "/data/gcp_001.tif"),
    ... ]
    >>> result = run_verification(config, Path("/tmp/work"), data_sets)
    >>> print(f"Passed: {result.passed}")
    """
    logger.info("=== WEEKLY VERIFICATION ===")
    logger.info(f"  Data sets: {len(data_sets)}")
    logger.info(f"  Threshold: {config.performance_threshold_m}m")
    logger.info(f"  Required: {config.performance_spec_percent}%")
    logger.info(f"  Mode: Verification (NO geolocation, NO kernel creation)")

    # Validate required functions are configured
    _validate_verification_config(config)

    # Extract functions from config
    gcp_pairing_func = config.gcp_pairing_func
    image_matching_func = config.image_matching_func

    # Step 1: GCP Pairing for all data sets
    logger.info("Step 1/3: GCP Pairing")
    science_keys = [sci_key for _, sci_key, _ in data_sets]
    gcp_pairs = gcp_pairing_func(science_keys)

    if not gcp_pairs:
        raise ValueError("No GCP pairs found. Cannot perform verification.")

    logger.info(f"  Found {len(gcp_pairs)} GCP pair(s)")

    # Step 2: Image Matching for each pair
    logger.info("Step 2/3: Image Matching")
    all_image_matching_results = []
    per_pair_metrics = []

    for pair_idx, (telemetry_key, science_key, gcp_key) in enumerate(data_sets):
        logger.info(f"  Processing pair {pair_idx + 1}/{len(data_sets)}: {science_key}")

        try:
            # Call image matching function (mission-specific implementation)
            # This measures offset between already-geolocated data and GCP
            # NO geolocation is performed here
            image_match_result = image_matching_func(
                telemetry_key=telemetry_key,
                science_key=science_key,
                gcp_key=gcp_key,
                config=config,
            )

            # Validate result is an xr.Dataset
            if not isinstance(image_match_result, xr.Dataset):
                logger.warning(f"  Image matching result is not xr.Dataset, skipping pair {pair_idx + 1}")
                continue

            all_image_matching_results.append(image_match_result)

            # Extract basic metrics
            per_pair_metrics.append(
                {
                    "pair_index": pair_idx,
                    "science_key": science_key,
                    "gcp_key": gcp_key,
                    "lat_error_deg": float(image_match_result.attrs.get("lat_error_deg", np.nan)),
                    "lon_error_deg": float(image_match_result.attrs.get("lon_error_deg", np.nan)),
                    "correlation_ccv": float(image_match_result.attrs.get("correlation_ccv", np.nan)),
                }
            )

            logger.debug(f"    Lat error: {per_pair_metrics[-1]['lat_error_deg']:.6f}°")
            logger.debug(f"    Lon error: {per_pair_metrics[-1]['lon_error_deg']:.6f}°")

        except Exception as e:
            logger.error(f"  Error processing pair {pair_idx + 1}: {e}")
            # Continue with other pairs
            continue

    if len(all_image_matching_results) == 0:
        raise ValueError("No valid image matching results. Cannot calculate statistics.")

    logger.info(f"  Successfully matched {len(all_image_matching_results)} pair(s)")

    # Step 3: Error Statistics
    logger.info("Step 3/3: Calculating Error Statistics")

    # Call error stats module (handles aggregation and nadir-equivalent calculation)
    aggregate_stats = call_error_stats_module(all_image_matching_results, correction_config=config)

    logger.info(f"  Calculated stats for {len(aggregate_stats.measurement)} measurements")

    # Update per-pair metrics with error stats if available
    per_pair_metrics = _enrich_per_pair_metrics(per_pair_metrics, aggregate_stats)

    # Check if performance meets threshold
    passed, percent = _check_threshold(aggregate_stats, config)
    logger.info(f"  Threshold check: {percent:.1f}% within {config.performance_threshold_m}m")
    logger.info(f"  Required: {config.performance_spec_percent}%")
    logger.info(f"  Result: {'PASSED' if passed else 'FAILED'}")

    # Generate warnings if verification failed
    warnings = _generate_warnings(passed, percent, config)
    if warnings:
        for warning in warnings:
            logger.warning(warning)

    # Create result object
    result = VerificationResult(
        passed=passed,
        aggregate_stats=aggregate_stats,
        per_pair_metrics=per_pair_metrics,
        warnings=warnings,
        timestamp=datetime.now(),
        config_summary=_extract_config_summary(config),
        percent_within_threshold=percent,
        threshold_m=config.performance_threshold_m,
        required_percent=config.performance_spec_percent,
    )

    logger.info("=== VERIFICATION COMPLETE ===")
    return result


def _validate_verification_config(config: CorrectionConfig) -> None:
    """
    Validate that required functions are configured for verification.

    Parameters
    ----------
    config : CorrectionConfig
        Configuration to validate

    Raises
    ------
    ValueError
        If required functions are not configured
    """
    if config.gcp_pairing_func is None:
        raise ValueError(
            "config.gcp_pairing_func is required for verification. "
            "Set this to your mission-specific GCP pairing function."
        )

    if config.image_matching_func is None:
        raise ValueError(
            "config.image_matching_func is required for verification. "
            "Set this to your mission-specific image matching function."
        )

    logger.debug("Verification config validated: required functions present")


def _enrich_per_pair_metrics(
    per_pair_metrics: list[dict[str, Any]], aggregate_stats: xr.Dataset
) -> list[dict[str, Any]]:
    """
    Enrich per-pair metrics with error statistics if available.

    Parameters
    ----------
    per_pair_metrics : list[dict]
        Basic per-pair metrics
    aggregate_stats : xr.Dataset
        Aggregate error statistics

    Returns
    -------
    list[dict]
        Enriched per-pair metrics
    """
    # If aggregate_stats has per-measurement data, try to map back to pairs
    # For now, just add aggregate-level info
    for metric in per_pair_metrics:
        if "nadir_equivalent_error_m" in aggregate_stats:
            # Could map individual measurements back to pairs
            # For simplicity, just note that nadir errors are in aggregate_stats
            metric["has_nadir_error"] = True

        # Add RMS info if available in attrs
        if hasattr(aggregate_stats, "attrs"):
            metric["aggregate_rms_error_m"] = aggregate_stats.attrs.get("rms_error_m", np.nan)
            metric["aggregate_mean_error_m"] = aggregate_stats.attrs.get("mean_error_m", np.nan)

    return per_pair_metrics


def _check_threshold(aggregate_stats: xr.Dataset, config: CorrectionConfig) -> tuple[bool, float]:
    """
    Check if performance meets the threshold requirement.

    Calculates what percentage of measurements have nadir-equivalent
    error below the threshold, and compares to the required percentage.

    Parameters
    ----------
    aggregate_stats : xr.Dataset
        Aggregate error statistics
        Must contain 'nadir_equiv_total_error_m' variable (total nadir-equivalent error)
    config : CorrectionConfig
        Configuration with threshold settings

    Returns
    -------
    passed : bool
        True if percentage meets or exceeds requirement
    percent_below : float
        Actual percentage of measurements below threshold
    """
    # Extract nadir-equivalent errors (total error)
    # The error stats module provides: nadir_equiv_total_error_m, nadir_equiv_vp_error_m, nadir_equiv_xvp_error_m
    # We use total error for verification
    nadir_errors = aggregate_stats["nadir_equiv_total_error_m"].values

    # Count measurements below threshold
    count_below = np.sum(nadir_errors < config.performance_threshold_m)

    # Calculate percentage
    percent_below = (count_below / len(nadir_errors)) * 100.0

    # Check if meets requirement
    passed = percent_below >= config.performance_spec_percent

    logger.debug(
        f"Threshold check: {count_below}/{len(nadir_errors)} below {config.performance_threshold_m}m "
        f"= {percent_below:.1f}% (required: {config.performance_spec_percent}%)"
    )

    return passed, percent_below


def _generate_warnings(passed: bool, percent_below: float, config: CorrectionConfig) -> list[str]:
    """
    Generate user-facing warning messages.

    If verification failed, creates a clear warning message recommending
    action to correct the performance degradation.

    Parameters
    ----------
    passed : bool
        Whether verification passed
    percent_below : float
        Actual percentage below threshold
    config : CorrectionConfig
        Configuration with threshold settings

    Returns
    -------
    list[str]
        List of warning messages (empty if passed)
    """
    warnings = []

    if not passed:
        warning = (
            f"⚠️  VERIFICATION FAILED: Only {percent_below:.1f}% of observations "
            f"meet the {config.performance_threshold_m}m nadir-equivalent error threshold "
            f"(required: {config.performance_spec_percent}%). "
            f"Recommend running the correction module to optimize calibration parameters."
        )
        warnings.append(warning)

    return warnings


def _extract_config_summary(config: CorrectionConfig) -> dict[str, Any]:
    """
    Extract key configuration values for logging and reproducibility.

    Parameters
    ----------
    config : CorrectionConfig
        Configuration object

    Returns
    -------
    dict
        Summary of key configuration values
    """
    return {
        "performance_threshold_m": config.performance_threshold_m,
        "performance_spec_percent": config.performance_spec_percent,
        "earth_radius_m": config.earth_radius_m,
        "has_gcp_pairing_func": config.gcp_pairing_func is not None,
        "has_image_matching_func": config.image_matching_func is not None,
    }
