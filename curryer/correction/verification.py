"""
Weekly Geolocation Verification Module.

This module provides a SIMPLE verification workflow that checks operational geolocation
performance against mission thresholds WITHOUT running parameter optimization or
creating/modifying SPICE kernels.

DESIGN PHILOSOPHY: This module is intentionally simple. It directly uses existing
pairing and image matching implementations without function injection or abstraction layers.

The verification process:
1. Takes ALREADY GEOLOCATED data files (no geolocation performed)
2. Performs GCP pairing using pair_files() from pairing.py (spatial overlap detection)
3. Runs image matching using image_matching() from correction.py (measure offsets)
4. Calculates error statistics using call_error_stats_module() (nadir-equivalent errors)
5. Checks if performance meets threshold (e.g., 39% within 250m)
6. Generates warnings if performance degrades

IMPORTANT: This module does NOT:
- Perform geolocation
- Create or modify SPICE kernels
- Load meta kernels
- Create dynamic or parameter-specific kernels
- Use function injection (unlike correction.py)

It only measures the accuracy of existing geolocated data by comparing to GCPs using
the real implementations from pairing.py, image_match.py, and correction.py.

Quick Start:
-----------
    from pathlib import Path
    from curryer.correction.verification import run_verification
    from curryer.correction.correction import CorrectionConfig

    # Configure for verification (NO function injection needed!)
    config = CorrectionConfig(
        seed=42,
        performance_threshold_m=250.0,
        performance_spec_percent=39.0,
        earth_radius_m=6378137.0,
        geo=geo_config,
        n_iterations=1,      # Not used
        parameters=[],       # Not used
    )

    # Provide paths to geolocated L1A files and GCP directory
    l1a_files = [
        Path("/data/geolocated_001.mat"),
        Path("/data/geolocated_002.mat"),
    ]
    gcp_directory = Path("/data/gcps/")

    # Run verification (no kernel creation, no function injection!)
    result = run_verification(
        config=config,
        work_dir=Path("/tmp/work"),
        l1a_files=l1a_files,
        gcp_directory=gcp_directory,
        telemetry=telemetry_df,
        calibration_dir=Path("/data/calibration"),
    )

    # Check results
    if result.passed:
        print(f"✅ {result.percent_within_threshold:.1f}% within {result.threshold_m}m")
    else:
        for warning in result.warnings:
            print(f"  {warning}")

See Also:
--------
- curryer.correction.pairing.pair_files() - GCP pairing (used directly)
- curryer.correction.correction.image_matching() - Image matching (used directly)
- curryer.correction.correction.call_error_stats_module() - Error statistics (used directly)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from curryer.correction.correction import CorrectionConfig, call_error_stats_module, image_matching
from curryer.correction.image_match import load_image_grid_from_mat
from curryer.correction.pairing import pair_files

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
    config: CorrectionConfig,
    work_dir: Path,
    l1a_files: list[Path],
    gcp_directory: Path,
    telemetry: pd.DataFrame,
    calibration_dir: Path,
    max_distance_m: float = 0.0,
    l1a_key: str = "subimage",
    gcp_key: str = "GCP",
    gcp_pattern: str = "*_resampled.mat",
) -> VerificationResult:
    """
    Run weekly verification on already-geolocated data.

    This function measures geolocation accuracy WITHOUT performing geolocation
    or creating/modifying SPICE kernels. It assumes input data is already
    geolocated and measures accuracy by comparing to GCPs.

    This is the SIMPLE approach: directly calls existing modules without
    function injection or abstraction layers.

    Workflow (NO kernel creation):
    1. GCP Pairing - Use pair_files() from pairing.py to find spatial overlaps
    2. Image Matching - Use image_matching() from correction.py to measure offsets
    3. Error Statistics - Use call_error_stats_module() to calculate nadir-equivalent errors
    4. Threshold Check - Determine if performance meets requirements

    Parameters
    ----------
    config : CorrectionConfig
        Configuration with verification settings:
        - performance_threshold_m: Error threshold (e.g., 250.0)
        - performance_spec_percent: Required percentage (e.g., 39.0)
        - earth_radius_m: Earth radius for error calculations
        - geo: GeolocationConfig with coordinate names
        Note: n_iterations, parameters, gcp_pairing_func, and image_matching_func are NOT used
    work_dir : Path
        Working directory for temporary files (if needed)
    l1a_files : list[Path]
        List of paths to geolocated L1A files (MATLAB .mat format)
        These are the already-geolocated science observations to verify
    gcp_directory : Path
        Directory containing GCP reference files
    telemetry : pd.DataFrame
        Spacecraft telemetry data (position, attitude, etc.)
        Required for image matching
    calibration_dir : Path
        Directory containing calibration files (LOS vectors, optical PSF)
        Required for image matching
    max_distance_m : float, optional
        Minimum margin for valid pairing (default 0.0):
        - 0.0: GCP center must be inside L1A footprint (strict)
        - >0: Allows GCP center up to this distance inside footprint
        - <0: Allows GCP center outside footprint (loose)
    l1a_key : str, optional
        MATLAB struct key for L1A data (default: "subimage")
    gcp_key : str, optional
        MATLAB struct key for GCP data (default: "GCP")
    gcp_pattern : str, optional
        File pattern for GCP discovery (default: "*_resampled.mat")

    Returns
    -------
    VerificationResult
        Structured results with pass/fail status, statistics, and warnings

    Raises
    ------
    ValueError
        If no valid GCP pairs found or no valid results

    Examples
    --------
    >>> from pathlib import Path
    >>> config = CorrectionConfig(
    ...     seed=42,
    ...     performance_threshold_m=250.0,
    ...     performance_spec_percent=39.0,
    ...     earth_radius_m=6378137.0,
    ...     geo=geo_config,
    ...     n_iterations=1,
    ...     parameters=[],
    ... )
    >>> l1a_files = [Path("/data/geo_001.mat"), Path("/data/geo_002.mat")]
    >>> result = run_verification(
    ...     config=config,
    ...     work_dir=Path("/tmp/work"),
    ...     l1a_files=l1a_files,
    ...     gcp_directory=Path("/data/gcps/"),
    ...     telemetry=telemetry_df,
    ...     calibration_dir=Path("/data/calibration"),
    ... )
    >>> print(f"Passed: {result.passed}")
    """
    logger.info("=== WEEKLY VERIFICATION ===")
    logger.info(f"  L1A files: {len(l1a_files)}")
    logger.info(f"  GCP directory: {gcp_directory}")
    logger.info(f"  Threshold: {config.performance_threshold_m}m")
    logger.info(f"  Required: {config.performance_spec_percent}%")
    logger.info(f"  Mode: Verification (NO geolocation, NO kernel creation)")

    # Step 1: GCP Pairing - Use REAL pair_files() from pairing.py
    logger.info("Step 1/3: GCP Pairing (using pair_files from pairing.py)")
    gcp_pairs = pair_files(
        l1a_files=l1a_files,
        gcp_directory=gcp_directory,
        max_distance_m=max_distance_m,
        l1a_key=l1a_key,
        gcp_key=gcp_key,
        gcp_pattern=gcp_pattern,
    )

    if not gcp_pairs:
        raise ValueError("No GCP pairs found. Cannot perform verification.")

    logger.info(f"  Found {len(gcp_pairs)} GCP pair(s)")

    # Step 2: Image Matching - Use REAL image_matching() from correction.py
    logger.info("Step 2/3: Image Matching (using image_matching from correction.py)")
    all_image_matching_results = []
    per_pair_metrics = []

    for pair_idx, (l1a_file, gcp_file) in enumerate(gcp_pairs):
        logger.info(f"  Processing pair {pair_idx + 1}/{len(gcp_pairs)}: {l1a_file.name}")

        try:
            # Load geolocated data from L1A file
            geolocated_data = load_image_grid_from_mat(l1a_file, key=l1a_key, as_named=False)

            # Convert ImageGrid to xr.Dataset for image_matching
            # The image_matching function expects an xr.Dataset with lat/lon
            geolocated_dataset = xr.Dataset(
                {
                    "lat": (["pixel"], geolocated_data.lat.flatten()),
                    "lon": (["pixel"], geolocated_data.lon.flatten()),
                    "data": (["pixel"], geolocated_data.data.flatten()),
                }
            )

            # Call REAL image_matching function from correction.py
            # This measures offset between already-geolocated data and GCP
            # NO geolocation is performed here
            image_match_result = image_matching(
                geolocated_data=geolocated_dataset,
                gcp_reference_file=gcp_file,
                telemetry=telemetry,
                calibration_dir=calibration_dir,
                params_info=[],  # No parameter sweep in verification
                config=config,
                los_vectors_cached=None,
                optical_psfs_cached=None,
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
                    "l1a_file": str(l1a_file),
                    "gcp_file": str(gcp_file),
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

    # Step 3: Error Statistics - Use REAL call_error_stats_module() from correction.py
    logger.info("Step 3/3: Calculating Error Statistics (using call_error_stats_module)")

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
    }
