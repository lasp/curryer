#!/usr/bin/env python
"""
Example: Weekly verification on geolocated observations.

This script demonstrates the production verification workflow:

1. Geolocation pipeline runs (produces geolocated observations)
2. Load previously-geolocated science data
3. Run full verification (image matching against ground truth → error stats → threshold check)
4. Report compliance results

Note: The geolocation itself is NOT done by the verification module.
The verification module runs image matching and error analysis on
output from your geolocation pipeline to check compliance.

Setup
-----
Before running, ensure you have:
- A CorrectionConfig with:
  - Proper kernel paths (SPICE kernels for geolocation)
  - Mission parameters (spacecraft, instrument names, etc.)
  - An image_matching_func (required - compares geolocated vs. ground truth)
  - Performance thresholds (performance_threshold_m, performance_spec_percent)
- Geolocated science data (output of your geolocation pipeline)

Example Usage
-------------
As a library::

    from scripts.example_weekly_verification import run_weekly_verification
    from pathlib import Path
    import xarray as xr

    # Load this week's geolocated observations (from your geolocation pipeline)
    geolocated = xr.open_dataset("weekly_2024-03-17.nc")

    # Run full verification (image matching + error stats + threshold check)
    report = run_weekly_verification(
        config=CorrectionConfig.from_json(Path("config/my_mission.json")),
        geolocated_data=geolocated,
        output_dir=Path("reports/2024-03-17/")
    )

    # Check result
    if not report.passed:
        send_alert(report.summary_table)

As a CLI::

    python scripts/example_weekly_verification.py \\
        --config config/my_mission.json \\
        --geolocated weekly_2024-03-17.nc \\
        --output-dir reports/2024-03-17/

    # Or with multiple files (concatenated):
    python scripts/example_weekly_verification.py \\
        --config config/my_mission.json \\
        --geolocated-dir weekly_data/ \\
        --pattern "*.nc" \\
        --output-dir reports/2024-03-17/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import xarray as xr

from curryer.correction.config import CorrectionConfig
from curryer.correction.verification import VerificationResult, verify

logger = logging.getLogger(__name__)


# ============================================================================
# Typical workflow: Load and verify pre-computed image-matching results
# ============================================================================


def run_weekly_verification(
    config: CorrectionConfig,
    geolocated_data: xr.Dataset | None = None,
    geolocated_files: list[Path] | None = None,
    output_dir: Path | None = None,
    save_report: bool = True,
) -> VerificationResult:
    """
    Run full verification on geolocated observations.

    This is the production workflow: load geolocated science data from your
    geolocation pipeline, run full verification (image matching → error stats →
    threshold check), and report compliance.

    The verification module does NOT perform geolocation. It verifies the
    quality of geolocation output by comparing geolocated measurements against
    ground truth using your image_matching_func.

    Parameters
    ----------
    config : CorrectionConfig
        Mission/instrument configuration with:
        - Proper SPICE kernels for geolocation
        - image_matching_func (required - compares geolocated vs. ground truth)
        - performance_threshold_m and performance_spec_percent
    geolocated_data : xr.Dataset, optional
        Pre-loaded geolocated science data (output of your geolocation pipeline).
        This is the recommended entry point when you have a single NetCDF file.
    geolocated_files : list[Path], optional
        List of NetCDF files to load and concatenate. Use when data spans
        multiple files. Ignored if geolocated_data is provided.
    output_dir : Path, optional
        Directory for saving the verification report. Created if missing.
        Defaults to ./verification_output.
    save_report : bool, default=True
        Whether to save JSON report and summary table to output_dir.

    Returns
    -------
    VerificationResult
        Structured result with pass/fail status, per-GCP errors, and
        human-readable summary table.

    Raises
    ------
    ValueError
        If neither geolocated_data nor geolocated_files is provided, or if
        config.image_matching_func is not set.
    """
    if output_dir is None:
        output_dir = Path.cwd() / "verification_output"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Verification: Running on Geolocated Observations")
    logger.info("=" * 70)

    # ----
    # Load geolocated data
    # ----
    if geolocated_data is None:
        if geolocated_files is None or not geolocated_files:
            raise ValueError(
                "Either geolocated_data or geolocated_files must be provided. "
                "Supply NetCDF file(s) with geolocated science observations."
            )
        logger.info(f"Loading geolocated data from {len(geolocated_files)} file(s)")
        datasets = [xr.open_dataset(f) for f in geolocated_files]
        # Concatenate along measurement dimension if multiple files
        if len(datasets) == 1:
            geolocated_data = datasets[0]
        else:
            geolocated_data = xr.concat(datasets, dim="measurement")
        logger.info(f"Combined {len(datasets)} file(s): total shape {geolocated_data.dims}")

    # ----
    # Run verification
    # ----
    result = verify(
        config=config,
        geolocated_data=geolocated_data,
        work_dir=output_dir if output_dir else None,  # Optional, defaults to ./verification_output
    )

    # ----
    # Print summary
    # ----
    print("\n" + result.summary_table)

    if result.warnings:
        print("\n⚠️  Warnings:")
        for w in result.warnings:
            print(f"  {w}")

    # ----
    # Save report if requested
    # ----
    if save_report:
        _save_verification_report(result, output_dir)

    return result


def _save_verification_report(result: VerificationResult, output_dir: Path) -> None:
    """Save verification result to JSON file and summary table text file."""
    # Save JSON report (excluding xarray Dataset which is not JSON-serializable)
    json_file = output_dir / "verification_report.json"
    dumped = result.model_dump(exclude={"aggregate_stats"})
    with open(json_file, "w") as f:
        json.dump(dumped, f, indent=2, default=str)  # default=str for datetime
    logger.info(f"Saved JSON report: {json_file}")

    # Save summary table
    table_file = output_dir / "verification_summary.txt"
    with open(table_file, "w") as f:
        f.write(result.summary_table)
    logger.info(f"Saved summary table: {table_file}")

    # Save xarray aggregate stats if available (useful for post-processing)
    if result.aggregate_stats is not None:
        stats_file = output_dir / "aggregate_stats.nc"
        result.aggregate_stats.to_netcdf(stats_file)
        logger.info(f"Saved aggregate statistics: {stats_file}")


# ============================================================================
# CLI entry point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Verification on geolocated observations (mission-agnostic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to CorrectionConfig JSON file.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--geolocated",
        type=Path,
        help="NetCDF file with geolocated science data from this week.",
    )
    input_group.add_argument(
        "--geolocated-dir",
        type=Path,
        help="Directory containing geolocated NetCDF files to concatenate.",
    )

    parser.add_argument(
        "--pattern",
        default="*.nc",
        help="Glob pattern for matching geolocated files (default: *.nc).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for verification report (default: ./verification_output).",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save JSON report and summary table.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config_dict = json.load(f)
    config = CorrectionConfig(**config_dict)

    # Check that image_matching_func is set
    if config.image_matching_func is None:
        logger.error(
            "config.image_matching_func is not set. "
            "Verification requires an image-matching function to run on geolocated data."
        )
        return 1

    # Load geolocated data
    if args.geolocated:
        logger.info(f"Loading geolocated data from {args.geolocated}")
        geolocated = xr.open_dataset(args.geolocated)

        # Run verification
        result = run_weekly_verification(
            config=config,
            geolocated_data=geolocated,
            output_dir=args.output_dir,
            save_report=not args.no_save,
        )

    else:  # args.geolocated_dir
        logger.info(f"Loading geolocated files from {args.geolocated_dir}")
        geolocated_dir = Path(args.geolocated_dir)
        geolocated_files = sorted(geolocated_dir.glob(args.pattern))
        logger.info(f"Found {len(geolocated_files)} matching files")

        if not geolocated_files:
            logger.error(f"No files matching pattern '{args.pattern}' in {args.geolocated_dir}")
            return 1

        # Run verification
        result = run_weekly_verification(
            config=config,
            geolocated_files=geolocated_files,
            output_dir=args.output_dir,
            save_report=not args.no_save,
        )

    # Exit with appropriate code
    return 0 if result.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
