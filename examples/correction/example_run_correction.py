#!/usr/bin/env python
"""
example_run_correction.py — Correction loop workflow template.

**This is a template, not a fully runnable demo.**

Use it as a starting point when setting up the correction loop for a new mission.
``example_verification.py`` in the same directory is the self-contained runnable demo.

Why is this a template?
-----------------------
The full correction loop regenerates SPICE kernels at every iteration, which
requires external tooling not bundled with the Python package:

  * SPICE kernel creation tools (``mkspk``, ``msopck``) on ``PATH`` or in
    ``bin/spice/<platform>/``
  * Generic SPICE data kernels in ``data/generic/``
    (see ``docs/source/users.md`` → "Data / Binary Files" for download links)
  * Preprocessed telemetry and science CSV files

When any of those are missing, this script detects the gap and exits cleanly
in **dry-run** mode, printing the API pattern without executing the loop.

Adapting for your mission
--------------------------
1. Copy ``examples/correction/example_config.json``, rename it, and fill in
   your mission's kernel paths and parameter specs.
2. Preprocess your raw telemetry into a single merged CSV.  Preprocessing
   is mission-specific; see your mission's ingestion scripts or the inline
   comments in :func:`_build_inputs` below for the expected file format.
3. Replace the ``_build_inputs()`` function below with your file paths.
4. Run::

       python examples/correction/example_run_correction.py
       python examples/correction/example_run_correction.py \\
           --config examples/correction/clarreo_config.json

Config values used here (n_iterations, parameter bounds, kernel paths, etc.)
are taken directly from ``examples/correction/clarreo_config.py``, which is
the canonical CLARREO config factory.

Run from the repo root::

    python examples/correction/example_run_correction.py

Requires ``lasp-curryer`` installed (``pip install -e ".[test,dev]"``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from curryer.correction import (
    CorrectionConfig,
    CorrectionInput,
    CorrectionResult,
    load_config_from_json,
    run_correction,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Repo-relative paths that mirror the CLARREO integration test fixtures.
_GCS_DIR = _REPO_ROOT / "tests" / "data" / "clarreo" / "gcs"
_IMAGE_MATCH_DIR = _REPO_ROOT / "tests" / "data" / "clarreo" / "image_match"
_GENERIC_DIR = _REPO_ROOT / "data" / "generic"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None) -> CorrectionConfig:
    """Load a :class:`CorrectionConfig` from JSON or via the CLARREO factory.

    Parameters
    ----------
    config_path : Path or None
        Explicit JSON config file.  When ``None``, the CLARREO config factory
        (``examples/correction/clarreo_config.py``) is used.

    Returns
    -------
    CorrectionConfig
    """
    if config_path is not None:
        print(f"  Loading config from: {config_path}")
        return load_config_from_json(config_path)

    # Use the CLARREO factory — see examples/correction/clarreo_config.py for the
    # full parameter list and inline comments explaining each field.
    print("  No --config supplied; using the CLARREO factory.")
    print("  See examples/correction/clarreo_config.py for the full parameter list.")

    # The sys.path manipulation is needed only when running this script directly;
    # installed packages (pip install -e .) can import as normal.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "clarreo_config",
        Path(__file__).parent / "clarreo_config.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.create_clarreo_config(data_dir=_GCS_DIR, generic_dir=_GENERIC_DIR)


# ---------------------------------------------------------------------------
# Input definition
# ---------------------------------------------------------------------------


def _build_inputs() -> list[CorrectionInput]:
    """Build the list of (telemetry, science, GCP) inputs for the loop.

    Each :class:`CorrectionInput` maps one preprocessed telemetry CSV and one
    science-timing CSV to one GCP reference image.  Add more entries to the
    list to incorporate additional overpasses / GCP sites.

    For CLARREO, the telemetry_file must be a single merged CSV that
    combines the four raw feeds (SC_SPK, SC_CK, ST_CK, AZEL_CK).  That
    merge step is mission-specific; implement it in your own preprocessing
    script (or reuse the logic in ``tests/test_correction/clarreo/``).
    The individual raw CSVs in ``tests/data/clarreo/gcs/`` are **not**
    directly suitable as telemetry_file — they must be merged first.

    **Replace these paths with your mission's preprocessed files.**

    Returns
    -------
    list[CorrectionInput]
    """
    return [
        CorrectionInput(
            # Merged/preprocessed telemetry CSV (SC_SPK, SC_CK, ST_CK, AZEL_CK merged).
            # Produce this file with your mission's preprocessing workflow.
            telemetry_file=_GCS_DIR / "clarreo_preprocessed_tlm.csv",
            # One row per science frame — contains 'corrected_timestamp' in GPS seconds.
            # This file is committed to the repo and can be used directly.
            science_file=_GCS_DIR / "openloop_tlm_5a_sci_times_20250521T225242.csv",
            # Ground-control-point reference image chip (.mat)
            gcp_file=_IMAGE_MATCH_DIR / "1" / "GCP12055Dili_resampled.mat",
        ),
        # Add more CorrectionInput entries here for additional GCP / overpass combos.
    ]


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run(config_path: Path | None = None, work_dir: Path | None = None) -> int:
    """Run the correction loop template.

    Loads config, defines inputs, and — when all data files are present —
    calls :func:`~curryer.correction.run_correction`.  Exits cleanly with a
    clear API-pattern summary when prerequisites are missing (dry-run).

    Parameters
    ----------
    config_path : Path or None
        JSON config file.  ``None`` → CLARREO factory is used.
    work_dir : Path or None
        Working directory for kernel files and output NetCDF.

    Returns
    -------
    int
        0 on success or dry-run, 1 on error.
    """
    work_dir = work_dir or Path("workdir_correction")
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  Correction Package — Full Correction Loop Template")
    print("=" * 68)

    # ------------------------------------------------------------------
    # Step 1: load / build config
    # ------------------------------------------------------------------
    print("\n[1/3] Loading configuration…")
    try:
        config = _load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load config: %s", exc)
        return 1

    print(f"  Iterations  : {config.n_iterations}")
    print(f"  Strategy    : {config.search_strategy.value}")
    print(f"  Parameters  : {len(config.parameters)}")
    print(f"  Threshold   : {config.performance_threshold_m} m")
    print(f"  Spec        : {config.performance_spec_percent}%")

    # To override search strategy or iteration count at runtime:
    #   from curryer.correction import SearchStrategy
    #   config = config.model_copy(update={
    #       "search_strategy": SearchStrategy.GRID_SEARCH,
    #       "n_iterations": 100,
    #   })

    # ------------------------------------------------------------------
    # Step 2: define inputs
    # ------------------------------------------------------------------
    print("\n[2/3] Defining input file sets…")
    inputs = _build_inputs()
    for i, inp in enumerate(inputs):
        print(f"  Input {i + 1}:  tlm={inp.telemetry_file.name}  sci={inp.science_file.name}  gcp={inp.gcp_file.name}")

    # Detect missing files / external prerequisites before attempting a real run.
    missing_files = [
        f for inp in inputs for f in [inp.telemetry_file, inp.science_file, inp.gcp_file] if not f.exists()
    ]
    missing_generic = not _GENERIC_DIR.exists() or not any(_GENERIC_DIR.iterdir())

    if missing_files or missing_generic:
        _print_dry_run_summary(missing_files, missing_generic)
        return 0

    # ------------------------------------------------------------------
    # Step 3: run the correction loop
    # ------------------------------------------------------------------
    print(f"\n[3/3] Running run_correction()…  (work_dir={work_dir})")
    result: CorrectionResult = run_correction(config, work_dir, inputs)

    if not result.results:
        logger.error("No results produced — check logs for errors.")
        return 1

    top_results = sorted(result.results, key=lambda r: r["rms_error_m"])[:3]
    print(f"\n  Parameter sets evaluated : {len(result.results)}")
    print(f"  Best RMS error           : {result.best_rms_m:.2f} m")

    print("\n  Top-3 by RMS error:")
    for r in top_results:
        print(f"    {r}")

    if result.netcdf_path is not None:
        print(f"\n  Results saved: {result.netcdf_path}")

    print("\n" + "=" * 68)
    print("  Correction loop complete.")
    print("  Next: run example_verification.py with the best parameter set.")
    print("=" * 68)
    return 0


def _print_dry_run_summary(missing_files: list[Path], missing_generic: bool) -> None:
    """Print a helpful dry-run message listing what prerequisites are absent."""
    print("\n  ── DRY-RUN: prerequisites not satisfied ──")
    if missing_files:
        print("\n  Missing data files:")
        for f in missing_files:
            print(f"    {f}")
        print("\n  Generate the preprocessed telemetry first.")
        print("    Preprocessing is mission-specific; use your mission's ingestion")
        print("    scripts or the _build_inputs() comments in this template.")
        print("    # Then point telemetry_file at the merged output CSV.")
    if missing_generic:
        print(f"\n  Missing / empty generic SPICE kernel dir: {_GENERIC_DIR}")
        print("  Download: see docs/source/users.md → 'Data / Binary Files'.")

    print("\n  The loop also requires SPICE kernel creation tools on PATH:")
    print("    mkspk, msopck  (https://naif.jpl.nasa.gov/naif/utilities.html)")

    print("\n  API pattern (runs when all prerequisites are present):")
    print()
    print("    from curryer.correction import run_correction, CorrectionInput, CorrectionResult")
    print()
    print("    result: CorrectionResult = run_correction(config, work_dir, inputs)")
    print()
    print("    # Inspect the best parameter set")
    print("    best = min(result.results, key=lambda r: r.rms_error_m)")
    print("    print(f'Best RMS: {best.rms_error_m:.2f} m')")
    print()
    print("    # Save full sweep results as NetCDF")
    print("    result.netcdf_data.to_netcdf('correction_results.nc')")
    print()
    print("  See also:")
    print("    examples/correction/example_verification.py  (fully runnable demo)")
    print("    docs/source/correction_user_guide.md          (full reference)")
    print()
    print("Template completed (dry-run — prerequisites not found).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correction loop workflow template (dry-run when data is missing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/correction/example_run_correction.py\n"
            "  python examples/correction/example_run_correction.py \\\n"
            "      --config examples/correction/clarreo_config.json"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON correction config (see examples/correction/clarreo_config.json).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("workdir_correction"),
        help="Working directory for kernel files and results (default: workdir_correction).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(config_path=args.config, work_dir=args.work_dir))
