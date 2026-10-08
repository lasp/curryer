#!/usr/bin/env python
"""
example_verification.py — End-to-end verification using the public API.

Demonstrates the recommended workflow:
  load image-matching results → call verify() → inspect VerificationResult
  → save it → review GCPs → recompute the statistics with the review applied

**Primary path:** uses CLARREO test image-match data from
``tests/data/clarreo/image_match/`` to run real image matching and produce
a realistic ``xr.Dataset``.

**Fallback:** if the test files are not present (e.g. fresh clone without
the large data files), synthetic matching results are used so the script
still runs end-to-end and demonstrates the full API surface.

In production, replace the data-loading block with the output of your own
image matching pipeline or ``run_image_matching()``.

Run from the repo root::

    python examples/correction/example_verification.py

Requires ``lasp-curryer`` installed (``pip install -e ".[test,dev]"``).
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from curryer.correction import (
    GeolocationConfig,
    GeolocationSetup,
    RequirementsConfig,
    VerificationResult,
    apply_review,
    compare_results,
    read_review_decisions,
    save_verification,
    verify,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo-root detection (scripts may be run from any working directory)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMAGE_MATCH_DIR = _REPO_ROOT / "tests" / "data" / "clarreo" / "image_match"


# ---------------------------------------------------------------------------
# Real-data path: load CLARREO test image-match result (case 1a)
# ---------------------------------------------------------------------------


def _load_clarreo_matching_result() -> xr.Dataset | None:
    """Run image matching on CLARREO test case 1a and return an xr.Dataset.

    Requires the ``.mat`` data files under ``tests/data/clarreo/image_match/``.
    Returns ``None`` when the files are absent.

    Returns
    -------
    xr.Dataset or None
        Image-matching result dataset with ``lat_error_deg``, ``lon_error_deg``,
        and the CLARREO spacecraft-state variables (``riss_ctrs``, ``bhat_hs``,
        ``t_hs2ctrs``), or ``None`` when data files are not found.
    """
    subimage_file = _IMAGE_MATCH_DIR / "1" / "TestCase1a_subimage.mat"
    gcp_file = _IMAGE_MATCH_DIR / "1" / "GCP12055Dili_resampled.mat"
    ancil_file = _IMAGE_MATCH_DIR / "1" / "R_ISS_midframe_TestCase1.mat"
    los_file = _IMAGE_MATCH_DIR / "b_HS.mat"
    psf_file = _IMAGE_MATCH_DIR / "optical_PSF_675nm_upsampled.mat"

    required = [subimage_file, gcp_file, ancil_file, los_file, psf_file]
    if not all(p.exists() for p in required):
        logger.warning("CLARREO test image-match files not found; falling back to synthetic data.")
        return None

    try:
        from scipy.io import loadmat

        from curryer.correction.config import PSFSamplingConfig, SearchConfig
        from curryer.correction.grid_types import ImageGrid
        from curryer.correction.image_io import load_image_grid, load_los_vectors, load_optical_psf
        from curryer.correction.image_match import integrated_image_match
    except ImportError as exc:
        logger.warning("Missing dependency for image matching (%s); using synthetic data.", exc)
        return None

    print("Loading CLARREO test case 1a (Dili) image-match data...")

    # Load subimage (instrument observation projected to lat/lon grid)
    subimage = load_image_grid(subimage_file, mat_key="subimage")

    # Load GCP reference image and ancillary spacecraft state
    gcp = load_image_grid(gcp_file, mat_key="GCP")
    mid_i, mid_j = gcp.mid_indices
    gcp_center_lat = float(gcp.lat[mid_i, mid_j])
    gcp_center_lon = float(gcp.lon[mid_i, mid_j])

    ancil = loadmat(str(ancil_file), squeeze_me=True)
    r_iss_midframe = ancil["R_ISS_midframe"].ravel()

    # Apply the known test-case geolocation error so image matching has
    # something to find (expected: ~3km lat, ~-3km lon for case 1).
    from curryer.compute import constants as _c

    lat_error_km, lon_error_km = 3.0, -3.0
    earth_radius_km = _c.WGS84_SEMI_MAJOR_AXIS_KM
    lat_offset_deg = lat_error_km / earth_radius_km * (180.0 / np.pi)
    lon_offset_deg = lon_error_km / (earth_radius_km * np.cos(np.deg2rad(gcp_center_lat))) * (180.0 / np.pi)
    subimage_shifted = ImageGrid(
        data=subimage.data.copy(),
        lat=subimage.lat + lat_offset_deg,
        lon=subimage.lon + lon_offset_deg,
        h=subimage.h.copy() if subimage.h is not None else None,
    )

    # Run image matching
    print("  Running integrated_image_match … (this may take 15–30 s)")
    los_vectors = load_los_vectors(los_file)
    optical_psfs = load_optical_psf(psf_file)

    result = integrated_image_match(
        subimage=subimage_shifted,
        gcp=gcp,
        r_iss_midframe_m=r_iss_midframe,
        los_vectors_hs=los_vectors,
        optical_psfs=optical_psfs,
        geolocation_config=PSFSamplingConfig(),
        search_config=SearchConfig(),
    )

    # Convert km-errors to degrees
    meas_lat_error_deg = np.rad2deg(result.lat_error_km / earth_radius_km)
    meas_lon_error_deg = np.rad2deg(result.lon_error_km / (earth_radius_km * np.cos(np.deg2rad(gcp_center_lat))))

    # Fixed (nominal) spacecraft state from the ancil file
    boresight = np.array([0.0, 0.0625969755450201, 0.99803888634292])
    t_matrix = np.array(
        [
            [-0.418977524967338, 0.748005379751721, 0.514728846515064],
            [-0.421890284446342, 0.341604851993858, -0.839830169131854],
            [-0.804031356019172, -0.569029065124742, 0.172451447025628],
        ]
    )

    ds = xr.Dataset(
        {
            "lat_error_deg": (["measurement"], [meas_lat_error_deg]),
            "lon_error_deg": (["measurement"], [meas_lon_error_deg]),
            # Match quality: the quality gates and weights use it
            "correlation": (["measurement"], [result.ccv_final]),
            # Spacecraft-state variables (required by ErrorStatsProcessor)
            "riss_ctrs": (["measurement", "xyz"], [r_iss_midframe]),
            "bhat_hs": (["measurement", "xyz"], [boresight]),
            "t_hs2ctrs": (["measurement", "xyz_from", "xyz_to"], t_matrix[np.newaxis, :, :]),
            # GCP reference location
            "gcp_lat_deg": (["measurement"], [gcp_center_lat]),
            "gcp_lon_deg": (["measurement"], [gcp_center_lon]),
            "gcp_alt": (["measurement"], [0.0]),
        },
        coords={
            "measurement": [0],
            "xyz": ["x", "y", "z"],
            "xyz_from": ["x", "y", "z"],
            "xyz_to": ["x", "y", "z"],
        },
        attrs={
            "sci_key": "TestCase1a_subimage",
            "gcp_key": "GCP12055Dili",
            "lat_error_km": result.lat_error_km,
            "lon_error_km": result.lon_error_km,
            "correlation_ccv": result.ccv_final,
        },
    )

    print(
        f"  Image matching complete: lat_err={result.lat_error_km:.2f} km, "
        f"lon_err={result.lon_error_km:.2f} km, ccv={result.ccv_final:.3f}"
    )
    return ds


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------


def _make_synthetic_matching_result(n: int = 50, seed: int = 42) -> xr.Dataset:
    """Create a synthetic image-matching result dataset.

    Used when test ``.mat`` files are not available. The structure
    exactly mirrors the output of a real image-matching run so that
    ``verify()`` processes it identically.

    Parameters
    ----------
    n : int
        Number of synthetic measurements.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Synthetic dataset with ``lat_error_deg``, ``lon_error_deg``, and the
        spacecraft-state variables expected by ``ErrorStatsProcessor``.
    """
    from curryer.compute.spatial import geodetic_to_ecef

    rng = np.random.RandomState(seed)

    # GCPs anywhere between 60S and 60N, each viewed from ~400 km directly above
    gcp_lat = rng.uniform(-60, 60, n)
    gcp_lon = rng.uniform(-180, 180, n)
    gcp_alt = rng.uniform(0, 2000, n)
    positions = geodetic_to_ecef(np.column_stack([gcp_lon, gcp_lat, np.full(n, 4.0e5)]), meters=True, degrees=True)

    # Instrument frame: +z towards the Earth centre; columns are the instrument
    # axes in CTRS, so t_hs2ctrs @ [0, 0, 1] points at nadir
    z_axis = -positions / np.linalg.norm(positions, axis=1, keepdims=True)
    x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    t_matrices = np.stack([x_axis, y_axis, z_axis], axis=2)

    # Boresights within ~1 degree of the instrument +z axis
    boresights = rng.normal([0, 0, 1], [0.01, 0.01, 0.0], (n, 3))
    boresights = boresights / np.linalg.norm(boresights, axis=1, keepdims=True)

    return xr.Dataset(
        {
            # Geolocation errors ~0.002° ≈ 222 m — straddles the 250 m threshold
            "lat_error_deg": (["measurement"], rng.normal(0.0, 0.002, n)),
            "lon_error_deg": (["measurement"], rng.normal(0.0, 0.002, n)),
            # Match quality: below setup.geo.minimum_correlation a GCP is rejected
            "correlation": (["measurement"], rng.uniform(0.5, 0.99, n)),
            # Spacecraft state — variable names must match the setup's name fields
            "riss_ctrs": (["measurement", "xyz"], positions),
            "bhat_hs": (["measurement", "xyz"], boresights),
            "t_hs2ctrs": (["measurement", "r", "c"], t_matrices),
            # GCP reference location
            "gcp_lat_deg": (["measurement"], gcp_lat),
            "gcp_lon_deg": (["measurement"], gcp_lon),
            "gcp_alt": (["measurement"], gcp_alt),
        },
        coords={"measurement": np.arange(n)},
        attrs={"sci_key": "synthetic_001", "gcp_key": "synthetic_gcp_001"},
    )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _make_setup() -> GeolocationSetup:
    """Build a ``GeolocationSetup`` for the verification example.

    The ``geo`` config uses placeholder kernel paths — they are NOT accessed
    when ``verify()`` is called with ``image_matching_results=``.  What
    matters for error-stats computation are the three variable-name fields
    (``spacecraft_position_name``, ``boresight_name``,
    ``transformation_matrix_name``), which must match the variable names
    in the ``xr.Dataset``.

    Returns
    -------
    GeolocationSetup
    """
    return GeolocationSetup(
        geo=GeolocationConfig(
            # Placeholder paths: NOT loaded when image_matching_results= is used.
            # Replace with real paths when running the full correction loop.
            meta_kernel_file=Path("tests/data/clarreo/gcs/cprs_v01.kernels.tm.json"),
            generic_kernel_dir=Path("data/generic"),
            instrument_name="CPRS_HYSICS",
            time_field="corrected_timestamp",
            # Quality gate: matches below this correlation are kept but rejected
            minimum_correlation=0.8,
        ),
        # CLARREO mission requirements
        requirements=RequirementsConfig(
            performance_threshold_m=250.0,  # Each measurement must be < 250 m nadir-equivalent error
            performance_spec_percent=39.0,  # At least 39% of measurements must pass
        ),
        # Variable names in the xr.Dataset — must match what image matching produced
        spacecraft_position_name="riss_ctrs",
        boresight_name="bhat_hs",
        transformation_matrix_name="t_hs2ctrs",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run end-to-end verification and print the result summary.

    Returns
    -------
    int
        0 if verification passed, 1 if failed.
    """
    print("=" * 68)
    print("  Correction Package — Verification Example")
    print("=" * 68)

    # ------------------------------------------------------------------
    # Step 1: Obtain image-matching results
    # ------------------------------------------------------------------
    # Priority: real CLARREO test data → synthetic fallback.
    # In production, replace this block with your pipeline's output from
    # run_image_matching() or a pre-computed NetCDF file.

    print("\n[1/4] Loading image-matching results…")
    clarreo_ds = _load_clarreo_matching_result()

    if clarreo_ds is not None:
        image_matching_results = [clarreo_ds]
        data_source = "CLARREO test case 1a (Dili)"
    else:
        print("  Using synthetic image-matching data (real data unavailable).")
        image_matching_results = [_make_synthetic_matching_result(n=50)]
        data_source = "synthetic (n=50)"

    print(f"  Data source  : {data_source}")
    print(f"  GCP pairs    : {len(image_matching_results)}")

    # ------------------------------------------------------------------
    # Step 2: Build config
    # ------------------------------------------------------------------
    print("\n[2/4] Building configuration…")
    setup = _make_setup()

    # Static instrument calibration and a custom image matcher are now attached
    # directly to the setup via dedicated fields:
    #
    #   from curryer.correction import CalibrationFiles
    #   setup.calibration = CalibrationFiles(
    #       los_vectors_file="tests/data/clarreo/image_match/b_HS.mat",
    #       psf_file="tests/data/clarreo/image_match/optical_PSF_675nm_upsampled.mat",
    #   )
    #   setup.image_matching_func = my_matcher  # signature: (geolocated_data, gcp_reference_file, ...)
    #
    # Not needed here because verify() is called with image_matching_results=.
    print(f"  Threshold    : {setup.requirements.performance_threshold_m} m")
    print(f"  Spec         : {setup.requirements.performance_spec_percent}%")
    print(f"  Instrument   : {setup.geo.instrument_name}")

    # ------------------------------------------------------------------
    # Step 3: Run verification (the public API call)
    # ------------------------------------------------------------------
    print("\n[3/4] Running verify()…")
    result: VerificationResult = verify(setup, image_matching_results=image_matching_results)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print("\n" + result.summary_table)

    print(f"\nOverall result : {'PASSED ✓' if result.passed else 'FAILED ✗'}")
    print(
        f"Within {setup.requirements.performance_threshold_m:.0f} m : {result.percent_within_threshold:.1f}%"
        f" (required: ≥ {setup.requirements.performance_spec_percent:.0f}%)"
    )
    print(f"Elapsed        : {result.elapsed_time_s:.2f} s" if result.elapsed_time_s else "")

    if result.weighted_percent_within_threshold is not None:
        print(f"Quality-weighted : {result.weighted_percent_within_threshold:.1f}% (reported, not used for pass/fail)")

    # Per-measurement detail (first 5): rejected GCPs stay in the result with a reason
    if result.per_gcp_errors:
        print("\nPer-measurement detail (first 5):")
        for err in result.per_gcp_errors[:5]:
            nadir = f"{err.nadir_equiv_error_m:.1f} m" if err.nadir_equiv_error_m is not None else "N/A"
            print(
                f"  #{err.gcp_index:>3}  lat={err.lat_error_deg:+.5f}°  "
                f"lon={err.lon_error_deg:+.5f}°  nadir={nadir}  {err.status.upper()}  {err.rejection_reason or ''}"
            )

    # ------------------------------------------------------------------
    # Step 4: Save, review, recompute
    # ------------------------------------------------------------------
    # save_verification writes summary.csv with an empty "review" column.  A
    # reviewer fills it with accept / reject (in a spreadsheet, MATLAB, or
    # code, as here); apply_review recomputes the statistics without
    # re-matching.
    print("\n[4/4] Saving and applying a review…")
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "verification"
        save_verification(result, saved)
        summary = saved / "summary.csv"
        table = pd.read_csv(summary, dtype={"review": str}, keep_default_na=False)
        decision = "accept" if table.loc[0, "status"] == "rejected" else "reject"
        table.loc[0, "review"] = decision
        table.to_csv(summary, index=False)
        print(f"  Reviewer marks GCP {table.loc[0, 'gcp_index']} ({table.loc[0, 'status']}) as {decision!r}")

        reviewed = apply_review(result, read_review_decisions(summary))
    print("\n" + reviewed.summary_table)

    # Demonstrate compare_results() with a "before" baseline
    # (here we reuse the same result as the baseline for illustration)
    print("\n--- compare_results() demo (before vs. after, same data) ---")
    print(compare_results(result, result))

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  {w}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
