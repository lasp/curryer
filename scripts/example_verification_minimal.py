#!/usr/bin/env python
"""
Minimal example: Run full verification on geolocated observations.

This example demonstrates the production verification workflow:
1. Load geolocated science data (output of geolocation pipeline)
2. Run image matching against ground truth
3. Compute error statistics
4. Check against geolocation requirements

This example uses synthetic data for demonstration. In production:
- Load real geolocated data from your pipeline
- Use actual ground truth GCP locations
- Run with real kernels and parameters

Use this to understand the verification API before deploying to production.
"""

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from curryer.correction.config import (
    CorrectionConfig,
    GeolocationConfig,
    ParameterConfig,
    ParameterType,
)
from curryer.correction.verification import verify


def create_synthetic_geolocated_data(n_measurements: int = 50, seed: int = 42) -> xr.Dataset:
    """Create a synthetic geolocated science dataset.

    In production, this would be the output from your geolocation pipeline.
    It should contain:
    - Geolocated pixel observations (latitude, longitude, altitude)
    - Spacecraft state data
    - Timing information

    Parameters
    ----------
    n_measurements : int
        Number of pixel measurements.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Geolocated science data with spacecraft state.
    """
    rng = np.random.RandomState(seed)

    # Simulated geolocated pixel observations
    # (What your geolocation algorithm produced using current kernels/parameters)
    geolocated_lat = rng.uniform(-90, 90, n_measurements)
    geolocated_lon = rng.uniform(-180, 180, n_measurements)
    geolocated_alt = rng.uniform(0, 5000, n_measurements)

    # Ground truth reference GCP locations
    # (Fixed reference points, never change)
    gcp_lat = rng.uniform(-90, 90, n_measurements)
    gcp_lon = rng.uniform(-180, 180, n_measurements)
    gcp_alt = rng.uniform(0, 5000, n_measurements)

    # Spacecraft state (from SPICE kernels)
    spacecraft_pos = rng.normal(6.371e6, 1e5, (n_measurements, 3))  # ECEF meters
    boresight = rng.normal(0, 1, (n_measurements, 3))
    boresight = boresight / np.linalg.norm(boresight, axis=1, keepdims=True)
    rotation_matrices = rng.normal(0, 0.1, (n_measurements, 3, 3))

    # Sensor/timing info
    band_data = rng.uniform(0, 65535, n_measurements).astype(np.uint16)

    ds = xr.Dataset(
        {
            # Geolocated measurements (from geolocation pipeline)
            "latitude": (["measurement"], geolocated_lat),
            "longitude": (["measurement"], geolocated_lon),
            "altitude": (["measurement"], geolocated_alt),
            # Ground truth for image matching
            "gcp_lat_deg": (["measurement"], gcp_lat),
            "gcp_lon_deg": (["measurement"], gcp_lon),
            "gcp_alt": (["measurement"], gcp_alt),
            # Spacecraft state (from SPICE kernels)
            "riss_ctrs": (["measurement", "component"], spacecraft_pos),
            "bhat_hs": (["measurement", "component"], boresight),
            "t_hs2ctrs": (["measurement", "matrix_row", "matrix_col"], rotation_matrices),
            # Sensor data
            "radiance": (["measurement"], band_data),
        },
        coords={"measurement": np.arange(n_measurements)},
    )

    ds.attrs["mission"] = "CLARREO"
    ds.attrs["date"] = "2024-03-17"

    return ds


def create_minimal_config() -> CorrectionConfig:
    """Create a minimal CorrectionConfig with a stub image_matching_func."""

    def stub_image_matching(data: xr.Dataset) -> xr.Dataset:
        """Stub image matching function.

        In production, this would run sophisticated image matching between
        geolocated observations and reference imagery.

        For this example, we just compute simple lat/lon differences.
        """
        n = len(data.measurement)

        # Compute errors (difference between geolocated and ground truth)
        lat_err = data["latitude"].values - data["gcp_lat_deg"].values
        lon_err = data["longitude"].values - data["gcp_lon_deg"].values

        result = xr.Dataset(
            {
                "lat_error_deg": (["measurement"], lat_err),
                "lon_error_deg": (["measurement"], lon_err),
                # Include spacecraft state for error_stats processor
                "riss_ctrs": (["measurement", "component"], data["riss_ctrs"].values),
                "bhat_hs": (["measurement", "component"], data["bhat_hs"].values),
                "t_hs2ctrs": (["measurement", "matrix_row", "matrix_col"], data["t_hs2ctrs"].values),
                "gcp_lat_deg": (["measurement"], data["gcp_lat_deg"].values),
                "gcp_lon_deg": (["measurement"], data["gcp_lon_deg"].values),
                "gcp_alt": (["measurement"], data["gcp_alt"].values),
            },
            coords={"measurement": np.arange(n)},
        )
        return result

    return CorrectionConfig(
        n_iterations=1,
        parameters=[
            ParameterConfig(
                ptype=ParameterType.CONSTANT_KERNEL,
                data={"current_value": [0.0, 0.0, 0.0], "bounds": [-300.0, 300.0]},
            )
        ],
        geo=GeolocationConfig(
            meta_kernel_file=Path("test.tm.json"),
            generic_kernel_dir=Path("data/generic"),
            instrument_name="TEST_INSTRUMENT",
            time_field="corrected_timestamp",
        ),
        performance_threshold_m=250.0,
        performance_spec_percent=39.0,
        earth_radius_m=6_378_140.0,
        spacecraft_position_name="riss_ctrs",
        boresight_name="bhat_hs",
        transformation_matrix_name="t_hs2ctrs",
        # This is required for the geolocated_data workflow
        image_matching_func=stub_image_matching,
    )


def main():
    """Run the production verification workflow."""
    print("=" * 70)
    print("CLARREO Weekly Verification (Production Workflow)")
    print("=" * 70)

    # Step 1: Load geolocated observations
    print("\n1. Loading geolocated observations...")
    geolocated = create_synthetic_geolocated_data(n_measurements=50, seed=42)
    print(f"   Loaded {len(geolocated.measurement)} measurements")
    print(f"   Variables: {list(geolocated.data_vars)}")

    # Step 2: Create config with image matching function
    print("\n2. Setting up verification config...")
    config = create_minimal_config()
    print(f"   image_matching_func: {config.image_matching_func.__name__}")
    print(f"   Threshold: {config.performance_threshold_m}m")
    print(f"   Spec: {config.performance_spec_percent}%")

    # Step 3: Run full verification
    print("\n3. Running verification...")
    print("   - Running image matching on geolocated data")
    print("   - Computing error statistics")
    print("   - Checking against requirements")
    print()

    # Use secure temporary directory (or omit work_dir to use default ./verification_output)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "verification_example"
        output_dir.mkdir(exist_ok=True)

        result = verify(
            config=config,
            geolocated_data=geolocated,
            work_dir=output_dir,  # Optional - defaults to ./verification_output if not provided
        )

        # Display results
        print("\n4. Verification Summary")
        print(result.summary_table)

        print("\n5. Result Details")
        print(f"   Status: {'PASSED ✓' if result.passed else 'FAILED ✗'}")
        print(f"   Percent within threshold: {result.percent_within_threshold:.1f}%")
        print(f"   Requirement: {result.requirements.performance_spec_percent}%")
        print(f"   Threshold: {result.requirements.performance_threshold_m}m")
        print(f"   Measurements analyzed: {len(result.per_gcp_errors)}")

        if result.warnings:
            print("\n6. Warnings")
            for warning in result.warnings:
                print(f"   ⚠️  {warning}")

    # Show per-measurement sample
    print("\n7. Sample Per-Measurement Errors (first 5)")
    for err in result.per_gcp_errors[:5]:
        status = "✓ PASS" if err.passed else "✗ FAIL"
        print(
            f"   #{err.gcp_index}: lat={err.lat_error_deg:+.5f}°, "
            f"lon={err.lon_error_deg:+.5f}°, nadir={err.nadir_equiv_error_m:.1f}m  {status}"
        )

    # Show output files
    print("\n8. Output Files")
    print(f"   Work directory: {output_dir}")
    if output_dir.exists():
        files = sorted(output_dir.glob("*"))
        if files:
            print("   Saved:")
            for f in files:
                if f.is_file():
                    size_mb = f.stat().st_size / (1024**2)
                    print(f"     - {f.name} ({size_mb:.2f} MB)")
        else:
            print("   (No files saved - use --save with CLI)")

    print("\n" + "=" * 70)
    print(f"Verification {'PASSED ✓' if result.passed else 'FAILED ✗'}")
    print("=" * 70)

    return 0 if result.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
