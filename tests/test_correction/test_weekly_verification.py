"""
Integration tests for weekly verification workflow.

This module provides end-to-end integration tests using CLARREO mission data
as a reference implementation. It demonstrates the complete verification workflow
from configuration to results analysis.

IMPORTANT: Verification does NOT create SPICE kernels or perform geolocation.
It only measures accuracy of already-geolocated data by comparing to GCPs.

Running Tests:
-------------
# Via pytest (recommended)
pytest tests/test_correction/test_weekly_verification.py -v

# Run specific test
pytest tests/test_correction/test_weekly_verification.py::TestWeeklyVerificationCLARREO::test_verification_mode_detection -v

# Skip slow integration tests
pytest tests/test_correction/test_weekly_verification.py -v -m "not slow"

Requirements:
-----------------
These tests demonstrate the verification workflow using test data patterns
from test_pairing.py and test_image_match.py. The full end-to-end test
requires mission-specific GCP pairing and image matching functions.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Import synthetic data generation functions from test_correction.py
from test_correction import (
    _generate_nadir_aligned_transforms,
    _generate_spherical_positions,
    _generate_synthetic_boresights,
)

from curryer import utils
from curryer.correction.correction import CorrectionConfig
from curryer.correction.verification import VerificationResult, run_verification

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])


# ==============================================================================
# Helper Functions - Wrap REAL pairing and image matching implementations
# REUSE: These functions use actual pairing.py and image_match.py components
# ==============================================================================


def create_real_gcp_pairing_func():
    """
    Create a GCP pairing function that uses REAL spatial overlap detection.

    Returns:
        Callable that takes science_keys and returns list of (sci_name, gcp_name) pairs
    """
    from scipy.io import loadmat

    from curryer.correction.data_structures import NamedImageGrid
    from curryer.correction.pairing import find_l1a_gcp_pairs

    def real_gcp_pairing(science_keys):
        """Real GCP pairing using actual spatial overlap detection."""
        logger.info(f"  Pairing: Loading {len(science_keys)} science files")

        # Load all science images using L1A_FILES list
        L1A_FILES = [
            ("1/TestCase1a_subimage.mat", "subimage"),
            ("1/TestCase1b_subimage.mat", "subimage"),
            ("2/TestCase2a_subimage.mat", "subimage"),
            ("2/TestCase2b_subimage.mat", "subimage"),
            ("3/TestCase3a_subimage.mat", "subimage"),
            ("3/TestCase3b_subimage.mat", "subimage"),
            ("4/TestCase4a_subimage.mat", "subimage"),
            ("4/TestCase4b_subimage.mat", "subimage"),
            ("5/TestCase5a_subimage.mat", "subimage"),
            ("5/TestCase5b_subimage.mat", "subimage"),
        ]

        # Get base directory from first science key
        test_dir = Path(science_keys[0]).parent.parent

        science_images = []
        for l1a_path, key in L1A_FILES:
            sci_file = test_dir / l1a_path
            mat = loadmat(str(sci_file), squeeze_me=True, struct_as_record=False)[key]
            science_images.append(
                NamedImageGrid(
                    data=np.asarray(mat.data),
                    lat=np.asarray(mat.lat),
                    lon=np.asarray(mat.lon),
                    h=np.asarray(mat.h) if hasattr(mat, "h") else None,
                    name=str(sci_file),
                )
            )

        logger.info(f"  Pairing: Loaded {len(science_images)} science images")

        # Load all GCP images using GCP_FILES list
        # REUSE: GCP file list from test_pairing.py
        GCP_FILES = [
            ("1/GCP12055Dili_resampled.mat", "GCP"),
            ("2/GCP10121Maracaibo_resampled.mat", "GCP"),
            ("3/GCP10665SantaRosa_resampled.mat", "GCP"),
            ("4/GCP20484Morocco_resampled.mat", "GCP"),
            ("5/GCP10087Titicaca_resampled.mat", "GCP"),
        ]

        # Get base directory from first science key
        test_dir = Path(science_keys[0]).parent.parent

        gcp_images = []
        for gcp_path, key in GCP_FILES:
            gcp_file = test_dir / gcp_path
            mat = loadmat(str(gcp_file), squeeze_me=True, struct_as_record=False)[key]
            gcp_images.append(
                NamedImageGrid(
                    data=np.asarray(mat.data),
                    lat=np.asarray(mat.lat),
                    lon=np.asarray(mat.lon),
                    h=np.asarray(mat.h) if hasattr(mat, "h") else None,
                    name=str(gcp_file),
                )
            )

        logger.info(f"  Pairing: Loaded {len(gcp_images)} GCP images")

        # REUSE: Call REAL find_l1a_gcp_pairs from pairing.py
        # Uses max_distance_m=0.0 (any overlap) - same as test_pairing.py
        pairing_result = find_l1a_gcp_pairs(science_images, gcp_images, max_distance_m=0.0)

        logger.info(f"  Pairing: Found {len(pairing_result.matches)} matches")

        # Extract pairs from PairingResult
        # REUSE: Pattern from test_pairing.py test_find_l1a_gcp_pairs()
        pairs = []
        for match in pairing_result.matches:
            l1a_name = pairing_result.l1a_images[match.l1a_index].name
            gcp_name = pairing_result.gcp_images[match.gcp_index].name
            pairs.append((l1a_name, gcp_name))

        logger.info(f"  Pairing: Returning {len(pairs)} pairs")
        return pairs

    return real_gcp_pairing


def create_real_image_matching_func():
    """
    Create an image matching function that uses REAL integrated_image_match.

    Returns:
        Callable that takes (telemetry_key, science_key, gcp_key, config) and returns xr.Dataset
    """
    from scipy.io import loadmat

    from curryer.correction.data_structures import GeolocationConfig as ImGeolocationConfig
    from curryer.correction.data_structures import ImageGrid, OpticalPSFEntry, SearchConfig
    from curryer.correction.image_match import integrated_image_match

    # Load shared calibration data once (PSF and LOS vectors)
    # REUSE: Pattern from test_image_match.py run_image_match()
    test_dir = Path(__file__).parent.parent / "data" / "clarreo" / "image_match"
    psf_file = test_dir / "optical_PSF_675nm_upsampled.mat"
    los_file = test_dir / "b_HS.mat"

    psf_struct = loadmat(str(psf_file), squeeze_me=True, struct_as_record=False)["PSF_struct_675nm"]
    psf_entries = [
        OpticalPSFEntry(
            data=np.asarray(entry.data),
            x=np.asarray(entry.x).ravel(),
            field_angle=np.asarray(entry.FA).ravel(),
        )
        for entry in np.atleast_1d(psf_struct)
    ]

    los_vectors = loadmat(str(los_file), squeeze_me=True)["b_HS"]

    def real_image_matching(telemetry_key, science_key, gcp_key, config):
        """Real image matching using actual test data and integrated_image_match."""
        import re

        import xarray as xr

        # REUSE: Load subimage pattern from test_image_match.py
        subimage_mat = loadmat(science_key, squeeze_me=True, struct_as_record=False)["subimage"]
        subimage = ImageGrid(
            data=np.asarray(subimage_mat.data),
            lat=np.asarray(subimage_mat.lat),
            lon=np.asarray(subimage_mat.lon),
            h=np.asarray(subimage_mat.h) if hasattr(subimage_mat, "h") else None,
        )

        # REUSE: Load GCP pattern from test_image_match.py
        # Use the gcp_key that was found by the pairing function
        gcp_mat = loadmat(gcp_key, squeeze_me=True, struct_as_record=False)["GCP"]
        gcp = ImageGrid(
            data=np.asarray(gcp_mat.data),
            lat=np.asarray(gcp_mat.lat),
            lon=np.asarray(gcp_mat.lon),
            h=np.asarray(gcp_mat.h) if hasattr(gcp_mat, "h") else None,
        )

        # Load spacecraft position (real data from test files)
        # REUSE: Pattern from test_image_match.py
        sci_path = Path(science_key)
        gcp_dir = sci_path.parent
        match = re.search(r"TestCase(\d+)", sci_path.name)
        if match:
            case_num = match.group(1)
            r_iss_file = gcp_dir / f"R_ISS_midframe_TestCase{case_num}.mat"
            if r_iss_file.exists():
                r_iss = loadmat(str(r_iss_file), squeeze_me=True)["R_ISS_midframe"].ravel()
            else:
                r_iss = np.array([7e6, 0, 0])
        else:
            r_iss = np.array([7e6, 0, 0])

        # REUSE: Call REAL integrated_image_match from image_match.py
        # Same usage as test_image_match.py run_image_match()
        result = integrated_image_match(
            subimage=subimage,
            gcp=gcp,
            r_iss_midframe_m=r_iss,
            los_vectors_hs=los_vectors,
            optical_psfs=psf_entries,
            geolocation_config=ImGeolocationConfig(),
            search_config=SearchConfig(),
        )

        # Convert result to xr.Dataset format expected by error stats
        n_meas = 1

        # REUSE: Synthetic geometry generators from test_correction.py
        synthetic_boresights = _generate_synthetic_boresights(n_meas, max_off_nadir_rad=0.07)

        # Use real spacecraft position if available
        if r_iss_file.exists():
            riss_ctrs = r_iss.reshape(1, 3)
        else:
            # REUSE: Generate synthetic position from test_correction.py
            synthetic_positions = _generate_spherical_positions(n_meas, radius_mean_m=6.8e6, radius_std_m=1e4)
            riss_ctrs = synthetic_positions

        # REUSE: Generate transformation matrix from test_correction.py
        t_hs2ctrs = _generate_nadir_aligned_transforms(n_meas, riss_ctrs, synthetic_boresights)

        # Create dataset with REAL image matching errors + synthetic geometry
        dataset = xr.Dataset(
            {
                # REAL errors from integrated_image_match
                "lat_error_deg": ("measurement", np.array([result.lat_error_km / 111.0])),
                "lon_error_deg": (
                    "measurement",
                    np.array([result.lon_error_km / (111.0 * np.cos(np.deg2rad(gcp.lat.mean())))]),
                ),
                # REAL GCP location from test data
                "gcp_lat_deg": ("measurement", np.array([float(gcp.lat.mean())])),
                "gcp_lon_deg": ("measurement", np.array([float(gcp.lon.mean())])),
                "gcp_alt": ("measurement", np.array([float(gcp.h.mean()) if gcp.h is not None else 0.0])),
                # REAL spacecraft position (when available) or synthetic
                "riss_ctrs": (("measurement", "xyz"), riss_ctrs),
                # Synthetic but realistic boresights and transforms
                "bhat_hs": (("measurement", "xyz"), synthetic_boresights),
                "t_hs2ctrs": (("measurement", "row", "col"), t_hs2ctrs),
            },
            coords={"measurement": np.arange(n_meas), "xyz": [0, 1, 2], "row": [0, 1, 2], "col": [0, 1, 2]},
        )

        dataset.attrs["lat_error_km"] = result.lat_error_km
        dataset.attrs["lon_error_km"] = result.lon_error_km
        dataset.attrs["correlation_ccv"] = result.ccv_final

        return dataset

    return real_image_matching


# ==============================================================================
# Test Classes
# ==============================================================================


class TestWeeklyVerificationCLARREO:
    """
    End-to-end integration test for weekly verification.

    REUSES test data loading patterns from test_image_match.py and test_pairing.py
    REUSES config creation from conftest.py fixtures

    Uses CLARREO mission test data as reference implementation.

    NOTE: Verification does NOT create SPICE kernels or perform geolocation.
    It measures accuracy of already-geolocated data by:
    1. GCP pairing (finding which GCPs correspond to data)
    2. Image matching (measuring offsets)
    3. Error statistics (calculating nadir-equivalent errors)
    4. Threshold checking (determining pass/fail)
    """

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Path to CLARREO test data.

        REUSE: Same pattern as test_image_match.py setUp()
        """
        root_dir = Path(__file__).parent.parent.parent
        test_dir = root_dir / "tests" / "data" / "clarreo" / "image_match"
        assert test_dir.is_dir(), f"Test data directory not found: {test_dir}"
        return test_dir

    @pytest.fixture
    def test_data_sets(self, test_data_dir: Path) -> list[tuple[str, str, str]]:
        """Load all available test case paths.

        REUSE: Data discovery pattern from test_pairing.py L1A_FILES and GCP_FILES

        NOTE: This fixture discovers test data files. The actual data loading
        happens via mission-specific functions (gcp_pairing_func, image_matching_func)
        provided in the config. No kernel creation or geolocation is performed.
        """
        data_sets = []

        # Test cases organized in directories 1-5
        for i in range(1, 6):  # 5 directories
            dir_path = test_data_dir / str(i)
            if not dir_path.exists():
                continue

            # Each directory has 'a' and 'b' test cases
            for suffix in ["a", "b"]:
                # REUSE: File naming pattern from test_image_match.py
                subimage_file = dir_path / f"TestCase{i}{suffix}_subimage.mat"

                if subimage_file.exists():
                    # For CLARREO test data, telemetry and science are in same file
                    # GCP file is shared per directory
                    gcp_file = self._find_gcp_file(dir_path)

                    if gcp_file:
                        data_sets.append(
                            (
                                str(subimage_file),  # telemetry_key
                                str(subimage_file),  # science_key (same file for test data)
                                str(gcp_file),  # gcp_key
                            )
                        )

        assert len(data_sets) > 0, "No test data sets found"
        logger.info(f"Found {len(data_sets)} test data sets")
        return data_sets

    @staticmethod
    def _find_gcp_file(directory: Path):
        """Find GCP file in directory.

        REUSE: Pattern from test_pairing.py GCP_FILES

        Returns:
            Path or None: GCP file path if found
        """
        # GCP files are named like "GCP12055Dili_resampled.mat"
        gcp_files = list(directory.glob("GCP*_resampled.mat"))
        return gcp_files[0] if gcp_files else None

    def test_verification_mode_detection(self):
        """Test that we can create a valid verification config.

        This is a simple smoke test that verifies config structure.
        Note: n_iterations and parameters are not actually used by verification,
        but are required by CorrectionConfig. The key requirement is that
        gcp_pairing_func and image_matching_func must be provided.
        """
        from curryer.correction.correction import GeolocationConfig

        # Create verification mode config
        geo_config = GeolocationConfig(
            meta_kernel_file=Path("dummy.json"),
            generic_kernel_dir=Path("data/generic"),
            dynamic_kernels=[],
            instrument_name="CLARREO",
            time_field="time",
        )

        config = CorrectionConfig(
            seed=42,
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
            earth_radius_m=6378137.0,
            geo=geo_config,
            n_iterations=1,  # Required by CorrectionConfig (not used by verification)
            parameters=[],  # Required by CorrectionConfig (not used by verification)
        )

        # Note: For actual verification, you would also need:
        # gcp_pairing_func=your_function,
        # image_matching_func=your_function,
        # But this test just verifies config structure

        # Config is valid (has required fields)
        assert config.n_iterations == 1
        assert len(config.parameters) == 0

    def test_data_sets_discovery(self, test_data_sets):
        """Test that we can discover test data files.

        This verifies the data discovery logic without running verification.
        """
        assert len(test_data_sets) >= 10, f"Expected at least 10 test cases, found {len(test_data_sets)}"

        # Verify structure
        for tel_key, sci_key, gcp_key in test_data_sets:
            assert Path(tel_key).exists(), f"Telemetry file not found: {tel_key}"
            assert Path(sci_key).exists(), f"Science file not found: {sci_key}"
            assert Path(gcp_key).exists(), f"GCP file not found: {gcp_key}"

        logger.info(f"✅ Discovered {len(test_data_sets)} valid test data sets")

    def test_weekly_verification_workflow_full(
        self,
        clarreo_config_programmatic,  # From conftest.py
        test_data_sets,
        tmp_path,
    ):
        """
        Test complete weekly verification workflow with REAL CLARREO functions.

        This test uses the ACTUAL pairing and image matching functions with
        real test data to verify the complete workflow end-to-end.

        NO KERNEL CREATION: Verification does not create or load SPICE kernels.
        It only measures accuracy of existing geolocated data.

        REUSE: Uses real pairing from pairing.py via helper
        REUSE: Uses real image matching from image_match.py via helper
        REUSE: Uses synthetic geometry generators from test_correction.py
        """
        # Get config
        config = clarreo_config_programmatic

        # These fields are required by CorrectionConfig but not used by verification
        config.n_iterations = 1
        config.parameters = []

        # Use helper functions that wrap the REAL pairing and matching implementations
        config.gcp_pairing_func = create_real_gcp_pairing_func()
        config.image_matching_func = create_real_image_matching_func()

        # Run verification (NO kernel creation, NO geolocation)
        result = run_verification(config=config, work_dir=tmp_path, data_sets=test_data_sets)

        # Verify result structure
        assert isinstance(result, VerificationResult)
        assert result.passed in [True, False]  # Check it's a boolean value
        assert isinstance(result.aggregate_stats, xr.Dataset)
        assert len(result.per_pair_metrics) > 0

        # Verify threshold calculation
        assert hasattr(result, "percent_within_threshold")
        assert 0.0 <= result.percent_within_threshold <= 100.0
        assert result.threshold_m == 250.0

        # Log the actual results for reference
        logger.info(f"✅ Verification completed: {result.percent_within_threshold:.1f}% within {result.threshold_m}m")
        logger.info(f"✅ Status: {'PASSED' if result.passed else 'FAILED'}")
        logger.info(f"✅ Processed {len(result.per_pair_metrics)} pairs successfully")
        assert result.required_percent == 39.0

        # Log results (demonstrates usage)
        print(f"\n{'=' * 60}")
        print(f"CLARREO Weekly Verification Results")
        print(f"{'=' * 60}")
        print(f"Total observations: {len(result.per_pair_metrics)}")
        print(f"Nadir error threshold: {result.threshold_m}m")
        print(f"Required success rate: {result.required_percent}%")
        print(f"Actual success rate: {result.percent_within_threshold:.1f}%")
        print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")

        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings:
                print(f"  {warning}")

        print(f"\nPer-Pair Summary (first 5):")
        for i, metric in enumerate(result.per_pair_metrics[:5]):
            rms = metric.get("aggregate_rms_error_m", metric.get("rms_error_m", np.nan))
            nadir = metric.get("nadir_equiv_error_m", np.nan)
            print(f"  Pair {i}: RMS={rms:.1f}m, Nadir={nadir:.1f}m")

        print(f"{'=' * 60}\n")

        # Basic sanity checks
        assert result.timestamp is not None
        assert result.config_summary is not None

    def test_verification_result_structure(self):
        """Test the VerificationResult dataclass structure.

        This tests the result object without running verification.
        """
        # Create a mock result
        mock_stats = xr.Dataset(
            {
                "nadir_equiv_total_error_m": ("measurement", np.array([200.0, 300.0, 150.0])),
            },
            coords={"measurement": np.arange(3)},
        )

        from datetime import datetime

        result = VerificationResult(
            passed=True,
            aggregate_stats=mock_stats,
            per_pair_metrics=[
                {"pair_index": 0, "rms_error_m": 200.0},
                {"pair_index": 1, "rms_error_m": 300.0},
            ],
            warnings=[],
            timestamp=datetime.now(),
            config_summary={"threshold": 250.0},
            percent_within_threshold=66.7,
            threshold_m=250.0,
            required_percent=39.0,
        )

        # Verify structure
        assert result.passed is True
        assert result.percent_within_threshold == 66.7
        assert result.threshold_m == 250.0
        assert result.required_percent == 39.0
        assert len(result.per_pair_metrics) == 2
        assert len(result.warnings) == 0
        assert result.timestamp is not None

        logger.info("✅ VerificationResult structure validated")


class TestVerificationDocumentation:
    """
    Tests that demonstrate how to use the verification module.

    These serve as executable documentation showing:
    - How to configure verification (requires gcp_pairing_func and image_matching_func)
    - How to run verification on already-geolocated data
    - How to interpret results

    NOTE: Verification does NOT create kernels or perform geolocation.
    It only measures accuracy of existing geolocated data.
    """

    def test_example_usage_basic(self, tmp_path):
        """Example: Basic verification usage with mocked data.

        This demonstrates the verification workflow:
        1. Configure with gcp_pairing_func and image_matching_func
        2. Provide already-geolocated data sets
        3. Run verification (NO kernel creation, NO geolocation)
        4. Check results
        """
        from unittest.mock import Mock, patch

        from curryer.correction.correction import GeolocationConfig

        # Step 1: Create verification config
        geo_config = GeolocationConfig(
            meta_kernel_file=Path("dummy.json"),
            generic_kernel_dir=Path("data/generic"),
            dynamic_kernels=[],
            instrument_name="CLARREO",
            time_field="time",
        )

        config = CorrectionConfig(
            seed=42,
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
            earth_radius_m=6378137.0,
            geo=geo_config,
            n_iterations=1,  # Required by CorrectionConfig
            parameters=[],  # Required by CorrectionConfig
            gcp_pairing_func=Mock(return_value=[("sci", "gcp")]),  # REQUIRED for verification
            image_matching_func=Mock(
                return_value=xr.Dataset(
                    {  # REQUIRED for verification
                        "lat_error_deg": ("measurement", np.array([0.001])),
                        "lon_error_deg": ("measurement", np.array([0.002])),
                    },
                    coords={"measurement": np.arange(1)},
                )
            ),
        )

        # Step 2: Prepare data sets (already-geolocated data)
        data_sets = [("tel_key", "sci_key", "gcp_key")]  # Already geolocated!

        # Step 3: Run verification (mock error stats calculation)
        with patch("curryer.correction.verification.call_error_stats_module") as mock_stats:
            # Mock successful verification (>39% within 250m)
            mock_stats.return_value = xr.Dataset(
                {"nadir_equiv_total_error_m": ("measurement", np.random.uniform(100, 400, 100))},
                coords={"measurement": np.arange(100)},
            )
            mock_stats.return_value.attrs["rms_error_m"] = 200.0

            result = run_verification(config, tmp_path, data_sets)

        # Step 4: Check results
        assert isinstance(result, VerificationResult)
        print(f"\nVerification {'PASSED' if result.passed else 'FAILED'}")
        print(f"Performance: {result.percent_within_threshold:.1f}% within {result.threshold_m}m")

        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        logger.info("✅ Example usage demonstrated")

    def test_example_interpreting_results(self):
        """Example: How to interpret verification results.

        This shows what each field in VerificationResult means.
        """
        from datetime import datetime

        # Create example result
        mock_stats = xr.Dataset(
            {
                "nadir_equiv_total_error_m": ("measurement", np.array([200.0, 300.0, 400.0, 150.0, 100.0])),
            },
            coords={"measurement": np.arange(5)},
        )

        result = VerificationResult(
            passed=True,
            aggregate_stats=mock_stats,
            per_pair_metrics=[
                {"pair_index": 0, "rms_error_m": 200.0, "nadir_equiv_error_m": 200.0},
                {"pair_index": 1, "rms_error_m": 300.0, "nadir_equiv_error_m": 300.0},
                {"pair_index": 2, "rms_error_m": 400.0, "nadir_equiv_error_m": 400.0},
                {"pair_index": 3, "rms_error_m": 150.0, "nadir_equiv_error_m": 150.0},
                {"pair_index": 4, "rms_error_m": 100.0, "nadir_equiv_error_m": 100.0},
            ],
            warnings=["Example warning"],
            timestamp=datetime(2026, 1, 12, 10, 30, 0),
            config_summary={"threshold": 250.0, "required": 39.0},
            percent_within_threshold=60.0,  # 3 out of 5 below 250m
            threshold_m=250.0,
            required_percent=39.0,
        )

        # Interpreting the result
        print("\n=== Interpreting VerificationResult ===")
        print(f"1. Overall Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
        print(f"   - {result.percent_within_threshold}% of measurements are within {result.threshold_m}m")
        print(f"   - Required: {result.required_percent}%")
        print()
        print(f"2. Aggregate Statistics:")
        print(f"   - Total measurements: {len(result.aggregate_stats.measurement)}")
        print(f"   - Nadir errors: {result.aggregate_stats['nadir_equiv_total_error_m'].values}")
        print()
        print(f"3. Per-Pair Metrics:")
        for metric in result.per_pair_metrics:
            status = "✓" if metric["nadir_equiv_error_m"] < result.threshold_m else "✗"
            print(f"   {status} Pair {metric['pair_index']}: {metric['nadir_equiv_error_m']:.1f}m")
        print()
        print(f"4. Warnings: {len(result.warnings)}")
        for warning in result.warnings:
            print(f"   - {warning}")
        print()
        print(f"5. Metadata:")
        print(f"   - Run time: {result.timestamp}")
        print(f"   - Config: {result.config_summary}")

        logger.info("✅ Result interpretation demonstrated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
