"""
Integration tests for weekly verification workflow.

This module provides integration tests that verify the complete verification workflow
works correctly with the refactored simplified design.

IMPORTANT: Verification does NOT create SPICE kernels or perform geolocation.
It only measures accuracy of already-geolocated data by comparing to GCPs.

The refactored verification module directly calls:
1. pair_files() from pairing.py - for GCP pairing
2. image_matching() from correction.py - for offset measurement
3. call_error_stats_module() from correction.py - for error statistics

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
These tests verify that the refactored verification workflow correctly:
1. Accepts concrete file paths instead of abstract keys
2. Calls real implementations without function injection
3. Properly processes results and applies threshold logic
"""

import logging
from pathlib import Path

import pytest

from curryer import utils

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])


# ==============================================================================
# Test Classes
# ==============================================================================


class TestWeeklyVerificationCLARREO:
    """
    Integration tests for weekly verification with refactored design.

    Tests verify that the refactored verification module:
    1. Accepts concrete file paths instead of abstract keys
    2. Directly calls real implementations (pair_files, image_matching, call_error_stats_module)
    3. Does not require function injection through config

    NOTE: Verification does NOT create SPICE kernels or perform geolocation.
    It measures accuracy of already-geolocated data.
    """

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Path to CLARREO test data."""
        root_dir = Path(__file__).parent.parent.parent
        test_dir = root_dir / "tests" / "data" / "clarreo" / "image_match"
        assert test_dir.is_dir(), f"Test data directory not found: {test_dir}"
        return test_dir

    def test_verification_mode_detection(self):
        """Test that we can create a valid verification config without function injection.

        This verifies the simplified config structure where gcp_pairing_func and
        image_matching_func are no longer required.
        """
        from curryer.correction.correction import CorrectionConfig, GeolocationConfig

        # Create verification mode config - NO function injection needed!
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
            n_iterations=1,
            parameters=[],
        )

        # Config is valid without function injection
        assert config.n_iterations == 1
        assert len(config.parameters) == 0
        # No need to check for gcp_pairing_func or image_matching_func!

    def test_data_discovery(self, test_data_dir):
        """Test that we can discover L1A and GCP files in test data.

        This verifies we can discover concrete file paths for verification.
        """
        # Discover L1A files
        l1a_files = []
        for i in range(1, 6):  # 5 directories
            dir_path = test_data_dir / str(i)
            if dir_path.exists():
                l1a_files.extend(list(dir_path.glob("TestCase*_subimage.mat")))

        assert len(l1a_files) > 0, "No L1A files found"
        logger.info(f"✅ Discovered {len(l1a_files)} L1A files")

        # Verify GCP directory exists
        gcp_files = []
        for i in range(1, 6):
            dir_path = test_data_dir / str(i)
            if dir_path.exists():
                gcp_files.extend(list(dir_path.glob("GCP*_resampled.mat")))

        assert len(gcp_files) > 0, "No GCP files found"
        logger.info(f"✅ Discovered {len(gcp_files)} GCP files")

    def test_verification_with_mocked_real_functions(self, test_data_dir, tmp_path):
        """Test verification workflow with mocked real function calls.

        This test verifies that run_verification correctly:
        1. Accepts concrete file paths
        2. Calls pair_files() from pairing.py
        3. Calls image_matching() from correction.py
        4. Calls call_error_stats_module() from correction.py
        5. Returns proper VerificationResult

        Mocking is used to avoid complexity of MATLAB data loading,
        but the flow through verification.py is real.
        """
        from unittest.mock import Mock, patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from curryer.correction.correction import CorrectionConfig, GeolocationConfig
        from curryer.correction.verification import VerificationResult, run_verification

        # Create config without function injection
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
            n_iterations=1,
            parameters=[],
        )

        # Create mock file paths
        l1a_files = [test_data_dir / "1" / "TestCase1a_subimage.mat"]
        gcp_directory = test_data_dir
        telemetry = pd.DataFrame({"time": [0.0]})
        calibration_dir = test_data_dir

        # Mock the real functions that verification.py calls
        with patch("curryer.correction.verification.pair_files") as mock_pair_files, \
             patch("curryer.correction.verification.load_image_grid_from_mat") as mock_load, \
             patch("curryer.correction.verification.image_matching") as mock_image_matching, \
             patch("curryer.correction.verification.call_error_stats_module") as mock_error_stats:

            # Setup mocks
            mock_pair_files.return_value = [(l1a_files[0], test_data_dir / "1" / "GCP_test.mat")]
            mock_load.return_value = Mock(
                lat=np.array([[0.0]]),
                lon=np.array([[0.0]]),
                data=np.array([[0.0]]),
            )

            # Mock image matching result
            mock_result = xr.Dataset(
                {"lat": (["pixel"], [0.0]), "lon": (["pixel"], [0.0])},
                attrs={"lat_error_deg": 0.001, "lon_error_deg": 0.002, "correlation_ccv": 0.95},
            )
            mock_image_matching.return_value = mock_result

            # Mock error stats with passing rate
            n_measurements = 100
            errors_below = np.random.uniform(50, 249, 42)  # 42% below 250m threshold
            errors_above = np.random.uniform(250, 500, 58)
            errors = np.concatenate([errors_below, errors_above])
            mock_error_stats.return_value = xr.Dataset(
                {
                    "nadir_equiv_total_error_m": ("measurement", errors),
                },
                coords={"measurement": np.arange(100)},
            )

            # Call verification - NOTE: No function injection!
            result = run_verification(
                config=config,
                work_dir=tmp_path,
                l1a_files=l1a_files,
                gcp_directory=gcp_directory,
                telemetry=telemetry,
                calibration_dir=calibration_dir,
            )

            # Verify result
            assert isinstance(result, VerificationResult)
            assert result.passed  # 42% > 39% required
            assert result.percent_within_threshold > 39.0

            # Verify real functions were called correctly
            mock_pair_files.assert_called_once_with(
                l1a_files=l1a_files,
                gcp_directory=gcp_directory,
                max_distance_m=0.0,
                l1a_key="subimage",
                gcp_key="GCP",
                gcp_pattern="*_resampled.mat",
            )
            mock_image_matching.assert_called()
            mock_error_stats.assert_called_once()

            logger.info(f"✅ Verification passed with {result.percent_within_threshold:.1f}% within threshold")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
