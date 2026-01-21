"""
Unit tests for verification.py module.

This module tests the weekly verification functionality:
- run_verification() orchestration with real function calls
- Threshold checking logic
- Warning message generation
- Per-pair metric extraction

The tests use mocking at the module level (pair_files, image_matching)
to isolate verification logic while still testing the real flow.

Running Tests:
-------------
# Via pytest (recommended)
pytest tests/test_correction/test_verification.py -v

# Run specific test
pytest tests/test_correction/test_verification.py::TestVerification::test_run_verification_with_valid_config -v

# Standalone execution
python -m pytest tests/test_correction/test_verification.py

Requirements:
-----------------
These tests validate that verification.py correctly:
1. Calls pair_files() from pairing.py
2. Calls image_matching() from correction.py
3. Calls call_error_stats_module() from correction.py
4. Applies threshold logic correctly
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from curryer import utils
from curryer.correction.correction import CorrectionConfig, GeolocationConfig
from curryer.correction.verification import (
    VerificationResult,
    _check_threshold,
    _generate_warnings,
    run_verification,
)

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])


class TestVerification:
    """Unit tests for verification module.

    Tests the simplified verification module that directly calls
    pair_files() and image_matching() without function injection.
    """

    def test_run_verification_with_valid_config(self, tmp_path):
        """Test successful verification run with mocked real functions.

        Mocks pair_files() and image_matching() at module level.
        """
        config = self._create_verification_config()
        
        # Create mock file paths
        l1a_files = [tmp_path / "geo_001.mat", tmp_path / "geo_002.mat"]
        gcp_directory = tmp_path / "gcps"
        telemetry = pd.DataFrame({"time": [0.0], "position": [[0, 0, 0]]})
        calibration_dir = tmp_path / "calibration"
        
        # Create mock image match result
        mock_image_match_result = self._create_mock_image_match_result()
        
        # Mock pair_files from pairing.py
        with patch("curryer.correction.verification.pair_files") as mock_pair_files, \
             patch("curryer.correction.verification.load_image_grid_from_mat") as mock_load, \
             patch("curryer.correction.verification.image_matching") as mock_image_matching, \
             patch("curryer.correction.verification.call_error_stats_module") as mock_error_stats:
            
            # Setup mocks
            mock_pair_files.return_value = [(l1a_files[0], tmp_path / "gcp_001.mat")]
            mock_load.return_value = Mock(lat=np.array([0.0]), lon=np.array([0.0]), data=np.array([0.0]))
            mock_image_matching.return_value = mock_image_match_result
            mock_error_stats.return_value = self._create_mock_aggregate_stats(pass_rate=0.42)
            
            result = run_verification(
                config=config,
                work_dir=tmp_path,
                l1a_files=l1a_files,
                gcp_directory=gcp_directory,
                telemetry=telemetry,
                calibration_dir=calibration_dir,
            )
            
            assert isinstance(result, VerificationResult)
            assert result.passed  # Should pass
            assert result.percent_within_threshold > 39.0
            assert len(result.warnings) == 0
            assert result.threshold_m == 250.0
            assert result.required_percent == 39.0
            
            # Verify real functions were called
            mock_pair_files.assert_called_once()
            mock_image_matching.assert_called()
            mock_error_stats.assert_called_once()

    def test_run_verification_with_failing_performance(self, tmp_path):
        """Test verification failure when performance is below threshold."""
        config = self._create_verification_config()
        
        # Create mock file paths
        l1a_files = [tmp_path / "geo_001.mat"]
        gcp_directory = tmp_path / "gcps"
        telemetry = pd.DataFrame({"time": [0.0], "position": [[0, 0, 0]]})
        calibration_dir = tmp_path / "calibration"
        
        mock_image_match_result = self._create_mock_image_match_result()
        
        # Mock real functions with error stats showing only 35% passing
        with patch("curryer.correction.verification.pair_files") as mock_pair_files, \
             patch("curryer.correction.verification.load_image_grid_from_mat") as mock_load, \
             patch("curryer.correction.verification.image_matching") as mock_image_matching, \
             patch("curryer.correction.verification.call_error_stats_module") as mock_error_stats:
            
            mock_pair_files.return_value = [(l1a_files[0], tmp_path / "gcp_001.mat")]
            mock_load.return_value = Mock(lat=np.array([0.0]), lon=np.array([0.0]), data=np.array([0.0]))
            mock_image_matching.return_value = mock_image_match_result
            mock_error_stats.return_value = self._create_mock_aggregate_stats(pass_rate=0.35)
            
            result = run_verification(
                config=config,
                work_dir=tmp_path,
                l1a_files=l1a_files,
                gcp_directory=gcp_directory,
                telemetry=telemetry,
                calibration_dir=calibration_dir,
            )
            
            assert isinstance(result, VerificationResult)
            assert not result.passed  # Should fail
            assert result.percent_within_threshold < 39.0
            assert len(result.warnings) > 0
            assert "VERIFICATION FAILED" in result.warnings[0]
            assert "35.0%" in result.warnings[0]

    def test_run_verification_no_gcp_pairs_found(self, tmp_path):
        """Test that verification fails gracefully when no GCP pairs found."""
        config = self._create_verification_config()
        
        l1a_files = [tmp_path / "geo_001.mat"]
        gcp_directory = tmp_path / "gcps"
        telemetry = pd.DataFrame({"time": [0.0], "position": [[0, 0, 0]]})
        calibration_dir = tmp_path / "calibration"
        
        # Mock pair_files to return empty list
        with patch("curryer.correction.verification.pair_files") as mock_pair_files:
            mock_pair_files.return_value = []  # No pairs found
            
            with pytest.raises(ValueError, match="No GCP pairs found"):
                run_verification(
                    config=config,
                    work_dir=tmp_path,
                    l1a_files=l1a_files,
                    gcp_directory=gcp_directory,
                    telemetry=telemetry,
                    calibration_dir=calibration_dir,
                )

    def test_check_threshold_exactly_at_requirement(self):
        """Test threshold logic when exactly at 39%."""
        config = self._create_verification_config()

        # Exactly 39% should pass
        stats = self._create_aggregate_stats_with_percent(0.39)
        passed, percent = _check_threshold(stats, config)

        assert passed  # Should pass
        assert np.isclose(percent, 39.0, atol=0.1)

    def test_check_threshold_just_below_requirement(self):
        """Test threshold logic just below requirement (38.9%)."""
        config = self._create_verification_config()

        # Just below (38.9%) - should fail
        stats = self._create_aggregate_stats_with_percent(0.389)
        passed, percent = _check_threshold(stats, config)

        assert not passed  # Should fail
        assert percent < 39.0
        assert np.isclose(percent, 38.9, atol=0.1)

    def test_check_threshold_well_above_requirement(self):
        """Test threshold logic well above requirement (50%)."""
        config = self._create_verification_config()

        # Well above (50%) - should pass
        stats = self._create_aggregate_stats_with_percent(0.50)
        passed, percent = _check_threshold(stats, config)

        assert passed  # Should pass
        assert percent > 39.0
        assert np.isclose(percent, 50.0, atol=0.1)

    def test_check_threshold_zero_percent(self):
        """Test threshold logic with 0% passing (edge case)."""
        config = self._create_verification_config()

        # All errors above threshold
        stats = self._create_aggregate_stats_with_percent(0.0)
        passed, percent = _check_threshold(stats, config)

        assert not passed  # Should fail
        assert percent == 0.0

    def test_check_threshold_hundred_percent(self):
        """Test threshold logic with 100% passing (edge case)."""
        config = self._create_verification_config()

        # All errors below threshold
        stats = self._create_aggregate_stats_with_percent(1.0)
        passed, percent = _check_threshold(stats, config)

        assert passed  # Should pass
        assert percent == 100.0

    def test_generate_warnings_on_failure(self):
        """Test warning generation when verification fails."""
        config = self._create_verification_config()

        warnings = _generate_warnings(False, 35.0, config)

        assert len(warnings) > 0
        assert "VERIFICATION FAILED" in warnings[0]
        assert "35.0%" in warnings[0]
        assert "250" in warnings[0]  # threshold
        assert "39.0%" in warnings[0]  # required
        assert "correction module" in warnings[0].lower()

    def test_generate_warnings_on_success(self):
        """Test no warnings when verification passes."""
        config = self._create_verification_config()

        warnings = _generate_warnings(True, 42.0, config)

        assert len(warnings) == 0

    def test_verification_result_dataclass(self):
        """Test VerificationResult dataclass structure."""
        aggregate_stats = self._create_mock_aggregate_stats(pass_rate=0.45)
        per_pair_metrics = [{"pair_index": 0, "rms_error_m": 200.0}]

        result = VerificationResult(
            passed=True,
            aggregate_stats=aggregate_stats,
            per_pair_metrics=per_pair_metrics,
            warnings=[],
            timestamp=None,
            config_summary={"threshold": 250.0},
            percent_within_threshold=45.0,
            threshold_m=250.0,
            required_percent=39.0,
        )

        assert result.passed is True
        assert result.percent_within_threshold == 45.0
        assert result.threshold_m == 250.0
        assert result.required_percent == 39.0
        assert isinstance(result.aggregate_stats, xr.Dataset)
        assert len(result.per_pair_metrics) == 1
        assert len(result.warnings) == 0

    # ========================================================================
    # Helper methods (REUSE patterns from test_image_match.py)
    # ========================================================================

    def _create_verification_config(self) -> CorrectionConfig:
        """Create valid verification mode config.

        No function injection needed - verification uses real functions directly.
        """
        geo_config = GeolocationConfig(
            meta_kernel_file=Path("dummy.json"),
            generic_kernel_dir=Path("data/generic"),
            dynamic_kernels=[],
            instrument_name="test_instrument",
            time_field="time",
            minimum_correlation=None,
        )

        return CorrectionConfig(
            seed=42,  # Required parameter
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
            earth_radius_m=6378137.0,
            geo=geo_config,
            n_iterations=1,  # Verification mode
            parameters=[],  # No parameter sweep
        )

    def _create_mock_aggregate_stats(self, pass_rate: float) -> xr.Dataset:
        """Create mock aggregate error stats dataset.

        REUSE: Structure matches ErrorStatsProcessor.process_geolocation_errors() output

        Parameters
        ----------
        pass_rate : float
            Fraction of measurements that should be below threshold (0.0-1.0)
        """
        n_measurements = 100
        n_passing = int(n_measurements * pass_rate)

        # Create realistic error distribution
        errors_below = np.random.uniform(50, 249, n_passing)  # Below 250m threshold
        errors_above = np.random.uniform(250, 500, n_measurements - n_passing)  # Above threshold
        errors = np.concatenate([errors_below, errors_above])
        np.random.shuffle(errors)

        ds = xr.Dataset(
            {
                "nadir_equiv_total_error_m": ("measurement", errors),
                "lat_error_deg": ("measurement", np.random.randn(n_measurements) * 0.01),
                "lon_error_deg": ("measurement", np.random.randn(n_measurements) * 0.01),
            },
            coords={"measurement": np.arange(n_measurements)},
        )

        # Add attributes (like ErrorStatsProcessor does)
        ds.attrs["rms_error_m"] = float(np.sqrt(np.mean(errors**2)))
        ds.attrs["mean_error_m"] = float(np.mean(errors))
        ds.attrs["max_error_m"] = float(np.max(errors))

        return ds

    def _create_mock_image_match_result(self) -> xr.Dataset:
        """Create mock image matching result.

        Returns xr.Dataset matching image_match output structure.
        """
        # Simple mock with error measurements
        ds = xr.Dataset(
            {
                "lat_error_deg": ("measurement", np.array([0.001])),
                "lon_error_deg": ("measurement", np.array([0.002])),
            },
            coords={"measurement": np.arange(1)},
        )

        # Add attributes
        ds.attrs["lat_error_deg"] = 0.001
        ds.attrs["lon_error_deg"] = 0.002
        ds.attrs["correlation_ccv"] = 0.95

        return ds

    def _create_aggregate_stats_with_percent(self, target_percent: float) -> xr.Dataset:
        """Create aggregate stats with exact percentage below threshold.

        Parameters
        ----------
        target_percent : float
            Exact fraction below threshold (0.0-1.0)
        """
        n_measurements = 1000  # Large number for precision
        n_below = int(n_measurements * target_percent)

        # Create exact distribution
        errors_below = np.full(n_below, 200.0)  # Below 250m
        errors_above = np.full(n_measurements - n_below, 300.0)  # Above 250m
        errors = np.concatenate([errors_below, errors_above])

        return xr.Dataset(
            {"nadir_equiv_total_error_m": ("measurement", errors)},
            coords={"measurement": np.arange(n_measurements)},
        )

    def _create_mock_netcdf_data(self, n_pairs: int, pass_rate: float) -> dict:
        """Create mock netcdf_data structure.

        REUSE: Structure matches _build_netcdf_structure() in correction.py

        Parameters
        ----------
        n_pairs : int
            Number of GCP pairs
        pass_rate : float
            Fraction of pairs that should have errors below threshold
        """
        n_param_sets = 1  # Verification mode

        # Generate realistic errors
        rms_errors = np.random.uniform(100, 400, (n_param_sets, n_pairs))
        nadir_errors = rms_errors * np.random.uniform(0.8, 1.2, (n_param_sets, n_pairs))

        return {
            "rms_error_m": rms_errors,
            "nadir_equiv_total_error_m": nadir_errors,
            "lat_error_km": np.random.uniform(-5, 5, (n_param_sets, n_pairs)),
            "lon_error_km": np.random.uniform(-5, 5, (n_param_sets, n_pairs)),
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
