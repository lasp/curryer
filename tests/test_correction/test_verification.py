"""
Unit tests for verification.py module.

This module tests the weekly verification functionality with mocked components:
- run_verification() orchestration
- Config validation (verification mode detection)
- Threshold checking logic
- Warning message generation
- Per-pair metric extraction

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
These tests use mocking to isolate verification logic from correction.loop().
They validate that verification.py correctly processes results and applies
threshold logic.
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from curryer import utils
from curryer.correction.correction import CorrectionConfig, GeolocationConfig
from curryer.correction.verification import (
    VerificationResult,
    _check_threshold,
    _generate_warnings,
    _validate_verification_config,
    run_verification,
)

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])


class TestVerification:
    """Unit tests for verification module.

    REUSES patterns from test_correction.py and test_image_match.py
    """

    def test_run_verification_with_valid_config(self, tmp_path):
        """Test successful verification run.

        REUSE: Mock pattern but now mock pairing and matching functions
        """
        config = self._create_verification_config()

        # Mock GCP pairing to return empty list (simple case)
        config.gcp_pairing_func = Mock(return_value=[("sci_key", "gcp_key")])

        # Mock image matching to return xr.Dataset with error measurements
        mock_image_match_result = self._create_mock_image_match_result()
        config.image_matching_func = Mock(return_value=mock_image_match_result)

        # Mock call_error_stats_module
        with patch("curryer.correction.verification.call_error_stats_module") as mock_error_stats:
            mock_error_stats.return_value = self._create_mock_aggregate_stats(pass_rate=0.42)

            result = run_verification(config, tmp_path, [("tel_key", "sci_key", "gcp_key")])

            assert isinstance(result, VerificationResult)
            assert result.passed  # Should pass
            assert result.percent_within_threshold > 39.0
            assert len(result.warnings) == 0
            assert result.threshold_m == 250.0
            assert result.required_percent == 39.0

            # Verify functions were called
            config.gcp_pairing_func.assert_called_once()
            config.image_matching_func.assert_called_once()
            mock_error_stats.assert_called_once()

    def test_run_verification_with_failing_performance(self, tmp_path):
        """Test verification failure when performance is below threshold."""
        config = self._create_verification_config()

        # Mock functions
        config.gcp_pairing_func = Mock(return_value=[("sci_key", "gcp_key")])
        mock_image_match_result = self._create_mock_image_match_result()
        config.image_matching_func = Mock(return_value=mock_image_match_result)

        # Mock error stats with only 35% passing (below 39% requirement)
        with patch("curryer.correction.verification.call_error_stats_module") as mock_error_stats:
            mock_error_stats.return_value = self._create_mock_aggregate_stats(pass_rate=0.35)

            result = run_verification(config, tmp_path, [("tel_key", "sci_key", "gcp_key")])

            assert isinstance(result, VerificationResult)
            assert not result.passed  # Should fail
            assert result.percent_within_threshold < 39.0
            assert len(result.warnings) > 0
            assert "VERIFICATION FAILED" in result.warnings[0]
            assert "35.0%" in result.warnings[0]

    def test_run_verification_rejects_missing_gcp_pairing_func(self, tmp_path):
        """Test that missing gcp_pairing_func is rejected."""
        config = self._create_verification_config()
        config.gcp_pairing_func = None  # Missing required function

        with pytest.raises(ValueError, match="gcp_pairing_func is required"):
            run_verification(config, tmp_path, [])

    def test_validate_config_missing_image_matching_func(self):
        """Test that validation rejects missing image_matching_func."""
        config = self._create_verification_config()
        config.image_matching_func = None

        with pytest.raises(ValueError, match="image_matching_func is required"):
            _validate_verification_config(config)

    def test_validate_config_accepts_valid_config(self):
        """Test that validation accepts config with required functions."""
        config = self._create_verification_config()
        _validate_verification_config(config)

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

        REUSE: Pattern from test_correction.py conftest fixtures
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
            telemetry_loader=Mock(),
            science_loader=Mock(),
            gcp_pairing_func=Mock(),
            image_matching_func=Mock(),
        )

    def _create_correction_config(self) -> CorrectionConfig:
        """Create correction mode config (for negative test)."""
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
            n_iterations=100,  # Correction mode (> 1)
            parameters=[Mock()],  # Has parameters
            telemetry_loader=Mock(),
            science_loader=Mock(),
            gcp_pairing_func=Mock(),
            image_matching_func=Mock(),
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
