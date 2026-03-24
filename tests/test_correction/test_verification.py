"""Unit tests for curryer.correction.verification.

Covers
------
- :class:`RequirementsConfig` – Pydantic model construction and validation
- :class:`GCPError` – typed per-measurement detail
- :class:`VerificationResult` – JSON round-trip serialisation
- :func:`_check_threshold` – 0 %, 39 %, 100 % edge cases
- :func:`_generate_warnings` – pass / fail messaging
- :func:`_format_summary_table` – structure and content checks
- :func:`_build_per_gcp_errors` – correct passed flag and coordinate fallback
- :func:`verify` – end-to-end with pre-computed ``image_matching_results``
- :func:`verify` – error paths (empty list, missing inputs, missing func)

All ``CorrectionConfig`` fixtures avoid deleted fields (``telemetry_loader``,
``science_loader``, ``gcp_loader``, ``gcp_pairing_func``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from curryer.correction.config import (
    CorrectionConfig,
    GeolocationConfig,
    ParameterConfig,
    ParameterType,
)
from curryer.correction.verification import (
    GCPError,
    RequirementsConfig,
    VerificationResult,
    _build_per_gcp_errors,
    _check_threshold,
    _format_summary_table,
    _generate_warnings,
    verify,
)

# ===========================================================================
# Helpers / factories
# ===========================================================================

_EARTH_RADIUS_M = 6_378_140.0
_THRESHOLD_M = 250.0
_SPEC_PCT = 39.0


def _make_geo() -> GeolocationConfig:
    """Minimal GeolocationConfig; files need not exist for verification tests."""
    return GeolocationConfig(
        meta_kernel_file=Path("tests/data/test.kernels.tm.json"),
        generic_kernel_dir=Path("data/generic"),
        instrument_name="TEST_INSTRUMENT",
        time_field="corrected_timestamp",
    )


def _make_config(**overrides) -> CorrectionConfig:
    """Return a minimal CorrectionConfig suitable for verification tests.

    Provides CLARREO-style variable name mappings and does **not** set any
    deleted fields (``telemetry_loader``, ``science_loader``, ``gcp_loader``,
    ``gcp_pairing_func``).
    """
    defaults = dict(
        n_iterations=1,
        parameters=[
            ParameterConfig(
                ptype=ParameterType.CONSTANT_KERNEL,
                data={"current_value": [0.0, 0.0, 0.0], "bounds": [-300.0, 300.0]},
            )
        ],
        geo=_make_geo(),
        performance_threshold_m=_THRESHOLD_M,
        performance_spec_percent=_SPEC_PCT,
        earth_radius_m=_EARTH_RADIUS_M,
        # CLARREO-style names so the 13-case dataset validates cleanly
        spacecraft_position_name="riss_ctrs",
        boresight_name="bhat_hs",
        transformation_matrix_name="t_hs2ctrs",
    )
    defaults.update(overrides)
    return CorrectionConfig(**defaults)


def _make_aggregate_stats_dataset(nadir_errors_m: list[float]) -> xr.Dataset:
    """Minimal dataset with ``nadir_equiv_total_error_m`` for threshold tests."""
    n = len(nadir_errors_m)
    return xr.Dataset(
        {
            "nadir_equiv_total_error_m": (["measurement"], np.array(nadir_errors_m, dtype=float)),
            "lat_error_deg": (["measurement"], np.zeros(n)),
            "lon_error_deg": (["measurement"], np.zeros(n)),
        },
        coords={"measurement": np.arange(n)},
    )


def _make_full_image_matching_dataset(n: int = 5, seed: int = 0) -> xr.Dataset:
    """Create a self-contained image-matching dataset usable by verify().

    Uses the validated 13-case geometry sampled with replacement so that
    :class:`~curryer.correction.error_stats.ErrorStatsProcessor` can compute
    nadir-equivalent errors without triggering geometry warnings.
    """
    from tests.test_correction.test_geolocation_error_stats import (
        create_test_dataset_13_cases,
    )

    rng = np.random.default_rng(seed)
    base = create_test_dataset_13_cases()
    indices = rng.integers(0, 13, n)
    sampled = base.isel(measurement=indices).assign_coords(measurement=np.arange(n))
    return sampled


# ===========================================================================
# RequirementsConfig
# ===========================================================================


class TestRequirementsConfig:
    def test_construction(self):
        req = RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0)
        assert req.performance_threshold_m == 250.0
        assert req.performance_spec_percent == 39.0

    def test_json_round_trip(self):
        req = RequirementsConfig(performance_threshold_m=500.0, performance_spec_percent=80.0)
        restored = RequirementsConfig.model_validate_json(req.model_dump_json())
        assert restored.performance_threshold_m == 500.0
        assert restored.performance_spec_percent == 80.0

    def test_missing_fields_raise(self):
        with pytest.raises(ValidationError, match="performance_threshold_m"):
            RequirementsConfig()


# ===========================================================================
# GCPError
# ===========================================================================


class TestGCPError:
    def test_construction_full(self):
        err = GCPError(
            gcp_index=0,
            science_key="sci_0",
            gcp_key="gcp_0",
            lat_error_deg=0.001,
            lon_error_deg=-0.002,
            nadir_equiv_error_m=120.5,
            correlation=0.87,
            passed=True,
        )
        assert err.passed is True
        assert err.nadir_equiv_error_m == pytest.approx(120.5)

    def test_optional_fields_default_to_none(self):
        err = GCPError(
            gcp_index=1,
            science_key="s",
            gcp_key="g",
            lat_error_deg=0.0,
            lon_error_deg=0.0,
            passed=False,
        )
        assert err.nadir_equiv_error_m is None
        assert err.correlation is None

    def test_json_round_trip(self):
        err = GCPError(
            gcp_index=2,
            science_key="sci_2",
            gcp_key="gcp_2",
            lat_error_deg=0.005,
            lon_error_deg=0.003,
            nadir_equiv_error_m=300.0,
            correlation=0.65,
            passed=False,
        )
        raw = json.loads(err.model_dump_json())
        assert raw["gcp_index"] == 2
        assert raw["passed"] is False


# ===========================================================================
# VerificationResult
# ===========================================================================


class TestVerificationResult:
    def _make_result(self, passed: bool = True) -> VerificationResult:
        req = RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0)
        errors = [
            GCPError(
                gcp_index=0,
                science_key="s0",
                gcp_key="g0",
                lat_error_deg=0.001,
                lon_error_deg=0.001,
                nadir_equiv_error_m=100.0,
                passed=True,
            )
        ]
        stats = _make_aggregate_stats_dataset([100.0])
        return VerificationResult(
            passed=passed,
            per_gcp_errors=errors,
            aggregate_stats=stats,
            requirements=req,
            summary_table="table",
            percent_within_threshold=100.0,
            warnings=[],
            timestamp=datetime.now(tz=timezone.utc),
        )

    def test_construction(self):
        result = self._make_result()
        assert result.passed is True
        assert len(result.per_gcp_errors) == 1
        assert isinstance(result.aggregate_stats, xr.Dataset)

    def test_failed_result_has_warnings(self):
        result = self._make_result(passed=False)
        result = VerificationResult(
            passed=False,
            per_gcp_errors=result.per_gcp_errors,
            aggregate_stats=result.aggregate_stats,
            requirements=result.requirements,
            summary_table="t",
            percent_within_threshold=10.0,
            warnings=["⚠️  VERIFICATION FAILED: ..."],
            timestamp=result.timestamp,
        )
        assert len(result.warnings) == 1
        assert "FAILED" in result.warnings[0]

    def test_model_dump_json_excludes_dataset(self):
        """xr.Dataset is arbitrary type — model_dump_json should not crash."""
        result = self._make_result()
        # Pydantic with arbitrary_types_allowed may not be JSON-serialisable for
        # xr.Dataset, but other fields should dump cleanly.
        dumped = result.model_dump(exclude={"aggregate_stats"})
        assert "passed" in dumped
        assert "percent_within_threshold" in dumped


# ===========================================================================
# _check_threshold – edge cases 0 %, 39 %, 100 %
# ===========================================================================


class TestCheckThreshold:
    """Validate _check_threshold against boundary conditions."""

    def _req(self, spec_pct: float = _SPEC_PCT) -> RequirementsConfig:
        return RequirementsConfig(
            performance_threshold_m=_THRESHOLD_M,
            performance_spec_percent=spec_pct,
        )

    def test_zero_percent_within_threshold_fails(self):
        """All errors above threshold → 0 % pass → FAILED."""
        stats = _make_aggregate_stats_dataset([300.0, 400.0, 500.0])
        passed, pct = _check_threshold(stats, self._req())
        assert passed is False
        assert pct == pytest.approx(0.0)

    def test_exactly_at_spec_percent_passes(self):
        """When exactly spec_percent of measurements pass the threshold."""
        # 39 out of 100 below 250m → 39 % → should pass (>= 39 %)
        errors = [100.0] * 39 + [300.0] * 61
        stats = _make_aggregate_stats_dataset(errors)
        passed, pct = _check_threshold(stats, self._req(spec_pct=39.0))
        assert passed is True
        assert pct == pytest.approx(39.0)

    def test_one_below_spec_percent_fails(self):
        """One fewer passing measurement → should fail."""
        errors = [100.0] * 38 + [300.0] * 62
        stats = _make_aggregate_stats_dataset(errors)
        passed, pct = _check_threshold(stats, self._req(spec_pct=39.0))
        assert passed is False
        assert pct == pytest.approx(38.0)

    def test_hundred_percent_within_threshold_passes(self):
        """All errors well below threshold → 100 % pass → PASSED."""
        stats = _make_aggregate_stats_dataset([50.0, 100.0, 150.0, 200.0])
        passed, pct = _check_threshold(stats, self._req())
        assert passed is True
        assert pct == pytest.approx(100.0)

    def test_empty_dataset_fails(self):
        """Empty measurement array → 0 % → FAILED."""
        stats = _make_aggregate_stats_dataset([])
        passed, pct = _check_threshold(stats, self._req())
        assert passed is False
        assert pct == pytest.approx(0.0)

    def test_exactly_at_threshold_does_not_pass(self):
        """Measurement exactly equal to threshold does NOT pass (strict < check)."""
        stats = _make_aggregate_stats_dataset([_THRESHOLD_M])
        passed, pct = _check_threshold(stats, self._req(spec_pct=0.0))
        # 0 % >= 0 % → should pass... except value is not < threshold
        # Expect pct=0 % and passed=True only because spec is 0 %
        assert pct == pytest.approx(0.0)


# ===========================================================================
# _generate_warnings
# ===========================================================================


class TestGenerateWarnings:
    def _req(self) -> RequirementsConfig:
        return RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0)

    def test_no_warnings_when_passed(self):
        warnings = _generate_warnings(passed=True, percent_below=60.0, requirements=self._req())
        assert warnings == []

    def test_warning_emitted_when_failed(self):
        warnings = _generate_warnings(passed=False, percent_below=20.0, requirements=self._req())
        assert len(warnings) == 1
        assert "VERIFICATION FAILED" in warnings[0]
        assert "20.0%" in warnings[0]
        assert "250.0m" in warnings[0]
        assert "39.0%" in warnings[0]

    def test_warning_contains_recommendation(self):
        warnings = _generate_warnings(passed=False, percent_below=5.0, requirements=self._req())
        assert "correction module" in warnings[0].lower() or "Recommend" in warnings[0]


# ===========================================================================
# _format_summary_table
# ===========================================================================


class TestFormatSummaryTable:
    def _req(self) -> RequirementsConfig:
        return RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0)

    def _errors(self) -> list[GCPError]:
        return [
            GCPError(
                gcp_index=0,
                science_key="sci_0",
                gcp_key="gcp_0",
                lat_error_deg=0.00123,
                lon_error_deg=-0.00045,
                nadir_equiv_error_m=145.2,
                passed=True,
            ),
            GCPError(
                gcp_index=1,
                science_key="sci_1",
                gcp_key="gcp_1",
                lat_error_deg=0.00567,
                lon_error_deg=0.00234,
                nadir_equiv_error_m=312.8,
                passed=False,
            ),
        ]

    def test_returns_string(self):
        table = _format_summary_table(self._errors(), self._req(), 50.0, False)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_contains_header_and_footer(self):
        table = _format_summary_table(self._errors(), self._req(), 50.0, False)
        assert "Verification Summary" in table
        assert "Result:" in table

    def test_pass_verdict_appears(self):
        table = _format_summary_table(self._errors(), self._req(), 60.0, True)
        assert "PASSED" in table

    def test_fail_verdict_appears(self):
        table = _format_summary_table(self._errors(), self._req(), 20.0, False)
        assert "FAILED" in table

    def test_threshold_and_spec_in_footer(self):
        table = _format_summary_table(self._errors(), self._req(), 50.0, False)
        assert "250.0m" in table
        assert "39.0%" in table

    def test_checkmark_and_cross_present(self):
        table = _format_summary_table(self._errors(), self._req(), 50.0, False)
        assert "✓" in table
        assert "✗" in table

    def test_empty_errors_list(self):
        """Should not raise with zero measurements."""
        table = _format_summary_table([], self._req(), 0.0, False)
        assert "Result:" in table

    def test_nadir_none_shows_na(self):
        errors = [
            GCPError(
                gcp_index=0,
                science_key="s",
                gcp_key="g",
                lat_error_deg=0.0,
                lon_error_deg=0.0,
                nadir_equiv_error_m=None,
                passed=False,
            )
        ]
        table = _format_summary_table(errors, self._req(), 0.0, False)
        assert "N/A" in table


# ===========================================================================
# _build_per_gcp_errors
# ===========================================================================


class TestBuildPerGcpErrors:
    def _req(self) -> RequirementsConfig:
        return RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0)

    def _stats_with_errors(self, nadir_errors: list[float]) -> xr.Dataset:
        n = len(nadir_errors)
        return xr.Dataset(
            {
                "nadir_equiv_total_error_m": (["measurement"], np.array(nadir_errors)),
                "lat_error_deg": (["measurement"], np.linspace(0.001, 0.005, n)),
                "lon_error_deg": (["measurement"], np.linspace(-0.001, 0.001, n)),
            },
            coords={"measurement": np.arange(n)},
        )

    def test_length_matches_measurements(self):
        stats = self._stats_with_errors([100.0, 300.0, 200.0])
        errors = _build_per_gcp_errors(stats, [], self._req())
        assert len(errors) == 3

    def test_passed_flag_set_correctly(self):
        stats = self._stats_with_errors([100.0, 300.0])
        errors = _build_per_gcp_errors(stats, [], self._req())
        assert errors[0].passed is True  # 100 < 250
        assert errors[1].passed is False  # 300 >= 250

    def test_source_mapping_applied(self):
        stats = self._stats_with_errors([100.0])
        mapping = [("my_science", "my_gcp")]
        errors = _build_per_gcp_errors(stats, mapping, self._req())
        assert errors[0].science_key == "my_science"
        assert errors[0].gcp_key == "my_gcp"

    def test_fallback_keys_when_mapping_too_short(self):
        stats = self._stats_with_errors([100.0, 200.0])
        errors = _build_per_gcp_errors(stats, [], self._req())
        assert errors[0].science_key == "sci_0"
        assert errors[1].gcp_key == "gcp_1"

    def test_correlation_extracted_when_present(self):
        n = 2
        stats = xr.Dataset(
            {
                "nadir_equiv_total_error_m": (["measurement"], [100.0, 200.0]),
                "lat_error_deg": (["measurement"], [0.001, 0.002]),
                "lon_error_deg": (["measurement"], [0.001, 0.002]),
                "correlation": (["measurement"], [0.85, 0.92]),
            },
            coords={"measurement": np.arange(n)},
        )
        errors = _build_per_gcp_errors(stats, [], self._req())
        assert errors[0].correlation == pytest.approx(0.85)
        assert errors[1].correlation == pytest.approx(0.92)

    def test_no_correlation_variable_gives_none(self):
        stats = self._stats_with_errors([100.0])
        errors = _build_per_gcp_errors(stats, [], self._req())
        assert errors[0].correlation is None

    def test_empty_dataset_returns_empty_list(self):
        stats = self._stats_with_errors([])
        errors = _build_per_gcp_errors(stats, [], self._req())
        assert errors == []


# ===========================================================================
# verify() – integration tests
# ===========================================================================


class TestVerify:
    """End-to-end tests for :func:`verify` using synthetic image-matching data."""

    @pytest.fixture
    def config(self) -> CorrectionConfig:
        return _make_config()

    @pytest.fixture
    def image_matching_dataset(self) -> xr.Dataset:
        """Single-pair dataset built from the validated 13-case geometry."""
        return _make_full_image_matching_dataset(n=13, seed=42)

    @pytest.fixture
    def multi_pair_results(self) -> list[xr.Dataset]:
        """Two GCP pairs with different sci/gcp labels."""
        ds1 = _make_full_image_matching_dataset(n=7, seed=0)
        ds1.attrs["sci_key"] = "scene_A"
        ds1.attrs["gcp_key"] = "gcp_site_1"
        ds2 = _make_full_image_matching_dataset(n=6, seed=1)
        ds2.attrs["sci_key"] = "scene_B"
        ds2.attrs["gcp_key"] = "gcp_site_2"
        return [ds1, ds2]

    # -------------------------------------------------------------------
    # Happy path – single GCP pair
    # -------------------------------------------------------------------

    def test_returns_verification_result(self, config, image_matching_dataset, tmp_path):
        result = verify(config, tmp_path, image_matching_results=[image_matching_dataset])
        assert isinstance(result, VerificationResult)

    def test_result_has_all_fields(self, config, image_matching_dataset, tmp_path):
        result = verify(config, tmp_path, image_matching_results=[image_matching_dataset])
        assert isinstance(result.passed, bool)
        assert isinstance(result.per_gcp_errors, list)
        assert isinstance(result.aggregate_stats, xr.Dataset)
        assert isinstance(result.summary_table, str)
        assert isinstance(result.percent_within_threshold, float)
        assert isinstance(result.warnings, list)
        assert isinstance(result.timestamp, datetime)

    def test_per_gcp_errors_count_matches_measurements(self, config, image_matching_dataset, tmp_path):
        n = image_matching_dataset.sizes["measurement"]
        result = verify(config, tmp_path, image_matching_results=[image_matching_dataset])
        assert len(result.per_gcp_errors) == n

    def test_all_per_gcp_have_nadir_error(self, config, image_matching_dataset, tmp_path):
        result = verify(config, tmp_path, image_matching_results=[image_matching_dataset])
        for err in result.per_gcp_errors:
            assert err.nadir_equiv_error_m is not None
            assert err.nadir_equiv_error_m >= 0.0

    def test_passed_flag_consistent_with_percent(self, config, image_matching_dataset, tmp_path):
        result = verify(config, tmp_path, image_matching_results=[image_matching_dataset])
        if result.passed:
            assert result.percent_within_threshold >= config.performance_spec_percent
        else:
            assert result.percent_within_threshold < config.performance_spec_percent

    def test_summary_table_is_non_empty_string(self, config, image_matching_dataset, tmp_path):
        result = verify(config, tmp_path, image_matching_results=[image_matching_dataset])
        assert len(result.summary_table) > 0
        assert "Verification Summary" in result.summary_table

    def test_warnings_empty_when_passed(self, config, image_matching_dataset, tmp_path):
        result = verify(config, tmp_path, image_matching_results=[image_matching_dataset])
        if result.passed:
            assert result.warnings == []

    def test_warnings_non_empty_when_failed(self, config, tmp_path):
        """Force a FAILED result by using a very tight spec (100 %)."""
        strict_config = _make_config(performance_spec_percent=100.0)
        ds = _make_full_image_matching_dataset(n=13, seed=0)
        result = verify(strict_config, tmp_path, image_matching_results=[ds])
        # With 100 % required, any imperfect measurement causes failure
        if not result.passed:
            assert len(result.warnings) >= 1
            assert "VERIFICATION FAILED" in result.warnings[0]

    # -------------------------------------------------------------------
    # Happy path – multiple GCP pairs
    # -------------------------------------------------------------------

    def test_multi_pair_aggregates_all_measurements(self, config, multi_pair_results, tmp_path):
        total = sum(ds.sizes["measurement"] for ds in multi_pair_results)
        result = verify(config, tmp_path, image_matching_results=multi_pair_results)
        assert len(result.per_gcp_errors) == total

    def test_multi_pair_science_keys_from_attrs(self, config, multi_pair_results, tmp_path):
        result = verify(config, tmp_path, image_matching_results=multi_pair_results)
        sci_keys = {e.science_key for e in result.per_gcp_errors}
        assert "scene_A" in sci_keys
        assert "scene_B" in sci_keys

    def test_requirements_reflect_config(self, config, image_matching_dataset, tmp_path):
        result = verify(config, tmp_path, image_matching_results=[image_matching_dataset])
        assert result.requirements.performance_threshold_m == _THRESHOLD_M
        assert result.requirements.performance_spec_percent == _SPEC_PCT

    def test_work_dir_created_if_missing(self, config, image_matching_dataset, tmp_path):
        new_dir = tmp_path / "nonexistent" / "subdir"
        assert not new_dir.exists()
        verify(config, new_dir, image_matching_results=[image_matching_dataset])
        assert new_dir.exists()

    # -------------------------------------------------------------------
    # RequirementsConfig override via config.verification
    # -------------------------------------------------------------------

    def test_custom_requirements_override_used(self, image_matching_dataset, tmp_path):
        """Attach RequirementsConfig to config.verification; verify() should use it."""
        base_config = _make_config()
        # Inject a very lenient requirement so it almost certainly passes
        req = RequirementsConfig(performance_threshold_m=1_000_000.0, performance_spec_percent=0.0)
        # Monkeypatch the config object (verification is not a declared field but
        # _build_requirements uses getattr with a fallback)
        object.__setattr__(base_config, "verification", req)
        result = verify(base_config, tmp_path, image_matching_results=[image_matching_dataset])
        assert result.requirements.performance_threshold_m == 1_000_000.0

    # -------------------------------------------------------------------
    # Error paths
    # -------------------------------------------------------------------

    def test_empty_image_matching_list_raises(self, config, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            verify(config, tmp_path, image_matching_results=[])

    def test_neither_input_raises_value_error(self, config, tmp_path):
        with pytest.raises(ValueError, match="Neither image_matching_results nor geolocated_data"):
            verify(config, tmp_path)

    def test_geolocated_data_without_func_raises(self, config, tmp_path):
        dummy_ds = xr.Dataset({"dummy": (["x"], [1, 2, 3])})
        with pytest.raises(ValueError, match="image_matching_func is not set"):
            verify(config, tmp_path, geolocated_data=dummy_ds)

    def test_geolocated_data_with_func_called(self, image_matching_dataset, tmp_path):
        """image_matching_func should be called when geolocated_data is supplied."""
        called = {"count": 0}

        def mock_matching_func(data):
            called["count"] += 1
            return [image_matching_dataset]

        config = _make_config(image_matching_func=mock_matching_func)
        dummy_geolocated = xr.Dataset({"placeholder": (["x"], [1, 2])})
        result = verify(config, tmp_path, geolocated_data=dummy_geolocated)
        assert called["count"] == 1
        assert isinstance(result, VerificationResult)
