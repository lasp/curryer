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

All ``GeolocationSetup`` fixtures avoid deleted fields (``telemetry_loader``,
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
    GeolocationConfig,
    GeolocationSetup,
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


def _make_setup(**overrides) -> GeolocationSetup:
    """Return a minimal GeolocationSetup suitable for verification tests.

    Provides CLARREO-style variable name mappings and does **not** set any
    deleted fields (``telemetry_loader``, ``science_loader``, ``gcp_loader``,
    ``gcp_pairing_func``).

    ``performance_threshold_m`` / ``performance_spec_percent`` overrides are
    folded into the nested :class:`RequirementsConfig`.
    """
    threshold_m = overrides.pop("performance_threshold_m", _THRESHOLD_M)
    spec_percent = overrides.pop("performance_spec_percent", _SPEC_PCT)
    defaults = dict(
        geo=_make_geo(),
        requirements=RequirementsConfig(
            performance_threshold_m=threshold_m,
            performance_spec_percent=spec_percent,
        ),
        # CLARREO-style names so the 13-case dataset validates cleanly
        spacecraft_position_name="riss_ctrs",
        boresight_name="bhat_hs",
        transformation_matrix_name="t_hs2ctrs",
    )
    defaults.update(overrides)
    return GeolocationSetup(**defaults)


def _make_aggregate_stats_dataset(nadir_errors_m: list[float], rejected: tuple[int, ...] = ()) -> xr.Dataset:
    """Minimal error-stats output for threshold tests; *rejected* indexes rejected measurements."""
    n = len(nadir_errors_m)
    accepted = np.ones(n, dtype=bool)
    accepted[list(rejected)] = False
    return xr.Dataset(
        {
            "nadir_equiv_total_error_m": (["measurement"], np.array(nadir_errors_m, dtype=float)),
            "lat_error_deg": (["measurement"], np.zeros(n)),
            "lon_error_deg": (["measurement"], np.zeros(n)),
            "accepted": (["measurement"], accepted),
            "rejection_reason": (["measurement"], np.where(accepted, "", "correlation 0.1000 < 0.5")),
        },
        coords={"measurement": np.arange(n)},
    )


def _make_full_image_matching_dataset(n: int = 5, seed: int = 0) -> xr.Dataset:
    """Create a self-contained image-matching dataset usable by verify().

    Uses the validated 13-case geometry sampled with replacement so that
    :class:`~curryer.correction.error_stats.ErrorStatsProcessor` can compute
    nadir-equivalent errors without triggering geometry warnings.
    """
    from test_error_stats import (
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

    def test_passed_contradicting_status_raises(self):
        with pytest.raises(ValueError, match="contradicts status"):
            GCPError(
                gcp_index=0,
                science_key="s",
                gcp_key="g",
                lat_error_deg=0.0,
                lon_error_deg=0.0,
                passed=True,
                status="fail",
            )

    @pytest.mark.parametrize(("status", "reason"), [("rejected", None), ("fail", "correlation 0.1000 < 0.5")])
    def test_rejection_reason_only_when_rejected(self, status, reason):
        with pytest.raises(ValueError, match="rejection_reason"):
            GCPError(
                gcp_index=0,
                science_key="s",
                gcp_key="g",
                lat_error_deg=0.0,
                lon_error_deg=0.0,
                passed=False,
                status=status,
                rejection_reason=reason,
            )

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
            status="pass",
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
            status="fail",
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
            status="fail",
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
                status="pass",
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

    def test_rejected_measurements_excluded(self):
        """Only accepted measurements enter the percentage."""
        stats = _make_aggregate_stats_dataset([100.0, 900.0, 950.0], rejected=(1, 2))
        passed, pct = _check_threshold(stats, self._req())
        assert pct == pytest.approx(100.0)
        assert passed

    def test_all_rejected_fails_at_zero(self):
        stats = _make_aggregate_stats_dataset([100.0, 200.0], rejected=(0, 1))
        assert _check_threshold(stats, self._req()) == (False, 0.0)

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
        """Value at threshold fails per-measurement check, but overall spec=0 % passes."""
        stats = _make_aggregate_stats_dataset([_THRESHOLD_M])
        passed, pct = _check_threshold(stats, self._req(spec_pct=0.0))
        # Per-measurement: value is not < threshold → 0 % of measurements pass.
        # Overall: 0 % >= spec_pct (0 %) → overall verification passes.
        assert pct == pytest.approx(0.0)
        assert passed is True


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
                status="pass",
            ),
            GCPError(
                gcp_index=1,
                science_key="sci_1",
                gcp_key="gcp_1",
                lat_error_deg=0.00567,
                lon_error_deg=0.00234,
                nadir_equiv_error_m=312.8,
                passed=False,
                status="fail",
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

    def test_status_names_and_rejection_count_present(self):
        rejected = GCPError(
            gcp_index=2,
            science_key="sci_2",
            gcp_key="gcp_2",
            lat_error_deg=0.01,
            lon_error_deg=0.01,
            nadir_equiv_error_m=1240.6,
            passed=False,
            status="rejected",
            rejection_reason="correlation 0.412 < 0.8",
        )
        table = _format_summary_table([*self._errors(), rejected], self._req(), 50.0, False)
        assert "PASS" in table
        assert "FAIL" in table
        assert "REJECTED" in table
        assert "2 accepted, 1 rejected" in table

    def test_rows_aligned_when_footer_is_widest(self):
        errors = [e.model_copy(update={"quality_weight": 4.0}) for e in self._errors()]
        table = _format_summary_table(errors, self._req(), 50.0, False, weighted_percent=50.0)
        assert "effective n = 2.0" in table
        assert len({len(line) for line in table.splitlines()}) == 1

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
                status="fail",
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
                "accepted": (["measurement"], np.ones(n, dtype=bool)),
                "rejection_reason": (["measurement"], np.full(n, "")),
            },
            coords={"measurement": np.arange(n)},
        )

    def test_rejected_measurement_has_status_and_reason(self):
        stats = self._stats_with_errors([100.0, 1300.0])
        stats["accepted"].values[1] = False
        stats["rejection_reason"] = (["measurement"], np.array(["", "peak margin 0.0100 < 0.05"]))
        errors = _build_per_gcp_errors(stats, [], self._req())
        assert [e.status for e in errors] == ["pass", "rejected"]
        assert errors[1].rejection_reason == "peak margin 0.0100 < 0.05"
        assert errors[1].nadir_equiv_error_m == pytest.approx(1300.0)

    def test_track_and_view_fields_extracted_when_present(self):
        stats = self._stats_with_errors([100.0, 300.0])
        stats["along_track_error_m"] = (["measurement"], np.array([90.0, -280.0]))
        stats["cross_track_error_m"] = (["measurement"], np.array([-40.0, 100.0]))
        stats["off_nadir_angle_deg"] = (["measurement"], np.array([1.5, 52.2]))
        stats["correlation_secondary"] = (["measurement"], np.array([0.4, -np.inf]))
        errors = _build_per_gcp_errors(stats, [], self._req())
        assert (errors[0].along_track_error_m, errors[0].cross_track_error_m) == (90.0, -40.0)
        assert errors[1].off_nadir_angle_deg == pytest.approx(52.2)
        assert errors[0].correlation_secondary == pytest.approx(0.4)
        assert errors[1].correlation_secondary is None

    def test_track_fields_none_without_azimuth(self):
        errors = _build_per_gcp_errors(self._stats_with_errors([100.0]), [], self._req())
        assert errors[0].along_track_error_m is None
        assert errors[0].cross_track_error_m is None

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
        stats = self._stats_with_errors([100.0, 200.0])
        stats["correlation"] = (["measurement"], [0.85, 0.92])
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
    def setup(self) -> GeolocationSetup:
        return _make_setup()

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

    def test_returns_verification_result(self, setup, image_matching_dataset, tmp_path):
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        assert isinstance(result, VerificationResult)

    def test_result_has_all_fields(self, setup, image_matching_dataset, tmp_path):
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        assert isinstance(result.passed, bool)
        assert isinstance(result.per_gcp_errors, list)
        assert isinstance(result.aggregate_stats, xr.Dataset)
        assert isinstance(result.summary_table, str)
        assert isinstance(result.percent_within_threshold, float)
        assert isinstance(result.warnings, list)
        assert isinstance(result.timestamp, datetime)

    def test_per_gcp_errors_count_matches_measurements(self, setup, image_matching_dataset, tmp_path):
        n = image_matching_dataset.sizes["measurement"]
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        assert len(result.per_gcp_errors) == n

    def test_all_per_gcp_have_nadir_error(self, setup, image_matching_dataset, tmp_path):
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        for err in result.per_gcp_errors:
            assert err.nadir_equiv_error_m is not None
            assert err.nadir_equiv_error_m >= 0.0

    def test_passed_flag_consistent_with_percent(self, setup, image_matching_dataset, tmp_path):
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        if result.passed:
            assert result.percent_within_threshold >= setup.requirements.performance_spec_percent
        else:
            assert result.percent_within_threshold < setup.requirements.performance_spec_percent

    def test_summary_table_is_non_empty_string(self, setup, image_matching_dataset, tmp_path):
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        assert len(result.summary_table) > 0
        assert "Verification Summary" in result.summary_table

    def test_warnings_empty_when_passed(self, setup, image_matching_dataset, tmp_path):
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        if result.passed:
            assert result.warnings == []

    def test_warnings_non_empty_when_failed(self, setup, tmp_path):
        """Force a FAILED result by using a very tight spec (100 %)."""
        strict_setup = _make_setup(performance_spec_percent=100.0)
        ds = _make_full_image_matching_dataset(n=13, seed=0)
        result = verify(strict_setup, image_matching_results=[ds], work_dir=tmp_path)
        # With 100 % required, any imperfect measurement causes failure
        if not result.passed:
            assert len(result.warnings) >= 1
            assert "VERIFICATION FAILED" in result.warnings[0]

    # -------------------------------------------------------------------
    # Happy path – multiple GCP pairs
    # -------------------------------------------------------------------

    def test_multi_pair_aggregates_all_measurements(self, setup, multi_pair_results, tmp_path):
        total = sum(ds.sizes["measurement"] for ds in multi_pair_results)
        result = verify(setup, image_matching_results=multi_pair_results, work_dir=tmp_path)
        assert len(result.per_gcp_errors) == total

    def test_multi_pair_science_keys_from_attrs(self, setup, multi_pair_results, tmp_path):
        result = verify(setup, image_matching_results=multi_pair_results, work_dir=tmp_path)
        sci_keys = {e.science_key for e in result.per_gcp_errors}
        assert "scene_A" in sci_keys
        assert "scene_B" in sci_keys

    def test_requirements_reflect_config(self, setup, image_matching_dataset, tmp_path):
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        assert result.requirements.performance_threshold_m == _THRESHOLD_M
        assert result.requirements.performance_spec_percent == _SPEC_PCT

    def test_work_dir_created_if_missing(self, setup, image_matching_dataset, tmp_path):
        new_dir = tmp_path / "nonexistent" / "subdir"
        assert not new_dir.exists()
        verify(setup, image_matching_results=[image_matching_dataset], work_dir=new_dir)
        assert new_dir.exists()

    # -------------------------------------------------------------------
    # RequirementsConfig override via setup.requirements
    # -------------------------------------------------------------------

    def test_custom_requirements_override_used(self, image_matching_dataset, tmp_path):
        """Set setup.requirements directly; verify() should use it."""
        setup = _make_setup()
        # Inject a very lenient requirement so it almost certainly passes
        setup.requirements = RequirementsConfig(performance_threshold_m=1_000_000.0, performance_spec_percent=0.0)
        result = verify(setup, image_matching_results=[image_matching_dataset], work_dir=tmp_path)
        assert result.requirements.performance_threshold_m == 1_000_000.0

    # -------------------------------------------------------------------
    # Error paths
    # -------------------------------------------------------------------

    def test_empty_image_matching_list_raises(self, setup, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            verify(setup, image_matching_results=[], work_dir=tmp_path)

    def test_neither_input_raises_value_error(self, setup, tmp_path):
        with pytest.raises(ValueError, match="Neither image_matching_results nor geolocated_data"):
            verify(setup, work_dir=tmp_path)

    def test_geolocated_data_without_func_raises(self, setup, tmp_path):
        """geolocated_data without gcp_directory/los_file/psf_file raises ValueError."""
        dummy_ds = xr.Dataset({"dummy": (["x"], [1, 2, 3])})
        with pytest.raises(ValueError, match="gcp_directory"):
            verify(setup, geolocated_data=dummy_ds, work_dir=tmp_path)

    def test_geolocated_data_primary_mode(self, image_matching_dataset, tmp_path):
        """Primary mode: geolocated_data + gcp_directory + los_file + psf_file auto-pairs and matches."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import numpy as np

        setup = _make_setup()

        # Build a minimal dataset with lat/lon so the footprint filter runs.
        lat = np.linspace(38.0, 39.0, 5)
        lon = np.linspace(-116.0, -115.0, 6)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        dummy_geolocated = xr.Dataset(
            {
                "band_data": (["y", "x"], np.ones((5, 6))),
                "lat": (["y", "x"], lat_grid),
                "lon": (["y", "x"], lon_grid),
            }
        )

        # Create a synthetic GCP chip that falls inside the footprint so at least
        # one chip is matched (otherwise verify() raises before calling the pipeline).
        with tempfile.TemporaryDirectory() as tmp_gcp_dir:
            gcp_dir = Path(tmp_gcp_dir)
            gcp_chip_path = gcp_dir / "chip_001_regridded.nc"
            # Centre of chip is inside the observation footprint.
            chip_lat = np.linspace(38.3, 38.7, 4)
            chip_lon = np.linspace(-115.8, -115.4, 4)
            chip_lat_g, chip_lon_g = np.meshgrid(chip_lat, chip_lon, indexing="ij")
            chip_ds = xr.Dataset(
                {
                    "band_data": (["y", "x"], np.ones((4, 4))),
                    "lat": (["y", "x"], chip_lat_g),
                    "lon": (["y", "x"], chip_lon_g),
                }
            )
            chip_ds.to_netcdf(gcp_chip_path)

            with (
                patch(
                    "curryer.correction.verification.match_geolocated_to_gcp_files",
                    return_value=[image_matching_dataset],
                ) as mock_fn,
                patch(
                    "curryer.correction.verification.load_los_vectors",
                    return_value=[[0.0, 0.0, 1.0]],
                ),
                patch(
                    "curryer.correction.verification.load_optical_psf",
                    return_value=[],
                ),
            ):
                result = verify(
                    setup,
                    geolocated_data=dummy_geolocated,
                    gcp_directory=gcp_dir,
                    los_file=tmp_path / "los.mat",
                    psf_file=tmp_path / "psf.mat",
                    work_dir=tmp_path,
                )
                mock_fn.assert_called_once()
                call_args = mock_fn.call_args
                # First positional arg is the geolocated dataset
                assert call_args.args[0] is dummy_geolocated
                # Second positional arg is the list of matched GCP paths
                assert gcp_chip_path in call_args.args[1]

        assert isinstance(result, VerificationResult)
        assert result.passed

    def test_geolocated_data_with_func_called(self, image_matching_dataset, tmp_path):
        """setup.image_matching_func should be called when geolocated_data is supplied."""
        called = {"count": 0}

        def mock_matching_func(
            geolocated_data,
            gcp_reference_file=None,
            telemetry=None,
            params_info=None,
            setup=None,
            los_vectors_cached=None,
            optical_psfs_cached=None,
            r_iss_midframe=None,
        ):
            called["count"] += 1
            return [image_matching_dataset]

        setup = _make_setup()
        setup.image_matching_func = mock_matching_func
        dummy_geolocated = xr.Dataset({"placeholder": (["x"], [1, 2])})
        result = verify(setup, geolocated_data=dummy_geolocated, work_dir=tmp_path)
        assert called["count"] == 1
        assert isinstance(result, VerificationResult)

    def test_gcp_pairs_raises_without_los_psf(self, setup, tmp_path):
        """gcp_pairs mode raises ValueError when los_file or psf_file is missing."""
        with pytest.raises(ValueError, match="los_file"):
            verify(setup, gcp_pairs=[("obs.mat", "gcp.mat")], work_dir=tmp_path)

    def test_observation_paths_raises_without_los_psf(self, setup, tmp_path):
        """observation_paths + gcp_directory mode raises ValueError when los_file/psf_file missing."""
        with pytest.raises(ValueError, match="los_file"):
            verify(setup, observation_paths=["obs.mat"], gcp_directory=tmp_path, work_dir=tmp_path)

    def test_gcp_directory_alone_raises_value_error(self, setup, tmp_path):
        """gcp_directory without observation_paths raises ValueError."""
        with pytest.raises(ValueError, match="observation_paths and gcp_directory"):
            verify(setup, gcp_directory=tmp_path, work_dir=tmp_path)


# ===========================================================================
# _log_pairing_summary
# ===========================================================================


class TestLogPairingSummary:
    """Tests for the _log_pairing_summary logging helper."""

    def test_all_paired(self, caplog):
        import logging

        from curryer.correction.verification import _log_pairing_summary

        pairs = [(Path("obs_001.mat"), Path("gcp_001.mat")), (Path("obs_002.mat"), Path("gcp_002.mat"))]
        with caplog.at_level(logging.INFO, logger="curryer.correction.verification"):
            _log_pairing_summary(pairs)

        log_text = "\n".join(caplog.messages)
        assert "obs_001.mat" in log_text
        assert "gcp_001.mat" in log_text
        assert "Proceeding with 2 observation(s)" in log_text

    def test_with_unpaired(self, caplog):
        import logging

        from curryer.correction.verification import _log_pairing_summary

        pairs = [(Path("obs_001.mat"), Path("gcp_001.mat"))]
        unpaired = [Path("obs_002.mat")]
        with caplog.at_level(logging.INFO, logger="curryer.correction.verification"):
            _log_pairing_summary(pairs, unpaired=unpaired)

        log_text = "\n".join(caplog.messages)
        assert "obs_002.mat" in log_text
        assert "No matching GCP" in log_text
        assert "Proceeding with 1 observation(s)" in log_text

    def test_empty_pairs(self, caplog):
        import logging

        from curryer.correction.verification import _log_pairing_summary

        with caplog.at_level(logging.INFO, logger="curryer.correction.verification"):
            _log_pairing_summary([])

        assert "Proceeding with 0 observation(s)" in "\n".join(caplog.messages)


# ===========================================================================
# Viewing geometry: no silent nadir fallback, no silently skipped pairs
# ===========================================================================


class TestViewingGeometryFailures:
    """Geometry failures raise instead of degrading to nadir or dropping a pair."""

    @staticmethod
    def _write_grid_nc(path: Path, lat0: float, lon0: float, position_m: np.ndarray | None = None) -> Path:
        lat, lon = np.meshgrid(lat0 + np.linspace(0.1, -0.1, 5), lon0 + np.linspace(-0.1, 0.1, 5), indexing="ij")
        data_vars = {
            "band_data": (["y", "x"], np.ones((5, 5))),
            "lat": (["y", "x"], lat),
            "lon": (["y", "x"], lon),
        }
        if position_m is not None:
            data_vars["position"] = (["xyz"], position_m)
        xr.Dataset(data_vars).to_netcdf(path)
        return path

    def test_gcp_pair_without_spacecraft_position_raises(self, tmp_path):
        from unittest.mock import patch

        obs = self._write_grid_nc(tmp_path / "obs.nc", 26.15, -102.33)
        gcp = self._write_grid_nc(tmp_path / "gcp_regridded.nc", 26.15, -102.33)
        with (
            patch("curryer.correction.image_io.load_los_vectors", return_value=np.tile([0.0, 0.0, 1.0], (5, 1))),
            patch("curryer.correction.image_io.load_optical_psf", return_value=[]),
            patch("curryer.correction.verification.integrated_image_match") as mock_match,
            pytest.raises(ValueError, match="Spacecraft ECEF position is required"),
        ):
            verify(
                _make_setup(),
                gcp_pairs=[(obs, gcp)],
                los_file=tmp_path / "los.mat",
                psf_file=tmp_path / "psf.mat",
                work_dir=tmp_path,
            )
        mock_match.assert_not_called()

    def test_gcp_pair_records_line_of_sight_in_gcp_column(self, tmp_path):
        """The boresight is aimed at the GCP's cross-track column, not the observation's centre."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from curryer.compute.spatial import geodetic_to_ecef
        from curryer.correction.verification import _run_image_matching_for_pairs

        obs_center = geodetic_to_ecef(np.array([-102.33, 26.15, 0.0]), meters=True, degrees=True)
        # GCP nearest pixel (3, 3); the mid-frame row is 2, so the viewed pixel is (2, 3).
        mid_row_gcp_column = geodetic_to_ecef(np.array([-102.28, 26.15, 0.0]), meters=True, degrees=True)
        east = np.array([-np.sin(np.deg2rad(-102.33)), np.cos(np.deg2rad(-102.33)), 0.0])
        r_sc = obs_center + 410_000.0 * obs_center / np.linalg.norm(obs_center) + 700_000.0 * east
        obs = self._write_grid_nc(tmp_path / "obs.nc", 26.15, -102.33, position_m=r_sc)
        gcp = self._write_grid_nc(tmp_path / "gcp_regridded.nc", 26.10, -102.28)
        match = SimpleNamespace(
            lat_error_km=0.1, lon_error_km=-0.2, ccv_final=0.9, ccv_secondary=0.3, final_grid_step_m=30.0
        )
        setup = _make_setup()
        with (
            patch("curryer.correction.image_io.load_los_vectors", return_value=np.tile([0.0, 0.0, 1.0], (5, 1))),
            patch("curryer.correction.image_io.load_optical_psf", return_value=[]),
            patch("curryer.correction.verification.integrated_image_match", return_value=match) as mock_match,
        ):
            (ds,), _ = _run_image_matching_for_pairs([(obs, gcp)], tmp_path / "los.mat", tmp_path / "psf.mat", setup)

        np.testing.assert_allclose(mock_match.call_args.kwargs["r_iss_midframe_m"], r_sc)
        expected = (mid_row_gcp_column - r_sc) / np.linalg.norm(mid_row_gcp_column - r_sc)
        np.testing.assert_allclose(ds[setup.boresight_name].values[0], expected, atol=1e-9)
        assert np.dot(expected, -r_sc / np.linalg.norm(r_sc)) < np.cos(np.deg2rad(30.0))
        to_obs_center = (obs_center - r_sc) / np.linalg.norm(obs_center - r_sc)
        assert np.rad2deg(np.arccos(np.dot(expected, to_obs_center))) > 0.1

    def test_error_degrees_round_trip_to_matched_meters(self, tmp_path):
        """km → deg here and deg → m in ErrorStatsProcessor use the same radius."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from curryer.compute.spatial import geodetic_to_ecef
        from curryer.correction.error_stats import _EARTH_RADIUS_M
        from curryer.correction.verification import _run_image_matching_for_pairs

        r_sc = geodetic_to_ecef(np.array([-102.33, 26.15, 410_000.0]), meters=True, degrees=True)
        obs = self._write_grid_nc(tmp_path / "obs.nc", 26.15, -102.33, position_m=r_sc)
        gcp = self._write_grid_nc(tmp_path / "gcp_regridded.nc", 26.15, -102.33)
        match = SimpleNamespace(
            lat_error_km=0.3, lon_error_km=-0.2, ccv_final=0.9, ccv_secondary=0.3, final_grid_step_m=30.0
        )
        with (
            patch("curryer.correction.image_io.load_los_vectors", return_value=np.tile([0.0, 0.0, 1.0], (5, 1))),
            patch("curryer.correction.image_io.load_optical_psf", return_value=[]),
            patch("curryer.correction.verification.integrated_image_match", return_value=match),
        ):
            (ds,), _ = _run_image_matching_for_pairs(
                [(obs, gcp)], tmp_path / "los.mat", tmp_path / "psf.mat", _make_setup()
            )

        ns_m = _EARTH_RADIUS_M * np.deg2rad(float(ds["lat_error_deg"].values[0]))
        ew_m = (
            _EARTH_RADIUS_M
            * np.cos(np.deg2rad(float(ds["gcp_lat_deg"].values[0])))
            * np.deg2rad(float(ds["lon_error_deg"].values[0]))
        )
        assert ns_m == pytest.approx(300.0, abs=1e-9)
        assert ew_m == pytest.approx(-200.0, abs=1e-9)

    @staticmethod
    def _image_matching_inputs(with_frame: bool):
        from curryer.correction.grid_types import ImageGrid

        lat, lon = np.meshgrid(np.linspace(26.2, 26.1, 3), np.linspace(-102.4, -102.3, 3), indexing="ij")
        grid = ImageGrid(data=np.ones((3, 3)), lat=lat, lon=lon)
        coords = {"frame": [1.4699e9, 1.4699e9 + 0.0667, 1.4699e9 + 0.1334]} if with_frame else {}
        geolocated = xr.Dataset(
            {"latitude": (["frame", "pixel"], lat), "longitude": (["frame", "pixel"], lon)}, coords=coords
        )
        return grid, geolocated

    def _call_image_matching(
        self, tmp_path, with_frame: bool, spice_side_effect=None, r_iss_midframe=(-1.5e6, -5.9e6, 3.0e6)
    ):
        from types import SimpleNamespace
        from unittest.mock import patch

        from curryer.correction.verification import image_matching

        grid, geolocated = self._image_matching_inputs(with_frame)
        match = SimpleNamespace(
            lat_error_km=0.1,
            lon_error_km=0.1,
            ccv_final=0.9,
            ccv_secondary=0.3,
            final_grid_step_m=30.0,
            final_index_row=1,
            final_index_col=1,
        )
        with (
            patch("curryer.correction.verification.geolocated_to_image_grid", return_value=grid),
            patch("curryer.correction.verification.load_image_grid", return_value=grid),
            patch("curryer.correction.verification.integrated_image_match", return_value=match) as mock_match,
            patch(
                "curryer.correction.verification._get_spice_boresight_and_rotation",
                side_effect=spice_side_effect,
            ),
        ):
            try:
                return image_matching(
                    geolocated_data=geolocated,
                    gcp_reference_file=tmp_path / "gcp.nc",
                    setup=_make_setup(),
                    los_vectors_cached=np.array([[0.0, 0.0, 1.0]]),
                    optical_psfs_cached=[],
                    r_iss_midframe=np.array(r_iss_midframe),
                )
            finally:
                self.image_match_calls = mock_match.call_count

    def test_image_matching_without_midframe_time_raises(self, tmp_path):
        with pytest.raises(ValueError, match="mid-frame time"):
            self._call_image_matching(tmp_path, with_frame=False)
        assert self.image_match_calls == 0

    def test_image_matching_spice_failure_propagates(self, tmp_path):
        from curryer import spicierpy as sp

        SpiceyError = sp.utils.exceptions.SpiceyError
        with pytest.raises(SpiceyError, match="no attitude coverage"):
            self._call_image_matching(tmp_path, with_frame=True, spice_side_effect=SpiceyError("no attitude coverage"))
        assert self.image_match_calls == 0

    def test_image_matching_rejects_position_in_kilometers(self, tmp_path):
        with pytest.raises(ValueError, match="inside the Earth"):
            self._call_image_matching(tmp_path, with_frame=True, r_iss_midframe=(-1.5e3, -5.9e3, 3.0e3))
        assert self.image_match_calls == 0

    def test_match_geolocated_to_gcp_files_propagates_failure(self, tmp_path):
        from unittest.mock import patch

        from curryer.correction.verification import match_geolocated_to_gcp_files

        _, geolocated = self._image_matching_inputs(with_frame=True)
        gcp_files = [tmp_path / "gcp_a.nc", tmp_path / "gcp_b.nc"]
        with (
            patch(
                "curryer.correction.verification.image_matching",
                side_effect=[xr.Dataset(), ValueError("match failed for gcp_b")],
            ) as mock_matching,
            pytest.raises(ValueError, match="match failed for gcp_b"),
        ):
            match_geolocated_to_gcp_files(geolocated, gcp_files, _make_setup())
        assert mock_matching.call_count == 2


# ===========================================================================
# Image-matching configuration and correlation carried through verification
# ===========================================================================


class TestMatchingConfigAndCorrelation:
    """Setup-level search/PSF configuration and the per-measurement correlation."""

    def test_setup_defaults_match_previous_hardcoded_values(self):
        setup = _make_setup()
        assert (setup.search.grid_size, setup.search.grid_span_km) == (44, 11.0)
        assert (setup.search.reduction_factor, setup.search.spacing_limit_m) == (0.8, 10.0)
        assert setup.search.peak_exclusion_km == 2.0
        assert setup.psf_sampling.psf_lat_sample_dist_deg == 2.4397105613972e-05
        assert setup.psf_sampling.psf_lon_sample_dist_deg == 2.8737038710207e-05

    def test_file_pair_matching_uses_setup_configs_and_records_correlation(self, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import patch

        from curryer.compute.spatial import geodetic_to_ecef
        from curryer.correction.config import PSFSamplingConfig, SearchConfig
        from curryer.correction.verification import _run_image_matching_for_pairs

        r_sc = geodetic_to_ecef(np.array([-102.33, 26.15, 410_000.0]), meters=True, degrees=True)
        obs = TestViewingGeometryFailures._write_grid_nc(tmp_path / "obs.nc", 26.15, -102.33, position_m=r_sc)
        gcp = TestViewingGeometryFailures._write_grid_nc(tmp_path / "gcp_regridded.nc", 26.15, -102.33)
        setup = _make_setup(
            psf_sampling=PSFSamplingConfig(psf_lat_sample_dist_deg=1e-4),
            search=SearchConfig(grid_size=20, grid_span_km=40.0),
        )
        match = SimpleNamespace(
            lat_error_km=0.1, lon_error_km=-0.2, ccv_final=0.83, ccv_secondary=0.4, final_grid_step_m=30.0
        )
        with (
            patch("curryer.correction.image_io.load_los_vectors", return_value=np.tile([0.0, 0.0, 1.0], (5, 1))),
            patch("curryer.correction.image_io.load_optical_psf", return_value=[]),
            patch("curryer.correction.verification.integrated_image_match", return_value=match) as mock_match,
        ):
            (ds,), _ = _run_image_matching_for_pairs([(obs, gcp)], tmp_path / "los.mat", tmp_path / "psf.mat", setup)

        assert mock_match.call_args.kwargs["geolocation_config"] is setup.psf_sampling
        assert mock_match.call_args.kwargs["search_config"] is setup.search
        assert float(ds["correlation"].values[0]) == pytest.approx(0.83)
        # The test grid's rows run north to south.
        assert float(ds["track_azimuth_deg"].values[0]) == pytest.approx(180.0, abs=0.01)

    @pytest.mark.parametrize("corr_name", ["correlation", "ccv", "im_ccv"])
    def test_aggregation_keeps_correlation(self, corr_name):
        from curryer.correction.verification import _aggregate_image_matching_results

        setup = _make_setup()
        results = [
            xr.Dataset(
                {
                    "lat_error_deg": (["measurement"], [0.001]),
                    "lon_error_deg": (["measurement"], [0.002]),
                    corr_name: (["measurement"], [ccv]),
                },
                coords={"measurement": [0]},
            )
            for ccv in (0.9, 0.4)
        ]
        aggregated = _aggregate_image_matching_results(results, setup)
        np.testing.assert_allclose(aggregated["correlation"].values, [0.9, 0.4])

    def test_file_pair_matching_uses_the_observation_detector_columns(self, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import patch

        from curryer.compute.spatial import geodetic_to_ecef
        from curryer.correction.verification import _run_image_matching_for_pairs

        r_sc = geodetic_to_ecef(np.array([-102.33, 26.15, 410_000.0]), meters=True, degrees=True)
        obs = TestViewingGeometryFailures._write_grid_nc(tmp_path / "obs.nc", 26.15, -102.33, position_m=r_sc)
        with xr.open_dataset(obs) as ds:
            cropped = ds.load().assign(detector_pixel=(["x"], np.arange(10, 15)))
        cropped.to_netcdf(tmp_path / "obs_crop.nc")
        gcp = TestViewingGeometryFailures._write_grid_nc(tmp_path / "gcp_regridded.nc", 26.15, -102.33)
        table = np.column_stack([np.zeros(20), np.linspace(-0.1, 0.1, 20), np.ones(20)])
        match = SimpleNamespace(
            lat_error_km=0.1, lon_error_km=-0.2, ccv_final=0.83, ccv_secondary=0.4, final_grid_step_m=30.0
        )
        with (
            patch("curryer.correction.image_io.load_los_vectors", return_value=table),
            patch("curryer.correction.image_io.load_optical_psf", return_value=[]),
            patch("curryer.correction.verification.integrated_image_match", return_value=match) as mock_match,
        ):
            (ds,), _ = _run_image_matching_for_pairs(
                [(tmp_path / "obs_crop.nc", gcp)], tmp_path / "los.mat", tmp_path / "psf.mat", _make_setup()
            )
        np.testing.assert_array_equal(mock_match.call_args.kwargs["los_vectors_hs"], table[10:15])
        assert float(ds["correlation_secondary"].values[0]) == pytest.approx(0.4)

    def test_aggregation_keeps_secondary_correlation(self):
        from curryer.correction.verification import _aggregate_image_matching_results

        results = [
            xr.Dataset(
                {
                    "lat_error_deg": (["measurement"], [0.001]),
                    "lon_error_deg": (["measurement"], [0.002]),
                    "correlation": (["measurement"], [ccv]),
                    "correlation_secondary": (["measurement"], [ccv - 0.1]),
                },
                coords={"measurement": [0]},
            )
            for ccv in (0.9, 0.4)
        ]
        aggregated = _aggregate_image_matching_results(results, _make_setup())
        np.testing.assert_allclose(aggregated["correlation_secondary"].values, [0.8, 0.3])
        with pytest.raises(ValueError, match="'correlation_secondary' is present in only some"):
            _aggregate_image_matching_results(
                [results[0], results[1].drop_vars("correlation_secondary")], _make_setup()
            )

    def test_aggregation_rejects_partial_correlation(self):
        from curryer.correction.verification import _aggregate_image_matching_results

        with_corr = xr.Dataset(
            {
                "lat_error_deg": (["measurement"], [0.001]),
                "lon_error_deg": (["measurement"], [0.002]),
                "correlation": (["measurement"], [0.9]),
            },
            coords={"measurement": [0]},
        )
        without_corr = with_corr.drop_vars("correlation")
        with pytest.raises(ValueError, match="present in only some"):
            _aggregate_image_matching_results([with_corr, without_corr], _make_setup())

    def test_aggregation_carries_track_azimuth_and_rejects_partial(self):
        from curryer.correction.verification import _aggregate_image_matching_results

        with_az = xr.Dataset(
            {
                "lat_error_deg": (["measurement"], [0.001]),
                "lon_error_deg": (["measurement"], [0.002]),
                "track_azimuth_deg": (["measurement"], [191.5]),
            },
            coords={"measurement": [0]},
        )
        aggregated = _aggregate_image_matching_results([with_az, with_az], _make_setup())
        np.testing.assert_array_equal(aggregated["track_azimuth_deg"].values, [191.5, 191.5])
        with pytest.raises(ValueError, match="'track_azimuth_deg' is present in only some"):
            _aggregate_image_matching_results([with_az, with_az.drop_vars("track_azimuth_deg")], _make_setup())

    def test_verify_with_every_measurement_rejected_reports_them(self, tmp_path):
        setup = _make_setup(geo=_make_geo().model_copy(update={"minimum_correlation": 0.95}))
        results = []
        for i in range(2):
            ds = _make_full_image_matching_dataset(n=1, seed=i)
            ds["correlation"] = (["measurement"], [0.3])
            results.append(ds)

        result = verify(setup, image_matching_results=results, work_dir=tmp_path)

        assert result.passed is False
        assert result.percent_within_threshold == 0.0
        assert [e.status for e in result.per_gcp_errors] == ["rejected", "rejected"]
        assert "total_measurements" not in result.aggregate_stats.attrs
        assert "0 accepted, 2 rejected" in result.summary_table

    def test_setup_configs_round_trip_through_json(self):
        import json

        from curryer.correction.config import PSFSamplingConfig, SearchConfig

        data = json.loads(_make_setup().model_dump_json())
        data["search"] = {"grid_size": 20, "grid_span_km": 40.0}
        data["psf_sampling"] = {"psf_lat_sample_dist_deg": 1e-4}
        setup = GeolocationSetup.model_validate_json(json.dumps(data))
        assert setup.search == SearchConfig(grid_size=20, grid_span_km=40.0)
        assert setup.psf_sampling == PSFSamplingConfig(psf_lat_sample_dist_deg=1e-4)
        assert GeolocationSetup.model_validate_json(setup.model_dump_json()) == setup

    @pytest.mark.parametrize("field", ["search", "psf_sampling"])
    def test_setup_config_rejects_unknown_keys(self, field):
        data = _make_setup().model_dump()
        data[field] = {"grid_spn_km": 40.0}
        with pytest.raises(ValidationError, match="grid_spn_km"):
            GeolocationSetup.model_validate(data)

    def test_image_matching_uses_setup_configs_and_records_correlation(self, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import patch

        from curryer.correction.config import PSFSamplingConfig, SearchConfig
        from curryer.correction.verification import image_matching

        grid, geolocated = TestViewingGeometryFailures._image_matching_inputs(with_frame=True)
        setup = _make_setup(
            psf_sampling=PSFSamplingConfig(psf_lat_sample_dist_deg=1e-4), search=SearchConfig(grid_size=20)
        )
        match = SimpleNamespace(
            lat_error_km=0.1,
            lon_error_km=0.1,
            ccv_final=0.77,
            ccv_secondary=0.5,
            final_grid_step_m=30.0,
            final_index_row=1,
            final_index_col=1,
        )
        with (
            patch("curryer.correction.verification.geolocated_to_image_grid", return_value=grid),
            patch("curryer.correction.verification.load_image_grid", return_value=grid),
            patch("curryer.correction.verification.integrated_image_match", return_value=match) as mock_match,
            patch(
                "curryer.correction.verification._get_spice_boresight_and_rotation",
                return_value=(np.array([0.0, 0.0, 1.0]), np.eye(3)),
            ),
        ):
            ds = image_matching(
                geolocated_data=geolocated,
                gcp_reference_file=tmp_path / "gcp.nc",
                setup=setup,
                los_vectors_cached=np.array([[0.0, 0.0, 1.0]]),
                optical_psfs_cached=[],
                r_iss_midframe=np.array([-1.5e6, -5.9e6, 3.0e6]),
            )
        assert mock_match.call_args.kwargs["geolocation_config"] is setup.psf_sampling
        assert mock_match.call_args.kwargs["search_config"] is setup.search
        assert float(ds["correlation"].values[0]) == pytest.approx(0.77)

    def test_verify_keeps_rejected_gcps_with_their_keys(self, tmp_path):
        """A rejected measurement stays in per-GCP errors, flagged, with its own keys."""
        setup = _make_setup(geo=_make_geo().model_copy(update={"minimum_correlation": 0.5}))
        results = []
        for i, ccv in enumerate((0.9, 0.1, 0.8)):
            ds = _make_full_image_matching_dataset(n=1, seed=i)
            ds["correlation"] = (["measurement"], [ccv])
            ds.attrs.update({"sci_key": f"sci_{i}", "gcp_key": f"gcp_{i}"})
            results.append(ds)

        result = verify(setup, image_matching_results=results, work_dir=tmp_path)

        rows = [(e.gcp_index, e.science_key, e.gcp_key, e.correlation) for e in result.per_gcp_errors]
        assert rows == [
            (0, "sci_0", "gcp_0", pytest.approx(0.9)),
            (1, "sci_1", "gcp_1", pytest.approx(0.1)),
            (2, "sci_2", "gcp_2", pytest.approx(0.8)),
        ]
        rejected = result.per_gcp_errors[1]
        assert rejected.status == "rejected"
        assert rejected.passed is False
        assert rejected.rejection_reason == "correlation 0.1000 < 0.5"
        assert rejected.nadir_equiv_error_m is not None
        assert all(e.status != "rejected" and e.rejection_reason is None for e in result.per_gcp_errors[::2])
        assert result.aggregate_stats.attrs["total_measurements"] == 2
        assert result.aggregate_stats.attrs["n_rejected"] == 1


class TestChipImagesSaveAndReview:
    """keep_images, save_verification / load_verification, and the review round trip."""

    @staticmethod
    def _result(tmp_path, correlations=(0.9, 0.1, 0.8)):
        setup = _make_setup(geo=_make_geo().model_copy(update={"minimum_correlation": 0.5}))
        results = []
        for i, ccv in enumerate(correlations):
            ds = _make_full_image_matching_dataset(n=1, seed=i)
            ds["correlation"] = (["measurement"], [ccv])
            ds.attrs.update({"sci_key": f"sci_{i}", "gcp_key": f"gcp_{i}"})
            results.append(ds)
        return verify(setup, image_matching_results=results, work_dir=tmp_path / "work")

    @staticmethod
    def _chips(n):
        return [
            xr.Dataset({"observed": (("row", "col"), np.full((2, 3), float(i)))}, attrs={"gcp_index": i})
            for i in range(n)
        ]

    def test_file_pair_matching_keeps_images(self, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import patch

        from curryer.compute.spatial import geodetic_to_ecef
        from curryer.correction.image_io import load_image_grid

        r_sc = geodetic_to_ecef(np.array([-102.33, 26.15, 410_000.0]), meters=True, degrees=True)
        obs = TestViewingGeometryFailures._write_grid_nc(tmp_path / "obs.nc", 26.15, -102.33, position_m=r_sc)
        gcp = TestViewingGeometryFailures._write_grid_nc(tmp_path / "gcp_regridded.nc", 26.15, -102.33)
        match = SimpleNamespace(
            lat_error_km=0.1,
            lon_error_km=-0.2,
            ccv_final=0.83,
            ccv_secondary=0.4,
            final_grid_step_m=30.0,
            convolved_gcp=load_image_grid(gcp),
        )
        from curryer.correction.verification import _run_image_matching_for_pairs

        with (
            patch("curryer.correction.image_io.load_los_vectors", return_value=np.tile([0.0, 0.0, 1.0], (5, 1))),
            patch("curryer.correction.image_io.load_optical_psf", return_value=[]),
            patch("curryer.correction.verification.integrated_image_match", return_value=match),
        ):
            _, kept = _run_image_matching_for_pairs(
                [(obs, gcp)], tmp_path / "los.mat", tmp_path / "psf.mat", _make_setup(), keep_images=True
            )
            _, not_kept = _run_image_matching_for_pairs(
                [(obs, gcp)], tmp_path / "los.mat", tmp_path / "psf.mat", _make_setup()
            )

        assert not_kept == []
        (chip,) = kept
        assert (chip.attrs["science_key"], chip.attrs["gcp_key"]) == ("obs.nc", "gcp_regridded.nc")
        assert chip["observed"].shape == (5, 5)
        assert chip.attrs["correlation"] == pytest.approx(0.83)

    def test_keep_images_with_precomputed_results_raises(self, tmp_path):
        with pytest.raises(ValueError, match="keep_images requires"):
            verify(
                _make_setup(),
                image_matching_results=[_make_full_image_matching_dataset(n=1)],
                work_dir=tmp_path,
                keep_images=True,
            )

    def test_chip_images_never_serialised(self, tmp_path):
        result = self._result(tmp_path).model_copy(update={"chip_images": self._chips(3)})
        assert "chip_images" not in json.loads(result.model_dump_json(exclude={"aggregate_stats"}))

    def test_save_and_load_round_trip(self, tmp_path):
        from curryer.correction import load_verification, save_verification

        result = self._result(tmp_path).model_copy(update={"chip_images": self._chips(3)})
        save_verification(result, tmp_path / "saved")

        assert sorted(p.name for p in (tmp_path / "saved").iterdir()) == [
            "aggregate_stats.nc",
            "chips",
            "result.json",
            "summary.csv",
        ]
        loaded = load_verification(tmp_path / "saved")
        assert loaded.per_gcp_errors == result.per_gcp_errors
        assert loaded.percent_within_threshold == result.percent_within_threshold
        np.testing.assert_array_equal(loaded.aggregate_stats["accepted"], result.aggregate_stats["accepted"])
        assert [int(c.attrs["gcp_index"]) for c in loaded.chip_images] == [0, 1, 2]
        np.testing.assert_array_equal(loaded.chip_images[2]["observed"], 2.0)

    def test_save_refuses_to_overwrite(self, tmp_path):
        from curryer.correction import save_verification

        result = self._result(tmp_path)
        save_verification(result, tmp_path / "saved")
        with pytest.raises(FileExistsError):
            save_verification(result, tmp_path / "saved")

    def test_summary_csv_review_column_round_trip(self, tmp_path):
        import pandas as pd

        from curryer.correction import read_review_decisions, save_verification

        save_verification(self._result(tmp_path), tmp_path / "saved")
        summary = pd.read_csv(tmp_path / "saved" / "summary.csv", keep_default_na=False)
        assert list(summary["status"])[1] == "rejected"
        assert summary["rejection_reason"][1] == "correlation 0.1000 < 0.5"
        assert list(summary["review"]) == ["", "", ""]
        assert read_review_decisions(tmp_path / "saved" / "summary.csv") == {}

        summary["review"] = ["reject", "accept", ""]
        summary.to_csv(tmp_path / "saved" / "summary.csv", index=False)
        assert read_review_decisions(tmp_path / "saved" / "summary.csv") == {0: "reject", 1: "accept"}

    @pytest.mark.parametrize(
        ("column", "value", "match"), [("review", "maybe", "expected 'accept'"), (None, None, "column")]
    )
    def test_read_review_decisions_rejects_bad_input(self, tmp_path, column, value, match):
        import pandas as pd

        from curryer.correction import read_review_decisions

        table = pd.DataFrame({"gcp_index": [0], "review": ["accept"]})
        if column is None:
            table = table.drop(columns="review")
        else:
            table[column] = value
        table.to_csv(tmp_path / "summary.csv", index=False)
        with pytest.raises(ValueError, match=match):
            read_review_decisions(tmp_path / "summary.csv")

    def test_apply_review_moves_gcps_in_and_out_of_the_statistics(self, tmp_path):
        from curryer.correction import apply_review

        result = self._result(tmp_path)
        reviewed = apply_review(result, {1: "accept", 2: "reject"})

        before = [e.status for e in result.per_gcp_errors]
        after = {e.gcp_index: e for e in reviewed.per_gcp_errors}
        assert before[1] == "rejected"
        assert after[1].status in ("pass", "fail")
        assert after[1].review == "accept"
        assert after[2].status == "rejected"
        assert after[2].rejection_reason == "rejected in review"
        assert after[0].review is None
        assert reviewed.aggregate_stats.attrs["total_measurements"] == 2
        assert reviewed.aggregate_stats.attrs["n_rejected"] == 1
        assert list(reviewed.aggregate_stats["review"].values) == ["", "accept", "reject"]
        assert [e.status for e in result.per_gcp_errors] == before  # input untouched

    def test_weighted_percent_reported_alongside(self, tmp_path):
        from curryer.correction.error_stats import match_snr_weight

        result = self._result(tmp_path)
        weights = {e.gcp_index: e.quality_weight for e in result.per_gcp_errors}
        assert weights[1] == 0.0
        assert weights[0] == pytest.approx(match_snr_weight(np.array([0.9]))[0])
        assert result.weighted_percent_within_threshold is not None
        assert "weighted" in result.summary_table
        assert "effective n" in result.summary_table

    def test_apply_review_reweights(self, tmp_path):
        from curryer.correction import apply_review
        from curryer.correction.error_stats import match_snr_weight

        reviewed = apply_review(self._result(tmp_path), {1: "accept", 2: "reject"})
        weights = {e.gcp_index: e.quality_weight for e in reviewed.per_gcp_errors}
        assert weights[1] == pytest.approx(match_snr_weight(np.array([0.1]))[0])
        assert weights[2] == 0.0
        assert reviewed.aggregate_stats.attrs["effective_measurements"] < 2.0

    def test_apply_review_rejecting_everything_drops_statistics(self, tmp_path):
        from curryer.correction import apply_review

        reviewed = apply_review(self._result(tmp_path), {0: "reject", 2: "reject"})
        assert reviewed.passed is False
        assert reviewed.percent_within_threshold == 0.0
        assert "total_measurements" not in reviewed.aggregate_stats.attrs
        assert "0 accepted, 3 rejected" in reviewed.summary_table
        assert reviewed.weighted_percent_within_threshold is None
        assert "weighted_mean_error_m" not in reviewed.aggregate_stats.attrs

    @pytest.mark.parametrize(("decisions", "match"), [({7: "accept"}, "not in the result"), ({0: "ok"}, "must be")])
    def test_apply_review_bad_decisions_raise(self, tmp_path, decisions, match):
        from curryer.correction import apply_review

        with pytest.raises(ValueError, match=match):
            apply_review(self._result(tmp_path), decisions)

    def test_review_contradicting_status_raises(self):
        with pytest.raises(ValueError, match="review='reject' contradicts"):
            GCPError(
                gcp_index=0,
                science_key="s",
                gcp_key="g",
                lat_error_deg=0.0,
                lon_error_deg=0.0,
                passed=True,
                status="pass",
                review="reject",
            )
