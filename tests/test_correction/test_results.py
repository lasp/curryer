"""Tests for ``curryer.correction.results``.

Covers
------
- :class:`ParameterSetResult` — construction and field access
- :class:`CorrectionResult` — construction, JSON serialisation,
  raw-data access, ``results``/``netcdf_data`` exclusion from JSON
- :func:`_format_correction_summary_table` — well-formed box-drawn output
- :func:`build_correction_result` — met/not-met threshold, all-NaN fallback
- :func:`compare_results` — side-by-side output format
- :class:`VerificationResult` provenance fields — defaults and population
- Backward-compat: ``loop()`` still returns a 2-tuple
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from curryer.correction.config import (
    CorrectionConfig,
    GeolocationConfig,
    ParameterConfig,
    ParameterType,
    RequirementsConfig,
)
from curryer.correction.results import (
    CorrectionResult,
    ParameterSetResult,
    _fmt_rms,
    _format_correction_summary_table,
    build_correction_result,
)
from curryer.correction.verification import (
    VerificationResult,
    compare_results,
)

# ===========================================================================
# Shared helpers
# ===========================================================================

_THRESHOLD_M = 250.0
_SPEC_PCT = 39.0


def _make_geo() -> GeolocationConfig:
    return GeolocationConfig(
        meta_kernel_file=Path("tests/data/test.kernels.tm.json"),
        generic_kernel_dir=Path("data/generic"),
        instrument_name="TEST_INSTRUMENT",
        time_field="corrected_timestamp",
    )


def _make_config(**overrides) -> CorrectionConfig:
    defaults = dict(
        n_iterations=3,
        parameters=[
            ParameterConfig(
                ptype=ParameterType.CONSTANT_KERNEL,
                spec={"current_value": [0.0, 0.0, 0.0], "bounds": [-300.0, 300.0]},
            )
        ],
        geo=_make_geo(),
        performance_threshold_m=_THRESHOLD_M,
        performance_spec_percent=_SPEC_PCT,
    )
    defaults.update(overrides)
    return CorrectionConfig(**defaults)


def _make_netcdf_data(n_params: int = 3, n_pairs: int = 2, threshold_m: float = 250.0) -> dict:
    """Synthetic netcdf_data mimicking the structure built by loop()."""
    rms_grid = np.array([[100.0 + i * 50 + j * 10 for j in range(n_pairs)] for i in range(n_params)])
    mean_rms = rms_grid.mean(axis=1)
    best_pair_rms = rms_grid.min(axis=1)
    worst_pair_rms = rms_grid.max(axis=1)
    return {
        "parameter_set_id": np.arange(n_params),
        "gcp_pair_id": np.arange(n_pairs),
        "param_test_kernel_roll": np.linspace(-1.0, 1.0, n_params),
        "param_test_kernel_pitch": np.linspace(-0.5, 0.5, n_params),
        "rms_error_m": rms_grid,
        "mean_error_m": rms_grid,
        "max_error_m": rms_grid * 1.2,
        "std_error_m": np.zeros((n_params, n_pairs)),
        "n_measurements": np.ones((n_params, n_pairs), dtype=int) * 5,
        "mean_rms_all_pairs": mean_rms,
        "best_pair_rms": best_pair_rms,
        "worst_pair_rms": worst_pair_rms,
        f"percent_under_{int(threshold_m)}m": np.zeros(n_params),
        "im_lat_error_km": np.zeros((n_params, n_pairs)),
        "im_lon_error_km": np.zeros((n_params, n_pairs)),
        "im_ccv": np.ones((n_params, n_pairs)) * 0.9,
        "im_grid_step_m": np.ones((n_params, n_pairs)) * 30.0,
    }


def _make_verification_result(
    passed: bool = True,
    pct: float = 60.0,
    stats: dict | None = None,
) -> VerificationResult:
    ds = xr.Dataset(
        {"nadir_equiv_total_error_m": (["measurement"], np.array([100.0, 200.0, 300.0]))},
        coords={"measurement": np.arange(3)},
    )
    if stats:
        ds.attrs.update(stats)
    return VerificationResult(
        passed=passed,
        per_gcp_errors=[],
        aggregate_stats=ds,
        requirements=RequirementsConfig(
            performance_threshold_m=_THRESHOLD_M,
            performance_spec_percent=_SPEC_PCT,
        ),
        summary_table="",
        percent_within_threshold=pct,
        warnings=[] if passed else ["FAILED"],
        timestamp=datetime.now(tz=timezone.utc),
    )


# ===========================================================================
# _fmt_rms
# ===========================================================================


def test_fmt_rms_finite():
    assert _fmt_rms(123.456) == "123.5m"


def test_fmt_rms_nan():
    assert _fmt_rms(float("nan")) == "N/A"


def test_fmt_rms_inf():
    assert _fmt_rms(float("inf")) == "N/A"


# ===========================================================================
# ParameterSetResult
# ===========================================================================


class TestParameterSetResult:
    def test_construction(self):
        ps = ParameterSetResult(
            index=2,
            parameter_values={"roll": 1.5, "pitch": -0.3},
            mean_rms_m=150.0,
            best_pair_rms_m=100.0,
            worst_pair_rms_m=200.0,
        )
        assert ps.index == 2
        assert ps.parameter_values == {"roll": 1.5, "pitch": -0.3}
        assert ps.mean_rms_m == pytest.approx(150.0)
        assert ps.best_pair_rms_m == pytest.approx(100.0)
        assert ps.worst_pair_rms_m == pytest.approx(200.0)

    def test_json_round_trip(self):
        ps = ParameterSetResult(
            index=0,
            parameter_values={"roll": 0.5},
            mean_rms_m=200.0,
            best_pair_rms_m=180.0,
            worst_pair_rms_m=220.0,
        )
        restored = ParameterSetResult.model_validate_json(ps.model_dump_json())
        assert restored.index == 0
        assert restored.mean_rms_m == pytest.approx(200.0)


# ===========================================================================
# CorrectionResult
# ===========================================================================


class TestCorrectionResult:
    def _make(self, **overrides) -> CorrectionResult:
        defaults = dict(
            best_parameter_set={"roll": 0.1},
            best_rms_m=120.0,
            best_index=0,
            worst_rms_m=300.0,
            mean_rms_m=200.0,
            n_parameter_sets=3,
            n_gcp_pairs=2,
            all_parameter_sets=[
                ParameterSetResult(
                    index=0,
                    parameter_values={"roll": 0.1},
                    mean_rms_m=120.0,
                    best_pair_rms_m=100.0,
                    worst_pair_rms_m=140.0,
                )
            ],
            met_threshold=True,
            recommendation="Update kernel files.",
            summary_table="(table)",
        )
        defaults.update(overrides)
        return CorrectionResult(**defaults)

    def test_construction_minimal(self):
        result = self._make()
        assert result.best_rms_m == pytest.approx(120.0)
        assert result.met_threshold is True
        assert result.n_parameter_sets == 3

    def test_optional_fields_defaults(self):
        result = self._make()
        assert result.netcdf_path is None
        assert result.config_snapshot == {}
        assert result.elapsed_time_s == pytest.approx(0.0)
        assert isinstance(result.timestamp, datetime)

    def test_raw_data_fields_accessible(self):
        raw_results = [{"iteration": 0, "rms_error_m": 100.0}]
        raw_netcdf = {"mean_rms_all_pairs": np.array([100.0, 150.0])}
        result = self._make(results=raw_results, netcdf_data=raw_netcdf)
        # Pydantic copies list/dict on model init; verify by value not identity
        assert result.results == raw_results
        assert list(result.netcdf_data.keys()) == list(raw_netcdf.keys())

    def test_json_excludes_raw_data(self):
        """results and netcdf_data must not appear in JSON output."""
        result = self._make(
            results=[{"iteration": 0}],
            netcdf_data={"mean_rms_all_pairs": np.array([100.0])},
        )
        d = result.model_dump()
        assert "results" not in d
        assert "netcdf_data" not in d

    def test_json_round_trip_scalar_fields(self):
        result = self._make(
            netcdf_path=Path("/tmp/out.nc"),
            config_snapshot={"seed": 42, "n_iterations": 10},
            elapsed_time_s=3.14,
        )
        json_str = result.model_dump_json()
        data = json.loads(json_str)
        assert data["best_rms_m"] == pytest.approx(120.0)
        assert data["met_threshold"] is True
        assert data["elapsed_time_s"] == pytest.approx(3.14)
        assert data["netcdf_path"] == "/tmp/out.nc"

    def test_timestamp_is_utc_datetime(self):
        result = self._make()
        assert isinstance(result.timestamp, datetime)
        # default_factory uses UTC
        assert result.timestamp.tzinfo is not None


# ===========================================================================
# _format_correction_summary_table
# ===========================================================================


def _make_ps(idx: int, rms: float) -> ParameterSetResult:
    return ParameterSetResult(
        index=idx,
        parameter_values={"roll": float(idx)},
        mean_rms_m=rms,
        best_pair_rms_m=rms * 0.9,
        worst_pair_rms_m=rms * 1.1,
    )


class TestFormatCorrectionSummaryTable:
    def test_basic_structure(self):
        sets = [_make_ps(0, 120.0), _make_ps(1, 200.0)]
        table = _format_correction_summary_table(sets, total_sets=5, n_gcp_pairs=2, met_threshold=True)
        lines = table.splitlines()
        # First line starts with ┌ and last with └
        assert lines[0].startswith("┌")
        assert lines[-1].startswith("└")
        # All lines have same length
        lengths = {len(line) for line in lines}
        assert len(lengths) == 1, f"Inconsistent line widths: {sorted(lengths)}"

    def test_met_threshold_shows_in_footer(self):
        sets = [_make_ps(0, 100.0)]
        table = _format_correction_summary_table(sets, 1, 1, met_threshold=True)
        assert "MET REQUIREMENTS" in table
        assert "✓" in table

    def test_not_met_shows_in_footer(self):
        sets = [_make_ps(0, 500.0)]
        table = _format_correction_summary_table(sets, 1, 1, met_threshold=False)
        assert "DID NOT MEET" in table
        assert "✗" in table

    def test_empty_sets_no_crash(self):
        table = _format_correction_summary_table([], total_sets=0, n_gcp_pairs=0, met_threshold=False)
        assert "No results available" in table
        lines = table.splitlines()
        lengths = {len(line) for line in lines}
        assert len(lengths) == 1

    def test_best_set_has_star_marker(self):
        sets = [_make_ps(0, 80.0), _make_ps(1, 150.0)]
        table = _format_correction_summary_table(sets, 2, 1, met_threshold=True)
        assert "★" in table

    def test_nan_values_display_as_na(self):
        sets = [_make_ps(0, float("nan"))]
        table = _format_correction_summary_table(sets, 1, 1, met_threshold=False)
        assert "N/A" in table

    def test_long_title_widens_table_consistently(self):
        """Wide title (many GCP pairs) must not break line-width consistency."""
        sets = [_make_ps(i, 100.0 + i * 10) for i in range(5)]
        table = _format_correction_summary_table(sets, total_sets=10000, n_gcp_pairs=99, met_threshold=True)
        lines = table.splitlines()
        lengths = {len(line) for line in lines}
        assert len(lengths) == 1

    def test_title_appears_in_output(self):
        sets = [_make_ps(0, 200.0)]
        table = _format_correction_summary_table(sets, total_sets=7, n_gcp_pairs=3, met_threshold=False)
        assert "7 sets" in table
        assert "3 pairs" in table


# ===========================================================================
# build_correction_result
# ===========================================================================


class TestBuildCorrectionResult:
    def test_basic_construction(self):
        config = _make_config()
        nc = _make_netcdf_data(n_params=3, n_pairs=2)
        result = build_correction_result(config, [], nc, Path("/tmp/out.nc"), elapsed_time_s=1.5)

        assert isinstance(result, CorrectionResult)
        assert result.n_parameter_sets == 3
        assert result.n_gcp_pairs == 2
        assert result.elapsed_time_s == pytest.approx(1.5)
        assert result.netcdf_path == Path("/tmp/out.nc")

    def test_best_index_is_lowest_mean_rms(self):
        config = _make_config()
        nc = _make_netcdf_data(n_params=3, n_pairs=2)
        result = build_correction_result(config, [], nc, None, 0.0)
        # Mean RMS values are 105, 155, 205 (from the helper); best is index 0
        assert result.best_index == 0
        assert result.best_rms_m < result.worst_rms_m

    def test_all_sets_sorted_ascending(self):
        config = _make_config()
        nc = _make_netcdf_data(n_params=5, n_pairs=2)
        result = build_correction_result(config, [], nc, None, 0.0)
        rms_values = [ps.mean_rms_m for ps in result.all_parameter_sets]
        assert rms_values == sorted(rms_values)

    def test_met_threshold_true_when_all_pairs_below(self):
        """If all pair RMS < threshold, pct_below = 100% ≥ spec → met."""
        # Use very large threshold so all errors are "below"
        config = _make_config(
            performance_threshold_m=10_000.0,
            performance_spec_percent=39.0,
        )
        nc = _make_netcdf_data(n_params=2, n_pairs=2, threshold_m=10_000.0)
        result = build_correction_result(config, [], nc, None, 0.0)
        assert result.met_threshold is True
        assert "meets performance requirements" in result.recommendation

    def test_met_threshold_false_when_no_pairs_below(self):
        """If no pair RMS < threshold, pct_below = 0% < spec → not met."""
        config = _make_config(
            performance_threshold_m=0.001,  # impossibly tight
            performance_spec_percent=39.0,
        )
        nc = _make_netcdf_data(n_params=2, n_pairs=2, threshold_m=0)
        result = build_correction_result(config, [], nc, None, 0.0)
        assert result.met_threshold is False
        assert "No parameter set met performance requirements" in result.recommendation

    def test_all_nan_rms_does_not_crash(self):
        """Degenerate case: all RMS values are NaN — build should not raise."""
        config = _make_config()
        nc = _make_netcdf_data(n_params=2, n_pairs=2)
        nc["mean_rms_all_pairs"] = np.array([float("nan"), float("nan")])
        nc["rms_error_m"][:] = float("nan")
        result = build_correction_result(config, [], nc, None, 0.0)
        assert math.isnan(result.best_rms_m)
        assert math.isnan(result.worst_rms_m)
        # Table should still render without crash
        assert "┌" in result.summary_table

    def test_raw_data_preserved(self):
        config = _make_config()
        nc = _make_netcdf_data()
        raw = [{"iteration": 0, "rms_error_m": 100.0}]
        result = build_correction_result(config, raw, nc, None, 0.0)
        # Pydantic copies list/dict on model init; verify by value not identity
        assert result.results == raw
        assert set(result.netcdf_data.keys()) == set(nc.keys())

    def test_config_snapshot_contains_key_fields(self):
        config = _make_config()
        nc = _make_netcdf_data()
        result = build_correction_result(config, [], nc, None, 0.0)
        snap = result.config_snapshot
        assert "n_iterations" in snap
        assert "performance_threshold_m" in snap
        assert "performance_spec_percent" in snap
        assert "search_strategy" in snap

    def test_summary_table_included(self):
        config = _make_config()
        nc = _make_netcdf_data()
        result = build_correction_result(config, [], nc, None, 0.0)
        assert "┌" in result.summary_table
        assert "Correction Sweep Summary" in result.summary_table


# ===========================================================================
# compare_results
# ===========================================================================


class TestCompareResults:
    def test_basic_output_format(self):
        before = _make_verification_result(passed=False, pct=25.0)
        after = _make_verification_result(passed=True, pct=65.0)
        output = compare_results(before, after)

        assert "Verification Comparison" in output
        assert "Before" in output
        assert "After" in output
        assert "Overall" in output
        assert "PASS" in output
        assert "FAIL" in output

    def test_percent_within_threshold_shown(self):
        before = _make_verification_result(pct=30.0)
        after = _make_verification_result(pct=70.0)
        output = compare_results(before, after)
        assert "percent_within_threshold" in output
        # Both values should appear
        assert "30.0%" in output
        assert "70.0%" in output

    def test_aggregate_stats_attrs_used(self):
        before = _make_verification_result(stats={"mean_error_m": 350.0, "rms_error_m": 400.0})
        after = _make_verification_result(stats={"mean_error_m": 120.0, "rms_error_m": 150.0})
        output = compare_results(before, after)
        assert "mean_error_m" in output
        assert "350.0" in output
        assert "120.0" in output

    def test_missing_stat_shows_na(self):
        """Stats not in aggregate_stats.attrs should display as N/A."""
        before = _make_verification_result(stats={})
        after = _make_verification_result(stats={})
        output = compare_results(before, after)
        # All stat rows should show N/A since attrs is empty
        assert "N/A" in output

    def test_returns_string(self):
        before = _make_verification_result()
        after = _make_verification_result()
        result = compare_results(before, after)
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# VerificationResult provenance fields
# ===========================================================================


class TestVerificationResultProvenanceFields:
    def test_default_files_processed_is_empty_list(self):
        vr = _make_verification_result()
        assert vr.files_processed == []

    def test_default_elapsed_time_s_is_none(self):
        vr = _make_verification_result()
        assert vr.elapsed_time_s is None

    def test_default_config_snapshot_is_none(self):
        vr = _make_verification_result()
        assert vr.config_snapshot is None

    def test_provenance_fields_set_explicitly(self):
        vr = _make_verification_result()
        vr_with_prov = vr.model_copy(
            update={
                "files_processed": ["sci_0+gcp_0", "sci_1+gcp_1"],
                "elapsed_time_s": 2.5,
                "config_snapshot": {"instrument_name": "TEST"},
            }
        )
        assert vr_with_prov.files_processed == ["sci_0+gcp_0", "sci_1+gcp_1"]
        assert vr_with_prov.elapsed_time_s == pytest.approx(2.5)
        assert vr_with_prov.config_snapshot == {"instrument_name": "TEST"}

    def test_existing_fields_unaffected_by_new_fields(self):
        """Old construction sites that don't supply provenance fields still work."""
        vr = VerificationResult(
            passed=True,
            per_gcp_errors=[],
            aggregate_stats=xr.Dataset(
                {"nadir_equiv_total_error_m": (["measurement"], np.array([100.0]))},
                coords={"measurement": np.arange(1)},
            ),
            requirements=RequirementsConfig(
                performance_threshold_m=250.0,
                performance_spec_percent=39.0,
            ),
            summary_table="",
            percent_within_threshold=100.0,
            warnings=[],
            timestamp=datetime.now(tz=timezone.utc),
            # provenance fields intentionally omitted
        )
        assert vr.passed is True
        assert vr.files_processed == []
        assert vr.elapsed_time_s is None
        assert vr.config_snapshot is None

    def test_json_round_trip_with_provenance_fields(self):
        vr = _make_verification_result()
        vr = vr.model_copy(
            update={
                "files_processed": ["sci+gcp"],
                "elapsed_time_s": 1.23,
                "config_snapshot": {"instrument_name": "CLARREO"},
            }
        )
        json_str = vr.model_dump_json(exclude={"aggregate_stats"})
        data = json.loads(json_str)
        assert data["files_processed"] == ["sci+gcp"]
        assert data["elapsed_time_s"] == pytest.approx(1.23)
        assert data["config_snapshot"]["instrument_name"] == "CLARREO"


# ===========================================================================
# Backward compatibility: loop() still returns (results, netcdf_data)
# ===========================================================================


def test_loop_return_annotation_is_not_correction_result():
    """loop() must NOT be annotated to return CorrectionResult.

    This is a static check — loop() is the internal workhorse and its
    2-tuple return type must remain stable.
    """
    import inspect

    from curryer.correction.pipeline import loop

    sig = inspect.signature(loop)
    ann = sig.return_annotation
    # No annotation (empty) is fine; CorrectionResult annotation is wrong
    assert ann is inspect.Parameter.empty or "CorrectionResult" not in str(ann)


def test_run_correction_return_annotation_is_correction_result():
    """run_correction() must be annotated to return CorrectionResult."""
    import inspect

    from curryer.correction.pipeline import run_correction

    sig = inspect.signature(run_correction)
    ann = sig.return_annotation
    assert "CorrectionResult" in str(ann)
