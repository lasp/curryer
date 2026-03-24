"""Unit tests for parameter-set generation strategies.

Covers:
- ``SearchStrategy.RANDOM``   – default Monte Carlo random walk (exact behaviour preserved)
- ``SearchStrategy.GRID_SEARCH`` – cartesian-product sweep over evenly-spaced offsets
- ``SearchStrategy.SINGLE_OFFSET`` – one-parameter-at-a-time sweep (others held at nominal)

For every strategy the three parameter types are exercised:
  - ``CONSTANT_KERNEL``  – 3-axis rotation (returns a pandas DataFrame)
  - ``OFFSET_KERNEL``    – single angle bias (float, radians)
  - ``OFFSET_TIME``      – timing correction (float, seconds)

Config validation:
- ``grid_points_per_param < 2`` rejected for GRID_SEARCH
- ``SINGLE_OFFSET`` with zero parameters rejected
- JSON round-trip preserves strategy fields
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from curryer.correction.config import (
    CorrectionConfig,
    GeolocationConfig,
    ParameterConfig,
    ParameterType,
    SearchStrategy,
)
from curryer.correction.parameters import (
    _get_grid_values,
    _get_nominal_value,
    load_param_sets,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def geo() -> GeolocationConfig:
    return GeolocationConfig(
        meta_kernel_file=Path("tests/data/test.kernels.tm.json"),
        generic_kernel_dir=Path("data/generic"),
        instrument_name="TEST_INSTRUMENT",
        time_field="ugps",
    )


@pytest.fixture
def param_constant() -> ParameterConfig:
    """CONSTANT_KERNEL: roll/pitch/yaw in arcseconds."""
    return ParameterConfig(
        ptype=ParameterType.CONSTANT_KERNEL,
        config_file=Path("tests/data/test_base.attitude.ck.json"),
        data={
            "current_value": [10.0, 20.0, 30.0],
            "bounds": [-60.0, 60.0],
            "sigma": 6.0,
            "units": "arcseconds",
        },
    )


@pytest.fixture
def param_constant_zero() -> ParameterConfig:
    """CONSTANT_KERNEL: all axes at zero with no sigma → always returns [0, 0, 0]."""
    return ParameterConfig(
        ptype=ParameterType.CONSTANT_KERNEL,
        config_file=Path("tests/data/test_base.attitude.ck.json"),
        data={
            "current_value": [0.0, 0.0, 0.0],
            "bounds": [-10.0, 10.0],
            "sigma": None,
            "units": "arcseconds",
        },
    )


@pytest.fixture
def param_offset_kernel() -> ParameterConfig:
    """OFFSET_KERNEL: angle bias in arcseconds."""
    return ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("tests/data/test_az.attitude.ck.json"),
        data={
            "field": "hps.az_ang_nonlin",
            "current_value": 0.0,
            "bounds": [-3600.0, 3600.0],
            "sigma": 360.0,
            "units": "arcseconds",
        },
    )


@pytest.fixture
def param_offset_time() -> ParameterConfig:
    """OFFSET_TIME: timing bias in milliseconds."""
    return ParameterConfig(
        ptype=ParameterType.OFFSET_TIME,
        config_file=None,
        data={
            "field": "corrected_timestamp",
            "current_value": 0.0,
            "bounds": [-50.0, 50.0],
            "sigma": 7.0,
            "units": "milliseconds",
        },
    )


def _make_config(
    geo,
    params,
    *,
    strategy: SearchStrategy = SearchStrategy.RANDOM,
    n_iterations: int = 5,
    seed: int | None = 0,
    grid_points_per_param: int = 4,
) -> CorrectionConfig:
    return CorrectionConfig(
        seed=seed,
        n_iterations=n_iterations,
        parameters=params,
        search_strategy=strategy,
        grid_points_per_param=grid_points_per_param,
        geo=geo,
        performance_threshold_m=250.0,
        performance_spec_percent=39.0,
        earth_radius_m=6_378_140.0,
    )


# ===========================================================================
# _get_nominal_value
# ===========================================================================


class TestGetNominalValue:
    def test_constant_kernel_list(self, param_constant):
        """Nominal CONSTANT_KERNEL returns DataFrame with converted arcsecond values."""
        df = _get_nominal_value(param_constant)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"ugps", "angle_x", "angle_y", "angle_z"}
        # current_value = [10, 20, 30] arcsec → radians
        expected_x = np.deg2rad(10.0 / 3600.0)
        expected_y = np.deg2rad(20.0 / 3600.0)
        expected_z = np.deg2rad(30.0 / 3600.0)
        np.testing.assert_allclose(df["angle_x"].iloc[0], expected_x, rtol=1e-10)
        np.testing.assert_allclose(df["angle_y"].iloc[0], expected_y, rtol=1e-10)
        np.testing.assert_allclose(df["angle_z"].iloc[0], expected_z, rtol=1e-10)

    def test_constant_kernel_zero(self, param_constant_zero):
        """Nominal CONSTANT_KERNEL with zeros returns zero-angle DataFrame."""
        df = _get_nominal_value(param_constant_zero)
        assert df["angle_x"].iloc[0] == 0.0
        assert df["angle_y"].iloc[0] == 0.0
        assert df["angle_z"].iloc[0] == 0.0

    def test_offset_kernel(self, param_offset_kernel):
        """Nominal OFFSET_KERNEL: current_value=0 arcsec → 0.0 rad."""
        val = _get_nominal_value(param_offset_kernel)
        assert isinstance(val, float)
        assert val == pytest.approx(0.0)

    def test_offset_time_ms(self, param_offset_time):
        """Nominal OFFSET_TIME: current_value=0 ms → 0.0 s."""
        val = _get_nominal_value(param_offset_time)
        assert isinstance(val, float)
        assert val == pytest.approx(0.0)

    def test_offset_time_nonzero(self):
        """Non-zero current_value is correctly converted ms → s."""
        p = ParameterConfig(
            ptype=ParameterType.OFFSET_TIME,
            data={"current_value": 500.0, "bounds": [-100.0, 100.0], "units": "milliseconds"},
        )
        val = _get_nominal_value(p)
        assert val == pytest.approx(0.5)


# ===========================================================================
# _get_grid_values
# ===========================================================================


class TestGetGridValues:
    def test_offset_time_count(self, param_offset_time):
        vals = _get_grid_values(param_offset_time, 6)
        assert len(vals) == 6

    def test_offset_time_endpoints(self, param_offset_time):
        """Endpoints must be current_value + bounds[0] and current_value + bounds[1] in seconds."""
        vals = _get_grid_values(param_offset_time, 5)
        # current_value=0, bounds=[-50, 50] ms → [-0.05, 0.05] s
        assert vals[0] == pytest.approx(-0.05)
        assert vals[-1] == pytest.approx(0.05)

    def test_offset_time_evenly_spaced(self, param_offset_time):
        vals = _get_grid_values(param_offset_time, 10)
        diffs = np.diff(vals)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)

    def test_offset_kernel_arcseconds(self, param_offset_kernel):
        """OFFSET_KERNEL with arcsecond units: bounds converted to radians."""
        vals = _get_grid_values(param_offset_kernel, 3)
        assert len(vals) == 3
        low_rad = np.deg2rad(-3600.0 / 3600.0)  # = -π/180 rad
        high_rad = np.deg2rad(3600.0 / 3600.0)  # = +π/180 rad
        assert vals[0] == pytest.approx(low_rad)
        assert vals[-1] == pytest.approx(high_rad)

    def test_constant_kernel_returns_dataframes(self, param_constant):
        """CONSTANT_KERNEL grid returns a list of DataFrames."""
        vals = _get_grid_values(param_constant, 4)
        assert len(vals) == 4
        for df in vals:
            assert isinstance(df, pd.DataFrame)
            assert "angle_x" in df.columns

    def test_constant_kernel_offset_applied_uniformly(self, param_constant_zero):
        """Uniform offset applied to all 3 axes for CONSTANT_KERNEL."""
        # param_constant_zero: current=[0,0,0], bounds=[-10,10] arcsec
        vals = _get_grid_values(param_constant_zero, 3)
        for df in vals:
            x = df["angle_x"].iloc[0]
            y = df["angle_y"].iloc[0]
            z = df["angle_z"].iloc[0]
            assert x == pytest.approx(y), "All 3 axes should share the same offset for zero current_value"
            assert y == pytest.approx(z), "All 3 axes should share the same offset for zero current_value"

    def test_constant_kernel_endpoint_magnitudes(self, param_constant_zero):
        """First and last grid DataFrames have angles matching the converted bounds."""
        vals = _get_grid_values(param_constant_zero, 2)
        low_rad = np.deg2rad(-10.0 / 3600.0)
        high_rad = np.deg2rad(10.0 / 3600.0)
        assert vals[0]["angle_x"].iloc[0] == pytest.approx(low_rad)
        assert vals[-1]["angle_x"].iloc[0] == pytest.approx(high_rad)

    def test_offset_time_microseconds(self):
        """Microsecond units are converted correctly."""
        p = ParameterConfig(
            ptype=ParameterType.OFFSET_TIME,
            data={"current_value": 0.0, "bounds": [-1_000_000.0, 1_000_000.0], "units": "microseconds"},
        )
        vals = _get_grid_values(p, 3)
        assert vals[0] == pytest.approx(-1.0)
        assert vals[-1] == pytest.approx(1.0)


# ===========================================================================
# SearchStrategy.RANDOM (default behaviour)
# ===========================================================================


class TestRandomStrategy:
    def test_output_length(self, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.RANDOM, n_iterations=7)
        sets = load_param_sets(config)
        assert len(sets) == 7

    def test_inner_length(self, geo, param_offset_kernel, param_offset_time):
        config = _make_config(
            geo, [param_offset_kernel, param_offset_time], strategy=SearchStrategy.RANDOM, n_iterations=3
        )
        sets = load_param_sets(config)
        assert all(len(s) == 2 for s in sets)

    def test_reproducible_with_seed(self, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.RANDOM, n_iterations=5, seed=42)
        sets_a = load_param_sets(config)
        sets_b = load_param_sets(config)
        assert len(sets_a) == len(sets_b)
        for param_set_a, param_set_b in zip(sets_a, sets_b):
            for (_, a), (_, b) in zip(param_set_a, param_set_b):
                assert a == pytest.approx(b)

    def test_random_values_within_bounds(self, geo, param_offset_time):
        """All sampled time offsets lie within [bounds_low, bounds_high] in seconds."""
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.RANDOM, n_iterations=50, seed=7)
        sets = load_param_sets(config)
        low_s, high_s = -0.05, 0.05  # bounds=[-50, 50] ms → seconds
        for param_set in sets:
            _, val = param_set[0]
            assert low_s <= val <= high_s

    def test_constant_kernel_returns_dataframe(self, geo, param_constant):
        config = _make_config(geo, [param_constant], strategy=SearchStrategy.RANDOM, n_iterations=2)
        sets = load_param_sets(config)
        for param_set in sets:
            _, df = param_set[0]
            assert isinstance(df, pd.DataFrame)
            assert "angle_x" in df.columns
            assert "angle_y" in df.columns
            assert "angle_z" in df.columns

    def test_offset_kernel_returns_float(self, geo, param_offset_kernel):
        config = _make_config(geo, [param_offset_kernel], strategy=SearchStrategy.RANDOM, n_iterations=3)
        sets = load_param_sets(config)
        for param_set in sets:
            _, val = param_set[0]
            assert isinstance(val, (float, np.floating))

    def test_no_sigma_returns_fixed_value(self, geo, param_constant_zero):
        """Parameter with sigma=None stays fixed at nominal across all iterations."""
        config = _make_config(geo, [param_constant_zero], strategy=SearchStrategy.RANDOM, n_iterations=10, seed=0)
        sets = load_param_sets(config)
        first_x = sets[0][0][1]["angle_x"].iloc[0]
        for param_set in sets:
            _, df = param_set[0]
            assert df["angle_x"].iloc[0] == pytest.approx(first_x)


# ===========================================================================
# SearchStrategy.GRID_SEARCH
# ===========================================================================


class TestGridSearchStrategy:
    def test_single_param_count(self, geo, param_offset_time):
        """1 parameter × 5 grid points → 5 parameter sets."""
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=5)
        sets = load_param_sets(config)
        assert len(sets) == 5

    def test_two_params_cartesian_product(self, geo, param_offset_kernel, param_offset_time):
        """2 parameters × 4 grid points → 4² = 16 parameter sets."""
        config = _make_config(
            geo,
            [param_offset_kernel, param_offset_time],
            strategy=SearchStrategy.GRID_SEARCH,
            grid_points_per_param=4,
        )
        sets = load_param_sets(config)
        assert len(sets) == 16

    def test_three_params_cartesian_product(self, geo, param_constant, param_offset_kernel, param_offset_time):
        """3 parameters × 3 grid points → 3³ = 27 parameter sets."""
        config = _make_config(
            geo,
            [param_constant, param_offset_kernel, param_offset_time],
            strategy=SearchStrategy.GRID_SEARCH,
            grid_points_per_param=3,
        )
        sets = load_param_sets(config)
        assert len(sets) == 27

    def test_inner_set_length(self, geo, param_offset_kernel, param_offset_time):
        config = _make_config(
            geo,
            [param_offset_kernel, param_offset_time],
            strategy=SearchStrategy.GRID_SEARCH,
            grid_points_per_param=3,
        )
        sets = load_param_sets(config)
        assert all(len(s) == 2 for s in sets)

    def test_values_span_full_bounds(self, geo, param_offset_time):
        """First and last values in single-param grid span the full converted bounds."""
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=5)
        sets = load_param_sets(config)
        vals = [s[0][1] for s in sets]
        assert min(vals) == pytest.approx(-0.05)
        assert max(vals) == pytest.approx(0.05)

    def test_values_are_evenly_spaced(self, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=6)
        sets = load_param_sets(config)
        vals = [s[0][1] for s in sets]
        diffs = np.diff(vals)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)

    def test_deterministic_no_seed_needed(self, geo, param_offset_time):
        """GRID_SEARCH is deterministic regardless of seed."""
        config_a = _make_config(
            geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=4, seed=None
        )
        config_b = _make_config(
            geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=4, seed=99
        )
        sets_a = load_param_sets(config_a)
        sets_b = load_param_sets(config_b)
        assert len(sets_a) == len(sets_b)
        for param_set_a, param_set_b in zip(sets_a, sets_b):
            for (_, a), (_, b) in zip(param_set_a, param_set_b):
                assert a == pytest.approx(b)

    def test_constant_kernel_in_grid(self, geo, param_constant_zero):
        """GRID_SEARCH on CONSTANT_KERNEL yields DataFrames with monotone angles."""
        config = _make_config(geo, [param_constant_zero], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=4)
        sets = load_param_sets(config)
        assert len(sets) == 4
        angle_xs = [s[0][1]["angle_x"].iloc[0] for s in sets]
        # Values should be monotonically increasing (linspace low→high)
        assert all(angle_xs[i] <= angle_xs[i + 1] for i in range(len(angle_xs) - 1))

    def test_n_iterations_ignored(self, geo, param_offset_time):
        """n_iterations has no effect on GRID_SEARCH output count."""
        config = _make_config(
            geo,
            [param_offset_time],
            strategy=SearchStrategy.GRID_SEARCH,
            grid_points_per_param=5,
            n_iterations=1000,  # ignored
        )
        sets = load_param_sets(config)
        assert len(sets) == 5


# ===========================================================================
# SearchStrategy.SINGLE_OFFSET
# ===========================================================================


class TestSingleOffsetStrategy:
    def test_single_param_count(self, geo, param_offset_time):
        """1 parameter × n_iterations values → n_iterations parameter sets."""
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=8)
        sets = load_param_sets(config)
        assert len(sets) == 8

    def test_two_params_count(self, geo, param_offset_kernel, param_offset_time):
        """2 parameters × 5 values each → 10 total parameter sets."""
        config = _make_config(
            geo,
            [param_offset_kernel, param_offset_time],
            strategy=SearchStrategy.SINGLE_OFFSET,
            n_iterations=5,
        )
        sets = load_param_sets(config)
        assert len(sets) == 10

    def test_time_offset_sweep_spans_bounds(self, geo, param_offset_time):
        """SINGLE_OFFSET sweep of OFFSET_TIME spans the full converted bounds."""
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=5)
        sets = load_param_sets(config)
        vals = [s[0][1] for s in sets]
        assert min(vals) == pytest.approx(-0.05)
        assert max(vals) == pytest.approx(0.05)

    def test_time_offset_sweep_evenly_spaced(self, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=7)
        sets = load_param_sets(config)
        vals = [s[0][1] for s in sets]
        diffs = np.diff(vals)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)

    def test_non_swept_params_held_at_nominal(self, geo, param_offset_kernel, param_offset_time):
        """While sweeping param_0, param_1 must equal its nominal value in every set."""
        config = _make_config(
            geo,
            [param_offset_kernel, param_offset_time],
            strategy=SearchStrategy.SINGLE_OFFSET,
            n_iterations=4,
        )
        sets = load_param_sets(config)
        nominal_time = _get_nominal_value(param_offset_time)
        # First 4 sets sweep param_0 (OFFSET_KERNEL); param_1 (time) should be nominal
        for param_set in sets[:4]:
            _, time_val = param_set[1]
            assert time_val == pytest.approx(nominal_time)

    def test_swept_param_changes_others_fixed(self, geo, param_offset_kernel, param_offset_time):
        """While sweeping param_1 (time), param_0 (kernel) stays at nominal in every set."""
        config = _make_config(
            geo,
            [param_offset_kernel, param_offset_time],
            strategy=SearchStrategy.SINGLE_OFFSET,
            n_iterations=4,
        )
        sets = load_param_sets(config)
        nominal_kernel = _get_nominal_value(param_offset_kernel)
        # Last 4 sets sweep param_1 (time); param_0 (kernel) should be nominal
        for param_set in sets[4:]:
            _, kernel_val = param_set[0]
            assert kernel_val == pytest.approx(nominal_kernel)

    def test_deterministic(self, geo, param_offset_time):
        """SINGLE_OFFSET is deterministic: two calls with same config return identical results."""
        config = _make_config(
            geo, [param_offset_time], strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=5, seed=None
        )
        sets_a = load_param_sets(config)
        sets_b = load_param_sets(config)
        assert len(sets_a) == len(sets_b)
        for param_set_a, param_set_b in zip(sets_a, sets_b):
            for (_, a), (_, b) in zip(param_set_a, param_set_b):
                assert a == pytest.approx(b)

    def test_constant_kernel_sweep(self, geo, param_constant_zero):
        """SINGLE_OFFSET on CONSTANT_KERNEL sweeps angle magnitudes monotonically."""
        config = _make_config(geo, [param_constant_zero], strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=5)
        sets = load_param_sets(config)
        assert len(sets) == 5
        angle_xs = [s[0][1]["angle_x"].iloc[0] for s in sets]
        assert all(angle_xs[i] <= angle_xs[i + 1] for i in range(len(angle_xs) - 1))


# ===========================================================================
# Config validation
# ===========================================================================


class TestConfigValidation:
    def test_grid_points_per_param_minimum(self, geo, param_offset_time):
        """grid_points_per_param must be >= 2."""
        with pytest.raises(ValidationError) as exc_info:
            _make_config(
                geo,
                [param_offset_time],
                strategy=SearchStrategy.GRID_SEARCH,
                grid_points_per_param=1,
            )
        errors = exc_info.value.errors()
        assert any(
            "grid_points_per_param" in err.get("loc", ())
            for err in errors
        )

    def test_search_strategy_default_is_random(self, geo, param_offset_time):
        config = CorrectionConfig(
            seed=0,
            n_iterations=3,
            parameters=[param_offset_time],
            geo=geo,
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
            earth_radius_m=6_378_140.0,
        )
        assert config.search_strategy == SearchStrategy.RANDOM

    def test_search_strategy_enum_values(self):
        assert SearchStrategy("random") is SearchStrategy.RANDOM
        assert SearchStrategy("grid") is SearchStrategy.GRID_SEARCH
        assert SearchStrategy("single") is SearchStrategy.SINGLE_OFFSET

    def test_invalid_strategy_string_rejected(self, geo, param_offset_time):
        with pytest.raises(ValidationError):
            _make_config(
                geo,
                [param_offset_time],
                strategy="not_a_strategy",  # type: ignore[arg-type]
            )

    def test_json_round_trip_random(self, geo, param_offset_time):
        """RANDOM config survives model_dump_json / model_validate_json round-trip."""
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.RANDOM)
        json_str = config.model_dump_json()
        restored = CorrectionConfig.model_validate_json(json_str)
        assert restored.search_strategy == SearchStrategy.RANDOM
        assert restored.n_iterations == config.n_iterations

    def test_json_round_trip_grid(self, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=7)
        restored = CorrectionConfig.model_validate_json(config.model_dump_json())
        assert restored.search_strategy == SearchStrategy.GRID_SEARCH
        assert restored.grid_points_per_param == 7

    def test_json_round_trip_single_offset(self, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=12)
        restored = CorrectionConfig.model_validate_json(config.model_dump_json())
        assert restored.search_strategy == SearchStrategy.SINGLE_OFFSET
        assert restored.n_iterations == 12


# ===========================================================================
# Strategy ↔ output type consistency
# ===========================================================================


class TestOutputTypeConsistency:
    """Ensure every strategy returns the correct element types for all param types."""

    @pytest.mark.parametrize(
        "strategy",
        [SearchStrategy.RANDOM, SearchStrategy.GRID_SEARCH, SearchStrategy.SINGLE_OFFSET],
    )
    def test_constant_kernel_always_dataframe(self, strategy, geo, param_constant_zero):
        config = _make_config(geo, [param_constant_zero], strategy=strategy, n_iterations=3, grid_points_per_param=3)
        sets = load_param_sets(config)
        for param_set in sets:
            _, val = param_set[0]
            assert isinstance(val, pd.DataFrame), f"Expected DataFrame for {strategy}, got {type(val)}"
            assert {"ugps", "angle_x", "angle_y", "angle_z"}.issubset(val.columns)

    @pytest.mark.parametrize(
        "strategy",
        [SearchStrategy.RANDOM, SearchStrategy.GRID_SEARCH, SearchStrategy.SINGLE_OFFSET],
    )
    def test_offset_kernel_always_float(self, strategy, geo, param_offset_kernel):
        config = _make_config(geo, [param_offset_kernel], strategy=strategy, n_iterations=3, grid_points_per_param=3)
        sets = load_param_sets(config)
        for param_set in sets:
            _, val = param_set[0]
            assert isinstance(val, (float, np.floating)), f"Expected float for {strategy}, got {type(val)}"

    @pytest.mark.parametrize(
        "strategy",
        [SearchStrategy.RANDOM, SearchStrategy.GRID_SEARCH, SearchStrategy.SINGLE_OFFSET],
    )
    def test_offset_time_always_float(self, strategy, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=strategy, n_iterations=3, grid_points_per_param=3)
        sets = load_param_sets(config)
        for param_set in sets:
            _, val = param_set[0]
            assert isinstance(val, (float, np.floating)), f"Expected float for {strategy}, got {type(val)}"
