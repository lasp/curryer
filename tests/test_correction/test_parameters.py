"""Unit tests for parameter-set generation strategies.

Covers:
- ``SearchStrategy.RANDOM``   – default Monte Carlo random walk (exact behaviour preserved)
- ``SearchStrategy.GRID_SEARCH`` – cartesian-product sweep over evenly-spaced offsets
- ``SearchStrategy.SINGLE_OFFSET`` – one-parameter-at-a-time sweep (others held at nominal)

For every strategy the three parameter types are exercised:
  - ``CONSTANT_KERNEL``  – one rotation angle of a frame (float, radians)
  - ``OFFSET_KERNEL``    – single angle bias (float, radians)
  - ``OFFSET_TIME``      – timing correction (float, seconds)

Config validation:
- ``grid_points_per_param < 2`` rejected for GRID_SEARCH
- JSON round-trip preserves strategy fields
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from curryer.correction.config import (
    GeolocationConfig,
    ParameterConfig,
    ParameterType,
    SearchStrategy,
    Sweep,
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


_AXES = ("angle_x", "angle_y", "angle_z")


def _frame(current_values, bounds, sigma) -> list[ParameterConfig]:
    """One CONSTANT_KERNEL parameter per axis of a test frame kernel, in arcseconds."""
    return [
        ParameterConfig(
            ptype=ParameterType.CONSTANT_KERNEL,
            config_file=Path("tests/data/test_base.attitude.ck.json"),
            spec={"field": axis, "current_value": value, "bounds": bounds, "sigma": sigma, "units": "arcseconds"},
        )
        for axis, value in zip(_AXES, current_values)
    ]


@pytest.fixture
def frame_constant() -> list[ParameterConfig]:
    """CONSTANT_KERNEL frame: roll/pitch/yaw 10/20/30 arcseconds, sampled."""
    return _frame([10.0, 20.0, 30.0], [-60.0, 60.0], 6.0)


@pytest.fixture
def frame_constant_zero() -> list[ParameterConfig]:
    """CONSTANT_KERNEL frame: all axes at zero with no sigma."""
    return _frame([0.0, 0.0, 0.0], [-10.0, 10.0], None)


def _arcsec(value):
    return np.deg2rad(np.asarray(value) / 3600.0)


@pytest.fixture
def param_offset_kernel() -> ParameterConfig:
    """OFFSET_KERNEL: angle bias in arcseconds."""
    return ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("tests/data/test_az.attitude.ck.json"),
        spec={
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
        spec={
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
) -> Sweep:
    # `geo` is retained for call-site readability but is not part of a Sweep.
    return Sweep(
        seed=seed,
        n_iterations=n_iterations,
        parameters=params,
        search_strategy=strategy,
        grid_points_per_param=grid_points_per_param,
    )


# ===========================================================================
# _get_nominal_value
# ===========================================================================


class TestGetNominalValue:
    def test_constant_kernel_axis_in_radians(self, frame_constant):
        """Each CONSTANT_KERNEL axis's nominal value is its own current_value in radians."""
        for param, expected in zip(frame_constant, [10.0, 20.0, 30.0]):
            assert _get_nominal_value(param) == pytest.approx(_arcsec(expected), rel=1e-12)

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
            spec={"current_value": 500.0, "bounds": [-100.0, 100.0], "units": "milliseconds"},
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

    def test_constant_kernel_spans_bounds_around_its_axis(self, frame_constant):
        """A CONSTANT_KERNEL axis's grid is its current_value plus the bounds, in radians."""
        vals = _get_grid_values(frame_constant[1], 3)
        np.testing.assert_allclose(vals, _arcsec([20.0 - 60.0, 20.0, 20.0 + 60.0]), rtol=1e-12)

    def test_offset_time_microseconds(self):
        """Microsecond units are converted correctly."""
        p = ParameterConfig(
            ptype=ParameterType.OFFSET_TIME,
            spec={"current_value": 0.0, "bounds": [-1_000_000.0, 1_000_000.0], "units": "microseconds"},
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

    def test_constant_kernel_axes_drawn_independently(self, geo, frame_constant):
        config = _make_config(geo, frame_constant, strategy=SearchStrategy.RANDOM, n_iterations=4)
        offsets = np.array([[val for _, val in param_set] for param_set in load_param_sets(config)])
        offsets -= _arcsec([10.0, 20.0, 30.0])
        assert np.all(np.abs(offsets) <= _arcsec(60.0))
        assert len(np.unique(np.round(offsets / _arcsec(1e-6)))) == offsets.size

    def test_offset_kernel_returns_float(self, geo, param_offset_kernel):
        config = _make_config(geo, [param_offset_kernel], strategy=SearchStrategy.RANDOM, n_iterations=3)
        sets = load_param_sets(config)
        for param_set in sets:
            _, val = param_set[0]
            assert isinstance(val, (float, np.floating))

    def test_no_sigma_returns_fixed_value(self, geo, frame_constant_zero):
        """Parameters with sigma=None stay fixed at nominal across all iterations."""
        config = _make_config(geo, frame_constant_zero, strategy=SearchStrategy.RANDOM, n_iterations=10, seed=0)
        for param_set in load_param_sets(config):
            assert [val for _, val in param_set] == [0.0, 0.0, 0.0]


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

    def test_five_params_cartesian_product(self, geo, frame_constant, param_offset_kernel, param_offset_time):
        """A frame's 3 axes + 2 parameters × 2 grid points → 2⁵ = 32 parameter sets."""
        config = _make_config(
            geo,
            [*frame_constant, param_offset_kernel, param_offset_time],
            strategy=SearchStrategy.GRID_SEARCH,
            grid_points_per_param=2,
        )
        sets = load_param_sets(config)
        assert len(sets) == 32

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

    def test_constant_kernel_axes_are_separate_grid_dimensions(self, geo, frame_constant_zero):
        """Each axis of a frame is its own grid dimension."""
        config = _make_config(geo, frame_constant_zero, strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=2)
        combos = {tuple(round(val / _arcsec(10.0)) for _, val in s) for s in load_param_sets(config)}
        assert combos == set(itertools.product([-1, 1], repeat=3))

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

    def test_guardrail_raises_when_total_exceeds_limit(self, geo, param_offset_kernel, param_offset_time):
        """GRID_SEARCH raises ValueError when cartesian product would exceed max_grid_sets."""
        # 10 points × 2 params = 100 sets → set limit to 99 to trigger the guard
        config = _make_config(
            geo,
            [param_offset_kernel, param_offset_time],
            strategy=SearchStrategy.GRID_SEARCH,
            grid_points_per_param=10,
        )
        # Override the limit below what the sweep would produce (100 sets)
        config = config.model_copy(update={"max_grid_sets": 99})
        with pytest.raises(ValueError, match="exceeds the safety limit"):
            load_param_sets(config)

    def test_guardrail_passes_when_limit_raised(self, geo, param_offset_time):
        """Explicitly raising max_grid_sets allows larger sweeps through."""
        # 5 points × 1 param = 5 sets; set limit to 5 exactly — should succeed
        config = _make_config(
            geo,
            [param_offset_time],
            strategy=SearchStrategy.GRID_SEARCH,
            grid_points_per_param=5,
        )
        config = config.model_copy(update={"max_grid_sets": 5})
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

    def test_constant_kernel_sweeps_one_axis_at_a_time(self, geo, frame_constant_zero):
        """SINGLE_OFFSET moves one axis of a frame while the other two stay at nominal."""
        config = _make_config(geo, frame_constant_zero, strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=3)
        sets = np.array([[val for _, val in s] for s in load_param_sets(config)]) / _arcsec(10.0)
        expected = np.zeros((9, 3))
        for axis in range(3):
            expected[3 * axis : 3 * axis + 3, axis] = [-1.0, 0.0, 1.0]
        np.testing.assert_allclose(sets, expected, atol=1e-12)


class TestHeldParameters:
    """A parameter with zero-width bounds is held at current_value by the deterministic strategies."""

    def _frame_with_yaw_held(self):
        frame = _frame([0.0, 0.0, 30.0], [-10.0, 10.0], None)
        frame[2] = frame[2].model_copy(update={"spec": frame[2].spec.model_copy(update={"bounds": [0.0, 0.0]})})
        return frame

    def test_grid_gives_a_held_axis_one_point(self, geo):
        config = _make_config(
            geo, self._frame_with_yaw_held(), strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=2
        )
        sets = load_param_sets(config)
        assert len(sets) == 4
        np.testing.assert_allclose([s[2][1] for s in sets], _arcsec(30.0))

    def test_single_offset_does_not_sweep_a_held_axis(self, geo):
        config = _make_config(geo, self._frame_with_yaw_held(), strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=3)
        sets = np.array([[val for _, val in s] for s in load_param_sets(config)])
        assert sets.shape == (6, 3)
        np.testing.assert_allclose(sets[:, 2], _arcsec(30.0))

    def test_single_offset_with_every_parameter_held_raises(self, geo):
        frame = [
            p.model_copy(update={"spec": p.spec.model_copy(update={"bounds": [0.0, 0.0]})})
            for p in _frame([0.0] * 3, [-1.0, 1.0], None)
        ]
        with pytest.raises(ValueError, match="non-zero-width bounds"):
            load_param_sets(_make_config(geo, frame, strategy=SearchStrategy.SINGLE_OFFSET))


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
        assert any("grid_points_per_param" in err.get("loc", ()) for err in errors)

    def test_search_strategy_default_is_random(self, param_offset_time):
        sweep = Sweep(
            seed=0,
            n_iterations=3,
            parameters=[param_offset_time],
        )
        assert sweep.search_strategy == SearchStrategy.RANDOM

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
        """RANDOM sweep survives model_dump_json / model_validate_json round-trip."""
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.RANDOM)
        json_str = config.model_dump_json()
        restored = Sweep.model_validate_json(json_str)
        assert restored.search_strategy == SearchStrategy.RANDOM
        assert restored.n_iterations == config.n_iterations

    def test_json_round_trip_grid(self, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=7)
        restored = Sweep.model_validate_json(config.model_dump_json())
        assert restored.search_strategy == SearchStrategy.GRID_SEARCH
        assert restored.grid_points_per_param == 7
        assert restored.max_grid_sets == config.max_grid_sets

    def test_json_round_trip_single_offset(self, geo, param_offset_time):
        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.SINGLE_OFFSET, n_iterations=12)
        restored = Sweep.model_validate_json(config.model_dump_json())
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
    def test_constant_kernel_always_float(self, strategy, geo, frame_constant_zero):
        config = _make_config(geo, frame_constant_zero, strategy=strategy, n_iterations=3, grid_points_per_param=3)
        for param_set in load_param_sets(config):
            for _, val in param_set:
                assert isinstance(val, (float, np.floating)), f"Expected float for {strategy}, got {type(val)}"

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


# ===========================================================================
# Logging behaviour
# ===========================================================================


class TestLogging:
    """Verify _log_param_set_summary emits per-set detail only at DEBUG."""

    def test_per_set_detail_suppressed_at_info(self, geo, param_offset_time, caplog):
        """With log level INFO, per-set lines must NOT appear in the log output."""
        import logging

        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=4)
        with caplog.at_level(logging.INFO, logger="curryer.correction.parameters"):
            load_param_sets(config)

        # The high-level count line should be present
        assert any("Generated 4 parameter sets" in r.message for r in caplog.records)
        # Individual "Set N:" detail lines must NOT appear at INFO
        assert not any(r.message.startswith("  Set ") for r in caplog.records)

    def test_per_set_detail_present_at_debug(self, geo, param_offset_time, caplog):
        """With log level DEBUG, per-set detail lines DO appear."""
        import logging

        config = _make_config(geo, [param_offset_time], strategy=SearchStrategy.GRID_SEARCH, grid_points_per_param=3)
        with caplog.at_level(logging.DEBUG, logger="curryer.correction.parameters"):
            load_param_sets(config)

        # Expect exactly 3 "Set N:" lines (one per grid point)
        set_lines = [r for r in caplog.records if r.message.strip().startswith("Set ")]
        assert len(set_lines) == 3
