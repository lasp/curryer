"""Parameter set generation for the correction pipeline.

This module provides :func:`load_param_sets`, which generates parameter sets
for correction analysis.  Three search strategies are supported:

``RANDOM`` (default)
    Monte Carlo random walk.  Each parameter is sampled from a normal
    distribution centred on ``current_value`` with the configured ``sigma``,
    clipped to ``bounds``.  Controlled by ``seed`` and ``n_iterations``.

``GRID_SEARCH``
    Deterministic cartesian-product sweep.  ``grid_points_per_param``
    evenly-spaced offset values are produced for each parameter (spanning its
    full ``bounds`` range) and the cartesian product of all per-parameter grids
    is enumerated.  ``n_iterations`` is ignored for this strategy.

``SINGLE_OFFSET``
    Deterministic single-parameter sweep.  Each parameter is varied
    independently across ``n_iterations`` evenly-spaced values while all other
    parameters are held at their nominal ``current_value``.

Supported parameter types:
- ``CONSTANT_KERNEL`` – 3-D attitude corrections (roll, pitch, yaw) stored as
  a ``pandas.DataFrame`` with ``ugps``, ``angle_x``, ``angle_y``, ``angle_z``.
- ``OFFSET_KERNEL`` – single-axis angle bias (float, in radians).
- ``OFFSET_TIME`` – timing correction (float, in seconds).
"""

import itertools
import logging
import typing

import numpy as np
import pandas as pd

from curryer.correction.config import CorrectionConfig, ParameterConfig, ParameterType, SearchStrategy

logger = logging.getLogger(__name__)

# ============================================================================
# Unit-conversion helpers
# ============================================================================

_UGPS_EPOCH_END = 2_209_075_218_000_000  # sentinel end-of-mission ugps for CK DataFrames


def _arcsec_to_rad(value: float) -> float:
    """Convert arcseconds to radians."""
    return np.deg2rad(value / 3600.0) if value != 0 else 0.0


def _bounds_to_rad(bounds: list[float], units: str | None) -> list[float]:
    if units == "arcseconds":
        return [np.deg2rad(bounds[0] / 3600.0), np.deg2rad(bounds[1] / 3600.0)]
    return list(bounds)


def _val_to_rad(value: float, units: str | None) -> float:
    if units == "arcseconds":
        return _arcsec_to_rad(value)
    return value


def _val_to_seconds(value: float, units: str | None) -> float:
    if units == "milliseconds":
        return value / 1_000.0
    if units == "microseconds":
        return value / 1_000_000.0
    return value


def _bounds_to_seconds(bounds: list[float], units: str | None) -> list[float]:
    if units == "milliseconds":
        return [bounds[0] / 1_000.0, bounds[1] / 1_000.0]
    if units == "microseconds":
        return [bounds[0] / 1_000_000.0, bounds[1] / 1_000_000.0]
    return list(bounds)


# ============================================================================
# DataFrame builder for CONSTANT_KERNEL
# ============================================================================


def _make_ck_dataframe(angle_vals: list[float]) -> pd.DataFrame:
    """Wrap ``[angle_x, angle_y, angle_z]`` (radians) into the pipeline DataFrame format."""
    return pd.DataFrame(
        {
            "ugps": [0, _UGPS_EPOCH_END],
            "angle_x": [angle_vals[0], angle_vals[0]],
            "angle_y": [angle_vals[1], angle_vals[1]],
            "angle_z": [angle_vals[2], angle_vals[2]],
        }
    )


# ============================================================================
# Scalar current-value extraction (handles list vs scalar current_value)
# ============================================================================


def _scalar_current_value(param: ParameterConfig) -> float:
    """Return ``param.data.current_value`` as a scalar float.

    Raises
    ------
    TypeError
        If ``current_value`` is not a scalar numeric type. This helps surface
        misconfigured parameters early instead of silently coercing them to 0.0.
    """
    cv = param.data.current_value
    if isinstance(cv, (int, float, np.number)):
        return float(cv)

    param_name = getattr(param, "name", "<unknown>")
    raise TypeError(
        f"Parameter '{param_name}' (type {getattr(param.ptype, 'name', param.ptype)}) "
        f"has non-scalar current_value of type {type(cv).__name__}; expected a scalar "
        "numeric value (int, float, or NumPy scalar)."
    )


# ============================================================================
# Nominal value (no offset applied) – used by SINGLE_OFFSET for held params
# ============================================================================


def _get_nominal_value(param: ParameterConfig) -> typing.Any:
    """Return the un-perturbed, unit-converted value for *param*.

    For ``CONSTANT_KERNEL``, returns a :class:`~pandas.DataFrame` with angles
    equal to the ``current_value`` in radians.  For ``OFFSET_KERNEL`` /
    ``OFFSET_TIME``, returns a float in radians / seconds respectively.
    """
    units = param.data.get("units")
    current_value = param.data.current_value

    if param.ptype == ParameterType.CONSTANT_KERNEL:
        if isinstance(current_value, list) and len(current_value) == 3:
            angle_vals = [_val_to_rad(v, units) for v in current_value]
        else:
            cv_rad = _val_to_rad(_scalar_current_value(param), units)
            angle_vals = [cv_rad, cv_rad, cv_rad]
        return _make_ck_dataframe(angle_vals)

    if param.ptype == ParameterType.OFFSET_KERNEL:
        return _val_to_rad(_scalar_current_value(param), units)

    if param.ptype == ParameterType.OFFSET_TIME:
        return _val_to_seconds(_scalar_current_value(param), units)

    return 0.0  # unreachable for known types


# ============================================================================
# Grid values (evenly-spaced across the offset range) – GRID_SEARCH / SINGLE_OFFSET
# ============================================================================


def _get_grid_values(param: ParameterConfig, n_points: int) -> list[typing.Any]:
    """Return *n_points* evenly-spaced sampled values for *param*.

    Offsets are linearly spaced over ``[bounds[0], bounds[1]]`` (in the
    parameter's native units before conversion) and added to the converted
    ``current_value``.

    For ``CONSTANT_KERNEL`` the scalar offset is applied uniformly to all
    three rotation axes.

    Parameters
    ----------
    param : ParameterConfig
        Parameter specification.
    n_points : int
        Number of evenly-spaced points (>= 2).

    Returns
    -------
    list
        List of *n_points* values; each element matches what the pipeline
        expects for that parameter type (DataFrame or float).
    """
    units = param.data.get("units")
    bounds = param.data.bounds
    current_value = param.data.current_value

    if param.ptype == ParameterType.CONSTANT_KERNEL:
        bounds_rad = _bounds_to_rad(bounds, units)
        offsets = np.linspace(bounds_rad[0], bounds_rad[1], n_points)

        if isinstance(current_value, list) and len(current_value) == 3:
            base_vals = [_val_to_rad(v, units) for v in current_value]
        else:
            cv_rad = _val_to_rad(_scalar_current_value(param), units)
            base_vals = [cv_rad, cv_rad, cv_rad]

        return [_make_ck_dataframe([bv + offset for bv in base_vals]) for offset in offsets]

    if param.ptype == ParameterType.OFFSET_KERNEL:
        cv_rad = _val_to_rad(_scalar_current_value(param), units)
        bounds_rad = _bounds_to_rad(bounds, units)
        offsets = np.linspace(bounds_rad[0], bounds_rad[1], n_points)
        return list(cv_rad + offsets)

    if param.ptype == ParameterType.OFFSET_TIME:
        cv_s = _val_to_seconds(_scalar_current_value(param), units)
        bounds_s = _bounds_to_seconds(bounds, units)
        offsets = np.linspace(bounds_s[0], bounds_s[1], n_points)
        return list(cv_s + offsets)

    raise ValueError(f"Unsupported parameter type for grid generation: {param.ptype!r}")


# ============================================================================
# Strategy implementations
# ============================================================================


def _generate_random(config: CorrectionConfig) -> list[list[tuple[ParameterConfig, typing.Any]]]:
    """Generate random parameter sets – exact current behaviour preserved."""
    if config.seed is not None:
        np.random.seed(config.seed)
        logger.info(f"Set random seed to {config.seed} for reproducible parameter generation")

    output = []

    for ith in range(config.n_iterations):
        out_set = []
        logger.debug(f"Generating parameter set {ith + 1}/{config.n_iterations}")

        for param_idx, param in enumerate(config.parameters):
            current_value = param.data.current_value
            bounds = param.data.bounds

            if param.ptype == ParameterType.CONSTANT_KERNEL:
                if isinstance(current_value, list) and len(current_value) == 3:
                    param_vals = []
                    for i, current_val in enumerate(current_value):
                        if "sigma" in param.data and param.data["sigma"] is not None and param.data["sigma"] > 0:
                            if param.data.get("units") == "arcseconds":
                                sigma_rad = np.deg2rad(param.data["sigma"] / 3600.0)
                                current_val_rad = np.deg2rad(current_val / 3600.0) if current_val != 0 else current_val
                                bounds_rad = [np.deg2rad(bounds[0] / 3600.0), np.deg2rad(bounds[1] / 3600.0)]
                            else:
                                sigma_rad = param.data["sigma"]
                                current_val_rad = current_val
                                bounds_rad = bounds
                            offset = np.random.normal(0, sigma_rad)
                            offset = np.clip(offset, bounds_rad[0], bounds_rad[1])
                            param_vals.append(current_val_rad + offset)
                        else:
                            if "sigma" not in param.data or param.data["sigma"] is None:
                                logger.debug(
                                    f"  Parameter {param_idx} axis {i}: No sigma specified, using fixed current_value"
                                )
                            elif param.data["sigma"] == 0:
                                logger.debug(f"  Parameter {param_idx} axis {i}: sigma=0, using fixed current_value")
                            if param.data.get("units") == "arcseconds":
                                current_val_rad = np.deg2rad(current_val / 3600.0) if current_val != 0 else current_val
                            else:
                                current_val_rad = current_val
                            param_vals.append(current_val_rad)
                else:
                    param_vals = [0.0, 0.0, 0.0]
                    if "sigma" in param.data and param.data["sigma"] is not None and param.data["sigma"] > 0:
                        if param.data.get("units") == "arcseconds":
                            sigma_rad = np.deg2rad(param.data["sigma"] / 3600.0)
                            bounds_rad = [np.deg2rad(bounds[0] / 3600.0), np.deg2rad(bounds[1] / 3600.0)]
                            current_val_rad = (
                                np.deg2rad(current_value / 3600.0) if current_value != 0 else current_value
                            )
                        else:
                            sigma_rad = param.data["sigma"]
                            bounds_rad = bounds
                            current_val_rad = current_value
                        for i in range(3):
                            offset = np.random.normal(0, sigma_rad)
                            offset = np.clip(offset, bounds_rad[0], bounds_rad[1])
                            param_vals[i] = current_val_rad + offset
                    else:
                        if "sigma" not in param.data or param.data["sigma"] is None:
                            logger.debug(f"  Parameter {param_idx}: No sigma specified, using fixed current_value")
                        elif param.data["sigma"] == 0:
                            logger.debug(f"  Parameter {param_idx}: sigma=0, using fixed current_value")
                        if param.data.get("units") == "arcseconds":
                            current_val_rad = (
                                np.deg2rad(current_value / 3600.0) if current_value != 0 else current_value
                            )
                        else:
                            current_val_rad = current_value
                        param_vals = [current_val_rad, current_val_rad, current_val_rad]

                param_vals = _make_ck_dataframe(param_vals)
                logger.debug(
                    f"  CONSTANT_KERNEL {param_idx}: angles=[{param_vals['angle_x'].iloc[0]:.6e}, "
                    f"{param_vals['angle_y'].iloc[0]:.6e}, {param_vals['angle_z'].iloc[0]:.6e}] rad"
                )

            elif param.ptype == ParameterType.OFFSET_KERNEL:
                if "sigma" in param.data and param.data["sigma"] is not None and param.data["sigma"] > 0:
                    if param.data.get("units") == "arcseconds":
                        sigma_rad = np.deg2rad(param.data["sigma"] / 3600.0)
                        current_val_rad = np.deg2rad(current_value / 3600.0) if current_value != 0 else current_value
                        bounds_rad = [np.deg2rad(bounds[0] / 3600.0), np.deg2rad(bounds[1] / 3600.0)]
                    else:
                        sigma_rad = param.data["sigma"]
                        current_val_rad = current_value
                        bounds_rad = bounds
                    offset = np.random.normal(0, sigma_rad)
                    offset = np.clip(offset, bounds_rad[0], bounds_rad[1])
                    param_vals = current_val_rad + offset
                else:
                    if "sigma" not in param.data or param.data["sigma"] is None:
                        logger.debug(f"  Parameter {param_idx}: No sigma specified, using fixed current_value")
                    elif param.data["sigma"] == 0:
                        logger.debug(f"  Parameter {param_idx}: sigma=0, using fixed current_value")
                    if param.data.get("units") == "arcseconds":
                        current_val_rad = np.deg2rad(current_value / 3600.0) if current_value != 0 else current_value
                    else:
                        current_val_rad = current_value
                    param_vals = current_val_rad
                logger.debug(f"  OFFSET_KERNEL {param_idx}: {param_vals:.6e} rad")

            elif param.ptype == ParameterType.OFFSET_TIME:
                if "sigma" in param.data and param.data["sigma"] is not None and param.data["sigma"] > 0:
                    if param.data.get("units") == "seconds":
                        sigma_time = param.data["sigma"]
                        current_val_time = current_value
                        bounds_time = bounds
                    elif param.data.get("units") == "milliseconds":
                        sigma_time = param.data["sigma"] / 1000.0
                        current_val_time = current_value / 1000.0
                        bounds_time = [bounds[0] / 1000.0, bounds[1] / 1000.0]
                    elif param.data.get("units") == "microseconds":
                        sigma_time = param.data["sigma"] / 1000000.0
                        current_val_time = current_value / 1000000.0
                        bounds_time = [bounds[0] / 1000000.0, bounds[1] / 1000000.0]
                    else:
                        sigma_time = param.data["sigma"]
                        current_val_time = current_value
                        bounds_time = bounds
                    offset = np.random.normal(0, sigma_time)
                    offset = np.clip(offset, bounds_time[0], bounds_time[1])
                    param_vals = current_val_time + offset
                else:
                    if "sigma" not in param.data or param.data["sigma"] is None:
                        logger.debug(f"  Parameter {param_idx}: No sigma specified, using fixed current_value")
                    elif param.data["sigma"] == 0:
                        logger.debug(f"  Parameter {param_idx}: sigma=0, using fixed current_value")
                    if param.data.get("units") == "milliseconds":
                        current_val_time = current_value / 1000.0
                    elif param.data.get("units") == "microseconds":
                        current_val_time = current_value / 1000000.0
                    else:
                        current_val_time = current_value
                    param_vals = current_val_time
                logger.debug(f"  OFFSET_TIME {param_idx}: {param_vals:.6e} seconds")

            out_set.append((param, param_vals))
        output.append(out_set)

    return output


def _generate_grid_search(config: CorrectionConfig) -> list[list[tuple[ParameterConfig, typing.Any]]]:
    """Generate parameter sets via deterministic cartesian-product grid sweep.

    Produces ``grid_points_per_param ^ len(parameters)`` parameter sets.
    ``n_iterations`` is not used for this strategy.
    """
    n = config.grid_points_per_param
    n_params = len(config.parameters)
    total = n**n_params
    logger.info(f"GRID_SEARCH: {n} points × {n_params} parameter(s) = {total} total parameter sets")

    per_param_values = [_get_grid_values(param, n) for param in config.parameters]

    output = []
    for combo in itertools.product(*per_param_values):
        out_set = list(zip(config.parameters, combo))
        output.append(out_set)

    logger.info(f"GRID_SEARCH: generated {len(output)} parameter sets")
    return output


def _generate_single_offset(config: CorrectionConfig) -> list[list[tuple[ParameterConfig, typing.Any]]]:
    """Generate parameter sets by sweeping one parameter at a time.

    For each parameter in ``config.parameters``:
    - ``n_iterations`` evenly-spaced values are generated spanning the
      parameter's full ``bounds`` offset range.
    - All other parameters are held at their nominal ``current_value``.

    Total parameter sets produced: ``len(parameters) × n_iterations``.
    """
    n = config.n_iterations
    n_params = len(config.parameters)
    logger.info(f"SINGLE_OFFSET: {n_params} parameter(s) × {n} values each = {n_params * n} total parameter sets")

    nominals = [_get_nominal_value(param) for param in config.parameters]

    output = []
    for sweep_idx, sweep_param in enumerate(config.parameters):
        sweep_values = _get_grid_values(sweep_param, n)
        param_name = sweep_param.config_file.name if sweep_param.config_file else f"param_{sweep_idx}"
        logger.debug(f"  SINGLE_OFFSET: sweeping parameter {sweep_idx} ({param_name}) with {len(sweep_values)} values")
        for val in sweep_values:
            out_set = []
            for param_idx, param in enumerate(config.parameters):
                out_set.append((param, val if param_idx == sweep_idx else nominals[param_idx]))
            output.append(out_set)

    logger.info(f"SINGLE_OFFSET: generated {len(output)} parameter sets")
    return output


# ============================================================================
# Logging helper
# ============================================================================


def _log_param_set_summary(output: list[list[tuple[ParameterConfig, typing.Any]]]) -> None:
    """Log a structured summary of all generated parameter sets."""
    if not output:
        return

    logger.info(f"Generated {len(output)} parameter sets with {len(output[0])} parameters each")
    logger.info("\nParameter Set Summary:")
    logger.info("-" * 100)
    for param_set_idx, param_set in enumerate(output):
        logger.info(f"  Set {param_set_idx}:")
        for param_idx, (param, param_vals) in enumerate(param_set):
            field_name = param.data.get("field", "unknown")
            ptype_name = param.ptype.name

            if param.ptype == ParameterType.CONSTANT_KERNEL:
                if isinstance(param_vals, pd.DataFrame) and "angle_x" in param_vals.columns:
                    angles = [
                        param_vals["angle_x"].iloc[0],
                        param_vals["angle_y"].iloc[0],
                        param_vals["angle_z"].iloc[0],
                    ]
                    logger.info(
                        f"    {ptype_name:16s} {field_name:25s}: "
                        f"[{angles[0]:+.6e}, {angles[1]:+.6e}, {angles[2]:+.6e}] rad"
                    )
                else:
                    logger.info(f"    {ptype_name:16s} {field_name:25s}: (constant kernel data)")
            elif param.ptype == ParameterType.OFFSET_KERNEL:
                units = param.data.get("units", "")
                if units == "arcseconds":
                    param_arcsec = np.rad2deg(param_vals) * 3600.0
                    logger.info(
                        f"    {ptype_name:16s} {field_name:25s}: {param_arcsec:+10.3f} arcsec ({param_vals:+.9f} rad)"
                    )
                else:
                    logger.info(f"    {ptype_name:16s} {field_name:25s}: {param_vals:+.9f} {units}")
            elif param.ptype == ParameterType.OFFSET_TIME:
                units = param.data.get("units", "")
                if units == "milliseconds":
                    param_ms = param_vals * 1000.0
                    logger.info(f"    {ptype_name:16s} {field_name:25s}: {param_ms:+10.3f} ms ({param_vals:+.9f} s)")
                else:
                    logger.info(f"    {ptype_name:16s} {field_name:25s}: {param_vals:+.9f} {units}")
    logger.info("-" * 100)


# ============================================================================
# Public API
# ============================================================================


def load_param_sets(config: CorrectionConfig) -> list[list[tuple[ParameterConfig, typing.Any]]]:
    """Generate parameter sets for correction iterations.

    Dispatches to the appropriate generator based on
    ``config.search_strategy``:

    - :attr:`~SearchStrategy.RANDOM` – Monte Carlo random walk (default).
    - :attr:`~SearchStrategy.GRID_SEARCH` – deterministic cartesian-product
      sweep across ``grid_points_per_param`` evenly-spaced values per parameter.
    - :attr:`~SearchStrategy.SINGLE_OFFSET` – deterministic single-parameter
      sweep; each parameter is varied independently while others stay at nominal.

    Parameters
    ----------
    config : CorrectionConfig
        Complete correction configuration including parameters, strategy, and
        sampling settings.

    Returns
    -------
    list[list[tuple[ParameterConfig, Any]]]
        Outer list: one element per parameter set (iteration).
        Inner list: one ``(ParameterConfig, sampled_value)`` pair per parameter.
        ``sampled_value`` is a :class:`~pandas.DataFrame` for
        ``CONSTANT_KERNEL`` and a ``float`` for ``OFFSET_KERNEL`` /
        ``OFFSET_TIME``.
    """
    strategy = config.search_strategy

    logger.info(
        f"Generating parameter sets for {len(config.parameters)} parameter(s) using strategy: {strategy.value!r}"
    )
    for i, param in enumerate(config.parameters):
        param_name = param.config_file.name if param.config_file else f"param_{i}"
        current_value = param.data.current_value
        bounds = param.data.bounds
        logger.info(
            f"  {i + 1}. {param_name} ({param.ptype.name}): "
            f"current_value={current_value}, sigma={param.data.get('sigma', 'N/A')}, "
            f"bounds={bounds}, units={param.data.get('units', 'N/A')}"
        )

    if strategy == SearchStrategy.RANDOM:
        output = _generate_random(config)
    elif strategy == SearchStrategy.GRID_SEARCH:
        output = _generate_grid_search(config)
    elif strategy == SearchStrategy.SINGLE_OFFSET:
        output = _generate_single_offset(config)
    else:
        raise ValueError(f"Unknown SearchStrategy: {strategy!r}. Valid values are: {[s.value for s in SearchStrategy]}")

    _log_param_set_summary(output)
    return output
