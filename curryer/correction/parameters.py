"""Parameter set generation for the correction pipeline.

This module provides :func:`load_param_sets`, which generates random
parameter sets for Monte Carlo-style sensitivity analysis.  Each parameter is
sampled according to its configured distribution, sigma, and bounds, and the
result is a list of ``(ParameterConfig, sampled_value)`` pairs consumed by
:mod:`curryer.correction.kernel_ops` to produce SPICE kernels or time offsets.

Supported parameter types:
- ``CONSTANT_KERNEL`` – 3-D attitude corrections (roll, pitch, yaw) stored as
  a ``pandas.DataFrame`` with ``ugps``, ``angle_x``, ``angle_y``, ``angle_z``.
- ``OFFSET_KERNEL`` – single-axis angle bias (float, in radians).
- ``OFFSET_TIME`` – timing correction (float, in seconds).
"""

import logging
import typing

import numpy as np
import pandas as pd

from curryer.correction.config import CorrectionConfig, ParameterConfig, ParameterType

logger = logging.getLogger(__name__)


def load_param_sets(config: CorrectionConfig) -> list[list[tuple[ParameterConfig, typing.Any]]]:
    """
    Generate random parameter sets for Correction iterations.
    Each parameter is sampled according to its distribution and bounds.

    The parameter generation works as follows:
    - current_value: The baseline/current parameter value
    - bounds: The limits for random offsets (in same units as current_value and sigma)
    - sigma: Standard deviation for normal distribution of offsets
    - Generated offsets are centered around 0, then applied to current_value
    - Final value = current_value + random_offset

    Handles all parameter types:
    - CONSTANT_KERNEL: 3D attitude corrections (roll, pitch, yaw)
    - OFFSET_KERNEL: Single angle biases for telemetry fields
    - OFFSET_TIME: Timing corrections for science frames
    """

    if config.seed is not None:
        np.random.seed(config.seed)
        logger.info(f"Set random seed to {config.seed} for reproducible parameter generation")

    output = []

    logger.info(f"Generating {config.n_iterations} parameter sets for {len(config.parameters)} parameters:")
    for i, param in enumerate(config.parameters):
        param_name = param.config_file.name if param.config_file else f"param_{i}"
        current_value = param.data.get("current_value", param.data.get("center", 0.0))
        bounds = param.data.get("bounds", param.data.get("arange", [-1.0, 1.0]))
        logger.info(
            f"  {i + 1}. {param_name} ({param.ptype.name}): "
            f"current_value={current_value}, sigma={param.data.get('sigma', 'N/A')}, "
            f"bounds={bounds}, units={param.data.get('units', 'N/A')}"
        )

    for ith in range(config.n_iterations):
        out_set = []
        logger.debug(f"Generating parameter set {ith + 1}/{config.n_iterations}")

        for param_idx, param in enumerate(config.parameters):
            current_value = param.data.get("current_value", param.data.get("center", 0.0))
            bounds = param.data.get("bounds", param.data.get("arange", [-1.0, 1.0]))

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

                param_vals = pd.DataFrame(
                    {
                        "ugps": [0, 2209075218000000],
                        "angle_x": [param_vals[0], param_vals[0]],
                        "angle_y": [param_vals[1], param_vals[1]],
                        "angle_z": [param_vals[2], param_vals[2]],
                    }
                )
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

    if output:
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
                            f"    {ptype_name:16s} {field_name:25s}: [{angles[0]:+.6e}, {angles[1]:+.6e}, {angles[2]:+.6e}] rad"
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
                        logger.info(
                            f"    {ptype_name:16s} {field_name:25s}: {param_ms:+10.3f} ms ({param_vals:+.9f} s)"
                        )
                    else:
                        logger.info(f"    {ptype_name:16s} {field_name:25s}: {param_vals:+.9f} {units}")
        logger.info("-" * 100)

    return output
