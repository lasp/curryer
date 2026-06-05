"""SPICE kernel file management for the correction pipeline.

This module creates and applies parameter-specific SPICE kernels:

- :func:`apply_offset` -- modifies telemetry/science data for
  ``OFFSET_KERNEL`` and ``OFFSET_TIME`` parameters.
- :func:`_create_dynamic_kernels` -- writes SC-SPK/SC-CK kernels from
  telemetry data (once per image pair, not per parameter set).
- :func:`_create_parameter_kernels` -- writes parameter-specific kernels
  and applies time offsets for each iteration.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from curryer.correction.config import CorrectionConfig, ParameterConfig, ParameterType
from curryer.kernels import create

logger = logging.getLogger(__name__)


def apply_offset(config: ParameterConfig, param_data, input_data):
    """
    Apply parameter offsets to input data based on parameter type.

    Args:
        config: ParameterConfig specifying how to apply the offset
        param_data: The parameter values to apply (offset amounts)
        input_data: The input dataset to modify

    Returns:
        Modified copy of input_data with parameter offsets applied
    """
    logger.info(f"Applying {config.ptype.name} offset to {config.spec.get('field', 'unknown field')}")

    # Make a copy to avoid modifying the original
    if isinstance(input_data, pd.DataFrame):
        modified_data = input_data.copy()
    else:
        modified_data = input_data.copy() if hasattr(input_data, "copy") else input_data

    if config.ptype == ParameterType.OFFSET_KERNEL:
        # Apply offset to telemetry fields for dynamic kernels (azimuth/elevation angles)
        # OFFSET_KERNEL is ONLY for angle biases, not time offsets
        # Valid units: "arcseconds" (converted to radians) or None (radians assumed)
        # For time offsets, use OFFSET_TIME instead
        field_name = config.spec.get("field")
        if not field_name:
            raise ValueError("OFFSET_KERNEL parameter requires 'field' to be specified in config")

        if field_name in modified_data.columns:
            # Convert parameter value to appropriate units
            # OFFSET_KERNEL is for angle biases only (azimuth/elevation angles)
            offset_value = param_data
            original_value = offset_value
            if config.spec.get("units") == "arcseconds":
                # Convert arcseconds to radians for application
                offset_value = np.deg2rad(param_data / 3600.0)
                logger.info(f"✓ Applying OFFSET_KERNEL to field '{field_name}'")
                logger.info(f"  Offset: {original_value:.6f} arcsec = {offset_value:.9f} rad")
            else:
                # No units specified - assume radians (direct application)
                logger.info(f"✓ Applying OFFSET_KERNEL to field '{field_name}'")
                logger.info(f"  Offset: {offset_value:.9f} rad (no unit conversion)")

            # Store original values for logging
            original_mean = modified_data[field_name].mean()

            # Apply additive offset
            modified_data[field_name] = modified_data[field_name] + offset_value

            # Log the effect
            new_mean = modified_data[field_name].mean()
            logger.info(f"  Original mean: {original_mean:.9f}")
            logger.info(f"  New mean:      {new_mean:.9f}")
            logger.info(f"  Delta:         {new_mean - original_mean:.9f}")
        else:
            available_cols = list(modified_data.columns) if hasattr(modified_data, "columns") else []
            logger.warning(f"Field '{field_name}' not found in telemetry data for offset application")
            logger.warning(f"Available columns: {available_cols}")

    elif config.ptype == ParameterType.OFFSET_TIME:
        # Apply time offset to science frame timing
        # NOTE: param_data is in seconds while target field (e.g., corrected_timestamp) is typically in microseconds
        field_name = config.spec.get("field", "corrected_timestamp")
        if hasattr(modified_data, "__getitem__") and field_name in modified_data:
            # param_data is already in seconds (converted by load_param_sets)
            # Convert seconds to microseconds for the timestamp field
            offset_value_seconds = param_data
            offset_value_us = param_data * 1000000.0  # seconds to microseconds

            logger.info(f"✓ Applying OFFSET_TIME to field '{field_name}'")
            units = config.spec.get("units", "seconds")
            if units == "milliseconds":
                logger.info(f"  Offset: {offset_value_seconds * 1000.0:.6f} ms (configured) = {offset_value_us:.6f} µs")
            elif units == "microseconds":
                logger.info(
                    f"  Offset: {offset_value_seconds * 1000000.0:.6f} µs (configured) = {offset_value_us:.6f} µs"
                )
            else:
                logger.info(f"  Offset: {offset_value_seconds:.6f} s = {offset_value_us:.6f} µs")

            # Store original values for logging
            if hasattr(modified_data[field_name], "mean"):
                original_mean = modified_data[field_name].mean()
            else:
                original_mean = np.mean(modified_data[field_name])

            # Apply additive offset in microseconds
            modified_data[field_name] = modified_data[field_name] + offset_value_us

            # Log the effect
            if hasattr(modified_data[field_name], "mean"):
                new_mean = modified_data[field_name].mean()
            else:
                new_mean = np.mean(modified_data[field_name])
            logger.info(f"  Original mean: {original_mean:.6f}")
            logger.info(f"  New mean:      {new_mean:.6f}")
            logger.info(f"  Delta:         {new_mean - original_mean:.6f}")
        else:
            logger.warning(f"Field '{field_name}' not found in science data for time offset application")

    elif config.ptype == ParameterType.CONSTANT_KERNEL:
        # For constant kernels, param_data should already be in the correct format
        # (DataFrame with ugps, angle_x, angle_y, angle_z columns)
        logger.info(
            f"Using constant kernel data with {len(param_data) if hasattr(param_data, '__len__') else 1} entries"
        )
        modified_data = param_data

    else:
        raise NotImplementedError(f"Parameter type {config.ptype} not implemented")

    return modified_data


def _create_dynamic_kernels(
    config: "CorrectionConfig",
    work_dir: Path,
    tlm_dataset: pd.DataFrame,
    creator: "create.KernelCreator",
) -> list[Path]:
    """Create dynamic SPICE kernels from telemetry data.

    Dynamic kernels (SC-SPK, SC-CK) are generated from spacecraft telemetry
    and do not change with parameter variations. In the current implementation,
    these are created once per image.

    Parameters
    ----------
    config : CorrectionConfig
        Configuration with geo settings and dynamic_kernels list
    work_dir : Path
        Working directory for kernel files
    tlm_dataset : pd.DataFrame
        Spacecraft state data with position, velocity, attitude, and time columns
    creator : create.KernelCreator
        KernelCreator instance for writing kernels

    Returns
    -------
    list[Path]
        List of kernel file paths created (e.g., [sc_ephemeris.bsp, sc_attitude.bc])

    Examples
    --------
    >>> from curryer.kernels import create
    >>> creator = create.KernelCreator(overwrite=True, append=False)
    >>> dynamic_kernels = _create_dynamic_kernels(config, work_dir, tlm_dataset, creator)
    >>> # Use in SPICE context
    >>> with sp.ext.load_kernel(dynamic_kernels):
    ...     # Perform geolocation
    ...     pass
    """
    logger.info("    Creating dynamic kernels from telemetry...")
    dynamic_kernels = []
    for kernel_config in config.geo.dynamic_kernels:
        dynamic_kernels.append(
            creator.write_from_json(
                kernel_config,
                output_kernel=work_dir,
                input_data=tlm_dataset,
            )
        )
    logger.info(f"    Created {len(dynamic_kernels)} dynamic kernels")
    return dynamic_kernels


def _create_parameter_kernels(
    params: list[tuple["ParameterConfig", Any]],
    work_dir: Path,
    tlm_dataset: pd.DataFrame,
    sci_dataset: pd.DataFrame,
    ugps_times: Any,
    config: "CorrectionConfig",
    creator: "create.KernelCreator",
) -> tuple[list[Path], Any]:
    """Create parameter-specific SPICE kernels and apply time offsets.

    This function applies parameter variations by creating modified kernels
    (CONSTANT_KERNEL, OFFSET_KERNEL) or modifying time tags (OFFSET_TIME).
    Each parameter set produces different kernels and/or time modifications.

    Parameters
    ----------
    params : list[tuple[ParameterConfig, Any]]
        List of (ParameterConfig, parameter_value) tuples for this iteration
    work_dir : Path
        Working directory for kernel files
    tlm_dataset : pd.DataFrame
        Spacecraft state data (may be modified for OFFSET_KERNEL) with position, velocity, attitude, and time columns
    sci_dataset : pd.DataFrame
        Science frame time data (may be modified for OFFSET_TIME), may include optional measurement columns
    ugps_times : array-like
        Original time array from science dataset
    config : CorrectionConfig
        Configuration with geo settings
    creator : create.KernelCreator
        KernelCreator instance for writing kernels

    Returns
    -------
    param_kernels : list[Path]
        List of parameter-specific kernel file paths
    ugps_times_modified : array-like
        Modified time array if OFFSET_TIME applied, otherwise original times

    Examples
    --------
    >>> param_kernels, times = _create_parameter_kernels(
    ...     params, work_dir, tlm_dataset, sci_dataset, ugps_times, config, creator
    ... )
    >>> # Use in SPICE context with dynamic kernels
    >>> with sp.ext.load_kernel([dynamic_kernels, param_kernels]):
    ...     geo = geolocate(times)
    """
    param_kernels = []
    ugps_times_modified = ugps_times.copy() if hasattr(ugps_times, "copy") else ugps_times

    # Apply each individual parameter change
    logger.info("    Applying parameter changes:")
    for a_param, p_data in params:  # [ParameterConfig, typing.Any]
        # Log parameter details
        param_name = a_param.spec.get("field", "unknown")
        if a_param.ptype == ParameterType.CONSTANT_KERNEL:
            logger.info(f"      {a_param.ptype.name}: {param_name} (constant kernel data)")
        elif a_param.ptype == ParameterType.OFFSET_KERNEL:
            units = a_param.spec.get("units", "")
            logger.info(
                f"      {a_param.ptype.name}: {param_name} = {p_data:.6f} "
                f"(internal units; configured units: {units or 'unspecified'})"
            )
        elif a_param.ptype == ParameterType.OFFSET_TIME:
            units = a_param.spec.get("units", "")
            logger.info(
                f"      {a_param.ptype.name}: {param_name} = {p_data:.6f} "
                f"(internal units; configured units: {units or 'unspecified'})"
            )

        # Create static changing SPICE kernels
        if a_param.ptype == ParameterType.CONSTANT_KERNEL:
            # Aka: BASE-CK, YOKE-CK, HYSICS-CK
            param_kernels.append(
                creator.write_from_json(
                    a_param.config_file,
                    output_kernel=work_dir,
                    input_data=p_data,
                )
            )

        # Create dynamic changing SPICE kernels
        elif a_param.ptype == ParameterType.OFFSET_KERNEL:
            # Aka: AZ-CK, EL-CK
            tlm_dataset_alt = apply_offset(a_param, p_data, tlm_dataset)
            param_kernels.append(
                creator.write_from_json(
                    a_param.config_file,
                    output_kernel=work_dir,
                    input_data=tlm_dataset_alt,
                )
            )

        # Alter non-kernel data
        elif a_param.ptype == ParameterType.OFFSET_TIME:
            # Aka: Frame-times...
            sci_dataset_alt = apply_offset(a_param, p_data, sci_dataset)
            ugps_times_modified = sci_dataset_alt[config.geo.time_field].values

        else:
            raise NotImplementedError(a_param.ptype)

    logger.info(f"    Created {len(param_kernels)} parameter-specific kernels")
    return param_kernels, ugps_times_modified
