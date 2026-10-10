"""SPICE kernel file management for the correction pipeline.

This module creates and applies parameter-specific SPICE kernels:

- :func:`apply_offset` -- modifies the telemetry (``OFFSET_KERNEL``) or the
  observation frame times (``OFFSET_TIME``).
- :func:`_create_dynamic_kernels` -- writes SC-SPK/SC-CK kernels from
  telemetry data (once per image pair, not per parameter set).
- :func:`_create_parameter_kernels` -- writes parameter-specific kernels
  and applies time offsets for each iteration.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from curryer.correction.config import _CONSTANT_KERNEL_AXES, ParameterConfig, ParameterType
from curryer.kernels import create

if TYPE_CHECKING:
    from curryer.correction.config import GeolocationSetup

logger = logging.getLogger(__name__)

# A CONSTANT_KERNEL CK holds its angles from the GPS epoch to this sentinel uGPS (2050).
_UGPS_EPOCH_END = 2_209_075_218_000_000


def apply_offset(config: ParameterConfig, param_data, input_data):
    """
    Apply parameter offsets to input data based on parameter type.

    Args:
        config: ParameterConfig specifying how to apply the offset
        param_data: The parameter values to apply (offset amounts), in the internal
            units of :func:`~curryer.correction.parameters.load_param_sets`: radians
            for OFFSET_KERNEL, seconds for OFFSET_TIME
        input_data: The input to modify: the telemetry DataFrame for
            OFFSET_KERNEL, the observation's per-frame uGPS times (ndarray) for
            OFFSET_TIME

    Returns:
        Modified copy of input_data with parameter offsets applied

    Raises:
        ValueError: If an OFFSET_KERNEL parameter names no ``field``.
        KeyError: If the OFFSET_KERNEL ``field`` is not a telemetry column.
        TypeError: If OFFSET_TIME *input_data* is not a numeric ndarray.
        NotImplementedError: For CONSTANT_KERNEL, whose angles are written to a
            kernel by :func:`_create_parameter_kernels`, not applied to data.
    """
    logger.info(f"Applying {config.ptype.name} offset to {config.spec.field or 'unknown field'}")

    # Make a copy to avoid modifying the original
    if isinstance(input_data, pd.DataFrame):
        modified_data = input_data.copy()
    else:
        modified_data = input_data.copy() if hasattr(input_data, "copy") else input_data

    if config.ptype == ParameterType.OFFSET_KERNEL:
        # Apply an angle bias (radians) to the telemetry field the AZ/EL kernels are built from
        field_name = config.spec.field
        if not field_name:
            raise ValueError("OFFSET_KERNEL parameter requires 'field' to be specified in config")

        if field_name in modified_data.columns:
            logger.info(f"✓ Applying OFFSET_KERNEL to field '{field_name}': {param_data:.9f} rad")
            modified_data[field_name] = modified_data[field_name] + param_data
        else:
            raise KeyError(
                f"OFFSET_KERNEL field '{field_name}' is not a telemetry column; available: {list(modified_data.columns)}"
            )

    elif config.ptype == ParameterType.OFFSET_TIME:
        # Shift the observation frame times; param_data is seconds, the times uGPS.
        if not isinstance(input_data, np.ndarray) or input_data.dtype.kind not in "iuf":
            raise TypeError(f"OFFSET_TIME applies to an ndarray of uGPS frame times, got {type(input_data).__name__}.")
        modified_data = input_data + param_data * 1e6
        logger.info(f"✓ Applying OFFSET_TIME: {param_data:.6f} s = {param_data * 1e6:.3f} µs to the frame times")

    else:
        raise NotImplementedError(f"Parameter type {config.ptype} not implemented")

    return modified_data


def _create_dynamic_kernels(
    setup: "GeolocationSetup",
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
    setup : GeolocationSetup
        Setup with geo settings and dynamic_kernels list
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
    for kernel_config in setup.geo.dynamic_kernels:
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
    frame_ugps: np.ndarray,
    creator: "create.KernelCreator",
) -> tuple[list[Path], np.ndarray]:
    """Create parameter-specific SPICE kernels and apply time offsets.

    This function applies parameter variations by creating modified kernels
    (CONSTANT_KERNEL, OFFSET_KERNEL) or modifying time tags (OFFSET_TIME).
    Each parameter set produces different kernels and/or time modifications.
    The CONSTANT_KERNEL parameters sharing a ``config_file`` (one per axis,
    see :class:`~curryer.correction.config.Sweep`) are written as one CK
    holding their angles from the GPS epoch to 2050.

    Parameters
    ----------
    params : list[tuple[ParameterConfig, Any]]
        List of (ParameterConfig, parameter_value) tuples for this iteration
    work_dir : Path
        Working directory for kernel files
    tlm_dataset : pd.DataFrame
        Spacecraft state data (may be modified for OFFSET_KERNEL) with position, velocity, attitude, and time columns
    frame_ugps : np.ndarray
        The observation's per-frame times, uGPS (shifted for OFFSET_TIME)
    creator : create.KernelCreator
        KernelCreator instance for writing kernels

    Returns
    -------
    param_kernels : list[Path]
        List of parameter-specific kernel file paths
    frame_ugps_modified : np.ndarray
        Frame times shifted by any OFFSET_TIME, otherwise *frame_ugps*

    Examples
    --------
    >>> param_kernels, times = _create_parameter_kernels(params, work_dir, tlm_dataset, frame_ugps, creator)
    >>> # Use in SPICE context with dynamic kernels
    >>> with sp.ext.load_kernel([dynamic_kernels, param_kernels]):
    ...     geo = geolocate(times)
    """
    param_kernels = []
    frame_ugps_modified = frame_ugps
    constant_angles: dict[Path, dict[str, float]] = {}

    # Apply each individual parameter change
    logger.info("    Applying parameter changes:")
    for a_param, p_data in params:  # [ParameterConfig, typing.Any]
        # Log parameter details
        param_name = a_param.spec.field or "unknown"
        units = a_param.spec.units or ""
        logger.info(
            f"      {a_param.ptype.name}: {param_name} = {p_data:.9f} "
            f"(internal units; configured units: {units or 'unspecified'})"
        )

        # Collect the frame's angles; its CK is written once all axes are known
        if a_param.ptype == ParameterType.CONSTANT_KERNEL:
            # Aka: BASE-CK, YOKE-CK, HYSICS-CK
            constant_angles.setdefault(a_param.config_file, {})[a_param.spec.field] = p_data

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
            frame_ugps_modified = apply_offset(a_param, p_data, frame_ugps_modified)

        else:
            raise NotImplementedError(a_param.ptype)

    for config_file, angles in constant_angles.items():
        ck_data = pd.DataFrame(
            {"ugps": [0, _UGPS_EPOCH_END], **{axis: [angles[axis]] * 2 for axis in _CONSTANT_KERNEL_AXES}}
        )
        # The two rows span the mission; gap chunking would split them into zero-length intervals.
        param_kernels.append(
            creator.write_from_json(
                config_file,
                output_kernel=work_dir,
                input_data=ck_data,
                overrides={"input_gap_threshold": None},
            )
        )

    logger.info(f"    Created {len(param_kernels)} parameter-specific kernels")
    return param_kernels, frame_ugps_modified
