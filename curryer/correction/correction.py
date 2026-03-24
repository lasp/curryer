"""Backward-compatibility re-export shim for the correction package.

**Do not add new code here.**

All implementation has been split into focused sub-modules:

- :mod:`curryer.correction.config`      -- config dataclasses & enums
- :mod:`curryer.correction.parameters`  -- parameter set generation
- :mod:`curryer.correction.kernel_ops`  -- SPICE kernel creation & offsets
- :mod:`curryer.correction.results_io`  -- NetCDF read/write
- :mod:`curryer.correction.pipeline`    -- main ``loop()`` orchestration

This module re-exports every public name so that existing code using
``from curryer.correction import correction`` continues to work without
modification.
"""

# Config dataclasses, enums, and JSON loader
from curryer.correction.config import (
    STANDARD_NETCDF_ATTRIBUTES,
    STANDARD_VAR_NAMES,
    CalibrationData,
    CorrectionConfig,
    DataConfig,
    GeolocationConfig,
    ImageMatchingContext,
    KernelContext,
    NetCDFConfig,
    NetCDFParameterMetadata,
    ParameterConfig,
    ParameterType,
    load_config_from_json,
)

# Kernel operations
from curryer.correction.kernel_ops import (
    _create_dynamic_kernels,
    _create_parameter_kernels,
    apply_offset,
)

# Parameter generation
from curryer.correction.parameters import load_param_sets

# Pipeline orchestration
from curryer.correction.pipeline import (
    _aggregate_image_matching_results,
    _compute_parameter_set_metrics,
    _extract_error_metrics,
    _extract_parameter_values,
    _extract_spacecraft_position_midframe,
    _geolocate_and_match,
    _geolocated_to_image_grid,
    _load_calibration_data,
    _load_file,
    _load_image_pair_data,
    _store_gcp_pair_results,
    _store_parameter_values,
    call_error_stats_module,
    image_matching,
    loop,
)

# NetCDF I/O
from curryer.correction.results_io import (
    _build_netcdf_structure,
    _cleanup_checkpoint,
    _load_checkpoint,
    _save_netcdf_checkpoint,
    _save_netcdf_results,
)

__all__ = [
    # Config
    "STANDARD_NETCDF_ATTRIBUTES",
    "STANDARD_VAR_NAMES",
    "CalibrationData",
    "CorrectionConfig",
    "DataConfig",
    "GeolocationConfig",
    "ImageMatchingContext",
    "KernelContext",
    "NetCDFConfig",
    "NetCDFParameterMetadata",
    "ParameterConfig",
    "ParameterType",
    # Parameters
    "load_param_sets",
    # Kernel ops
    "_create_dynamic_kernels",
    "_create_parameter_kernels",
    "apply_offset",
    # Results I/O
    "_build_netcdf_structure",
    "_cleanup_checkpoint",
    "_load_checkpoint",
    "_save_netcdf_checkpoint",
    "_save_netcdf_results",
    # Pipeline
    "_aggregate_image_matching_results",
    "_compute_parameter_set_metrics",
    "_extract_error_metrics",
    "_extract_parameter_values",
    "_extract_spacecraft_position_midframe",
    "_geolocate_and_match",
    "_geolocated_to_image_grid",
    "_load_calibration_data",
    "_load_file",
    "_load_image_pair_data",
    "_store_gcp_pair_results",
    "_store_parameter_values",
    "call_error_stats_module",
    "image_matching",
    "load_config_from_json",
    "loop",
]
