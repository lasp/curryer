"""Backward-compatibility re-export shim for the correction package.

**Do not add new code here.**

All implementation has been split into focused sub-modules:

- :mod:`curryer.correction.config`       -- config dataclasses & enums
- :mod:`curryer.correction.parameters`   -- parameter set generation
- :mod:`curryer.correction.kernel_ops`   -- SPICE kernel creation & offsets
- :mod:`curryer.correction.results_io`   -- NetCDF read/write
- :mod:`curryer.correction.pipeline`     -- main ``loop()`` orchestration
- :mod:`curryer.correction.verification` -- image matching, GCP pairing, error stats
- :mod:`curryer.correction.image_io`     -- image grid I/O utilities
"""

# Config dataclasses, enums, and JSON loader
from curryer.correction.config import (
    DEFAULT_NETCDF_ATTRIBUTES,
    CalibrationData,
    CalibrationFiles,
    DataConfig,
    GeolocationConfig,
    GeolocationSetup,
    ImageMatchingContext,
    KernelContext,
    NetCDFConfig,
    NetCDFParameterMetadata,
    OutputConfig,
    ParameterConfig,
    ParameterSpec,
    ParameterType,
    RequirementsConfig,
    SearchStrategy,
    Sweep,
    load_config_files,
    load_setup_from_json,
    load_sweep_from_json,
)

# Image I/O utilities
from curryer.correction.image_io import geolocated_to_image_grid

# Kernel operations
from curryer.correction.kernel_ops import (
    _create_dynamic_kernels,
    _create_parameter_kernels,
    apply_offset,
)

# Parameter generation
from curryer.correction.parameters import load_param_sets

# Pipeline orchestration (functions that genuinely live in pipeline)
from curryer.correction.pipeline import (
    _compute_parameter_set_metrics,
    _extract_error_metrics,
    _extract_parameter_values,
    _geolocate_and_match,
    _load_calibration_data,
    _load_file,
    _load_image_pair_data,
    _resolve_gcp_pairs,
    _store_gcp_pair_results,
    _store_parameter_values,
    call_error_stats_module,
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

# Verification pipeline (image matching, GCP pairing, error aggregation)
from curryer.correction.verification import (
    _aggregate_image_matching_results,
    _extract_spacecraft_position_midframe,
    image_matching,
    match_geolocated_to_gcp_files,
)

__all__ = [
    # Config
    "DEFAULT_NETCDF_ATTRIBUTES",
    "CalibrationData",
    "CalibrationFiles",
    "DataConfig",
    "GeolocationConfig",
    "GeolocationSetup",
    "ImageMatchingContext",
    "KernelContext",
    "NetCDFConfig",
    "NetCDFParameterMetadata",
    "OutputConfig",
    "ParameterConfig",
    "ParameterSpec",
    "ParameterType",
    "RequirementsConfig",
    "SearchStrategy",
    "Sweep",
    "load_config_files",
    "load_setup_from_json",
    "load_sweep_from_json",
    # Parameters
    "load_param_sets",
    # Kernel ops
    "_create_dynamic_kernels",
    "_create_parameter_kernels",
    "apply_offset",
    # Image I/O
    "geolocated_to_image_grid",
    # Results I/O
    "_build_netcdf_structure",
    "_cleanup_checkpoint",
    "_load_checkpoint",
    "_save_netcdf_checkpoint",
    "_save_netcdf_results",
    # Pipeline
    "_compute_parameter_set_metrics",
    "_extract_error_metrics",
    "_extract_parameter_values",
    "_geolocate_and_match",
    "_load_calibration_data",
    "_load_file",
    "_load_image_pair_data",
    "_resolve_gcp_pairs",
    "_store_gcp_pair_results",
    "_store_parameter_values",
    "call_error_stats_module",
    "loop",
    # Verification
    "_aggregate_image_matching_results",
    "_extract_spacecraft_position_midframe",
    "image_matching",
    "match_geolocated_to_gcp_files",
]
