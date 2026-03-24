"""Correction module for iterative geolocation alignment.

Provides tools for Monte Carlo-style correction loops, image matching
against ground control points, PSF modelling, and error statistics.

Sub-module layout
-----------------
config
    Config dataclasses (``CorrectionConfig``, ``ParameterConfig``, etc.)
    and the ``ParameterType`` enum.
parameters
    Random parameter-set generation (:func:`load_param_sets`).
kernel_ops
    SPICE kernel creation and telemetry/time-offset application.
results_io
    NetCDF result file read/write and checkpoint support.
pipeline
    Main :func:`loop` orchestration and all per-iteration helpers.
correction
    Thin re-export shim -- keeps all existing
    ``from curryer.correction import correction`` import paths working.
correction_config
    Utilities for reading and validating JSON config files.
data_structures
    Shared data-container dataclasses (``ImageGrid``, ``PSFGrid``, ...).
dataio
    Validation helpers and S3 data-access utilities.
error_stats
    Error statistics computation (``ErrorStatsProcessor``).
image_match
    Image-matching algorithm (``integrated_image_match``).
pairing
    Ground-control-point pairing utilities.
psf
    Point-spread-function modelling.
search
    Image-search / correlation routines.
verification
    Standalone geolocation compliance check (:func:`verify`).
"""

# Sub-modules (ensure `curryer.correction.psf` etc. work as attributes)
from . import (
    config,
    correction,
    correction_config,
    data_structures,
    dataio,
    error_stats,
    image_match,
    kernel_ops,
    pairing,
    parameters,
    pipeline,
    psf,
    results_io,
    search,
    verification,
)

# Key public names lifted to package level
from .config import (
    CorrectionConfig,
    DataConfig,
    GeolocationConfig,
    NetCDFConfig,
    NetCDFParameterMetadata,
    ParameterConfig,
    ParameterType,
    SearchStrategy,
    load_config_from_json,
)
from .data_structures import ImageGrid, PSFGrid, PSFSamplingConfig, SearchConfig
from .error_stats import ErrorStatsConfig, ErrorStatsProcessor
from .pipeline import loop
from .verification import GCPError, RequirementsConfig, VerificationResult, verify

__all__ = [
    # Sub-modules
    "config",
    "correction",
    "correction_config",
    "data_structures",
    "dataio",
    "error_stats",
    "image_match",
    "kernel_ops",
    "pairing",
    "parameters",
    "pipeline",
    "psf",
    "results_io",
    "search",
    "verification",
    # Config
    "CorrectionConfig",
    "DataConfig",
    "GeolocationConfig",
    "NetCDFConfig",
    "NetCDFParameterMetadata",
    "ParameterConfig",
    "ParameterType",
    "SearchStrategy",
    "load_config_from_json",
    # Pipeline entry point
    "loop",
    # Data structures
    "ImageGrid",
    "PSFGrid",
    "PSFSamplingConfig",
    "SearchConfig",
    # Error stats
    "ErrorStatsConfig",
    "ErrorStatsProcessor",
    # Verification
    "GCPError",
    "RequirementsConfig",
    "VerificationResult",
    "verify",
]
