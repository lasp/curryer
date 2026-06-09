"""Correction module for iterative geolocation alignment.

Provides tools for Monte Carlo-style correction loops, image matching
against ground control points, PSF modelling, and error statistics.

Sub-module layout
-----------------
config
    Config models (``GeolocationSetup``, ``Sweep``, ``OutputConfig``,
    ``ParameterConfig``, etc.) and the ``ParameterType`` enum.
io_config
    NetCDF output configuration and standard attribute definitions
    (:class:`NetCDFConfig`, :class:`NetCDFParameterMetadata`,
    :data:`DEFAULT_NETCDF_ATTRIBUTES`).
parameters
    Random parameter-set generation (:func:`load_param_sets`).
kernel_ops
    SPICE kernel creation and telemetry/time-offset application.
results_io
    NetCDF result file read/write and checkpoint support.
pipeline
    Main :func:`loop` orchestration and all per-iteration helpers.
    Preferred-name aliases: :func:`run_correction`, :func:`run_image_matching`,
    :func:`compute_error_stats`.
    :func:`run_correction` also accepts :class:`CorrectionInput` objects.
grid_types
    Pure grid data containers (``ImageGrid``, ``PSFGrid``, ...).
dataio
    Validation helpers and S3 data-access utilities.  The S3 utilities
    (``S3Configuration``, ``find_netcdf_objects``, ``download_netcdf_objects``)
    are optional convenience helpers for mission-specific data pipelines;
    the core correction API does not depend on them.
error_stats
    Error statistics computation (``ErrorStatsProcessor``).
image_match
    Image-matching algorithm (``integrated_image_match``).
io
    Unified path resolution (``resolve_path``).  Transparently handles
    local paths and S3 URIs (``s3://…``) when ``boto3`` is installed.
    Optional convenience — the public API contract is local ``Path``
    objects; S3 support is opt-in.
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
    dataio,
    error_stats,
    grid_types,
    image_io,
    image_match,
    io,
    io_config,
    kernel_ops,
    pairing,
    parameters,
    pipeline,
    psf,
    regrid,
    results,
    results_io,
    search,
    verification,
)

# Key public names lifted to package level
from .config import (
    CalibrationFiles,
    CorrectionInput,
    DataConfig,
    GeolocationConfig,
    GeolocationSetup,
    NetCDFConfig,
    NetCDFParameterMetadata,
    OutputConfig,
    ParameterConfig,
    ParameterSpec,
    ParameterType,
    PSFSamplingConfig,
    RegridConfig,
    RequirementsConfig,
    SearchConfig,
    SearchStrategy,
    Sweep,
    load_config_files,
    load_setup_from_json,
    load_sweep_from_json,
)
from .error_stats import ErrorStatsConfig, ErrorStatsProcessor, compute_percent_below
from .grid_types import ImageGrid, PSFGrid
from .image_io import (
    geolocated_to_image_grid,
    infer_spacecraft_state,
    load_image_grid,
    load_los_vectors,
    load_named_image_grid,
    load_observation_file,
    load_optical_psf,
    save_image_grid,
)
from .io import resolve_path
from .pipeline import compute_error_stats, loop, run_correction, run_image_matching
from .results import CorrectionResult, ParameterSetResult
from .verification import GCPError, VerificationResult, compare_results, match_geolocated_to_gcp_files, verify

__all__ = [
    # Sub-modules
    "config",
    "dataio",
    "error_stats",
    "grid_types",
    "image_io",
    "image_match",
    "io",
    "io_config",
    "kernel_ops",
    "pairing",
    "parameters",
    "pipeline",
    "psf",
    "regrid",
    "results",
    "results_io",
    "search",
    "verification",
    # Config
    "CalibrationFiles",
    "CorrectionInput",
    "DataConfig",
    "GeolocationConfig",
    "GeolocationSetup",
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
    # Pipeline entry points
    "loop",
    "run_correction",
    "compute_error_stats",
    "run_image_matching",
    "match_geolocated_to_gcp_files",
    # Data structures
    "ImageGrid",
    "PSFGrid",
    "PSFSamplingConfig",
    "RegridConfig",
    "SearchConfig",
    # Image I/O helpers
    "load_image_grid",
    "load_named_image_grid",
    "load_observation_file",
    "load_los_vectors",
    "load_optical_psf",
    "save_image_grid",
    "infer_spacecraft_state",
    "geolocated_to_image_grid",
    # Error stats
    "ErrorStatsConfig",
    "ErrorStatsProcessor",
    "compute_percent_below",
    # IO
    "resolve_path",
    # Verification
    "GCPError",
    "VerificationResult",
    "compare_results",
    "verify",
    # Structured results
    "CorrectionResult",
    "ParameterSetResult",
]
