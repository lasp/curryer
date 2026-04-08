"""Configuration models and enumerations for the geolocation correction pipeline.

This module defines the data structures that represent the complete configuration
for a correction analysis run, including:

- ``ParameterType`` – enum of the three parameter variation strategies
- ``ParameterData`` – typed container for a parameter's sampling spec
- ``ParameterConfig`` – a single parameter to vary (kernel or time offset)
- ``GeolocationConfig`` – SPICE kernel paths and instrument settings
- ``NetCDFParameterMetadata`` / ``NetCDFConfig`` – NetCDF output metadata
- ``CorrectionConfig`` – the single top-level config object passed to ``pipeline.loop()``
- ``KernelContext``, ``CalibrationData``, ``ImageMatchingContext`` – lightweight NamedTuples
  used to pass state between pipeline helper functions
- ``load_config_from_json`` – build a ``CorrectionConfig`` from a JSON file

All mission-specific values (kernel filenames, parameter ranges, instrument names)
live in mission configuration modules (e.g. ``tests/test_correction/clarreo_config.py``)
and are injected via ``CorrectionConfig``.

All config objects are ``pydantic.BaseModel`` subclasses which provide:
- Automatic type validation and clear ``ValidationError`` messages on construction
- Free JSON serialization via ``model_dump_json()`` / ``model_validate_json()``
- IDE autocomplete on every field
"""

import json
import logging
import warnings
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

if TYPE_CHECKING:
    from curryer import meta

# ============================================================================
# Standard NetCDF Variable Attributes (Mission-Agnostic)
# ============================================================================

STANDARD_NETCDF_ATTRIBUTES = {
    # Geolocation error metrics (per GCP pair)
    "rms_error_m": {"units": "meters", "long_name": "RMS geolocation error"},
    "mean_error_m": {"units": "meters", "long_name": "Mean geolocation error"},
    "max_error_m": {"units": "meters", "long_name": "Maximum geolocation error"},
    "std_error_m": {"units": "meters", "long_name": "Standard deviation of geolocation error"},
    "n_measurements": {"units": "count", "long_name": "Number of measurement points"},
    # Aggregate performance metrics (per parameter set)
    "mean_rms_all_pairs": {"units": "meters", "long_name": "Mean RMS error across all GCP pairs"},
    "worst_pair_rms": {"units": "meters", "long_name": "Worst performing GCP pair RMS error"},
    "best_pair_rms": {"units": "meters", "long_name": "Best performing GCP pair RMS error"},
    # Image matching metrics (per GCP pair)
    "im_lat_error_km": {"units": "kilometers", "long_name": "Image matching latitude error"},
    "im_lon_error_km": {"units": "kilometers", "long_name": "Image matching longitude error"},
    "im_ccv": {"units": "dimensionless", "long_name": "Image matching correlation coefficient"},
    "im_grid_step_m": {"units": "meters", "long_name": "Image matching final grid step size"},
}


# ============================================================================
# Standard Data Variable Names (Mission-Agnostic Keys)
# ============================================================================

# Standard variable names that should be present in image matching results.
# Used for extracting data from xarray.Dataset objects.
STANDARD_VAR_NAMES = {
    # Error measurements (required)
    "lat_error_deg": "lat_error_deg",
    "lon_error_deg": "lon_error_deg",
    # Spacecraft state (configurable names)
    "spacecraft_position": "sc_position",  # Generic default
    "boresight": "boresight",  # Generic default
    "transformation_matrix": "t_inst2ref",  # Generic default
    # Control point location (optional)
    "gcp_lat_deg": "gcp_lat_deg",
    "gcp_lon_deg": "gcp_lon_deg",
    "gcp_alt": "gcp_alt",
}


# ============================================================================
# Pipeline Helper NamedTuples (not config – pass-through state only)
# ============================================================================


class KernelContext(NamedTuple):
    """Context for SPICE kernel loading during geolocation."""

    mkrn: "meta.MetaKernel"
    dynamic_kernels: list[Path]
    param_kernels: list[Path]


class CalibrationData(NamedTuple):
    """Pre-loaded calibration data for image matching."""

    los_vectors: np.ndarray | None
    optical_psfs: list | None


class ImageMatchingContext(NamedTuple):
    """Context data needed for image matching operations."""

    gcp_pairs: list[tuple]
    params: list[tuple]
    pair_idx: int
    sci_key: str


# ============================================================================
# Data Loading Configuration
# ============================================================================


class DataConfig(BaseModel):
    """Configuration for config-driven internal data loading.

    Replaces mission-specific loader callables with a declarative specification
    of how files should be read.  The pipeline reads telemetry and science data
    directly from the provided file paths using pandas/xarray, applying the
    ``time_scale_factor`` to convert the science time column to uGPS.

    Attributes
    ----------
    file_format
        File format for both telemetry and science data files.
        ``"csv"`` uses :func:`pandas.read_csv`; ``"netcdf"`` converts via
        :func:`xarray.open_dataset`; ``"hdf5"`` uses :func:`pandas.read_hdf`.
    time_scale_factor
        Multiply science timestamps by this factor to obtain uGPS
        (microseconds since GPS epoch).  For example, ``1e6`` converts GPS
        seconds to uGPS; ``1.0`` means the file already contains uGPS.
        The time column name is taken from :attr:`GeolocationConfig.time_field`
        (single source of truth).
    """

    file_format: Literal["csv", "netcdf", "hdf5"] = "csv"
    time_scale_factor: float = 1.0
    # Explicit column name mappings for telemetry spacecraft-position data.
    # e.g. ["sc_pos_x", "sc_pos_y", "sc_pos_z"].  None means use mission defaults.
    position_columns: list[str] | None = None


# ============================================================================
# Parameter Configuration
# ============================================================================


class ParameterType(Enum):
    CONSTANT_KERNEL = auto()  # Set a specific value.
    OFFSET_KERNEL = auto()  # Modify input kernel data by an offset.
    OFFSET_TIME = auto()  # Modify input timetags by an offset


class SearchStrategy(str, Enum):
    """Strategy used to generate parameter sets during correction analysis.

    Attributes
    ----------
    RANDOM
        Monte Carlo random walk (current default).  Each iteration draws an
        independent sample from a normal distribution centred on the
        parameter's ``current_value`` with the specified ``sigma``, clipped
        to ``bounds``.  Requires ``seed`` and ``n_iterations`` on
        :class:`CorrectionConfig`.
    GRID_SEARCH
        Deterministic cartesian-product sweep.  For every parameter,
        ``grid_points_per_param`` evenly-spaced values are generated across
        the full ``bounds`` offset range and the cartesian product of all
        per-parameter grids is enumerated.  ``n_iterations`` is ignored.
    SINGLE_OFFSET
        Deterministic single-parameter sweep.  Each parameter is varied
        independently across ``n_iterations`` evenly-spaced values (spanning
        its ``bounds`` offset range) while all other parameters are held at
        their nominal ``current_value``.  Total parameter sets produced:
        ``len(parameters) × n_iterations``.
    """

    RANDOM = "random"
    GRID_SEARCH = "grid"
    SINGLE_OFFSET = "single"


class ParameterData(BaseModel):
    """Typed sampling specification for a single correction parameter.

    Supports dict-style access (``get``, ``__getitem__``, ``__contains__``)
    for backward compatibility with code written against the old ``dict``-based
    ``ParameterConfig.data`` API.

    Attributes
    ----------
    current_value
        Baseline parameter value(s).  A scalar for OFFSET_KERNEL/OFFSET_TIME
        and a 3-element list ``[roll, pitch, yaw]`` for CONSTANT_KERNEL.
    bounds
        ``[min, max]`` offset limits (same units as ``sigma``).
    sigma
        Standard deviation for normal-distribution sampling.  ``None`` means
        the parameter is held fixed at ``current_value``.
    units
        Physical units string, e.g. ``"arcseconds"`` or ``"milliseconds"``.
    distribution
        Sampling distribution name.  Stored for documentation purposes;
        the current implementation always uses a normal distribution.
    field
        Telemetry / science DataFrame column that this parameter modifies
        (required for ``OFFSET_KERNEL`` and ``OFFSET_TIME``).
    transformation_type
        Optional hint consumed by kernel-creation routines (e.g.
        ``"dcm_rotation"`` or ``"angle_bias"``).
    coordinate_frames
        Optional list of SPICE frame names affected by this parameter.
    """

    model_config = ConfigDict(extra="allow")

    current_value: float | list[float] = 0.0
    bounds: list[float] = Field(default_factory=lambda: [-1.0, 1.0])
    sigma: float | None = None
    units: str | None = None
    distribution: str = "normal"
    field: str | None = None
    transformation_type: str | None = None
    coordinate_frames: list[str] | None = None

    # ------------------------------------------------------------------
    # Backward-compatible dict-style access
    # ------------------------------------------------------------------

    def _get_raw(self, key: str) -> Any:
        """Return the raw value for *key* from declared fields or extra fields."""
        if key in type(self).model_fields:
            return getattr(self, key, None)
        extra = self.__pydantic_extra__ or {}
        return extra.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """``dict.get()`` shim for backward compatibility.

        Returns *default* when the value is ``None`` (i.e. field was not
        explicitly set), mirroring ``dict.get`` on a mapping that only
        contains keys with non-``None`` values.
        """
        val = self._get_raw(key)
        return default if val is None else val

    def __contains__(self, key: str) -> bool:
        """``key in data`` shim – ``True`` when the value is not ``None``."""
        return self._get_raw(key) is not None

    def __getitem__(self, key: str) -> Any:
        """``data[key]`` shim for backward compatibility."""
        if key in type(self).model_fields:
            return getattr(self, key)
        extra = self.__pydantic_extra__ or {}
        if key in extra:
            return extra[key]
        raise KeyError(key)


class ParameterConfig(BaseModel):
    """A single parameter to vary during correction analysis.

    Attributes
    ----------
    ptype
        How this parameter is applied (constant kernel, offset kernel, or
        time offset).
    config_file
        Path to the SPICE kernel JSON template, or ``None`` for time
        offsets that require no kernel file.
    data
        Sampling specification.  Accepts a plain ``dict`` or ``None`` on
        construction (Pydantic coerces both to :class:`ParameterData`
        automatically; ``None`` becomes an empty ``ParameterData()``).
    """

    ptype: ParameterType
    config_file: Path | None = None
    data: ParameterData = Field(default_factory=ParameterData)

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_data(cls, values: Any) -> Any:
        """Convert ``data=None`` to an empty ``ParameterData`` (backward compat)."""
        if isinstance(values, dict) and values.get("data") is None:
            values = dict(values)
            values["data"] = {}
        return values


# ============================================================================
# Geolocation Configuration
# ============================================================================


class GeolocationConfig(BaseModel):
    """SPICE kernel paths and instrument settings for geolocation.

    Attributes
    ----------
    meta_kernel_file
        Path to the mission meta-kernel JSON file.
    generic_kernel_dir
        Directory containing generic/shared SPICE kernels.
    dynamic_kernels
        Kernels regenerated from telemetry each run (SC-SPK, SC-CK, etc.)
        but *not* altered by parameter variations.
    instrument_name
        SPICE instrument name (e.g. ``"CPRS_HYSICS"``).
    time_field
        Column name in the science DataFrame that holds uGPS timestamps.
    minimum_correlation
        Optional image-matching quality filter threshold (0.0–1.0).
    """

    meta_kernel_file: Path
    generic_kernel_dir: Path
    dynamic_kernels: list[Path] = Field(default_factory=list)
    instrument_name: str
    time_field: str
    minimum_correlation: float | None = None


# ============================================================================
# NetCDF Output Configuration
# ============================================================================


class NetCDFParameterMetadata(BaseModel):
    """NetCDF metadata for a single output parameter variable."""

    variable_name: str
    units: str
    long_name: str


class NetCDFConfig(BaseModel):
    """Configuration for NetCDF output structure and metadata.

    Attributes
    ----------
    performance_threshold_m
        Accuracy threshold in metres used to derive threshold-specific
        variable names (e.g. ``"percent_under_250m"``).
    title
        Global title attribute for the output NetCDF file.
    description
        Global description attribute for the output NetCDF file.
    parameter_metadata
        Optional mapping of parameter key → :class:`NetCDFParameterMetadata`.
        Auto-generated from ``CorrectionConfig.parameters`` when ``None``.
    standard_attributes
        Optional mission-specific attribute overrides.  Falls back to the
        module-level :data:`STANDARD_NETCDF_ATTRIBUTES` when ``None``.
    """

    performance_threshold_m: float
    title: str = "Correction Geolocation Analysis Results"
    description: str = "Parameter sensitivity analysis"
    parameter_metadata: dict[str, NetCDFParameterMetadata] | None = None
    standard_attributes: dict[str, dict[str, str]] | None = None

    def get_threshold_metric_name(self) -> str:
        """Generate metric name dynamically from threshold."""
        threshold_m = int(self.performance_threshold_m)
        return f"percent_under_{threshold_m}m"

    def get_standard_attributes(self) -> dict[str, dict[str, str]]:
        """Get standard variable attributes, using mission overrides if provided."""
        if self.standard_attributes is not None:
            return self.standard_attributes
        return STANDARD_NETCDF_ATTRIBUTES.copy()

    def get_parameter_netcdf_metadata(
        self, param_config: "ParameterConfig", angle_type: str | None = None
    ) -> "NetCDFParameterMetadata":
        """Get NetCDF metadata for a parameter."""
        if param_config.config_file:
            param_stem = param_config.config_file.stem
            lookup_key = f"{param_stem}_{angle_type}" if angle_type else param_stem
        else:
            lookup_key = f"param_{param_config.ptype.name.lower()}"

        if self.parameter_metadata and lookup_key in self.parameter_metadata:
            return self.parameter_metadata[lookup_key]

        return self._auto_generate_metadata(param_config, angle_type, lookup_key)

    def _auto_generate_metadata(
        self, param_config: "ParameterConfig", angle_type: str | None, base_key: str
    ) -> "NetCDFParameterMetadata":
        """Auto-generate NetCDF metadata from parameter configuration."""
        if param_config.ptype == ParameterType.CONSTANT_KERNEL:
            units = "arcseconds"
        elif param_config.ptype == ParameterType.OFFSET_KERNEL:
            units = "arcseconds"
        elif param_config.ptype == ParameterType.OFFSET_TIME:
            units = "milliseconds"
        else:
            units = "unknown"

        # Use declared units field (replaces old isinstance(data, dict) check)
        if param_config.data.units is not None:
            units = param_config.data.units

        var_name = base_key.replace(".", "_").replace("-", "_")
        if not var_name.startswith("param_"):
            var_name = f"param_{var_name}"

        if param_config.config_file:
            file_stem = param_config.config_file.stem
            clean_name = file_stem.replace("_v01", "").replace("_v02", "").replace(".attitude.ck", "")
            clean_name = clean_name.replace("_", " ").title()
            if angle_type:
                long_name = f"{clean_name} {angle_type} correction"
            else:
                long_name = f"{clean_name} correction"
        else:
            long_name = f"{param_config.ptype.name.replace('_', ' ').title()} parameter"

        return NetCDFParameterMetadata(variable_name=var_name, units=units, long_name=long_name)


# ============================================================================
# Verification Requirements Configuration
# ============================================================================


class RequirementsConfig(BaseModel):
    """Verification requirements / thresholds.

    Can be attached as an optional ``verification`` field on
    :class:`CorrectionConfig`, or passed directly to
    :func:`~curryer.correction.verification.verify`.  When neither is supplied,
    :func:`~curryer.correction.verification.verify` falls back to
    :attr:`CorrectionConfig.performance_threshold_m` and
    :attr:`CorrectionConfig.performance_spec_percent`.

    Attributes
    ----------
    performance_threshold_m : float
        Per-measurement nadir-equivalent error limit in metres.
        A measurement *passes* when its error is **below** this value.
    performance_spec_percent : float
        Minimum fraction of measurements (0–100) that must pass for the
        overall verification to be considered successful.
    """

    performance_threshold_m: float
    performance_spec_percent: float


# ============================================================================
# Top-Level Correction Configuration
# ============================================================================


class CorrectionConfig(BaseModel):
    """The configuration object for geolocation correction analysis.

    This config contains everything needed for a Correction run:
    - What parameters to vary (parameters list)
    - How to vary them (seed, n_iterations)
    - How to load data (telemetry_loader, science_loader)
    - How to process data (gcp_pairing_func, image_matching_func)
    - Geolocation settings (geo: GeolocationConfig)
    - Success criteria (performance_threshold_m, performance_spec_percent)
    - Output configuration (netcdf: NetCDFConfig, output_filename)

    Create one CorrectionConfig object and pass it to pipeline.loop() to run.

    Serialisation
    -------------
    ``model_dump_json()`` / ``model_validate_json()`` provide lossless
    JSON round-trips for all typed fields.  Callable fields (loaders,
    pairing/matching functions) are **excluded** from serialisation because
    they cannot be represented as JSON; re-attach them after deserialising.

    Parameters
    ----------
    CORE CORRECTION SETTINGS:
        seed : int | None
            Random seed for reproducibility, or None for non-reproducible runs.
        n_iterations : int
            Number of parameter set iterations.
        parameters : list[ParameterConfig]
            Parameters to vary (defines sensitivity analysis).

    GEOLOCATION & PERFORMANCE REQUIREMENTS:
        geo : GeolocationConfig
        performance_threshold_m : float
        performance_spec_percent : float

    DATA LOADING CONFIGURATION:
        data : DataConfig | None
            Specifies file format, time field, scale factor, and optional GCP
            discovery settings.  When provided, telemetry and science files are
            read internally by the pipeline from the paths supplied in
            ``tlm_sci_gcp_sets``.

    PROCESSING FUNCTION (optional override):
        image_matching_func
            Defaults to the built-in ``pipeline.image_matching`` when ``None``.
            Override only for missions with fundamentally different matching.

    OUTPUT CONFIGURATION:
        netcdf : NetCDFConfig | None
        output_filename : str | None

    CALIBRATION CONFIGURATION:
        calibration_dir : Path | None
        calibration_file_names : dict[str, str] | None

    MISSION-SPECIFIC NAMING:
        spacecraft_position_name, boresight_name, transformation_matrix_name
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # CORE CORRECTION SETTINGS
    seed: int | None = None
    n_iterations: int
    parameters: list[ParameterConfig]

    # SEARCH STRATEGY
    search_strategy: SearchStrategy = SearchStrategy.RANDOM
    grid_points_per_param: int = Field(
        default=10,
        ge=2,
        description="Number of evenly-spaced grid points per parameter for GRID_SEARCH strategy.",
    )
    max_grid_sets: int = Field(
        default=100_000,
        ge=1,
        description=(
            "Hard upper bound on the total number of parameter sets that GRID_SEARCH may materialise. "
            "Prevents accidental out-of-memory runs caused by large cartesian products "
            "(e.g. 10 points × 6 params = 1,000,000 sets). "
            "Raise this value deliberately, or switch to SINGLE_OFFSET for high-dimensional sweeps."
        ),
    )

    # GEOLOCATION & PERFORMANCE REQUIREMENTS
    geo: GeolocationConfig
    performance_threshold_m: float
    performance_spec_percent: float

    # DATA LOADING CONFIGURATION (config-driven; replaces mission-specific loader callables)
    data: DataConfig | None = None

    # Private test-injection override for image matching.
    # Not part of the public API; not serialised to JSON (PrivateAttr is always excluded).
    # Usage: config._image_matching_override = your_func
    # TODO(#151): Add Requirement model with evaluate_all() for multi-metric requirements.
    _image_matching_override: Any = PrivateAttr(default=None)

    @property
    def image_matching_func(self) -> Any:
        """Deprecated — use ``_image_matching_override`` for test injection.

        .. deprecated::
            Set ``config._image_matching_override = func`` instead.
            This property will be removed in a future release.
        """
        warnings.warn(
            "image_matching_func is deprecated. Use config._image_matching_override = func for test injection.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._image_matching_override

    @image_matching_func.setter
    def image_matching_func(self, value: Any) -> None:
        warnings.warn(
            "image_matching_func is deprecated. Use config._image_matching_override = func for test injection.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._image_matching_override = value

    # OUTPUT CONFIGURATION
    netcdf: NetCDFConfig | None = None
    output_filename: str | None = None

    # CALIBRATION CONFIGURATION
    calibration_dir: Path | None = None
    calibration_file_names: dict[str, str] | None = None
    # Direct calibration file paths (alternative to calibration_dir + calibration_file_names)
    psf_file: Path | None = None
    los_vectors_file: Path | None = None

    # MISSION-SPECIFIC NAMING
    spacecraft_position_name: str = "sc_position"
    boresight_name: str = "boresight"
    transformation_matrix_name: str = "t_inst2ref"

    @model_validator(mode="after")
    def _validate_search_strategy(self) -> "CorrectionConfig":
        """Ensure strategy-specific settings are consistent."""
        if self.search_strategy in (SearchStrategy.GRID_SEARCH, SearchStrategy.SINGLE_OFFSET):
            if not self.parameters:
                raise ValueError(
                    f"SearchStrategy.{self.search_strategy.name} requires at least one parameter in `parameters`."
                )
        return self

    def get_calibration_file(self, file_type: str, default: str = None) -> str:
        """Get calibration filename for given type with fallback to default."""
        if self.calibration_file_names and file_type in self.calibration_file_names:
            return self.calibration_file_names[file_type]
        if default:
            return default
        raise ValueError(f"No calibration file configured for type: {file_type}")

    def validate(self, check_loaders: bool = False):
        """Validate that all required configuration values are present.

        Args:
            check_loaders: Deprecated – accepted for backward compatibility but
                           has no effect.  Loader callables no longer exist on
                           this config; data loading is driven by the ``data``
                           field (:class:`DataConfig`).

        Raises:
            ValueError: If any required fields are missing or invalid
        """
        import logging

        logger = logging.getLogger(__name__)
        errors = []

        if self.n_iterations is None or self.n_iterations <= 0:
            errors.append("n_iterations must be a positive integer")

        if self.parameters is None or len(self.parameters) == 0:
            errors.append("parameters list cannot be empty")

        if self.geo is None:
            errors.append("geo (GeolocationConfig) is required")

        if self.performance_threshold_m is None or self.performance_threshold_m <= 0:
            errors.append("performance_threshold_m must be a positive number (e.g., 250.0 meters)")

        if self.performance_spec_percent is None or not (0 <= self.performance_spec_percent <= 100):
            errors.append("performance_spec_percent must be between 0 and 100 (e.g., 39.0)")

        if errors:
            error_msg = "CorrectionConfig validation failed:\n  - " + "\n  - ".join(errors)
            error_msg += "\n\nThese values must be provided in your mission configuration."
            error_msg += "\nSee tests/test_correction/clarreo_config.py for an example."
            raise ValueError(error_msg)

        logger.debug("CorrectionConfig validation passed")

    def ensure_netcdf_config(self):
        """Ensure NetCDFConfig exists, creating with defaults if needed."""
        if self.netcdf is None:
            self.netcdf = NetCDFConfig(performance_threshold_m=self.performance_threshold_m)

    def get_output_filename(self, default: str = "correction_results.nc") -> str:
        """Get output filename with optional auto-generation."""
        if self.output_filename:
            return self.output_filename
        return default

    @staticmethod
    def generate_timestamped_filename(prefix: str = "correction", suffix: str = "") -> str:
        """Generate a timestamped output filename for production use."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            return f"{prefix}_{timestamp}_{suffix}.nc"
        return f"{prefix}_{timestamp}.nc"


# ============================================================================
# JSON Config Loading
# ============================================================================

# Imported here (not at module top) to avoid a circular import with the
# correction_config sibling module, which itself has no dependency on config.
from curryer.correction import correction_config as _correction_config  # noqa: E402

_config_logger = logging.getLogger(__name__)


def load_config_from_json(config_path: Path) -> "CorrectionConfig":
    """Load correction configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file (e.g., gcs_config.json)

    Returns:
        CorrectionConfig object populated from the JSON file

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
        KeyError: If required config sections are missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    _config_logger.info(f"Loading Correction configuration from: {config_path}")

    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")

    # Extract mission configuration and kernel mappings
    mission_config = _correction_config.extract_mission_config(config_data)
    constant_kernel_map = _correction_config.get_kernel_mapping(config_data, "constant_kernel")
    offset_kernel_map = _correction_config.get_kernel_mapping(config_data, "offset_kernel")

    _config_logger.debug(f"Mission: {mission_config.get('mission_name', 'UNKNOWN')}")
    _config_logger.debug(f"Constant kernel mappings: {constant_kernel_map}")
    _config_logger.debug(f"Offset kernel mappings: {offset_kernel_map}")

    # Validate required sections exist
    if "correction" not in config_data:
        raise KeyError("Missing required 'correction' section in config file")
    if "geolocation" not in config_data:
        raise KeyError("Missing required 'geolocation' section in config file")

    # Extract correction section
    corr_config = config_data.get("correction", {})
    geo_config = config_data.get("geolocation", {})

    # Validate correction section
    if "parameters" not in corr_config:
        raise KeyError("Missing required 'parameters' in correction section")
    if not isinstance(corr_config["parameters"], list):
        raise ValueError("'parameters' must be a list")
    if len(corr_config["parameters"]) == 0:
        raise ValueError("No parameters defined in configuration")

    # Parse parameters and group related ones together
    parameters = []
    param_groups = {}

    # First pass: group parameters by their base name and type
    for param_dict in corr_config.get("parameters", []):
        param_name = param_dict.get("name", "")
        ptype_str = param_dict.get("parameter_type", "CONSTANT_KERNEL")
        ptype = ParameterType[ptype_str]

        # Group CONSTANT_KERNEL parameters by their base frame name
        if ptype == ParameterType.CONSTANT_KERNEL:
            # Extract base name (e.g., "hysics_to_cradle" from "hysics_to_cradle_roll")
            if "_roll" in param_name:
                base_name = param_name.replace("_roll", "")
                angle_type = "roll"
            elif "_pitch" in param_name:
                base_name = param_name.replace("_pitch", "")
                angle_type = "pitch"
            elif "_yaw" in param_name:
                base_name = param_name.replace("_yaw", "")
                angle_type = "yaw"
            else:
                base_name = param_name
                angle_type = "single"

            if base_name not in param_groups:
                param_groups[base_name] = {"type": ptype, "angles": {}, "template": param_dict, "config_file": None}

            param_groups[base_name]["angles"][angle_type] = param_dict.get("initial_value", 0.0)

            # Determine config file based on kernel mapping from config
            kernel_file = _correction_config.find_kernel_file(base_name, constant_kernel_map)
            if kernel_file:
                param_groups[base_name]["config_file"] = Path(kernel_file)
                _config_logger.debug(f"Mapped CONSTANT_KERNEL '{base_name}' → {kernel_file}")
            else:
                _config_logger.warning(f"No kernel mapping found for CONSTANT_KERNEL parameter: {base_name}")

        else:
            # OFFSET_KERNEL and OFFSET_TIME parameters are individual
            param_groups[param_name] = {"type": ptype, "param_dict": param_dict, "config_file": None}

            if ptype == ParameterType.OFFSET_KERNEL:
                kernel_file = _correction_config.find_kernel_file(param_name, offset_kernel_map)
                if kernel_file:
                    param_groups[param_name]["config_file"] = Path(kernel_file)
                    _config_logger.debug(f"Mapped OFFSET_KERNEL '{param_name}' → {kernel_file}")
                elif param_dict.get("config_file"):
                    param_groups[param_name]["config_file"] = Path(param_dict["config_file"])
                    _config_logger.debug(f"Using explicit config_file for OFFSET_KERNEL '{param_name}'")
                else:
                    _config_logger.warning(f"No kernel mapping found for OFFSET_KERNEL parameter: {param_name}")

    # Second pass: create ParameterConfig objects from groups
    for group_name, group_data in param_groups.items():
        if group_data["type"] == ParameterType.CONSTANT_KERNEL:
            template = group_data["template"]
            angles = group_data["angles"]
            center_values = [angles.get("roll", 0.0), angles.get("pitch", 0.0), angles.get("yaw", 0.0)]
            param_data = {
                "current_value": center_values,
                "bounds": template.get("bounds", [-100, 100]),
                "sigma": template.get("sigma"),
                "units": template.get("units", "arcseconds"),
                "distribution": template.get("distribution_type", "normal"),
                "field": template.get("application_target", {}).get("field_name", None),
            }
        else:
            param_dict = group_data["param_dict"]
            param_data = {
                "current_value": param_dict.get("current_value", param_dict.get("initial_value", 0.0)),
                "bounds": param_dict.get("bounds", [-100, 100]),
                "sigma": param_dict.get("sigma"),
                "units": param_dict.get("units", "radians"),
                "distribution": param_dict.get("distribution_type", "normal"),
                "field": (param_dict.get("field") or param_dict.get("application_target", {}).get("field_name", None)),
            }

        parameters.append(
            ParameterConfig(ptype=group_data["type"], config_file=group_data["config_file"], data=param_data)
        )

    _config_logger.info(
        f"Loaded {len(parameters)} parameter groups from {len(corr_config.get('parameters', []))} individual parameters"
    )

    # Parse geolocation configuration
    default_instrument = mission_config.get("instrument_name")
    instrument_name = geo_config.get("instrument_name", default_instrument)
    if instrument_name is None:
        raise ValueError("instrument_name must be specified in config (either in geolocation or mission section)")

    time_field = geo_config.get("time_field")
    if time_field is None:
        raise ValueError("time_field must be specified in geolocation config")

    geo = GeolocationConfig(
        meta_kernel_file=Path(geo_config.get("meta_kernel_file", "")),
        generic_kernel_dir=Path(geo_config.get("generic_kernel_dir", "")),
        dynamic_kernels=[Path(k) for k in geo_config.get("dynamic_kernels", [])],
        instrument_name=instrument_name,
        time_field=time_field,
    )

    # Extract required mission-specific parameters from correction section
    earth_radius_m = corr_config.get("earth_radius_m")
    if earth_radius_m is not None:
        _config_logger.warning(
            "earth_radius_m in config is deprecated and ignored. "
            "The WGS84 value from curryer.compute.constants is used instead."
        )

    performance_threshold_m = corr_config.get("performance_threshold_m")
    if performance_threshold_m is None:
        raise KeyError(
            "Missing required 'performance_threshold_m' in correction config section. "
            "This must be specified for your mission (e.g., 250.0 meters for CLARREO)."
        )

    performance_spec_percent = corr_config.get("performance_spec_percent")
    if performance_spec_percent is None:
        raise KeyError(
            "Missing required 'performance_spec_percent' in correction config section. "
            "This must be specified for your mission (e.g., 39.0 percent for CLARREO)."
        )

    config = CorrectionConfig(
        seed=corr_config.get("seed"),
        n_iterations=corr_config.get("n_iterations", 10),
        parameters=parameters,
        geo=geo,
        performance_threshold_m=performance_threshold_m,
        performance_spec_percent=performance_spec_percent,
    )

    config.validate()

    _config_logger.info(
        f"Configuration loaded and validated: {config.n_iterations} iterations, "
        f"{len(config.parameters)} parameter groups"
    )
    return config


# ============================================================================
# Typed Input Structure
# ============================================================================


class CorrectionInput(BaseModel):
    """A single input set for the correction loop.

    Replaces the positional tuple ``(telemetry_path, science_path, gcp_path)``
    with named fields for clarity and IDE autocomplete.

    Parameters
    ----------
    telemetry_file : Path
        Path to the telemetry CSV (or NetCDF/HDF5) file.
    science_file : Path
        Path to the science/timing CSV (or NetCDF/HDF5) file.
    gcp_file : Path
        Path to the GCP reference image (``.mat`` file).

    Examples
    --------
    >>> from curryer.correction import CorrectionInput, run_correction
    >>> inputs = [
    ...     CorrectionInput(
    ...         telemetry_file="data/tlm_20240317.csv",
    ...         science_file="data/sci_20240317.csv",
    ...         gcp_file="gcps/landsat_chip_001.mat",
    ...     )
    ... ]
    >>> results, netcdf_data = run_correction(config, work_dir, inputs)
    """

    telemetry_file: Path
    science_file: Path
    gcp_file: Path
