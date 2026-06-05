"""Configuration models and enumerations for the geolocation correction pipeline.

This module defines the data structures that represent the complete configuration
for a correction analysis run, including:

- ``ParameterType`` – enum of the three parameter application strategies
- ``ParameterSpec`` – sampling specification for a single correction parameter
- ``ParameterConfig`` – assembles a parameter's type, kernel file, and sampling spec
- ``GeolocationConfig`` – SPICE kernel paths and instrument settings
- ``NetCDFParameterMetadata`` / ``NetCDFConfig`` – NetCDF output metadata (re-exported from io_config)
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

Parameter configuration — three cooperating classes
----------------------------------------------------
These three classes each capture a distinct, orthogonal concern:

``ParameterType`` (enum)
    *How the value is applied to the pipeline.*  Each member maps to a different
    pipeline code path:

    - ``CONSTANT_KERNEL`` — replace the kernel value with the sampled value
    - ``OFFSET_KERNEL``   — shift the existing kernel value by the sampled offset
    - ``OFFSET_TIME``     — shift the input timestamps by the sampled offset

    ``ParameterType`` belongs on :class:`ParameterConfig` (not on
    :class:`ParameterSpec`) because it describes application mechanics, not
    sampling statistics.

``ParameterSpec``
    *How to draw samples.*  Holds the statistical description of the search
    space: the nominal ``current_value``, the ``bounds`` that clip samples, the
    ``sigma`` for normal-distribution sampling, physical ``units``, and optional
    ``field`` / ``coordinate_frames`` hints consumed by kernel-creation routines.
    ``ParameterSpec`` is intentionally agnostic about how any drawn value will
    be applied — that is ``ParameterType``'s job.

``ParameterConfig``
    *Assembles the three concerns.*  A single ``ParameterConfig`` answers:
    "vary *this* kernel file (``config_file``), applied as *this kind* of change
    (``ptype``), drawing samples according to *this spec* (``spec``)."
    ``ParameterConfig`` is also the right place for cross-field validation (e.g.
    confirming that ``data.field`` is provided whenever ``ptype`` requires it).

Typical construction::

    ParameterConfig(
        ptype=ParameterType.OFFSET_TIME,
        config_file=None,          # time offsets need no kernel file
        spec=ParameterSpec(
            current_value=0.0,
            bounds=[-50.0, 50.0],
            sigma=10.0,
            units="milliseconds",
            field="time_ugps",     # required for OFFSET_KERNEL / OFFSET_TIME
        ),
    )
"""

import json
import logging
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

import curryer.correction.correction_config as _correction_config

if TYPE_CHECKING:
    from curryer import meta

from curryer.correction.io_config import (  # noqa: E402, F401
    DEFAULT_NETCDF_ATTRIBUTES,
    STANDARD_VAR_NAMES,
    NetCDFConfig,
    NetCDFParameterMetadata,
)

logger = logging.getLogger(__name__)

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
    position_columns
        Explicit column name mappings for telemetry spacecraft-position data,
        e.g. ``["sc_pos_x", "sc_pos_y", "sc_pos_z"]``.  ``None`` means use
        mission defaults from the geolocation configuration.
    """

    file_format: Literal["csv", "netcdf", "hdf5"] = "csv"
    time_scale_factor: float = 1.0
    # Explicit column name mappings for telemetry spacecraft-position data.
    # e.g. ["sc_pos_x", "sc_pos_y", "sc_pos_z"]. None means use mission defaults from the geolocation configuration.
    position_columns: list[str] | None = None


# ============================================================================
# Parameter Configuration
# ============================================================================


class ParameterType(Enum):
    """Parameter types used in the correction configuration.

    Specifies how a parameter is applied during geolocation analysis:
    whether as a constant kernel value, a kernel offset, or a time offset.

    Attributes
    ----------
    CONSTANT_KERNEL
        Set a specific kernel value (e.g., fixed rotation angles).
    OFFSET_KERNEL
        Modify input kernel data by an offset.
    OFFSET_TIME
        Modify input timetags by an offset.
    """

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


class ParameterSpec(BaseModel):
    """Typed sampling specification for a single correction parameter.

    Supports dict-style access (``get``, ``__getitem__``, ``__contains__``)
    for backward compatibility with code written against the old ``dict``-based
    ``ParameterConfig.spec`` API.

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
    spec
        Sampling specification.  Accepts a plain ``dict`` or ``None`` on
        construction (Pydantic coerces both to :class:`ParameterSpec`
        automatically; ``None`` becomes an empty ``ParameterSpec()``).
    """

    ptype: ParameterType
    config_file: Path | None = None
    spec: ParameterSpec = Field(default_factory=ParameterSpec)

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_spec(cls, values: Any) -> Any:
        """Convert ``spec=None`` to an empty ``ParameterSpec`` (backward compat)."""
        if isinstance(values, dict) and values.get("spec") is None:
            values = dict(values)
            values["spec"] = {}
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
# Image-Matching Configuration
# ============================================================================


@dataclass
class PSFSamplingConfig:
    """Configuration for PSF sampling during image matching.

    Default values are calibrated for Landsat 30 m GCPs, which are the
    standard reference for GCP-based geolocation verification. Override
    all fields for instruments with different ground resolution or motion
    characteristics.

    Parameters
    ----------
    gcp_step_m : float, optional
        Ground control point step size in metres. Default is 30.0.
    motion_convolution_step_m : float or None, optional
        Step size for spacecraft motion convolution in metres.
        If ``None`` (default), derived as ``gcp_step_m / 20.0`` in
        ``__post_init__``.
    psf_lat_sample_dist_deg : float, optional
        PSF sample distance in the latitude direction in degrees.
        Default approximately 2.7 m at the equator (Landsat calibration).
    psf_lon_sample_dist_deg : float, optional
        PSF sample distance in the longitude direction in degrees.
        Default approximately 2.7 m at the equator (Landsat calibration).
    """

    gcp_step_m: float = 30.0
    motion_convolution_step_m: float | None = None  # defaults to gcp_step_m / 20.0 if None
    psf_lat_sample_dist_deg: float = 2.4397105613972e-05
    psf_lon_sample_dist_deg: float = 2.8737038710207e-05

    def __post_init__(self) -> None:
        if self.motion_convolution_step_m is None:
            self.motion_convolution_step_m = self.gcp_step_m / 20.0


@dataclass
class SearchConfig:
    """Configuration for the image-matching correlation search grid.

    Default values are tuned for Landsat 30 m GCPs. Override all fields
    for instruments with different ground resolution.

    Parameters
    ----------
    grid_size : int, optional
        Number of grid points per axis in the correlation search grid.
        Default 44 (Landsat-tuned).
    grid_span_km : float, optional
        Half-width of the search grid in kilometres. Default 11.0.
    reduction_factor : float, optional
        Multiplicative reduction applied to grid spacing each iteration.
        Default 0.8.
    spacing_limit_m : float, optional
        Minimum grid spacing in metres; search stops when reached.
        Default 10.0 (Landsat-tuned).
    """

    grid_size: int = 44
    grid_span_km: float = 11.0
    reduction_factor: float = 0.8
    spacing_limit_m: float = 10.0


class RegridConfig(BaseModel):
    """Configuration for GCP chip regridding.

    Specifies output grid parameters for transforming irregular geodetic grids
    to regular latitude/longitude grids. ECEF → geodetic conversion always
    uses the WGS84 ellipsoid, which is the only ellipsoid supported by
    ``curryer.compute.spatial.ecef_to_geodetic``.

    Parameters
    ----------
    output_grid_size : tuple[int, int], optional
        Desired output grid dimensions as (nrows, ncols). Mutually exclusive
        with ``output_resolution_deg``.
    output_resolution_deg : tuple[float, float], optional
        Desired output resolution as (dlat, dlon) in degrees. Mutually
        exclusive with ``output_grid_size``. Required when ``output_bounds``
        is set.
    output_bounds : tuple[float, float, float, float], optional
        Explicit output grid bounds as (minlon, maxlon, minlat, maxlat) in
        degrees. Requires ``output_resolution_deg``.
    conservative_bounds : bool, default=True
        If True, shrink bounds to ensure all output points lie within the
        input irregular grid (avoids edge extrapolation).
    interpolation_method : str, default="bilinear"
        Interpolation method; one of ``"bilinear"`` or ``"nearest"``.
    fill_value : float, default=NaN
        Value assigned to output points that fall outside the input grid.
    """

    output_grid_size: tuple[int, int] | None = None
    output_resolution_deg: tuple[float, float] | None = None
    output_bounds: tuple[float, float, float, float] | None = None
    conservative_bounds: bool = True
    interpolation_method: str = "bilinear"
    fill_value: float = float("nan")

    @field_validator("interpolation_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate interpolation method name."""
        valid = {"bilinear", "nearest"}
        if v not in valid:
            raise ValueError(f"interpolation_method must be one of {valid}, got '{v}'")
        return v

    @field_validator("output_grid_size")
    @classmethod
    def validate_grid_size(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        """Validate that grid size has at least 2 rows and 2 columns."""
        if v is not None:
            if v[0] < 2:
                raise ValueError(f"Grid size must have at least 2 rows and 2 columns, got {v}")
            if v[1] < 2:
                raise ValueError(f"Grid size must have at least 2 rows and 2 columns, got {v}")
        return v

    @field_validator("output_resolution_deg")
    @classmethod
    def validate_resolution(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        """Validate that resolution values are positive."""
        if v is not None:
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError(f"Resolution values must be positive (dlat, dlon), got {v}")
        return v

    @field_validator("output_bounds")
    @classmethod
    def validate_bounds(cls, v: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
        """Validate that bounds are properly ordered."""
        if v is not None:
            minlon, maxlon, minlat, maxlat = v
            if minlon >= maxlon:
                raise ValueError(f"minlon must be < maxlon, got {minlon} >= {maxlon}")
            if minlat >= maxlat:
                raise ValueError(f"minlat must be < maxlat, got {minlat} >= {maxlat}")
        return v

    @model_validator(mode="after")
    def validate_grid_spec(self) -> "RegridConfig":
        """Validate that grid specification options are mutually consistent."""
        has_size = self.output_grid_size is not None
        has_res = self.output_resolution_deg is not None
        has_bounds = self.output_bounds is not None

        if has_size and has_res:
            raise ValueError("Cannot specify both output_grid_size and output_resolution_deg")
        if has_bounds and not has_res:
            raise ValueError("output_bounds requires output_resolution_deg")
        if has_bounds and has_size:
            raise ValueError("Cannot specify both output_bounds and output_grid_size")
        return self


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
        data_config : DataConfig | None
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
    n_iterations: int = Field(gt=0)
    parameters: list[ParameterConfig] = Field(min_length=1)

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
    performance_threshold_m: float = Field(gt=0)
    performance_spec_percent: float = Field(ge=0, le=100)

    # DATA LOADING CONFIGURATION (config-driven; replaces mission-specific loader callables)
    data_config: DataConfig | None = None

    # Private test-injection override for image matching.
    # Not part of the public API; not serialised to JSON (PrivateAttr is always excluded).
    # Usage: config._image_matching_override = your_func
    # TODO(#151): Add Requirement model with evaluate_all() for multi-metric requirements.
    _image_matching_override: Any = PrivateAttr(default=None)

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_search_strategy(self) -> "CorrectionConfig":
        """Ensure strategy-specific settings are consistent."""
        if self.search_strategy in (SearchStrategy.GRID_SEARCH, SearchStrategy.SINGLE_OFFSET):
            if not self.parameters:
                raise ValueError(
                    f"SearchStrategy.{self.search_strategy.name} requires at least one parameter in `parameters`."
                )
        return self

    @model_validator(mode="after")
    def _populate_netcdf_config(self) -> "CorrectionConfig":
        """Auto-populate :attr:`netcdf` with defaults when it is not supplied.

        Guarantees ``config.netcdf`` is always a usable :class:`NetCDFConfig`,
        so downstream result/IO code can read it without a ``None`` guard.  The
        default threshold is inherited from :attr:`performance_threshold_m`.
        """
        if self.netcdf is None:
            self.netcdf = NetCDFConfig(performance_threshold_m=self.performance_threshold_m)
        return self

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def get_calibration_file(self, file_type: str, default: str | None = None) -> str:
        """Return the configured calibration filename for *file_type*.

        Parameters
        ----------
        file_type : str
            Calibration file key (e.g. ``"psf"`` or ``"los_vectors"``) to look
            up in :attr:`calibration_file_names`.
        default : str, optional
            Filename returned when *file_type* is not present in
            :attr:`calibration_file_names`.

        Raises
        ------
        ValueError
            If *file_type* is unconfigured and no *default* is given.
        """
        if self.calibration_file_names and file_type in self.calibration_file_names:
            return self.calibration_file_names[file_type]
        if default:
            return default
        raise ValueError(f"No calibration file configured for type: {file_type}")

    def validate(self, check_loaders: bool = False):
        """Validate that all required configuration values are present.

        Parameters
        ----------
        check_loaders : bool, optional
            Accepted for backward compatibility but has no effect.  Loader
            callables no longer exist on this config; data loading is driven
            by the ``data_config`` field (:class:`DataConfig`).
        """
        logger.debug("CorrectionConfig validation passed")

    def ensure_netcdf_config(self):
        """Ensure :attr:`netcdf` exists, creating it with defaults if needed.

        Retained for backward compatibility.  As of the ``_populate_netcdf_config``
        model validator, :attr:`netcdf` is auto-populated at construction, so this
        is normally a no-op; it still guards against ``netcdf`` being reset to
        ``None`` after construction.
        """
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

    logger.info(f"Loading Correction configuration from: {config_path}")

    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")

    # Extract mission configuration and kernel mappings
    mission_config = _correction_config.extract_mission_config(config_data)
    constant_kernel_map = _correction_config.get_kernel_mapping(config_data, "constant_kernel")
    offset_kernel_map = _correction_config.get_kernel_mapping(config_data, "offset_kernel")

    logger.debug(f"Mission: {mission_config.get('mission_name', 'UNKNOWN')}")
    logger.debug(f"Constant kernel mappings: {constant_kernel_map}")
    logger.debug(f"Offset kernel mappings: {offset_kernel_map}")

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
            if param_name.endswith("_roll"):
                base_name = param_name[:-5]
                angle_type = "roll"
            elif param_name.endswith("_pitch"):
                base_name = param_name[:-6]
                angle_type = "pitch"
            elif param_name.endswith("_yaw"):
                base_name = param_name[:-4]
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
                logger.debug(f"Mapped CONSTANT_KERNEL '{base_name}' → {kernel_file}")
            else:
                logger.warning(f"No kernel mapping found for CONSTANT_KERNEL parameter: {base_name}")

        else:
            # OFFSET_KERNEL and OFFSET_TIME parameters are individual
            param_groups[param_name] = {"type": ptype, "param_dict": param_dict, "config_file": None}

            if ptype == ParameterType.OFFSET_KERNEL:
                kernel_file = _correction_config.find_kernel_file(param_name, offset_kernel_map)
                if kernel_file:
                    param_groups[param_name]["config_file"] = Path(kernel_file)
                    logger.debug(f"Mapped OFFSET_KERNEL '{param_name}' → {kernel_file}")
                elif param_dict.get("config_file"):
                    param_groups[param_name]["config_file"] = Path(param_dict["config_file"])
                    logger.debug(f"Using explicit config_file for OFFSET_KERNEL '{param_name}'")
                else:
                    logger.warning(f"No kernel mapping found for OFFSET_KERNEL parameter: {param_name}")

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
            ParameterConfig(ptype=group_data["type"], config_file=group_data["config_file"], spec=param_data)
        )

    logger.info(
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

    if "meta_kernel_file" not in geo_config:
        raise KeyError("Missing required 'meta_kernel_file' in geolocation config section.")
    if "generic_kernel_dir" not in geo_config:
        raise KeyError("Missing required 'generic_kernel_dir' in geolocation config section.")
    geo = GeolocationConfig(
        meta_kernel_file=Path(geo_config["meta_kernel_file"]),
        generic_kernel_dir=Path(geo_config["generic_kernel_dir"]),
        dynamic_kernels=[Path(k) for k in geo_config.get("dynamic_kernels", [])],
        instrument_name=instrument_name,
        time_field=time_field,
    )

    # Extract required mission-specific parameters from correction section
    earth_radius_m = corr_config.get("earth_radius_m")
    if earth_radius_m is not None:
        logger.warning(
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

    # Optional calibration paths (direct file paths or directory + filename map)
    calibration_dir_raw = corr_config.get("calibration_dir")
    calibration_dir = Path(calibration_dir_raw) if calibration_dir_raw else None
    calibration_file_names = corr_config.get("calibration_file_names")
    los_vectors_file_raw = corr_config.get("los_vectors_file")
    los_vectors_file = Path(los_vectors_file_raw) if los_vectors_file_raw else None
    psf_file_raw = corr_config.get("psf_file")
    psf_file = Path(psf_file_raw) if psf_file_raw else None

    config = CorrectionConfig(
        seed=corr_config.get("seed"),
        n_iterations=corr_config.get("n_iterations", 10),
        parameters=parameters,
        geo=geo,
        performance_threshold_m=performance_threshold_m,
        performance_spec_percent=performance_spec_percent,
        calibration_dir=calibration_dir,
        calibration_file_names=calibration_file_names,
        los_vectors_file=los_vectors_file,
        psf_file=psf_file,
    )

    config.validate()

    logger.info(
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
    >>> result = run_correction(config, work_dir, inputs)
    >>> results = result.results
    >>> netcdf_data = result.netcdf_data
    """

    telemetry_file: Path
    science_file: Path
    gcp_file: Path
