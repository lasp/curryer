"""Configuration models and enumerations for the geolocation correction pipeline.

This module defines the data structures that represent the complete configuration
for a correction analysis run, including:

- ``ParameterType`` – enum of the three parameter application strategies
- ``ParameterSpec`` – sampling specification for a single correction parameter
- ``ParameterConfig`` – assembles a parameter's type, kernel file, and sampling spec
- ``GeolocationConfig`` – SPICE kernel paths and instrument settings
- ``NetCDFParameterMetadata`` / ``NetCDFConfig`` – NetCDF output metadata (re-exported from io_config)
- ``GeolocationSetup`` – durable, mission-specific setup (built once, reused across sweeps)
- ``Sweep`` – the lightweight, frequently-varied parameter experiment
- ``OutputConfig`` – output settings (NetCDF metadata + filename)
- ``KernelContext``, ``CalibrationData``, ``ImageMatchingContext`` – lightweight NamedTuples
  used to pass state between pipeline helper functions
- ``load_setup_from_json`` / ``load_sweep_from_json`` / ``load_config_files`` – build the
  ``setup`` / ``sweep`` / ``output`` models from a JSON file

All mission-specific values (kernel filenames, parameter ranges, instrument names)
live in mission configuration modules (e.g. ``tests/test_correction/clarreo_config.py``)
and are injected via ``GeolocationSetup`` / ``Sweep``.

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
from collections.abc import Callable  # noqa: E402  (kept adjacent to other stdlib usage)
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from curryer import meta

from curryer.correction.io_config import (  # noqa: E402, F401
    DEFAULT_NETCDF_ATTRIBUTES,
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


class ParameterType(str, Enum):
    """Parameter types used in the correction configuration.

    Specifies how a parameter is applied during geolocation analysis:
    whether as a constant kernel value, a kernel offset, or a time offset.
    String-valued so JSON configs use readable names (``"OFFSET_TIME"``).

    Attributes
    ----------
    CONSTANT_KERNEL
        Set a specific kernel value (e.g., fixed rotation angles).
    OFFSET_KERNEL
        Modify input kernel data by an offset.
    OFFSET_TIME
        Modify input timetags by an offset.
    """

    CONSTANT_KERNEL = "CONSTANT_KERNEL"  # Set a specific value.
    OFFSET_KERNEL = "OFFSET_KERNEL"  # Modify input kernel data by an offset.
    OFFSET_TIME = "OFFSET_TIME"  # Modify input timetags by an offset


class SearchStrategy(str, Enum):
    """Strategy used to generate parameter sets during correction analysis.

    Attributes
    ----------
    RANDOM
        Monte Carlo random walk (current default).  Each iteration draws an
        independent sample from a normal distribution centred on the
        parameter's ``current_value`` with the specified ``sigma``, clipped
        to ``bounds``.  Requires ``seed`` and ``n_iterations`` on
        :class:`Sweep`.
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

    Strict by design (``extra="forbid"``): unknown fields raise a
    ``ValidationError`` so typos surface immediately.  Mission-specific extras
    that the pipeline does not interpret go in :attr:`metadata`.

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
    metadata
        Free-form mission-specific extras not interpreted by the pipeline
        (e.g. a display ``"name"``, provenance, calibration date).
    """

    model_config = ConfigDict(extra="forbid")

    current_value: float | list[float] = 0.0
    bounds: list[float] = Field(default_factory=lambda: [-1.0, 1.0])
    sigma: float | None = None
    units: str | None = None
    distribution: str = "normal"
    field: str | None = None
    transformation_type: str | None = None
    coordinate_frames: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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

    Held as the required ``requirements`` field on :class:`GeolocationSetup`
    and consumed by :func:`~curryer.correction.verification.verify` and the
    correction verdict.

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
# Setup / Sweep / Output — the config surface
# ============================================================================
#
# ``GeolocationSetup`` holds the durable, mission-specific setup (built once);
# ``Sweep`` is the lightweight experiment varied between runs; ``OutputConfig``
# holds output settings.  A run is ``run_correction(setup, sweep, inputs, work_dir)``.


class CalibrationFiles(BaseModel):
    """Direct paths to instrument calibration inputs.

    Both fields are optional and interim: real line-of-sight vectors and
    spacecraft geometry will be SPICE-derived from telemetry rather than loaded
    from files, so nothing in the pipeline *requires* these.

    Attributes
    ----------
    los_vectors_file
        Per-detector line-of-sight unit vectors (instrument frame).
    psf_file
        Optical point-spread-function calibration.
    """

    los_vectors_file: Path | None = None
    psf_file: Path | None = None


class GeolocationSetup(BaseModel):
    """Durable, mission-specific setup for geolocation correction/verification.

    Built once per mission and reused across many :class:`Sweep` runs.  Holds
    everything that does *not* change when you vary which parameters are swept:
    SPICE kernels and instrument identity (:class:`GeolocationConfig`), the
    pass/fail :class:`RequirementsConfig`, how input data is read
    (:class:`DataConfig`), static instrument calibration
    (:class:`CalibrationFiles`), the science-Dataset variable names, and an
    optional custom image-matching implementation.

    Attributes
    ----------
    geo
        SPICE kernels, instrument name, and science time field.
    requirements
        Pass/fail thresholds used by verification and the correction verdict.
    data_config
        How telemetry/science files are read.  ``None`` uses CSV defaults.
    calibration
        Optional direct calibration file paths.  ``None`` when geometry is
        supplied another way (e.g. SPICE-derived).
    spacecraft_position_name, boresight_name, transformation_matrix_name
        Variable names for the spacecraft-state fields in the image-matching
        ``xr.Dataset`` (mission-configurable; generic defaults).
    image_matching_func
        Optional custom image-matching callable.  ``None`` uses the built-in
        :func:`~curryer.correction.verification.image_matching`.  Excluded from
        JSON serialisation because callables are not serialisable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    geo: GeolocationConfig
    requirements: RequirementsConfig
    data_config: DataConfig | None = None
    calibration: CalibrationFiles | None = None

    spacecraft_position_name: str = "sc_position"
    boresight_name: str = "boresight"
    transformation_matrix_name: str = "t_inst2ref"

    image_matching_func: Callable | None = Field(default=None, exclude=True)


class Sweep(BaseModel):
    """A parameter-variation experiment run against a :class:`GeolocationSetup`.

    Lightweight and cheap to copy, so a setup can be held fixed while rapidly
    trying parameter variations.

    Attributes
    ----------
    parameters
        The parameters to vary (at least one).
    search_strategy
        How parameter sets are generated (RANDOM / GRID_SEARCH / SINGLE_OFFSET).
    n_iterations
        Iterations for RANDOM and values-per-parameter for SINGLE_OFFSET;
        ignored by GRID_SEARCH.
    seed
        Random seed for reproducible RANDOM sweeps.
    grid_points_per_param
        Evenly-spaced points per parameter for GRID_SEARCH.
    max_grid_sets
        Safety cap on total GRID_SEARCH parameter sets.
    """

    parameters: list[ParameterConfig] = Field(min_length=1)
    search_strategy: SearchStrategy = SearchStrategy.RANDOM
    n_iterations: int = Field(default=10, gt=0)
    seed: int | None = None
    grid_points_per_param: int = Field(default=10, ge=2)
    max_grid_sets: int = Field(default=100_000, ge=1)

    @model_validator(mode="after")
    def _validate_search_strategy(self) -> "Sweep":
        """Ensure strategy-specific settings are consistent."""
        if self.search_strategy in (SearchStrategy.GRID_SEARCH, SearchStrategy.SINGLE_OFFSET):
            if not self.parameters:
                raise ValueError(
                    f"SearchStrategy.{self.search_strategy.name} requires at least one parameter in `parameters`."
                )
        return self


class OutputConfig(BaseModel):
    """Output settings for a correction run.

    Attributes
    ----------
    netcdf
        NetCDF structure/metadata config.  ``None`` is auto-populated by
        :func:`~curryer.correction.pipeline.run_correction` from the setup's
        performance threshold.
    output_filename
        Output NetCDF filename.  ``None`` falls back to the default in
        :meth:`get_output_filename`.
    """

    netcdf: NetCDFConfig | None = None
    output_filename: str | None = None

    def get_output_filename(self, default: str = "correction_results.nc") -> str:
        """Return :attr:`output_filename` if set, otherwise *default*."""
        return self.output_filename or default


# ============================================================================
# Typed Input Structure
# ============================================================================


class CorrectionInput(BaseModel):
    """A single input set for the correction loop.

    Replaces the positional tuple ``(telemetry_path, science_path, gcp_path)``
    with named fields for clarity and IDE autocomplete.

    The reader for each file is chosen by :attr:`DataConfig.file_format`, so the
    inputs are format-agnostic.  The first-class real-data path is a NetCDF
    image observation (radiance as the science variable) carrying telemetry,
    metadata, and science times; ``.mat`` files are interim test scaffolding.

    Parameters
    ----------
    telemetry_file : Path
        Telemetry observation file (NetCDF for real data; CSV/HDF5 also read).
    science_file : Path
        Science/timing observation file (NetCDF for real data; CSV/HDF5 also read).
    gcp_file : Path
        GCP reference-image file (NetCDF or ``.mat``).

    Examples
    --------
    >>> from curryer.correction import CorrectionInput
    >>> inputs = [
    ...     CorrectionInput(
    ...         telemetry_file="data/obs_20240317.nc",
    ...         science_file="data/obs_20240317.nc",
    ...         gcp_file="gcps/landsat_chip_001.nc",
    ...     )
    ... ]
    """

    telemetry_file: Path
    science_file: Path
    gcp_file: Path


# ============================================================================
# Setup / Sweep / Output JSON loading
# ============================================================================


def _read_config_json(config_path: Path) -> dict:
    """Read and parse a JSON config file, raising clear errors on failure."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}") from e


def load_setup_from_json(config_path: Path) -> GeolocationSetup:
    """Load a :class:`GeolocationSetup` from the ``"setup"`` section of a JSON file."""
    data = _read_config_json(config_path)
    if "setup" not in data:
        raise KeyError(f"Missing required 'setup' section in {config_path}")
    return GeolocationSetup.model_validate(data["setup"])


def load_sweep_from_json(config_path: Path) -> Sweep:
    """Load a :class:`Sweep` from the ``"sweep"`` section of a JSON file."""
    data = _read_config_json(config_path)
    if "sweep" not in data:
        raise KeyError(f"Missing required 'sweep' section in {config_path}")
    return Sweep.model_validate(data["sweep"])


def load_config_files(config_path: Path) -> tuple[GeolocationSetup, Sweep, OutputConfig]:
    """Load ``(GeolocationSetup, Sweep, OutputConfig)`` from one JSON file.

    The file has three top-level sections — ``"setup"``, ``"sweep"``, and an
    optional ``"output"`` — each validated directly against its model.  The
    ``"sweep".parameters`` entries mirror :class:`ParameterConfig` (``ptype`` /
    ``config_file`` / ``spec``); rotation frames are authored as a single
    ``CONSTANT_KERNEL`` parameter with ``spec.current_value = [roll, pitch, yaw]``.
    """
    data = _read_config_json(config_path)
    for section in ("setup", "sweep"):
        if section not in data:
            raise KeyError(f"Missing required '{section}' section in {config_path}")
    setup = GeolocationSetup.model_validate(data["setup"])
    sweep = Sweep.model_validate(data["sweep"])
    output = OutputConfig.model_validate(data.get("output", {}))
    return setup, sweep, output
