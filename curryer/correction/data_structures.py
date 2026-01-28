from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageGrid:
    """Container for image data sampled on a latitude/longitude grid."""

    data: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    h: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        self.lat = np.asarray(self.lat, dtype=float)
        self.lon = np.asarray(self.lon, dtype=float)
        if self.h is not None:
            self.h = np.asarray(self.h, dtype=float)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.lat.shape != self.lon.shape:
            raise ValueError("Latitude and longitude grids must have matching shapes.")
        if self.lat.shape != self.data.shape:
            raise ValueError("Data array must match the shape of the latitude/longitude grids.")
        if self.h is not None and self.h.shape != self.data.shape:
            raise ValueError("Height grid must match the shape of the data array.")

    @property
    def mid_indices(self) -> tuple[int, int]:
        """Return the row/column indices of the central pixel."""
        rows, cols = self.data.shape
        return rows // 2, cols // 2


@dataclass
class NamedImageGrid(ImageGrid):
    """Image grid with an associated descriptive name."""

    name: str | None = None


@dataclass
class PSFGrid:
    """Point spread function sampled on a latitude/longitude grid."""

    data: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        self.lat = np.asarray(self.lat, dtype=float)
        self.lon = np.asarray(self.lon, dtype=float)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        data_shape = self.data.shape
        valid_lat = self.lat.shape == data_shape or (self.lat.ndim == 1 and self.lat.size in data_shape)
        valid_lon = self.lon.shape == data_shape or (self.lon.ndim == 1 and self.lon.size in data_shape)
        if not valid_lat or not valid_lon:
            raise ValueError(
                "Latitude/longitude arrays must either match PSF data shape or "
                "represent one-dimensional coordinate axes."
            )


@dataclass
class ProjectedPSF(PSFGrid):
    """PSF sampled on the Earth's surface, optionally carrying heights."""

    height: np.ndarray | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.height is not None:
            self.height = np.asarray(self.height, dtype=float)
            if self.height.shape != self.data.shape:
                raise ValueError("Height array must match PSF data shape.")


@dataclass
class OpticalPSFEntry:
    """Optical PSF sample defined by cross/along slit angles."""

    data: np.ndarray
    x: np.ndarray
    field_angle: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.field_angle = np.asarray(self.field_angle, dtype=float)
        if self.data.shape != (self.field_angle.size, self.x.size):
            raise ValueError("Optical PSF data must have shape (len(field_angle), len(x)).")


@dataclass
class GeolocationConfig:
    """Configuration parameters for PSF geolocation modelling."""

    gcp_step_m: float = 30.0
    motion_convolution_step_m: float = gcp_step_m / 20.0
    psf_lat_sample_dist_deg: float = 2.4397105613972e-05
    psf_lon_sample_dist_deg: float = 2.8737038710207e-05


@dataclass
class SearchConfig:
    """Parameters controlling the image matching search grid."""

    grid_size: int = 44
    grid_span_km: float = 11.0
    reduction_factor: float = 0.8
    spacing_limit_m: float = 10.0


@dataclass
class RegridConfig:
    """Configuration for GCP chip regridding.

    Specifies output grid parameters for transforming irregular geodetic grids
    to regular latitude/longitude grids. Exactly one of the grid specification
    methods should be provided.

    Parameters
    ----------
    output_grid_size : tuple[int, int], optional
        Output dimensions (nrows, ncols). If provided, resolution is calculated
        automatically from input bounds.
    output_resolution_deg : tuple[float, float], optional
        Output resolution (dlat, dlon) in degrees. If provided, grid size is
        calculated automatically from input bounds. Recommended for mission-specific
        configuration (e.g., match detector pixel size).
    output_bounds : tuple[float, float, float, float], optional
        Explicit output bounds (minlon, maxlon, minlat, maxlat) in degrees.
        Must be combined with output_resolution_deg.
    conservative_bounds : bool, default=True
        If True, shrink output bounds to ensure all points lie within input grid
        (no extrapolation). If False, use full extent of input (may require filling).
    interpolation_method : str, default="bilinear"
        Interpolation method: "bilinear", "nearest", or "cubic".
    fill_value : float, default=np.nan
        Value for output points that fall outside input grid.
    ellipsoid : str, default="WGS84"
        Geodetic reference ellipsoid for coordinate conversions.

    Examples
    --------
    Resolution-based (recommended for missions):

    >>> config = RegridConfig(output_resolution_deg=(0.0009, 0.0009))  # ~100m

    Size-based (fixed dimensions):

    >>> config = RegridConfig(output_grid_size=(500, 500))

    Bounds + resolution (advanced):

    >>> config = RegridConfig(
    ...     output_bounds=(-116.0, -115.0, 38.0, 39.0),
    ...     output_resolution_deg=(0.001, 0.001)
    ... )
    """

    output_grid_size: tuple[int, int] | None = None
    output_resolution_deg: tuple[float, float] | None = None
    output_bounds: tuple[float, float, float, float] | None = None
    conservative_bounds: bool = True
    interpolation_method: str = "bilinear"
    fill_value: float = float("nan")
    ellipsoid: str = "WGS84"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Count how many grid specification methods are provided
        has_size = self.output_grid_size is not None
        has_resolution = self.output_resolution_deg is not None
        has_bounds = self.output_bounds is not None

        # Validate mutual exclusivity
        if has_bounds and not has_resolution:
            raise ValueError("output_bounds requires output_resolution_deg to be specified")

        if has_bounds and has_size:
            raise ValueError(
                "Cannot specify both output_bounds and output_grid_size. "
                "Use output_bounds + output_resolution_deg instead."
            )

        if has_size and has_resolution:
            raise ValueError("Cannot specify both output_grid_size and output_resolution_deg. Choose one method.")

        # Validate interpolation method
        valid_methods = {"bilinear", "nearest", "cubic"}
        if self.interpolation_method not in valid_methods:
            raise ValueError(f"interpolation_method must be one of {valid_methods}, got '{self.interpolation_method}'")

        # Validate grid size if provided
        if has_size:
            nrows, ncols = self.output_grid_size
            if nrows < 2 or ncols < 2:
                raise ValueError(
                    f"output_grid_size must have at least 2 rows and 2 columns, got {self.output_grid_size}"
                )

        # Validate resolution if provided
        if has_resolution:
            dlat, dlon = self.output_resolution_deg
            if dlat <= 0 or dlon <= 0:
                raise ValueError(f"output_resolution_deg must be positive, got {self.output_resolution_deg}")

        # Validate bounds if provided
        if has_bounds:
            minlon, maxlon, minlat, maxlat = self.output_bounds
            if minlon >= maxlon:
                raise ValueError(f"minlon must be < maxlon, got {minlon} >= {maxlon}")
            if minlat >= maxlat:
                raise ValueError(f"minlat must be < maxlat, got {minlat} >= {maxlat}")
