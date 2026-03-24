from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, field_validator, model_validator


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
class PSFSamplingConfig:
    """Configuration parameters for PSF sampling during image matching.

    Parameters
    ----------
    gcp_step_m : float, optional
        Ground control point step size in meters. Default is 30.0.
    motion_convolution_step_m : float, optional
        Step size for spacecraft motion convolution in meters.
        Default is ``gcp_step_m / 20.0``.
    psf_lat_sample_dist_deg : float, optional
        PSF sample distance in the latitude direction in degrees.
        Default is 2.4397105613972e-05.
    psf_lon_sample_dist_deg : float, optional
        PSF sample distance in the longitude direction in degrees.
        Default is 2.8737038710207e-05.
    """

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


class RegridConfig(BaseModel):
    """Configuration for GCP chip regridding.

    Specifies output grid parameters for transforming irregular geodetic grids
    to regular latitude/longitude grids.

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
        Interpolation method; one of ``"bilinear"``, ``"nearest"``,
        or ``"cubic"``.
    fill_value : float, default=NaN
        Value assigned to output points that fall outside the input grid.
    ellipsoid : str, default="WGS84"
        Reference ellipsoid used for ECEF ↔ geodetic conversions.
    """

    output_grid_size: tuple[int, int] | None = None
    output_resolution_deg: tuple[float, float] | None = None
    output_bounds: tuple[float, float, float, float] | None = None
    conservative_bounds: bool = True
    interpolation_method: str = "bilinear"
    fill_value: float = float("nan")
    ellipsoid: str = "WGS84"

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
    def validate_grid_spec(self) -> RegridConfig:
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
