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
