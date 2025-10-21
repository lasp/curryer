"""Utilities for pairing L1A images with nearby GCP chips.

The routines in this module describe each image footprint using the
``NamedImageGrid`` metadata, convert the corners to a local East-North-Up frame,
and compute the distance between a GCP center point and the nearest edge of
each L1A footprint.  The core entry point is :func:`find_l1a_gcp_pairs`, which
returns a many-to-many mapping between the supplied L1A and GCP collections.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .data_structures import ImageGrid, NamedImageGrid
from ..compute.spatial import geodetic_to_ecef


logger = logging.getLogger(__name__)


def enu_rotation_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the rotation matrix from ECEF deltas to local ENU
    (East–North–Up, local tangent-coordinate frame).

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude of the origin in degrees.
    lon_deg : float
        Geodetic longitude of the origin in degrees.

    Returns
    -------
    ndarray, shape (3, 3)
        Matrix that converts an ECEF delta vector into east, north, and up
        components with respect to the specified origin.
    """

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )


def geodetic_to_enu(
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        h_m: np.ndarray,
        origin_lat_deg: float,
        origin_lon_deg: float,
        origin_h_m: float = 0.0,
) -> np.ndarray:
    """Convert geodetic coordinates to local ENU (East–North–Up) coordinates.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Geodetic latitude and longitude (degrees) of the points to convert.
    h_m : array_like
        Heights above the WGS-84 ellipsoid (meters) for each point.
    origin_lat_deg, origin_lon_deg : float
        Geodetic latitude/longitude of the ENU frame origin in degrees.
    origin_h_m : float, optional
        Height of the origin point in meters. Defaults to ``0``.

    Returns
    -------
    ndarray, shape (..., 3)
        East, north, and up coordinates (meters) of the input points relative
        to the specified origin.
    """

    origin_ecef = geodetic_to_ecef(np.array([origin_lon_deg, origin_lat_deg, origin_h_m]), degrees=True, meters=True)
    points_ecef = geodetic_to_ecef(np.vstack([lon_deg, lat_deg, h_m]).T, degrees=True, meters=True)
    deltas = points_ecef - origin_ecef
    rot = enu_rotation_matrix(origin_lat_deg, origin_lon_deg)
    flat_deltas = deltas.reshape(-1, 3).T
    enu = rot @ flat_deltas
    return enu.T.reshape(points_ecef.shape)


@dataclass
class ImageMetadata:
    """Metadata describing an image footprint.

    Parameters
    ----------
    index
        Position of the image inside the original input list.
    name
        Identifier associated with the image (e.g., filename).
    corners
        Four corner latitude/longitude tuples ordered clockwise.
    center
        Latitude/longitude of the image center pixel.
    bbox
        Bounding box expressed as ``(lat_min, lat_max, lon_min, lon_max)``.
    """

    index: int
    name: str
    corners: List[tuple[float, float]]
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]

    def corner_array(self) -> np.ndarray:
        """Return the corner coordinates as a ``(4, 2)`` NumPy array."""

        return np.asarray(self.corners, dtype=float)


@dataclass
class GCPMetadata(ImageMetadata):
    """Metadata describing a GCP image footprint.

    Extends :class:`ImageMetadata` with the ECEF coordinates of the GCP
    center point to simplify subsequent distance calculations.
    """

    center_point_ecef: np.ndarray


@dataclass
class PairMatch:
    """Relationship between an L1A image and a GCP chip.

    The ``distance_m`` field stores the signed margin between the GCP
    center and the closest edge of the L1A footprint in meters.  Positive
    values indicate the center lies inside the footprint, while negative
    values mean it lies outside.
    """

    l1a_index: int
    gcp_index: int
    distance_m: float


@dataclass
class PairingResult:
    """Container for the output of :func:`find_l1a_gcp_pairs`."""

    l1a_images: List[ImageMetadata]
    gcp_images: List[GCPMetadata]
    matches: List[PairMatch]


def _image_corners(image: ImageGrid) -> List[tuple[float, float]]:
    """Return the four corner latitude/longitude pairs of ``image``."""

    lat = image.lat
    lon = image.lon
    return [
        (float(lat[0, 0]), float(lon[0, 0])),
        (float(lat[0, -1]), float(lon[0, -1])),
        (float(lat[-1, -1]), float(lon[-1, -1])),
        (float(lat[-1, 0]), float(lon[-1, 0])),
    ]


def _image_center(image: ImageGrid) -> tuple[float, float, float]:
    """Return the latitude, longitude, and height of the center pixel."""

    mid_i, mid_j = image.mid_indices
    lat = float(image.lat[mid_i, mid_j])
    lon = float(image.lon[mid_i, mid_j])
    if image.h is not None:
        h = float(image.h[mid_i, mid_j])
    else:
        h = 0.0
    return lat, lon, h


def _image_bbox(image: ImageGrid) -> tuple[float, float, float, float]:
    """Return the latitude/longitude bounding box of ``image``."""

    lat_min = float(np.min(image.lat))
    lat_max = float(np.max(image.lat))
    lon_min = float(np.min(image.lon))
    lon_max = float(np.max(image.lon))
    return lat_min, lat_max, lon_min, lon_max


def _point_in_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    """Return ``True`` if ``point_xy`` lies inside ``polygon_xy``.

    Uses the winding-number algorithm with a ray cast along the positive
    *x*-axis.
    """

    winding = 0
    x, y = point_xy
    for i in range(len(polygon_xy)):
        x1, y1 = polygon_xy[i]
        x2, y2 = polygon_xy[(i + 1) % len(polygon_xy)]
        if y1 <= y:
            if y2 > y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0:
                winding += 1
        else:
            if y2 <= y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0:
                winding -= 1
    return winding != 0


def _distance_point_to_segment(point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    """Return the minimum distance from ``point_xy`` to segment ``ab``."""

    ap = point_xy - a_xy
    ab = b_xy - a_xy
    ab_norm_sq = float(np.dot(ab, ab))
    if ab_norm_sq == 0.0:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab)) / ab_norm_sq
    t_clamped = max(0.0, min(1.0, t))
    closest = a_xy + t_clamped * ab
    return float(np.linalg.norm(point_xy - closest))


def _distance_point_to_polygon_m(
        point_lat: float,
        point_lon: float,
        point_h: float,
        polygon_latlon: Sequence[tuple[float, float]],
) -> float:
    """Return the distance from a point to a polygon in meters.

    Parameters
    ----------
    point_lat, point_lon, point_h
        Geodetic coordinates of the query location.
    polygon_latlon
        Sequence of latitude/longitude tuples describing polygon corners.

    Returns
    -------
    float
        Signed distance between the point and the polygon boundary.  Positive
        values indicate the point is inside the polygon and represent the
        margin to the nearest edge.  Negative values indicate the point lies
        outside the polygon and represent the distance to the closest edge.
    """

    poly = np.asarray(polygon_latlon, dtype=float)
    origin_lat = float(np.mean(poly[:, 0]))
    origin_lon = float(np.mean(poly[:, 1]))
    origin_h = 0.0

    polygon_enu = geodetic_to_enu(
        poly[:, 0],
        poly[:, 1],
        np.zeros(len(poly)),
        origin_lat,
        origin_lon,
        origin_h,
    )[:, :2]

    point_enu = geodetic_to_enu(
        np.array([point_lat]),
        np.array([point_lon]),
        np.array([point_h]),
        origin_lat,
        origin_lon,
        origin_h,
    )[0, :2]

    inside = _point_in_polygon(point_enu, polygon_enu)

    min_distance = float("inf")
    for i in range(len(polygon_enu)):
        a = polygon_enu[i]
        b = polygon_enu[(i + 1) % len(polygon_enu)]
        distance = _distance_point_to_segment(point_enu, a, b)
        if distance < min_distance:
            min_distance = distance

    if inside:
        return min_distance
    return -min_distance


def _build_image_metadata(index: int, image: NamedImageGrid) -> ImageMetadata:
    """Construct :class:`ImageMetadata` for ``image``."""

    corners = _image_corners(image)
    lat_c, lon_c, _ = _image_center(image)
    bbox = _image_bbox(image)
    name = image.name if image.name is not None else f"image_{index}"
    return ImageMetadata(
        index=index,
        name=name,
        corners=corners,
        center=(lat_c, lon_c),
        bbox=bbox,
    )


def _build_gcp_metadata(index: int, image: NamedImageGrid) -> GCPMetadata:
    """Construct :class:`GCPMetadata` for ``image``."""

    base = _build_image_metadata(index, image)
    lat_c, lon_c, h_c = _image_center(image)
    center_ecef = geodetic_to_ecef(np.array([lon_c, lat_c, h_c]), degrees=True, meters=True)
    return GCPMetadata(
        index=base.index,
        name=base.name,
        corners=base.corners,
        center=base.center,
        bbox=base.bbox,
        center_point_ecef=center_ecef,
    )


def find_l1a_gcp_pairs(
        l1a_images: Iterable[NamedImageGrid],
        gcp_images: Iterable[NamedImageGrid],
        max_distance_m: float,
) -> PairingResult:
    """Find all L1A/GCP pairs within a distance threshold.

    Parameters
    ----------
    l1a_images
        Iterable of :class:`NamedImageGrid` instances representing L1A imagery.
    gcp_images
        Iterable of :class:`NamedImageGrid` instances representing GCP chips.
    max_distance_m
        Minimum margin (meters) required between the GCP center and the
        nearest L1A edge.  Only pairs with ``margin >= max_distance_m`` are
        returned.

    Returns
    -------
    PairingResult
        Metadata for the supplied images together with any pairs that fall
        within ``max_distance_m``.
    """

    l1a_meta: List[ImageMetadata] = []
    gcp_meta: List[GCPMetadata] = []

    for idx, image in enumerate(l1a_images):
        l1a_meta.append(_build_image_metadata(idx, image))

    for idx, image in enumerate(gcp_images):
        gcp_meta.append(_build_gcp_metadata(idx, image))

    matches: List[PairMatch] = []
    for l1a in l1a_meta:
        for gcp in gcp_meta:
            lat_c, lon_c = gcp.center
            margin = _distance_point_to_polygon_m(lat_c, lon_c, 0.0, l1a.corners)
            if margin >= max_distance_m:
                logger.info('Paired [%s] to [%s] within [%s m]', l1a.name, gcp.name, margin)
                matches.append(PairMatch(l1a.index, gcp.index, margin))
            else:
                logger.debug('No-pair [%s] to [%s] because [%s m] >= [%s m]',
                             l1a.name, gcp.name, margin, max_distance_m)

    return PairingResult(l1a_meta, gcp_meta, matches)
