"""Utilities for pairing L1A images with nearby GCP chips.

The routines in this module describe each image footprint using the
``NamedImageGrid`` metadata, convert the corners to a local East-North-Up frame,
and compute the distance between a GCP center point and the nearest edge of
each L1A footprint.  The core entry point is :func:`find_l1a_gcp_pairs`, which
returns a many-to-many mapping between the supplied L1A and GCP collections.

File-based utilities (discover_gcp_files, pair_files) provide higher-level
wrappers for working with MATLAB .mat files on disk.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..compute.spatial import geodetic_to_ecef
from .data_structures import ImageGrid, NamedImageGrid

logger = logging.getLogger(__name__)


# ============================================================================
# GCP Pairing Interface Protocol
# ============================================================================


class GCPPairingFunc(Protocol):
    """
    Protocol for GCP pairing functions in Monte Carlo pipeline.

    Pairing functions determine which science observations (L1A images)
    overlap with which ground control points (GCP reference images).

    Standard Signature:
        def pair_gcps(science_keys: List[str]) -> List[Tuple[str, str]]

    Returns:
        List of (science_key, gcp_reference_path) tuples, one per valid pair

    Note:
        This is a simplified interface for Monte Carlo compatibility.
        Real implementations (like find_l1a_gcp_pairs below) may use more
        sophisticated spatial algorithms internally, but must return results
        in this simple tuple format.

    Examples:
        # Real spatial pairing
        def spatial_gcp_pairing(science_keys):
            l1a_images = load_images(science_keys)
            gcp_images = discover_gcps()
            pairs = find_spatial_overlaps(l1a_images, gcp_images)
            return [(l1a.name, gcp.path) for l1a, gcp in pairs]

        # Test/synthetic pairing
        def synthetic_gcp_pairing(science_keys):
            return [(key, f"synthetic_gcp_{i}.tif")
                    for i, key in enumerate(science_keys)]
    """

    def __call__(self, science_keys: list[str]) -> list[tuple[str, str]]:
        """Find GCP pairs for given science observations."""
        ...


def validate_pairing_output(pairs: list[tuple[str, str]]) -> None:
    """
    Validate that GCP pairing output conforms to expected format.

    Args:
        pairs: List of (science_key, gcp_path) tuples

    Raises:
        TypeError: If structure is invalid
        ValueError: If tuple elements have wrong types

    Example:
        >>> pairs = gcp_pairing_func(["sci_001", "sci_002"])
        >>> validate_pairing_output(pairs)
    """
    if not isinstance(pairs, list):
        raise TypeError(f"GCP pairing must return list, got {type(pairs)}")

    for i, pair in enumerate(pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                f"GCP pairing output[{i}] must be (str, str) tuple, "
                f"got {type(pair)} with length {len(pair) if isinstance(pair, tuple) else 'N/A'}"
            )
        sci_key, gcp_path = pair
        if not isinstance(sci_key, str) or not isinstance(gcp_path, str | Path):
            raise ValueError(
                f"GCP pairing output[{i}] = ({type(sci_key).__name__}, {type(gcp_path).__name__}), expected (str, str)"
            )


# ============================================================================
# Spatial Pairing Implementation
# ============================================================================


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
    corners: list[tuple[float, float]]
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

    l1a_images: list[ImageMetadata]
    gcp_images: list[GCPMetadata]
    matches: list[PairMatch]


def _image_corners(image: ImageGrid) -> list[tuple[float, float]]:
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

    l1a_meta: list[ImageMetadata] = []
    gcp_meta: list[GCPMetadata] = []

    for idx, image in enumerate(l1a_images):
        l1a_meta.append(_build_image_metadata(idx, image))

    for idx, image in enumerate(gcp_images):
        gcp_meta.append(_build_gcp_metadata(idx, image))

    matches: list[PairMatch] = []
    for l1a in l1a_meta:
        for gcp in gcp_meta:
            lat_c, lon_c = gcp.center
            margin = _distance_point_to_polygon_m(lat_c, lon_c, 0.0, l1a.corners)
            if margin >= max_distance_m:
                logger.info("Paired [%s] to [%s] within [%s m]", l1a.name, gcp.name, margin)
                matches.append(PairMatch(l1a.index, gcp.index, margin))
            else:
                logger.debug(
                    "No-pair [%s] to [%s] because [%s m] >= [%s m]", l1a.name, gcp.name, margin, max_distance_m
                )

    return PairingResult(l1a_meta, gcp_meta, matches)


# ============================================================================
# File-Based Pairing Utilities
# ============================================================================


def discover_gcp_files(gcp_directory: Path, pattern: str = "*_resampled.mat") -> list[Path]:
    """
    Find all GCP files in a directory matching a pattern.

    Args:
        gcp_directory: Directory to search
        pattern: Glob pattern for GCP files (default: "*_resampled.mat")

    Returns:
        Sorted list of Path objects for GCP files

    Example:
        >>> gcp_files = discover_gcp_files(Path("tests/data/clarreo/image_match"))
        >>> print(f"Found {len(gcp_files)} GCP files")
    """
    gcp_dir = Path(gcp_directory)
    if not gcp_dir.is_dir():
        logger.warning(f"GCP directory not found: {gcp_dir}")
        return []

    # Search recursively in subdirectories too
    gcp_files = sorted(gcp_dir.rglob(pattern))
    logger.info(f"Discovered {len(gcp_files)} GCP files in {gcp_dir} with pattern '{pattern}'")

    return gcp_files


def pair_files(
    l1a_files: list[Path],
    gcp_directory: Path,
    max_distance_m: float = 0.0,
    l1a_key: str = "subimage",
    gcp_key: str = "GCP",
    gcp_pattern: str = "*_resampled.mat",
) -> list[tuple[Path, Path]]:
    """
    Find L1A-GCP pairs based on spatial overlap and return as file path tuples.

    This is the production replacement for placeholder_gcp_pairing().
    Uses the find_l1a_gcp_pairs() algorithm for spatial matching.

    Args:
        l1a_files: List of L1A file paths to pair
        gcp_directory: Directory containing GCP reference files
        max_distance_m: Minimum margin for valid pairing (default: 0.0)
            - 0.0: Requires GCP center inside L1A footprint (strict)
            - >0: Allows GCP center up to this distance inside footprint
            - <0: Allows GCP center outside footprint (loose)
        l1a_key: MATLAB struct key for L1A data (default: "subimage")
        gcp_key: MATLAB struct key for GCP data (default: "GCP")
        gcp_pattern: File pattern for GCP discovery (default: "*_resampled.mat")

    Returns:
        List of (l1a_file, gcp_file) tuples for all valid spatial pairs
        Note: One L1A can pair with multiple GCPs (many-to-many)

    Raises:
        FileNotFoundError: If gcp_directory doesn't exist
        ValueError: If no valid pairs found

    Example:
        >>> l1a_files = [Path("test1.mat"), Path("test2.mat")]
        >>> pairs = pair_files(l1a_files, Path("gcp_chips"), max_distance_m=0.0)
        >>> print(f"Found {len(pairs)} valid L1A-GCP pairs")
        >>> for l1a, gcp in pairs:
        ...     print(f"  {l1a.name} → {gcp.name}")
    """
    from .image_match import load_image_grid_from_mat

    gcp_dir = Path(gcp_directory)
    if not gcp_dir.is_dir():
        raise FileNotFoundError(f"GCP directory not found: {gcp_dir}")

    logger.info(f"GCP Pairing: Loading {len(l1a_files)} L1A images...")

    # Load L1A images as NamedImageGrid
    l1a_images = []
    for l1a_file in l1a_files:
        try:
            img = load_image_grid_from_mat(l1a_file, key=l1a_key, name=str(l1a_file), as_named=True)
            l1a_images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load L1A file {l1a_file}: {e}")

    if not l1a_images:
        raise ValueError("No L1A images loaded successfully")

    # Discover and load GCP images
    gcp_files = discover_gcp_files(gcp_dir, pattern=gcp_pattern)
    if not gcp_files:
        raise ValueError(f"No GCP files found in {gcp_dir} with pattern '{gcp_pattern}'")

    logger.info(f"GCP Pairing: Found {len(gcp_files)} GCP files")

    gcp_images = []
    for gcp_file in gcp_files:
        try:
            img = load_image_grid_from_mat(gcp_file, key=gcp_key, name=str(gcp_file), as_named=True)
            gcp_images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load GCP file {gcp_file}: {e}")

    if not gcp_images:
        raise ValueError("No GCP images loaded successfully")

    # Run spatial pairing algorithm
    logger.info(f"GCP Pairing: Finding spatial overlaps (max_distance={max_distance_m}m)...")
    result = find_l1a_gcp_pairs(l1a_images, gcp_images, max_distance_m)

    if not result.matches:
        logger.warning(
            f"No valid pairs found with max_distance={max_distance_m}m. "
            f"Try increasing max_distance_m or check spatial coverage."
        )
        return []

    logger.info(f"GCP Pairing: Found {len(result.matches)} valid pairs")

    # Convert PairingResult to file path tuples
    pairs = []
    for match in result.matches:
        l1a_file = l1a_files[match.l1a_index]
        gcp_file = gcp_files[match.gcp_index]
        pairs.append((l1a_file, gcp_file))
        logger.debug(f"  Paired: {l1a_file.name} → {gcp_file.name} (margin={match.distance_m:.1f}m)")

    return pairs
