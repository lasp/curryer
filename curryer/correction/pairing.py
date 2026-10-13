"""Utilities for pairing L1A images with nearby GCP chips.

Each L1A footprint is the polygon traced around its geolocated pixels
(:func:`curryer.compute.footprint.footprint_polygon`), so it follows curved
swaths. Its edge lies half a pixel plus the footprint tolerance outside the edge
pixel centres (one pixel at the default tolerance), moved by up to the tolerance
either way by simplification. A GCP chip pairs with an L1A
image by the signed great-circle distance from the chip's center pixel to that
polygon's edge (:func:`curryer.compute.footprint.footprint_margin_m`). The core
entry point is :func:`find_l1a_gcp_pairs`, which returns a many-to-many mapping
between the supplied L1A and GCP collections.

File-based utilities (discover_gcp_files, pair_files) provide higher-level
wrappers for working with ``.mat`` or NetCDF ``.nc`` files on disk.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from ..compute.footprint import FootprintGeometryError, convex_footprint, footprint_margin_m, swath_footprint
from ..compute.spatial import geodetic_to_ecef
from .grid_types import ImageGrid, NamedImageGrid

logger = logging.getLogger(__name__)


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
class L1AMetadata(ImageMetadata):
    """Metadata describing an L1A image footprint.

    Extends :class:`ImageMetadata` with the traced footprint polygon that
    pairing tests GCP centers against: (longitude, latitude) degrees, shape
    (N + 1, 2), closed, as returned by
    :func:`curryer.compute.footprint.footprint_polygon`. ``footprint`` is the
    image's extent; ``corners``, ``center`` and ``bbox`` are raw pixel values,
    NaN where those pixels are fill, and ``bbox`` does not account for the
    antimeridian.
    """

    footprint: np.ndarray


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

    l1a_images: list[L1AMetadata]
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


def _build_l1a_metadata(index: int, image: NamedImageGrid, footprint_tolerance_km: float | None) -> L1AMetadata:
    """Construct :class:`L1AMetadata` for ``image``, tracing its footprint.

    The footprint is :func:`~curryer.compute.footprint.swath_footprint`, or
    :func:`~curryer.compute.footprint.convex_footprint` with a warning naming the
    image where the edge cannot be traced (as
    :func:`~curryer.compute.footprint.footprint_polygon` does). It is built before
    the corner metadata, so geolocation that is not 2-D raises ``ValueError``.
    """
    name = image.name if image.name is not None else f"image_{index}"
    try:
        try:
            footprint = swath_footprint(image.lat, image.lon, tolerance_km=footprint_tolerance_km)
        except FootprintGeometryError as e:
            logger.warning(
                "L1A image %r: could not trace the swath edge (%s); using the buffered convex hull.", name, e
            )
            footprint = convex_footprint(image.lat, image.lon, tolerance_km=footprint_tolerance_km)
    except FootprintGeometryError:
        raise
    except ValueError as e:
        raise ValueError(f"Footprint of L1A image {name!r}: {e}") from e
    base = _build_image_metadata(index, image)
    return L1AMetadata(
        index=base.index,
        name=base.name,
        corners=base.corners,
        center=base.center,
        bbox=base.bbox,
        footprint=footprint,
    )


def _build_gcp_metadata(index: int, image: NamedImageGrid) -> GCPMetadata:
    """Construct :class:`GCPMetadata` for ``image``; raises ``ValueError`` naming it if its center is invalid."""

    base = _build_image_metadata(index, image)
    lat_c, lon_c, h_c = _image_center(image)
    if not (np.isfinite(lat_c) and np.isfinite(lon_c) and abs(lat_c) <= 90 and abs(lon_c) <= 180):
        raise ValueError(
            f"GCP chip {base.name!r}: center pixel latitude {lat_c} and longitude {lon_c} must be finite, "
            "within [-90, 90] and [-180, 180]"
        )
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
    footprint_tolerance_km: float | None = None,
) -> PairingResult:
    """Find all L1A/GCP pairs whose signed footprint margin meets a threshold.

    Parameters
    ----------
    l1a_images
        Iterable of :class:`NamedImageGrid` instances representing L1A imagery.
        Latitude and longitude are 2-D (rows, columns) in degrees, NaN for fill.
        Each footprint is the polygon traced around the pixels, its edge half a
        pixel plus ``footprint_tolerance_km`` outside the edge pixel centres
        (one pixel at the default), moved by up to ``footprint_tolerance_km``
        either way by simplification. Where the edge cannot be traced (folded
        or overlapping grids), it is the buffered convex hull of the pixels,
        logged as a warning naming the image; the hull reaches past a concave
        swath, so it can pair chips just outside the pixels.
    gcp_images
        Iterable of :class:`NamedImageGrid` instances representing GCP chips.
        The center pixel's latitude and longitude are the chip's position.
    max_distance_m
        Minimum signed margin (meters, finite) of the GCP center from the
        L1A footprint edge, positive inside.  Only pairs with
        ``margin >= max_distance_m`` are returned: ``0`` pairs centers inside
        or on the edge, a positive value requires them at least that far
        inside, a negative value allows them up to ``abs(max_distance_m)``
        outside.
    footprint_tolerance_km
        Tolerance of each L1A footprint polygon, km: the edge's offset beyond
        the half-pixel line and how far simplification may move it, so a
        larger value also widens the footprint. ``None`` uses half the image's
        smaller median pixel spacing
        (:func:`curryer.compute.footprint.footprint_polygon`).

    Returns
    -------
    PairingResult
        Metadata for the supplied images together with the pairs whose signed
        margin is at least ``max_distance_m``.

    Raises
    ------
    ValueError
        Naming the image, if an L1A footprint cannot be built: geolocation that
        is not 2-D floating, holds infinite or out-of-range values (a fill value
        other than NaN, such as -9999, is out of range), has no valid pixels or
        no two adjacent valid pixels, extends more than 60 degrees from its
        center, or encloses a pole; a grid axis with no two adjacent valid
        pixels (a single row, say), along which the pixel size is unknown; an
        invalid ``footprint_tolerance_km``; or, with
        ``footprint_tolerance_km=None``, a zero median pixel spacing (as with
        repeated frames). If ``max_distance_m`` is not a finite number. Naming the chip, if a GCP chip's center pixel
        has a non-finite or out-of-range latitude or longitude.
    FootprintGeometryError
        Only if a convex-hull footprint fails its own verification, which
        indicates a defect in :mod:`curryer.compute.footprint`.
    """
    if (
        isinstance(max_distance_m, bool)
        or not isinstance(max_distance_m, (int, float))
        or not np.isfinite(max_distance_m)
    ):
        raise ValueError(f"max_distance_m must be a finite number, got {max_distance_m!r}")
    l1a_meta: list[L1AMetadata] = []
    gcp_meta: list[GCPMetadata] = []
    for idx, image in enumerate(l1a_images):
        l1a_meta.append(_build_l1a_metadata(idx, image, footprint_tolerance_km))
    for idx, image in enumerate(gcp_images):
        gcp_meta.append(_build_gcp_metadata(idx, image))

    matches: list[PairMatch] = []
    if not gcp_meta:
        return PairingResult(l1a_meta, gcp_meta, matches)
    gcp_lat = np.array([gcp.center[0] for gcp in gcp_meta], dtype=float)
    gcp_lon = np.array([gcp.center[1] for gcp in gcp_meta], dtype=float)
    for l1a in l1a_meta:
        margins = footprint_margin_m(l1a.footprint, gcp_lat, gcp_lon)
        for gcp, margin in zip(gcp_meta, margins):
            if margin >= max_distance_m:
                logger.info("Paired [%s] to [%s] within [%s m]", l1a.name, gcp.name, margin)
                matches.append(PairMatch(l1a.index, gcp.index, float(margin)))
            else:
                logger.debug("No-pair [%s] to [%s] because [%s m] < [%s m]", l1a.name, gcp.name, margin, max_distance_m)

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
        >>> gcp_files = discover_gcp_files(Path("gcp_chips"))
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
    footprint_tolerance_km: float | None = None,
) -> list[tuple[Path, Path]]:
    """
    Find L1A-GCP pairs based on spatial overlap and return as file path tuples.

    Both ``.mat`` and NetCDF (``.nc``) files are supported for L1A observations
    and GCP chips.  The format is detected automatically from the file extension
    by :func:`~curryer.correction.image_io.load_named_image_grid`.

    Args:
        l1a_files: List of L1A (or observation) file paths to pair.
            Supports ``.mat`` and ``.nc`` files.
        gcp_directory: Directory containing GCP reference files.
        max_distance_m: Minimum signed margin of the GCP center from the
            footprint edge, meters, positive inside (default: 0.0)
            - 0.0: GCP center inside the L1A footprint or on its edge
            - >0: GCP center at least this far inside the footprint
            - <0: GCP center may lie up to abs(max_distance_m) outside
        l1a_key: MATLAB struct key for L1A data (default: "subimage").
            Ignored for NetCDF files.
        gcp_key: MATLAB struct key for GCP data (default: "GCP").
            Ignored for NetCDF files.
        gcp_pattern: File pattern for GCP discovery
            (default: "*_resampled.mat"; use "*_regridded.nc" for NetCDF chips).
        footprint_tolerance_km: Tolerance of each L1A footprint polygon, km
            (its offset and simplification; see :func:`find_l1a_gcp_pairs`);
            None (default) uses half the image's smaller median pixel spacing.

    Returns:
        List of (l1a_file, gcp_file) tuples for all valid spatial pairs.
        One L1A can pair with multiple GCPs (many-to-many).

    Raises:
        FileNotFoundError: If gcp_directory doesn't exist.
        ValueError: If l1a_files is empty, no GCP files match gcp_pattern, or
            any L1A or GCP file fails to load, including a missing file (the
            message names the file; the original exception is chained as
            ``__cause__``), or an L1A footprint cannot be built (see
            :func:`find_l1a_gcp_pairs`). An empty pair list is returned, not
            raised, when the files load but nothing overlaps.

    Example:
        >>> # .mat subimages paired with .mat GCPs (original behaviour)
        >>> pairs = pair_files(mat_files, Path("gcp_chips"))

        >>> # NetCDF observations paired with regridded NetCDF GCP chips
        >>> pairs = pair_files(
        ...     nc_obs_files, Path("gcp_chips"),
        ...     gcp_pattern="*_regridded.nc",
        ... )
    """
    from .image_io import load_named_image_grid

    gcp_dir = Path(gcp_directory)
    if not gcp_dir.is_dir():
        raise FileNotFoundError(f"GCP directory not found: {gcp_dir}")

    if not l1a_files:
        raise ValueError("No L1A files provided")

    logger.info(f"GCP Pairing: Loading {len(l1a_files)} L1A images...")

    l1a_images: list[NamedImageGrid] = []
    for l1a_file in l1a_files:
        mat_key = l1a_key if Path(l1a_file).suffix.lower() == ".mat" else "subimage"
        try:
            l1a_images.append(load_named_image_grid(l1a_file, mat_key=mat_key))
        except Exception as e:
            raise ValueError(f"Failed to load L1A file {l1a_file}: {e}") from e

    gcp_files = discover_gcp_files(gcp_dir, pattern=gcp_pattern)
    if not gcp_files:
        raise ValueError(f"No GCP files found in {gcp_dir} with pattern '{gcp_pattern}'")

    logger.info(f"GCP Pairing: Found {len(gcp_files)} GCP files")

    gcp_images: list[NamedImageGrid] = []
    for gcp_file in gcp_files:
        mat_key = gcp_key if Path(gcp_file).suffix.lower() == ".mat" else "GCP"
        try:
            gcp_images.append(load_named_image_grid(gcp_file, mat_key=mat_key))
        except Exception as e:
            raise ValueError(f"Failed to load GCP file {gcp_file}: {e}") from e

    logger.info(f"GCP Pairing: Finding spatial overlaps (max_distance={max_distance_m}m)...")
    result = find_l1a_gcp_pairs(l1a_images, gcp_images, max_distance_m, footprint_tolerance_km)

    if not result.matches:
        logger.warning(
            f"No valid pairs found with max_distance={max_distance_m}m. "
            f"Try increasing max_distance_m or check spatial coverage."
        )
        return []

    logger.info(f"GCP Pairing: Found {len(result.matches)} valid pairs")

    pairs: list[tuple[Path, Path]] = []
    for match in result.matches:
        l1a_file = l1a_files[match.l1a_index]
        gcp_file = gcp_files[match.gcp_index]
        pairs.append((l1a_file, gcp_file))
        logger.debug(f"  Paired: {l1a_file.name} → {gcp_file.name} (margin={match.distance_m:.1f}m)")

    return pairs


def pair_geolocated_dataset_with_gcp_files(
    geolocated_data: xr.Dataset,
    gcp_files: list[Path],
    max_distance_m: float = 0.0,
    gcp_key: str = "GCP",
    footprint_tolerance_km: float | None = None,
) -> list[Path]:
    """Find GCP chip files that spatially overlap an in-memory geolocated dataset.

    This is the in-memory-observation counterpart of :func:`pair_files`. Both
    delegate to the same core spatial-pairing algorithm
    (:func:`find_l1a_gcp_pairs`): the GCP center's signed distance to the
    observation's traced footprint polygon.

    Parameters
    ----------
    geolocated_data : xr.Dataset
        Already-geolocated observation dataset with ``latitude`` and
        ``longitude`` variables.
    gcp_files : list[Path]
        GCP chip file paths to test for spatial overlap.
    max_distance_m : float, optional
        Minimum signed margin (meters) required between the GCP chip center
        and the nearest observation footprint edge.  ``0.0`` (default) means
        the GCP center must fall inside the footprint or on its edge;
        negative values allow the center to lie *outside* the footprint by up
        to ``abs(max_distance_m)`` meters.
    gcp_key : str, optional
        MATLAB struct key used when loading ``.mat`` GCP chips.  Defaults to
        ``"GCP"``.  Ignored for NetCDF files.
    footprint_tolerance_km : float or None, optional
        Tolerance of the observation's footprint polygon, km (its offset and
        simplification; see :func:`find_l1a_gcp_pairs`); ``None`` (default)
        uses half its smaller median pixel spacing.

    Returns
    -------
    list[Path]
        Subset of *gcp_files* whose center-points overlap the observation
        footprint, ordered by their original position in *gcp_files*.
        Empty when no overlap is found or when *gcp_files* is empty.

    Raises
    ------
    ValueError
        If any GCP file fails to load (the message names the file; the
        original exception is chained as ``__cause__``), or the observation's
        footprint cannot be built (see :func:`find_l1a_gcp_pairs`).

    Examples
    --------
    >>> gcp_dir = Path("data/gcp_chips")
    >>> all_chips = sorted(gcp_dir.glob("*_regridded.nc"))
    >>> matched = pair_geolocated_dataset_with_gcp_files(geo_ds, all_chips, max_distance_m=-500.0)
    >>> print(f"{len(matched)}/{len(all_chips)} chips overlap the observation footprint")
    """

    from .image_io import geolocated_to_image_grid, load_named_image_grid  # noqa: PLC0415

    if not gcp_files:
        return []

    # Convert the in-memory dataset to a NamedImageGrid so it can be passed
    # directly to find_l1a_gcp_pairs.
    obs_grid = geolocated_to_image_grid(geolocated_data)
    obs_named = NamedImageGrid(
        data=obs_grid.data,
        lat=obs_grid.lat,
        lon=obs_grid.lon,
        h=obs_grid.h,
        name="observation",
    )

    gcp_named: list[NamedImageGrid] = []
    for gcp_file in gcp_files:
        mat_key = gcp_key if Path(gcp_file).suffix.lower() == ".mat" else "GCP"
        try:
            gcp_named.append(load_named_image_grid(gcp_file, mat_key=mat_key))
        except Exception as e:
            raise ValueError(f"Failed to load GCP file {gcp_file}: {e}") from e

    pairing_result = find_l1a_gcp_pairs([obs_named], gcp_named, max_distance_m, footprint_tolerance_km)
    matched_indices = {m.gcp_index for m in pairing_result.matches}
    return [Path(gcp_files[i]) for i in sorted(matched_indices)]
