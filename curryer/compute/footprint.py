"""Footprint polygons of 2-D geolocation grids.

Traces the outer edge of a grid of geolocated pixels (rows x columns, such as frames x
cross-track pixels) and simplifies it to a closed, counter-clockwise polygon whose edges are
great-circle arcs. :func:`footprint_margin_m` measures how far points lie inside or outside such
a polygon, for example to pair ground control points with the observations that see them.

Great-circle edges are how CMR interprets a UMM-G GPolygon in a collection whose spatial
representation is GEODETIC, so a footprint can be written to granule metadata as-is, including
when it crosses the antimeridian. The Earth is treated as a sphere of the IUGG mean radius,
matching that geodetic model.

@author: Matthew Maclay
"""

import logging

import numpy as np
from scipy.spatial import ConvexHull

from .constants import WGS84_SEMI_MAJOR_AXIS_KM, WGS84_SEMI_MINOR_AXIS_KM

logger = logging.getLogger(__name__)

# IUGG mean radius R1 = (2a + b) / 3, 6371.0088 km
EARTH_MEAN_RADIUS_KM = (2 * WGS84_SEMI_MAJOR_AXIS_KM + WGS84_SEMI_MINOR_AXIS_KM) / 3
MAX_ANGULAR_RADIUS_DEG = 60.0


class FootprintGeometryError(ValueError):
    """Valid geolocation whose swath edge cannot be traced into a simple polygon."""


def footprint_polygon(latitude: np.ndarray, longitude: np.ndarray, *, tolerance_km: float | None = None) -> np.ndarray:
    """Return the footprint of a swath: its traced edge, or its buffered convex hull.

    Calls :func:`swath_footprint`. Where the edge cannot be traced (:class:`FootprintGeometryError`:
    folded, overlapping or single-row grids), logs a warning and returns :func:`convex_footprint`
    instead, which contains every valid pixel center but overreaches the swath where it is concave.

    Parameters and returns are as for :func:`swath_footprint`.

    Raises
    ------
    ValueError
        For invalid geolocation or ``tolerance_km``, as for :func:`swath_footprint`. Also where the
        fallback raises it: no two adjacent valid pixels along either grid axis (such as a single
        valid pixel), which :func:`swath_footprint` reports as :class:`FootprintGeometryError`; or,
        with ``tolerance_km=None``, a zero median pixel spacing.
    FootprintGeometryError
        Only if the convex hull fails verification, which indicates a defect in this module.
    """
    try:
        return swath_footprint(latitude, longitude, tolerance_km=tolerance_km)
    except FootprintGeometryError as e:
        logger.warning("Could not trace the swath edge (%s); using the buffered convex hull.", e)
        return convex_footprint(latitude, longitude, tolerance_km=tolerance_km)


def swath_footprint(latitude: np.ndarray, longitude: np.ndarray, *, tolerance_km: float | None = None) -> np.ndarray:
    """Return a polygon enclosing a 2-D swath of geolocated pixels.

    Each outer pixel is pushed outward by half a pixel plus ``tolerance_km``, giving a ring just
    outside the pixel edges. The outward direction is perpendicular to the local direction of the
    pixel's side; the half-pixel length is half the step from its inner neighbor, measured along
    that direction. Each corner is placed where the offset lines of its two sides meet. Each side
    of the ring (first row, last column, last row, first column) is then simplified with
    Douglas-Peucker, using the distance to the great-circle arc between the kept vertices.

    Along the sides, the polygon edge lies between 0 and 2 * ``tolerance_km`` outside the
    half-pixel edge. At a corner where the grid axes meet at angle theta, the vertex can lie up to
    about ``tolerance_km / sin(theta / 2)`` beyond the pixel corner, which exceeds 2 *
    ``tolerance_km`` when theta is under 60 degrees.

    Fill is NaN. Rows and columns that are entirely fill are cropped from the ends of the grid.
    Outer pixels that are fill are dropped from the ring. Where an inner neighbor is fill, or at
    the same position as its outer pixel, the half-pixel length is taken from the nearest complete
    pair on the same side, or from the grid's median pixel spacing if the side has none.

    Parameters
    ----------
    latitude : np.ndarray
        Geodetic latitude of pixel centers, degrees in [-90, 90], shape (rows, columns), floating
        dtype, NaN for fill.
    longitude : np.ndarray
        Longitude of pixel centers, degrees in [-180, 180], same shape and fill as ``latitude``.
    tolerance_km : float or None
        Maximum distance, in km, that simplification may move the polygon edge; positive and finite.
        ``None`` uses half the smaller of the grid's median pixel spacings along its two axes, so the
        polygon follows the swath as closely at any pixel size.

    Returns
    -------
    np.ndarray
        Vertices as (longitude, latitude) in degrees, shape (N + 1, 2), float64, counter-clockwise
        seen from above, closed by repeating the first vertex. Longitudes are in [-180, 180).

    Raises
    ------
    ValueError
        If the geolocation is invalid: the inputs are not 2-D floating arrays of the same shape,
        contain infinite or out-of-range values, or have no valid pixels; the valid pixels extend
        more than ``MAX_ANGULAR_RADIUS_DEG`` from their center, have no well-defined center, or the
        footprint contains a pole. Also if ``tolerance_km`` is neither None nor a positive finite
        number, or is None and the median pixel spacing along a grid axis is zero (more than half of
        the adjacent pixels coincide, as with repeated frames); pass ``tolerance_km`` then.
    FootprintGeometryError
        If the geolocation is valid but its edge cannot be traced: fewer than 2 valid rows or
        columns; a side with fewer than 2 valid pixels, or whose valid pixels are all at one
        position; no two adjacent valid pixels along a grid axis; outer pixels enclosing no area;
        two sides meeting at a corner without an angle between them; or a resulting polygon with
        fewer than 3 distinct vertices, that crosses itself, or that does not contain every valid
        pixel center and every point of the half-pixel edge ring. ``convex_footprint`` handles
        these swaths.
    """
    lat, lon = _validate(latitude, longitude, tolerance_km)
    if lat.shape[0] < 2 or lat.shape[1] < 2:
        raise FootprintGeometryError("at least 2 rows and 2 columns must contain valid pixels")
    xyz = _to_unit_vectors(lat, lon)
    valid = ~np.isnan(lat)
    if tolerance_km is None:
        tol = _default_tolerance([_median_spacing(xyz, valid, 0), _median_spacing(xyz, valid, 1)])
    else:
        tol = tolerance_km / EARTH_MEAN_RADIUS_KM

    vertices = _join([_simplify(side, tol) for side in _edge_ring_sides(xyz, valid, tol) if len(side)])
    edge_ring = _join([side for side in _edge_ring_sides(xyz, valid, 0.0) if len(side)])

    vertices = _verify(vertices, xyz[valid], edge_ring)
    return _to_lon_lat(np.concatenate([vertices, vertices[:1]]))


def _validate(latitude: np.ndarray, longitude: np.ndarray, tolerance_km: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Check the input contract and return float64 arrays cropped to their valid rows and columns."""
    for name, arr in (("latitude", latitude), ("longitude", longitude)):
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"{name} must be a 2-D numpy array")
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"{name} must have a floating dtype, got {arr.dtype}")
    if latitude.shape != longitude.shape:
        raise ValueError(f"latitude shape {latitude.shape} does not match longitude shape {longitude.shape}")
    if tolerance_km is not None and (
        isinstance(tolerance_km, bool)
        or not isinstance(tolerance_km, (int, float))
        or not np.isfinite(tolerance_km)
        or tolerance_km <= 0
    ):
        raise ValueError(f"tolerance_km must be None or a positive finite number, got {tolerance_km!r}")

    lat = latitude.astype(np.float64)
    lon = longitude.astype(np.float64)
    if np.isinf(lat).any() or np.isinf(lon).any():
        raise ValueError("latitude and longitude must not contain infinite values")

    fill = np.isnan(lat) | np.isnan(lon)
    if (np.abs(lat[~fill]) > 90).any():
        raise ValueError("latitude values must be within [-90, 90]")
    if (np.abs(lon[~fill]) > 180).any():
        raise ValueError("longitude values must be within [-180, 180]")

    rows = np.flatnonzero(~fill.all(axis=1))
    cols = np.flatnonzero(~fill.all(axis=0))
    if rows.size == 0:
        raise ValueError("latitude and longitude contain no valid pixels")
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    lat, lon, fill = lat[crop], lon[crop], fill[crop]
    lat[fill] = np.nan
    lon[fill] = np.nan
    return lat, lon


def _default_tolerance(spacings: list[float]) -> float:
    """Half the smallest median pixel spacing, on the unit sphere; raises if it is zero."""
    tol = 0.5 * min(spacings)
    if tol <= 0:
        raise ValueError("the median pixel spacing along a grid axis is zero; pass tolerance_km")
    return tol


def _to_unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)], axis=-1)


def _to_lon_lat(xyz: np.ndarray) -> np.ndarray:
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lon = np.where(lon >= 180.0, lon - 360.0, lon)
    return np.stack([lon, lat], axis=-1)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _half_lengths(
    outer: np.ndarray, inner: np.ndarray, normals: np.ndarray, pair_valid: np.ndarray, fallback: float
) -> np.ndarray:
    """Half the reach of each outer pixel along its outward normal, on the unit sphere.

    The reach is the component of the step from the inner neighbor to the outer pixel along the
    normal, which is the pixel's extent beyond its center on a skewed grid. Where the pair has fill
    or zero reach, the value is taken from the nearest pair along the side that has neither; if the
    side has no such pair, ``fallback`` is used.
    """
    lengths = np.abs(np.sum((outer - inner) * normals, axis=1))
    idx = np.flatnonzero(pair_valid & (lengths > 0))
    if idx.size == 0:
        return np.full(len(outer), 0.5 * fallback)
    positions = np.arange(len(outer))
    after = np.clip(np.searchsorted(idx, positions), 0, idx.size - 1)
    before = np.clip(after - 1, 0, idx.size - 1)
    nearest = np.where(np.abs(idx[before] - positions) <= np.abs(idx[after] - positions), idx[before], idx[after])
    return 0.5 * lengths[nearest]


def _side_normals(outer: np.ndarray, outer_valid: np.ndarray, side: str) -> np.ndarray:
    """Unit vectors tangent to the sphere and perpendicular to the side, to the right of its direction.

    The side tangent at each valid pixel is the difference of its valid neighbors along the side
    (one-sided at the ends). Positions of fill pixels are NaN.
    """
    idx = np.flatnonzero(outer_valid)
    if idx.size < 2:
        raise FootprintGeometryError(f"the {side} of the grid has fewer than 2 valid pixels")
    pts = outer[idx]
    k = np.arange(len(pts))
    normal = np.zeros_like(pts)
    norm = np.zeros(len(pts))
    # Widen the difference until it spans distinct points, so repeated pixels do not zero the tangent.
    for width in range(1, len(pts)):
        todo = norm == 0
        if not todo.any():
            break
        tangent = pts[np.minimum(k + width, len(pts) - 1)] - pts[np.maximum(k - width, 0)]
        normal[todo] = np.cross(tangent[todo], pts[todo])
        norm[todo] = np.linalg.norm(normal[todo], axis=1)
    if (norm == 0).any():
        raise FootprintGeometryError(f"all valid pixels on the {side} of the grid are at the same position")
    normals = np.full_like(outer, np.nan)
    normals[idx] = normal / norm[:, None]
    return normals


def _median_spacing(xyz: np.ndarray, valid: np.ndarray, axis: int) -> float:
    """Median chord between adjacent valid pixels along ``axis``."""
    a = np.moveaxis(xyz, axis, 0)
    v = np.moveaxis(valid, axis, 0)
    pair = v[1:] & v[:-1]
    if not pair.any():
        raise FootprintGeometryError(f"no two adjacent valid pixels along grid axis {axis}")
    return float(np.median(np.linalg.norm(a[1:] - a[:-1], axis=-1)[pair]))


def _traversal_sign(xyz: np.ndarray, valid: np.ndarray) -> float:
    """+1 if the outer pixels, traced first row, last column, last row, first column, run counter-clockwise."""
    ring = np.concatenate(
        [xyz[0][valid[0]], xyz[:, -1][valid[:, -1]], xyz[-1, ::-1][valid[-1, ::-1]], xyz[::-1, 0][valid[::-1, 0]]]
    )
    center = _normalize(xyz[valid].mean(axis=0))
    area = _signed_area(_project(ring, center, *_gnomonic_basis(center)))
    # 1e-15 square radians is about 0.04 m^2 on the ground.
    if abs(area) < 1e-15:
        raise FootprintGeometryError("outer pixels of the grid enclose no area")
    return 1.0 if area > 0 else -1.0


def _edge_ring_sides(xyz: np.ndarray, valid: np.ndarray, tol: float) -> list[np.ndarray]:
    """Return the four sides of the pushed-out ring, each running corner to corner.

    Sides are the first row, last column, last row reversed, and first column reversed. Each side
    starts at the corner shared with the previous side. Every valid outer pixel is pushed along the
    side's outward normal by half its pixel size plus ``tol``. Each corner is placed where the offset
    lines of its two sides meet, which accounts for grids whose rows and columns are not at right
    angles. Fill pixels are dropped.
    """
    row_spacing = _median_spacing(xyz, valid, 0)
    col_spacing = _median_spacing(xyz, valid, 1)

    def side(outer, inner, outer_valid, inner_valid, forward, fallback, name):
        # Normals are computed in traversal order; top and right run forward, bottom and left backward.
        order = slice(None) if forward else slice(None, None, -1)
        normals = _side_normals(outer[order], outer_valid[order], name)[order]
        half = _half_lengths(outer, inner, normals, outer_valid & inner_valid, fallback)
        return normals, half

    top = side(xyz[0], xyz[1], valid[0], valid[1], True, row_spacing, "first row")
    right = side(xyz[:, -1], xyz[:, -2], valid[:, -1], valid[:, -2], True, col_spacing, "last column")
    bottom = side(xyz[-1], xyz[-2], valid[-1], valid[-2], False, row_spacing, "last row")
    left = side(xyz[:, 0], xyz[:, 1], valid[:, 0], valid[:, 1], False, col_spacing, "first column")
    # With the swath on the left of the traversal, the right-hand normals point outward.
    sign = _traversal_sign(xyz, valid)
    top, right, bottom, left = ((sign * n, h) for n, h in (top, right, bottom, left))

    def offset(n_half):
        normals, half = n_half
        return normals * (half + tol)[:, None]

    def corner(p, a, i, b, j):
        """Point where the offset lines of sides ``a`` (at ``i``) and ``b`` (at ``j``) meet."""
        na, nb = a[0][i], b[0][j]
        da, db = a[1][i] + tol, b[1][j] + tol
        c = float(na @ nb)
        # Sides within about 0.06 degrees of parallel or antiparallel have no usable intersection.
        if 1.0 - c * c < 1e-6:
            raise FootprintGeometryError("two sides of the grid meet at a corner without an angle between them")
        alpha = (da - c * db) / (1.0 - c * c)
        beta = (db - c * da) / (1.0 - c * c)
        return p + alpha * na + beta * nb

    top_pts = xyz[0] + offset(top)
    right_pts = xyz[:, -1] + offset(right)
    bottom_pts = xyz[-1] + offset(bottom)
    left_pts = xyz[:, 0] + offset(left)
    with np.errstate(invalid="ignore"):
        corners = [
            corner(xyz[0, 0], top, 0, left, 0),
            corner(xyz[0, -1], top, -1, right, 0),
            corner(xyz[-1, -1], bottom, -1, right, -1),
            corner(xyz[-1, 0], bottom, 0, left, -1),
        ]
    ok_corners = [valid[0, 0], valid[0, -1], valid[-1, -1], valid[-1, 0]]

    runs = [
        (top_pts[1:-1], valid[0, 1:-1]),
        (right_pts[1:-1], valid[1:-1, -1]),
        (bottom_pts[-2:0:-1], valid[-1, -2:0:-1]),
        (left_pts[-2:0:-1], valid[-2:0:-1, 0]),
    ]
    sides = []
    for k, (pts, ok) in enumerate(runs):
        start, end = corners[k], corners[(k + 1) % 4]
        ok_start, ok_end = ok_corners[k], ok_corners[(k + 1) % 4]
        side_pts = np.concatenate([start[None][[ok_start]], pts[ok], end[None][[ok_end]]])
        sides.append(_normalize(side_pts) if len(side_pts) else side_pts)
    return sides


def _join(sides: list[np.ndarray]) -> np.ndarray:
    """Concatenate sides into an open ring, dropping points equal to their predecessor (cyclically)."""
    ring = np.concatenate(sides)
    repeated = (ring == np.roll(ring, 1, axis=0)).all(axis=1)
    return ring[~repeated]


def _arc_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angular distance, in radians, from each point to the minor great-circle arc from ``a`` to ``b``."""

    def angle(p, q):
        return np.arctan2(np.linalg.norm(np.cross(p, q), axis=-1), p @ q)

    to_ends = np.minimum(angle(points, a), angle(points, b))
    n = np.cross(a, b)
    norm = np.linalg.norm(n)
    if norm == 0.0:
        return to_ends
    n = n / norm
    # A point is abreast of the arc when its projection onto the arc's plane lies between a and b.
    abreast = (np.cross(a, points) @ n >= 0) & (np.cross(points, b) @ n >= 0)
    return np.where(abreast, np.abs(np.arcsin(np.clip(points @ n, -1.0, 1.0))), to_ends)


def _simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on the sphere, keeping the first and last points."""
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d = _arc_distance(points[i + 1 : j], points[i], points[j])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return points[keep]


def _gnomonic_basis(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East and north unit vectors at ``center``, right-handed with ``center`` pointing outward."""
    east = np.cross([0.0, 0.0, 1.0], center)
    if np.linalg.norm(east) < 1e-12:
        east = np.array([0.0, 1.0, 0.0])
    east = _normalize(east)
    return east, np.cross(center, east)


def _project(xyz: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Gnomonic projection: great circles map to straight lines."""
    d = xyz @ center
    return np.stack([(xyz @ east) / d, (xyz @ north) / d], axis=-1)


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _points_in_ring(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Crossing-number test of ``points`` (N, 2) against the open ring ``ring`` (M, 2)."""
    inside = np.zeros(len(points), dtype=bool)
    px, py = points[:, 0], points[:, 1]
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        straddles = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddles & (px < x_cross)
    return inside


def _self_intersects(ring: np.ndarray) -> bool:
    """True if any two non-adjacent edges of the open ring ``ring`` intersect."""
    a = ring
    b = np.roll(ring, -1, axis=0)
    n = len(ring)

    def orient(p, q, r):
        return np.sign(
            (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
        )

    i, j = np.triu_indices(n, k=2)
    nonadjacent = ~((i == 0) & (j == n - 1))
    i, j = i[nonadjacent], j[nonadjacent]
    o1 = orient(a[i], b[i], a[j])
    o2 = orient(a[i], b[i], b[j])
    o3 = orient(a[j], b[j], a[i])
    o4 = orient(a[j], b[j], b[i])

    def on_segment(p, q, r):
        """For collinear p, q, r: True if r lies within the bounding box of segment p-q."""
        return (
            (np.minimum(p[..., 0], q[..., 0]) <= r[..., 0])
            & (r[..., 0] <= np.maximum(p[..., 0], q[..., 0]))
            & (np.minimum(p[..., 1], q[..., 1]) <= r[..., 1])
            & (r[..., 1] <= np.maximum(p[..., 1], q[..., 1]))
        )

    general = (o1 != o2) & (o3 != o4)
    collinear = (
        ((o1 == 0) & on_segment(a[i], b[i], a[j]))
        | ((o2 == 0) & on_segment(a[i], b[i], b[j]))
        | ((o3 == 0) & on_segment(a[j], b[j], a[i]))
        | ((o4 == 0) & on_segment(a[j], b[j], b[i]))
    )
    return bool(np.any(general | collinear))


def _center(pixels: np.ndarray) -> np.ndarray:
    """Unit vector at the mean of the pixels, checking that they fit the gnomonic projection around it."""
    center = pixels.mean(axis=0)
    if np.linalg.norm(center) < 1e-6:
        raise ValueError("swath pixels have no well-defined center")
    center = _normalize(center)
    _check_extent(pixels, center)
    return center


def _check_extent(points: np.ndarray, center: np.ndarray) -> None:
    if (points @ center).min() < np.cos(np.radians(MAX_ANGULAR_RADIUS_DEG)):
        raise ValueError(f"points extend more than {MAX_ANGULAR_RADIUS_DEG} degrees from their center")


def _verify(vertices: np.ndarray, pixels: np.ndarray, edge_ring: np.ndarray) -> np.ndarray:
    """Check the polygon against the pixels and return its vertices in counter-clockwise order."""
    center = _center(pixels)
    _check_extent(vertices, center)

    east, north = _gnomonic_basis(center)
    ring = _project(vertices, center, east, north)

    if len(ring) < 3 or (np.linalg.norm(ring - np.roll(ring, -1, axis=0), axis=1) == 0).any():
        raise FootprintGeometryError("footprint has fewer than 3 distinct vertices")
    if _signed_area(ring) < 0:
        vertices, ring = vertices[::-1], ring[::-1]
    if _self_intersects(ring):
        raise FootprintGeometryError("footprint polygon crosses itself")

    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    poles = poles[poles @ center > 0]
    if len(poles) and _points_in_ring(_project(poles, center, east, north), ring).any():
        raise ValueError("footprint polygon contains a pole")

    for name, pts in (("valid pixel centers", pixels), ("half-pixel edge ring points", edge_ring)):
        outside = int((~_points_in_ring(_project(pts, center, east, north), ring)).sum())
        if outside:
            raise FootprintGeometryError(f"{outside} {name} fall outside the footprint polygon")
    return vertices


def convex_footprint(latitude: np.ndarray, longitude: np.ndarray, *, tolerance_km: float | None = None) -> np.ndarray:
    """Return the buffered spherical convex hull of a swath of geolocated pixels.

    For swaths whose edge ``swath_footprint`` cannot trace, such as folded, overlapping or
    single-row grids. The hull of the valid pixel centers is taken in a gnomonic projection, where
    great circles are straight lines, so it is the spherical convex hull and its edges are the arcs
    CMR draws. Each hull vertex is then surrounded by an octagon whose edges lie at least ``r``
    from it, with ``r`` half the pixel diagonal plus ``tolerance_km``, and the hull is taken again.
    Douglas-Peucker then drops hull vertices, which cuts into the convex hull by at most
    ``tolerance_km``. The polygon therefore contains every valid pixel center with a margin of at
    least half the median pixel diagonal, whatever the grid layout. It overreaches the swath where the swath is
    concave, such as the inner side of a curved swath or around a fold.

    The pixel diagonal comes from the median spacing of adjacent valid pixels along each grid axis.
    If one axis has no adjacent valid pixels, as in a single row, the other axis's spacing is used
    for both.

    Parameters and returns are as for ``swath_footprint``. With ``tolerance_km=None``, the
    tolerance is half the smaller pixel spacing, as there.

    Raises
    ------
    ValueError
        For invalid geolocation, as for ``swath_footprint``. Also if no two adjacent valid pixels
        exist along either grid axis, such as a single valid pixel: the geolocation is valid, but
        the pixel size, and so the margin, is unknown. With ``tolerance_km=None``, also if a median
        pixel spacing is zero, as for ``swath_footprint``.
    FootprintGeometryError
        If the hull fails verification, which indicates a defect in this function.
    """
    lat, lon = _validate(latitude, longitude, tolerance_km)
    xyz = _to_unit_vectors(lat, lon)
    valid = ~np.isnan(lat)
    pixels = xyz[valid]

    spacings = []
    for axis in (0, 1):
        if lat.shape[axis] > 1 and (np.moveaxis(valid, axis, 0)[1:] & np.moveaxis(valid, axis, 0)[:-1]).any():
            spacings.append(_median_spacing(xyz, valid, axis))
    if not spacings:
        raise ValueError("no two adjacent valid pixels along either grid axis")
    row_spacing, col_spacing = spacings * 2 if len(spacings) == 1 else spacings
    tol = _default_tolerance(spacings) if tolerance_km is None else tolerance_km / EARTH_MEAN_RADIUS_KM
    radius = 0.5 * np.hypot(row_spacing, col_spacing) + tol

    center = _center(pixels)
    east, north = _gnomonic_basis(center)
    # Joggling keeps qhull working when the centers are collinear; the octagons give the second hull area.
    if len(pixels) < 3:
        corners = pixels
    else:
        corners = pixels[ConvexHull(_project(pixels, center, east, north), qhull_options="QJ").vertices]
    buffered = _octagons(corners, radius / np.cos(np.pi / 8), east)
    _check_extent(buffered, center)
    hull = buffered[ConvexHull(_project(buffered, center, east, north)).vertices]
    # Split the closed ring at its first vertex and the vertex farthest from it, and simplify each half.
    far = int(np.argmax(np.linalg.norm(hull - hull[0], axis=1)))
    vertices = np.concatenate(
        [_simplify(hull[: far + 1], tol)[:-1], _simplify(np.concatenate([hull[far:], hull[:1]]), tol)[:-1]]
    )

    vertices = _verify(vertices, pixels, np.empty((0, 3)))
    return _to_lon_lat(np.concatenate([vertices, vertices[:1]]))


def footprint_margin_m(footprint: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Return the signed distance, in metres, from each point to the edge of a footprint polygon.

    The distance is along the sphere of ``EARTH_MEAN_RADIUS_KM`` to the nearest polygon edge, each
    edge being the minor great-circle arc between its vertices. It is positive for points inside the
    polygon and negative for points outside.

    Parameters
    ----------
    footprint : np.ndarray
        Polygon as returned by :func:`footprint_polygon`: (longitude, latitude) in degrees, shape
        (N + 1, 2), floating dtype, closed by repeating the first vertex, at least 3 distinct
        vertices, within ``MAX_ANGULAR_RADIUS_DEG`` of the mean of its vertices, not crossing itself.
    latitude : np.ndarray
        Geodetic latitude of the points, degrees in [-90, 90], floating dtype, any shape.
    longitude : np.ndarray
        Longitude of the points, degrees in [-180, 180], same shape as ``latitude``.

    Returns
    -------
    np.ndarray
        Signed distances in metres, float64, shape of ``latitude``.

    Raises
    ------
    ValueError
        If ``footprint`` is not a closed (N + 1, 2) floating array of finite, in-range coordinates
        with at least 3 distinct vertices, extends more than ``MAX_ANGULAR_RADIUS_DEG`` from its
        center, or crosses itself; or if ``latitude`` and ``longitude`` are not floating arrays of one shape holding
        finite, in-range values.
    """
    if not isinstance(footprint, np.ndarray) or footprint.ndim != 2 or footprint.shape[1] != 2:
        raise ValueError("footprint must be a numpy array of shape (N + 1, 2)")
    if not np.issubdtype(footprint.dtype, np.floating):
        raise ValueError(f"footprint must have a floating dtype, got {footprint.dtype}")
    if not np.isfinite(footprint).all():
        raise ValueError("footprint must contain only finite values")
    if (np.abs(footprint[:, 0]) > 180).any() or (np.abs(footprint[:, 1]) > 90).any():
        raise ValueError("footprint longitudes must be within [-180, 180] and latitudes within [-90, 90]")
    if len(footprint) < 2 or not np.array_equal(footprint[0], footprint[-1]):
        raise ValueError("footprint must be closed by repeating its first vertex")
    for name, arr in (("latitude", latitude), ("longitude", longitude)):
        if not isinstance(arr, np.ndarray) or not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"{name} must be a numpy array with a floating dtype")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
    if latitude.shape != longitude.shape:
        raise ValueError(f"latitude shape {latitude.shape} does not match longitude shape {longitude.shape}")
    if (np.abs(latitude) > 90).any() or (np.abs(longitude) > 180).any():
        raise ValueError("latitude must be within [-90, 90] and longitude within [-180, 180]")

    vertices = _join([_to_unit_vectors(footprint[:-1, 1], footprint[:-1, 0])])
    if len(vertices) < 3:
        raise ValueError("footprint has fewer than 3 distinct vertices")
    center = _normalize(vertices.mean(axis=0))
    _check_extent(vertices, center)

    east, north = _gnomonic_basis(center)
    ring = _project(vertices, center, east, north)
    if _self_intersects(ring):
        raise ValueError("footprint polygon crosses itself")

    points = _to_unit_vectors(latitude.astype(np.float64).ravel(), longitude.astype(np.float64).ravel())
    # A point more than 90 degrees from the center cannot be inside a polygon within 60 degrees of it,
    # and the gnomonic projection does not reach it.
    near = points @ center > 0
    inside = np.zeros(len(points), dtype=bool)
    inside[near] = _points_in_ring(_project(points[near], center, east, north), ring)

    distance = np.full(len(points), np.inf)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        distance = np.minimum(distance, _arc_distance(points, a, b))
    margin_m = np.where(inside, distance, -distance) * EARTH_MEAN_RADIUS_KM * 1e3
    return margin_m.reshape(latitude.shape)


def _octagons(points: np.ndarray, circumradius: float, east: np.ndarray) -> np.ndarray:
    """Vertices of a regular spherical octagon around each point, ``circumradius`` in radians."""
    u = _normalize(east - (points @ east)[:, None] * points)
    w = np.cross(points, u)
    theta = np.arange(8) * np.pi / 4
    offsets = np.cos(theta)[None, :, None] * u[:, None] + np.sin(theta)[None, :, None] * w[:, None]
    ring = np.cos(circumradius) * points[:, None] + np.sin(circumradius) * offsets
    return ring.reshape(-1, 3)
