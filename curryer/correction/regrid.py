"""GCP chip regridding algorithms.

This module provides functionality to transform GCP chips from irregular
geodetic grids (derived from ECEF coordinates) to regular latitude/longitude
grids. The regridding process is mission-agnostic and configurable via RegridConfig.

The main workflow is:
1. Load raw GCP chip with ECEF coordinates (from HDF file)
2. Convert ECEF → geodetic (lon, lat, h)
3. Determine output grid bounds and spacing
4. Interpolate data onto regular grid using bilinear interpolation
5. Return ImageGrid with regular lat/lon coordinates

@author: Brandon Stone, NASA Langley Research Center
"""

from __future__ import annotations

import logging

import numpy as np

from ..compute.spatial import ecef_to_geodetic
from .data_structures import ImageGrid, RegridConfig

logger = logging.getLogger(__name__)


def compute_regular_grid_bounds(
    lon_irregular: np.ndarray,
    lat_irregular: np.ndarray,
    conservative: bool = True,
) -> tuple[float, float, float, float]:
    """
    Compute bounds for regular output grid from irregular input.

    Parameters
    ----------
    lon_irregular, lat_irregular : np.ndarray
        2D arrays of irregular grid coordinates (degrees).
    conservative : bool, default=True
        If True, shrink bounds to ensure all output points are within input.
        Conservative bounds avoid extrapolation at edges by taking the maximum
        of left/bottom edges and minimum of right/top edges.

    Returns
    -------
    minlon, maxlon, minlat, maxlat : float
        Bounding box for regular grid (degrees).

    Notes
    -----
    Conservative bounds (default) follow MATLAB behavior:
    - minlon = max(bottom_left_lon, top_left_lon)
    - maxlon = min(bottom_right_lon, top_right_lon)
    - minlat = max(bottom_left_lat, bottom_right_lat)
    - maxlat = min(top_left_lat, top_right_lat)

    This ensures the regular grid lies entirely within the irregular grid.
    """
    if conservative:
        # Get corner coordinates (assuming row increases south, col increases east)
        # Corners: [0,0]=top_left, [0,-1]=top_right, [-1,0]=bottom_left, [-1,-1]=bottom_right
        minlon = max(lon_irregular[-1, 0], lon_irregular[0, 0])  # bottom-left, top-left
        maxlon = min(lon_irregular[0, -1], lon_irregular[-1, -1])  # top-right, bottom-right
        minlat = max(lat_irregular[-1, 0], lat_irregular[-1, -1])  # bottom corners
        maxlat = min(lat_irregular[0, 0], lat_irregular[0, -1])  # top corners
    else:
        # Use full extent
        minlon = float(lon_irregular.min())
        maxlon = float(lon_irregular.max())
        minlat = float(lat_irregular.min())
        maxlat = float(lat_irregular.max())

    return minlon, maxlon, minlat, maxlat


def create_regular_grid(
    bounds: tuple[float, float, float, float],
    grid_size: tuple[int, int] | None = None,
    resolution: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create regular lat/lon grid.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        (minlon, maxlon, minlat, maxlat) in degrees.
    grid_size : tuple[int, int], optional
        (nrows, ncols). If None, derive from resolution.
    resolution : tuple[float, float], optional
        (dlat, dlon) in degrees. If None, use grid_size.

    Returns
    -------
    lon_regular, lat_regular : np.ndarray
        2D arrays of regular grid coordinates (degrees), shape (nrows, ncols).

    Notes
    -----
    Exactly one of grid_size or resolution must be provided.
    Grid follows MATLAB convention:
    - Row index increases going south (latitude decreases)
    - Column index increases going east (longitude increases)
    """
    minlon, maxlon, minlat, maxlat = bounds

    if grid_size is not None and resolution is not None:
        raise ValueError("Specify only one of grid_size or resolution, not both")
    if grid_size is None and resolution is None:
        raise ValueError("Must specify either grid_size or resolution")

    if grid_size is not None:
        nrows, ncols = grid_size
        dlat = (maxlat - minlat) / (nrows - 1)
        dlon = (maxlon - minlon) / (ncols - 1)
    else:
        dlat, dlon = resolution
        nrows = int((maxlat - minlat) / dlat) + 1
        ncols = int((maxlon - minlon) / dlon) + 1

    # Create 1D coordinate arrays
    # Latitude: starts at maxlat (north) and decreases
    lat_1d = maxlat - np.arange(nrows) * dlat
    # Longitude: starts at minlon (west) and increases
    lon_1d = minlon + np.arange(ncols) * dlon

    # Create 2D meshgrid
    lon_regular, lat_regular = np.meshgrid(lon_1d, lat_1d)

    logger.debug(f"Created regular grid: {nrows}×{ncols}, resolution: ({dlat:.6f}°, {dlon:.6f}°)")

    return lon_regular, lat_regular


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product: a[0]*b[1] - a[1]*b[0]."""
    return a[0] * b[1] - a[1] * b[0]


def point_in_triangle(
    point: np.ndarray,
    triangle: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """
    Check if point is inside triangle using barycentric coordinates.

    Parameters
    ----------
    point : np.ndarray
        Point [x, y] to test.
    triangle : np.ndarray
        Triangle vertices, shape (3, 2): [[x1, y1], [x2, y2], [x3, y3]].

    Returns
    -------
    inside : bool
        True if point is inside triangle (barycentric coords all in (0, 1)).
    barycentric_coords : np.ndarray
        Barycentric coordinates [w1, w2, w3].

    Notes
    -----
    Uses the cross product method from MATLAB bandval function.
    Point P is inside triangle ABC if all barycentric weights are in (0, 1).
    """
    A, B, C = triangle

    # Compute denominator (area of triangle * 2)
    # MATLAB: d=cross2(A,B)+cross2(B,C)+cross2(C,A)
    d = _cross2(A - C, B - C)  # Simplified: same as above

    if abs(d) < 1e-14:  # Degenerate triangle
        return False, np.array([0.0, 0.0, 0.0])

    # Compute barycentric coordinates following MATLAB bandval logic
    # MATLAB: wA=(cross2(B,C)+cross2(P,B-C))/d
    wA = _cross2(point - C, B - C) / d
    wB = _cross2(A - C, point - C) / d
    wC = 1.0 - wA - wB

    # Check if point is inside (all weights in [0, 1])
    # Use small tolerance for boundary cases
    # MATLAB uses strict inequalities (0 < w < 1), but for numerical stability
    # and to handle edge cases in regridding, we use tolerant checks
    tol = 1e-10
    inside = (-tol <= wA <= 1 + tol) and (-tol <= wB <= 1 + tol) and (-tol <= wC <= 1 + tol)

    return inside, np.array([wA, wB, wC])


def bilinear_interpolate_quad(
    point: np.ndarray,
    corners_lon: np.ndarray,
    corners_lat: np.ndarray,
    corner_values: np.ndarray,
) -> float:
    """
    Bilinear interpolation within an irregular quadrilateral.

    Parameters
    ----------
    point : np.ndarray
        Target point [lon, lat] (degrees).
    corners_lon, corners_lat : np.ndarray
        Coordinates of 4 corners, ordered clockwise from top-left:
        [top-left, top-right, bottom-right, bottom-left].
    corner_values : np.ndarray
        Values at the 4 corners.

    Returns
    -------
    interpolated_value : float
        Interpolated value at target point.

    Notes
    -----
    Uses matrix inversion method from MATLAB code:
    Solves [1, lon, lat, lon*lat]^T = M * [w1, w2, w3, w4]^T
    where M is constructed from corner coordinates.
    """
    lon_p, lat_p = point

    # Build interpolation matrix (4x4)
    # M = [[1,    1,    1,    1   ]
    #      [lon1, lon2, lon3, lon4]
    #      [lat1, lat2, lat3, lat4]
    #      [lon1*lat1, lon2*lat2, lon3*lat3, lon4*lat4]]
    M = np.ones((4, 4))
    M[1, :] = corners_lon
    M[2, :] = corners_lat
    M[3, :] = corners_lon * corners_lat

    # Right-hand side vector
    E = np.array([1.0, lon_p, lat_p, lon_p * lat_p])

    # Solve for weights
    try:
        weights = np.linalg.solve(M, E)
    except np.linalg.LinAlgError:
        # Singular matrix - degenerate quadrilateral
        # Fall back to simple average
        logger.warning("Singular matrix in bilinear interpolation, using average")
        return float(np.mean(corner_values))

    # Compute interpolated value
    value = np.dot(weights, corner_values)

    return float(value)


def _check_point_in_cell(
    point_lon: float,
    point_lat: float,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    i: int,
    j: int,
) -> bool:
    """Fast check if point is in cell [i,j]. Returns True/False."""
    tol = 1e-10

    # Get corner coordinates
    lon_tl, lat_tl = lon_grid[i, j], lat_grid[i, j]
    lon_tr, lat_tr = lon_grid[i, j + 1], lat_grid[i, j + 1]
    lon_br, lat_br = lon_grid[i + 1, j + 1], lat_grid[i + 1, j + 1]
    lon_bl, lat_bl = lon_grid[i + 1, j], lat_grid[i + 1, j]

    # Test upper-left triangle (TL, TR, BL)
    d_ul = (lon_tl - lon_bl) * (lat_tr - lat_bl) - (lat_tl - lat_bl) * (lon_tr - lon_bl)

    if abs(d_ul) > 1e-14:
        wA = ((point_lon - lon_bl) * (lat_tr - lat_bl) - (point_lat - lat_bl) * (lon_tr - lon_bl)) / d_ul
        wB = ((lon_tl - lon_bl) * (point_lat - lat_bl) - (lat_tl - lat_bl) * (point_lon - lon_bl)) / d_ul
        wC = 1.0 - wA - wB

        if (-tol <= wA <= 1 + tol) and (-tol <= wB <= 1 + tol) and (-tol <= wC <= 1 + tol):
            return True

    # Test lower-right triangle (TR, BR, BL)
    d_lr = (lon_tr - lon_bl) * (lat_br - lat_bl) - (lat_tr - lat_bl) * (lon_br - lon_bl)

    if abs(d_lr) > 1e-14:
        wA = ((point_lon - lon_bl) * (lat_br - lat_bl) - (point_lat - lat_bl) * (lon_br - lon_bl)) / d_lr
        wB = ((lon_tr - lon_bl) * (point_lat - lat_bl) - (lat_tr - lat_bl) * (point_lon - lon_bl)) / d_lr
        wC = 1.0 - wA - wB

        if (-tol <= wA <= 1 + tol) and (-tol <= wB <= 1 + tol) and (-tol <= wC <= 1 + tol):
            return True

    return False


def find_containing_cell(
    point: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    start_cell: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """
    Find which cell in irregular grid contains the target point.

    Parameters
    ----------
    point : np.ndarray
        Target point [lon, lat] (degrees).
    lon_grid, lat_grid : np.ndarray
        2D arrays of irregular grid coordinates (degrees).
    start_cell : tuple[int, int], optional
        Starting cell (i, j) for search (optimization hint).

    Returns
    -------
    cell_indices : tuple[int, int] or None
        (i, j) of cell containing point, or None if not found.

    Notes
    -----
    Uses barycentric coordinate test to check if point is inside
    quadrilateral. For each cell, tests two triangles (upper-left and
    lower-right) that together form the quadrilateral.

    Search strategy follows MATLAB optimization:
    - Start from hint if provided
    - Check cells near last found cell (spatial locality)
    - If not found, search all cells

    Optimization: Inline triangle test to avoid array allocations.
    """
    nrows, ncols = lon_grid.shape
    max_i, max_j = nrows - 1, ncols - 1

    point_lon, point_lat = point[0], point[1]
    tol = 1e-10

    def check_cell(i: int, j: int) -> bool:
        """Check if point is in cell [i,j]. Optimized inline version."""
        # Get corner coordinates (avoid array allocation)
        lon_tl, lat_tl = lon_grid[i, j], lat_grid[i, j]
        lon_tr, lat_tr = lon_grid[i, j + 1], lat_grid[i, j + 1]
        lon_br, lat_br = lon_grid[i + 1, j + 1], lat_grid[i + 1, j + 1]
        lon_bl, lat_bl = lon_grid[i + 1, j], lat_grid[i + 1, j]

        # Test upper-left triangle (TL, TR, BL)
        # Inline barycentric coordinate calculation
        d_ul = (lon_tl - lon_bl) * (lat_tr - lat_bl) - (lat_tl - lat_bl) * (lon_tr - lon_bl)

        if abs(d_ul) > 1e-14:
            wA = ((point_lon - lon_bl) * (lat_tr - lat_bl) - (point_lat - lat_bl) * (lon_tr - lon_bl)) / d_ul
            wB = ((lon_tl - lon_bl) * (point_lat - lat_bl) - (lat_tl - lat_bl) * (point_lon - lon_bl)) / d_ul
            wC = 1.0 - wA - wB

            if (-tol <= wA <= 1 + tol) and (-tol <= wB <= 1 + tol) and (-tol <= wC <= 1 + tol):
                return True

        # Test lower-right triangle (TR, BR, BL)
        d_lr = (lon_tr - lon_bl) * (lat_br - lat_bl) - (lat_tr - lat_bl) * (lon_br - lon_bl)

        if abs(d_lr) > 1e-14:
            wA = ((point_lon - lon_bl) * (lat_br - lat_bl) - (point_lat - lat_bl) * (lon_br - lon_bl)) / d_lr
            wB = ((lon_tr - lon_bl) * (point_lat - lat_bl) - (lat_tr - lat_bl) * (point_lon - lon_bl)) / d_lr
            wC = 1.0 - wA - wB

            if (-tol <= wA <= 1 + tol) and (-tol <= wB <= 1 + tol) and (-tol <= wC <= 1 + tol):
                return True

        return False

    # Determine search start
    if start_cell is not None:
        start_i = max(0, min(start_cell[0] - 1, max_i - 1))
        start_j = max(0, min(start_cell[1] - 1, max_j - 1))

        # Search in a small window around hint first
        window_size = 3
        for di in range(-window_size, window_size + 1):
            for dj in range(-window_size, window_size + 1):
                i = start_i + di
                j = start_j + dj
                if 0 <= i < max_i and 0 <= j < max_j:
                    if check_cell(i, j):
                        return (i, j)

        # Not found in window - return None (caller will use spatial index or expand search)
        return None
    else:
        # No hint provided - do full search (only happens for very first point or edge cases)
        # This is expensive but necessary for correctness when no hint is available
        for i in range(max_i):
            for j in range(max_j):
                if check_cell(i, j):
                    return (i, j)

    # Not found
    return None


def regrid_irregular_to_regular(
    data_irregular: np.ndarray,
    lon_irregular: np.ndarray,
    lat_irregular: np.ndarray,
    lon_regular: np.ndarray,
    lat_regular: np.ndarray,
    method: str = "bilinear",
    fill_value: float = np.nan,
    use_spatial_index: bool = True,
) -> np.ndarray:
    """
    Regrid data from irregular geodetic grid to regular lat/lon grid.

    This is the core algorithm: for each point in the regular output grid,
    find the corresponding quadrilateral cell in the irregular input grid
    and interpolate the value.

    Parameters
    ----------
    data_irregular : np.ndarray
        2D array of values on irregular grid.
    lon_irregular, lat_irregular : np.ndarray
        2D arrays of irregular grid coordinates (degrees).
    lon_regular, lat_regular : np.ndarray
        2D arrays of regular grid coordinates (degrees).
    method : str, default="bilinear"
        Interpolation method: "bilinear" or "nearest".
    fill_value : float, default=np.nan
        Value for output points that fall outside input grid.
    use_spatial_index : bool, default=True
        If True, build a spatial index (KD-tree) for faster cell finding.
        Recommended for large grids (>100×100). Adds ~0.1s overhead.

    Returns
    -------
    data_regular : np.ndarray
        2D array of interpolated values on regular grid.

    Notes
    -----
    Algorithm (follows MATLAB Chip_regrid2.m):
    1. For each point P in regular grid:
       a. Search for containing quadrilateral in irregular grid
       b. Perform bilinear interpolation using 4 corner values
    2. Optimization: Use spatial locality (start search near last found cell)
    3. Points outside irregular grid are filled with fill_value

    Performance: O(n²) worst case, O(n²/k) typical with spatial locality.

    Optimizations applied:
    - Minimize array allocations in inner loop
    - Extract corner data once per cell
    - Use scalar operations where possible
    - Optional spatial index for O(log n) nearest neighbor queries
    """
    nrows_out, ncols_out = lon_regular.shape
    data_regular = np.full((nrows_out, ncols_out), fill_value)

    # Build spatial index if enabled (speeds up cell finding for large grids)
    kdtree = None
    cell_centers = None
    if use_spatial_index and lon_irregular.shape[0] > 50:
        try:
            from scipy.spatial import cKDTree

            # Compute cell centers for spatial index
            nrows_in, ncols_in = lon_irregular.shape
            cell_centers_list = []
            cell_indices_list = []

            for i in range(nrows_in - 1):
                for j in range(ncols_in - 1):
                    # Cell center (approximate)
                    center_lon = 0.25 * (
                        lon_irregular[i, j]
                        + lon_irregular[i, j + 1]
                        + lon_irregular[i + 1, j]
                        + lon_irregular[i + 1, j + 1]
                    )
                    center_lat = 0.25 * (
                        lat_irregular[i, j]
                        + lat_irregular[i, j + 1]
                        + lat_irregular[i + 1, j]
                        + lat_irregular[i + 1, j + 1]
                    )
                    cell_centers_list.append([center_lon, center_lat])
                    cell_indices_list.append((i, j))

            cell_centers = np.array(cell_centers_list)
            cell_indices_map = cell_indices_list
            kdtree = cKDTree(cell_centers)
            logger.debug(f"Built spatial index with {len(cell_indices_map)} cells")

        except ImportError:
            logger.debug("scipy.spatial.cKDTree not available, using sequential search")
            kdtree = None

    # Track last found cell for optimization (spatial locality)
    last_cell = None
    first_cell_of_row = None

    logger.info(f"Regridding {nrows_out}×{ncols_out} points using {method} interpolation...")

    # Iterate through regular grid points
    for ii in range(nrows_out):
        if ii % 50 == 0:
            logger.debug(f"  Processing row {ii + 1}/{nrows_out}")

        # Reset search hint at start of each row (follow MATLAB pattern)
        if first_cell_of_row is not None:
            last_cell = first_cell_of_row

        for jj in range(ncols_out):
            point_lon = lon_regular[ii, jj]
            point_lat = lat_regular[ii, jj]
            point = np.array([point_lon, point_lat])

            cell = None

            # Strategy:
            # 1. If we have a hint from previous point, use windowed search
            # 2. If windowed search fails or no hint, use spatial index
            # 3. If no spatial index, do full search (slow, but correct)

            if last_cell is not None:
                # Try windowed search around last cell
                cell = find_containing_cell(point, lon_irregular, lat_irregular, last_cell)

            # If windowed search failed or no hint, use spatial index
            if cell is None and kdtree is not None:
                # Query k nearest neighbors to find containing cell
                distances, indices = kdtree.query(point, k=min(9, len(cell_centers)))

                for idx in indices:
                    candidate_i, candidate_j = cell_indices_map[idx]
                    # Direct check if point is in this candidate cell (fast)
                    if _check_point_in_cell(
                        point_lon, point_lat, lon_irregular, lat_irregular, candidate_i, candidate_j
                    ):
                        cell = (candidate_i, candidate_j)
                        break

            # Last resort: full search (only if no spatial index available)
            if cell is None and kdtree is None:
                cell = find_containing_cell(point, lon_irregular, lat_irregular, None)

            if cell is None:
                # Point outside irregular grid, leave as fill_value
                continue

            i, j = cell

            # Get corner coordinates and values (extract once)
            # Clockwise from top-left: TL, TR, BR, BL
            lon_tl, lat_tl = lon_irregular[i, j], lat_irregular[i, j]
            lon_tr, lat_tr = lon_irregular[i, j + 1], lat_irregular[i, j + 1]
            lon_br, lat_br = lon_irregular[i + 1, j + 1], lat_irregular[i + 1, j + 1]
            lon_bl, lat_bl = lon_irregular[i + 1, j], lat_irregular[i + 1, j]

            val_tl = data_irregular[i, j]
            val_tr = data_irregular[i, j + 1]
            val_br = data_irregular[i + 1, j + 1]
            val_bl = data_irregular[i + 1, j]

            # Interpolate
            if method == "bilinear":
                # Inline bilinear interpolation (avoid function call overhead)
                # Build interpolation matrix (4x4) and solve
                M = np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [lon_tl, lon_tr, lon_br, lon_bl],
                        [lat_tl, lat_tr, lat_br, lat_bl],
                        [lon_tl * lat_tl, lon_tr * lat_tr, lon_br * lat_br, lon_bl * lat_bl],
                    ]
                )

                E = np.array([1.0, point_lon, point_lat, point_lon * point_lat])

                try:
                    weights = np.linalg.solve(M, E)
                    data_regular[ii, jj] = (
                        weights[0] * val_tl + weights[1] * val_tr + weights[2] * val_br + weights[3] * val_bl
                    )
                except np.linalg.LinAlgError:
                    # Singular matrix - use simple average
                    data_regular[ii, jj] = 0.25 * (val_tl + val_tr + val_br + val_bl)

            elif method == "nearest":
                # Use nearest corner (avoid extra array allocations)
                dist_tl = (lon_tl - point_lon) ** 2 + (lat_tl - point_lat) ** 2
                dist_tr = (lon_tr - point_lon) ** 2 + (lat_tr - point_lat) ** 2
                dist_br = (lon_br - point_lon) ** 2 + (lat_br - point_lat) ** 2
                dist_bl = (lon_bl - point_lon) ** 2 + (lat_bl - point_lat) ** 2

                min_dist = min(dist_tl, dist_tr, dist_br, dist_bl)
                if min_dist == dist_tl:
                    data_regular[ii, jj] = val_tl
                elif min_dist == dist_tr:
                    data_regular[ii, jj] = val_tr
                elif min_dist == dist_br:
                    data_regular[ii, jj] = val_br
                else:
                    data_regular[ii, jj] = val_bl

            # Update search hint
            last_cell = cell
            if jj == 0:
                first_cell_of_row = cell

    logger.info("Regridding complete")

    return data_regular


def regrid_gcp_chip(
    band_data: np.ndarray,
    ecef_coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: RegridConfig,
) -> ImageGrid:
    """
    High-level function: Regrid GCP chip from ECEF to regular lat/lon grid.

    This is the main entry point for GCP chip regridding. It handles the complete
    workflow from ECEF coordinates to a regular geodetic grid.

    Parameters
    ----------
    band_data : np.ndarray
        2D array of radiometric values.
    ecef_coords : tuple[np.ndarray, np.ndarray, np.ndarray]
        (X, Y, Z) ECEF coordinate arrays (meters), each shape (nrows, ncols).
    config : RegridConfig
        Regridding configuration.

    Returns
    -------
    regridded_chip : ImageGrid
        Regridded data on regular lat/lon grid.

    Workflow
    --------
    1. Convert ECEF → geodetic (lon, lat, h) using curryer.compute.spatial
    2. Compute regular grid bounds (conservative or full extent)
    3. Create regular output grid (from resolution or size)
    4. Regrid data using bilinear interpolation
    5. Return ImageGrid with (data, lat, lon, h)

    Examples
    --------
    >>> from curryer.correction.image_io import load_gcp_chip_from_hdf
    >>> from curryer.correction.regrid import regrid_gcp_chip, RegridConfig
    >>> band, x, y, z = load_gcp_chip_from_hdf("chip.hdf")
    >>> config = RegridConfig(output_resolution_deg=(0.001, 0.001))
    >>> regridded = regrid_gcp_chip(band, (x, y, z), config)
    """
    ecef_x, ecef_y, ecef_z = ecef_coords

    # Validate input shapes
    if not (band_data.shape == ecef_x.shape == ecef_y.shape == ecef_z.shape):
        raise ValueError(
            f"Shape mismatch: band={band_data.shape}, x={ecef_x.shape}, y={ecef_y.shape}, z={ecef_z.shape}"
        )

    logger.info(f"Regridding GCP chip: input shape {band_data.shape}")

    # Step 1: Convert ECEF → geodetic
    logger.debug("Converting ECEF → geodetic coordinates...")
    nrows, ncols = band_data.shape

    # Flatten for vectorized conversion
    ecef_flat = np.stack([ecef_x.ravel(), ecef_y.ravel(), ecef_z.ravel()], axis=1)

    # Convert using Curryer's spatial module (vectorized, uses WGS84)
    lla_flat = ecef_to_geodetic(ecef_flat, meters=True, degrees=True)

    # Reshape back to 2D grids
    lon_irregular = lla_flat[:, 0].reshape(nrows, ncols)
    lat_irregular = lla_flat[:, 1].reshape(nrows, ncols)

    logger.debug(
        f"Geodetic range: lon=[{lon_irregular.min():.4f}, {lon_irregular.max():.4f}], "
        f"lat=[{lat_irregular.min():.4f}, {lat_irregular.max():.4f}]"
    )

    # Step 2: Determine output grid bounds
    if config.output_bounds is not None:
        bounds = config.output_bounds
        logger.debug(f"Using explicit bounds: {bounds}")
    else:
        bounds = compute_regular_grid_bounds(lon_irregular, lat_irregular, config.conservative_bounds)
        logger.debug(f"Computed bounds: {bounds}")

    # Step 3: Create regular grid
    if config.output_resolution_deg is not None:
        lon_regular, lat_regular = create_regular_grid(bounds, resolution=config.output_resolution_deg)
    elif config.output_grid_size is not None:
        lon_regular, lat_regular = create_regular_grid(bounds, grid_size=config.output_grid_size)
    else:
        # Auto mode: use input size
        logger.debug("Auto mode: using input grid size")
        lon_regular, lat_regular = create_regular_grid(bounds, grid_size=band_data.shape)

    # Step 4: Regrid data
    data_regular = regrid_irregular_to_regular(
        band_data,
        lon_irregular,
        lat_irregular,
        lon_regular,
        lat_regular,
        method=config.interpolation_method,
        fill_value=config.fill_value,
    )

    # Step 5: Create ImageGrid
    # Note: h is not regridded (would need separate interpolation), set to None
    regridded_chip = ImageGrid(
        data=data_regular,
        lat=lat_regular,
        lon=lon_regular,
        h=None,  # Height not interpolated
    )

    logger.info(f"Regridding complete: output shape {data_regular.shape}")

    return regridded_chip
