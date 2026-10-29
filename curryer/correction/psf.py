from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    RegularGridInterpolator,
)
from scipy.ndimage import map_coordinates
from scipy.signal import convolve2d, fftconvolve

from ..compute import constants
from ..compute.spatial import calc_azimuth, ecef_to_geodetic, geodetic_to_ecef
from .config import PSFSamplingConfig
from .grid_types import (
    ImageGrid,
    OpticalPSFEntry,
    ProjectedPSF,
    PSFGrid,
)

logger = logging.getLogger(__name__)


_CENTRING_TOLERANCE_PX = 1e-6
_MAX_CENTRING_ITERATIONS = 20


def centroid(weights: np.ndarray) -> float:
    """
    Compute center-of-mass index for a one-dimensional weight vector.

    Parameters
    ----------
    weights : np.ndarray
        One-dimensional weight array.

    Returns
    -------
    float
        Center-of-mass index position.

    Raises
    ------
    ValueError
        If weight vector has zero total mass.
    """

    w = np.asarray(weights, dtype=float).ravel()
    total = w.sum()
    if total == 0.0:
        raise ValueError("Cannot compute centroid of zero-mass vector.")
    indices = np.arange(w.size, dtype=float)
    return float(np.dot(indices, w) / total)


def project_psf(
    r_iss_ctrs_m: np.ndarray,
    optical_psfs: Iterable[OpticalPSFEntry],
    subimage: ImageGrid,
    los_set_hs: np.ndarray,
) -> ProjectedPSF:
    """
    Project optical PSF onto Earth's surface using vectorized ray tracing.

    Parameters
    ----------
    r_iss_ctrs_m : np.ndarray
        Spacecraft position in ECEF coordinates, shape (3,), units: meters.
    optical_psfs : Iterable[OpticalPSFEntry]
        Collection of optical PSF samples at different field angles.
    subimage : ImageGrid
        Image grid defining the observation geometry.
    los_set_hs : np.ndarray
        Line-of-sight unit vectors in instrument frame, shape (n_pixels, 3).

    Returns
    -------
    ProjectedPSF
        PSF projected onto Earth's surface with lat, lon, height grids.

    Raises
    ------
    ValueError
        If no optical PSF entries provided.
    RuntimeError
        If ray-ellipsoid intersection fails.
    """

    r_iss = np.asarray(r_iss_ctrs_m, dtype=float).reshape(3)
    los_set = np.asarray(los_set_hs, dtype=float)

    psf_list = list(optical_psfs)
    if not psf_list:
        raise ValueError("At least one optical PSF entry is required.")

    nframes, nrows = subimage.data.shape
    midframe = nframes // 2
    midrow = nrows // 2

    center_los = los_set[midrow]
    center_field_angle = np.arctan2(center_los[1], center_los[2])

    psf_diffs = [abs(np.deg2rad(np.mean(entry.field_angle)) - center_field_angle) for entry in psf_list]
    psf_idx = int(np.argmin(psf_diffs))
    entry = psf_list[psf_idx]

    xang = np.deg2rad(entry.x.copy())
    yang = np.deg2rad(entry.field_angle.copy())
    psf_values = entry.data.copy()

    lat = subimage.lat
    lon = subimage.lon
    if subimage.h is None:
        height = np.zeros_like(lat)
    else:
        height = subimage.h

    pix1 = 0
    pix2 = nrows - 1

    b1_hs = los_set[pix1]
    b2_hs = los_set[pix2]

    # llh1 = np.array([lat[midframe, pix1], lon[midframe, pix1], height[midframe, pix1]])
    # llh2 = np.array([lat[midframe, pix2], lon[midframe, pix2], height[midframe, pix2]])
    # p1 = lla_to_ecef(llh1[0], llh1[1], llh1[2])
    # p2 = lla_to_ecef(llh2[0], llh2[1], llh2[2])

    llh1 = np.array([lon[midframe, pix1], lat[midframe, pix1], height[midframe, pix1]])
    llh2 = np.array([lon[midframe, pix2], lat[midframe, pix2], height[midframe, pix2]])
    p1 = geodetic_to_ecef(llh1, meters=True, degrees=True)
    p2 = geodetic_to_ecef(llh2, meters=True, degrees=True)

    b1_ctrs = (p1 - r_iss) / np.linalg.norm(p1 - r_iss)
    b2_ctrs = (p2 - r_iss) / np.linalg.norm(p2 - r_iss)

    c1_hs = np.cross(b1_hs, b2_hs)
    c1_hs /= np.linalg.norm(c1_hs)
    c1_ctrs = np.cross(b1_ctrs, b2_ctrs)
    c1_ctrs /= np.linalg.norm(c1_ctrs)

    m_ctrs = np.column_stack((b1_ctrs, c1_ctrs, np.cross(b1_ctrs, c1_ctrs)))
    m_hs = np.column_stack((b1_hs, c1_hs, np.cross(b1_hs, c1_hs)))

    t_hs_to_ctrs = m_ctrs @ np.linalg.inv(m_hs)

    xang -= np.mean(xang)
    yang -= np.mean(yang)

    xangb = np.arcsin(center_los[0])
    yangb = np.arcsin(center_los[1])

    ny, nx = psf_values.shape

    # re = 6_378_140.0
    # rp = 6_356_750.0
    # Optimize: Vectorize nested loops using NumPy broadcasting
    # Create meshgrids for vectorized computation
    re = constants.WGS84_SEMI_MAJOR_AXIS_KM * 1000.0
    rp = constants.WGS84_SEMI_MINOR_AXIS_KM * 1000.0

    # Create full meshgrids for proper broadcasting
    xang_mesh, yang_mesh = np.meshgrid(xang, yang, indexing="xy")  # Both shape: (ny, nx)

    # Compute LOS vectors for all PSF pixels at once
    # plos_hs will be shape (ny, nx, 3)
    plos_x = np.sin(xangb + xang_mesh)  # Shape: (ny, nx)
    plos_y = np.sin(yangb + yang_mesh)  # Shape: (ny, nx)
    plos_z = np.sqrt(np.maximum(0.0, 1.0 - plos_x**2 - plos_y**2))  # Shape: (ny, nx)

    # Stack to create (ny, nx, 3) array
    plos_hs = np.stack([plos_x, plos_y, plos_z], axis=-1)

    # Transform all LOS vectors at once: (ny, nx, 3) @ (3, 3)^T = (ny, nx, 3)
    # Using Einstein summation for efficient matrix multiplication
    plos_ctrs = np.einsum("ijk,lk->ijl", plos_hs, t_hs_to_ctrs)

    # Vectorized ray-ellipsoid intersection
    re2 = re * re
    rp2 = rp * rp

    a = (plos_ctrs[..., 0] / re) ** 2 + (plos_ctrs[..., 1] / re) ** 2 + (plos_ctrs[..., 2] / rp) ** 2
    b = (
        2.0 * (r_iss[0] * plos_ctrs[..., 0] + r_iss[1] * plos_ctrs[..., 1]) / re2
        + 2.0 * r_iss[2] * plos_ctrs[..., 2] / rp2
    )
    c = (r_iss[0] / re) ** 2 + (r_iss[1] / re) ** 2 + (r_iss[2] / rp) ** 2 - 1.0

    discriminant = b * b - 4.0 * a * c
    if np.any(discriminant < 0):
        raise RuntimeError("Pseudo line of sight does not intersect the Earth ellipsoid.")

    slant_range = (-b - np.sqrt(discriminant)) / (2.0 * a)  # Shape: (ny, nx)

    # Compute surface intersection points: (ny, nx, 3)
    p_surface = r_iss[np.newaxis, np.newaxis, :] + slant_range[..., np.newaxis] * plos_ctrs

    # Reshape to (ny*nx, 3) for batch geodetic conversion
    p_surface_flat = p_surface.reshape(-1, 3)

    # Convert all points at once using vectorized ecef_to_geodetic
    # ecef_to_geodetic already supports batch processing
    llh_array = ecef_to_geodetic(p_surface_flat, meters=True, degrees=True)  # Shape: (ny*nx, 3)

    # Reshape back to (ny, nx) grids
    psf_lon = llh_array[:, 0].reshape(ny, nx)
    psf_lat = llh_array[:, 1].reshape(ny, nx)
    psf_h = llh_array[:, 2].reshape(ny, nx)

    return ProjectedPSF(data=psf_values, lat=psf_lat, lon=psf_lon, height=psf_h)


def convolve_gcp_with_psf(gcp: ImageGrid, psf: PSFGrid) -> ImageGrid:
    """
    Convolve GCP reference image with dynamic PSF using FFT.

    Parameters
    ----------
    gcp : ImageGrid
        Ground control point reference image.
    psf : PSFGrid
        Point spread function to convolve with.

    Returns
    -------
    ImageGrid
        Convolved GCP image with same coordinate grids.
    """

    kernel = np.flipud(psf.data)
    convolved = fftconvolve(gcp.data, kernel, mode="same")
    return ImageGrid(data=convolved, lat=gcp.lat, lon=gcp.lon, h=gcp.h)


def crop_to_full_psf_overlap(convolved: ImageGrid, psf: PSFGrid) -> ImageGrid:
    """
    Crop a GCP convolved by :func:`convolve_gcp_with_psf` to where the PSF lies wholly inside the chip.

    The convolution pads the chip with zeros, so within half a kernel of its edge the convolved
    chip fades toward zero. The cropped grid holds only pixels computed from chip data alone, the
    ``mode="valid"`` region of the convolution.

    Parameters
    ----------
    convolved : ImageGrid
        Output of :func:`convolve_gcp_with_psf`.
    psf : PSFGrid
        The PSF it was convolved with.

    Returns
    -------
    ImageGrid
        ``convolved`` with ``psf.data.shape[i] - 1`` rows and columns removed in total, split
        between the two edges as ``fftconvolve(mode="same")`` centres the kernel.

    Raises
    ------
    ValueError
        If the PSF kernel is not smaller than the chip along both axes.
    """

    (n_rows, n_cols), (k_rows, k_cols) = convolved.data.shape, psf.data.shape
    if k_rows >= n_rows or k_cols >= n_cols:
        raise ValueError(f"PSF kernel {psf.data.shape} is not smaller than the GCP chip {convolved.data.shape}.")
    rows = slice(k_rows // 2, n_rows - (k_rows - 1) // 2)
    cols = slice(k_cols // 2, n_cols - (k_cols - 1) // 2)
    h = None if convolved.h is None else convolved.h[rows, cols]
    return ImageGrid(data=convolved.data[rows, cols], lat=convolved.lat[rows, cols], lon=convolved.lon[rows, cols], h=h)


def convolve_psf_with_spacecraft_motion(
    psf: ProjectedPSF,
    composite_img: ImageGrid,
    config: PSFSamplingConfig,
) -> PSFGrid:
    """
    Apply spacecraft motion blur to projected PSF.

    The footprint moves along track during a frame, so the PSF is blurred by the
    mean lat/lon step between successive frames (rows) of *composite_img*.  The PSF
    is rotated so the motion runs along one grid axis, blurred along it, and
    rotated back onto a lat/lon grid that covers the whole blurred PSF.  The blur
    is the whole frame step, so the detector is taken to integrate for the whole
    frame period.

    Parameters
    ----------
    psf : ProjectedPSF
        Projected PSF on Earth's surface.
    composite_img : ImageGrid
        Observation subimage, rows = frames (along track); its frame step sets the
        direction and length of the blur.
    config : PSFSamplingConfig
        Configuration with PSF sampling parameters.

    Returns
    -------
    PSFGrid
        PSF convolved with spacecraft motion blur, on a regular lat/lon grid at the
        configured sample distances; zero outside the sampled PSF.

    Raises
    ------
    ValueError
        If *composite_img* has fewer than two frames, or its mean frame step is
        not finite or is zero.
    """
    if composite_img.lat.shape[0] < 2:
        raise ValueError("The motion blur needs at least two frames to give the frame step.")
    lat_motion = np.mean(np.diff(composite_img.lat, axis=0))
    lon_motion = np.mean(np.diff(composite_img.lon, axis=0))
    dist = np.hypot(lat_motion, lon_motion)
    if not (np.isfinite(dist) and dist > 0):
        raise ValueError(f"The mean frame step ({lat_motion}, {lon_motion}) deg must be finite and nonzero.")

    theta = np.arctan2(lat_motion, lon_motion)
    dlat = config.psf_lat_sample_dist_deg
    dlon = config.psf_lon_sample_dist_deg

    mid = np.array([np.mean(psf.lon), np.mean(psf.lat)])
    to_motion = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    samples = to_motion @ np.vstack((psf.lon.ravel() - mid[0], psf.lat.ravel() - mid[1]))

    x = np.arange(samples[0].min() - dist / 2.0, samples[0].max() + dist / 2.0 + dlon, dlon)
    y = np.arange(samples[1].min(), samples[1].max() + dlat, dlat)
    X, Y = np.meshgrid(x, y)
    sampled = LinearNDInterpolator(samples.T, psf.data.ravel(), fill_value=0.0)(X, Y)

    num_steps = max(1, int(round(dist / dlon)))
    psf_map = convolve2d(sampled, np.ones((1, num_steps)), mode="same", boundary="fill", fillvalue=0.0)

    corners = to_motion.T @ np.array([[x[0], x[-1], x[0], x[-1]], [y[0], y[0], y[-1], y[-1]]])
    lon = mid[0] + np.arange(corners[0].min(), corners[0].max() + dlon, dlon)
    lat = mid[1] + np.arange(corners[1].min(), corners[1].max() + dlat, dlat)
    LON, LAT = np.meshgrid(lon, lat)
    in_motion = to_motion @ np.vstack((LON.ravel() - mid[0], LAT.ravel() - mid[1]))
    data = map_coordinates(
        psf_map, [(in_motion[1] - y[0]) / dlat, (in_motion[0] - x[0]) / dlon], order=1, mode="constant", cval=0.0
    ).reshape(LON.shape)
    return PSFGrid(data=data, lat=lat, lon=lon)


def zero_pad_psf(psf: PSFGrid) -> PSFGrid:
    """
    Zero-pad PSF to center its centroid on the grid.

    Parameters
    ----------
    psf : PSFGrid
        Point spread function on rectilinear grid.

    Returns
    -------
    PSFGrid
        Zero-padded PSF with centered centroid.

    Raises
    ------
    ValueError
        If PSF lat/lon are not one-dimensional.
    """

    x = np.asarray(psf.lon, dtype=float)
    y = np.asarray(psf.lat, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("PSF latitude/longitude must be one-dimensional after motion blur.")

    grid_step_x = float(np.mean(np.diff(x)))
    grid_step_y = float(np.mean(np.diff(y)))

    sum_x = psf.data.sum(axis=0)
    sum_y = psf.data.sum(axis=1)

    x_center_index = centroid(sum_x)
    y_center_index = centroid(sum_y)

    x_center = np.interp(x_center_index, np.arange(x.size), x)
    y_center = np.interp(y_center_index, np.arange(y.size), y)

    max_left_x = (x_center - x[0]) / grid_step_x
    max_right_x = (x[-1] - x_center) / grid_step_x
    half_count_x = int(np.ceil(max(max_left_x, max_right_x)))

    max_left_y = (y_center - y[0]) / grid_step_y
    max_right_y = (y[-1] - y_center) / grid_step_y
    half_count_y = int(np.ceil(max(max_left_y, max_right_y)))

    x_new = x_center + np.arange(-half_count_x, half_count_x + 1) * grid_step_x
    y_new = y_center + np.arange(-half_count_y, half_count_y + 1) * grid_step_y

    interpolant = RegularGridInterpolator((y, x), psf.data, bounds_error=False, fill_value=0.0)
    Y_new, X_new = np.meshgrid(y_new, x_new, indexing="ij")
    data_new = interpolant(np.stack((Y_new.ravel(), X_new.ravel()), axis=-1)).reshape(Y_new.shape)

    return PSFGrid(data=data_new, lat=y_new, lon=x_new)


def resample_psf_to_gcp_resolution(psf: PSFGrid, gcp: ImageGrid) -> PSFGrid:
    """
    Resample PSF to match GCP reference image resolution.

    The resampled grid has an odd number of samples on each axis, and its
    sample positions are offset (by less than a pixel) so that the weighted
    centre of the sampled weights is the middle sample to within
    ``1e-6`` pixel; :func:`convolve_gcp_with_psf` then blurs the GCP without
    shifting it. Sampling at the PSF centroid alone leaves the discrete
    weights' centre off the middle sample by up to about 0.1 pixel on a
    coarse grid. The returned coordinates put the middle sample at the PSF
    centroid. The GCP pixel must be no wider than the PSF's RMS width on each
    axis, so that the samples represent the PSF.

    Parameters
    ----------
    psf : PSFGrid
        Point spread function on rectilinear grid.
    gcp : ImageGrid
        Ground control point reference defining target resolution.

    Returns
    -------
    PSFGrid
        Resampled PSF at GCP resolution, covering the whole input PSF.

    Raises
    ------
    ValueError
        If PSF is not on a rectilinear grid, its weights do not have a positive
        sum, the GCP pixel is wider than the PSF's RMS width on either axis, or
        the sampled weights cannot be centred.
    """

    if psf.lat.ndim != 1 or psf.lon.ndim != 1:
        raise ValueError("PSF must be on a rectilinear grid before resampling.")
    total = psf.data.sum()
    if not total > 0:
        raise ValueError(f"PSF weights sum to {total}; the centroid is undefined.")

    dlon = float(np.abs(np.mean(np.diff(gcp.lon, axis=1))))
    dlat = float(np.abs(np.mean(np.diff(gcp.lat, axis=0))))

    lat_c = float(np.sum(psf.data.sum(axis=1) * psf.lat) / total)
    lon_c = float(np.sum(psf.data.sum(axis=0) * psf.lon) / total)
    rms_lat = float(np.sqrt(np.sum(psf.data.sum(axis=1) * (psf.lat - lat_c) ** 2) / total))
    rms_lon = float(np.sqrt(np.sum(psf.data.sum(axis=0) * (psf.lon - lon_c) ** 2) / total))
    if dlat > rms_lat or dlon > rms_lon:
        raise ValueError(
            f"GCP pixel ({dlat:.3g}, {dlon:.3g}) deg lat/lon is wider than the PSF's RMS width "
            f"({rms_lat:.3g}, {rms_lon:.3g}) deg; too few samples to represent the PSF."
        )
    extent_x = max(lon_c - psf.lon.min(), psf.lon.max() - lon_c)
    extent_y = max(lat_c - psf.lat.min(), psf.lat.max() - lat_c)
    interpolant = RegularGridInterpolator((psf.lat, psf.lon), psf.data, bounds_error=False, fill_value=0.0)

    # Shift the samples until the discrete weights' centre is the middle sample; the offset
    # moves that centre by about offset / step, so each step removes the residual to first order.
    shift_x = shift_y = 0.0
    for _ in range(_MAX_CENTRING_ITERATIONS):
        half_x = int(np.ceil((extent_x + abs(shift_x)) / dlon))
        half_y = int(np.ceil((extent_y + abs(shift_y)) / dlat))
        k_x = np.arange(-half_x, half_x + 1)
        k_y = np.arange(-half_y, half_y + 1)
        Y_new, X_new = np.meshgrid(lat_c + shift_y + k_y * dlat, lon_c + shift_x + k_x * dlon, indexing="ij")
        data = interpolant(np.stack((Y_new.ravel(), X_new.ravel()), axis=-1)).reshape(Y_new.shape)
        off_y = centroid(data.sum(axis=1)) - half_y
        off_x = centroid(data.sum(axis=0)) - half_x
        if abs(off_x) < _CENTRING_TOLERANCE_PX and abs(off_y) < _CENTRING_TOLERANCE_PX:
            return PSFGrid(data=data, lat=lat_c + k_y * dlat, lon=lon_c + k_x * dlon)
        shift_x += off_x * dlon
        shift_y += off_y * dlat
    raise ValueError(
        f"Resampled PSF weights stay ({off_y:.3g}, {off_x:.3g}) pixel lat/lon off the middle sample after "
        f"{_MAX_CENTRING_ITERATIONS} centring steps."
    )


def normalize_psf(psf: PSFGrid) -> PSFGrid:
    """
    Normalize PSF to unit total power.

    Parameters
    ----------
    psf : PSFGrid
        Point spread function to normalize.

    Returns
    -------
    PSFGrid
        Normalized PSF with total power = 1.

    Raises
    ------
    ValueError
        If PSF has zero total power.
    """

    total = np.nansum(psf.data)
    if total == 0.0:
        raise ValueError("Cannot normalize PSF with zero total power.")
    psf.data = psf.data / total
    return psf


def validate_spacecraft_ecef_m(r_spacecraft_m: np.ndarray) -> np.ndarray:
    """Check that a spacecraft position is an ECEF vector in meters above the Earth.

    Catches the common unit and shape mistakes (kilometers instead of meters,
    a column vector, NaN fill) at the point where the position enters image
    matching or error statistics.  It cannot detect an inertial (e.g. J2000)
    position passed as ECEF, since both have Earth-orbit magnitudes.

    Parameters
    ----------
    r_spacecraft_m : ndarray of shape (3,)
        Spacecraft ECEF position in meters.

    Returns
    -------
    ndarray, shape (3,)
        The position as a float64 array.

    Raises
    ------
    ValueError
        If the position is not shape ``(3,)``, not finite, inside the Earth
        (shorter than the WGS-84 semi-minor axis), or not above the WGS-84
        ellipsoid.
    """
    r = np.asarray(r_spacecraft_m, dtype=float)
    if r.shape != (3,):
        raise ValueError(f"Spacecraft position must have shape (3,), got {r.shape}.")
    if not np.all(np.isfinite(r)):
        raise ValueError(f"Spacecraft position must be finite, got {r}.")
    if np.linalg.norm(r) < constants.WGS84_SEMI_MINOR_AXIS_KM * 1_000.0:
        raise ValueError(
            f"Spacecraft position {r} is {np.linalg.norm(r):.0f} m from the Earth's center, inside the Earth; "
            "expected an ECEF position in meters."
        )
    altitude_m = ecef_to_geodetic(r, meters=True)[2]
    if not altitude_m > 0.0:
        raise ValueError(
            f"Spacecraft position {r} m is {altitude_m:.0f} m relative to the WGS-84 ellipsoid; "
            "expected an ECEF position in meters above the surface."
        )
    return r


def _nearest_grid_pixel(grid: ImageGrid, lat_deg: float, lon_deg: float) -> tuple[int, int]:
    """Return the ``(row, col)`` of the grid pixel nearest a geodetic point.

    Raises
    ------
    ValueError
        If the point is farther from its nearest pixel than that pixel is
        from its farthest adjacent pixel, i.e. outside the grid.
    """
    coslat = np.cos(np.deg2rad(lat_deg))

    def dist2(lat, lon, lat0, lon0):
        return (lat - lat0) ** 2 + (((lon - lon0 + 180.0) % 360.0 - 180.0) * coslat) ** 2

    d2 = dist2(grid.lat, grid.lon, lat_deg, lon_deg)
    if not np.any(np.isfinite(d2)):
        raise ValueError("Observation grid has no finite lat/lon.")
    i, j = np.unravel_index(np.nanargmin(d2), d2.shape)
    rows, cols = d2.shape
    neighbours = [
        (i + di, j + dj) for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)) if 0 <= i + di < rows and 0 <= j + dj < cols
    ]
    spacing2 = np.array([dist2(grid.lat[n], grid.lon[n], grid.lat[i, j], grid.lon[i, j]) for n in neighbours])
    if not spacing2.size or not np.any(np.isfinite(spacing2)) or not d2[i, j] <= np.nanmax(spacing2):
        raise ValueError(
            f"Point (lat {lat_deg}, lon {lon_deg}) is outside the observation grid: its nearest pixel "
            f"({i}, {j}) is farther away than that pixel's adjacent pixels."
        )
    return int(i), int(j)


def resolve_spacecraft_ecef(
    grid: ImageGrid,
    r_spacecraft_m: np.ndarray,
    target_lat_deg: float,
    target_lon_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(r_spacecraft_m, boresight, t_matrix)`` for image-matching.

    The boresight is the unit line of sight from the spacecraft to the
    observation pixel in the mid-frame row and in the column of the pixel
    nearest the target (the GCP centre), using that pixel's lat/lon/height.
    The nadir-equivalent error is defined on the line of sight of the pixel
    that viewed the GCP.  With one mid-frame spacecraft position, the
    pixel in the GCP's cross-track column of the mid-frame row has the GCP's
    off-nadir angle; aiming at the GCP itself from the mid-frame position
    would not, when the GCP was imaged at a different frame.  The view-plane
    azimuth still differs from the GCP's by the along-track change in
    viewing direction between the mid-frame and the GCP's frame.

    The rotation matrix is the ``3×3`` identity because the boresight is
    already expressed in the CTRS (ECEF) frame.

    Parameters
    ----------
    grid : ImageGrid
        Observation grid with rows = frames (along track) and columns =
        cross-track pixels; its middle row is the frame of *r_spacecraft_m*.
        ``grid.lat``/``grid.lon`` in degrees, ``grid.h`` in meters above the
        WGS-84 ellipsoid (treated as 0 when ``None``).
    r_spacecraft_m : ndarray of shape (3,)
        Spacecraft ECEF position in meters at the observation mid-frame.
    target_lat_deg, target_lon_deg : float
        Geodetic latitude and longitude of the GCP centre, degrees.  Must lie
        within *grid*.

    Returns
    -------
    r_spacecraft_m : ndarray, shape (3,)
        ECEF spacecraft position in meters (float copy of the input).
    boresight : ndarray, shape (3,)
        Unit vector in ECEF from the spacecraft to the mid-frame pixel in the
        target's column.
    t_matrix : ndarray, shape (3, 3)
        Identity rotation matrix.

    Raises
    ------
    ValueError
        If *r_spacecraft_m* is ``None``, not shape ``(3,)``, not finite, or
        not above the WGS-84 ellipsoid; if the target lat/lon is not finite or
        lies outside *grid*; or if the lat/lon/height of the mid-frame pixel
        in the target's column is not finite.
    """
    if r_spacecraft_m is None:
        raise ValueError(
            "Spacecraft ECEF position is required to resolve the viewing geometry; "
            "supply it in the observation file ('position' for NetCDF, 'R_ISS_midframe' for .mat)."
        )
    r = validate_spacecraft_ecef_m(r_spacecraft_m)

    if not (np.isfinite(target_lat_deg) and np.isfinite(target_lon_deg)):
        raise ValueError(f"Target lat/lon must be finite, got ({target_lat_deg}, {target_lon_deg}).")
    _, col = _nearest_grid_pixel(grid, target_lat_deg, target_lon_deg)
    row = grid.mid_indices[0]
    height_m = 0.0 if grid.h is None else float(grid.h[row, col])
    lon_lat_h = np.array([grid.lon[row, col], grid.lat[row, col], height_m], dtype=float)
    if not np.all(np.isfinite(lon_lat_h)):
        raise ValueError(f"Mid-frame pixel ({row}, {col}) lon/lat/height must be finite, got {lon_lat_h}.")
    p_view = geodetic_to_ecef(lon_lat_h, meters=True, degrees=True)

    line_of_sight = p_view - r
    boresight = line_of_sight / np.linalg.norm(line_of_sight)
    return r, boresight, np.eye(3)


def ground_track_azimuth_deg(grid: ImageGrid, target_lat_deg: float, target_lon_deg: float) -> float:
    """Return the ground azimuth of increasing row at the grid pixel nearest a target.

    With rows = frames, this is the direction in which successive frames
    advance over the ground at the target: the along-track direction used to
    resolve a geolocation error into along-track and cross-track components.
    It is the azimuth, at the previous row, of the ellipsoid point of the next
    row in the target's column (one-sided at the first and last rows).

    Parameters
    ----------
    grid : ImageGrid
        Observation grid with rows = frames (along track) and columns =
        cross-track pixels; ``grid.lat``/``grid.lon`` in degrees.
    target_lat_deg, target_lon_deg : float
        Geodetic latitude and longitude of the target, degrees.  Must lie
        within *grid*.

    Returns
    -------
    float
        Azimuth in degrees clockwise from north, in ``[0, 360)``.

    Raises
    ------
    ValueError
        If the target lat/lon is not finite or lies outside *grid*; if *grid*
        has fewer than two rows; or if the two pixels used are not finite or
        coincide.
    """
    if not (np.isfinite(target_lat_deg) and np.isfinite(target_lon_deg)):
        raise ValueError(f"Target lat/lon must be finite, got ({target_lat_deg}, {target_lon_deg}).")
    n_rows = grid.lat.shape[0]
    if n_rows < 2:
        raise ValueError(f"Observation grid has {n_rows} row; the ground-track direction needs at least two.")
    row, col = _nearest_grid_pixel(grid, target_lat_deg, target_lon_deg)
    rows = (max(row - 1, 0), min(row + 1, n_rows - 1))
    lon_lat = np.array([[grid.lon[r, col], grid.lat[r, col], 0.0] for r in rows], dtype=float)
    if not np.all(np.isfinite(lon_lat)):
        raise ValueError(f"Pixels in rows {rows}, column {col} must have finite lat/lon, got {lon_lat[:, 1::-1]}.")
    p_from, p_to = geodetic_to_ecef(lon_lat, meters=False, degrees=True)
    if np.array_equal(p_from, p_to):
        raise ValueError(f"Pixels in rows {rows}, column {col} coincide; the ground-track direction is undefined.")
    return float(calc_azimuth(p_from, p_to, degrees=True)[0])
