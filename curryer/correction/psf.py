from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from scipy.interpolate import (
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from scipy.ndimage import map_coordinates
from scipy.signal import convolve2d, fftconvolve

from ..compute import constants
from ..compute.spatial import ecef_to_geodetic, geodetic_to_ecef
from .data_structures import (
    GeolocationConfig,
    ImageGrid,
    OpticalPSFEntry,
    ProjectedPSF,
    PSFGrid,
)

logger = logging.getLogger(__name__)


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


def convolve_psf_with_spacecraft_motion(
    psf: ProjectedPSF,
    composite_img: ImageGrid,
    config: GeolocationConfig,
) -> PSFGrid:
    """
    Apply spacecraft motion blur to projected PSF.

    Parameters
    ----------
    psf : ProjectedPSF
        Projected PSF on Earth's surface.
    composite_img : ImageGrid
        Composite image defining spacecraft motion direction.
    config : GeolocationConfig
        Configuration with PSF sampling parameters.

    Returns
    -------
    PSFGrid
        PSF convolved with spacecraft motion blur.
    """

    logger.debug("Convolve with SC - Init")
    lat_motion = np.mean(np.diff(composite_img.lat, axis=1))
    lon_motion = np.mean(np.diff(composite_img.lon, axis=1))

    theta_deg = np.degrees(np.arctan2(lat_motion, lon_motion))
    dist = np.hypot(lat_motion, lon_motion)

    psf_lon = psf.lon
    psf_lat = psf.lat

    psf_mid_lon = np.mean(psf_lon)
    psf_mid_lat = np.mean(psf_lat)

    rot_angle = np.deg2rad(-theta_deg)
    rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])

    logger.debug("Convolve with SC - Forward prep")
    coords = np.vstack((psf_lon.ravel() - psf_mid_lon, psf_lat.ravel() - psf_mid_lat))
    rotated = rot_mat @ coords
    psf_lon_rot = rotated[0].reshape(psf_lon.shape) + psf_mid_lon
    psf_lat_rot = rotated[1].reshape(psf_lat.shape) + psf_mid_lat

    dlat = config.psf_lat_sample_dist_deg
    dlon = config.psf_lon_sample_dist_deg

    x = np.arange(
        psf_lon_rot.min() - dist / 2.0,
        psf_lon_rot.max() + dist / 2.0 + dlon,
        dlon,
    )
    y = np.arange(psf_lat_rot.min(), psf_lat_rot.max() + dlat, dlat)
    X, Y = np.meshgrid(x, y)

    logger.debug("Convolve with SC - Nearest interp")
    nearest = NearestNDInterpolator(np.column_stack((psf_lon_rot.ravel(), psf_lat_rot.ravel())), psf.data.ravel())
    psf_interp = nearest(X, Y)
    psf_interp = np.nan_to_num(psf_interp, nan=0.0)

    logger.debug("Convolve with SC - Convolve 2d")
    num_steps = max(1, int(round(abs(dist / dlon))))
    kernel = np.ones((1, num_steps), dtype=float)
    psf_map = convolve2d(psf_interp, kernel, mode="same", boundary="fill", fillvalue=0.0)

    logger.debug("Convolve with SC - Linear interp")  # TODO: Orig impl was slow!!!
    # # Original implementation.
    # rot_angle_back = np.deg2rad(theta_deg)
    # rot_mat_back = np.array(
    #     [[np.cos(rot_angle_back), -np.sin(rot_angle_back)], [np.sin(rot_angle_back), np.cos(rot_angle_back)]]
    # )
    # coords_back = np.vstack((X.ravel() - psf_mid_lon, Y.ravel() - psf_mid_lat))
    # rotated_back = rot_mat_back @ coords_back
    # X_new = rotated_back[0] + psf_mid_lon
    # Y_new = rotated_back[1] + psf_mid_lat
    #
    # linear = LinearNDInterpolator(
    #     np.column_stack((X_new, Y_new)), psf_map.ravel(), fill_value=0.0
    # )
    # data = linear(X, Y)
    # data = np.nan_to_num(data, nan=0.0)

    # # New implementation (v1).
    # # Convert back to fractional indices on the uniform grid
    # y_coords = (Y_new - y[0]) / dlat
    # x_coords = (X_new - x[0]) / dlon
    #
    # # Bilinear interpolation with zero padding outside the grid
    # data = map_coordinates(
    #     psf_map,
    #     [y_coords.ravel(), x_coords.ravel()],
    #     order=1,
    #     mode="constant",
    #     cval=0.0,
    # ).reshape(Y.shape)

    # New implementation (v2).
    coords = np.vstack((X.ravel() - psf_mid_lon, Y.ravel() - psf_mid_lat))
    rotated = rot_mat @ coords  # rot_mat is the earlier -theta rotation
    X_rot = rotated[0] + psf_mid_lon
    Y_rot = rotated[1] + psf_mid_lat

    # Convert to fractional indices in the rotated grid
    x_idx = (X_rot - x[0]) / dlon
    y_idx = (Y_rot - y[0]) / dlat

    data = map_coordinates(
        psf_map,
        [y_idx, x_idx],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(X.shape)

    logger.debug("Convolve with SC - End")
    return PSFGrid(data=data, lat=y, lon=x)


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

    Parameters
    ----------
    psf : PSFGrid
        Point spread function on rectilinear grid.
    gcp : ImageGrid
        Ground control point reference defining target resolution.

    Returns
    -------
    PSFGrid
        Resampled PSF at GCP resolution.

    Raises
    ------
    ValueError
        If PSF is not on a rectilinear grid.
    """

    if psf.lat.ndim != 1 or psf.lon.ndim != 1:
        raise ValueError("PSF must be on a rectilinear grid before resampling.")

    dlon = float(np.abs(np.mean(np.diff(gcp.lon, axis=1))))
    dlat = float(np.abs(np.mean(np.diff(gcp.lat, axis=0))))

    x = np.arange(psf.lon.min(), psf.lon.max() + dlon, dlon)
    y = np.arange(psf.lat.min(), psf.lat.max() + dlat, dlat)
    Y_new, X_new = np.meshgrid(y, x, indexing="ij")

    interpolant = RegularGridInterpolator((psf.lat, psf.lon), psf.data, bounds_error=False, fill_value=0.0)
    data = interpolant(np.stack((Y_new.ravel(), X_new.ravel()), axis=-1)).reshape(Y_new.shape)

    return PSFGrid(data=data, lat=y, lon=x)


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
