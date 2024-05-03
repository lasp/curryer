"""SPICE Spatial computations.

@author: Brandon Stone
"""
import logging
import time

import numpy as np
import pandas as pd
import xarray as xr
from spiceypy.utils.exceptions import SpiceyError, NotFoundError

from . import constants, elevation
from .constants import SpatialQualityFlags as SQF
from .. import spicierpy, spicetime


logger = logging.getLogger(__name__)


def pixel_vectors(instrument):
    """Load the pixel or boresight vector(s) for a given instrument.

    Boresight vector is queried from the instrument kernel, but superseded by
    pixel vectors if they are available.

    Pixel vectors are queried from the SPICE kernel variable pool using the
    non-standard (but established) variables:
        INS{instrument_id}_PIXEL_COUNT
        INS{instrument_id}_PIXEL_VECTORS
    where the former is the pixel index and the latter is three doubles.

    Parameters
    ----------
    instrument : spicierpy.obj.Body or int or str
        Instrument to query details about.

    Returns
    -------
    int
        Number of boresight (1) or pixel vectors (1+).
    np.ndarray
        2D array of boresight (1x3) or pixel vectors (1+,3).

    """
    if not isinstance(instrument, spicierpy.obj.Body):
        instrument = spicierpy.obj.Body(instrument, frame=True)

    boresight_vector = spicierpy.ext.instrument_boresight(instrument)

    inst_var_base = f'INS{instrument.id}'
    try:
        count = spicierpy.gipool(f'{inst_var_base}_PIXEL_COUNT', 0, 1)
        count = int(count[0])  # Later calls will fail if a numpy int instead of native.

        vectors = spicierpy.gdpool(f'{inst_var_base}_PIXEL_VECTORS', 0, count * 3)
        vectors = vectors.reshape((count, 3))

    except NotFoundError:
        count = 1
        vectors = np.array([boresight_vector])

    logger.debug('Read [%d] pixel vectors for instrument [%s]', count, instrument)
    return count, vectors


def instrument_intersect_ellipsoid(ugps_times, instrument, correction=None, allow_nans=True, boresight_vector=None,
                                   geodetic=False, degrees=False):
    """Geolocate the boresight on the Earth's surface (WGS84 ellipsoid).
    """
    # Prepare the SPICE objects.
    if not isinstance(instrument, spicierpy.obj.Body):
        instrument = spicierpy.obj.Body(instrument, frame=True)

    # Prepare arguments for spice.
    observer_id = spicierpy.obj.Body('EARTH').id
    fixed_frame_name = 'ITRF93'  # ECEF (high-precision)
    target_id = instrument.id
    boresight_frame_name = instrument.frame.name

    et_times = spicetime.adapt(ugps_times, to='et')
    if correction is None:
        correction = 'NONE'
    nan_result = (np.full((3, 3), np.nan), np.full((3,), np.nan))

    # Optional multi-pixel support...
    if boresight_vector is None:
        pix_count, pix_vectors = pixel_vectors(instrument)
    else:
        pix_count = 1
        pix_vectors = np.array(boresight_vector)[None, ...]

    logger.debug('Calculating [%s x %s] Earth ellipsoid intercepts for [%s]', pix_count, len(et_times), instrument)

    # Query the rotation and position in fixed reference frame, but optionally
    # map SPICE errors to NaNs and quality flags.
    @spicierpy.ext.spice_error_to_val(
        err_value=nan_result, err_flag=SQF.from_spice_error, pass_flag=SQF.GOOD, disable=not allow_nans)
    def _query(sample_et):
        rot_to_fix = spicierpy.pxform(boresight_frame_name, fixed_frame_name, sample_et)
        position, _ = spicierpy.spkezp(
            target_id, sample_et, ref=fixed_frame_name, abcorr=correction, obs=observer_id
        )
        return rot_to_fix, position

    surf_points = np.full((et_times.size * pix_count, 3), np.nan)
    sc_positions = np.full((et_times.size * pix_count, 3), np.nan)
    quality_flags = np.zeros((et_times.size * pix_count,), dtype=np.int64)

    for ith, time in enumerate(et_times):
        # Query the boresight pointing and position in ECEF.
        (fixed_rotation, fixed_position), qf_val = _query(time)

        # Enough data was found to check individual pixels.
        if qf_val == SQF.GOOD:
            fixed_vectors = (fixed_rotation @ pix_vectors.T).T
            fixed_points = ray_intersect_ellipsoid(fixed_vectors, fixed_position, geodetic=geodetic, degrees=degrees)
            surf_points[ith * pix_count: (ith + 1) * pix_count, :] = fixed_points

            sc_positions[ith * pix_count: (ith + 1) * pix_count, :] = fixed_position

            qf_values = np.full((pix_count,), qf_val)
            qf_values[~np.isfinite(fixed_points).all(axis=1)] |= SQF.CALC_ELLIPS_NO_INTERSECT
            quality_flags[ith * pix_count: (ith + 1) * pix_count] = qf_values

        else:
            # Leave the surface and S/C data points as NaNs.
            qf_val |= SQF.CALC_ELLIPS_INSUFF_DATA
            quality_flags[ith * pix_count: (ith + 1) * pix_count] = qf_val

    if pix_count == 1:
        index = pd.Index(ugps_times, name='ugps')
    else:
        index = pd.MultiIndex.from_product([ugps_times, np.arange(pix_count) + 1], names=['ugps', 'pixel'])

    columns = ['lon', 'lat', 'alt'] if geodetic else ['x', 'y', 'z']
    surf_data = pd.DataFrame(surf_points, columns=columns, index=index)
    sc_data = pd.DataFrame(sc_positions, columns=['x', 'y', 'z'], index=index)
    qf_data = pd.Series(quality_flags, name='qf', index=index)

    surf_data.columns.name = f'Ellipsoid[{instrument.name}]@{fixed_frame_name}'
    sc_data.columns.name = f'Position[{instrument.name}]@{fixed_frame_name}'

    logger.info('Completed [%s x %s] Earth ellipsoid intercepts for [%s]', pix_count, len(et_times), instrument)
    return surf_data, sc_data, qf_data


def ray_intersect_ellipsoid(vector, position, geodetic=False, degrees=False, a=None, b=None, e2=None):
    # Source:
    #   https://modis.gsfc.nasa.gov/data/atbd/atbd_mod28_v3.pdf
    if a is b is e2 is None:
        a = constants.WGS84_SEMI_MAJOR_AXIS_KM
        b = constants.WGS84_SEMI_MINOR_AXIS_KM
        e2 = constants.WGS84_ECCENTRICITY2

    elif any(v is not None for v in (a, b, e2)):
        raise TypeError('Must specify `a`, `b`, and `e2`, or none of them!')

    pairwise = True
    if vector.ndim == 2 and position.ndim == 1:
        pairwise = False
    elif vector.ndim != position.ndim or vector.size != position.size:
        raise ValueError('`vector` and `position` must have the same dim and size!')

    given_1d = vector.ndim == 1
    if given_1d:
        vector = vector[None, ...]
        position = position[None, ...]

    if vector.shape[-1] != 3 or vector.shape[-1] != position.shape[-1]:
        raise ValueError('`vector` and `position` must have 3 values per point!')

    aab = np.array([a, a, b])
    if pairwise:
        uu_dot_pp = np.prod([vector / aab, position / aab], axis=0).sum(axis=1)[..., None]
        pp_norm2 = np.linalg.norm(position / aab, axis=1)[..., None] ** 2
    else:
        uu_dot_pp = ((vector / aab) * (position / aab)).sum(axis=1)[..., None]
        pp_norm2 = np.linalg.norm(position / aab)[..., None, None] ** 2
    uu_norm2 = np.linalg.norm(vector / aab, axis=1)[..., None] ** 2
    dist = ((-uu_dot_pp - np.sqrt(uu_dot_pp ** 2 - uu_norm2 * (pp_norm2 - 1))) / uu_norm2)

    # No intersect computes to nan, but inverse direction misses are negative.
    dist[dist < 0] = np.nan

    xyz = position + dist * vector
    if not geodetic:
        return xyz.ravel() if given_1d else xyz

    # Compute lon/lat. Special case where direct computation is allowed since
    # height is assumed to be zero.
    bad_xyz = np.isnan(xyz).any(axis=1)
    lla = np.stack([
        np.arctan(xyz[:, 1] / xyz[:, 0]),
        np.arctan((xyz[:, 2] / (1 - e2)) / np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)),
        np.where(bad_xyz, np.nan, 0.0)
    ], axis=1)

    # Handle edge cases where div-zero creates NaNs instead of 0.0.
    fix_lla = np.isnan(lla).any(axis=1)
    if fix_lla.any():
        good_fix = ~bad_xyz & fix_lla
        lla[good_fix, :] = np.nan_to_num(lla[good_fix, :], nan=0.0, posinf=0.0, neginf=0.0)

    if degrees:
        lla[:, :2] = np.rad2deg(lla[:, :2])
    return lla.ravel() if given_1d else lla


def ecef_to_geodetic(xyz, meters=False, degrees=True):
    # TODO: Any edge cases that need checking (e.g., small values that
    #   result in a NaN but should be zero or 90/180)???
    # Source:
    #   https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#The_application_of_Ferrari's_solution
    # Note:
    #   With 1k points, this is ~0.2 ms, vs. SPICE at ~20ms.

    given_1d = xyz.ndim == 1
    if given_1d:
        xyz = xyz[None, ...]

    a = constants.WGS84_SEMI_MAJOR_AXIS_KM
    b = constants.WGS84_SEMI_MINOR_AXIS_KM
    if meters:
        a *= 1e3
        b *= 1e3

    e2 = (a ** 2 - b ** 2) / a ** 2
    ep2 = (a ** 2 - b ** 2) / b ** 2

    p = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    ff = 54 * b ** 2 * xyz[:, 2] ** 2
    gg = p ** 2 + (1 - e2) * xyz[:, 2] ** 2 - e2 * (a ** 2 - b ** 2)
    c = (e2 * e2) * ff * p ** 2 / gg ** 3
    s = (1 + c + np.sqrt(c ** 2 + 2 * c)) ** (1 / 3)
    k = s + 1 + 1 / s
    pp = ff / (3 * k ** 2 * gg ** 2)
    qq = np.sqrt(1 + 2 * (e2 * e2) * pp)

    r0 = (-1 * pp * e2 * p) / (1 + qq) + np.sqrt(
        (1 / 2) * a ** 2 * (1 + 1 / qq)
        - (pp * (1 - e2) * xyz[:, 2] ** 2) / (qq * (1 + qq))
        - (1 / 2) * pp * p ** 2
    )
    uu = np.sqrt((p - e2 * r0) ** 2 + xyz[:, 2] ** 2)
    vv = np.sqrt((p - e2 * r0) ** 2 + (1 - e2) * xyz[:, 2] ** 2)
    z0 = b ** 2 * xyz[:, 2] / (a * vv)

    h = uu * (1 - b ** 2 / (a * vv))
    phi = np.arctan((xyz[:, 2] + ep2 * z0) / p)
    lam = np.arctan2(xyz[:, 1], xyz[:, 0])

    lla = np.stack([lam, phi, h], axis=1)

    if degrees:
        lla[:, :2] = np.rad2deg(lla[:, :2])

    return lla.ravel() if given_1d else lla


def geodetic_to_ecef(lon_lat_alt, meters=False, degrees=True):
    # Source:
    #   https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    given_1d = lon_lat_alt.ndim == 1
    if given_1d:
        lon_lat_alt = lon_lat_alt[None, ...]

    if degrees:
        lon_lat_alt = lon_lat_alt.copy()
        lon_lat_alt[:, :2] = np.deg2rad(lon_lat_alt[:, :2])

    a = constants.WGS84_SEMI_MAJOR_AXIS_KM
    b = constants.WGS84_SEMI_MINOR_AXIS_KM
    if meters:
        a *= 1e3
        b *= 1e3

    e2 = 1 - b ** 2 / a ** 2
    f = 1 - b / a

    nn = a / np.sqrt(1 - e2 * np.sin(lon_lat_alt[:, 1]) ** 2)

    x = (nn + lon_lat_alt[:, 2]) * np.cos(lon_lat_alt[:, 1]) * np.cos(lon_lat_alt[:, 0])
    y = (nn + lon_lat_alt[:, 2]) * np.cos(lon_lat_alt[:, 1]) * np.sin(lon_lat_alt[:, 0])
    z = ((1 - f) ** 2 * nn + lon_lat_alt[:, 2]) * np.sin(lon_lat_alt[:, 1])

    xyz = np.stack([x, y, z], axis=1)
    return xyz.ravel() if given_1d else xyz


def terrain_correct(elev: elevation.Elevation, ec_srf_pos, ec_sat_pos, local_minmax=None):
    if elev.meters or elev.degrees:
        raise ValueError('Elevation instance must be configured for KM (meters=False) and RADIANS (degrees=False)')

    if ec_sat_pos.ndim != ec_srf_pos.ndim or ec_sat_pos.size != ec_srf_pos.size:
        raise ValueError('`ec_sat_pos` and `ec_srf_pos` must have the same dim and size!')

    given_1d = ec_sat_pos.ndim == 1
    if given_1d:
        ec_sat_pos = ec_sat_pos[None, ...]
        ec_srf_pos = ec_srf_pos[None, ...]
    npts = ec_sat_pos.shape[0]

    if ec_sat_pos.shape[1] != 3 or ec_sat_pos.shape[1] != ec_srf_pos.shape[1]:
        raise ValueError('`ec_sat_pos` and `ec_srf_pos` must have 3 values per point!')

    qf_values = np.full((npts,), SQF.GOOD)

    # 0) Geodetic intersection on the ellipsoid.
    gd_xyz = ecef_to_geodetic(ec_srf_pos, degrees=False)  # rad

    # 0) Determine local min/max elevation.
    if local_minmax is None:
        alt_srf_min, alt_srf_max = elev.local_minmax(gd_xyz[0, 0], gd_xyz[0, 1], pad=10)  # km
    else:
        alt_srf_min, alt_srf_max = local_minmax

    if logger.isEnabledFor(logging.DEBUG):
        gd_sat_pos = ecef_to_geodetic(ec_sat_pos, degrees=True)
        sc_minmax_lla = []
        pt_minmax_lla = []
        for ith in range(3):
            sc_minmax_lla.extend([gd_sat_pos[:, ith].min(), gd_sat_pos[:, ith].max()])
            vals = [gd_xyz[:, ith].min(), gd_xyz[:, ith].max()]
            if ith < 2:
                vals = np.rad2deg(vals)
            pt_minmax_lla.extend(vals)

        logger.debug('Terrain intersection init-S/C: lon=[%11.6f, %11.6f], lat=[%10.6f, %10.6f], alt=[%7.3f, %7.3f]'
                     ' (min, max)', *sc_minmax_lla)
        logger.debug('Terrain intersection init-Pnt: lon=[%11.6f, %11.6f], lat=[%10.6f, %10.6f], alt=[%7.3f, %7.3f],'
                     ' dem=[%7.3f, %7.3f] (min, max)',
                     *pt_minmax_lla, alt_srf_min, alt_srf_max)

    # 1) Local ellipsoid normal unit vector from geodetic lat/lon.
    ellips_norm_vec = np.array([
        np.cos(gd_xyz[:, 1]) * np.cos(gd_xyz[:, 0]),
        np.cos(gd_xyz[:, 1]) * np.sin(gd_xyz[:, 0]),
        np.sin(gd_xyz[:, 1])
    ]).T

    # 2) ECR unit vector from ground to satellite.
    ec_sat_vec = ec_srf_pos - ec_sat_pos
    ec_sat_vec = (ec_sat_vec / np.linalg.norm(ec_sat_vec, axis=1)[..., None]) * -1

    # 3) Local vertical component of the vector.
    cos_sat = np.prod([ec_sat_vec, ellips_norm_vec], axis=0).sum(axis=1)[..., None]

    # Skip (or exit if all) within 5-deg of horizontal! Too off nadir.
    v_zenith_ang = np.arccos(cos_sat)  # rad
    idx_ang_valid = (v_zenith_ang <= np.deg2rad(85)).ravel()
    n_ang_valid = idx_ang_valid.sum()
    if n_ang_valid != npts:
        logger.info('Unable to perform terrain correction on [%d/%d] points! Zenith angle too high!',
                    npts - n_ang_valid, npts)
        qf_values[~idx_ang_valid] |= SQF.CALC_TERRAIN_EXTREME_ZENITH

        if n_ang_valid == 0:
            gd_final = np.array([np.nan, np.nan, np.nan])
            if not given_1d:
                gd_final = np.tile(gd_final, (npts, 1))
            else:
                qf_values = qf_values.ravel()
            return gd_final, qf_values

    # 4) Distance along satellite vector for MAX local height.
    dist_srf_max = alt_srf_max / cos_sat

    # 5) ECR point along the viewing vector for MAX height.
    ec_max_pos = ec_srf_pos + dist_srf_max * ec_sat_vec

    # TODO: Unused? Meant to be a limit?
    # # 6) Distance along satellite vector for MIN local height.
    # dist_srf_min = alt_srf_min / cos_sat
    #
    # # 7) ECR point along the viewing vector for MIN height.
    # ec_min_pos = ec_srf_pos + dist_srf_min * ec_sat_vec

    # 8) Geodetic position for MAX.
    gd_max_xyz = ecef_to_geodetic(ec_max_pos, degrees=False)

    # 9) Initialize to MAX position on the ellipsoid.
    # gd_cur_lon, gd_cur_lat, gd_cur_alt = gd_max_xyz[:, 0], gd_max_xyz[:, 1], np.zeros(npts)
    gd_cur_xyz = gd_max_xyz.copy()  # TODO: Unused after, not needed? Check others!
    gd_cur_xyz[:, 2] = np.zeros(npts)

    # 10) Terrain intersection.
    gd_cur_h = elev.query(gd_cur_xyz[:, 0], gd_cur_xyz[:, 1])
    gd_prev_h = gd_cur_h.copy()

    # Initialize to the max local height.
    gd_cur_alt = np.full(npts, alt_srf_max)
    gd_prev_alt = gd_cur_alt.copy()

    # Distance step size. Must consider DEM resolution!
    d_step = 0.5  # km
    delta_dist = d_step / np.sin(v_zenith_ang)
    delta_dist = np.min([delta_dist, dist_srf_max], axis=0)  # Added due to issues around pure-nadir.
    dist_cur = dist_srf_max.copy()

    # Iterate until the point is below the elevation.
    idx_cont = idx_ang_valid & (gd_cur_h < gd_cur_alt)
    iloop = 0
    while idx_cont.any():
        iloop += 1

        # Step along the vector towards the surface, update location.
        dist_cur[idx_cont] -= delta_dist[idx_cont]
        ec_cur_pos_subset = ec_srf_pos[idx_cont, :] + dist_cur[idx_cont] * ec_sat_vec[idx_cont, :]

        # Convert to geodetic lon, lat and altitude (above ellipsoid).
        gd_cur_pos_subset = ecef_to_geodetic(ec_cur_pos_subset, degrees=False)
        gd_prev_alt[idx_cont] = gd_cur_alt[idx_cont]
        gd_cur_alt[idx_cont] = gd_cur_pos_subset[:, 2]

        # Look up the ellipsoidal height (surface) for this position.
        gd_prev_h = gd_cur_h
        gd_cur_h[idx_cont] = elev.query(gd_cur_pos_subset[:, 0], gd_cur_pos_subset[:, 1])  # alt=0

        if logger.isEnabledFor(logging.DEBUG):
            pt_minmax_ll = np.rad2deg([gd_cur_pos_subset[:, 0].min(), gd_cur_pos_subset[:, 0].max(),
                                       gd_cur_pos_subset[:, 1].min(), gd_cur_pos_subset[:, 1].max()])
            logger.debug('Terrain intersection iter=[%d]: lon=[%11.6f, %11.6f], lat=[%10.6f, %10.6f],'
                         ' alt=[%7.3f, %7.3f], dem=[%7.3f, %7.3f] (min, max), [%d/%d] remain',
                         iloop, *pt_minmax_ll, gd_cur_alt[idx_cont].min(), gd_cur_alt[idx_cont].max(),
                         gd_cur_h[idx_cont].min(), gd_cur_h[idx_cont].max(), idx_cont.sum(), idx_cont.size)

        # Update if we've stepped below the surface (ends loop after all points).
        idx_cont = idx_ang_valid & (gd_cur_h < gd_cur_alt)

        # Fail-safe to prevent infinite looping.
        if iloop > 1000:
            logger.warning('Failed to find a terrain intersection after [%d] loops for [%d/%d] points!',
                           iloop, idx_cont.sum(), idx_cont.size)
            qf_values[idx_cont] |= SQF.CALC_TERRAIN_MAX_ITER

            gd_final = np.array([np.nan, np.nan, np.nan])
            if not given_1d:
                gd_final = np.tile(gd_final, (npts, 1))
            else:
                qf_values = qf_values.ravel()
            return gd_final, qf_values

    # 11) Precise terrain intersection from last two iterations.
    delta_gd_cur_alt = gd_cur_alt - gd_prev_alt
    alpha = (gd_prev_alt - gd_prev_h) / (gd_cur_h - gd_prev_h - delta_gd_cur_alt)

    # Final DEM-based height above the ellipsoid.
    gd_final_pos = np.full((npts, 3), np.nan)
    gd_final_pos[idx_ang_valid, 2] = (alpha * gd_cur_h + (1 - alpha) * gd_prev_h)[idx_ang_valid]

    # 12) Final geodetic coordinates. Note that the DEM height is used over the
    # computed geodetic altitude (discarded).
    dist_final = dist_cur + (1 - alpha[..., None]) * delta_dist
    ec_final_pos = ec_srf_pos + dist_final * ec_sat_vec
    gd_final_pos[idx_ang_valid, :2] = ecef_to_geodetic(ec_final_pos[idx_ang_valid, :], degrees=True)[:, :2]

    # 13) Set final quality flag if any nans (shouldn't happen).
    idx_out_valid = np.isfinite(gd_final_pos).all(axis=1)
    qf_values[~idx_out_valid & idx_ang_valid] |= SQF.CALC_TERRAIN_NOT_FINITE

    if logger.isEnabledFor(logging.DEBUG):
        if idx_out_valid.sum():
            pt_minmax_lla = []
            for ith in range(3):
                pt_minmax_lla.extend([gd_final_pos[idx_out_valid, ith].min(), gd_final_pos[idx_out_valid, ith].max()])
            pt_ellps_ht = ecef_to_geodetic(ec_final_pos[idx_ang_valid, :])[:, 2]
            pt_ellps_ht = [pt_ellps_ht.min(), pt_ellps_ht.max()]
        else:
            pt_minmax_lla = [np.nan for _ in range(6)]
            pt_ellps_ht = [np.nan for _ in range(2)]

        logger.debug('Terrain intersection done-Pnt: lon=[%11.6f, %11.6f], lat=[%10.6f, %10.6f], alt=[%7.3f, %7.3f],'
                     ' dem=[%7.3f, %7.3f] (min, max), after [%d] iterations',
                     *pt_minmax_lla[:4], *pt_ellps_ht, *pt_minmax_lla[4:], iloop)

    return (gd_final_pos.ravel(), qf_values.ravel()) if given_1d else (gd_final_pos, qf_values)


def calc_azimuth(obs_position, trg_position, degrees=False):
    if obs_position.ndim == 2 and trg_position.ndim == 1:
        pass  # Many-to-one (non-pairwise) is supported.
    elif obs_position.ndim != trg_position.ndim or obs_position.size != trg_position.size:
        raise ValueError('`obs_position` and `trg_position` must have the same dim and size!')

    given_1d = obs_position.ndim == 1
    if given_1d:
        obs_position = obs_position[None, ...]
        trg_position = trg_position[None, ...]

    if obs_position.shape[-1] != 3 or obs_position.shape[-1] != trg_position.shape[-1]:
        raise ValueError('`obs_position` and `trg_position` must have 3 values per point!')

    xy_ang = np.arctan2(trg_position[..., 1], trg_position[..., 0]) - np.arctan2(obs_position[:, 1], obs_position[:, 0])
    xy_dist = np.sin(xy_ang) * np.sqrt(trg_position[..., 0] ** 2 + trg_position[..., 1] ** 2)
    az_ang = np.arctan2(xy_dist, trg_position[..., 2] - obs_position[..., 2])
    az_ang[az_ang < 0] += np.pi * 2

    if degrees:
        az_ang = np.rad2deg(az_ang)
    return az_ang.ravel() if given_1d else az_ang


def calc_zenith(obs_position, trg_position, degrees=False):
    pairwise = True
    if obs_position.ndim == 2 and trg_position.ndim == 1:
        pairwise = False
    elif obs_position.ndim != trg_position.ndim or obs_position.size != trg_position.size:
        raise ValueError('`obs_position` and `trg_position` must have the same dim and size!')

    given_1d = obs_position.ndim == 1
    if given_1d:
        obs_position = obs_position[None, ...]
        trg_position = trg_position[None, ...]

    if obs_position.shape[-1] != 3 or obs_position.shape[-1] != trg_position.shape[-1]:
        raise ValueError('`obs_position` and `trg_position` must have 3 values per point!')

    trg_position = trg_position - obs_position
    trg_position /= np.linalg.norm(trg_position, axis=1)[..., None]
    obs_position = obs_position / np.linalg.norm(obs_position, axis=1)[..., None]

    if pairwise:
        zenith_ang = np.prod([obs_position, trg_position], axis=0)
    else:
        zenith_ang = obs_position * trg_position
    zenith_ang = np.arccos(zenith_ang.sum(axis=1))

    if degrees:
        zenith_ang = np.rad2deg(zenith_ang)
    return zenith_ang.ravel() if given_1d else zenith_ang


def surface_angles(surface_positions, target_positions=None, target_obj=None, degrees=False, allow_nans=False):
    if not isinstance(surface_positions, pd.DataFrame):
        raise TypeError(f'`surface_positions` must be a DataFrame, not: {type(surface_positions)}')
    if target_positions is not None and not isinstance(target_positions, pd.DataFrame):
        raise TypeError(f'`target_positions` must be a DataFrame, not: {type(target_positions)}')
    if target_positions is target_obj is None or (target_positions is not None and target_obj is not None):
        raise ValueError('Must specify either `target_positions` or `target_obj`, and not both')

    pairwise = True
    if target_obj is not None:
        ugps_times = surface_positions.index
        if isinstance(ugps_times, pd.MultiIndex):
            pairwise = False
            ugps_times = ugps_times.unique(level=0)

        target_positions = spicierpy.ext.query_ephemeris(
            ugps_times, target=target_obj, observer='EARTH', ref_frame='ITRF93', allow_nans=allow_nans
        )

    elif target_positions.shape[0] != surface_positions.shape[0]:
        pairwise = False

    # Use the same target position for every pixel at each time.
    if not pairwise:
        target_positions = target_positions.reindex(surface_positions.index, level=0)

    azimuth_ang = calc_azimuth(surface_positions.values, target_positions.values, degrees=degrees)
    zenith_ang = calc_zenith(surface_positions.values, target_positions.values, degrees=degrees)

    angles = pd.DataFrame({'azimuth': azimuth_ang, 'zenith': zenith_ang},
                          index=surface_positions.index)
    angles.columns.name = target_positions.columns.name
    return angles


class Geolocate:

    def __init__(self, instrument, dem_data_dir=None):
        self.instrument = spicierpy.obj.Body(instrument, frame=True)  # E.g. "CPRS_HYSICS"
        self.sc_state_frame = spicierpy.obj.Frame('J2000')  # TODO: Or ECEF (ITRF93)?
        self.elevation = elevation.Elevation(dem_data_dir, meters=False, degrees=False)
        self._step_time = None

    def _log_step(self, msg, qf_ds=None):
        if msg is not None:
            t0 = self._step_time
            t1 = time.time()
            if t0 is None:
                t0 = t1

            extra = ''
            if qf_ds is not None:
                extra = '\n\tQuality Flags Summary (count, hex, label):'
                for val, cnt in qf_ds.value_counts().sort_index().items():
                    extra += f'\n\t{cnt:7d}: 0x{val:<6X} - {str(SQF(val)).split(".", 1)[-1]}'

            logger.info('Geolocation step [%s] took [%.3f] sec%s', msg, t1 - t0, extra)

        self._step_time = time.time()

    def __call__(self, ugps_times):
        logger.info('Geolocation starting processing of [%d] times for instrument=[%s]',
                    len(ugps_times), self.instrument)

        # Sanity check that kernels are loaded.
        try:
            _ = spicierpy.obj.Body(self.instrument, frame=True)
        except SpiceyError:
            logger.exception('Suppressing exception as cause is likely kernels were not loaded.')
            raise ValueError(f'Unable to look up the frame for instrument=[{self.instrument}].'
                             ' The SPICE kernels were likely not loaded!')

        t0 = time.time()
        self._log_step(None)  # Start sub-timer.

        # Note that `sc_xyz_df` is in a different frame than `sc_state_df`.
        ellips_lla_df, sc_xyz_df, ellips_qf_ds = self.intersect_ellipsoid(ugps_times)
        self._log_step('ellipsoid intersect', ellips_qf_ds)

        terrain_lla_df, terrain_qf_ds = self.correct_terrain(ellips_lla_df, sc_xyz_df, ellips_qf_ds)
        self._log_step('terrain intersect', terrain_qf_ds)

        solar_angles_df, view_angles_df, sc_state_df, ancil_qf_ds = self.calc_ancillary(terrain_lla_df, sc_xyz_df)
        self._log_step('ancillary fields', ancil_qf_ds)

        # Combine ancillary QFs with the rest of them.
        all_qfs_ds = terrain_qf_ds | ancil_qf_ds

        # Convert KM to meters.
        sc_state_df.loc[:, ['x', 'y', 'z', 'vx', 'vy', 'vz']] *= 1e3
        terrain_lla_df.loc[:, 'alt'] *= 1e3
        ellips_lla_df.loc[:, 'alt'] *= 1e3

        # Unstack time x pixel.
        pixel_ids = ellips_lla_df.index.unique(level=1)
        terrain_lla_df = terrain_lla_df.unstack(level=1)
        ellips_lla_df = ellips_lla_df.unstack(level=1)
        solar_angles_df = solar_angles_df.unstack(level=1)
        view_angles_df = view_angles_df.unstack(level=1)
        all_qfs_ds = all_qfs_ds.unstack(level=1)

        dataset = xr.Dataset(
            {
                'attitude': (['frame', 'euclidian_dim'], sc_state_df[['ex', 'ey', 'ez']].values),
                'position': (['frame', 'euclidian_dim'], sc_state_df[['x', 'y', 'z']].values),
                'velocity': (['frame', 'euclidian_dim'], sc_state_df[['vx', 'vy', 'vz']].values),
                # TODO: Now they want both ellipsoid and terrain?
                'altitude_ellipsoidal': (['frame', 'spatial_pixel'], ellips_lla_df['alt'].values),
                'latitude_ellipsoidal': (['frame', 'spatial_pixel'], ellips_lla_df['lat'].values),
                'longitude_ellipsoidal': (['frame', 'spatial_pixel'], ellips_lla_df['lon'].values),
                # TODO: Different than ellips height for terrain corr'd point!
                'altitude': (['frame', 'spatial_pixel'], terrain_lla_df['alt'].values),
                'latitude': (['frame', 'spatial_pixel'], terrain_lla_df['lat'].values),
                'longitude': (['frame', 'spatial_pixel'], terrain_lla_df['lon'].values),
                # TODO: Drop per pixel dimension? Waste of space!
                'solar_azimuth': (['frame', 'spatial_pixel'], solar_angles_df['azimuth'].values),
                'solar_zenith': (['frame', 'spatial_pixel'], solar_angles_df['zenith'].values),
                'view_azimuth': (['frame', 'spatial_pixel'], view_angles_df['azimuth'].values),
                'view_zenith': (['frame', 'spatial_pixel'], view_angles_df['zenith'].values),
                'quality_flags': (['frame', 'spatial_pixel'], all_qfs_ds.values),
            },
            coords={
                'euclidian_dim': ('euclidian_dim', ['x', 'y', 'z']),
                'frame': ('frame', ugps_times / 1e6),
                'spatial_pixel': ('spatial_pixel', pixel_ids.values),
                'spectral_pixel': ('spectral_pixel', []),
            },
            attrs={
                'instrument': self.instrument.name,  # TODO: Use expected attrs!
                'state_frame': self.sc_state_frame.name,  # TODO: Use expected attrs!
            }
        )

        self._log_step('create product', all_qfs_ds.stack())
        logger.info('Geolocation completed processing of [%d] times in [%.3f] sec:\n%s',
                    len(ugps_times), time.time() - t0, dataset)
        return dataset

    def intersect_ellipsoid(self, ugps_times):
        ellips_lla_df, sc_xyz_df, ellips_qf_ds = instrument_intersect_ellipsoid(
            ugps_times, self.instrument, geodetic=True, degrees=True)
        return ellips_lla_df, sc_xyz_df, ellips_qf_ds

    def correct_terrain(self, ellips_lla_df, sc_xyz_df, ellips_qf_ds, pad_degrees=1.0):
        # TODO: Assert shapes!!!

        is_valid = (ellips_qf_ds == SQF.GOOD).values
        data_arr = np.full((ellips_lla_df.shape[0], 3), np.nan)
        qf_arr = ellips_qf_ds.values.copy()

        if is_valid.any():
            # Pre-compute regional DEM for faster elevation look-ups.
            minmax_lonlat = [ellips_lla_df['lon'].min() - pad_degrees,
                             ellips_lla_df['lon'].max() + pad_degrees,
                             ellips_lla_df['lat'].min() - pad_degrees,
                             ellips_lla_df['lat'].max() + pad_degrees]
            elev_region = self.elevation.local_region(*np.deg2rad(minmax_lonlat))

            ellips_xyz_arr_sub = geodetic_to_ecef(ellips_lla_df.values[is_valid], degrees=True)
            terrain_lla_arr_sub, terrain_qf_arr_sub = terrain_correct(
                elev=elev_region, ec_srf_pos=ellips_xyz_arr_sub, ec_sat_pos=sc_xyz_df.values[is_valid],
            )
            # TODO: Also compute the ellipsoidal height?

            data_arr[is_valid, :] = terrain_lla_arr_sub
            qf_arr[is_valid] |= terrain_qf_arr_sub

        terrain_lla_df = pd.DataFrame(data_arr, columns=['lon', 'lat', 'alt'], index=ellips_lla_df.index)
        terrain_qf_ds = pd.Series(qf_arr, name='qf', index=ellips_lla_df.index)

        terrain_lla_df.columns.name = f'Terrain[{self.instrument.name}]@ITRF93'
        return terrain_lla_df, terrain_qf_ds

    def calc_ancillary(self, terrain_lla_df, sc_xyz_df):
        terrain_xyz_arr = geodetic_to_ecef(terrain_lla_df.values, degrees=True)
        terrain_xyz_df = pd.DataFrame(terrain_xyz_arr, columns=['x', 'y', 'z'], index=terrain_lla_df.index)

        solar_angles_df = surface_angles(terrain_xyz_df, target_obj='SUN', degrees=True)
        view_angles_df = surface_angles(terrain_xyz_df, target_positions=sc_xyz_df, degrees=True)

        ancil_qf_arr = np.zeros(terrain_lla_df.shape[0], dtype=np.int64)
        ancil_qf_ds = pd.Series(ancil_qf_arr, name='qf', index=terrain_lla_df.index)

        # Query the S/C attitude, position, and velocity, potentially in a
        # different reference frame than was provided.
        boresight_frame_name = self.instrument.frame.name  # TODO: Do they want instr or pure S/C att?
        out_frame_name = self.sc_state_frame.name
        observer_id = spicierpy.obj.Body('EARTH').id
        target_id = self.instrument.id
        nan_result = (np.full((3,), np.nan), np.full((6,), np.nan))

        @spicierpy.ext.spice_error_to_val(
            err_value=nan_result, err_flag=SQF.from_spice_error, pass_flag=SQF.GOOD)
        def _query(sample_et):
            rotation = spicierpy.pxform(boresight_frame_name, out_frame_name, sample_et)
            rotation = spicierpy.m2eul(rotation, 1, 2, 3)  # TODO: Check order!
            # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/m2eul_c.html
            position_velocity, _ = spicierpy.spkezr(
                target_id, sample_et, ref=out_frame_name, abcorr='NONE', obs=observer_id)
            return rotation, position_velocity

        ugps_times = terrain_lla_df.index.unique(level=0)
        sc_state_arr = np.empty((ugps_times.size, 9))

        for ith, et_time in enumerate(spicetime.adapt(ugps_times, 'ugps', 'et')):
            (sc_rot, sc_posvel), qf_val = _query(et_time)
            sc_state_arr[ith, :6] = sc_posvel
            sc_state_arr[ith, 6:] = sc_rot
            if qf_val != SQF.GOOD:
                # Dev note: Can't bitwise "or" b/c slice converts it to a float.
                ancil_qf_ds.loc[(ugps_times[ith], slice(None, None))] = SQF.CALC_ANCIL_INSUFF_DATA.value

        sc_state_arr[:, 6:] = np.rad2deg(sc_state_arr[:, 6:])
        sc_state_df = pd.DataFrame(
            sc_state_arr, columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'ex', 'ey', 'ez'], index=ugps_times
        )
        sc_state_df.columns.name = f'State[{self.instrument.name}]@{out_frame_name}'

        # Dev note: Important to do this check last, since above logic requires
        # direct assignment instead of OR'ing.
        angles_invalid = (solar_angles_df.isna().any(axis=1) | view_angles_df.isna().any(axis=1))
        ancil_qf_ds.loc[angles_invalid] |= SQF.CALC_ANCIL_NOT_FINITE

        return solar_angles_df, view_angles_df, sc_state_df, ancil_qf_ds


# =============================================================================
# Obsolete implementations kept for reference / validation.
# -----------------------------------------------------------------------------
# Replaced with vectorized impl. Kept for independent validation.
def spice_angles(ugps_times, surface_positions, target_obj, degrees=False):
    """Compute azimuth and zenith angles using SPICE.
    """
    et_times = spicetime.adapt(ugps_times, 'ugps', 'et')

    if len(et_times) == 1:
        et_times = [et_times]
        surface_positions = [surface_positions]

    azimuth_ang = []
    elevation_ang = []
    for ith in range(et_times.size):
        # ( r, az, el, dr/dt, daz/dt, del/dt )
        output, lt = spicierpy.azlcpo(
            et=et_times[ith], abcorr='NONE', method='ELLIPSOID',
            target=target_obj, azccw=False, elplsz=True,
            obspos=surface_positions[ith],
            obsref='ITRF93', obsctr='EARTH',
        )
        azimuth_ang.append(output[1])
        elevation_ang.append(output[2])

    azimuth_ang = np.array(azimuth_ang)
    zenith_ang = np.pi / 2 - np.array(elevation_ang)

    if degrees:
        azimuth_ang = np.rad2deg(azimuth_ang)
        zenith_ang = np.rad2deg(zenith_ang)

    return azimuth_ang, zenith_ang


# Replaced with vectorized impl, which is MUCH faster.
def terrain_correct_single(elev: elevation.Elevation, ec_srf_pos, ec_sat_pos, local_minmax=None):
    """Apply terrain correction to a single point.
    """
    if elev.meters or elev.degrees:
        raise ValueError('Elevation instance must be configured for KM (meters=False) and RADIANS (degrees=False)')

    # 0) Geodetic intersection on the ellipsoid.
    gd_lon, gd_lat, gd_alt = spicierpy.recgeo(
        rectan=ec_srf_pos,
        re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
        f=constants.WGS84_INVERSE_FLATTENING,
    )  # Radians.

    # 0) Determine local min/max elevation.
    if local_minmax is None:
        alt_srf_min, alt_srf_max = elev.local_minmax(gd_lon, gd_lat, pad=10)  # km
    else:
        alt_srf_min, alt_srf_max = local_minmax

    if logger.isEnabledFor(5):
        gd_sat_pos = spicierpy.recgeo(
            rectan=ec_sat_pos,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )
        logger.log(5, 'Terrain intersection init-S/C: lon=[%11.6f], lat=[%10.6f], alt=[%5.3f]',
                   np.rad2deg(gd_sat_pos[0]), np.rad2deg(gd_sat_pos[1]), gd_sat_pos[2])
        logger.log(5, 'Terrain intersection init-Pnt: lon=[%11.6f], lat=[%10.6f], alt=[%5.3f], dem=[%5.3f]',
                   np.rad2deg(gd_lon), np.rad2deg(gd_lat), gd_alt, elev.query(gd_lon, gd_lat))

    # 1) Local ellipsoid normal unit vector from geodetic lat/lon.
    ellips_norm_vec = np.array([
        np.cos(gd_lat) * np.cos(gd_lon),
        np.cos(gd_lat) * np.sin(gd_lon),
        np.sin(gd_lat)
    ])

    # 2) ECR unit vector from ground to satellite.
    ec_sat_vec = ec_srf_pos - ec_sat_pos
    ec_sat_vec = (ec_sat_vec / np.linalg.norm(ec_sat_vec)) * -1

    # 3) Local vertical component of the vector.
    cos_sat = ec_sat_vec @ ellips_norm_vec

    # Exit early if within 5-deg of horizontal! Too off nadir.
    v_zenith_ang = np.arccos(cos_sat)  # rad
    if v_zenith_ang > np.deg2rad(85):
        logger.info('Unable to perform terrain correction! Zenith angle [%s] (degrees) too high!',
                    np.rad2deg(v_zenith_ang))
        return np.array([np.nan, np.nan, np.nan])

    # 4) Distance along satellite vector for MAX local height.
    dist_srf_max = alt_srf_max / cos_sat

    # 5) ECR point along the viewing vector for MAX height.
    ec_max_pos = ec_srf_pos + dist_srf_max * ec_sat_vec

    # TODO: Unused? Meant to be a limit?
    # # 6) Distance along satellite vector for MIN local height.
    # dist_srf_min = alt_srf_min / cos_sat
    #
    # # 7) ECR point along the viewing vector for MIN height.
    # ec_min_pos = ec_srf_pos + dist_srf_min * ec_sat_vec

    # 8) Geodetic position for MAX.
    gd_max_lon, gd_max_lat, gd_max_alt = spicierpy.recgeo(
        rectan=ec_max_pos,
        re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
        f=constants.WGS84_INVERSE_FLATTENING,
    )
    assert 0.9 * alt_srf_max < gd_max_alt < alt_srf_max * 1.1  # TODO: How roughly?

    # 9) Initialize to MAX position on the ellipsoid.
    gd_cur_lon, gd_cur_lat, gd_cur_alt = gd_max_lon, gd_max_lat, 0.0
    assert gd_cur_alt == 0.0

    # 10) Terrain intersection.
    gd_cur_h = elev.query(gd_cur_lon, gd_cur_lat)
    gd_prev_h = gd_cur_h

    gd_cur_alt = alt_srf_max
    gd_prev_alt = gd_cur_alt

    # Distance step size. Must consider DEM resolution!
    d_step = 0.5  # km
    delta_dist = d_step / np.sin(v_zenith_ang)
    delta_dist = np.min([delta_dist, dist_srf_max])  # Added due to issues around pure-nadir.
    dist_cur = dist_srf_max

    # Iterate until the point is below the elevation.
    i = 0
    while gd_cur_h < gd_cur_alt:
        i += 1
        dist_cur -= delta_dist
        ec_cur_pos = ec_srf_pos + dist_cur * ec_sat_vec

        gd_cur_pos = spicierpy.recgeo(
            rectan=ec_cur_pos,
            re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
            f=constants.WGS84_INVERSE_FLATTENING,
        )
        gd_cur_lon, gd_cur_lat, gd_cur_alt = gd_cur_pos  # Key that `gd_cur_alt` is updated!

        gd_prev_h = gd_cur_h
        gd_cur_h = elev.query(gd_cur_lon, gd_cur_lat)  # alt=0

        if logger.isEnabledFor(5):
            logger.log(5, 'Terrain intersection iter=[%d]: lon=[%11.6f], lat=[%10.6f], alt=[%5.3f], dem=[%5.3f]',
                       i, np.rad2deg(gd_cur_lon), np.rad2deg(gd_cur_lat), gd_cur_alt, gd_cur_h)

        # Fail-safe to prevent infinite looping.
        if i > 1000:
            logger.warning('Failed to find a terrain intersection after [%d] loops! Spacecraft=[%s], surface=[%s]',
                           i, ec_sat_pos, ec_srf_pos)
            return np.array([np.nan, np.nan, np.nan])

    # 11) Precise terrain intersection from last two iterations.
    delta_gd_cur_alt = (gd_cur_alt - gd_prev_alt)
    alpha = (gd_prev_alt - gd_prev_h) / (gd_cur_h - gd_prev_h - delta_gd_cur_alt)
    # one_alpha = (gd_cur_h - gd_cur_alt) / (gd_cur_h - gd_prev_h - delta_gd_cur_alt)
    # NOTE: Why is 1-alpha is a separate formula?

    # gd_final_h1 = alpha * gd_cur_h + (1 - alpha) * gd_prev_h
    # gd_final_h2 = alpha * gd_cur_alt + (1 - alpha) * gd_prev_alt

    dist_final = dist_cur + (1 - alpha) * delta_dist
    ec_final_pos = ec_srf_pos + dist_final * ec_sat_vec

    gd_final_pos = spicierpy.recgeo(
        rectan=ec_final_pos,
        re=constants.WGS84_SEMI_MAJOR_AXIS_KM,
        f=constants.WGS84_INVERSE_FLATTENING,
    )
    gd_final_lon, gd_final_lat, gd_final_alt = gd_final_pos

    gd_final_h = alpha * gd_cur_h + (1 - alpha) * gd_prev_h

    # 12) Final geodetic coordinates.
    gd_final = np.array([np.rad2deg(gd_final_lon), np.rad2deg(gd_final_lat), gd_final_h])  # Note: `h` not `alt`!

    if logger.isEnabledFor(5):
        logger.log(5, 'Terrain intersection done=[%d]: lon=[%11.6f], lat=[%10.6f], alt=[%5.3f], dem=[%5.3f]',
                   i, gd_final[0], gd_final[1], gd_final_alt, gd_final[2])

    return gd_final


# A little faster than the current impl when there's a single boresight/pixel
# vector per time step, but MUCH slower when there's multiple vectors!
def legacy_intersect_ellipsoid(ugps_times, instrument, correction=None, allow_nans=True, boresight_vector=None,
                               geodetic=False):
    """Geolocate the boresight on the Earth's surface (WGS84 ellipsoid).
    """
    # Prepare the SPICE objects.
    if not isinstance(instrument, spicierpy.obj.Body):
        instrument = spicierpy.obj.Body(instrument, frame=True)
    name = f'Surf[{instrument.name}]'

    # Prepare arguments for spice.
    target_name = 'EARTH'
    fixed_frame_name = 'ITRF93'  # ECEF (high-precision)
    observer_name = instrument.name
    boresight_frame_name = instrument.frame.name

    et_times = spicetime.adapt(ugps_times, to='et')
    if correction is None:
        correction = 'NONE'
    nan_result3 = (np.nan, np.nan, np.nan)

    # Optional multi-pixel support...
    if boresight_vector is None:
        pix_count, pix_vectors = pixel_vectors(instrument)
    else:
        pix_count = 1
        pix_vectors = [boresight_vector]

    # Function to handle insufficient data errors.
    def _query(sample_et, vector):
        try:
            # Compute surface intercept lat/lon.
            pt_surf, _, vec_surf = spicierpy.sincpt(
                et=sample_et, abcorr=correction, method='ELLIPSOID',
                target=target_name, fixref=fixed_frame_name,
                obsrvr=observer_name, dref=boresight_frame_name, dvec=vector,
            )
            return pt_surf

        except SpiceyError as e:
            if not allow_nans:
                raise
            if 'returns not found' in e.message:
                return nan_result3  # No surface intercept.
            if 'SPICE(SPKINSUFFDATA)' in e.short:
                return nan_result3  # Lacks ephemeris (generally).
            if 'SPICE(NOFRAMECONNECT)' in e.short:
                return nan_result3  # Lacks attitude (generally).
            if 'SPICE(NOTDISJOINT)' in e.short:
                # Interp through an invalid ephemeris (generally a gap).
                return nan_result3  # Viewpoint is inside target.
            raise e

    logger.debug('Checking for [%s x %s] Earth surface intercepts: %s', pix_count, len(et_times), name)

    points = [_query(time, pix_vectors[ipix]) for time in et_times for ipix in range(pix_count)]
    points = np.asarray(points)

    if pix_count == 1:
        index = pd.Index(ugps_times, name='ugps')
    else:
        index = pd.MultiIndex.from_product([ugps_times, np.arange(pix_count) + 1], names=['ugps', 'pixel'])

    if geodetic:
        lonlatalt = ecef_to_geodetic(points, degrees=True)
        data = pd.DataFrame(lonlatalt, columns=['lon', 'lat', 'alt'], index=index)
    else:
        data = pd.DataFrame(points, columns=['x', 'y', 'z'], index=index)

    data.columns.name = name
    return data
