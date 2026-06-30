"""Synthetic test-data helpers shared by test_pipeline.py and clarreo e2e tests.

These functions generate realistic-looking but entirely synthetic sensor data
(boresight vectors, spacecraft positions, transformation matrices, GCP pairs)
so that upstream pipeline tests can run without any real instrument data.

**These are test infrastructure helpers – not pytest tests.**
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class _PlaceholderConfig:
    """Default parameters controlling synthetic data generation."""

    base_error_m: float = 50.0
    param_error_scale: float = 10.0
    max_measurements: int = 100
    min_measurements: int = 10
    orbit_radius_mean_m: float = 6.78e6
    orbit_radius_std_m: float = 4e3
    latitude_range: tuple = (-60.0, 60.0)
    longitude_range: tuple = (-180.0, 180.0)
    altitude_range: tuple = (0.0, 1000.0)
    max_off_nadir_rad: float = 0.1


def synthetic_gcp_pairing(science_data_files):
    """Return SYNTHETIC GCP pairs for upstream testing (no real GCP data needed)."""
    logger.warning("USING SYNTHETIC GCP PAIRING - FAKE DATA!")
    return [(str(f), f"landsat_gcp_{i:03d}.tif") for i, f in enumerate(science_data_files)]


def synthetic_image_matching(
    geolocated_data,
    gcp_reference_file,
    telemetry,
    params_info,
    setup,
    los_vectors_cached=None,
    optical_psfs_cached=None,
):
    """Return SYNTHETIC image-matching results for upstream testing.

    Accepts (and ignores) the same signature as the real ``image_matching``
    function so the pipeline loop can call it transparently.
    """
    logger.warning("USING SYNTHETIC IMAGE MATCHING - FAKE DATA!")
    placeholder_cfg = setup.placeholder if hasattr(setup, "placeholder") and setup.placeholder else _PlaceholderConfig()
    sc_pos_name = getattr(setup, "spacecraft_position_name", "sc_position")
    boresight_name = getattr(setup, "boresight_name", "boresight")
    transform_name = getattr(setup, "transformation_matrix_name", "t_inst2ref")

    valid_mask = ~np.isnan(geolocated_data["latitude"].values).any(axis=1)
    n_valid = int(valid_mask.sum())
    n_meas = placeholder_cfg.min_measurements if n_valid == 0 else min(n_valid, placeholder_cfg.max_measurements)

    riss_ctrs = _generate_spherical_positions(
        n_meas, placeholder_cfg.orbit_radius_mean_m, placeholder_cfg.orbit_radius_std_m
    )
    boresights = _generate_synthetic_boresights(n_meas, placeholder_cfg.max_off_nadir_rad)
    t_matrices = _generate_nadir_aligned_transforms(n_meas, riss_ctrs, boresights)

    param_contribution = (
        sum(abs(p) if isinstance(p, (int, float)) else np.linalg.norm(p) for _, p in params_info)
        * placeholder_cfg.param_error_scale
    )
    error_magnitude = placeholder_cfg.base_error_m + param_contribution
    lat_errors = np.random.normal(0, error_magnitude / 111_000, n_meas)
    lon_errors = np.random.normal(0, error_magnitude / 111_000, n_meas)

    if n_valid > 0:
        idx = np.where(valid_mask)[0][:n_meas]
        gcp_lat = geolocated_data["latitude"].values[idx, 0]
        gcp_lon = geolocated_data["longitude"].values[idx, 0]
    else:
        gcp_lat = np.random.uniform(*placeholder_cfg.latitude_range, n_meas)
        gcp_lon = np.random.uniform(*placeholder_cfg.longitude_range, n_meas)

    gcp_alt = np.random.uniform(*placeholder_cfg.altitude_range, n_meas)

    return xr.Dataset(
        {
            "lat_error_deg": (["measurement"], lat_errors),
            "lon_error_deg": (["measurement"], lon_errors),
            sc_pos_name: (["measurement", "xyz"], riss_ctrs),
            boresight_name: (["measurement", "xyz"], boresights),
            transform_name: (["measurement", "xyz_from", "xyz_to"], t_matrices),
            "gcp_lat_deg": (["measurement"], gcp_lat),
            "gcp_lon_deg": (["measurement"], gcp_lon),
            "gcp_alt": (["measurement"], gcp_alt),
        },
        coords={
            "measurement": range(n_meas),
            "xyz": ["x", "y", "z"],
            "xyz_from": ["x", "y", "z"],
            "xyz_to": ["x", "y", "z"],
        },
    )


def _generate_synthetic_boresights(n, max_off_nadir_rad=0.07):
    """Return *n* unit boresight vectors with small off-nadir angles."""
    b = np.zeros((n, 3))
    for i in range(n):
        th = np.random.uniform(-max_off_nadir_rad, max_off_nadir_rad)
        b[i] = [0.0, np.sin(th), np.cos(th)]
    return b


def _generate_spherical_positions(n, radius_mean_m, radius_std_m):
    """Return *n* random points on a sphere (spacecraft orbit positions)."""
    pos = np.zeros((n, 3))
    for i in range(n):
        r = np.random.normal(radius_mean_m, radius_std_m)
        phi = np.random.uniform(0, 2 * np.pi)
        ct = np.random.uniform(-1, 1)
        st = np.sqrt(max(0.0, 1 - ct**2))
        pos[i] = [r * st * np.cos(phi), r * st * np.sin(phi), r * ct]
    return pos


def _generate_nadir_aligned_transforms(n, riss_ctrs, boresights_hs):
    """Return *n* rotation matrices aligning ``boresights_hs`` toward nadir."""
    T = np.zeros((n, 3, 3))
    for i in range(n):
        nadir = -riss_ctrs[i] / np.linalg.norm(riss_ctrs[i])
        bhat = boresights_hs[i] / np.linalg.norm(boresights_hs[i])
        ax = np.cross(bhat, nadir)
        ax_norm = np.linalg.norm(ax)
        if ax_norm < 1e-6:
            if np.dot(bhat, nadir) > 0:
                T[i] = np.eye(3)
            else:
                perp = np.array([1, 0, 0]) if abs(bhat[0]) < 0.9 else np.array([0, 1, 0])
                ax = np.cross(bhat, perp)
                ax /= np.linalg.norm(ax)
                K = _skew(ax)
                T[i] = np.eye(3) + 2 * K @ K
        else:
            ax /= ax_norm
            angle = np.arccos(np.clip(np.dot(bhat, nadir), -1.0, 1.0))
            K = _skew(ax)
            T[i] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return T


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
