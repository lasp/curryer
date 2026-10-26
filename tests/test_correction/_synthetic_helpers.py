"""Synthetic test-data helpers shared by test_pipeline.py and clarreo e2e tests.

These functions generate realistic-looking but entirely synthetic sensor data
(boresight vectors, spacecraft positions, transformation matrices, GCP pairs)
so that upstream pipeline tests can run without any real instrument data.

**These are test infrastructure helpers – not pytest tests.**
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def synthetic_gcp_pairing(science_data_files):
    """Return SYNTHETIC GCP pairs for upstream testing (no real GCP data needed)."""
    logger.warning("USING SYNTHETIC GCP PAIRING - FAKE DATA!")
    return [(str(f), f"landsat_gcp_{i:03d}.tif") for i, f in enumerate(science_data_files)]


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
