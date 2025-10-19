"""Minimal adapters to connect Monte Carlo pipeline to image matching module.

This module provides data conversion functions to integrate the existing
integrated_image_match() function with the Monte Carlo GCS pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.io import loadmat
from typing import List
import logging

from .data_structures import ImageGrid, OpticalPSFEntry

logger = logging.getLogger(__name__)


def geolocated_to_image_grid(geo_dataset: xr.Dataset) -> ImageGrid:
    """
    Convert Monte Carlo geolocated dataset to ImageGrid.

    Args:
        geo_dataset: xarray.Dataset from geolocation step with latitude, longitude, altitude

    Returns:
        ImageGrid object suitable for image matching
    """
    lat = geo_dataset['latitude'].values
    lon = geo_dataset['longitude'].values

    # Try different field names for altitude/height
    if 'altitude' in geo_dataset:
        h = geo_dataset['altitude'].values
    elif 'height' in geo_dataset:
        h = geo_dataset['height'].values
    else:
        h = np.zeros_like(lat)

    # Get actual radiance/reflectance data when available
    # For now, use ones as placeholder (image matching uses this for intensity)
    if 'radiance' in geo_dataset:
        data = geo_dataset['radiance'].values
    elif 'reflectance' in geo_dataset:
        data = geo_dataset['reflectance'].values
    else:
        data = np.ones_like(lat)

    return ImageGrid(data=data, lat=lat, lon=lon, h=h)


def extract_spacecraft_position_midframe(telemetry: pd.DataFrame) -> np.ndarray:
    """
    Extract spacecraft position at mid-frame from telemetry.

    Args:
        telemetry: Telemetry dataframe with spacecraft position columns

    Returns:
        np.ndarray shape (3,) - [x, y, z] in meters (J2000 frame)

    Raises:
        ValueError: If position columns cannot be found
    """
    mid_idx = len(telemetry) // 2

    # Try common column name patterns
    position_patterns = [
        ['sc_pos_x', 'sc_pos_y', 'sc_pos_z'],
        ['position_x', 'position_y', 'position_z'],
        ['r_x', 'r_y', 'r_z'],
        ['pos_x', 'pos_y', 'pos_z'],
    ]

    for cols in position_patterns:
        if all(c in telemetry.columns for c in cols):
            position = telemetry[cols].iloc[mid_idx].values.astype(np.float64)
            logger.debug(f"Extracted spacecraft position from columns {cols}: {position}")
            return position

    # If patterns don't match, try to find any column containing 'pos' or 'r_'
    pos_cols = [c for c in telemetry.columns if 'pos' in c.lower() or c.startswith('r_')]
    if len(pos_cols) >= 3:
        logger.warning(f"Using first 3 position-like columns: {pos_cols[:3]}")
        return telemetry[pos_cols[:3]].iloc[mid_idx].values.astype(np.float64)

    raise ValueError(
        f"Cannot find position columns in telemetry. "
        f"Available columns: {telemetry.columns.tolist()}"
    )


def load_los_vectors_from_mat(mat_file: Path) -> np.ndarray:
    """
    Load line-of-sight vectors from MATLAB file.

    Args:
        mat_file: Path to MATLAB file containing LOS vectors

    Returns:
        np.ndarray shape (n_pixels, 3) - LOS vectors in instrument frame

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If expected data not found in file
    """
    if not mat_file.exists():
        raise FileNotFoundError(f"LOS vector file not found: {mat_file}")

    mat_data = loadmat(str(mat_file))

    # Try common field names
    for key in ['b_HS', 'los_vectors', 'pixel_vectors']:
        if key in mat_data:
            b_hs = mat_data[key]  # Shape: (3, n_pixels) or (n_pixels, 3)

            # Ensure shape is (n_pixels, 3)
            if b_hs.shape[0] == 3 and b_hs.shape[1] > 3:
                b_hs = b_hs.T

            logger.info(f"Loaded LOS vectors from {mat_file.name}: shape {b_hs.shape}")
            return b_hs

    raise KeyError(f"Cannot find LOS vectors in {mat_file}. Available keys: {list(mat_data.keys())}")


def load_optical_psf_from_mat(mat_file: Path) -> List[OpticalPSFEntry]:
    """
    Load optical PSF from MATLAB file.

    Args:
        mat_file: Path to MATLAB file containing optical PSF data

    Returns:
        List of OpticalPSFEntry objects

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If expected data not found in file
    """
    if not mat_file.exists():
        raise FileNotFoundError(f"Optical PSF file not found: {mat_file}")

    mat_data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

    # Try common field names
    for key in ['PSF_struct_675nm', 'optical_PSF', 'PSF']:
        if key in mat_data:
            psf_struct = mat_data[key]

            # Convert to list if single entry
            psf_entries_raw = np.atleast_1d(psf_struct)

            # Create OpticalPSFEntry objects
            psf_entries = []
            for entry in psf_entries_raw:
                psf_entry = OpticalPSFEntry(
                    data=np.asarray(entry.data),
                    x=np.asarray(entry.x).ravel(),
                    field_angle=np.asarray(entry.FA if hasattr(entry, 'FA') else entry.field_angle).ravel(),
                )
                psf_entries.append(psf_entry)

            logger.info(f"Loaded {len(psf_entries)} optical PSF entries from {mat_file.name}")
            return psf_entries

    raise KeyError(f"Cannot find optical PSF in {mat_file}. Available keys: {list(mat_data.keys())}")


def load_gcp_from_mat(mat_file: Path) -> ImageGrid:
    """
    Load Ground Control Point (GCP) reference image from MATLAB file.

    Args:
        mat_file: Path to MATLAB file containing GCP data

    Returns:
        ImageGrid with GCP reference data

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If expected data not found in file
    """
    if not mat_file.exists():
        raise FileNotFoundError(f"GCP file not found: {mat_file}")

    mat_data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

    # Try common field names
    for key in ['GCP', 'gcp', 'reference', 'ref_image']:
        if key in mat_data:
            gcp_struct = mat_data[key]

            # Convert struct to ImageGrid
            gcp = ImageGrid(
                data=np.asarray(gcp_struct.data),
                lat=np.asarray(gcp_struct.lat),
                lon=np.asarray(gcp_struct.lon),
                h=np.asarray(gcp_struct.h) if hasattr(gcp_struct, 'h') else None,
            )

            logger.info(f"Loaded GCP from {mat_file.name}: shape {gcp.data.shape}")
            return gcp

    raise KeyError(f"Cannot find GCP data in {mat_file}. Available keys: {list(mat_data.keys())}")


def get_gcp_center_location(gcp: ImageGrid) -> tuple[float, float]:
    """
    Get the center latitude and longitude of a GCP.

    Args:
        gcp: ImageGrid containing GCP data

    Returns:
        Tuple of (center_lat, center_lon) in degrees
    """
    mid_row = gcp.lat.shape[0] // 2
    mid_col = gcp.lat.shape[1] // 2

    center_lat = float(gcp.lat[mid_row, mid_col])
    center_lon = float(gcp.lon[mid_row, mid_col])

    return center_lat, center_lon

