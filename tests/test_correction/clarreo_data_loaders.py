#!/usr/bin/env python3
"""
CLARREO-specific data loading functions for Monte Carlo testing.

This module contains all CLARREO/HySICS-specific data loading logic that was
previously in monte_carlo.py. These functions handle the specific file formats,
naming conventions, and data transformations needed for CLARREO Pathfinder data.

Other missions should create similar modules with their own data loading logic.

Usage:
    from clarreo_data_loaders import load_clarreo_telemetry, load_clarreo_science

    tlm_data = load_clarreo_telemetry(tlm_key, config)
    sci_data = load_clarreo_science(sci_key, config)

Author: Mission-Agnostic Monte Carlo Implementation
Date: October 28, 2025
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from curryer import spicierpy as sp

logger = logging.getLogger(__name__)


def load_clarreo_telemetry(tlm_key: str, config) -> pd.DataFrame:
    """
    Load CLARREO telemetry data from multiple CSV files.

    CLARREO-specific implementation that:
    - Loads 4 separate CSV files (SC_SPK, SC_CK, ST_CK, AZEL_CK)
    - Reverses azimuth direction
    - Converts star-tracker DCM to quaternions
    - Merges all telemetry sources
    - Computes combined time tags

    Args:
        tlm_key: Path to telemetry file or identifier (used to construct paths)
        config: Monte Carlo configuration

    Returns:
        DataFrame with merged telemetry data
    """
    # Extract the base path from config or construct from tlm_key
    # For test cases, tlm_key is often just a string identifier like 'telemetry_5a'
    # The actual data location comes from config.geo.meta_kernel_file or we use default
    if hasattr(config.geo, "meta_kernel_file") and config.geo.meta_kernel_file:
        # Get directory from meta kernel file path
        base_path = config.geo.meta_kernel_file.parent
    elif isinstance(tlm_key, Path) and tlm_key.is_dir():
        base_path = tlm_key
    elif isinstance(tlm_key, Path) and tlm_key.parent.exists():
        base_path = tlm_key.parent
    else:
        # Fallback: construct absolute path to test data
        script_dir = Path(__file__).parent.parent.parent
        base_path = script_dir / "tests" / "data" / "clarreo" / "gcs"

    logger.info(f"Loading CLARREO telemetry data from: {base_path}")

    # Verify the directory exists and has data
    if not base_path.exists():
        raise FileNotFoundError(f"Telemetry data directory not found: {base_path}")

    # Load the 4 telemetry CSVs (CLARREO-specific files)
    sc_spk_df = pd.read_csv(base_path / "openloop_tlm_5a_sc_spk_20250521T225242.csv", index_col=0)
    sc_ck_df = pd.read_csv(base_path / "openloop_tlm_5a_sc_ck_20250521T225242.csv", index_col=0)
    st_ck_df = pd.read_csv(base_path / "openloop_tlm_5a_st_ck_20250521T225242.csv", index_col=0)
    azel_ck_df = pd.read_csv(base_path / "openloop_tlm_5a_azel_ck_20250521T225242.csv", index_col=0)

    logger.info(
        f"Loaded telemetry CSVs - SC_SPK: {sc_spk_df.shape}, SC_CK: {sc_ck_df.shape}, "
        f"ST_CK: {st_ck_df.shape}, AZEL_CK: {azel_ck_df.shape}"
    )

    # CLARREO-specific: Reverse the direction of the Azimuth element
    azel_ck_df["hps.az_ang_nonlin"] = azel_ck_df["hps.az_ang_nonlin"] * -1

    # CLARREO-specific: Convert star-tracker from rotation matrix to quaternion
    tlm_st_rot = np.vstack(
        [
            st_ck_df["hps.dcm_base_iss_1_1"].values,
            st_ck_df["hps.dcm_base_iss_1_2"].values,
            st_ck_df["hps.dcm_base_iss_1_3"].values,
            st_ck_df["hps.dcm_base_iss_2_1"].values,
            st_ck_df["hps.dcm_base_iss_2_2"].values,
            st_ck_df["hps.dcm_base_iss_2_3"].values,
            st_ck_df["hps.dcm_base_iss_3_1"].values,
            st_ck_df["hps.dcm_base_iss_3_2"].values,
            st_ck_df["hps.dcm_base_iss_3_3"].values,
        ]
    ).T
    tlm_st_rot = np.reshape(tlm_st_rot, (-1, 3, 3)).copy()

    tlm_st_rot_q = np.vstack([sp.m2q(tlm_st_rot[i, :, :]) for i in range(tlm_st_rot.shape[0])])
    st_ck_df["hps.dcm_base_iss_s"] = tlm_st_rot_q[:, 0]
    st_ck_df["hps.dcm_base_iss_i"] = tlm_st_rot_q[:, 1]
    st_ck_df["hps.dcm_base_iss_j"] = tlm_st_rot_q[:, 2]
    st_ck_df["hps.dcm_base_iss_k"] = tlm_st_rot_q[:, 3]

    # CLARREO-specific: Merge all telemetry sources with outer joins
    left_df = sc_spk_df
    for right_df in [sc_ck_df, st_ck_df, azel_ck_df]:
        left_df = pd.merge(left_df, right_df, on="ert", how="outer")
    left_df = left_df.sort_values("ert")

    # CLARREO-specific: Compute combined second and subsecond timetags
    for col in list(left_df):
        if col in ("hps.bad_ps_tms", "hps.corrected_tms", "hps.resolver_tms", "hps.st_quat_coi_tms"):
            assert col + "s" in left_df.columns, f"Missing subsecond column for {col}"

            if col == "hps.bad_ps_tms":
                left_df[col + "_tmss"] = left_df[col] + left_df[col + "s"] / 256
            elif col in ("hps.corrected_tms", "hps.resolver_tms", "hps.st_quat_coi_tms"):
                left_df[col + "_tmss"] = left_df[col] + left_df[col + "s"] / 2**32
            else:
                raise ValueError(f"Missing conversion for expected column: {col}")

    logger.info(f"Final CLARREO telemetry shape: {left_df.shape}")

    # Validate output format
    from curryer.correction.dataio import validate_telemetry_output

    validate_telemetry_output(left_df, config)

    return left_df


def load_clarreo_science(sci_key: str, config) -> pd.DataFrame:
    """
    Load CLARREO science frame timing data.

    CLARREO-specific implementation that:
    - Loads science frame timestamps from CSV
    - Converts GPS seconds to uGPS (microseconds)

    Args:
        sci_key: Path to science file or identifier
        config: Monte Carlo configuration

    Returns:
        DataFrame with science frame timestamps
    """
    # Extract the base path from config or construct from sci_key
    if hasattr(config.geo, "meta_kernel_file") and config.geo.meta_kernel_file:
        base_path = config.geo.meta_kernel_file.parent
    elif isinstance(sci_key, Path) and sci_key.is_dir():
        base_path = sci_key
    elif isinstance(sci_key, Path) and sci_key.parent.exists():
        base_path = sci_key.parent
    else:
        # Fallback: construct absolute path to test data
        script_dir = Path(__file__).parent.parent.parent
        base_path = script_dir / "tests" / "data" / "clarreo" / "gcs"

    logger.info(f"Loading CLARREO science data from: {base_path}")

    # CLARREO-specific: Load science frame timing CSV
    sci_time_df = pd.read_csv(base_path / "openloop_tlm_5a_sci_times_20250521T225242.csv", index_col=0)

    # CLARREO-specific: Frame times are GPS seconds, geolocation expects uGPS (microseconds)
    sci_time_df["corrected_timestamp"] *= 1e6

    logger.info(f"CLARREO science data shape: {sci_time_df.shape}")
    logger.info(
        f"corrected_timestamp range: {sci_time_df['corrected_timestamp'].min():.2e} to "
        f"{sci_time_df['corrected_timestamp'].max():.2e} uGPS"
    )

    # Validate output format
    from curryer.correction.dataio import validate_science_output

    validate_science_output(sci_time_df, config)

    return sci_time_df


def load_clarreo_gcp(gcp_key: str, config):
    """
    Load CLARREO Ground Control Point (GCP) reference data.

    PLACEHOLDER - Real implementation will:
    - Load Landsat GCP reference images/coordinates
    - Extract georeferenced control points
    - Return spatially/temporally matched reference data

    Args:
        gcp_key: Path to GCP file or identifier
        config: Monte Carlo configuration

    Returns:
        GCP reference data (format TBD based on GCP pairing module requirements)
    """
    logger.info(f"Loading CLARREO GCP data from: {gcp_key} (PLACEHOLDER)")

    # For testing purposes, return None - the GCP pairing module will handle this
    # In real implementation, this would load and process GCP reference data
    return
