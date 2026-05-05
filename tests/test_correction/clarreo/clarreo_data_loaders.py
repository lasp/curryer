#!/usr/bin/env python3
"""CLARREO-specific data PREPROCESSING scripts — **test fixture**.

These functions are **test infrastructure helpers**, not user-facing examples.

For user-facing examples and documentation, see:
  - ``examples/correction/example_verification.py`` — runnable verification demo
  - ``examples/correction/example_run_correction.py`` — correction loop template
  - ``docs/source/correction_user_guide.md``          — full reference

.. deprecated::
    The Protocol-based loader pattern (``TelemetryLoader``, ``ScienceLoader``,
    ``GCPLoader``) has been removed.  Data loading is now config-driven via
    :class:`~curryer.correction.config.DataConfig`.

    This file is kept as a shim so existing ``from clarreo_data_loaders import …``
    imports in tests continue to resolve.  New code should call
    :mod:`scripts.clarreo_preprocess` directly and pass preprocessed CSV file
    paths to the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_clarreo_telemetry(data_dir: Path | str) -> pd.DataFrame:
    """Preprocess CLARREO raw telemetry CSVs into a single clean DataFrame.

    Performs CLARREO-specific steps:
    - Loads 4 raw CSVs (SC_SPK, SC_CK, ST_CK, AZEL_CK)
    - Reverses azimuth sign
    - Converts star-tracker DCM → quaternion
    - Outer-joins and sorts on ``ert``
    - Computes combined second + sub-second timetags

    Parameters
    ----------
    data_dir : Path
        Directory containing the raw CLARREO telemetry CSV files.
    """
    from curryer import spicierpy as sp

    data_dir = Path(data_dir)
    logger.info(f"Loading CLARREO telemetry from: {data_dir}")

    sc_spk_df = pd.read_csv(data_dir / "openloop_tlm_5a_sc_spk_20250521T225242.csv", index_col=0)
    sc_ck_df = pd.read_csv(data_dir / "openloop_tlm_5a_sc_ck_20250521T225242.csv", index_col=0)
    st_ck_df = pd.read_csv(data_dir / "openloop_tlm_5a_st_ck_20250521T225242.csv", index_col=0)
    azel_ck_df = pd.read_csv(data_dir / "openloop_tlm_5a_azel_ck_20250521T225242.csv", index_col=0)

    # Reverse azimuth direction
    azel_ck_df["hps.az_ang_nonlin"] = azel_ck_df["hps.az_ang_nonlin"] * -1

    # Convert star-tracker DCM → quaternion
    tlm_st_rot = np.vstack([st_ck_df[f"hps.dcm_base_iss_{r}_{c}"].values for r in range(1, 4) for c in range(1, 4)]).T
    tlm_st_rot = np.reshape(tlm_st_rot, (-1, 3, 3)).copy()
    tlm_st_rot_q = np.vstack([sp.m2q(tlm_st_rot[i]) for i in range(tlm_st_rot.shape[0])])
    st_ck_df["hps.dcm_base_iss_s"] = tlm_st_rot_q[:, 0]
    st_ck_df["hps.dcm_base_iss_i"] = tlm_st_rot_q[:, 1]
    st_ck_df["hps.dcm_base_iss_j"] = tlm_st_rot_q[:, 2]
    st_ck_df["hps.dcm_base_iss_k"] = tlm_st_rot_q[:, 3]

    # Outer-join all four sources and sort by ERT
    left_df = sc_spk_df
    for right_df in [sc_ck_df, st_ck_df, azel_ck_df]:
        left_df = pd.merge(left_df, right_df, on="ert", how="outer")
    left_df = left_df.sort_values("ert")

    # Compute combined second + sub-second timetags
    for col in list(left_df):
        if col in ("hps.bad_ps_tms", "hps.corrected_tms", "hps.resolver_tms", "hps.st_quat_coi_tms"):
            sub_col = col + "s"
            if sub_col not in left_df.columns:
                raise ValueError(f"Missing sub-second column for {col}")
            if col == "hps.bad_ps_tms":
                left_df[col + "_tmss"] = left_df[col] + left_df[sub_col] / 256
            else:
                left_df[col + "_tmss"] = left_df[col] + left_df[sub_col] / 2**32

    logger.info(f"Final CLARREO telemetry shape: {left_df.shape}")
    return left_df


def load_clarreo_science(data_dir: Path | str) -> pd.DataFrame:
    """Load CLARREO science frame timing CSV.

    Returns ``corrected_timestamp`` in GPS seconds (the raw instrument unit).
    Set ``DataConfig.time_scale_factor = 1e6`` so the pipeline converts to uGPS.

    Parameters
    ----------
    data_dir : Path
        Directory containing the science timing CSV.
    """
    data_dir = Path(data_dir)
    logger.info(f"Loading CLARREO science from: {data_dir}")
    sci_df = pd.read_csv(data_dir / "openloop_tlm_5a_sci_times_20250521T225242.csv", index_col=0)
    logger.info(f"Science shape: {sci_df.shape}")
    return sci_df


def load_clarreo_gcp(gcp_key: str, config=None) -> None:  # noqa: ANN001
    """No-op placeholder – GCPLoader protocol has been removed.

    Pass GCP file paths directly as the third element of each
    ``tlm_sci_gcp_sets`` tuple instead.
    """
    logger.info("load_clarreo_gcp is a no-op placeholder (GCPLoader protocol removed).")
    return None


__all__ = ["load_clarreo_telemetry", "load_clarreo_science", "load_clarreo_gcp"]
