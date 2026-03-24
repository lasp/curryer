#!/usr/bin/env python3
"""CLARREO Pathfinder telemetry and science preprocessing script.

This script converts CLARREO-specific raw telemetry CSV files into clean,
pipeline-ready CSVs that the correction pipeline can load directly via
:class:`~curryer.correction.config.DataConfig`.

CLARREO-specific preprocessing steps
--------------------------------------
Telemetry:
  - Load four raw CSVs: ``SC_SPK``, ``SC_CK``, ``ST_CK``, ``AZEL_CK``
  - Reverse the azimuth sign (``hps.az_ang_nonlin *= -1``)
  - Convert star-tracker DCM columns to quaternion columns
  - Outer-join and sort all four sources on ``ert``
  - Compute combined second + sub-second timetags

Science:
  - Load science-frame timing CSV
  - **No** time scaling is applied here — the pipeline handles that via
    ``DataConfig.time_scale_factor``.  The output ``corrected_timestamp``
    column contains GPS seconds (the raw unit from the instrument).

Usage
-----
As a library (called from tests or other scripts)::

    from scripts.clarreo_preprocess import preprocess_clarreo_telemetry, preprocess_clarreo_science
    import pandas as pd, tempfile, pathlib

    data_dir = pathlib.Path("tests/data/clarreo/gcs")
    tlm_df = preprocess_clarreo_telemetry(data_dir)   # → pd.DataFrame
    sci_df = preprocess_clarreo_science(data_dir)     # → pd.DataFrame

    # Save to CSVs for the pipeline
    tlm_df.to_csv("telemetry.csv")
    sci_df.to_csv("science.csv")

As a CLI::

    python scripts/clarreo_preprocess.py --data-dir tests/data/clarreo/gcs --output-dir /tmp/clarreo_out
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public preprocessing functions
# ---------------------------------------------------------------------------


def preprocess_clarreo_telemetry(data_dir: Path | str) -> pd.DataFrame:
    """Preprocess CLARREO raw telemetry CSVs into a single clean DataFrame.

    Parameters
    ----------
    data_dir : Path
        Directory containing the four CLARREO telemetry CSV files.

    Returns
    -------
    pd.DataFrame
        Merged and preprocessed telemetry DataFrame ready for the correction
        pipeline.
    """
    try:
        from curryer import spicierpy as sp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("curryer.spicierpy is required for DCM→quaternion conversion.") from exc

    data_dir = Path(data_dir)
    logger.info(f"Loading CLARREO telemetry CSVs from: {data_dir}")

    sc_spk_df = pd.read_csv(data_dir / "openloop_tlm_5a_sc_spk_20250521T225242.csv", index_col=0)
    sc_ck_df = pd.read_csv(data_dir / "openloop_tlm_5a_sc_ck_20250521T225242.csv", index_col=0)
    st_ck_df = pd.read_csv(data_dir / "openloop_tlm_5a_st_ck_20250521T225242.csv", index_col=0)
    azel_ck_df = pd.read_csv(data_dir / "openloop_tlm_5a_azel_ck_20250521T225242.csv", index_col=0)

    logger.info(
        f"Loaded – SC_SPK: {sc_spk_df.shape}, SC_CK: {sc_ck_df.shape}, "
        f"ST_CK: {st_ck_df.shape}, AZEL_CK: {azel_ck_df.shape}"
    )

    # Reverse the direction of the Azimuth element (CLARREO-specific)
    azel_ck_df["hps.az_ang_nonlin"] = azel_ck_df["hps.az_ang_nonlin"] * -1

    # Convert star-tracker DCM to quaternion (CLARREO-specific)
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

    # Outer-join all four sources and sort by ERT
    left_df = sc_spk_df
    for right_df in [sc_ck_df, st_ck_df, azel_ck_df]:
        left_df = pd.merge(left_df, right_df, on="ert", how="outer")
    left_df = left_df.sort_values("ert")

    # Compute combined second + sub-second timetags (CLARREO-specific)
    for col in list(left_df):
        if col in ("hps.bad_ps_tms", "hps.corrected_tms", "hps.resolver_tms", "hps.st_quat_coi_tms"):
            if col + "s" not in left_df.columns:
                raise ValueError(f"Missing sub-second column for {col}")
            if col == "hps.bad_ps_tms":
                left_df[col + "_tmss"] = left_df[col] + left_df[col + "s"] / 256
            elif col in ("hps.corrected_tms", "hps.resolver_tms", "hps.st_quat_coi_tms"):
                left_df[col + "_tmss"] = left_df[col] + left_df[col + "s"] / 2**32
            else:
                raise ValueError(f"Missing conversion for expected column: {col}")

    logger.info(f"Final telemetry shape: {left_df.shape}")
    return left_df


def preprocess_clarreo_science(data_dir: Path | str) -> pd.DataFrame:
    """Preprocess CLARREO science frame timing CSV into a clean DataFrame.

    The ``corrected_timestamp`` column is returned in GPS seconds (the raw
    instrument unit).  Set ``DataConfig.time_scale_factor = 1e6`` in your
    :class:`~curryer.correction.config.CorrectionConfig` to convert to uGPS
    during pipeline loading.

    Parameters
    ----------
    data_dir : Path
        Directory containing the science timing CSV.

    Returns
    -------
    pd.DataFrame
        Science DataFrame with ``corrected_timestamp`` in GPS seconds.
    """
    data_dir = Path(data_dir)
    logger.info(f"Loading CLARREO science CSV from: {data_dir}")

    sci_time_df = pd.read_csv(data_dir / "openloop_tlm_5a_sci_times_20250521T225242.csv", index_col=0)

    logger.info(
        f"Science data shape: {sci_time_df.shape}, "
        f"corrected_timestamp range: "
        f"{sci_time_df['corrected_timestamp'].min():.3f} – "
        f"{sci_time_df['corrected_timestamp'].max():.3f} GPS sec"
    )
    return sci_time_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess CLARREO Pathfinder raw telemetry/science CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the raw CLARREO CSV files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where preprocessed CSVs will be written.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tlm_path = output_dir / "clarreo_telemetry_preprocessed.csv"
    sci_path = output_dir / "clarreo_science_preprocessed.csv"

    logger.info("Preprocessing CLARREO telemetry…")
    tlm_df = preprocess_clarreo_telemetry(args.data_dir)
    tlm_df.to_csv(tlm_path)
    logger.info(f"  → {tlm_path}")

    logger.info("Preprocessing CLARREO science frame timing…")
    sci_df = preprocess_clarreo_science(args.data_dir)
    sci_df.to_csv(sci_path)
    logger.info(f"  → {sci_path}")

    logger.info("Done.")
    print(f"Telemetry: {tlm_path}")
    print(f"Science:   {sci_path}")


if __name__ == "__main__":
    main()
