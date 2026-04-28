"""
clarreo_config.py — CLARREO correction configuration factory.

This module provides a clean, public-API-only way to create a
``CorrectionConfig`` for the CLARREO Pathfinder mission. Use this as a
reference when creating your own mission configuration.

Prefer loading from JSON in production::

    from curryer.correction import load_config_from_json
    config = load_config_from_json("examples/correction/clarreo_config.json")

Use this Python script when you need to build the config programmatically
or inject it into tests.

Note
----
For the test fixture version (which also saves the config to JSON), see
``tests/test_correction/clarreo/clarreo_config.py``. That file is a test
infrastructure helper; this file is the user-facing example.
"""

from __future__ import annotations

import logging
from pathlib import Path

from curryer.correction import (
    CorrectionConfig,
    GeolocationConfig,
    NetCDFConfig,
    ParameterConfig,
    ParameterType,
)

logger = logging.getLogger(__name__)

# Default repo-relative paths — override by passing data_dir / generic_dir
_DEFAULT_DATA_DIR = Path("tests/data/clarreo/gcs")
_DEFAULT_GENERIC_DIR = Path("data/generic")


def create_clarreo_config(
    data_dir: Path | None = None,
    generic_dir: Path | None = None,
) -> CorrectionConfig:
    """Create the CLARREO Pathfinder geolocation correction configuration.

    Defines all 12 CLARREO-specific correction parameters:
    - 9 CONSTANT_KERNEL parameters (3 frames x 3 attitude angles each)
    - 2 OFFSET_KERNEL parameters (azimuth and elevation angle biases)
    - 1 OFFSET_TIME parameter (science timing correction)

    Parameters
    ----------
    data_dir : Path or None
        Directory containing the CLARREO GCS kernel JSON files
        (``cprs_*_v01.attitude.ck.json``, etc.).
        Defaults to ``tests/data/clarreo/gcs`` (repo-relative).
    generic_dir : Path or None
        Directory containing the generic SPICE kernels (leap-second, etc.).
        Defaults to ``data/generic`` (repo-relative).

    Returns
    -------
    CorrectionConfig
        Validated configuration object ready to pass to ``run_correction()``
        or ``verify()``.

    Examples
    --------
    >>> from examples.correction.clarreo_config import create_clarreo_config
    >>> config = create_clarreo_config()
    >>> print(config.performance_threshold_m)
    250.0
    >>> print(len(config.parameters))
    6

    Or load from the JSON file (preferred for production):

    >>> from curryer.correction import load_config_from_json
    >>> config = load_config_from_json("examples/correction/clarreo_config.json")
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    generic_dir = Path(generic_dir) if generic_dir is not None else _DEFAULT_GENERIC_DIR

    parameters = [
        # ----------------------------------------------------------------
        # CONSTANT_KERNEL parameters
        # Each ParameterConfig represents one SPICE frame kernel.
        # current_value = [roll, pitch, yaw] baseline offsets (arcseconds).
        # The correction loop perturbs these values to find a better alignment.
        # ----------------------------------------------------------------
        # BASE mechanical frame — mounts the ISS pointing platform
        ParameterConfig(
            ptype=ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_base_v01.attitude.ck.json",
            data={
                "current_value": [0.0, 0.0, 0.0],
                "bounds": [-300.0, 300.0],
                "sigma": 30.0,
                "units": "arcseconds",
                "transformation_type": "dcm_rotation",
                "coordinate_frames": ["BASE_AZIMUTH", "BASE_CUBE"],
            },
        ),
        # YOKE frame — rotational axis between azimuth and elevation stages
        ParameterConfig(
            ptype=ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_yoke_v01.attitude.ck.json",
            data={
                "current_value": [0.0, 0.0, 0.0],
                "bounds": [-200.0, 200.0],
                "sigma": 20.0,
                "units": "arcseconds",
                "transformation_type": "dcm_rotation",
                "coordinate_frames": ["YOKE_ELEVATION", "YOKE_AZIMUTH"],
            },
        ),
        # HYSICS focal-plane instrument frame
        ParameterConfig(
            ptype=ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_hysics_v01.attitude.ck.json",
            data={
                "current_value": [0.0, 0.0, 0.0],
                "bounds": [-300.0, 300.0],
                "sigma": 30.0,
                "units": "arcseconds",
                "transformation_type": "dcm_rotation",
                "coordinate_frames": ["HYSICS_SLIT", "CRADLE_ELEVATION"],
            },
        ),
        # ----------------------------------------------------------------
        # OFFSET_KERNEL parameters
        # A scalar bias is added to a named telemetry column before
        # the dynamic kernel is regenerated each iteration.
        # ----------------------------------------------------------------
        # Azimuth angle bias
        ParameterConfig(
            ptype=ParameterType.OFFSET_KERNEL,
            config_file=data_dir / "cprs_az_v01.attitude.ck.json",
            data={
                "field": "hps.az_ang_nonlin",
                "current_value": 0.0,
                "bounds": [-300.0, 300.0],
                "sigma": 30.0,
                "units": "arcseconds",
                "transformation_type": "angle_bias",
                "coordinate_frames": ["YOKE_AZIMUTH"],
            },
        ),
        # Elevation angle bias
        ParameterConfig(
            ptype=ParameterType.OFFSET_KERNEL,
            config_file=data_dir / "cprs_el_v01.attitude.ck.json",
            data={
                "field": "hps.el_ang_nonlin",
                "current_value": 0.0,
                "bounds": [-300.0, 300.0],
                "sigma": 30.0,
                "units": "arcseconds",
                "transformation_type": "angle_bias",
                "coordinate_frames": ["YOKE_ELEVATION"],
            },
        ),
        # ----------------------------------------------------------------
        # OFFSET_TIME parameter
        # All science timestamps are shifted by a constant offset.
        # No kernel file is needed — the pipeline modifies the data directly.
        # ----------------------------------------------------------------
        ParameterConfig(
            ptype=ParameterType.OFFSET_TIME,
            config_file=None,
            data={
                "field": "corrected_timestamp",
                "current_value": 0.0,
                "bounds": [-50.0, 50.0],
                "sigma": 7.0,
                "units": "milliseconds",
            },
        ),
    ]

    geo = GeolocationConfig(
        meta_kernel_file=data_dir / "cprs_v01.kernels.tm.json",
        generic_kernel_dir=generic_dir,
        dynamic_kernels=[
            data_dir / "iss_sc_v01.ephemeris.spk.json",
            data_dir / "iss_sc_v01.attitude.ck.json",
            data_dir / "cprs_st_v01.attitude.ck.json",
        ],
        instrument_name="CPRS_HYSICS",
        time_field="corrected_timestamp",
    )

    netcdf = NetCDFConfig(
        title="CLARREO Pathfinder Geolocation Correction Analysis",
        description="Parameter sensitivity analysis for CLARREO Pathfinder on ISS",
        performance_threshold_m=250.0,
    )

    config = CorrectionConfig(
        seed=42,
        n_iterations=5,
        parameters=parameters,
        geo=geo,
        # CLARREO mission performance requirements
        performance_threshold_m=250.0,  # Each measurement must be < 250 m nadir-equivalent
        performance_spec_percent=39.0,  # At least 39% of measurements must pass
        netcdf=netcdf,
        # HySICS calibration files (relative to calibration_dir)
        calibration_file_names={
            "los_vectors": "b_HS.mat",
            "optical_psf": "optical_PSF_675nm_upsampled.mat",
        },
        # Variable names in image-matching xr.Dataset output
        spacecraft_position_name="riss_ctrs",
        boresight_name="bhat_hs",
        transformation_matrix_name="t_hs2ctrs",
    )

    config.validate()
    logger.info("CLARREO config created: %d parameters", len(config.parameters))
    return config
