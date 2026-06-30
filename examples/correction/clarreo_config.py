"""
clarreo_config.py — CLARREO correction configuration factory.

This module is the public-API-only way to build the CLARREO Pathfinder
correction configuration as the redesigned three-object surface:
``GeolocationSetup`` (durable mission setup), ``Sweep`` (the parameter
experiment you vary between runs), and ``OutputConfig`` (output settings).
Use it as a reference when creating your own mission configuration.

Prefer loading from JSON in production::

    from curryer.correction import load_config_files
    setup, sweep, output = load_config_files("examples/correction/clarreo_config.json")

Use this Python factory when you want to build the config programmatically
or inject it into tests, and when you want to iterate quickly — hold ``setup``
fixed and cheaply vary ``sweep`` with ``Sweep.with_strategy`` /
``Sweep.update_param``.

Note
----
For the test fixture version, see ``tests/test_correction/clarreo/clarreo_config.py``.
That file is a test-infrastructure helper; this file is the user-facing example.
"""

from __future__ import annotations

import logging
from pathlib import Path

from curryer.correction import (
    GeolocationConfig,
    GeolocationSetup,
    NetCDFConfig,
    OutputConfig,
    ParameterConfig,
    ParameterType,
    RequirementsConfig,
    Sweep,
)

logger = logging.getLogger(__name__)

# Default repo-relative paths — override by passing data_dir / generic_dir
_DEFAULT_DATA_DIR = Path("tests/data/clarreo/gcs")
_DEFAULT_GENERIC_DIR = Path("data/generic")


def create_clarreo_config(
    data_dir: Path | None = None,
    generic_dir: Path | None = None,
) -> tuple[GeolocationSetup, Sweep, OutputConfig]:
    """Create the CLARREO Pathfinder ``(GeolocationSetup, Sweep, OutputConfig)``.

    The sweep defines 6 CLARREO-specific ``ParameterConfig`` entries covering 12
    underlying scalar correction values:
    - 3 CONSTANT_KERNEL entries (one per frame, each with a 3-angle attitude vector)
    - 2 OFFSET_KERNEL entries (azimuth and elevation angle biases)
    - 1 OFFSET_TIME entry (science timing correction)

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
    tuple
        ``(setup, sweep, output)`` — pass directly to
        ``run_correction(setup, sweep, inputs, work_dir, output)`` or
        ``verify(setup, ...)``.

    Examples
    --------
    >>> from examples.correction.clarreo_config import create_clarreo_config
    >>> setup, sweep, output = create_clarreo_config()
    >>> setup.requirements.performance_threshold_m
    250.0
    >>> len(sweep.parameters)
    6

    Hold the setup fixed and try a different experiment cheaply:

    >>> grid = sweep.with_strategy("grid", grid_points_per_param=5)
    >>> wider = sweep.update_param("hps.az_ang_nonlin", bounds=[-100.0, 100.0])

    Or load everything from the JSON file (preferred for production):

    >>> from curryer.correction import load_config_files
    >>> setup, sweep, output = load_config_files("examples/correction/clarreo_config.json")
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
            spec={
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
            spec={
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
            spec={
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
            spec={
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
            spec={
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
            spec={
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

    # Durable, mission-specific setup — built once, reused across many sweeps.
    setup = GeolocationSetup(
        geo=geo,
        requirements=RequirementsConfig(
            performance_threshold_m=250.0,  # Each measurement must be < 250 m nadir-equivalent
            performance_spec_percent=39.0,  # At least 39% of measurements must pass
        ),
        # Variable names in the image-matching xr.Dataset output (ISS/HySICS specific)
        spacecraft_position_name="riss_ctrs",
        boresight_name="bhat_hs",
        transformation_matrix_name="t_hs2ctrs",
    )

    # The lightweight experiment — vary this between runs.
    sweep = Sweep(
        seed=42,
        n_iterations=5,
        parameters=parameters,
    )

    output = OutputConfig(
        netcdf=NetCDFConfig(
            title="CLARREO Pathfinder Geolocation Correction Analysis",
            description="Parameter sensitivity analysis for CLARREO Pathfinder on ISS",
            performance_threshold_m=250.0,
        ),
    )

    logger.info("CLARREO config created: %d parameters", len(sweep.parameters))
    return setup, sweep, output
