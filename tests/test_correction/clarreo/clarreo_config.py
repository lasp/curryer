#!/usr/bin/env python3
"""CLARREO-specific correction configuration generation — **test fixture**.

This module is a **test infrastructure helper**. It is not part of the
public API and is not intended as a user-facing example.

For user-facing examples and documentation, see:
  - ``examples/correction/clarreo_config.py``    — public-API config factory
  - ``examples/correction/clarreo_config.json``  — JSON config (loadable)
  - ``examples/correction/example_verification.py`` — runnable demo
  - ``docs/source/correction_user_guide.md``     — full reference

This module builds the CLARREO ``GeolocationSetup`` / ``Sweep`` / ``OutputConfig``
used by integration tests.  All CLARREO-specific values (kernel names,
parameters, instruments) are defined here and injected into tests via fixtures.

Usage::

    from clarreo_config import create_clarreo_setup_sweep

    data_dir = Path('tests/data/clarreo/gcs')
    generic_dir = Path('data/generic')
    setup, sweep, output = create_clarreo_setup_sweep(data_dir, generic_dir)
"""

import logging

from curryer.correction import correction

logger = logging.getLogger(__name__)


def create_clarreo_setup_sweep(data_dir, generic_dir):
    """Create the CLARREO ``(GeolocationSetup, Sweep, OutputConfig)`` test fixture.

    Defines all 12 CLARREO geolocation-correction parameters (9 CONSTANT_KERNEL,
    2 OFFSET_KERNEL, 1 OFFSET_TIME), the SPICE/instrument setup, the pass/fail
    requirements, and the NetCDF output metadata.

    Parameters
    ----------
    data_dir : Path
        Path to the CLARREO GCS data directory.
    generic_dir : Path
        Path to the generic SPICE kernels directory.

    Returns
    -------
    tuple
        ``(GeolocationSetup, Sweep, OutputConfig)``.
    """
    logger.info("=== CREATING CLARREO CORRECTION CONFIGURATION ===")

    # Define the 12 parameters for CLARREO geolocation correction
    parameters = [
        # ===== CONSTANT_KERNEL Parameters (9 total) =====
        # BASE frame corrections (roll, pitch, yaw)
        correction.ParameterConfig(
            ptype=correction.ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_base_v01.attitude.ck.json",
            spec=dict(
                current_value=[0.0, 0.0, 0.0],  # [roll, pitch, yaw] baseline values in arcseconds
                bounds=[-300.0, 300.0],  # Offset limits in arcseconds (around 0)
                sigma=30.0,  # Standard deviation for offset sampling (arcseconds)
                units="arcseconds",
                distribution="normal",
                transformation_type="dcm_rotation",
                coordinate_frames=["BASE_AZIMUTH", "BASE_CUBE"],
            ),
        ),
        # YOKE frame corrections (roll, pitch, yaw)
        correction.ParameterConfig(
            ptype=correction.ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_yoke_v01.attitude.ck.json",
            spec=dict(
                current_value=[0.0, 0.0, 0.0],  # [roll, pitch, yaw] baseline values
                bounds=[-200.0, 200.0],  # Smaller offset range for yoke
                sigma=20.0,
                units="arcseconds",
                distribution="normal",
                transformation_type="dcm_rotation",
                coordinate_frames=["YOKE_ELEVATION", "YOKE_AZIMUTH"],
            ),
        ),
        # HYSICS frame corrections (roll, pitch, yaw)
        correction.ParameterConfig(
            ptype=correction.ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_hysics_v01.attitude.ck.json",
            spec=dict(
                current_value=[0.0, 0.0, 0.0],  # [roll, pitch, yaw] baseline values
                bounds=[-300.0, 300.0],  # Offset range for HySICS instrument
                sigma=30.0,
                units="arcseconds",
                distribution="normal",
                transformation_type="dcm_rotation",
                coordinate_frames=["HYSICS_SLIT", "CRADLE_ELEVATION"],
            ),
        ),
        # ===== OFFSET_KERNEL Parameters (2 total) =====
        # Azimuth angle bias correction
        correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=data_dir / "cprs_az_v01.attitude.ck.json",
            spec=dict(
                field="hps.az_ang_nonlin",  # Telemetry field to modify
                current_value=0.0,  # Baseline azimuth bias (arcseconds)
                bounds=[-300.0, 300.0],  # Offset limits in arcseconds (around 0)
                sigma=30.0,  # Standard deviation for offset sampling
                units="arcseconds",
                distribution="normal",
                transformation_type="angle_bias",
                coordinate_frames=["YOKE_AZIMUTH"],
            ),
        ),
        # Elevation angle bias correction
        correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_KERNEL,
            config_file=data_dir / "cprs_el_v01.attitude.ck.json",
            spec=dict(
                field="hps.el_ang_nonlin",  # Telemetry field to modify
                current_value=0.0,  # Baseline elevation bias (arcseconds)
                bounds=[-300.0, 300.0],  # Offset limits in arcseconds (around 0)
                sigma=30.0,
                units="arcseconds",
                distribution="normal",
                transformation_type="angle_bias",
                coordinate_frames=["YOKE_ELEVATION"],
            ),
        ),
        # ===== OFFSET_TIME Parameters (1 total) =====
        # Science frame timing correction
        correction.ParameterConfig(
            ptype=correction.ParameterType.OFFSET_TIME,
            config_file=None,  # No config file needed for time corrections
            spec=dict(
                field="corrected_timestamp",  # Science timing field to modify
                current_value=0.0,  # Baseline timing offset (milliseconds)
                bounds=[-50.0, 50.0],  # Offset limits in milliseconds (around 0)
                sigma=7.0,  # Standard deviation for offset sampling
                units="milliseconds",
                distribution="normal",
            ),
        ),
    ]

    # Geolocation configuration
    geo_config = correction.GeolocationConfig(
        meta_kernel_file=data_dir / "cprs_v01.kernels.tm.json",
        generic_kernel_dir=generic_dir,
        dynamic_kernels=[
            data_dir / "iss_sc_v01.ephemeris.spk.json",  # Spacecraft position kernel
            data_dir / "iss_sc_v01.attitude.ck.json",  # Spacecraft attitude kernel
            data_dir / "cprs_st_v01.attitude.ck.json",  # Star tracker attitude kernel
        ],
        instrument_name="CPRS_HYSICS",
        time_field="corrected_timestamp",
    )

    # NetCDF output configuration
    netcdf_config = correction.NetCDFConfig(
        title="CLARREO Pathfinder Geolocation Correction Analysis",
        description="Parameter sensitivity analysis for CLARREO Pathfinder on ISS",
        performance_threshold_m=250.0,  # CLARREO requirement
        parameter_metadata=None,  # Auto-generate from parameters
    )

    setup = correction.GeolocationSetup(
        geo=geo_config,
        requirements=correction.RequirementsConfig(
            performance_threshold_m=250.0,  # CLARREO accuracy requirement (meters)
            performance_spec_percent=39.0,  # 39% of measurements under threshold
        ),
        # Coordinate variable names (ISS/HySICS specific for backward compatibility)
        spacecraft_position_name="riss_ctrs",  # ISS position in CTRS frame
        boresight_name="bhat_hs",  # HySICS boresight
        transformation_matrix_name="t_hs2ctrs",  # HySICS to CTRS transformation
    )

    sweep = correction.Sweep(
        seed=42,  # For reproducible results
        n_iterations=5,  # Number of parameter sets to test
        parameters=parameters,
    )

    output = correction.OutputConfig(netcdf=netcdf_config)

    logger.info(f"CLARREO configuration created with {len(parameters)} parameters:")
    for i, param in enumerate(parameters):
        param_name = param.config_file.name if param.config_file else "time_correction"
        logger.info(f"  {i + 1}. {param_name} ({param.ptype.name})")

    return setup, sweep, output
