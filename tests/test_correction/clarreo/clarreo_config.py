#!/usr/bin/env python3
"""CLARREO-specific Correction configuration generation — **test fixture**.

This module is a **test infrastructure helper**. It is not part of the
public API and is not intended as a user-facing example.

For user-facing examples and documentation, see:
  - ``examples/correction/clarreo_config.py``    — public-API config factory
  - ``examples/correction/clarreo_config.json``  — JSON config (loadable)
  - ``examples/correction/example_verification.py`` — runnable demo
  - ``docs/source/correction_user_guide.md``     — full reference

This module creates the CLARREO CorrectionConfig used by integration tests.
All CLARREO-specific values (kernel names, parameters, instruments) are
defined here and injected into tests via fixtures.

Usage:
    from tests.test_correction.clarreo_config import create_clarreo_correction_config

    data_dir = Path('tests/data/clarreo/gcs')
    generic_dir = Path('data/generic')
    config = create_clarreo_correction_config(data_dir, generic_dir)

    # Optionally save to JSON
    config_path = Path('tests/test_correction/configs/clarreo_correction_config.json')
    create_clarreo_correction_config(data_dir, generic_dir, config_output_path=config_path)

Author: Matthew Maclay
Date: October 27, 2025
"""

import json
import logging
from pathlib import Path

from curryer.correction import correction

logger = logging.getLogger(__name__)


def create_clarreo_correction_config(data_dir, generic_dir, config_output_path=None):
    """
    Create the CLARREO Correction configuration with all 12 parameters.

    This is mission-specific configuration. It defines all CLARREO-specific
    kernel files, parameter ranges, and iteration settings.

    USER CONFIGURATION: This is the main place to adjust Correction parameters.
    You can modify:
    - current_value: Starting parameter values
    - bounds: Limits for parameter variation
    - sigma: Standard deviation for random sampling
    - n_iterations: Number of parameter sets to test

    This configuration defines:
    - 9 CONSTANT_KERNEL parameters (3 frames × 3 attitudes each)
    - 2 OFFSET_KERNEL parameters (azimuth and elevation biases)
    - 1 OFFSET_TIME parameter (timing correction)

    Returns:
        CorrectionConfig: THE single config object containing all CLARREO settings.

    IMPORTANT - Config-Centric Design:
        This function creates the base CorrectionConfig with all CLARREO-specific
        parameters and settings.

        After creation, attach a :class:`~curryer.correction.config.DataConfig`
        and (optionally) an ``_image_matching_override`` for test injection::

            config = create_clarreo_correction_config(data_dir, generic_dir)
            config.data_config = DataConfig(file_format="csv", time_scale_factor=1e6)
            config._image_matching_override = your_matching_func  # optional (tests only)
            results = correction.loop(config, work_dir, data_sets)

    Args:
        data_dir: Path to the CLARREO GCS data directory
        generic_dir: Path to the generic SPICE kernels directory
        config_output_path: Optional path to save the configuration as JSON file

    Returns:
        CorrectionConfig object (you must add loaders before calling correction.loop())

    Example:
        config = create_clarreo_correction_config(data_dir, generic_dir)

        # Attach data loading config
        from curryer.correction.config import DataConfig
        config.data_config = DataConfig(file_format="csv", time_scale_factor=1e6)

        # Optionally override image matching (tests only)
        config._image_matching_override = synthetic_image_matching

        # Now ready to use
        results = correction.loop(config, work_dir, data_sets)
    """
    logger.info("=== CREATING CLARREO CORRECTION CONFIGURATION ===")

    # Define the 12 parameters for CLARREO geolocation correction
    parameters = [
        # ===== CONSTANT_KERNEL Parameters (9 total) =====
        # These are fixed attitude corrections for instrument frames
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
        # These are dynamic angle biases applied to telemetry
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
        # Timing corrections for science frames
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

    # NetCDF output configuration (NEW - Phase 1)
    netcdf_config = correction.NetCDFConfig(
        title="CLARREO Pathfinder Geolocation Correction Analysis",
        description="Parameter sensitivity analysis for CLARREO Pathfinder on ISS",
        performance_threshold_m=250.0,  # CLARREO requirement
        parameter_metadata=None,  # Auto-generate from parameters
    )

    # Create complete Correction configuration
    config = correction.CorrectionConfig(
        seed=42,  # For reproducible results
        n_iterations=5,  # Number of parameter sets to test
        parameters=parameters,
        geo=geo_config,
        # Performance metrics (CLARREO requirements)
        performance_threshold_m=250.0,  # CLARREO accuracy requirement (meters)
        performance_spec_percent=39.0,  # CLARREO requirement: 39% of measurements under threshold
        # NetCDF output configuration
        netcdf=netcdf_config,
        # Calibration file names (CLARREO/HySICS specific)
        calibration_file_names={
            "los_vectors": "b_HS.mat",  # HySICS boresight vectors
            "optical_psf": "optical_PSF_675nm_upsampled.mat",  # 675nm wavelength PSF
        },
        # Coordinate variable names (ISS/HySICS specific for backward compatibility)
        spacecraft_position_name="riss_ctrs",  # ISS position in CTRS frame
        boresight_name="bhat_hs",  # HySICS boresight
        transformation_matrix_name="t_hs2ctrs",  # HySICS to CTRS transformation
    )

    logger.info(f"CLARREO configuration created with {len(parameters)} parameters:")
    for i, param in enumerate(parameters):
        param_name = param.config_file.name if param.config_file else "time_correction"
        logger.info(f"  {i + 1}. {param_name} ({param.ptype.name})")

    # Save configuration to file if path is provided
    if config_output_path is not None:
        config_output_path = Path(config_output_path)
        logger.info(f"Saving CLARREO configuration to: {config_output_path}")

        # Convert configuration to JSON-serializable format
        config_dict = {
            "mission_config": {
                "mission_name": "CLARREO_Pathfinder",
                "instrument_name": "CPRS_HYSICS",
                "kernel_mappings": {
                    "constant_kernel": {
                        "hysics": "cprs_hysics_v01.attitude.ck.json",
                        "yoke": "cprs_yoke_v01.attitude.ck.json",
                        "base": "cprs_base_v01.attitude.ck.json",
                    },
                    "offset_kernel": {
                        "azimuth": "cprs_az_v01.attitude.ck.json",
                        "elevation": "cprs_el_v01.attitude.ck.json",
                    },
                },
            },
            "correction": {
                "seed": config.seed,
                "n_iterations": config.n_iterations,
                "performance_threshold_m": config.performance_threshold_m,
                "performance_spec_percent": config.performance_spec_percent,
                "parameters": [],
            },
            "geolocation": {
                "meta_kernel_file": str(config.geo.meta_kernel_file),
                "generic_kernel_dir": str(config.geo.generic_kernel_dir),
                "dynamic_kernels": [str(k) for k in config.geo.dynamic_kernels],
                "instrument_name": config.geo.instrument_name,
                "time_field": config.geo.time_field,
            },
        }

        # Convert parameters to JSON format
        for i, param in enumerate(config.parameters):
            param_dict = {
                "name": param.config_file.name if param.config_file else f"time_correction_{i}",
                "parameter_type": param.ptype.name,
                "config_file": str(param.config_file) if param.config_file else None,
                "current_value": param.spec.get("current_value"),
                "bounds": param.spec.get("bounds"),
                "sigma": param.spec.get("sigma"),
                "units": param.spec.get("units"),
                "distribution": param.spec.get("distribution"),
                "field": param.spec.get("field"),
            }

            # Add optional fields if they exist
            if "transformation_type" in param.spec:
                param_dict["transformation_type"] = param.spec["transformation_type"]
            if "coordinate_frames" in param.spec:
                param_dict["coordinate_frames"] = param.spec["coordinate_frames"]

            config_dict["correction"]["parameters"].append(param_dict)

        # Write to file
        config_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_output_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to: {config_output_path}")
        logger.info(f"File size: {config_output_path.stat().st_size / 1024:.1f} KB")

    return config
