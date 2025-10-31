#!/usr/bin/env python3
"""
CLARREO Monte Carlo GCS Test Runner

This script demonstrates the complete Monte Carlo Ground Control System (GCS) workflow
for CLARREO Pathfinder geolocation parameter optimization. It follows the test_scenario_5a
structure but is organized for easy understanding and modification.

GETTING STARTED:
1. Ensure data directories exist: tests/data/clarreo/gcs/ and data/generic/
2. Adjust parameters in tests/test_correction/clarreo_config.py if needed
3. Run: python run_monte_carlo.py
4. Review results in curryer/correction/monte_carlo_results/

USER CONFIGURATION POINTS:
- tests/test_correction/clarreo_config.py: Modify parameters, bounds, sigmas, iterations
- setup_directories(): Change data/output directory paths
- prepare_gcp_data_sets(): Add/modify data sets for processing

Usage:
    python run_monte_carlo.py

Author: Matthew Maclay
Date: October 7, 2025
"""

import logging
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

# Add curryer to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from curryer import utils, meta
from curryer import spicierpy as sp
from curryer.correction import monte_carlo as mc
from curryer.kernels import create

# Import CLARREO-specific config and data loaders
from clarreo_config import create_clarreo_monte_carlo_config
from clarreo_data_loaders import load_clarreo_telemetry, load_clarreo_science, load_clarreo_gcp

# Configuration and Setup Functions

def setup_logging():
    """Configure logging for detailed output."""
    utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])
    xr.set_options(display_width=120, display_max_rows=30)
    np.set_printoptions(linewidth=120)
    return logging.getLogger(__name__)

def setup_directories():
    """Setup working directories and validate paths.

    USER CONFIGURATION: Modify these paths to point to your data locations:
    - generic_dir: SPICE generic kernels (LSK, PCK, etc.)
    - data_dir: GCS test data (telemetry, science, kernel configs)
    - output_dir: Where results will be saved

    Raises:
        FileNotFoundError: If required data directories don't exist
        PermissionError: If unable to create output directory
    """
    logger = logging.getLogger(__name__)

    # Define directory paths
    root_dir = Path(__file__).parent.parent.parent
    generic_dir = root_dir / 'data' / 'generic'
    data_dir = root_dir / 'tests' / 'data' / 'clarreo' / 'gcs'

    # Validate directories exist with clear error messages
    if not generic_dir.is_dir():
        raise FileNotFoundError(
            f"Generic data directory not found: {generic_dir}\n"
            f"Please ensure SPICE generic kernels are available at this location."
        )
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"GCS data directory not found: {data_dir}\n"
            f"Please ensure test data files are available at this location."
        )

    # Create output directory with error handling
    output_dir = root_dir / 'tests' / 'test_correction' / 'monte_carlo_results'
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(
            f"Unable to create output directory: {output_dir}\n"
            f"Please check write permissions for this location."
        )

    logger.info(f"Generic data directory: {generic_dir}")
    logger.info(f"GCS data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    return generic_dir, data_dir, output_dir

# =============================================================================
# Monte Carlo Configuration
# =============================================================================
# NOTE: Configuration has been moved to tests/test_correction/clarreo_config.py
# The function below is kept for reference but is no longer used.
# Use create_clarreo_monte_carlo_config() from clarreo_config.py instead.
# =============================================================================

def create_monte_carlo_config(data_dir, generic_dir, config_output_path=None):
    """
    DEPRECATED: Use create_clarreo_monte_carlo_config() from clarreo_config.py instead.

    Create the Monte Carlo configuration with all 12 parameters.

    USER CONFIGURATION: This is the main place to adjust Monte Carlo parameters.
    You can modify:
    - current_value: Starting parameter values
    - bounds: Limits for parameter variation
    - sigma: Standard deviation for random sampling
    - n_iterations: Number of parameter sets to test

    This configuration defines:
    - 9 CONSTANT_KERNEL parameters (3 frames × 3 attitudes each)
    - 2 OFFSET_KERNEL parameters (azimuth and elevation biases)
    - 1 OFFSET_TIME parameter (timing correction)

    Args:
        data_dir: Path to the GCS data directory
        generic_dir: Path to the generic SPICE kernels directory
        config_output_path: Optional path to save the configuration as JSON file

    Returns:
        MonteCarloConfig object
    """
    logger = logging.getLogger(__name__)
    logger.info("=== CREATING MONTE CARLO CONFIGURATION ===")

    # Define the 12 parameters for CLARREO geolocation correction
    parameters = [
        # ===== CONSTANT_KERNEL Parameters (9 total) =====
        # These are fixed attitude corrections for instrument frames

        # BASE frame corrections (roll, pitch, yaw)
        mc.ParameterConfig(
            ptype=mc.ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_base_v01.attitude.ck.json",
            data=dict(
                current_value=[0.0, 0.0, 0.0],  # [roll, pitch, yaw] baseline values in arcseconds
                bounds=[-300.0, 300.0],  # Offset limits in arcseconds (around 0)
                sigma=30.0,  # Standard deviation for offset sampling (arcseconds)
                units='arcseconds',
                distribution='normal'
            ),
        ),

        # YOKE frame corrections (roll, pitch, yaw)
        mc.ParameterConfig(
            ptype=mc.ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_yoke_v01.attitude.ck.json",
            data=dict(
                current_value=[0.0, 0.0, 0.0],  # [roll, pitch, yaw] baseline values
                bounds=[-200.0, 200.0],  # Smaller offset range for yoke
                sigma=20.0,
                units='arcseconds',
                distribution='normal'
            ),
        ),

        # HYSICS frame corrections (roll, pitch, yaw)
        mc.ParameterConfig(
            ptype=mc.ParameterType.CONSTANT_KERNEL,
            config_file=data_dir / "cprs_hysics_v01.attitude.ck.json",
            data=dict(
                current_value=[0.0, 0.0, 0.0],  # [roll, pitch, yaw] baseline values
                bounds=[-300.0, 300.0],  # Offset range for HySICS instrument
                sigma=30.0,
                units='arcseconds',
                distribution='normal'
            ),
        ),

        # ===== OFFSET_KERNEL Parameters (2 total) =====
        # These are dynamic angle biases applied to telemetry

        # Azimuth angle bias correction
        mc.ParameterConfig(
            ptype=mc.ParameterType.OFFSET_KERNEL,
            config_file=data_dir / "cprs_az_v01.attitude.ck.json",
            data=dict(
                field="hps.az_ang_nonlin",  # Telemetry field to modify
                current_value=0.0,  # Baseline azimuth bias (arcseconds)
                bounds=[-300.0, 300.0],  # Offset limits in arcseconds (around 0)
                sigma=30.0,  # Standard deviation for offset sampling
                units='arcseconds',
                distribution='normal'
            ),
        ),

        # Elevation angle bias correction
        mc.ParameterConfig(
            ptype=mc.ParameterType.OFFSET_KERNEL,
            config_file=data_dir / "cprs_el_v01.attitude.ck.json",
            data=dict(
                field="hps.el_ang_nonlin",  # Telemetry field to modify
                current_value=0.0,  # Baseline elevation bias (arcseconds)
                bounds=[-300.0, 300.0],  # Offset limits in arcseconds (around 0)
                sigma=30.0,
                units='arcseconds',
                distribution='normal'
            ),
        ),

        # ===== OFFSET_TIME Parameters (1 total) =====
        # Timing corrections for science frames

        # Science frame timing correction
        mc.ParameterConfig(
            ptype=mc.ParameterType.OFFSET_TIME,
            config_file=None,  # No config file needed for time corrections
            data=dict(
                field="corrected_timestamp",  # Science timing field to modify
                current_value=0.0,  # Baseline timing offset (milliseconds)
                bounds=[-50.0, 50.0],  # Offset limits in milliseconds (around 0)
                sigma=7.0,  # Standard deviation for offset sampling
                units='milliseconds',
                distribution='normal'
            ),
        ),
    ]

    # Geolocation configuration
    geo_config = mc.GeolocationConfig(
        meta_kernel_file=data_dir / 'cprs_v01.kernels.tm.json',
        generic_kernel_dir=generic_dir,  # Use the correct generic_dir variable
        dynamic_kernels=[
            data_dir / "iss_sc_v01.ephemeris.spk.json",  # Spacecraft position kernel
            data_dir / "iss_sc_v01.attitude.ck.json",   # Spacecraft attitude kernel
            data_dir / "cprs_st_v01.attitude.ck.json",  # Star tracker attitude kernel
        ],
        instrument_name='CPRS_HYSICS',
        time_field='corrected_timestamp',
    )

    # Create complete Monte Carlo configuration
    config = mc.MonteCarloConfig(
        seed=42,  # For reproducible results
        n_iterations=5,  # Number of parameter sets to test
        parameters=parameters,
        geo=geo_config,
    )

    logger.info(f"Configuration created with {len(parameters)} parameters:")
    for i, param in enumerate(parameters):
        param_name = param.config_file.name if param.config_file else "time_correction"
        logger.info(f"  {i+1}. {param_name} ({param.ptype.name})")

    # Save configuration to file if path is provided
    if config_output_path is not None:
        import json
        from pathlib import Path

        config_output_path = Path(config_output_path)
        logger.info(f"Saving configuration to: {config_output_path}")

        # Convert configuration to JSON-serializable format
        config_dict = {
            "monte_carlo": {
                "seed": config.seed,
                "n_iterations": config.n_iterations,
                "parameters": []
            },
            "geolocation": {
                "meta_kernel_file": str(config.geo.meta_kernel_file),
                "generic_kernel_dir": str(config.geo.generic_kernel_dir),
                "dynamic_kernels": [str(k) for k in config.geo.dynamic_kernels],
                "instrument_name": config.geo.instrument_name,
                "time_field": config.geo.time_field
            }
        }

        # Convert parameters to JSON format
        for i, param in enumerate(config.parameters):
            param_dict = {
                "name": param.config_file.name if param.config_file else f"time_correction_{i}",
                "parameter_type": param.ptype.name,
                "config_file": str(param.config_file) if param.config_file else None,
                "current_value": param.data.get('current_value'),
                "bounds": param.data.get('bounds'),
                "sigma": param.data.get('sigma'),
                "units": param.data.get('units'),
                "distribution": param.data.get('distribution'),
                "field": param.data.get('field')
            }
            config_dict["monte_carlo"]["parameters"].append(param_dict)

        # Write to file
        config_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to: {config_output_path}")
        logger.info(f"File size: {config_output_path.stat().st_size / 1024:.1f} KB")

    return config

# =============================================================================
# Static Kernel Creation
# =============================================================================

def create_static_kernels(data_dir, generic_dir, work_dir):
    """
    Create the static SPICE kernels required for geolocation.
    These kernels contain fixed instrument and spacecraft geometry.
    """
    logger = logging.getLogger(__name__)
    logger.info("=== CREATING STATIC KERNELS ===")

    # Load meta kernel to initialize SPICE environment
    mkrn = meta.MetaKernel.from_json(
        data_dir / 'cprs_v01.kernels.tm.json',
        relative=True,
        sds_dir=generic_dir,
    )
    logger.info(f"Loaded meta kernel: {mkrn}")

    # Define static kernel configurations
    static_kernel_configs = [
        data_dir / "cprs_st_v01.fixed_offset.spk.json",      # Star tracker
        data_dir / "cprs_base_v01.fixed_offset.spk.json",    # Base frame
        data_dir / "cprs_pede_v01.fixed_offset.spk.json",    # Pedestal
        data_dir / "cprs_az_v01.fixed_offset.spk.json",      # Azimuth
        data_dir / "cprs_yoke_v01.fixed_offset.spk.json",    # Yoke
        data_dir / "cprs_el_v01.fixed_offset.spk.json",      # Elevation
        data_dir / "cprs_hysics_v01.fixed_offset.spk.json",  # HySICS instrument
    ]

    # Create kernel writer
    creator = create.KernelCreator(overwrite=True, append=False)

    # Generate static kernels
    generated_kernels = []
    for config_file in static_kernel_configs:
        if not config_file.is_file():
            logger.warning(f"Static kernel config not found: {config_file}")
            continue

        kernel_path = creator.write_from_json(config_file, output_kernel=work_dir)
        generated_kernels.append(kernel_path)
        logger.info(f"Created static kernel: {kernel_path.name}")

    logger.info(f"Created {len(generated_kernels)} static kernels")
    return generated_kernels

# =============================================================================
# Data Preparation
# =============================================================================

def prepare_gcp_data_sets(data_dir):
    """
    Prepare the list of telemetry, science, and GCP data sets for processing.
    Each tuple represents one image-to-reference pair for geolocation testing.

    USER CONFIGURATION: Add more data sets here as they become available.
    Format: (telemetry_file, science_file, gcp_reference_file)
    """
    logger = logging.getLogger(__name__)
    logger.info("=== PREPARING GCP DATA SETS ===")

    # For this test, we'll use the 5a scenario data
    # In a real system, this would come from the GCP pairing module
    tlm_sci_gcp_sets = [
        (
            str(data_dir / "openloop_tlm_5a_sc_spk_20250521T225242.csv"),  # Telemetry key
            str(data_dir / "openloop_tlm_5a_sci_times_20250521T225242.csv"),  # Science key
            "synthetic_gcp_001.tif"  # GCP reference (placeholder)
        ),
        # Add more data sets here as they become available
    ]

    logger.info(f"Prepared {len(tlm_sci_gcp_sets)} GCP data sets:")
    for i, (tlm, sci, gcp) in enumerate(tlm_sci_gcp_sets):
        logger.info(f"  {i+1}. TLM: {Path(tlm).name}, SCI: {Path(sci).name}, GCP: {gcp}")

    return tlm_sci_gcp_sets

# =============================================================================
# Monte Carlo Execution
# =============================================================================

def run_monte_carlo_analysis(config, work_dir, tlm_sci_gcp_sets):
    """
    Execute the complete Monte Carlo analysis workflow.

    This function orchestrates:
    1. Parameter set generation
    2. Data loading and processing
    3. Kernel creation and modification
    4. Geolocation computation
    5. Image matching and error analysis
    6. Results compilation
    """
    logger = logging.getLogger(__name__)
    logger.info("=== RUNNING MONTE CARLO ANALYSIS ===")
    logger.info(f"Configuration: {config.n_iterations} iterations, {len(config.parameters)} parameters")
    logger.info(f"Data sets: {len(tlm_sci_gcp_sets)} GCP pairs")

    # Execute the main Monte Carlo loop with CLARREO-specific data loaders (Phase 3)
    results, netcdf_data = mc.loop(
        config,
        work_dir,
        tlm_sci_gcp_sets,
        telemetry_loader=load_clarreo_telemetry,
        science_loader=load_clarreo_science,
        gcp_loader=load_clarreo_gcp
    )

    logger.info("=== MONTE CARLO ANALYSIS COMPLETE ===")
    logger.info(f"Processed {len(results)} total iterations")
    logger.info(f"Generated results for {len(netcdf_data['parameter_set_id'])} parameter sets")

    return results, netcdf_data

# =============================================================================
# Results Analysis
# =============================================================================

def analyze_results(netcdf_data, config):
    """
    Analyze the Monte Carlo results and identify the best parameter set.

    Args:
        netcdf_data: Dictionary with NetCDF variables and data
        config: MonteCarloConfig with performance threshold and metadata
    """
    logger = logging.getLogger(__name__)
    logger.info("=== ANALYZING RESULTS ===")

    # Find the best parameter set based on mean RMS error
    mean_rms_errors = netcdf_data['mean_rms_all_pairs']
    valid_errors = mean_rms_errors[~np.isnan(mean_rms_errors)]

    if len(valid_errors) == 0:
        logger.error("No valid results found!")
        return None

    best_idx = np.nanargmin(mean_rms_errors)
    best_rms = mean_rms_errors[best_idx]

    # Use dynamic threshold metric name (Phase 2)
    threshold_metric = None
    for key in netcdf_data.keys():
        if key.startswith('percent_under_') and key.endswith('m'):
            threshold_metric = key
            break

    if threshold_metric:
        best_percent_under_threshold = netcdf_data[threshold_metric][best_idx]
        threshold_value = config.performance_threshold_m
        logger.info(f"Best parameter set: #{best_idx}")
        logger.info(f"  Mean RMS error: {best_rms:.2f} meters")
        logger.info(f"  Percent under {threshold_value}m: {best_percent_under_threshold:.1f}%")
        logger.info(f"  Meets CLARREO requirement: {'YES' if best_percent_under_threshold >= 39.0 else 'NO'}")
    else:
        logger.warning("Threshold metric not found in netcdf_data")
        logger.info(f"Best parameter set: #{best_idx}")
        logger.info(f"  Mean RMS error: {best_rms:.2f} meters")

    # Log best parameter values dynamically (Phase 2)
    logger.info("Best parameter values:")

    # Get all parameter variable names from netcdf_data
    param_vars = sorted([k for k in netcdf_data.keys() if k.startswith('param_')])

    for param_var in param_vars:
        value = netcdf_data[param_var][best_idx]
        if not np.isnan(value):
            # Clean up variable name for display
            display_name = param_var.replace('param_', '').replace('_', ' ')
            logger.info(f"  {display_name}: {value:.6f}")

    # Performance summary
    logger.info("\nPerformance Summary:")
    logger.info(f"  Best RMS error: {netcdf_data['best_pair_rms'][best_idx]:.2f}m")
    logger.info(f"  Worst RMS error: {netcdf_data['worst_pair_rms'][best_idx]:.2f}m")
    logger.info(f"  Mean across all pairs: {best_rms:.2f}m")

    return best_idx

def save_results(netcdf_data, work_dir, config):
    """
    Save results to files and optionally update configuration.
    """
    logger = logging.getLogger(__name__)
    logger.info("=== SAVING RESULTS ===")

    # Results are already saved to NetCDF by the monte_carlo.loop function
    results_file = work_dir / "monte_carlo_results.nc"
    if results_file.exists():
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"File size: {results_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        logger.warning("NetCDF results file not found!")

    # TODO: Implement saving best parameters back to gcs_config.json
    # This would be done with mc.save_best_parameters_to_config() once implemented

    return results_file

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main execution function that orchestrates the complete Monte Carlo GCS workflow.

    This function runs the complete workflow:
    1. Setup logging and directories
    2. Create Monte Carlo configuration
    3. Generate static kernels
    4. Prepare data sets
    5. Execute Monte Carlo analysis
    6. Analyze and save results

    Results are saved to: curryer/correction/monte_carlo_results/
    """
    print("="*80)
    print("CLARREO MONTE CARLO GCS TEST RUNNER")
    print("="*80)

    try:
        # Setup
        logger = setup_logging()
        generic_dir, data_dir, work_dir = setup_directories()

        # Configuration
        logger.info("Creating CLARREO Monte Carlo configuration...")
        config = create_clarreo_monte_carlo_config(
            data_dir,
            generic_dir,
            config_output_path=Path(__file__).parent / 'configs' / 'run_monte_carlo_config.json'
        )

        # Static kernels
        static_kernels = create_static_kernels(data_dir, generic_dir, work_dir)

        # Data preparation
        tlm_sci_gcp_sets = prepare_gcp_data_sets(data_dir)

        # Monte Carlo execution
        results, netcdf_data = run_monte_carlo_analysis(config, work_dir, tlm_sci_gcp_sets)

        # Results analysis (Phase 2: Pass config for dynamic threshold)
        best_idx = analyze_results(netcdf_data, config)

        # Save results
        results_file = save_results(netcdf_data, work_dir, config)

        # Final summary
        print("\n" + "="*80)
        print("MONTE CARLO GCS ANALYSIS COMPLETE")
        print("="*80)
        print(f"Working directory: {work_dir}")
        print(f"Results file: {results_file}")
        if best_idx is not None:
            best_rms = netcdf_data['mean_rms_all_pairs'][best_idx]

            # Find threshold metric dynamically (Phase 2)
            threshold_metric = None
            for key in netcdf_data.keys():
                if key.startswith('percent_under_') and key.endswith('m'):
                    threshold_metric = key
                    break

            if threshold_metric:
                best_percent = netcdf_data[threshold_metric][best_idx]
                threshold_value = config.performance_threshold_m
                print(f"Best parameter set: #{best_idx} (RMS: {best_rms:.2f}m, {best_percent:.1f}% under {threshold_value}m)")
            else:
                print(f"Best parameter set: #{best_idx} (RMS: {best_rms:.2f}m)")
        print("="*80)

        return 0

    except Exception as e:
        logger.error(f"Monte Carlo analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
