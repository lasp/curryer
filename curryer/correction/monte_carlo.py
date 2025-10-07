import logging
import typing
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from curryer import meta
from curryer import spicierpy as sp
from curryer.compute import spatial
from curryer.kernels import create


logger = logging.getLogger(__name__)


# ============================================================================
# GCS MODULE PLACEHOLDERS
# ============================================================================

def placeholder_gcp_pairing(clarreo_data_files):
    """
    PLACEHOLDER for GCP pairing module.

    Real implementation will:
    - Take CLARREO L1 image files
    - Find spatially/temporally overlapping Landsat GCP scenes
    - Return list of (clarreo_file, gcp_file) pairs

    For now: Return synthetic pairs for testing
    """
    logger.info("GCP Pairing: Finding overlapping image pairs (PLACEHOLDER)")
    # Simulate finding GCP pairs
    synthetic_pairs = [(f"clarreo_{i:03d}.nc", f"landsat_gcp_{i:03d}.tif") for i in range(len(clarreo_data_files))]
    return synthetic_pairs


def placeholder_image_matching(geolocated_data, gcp_reference_file, params_info):
    """
    PLACEHOLDER for image matching module.

    Real implementation will:
    - Compare CLARREO geolocated pixels with Landsat GCP references
    - Perform image correlation/matching
    - Return spatial errors in format expected by error_stats module

    For now: Generate synthetic error data matching error_stats test format
    """
    logger.info(f"Image Matching: Comparing geolocated pixels with {gcp_reference_file} (PLACEHOLDER)")

    # Extract valid geolocation points (non-NaN)
    valid_mask = ~np.isnan(geolocated_data['latitude'].values).any(axis=1)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        logger.warning("No valid geolocation points found for image matching")
        n_measurements = 10  # Minimum synthetic measurements
    else:
        n_measurements = min(n_valid, 100)  # Limit for testing

    # Generate realistic transformation matrices (from error_stats tests)
    t_matrices = np.zeros((3, 3, n_measurements))
    for i in range(n_measurements):
        if i % 3 == 0:
            t_matrices[:, :, i] = np.eye(3)  # Identity
        elif i % 3 == 1:
            t_matrices[:, :, i] = [[0.9, 0.1, 0], [-0.1, 0.9, 0], [0, 0, 1]]  # Simple rotation
        else:
            t_matrices[:, :, i] = [[0.8, 0, 0.2], [0, 1, 0], [-0.2, 0, 0.8]]  # Another rotation

    # Generate synthetic errors based on parameter variations
    # Errors should vary based on how far parameters are from optimal values
    base_error = 50.0  # Base error in meters
    param_contribution = sum(abs(p) if isinstance(p, (int, float)) else np.linalg.norm(p)
                           for _, p in params_info) * 10.0  # Scale parameter deviations

    error_magnitude = base_error + param_contribution

    # Generate errors with spatial correlation
    lat_errors = np.random.normal(0, error_magnitude / 111000, n_measurements)  # Convert m to degrees
    lon_errors = np.random.normal(0, error_magnitude / 111000, n_measurements)

    # Generate realistic boresight vectors
    boresights = _generate_realistic_boresights(n_measurements)

    # Generate spacecraft position vectors (ISS altitude ~400km)
    riss_ctrs = np.random.uniform(6378e3, 6778e3, (n_measurements, 3))  # Earth radius + altitude

    # Extract corresponding geolocation data
    if n_valid > 0:
        valid_indices = np.where(valid_mask)[0][:n_measurements]
        cp_lat = geolocated_data['latitude'].values[valid_indices, 0]  # Use first pixel
        cp_lon = geolocated_data['longitude'].values[valid_indices, 0]
    else:
        cp_lat = np.random.uniform(-60, 60, n_measurements)
        cp_lon = np.random.uniform(-180, 180, n_measurements)

    cp_alt = np.random.uniform(0, 1000, n_measurements)

    return xr.Dataset({
        'lat_error_deg': (['measurement'], lat_errors),
        'lon_error_deg': (['measurement'], lon_errors),
        'riss_ctrs': (['measurement', 'xyz'], riss_ctrs),
        'bhat_hs': (['measurement', 'xyz'], boresights),
        't_hs2ctrs': (['xyz_from', 'xyz_to', 'measurement'], t_matrices),
        'cp_lat_deg': (['measurement'], cp_lat),
        'cp_lon_deg': (['measurement'], cp_lon),
        'cp_alt': (['measurement'], cp_alt)
    }, coords={
        'measurement': range(n_measurements),
        'xyz': ['x', 'y', 'z'],
        'xyz_from': ['x', 'y', 'z'],
        'xyz_to': ['x', 'y', 'z']
    })


def _generate_realistic_boresights(n_measurements):
    """Generate realistic boresight vectors for testing."""
    boresights = np.zeros((n_measurements, 3))
    for i in range(n_measurements):
        # Small off-nadir angles (typical for Earth observation)
        theta = np.random.uniform(0, 0.1)  # 0-6 degrees off-nadir
        phi = np.random.uniform(0, 2*np.pi)
        boresights[i] = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    return boresights


def call_error_stats_module(image_matching_output):
    """
    Call the REAL error_stats module with image matching output.

    This demonstrates the actual integration with the completed error_stats module.
    """
    try:
        from curryer.correction.error_stats.geolocation_error_stats import ErrorStatsProcessor

        logger.info("Error Statistics: Processing geolocation errors (REAL MODULE)")
        processor = ErrorStatsProcessor()
        error_results = processor.process_geolocation_errors(image_matching_output)

        return error_results
    except ImportError as e:
        logger.warning(f"Error stats module not available: {e}")
        logger.info("Error Statistics: Using placeholder calculations")

        # Fallback: compute basic statistics
        lat_errors = image_matching_output['lat_error_deg'].values
        lon_errors = image_matching_output['lon_error_deg'].values

        # Convert to meters (approximate)
        lat_error_m = lat_errors * 111000
        lon_error_m = lon_errors * 111000
        total_error_m = np.sqrt(lat_error_m**2 + lon_error_m**2)

        mean_error = float(np.mean(total_error_m))
        rms_error = float(np.sqrt(np.mean(total_error_m**2)))
        std_error = float(np.std(total_error_m))

        return xr.Dataset({
            'mean_error': mean_error,
            'rms_error': rms_error,
            'std_error': std_error,
            'max_error': float(np.max(total_error_m)),
            'min_error': float(np.min(total_error_m)),
        }, attrs={
            'mean_error_m': mean_error,
            'rms_error_m': rms_error,
            'total_measurements': len(lat_errors),
            'performance_threshold_m': 300.0,  # CLARREO requirement
        })


# ============================================================================
# ORIGINAL FUNCTIONS
# ============================================================================

class ParameterType(Enum):
    CONSTANT_KERNEL = auto()  # Set a specific value.
    OFFSET_KERNEL = auto()  # Modify input kernel data by an offset.
    OFFSET_TIME = auto()  # Modify input timetags by an offset


@dataclass
class ParameterConfig:
    ptype: ParameterType
    config_file: typing.Optional[Path]
    data: typing.Any


@dataclass
class GeolocationConfig:
    meta_kernel_file: Path
    generic_kernel_dir: Path
    dynamic_kernels: [Path]  # Kernels that are dynamic but *NOT* altered by param!
    instrument_name: str
    time_field: str


@dataclass
class MonteCarloConfig:
    seed: typing.Optional[int]  # Used to make param results reproducible.
    n_iterations: int
    parameters: typing.List[ParameterConfig]
    geo: GeolocationConfig
    # match: ImageMatchConfig
    # stats: ErrorStatsConfig


def load_param_sets(config: MonteCarloConfig) -> [ParameterConfig, typing.Any]:
    """
    Generate random parameter sets for Monte Carlo iterations.
    Each parameter is sampled according to its distribution and bounds.
    """

    if config.seed is not None:
        np.random.seed(config.seed)

    output = []
    for ith in range(config.n_iterations):
        out_set = []
        for param in config.parameters:
            center = param.data['center']
            arange = param.data['arange']

            # Generate random values based on parameter type
            if isinstance(center, list):  # Multi-dimensional parameter (e.g., Euler angles)
                param_vals = []
                for i, c in enumerate(center):
                    # Use normal distribution sampling within bounds if sigma provided
                    if 'sigma' in param.data:
                        # Convert arcsec to radians for sampling
                        sigma_rad = np.deg2rad(param.data['sigma'] / 3600.0)  # arcsec to radians
                        offset = np.random.normal(0, sigma_rad)
                        # Clip to bounds
                        bound_rad = [np.deg2rad(arange[0] / 3600.0), np.deg2rad(arange[1] / 3600.0)]
                        offset = np.clip(offset, bound_rad[0], bound_rad[1])
                    else:
                        # Fallback to uniform distribution
                        offset = np.random.uniform(arange[0], arange[1])
                    param_vals.append(c + offset)
            else:  # Single parameter
                if 'sigma' in param.data:
                    # Use appropriate sigma based on parameter units
                    if param.data.get('units') == 'arcseconds':
                        sigma = np.deg2rad(param.data['sigma'] / 3600.0)  # arcsec to radians
                        bounds = [np.deg2rad(arange[0] / 3600.0), np.deg2rad(arange[1] / 3600.0)]
                    elif param.data.get('units') == 'milliseconds':
                        sigma = param.data['sigma'] / 1000.0  # msec to seconds
                        bounds = [arange[0] / 1000.0, arange[1] / 1000.0]  # msec to seconds
                    else:
                        sigma = param.data['sigma']
                        bounds = arange

                    offset = np.random.normal(0, sigma)
                    offset = np.clip(offset, bounds[0], bounds[1])
                else:
                    # Fallback to uniform distribution
                    offset = np.random.uniform(arange[0], arange[1])
                param_vals = center + offset

            # Convert to DataFrame for CONSTANT_KERNEL parameters
            if param.ptype is ParameterType.CONSTANT_KERNEL and param.config_file and '.ck.' in param.config_file.name:
                param_vals = pd.DataFrame({
                    "ugps": [0, 2209075218000000],
                    "angle_x": [param_vals[0], param_vals[0]],
                    "angle_y": [param_vals[1], param_vals[1]],
                    "angle_z": [param_vals[2], param_vals[2]],
                })

            out_set.append((param, param_vals))
        output.append(out_set)

    logger.info(f"Generated {len(output)} parameter sets with {len(config.parameters)} parameters each")
    return output


def load_telemetry(tlm_key: str, config: MonteCarloConfig) -> pd.DataFrame:
    raise NotImplementedError


def load_science(sci_key: str, config: MonteCarloConfig) -> xr.Dataset:
    raise NotImplementedError


def load_gcp(gcp_key: str, config: MonteCarloConfig) -> xr.Dataset:
    raise NotImplementedError


def apply_offset(config: ParameterConfig, param_data, input_data):
    raise NotImplementedError


def loop(config: MonteCarloConfig, work_dir: Path, tlm_sci_gcp_sets: [(str, str, str)]):
    # Initialize the entire set of parameters.
    params_set = load_param_sets(config)

    # Initialize return data structure...
    results = []

    # Prepare meta kernel details and kernel writer.
    mkrn = meta.MetaKernel.from_json(
        config.geo.meta_kernel_file, relative=True, sds_dir=config.geo.generic_kernel_dir,
    )
    creator = create.KernelCreator(overwrite=True, append=False)

    # Process each pairing of image data to a GCP.
    for pair_idx, (tlm_key, sci_key, gcp_key) in enumerate(tlm_sci_gcp_sets):
        logger.info(f"Processing data pair {pair_idx + 1}/{len(tlm_sci_gcp_sets)}: {sci_key}")

        # Load telemetry (L1) telemetry...
        tlm_dataset = load_telemetry(tlm_key, config)

        # Load science (L1A) dataset...
        sci_dataset = load_science(sci_key, config)
        ugps_times = sci_dataset[config.geo.time_field]  # Can be altered by later steps.

        # === GCP PAIRING MODULE (PLACEHOLDER) ===
        # Real implementation will find overlapping CLARREO-Landsat pairs
        logger.info("=== GCP PAIRING MODULE ===")
        gcp_pairs = placeholder_gcp_pairing([sci_key])
        logger.info(f"Found {len(gcp_pairs)} GCP pairs for processing")

        # Create dynamic unmodified SPICE kernels...
        #   Aka: SC-SPK, SC-CK
        logger.info("Creating dynamic kernels from telemetry...")
        dynamic_kernels = []
        for kernel_config in config.geo.dynamic_kernels:
            dynamic_kernels.append(creator.write_from_json(
                kernel_config, output_kernel=work_dir, input_data=tlm_dataset,
            ))
        logger.info(f"Created {len(dynamic_kernels)} dynamic kernels")

        # Loop for each parameter set.
        for param_idx, params in enumerate(params_set):
            logger.info(f"=== Monte Carlo Iteration {param_idx + 1}/{len(params_set)} ===")
            param_kernels = []
            ugps_times_modified = ugps_times.copy() if hasattr(ugps_times, 'copy') else ugps_times

            # Apply each individual parameter change.
            for a_param, p_data in params:  # [ParameterConfig, typing.Any]

                # Create static changing SPICE kernels.
                if a_param.ptype == ParameterType.CONSTANT_KERNEL:
                    # Aka: BASE-CK, YOKE-CK, HYSICS-CK
                    param_kernels.append(creator.write_from_json(
                        a_param.config_file, output_kernel=work_dir, input_data=p_data,
                    ))

                # Create dynamic changing SPICE kernels.
                elif a_param.ptype == ParameterType.OFFSET_KERNEL:
                    # Aka: AZ-CK, EL-CK
                    tlm_dataset_alt = apply_offset(a_param, p_data, tlm_dataset)
                    param_kernels.append(creator.write_from_json(
                        a_param.config_file, output_kernel=work_dir, input_data=tlm_dataset_alt,
                    ))

                # Alter non-kernel data.
                elif a_param.ptype == ParameterType.OFFSET_TIME:
                    # Aka: Frame-times...
                    sci_dataset_alt = apply_offset(a_param, p_data, sci_dataset)
                    ugps_times_modified = sci_dataset_alt[config.geo.time_field].values

                else:
                    raise NotImplementedError(a_param.ptype)

            logger.info(f"Created {len(param_kernels)} parameter-specific kernels")

            # Geolocate.
            logger.info("Performing geolocation...")
            with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, dynamic_kernels, param_kernels]):
                geoloc_inst = spatial.Geolocate(config.geo.instrument_name)
                geo_dataset = geoloc_inst(ugps_times_modified)

                # === IMAGE MATCHING MODULE (PLACEHOLDER) ===
                logger.info("=== IMAGE MATCHING MODULE ===")
                image_matching_output = placeholder_image_matching(
                    geo_dataset,
                    gcp_pairs[0][1] if gcp_pairs else "synthetic_gcp.tif",
                    params
                )
                logger.info(f"Generated error measurements for {len(image_matching_output.measurement)} points")

                # === ERROR STATISTICS MODULE (REAL) ===
                logger.info("=== ERROR STATISTICS MODULE ===")
                stats_dataset = call_error_stats_module(image_matching_output)
                rms_error = stats_dataset.attrs.get('rms_error_m', stats_dataset.attrs.get('mean_error_m', float('inf')))
                logger.info(f"Computed error statistics - RMS: {rms_error:.2f}m")

                # Store comprehensive results with clear module outputs
                iteration_result = {
                    'iteration': len(results),
                    'pair_index': pair_idx,
                    'param_index': param_idx,
                    'parameters': {
                        (param[0].config_file.name if param[0].config_file else f'param_{i}'): param[1]
                        for i, param in enumerate(params)
                    },
                    'geolocation': geo_dataset,
                    'gcp_pairs': gcp_pairs,  # From GCP pairing module
                    'image_matching': image_matching_output,  # From image matching module
                    'error_stats': stats_dataset,  # From error stats module
                    'rms_error_m': rms_error
                }
                results.append(iteration_result)
                logger.info(f"Iteration complete: RMS error = {rms_error:.2f}m")

    logger.info(f"=== GCS Loop Complete: Processed {len(results)} total iterations ===")
    return results
