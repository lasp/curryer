import logging
import typing
import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import xarray as xr

from curryer import meta
from curryer import spicierpy as sp
from curryer.compute import spatial
from curryer.kernels import create


logger = logging.getLogger(__name__)


# Configuration Loading Functions

def load_config_from_json(config_path: Path) -> 'MonteCarloConfig':
    """Load Monte Carlo configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file (e.g., gcs_config.json)

    Returns:
        MonteCarloConfig object populated from the JSON file

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
        KeyError: If required config sections are missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading Monte Carlo configuration from: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")

    # Validate required sections exist
    if 'monte_carlo' not in config_data:
        raise KeyError("Missing required 'monte_carlo' section in config file")
    if 'geolocation' not in config_data:
        raise KeyError("Missing required 'geolocation' section in config file")

    # Extract monte_carlo section
    mc_config = config_data.get('monte_carlo', {})
    geo_config = config_data.get('geolocation', {})

    # Validate monte_carlo section
    if 'parameters' not in mc_config:
        raise KeyError("Missing required 'parameters' in monte_carlo section")
    if not isinstance(mc_config['parameters'], list):
        raise ValueError("'parameters' must be a list")
    if len(mc_config['parameters']) == 0:
        raise ValueError("No parameters defined in configuration")

    # Parse parameters and group related ones together
    parameters = []
    param_groups = {}

    # First pass: group parameters by their base name and type
    for param_dict in mc_config.get('parameters', []):
        param_name = param_dict.get('name', '')
        ptype_str = param_dict.get('parameter_type', 'CONSTANT_KERNEL')
        ptype = ParameterType[ptype_str]

        # Group CONSTANT_KERNEL parameters by their base frame name
        if ptype == ParameterType.CONSTANT_KERNEL:
            # Extract base name (e.g., "hysics_to_cradle" from "hysics_to_cradle_roll")
            if '_roll' in param_name:
                base_name = param_name.replace('_roll', '')
                angle_type = 'roll'
            elif '_pitch' in param_name:
                base_name = param_name.replace('_pitch', '')
                angle_type = 'pitch'
            elif '_yaw' in param_name:
                base_name = param_name.replace('_yaw', '')
                angle_type = 'yaw'
            else:
                base_name = param_name
                angle_type = 'single'

            if base_name not in param_groups:
                param_groups[base_name] = {
                    'type': ptype,
                    'angles': {},
                    'template': param_dict,
                    'config_file': None
                }

            param_groups[base_name]['angles'][angle_type] = param_dict.get('initial_value', 0.0)

            # Determine config file based on base name
            if 'hysics' in base_name.lower():
                param_groups[base_name]['config_file'] = Path('cprs_hysics_v01.attitude.ck.json')
            elif 'yoke' in base_name.lower():
                param_groups[base_name]['config_file'] = Path('cprs_yoke_v01.attitude.ck.json')
            elif 'base' in base_name.lower():
                param_groups[base_name]['config_file'] = Path('cprs_base_v01.attitude.ck.json')

        else:
            # OFFSET_KERNEL and OFFSET_TIME parameters are individual
            param_groups[param_name] = {
                'type': ptype,
                'param_dict': param_dict,
                'config_file': None
            }

            if ptype == ParameterType.OFFSET_KERNEL:
                if 'azimuth' in param_name.lower():
                    param_groups[param_name]['config_file'] = Path('cprs_az_v01.attitude.ck.json')
                elif 'elevation' in param_name.lower():
                    param_groups[param_name]['config_file'] = Path('cprs_el_v01.attitude.ck.json')

    # Second pass: create ParameterConfig objects from groups
    for group_name, group_data in param_groups.items():
        if group_data['type'] == ParameterType.CONSTANT_KERNEL:
            # For CONSTANT_KERNEL, combine roll/pitch/yaw into a single parameter
            template = group_data['template']
            angles = group_data['angles']

            # Create center values array [roll, pitch, yaw] with defaults of 0.0
            center_values = [
                angles.get('roll', 0.0),
                angles.get('pitch', 0.0),
                angles.get('yaw', 0.0)
            ]

            param_data = {
                'center': center_values,
                'arange': template.get('bounds', [-100, 100]),
                'sigma': template.get('sigma'),
                'units': template.get('units', 'arcseconds'),
                'distribution': template.get('distribution_type', 'normal'),
                'field': template.get('application_target', {}).get('field_name', None),
            }

        else:
            # For OFFSET_KERNEL and OFFSET_TIME, use the parameter as-is
            param_dict = group_data['param_dict']
            param_data = {
                'center': param_dict.get('initial_value', 0.0),
                'arange': param_dict.get('bounds', [-100, 100]),
                'sigma': param_dict.get('sigma'),
                'units': param_dict.get('units', 'radians'),
                'distribution': param_dict.get('distribution_type', 'normal'),
                'field': param_dict.get('application_target', {}).get('field_name', None),
            }

        parameters.append(ParameterConfig(
            ptype=group_data['type'],
            config_file=group_data['config_file'],
            data=param_data
        ))

    logger.info(f"Loaded {len(parameters)} parameter groups from {len(mc_config.get('parameters', []))} individual parameters")

    # Parse geolocation configuration
    geo = GeolocationConfig(
        meta_kernel_file=Path(geo_config.get('meta_kernel_file', '')),
        generic_kernel_dir=Path(geo_config.get('generic_kernel_dir', '')),
        dynamic_kernels=[Path(k) for k in geo_config.get('dynamic_kernels', [])],
        instrument_name=geo_config.get('instrument_name', 'CPRS_HYSICS'),
        time_field=geo_config.get('time_field', 'corrected_timestamp'),
    )

    # Create MonteCarloConfig
    config = MonteCarloConfig(
        seed=mc_config.get('seed'),
        n_iterations=mc_config.get('n_iterations', 10),
        parameters=parameters,
        geo=geo,
    )

    logger.info(f"Configuration loaded: {config.n_iterations} iterations, {len(config.parameters)} parameter groups")
    return config


# GCS Module Placeholder Functions

def diagnose_telemetry_data(tlm_dataset: pd.DataFrame, dataset_name: str = "telemetry"):
    """
    Diagnose telemetry data for validity issues.

    Checks for:
    - NaN values
    - Zero/constant values
    - Data range issues
    - Missing required columns
    """
    print(f"\n{'='*80}")
    print(f"=== DIAGNOSING {dataset_name.upper()} DATA ===")
    print(f"{'='*80}")

    # Basic info
    print(f"Dataset shape: {tlm_dataset.shape}")
    print(f"Total columns: {len(tlm_dataset.columns)}")
    print(f"Index: {tlm_dataset.index.name} (range: {tlm_dataset.index.min()} to {tlm_dataset.index.max()})")

    # Show first few column names
    print(f"\nFirst 10 columns: {list(tlm_dataset.columns[:10])}")
    print(f"Last 10 columns: {list(tlm_dataset.columns[-10:])}")

    # Check for NaN values
    nan_counts = tlm_dataset.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\n Found NaN values in {len(nan_cols)} columns:")
        for col, count in nan_cols.items():
            pct = (count / len(tlm_dataset)) * 100
            print(f"  {col}: {count} NaNs ({pct:.1f}%)")
    else:
        print("\n✓ No NaN values found")

    # Check for zero/constant values in critical columns
    print(f"\n{'='*80}")
    print("=== CHECKING CRITICAL COLUMN VALUES ===")
    print(f"{'='*80}")

    critical_patterns = ['quat', 'ang', 'dcm', 'position', 'velocity']
    for pattern in critical_patterns:
        matching_cols = [col for col in tlm_dataset.columns if pattern in col.lower()]
        if matching_cols:
            print(f"\n{pattern.upper()} columns ({len(matching_cols)} found):")
            for col in matching_cols:
                values = tlm_dataset[col].dropna()
                if len(values) > 0:
                    # Check if all zeros
                    if (values == 0).all():
                        print(f"{col}: ALL ZEROS!")
                    # Check if constant
                    elif values.nunique() == 1:
                        print(f"{col}: CONSTANT VALUE = {values.iloc[0]}")
                    # Check if mostly zeros
                    elif (values == 0).sum() / len(values) > 0.9:
                        zero_pct = (values == 0).sum() / len(values) * 100
                        print(f"{col}: {zero_pct:.1f}% zeros (range: {values.min():.6e} to {values.max():.6e})")
                    else:
                        print(f"{col}: Valid range [{values.min():.6e}, {values.max():.6e}]")
                else:
                    print(f"{col}: No valid (non-NaN) values!")

    # Check for quaternion validity (should have unit norm)
    print(f"\n{'='*80}")
    print("=== CHECKING QUATERNION VALIDITY ===")
    print(f"{'='*80}")

    quat_cols = [col for col in tlm_dataset.columns if 'quat' in col.lower() or '_s' in col or '_i' in col or '_j' in col or '_k' in col]
    if len(quat_cols) >= 4:
        # Try to identify quaternion sets
        quat_sets = {}
        for col in quat_cols:
            base_name = col.rsplit('_', 1)[0] if '_' in col else col.rsplit('.', 1)[0]
            if base_name not in quat_sets:
                quat_sets[base_name] = []
            quat_sets[base_name].append(col)

        for base_name, cols in quat_sets.items():
            if len(cols) == 4:
                print(f"\nChecking quaternion set: {base_name}")
                print(f"  Components: {cols}")
                quat_data = tlm_dataset[cols].dropna()
                if len(quat_data) > 0:
                    norms = np.sqrt((quat_data ** 2).sum(axis=1))
                    if (norms == 0).any():
                        print(f"{base_name}: Contains zero-norm quaternions!")
                    elif not np.allclose(norms, 1.0, atol=1e-3):
                        print(f"{base_name}: Quaternion norms deviate from 1.0")
                        print(f"Norm range: {norms.min():.6f} to {norms.max():.6f}")
                    else:
                        print(f"{base_name}: Valid unit quaternions (norm ≈ 1.0)")
                else:
                    print(f"{base_name}: No valid (non-NaN) quaternion data!")
    else:
        print("No quaternion columns found")

    # Check time columns
    print(f"\n{'='*80}")
    print("=== CHECKING TIME COLUMNS ===")
    print(f"{'='*80}")

    time_cols = [col for col in tlm_dataset.columns if 'time' in col.lower() or 'tms' in col.lower()]
    if time_cols:
        for col in time_cols:
            values = tlm_dataset[col].dropna()
            if len(values) > 0:
                print(f"\n{col}:")
                print(f"  Valid values: {len(values)}/{len(tlm_dataset)} ({len(values)/len(tlm_dataset)*100:.1f}%)")
                print(f"  Range: [{values.min():.6e}, {values.max():.6e}]")
                print(f"  Mean: {values.mean():.6e}")
            else:
                print(f"\n{col}: No valid values!")
    else:
        print("No time columns found")

    print(f"\n{'='*80}")
    print(f"=== END {dataset_name.upper()} DIAGNOSIS ===")
    print(f"{'='*80}\n")


def diagnose_science_data(sci_dataset, dataset_name: str = "science"):
    """
    Diagnose science data for validity issues.
    """
    msg = f"=== DIAGNOSING {dataset_name.upper()} DATA ==="
    logger.info(msg)
    print(msg)

    if isinstance(sci_dataset, pd.DataFrame):
        msg = f"Dataset type: DataFrame"
        logger.info(msg)
        print(msg)
        msg = f"Dataset shape: {sci_dataset.shape}"
        logger.info(msg)
        print(msg)
        msg = f"Columns: {list(sci_dataset.columns)}"
        logger.info(msg)
        print(msg)

        # Check time field
        time_cols = [col for col in sci_dataset.columns if 'time' in col.lower()]
        for col in time_cols:
            values = sci_dataset[col].dropna()
            if len(values) > 0:
                msg = f"Time column {col}: {len(values)} values, range [{values.min():.6e}, {values.max():.6e}]"
                logger.info(msg)
                print(msg)

                # Check for zeros
                if (values == 0).any():
                    zero_count = (values == 0).sum()
                    msg = f"  {col}: Contains {zero_count} zero values!"
                    logger.warning(msg)
                    print(f"⚠️  {msg}")

    elif isinstance(sci_dataset, xr.Dataset):
        msg = f"Dataset type: xarray.Dataset"
        logger.info(msg)
        print(msg)
        msg = f"Dimensions: {dict(sci_dataset.dims)}"
        logger.info(msg)
        print(msg)
        msg = f"Variables: {list(sci_dataset.data_vars)}"
        logger.info(msg)
        print(msg)
        msg = f"Coordinates: {list(sci_dataset.coords)}"
        logger.info(msg)
        print(msg)

        for var in sci_dataset.data_vars:
            values = sci_dataset[var].values
            msg = f"Variable {var}: shape={values.shape}, dtype={values.dtype}"
            logger.info(msg)
            print(msg)

    msg = f"=== END {dataset_name.upper()} DIAGNOSIS ===\n"
    logger.info(msg)
    print(msg)


def diagnose_kernel_inputs(param_data, param_config: 'ParameterConfig'):
    """
    Diagnose kernel input data before creation.
    """
    logger.info(f"=== DIAGNOSING KERNEL INPUT: {param_config.config_file.name if param_config.config_file else 'N/A'} ===")

    if isinstance(param_data, pd.DataFrame):
        logger.info(f"Data type: DataFrame, shape={param_data.shape}")
        logger.info(f"Columns: {list(param_data.columns)}")

        # Check each column
        for col in param_data.columns:
            values = param_data[col]
            logger.info(f"Column {col}:")
            logger.info(f"  Range: [{values.min():.6e}, {values.max():.6e}]")
            logger.info(f"  Unique values: {values.nunique()}")

            # Warn if all same value
            if values.nunique() == 1:
                logger.warning(f"All values are constant: {values.iloc[0]}")

            # Warn if all zeros
            if (values == 0).all():
                logger.error(f"All values are ZERO!")

        # For attitude kernels, check angle values
        angle_cols = [col for col in param_data.columns if 'angle' in col.lower()]
        if angle_cols:
            logger.info("Angle values (should be in radians):")
            for col in angle_cols:
                values = param_data[col]
                logger.info(f"  {col}: {values.iloc[0]:.6e} rad = {np.degrees(values.iloc[0]):.6f} deg")

    elif isinstance(param_data, (int, float)):
        logger.info(f"Data type: {type(param_data).__name__}, value={param_data}")

    elif isinstance(param_data, (list, np.ndarray)):
        logger.info(f"Data type: {type(param_data).__name__}, length={len(param_data)}")
        logger.info(f"Values: {param_data}")

    logger.info(f"=== END KERNEL INPUT DIAGNOSIS ===\n")


def placeholder_gcp_pairing(clarreo_data_files):
    """
    PLACEHOLDER for GCP pairing module.

    TODO: Replace with real GCP pairing implementation

    Expected inputs:
    - clarreo_data_files (list): List of CLARREO L1A image file paths

    Expected outputs:
    - List of tuples: [(clarreo_file, gcp_reference_file), ...]

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

    TODO: Replace with real image matching implementation

    Expected inputs:
    - geolocated_data (xarray.Dataset): CLARREO geolocated pixels with lat/lon coordinates
    - gcp_reference_file (str): Path to Landsat GCP reference image
    - params_info (list): Current parameter values for error correlation

    Expected outputs:
    - xarray.Dataset with error measurements matching error_stats module format:
      - lat_error_deg, lon_error_deg: Spatial errors in degrees
      - riss_ctrs, bhat_hs, t_hs2ctrs: Coordinate transformation data
      - cp_lat_deg, cp_lon_deg, cp_alt: Control point coordinates

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


def call_error_stats_module(image_matching_results):
    """
    Call the REAL error_stats module with image matching output.

    Args:
        image_matching_results: Either a single image matching result (xarray.Dataset)
                              or a list of image matching results from multiple GCP pairs

    Returns:
        Aggregate error statistics dataset
    """
    # Handle both single result (backward compatibility) and list of results
    if not isinstance(image_matching_results, list):
        image_matching_results = [image_matching_results]

    try:
        from curryer.correction.error_stats.geolocation_error_stats import ErrorStatsProcessor

        logger.info(f"Error Statistics: Processing geolocation errors from {len(image_matching_results)} GCP pairs (REAL MODULE)")
        processor = ErrorStatsProcessor()

        if len(image_matching_results) == 1:
            # Single GCP pair case
            error_results = processor.process_geolocation_errors(image_matching_results[0])
        else:
            # Multiple GCP pairs - aggregate the data first
            aggregated_data = _aggregate_image_matching_results(image_matching_results)
            error_results = processor.process_geolocation_errors(aggregated_data)

        return error_results
    except ImportError as e:
        logger.warning(f"Error stats module not available: {e}")
        logger.info(f"Error Statistics: Using placeholder calculations for {len(image_matching_results)} GCP pairs")

        # Fallback: compute basic statistics across all GCP pairs
        all_lat_errors = []
        all_lon_errors = []
        total_measurements = 0

        for result in image_matching_results:
            lat_errors = result['lat_error_deg'].values
            lon_errors = result['lon_error_deg'].values
            all_lat_errors.extend(lat_errors)
            all_lon_errors.extend(lon_errors)
            total_measurements += len(lat_errors)

        all_lat_errors = np.array(all_lat_errors)
        all_lon_errors = np.array(all_lon_errors)

        # Convert to meters (approximate)
        lat_error_m = all_lat_errors * 111000
        lon_error_m = all_lon_errors * 111000
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
            'total_measurements': total_measurements,
            'gcp_pairs_processed': len(image_matching_results),
            'performance_threshold_m': 300.0,  # CLARREO requirement
        })


def _aggregate_image_matching_results(image_matching_results):
    """
    Aggregate multiple image matching results into a single dataset for error stats processing.

    Args:
        image_matching_results: List of xarray.Dataset objects from image matching

    Returns:
        Single aggregated xarray.Dataset with all measurements combined
    """
    logger.info(f"Aggregating {len(image_matching_results)} image matching results")

    # Combine all measurements into single arrays
    all_lat_errors = []
    all_lon_errors = []
    all_riss_ctrs = []
    all_bhat_hs = []
    all_t_hs2ctrs = []
    all_cp_lats = []
    all_cp_lons = []
    all_cp_alts = []

    for i, result in enumerate(image_matching_results):
        # Add GCP pair identifier to track source
        n_measurements = len(result['lat_error_deg'])

        all_lat_errors.extend(result['lat_error_deg'].values)
        all_lon_errors.extend(result['lon_error_deg'].values)

        # Handle coordinate transformation data
        if 'riss_ctrs' in result:
            all_riss_ctrs.extend(result['riss_ctrs'].values)
        if 'bhat_hs' in result:
            all_bhat_hs.extend(result['bhat_hs'].values)
        if 't_hs2ctrs' in result:
            all_t_hs2ctrs.extend(result['t_hs2ctrs'].values)
        if 'cp_lat_deg' in result:
            all_cp_lats.extend(result['cp_lat_deg'].values)
        if 'cp_lon_deg' in result:
            all_cp_lons.extend(result['cp_lon_deg'].values)
        if 'cp_alt' in result:
            all_cp_alts.extend(result['cp_alt'].values)

    # Create aggregated dataset
    aggregated = xr.Dataset({
        'lat_error_deg': (['measurement'], np.array(all_lat_errors)),
        'lon_error_deg': (['measurement'], np.array(all_lon_errors)),
    }, coords={
        'measurement': np.arange(len(all_lat_errors))
    })

    # Add optional coordinate transformation data if available
    if all_riss_ctrs:
        aggregated['riss_ctrs'] = (['measurement', 'coord'], np.array(all_riss_ctrs))
    if all_bhat_hs:
        aggregated['bhat_hs'] = (['measurement', 'coord'], np.array(all_bhat_hs))
    if all_t_hs2ctrs:
        aggregated['t_hs2ctrs'] = (['measurement', 'matrix_i', 'matrix_j'], np.array(all_t_hs2ctrs))
    if all_cp_lats:
        aggregated['cp_lat_deg'] = (['measurement'], np.array(all_cp_lats))
    if all_cp_lons:
        aggregated['cp_lon_deg'] = (['measurement'], np.array(all_cp_lons))
    if all_cp_alts:
        aggregated['cp_alt'] = (['measurement'], np.array(all_cp_alts))

    aggregated.attrs['source_gcp_pairs'] = len(image_matching_results)
    aggregated.attrs['total_measurements'] = len(all_lat_errors)

    return aggregated


# Original Functions

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

    The parameter generation now works as follows:
    - current_value: The baseline/current parameter value
    - bounds: The limits for random offsets (in same units as current_value and sigma)
    - sigma: Standard deviation for normal distribution of offsets
    - Generated offsets are centered around 0, then applied to current_value
    - Final value = current_value + random_offset

    Handles all parameter types:
    - CONSTANT_KERNEL: 3D attitude corrections (roll, pitch, yaw)
    - OFFSET_KERNEL: Single angle biases for telemetry fields
    - OFFSET_TIME: Timing corrections for science frames
    """

    if config.seed is not None:
        np.random.seed(config.seed)
        logger.info(f"Set random seed to {config.seed} for reproducible parameter generation")

    output = []

    logger.info(f"Generating {config.n_iterations} parameter sets for {len(config.parameters)} parameters:")
    for i, param in enumerate(config.parameters):
        param_name = param.config_file.name if param.config_file else f"param_{i}"
        current_value = param.data.get('current_value', param.data.get('center', 0.0))
        bounds = param.data.get('bounds', param.data.get('arange', [-1.0, 1.0]))
        logger.info(f"  {i+1}. {param_name} ({param.ptype.name}): "
                   f"current_value={current_value}, sigma={param.data.get('sigma', 'N/A')}, "
                   f"bounds={bounds}, units={param.data.get('units', 'N/A')}")

    for ith in range(config.n_iterations):
        out_set = []
        logger.debug(f"Generating parameter set {ith + 1}/{config.n_iterations}")

        for param_idx, param in enumerate(config.parameters):
            # Get parameter configuration with backward compatibility
            current_value = param.data.get('current_value', param.data.get('center', 0.0))
            bounds = param.data.get('bounds', param.data.get('arange', [-1.0, 1.0]))

            # Handle different parameter structure types
            if param.ptype == ParameterType.CONSTANT_KERNEL:
                # CONSTANT_KERNEL parameters are 3D attitude corrections (roll, pitch, yaw)
                if isinstance(current_value, list) and len(current_value) == 3:
                    # Multi-dimensional parameter (roll, pitch, yaw)
                    param_vals = []
                    for i, current_val in enumerate(current_value):
                        # Check if parameter should be varied
                        if ('sigma' in param.data and param.data['sigma'] is not None and
                            param.data['sigma'] > 0):
                            # Apply variation: Generate offset around 0, then apply to current_value
                            if param.data.get('units') == 'arcseconds':
                                # Convert arcsec to radians for sampling
                                sigma_rad = np.deg2rad(param.data['sigma'] / 3600.0)
                                current_val_rad = np.deg2rad(current_val / 3600.0) if current_val != 0 else current_val
                                # Convert bounds from arcsec to radians (these are offset bounds around 0)
                                bounds_rad = [np.deg2rad(bounds[0] / 3600.0), np.deg2rad(bounds[1] / 3600.0)]
                            else:
                                # Assume all values are already in radians
                                sigma_rad = param.data['sigma']
                                current_val_rad = current_val
                                bounds_rad = bounds

                            # Generate offset around 0, clamp to bounds, and add to current value
                            offset = np.random.normal(0, sigma_rad)
                            offset = np.clip(offset, bounds_rad[0], bounds_rad[1])
                            param_vals.append(current_val_rad + offset)
                        else:
                            # No variation: use current_value directly
                            if 'sigma' not in param.data or param.data['sigma'] is None:
                                logger.debug(f"  Parameter {param_idx} axis {i}: No sigma specified, using fixed current_value")
                            elif param.data['sigma'] == 0:
                                logger.debug(f"  Parameter {param_idx} axis {i}: sigma=0, using fixed current_value")

                            # Convert to appropriate units if needed
                            if param.data.get('units') == 'arcseconds':
                                current_val_rad = np.deg2rad(current_val / 3600.0) if current_val != 0 else current_val
                            else:
                                current_val_rad = current_val
                            param_vals.append(current_val_rad)
                else:
                    # Single angle or default to zero for each axis
                    param_vals = [0.0, 0.0, 0.0]  # [roll, pitch, yaw]
                    if ('sigma' in param.data and param.data['sigma'] is not None and
                        param.data['sigma'] > 0):
                        # Apply variation
                        if param.data.get('units') == 'arcseconds':
                            sigma_rad = np.deg2rad(param.data['sigma'] / 3600.0)
                            bounds_rad = [np.deg2rad(bounds[0] / 3600.0), np.deg2rad(bounds[1] / 3600.0)]
                            current_val_rad = np.deg2rad(current_value / 3600.0) if current_value != 0 else current_value
                        else:
                            sigma_rad = param.data['sigma']
                            bounds_rad = bounds
                            current_val_rad = current_value

                        for i in range(3):
                            # Generate offset around 0, clamp to bounds, add to current value
                            offset = np.random.normal(0, sigma_rad)
                            offset = np.clip(offset, bounds_rad[0], bounds_rad[1])
                            param_vals[i] = current_val_rad + offset
                    else:
                        # No variation: use current_value directly for all axes
                        if 'sigma' not in param.data or param.data['sigma'] is None:
                            logger.debug(f"  Parameter {param_idx}: No sigma specified, using fixed current_value")
                        elif param.data['sigma'] == 0:
                            logger.debug(f"  Parameter {param_idx}: sigma=0, using fixed current_value")

                        # Convert to appropriate units if needed
                        if param.data.get('units') == 'arcseconds':
                            current_val_rad = np.deg2rad(current_value / 3600.0) if current_value != 0 else current_value
                        else:
                            current_val_rad = current_value

                        # Use same value for all three axes (this handles scalar current_value)
                        param_vals = [current_val_rad, current_val_rad, current_val_rad]

                # Convert to DataFrame format expected by kernel creation
                param_vals = pd.DataFrame({
                    "ugps": [0, 2209075218000000],  # Start and end times
                    "angle_x": [param_vals[0], param_vals[0]],  # Roll (constant over time)
                    "angle_y": [param_vals[1], param_vals[1]],  # Pitch (constant over time)
                    "angle_z": [param_vals[2], param_vals[2]],  # Yaw (constant over time)
                })

                logger.debug(f"  CONSTANT_KERNEL {param_idx}: angles=[{param_vals['angle_x'].iloc[0]:.6e}, "
                           f"{param_vals['angle_y'].iloc[0]:.6e}, {param_vals['angle_z'].iloc[0]:.6e}] rad")

            elif param.ptype == ParameterType.OFFSET_KERNEL:
                # OFFSET_KERNEL parameters are angle biases (single values)
                if ('sigma' in param.data and param.data['sigma'] is not None and
                    param.data['sigma'] > 0):
                    # Apply variation: Generate offset around 0, then apply to current_value
                    if param.data.get('units') == 'arcseconds':
                        # Convert arcsec to radians for sampling
                        sigma_rad = np.deg2rad(param.data['sigma'] / 3600.0)
                        current_val_rad = np.deg2rad(current_value / 3600.0) if current_value != 0 else current_value
                        # Convert bounds from arcsec to radians (these are offset bounds around 0)
                        bounds_rad = [np.deg2rad(bounds[0] / 3600.0), np.deg2rad(bounds[1] / 3600.0)]
                    else:
                        # Assume all values are already in radians
                        sigma_rad = param.data['sigma']
                        current_val_rad = current_value
                        bounds_rad = bounds

                    # Generate offset around 0, clamp to bounds, and add to current value
                    offset = np.random.normal(0, sigma_rad)
                    offset = np.clip(offset, bounds_rad[0], bounds_rad[1])
                    param_vals = current_val_rad + offset
                else:
                    # No variation: use current_value directly
                    if 'sigma' not in param.data or param.data['sigma'] is None:
                        logger.debug(f"  Parameter {param_idx}: No sigma specified, using fixed current_value")
                    elif param.data['sigma'] == 0:
                        logger.debug(f"  Parameter {param_idx}: sigma=0, using fixed current_value")

                    # Convert to appropriate units if needed
                    if param.data.get('units') == 'arcseconds':
                        current_val_rad = np.deg2rad(current_value / 3600.0) if current_value != 0 else current_value
                    else:
                        current_val_rad = current_value
                    param_vals = current_val_rad

                logger.debug(f"  OFFSET_KERNEL {param_idx}: {param_vals:.6e} rad")

            elif param.ptype == ParameterType.OFFSET_TIME:
                # OFFSET_TIME parameters are timing corrections (single values)
                if ('sigma' in param.data and param.data['sigma'] is not None and
                    param.data['sigma'] > 0):
                    # Apply variation: Generate offset around 0, then apply to current_value
                    if param.data.get('units') == 'seconds':
                        # Time parameters typically use seconds, no conversion needed
                        sigma_time = param.data['sigma']
                        current_val_time = current_value
                        bounds_time = bounds
                    elif param.data.get('units') == 'milliseconds':
                        # Convert milliseconds to seconds
                        sigma_time = param.data['sigma'] / 1000.0
                        current_val_time = current_value / 1000.0
                        bounds_time = [bounds[0] / 1000.0, bounds[1] / 1000.0]
                    elif param.data.get('units') == 'microseconds':
                        # Convert microseconds to seconds
                        sigma_time = param.data['sigma'] / 1000000.0
                        current_val_time = current_value / 1000000.0
                        bounds_time = [bounds[0] / 1000000.0, bounds[1] / 1000000.0]
                    else:
                        # Default to seconds if units not specified
                        sigma_time = param.data['sigma']
                        current_val_time = current_value
                        bounds_time = bounds

                    # Generate offset around 0, clamp to bounds, then add to current value
                    offset = np.random.normal(0, sigma_time)
                    offset = np.clip(offset, bounds_time[0], bounds_time[1])
                    param_vals = current_val_time + offset
                else:
                    # No variation: use current_value directly
                    if 'sigma' not in param.data or param.data['sigma'] is None:
                        logger.debug(f"  Parameter {param_idx}: No sigma specified, using fixed current_value")
                    elif param.data['sigma'] == 0:
                        logger.debug(f"  Parameter {param_idx}: sigma=0, using fixed current_value")

                    # Convert to appropriate units if needed
                    if param.data.get('units') == 'milliseconds':
                        current_val_time = current_value / 1000.0
                    elif param.data.get('units') == 'microseconds':
                        current_val_time = current_value / 1000000.0
                    else:
                        current_val_time = current_value
                    param_vals = current_val_time

                logger.debug(f"  OFFSET_TIME {param_idx}: {param_vals:.6e} seconds")

            out_set.append((param, param_vals))
        output.append(out_set)

    logger.info(f"Generated {len(output)} parameter sets with {len(output[0])} parameters each")
    return output


def load_telemetry(tlm_key: str, config: MonteCarloConfig) -> pd.DataFrame:
    """
    Load telemetry data following the example patterns for robust processing.

    Args:
        tlm_key: Path to telemetry file or identifier (used to construct paths)
        config: Monte Carlo configuration (currently unused, for future flexibility)

    Returns:
        DataFrame with merged telemetry data using the example patterns
    """
    # Extract the base path from tlm_key or use default test data location
    if isinstance(tlm_key, (str, Path)):
        base_path = Path(tlm_key).parent if Path(tlm_key).parent.exists() else Path('tests/data/clarreo/gcs')
    else:
        base_path = Path('tests/data/clarreo/gcs')

    logger.info(f"Loading telemetry data from: {base_path}")

    # Load the 4 telemetry CSVs (following example pattern)
    sc_spk_df = pd.read_csv(base_path / "openloop_tlm_5a_sc_spk_20250521T225242.csv", index_col=0)
    sc_ck_df = pd.read_csv(base_path / "openloop_tlm_5a_sc_ck_20250521T225242.csv", index_col=0)
    st_ck_df = pd.read_csv(base_path / "openloop_tlm_5a_st_ck_20250521T225242.csv", index_col=0)
    azel_ck_df = pd.read_csv(base_path / "openloop_tlm_5a_azel_ck_20250521T225242.csv", index_col=0)

    logger.info(f"Loaded telemetry CSVs - SC_SPK: {sc_spk_df.shape}, SC_CK: {sc_ck_df.shape}, "
                f"ST_CK: {st_ck_df.shape}, AZEL_CK: {azel_ck_df.shape}")

    # Reverse the direction of the Azimuth element
    azel_ck_df['hps.az_ang_nonlin'] = azel_ck_df['hps.az_ang_nonlin'] * -1

    # Convert star-tracker from rotation matrix to quaternion (following example pattern)
    tlm_st_rot = np.vstack([st_ck_df['hps.dcm_base_iss_1_1'].values,
                            st_ck_df['hps.dcm_base_iss_1_2'].values,
                            st_ck_df['hps.dcm_base_iss_1_3'].values,
                            st_ck_df['hps.dcm_base_iss_2_1'].values,
                            st_ck_df['hps.dcm_base_iss_2_2'].values,
                            st_ck_df['hps.dcm_base_iss_2_3'].values,
                            st_ck_df['hps.dcm_base_iss_3_1'].values,
                            st_ck_df['hps.dcm_base_iss_3_2'].values,
                            st_ck_df['hps.dcm_base_iss_3_3'].values]).T
    tlm_st_rot = np.reshape(tlm_st_rot, (-1, 3, 3)).copy()

    # Import spicierpy for quaternion conversion
    from curryer import spicierpy as sp
    tlm_st_rot_q = np.vstack([sp.m2q(tlm_st_rot[i, :, :]) for i in range(tlm_st_rot.shape[0])])
    st_ck_df['hps.dcm_base_iss_s'] = tlm_st_rot_q[:, 0]
    st_ck_df['hps.dcm_base_iss_i'] = tlm_st_rot_q[:, 1]
    st_ck_df['hps.dcm_base_iss_j'] = tlm_st_rot_q[:, 2]
    st_ck_df['hps.dcm_base_iss_k'] = tlm_st_rot_q[:, 3]

    # Use example pattern: start with left_df and merge with outer joins
    left_df = sc_spk_df
    for right_df in [sc_ck_df, st_ck_df, azel_ck_df]:
        left_df = pd.merge(left_df, right_df, on='ert', how='outer')
    left_df = left_df.sort_values('ert')

    # Compute combined second and subsecond timetags (following example pattern exactly)
    for col in list(left_df):
        if col in ('hps.bad_ps_tms', 'hps.corrected_tms', 'hps.resolver_tms', 'hps.st_quat_coi_tms'):
            assert col + 's' in left_df.columns, col

            if col == 'hps.bad_ps_tms':
                left_df[col + '_tmss'] = left_df[col] + left_df[col + 's'] / 256
            elif col in ('hps.corrected_tms', 'hps.resolver_tms', 'hps.st_quat_coi_tms'):
                left_df[col + '_tmss'] = left_df[col] + left_df[col + 's'] / 2 ** 32
            else:
                raise ValueError('Missing if for expected column...')

    logger.info(f"Final telemetry shape: {left_df.shape}")
    return left_df



def load_science(sci_key: str, config: MonteCarloConfig) -> pd.DataFrame:
    """
    Load science frame timing data.

    Args:
        sci_key: Path to science file or identifier
        config: Monte Carlo configuration

    Returns:
        DataFrame with science frame timestamps
    """
    # Extract the base path from sci_key or use default test data location
    if isinstance(sci_key, (str, Path)):
        base_path = Path(sci_key).parent if Path(sci_key).parent.exists() else Path('tests/data/clarreo/gcs')
    else:
        base_path = Path('tests/data/clarreo/gcs')

    logger.info(f"Loading science data from: {base_path}")

    sci_time_df = pd.read_csv(base_path / "openloop_tlm_5a_sci_times_20250521T225242.csv", index_col=0)

    # Frame times are GPS seconds, geolocation expects uGPS (microseconds)
    sci_time_df['corrected_timestamp'] *= 1e6

    logger.info(f"Science data shape: {sci_time_df.shape}")
    logger.info(f"corrected_timestamp range: {sci_time_df['corrected_timestamp'].min():.2e} to "
                f"{sci_time_df['corrected_timestamp'].max():.2e} uGPS")

    return sci_time_df


def load_gcp(gcp_key: str, config: MonteCarloConfig):
    """
    Load Ground Control Point (GCP) reference data.

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
    logger.info(f"Loading GCP data from: {gcp_key} (PLACEHOLDER)")

    # For testing purposes, return None - the GCP pairing module will handle this
    # In real implementation, this would load and process GCP reference data
    return None


def apply_offset(config: ParameterConfig, param_data, input_data):
    """
    Apply parameter offsets to input data based on parameter type.

    Args:
        config: ParameterConfig specifying how to apply the offset
        param_data: The parameter values to apply (offset amounts)
        input_data: The input dataset to modify

    Returns:
        Modified copy of input_data with parameter offsets applied
    """
    logger.info(f"Applying {config.ptype.name} offset to {config.data.get('field', 'unknown field')}")

    # Make a copy to avoid modifying the original
    if isinstance(input_data, pd.DataFrame):
        modified_data = input_data.copy()
    else:
        modified_data = input_data.copy() if hasattr(input_data, 'copy') else input_data

    if config.ptype == ParameterType.OFFSET_KERNEL:
        # Apply offset to telemetry fields for dynamic kernels (azimuth/elevation angles)
        field_name = config.data.get('field')
        if field_name and field_name in modified_data.columns:
            # Convert parameter value to appropriate units
            offset_value = param_data
            if config.data.get('units') == 'arcseconds':
                # Convert arcseconds to radians for application
                offset_value = np.deg2rad(param_data / 3600.0)
            elif config.data.get('units') == 'milliseconds':
                # Convert milliseconds to seconds
                offset_value = param_data / 1000.0

            # Apply additive offset
            logger.info(f"Applying offset {offset_value} to field {field_name}")
            modified_data[field_name] = modified_data[field_name] + offset_value

        else:
            logger.warning(f"Field {field_name} not found in telemetry data for offset application")

    elif config.ptype == ParameterType.OFFSET_TIME:
        # Apply time offset to science frame timing
        field_name = config.data.get('field', 'corrected_timestamp')
        if hasattr(modified_data, '__getitem__') and field_name in modified_data:
            offset_value = param_data
            if config.data.get('units') == 'milliseconds':
                # Convert milliseconds to microseconds (uGPS)
                offset_value = param_data * 1000.0

            logger.info(f"Applying time offset {offset_value} to field {field_name}")
            modified_data[field_name] = modified_data[field_name] + offset_value

    elif config.ptype == ParameterType.CONSTANT_KERNEL:
        # For constant kernels, param_data should already be in the correct format
        # (DataFrame with ugps, angle_x, angle_y, angle_z columns)
        logger.info(f"Using constant kernel data with {len(param_data) if hasattr(param_data, '__len__') else 1} entries")
        modified_data = param_data

    else:
        raise NotImplementedError(f"Parameter type {config.ptype} not implemented")

    return modified_data


def loop(config: MonteCarloConfig, work_dir: Path, tlm_sci_gcp_sets: [(str, str, str)]):
    # Initialize the entire set of parameters.
    params_set = load_param_sets(config)

    # Initialize return data structure...
    results = []

    # Prepare NetCDF data structure for hierarchical output
    n_param_sets = len(params_set)
    n_gcp_pairs = len(tlm_sci_gcp_sets)

    # Initialize arrays for NetCDF output
    netcdf_data = {
        # Parameter set dimension (outer layer)
        'parameter_set_id': np.arange(n_param_sets),
        'gcp_pair_id': np.arange(n_gcp_pairs),

        # Parameter values (12 parameters) - stored once per parameter set
        'param_hysics_roll': np.full(n_param_sets, np.nan),
        'param_hysics_pitch': np.full(n_param_sets, np.nan),
        'param_hysics_yaw': np.full(n_param_sets, np.nan),
        'param_yoke_roll': np.full(n_param_sets, np.nan),
        'param_yoke_pitch': np.full(n_param_sets, np.nan),
        'param_yoke_yaw': np.full(n_param_sets, np.nan),
        'param_base_roll': np.full(n_param_sets, np.nan),
        'param_base_pitch': np.full(n_param_sets, np.nan),
        'param_base_yaw': np.full(n_param_sets, np.nan),
        'param_azimuth_bias': np.full(n_param_sets, np.nan),
        'param_elevation_bias': np.full(n_param_sets, np.nan),
        'param_time_correction': np.full(n_param_sets, np.nan),

        # Error statistics per GCP pair (2D arrays: [parameter_set_id, gcp_pair_id])
        'rms_error_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'mean_error_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'max_error_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'std_error_m': np.full((n_param_sets, n_gcp_pairs), np.nan),
        'n_measurements': np.full((n_param_sets, n_gcp_pairs), 0, dtype=int),

        # Overall performance metrics per parameter set
        'percent_under_250m': np.full(n_param_sets, np.nan),
        'mean_rms_all_pairs': np.full(n_param_sets, np.nan),
        'worst_pair_rms': np.full(n_param_sets, np.nan),
        'best_pair_rms': np.full(n_param_sets, np.nan),
    }

    logger.info(f"Initialized NetCDF data structure: {n_param_sets} parameter sets × {n_gcp_pairs} GCP pairs")

    # Prepare meta kernel details and kernel writer.
    mkrn = meta.MetaKernel.from_json(
        config.geo.meta_kernel_file, relative=True, sds_dir=config.geo.generic_kernel_dir,
    )
    creator = create.KernelCreator(overwrite=True, append=False)

    # Process each parameter set (Monte Carlo iteration)
    for param_idx, params in enumerate(params_set):
        logger.info(f"=== Parameter Set {param_idx + 1}/{len(params_set)} ===")

        # Extract and store parameter values for this set
        param_values = _extract_parameter_values(params)
        _store_parameter_values(netcdf_data, param_idx, param_values)

        # Collect image matching results from all GCP pairs for this parameter set
        image_matching_results = []
        gcp_pair_geolocation_data = []

        # Process each pairing of image data to a GCP for this parameter set
        for pair_idx, (tlm_key, sci_key, gcp_key) in enumerate(tlm_sci_gcp_sets):
            logger.info(f"  Processing GCP pair {pair_idx + 1}/{len(tlm_sci_gcp_sets)}: {sci_key}")

            # Load telemetry (L1) telemetry...
            tlm_dataset = load_telemetry(tlm_key, config)

            # Load science (L1A) dataset...
            sci_dataset = load_science(sci_key, config)
            ugps_times = sci_dataset[config.geo.time_field]  # Can be altered by later steps.

            # === GCP PAIRING MODULE (PLACEHOLDER) ===
            logger.info("    === GCP PAIRING MODULE ===")
            gcp_pairs = placeholder_gcp_pairing([sci_key])
            logger.info(f"    Found {len(gcp_pairs)} GCP pairs for processing")

            # Create dynamic unmodified SPICE kernels...
            #   Aka: SC-SPK, SC-CK
            logger.info("    Creating dynamic kernels from telemetry...")
            dynamic_kernels = []
            for kernel_config in config.geo.dynamic_kernels:
                dynamic_kernels.append(creator.write_from_json(
                    kernel_config, output_kernel=work_dir, input_data=tlm_dataset,
                ))
            logger.info(f"    Created {len(dynamic_kernels)} dynamic kernels")

            # Apply parameter changes for this parameter set
            param_kernels = []
            ugps_times_modified = ugps_times.copy() if hasattr(ugps_times, 'copy') else ugps_times

            # Apply each individual parameter change.
            print('Applying parameter changes:')
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

            logger.info(f"    Created {len(param_kernels)} parameter-specific kernels")

            # Geolocate.
            logger.info("    Performing geolocation...")
            with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, dynamic_kernels, param_kernels]):
                geoloc_inst = spatial.Geolocate(config.geo.instrument_name)
                geo_dataset = geoloc_inst(ugps_times_modified)

                # === IMAGE MATCHING MODULE (PLACEHOLDER) ===
                logger.info("    === IMAGE MATCHING MODULE ===")
                image_matching_output = placeholder_image_matching(
                    geo_dataset,
                    gcp_pairs[0][1] if gcp_pairs else "synthetic_gcp.tif",
                    params
                )
                logger.info(f"    Generated error measurements for {len(image_matching_output.measurement)} points")

                # Store image matching result for aggregate processing
                image_matching_output.attrs['gcp_pair_index'] = pair_idx
                image_matching_output.attrs['gcp_pair_id'] = f"{sci_key}_pair_{pair_idx}"
                image_matching_results.append(image_matching_output)

                # Store geolocation data for backward compatibility
                gcp_pair_geolocation_data.append({
                    'pair_index': pair_idx,
                    'geolocation': geo_dataset,
                    'gcp_pairs': gcp_pairs,
                })

                logger.info(f"    GCP pair {pair_idx + 1} image matching complete")

        # === ERROR STATISTICS MODULE (AGGREGATE PROCESSING) ===
        logger.info(f"  === ERROR STATISTICS MODULE (AGGREGATE) ===")
        logger.info(f"  Processing aggregate statistics from {len(image_matching_results)} GCP pairs")

        # Call error stats module on aggregate of all image matching results
        aggregate_stats = call_error_stats_module(image_matching_results)

        # Extract aggregate error metrics
        aggregate_error_metrics = _extract_error_metrics(aggregate_stats)

        logger.info(f"  Aggregate error statistics: RMS = {aggregate_error_metrics['rms_error_m']:.2f}m, "
                   f"measurements = {aggregate_error_metrics['n_measurements']}")

        # Process individual GCP pair results for detailed NetCDF storage
        pair_errors = []
        for pair_idx, image_matching_result in enumerate(image_matching_results):
            # Get individual pair error metrics from the single result
            individual_stats = call_error_stats_module(image_matching_result)
            individual_metrics = _extract_error_metrics(individual_stats)

            pair_errors.append(individual_metrics['rms_error_m'])

            # Store individual results in NetCDF structure
            _store_gcp_pair_results(netcdf_data, param_idx, pair_idx, individual_metrics)

            logger.info(f"    GCP pair {pair_idx + 1}: RMS = {individual_metrics['rms_error_m']:.2f}m")

        # Store comprehensive results for backward compatibility
        for pair_idx, (image_matching_result, geo_data) in enumerate(zip(image_matching_results, gcp_pair_geolocation_data)):
            individual_stats = call_error_stats_module(image_matching_result)
            individual_metrics = _extract_error_metrics(individual_stats)

            iteration_result = {
                'iteration': len(results),
                'pair_index': pair_idx,
                'param_index': param_idx,
                'parameters': param_values,
                'geolocation': geo_data['geolocation'],
                'gcp_pairs': geo_data['gcp_pairs'],
                'image_matching': image_matching_result,
                'error_stats': individual_stats,
                'aggregate_error_stats': aggregate_stats,  # NEW: Include aggregate statistics
                'rms_error_m': individual_metrics['rms_error_m'],
                'aggregate_rms_error_m': aggregate_error_metrics['rms_error_m']  # NEW: Include aggregate RMS
            }
            results.append(iteration_result)

        # Compute overall performance metrics for this parameter set
        _compute_parameter_set_metrics(netcdf_data, param_idx, pair_errors)

        # Log parameter set summary with both individual and aggregate metrics
        percent_under_250 = netcdf_data['percent_under_250m'][param_idx]
        mean_rms = netcdf_data['mean_rms_all_pairs'][param_idx]
        logger.info(f"Parameter set {param_idx + 1} complete:")
        logger.info(f"  Individual pairs - {percent_under_250:.1f}% under 250m, mean RMS = {mean_rms:.2f}m")
        logger.info(f"  Aggregate - RMS = {aggregate_error_metrics['rms_error_m']:.2f}m, "
                   f"total measurements = {aggregate_error_metrics['n_measurements']}")

    # Save NetCDF results
    output_file = work_dir / "monte_carlo_results.nc"
    _save_netcdf_results(netcdf_data, output_file, config)
    logger.info(f"Saved NetCDF results to: {output_file}")

    logger.info(f"=== GCS Loop Complete: Processed {len(params_set)} parameter sets × {len(tlm_sci_gcp_sets)} GCP pairs ===")
    return results, netcdf_data


def _extract_parameter_values(params):
    """Extract parameter values from a parameter set into a dictionary."""
    param_values = {}

    for param_config, param_data in params:
        if param_config.config_file:
            param_name = param_config.config_file.stem

            if param_config.ptype == ParameterType.CONSTANT_KERNEL:
                # Extract roll, pitch, yaw from DataFrame
                if isinstance(param_data, pd.DataFrame) and 'angle_x' in param_data.columns:
                    # Convert back to arcseconds for storage
                    param_values[f"{param_name}_roll"] = np.degrees(param_data['angle_x'].iloc[0]) * 3600
                    param_values[f"{param_name}_pitch"] = np.degrees(param_data['angle_y'].iloc[0]) * 3600
                    param_values[f"{param_name}_yaw"] = np.degrees(param_data['angle_z'].iloc[0]) * 3600

            elif param_config.ptype == ParameterType.OFFSET_KERNEL:
                # Single bias value (keep in original units)
                param_values[param_name] = param_data

            elif param_config.ptype == ParameterType.OFFSET_TIME:
                # Time correction (keep in original units)
                param_values[param_name] = param_data

    return param_values


def _store_parameter_values(netcdf_data, param_idx, param_values):
    """Store parameter values in the NetCDF data structure."""
    # Map parameter names to NetCDF variable names
    param_mapping = {
        'cprs_hysics_v01.attitude.ck_roll': 'param_hysics_roll',
        'cprs_hysics_v01.attitude.ck_pitch': 'param_hysics_pitch',
        'cprs_hysics_v01.attitude.ck_yaw': 'param_hysics_yaw',
        'cprs_yoke_v01.attitude.ck_roll': 'param_yoke_roll',
        'cprs_yoke_v01.attitude.ck_pitch': 'param_yoke_pitch',
        'cprs_yoke_v01.attitude.ck_yaw': 'param_yoke_yaw',
        'cprs_base_v01.attitude.ck_roll': 'param_base_roll',
        'cprs_base_v01.attitude.ck_pitch': 'param_base_pitch',
        'cprs_base_v01.attitude.ck_yaw': 'param_base_yaw',
        'cprs_az_v01.attitude.ck': 'param_azimuth_bias',
        'cprs_el_v01.attitude.ck': 'param_elevation_bias',
        'time_correction': 'param_time_correction',
    }

    for param_name, value in param_values.items():
        if param_name in param_mapping:
            netcdf_var = param_mapping[param_name]
            if netcdf_var in netcdf_data:
                netcdf_data[netcdf_var][param_idx] = value


def _extract_error_metrics(stats_dataset):
    """Extract error metrics from error statistics dataset."""
    if hasattr(stats_dataset, 'attrs'):
        # Real error stats module
        return {
            'rms_error_m': stats_dataset.attrs.get('rms_error_m', np.nan),
            'mean_error_m': stats_dataset.attrs.get('mean_error_m', np.nan),
            'max_error_m': stats_dataset.attrs.get('max_error_m', np.nan),
            'std_error_m': stats_dataset.attrs.get('std_error_m', np.nan),
            'n_measurements': stats_dataset.attrs.get('total_measurements', 0),
        }
    else:
        # Fallback for placeholder
        return {
            'rms_error_m': float(stats_dataset.get('rms_error', np.nan)),
            'mean_error_m': float(stats_dataset.get('mean_error', np.nan)),
            'max_error_m': float(stats_dataset.get('max_error', np.nan)),
            'std_error_m': float(stats_dataset.get('std_error', np.nan)),
            'n_measurements': int(stats_dataset.get('n_measurements', 0)),
        }


def _store_gcp_pair_results(netcdf_data, param_idx, pair_idx, error_metrics):
    """Store GCP pair results in the NetCDF data structure."""
    netcdf_data['rms_error_m'][param_idx, pair_idx] = error_metrics['rms_error_m']
    netcdf_data['mean_error_m'][param_idx, pair_idx] = error_metrics['mean_error_m']
    netcdf_data['max_error_m'][param_idx, pair_idx] = error_metrics['max_error_m']
    netcdf_data['std_error_m'][param_idx, pair_idx] = error_metrics['std_error_m']
    netcdf_data['n_measurements'][param_idx, pair_idx] = error_metrics['n_measurements']


def _compute_parameter_set_metrics(netcdf_data, param_idx, pair_errors):
    """Compute overall performance metrics for a parameter set."""
    pair_errors = np.array(pair_errors)
    valid_errors = pair_errors[~np.isnan(pair_errors)]

    if len(valid_errors) > 0:
        # Percentage of pairs with error < 250m
        percent_under_250 = (valid_errors < 250.0).sum() / len(valid_errors) * 100
        netcdf_data['percent_under_250m'][param_idx] = percent_under_250

        # Mean RMS across all pairs
        netcdf_data['mean_rms_all_pairs'][param_idx] = np.mean(valid_errors)

        # Best and worst pair performance
        netcdf_data['best_pair_rms'][param_idx] = np.min(valid_errors)
        netcdf_data['worst_pair_rms'][param_idx] = np.max(valid_errors)


def _save_netcdf_results(netcdf_data, output_file, config):
    """Save results to NetCDF file with proper dimensions and metadata."""
    import xarray as xr

    # Create coordinate arrays
    coords = {
        'parameter_set_id': netcdf_data['parameter_set_id'],
        'gcp_pair_id': netcdf_data['gcp_pair_id'],
    }

    # Create data variables
    data_vars = {}

    # Parameter values (1D: parameter_set_id)
    param_vars = ['param_hysics_roll', 'param_hysics_pitch', 'param_hysics_yaw',
                  'param_yoke_roll', 'param_yoke_pitch', 'param_yoke_yaw',
                  'param_base_roll', 'param_base_pitch', 'param_base_yaw',
                  'param_azimuth_bias', 'param_elevation_bias', 'param_time_correction']

    for var in param_vars:
        if var in netcdf_data:
            data_vars[var] = (['parameter_set_id'], netcdf_data[var])

    # Error metrics (2D: parameter_set_id, gcp_pair_id)
    error_vars = ['rms_error_m', 'mean_error_m', 'max_error_m', 'std_error_m', 'n_measurements']
    for var in error_vars:
        if var in netcdf_data:
            data_vars[var] = (['parameter_set_id', 'gcp_pair_id'], netcdf_data[var])

    # Overall metrics (1D: parameter_set_id)
    overall_vars = ['percent_under_250m', 'mean_rms_all_pairs', 'worst_pair_rms', 'best_pair_rms']
    for var in overall_vars:
        if var in netcdf_data:
            data_vars[var] = (['parameter_set_id'], netcdf_data[var])

    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)

    # Add metadata - handle None values properly for NetCDF
    ds.attrs.update({
        'title': 'CLARREO Geolocation Monte Carlo Analysis Results',
        'description': 'Parameter sensitivity analysis for CLARREO geolocation system',
        'created': pd.Timestamp.now().isoformat(),
        'monte_carlo_iterations': config.n_iterations,
        'performance_threshold_m': 250.0,
        'parameter_count': len(config.parameters),
        'random_seed': config.seed if config.seed is not None else 'None',  # Convert None to string
    })

    # Add variable attributes
    var_attrs = {
        'param_hysics_roll': {'units': 'arcseconds', 'long_name': 'HySICS to cradle roll correction'},
        'param_hysics_pitch': {'units': 'arcseconds', 'long_name': 'HySICS to cradle pitch correction'},
        'param_hysics_yaw': {'units': 'arcseconds', 'long_name': 'HySICS to cradle yaw correction'},
        'param_yoke_roll': {'units': 'arcseconds', 'long_name': 'Yoke elevation to azimuth roll correction'},
        'param_yoke_pitch': {'units': 'arcseconds', 'long_name': 'Yoke elevation to azimuth pitch correction'},
        'param_yoke_yaw': {'units': 'arcseconds', 'long_name': 'Yoke elevation to azimuth yaw correction'},
        'param_base_roll': {'units': 'arcseconds', 'long_name': 'Base azimuth to cube roll correction'},
        'param_base_pitch': {'units': 'arcseconds', 'long_name': 'Base azimuth to cube pitch correction'},
        'param_base_yaw': {'units': 'arcseconds', 'long_name': 'Base azimuth to cube yaw correction'},
        'param_azimuth_bias': {'units': 'arcseconds', 'long_name': 'Azimuth angle bias correction'},
        'param_elevation_bias': {'units': 'arcseconds', 'long_name': 'Elevation angle bias correction'},
        'param_time_correction': {'units': 'milliseconds', 'long_name': 'Science frame time correction'},
        'rms_error_m': {'units': 'meters', 'long_name': 'RMS geolocation error'},
        'mean_error_m': {'units': 'meters', 'long_name': 'Mean geolocation error'},
        'max_error_m': {'units': 'meters', 'long_name': 'Maximum geolocation error'},
        'std_error_m': {'units': 'meters', 'long_name': 'Standard deviation of geolocation error'},
        'n_measurements': {'units': 'count', 'long_name': 'Number of measurement points'},
        'percent_under_250m': {'units': 'percent', 'long_name': 'Percentage of pairs with error < 250m'},
        'mean_rms_all_pairs': {'units': 'meters', 'long_name': 'Mean RMS error across all GCP pairs'},
        'worst_pair_rms': {'units': 'meters', 'long_name': 'Worst performing GCP pair RMS error'},
        'best_pair_rms': {'units': 'meters', 'long_name': 'Best performing GCP pair RMS error'},
    }

    for var, attrs in var_attrs.items():
        if var in ds.data_vars:
            ds[var].attrs.update(attrs)

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file)
    logger.info(f"NetCDF file saved: {ds.sizes}")
    logger.info(f"Data variables: {list(ds.data_vars.keys())}")
