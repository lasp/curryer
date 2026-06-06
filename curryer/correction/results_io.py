"""NetCDF I/O helpers for the correction pipeline.

This module owns all read/write operations on the NetCDF result file:

- :func:`_build_netcdf_structure` -- initialise the in-memory data dict.
- :func:`_save_netcdf_results` -- write the final NetCDF output file.
- :func:`_save_netcdf_checkpoint` -- write a partial checkpoint after each GCP pair.
- :func:`_load_checkpoint` -- reload a checkpoint to resume an interrupted run.
- :func:`_cleanup_checkpoint` -- delete the checkpoint file after a successful run.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from curryer.correction.config import GeolocationSetup, NetCDFConfig, ParameterType, Sweep

logger = logging.getLogger(__name__)


def _build_netcdf_structure(
    setup: GeolocationSetup,
    sweep: Sweep,
    netcdf_config: NetCDFConfig,
    n_param_sets: int,
    n_gcp_pairs: int,
) -> dict:
    """
    Build NetCDF data structure dynamically from the sweep + output config.

    This creates the netcdf_data dictionary with proper variable names based on
    the parameters defined in the sweep, avoiding hardcoded mission-specific names.

    Args:
        setup: GeolocationSetup (performance threshold for metric naming)
        sweep: Sweep with the parameters to vary
        netcdf_config: Resolved NetCDFConfig with metadata
        n_param_sets: Number of parameter sets (iterations)
        n_gcp_pairs: Number of GCP pairs

    Returns:
        Dictionary with initialized arrays for all NetCDF variables
    """
    logger.info(f"Building NetCDF data structure for {n_param_sets} parameter sets × {n_gcp_pairs} GCP pairs")

    # Start with coordinate dimensions
    netcdf_data = {
        "parameter_set_id": np.arange(n_param_sets),
        "gcp_pair_id": np.arange(n_gcp_pairs),
    }

    # Add parameter variables dynamically based on sweep.parameters
    param_count = 0
    for param in sweep.parameters:
        if param.ptype == ParameterType.CONSTANT_KERNEL:
            # CONSTANT_KERNEL parameters have roll, pitch, yaw components
            for angle in ["roll", "pitch", "yaw"]:
                metadata = netcdf_config.get_parameter_netcdf_metadata(param, angle)
                var_name = metadata.variable_name
                netcdf_data[var_name] = np.full(n_param_sets, np.nan)
                logger.debug(f"  Added parameter variable: {var_name} ({metadata.long_name})")
                param_count += 1
        else:
            # OFFSET_KERNEL and OFFSET_TIME are single values
            metadata = netcdf_config.get_parameter_netcdf_metadata(param)
            var_name = metadata.variable_name
            netcdf_data[var_name] = np.full(n_param_sets, np.nan)
            logger.debug(f"  Added parameter variable: {var_name} ({metadata.long_name})")
            param_count += 1

    logger.info(f"  Created {param_count} parameter variables from {len(sweep.parameters)} parameter configs")

    # Add standard error statistics (2D: parameter_set_id × gcp_pair_id)
    error_metrics = {
        "rms_error_m": "RMS geolocation error",
        "mean_error_m": "Mean geolocation error",
        "max_error_m": "Maximum geolocation error",
        "std_error_m": "Standard deviation of geolocation error",
        "n_measurements": "Number of measurement points",
    }

    for var_name, description in error_metrics.items():
        if var_name == "n_measurements":
            netcdf_data[var_name] = np.full((n_param_sets, n_gcp_pairs), 0, dtype=int)
        else:
            netcdf_data[var_name] = np.full((n_param_sets, n_gcp_pairs), np.nan)
        logger.debug(f"  Added error metric: {var_name}")

    # Add image matching results (2D: parameter_set_id × gcp_pair_id)
    image_match_vars = {
        "im_lat_error_km": "Image matching latitude error",
        "im_lon_error_km": "Image matching longitude error",
        "im_ccv": "Image matching correlation coefficient",
        "im_grid_step_m": "Image matching final grid step size",
    }

    for var_name, description in image_match_vars.items():
        netcdf_data[var_name] = np.full((n_param_sets, n_gcp_pairs), np.nan)
        logger.debug(f"  Added image matching variable: {var_name}")

    # Add overall performance metrics (1D: parameter_set_id)
    # Use dynamic threshold metric name
    threshold_metric = netcdf_config.threshold_metric_name
    overall_metrics = {
        threshold_metric: f"Percentage of pairs with error < {setup.requirements.performance_threshold_m}m",
        "mean_rms_all_pairs": "Mean RMS error across all GCP pairs",
        "worst_pair_rms": "Worst performing GCP pair RMS error",
        "best_pair_rms": "Best performing GCP pair RMS error",
    }

    for var_name, description in overall_metrics.items():
        netcdf_data[var_name] = np.full(n_param_sets, np.nan)
        logger.debug(f"  Added overall metric: {var_name}")

    logger.info(f"NetCDF data structure created with {len(netcdf_data)} variables")

    return netcdf_data


def _save_netcdf_checkpoint(netcdf_data, output_file, setup, sweep, netcdf_config, pair_idx_completed):
    """
    Save NetCDF checkpoint with partial results after each GCP pair completes.

    This enables resuming correction runs if they are interrupted.
    Adapted for pair-outer loop order where each pair processes all parameters.

    Args:
        netcdf_data: Dictionary with current NetCDF data
        output_file: Path to final output file (checkpoint uses .checkpoint.nc suffix)
        setup: GeolocationSetup (performance threshold for metric naming)
        sweep: Sweep with parameters/iterations/seed
        netcdf_config: Resolved NetCDFConfig with metadata
        pair_idx_completed: Index of the last completed GCP pair (for pair-outer loop)
    """

    checkpoint_file = output_file.parent / f"{output_file.stem}_checkpoint.nc"

    # Create coordinate arrays
    coords = {
        "parameter_set_id": netcdf_data["parameter_set_id"],
        "gcp_pair_id": netcdf_data["gcp_pair_id"],
    }

    # Build variable list dynamically from netcdf_data keys
    data_vars = {}
    for var_name, var_data in netcdf_data.items():
        if var_name not in coords:
            if isinstance(var_data, np.ndarray):
                if var_data.ndim == 1:
                    data_vars[var_name] = (["parameter_set_id"], var_data)
                elif var_data.ndim == 2:
                    data_vars[var_name] = (["parameter_set_id", "gcp_pair_id"], var_data)

    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)

    # Add regular metadata
    ds.attrs.update(
        {
            "title": netcdf_config.title,
            "description": netcdf_config.description,
            "created": pd.Timestamp.now().isoformat(),
            "correction_iterations": sweep.n_iterations,
            "performance_threshold_m": netcdf_config.performance_threshold_m,
            "parameter_count": len(sweep.parameters),
            "random_seed": str(sweep.seed) if sweep.seed is not None else "None",
        }
    )

    # Add checkpoint-specific metadata (NetCDF-compatible types)
    ds.attrs["checkpoint"] = 1  # Use integer instead of boolean for NetCDF compatibility
    ds.attrs["completed_gcp_pairs"] = int(pair_idx_completed + 1)
    ds.attrs["total_gcp_pairs"] = int(len(netcdf_data["gcp_pair_id"]))
    ds.attrs["checkpoint_timestamp"] = pd.Timestamp.now().isoformat()

    # Add parameter variable attributes from config
    for param in sweep.parameters:
        if param.ptype == ParameterType.CONSTANT_KERNEL:
            for angle in ["roll", "pitch", "yaw"]:
                metadata = netcdf_config.get_parameter_netcdf_metadata(param, angle)
                if metadata.variable_name in ds.data_vars:
                    ds[metadata.variable_name].attrs.update({"units": metadata.units, "long_name": metadata.long_name})
        else:
            metadata = netcdf_config.get_parameter_netcdf_metadata(param)
            if metadata.variable_name in ds.data_vars:
                ds[metadata.variable_name].attrs.update({"units": metadata.units, "long_name": metadata.long_name})

    # Add standard metric attributes
    standard_attrs = netcdf_config.standard_attributes_dict
    threshold_metric = netcdf_config.threshold_metric_name
    standard_attrs[threshold_metric] = {
        "units": "percent",
        "long_name": f"Percentage of pairs with error < {setup.requirements.performance_threshold_m}m",
    }
    for var, attrs in standard_attrs.items():
        if var in ds.data_vars:
            ds[var].attrs.update(attrs)

    # Save to file in one operation
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(checkpoint_file, mode="w")  # Force overwrite mode
    ds.close()

    logger.info(f"  Checkpoint saved: {pair_idx_completed + 1}/{len(netcdf_data['gcp_pair_id'])} GCP pairs complete")


def _load_checkpoint(output_file):
    """
    Load checkpoint if it exists and convert back to netcdf_data dict.

    Args:
        output_file: Path to final output file (will check for .checkpoint.nc)

    Returns:
        Tuple of (netcdf_data dict, start_idx) or (None, 0) if no checkpoint
    """

    checkpoint_file = output_file.parent / f"{output_file.stem}_checkpoint.nc"

    if not checkpoint_file.exists():
        return None, 0

    logger.info(f"Found checkpoint file: {checkpoint_file}")

    try:
        ds = xr.open_dataset(checkpoint_file, decode_timedelta=False)

        # Verify this is actually a checkpoint (checkpoint attribute is 1 for true, 0 or missing for false)
        checkpoint_flag = ds.attrs.get("checkpoint", 0)
        if not checkpoint_flag:  # Will be True if checkpoint=1, False if checkpoint=0 or missing
            logger.warning("File exists but is not marked as checkpoint, ignoring")
            ds.close()
            return None, 0

        completed = ds.attrs.get("completed_gcp_pairs", 0)
        total = ds.attrs.get("total_gcp_pairs", 0)
        timestamp = ds.attrs.get("checkpoint_timestamp", "unknown")

        logger.info(f"Checkpoint from {timestamp}: {completed}/{total} GCP pairs complete")

        # Convert xarray.Dataset back to netcdf_data dictionary
        netcdf_data = {}

        # Add coordinates
        netcdf_data["parameter_set_id"] = ds.coords["parameter_set_id"].values
        netcdf_data["gcp_pair_id"] = ds.coords["gcp_pair_id"].values

        # Add all data variables
        for var_name in ds.data_vars:
            netcdf_data[var_name] = ds[var_name].values

        ds.close()

        logger.info(f"Checkpoint loaded successfully, resuming from GCP pair {completed}")

        return netcdf_data, completed

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None, 0


def _cleanup_checkpoint(output_file):
    """
    Remove checkpoint file after successful completion.

    Args:
        output_file: Path to final output file (will remove .checkpoint.nc)
    """
    checkpoint_file = output_file.parent / f"{output_file.stem}_checkpoint.nc"

    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            logger.info(f"Checkpoint file cleaned up: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint file: {e}")


def _save_netcdf_results(netcdf_data, output_file, setup, sweep, netcdf_config):
    """
    Save results to NetCDF file using config-driven metadata.

    This function dynamically builds the NetCDF file structure from the
    netcdf_data dictionary, using configuration for all metadata rather
    than hardcoding mission-specific values.

    Args:
        netcdf_data: Dictionary with all NetCDF variables and data
        output_file: Path to output NetCDF file
        setup: GeolocationSetup (performance threshold for metric naming)
        sweep: Sweep with parameters/iterations/seed
        netcdf_config: Resolved NetCDFConfig with metadata
    """

    logger.info(f"Saving NetCDF results to: {output_file}")

    # Create coordinate arrays
    coords = {
        "parameter_set_id": netcdf_data["parameter_set_id"],
        "gcp_pair_id": netcdf_data["gcp_pair_id"],
    }

    # Build variable list dynamically from netcdf_data keys
    data_vars = {}

    # Add all non-coordinate variables, determining dimensions from array shape
    for var_name, var_data in netcdf_data.items():
        if var_name not in coords:
            # Determine dimensions from array shape
            if isinstance(var_data, np.ndarray):
                if var_data.ndim == 1:
                    data_vars[var_name] = (["parameter_set_id"], var_data)
                elif var_data.ndim == 2:
                    data_vars[var_name] = (["parameter_set_id", "gcp_pair_id"], var_data)

    logger.info(f"  Creating dataset with {len(data_vars)} data variables")

    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)

    # Add global metadata from config
    ds.attrs.update(
        {
            "title": netcdf_config.title,
            "description": netcdf_config.description,
            "created": pd.Timestamp.now().isoformat(),
            "correction_iterations": sweep.n_iterations,
            "performance_threshold_m": netcdf_config.performance_threshold_m,
            "parameter_count": len(sweep.parameters),
            "random_seed": str(sweep.seed) if sweep.seed is not None else "None",
        }
    )

    # Add parameter variable attributes from config
    for param in sweep.parameters:
        if param.ptype == ParameterType.CONSTANT_KERNEL:
            # Add metadata for roll, pitch, yaw components
            for angle in ["roll", "pitch", "yaw"]:
                metadata = netcdf_config.get_parameter_netcdf_metadata(param, angle)
                if metadata.variable_name in ds.data_vars:
                    ds[metadata.variable_name].attrs.update({"units": metadata.units, "long_name": metadata.long_name})
        else:
            # Add metadata for single-value parameters
            metadata = netcdf_config.get_parameter_netcdf_metadata(param)
            if metadata.variable_name in ds.data_vars:
                ds[metadata.variable_name].attrs.update({"units": metadata.units, "long_name": metadata.long_name})

    # Add standard metric attributes from config (allows mission overrides)
    standard_attrs = netcdf_config.standard_attributes_dict

    # Add dynamic threshold metric
    threshold_metric = netcdf_config.threshold_metric_name
    standard_attrs[threshold_metric] = {
        "units": "percent",
        "long_name": f"Percentage of pairs with error < {setup.requirements.performance_threshold_m}m",
    }

    for var, attrs in standard_attrs.items():
        if var in ds.data_vars:
            ds[var].attrs.update(attrs)

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file)

    logger.info(f"  NetCDF file saved successfully")
    logger.info(f"  Dimensions: {dict(ds.sizes)}")
    logger.info(f"  Data variables: {len(list(ds.data_vars.keys()))}")
    logger.info(f"  File: {output_file}")
