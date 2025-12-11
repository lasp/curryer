"""
Geolocation statistics processor with Xarray inputs and outputs.

This module processes geolocation errors from the GCS image matching algorithm and
produces performance verification metrics, specifically the nadir-equivalent geolocation errors
which pass when less than 250m.

The main processing pipeline:
1) Convert angular errors to N-S and E-W distances
2) Transform error components to view-plane/cross-view-plane distances
3) Scale to nadir-equivalent using geometric factors
4) Compute statistical performance metrics
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class GeolocationConfig:
    """Configuration parameters for geolocation processing.

    All values should be provided from MonteCarloConfig - no hardcoded defaults.
    """

    earth_radius_m: float  # Earth radius in meters (e.g., WGS84: 6378140.0)
    performance_threshold_m: float  # Accuracy threshold (e.g., 250.0)
    performance_spec_percent: float  # Performance requirement percentage (e.g., 39.0)
    minimum_correlation: float | None = None  # Filter threshold (0.0-1.0)

    # Mission-agnostic variable name mappings
    # Maps semantic names to actual variable names in the dataset
    variable_names: dict[str, str] | None = None  # If None, uses CLARREO defaults

    @classmethod
    def from_monte_carlo_config(cls, mc_config) -> "GeolocationConfig":
        """
        Create GeolocationConfig from MonteCarloConfig.

        This is the preferred way to create this config - extracts all settings
        from the single source of truth (MonteCarloConfig).

        Args:
            mc_config: MonteCarloConfig instance

        Returns:
            GeolocationConfig with settings from MonteCarloConfig
        """
        # Create variable name mapping
        variable_names = {
            "spacecraft_position": mc_config.spacecraft_position_name,
            "boresight": mc_config.boresight_name,
            "transformation_matrix": mc_config.transformation_matrix_name,
        }

        return cls(
            earth_radius_m=mc_config.earth_radius_m,
            performance_threshold_m=mc_config.performance_threshold_m,
            performance_spec_percent=mc_config.performance_spec_percent,
            minimum_correlation=mc_config.geo.minimum_correlation,
            variable_names=variable_names,
        )

    def get_variable_name(self, semantic_name: str) -> str:
        """
        Get actual variable name for a semantic concept.

        Args:
            semantic_name: Semantic name like 'spacecraft_position', 'boresight', etc.

        Returns:
            Actual variable name in the dataset

        Raises:
            ValueError: If variable_names not provided and semantic_name not found
        """
        if self.variable_names is None:
            raise ValueError(
                f"GeolocationConfig.variable_names is None. "
                f"Use GeolocationConfig.from_monte_carlo_config() to create config with proper variable names."
            )

        if semantic_name not in self.variable_names:
            raise ValueError(
                f"Variable name mapping for '{semantic_name}' not found in config. "
                f"Available mappings: {list(self.variable_names.keys())}"
            )

        return self.variable_names[semantic_name]


class ErrorStatsProcessor:
    """Production-ready processor for geolocation error statistics."""

    def __init__(self, config: GeolocationConfig):
        """
        Initialize processor with configuration.

        Args:
            config: GeolocationConfig (required) - use GeolocationConfig.from_monte_carlo_config()
                   to create from MonteCarloConfig
        """
        if config is None:
            raise ValueError(
                "GeolocationConfig is required. Use GeolocationConfig.from_monte_carlo_config(mc_config) to create."
            )
        self.config = config

    def _filter_by_correlation(self, data: xr.Dataset) -> xr.Dataset:
        """
        Filter measurements by correlation coefficient threshold.

        Args:
            data: Input dataset with optional 'correlation' or 'ccv' variable

        Returns:
            Filtered dataset with low-correlation measurements removed
        """
        if self.config.minimum_correlation is None:
            return data

        # Check for correlation variable (try multiple names)
        corr_var = None
        for var_name in ["correlation", "ccv", "im_ccv"]:
            if var_name in data.data_vars:
                corr_var = var_name
                break

        if corr_var is None:
            logger.warning("No correlation variable found; skipping filtering")
            return data

        # Apply filter
        valid_mask = data[corr_var] >= self.config.minimum_correlation
        n_before = len(data.measurement)
        filtered_data = data.where(valid_mask, drop=True)
        n_after = len(filtered_data.measurement)

        logger.info(
            f"Correlation filtering: {n_before} → {n_after} measurements (threshold={self.config.minimum_correlation})"
        )

        return filtered_data

    def process_geolocation_errors(self, input_data: xr.Dataset) -> xr.Dataset:
        """
        Process geolocation errors from input dataset to nadir-equivalent statistics.

        Args:
            input_data: Xarray Dataset with required error measurement variables

        Returns:
            Xarray Dataset with processed results and statistics
        """
        # Validate input data
        self._validate_input_data(input_data)

        # NEW: Apply correlation filtering if configured
        filtered_data = self._filter_by_correlation(input_data)

        if len(filtered_data.measurement) == 0:
            raise ValueError("No measurements remaining after correlation filtering")

        # Extract data arrays (now using filtered_data)
        n_measurements = len(filtered_data.measurement)

        # Get actual variable names from config
        sc_pos_var = self.config.get_variable_name("spacecraft_position")
        boresight_var = self.config.get_variable_name("boresight")
        transform_var = self.config.get_variable_name("transformation_matrix")

        # Convert angular errors to distance errors
        lat_error_rad = np.deg2rad(filtered_data.lat_error_deg.values)
        lon_error_rad = np.deg2rad(filtered_data.lon_error_deg.values)
        gcp_lat_rad = np.deg2rad(filtered_data.gcp_lat_deg.values)
        gcp_lon_rad = np.deg2rad(filtered_data.gcp_lon_deg.values)

        # Calculate N-S and E-W error distances in meters
        ns_error_dist_m = self.config.earth_radius_m * lat_error_rad
        ew_error_dist_m = self.config.earth_radius_m * np.cos(gcp_lat_rad) * lon_error_rad

        # Transform boresight vectors using configurable variable names
        bhat_ctrs = self._transform_boresight_vectors(
            filtered_data[boresight_var].values, filtered_data[transform_var].values
        )

        # Process each measurement to nadir-equivalent using configurable variable name
        results = self._process_to_nadir_equivalent(
            ns_error_dist_m,
            ew_error_dist_m,
            filtered_data[sc_pos_var].values,
            bhat_ctrs,
            gcp_lat_rad,
            gcp_lon_rad,
            n_measurements,
        )

        # Create output dataset
        output_data = self._create_output_dataset(filtered_data, results)

        # Add statistics as global attributes
        stats = self._calculate_statistics(results["nadir_equiv_total_error_m"])
        output_data.attrs.update(stats)

        return output_data

    def _validate_input_data(self, data: xr.Dataset) -> None:
        """Validate that input dataset contains all required variables."""
        # Get actual variable names from config
        sc_pos_var = self.config.get_variable_name("spacecraft_position")
        boresight_var = self.config.get_variable_name("boresight")
        transform_var = self.config.get_variable_name("transformation_matrix")

        required_vars = [
            "lat_error_deg",
            "lon_error_deg",
            sc_pos_var,
            boresight_var,
            transform_var,
            "gcp_lat_deg",
            "gcp_lon_deg",
            "gcp_alt",
        ]

        missing_vars = [var for var in required_vars if var not in data.data_vars]
        if missing_vars:
            raise ValueError(f"Missing required input variables: {missing_vars}")

        # Check dimensions
        if "measurement" not in data.dims:
            raise ValueError("Input data must have 'measurement' dimension")

    def _transform_boresight_vectors(self, bhat_hs: np.ndarray, t_hs2ctrs: np.ndarray) -> np.ndarray:
        """Transform boresight vectors from HS to CTRS coordinate system."""
        n_measurements = bhat_hs.shape[0]
        bhat_ctrs = np.zeros((n_measurements, 3))

        for i in range(n_measurements):
            bhat_ctrs[i] = bhat_hs[i] @ t_hs2ctrs[i, :, :].T
        return bhat_ctrs

    def _process_to_nadir_equivalent(
        self,
        ns_error_m: np.ndarray,
        ew_error_m: np.ndarray,
        riss_ctrs: np.ndarray,
        bhat_ctrs: np.ndarray,
        gcp_lat_rad: np.ndarray,
        gcp_lon_rad: np.ndarray,
        n_measurements: int,
    ) -> dict[str, np.ndarray]:
        """Process error measurements to nadir-equivalent values."""

        # Initialize result arrays
        results = {
            "vp_error_m": np.zeros(n_measurements),
            "xvp_error_m": np.zeros(n_measurements),
            "off_nadir_angle_rad": np.zeros(n_measurements),
            "vp_scaling_factor": np.zeros(n_measurements),
            "xvp_scaling_factor": np.zeros(n_measurements),
            "nadir_equiv_vp_error_m": np.zeros(n_measurements),
            "nadir_equiv_xvp_error_m": np.zeros(n_measurements),
            "nadir_equiv_total_error_m": np.zeros(n_measurements),
        }

        for i in range(n_measurements):
            # Create transformation matrix from CTRS to Up-East-North (UEN)
            t_ctrs2uen = self._create_ctrs_to_uen_transform(gcp_lat_rad[i], gcp_lon_rad[i])

            # Transform boresight vector to UEN coordinates
            bhat_uen = bhat_ctrs[i] @ t_ctrs2uen.T

            # Calculate view-plane and cross-view-plane unit vectors in UEN
            v_uen, x_uen = self._calculate_view_plane_vectors(bhat_uen)

            # Create UEN to UXV transformation matrix
            t_uen2uxv = np.eye(3)
            t_uen2uxv[1] = x_uen  # Cross-view-plane direction
            t_uen2uxv[2] = v_uen  # View-plane direction

            # Transform error distances to view-plane coordinates
            error_uen = np.array([0, ew_error_m[i], ns_error_m[i]])
            error_uxv = error_uen @ t_uen2uxv.T
            results["xvp_error_m"][i] = error_uxv[1]  # Cross-view-plane error
            results["vp_error_m"][i] = error_uxv[2]  # View-plane error

            # Calculate off-nadir angle and scaling factors
            rhat = riss_ctrs[i] / np.linalg.norm(riss_ctrs[i])
            # Clip dot product to avoid tiny rounding errors outside [-1, 1]
            dot_product = np.clip(np.dot(bhat_ctrs[i], -rhat), -1.0, 1.0)
            results["off_nadir_angle_rad"][i] = np.arccos(dot_product)

            # Calculate nadir-equivalent scaling factors
            scaling_factors = self._calculate_scaling_factors(riss_ctrs[i], results["off_nadir_angle_rad"][i])
            results["vp_scaling_factor"][i] = scaling_factors[0]
            results["xvp_scaling_factor"][i] = scaling_factors[1]

            # Apply scaling to get nadir-equivalent errors
            results["nadir_equiv_vp_error_m"][i] = results["vp_error_m"][i] * scaling_factors[0]
            results["nadir_equiv_xvp_error_m"][i] = results["xvp_error_m"][i] * scaling_factors[1]
            results["nadir_equiv_total_error_m"][i] = np.sqrt(
                results["nadir_equiv_vp_error_m"][i] ** 2 + results["nadir_equiv_xvp_error_m"][i] ** 2
            )

        return results

    def _create_ctrs_to_uen_transform(self, lat_rad: float, lon_rad: float) -> np.ndarray:
        """Create transformation matrix from CTRS to Up-East-North coordinates."""
        t_ctrs2uen = np.zeros((3, 3))

        # Up direction (radial outward)
        t_ctrs2uen[0] = [np.cos(lon_rad) * np.cos(lat_rad), np.sin(lon_rad) * np.cos(lat_rad), np.sin(lat_rad)]

        # East direction
        t_ctrs2uen[1] = [-np.sin(lon_rad), np.cos(lon_rad), 0]

        # North direction
        t_ctrs2uen[2] = [-np.cos(lon_rad) * np.sin(lat_rad), -np.sin(lon_rad) * np.sin(lat_rad), np.cos(lat_rad)]

        return t_ctrs2uen

    def _calculate_view_plane_vectors(self, bhat_uen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate view-plane and cross-view-plane unit vectors in UEN coordinates."""
        # Calculate normalization factor for horizontal components
        norm_factor = np.sqrt(bhat_uen[1] ** 2 + bhat_uen[2] ** 2)

        # View-plane direction (in the direction of boresight horizontal projection)
        v_uen = np.array([0, bhat_uen[1], bhat_uen[2]]) / norm_factor

        # Cross-view-plane direction (perpendicular to view-plane in horizontal)
        x_uen = np.array([0, bhat_uen[2], -bhat_uen[1]]) / norm_factor

        return v_uen, x_uen

    def _calculate_scaling_factors(self, riss_ctrs: np.ndarray, theta: float) -> tuple[float, float]:
        """Calculate scaling factors for nadir-equivalent transformation."""
        r_magnitude = np.linalg.norm(riss_ctrs)
        f = r_magnitude / self.config.earth_radius_m
        h = r_magnitude - self.config.earth_radius_m

        # Calculate discriminant for sqrt - should be positive for physically valid geometries
        discriminant = 1 - f**2 * np.sin(theta) ** 2

        # Check for suspicious geometries
        if discriminant < 0:  # Significantly negative suggests bad input data
            logger.error(
                f"Suspicious geometry: discriminant={discriminant:.6f} for f={f:.3f}, theta={np.rad2deg(theta):.1f}°. "
                f"This suggests Invalid geometry (no-intersection)."
            )

        temp1 = np.sqrt(discriminant)

        # Add small epsilon to prevent division by zero for extreme cases
        # (when discriminant rounds to exactly 0)
        temp1 = np.maximum(temp1, 1e-10)

        # View-plane scaling factor
        vp_factor = h / self.config.earth_radius_m / (-1 + f * np.cos(theta) / temp1)

        # Cross-view-plane scaling factor
        xvp_factor = h / self.config.earth_radius_m / np.cos(theta) / (f * np.cos(theta) - temp1)

        return vp_factor, xvp_factor

    def _create_output_dataset(self, input_data: xr.Dataset, results: dict[str, np.ndarray]) -> xr.Dataset:
        """Create output Xarray Dataset with processing results."""

        # Create data variables for output
        data_vars = {}

        # Nadir-equivalent errors (main results)
        data_vars["nadir_equiv_total_error_m"] = (
            ["measurement"],
            results["nadir_equiv_total_error_m"],
            {"units": "meters", "long_name": "Total nadir-equivalent geolocation error"},
        )

        data_vars["nadir_equiv_vp_error_m"] = (
            ["measurement"],
            results["nadir_equiv_vp_error_m"],
            {"units": "meters", "long_name": "View-plane nadir-equivalent error"},
        )

        data_vars["nadir_equiv_xvp_error_m"] = (
            ["measurement"],
            results["nadir_equiv_xvp_error_m"],
            {"units": "meters", "long_name": "Cross-view-plane nadir-equivalent error"},
        )

        # Intermediate processing results
        data_vars["vp_error_m"] = (
            ["measurement"],
            results["vp_error_m"],
            {"units": "meters", "long_name": "View-plane error distance"},
        )

        data_vars["xvp_error_m"] = (
            ["measurement"],
            results["xvp_error_m"],
            {"units": "meters", "long_name": "Cross-view-plane error distance"},
        )

        data_vars["off_nadir_angle_deg"] = (
            ["measurement"],
            np.rad2deg(results["off_nadir_angle_rad"]),
            {"units": "degrees", "long_name": "Off-nadir viewing angle"},
        )

        data_vars["vp_scaling_factor"] = (
            ["measurement"],
            results["vp_scaling_factor"],
            {"units": "dimensionless", "long_name": "View-plane nadir scaling factor"},
        )

        data_vars["xvp_scaling_factor"] = (
            ["measurement"],
            results["xvp_scaling_factor"],
            {"units": "dimensionless", "long_name": "Cross-view-plane nadir scaling factor"},
        )

        # Preserve original input data as reference
        for var in input_data.data_vars:
            if var not in data_vars:  # Don't duplicate variables
                data_vars[var] = input_data[var]

        # Create output dataset
        output_ds = xr.Dataset(
            data_vars=data_vars,
            coords=input_data.coords,
            attrs={
                "title": "Geolocation Error Statistics Results",
                "processing_timestamp": np.datetime64("now"),
                "earth_radius_m": self.config.earth_radius_m,
                "performance_threshold_m": self.config.performance_threshold_m,
            },
        )

        # Add correlation filtering metadata if applied
        if self.config.minimum_correlation is not None:
            output_ds.attrs["minimum_correlation_threshold"] = self.config.minimum_correlation
            output_ds.attrs["correlation_filtering_applied"] = True

        return output_ds

    def _calculate_statistics(self, nadir_equiv_errors_m: np.ndarray) -> dict[str, float | int]:
        """Calculate performance statistics on nadir-equivalent errors."""

        # Count errors below threshold
        num_below_threshold = np.sum(nadir_equiv_errors_m < self.config.performance_threshold_m)

        # Calculate statistics
        stats = {
            "mean_error_distance_m": float(np.mean(nadir_equiv_errors_m)),
            "std_error_distance_m": float(np.std(nadir_equiv_errors_m)),
            "min_error_distance_m": float(np.min(nadir_equiv_errors_m)),
            "max_error_distance_m": float(np.max(nadir_equiv_errors_m)),
            "percent_below_250m": float(num_below_threshold / len(nadir_equiv_errors_m) * 100),
            "num_below_250m": int(num_below_threshold),
            "total_measurements": int(len(nadir_equiv_errors_m)),
            "performance_spec_met": bool(
                num_below_threshold / len(nadir_equiv_errors_m) * 100 > self.config.performance_spec_percent
            ),
        }

        return stats

    def process_from_netcdf(self, filepath: Union[str, "Path"], minimum_correlation: float | None = None) -> xr.Dataset:
        """
        Load previous results from NetCDF and reprocess error statistics.

        This enables iterative post-processing of Monte Carlo results without
        re-running expensive image matching operations.

        Args:
            filepath: Path to NetCDF file from previous Monte Carlo run
            minimum_correlation: Override correlation threshold (if provided)

        Returns:
            Xarray Dataset with reprocessed error statistics

        Example:
            >>> processor = ErrorStatsProcessor()
            >>> # Try different correlation thresholds
            >>> results_50 = processor.process_from_netcdf(
            ...     "monte_carlo_results/run_001.nc",
            ...     minimum_correlation=0.5
            ... )
            >>> results_70 = processor.process_from_netcdf(
            ...     "monte_carlo_results/run_001.nc",
            ...     minimum_correlation=0.7
            ... )
        """

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"NetCDF file not found: {filepath}")

        logger.info(f"Loading NetCDF results from: {filepath}")
        input_data = xr.open_dataset(filepath)

        # Override correlation threshold if provided
        original_threshold = self.config.minimum_correlation
        if minimum_correlation is not None:
            self.config.minimum_correlation = minimum_correlation
            logger.info(f"Overriding correlation threshold: {original_threshold} → {minimum_correlation}")

        # Validate that required variables exist
        try:
            self._validate_input_data(input_data)
        except ValueError as e:
            raise ValueError(
                f"NetCDF file missing required variables for error stats: {e}\n"
                f"Available variables: {list(input_data.data_vars.keys())}"
            )

        # Reprocess with current configuration
        results = self.process_geolocation_errors(input_data)

        # Add metadata about reprocessing
        results.attrs["reprocessed_from"] = str(filepath)
        results.attrs["reprocessing_date"] = str(np.datetime64("now"))
        if minimum_correlation is not None:
            results.attrs["correlation_threshold_override"] = minimum_correlation

        # Restore original threshold
        self.config.minimum_correlation = original_threshold

        return results
