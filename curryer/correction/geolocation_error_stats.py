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
import numpy as np
import xarray as xr
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeolocationConfig:
    """Configuration parameters for geolocation processing."""
    earth_radius_m: float = 6378140.0
    performance_threshold_m: float = 250.0
    performance_spec_percent: float = 39.0
    minimum_correlation: Optional[float] = None  # NEW: Filter threshold (0.0-1.0)


class ErrorStatsProcessor:
    """Production-ready processor for geolocation error statistics."""

    def __init__(self, config: Optional[GeolocationConfig] = None):
        """Initialize processor with configuration."""
        self.config = config or GeolocationConfig()

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
        for var_name in ['correlation', 'ccv', 'im_ccv']:
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

        logger.info(f"Correlation filtering: {n_before} → {n_after} measurements "
                    f"(threshold={self.config.minimum_correlation})")

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

        # Convert angular errors to distance errors
        lat_error_rad = np.deg2rad(filtered_data.lat_error_deg.values)
        lon_error_rad = np.deg2rad(filtered_data.lon_error_deg.values)
        cp_lat_rad = np.deg2rad(filtered_data.cp_lat_deg.values)
        cp_lon_rad = np.deg2rad(filtered_data.cp_lon_deg.values)

        # Calculate N-S and E-W error distances in meters
        ns_error_dist_m = self.config.earth_radius_m * lat_error_rad
        ew_error_dist_m = self.config.earth_radius_m * np.cos(cp_lat_rad) * lon_error_rad

        # Transform boresight vectors from HS to CTRS coordinates
        bhat_ctrs = self._transform_boresight_vectors(
            filtered_data.bhat_hs.values,
            filtered_data.t_hs2ctrs.values
        )

        # Process each measurement to nadir-equivalent
        results = self._process_to_nadir_equivalent(
            ns_error_dist_m, ew_error_dist_m,
            filtered_data.riss_ctrs.values, bhat_ctrs,
            cp_lat_rad, cp_lon_rad, n_measurements
        )

        # Create output dataset
        output_data = self._create_output_dataset(filtered_data, results)

        # Add statistics as global attributes
        stats = self._calculate_statistics(results['nadir_equiv_total_error_m'])
        output_data.attrs.update(stats)

        return output_data

    def _validate_input_data(self, data: xr.Dataset) -> None:
        """Validate that input dataset contains all required variables."""
        required_vars = [
            'lat_error_deg', 'lon_error_deg', 'riss_ctrs', 'bhat_hs',
            't_hs2ctrs', 'cp_lat_deg', 'cp_lon_deg', 'cp_alt'
        ]

        missing_vars = [var for var in required_vars if var not in data.data_vars]
        if missing_vars:
            raise ValueError(f"Missing required input variables: {missing_vars}")

        # Check dimensions
        if 'measurement' not in data.dims:
            raise ValueError("Input data must have 'measurement' dimension")

    def _transform_boresight_vectors(self, bhat_hs: np.ndarray, t_hs2ctrs: np.ndarray) -> np.ndarray:
        """Transform boresight vectors from HS to CTRS coordinate system."""
        n_measurements = bhat_hs.shape[0]
        bhat_ctrs = np.zeros((n_measurements, 3))

        for i in range(n_measurements):
            bhat_ctrs[i] = bhat_hs[i] @ t_hs2ctrs[:, :, i].T

        return bhat_ctrs

    def _process_to_nadir_equivalent(self, ns_error_m: np.ndarray, ew_error_m: np.ndarray,
                                   riss_ctrs: np.ndarray, bhat_ctrs: np.ndarray,
                                   cp_lat_rad: np.ndarray, cp_lon_rad: np.ndarray,
                                   n_measurements: int) -> Dict[str, np.ndarray]:
        """Process error measurements to nadir-equivalent values."""

        # Initialize result arrays
        results = {
            'vp_error_m': np.zeros(n_measurements),
            'xvp_error_m': np.zeros(n_measurements),
            'off_nadir_angle_rad': np.zeros(n_measurements),
            'vp_scaling_factor': np.zeros(n_measurements),
            'xvp_scaling_factor': np.zeros(n_measurements),
            'nadir_equiv_vp_error_m': np.zeros(n_measurements),
            'nadir_equiv_xvp_error_m': np.zeros(n_measurements),
            'nadir_equiv_total_error_m': np.zeros(n_measurements)
        }

        for i in range(n_measurements):
            # Create transformation matrix from CTRS to Up-East-North (UEN)
            t_ctrs2uen = self._create_ctrs_to_uen_transform(cp_lat_rad[i], cp_lon_rad[i])

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
            results['xvp_error_m'][i] = error_uxv[1]  # Cross-view-plane error
            results['vp_error_m'][i] = error_uxv[2]   # View-plane error

            # Calculate off-nadir angle and scaling factors
            rhat = riss_ctrs[i] / np.linalg.norm(riss_ctrs[i])
            results['off_nadir_angle_rad'][i] = np.arccos(np.dot(bhat_ctrs[i], -rhat))

            # Calculate nadir-equivalent scaling factors
            scaling_factors = self._calculate_scaling_factors(
                riss_ctrs[i], results['off_nadir_angle_rad'][i]
            )
            results['vp_scaling_factor'][i] = scaling_factors[0]
            results['xvp_scaling_factor'][i] = scaling_factors[1]

            # Apply scaling to get nadir-equivalent errors
            results['nadir_equiv_vp_error_m'][i] = results['vp_error_m'][i] * scaling_factors[0]
            results['nadir_equiv_xvp_error_m'][i] = results['xvp_error_m'][i] * scaling_factors[1]
            results['nadir_equiv_total_error_m'][i] = np.sqrt(
                results['nadir_equiv_vp_error_m'][i]**2 +
                results['nadir_equiv_xvp_error_m'][i]**2
            )

        return results

    def _create_ctrs_to_uen_transform(self, lat_rad: float, lon_rad: float) -> np.ndarray:
        """Create transformation matrix from CTRS to Up-East-North coordinates."""
        t_ctrs2uen = np.zeros((3, 3))

        # Up direction (radial outward)
        t_ctrs2uen[0] = [
            np.cos(lon_rad) * np.cos(lat_rad),
            np.sin(lon_rad) * np.cos(lat_rad),
            np.sin(lat_rad)
        ]

        # East direction
        t_ctrs2uen[1] = [
            -np.sin(lon_rad),
            np.cos(lon_rad),
            0
        ]

        # North direction
        t_ctrs2uen[2] = [
            -np.cos(lon_rad) * np.sin(lat_rad),
            -np.sin(lon_rad) * np.sin(lat_rad),
            np.cos(lat_rad)
        ]

        return t_ctrs2uen

    def _calculate_view_plane_vectors(self, bhat_uen: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate view-plane and cross-view-plane unit vectors in UEN coordinates."""
        # Calculate normalization factor for horizontal components
        norm_factor = np.sqrt(bhat_uen[1]**2 + bhat_uen[2]**2)

        # View-plane direction (in the direction of boresight horizontal projection)
        v_uen = np.array([0, bhat_uen[1], bhat_uen[2]]) / norm_factor

        # Cross-view-plane direction (perpendicular to view-plane in horizontal)
        x_uen = np.array([0, bhat_uen[2], -bhat_uen[1]]) / norm_factor

        return v_uen, x_uen

    def _calculate_scaling_factors(self, riss_ctrs: np.ndarray, theta: float) -> Tuple[float, float]:
        """Calculate scaling factors for nadir-equivalent transformation."""
        r_magnitude = np.linalg.norm(riss_ctrs)
        f = r_magnitude / self.config.earth_radius_m
        h = r_magnitude - self.config.earth_radius_m

        temp1 = np.sqrt(1 - f**2 * np.sin(theta)**2)

        # View-plane scaling factor
        vp_factor = h / self.config.earth_radius_m / (-1 + f * np.cos(theta) / temp1)

        # Cross-view-plane scaling factor
        xvp_factor = (h / self.config.earth_radius_m / np.cos(theta) /
                     (f * np.cos(theta) - temp1))

        return vp_factor, xvp_factor

    def _create_output_dataset(self, input_data: xr.Dataset,
                             results: Dict[str, np.ndarray]) -> xr.Dataset:
        """Create output Xarray Dataset with processing results."""

        # Create data variables for output
        data_vars = {}

        # Nadir-equivalent errors (main results)
        data_vars['nadir_equiv_total_error_m'] = (
            ['measurement'], results['nadir_equiv_total_error_m'],
            {'units': 'meters', 'long_name': 'Total nadir-equivalent geolocation error'}
        )

        data_vars['nadir_equiv_vp_error_m'] = (
            ['measurement'], results['nadir_equiv_vp_error_m'],
            {'units': 'meters', 'long_name': 'View-plane nadir-equivalent error'}
        )

        data_vars['nadir_equiv_xvp_error_m'] = (
            ['measurement'], results['nadir_equiv_xvp_error_m'],
            {'units': 'meters', 'long_name': 'Cross-view-plane nadir-equivalent error'}
        )

        # Intermediate processing results
        data_vars['vp_error_m'] = (
            ['measurement'], results['vp_error_m'],
            {'units': 'meters', 'long_name': 'View-plane error distance'}
        )

        data_vars['xvp_error_m'] = (
            ['measurement'], results['xvp_error_m'],
            {'units': 'meters', 'long_name': 'Cross-view-plane error distance'}
        )

        data_vars['off_nadir_angle_deg'] = (
            ['measurement'], np.rad2deg(results['off_nadir_angle_rad']),
            {'units': 'degrees', 'long_name': 'Off-nadir viewing angle'}
        )

        data_vars['vp_scaling_factor'] = (
            ['measurement'], results['vp_scaling_factor'],
            {'units': 'dimensionless', 'long_name': 'View-plane nadir scaling factor'}
        )

        data_vars['xvp_scaling_factor'] = (
            ['measurement'], results['xvp_scaling_factor'],
            {'units': 'dimensionless', 'long_name': 'Cross-view-plane nadir scaling factor'}
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
                'title': 'Geolocation Error Statistics Results',
                'processing_timestamp': np.datetime64('now'),
                'earth_radius_m': self.config.earth_radius_m,
                'performance_threshold_m': self.config.performance_threshold_m
            }
        )

        # Add correlation filtering metadata if applied
        if self.config.minimum_correlation is not None:
            output_ds.attrs['minimum_correlation_threshold'] = self.config.minimum_correlation
            output_ds.attrs['correlation_filtering_applied'] = True

        return output_ds

    def _calculate_statistics(self, nadir_equiv_errors_m: np.ndarray) -> Dict[str, Union[float, int]]:
        """Calculate performance statistics on nadir-equivalent errors."""

        # Count errors below threshold
        num_below_threshold = np.sum(nadir_equiv_errors_m < self.config.performance_threshold_m)

        # Calculate statistics
        stats = {
            'mean_error_distance_m': float(np.mean(nadir_equiv_errors_m)),
            'std_error_distance_m': float(np.std(nadir_equiv_errors_m)),
            'min_error_distance_m': float(np.min(nadir_equiv_errors_m)),
            'max_error_distance_m': float(np.max(nadir_equiv_errors_m)),
            'percent_below_250m': float(num_below_threshold / len(nadir_equiv_errors_m) * 100),
            'num_below_250m': int(num_below_threshold),
            'total_measurements': int(len(nadir_equiv_errors_m)),
            'performance_spec_met': bool(
                num_below_threshold / len(nadir_equiv_errors_m) * 100 > self.config.performance_spec_percent
            )
        }

        return stats

    def process_from_netcdf(self,
                           filepath: Union[str, 'Path'],
                           minimum_correlation: Optional[float] = None) -> xr.Dataset:
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
        from pathlib import Path

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"NetCDF file not found: {filepath}")

        logger.info(f"Loading NetCDF results from: {filepath}")
        input_data = xr.open_dataset(filepath)

        # Override correlation threshold if provided
        original_threshold = self.config.minimum_correlation
        if minimum_correlation is not None:
            self.config.minimum_correlation = minimum_correlation
            logger.info(f"Overriding correlation threshold: "
                       f"{original_threshold} → {minimum_correlation}")

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
        results.attrs['reprocessed_from'] = str(filepath)
        results.attrs['reprocessing_date'] = str(np.datetime64('now'))
        if minimum_correlation is not None:
            results.attrs['correlation_threshold_override'] = minimum_correlation

        # Restore original threshold
        self.config.minimum_correlation = original_threshold

        return results

    @staticmethod
    def create_test_dataset() -> xr.Dataset:
        """Create test dataset with the original 13 hardcoded test cases."""

        # Test case data (from original script)
        test_data = {
            'lat_error_deg': [0.026980, -0.0269188, 0.009040, -0.008925, 0.022515,
                             0.001, -0.0015, -0.002, 0.002, 0.001, -0.001, 0.0011, 0.0025],
            'lon_error_deg': [-0.027266, 0.018384, 0.010851, -0.026241, 0.000992,
                             0.0005, 0.0015, 0.0, -0.0005, 0.001, -0.0015, 0.0011, 0.0],
            'cp_lat_deg': [-8.57802802047, 10.9913499301, 33.9986324792, 31.1629017783, -69.6971234613,
                          34.2, -19.7, -49.5, -8.2, 46.0, 32.5, -20.33, -47.6],
            'cp_lon_deg': [125.482222317, -71.8457829833, -120.248435967, -8.7815788192, -15.2311156066,
                          42.5, -51.1, -178.3, 70.0, -29.0, -170.4, 95.4, -30.87],
            'cp_alt': [44, 4, 0, 894, 3925, 0, 1000, 500, 500, 50, 100, 100, 100]
        }

        # RISS_CTRS positions (satellite positions)
        riss_ctrs_data = np.array([
            [-3888220.86746399, 5466997.0490439, -1000356.92985575],
            [2138128.91767507, -6313660.02871594, 1241996.71916521],
            [-2836930.06048711, -4869372.01247407, 3765186.91739563],
            [5764626.80186185, -843462.027662883, 3457275.08087601],
            [2210828.23546441, -6156903.77352567, -1818743.37976767],
            [4160733.71254889, 3708441.12891715, 3850046.48797648],
            [4060487.97522754, -4920200.36807653, -2308736.58835498],
            [-4274543.69565126, -116765.394831108, -5276242.10262264],
            [2520101.83352962, 6230331.37805726, -961492.530214298],
            [4248835.7920035, -2447631.1800248, 4676942.35070364],
            [-5515282.275281, -925908.369707886, 3822512.18293707],
            [-824875.850002718, 6312906.79811629, -2344264.22196647],
            [3675746.11507236, -2198122.65541618, -5270960.3157354]
        ])

        # Boresight vectors in HS coordinate system
        bhat_hs_data = np.array([
            [0, 0.0625969755450201, 0.99803888634292],
            [0, 0.0313138440396569, 0.999509601340307],
            [0, 0.000368458389306164, 0.999999932119205],
            [0, -0.0368375032472, 0.999321268839262],
            [0, -0.0699499255109834, 0.997550503945042],
            [0, -0.0368375032472, 0.999321268839262],
            [0, -0.0257892283106606, 0.999667402541035],
            [0, 0.0257892283106606, 0.999667402541035],
            [0, 0.0625969755450201, 0.99803888634292],
            [0, 0.0552406262884485, 0.998473070847311],
            [0, -0.0147378023382108, 0.999891392693346],
            [0, -0.0221057030926655, 0.999755639089262],
            [0, -0.0221057030926655, 0.999755639089262]
        ])

        # Transformation matrices from HS to CTRS (3x3x13)
        t_hs2ctrs_data = np.zeros((3, 3, 13))

        # Test case 1
        t_hs2ctrs_data[:, :, 0] = [
            [-0.418977524967338, 0.748005379751721, 0.514728846515064],
            [-0.421890284446342, 0.341604851993858, -0.839830169131854],
            [-0.804031356019172, -0.569029065124742, 0.172451447025628]
        ]

        # Test case 2
        t_hs2ctrs_data[:, :, 1] = [
            [0.509557370616697, 0.714990103896663, -0.478686157497828],
            [0.336198439435013, 0.346660121582392, 0.875669669125261],
            [0.792036549265032, -0.607137473258174, -0.0637353370903461]
        ]

        # Test case 3
        t_hs2ctrs_data[:, :, 2] = [
            [0.436608377090994, -0.795688667243495, 0.419824570355571],
            [-0.682818757213707, 0.0107593091164333, 0.730508577680278],
            [-0.585774418911493, -0.605610255930006, -0.53861354240429]
        ]

        # Test case 4
        t_hs2ctrs_data[:, :, 3] = [
            [-0.275228112982228, 0.368161232084539, -0.888091658002842],
            [0.740939532874243, 0.669849578957866, 0.0480640218257623],
            [0.612583132683508, -0.644793648200697, -0.457146646921637]
        ]

        # Test case 5
        t_hs2ctrs_data[:, :, 4] = [
            [0.497596843733441, -0.8343127195548, -0.237317650198193],
            [0.404893735025568, -0.0185495841473054, 0.914175571903453],
            [-0.767110451267327, -0.550979308973609, 0.328577778675617]
        ]

        # Test case 6
        t_hs2ctrs_data[:, :, 5] = [
            [-0.765506977045252, 0.0328563789337692, -0.642588135250651],
            [0.295444324153605, 0.905137001368494, -0.305678298647175],
            [0.571587018239969, -0.423847628778339, -0.702596052277786]
        ]

        # Test case 7
        t_hs2ctrs_data[:, :, 6] = [
            [0.629603159973548, 0.368109063956699, -0.684174861189846],
            [0.215915166022854, 0.763032030046674, 0.609230735287836],
            [0.746311263503805, -0.531296841841519, 0.400927218783918]
        ]

        # Test case 8
        t_hs2ctrs_data[:, :, 7] = [
            [0.194530273749036, 0.949748975936332, 0.245223854916003],
            [-0.978512013106359, 0.205316430746179, -0.0189577388931897],
            [-0.0683535866042618, -0.236266544212139, 0.969281089206121]
        ]

        # Test case 9
        t_hs2ctrs_data[:, :, 8] = [
            [-0.446421529583839, 0.413968410219497, -0.793307812225063],
            [0.384674732015686, -0.711668847399292, -0.587837230750712],
            [-0.807919035840032, -0.56758835193185, 0.158460875505101]
        ]

        # Test case 10
        t_hs2ctrs_data[:, :, 9] = [
            [0.632159685228781, -0.204512480192889, -0.747361135669863],
            [0.598189792213041, -0.48424547358001, 0.638493680968301],
            [-0.492486654503011, -0.850693598485042, -0.183784021560657]
        ]

        # Test case 11
        t_hs2ctrs_data[:, :, 10] = [
            [0.753428906287479, -0.49153961754033, 0.436730605865421],
            [-0.565149851875981, -0.823589060852856, 0.0480236900131712],
            [0.336081133202996, -0.283000352923251, -0.898309519920648]
        ]

        # Test case 12
        t_hs2ctrs_data[:, :, 11] = [
            [-0.585265557251293, -0.595045400433036, 0.550803349451662],
            [-0.109341614649782, -0.615175192945938, -0.780771245706364],
            [0.803435522491452, -0.517183787785386, 0.294977548217097]
        ]

        # Test case 13
        t_hs2ctrs_data[:, :, 12] = [
            [0.292122841971449, -0.95622050459562, 0.017506859615382],
            [0.95633436246494, 0.291879296125504, -0.0152004494429911],
            [0.00942509242927969, 0.0211828985704245, 0.999731155222533]
        ]

        # Create coordinates
        measurements = np.arange(13)

        # Create dataset
        dataset = xr.Dataset(
            {
                'lat_error_deg': (['measurement'], test_data['lat_error_deg']),
                'lon_error_deg': (['measurement'], test_data['lon_error_deg']),
                'riss_ctrs': (['measurement', 'xyz'], riss_ctrs_data),
                'bhat_hs': (['measurement', 'xyz'], bhat_hs_data),
                't_hs2ctrs': (['xyz_from', 'xyz_to', 'measurement'], t_hs2ctrs_data),
                'cp_lat_deg': (['measurement'], test_data['cp_lat_deg']),
                'cp_lon_deg': (['measurement'], test_data['cp_lon_deg']),
                'cp_alt': (['measurement'], test_data['cp_alt'])
            },
            coords={
                'measurement': measurements,
                'xyz': ['x', 'y', 'z'],
                'xyz_from': ['x', 'y', 'z'],
                'xyz_to': ['x', 'y', 'z']
            },
            attrs={
                'title': 'Test Dataset for Geolocation Error Statistics',
                'description': 'Original 13 hardcoded test cases from MATLAB implementation',
                'n_measurements': 13
            }
        )

        return dataset


# Convenience functions for backward compatibility and easy testing
def process_test_data(display_results: bool = True) -> xr.Dataset:
    """Process the original 13 test cases using the new production processor."""
    processor = ErrorStatsProcessor()
    test_data = ErrorStatsProcessor.create_test_dataset()

    results = processor.process_geolocation_errors(test_data)

    if display_results:
        print_results_summary(results)

    return results


def print_results_summary(results: xr.Dataset) -> None:
    """Print a summary of processing results."""
    print(f"Processing Results Summary:")
    print(f"=" * 50)
    print(f"Total measurements: {results.attrs['total_measurements']}")
    print(f"Mean error distance: {results.attrs['mean_error_distance_m']:.2f} m")
    print(f"Std error distance: {results.attrs['std_error_distance_m']:.2f} m")
    print(f"Min/Max error: {results.attrs['min_error_distance_m']:.2f} / {results.attrs['max_error_distance_m']:.2f} m")
    print(f"Errors < 250m: {results.attrs['num_below_250m']} ({results.attrs['percent_below_250m']:.1f}%)")

    spec_status = "✓ PASS" if results.attrs['performance_spec_met'] else "✗ FAIL"
    print(f"Performance spec (>39% < 250m): {spec_status}")


def main():
    """Main function demonstrating the production processor with test data."""
    print("Production Geolocation Statistics Processor")
    print("=" * 50)

    # Process test data
    results = process_test_data(display_results=True)

    # Show some detailed results
    print(f"\nDetailed Results (first 15 measurements):")
    print(f"{'Measurement':<12} {'Total Error (m)':<15} {'Off-Nadir (°)':<15}")
    print(f"-" * 42)

    for i in range(min(15, len(results.measurement))):
        total_err = results.nadir_equiv_total_error_m.values[i]
        off_nadir = results.off_nadir_angle_deg.values[i]
        print(f"{i+1:<12} {total_err:<15.3f} {off_nadir:<15.3f}")

    return results


if __name__ == "__main__":
    main()
