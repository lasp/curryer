"""
Geolocation statistics processor with Xarray inputs and outputs.

This module processes geolocation errors from the image matching algorithm and
produces nadir-equivalent geolocation errors together with mission-agnostic
summary statistics.

The main processing pipeline:

1. Convert angular errors to N-S and E-W distances.
2. Transform error components to view-plane / cross-view-plane distances.
3. Scale to nadir-equivalent using geometric factors.
4. (Optional) Compute comprehensive statistics across all measurements.

Pass/fail evaluation is intentionally **not** included here — whether the
statistics meet mission requirements is the caller's responsibility.  Use
:func:`compute_percent_below` for custom threshold queries, or compare the
fixed threshold-table entries (``percent_below_100m``, ``percent_below_250m``,
etc.) directly.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import xarray as xr

from curryer.compute import constants

logger = logging.getLogger(__name__)

# WGS84 Earth radius in meters – single source of truth from curryer.compute.constants.
_EARTH_RADIUS_M: float = constants.WGS84_SEMI_MAJOR_AXIS_KM * 1000.0


class ViewPlaneVectors(NamedTuple):
    """Unit vectors spanning the view plane in UEN coordinates."""

    v_uen: np.ndarray
    x_uen: np.ndarray


class ScalingFactors(NamedTuple):
    """Scaling factors for nadir-equivalent error projection."""

    vp_factor: float
    xvp_factor: float


def compute_percent_below(errors: np.ndarray, threshold_m: float) -> float:
    """Compute the percentage of errors below a given threshold.

    Useful for evaluating custom thresholds not in the standard table
    produced by :meth:`ErrorStatsProcessor._calculate_statistics`.

    Parameters
    ----------
    errors : np.ndarray
        Array of nadir-equivalent geolocation errors in meters.
    threshold_m : float
        Threshold in meters.

    Returns
    -------
    float
        Percentage (0–100) of errors strictly below *threshold_m*.
        Returns ``0.0`` when *errors* is empty.
    """
    if len(errors) == 0:
        return 0.0
    return float(np.sum(errors < threshold_m) / len(errors) * 100)


@dataclass
class ErrorStatsConfig:
    """Configuration for geolocation error statistics processing.

    Parameters
    ----------
    minimum_correlation : float or None, optional
        Minimum correlation (0.0–1.0).  Measurements whose correlation score
        falls below it are rejected: kept in the output with ``accepted``
        False and a ``rejection_reason``, and left out of the statistics.
        When set, the input must carry a ``correlation`` (or ``ccv`` /
        ``im_ccv``) variable or processing raises.  Default is ``None`` (no
        gate).
    minimum_peak_margin : float or None, optional
        Minimum amount by which a measurement's correlation must exceed
        ``correlation_secondary`` (the strongest competing correlation away
        from the peak).  Measurements below it are rejected as for
        ``minimum_correlation``; the input must then carry
        ``correlation_secondary`` and a correlation variable or processing
        raises.  Default is ``None``.
    variable_names : dict of str to str or None, optional
        Mission-agnostic variable name mappings from semantic names to actual
        dataset variable names.  If ``None``, generic defaults are used.

    Notes
    -----
    Pass/fail thresholds are **not** part of this config.
    ``ErrorStatsProcessor`` computes statistics only; whether those numbers
    meet mission requirements is the caller's responsibility.

    Earth radius is not a config field either.  ``_EARTH_RADIUS_M`` (derived
    from ``curryer.compute.constants.WGS84_SEMI_MAJOR_AXIS_KM``) is used
    directly in all calculations.
    """

    minimum_correlation: float | None = None
    minimum_peak_margin: float | None = None

    # Mission-agnostic variable name mappings
    # Maps semantic names to actual variable names in the dataset
    variable_names: dict[str, str] | None = None  # If None, uses generic defaults

    @classmethod
    def from_setup(cls, setup) -> "ErrorStatsConfig":
        """Create an :class:`ErrorStatsConfig` from a :class:`GeolocationSetup`.

        Extracts the science-Dataset variable names, ``minimum_correlation`` and
        ``minimum_peak_margin`` from the setup, the single source of truth for those settings.

        Parameters
        ----------
        setup : GeolocationSetup
            The geolocation setup (variable names + geo settings).

        Returns
        -------
        ErrorStatsConfig
        """
        variable_names = {
            "spacecraft_position": setup.spacecraft_position_name,
            "boresight": setup.boresight_name,
            "transformation_matrix": setup.transformation_matrix_name,
        }

        return cls(
            minimum_correlation=setup.geo.minimum_correlation,
            minimum_peak_margin=setup.geo.minimum_peak_margin,
            variable_names=variable_names,
        )

    def get_variable_name(self, semantic_name: str) -> str:
        """
        Get actual variable name for a semantic concept.

        Parameters
        ----------
        semantic_name : str
            Semantic name like 'spacecraft_position', 'boresight', etc.

        Returns
        -------
        str
            Actual variable name in the dataset.

        Raises
        ------
        ValueError
            If variable_names is None or semantic_name is not found.
        """
        if self.variable_names is None:
            raise ValueError(
                f"ErrorStatsConfig.variable_names is None. "
                f"Use ErrorStatsConfig.from_setup() to create config with proper variable names."
            )

        if semantic_name not in self.variable_names:
            raise ValueError(
                f"Variable name mapping for '{semantic_name}' not found in config. "
                f"Available mappings: {list(self.variable_names.keys())}"
            )

        return self.variable_names[semantic_name]


class ErrorStatsProcessor:
    """Production-ready processor for geolocation error statistics."""

    def __init__(self, config: ErrorStatsConfig):
        """
        Initialize processor with configuration.

        Parameters
        ----------
        config : ErrorStatsConfig
            Configuration for error statistics processing. Use
            ``ErrorStatsConfig.from_setup()`` to create from a
            GeolocationSetup.
        """
        if config is None:
            raise ValueError("ErrorStatsConfig is required. Use ErrorStatsConfig.from_setup(setup) to create.")
        self.config = config

    def rejection_reasons(self, data: xr.Dataset) -> np.ndarray:
        """Return why each measurement fails the match-quality gates.

        The gates are ``minimum_correlation`` (correlation at the match) and
        ``minimum_peak_margin`` (correlation minus ``correlation_secondary``).
        A rejected measurement is a match that worked but is not trusted, e.g.
        a cloud-covered or featureless scene; it stays in the output, flagged,
        and is left out of the statistics.

        Parameters
        ----------
        data : xr.Dataset
            Measurements with a ``measurement`` dimension; must carry a
            ``correlation``, ``ccv`` or ``im_ccv`` variable when either gate is
            set, and ``correlation_secondary`` when ``minimum_peak_margin`` is set.

        Returns
        -------
        np.ndarray of str, shape (n_measurements,)
            ``""`` for an accepted measurement; otherwise each failed gate with
            its value and threshold, joined by ``"; "`` (e.g.
            ``"correlation 0.4120 < 0.8"``).  A NaN correlation fails both gates.

        Raises
        ------
        ValueError
            If a gate is set and *data* lacks the variable it needs.
        """
        n = data.sizes["measurement"]
        if self.config.minimum_correlation is None and self.config.minimum_peak_margin is None:
            return np.full(n, "", dtype=object)

        corr_var = next((name for name in ("correlation", "ccv", "im_ccv") if name in data.data_vars), None)
        if corr_var is None:
            raise ValueError(
                f"minimum_correlation={self.config.minimum_correlation} / minimum_peak_margin="
                f"{self.config.minimum_peak_margin} is set but the input has no correlation variable "
                "('correlation', 'ccv' or 'im_ccv'); the threshold cannot be applied."
            )
        corr = data[corr_var].values
        failed: list[list[str]] = [[] for _ in range(n)]
        if self.config.minimum_correlation is not None:
            for i in np.flatnonzero(~(corr >= self.config.minimum_correlation)):
                failed[i].append(f"correlation {corr[i]:.4f} < {self.config.minimum_correlation}")
        if self.config.minimum_peak_margin is not None:
            if "correlation_secondary" not in data.data_vars:
                raise ValueError(
                    f"minimum_peak_margin={self.config.minimum_peak_margin} is set but the input has no "
                    "'correlation_secondary' variable; the threshold cannot be applied."
                )
            margin = corr - data["correlation_secondary"].values
            for i in np.flatnonzero(~(margin >= self.config.minimum_peak_margin)):
                failed[i].append(f"peak margin {margin[i]:.4f} < {self.config.minimum_peak_margin}")
        return np.array(["; ".join(reasons) for reasons in failed], dtype=object)

    def compute_nadir_equivalent_errors(self, input_data: xr.Dataset) -> xr.Dataset:
        """Compute per-measurement nadir-equivalent errors WITHOUT aggregate statistics.

        This is the method to call inside the correction loop — it requires
        observation geometry (spacecraft position, boresight, transformation
        matrix) that is only available during each iteration, and produces
        nadir-equivalent errors for each measurement.  No aggregate statistics
        are computed (meaningless for a single GCP pair in isolation).

        Use this inside the loop for checkpoint/resume support.
        Call :meth:`process_geolocation_errors` for the final aggregate pass
        (nadir-equivalent + comprehensive statistics).

        Every input measurement is in the output.  Those failing the
        match-quality gates (:meth:`rejection_reasons`) have ``accepted`` False
        and keep their computed errors for review.

        Parameters
        ----------
        input_data : xr.Dataset
            Dataset with required error measurement variables and a
            ``measurement`` dimension.  When it carries ``track_azimuth_deg``
            (ground azimuth of the along-track direction at each GCP, degrees
            clockwise from north), the error is also resolved into
            along-track and cross-track components.

        Returns
        -------
        xr.Dataset
            Dataset with ``nadir_equiv_total_error_m`` and related intermediate
            variables, ``accepted`` (bool), ``rejection_reason`` (str, ``""``
            when accepted), and, when the input carries ``track_azimuth_deg``,
            ``along_track_error_m`` (positive along the azimuth) and
            ``cross_track_error_m`` (positive 90° clockwise from it), in meters,
            from the same north/east error as the view-plane components.
            Attributes ``n_matched``, ``n_accepted`` and ``n_rejected``; no
            statistics.

        Raises
        ------
        ValueError
            If required variables are missing, or a match-quality gate is set
            but the input lacks the variable it needs.
        """
        self._validate_input_data(input_data)
        reasons = self.rejection_reasons(input_data)

        n_measurements = len(input_data.measurement)

        sc_pos_var = self.config.get_variable_name("spacecraft_position")
        boresight_var = self.config.get_variable_name("boresight")
        transform_var = self.config.get_variable_name("transformation_matrix")

        lat_error_rad = np.deg2rad(input_data.lat_error_deg.values)
        lon_error_rad = np.deg2rad(input_data.lon_error_deg.values)
        gcp_lat_rad = np.deg2rad(input_data.gcp_lat_deg.values)
        gcp_lon_rad = np.deg2rad(input_data.gcp_lon_deg.values)

        ns_error_dist_m = _EARTH_RADIUS_M * lat_error_rad
        ew_error_dist_m = _EARTH_RADIUS_M * np.cos(gcp_lat_rad) * lon_error_rad

        bhat_ctrs = self._transform_boresight_vectors(
            input_data[boresight_var].values, input_data[transform_var].values
        )

        results = self._process_to_nadir_equivalent(
            ns_error_dist_m,
            ew_error_dist_m,
            input_data[sc_pos_var].values,
            bhat_ctrs,
            gcp_lat_rad,
            gcp_lon_rad,
            n_measurements,
        )
        if "track_azimuth_deg" in input_data.data_vars:
            azimuth_rad = np.deg2rad(input_data["track_azimuth_deg"].values)
            results["along_track_error_m"] = ns_error_dist_m * np.cos(azimuth_rad) + ew_error_dist_m * np.sin(
                azimuth_rad
            )
            results["cross_track_error_m"] = ew_error_dist_m * np.cos(azimuth_rad) - ns_error_dist_m * np.sin(
                azimuth_rad
            )

        output = self._create_output_dataset(input_data, results, reasons)
        logger.info(
            "Match-quality gates: %d of %d measurements accepted (minimum_correlation=%s, minimum_peak_margin=%s)",
            output.attrs["n_accepted"],
            n_measurements,
            self.config.minimum_correlation,
            self.config.minimum_peak_margin,
        )
        return output

    def add_statistics(self, per_measurement: xr.Dataset) -> xr.Dataset:
        """Add aggregate statistics over the accepted measurements as attributes.

        Parameters
        ----------
        per_measurement : xr.Dataset
            Output of :meth:`compute_nadir_equivalent_errors`.

        Returns
        -------
        xr.Dataset
            *per_measurement* with the statistics of
            :meth:`_calculate_statistics` added to its attributes;
            ``total_measurements`` counts accepted measurements only.

        Raises
        ------
        ValueError
            If no measurement is accepted.
        """
        accepted = per_measurement["accepted"].values
        if not accepted.any():
            raise ValueError(
                f"No measurements remaining after correlation filtering: all {accepted.size} were rejected."
            )
        output = per_measurement.copy()
        output.attrs.update(self._calculate_statistics(output["nadir_equiv_total_error_m"].values[accepted]))
        return output

    def process_geolocation_errors(self, input_data: xr.Dataset) -> xr.Dataset:
        """Full processing: nadir-equivalent errors + aggregate statistics.

        Use this for final aggregation after the loop, or in :func:`verify`.
        For per-iteration computation (single GCP pair), prefer
        :meth:`compute_nadir_equivalent_errors` to avoid computing aggregate
        statistics on a small or single-measurement sample.

        Parameters
        ----------
        input_data : xr.Dataset
            Dataset with required error measurement variables.

        Returns
        -------
        xr.Dataset
            Output of :meth:`compute_nadir_equivalent_errors` for every
            measurement, with statistics over the accepted measurements as
            global attributes (:meth:`add_statistics`).

        Raises
        ------
        ValueError
            As :meth:`compute_nadir_equivalent_errors`, or if no measurement is
            accepted.
        """
        return self.add_statistics(self.compute_nadir_equivalent_errors(input_data))

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

    def _calculate_view_plane_vectors(self, bhat_uen: np.ndarray) -> ViewPlaneVectors:
        """Calculate view-plane and cross-view-plane unit vectors in UEN coordinates."""
        # Calculate normalization factor for horizontal components
        norm_factor = np.sqrt(bhat_uen[1] ** 2 + bhat_uen[2] ** 2)

        # View-plane direction (in the direction of boresight horizontal projection)
        v_uen = np.array([0, bhat_uen[1], bhat_uen[2]]) / norm_factor

        # Cross-view-plane direction (perpendicular to view-plane in horizontal)
        x_uen = np.array([0, bhat_uen[2], -bhat_uen[1]]) / norm_factor

        return ViewPlaneVectors(v_uen=v_uen, x_uen=x_uen)

    def _calculate_scaling_factors(self, riss_ctrs: np.ndarray, theta: float) -> ScalingFactors:
        """Calculate scaling factors for nadir-equivalent transformation."""
        r_magnitude = np.linalg.norm(riss_ctrs)
        f = r_magnitude / _EARTH_RADIUS_M
        h = r_magnitude - _EARTH_RADIUS_M

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
        vp_factor = h / _EARTH_RADIUS_M / (-1 + f * np.cos(theta) / temp1)

        # Cross-view-plane scaling factor
        xvp_factor = h / _EARTH_RADIUS_M / np.cos(theta) / (f * np.cos(theta) - temp1)

        return ScalingFactors(vp_factor=vp_factor, xvp_factor=xvp_factor)

    def _create_output_dataset(
        self, input_data: xr.Dataset, results: dict[str, np.ndarray], reasons: np.ndarray
    ) -> xr.Dataset:
        """Create output Xarray Dataset with processing results."""

        # Create data variables for output
        data_vars = {}

        accepted = reasons == ""
        data_vars["accepted"] = (
            ["measurement"],
            accepted.astype(bool),
            {"long_name": "Measurement passed the match-quality gates and enters the statistics"},
        )
        data_vars["rejection_reason"] = (
            ["measurement"],
            reasons.astype(str),
            {"long_name": "Match-quality gates failed; empty when accepted"},
        )
        if "along_track_error_m" in results:
            data_vars["along_track_error_m"] = (
                ["measurement"],
                results["along_track_error_m"],
                {"units": "meters", "long_name": "Error component along the ground-track azimuth"},
            )
            data_vars["cross_track_error_m"] = (
                ["measurement"],
                results["cross_track_error_m"],
                {"units": "meters", "long_name": "Error component 90 degrees clockwise from the ground-track azimuth"},
            )

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
                "processing_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "earth_radius_m": _EARTH_RADIUS_M,
                "n_matched": int(accepted.size),
                "n_accepted": int(accepted.sum()),
                "n_rejected": int((~accepted).sum()),
            },
        )

        # Add correlation filtering metadata if applied
        if self.config.minimum_correlation is not None:
            output_ds.attrs["minimum_correlation_threshold"] = self.config.minimum_correlation
            output_ds.attrs["correlation_filtering_applied"] = 1  # NetCDF attributes cannot hold a bool
        if self.config.minimum_peak_margin is not None:
            output_ds.attrs["minimum_peak_margin_threshold"] = self.config.minimum_peak_margin

        return output_ds

    def _calculate_statistics(self, nadir_equiv_errors_m: np.ndarray) -> dict[str, float | int]:
        """Calculate comprehensive, mission-agnostic performance statistics.

        This method intentionally does NOT include any pass/fail evaluation.
        Whether these statistics meet mission requirements is the caller's
        responsibility.  Use :func:`compute_percent_below` for custom threshold
        queries not covered by the standard table.

        Parameters
        ----------
        nadir_equiv_errors_m : np.ndarray
            Array of nadir-equivalent geolocation errors in meters.

        Returns
        -------
        dict[str, float | int]
            Keys: central tendency (``mean_error_m``, ``median_error_m``,
            ``rms_error_m``), spread (``std_error_m``, ``min_error_m``,
            ``max_error_m``), percentiles (``p25_error_m`` … ``p99_error_m``),
            count (``total_measurements``), and a threshold table at standard
            intervals (``percent_below_100m`` … ``percent_below_1000m``).
        """
        n = len(nadir_equiv_errors_m)
        return {
            # Central tendency
            "mean_error_m": float(np.mean(nadir_equiv_errors_m)),
            "median_error_m": float(np.median(nadir_equiv_errors_m)),
            "rms_error_m": float(np.sqrt(np.mean(nadir_equiv_errors_m**2))),
            # Spread
            "std_error_m": float(np.std(nadir_equiv_errors_m)),
            "min_error_m": float(np.min(nadir_equiv_errors_m)),
            "max_error_m": float(np.max(nadir_equiv_errors_m)),
            # Percentiles
            "p25_error_m": float(np.percentile(nadir_equiv_errors_m, 25)),
            "p75_error_m": float(np.percentile(nadir_equiv_errors_m, 75)),
            "p90_error_m": float(np.percentile(nadir_equiv_errors_m, 90)),
            "p95_error_m": float(np.percentile(nadir_equiv_errors_m, 95)),
            "p99_error_m": float(np.percentile(nadir_equiv_errors_m, 99)),
            # Count
            "total_measurements": int(n),
            # Threshold table (standard intervals, for quick reference)
            "percent_below_100m": float(np.sum(nadir_equiv_errors_m < 100.0) / n * 100),
            "percent_below_250m": float(np.sum(nadir_equiv_errors_m < 250.0) / n * 100),
            "percent_below_500m": float(np.sum(nadir_equiv_errors_m < 500.0) / n * 100),
            "percent_below_750m": float(np.sum(nadir_equiv_errors_m < 750.0) / n * 100),
            "percent_below_1000m": float(np.sum(nadir_equiv_errors_m < 1000.0) / n * 100),
        }

    def process_from_netcdf(self, filepath: Union[str, "Path"], minimum_correlation: float | None = None) -> xr.Dataset:
        """
        Load previous results from NetCDF and reprocess error statistics.

        This enables iterative post-processing of Correction results without
        re-running expensive image matching operations.

        Args:
            filepath: Path to NetCDF file from previous Correction run
            minimum_correlation: Override correlation threshold (if provided)

        Returns:
            Xarray Dataset with reprocessed error statistics

        Example:
            >>> processor = ErrorStatsProcessor()
            >>> # Try different correlation thresholds
            >>> results_50 = processor.process_from_netcdf(
            ...     "correction_results/run_001.nc",
            ...     minimum_correlation=0.5
            ... )
            >>> results_70 = processor.process_from_netcdf(
            ...     "correction_results/run_001.nc",
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
