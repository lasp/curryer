"""NetCDF output configuration and standard attribute definitions.

Separates I/O-specific configuration from the core correction config models.
"""

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from curryer.correction.config import ParameterConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Standard NetCDF Variable Attributes (Mission-Agnostic)
# ============================================================================

DEFAULT_NETCDF_ATTRIBUTES = {
    # Geolocation error metrics (per GCP pair)
    "rms_error_m": {"units": "meters", "long_name": "RMS geolocation error"},
    "mean_error_m": {"units": "meters", "long_name": "Mean geolocation error"},
    "max_error_m": {"units": "meters", "long_name": "Maximum geolocation error"},
    "std_error_m": {"units": "meters", "long_name": "Standard deviation of geolocation error"},
    "n_measurements": {"units": "count", "long_name": "Number of measurement points"},
    # Aggregate performance metrics (per parameter set)
    "mean_rms_all_pairs": {"units": "meters", "long_name": "Mean RMS error across all GCP pairs"},
    "worst_pair_rms": {"units": "meters", "long_name": "Worst performing GCP pair RMS error"},
    "best_pair_rms": {"units": "meters", "long_name": "Best performing GCP pair RMS error"},
    # Image matching metrics (per GCP pair)
    "im_lat_error_km": {"units": "kilometers", "long_name": "Image matching latitude error"},
    "im_lon_error_km": {"units": "kilometers", "long_name": "Image matching longitude error"},
    "im_ccv": {"units": "dimensionless", "long_name": "Image matching correlation coefficient"},
    "im_grid_step_m": {"units": "meters", "long_name": "Image matching final grid step size"},
}


# ============================================================================
# Standard Data Variable Names (Mission-Agnostic Keys)
# ============================================================================

# Standard variable names that should be present in image matching results.
# Used for extracting data from xarray.Dataset objects.
STANDARD_VAR_NAMES = {
    # Error measurements (required)
    "lat_error_deg": "lat_error_deg",
    "lon_error_deg": "lon_error_deg",
    # Spacecraft state (configurable names)
    "spacecraft_position": "sc_position",  # Generic default
    "boresight": "boresight",  # Generic default
    "transformation_matrix": "t_inst2ref",  # Generic default
    # Control point location (optional)
    "gcp_lat_deg": "gcp_lat_deg",
    "gcp_lon_deg": "gcp_lon_deg",
    "gcp_alt": "gcp_alt",
}


class NetCDFParameterMetadata(BaseModel):
    """NetCDF metadata for a single output parameter variable."""

    variable_name: str
    units: str
    long_name: str


class NetCDFConfig(BaseModel):
    """Configuration for NetCDF output structure and metadata.

    Attributes
    ----------
    performance_threshold_m
        Accuracy threshold in metres used to derive threshold-specific
        variable names (e.g. ``"percent_under_250m"``).
    title
        Global title attribute for the output NetCDF file.
    description
        Global description attribute for the output NetCDF file.
    parameter_metadata
        Optional mapping of parameter key → :class:`NetCDFParameterMetadata`.
        Auto-generated from ``CorrectionConfig.parameters`` when ``None``.
    standard_attributes
        Optional mission-specific attribute overrides.  Falls back to the
        module-level :data:`DEFAULT_NETCDF_ATTRIBUTES` when ``None``.
    """

    performance_threshold_m: float
    title: str = "Correction Geolocation Analysis Results"
    description: str = "Parameter sensitivity analysis"
    parameter_metadata: dict[str, NetCDFParameterMetadata] | None = None
    standard_attributes: dict[str, dict[str, str]] | None = None

    @property
    def threshold_metric_name(self) -> str:
        """Threshold-derived metric variable name, e.g. ``"percent_under_250m"``."""
        threshold_m = int(self.performance_threshold_m)
        return f"percent_under_{threshold_m}m"

    @property
    def standard_attributes_dict(self) -> dict[str, dict[str, str]]:
        """Standard variable attributes, with optional mission-specific overrides."""
        if self.standard_attributes is not None:
            return self.standard_attributes
        return DEFAULT_NETCDF_ATTRIBUTES.copy()

    def get_parameter_netcdf_metadata(
        self, param_config: "ParameterConfig", angle_type: str | None = None
    ) -> "NetCDFParameterMetadata":
        """Get NetCDF metadata for a parameter."""
        if param_config.config_file:
            param_stem = param_config.config_file.stem
            lookup_key = f"{param_stem}_{angle_type}" if angle_type else param_stem
        else:
            identity = (
                param_config.spec.metadata.get("name") or param_config.spec.field or param_config.ptype.name.lower()
            )
            lookup_key = identity

        if self.parameter_metadata and lookup_key in self.parameter_metadata:
            return self.parameter_metadata[lookup_key]

        return self._auto_generate_metadata(param_config, angle_type, lookup_key)

    def _auto_generate_metadata(
        self, param_config: "ParameterConfig", angle_type: str | None, base_key: str
    ) -> "NetCDFParameterMetadata":
        """Auto-generate NetCDF metadata from parameter configuration."""
        from curryer.correction.config import ParameterType

        if param_config.ptype == ParameterType.CONSTANT_KERNEL:
            units = "arcseconds"
        elif param_config.ptype == ParameterType.OFFSET_KERNEL:
            units = "arcseconds"
        elif param_config.ptype == ParameterType.OFFSET_TIME:
            units = "milliseconds"
        else:
            units = "unknown"

        # Use declared units field (replaces old isinstance(data, dict) check)
        if param_config.spec.units is not None:
            units = param_config.spec.units

        var_name = base_key.replace(".", "_").replace("-", "_")
        if not var_name.startswith("param_"):
            var_name = f"param_{var_name}"

        if param_config.config_file:
            file_stem = param_config.config_file.stem
            # Cosmetic only: strip kernel version suffixes (``_v01``/``_v02``) and the
            # ``.attitude.ck`` format tag so the NetCDF ``long_name`` reads cleanly
            # (e.g. ``cprs_hysics_v01.attitude.ck`` -> "Cprs Hysics correction").
            # These tokens are conventional in mission kernel filenames; unmatched
            # names simply pass through unchanged, so this is safe for any mission.
            clean_name = file_stem.replace("_v01", "").replace("_v02", "").replace(".attitude.ck", "")
            clean_name = clean_name.replace("_", " ").title()
            if angle_type:
                long_name = f"{clean_name} {angle_type} correction"
            else:
                long_name = f"{clean_name} correction"
        else:
            long_name = f"{param_config.ptype.name.replace('_', ' ').title()} parameter"

        return NetCDFParameterMetadata(variable_name=var_name, units=units, long_name=long_name)
