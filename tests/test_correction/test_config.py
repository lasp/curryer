"""Tests for the Pydantic-based correction configuration models.

Covers:
- Construction and field validation for all BaseModel subclasses
- ``DataConfig`` config-driven data loading configuration
- ``ParameterData`` backward-compatible dict-style access
- JSON round-trip: ``config == CorrectionConfig.model_validate_json(config.model_dump_json())``
- ``ValidationError`` raised with field-level messages for invalid inputs
- Callable / loader fields excluded from JSON serialisation
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from curryer.correction.config import (
    CorrectionConfig,
    DataConfig,
    GeolocationConfig,
    NetCDFConfig,
    NetCDFParameterMetadata,
    ParameterConfig,
    ParameterData,
    ParameterType,
    load_config_from_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def geo() -> GeolocationConfig:
    return GeolocationConfig(
        meta_kernel_file=Path("tests/data/test.kernels.tm.json"),
        generic_kernel_dir=Path("data/generic"),
        dynamic_kernels=[Path("tests/data/sc.spk.json"), Path("tests/data/sc.ck.json")],
        instrument_name="TEST_INSTRUMENT",
        time_field="corrected_timestamp",
    )


@pytest.fixture
def param_constant(geo) -> ParameterConfig:
    return ParameterConfig(
        ptype=ParameterType.CONSTANT_KERNEL,
        config_file=Path("tests/data/test_base.attitude.ck.json"),
        data={
            "current_value": [0.0, 0.0, 0.0],
            "bounds": [-300.0, 300.0],
            "sigma": 30.0,
            "units": "arcseconds",
            "distribution": "normal",
            "transformation_type": "dcm_rotation",
            "coordinate_frames": ["FRAME_A", "FRAME_B"],
        },
    )


@pytest.fixture
def param_offset_kernel() -> ParameterConfig:
    return ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("tests/data/test_az.attitude.ck.json"),
        data={
            "field": "hps.az_ang_nonlin",
            "current_value": 0.0,
            "bounds": [-300.0, 300.0],
            "sigma": 30.0,
            "units": "arcseconds",
        },
    )


@pytest.fixture
def param_offset_time() -> ParameterConfig:
    return ParameterConfig(
        ptype=ParameterType.OFFSET_TIME,
        config_file=None,
        data={
            "field": "corrected_timestamp",
            "current_value": 0.0,
            "bounds": [-50.0, 50.0],
            "sigma": 7.0,
            "units": "milliseconds",
        },
    )


@pytest.fixture
def netcdf_cfg() -> NetCDFConfig:
    return NetCDFConfig(
        performance_threshold_m=250.0,
        title="Test Geolocation Analysis",
        description="Unit-test run",
    )


@pytest.fixture
def minimal_config(geo, param_constant) -> CorrectionConfig:
    """Minimal CorrectionConfig with no loaders (fully serialisable)."""
    return CorrectionConfig(
        seed=42,
        n_iterations=5,
        parameters=[param_constant],
        geo=geo,
        performance_threshold_m=250.0,
        performance_spec_percent=39.0,
    )


@pytest.fixture
def full_config(geo, param_constant, param_offset_kernel, param_offset_time, netcdf_cfg) -> CorrectionConfig:
    """Full CorrectionConfig with all optional fields populated."""
    return CorrectionConfig(
        seed=0,
        n_iterations=10,
        parameters=[param_constant, param_offset_kernel, param_offset_time],
        geo=geo,
        performance_threshold_m=250.0,
        performance_spec_percent=39.0,
        netcdf=netcdf_cfg,
        output_filename="test_results.nc",
        calibration_dir=Path("tests/data/calibration"),
        calibration_file_names={"los_vectors": "b_HS.mat", "optical_psf": "psf.mat"},
        spacecraft_position_name="riss_ctrs",
        boresight_name="bhat_hs",
        transformation_matrix_name="t_hs2ctrs",
    )


# ===========================================================================
# DataConfig – construction and typed fields
# ===========================================================================


class TestDataConfig:
    def test_defaults(self):
        dc = DataConfig()
        assert dc.file_format == "csv"
        assert dc.time_scale_factor == 1.0

    def test_custom_values(self):
        dc = DataConfig(file_format="netcdf", time_scale_factor=1e6)
        assert dc.file_format == "netcdf"
        assert dc.time_scale_factor == 1e6

    def test_invalid_file_format(self):
        with pytest.raises(ValidationError):
            DataConfig(file_format="xml")

    def test_json_round_trip(self):
        dc = DataConfig(file_format="hdf5", time_scale_factor=1.0)
        restored = DataConfig.model_validate_json(dc.model_dump_json())
        assert restored.file_format == "hdf5"
        assert restored.time_scale_factor == 1.0

    def test_embedded_in_correction_config(self, geo, param_constant):
        """DataConfig round-trips through CorrectionConfig serialisation."""
        cfg = CorrectionConfig(
            n_iterations=1,
            parameters=[param_constant],
            geo=geo,
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
            data=DataConfig(file_format="csv", time_scale_factor=1e6),
        )
        json_str = cfg.model_dump_json()
        restored = CorrectionConfig.model_validate_json(json_str)
        assert restored.data is not None
        assert restored.data.file_format == "csv"
        assert restored.data.time_scale_factor == 1e6

    def test_none_data_field_is_valid(self, geo, param_constant):
        """CorrectionConfig.data may be None (backward compat)."""
        cfg = CorrectionConfig(
            n_iterations=1,
            parameters=[param_constant],
            geo=geo,
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
        )
        assert cfg.data is None


# ===========================================================================
# ParameterData – construction and typed fields
# ===========================================================================


class TestParameterData:
    def test_construction_with_all_fields(self):
        pd = ParameterData(
            current_value=[1.0, 2.0, 3.0],
            bounds=[-100.0, 100.0],
            sigma=10.0,
            units="arcseconds",
            distribution="normal",
            field="some_field",
            transformation_type="dcm_rotation",
            coordinate_frames=["F1", "F2"],
        )
        assert pd.current_value == [1.0, 2.0, 3.0]
        assert pd.sigma == 10.0
        assert pd.units == "arcseconds"

    def test_defaults(self):
        pd = ParameterData()
        assert pd.current_value == 0.0
        assert pd.sigma is None
        assert pd.units is None
        assert pd.distribution == "normal"

    # -- dict-style backward compat -------------------------------------------

    def test_get_returns_value(self):
        pd = ParameterData(sigma=30.0, units="arcseconds")
        assert pd.get("sigma") == 30.0
        assert pd.get("units") == "arcseconds"

    def test_get_returns_default_for_none_field(self):
        pd = ParameterData()  # sigma=None by default
        assert pd.get("sigma", "N/A") == "N/A"
        assert pd.get("sigma") is None

    def test_get_nonexistent_key_returns_default(self):
        pd = ParameterData()
        assert pd.get("no_such_key", "MISSING") == "MISSING"

    def test_contains_true_for_non_none_field(self):
        pd = ParameterData(sigma=30.0)
        assert "sigma" in pd

    def test_contains_false_for_none_field(self):
        pd = ParameterData()  # sigma=None
        assert "sigma" not in pd

    def test_contains_true_for_zero_sigma(self):
        """sigma=0.0 is explicitly set and must be found by 'in'."""
        pd = ParameterData(sigma=0.0)
        assert "sigma" in pd

    def test_getitem_returns_value(self):
        pd = ParameterData(sigma=5.0)
        assert pd["sigma"] == 5.0

    def test_getitem_raises_keyerror_for_missing_key(self):
        pd = ParameterData()
        with pytest.raises(KeyError):
            _ = pd["totally_missing"]

    def test_extra_fields_allowed_and_accessible(self):
        pd = ParameterData(my_custom_field="hello")
        assert pd.get("my_custom_field") == "hello"
        assert "my_custom_field" in pd
        assert pd["my_custom_field"] == "hello"

    def test_validation_error_for_non_numeric_sigma(self):
        with pytest.raises(ValidationError) as exc_info:
            ParameterData(sigma="not-a-float")
        assert "sigma" in str(exc_info.value)


# ===========================================================================
# ParameterConfig
# ===========================================================================


class TestParameterConfig:
    def test_dict_coercion(self, param_constant):
        """Passing data as a plain dict must produce a ParameterData instance."""
        assert isinstance(param_constant.data, ParameterData)
        assert param_constant.data.sigma == 30.0

    def test_none_data_becomes_empty_parameter_data(self):
        """data=None (old API) must be accepted and become a default ParameterData."""
        pc = ParameterConfig(
            ptype=ParameterType.CONSTANT_KERNEL,
            config_file=Path("kernel.json"),
            data=None,
        )
        assert isinstance(pc.data, ParameterData)

    def test_no_config_file(self):
        pc = ParameterConfig(ptype=ParameterType.OFFSET_TIME, config_file=None)
        assert pc.config_file is None

    def test_invalid_ptype_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            ParameterConfig(ptype="INVALID_TYPE", config_file=None)
        assert "ptype" in str(exc_info.value)

    def test_all_parameter_types_accepted(self):
        for ptype in ParameterType:
            pc = ParameterConfig(ptype=ptype)
            assert pc.ptype == ptype


# ===========================================================================
# GeolocationConfig
# ===========================================================================


class TestGeolocationConfig:
    def test_basic_construction(self, geo):
        assert geo.instrument_name == "TEST_INSTRUMENT"
        assert geo.time_field == "corrected_timestamp"
        assert len(geo.dynamic_kernels) == 2

    def test_dynamic_kernels_default_empty(self):
        g = GeolocationConfig(
            meta_kernel_file=Path("x.json"),
            generic_kernel_dir=Path("data"),
            instrument_name="INST",
            time_field="ts",
        )
        assert g.dynamic_kernels == []

    def test_path_fields_coerce_strings(self):
        g = GeolocationConfig(
            meta_kernel_file="path/to/mk.json",
            generic_kernel_dir="data/generic",
            instrument_name="I",
            time_field="t",
        )
        assert isinstance(g.meta_kernel_file, Path)
        assert isinstance(g.generic_kernel_dir, Path)

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            GeolocationConfig(
                meta_kernel_file=Path("x.json"),
                generic_kernel_dir=Path("data"),
                # instrument_name missing
                time_field="ts",
            )
        assert "instrument_name" in str(exc_info.value)

    def test_minimum_correlation_optional(self, geo):
        assert geo.minimum_correlation is None
        geo_with_corr = GeolocationConfig(
            meta_kernel_file=Path("x.json"),
            generic_kernel_dir=Path("d"),
            instrument_name="I",
            time_field="t",
            minimum_correlation=0.7,
        )
        assert geo_with_corr.minimum_correlation == 0.7


# ===========================================================================
# NetCDFParameterMetadata
# ===========================================================================


class TestNetCDFParameterMetadata:
    def test_construction(self):
        m = NetCDFParameterMetadata(
            variable_name="param_hysics_roll",
            units="arcseconds",
            long_name="HySICS roll correction",
        )
        assert m.variable_name == "param_hysics_roll"

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            NetCDFParameterMetadata(variable_name="x", units="m")  # long_name missing
        assert "long_name" in str(exc_info.value)


# ===========================================================================
# NetCDFConfig
# ===========================================================================


class TestNetCDFConfig:
    def test_threshold_metric_name(self, netcdf_cfg):
        assert netcdf_cfg.get_threshold_metric_name() == "percent_under_250m"

    def test_threshold_metric_name_round(self):
        nc = NetCDFConfig(performance_threshold_m=500.0)
        assert nc.get_threshold_metric_name() == "percent_under_500m"

    def test_get_standard_attributes_defaults(self, netcdf_cfg):
        attrs = netcdf_cfg.get_standard_attributes()
        assert "rms_error_m" in attrs
        assert attrs["rms_error_m"]["units"] == "meters"

    def test_get_standard_attributes_override(self):
        custom = {"my_var": {"units": "km", "long_name": "My Variable"}}
        nc = NetCDFConfig(performance_threshold_m=100.0, standard_attributes=custom)
        assert nc.get_standard_attributes() == custom

    def test_auto_generate_metadata_constant_kernel(self, netcdf_cfg, param_constant):
        meta = netcdf_cfg.get_parameter_netcdf_metadata(param_constant, angle_type="roll")
        assert "roll" in meta.long_name
        assert meta.units == "arcseconds"
        assert meta.variable_name.startswith("param_")

    def test_auto_generate_metadata_offset_time(self, netcdf_cfg, param_offset_time):
        meta = netcdf_cfg.get_parameter_netcdf_metadata(param_offset_time)
        assert meta.units == "milliseconds"

    def test_missing_threshold_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            NetCDFConfig()  # performance_threshold_m required
        assert "performance_threshold_m" in str(exc_info.value)


# ===========================================================================
# CorrectionConfig
# ===========================================================================


class TestCorrectionConfig:
    def test_basic_construction(self, minimal_config):
        assert minimal_config.n_iterations == 5
        assert minimal_config.seed == 42
        assert len(minimal_config.parameters) == 1

    def test_callable_fields_default_none(self, minimal_config):
        """_image_matching_override defaults to None."""
        assert minimal_config._image_matching_override is None

    def test_image_matching_override_can_be_set(self, minimal_config):
        """_image_matching_override accepts any callable."""

        def my_func(*args, **kwargs):
            return None

        minimal_config._image_matching_override = my_func
        assert minimal_config._image_matching_override is my_func
        minimal_config._image_matching_override = None

    def test_image_matching_func_deprecated_getter(self, minimal_config):
        """Accessing image_matching_func property emits DeprecationWarning."""
        import warnings

        minimal_config._image_matching_override = lambda: "x"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = minimal_config.image_matching_func
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert val is minimal_config._image_matching_override
        minimal_config._image_matching_override = None

    def test_image_matching_func_deprecated_setter(self, minimal_config):
        """Setting image_matching_func via deprecated property emits DeprecationWarning."""
        import warnings

        def my_func(*args, **kwargs):
            return None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minimal_config.image_matching_func = my_func
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert minimal_config._image_matching_override is my_func
        minimal_config._image_matching_override = None

    def test_mutable_fields(self, minimal_config):
        minimal_config.n_iterations = 99
        assert minimal_config.n_iterations == 99
        minimal_config.n_iterations = 5

    def test_missing_required_field_raises(self, geo, param_constant):
        with pytest.raises(ValidationError) as exc_info:
            CorrectionConfig(
                n_iterations=5,
                parameters=[param_constant],
                geo=geo,
                # performance_threshold_m missing
                performance_spec_percent=39.0,
            )
        assert "performance_threshold_m" in str(exc_info.value)

    def test_invalid_n_iterations_type_raises(self, geo, param_constant):
        with pytest.raises(ValidationError) as exc_info:
            CorrectionConfig(
                n_iterations="not-an-int",
                parameters=[param_constant],
                geo=geo,
                performance_threshold_m=250.0,
                performance_spec_percent=39.0,
            )
        assert "n_iterations" in str(exc_info.value)

    def test_validate_method_passes_for_valid_config(self, minimal_config):
        minimal_config.validate()  # must not raise

    def test_validate_method_accepts_legacy_check_loaders_kwarg(self, minimal_config):
        """check_loaders is accepted for backward compat but has no effect."""
        minimal_config.validate(check_loaders=False)  # must not raise
        minimal_config.validate(check_loaders=True)  # must also not raise

    def test_validate_method_raises_for_bad_n_iterations(self, minimal_config):
        minimal_config.n_iterations = -1
        with pytest.raises(ValueError, match="n_iterations"):
            minimal_config.validate()
        minimal_config.n_iterations = 5

    def test_correction_config_requires_no_earth_radius_m(self, geo, param_constant):
        """CorrectionConfig constructs successfully without earth_radius_m (field removed)."""
        config = CorrectionConfig(
            n_iterations=5,
            parameters=[param_constant],
            geo=geo,
            performance_threshold_m=250.0,
            performance_spec_percent=39.0,
        )
        assert not hasattr(config, "earth_radius_m") or config.model_fields.get("earth_radius_m") is None

    def test_ensure_netcdf_config_creates_default(self, minimal_config):
        assert minimal_config.netcdf is None
        minimal_config.ensure_netcdf_config()
        assert isinstance(minimal_config.netcdf, NetCDFConfig)
        assert minimal_config.netcdf.performance_threshold_m == 250.0

    def test_ensure_netcdf_config_idempotent(self, minimal_config):
        minimal_config.ensure_netcdf_config()
        first = minimal_config.netcdf
        minimal_config.ensure_netcdf_config()
        assert minimal_config.netcdf is first  # same object

    def test_get_output_filename_default(self, minimal_config):
        assert minimal_config.get_output_filename() == "correction_results.nc"

    def test_get_output_filename_custom(self, minimal_config):
        minimal_config.output_filename = "my_results.nc"
        assert minimal_config.get_output_filename() == "my_results.nc"
        minimal_config.output_filename = None

    def test_get_calibration_file(self, full_config):
        assert full_config.get_calibration_file("los_vectors") == "b_HS.mat"

    def test_get_calibration_file_missing_raises(self, full_config):
        with pytest.raises(ValueError, match="No calibration file configured"):
            full_config.get_calibration_file("nonexistent_type")

    def test_get_calibration_file_default_fallback(self, full_config):
        assert full_config.get_calibration_file("nonexistent_type", default="fallback.mat") == "fallback.mat"


# ===========================================================================
# JSON Round-Trip (acceptance criterion)
# ===========================================================================


class TestJsonRoundTrip:
    """config == CorrectionConfig.model_validate_json(config.model_dump_json())"""

    def test_minimal_config_roundtrip(self, minimal_config):
        json_str = minimal_config.model_dump_json()
        reloaded = CorrectionConfig.model_validate_json(json_str)
        assert minimal_config == reloaded

    def test_full_config_roundtrip(self, full_config):
        json_str = full_config.model_dump_json()
        reloaded = CorrectionConfig.model_validate_json(json_str)
        assert full_config == reloaded

    def test_callable_fields_excluded_from_json(self, minimal_config):
        """_image_matching_override (PrivateAttr) is always excluded from JSON serialisation."""
        minimal_config._image_matching_override = lambda: None
        json_str = minimal_config.model_dump_json()
        assert "image_matching_func" not in json_str
        assert "_image_matching_override" not in json_str
        # clean up
        minimal_config._image_matching_override = None

    def test_json_contains_expected_keys(self, minimal_config):
        import json

        data = json.loads(minimal_config.model_dump_json())
        assert "n_iterations" in data
        assert "parameters" in data
        assert "geo" in data
        assert "performance_threshold_m" in data
        assert "performance_spec_percent" in data
        assert "earth_radius_m" not in data

    def test_path_fields_survive_roundtrip(self, minimal_config):
        reloaded = CorrectionConfig.model_validate_json(minimal_config.model_dump_json())
        assert isinstance(reloaded.geo.meta_kernel_file, Path)
        assert reloaded.geo.meta_kernel_file == minimal_config.geo.meta_kernel_file
        assert reloaded.geo.generic_kernel_dir == minimal_config.geo.generic_kernel_dir

    def test_parameter_type_enum_survives_roundtrip(self, full_config):
        reloaded = CorrectionConfig.model_validate_json(full_config.model_dump_json())
        for orig, reld in zip(full_config.parameters, reloaded.parameters):
            assert orig.ptype == reld.ptype

    def test_parameter_data_fields_survive_roundtrip(self, full_config):
        reloaded = CorrectionConfig.model_validate_json(full_config.model_dump_json())
        for orig, reld in zip(full_config.parameters, reloaded.parameters):
            assert orig.data.sigma == reld.data.sigma
            assert orig.data.units == reld.data.units
            assert orig.data.bounds == reld.data.bounds
            assert orig.data.field == reld.data.field

    def test_netcdf_config_survives_roundtrip(self, full_config):
        reloaded = CorrectionConfig.model_validate_json(full_config.model_dump_json())
        assert reloaded.netcdf is not None
        assert reloaded.netcdf.performance_threshold_m == full_config.netcdf.performance_threshold_m
        assert reloaded.netcdf.title == full_config.netcdf.title

    def test_geo_model_roundtrip_standalone(self, geo):
        json_str = geo.model_dump_json()
        reloaded = GeolocationConfig.model_validate_json(json_str)
        assert geo == reloaded

    def test_netcdf_config_roundtrip_standalone(self, netcdf_cfg):
        json_str = netcdf_cfg.model_dump_json()
        reloaded = NetCDFConfig.model_validate_json(json_str)
        assert netcdf_cfg == reloaded

    def test_parameter_config_roundtrip_standalone(self, param_constant):
        json_str = param_constant.model_dump_json()
        reloaded = ParameterConfig.model_validate_json(json_str)
        assert param_constant == reloaded

    def test_parameter_data_roundtrip_standalone(self):
        pd = ParameterData(
            current_value=[1.0, 2.0, 3.0],
            bounds=[-300.0, 300.0],
            sigma=30.0,
            units="arcseconds",
            coordinate_frames=["F1", "F2"],
        )
        reloaded = ParameterData.model_validate_json(pd.model_dump_json())
        assert pd == reloaded

    def test_roundtrip_with_none_seed(self, geo, param_constant):
        config = CorrectionConfig(
            seed=None,
            n_iterations=3,
            parameters=[param_constant],
            geo=geo,
            performance_threshold_m=100.0,
            performance_spec_percent=50.0,
        )
        reloaded = CorrectionConfig.model_validate_json(config.model_dump_json())
        assert reloaded.seed is None
        assert config == reloaded


# ===========================================================================
# load_config_from_json – earth_radius_m deprecation
# ===========================================================================


class TestLoadConfigFromJsonEarthRadius:
    """Verify that earth_radius_m in JSON is accepted (with a warning) and ignored."""

    def _minimal_json(self, tmp_path, *, include_earth_radius: bool) -> Path:
        """Write a minimal valid correction config JSON to a temp file."""
        import json

        # Use the parameter format that load_config_from_json expects
        # (name + parameter_type, not ptype).
        payload = {
            "mission_config": {
                "mission_name": "TEST",
                "kernel_mappings": {
                    "constant_kernel": {},
                    "offset_kernel": {},
                },
            },
            "geolocation": {
                "instrument_name": "TEST_INST",
                "time_field": "ugps_time",
                "meta_kernel_file": str(tmp_path / "test.kernels.tm.json"),
                "generic_kernel_dir": str(tmp_path),
            },
            "correction": {
                "n_iterations": 2,
                "performance_threshold_m": 250.0,
                "performance_spec_percent": 39.0,
                "parameters": [
                    {
                        "name": "time_correction",
                        "parameter_type": "OFFSET_TIME",
                        "initial_value": 0.0,
                        "bounds": [-50.0, 50.0],
                        "sigma": 7.0,
                        "units": "milliseconds",
                        "field": "ugps_time",
                    }
                ],
            },
        }
        if include_earth_radius:
            payload["correction"]["earth_radius_m"] = 6_378_140.0

        path = tmp_path / "config.json"
        path.write_text(json.dumps(payload))
        return path

    def test_json_without_earth_radius_loads_fine(self, tmp_path):
        """Config without earth_radius_m should load without error."""
        config_path = self._minimal_json(tmp_path, include_earth_radius=False)
        config = load_config_from_json(config_path)
        assert config.performance_threshold_m == 250.0

    def test_json_with_earth_radius_loads_with_warning(self, tmp_path, caplog):
        """Config with legacy earth_radius_m loads but emits a deprecation warning."""
        import logging

        config_path = self._minimal_json(tmp_path, include_earth_radius=True)
        with caplog.at_level(logging.WARNING, logger="curryer.correction.config"):
            config = load_config_from_json(config_path)

        assert config.performance_threshold_m == 250.0
        assert any("earth_radius_m" in msg and "deprecated" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# CorrectionInput
# ---------------------------------------------------------------------------


class TestCorrectionInput:
    """Tests for the typed CorrectionInput model."""

    def test_basic_construction(self, tmp_path):
        from curryer.correction.config import CorrectionInput

        inp = CorrectionInput(
            telemetry_file=tmp_path / "tlm.csv",
            science_file=tmp_path / "sci.csv",
            gcp_file=tmp_path / "gcp.mat",
        )
        assert inp.telemetry_file == tmp_path / "tlm.csv"
        assert inp.science_file == tmp_path / "sci.csv"
        assert inp.gcp_file == tmp_path / "gcp.mat"

    def test_string_paths_coerced_to_path(self, tmp_path):
        from curryer.correction.config import CorrectionInput

        inp = CorrectionInput(
            telemetry_file="data/tlm.csv",
            science_file="data/sci.csv",
            gcp_file="gcps/chip.mat",
        )
        assert isinstance(inp.telemetry_file, Path)
        assert isinstance(inp.science_file, Path)
        assert isinstance(inp.gcp_file, Path)

    def test_run_correction_accepts_correction_input(self, minimal_config, tmp_path):
        """run_correction() normalises CorrectionInput to tuples before calling loop()."""
        from unittest.mock import patch

        from curryer.correction.config import CorrectionInput
        from curryer.correction.pipeline import run_correction

        inp = CorrectionInput(
            telemetry_file=tmp_path / "tlm.csv",
            science_file=tmp_path / "sci.csv",
            gcp_file=tmp_path / "gcp.mat",
        )

        with patch("curryer.correction.pipeline.loop") as mock_loop:
            mock_loop.return_value = ([], {})
            run_correction(minimal_config, tmp_path, [inp])

        mock_loop.assert_called_once()
        call_args = mock_loop.call_args
        normalized_inputs = call_args[0][2]  # third positional arg
        assert len(normalized_inputs) == 1
        assert normalized_inputs[0] == (
            str(tmp_path / "tlm.csv"),
            str(tmp_path / "sci.csv"),
            str(tmp_path / "gcp.mat"),
        )

    def test_run_correction_accepts_legacy_tuples(self, minimal_config, tmp_path):
        """run_correction() passes legacy tuples through unchanged."""
        from unittest.mock import patch

        from curryer.correction.pipeline import run_correction

        tuples = [("tlm.csv", "sci.csv", "gcp.mat")]

        with patch("curryer.correction.pipeline.loop") as mock_loop:
            mock_loop.return_value = ([], {})
            run_correction(minimal_config, tmp_path, tuples)

        call_args = mock_loop.call_args
        assert call_args[0][2] == tuples
