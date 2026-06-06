"""Tests for the Pydantic-based correction configuration models.

Covers:
- Construction and field validation for all BaseModel subclasses
- ``DataConfig`` config-driven data loading configuration
- ``GeolocationSetup`` / ``Sweep`` / ``OutputConfig`` — the config surface
- JSON round-trip: ``model == Model.model_validate_json(model.model_dump_json())``
- ``ValidationError`` raised with field-level messages for invalid inputs
- Callable / loader fields excluded from JSON serialisation
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from curryer.correction.config import (
    CalibrationFiles,
    DataConfig,
    GeolocationConfig,
    GeolocationSetup,
    NetCDFConfig,
    NetCDFParameterMetadata,
    OutputConfig,
    ParameterConfig,
    ParameterSpec,
    ParameterType,
    RequirementsConfig,
    SearchStrategy,
    Sweep,
    load_config_files,
    load_setup_from_json,
    load_sweep_from_json,
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
        spec={
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
        spec={
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
        spec={
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
def minimal_setup(geo) -> GeolocationSetup:
    """Minimal GeolocationSetup with no calibration/loaders."""
    return GeolocationSetup(
        geo=geo,
        requirements=RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0),
    )


@pytest.fixture
def minimal_sweep(param_constant) -> Sweep:
    """Minimal Sweep with a single parameter."""
    return Sweep(seed=42, n_iterations=5, parameters=[param_constant])


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

    def test_embedded_in_setup(self, geo):
        """DataConfig round-trips through GeolocationSetup serialisation."""
        setup = GeolocationSetup(
            geo=geo,
            requirements=RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0),
            data_config=DataConfig(file_format="csv", time_scale_factor=1e6),
        )
        json_str = setup.model_dump_json()
        restored = GeolocationSetup.model_validate_json(json_str)
        assert restored.data_config is not None
        assert restored.data_config.file_format == "csv"
        assert restored.data_config.time_scale_factor == 1e6

    def test_none_data_field_is_valid(self, geo):
        """GeolocationSetup.data_config defaults to None."""
        setup = GeolocationSetup(
            geo=geo,
            requirements=RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0),
        )
        assert setup.data_config is None


# ===========================================================================
# ParameterSpec – construction and typed fields
# ===========================================================================


class TestParameterSpec:
    def test_construction_with_all_fields(self):
        pd = ParameterSpec(
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
        pd = ParameterSpec()
        assert pd.current_value == 0.0
        assert pd.sigma is None
        assert pd.units is None
        assert pd.distribution == "normal"

    # -- strict validation + metadata -----------------------------------------

    def test_unknown_field_rejected(self):
        """extra='forbid' surfaces typos as a ValidationError."""
        with pytest.raises(ValidationError):
            ParameterSpec(boundes=[-1.0, 1.0])  # typo for 'bounds'

    def test_metadata_holds_mission_extras(self):
        pd = ParameterSpec(metadata={"name": "az_bias", "source": "vendor"})
        assert pd.metadata["name"] == "az_bias"
        assert pd.metadata["source"] == "vendor"

    def test_metadata_defaults_empty(self):
        assert ParameterSpec().metadata == {}

    def test_validation_error_for_non_numeric_sigma(self):
        with pytest.raises(ValidationError) as exc_info:
            ParameterSpec(sigma="not-a-float")
        assert "sigma" in str(exc_info.value)


# ===========================================================================
# ParameterConfig
# ===========================================================================


class TestParameterConfig:
    def test_dict_coercion(self, param_constant):
        """Passing spec as a plain dict must produce a ParameterSpec instance."""
        assert isinstance(param_constant.spec, ParameterSpec)
        assert param_constant.spec.sigma == 30.0

    def test_none_data_becomes_empty_parameter_data(self):
        """spec=None (old API) must be accepted and become a default ParameterSpec."""
        pc = ParameterConfig(
            ptype=ParameterType.CONSTANT_KERNEL,
            config_file=Path("kernel.json"),
            spec=None,
        )
        assert isinstance(pc.spec, ParameterSpec)

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
        assert netcdf_cfg.threshold_metric_name == "percent_under_250m"

    def test_threshold_metric_name_round(self):
        nc = NetCDFConfig(performance_threshold_m=500.0)
        assert nc.threshold_metric_name == "percent_under_500m"

    def test_get_standard_attributes_defaults(self, netcdf_cfg):
        attrs = netcdf_cfg.standard_attributes_dict
        assert "rms_error_m" in attrs
        assert attrs["rms_error_m"]["units"] == "meters"

    def test_get_standard_attributes_override(self):
        custom = {"my_var": {"units": "km", "long_name": "My Variable"}}
        nc = NetCDFConfig(performance_threshold_m=100.0, standard_attributes=custom)
        assert nc.standard_attributes_dict == custom

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

    def test_run_correction_accepts_correction_input(self, minimal_setup, minimal_sweep, tmp_path):
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
            run_correction(minimal_setup, minimal_sweep, [inp], tmp_path)

        mock_loop.assert_called_once()
        call_args = mock_loop.call_args
        # loop(setup, sweep, work_dir, normalized_inputs, output, resume) — inputs is the 4th positional arg.
        normalized_inputs = call_args[0][3]
        assert len(normalized_inputs) == 1
        assert normalized_inputs[0] == (
            str(tmp_path / "tlm.csv"),
            str(tmp_path / "sci.csv"),
            str(tmp_path / "gcp.mat"),
        )

    def test_run_correction_accepts_legacy_tuples(self, minimal_setup, minimal_sweep, tmp_path):
        """run_correction() passes legacy tuples through unchanged."""
        from unittest.mock import patch

        from curryer.correction.pipeline import run_correction

        tuples = [("tlm.csv", "sci.csv", "gcp.mat")]

        with patch("curryer.correction.pipeline.loop") as mock_loop:
            mock_loop.return_value = ([], {})
            run_correction(minimal_setup, minimal_sweep, tuples, tmp_path)

        call_args = mock_loop.call_args
        assert call_args[0][3] == tuples


class TestSetupSweepOutput:
    """The redesigned config surface: GeolocationSetup / Sweep / OutputConfig."""

    def _geo(self) -> GeolocationConfig:
        return GeolocationConfig(
            meta_kernel_file=Path("tests/data/test.kernels.tm.json"),
            generic_kernel_dir=Path("data/generic"),
            instrument_name="CPRS_HYSICS",
            time_field="corrected_timestamp",
        )

    def _setup(self) -> GeolocationSetup:
        return GeolocationSetup(
            geo=self._geo(),
            requirements=RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0),
            data_config=DataConfig(file_format="netcdf", time_scale_factor=1.0),
            calibration=CalibrationFiles(psf_file=Path("psf.mat"), los_vectors_file=Path("los.mat")),
            spacecraft_position_name="riss_ctrs",
            boresight_name="bhat_hs",
            transformation_matrix_name="t_hs2ctrs",
        )

    def _sweep(self) -> Sweep:
        return Sweep(
            parameters=[
                ParameterConfig(
                    ptype=ParameterType.CONSTANT_KERNEL,
                    config_file=Path("k.json"),
                    spec={
                        "current_value": [0.0, 0.0, 0.0],
                        "bounds": [-300.0, 300.0],
                        "sigma": 30.0,
                        "units": "arcseconds",
                    },
                )
            ],
            search_strategy=SearchStrategy.RANDOM,
            n_iterations=5,
            seed=42,
        )

    def test_setup_construction_and_defaults(self):
        setup = GeolocationSetup(
            geo=self._geo(),
            requirements=RequirementsConfig(performance_threshold_m=250.0, performance_spec_percent=39.0),
        )
        assert setup.spacecraft_position_name == "sc_position"
        assert setup.boresight_name == "boresight"
        assert setup.transformation_matrix_name == "t_inst2ref"
        assert setup.calibration is None
        assert setup.image_matching_func is None

    def test_setup_requires_geo_and_requirements(self):
        with pytest.raises(ValidationError):
            GeolocationSetup(geo=self._geo())  # missing requirements

    def test_setup_json_round_trip_excludes_callable(self):
        setup = self._setup()
        setup.image_matching_func = lambda *a, **k: None  # callable hook
        json_str = setup.model_dump_json()
        assert "image_matching_func" not in json_str
        restored = GeolocationSetup.model_validate_json(json_str)
        assert restored.geo.instrument_name == "CPRS_HYSICS"
        assert restored.requirements.performance_threshold_m == 250.0
        assert restored.spacecraft_position_name == "riss_ctrs"
        assert restored.calibration.psf_file == Path("psf.mat")

    def test_sweep_defaults_and_round_trip(self):
        sweep = self._sweep()
        assert sweep.search_strategy is SearchStrategy.RANDOM
        assert sweep.n_iterations == 5
        assert sweep.grid_points_per_param == 10
        restored = Sweep.model_validate_json(sweep.model_dump_json())
        assert len(restored.parameters) == 1
        assert restored.seed == 42

    def test_sweep_requires_at_least_one_parameter(self):
        with pytest.raises(ValidationError):
            Sweep(parameters=[])

    def test_output_config_filename_default_and_override(self):
        assert OutputConfig().get_output_filename() == "correction_results.nc"
        assert OutputConfig(output_filename="run.nc").get_output_filename() == "run.nc"

    def test_load_config_files_from_json(self, tmp_path):
        """load_config_files() parses setup/sweep/output sections of one JSON file."""
        import json

        cfg = {
            "setup": {
                "geo": {
                    "meta_kernel_file": "m.json",
                    "generic_kernel_dir": "data/generic",
                    "instrument_name": "CPRS_HYSICS",
                    "time_field": "corrected_timestamp",
                },
                "requirements": {"performance_threshold_m": 250.0, "performance_spec_percent": 39.0},
                "data_config": {"file_format": "netcdf", "time_scale_factor": 1.0},
                "calibration": {"psf_file": "psf.mat", "los_vectors_file": "los.mat"},
                "spacecraft_position_name": "riss_ctrs",
            },
            "sweep": {
                "search_strategy": "grid",
                "grid_points_per_param": 7,
                "parameters": [
                    {
                        "ptype": "CONSTANT_KERNEL",
                        "config_file": "frame.attitude.ck.json",
                        "spec": {
                            "current_value": [0.0, 0.0, 0.0],
                            "bounds": [-300.0, 300.0],
                            "sigma": 30.0,
                            "units": "arcseconds",
                        },
                    },
                    {
                        "ptype": "OFFSET_TIME",
                        "config_file": None,
                        "spec": {
                            "field": "corrected_timestamp",
                            "current_value": 0.0,
                            "bounds": [-50.0, 50.0],
                            "sigma": 7.0,
                            "units": "milliseconds",
                        },
                    },
                ],
            },
            "output": {"output_filename": "results.nc"},
        }
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(cfg))

        setup, sweep, output = load_config_files(path)
        assert setup.geo.instrument_name == "CPRS_HYSICS"
        assert setup.requirements.performance_threshold_m == 250.0
        assert setup.spacecraft_position_name == "riss_ctrs"
        assert setup.calibration.psf_file == Path("psf.mat")
        assert sweep.search_strategy is SearchStrategy.GRID_SEARCH
        assert sweep.grid_points_per_param == 7
        assert len(sweep.parameters) == 2
        assert sweep.parameters[1].ptype is ParameterType.OFFSET_TIME
        assert output.get_output_filename() == "results.nc"

    def test_load_setup_and_sweep_separately(self, tmp_path):
        import json

        cfg = {
            "setup": {
                "geo": {
                    "meta_kernel_file": "m.json",
                    "generic_kernel_dir": "g",
                    "instrument_name": "X",
                    "time_field": "t",
                },
                "requirements": {"performance_threshold_m": 100.0, "performance_spec_percent": 50.0},
            },
            "sweep": {"parameters": [{"ptype": "OFFSET_TIME", "spec": {"field": "t"}}]},
        }
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(cfg))
        assert load_setup_from_json(path).geo.instrument_name == "X"
        assert load_sweep_from_json(path).parameters[0].ptype is ParameterType.OFFSET_TIME
