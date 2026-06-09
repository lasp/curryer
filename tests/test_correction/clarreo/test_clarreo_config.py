"""Tests for CLARREO correction configuration generation and JSON serialisation."""

from __future__ import annotations

import json

import pytest
from clarreo_config import create_clarreo_setup_sweep

from curryer.correction.config import DataConfig, ParameterType, load_config_files


def test_generate_clarreo_config_json(tmp_path, clarreo_gcs_data_dir, clarreo_generic_dir):
    """Serialise the CLARREO setup/sweep/output and reload via the public loader."""
    output_path = tmp_path / "configs" / "clarreo_correction_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup, sweep, output = create_clarreo_setup_sweep(clarreo_gcs_data_dir, clarreo_generic_dir)

    config_data = {
        "setup": setup.model_dump(mode="json"),
        "sweep": sweep.model_dump(mode="json"),
        "output": output.model_dump(mode="json"),
    }
    with open(output_path, "w") as fh:
        json.dump(config_data, fh)

    assert output_path.exists()

    assert "setup" in config_data
    assert "sweep" in config_data
    assert config_data["setup"]["geo"]["instrument_name"] == "CPRS_HYSICS"

    sweep_data = config_data["sweep"]
    assert isinstance(sweep_data.get("parameters"), list)
    assert len(sweep_data["parameters"]) > 0
    assert config_data["setup"]["requirements"]["performance_threshold_m"] == 250.0
    assert config_data["setup"]["requirements"]["performance_spec_percent"] == 39.0

    reloaded_setup, reloaded_sweep, _reloaded_output = load_config_files(output_path)
    assert reloaded_setup.geo.instrument_name == "CPRS_HYSICS"
    assert reloaded_sweep.n_iterations == sweep.n_iterations
    assert len(reloaded_sweep.parameters) == len(sweep.parameters)

    # Verify each reloaded parameter has the expected ptype, config_file (for
    # kernel-based params), and data.field (for OFFSET_KERNEL / OFFSET_TIME),
    # so this test will fail if the JSON can't be used to run loop().
    valid_ptypes = {
        ParameterType.CONSTANT_KERNEL,
        ParameterType.OFFSET_KERNEL,
        ParameterType.OFFSET_TIME,
    }
    for param in reloaded_sweep.parameters:
        assert param.ptype in valid_ptypes, f"Unexpected ptype: {param.ptype}"

        if param.ptype in (ParameterType.CONSTANT_KERNEL, ParameterType.OFFSET_KERNEL):
            assert param.config_file is not None, f"{param.ptype.name} parameter must have a config_file, got None"

        if param.ptype in (ParameterType.OFFSET_KERNEL, ParameterType.OFFSET_TIME):
            assert param.spec.field is not None, f"{param.ptype.name} parameter must have spec.field set, got None"

    # Count parameters by type to ensure the expected composition is preserved.
    ptypes = [p.ptype for p in reloaded_sweep.parameters]
    assert ptypes.count(ParameterType.CONSTANT_KERNEL) >= 1
    assert ptypes.count(ParameterType.OFFSET_KERNEL) >= 1
    assert ptypes.count(ParameterType.OFFSET_TIME) >= 1


class TestClarreoConfiguration:
    """Smoke tests for the CLARREO setup/sweep models."""

    @pytest.fixture(autouse=True)
    def _setup(self, clarreo_gcs_data_dir, clarreo_generic_dir):
        self.data_dir = clarreo_gcs_data_dir
        self.generic_dir = clarreo_generic_dir

    def test_config_validates(self):
        setup, sweep, _output = create_clarreo_setup_sweep(self.data_dir, self.generic_dir)
        setup.data_config = DataConfig(file_format="csv", time_scale_factor=1e6)
        assert setup.geo.instrument_name == "CPRS_HYSICS"
        assert sweep.seed == 42

    def test_parameter_count(self):
        _setup, sweep, _output = create_clarreo_setup_sweep(self.data_dir, self.generic_dir)
        assert len(sweep.parameters) == 6

    def test_performance_thresholds(self):
        setup, _sweep, _output = create_clarreo_setup_sweep(self.data_dir, self.generic_dir)
        assert setup.requirements.performance_threshold_m == 250.0
        assert setup.requirements.performance_spec_percent == 39.0
