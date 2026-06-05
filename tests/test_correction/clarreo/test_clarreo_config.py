"""Tests for CLARREO correction configuration generation and JSON serialisation."""

from __future__ import annotations

import json

import pytest
from clarreo_config import create_clarreo_correction_config

from curryer.correction import correction
from curryer.correction.config import DataConfig


def test_generate_clarreo_config_json(tmp_path, clarreo_gcs_data_dir, clarreo_generic_dir):
    """Generate the CLARREO config JSON and validate structure end-to-end."""
    output_path = tmp_path / "configs" / "clarreo_correction_config.json"

    config = create_clarreo_correction_config(clarreo_gcs_data_dir, clarreo_generic_dir, config_output_path=output_path)

    assert output_path.exists()

    with open(output_path) as fh:
        config_data = json.load(fh)

    assert "mission_config" in config_data
    assert "correction" in config_data
    assert "geolocation" in config_data
    assert config_data["mission_config"]["mission_name"] == "CLARREO_Pathfinder"

    corr = config_data["correction"]
    assert isinstance(corr.get("parameters"), list)
    assert len(corr["parameters"]) > 0
    assert corr["performance_threshold_m"] == 250.0
    assert corr["performance_spec_percent"] == 39.0
    assert config_data["geolocation"]["instrument_name"] == "CPRS_HYSICS"

    reloaded = correction.load_config_from_json(output_path)
    assert reloaded.n_iterations == config.n_iterations
    assert len(reloaded.parameters) == len(config.parameters)

    # Verify each reloaded parameter has the expected ptype, config_file (for
    # kernel-based params), and data.field (for OFFSET_KERNEL / OFFSET_TIME),
    # so this test will fail if the JSON can't be used to run correction.loop().
    valid_ptypes = {
        correction.ParameterType.CONSTANT_KERNEL,
        correction.ParameterType.OFFSET_KERNEL,
        correction.ParameterType.OFFSET_TIME,
    }
    for param in reloaded.parameters:
        assert param.ptype in valid_ptypes, f"Unexpected ptype: {param.ptype}"

        if param.ptype in (correction.ParameterType.CONSTANT_KERNEL, correction.ParameterType.OFFSET_KERNEL):
            assert param.config_file is not None, f"{param.ptype.name} parameter must have a config_file, got None"

        if param.ptype in (correction.ParameterType.OFFSET_KERNEL, correction.ParameterType.OFFSET_TIME):
            assert param.spec.field is not None, f"{param.ptype.name} parameter must have spec.field set, got None"

    # Count parameters by type to ensure the expected composition is preserved.
    ptypes = [p.ptype for p in reloaded.parameters]
    assert ptypes.count(correction.ParameterType.CONSTANT_KERNEL) >= 1
    assert ptypes.count(correction.ParameterType.OFFSET_KERNEL) >= 1
    assert ptypes.count(correction.ParameterType.OFFSET_TIME) >= 1


class TestClarreoConfiguration:
    """Smoke tests for the CLARREO CorrectionConfig object."""

    @pytest.fixture(autouse=True)
    def _setup(self, clarreo_gcs_data_dir, clarreo_generic_dir):
        self.data_dir = clarreo_gcs_data_dir
        self.generic_dir = clarreo_generic_dir

    def test_config_validates(self):
        config = create_clarreo_correction_config(self.data_dir, self.generic_dir)
        config.data_config = DataConfig(file_format="csv", time_scale_factor=1e6)
        assert config.geo.instrument_name == "CPRS_HYSICS"
        assert config.seed == 42

    def test_parameter_count(self):
        config = create_clarreo_correction_config(self.data_dir, self.generic_dir)
        assert len(config.parameters) == 6

    def test_performance_thresholds(self):
        config = create_clarreo_correction_config(self.data_dir, self.generic_dir)
        assert config.performance_threshold_m == 250.0
        assert config.performance_spec_percent == 39.0
