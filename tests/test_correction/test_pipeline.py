"""Tests for ``curryer.correction.pipeline``.

Covers:
- ``_extract_parameter_values``
- ``_extract_error_metrics``
- ``_store_parameter_values``
- ``_store_gcp_pair_results``
- ``_compute_parameter_set_metrics``
- ``_load_image_pair_data``
- ``_extract_spacecraft_position_midframe`` (position_columns feature)
- ``loop`` (optimised pair-outer, ``@pytest.mark.extra``)
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _synthetic_helpers import synthetic_image_matching
from clarreo_config import create_clarreo_setup_sweep
from clarreo_data_loaders import load_clarreo_science, load_clarreo_telemetry

from curryer.correction.config import CalibrationData, DataConfig, ParameterConfig, ParameterType
from curryer.correction.io_config import NetCDFConfig
from curryer.correction.pipeline import (
    _compute_parameter_set_metrics,
    _extract_error_metrics,
    _extract_parameter_values,
    _load_image_pair_data,
    _require_image_matching_inputs,
    _resolve_netcdf_config,
    _store_gcp_pair_results,
    _store_parameter_values,
    loop,
)
from curryer.correction.verification import _extract_spacecraft_position_midframe

# ── shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def clarreo_cfg(root_dir):
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
    generic_dir = root_dir / "data" / "generic"
    setup, sweep, output = create_clarreo_setup_sweep(data_dir, generic_dir)
    return setup, sweep, output


# ── tests ─────────────────────────────────────────────────────────────────────


def test_extract_parameter_values():
    """_extract_parameter_values returns roll/pitch/yaw keys."""
    param_config = ParameterConfig(ptype=ParameterType.CONSTANT_KERNEL, config_file=Path("test_kernel.json"), spec=None)
    param_data = pd.DataFrame(
        {
            "angle_x": [np.radians(1.0 / 3600)],
            "angle_y": [np.radians(2.0 / 3600)],
            "angle_z": [np.radians(3.0 / 3600)],
        }
    )
    result = _extract_parameter_values([(param_config, param_data)])
    assert isinstance(result, dict)
    assert len(result) == 3
    assert "test_kernel_roll" in result
    assert "test_kernel_pitch" in result
    assert "test_kernel_yaw" in result


def test_extract_error_metrics():
    """_extract_error_metrics pulls named metrics from a Dataset."""
    ds = xr.Dataset({"lat_error_deg": (["pt"], [0.001, 0.002])})
    ds.attrs.update(
        {
            "rms_error_m": 150.0,
            "mean_error_m": 140.0,
            "max_error_m": 200.0,
            "std_error_m": 10.0,
            "total_measurements": 2,
        }
    )
    m = _extract_error_metrics(ds)
    assert m["rms_error_m"] == 150.0
    assert m["n_measurements"] == 2


def test_store_parameter_values():
    """_store_parameter_values writes values at the correct index."""
    netcdf_data = {"parameter_set_id": np.zeros(3, dtype=int), "param_foo": np.zeros(3)}
    _store_parameter_values(netcdf_data, param_idx=1, param_values={"foo": 2.5})
    assert netcdf_data["param_foo"][1] == pytest.approx(2.5)


def test_store_gcp_pair_results():
    """_store_gcp_pair_results populates all metric arrays correctly."""
    nc = {k: np.zeros((2, 2)) for k in ("rms_error_m", "mean_error_m", "max_error_m", "std_error_m")}
    nc["n_measurements"] = np.zeros((2, 2), dtype=int)
    metrics = {
        "rms_error_m": 150.0,
        "mean_error_m": 140.0,
        "max_error_m": 200.0,
        "std_error_m": 10.0,
        "n_measurements": 10,
    }
    _store_gcp_pair_results(nc, param_idx=0, pair_idx=1, error_metrics=metrics)
    assert nc["rms_error_m"][0, 1] == 150.0
    assert nc["std_error_m"][0, 1] == 10.0
    assert nc["n_measurements"][0, 1] == 10


def test_compute_parameter_set_metrics():
    """_compute_parameter_set_metrics populates aggregate stats."""
    nc = {
        "percent_under_250m": np.zeros(2),
        "mean_rms_all_pairs": np.zeros(2),
        "best_pair_rms": np.zeros(2),
        "worst_pair_rms": np.zeros(2),
    }
    _compute_parameter_set_metrics(nc, param_idx=0, pair_errors=[100.0, 200.0, 300.0], threshold_m=250.0)
    assert nc["percent_under_250m"][0] > 0
    assert nc["best_pair_rms"][0] == 100.0
    assert nc["worst_pair_rms"][0] == 300.0


class TestRequireImageMatchingInputs:
    """_require_image_matching_inputs fail-fast guard for the built-in matcher."""

    def test_builtin_matcher_without_calibration_raises(self):
        """No override + no calibration -> clear early ValueError."""
        setup = SimpleNamespace(image_matching_func=None)
        calibration = CalibrationData(los_vectors=None, optical_psfs=None)
        with pytest.raises(ValueError, match="no calibration data is configured"):
            _require_image_matching_inputs(setup, calibration)

    def test_override_without_calibration_allowed(self):
        """A custom override may legitimately need no calibration."""
        setup = SimpleNamespace(image_matching_func=lambda **_: None)
        calibration = CalibrationData(los_vectors=None, optical_psfs=None)
        _require_image_matching_inputs(setup, calibration)  # no raise

    def test_builtin_matcher_with_calibration_allowed(self):
        """No override but calibration present -> built-in matcher can run."""
        setup = SimpleNamespace(image_matching_func=None)
        calibration = CalibrationData(los_vectors=np.zeros((4, 3)), optical_psfs=[object()])
        _require_image_matching_inputs(setup, calibration)  # no raise


class TestResolveNetcdfConfig:
    """_resolve_netcdf_config pins the output threshold to the requirement."""

    def test_defaults_threshold_from_requirements(self):
        setup = SimpleNamespace(requirements=SimpleNamespace(performance_threshold_m=300.0))
        output = SimpleNamespace(netcdf=None)
        resolved = _resolve_netcdf_config(setup, output)
        assert resolved.performance_threshold_m == 300.0

    def test_pins_threshold_over_caller_override(self):
        """A divergent output.netcdf threshold is pinned to the requirement, so the
        written variable name/metadata match the computed threshold; other fields stay."""
        setup = SimpleNamespace(requirements=SimpleNamespace(performance_threshold_m=300.0))
        output = SimpleNamespace(netcdf=NetCDFConfig(performance_threshold_m=250.0, title="Custom"))
        resolved = _resolve_netcdf_config(setup, output)
        assert resolved.performance_threshold_m == 300.0
        assert resolved.threshold_metric_name == "percent_under_300m"
        assert resolved.title == "Custom"


def test_load_image_pair_data(root_dir, clarreo_cfg, tmp_path):
    """_load_image_pair_data returns DataFrames for tlm and sci."""
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
    tlm_csv = tmp_path / "tlm.csv"
    sci_csv = tmp_path / "sci.csv"
    load_clarreo_telemetry(data_dir).to_csv(tlm_csv)
    load_clarreo_science(data_dir).to_csv(sci_csv)
    setup, _sweep, _output = clarreo_cfg
    setup = setup.model_copy(deep=True)
    setup.data_config = DataConfig(file_format="csv", time_scale_factor=1e6)
    tlm_ds, sci_ds, ugps = _load_image_pair_data(str(tlm_csv), str(sci_csv), setup)
    assert isinstance(tlm_ds, pd.DataFrame)
    assert isinstance(sci_ds, pd.DataFrame)
    assert ugps is not None


@pytest.mark.extra
def test_loop_optimized(root_dir, tmp_path):
    """loop() produces correct result structure. Requires GMTED – ``--run-extra``."""
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
    generic_dir = root_dir / "data" / "generic"
    setup, sweep, output = create_clarreo_setup_sweep(data_dir, generic_dir)
    sweep.n_iterations = 2
    output.output_filename = "test_loop.nc"
    work = tmp_path / "loop"
    work.mkdir()
    tlm_csv, sci_csv = work / "tlm.csv", work / "sci.csv"
    load_clarreo_telemetry(data_dir).to_csv(tlm_csv)
    load_clarreo_science(data_dir).to_csv(sci_csv)
    setup.data_config = DataConfig(file_format="csv", time_scale_factor=1e6)
    setup.image_matching_func = synthetic_image_matching
    sets = [(str(tlm_csv), str(sci_csv), "synthetic_gcp.mat")]
    np.random.seed(42)
    results, nc = loop(setup, sweep, work, sets, output=output, resume_from_checkpoint=False)
    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) == sweep.n_iterations * len(sets)
    assert nc["rms_error_m"].shape == (sweep.n_iterations, len(sets))
    for r in results:
        assert "param_index" in r
        assert "pair_index" in r
        assert "rms_error_m" in r
        assert r["aggregate_rms_error_m"] is not None
        assert isinstance(r["aggregate_rms_error_m"], (int, float, np.number))


# ── _extract_spacecraft_position_midframe ─────────────────────────────────────


def _make_telemetry() -> pd.DataFrame:
    """Return a 3-row telemetry DataFrame with standard column names."""
    return pd.DataFrame(
        {
            "sc_pos_x": [1.0, 2.0, 3.0],
            "sc_pos_y": [4.0, 5.0, 6.0],
            "sc_pos_z": [7.0, 8.0, 9.0],
        }
    )


class TestExtractSpacecraftPositionMidframe:
    """Tests for _extract_spacecraft_position_midframe with position_columns."""

    def test_explicit_position_columns_used(self):
        """config.data_config.position_columns should be used directly."""
        telemetry = pd.DataFrame(
            {
                "my_x": [1.0, 2.0, 3.0],
                "my_y": [4.0, 5.0, 6.0],
                "my_z": [7.0, 8.0, 9.0],
            }
        )
        config = MagicMock()
        config.data_config = DataConfig(position_columns=["my_x", "my_y", "my_z"])

        result = _extract_spacecraft_position_midframe(telemetry, setup=config)

        np.testing.assert_array_equal(result, [2.0, 5.0, 8.0])  # mid_idx = 1

    def test_explicit_position_columns_returns_float64(self):
        """Result should be a float64 ndarray of shape (3,)."""
        telemetry = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        config = MagicMock()
        config.data_config = DataConfig(position_columns=["a", "b", "c"])

        result = _extract_spacecraft_position_midframe(telemetry, setup=config)

        assert result.shape == (3,)
        assert result.dtype == np.float64

    def test_position_columns_wrong_length_raises_valueerror(self):
        """position_columns with != 3 entries should raise ValueError."""
        telemetry = pd.DataFrame({"x": [1.0], "y": [2.0]})
        config = MagicMock()
        config.data_config = DataConfig(position_columns=["x", "y"])

        with pytest.raises(ValueError, match="exactly 3 entries"):
            _extract_spacecraft_position_midframe(telemetry, setup=config)

    def test_position_columns_missing_column_raises_valueerror(self):
        """position_columns referencing nonexistent columns should raise ValueError."""
        telemetry = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
        config = MagicMock()
        config.data_config = DataConfig(position_columns=["x", "y", "MISSING"])

        with pytest.raises(ValueError, match="not found in telemetry"):
            _extract_spacecraft_position_midframe(telemetry, setup=config)

    def test_no_position_columns_falls_back_with_warning(self, caplog):
        """When position_columns is None, fall back to pattern-guessing with warning."""
        telemetry = _make_telemetry()
        config = MagicMock()
        config.data_config = None  # position_columns not configured

        with caplog.at_level(logging.WARNING, logger="curryer.correction.pipeline"):
            result = _extract_spacecraft_position_midframe(telemetry, setup=config)

        assert "position_columns not configured" in caplog.text
        np.testing.assert_array_equal(result, [2.0, 5.0, 8.0])

    def test_no_config_falls_back_to_pattern_guessing(self, caplog):
        """When config=None entirely, pattern-guessing is used (backward compat)."""
        telemetry = _make_telemetry()

        with caplog.at_level(logging.WARNING, logger="curryer.correction.verification"):
            result = _extract_spacecraft_position_midframe(telemetry, setup=None)

        assert "position_columns not configured" in caplog.text
        np.testing.assert_array_equal(result, [2.0, 5.0, 8.0])
