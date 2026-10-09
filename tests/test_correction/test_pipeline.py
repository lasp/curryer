"""Tests for ``curryer.correction.pipeline``.

Covers:
- ``_extract_parameter_values``
- ``_extract_error_metrics``
- ``_store_parameter_values``
- ``_store_gcp_pair_results``
- ``_compute_parameter_set_metrics``
- ``_usable_pairs_and_valid_sets``
- ``load_loop_observation``
- ``_extract_spacecraft_position_midframe`` (position_columns feature)
- ``loop`` with SPICE geolocation and image matching (``@pytest.mark.extra``)
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
from _loop_scene import build_loop_scene

from curryer.correction.config import CalibrationData, DataConfig, ParameterConfig, ParameterType
from curryer.correction.io_config import NetCDFConfig
from curryer.correction.pipeline import (
    _compute_parameter_set_metrics,
    _extract_error_metrics,
    _extract_parameter_values,
    _per_pair_error_processor,
    _require_image_matching_inputs,
    _resolve_netcdf_config,
    _store_gcp_pair_results,
    _store_parameter_values,
    _usable_pairs_and_valid_sets,
    load_loop_observation,
    loop,
)
from curryer.correction.verification import _extract_spacecraft_position_midframe

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


def test_extract_parameter_values_offsets_in_configured_units():
    """OFFSET_KERNEL (radians internally) and OFFSET_TIME (seconds) are stored in their configured units."""
    angle = ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL, config_file=Path("az.json"), spec=dict(field="az", units="arcseconds")
    )
    time = ParameterConfig(ptype=ParameterType.OFFSET_TIME, spec=dict(field="t", units="milliseconds"))
    result = _extract_parameter_values([(angle, np.deg2rad(30.0 / 3600.0)), (time, 0.25)])
    assert result["az"] == pytest.approx(30.0)
    assert result["t"] == pytest.approx(250.0)


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


def test_store_parameter_values_unknown_variable_raises():
    netcdf_data = {"parameter_set_id": np.zeros(3, dtype=int), "param_foo": np.zeros(3)}
    with pytest.raises(KeyError, match="param_bar"):
        _store_parameter_values(netcdf_data, param_idx=1, param_values={"bar": 2.5})


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
    """_require_image_matching_inputs: the loop always needs LOS; the built-in matcher also needs a PSF."""

    def test_no_los_raises(self):
        setup = SimpleNamespace(observation_matching_func=lambda *_: None)
        with pytest.raises(ValueError, match="no calibration data is configured"):
            _require_image_matching_inputs(setup, CalibrationData(los_vectors=None, optical_psfs=None))

    def test_builtin_matcher_without_psf_raises(self):
        setup = SimpleNamespace(observation_matching_func=None)
        with pytest.raises(ValueError, match="no optical PSF is configured"):
            _require_image_matching_inputs(setup, CalibrationData(los_vectors=np.zeros((4, 3)), optical_psfs=None))

    def test_override_without_psf_allowed(self):
        setup = SimpleNamespace(observation_matching_func=lambda *_: None)
        _require_image_matching_inputs(setup, CalibrationData(los_vectors=np.zeros((4, 3)), optical_psfs=None))

    def test_builtin_matcher_with_calibration_allowed(self):
        setup = SimpleNamespace(observation_matching_func=None)
        _require_image_matching_inputs(setup, CalibrationData(los_vectors=np.zeros((4, 3)), optical_psfs=[object()]))


class TestUsablePairsAndValidSets:
    """_usable_pairs_and_valid_sets: which pairs and parameter sets the selection compares."""

    def test_all_accepted(self):
        pairs, valid = _usable_pairs_and_valid_sets(np.ones((3, 2), dtype=bool))
        np.testing.assert_array_equal(pairs, [0, 1])
        np.testing.assert_array_equal(valid, [True, True, True])

    def test_pair_rejected_everywhere_is_not_used(self):
        """A pair no parameter set matches (e.g. cloud) leaves every set valid."""
        accepted = np.array([[True, False], [True, False]])
        pairs, valid = _usable_pairs_and_valid_sets(accepted)
        np.testing.assert_array_equal(pairs, [0])
        np.testing.assert_array_equal(valid, [True, True])

    def test_set_losing_a_usable_pair_is_invalid(self):
        """A parameter set that moves a matchable pair out of matching range is not compared."""
        accepted = np.array([[True, True], [True, False]])
        pairs, valid = _usable_pairs_and_valid_sets(accepted)
        np.testing.assert_array_equal(pairs, [0, 1])
        np.testing.assert_array_equal(valid, [True, False])

    def test_no_usable_pair_raises(self):
        with pytest.raises(ValueError, match="under any parameter set"):
            _usable_pairs_and_valid_sets(np.zeros((2, 3), dtype=bool))

    def test_no_valid_set_raises(self):
        accepted = np.array([[True, False], [False, True]])
        with pytest.raises(ValueError, match=r"on every usable GCP pair; .* \[1, 1\] of 2"):
            _usable_pairs_and_valid_sets(accepted)


def _write_observation(path, n_frames=4, n_columns=3, **overrides):
    data = {
        "band_data": (["frame", "pixel"], np.arange(n_frames * n_columns, dtype=float).reshape(n_frames, n_columns)),
        "ugps": (["frame"], 1_000_000_000_000 + 66_667 * np.arange(n_frames, dtype=np.int64)),
        "detector_pixel": (["pixel"], np.arange(1, n_columns + 1)),
    }
    data.update(overrides)
    xr.Dataset({k: v for k, v in data.items() if v is not None}).to_netcdf(path)
    return path


class TestLoadLoopObservation:
    """load_loop_observation: the loop's observation contract."""

    los = np.eye(3)[[0, 1, 2, 0, 1]].astype(float)

    def test_reads_radiance_times_and_los_rows(self, tmp_path):
        obs = load_loop_observation(_write_observation(tmp_path / "obs.nc"), self.los)
        assert obs.radiance.shape == (4, 3)
        assert obs.frame_ugps.dtype == np.int64
        np.testing.assert_array_equal(obs.los_vectors, self.los[[1, 2, 3]])

    def test_missing_ugps_raises(self, tmp_path):
        path = _write_observation(tmp_path / "obs.nc", ugps=None)
        with pytest.raises(ValueError, match="must carry 'band_data' and 'ugps'"):
            load_loop_observation(path, self.los)

    def test_float_times_raise(self, tmp_path):
        path = _write_observation(tmp_path / "obs.nc", ugps=(["frame"], np.arange(4, dtype=float) * 1e5))
        with pytest.raises(ValueError, match="strictly increasing integers"):
            load_loop_observation(path, self.los)

    def test_repeated_time_raises(self, tmp_path):
        path = _write_observation(tmp_path / "obs.nc", ugps=(["frame"], np.array([10, 20, 20, 30], dtype=np.int64)))
        with pytest.raises(ValueError, match="strictly increasing integers"):
            load_loop_observation(path, self.los)

    def test_decreasing_unsigned_times_raise(self, tmp_path):
        """uint64 differences wrap instead of going negative; decreasing times still raise."""
        path = _write_observation(tmp_path / "obs.nc", ugps=(["frame"], np.array([40, 30, 20, 10], dtype=np.uint64)))
        with pytest.raises(ValueError, match="strictly increasing integers"):
            load_loop_observation(path, self.los)

    def test_non_finite_radiance_raises(self, tmp_path):
        radiance = np.ones((4, 3))
        radiance[2, 1] = np.nan
        path = _write_observation(tmp_path / "obs.nc", band_data=(["frame", "pixel"], radiance))
        with pytest.raises(ValueError, match="finite 2-D"):
            load_loop_observation(path, self.los)

    def test_detector_pixel_outside_los_table_raises(self, tmp_path):
        path = _write_observation(tmp_path / "obs.nc", detector_pixel=(["pixel"], np.array([3, 4, 5])))
        with pytest.raises(ValueError, match="detector_pixel"):
            load_loop_observation(path, self.los)


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


def test_per_pair_error_processor_ignores_correlation_threshold():
    """Per-pair loop errors keep a below-threshold measurement; the setup's names are kept."""
    from test_error_stats import create_test_dataset_13_cases

    setup = SimpleNamespace(
        geo=SimpleNamespace(minimum_correlation=0.5, minimum_peak_margin=0.1),
        spacecraft_position_name="riss_ctrs",
        boresight_name="bhat_hs",
        transformation_matrix_name="t_hs2ctrs",
    )
    processor = _per_pair_error_processor(setup)
    data = create_test_dataset_13_cases()
    correlation = np.full(data.sizes["measurement"], 0.9)
    correlation[0] = 0.1
    data["correlation"] = (["measurement"], correlation)
    data["correlation_secondary"] = (["measurement"], correlation)

    out = processor.compute_nadir_equivalent_errors(data)

    assert processor.config.minimum_correlation is None
    assert processor.config.minimum_peak_margin is None
    assert out.sizes["measurement"] == data.sizes["measurement"]


def test_loop_refuses_to_resume_completed_pairs(root_dir, tmp_path, monkeypatch):
    """A checkpoint with completed pairs cannot be resumed (CURRYER-100); the loop says so before any work."""
    from clarreo_config import create_clarreo_setup_sweep

    from curryer.correction import pipeline

    setup, sweep, output = create_clarreo_setup_sweep(root_dir / "tests" / "data" / "clarreo" / "gcs", root_dir)
    monkeypatch.setattr(pipeline, "_load_checkpoint", lambda _path: ({}, 1))
    with pytest.raises(NotImplementedError, match="CURRYER-100"):
        loop(setup, sweep, tmp_path, [("t.csv", "o.nc", "c.nc")] * 2, output=output, resume_from_checkpoint=True)


def _reject_first_parameter_set(n_param_sets):
    """Built-in matching, with correlation 0 for parameter set 0 (calls arrive in parameter-set order per pair)."""
    from curryer.correction.verification import match_observation

    calls = iter(range(10**6))

    def matcher(*args):
        ds = match_observation(*args)
        if next(calls) % n_param_sets == 0:
            ds["correlation"] = ds["correlation"] * 0.0
        return ds

    return matcher


@pytest.fixture(scope="module")
def loop_run(root_dir, tmp_path_factory):
    """One loop over the synthetic scene seen 0.2 s after its frame times, with parameter set 0
    failing the correlation gate. Requires GMTED."""
    from curryer.correction.parameters import load_param_sets

    work = tmp_path_factory.mktemp("loop")
    setup, sweep, output, sets = build_loop_scene(root_dir, work, time_offset_s=0.2)
    setup.observation_matching_func = _reject_first_parameter_set(len(load_param_sets(sweep)))
    results, netcdf_data = loop(setup, sweep, work, sets, output=output)
    return setup, sweep, output, sets, work, results, netcdf_data


@pytest.mark.extra
def test_loop_selects_the_true_time_offset(loop_run):
    """With SPICE geolocation and image matching, the +0.2 s parameter set re-geolocates onto the truth."""
    *_, results, nc = loop_run
    best = int(np.nanargmin(nc["mean_rms_all_pairs"]))
    assert nc["param_corrected_timestamp"][best] == pytest.approx(200.0)
    assert nc["rms_error_m"][best, 0] < 300.0
    worst = int(np.nanargmax(nc["mean_rms_all_pairs"]))
    assert nc["param_corrected_timestamp"][worst] == pytest.approx(-200.0)


@pytest.mark.extra
def test_loop_does_not_compare_a_set_failing_the_gates(loop_run):
    """Parameter set 0 fails the gate on the only (usable) pair: it is invalid, unscored and not selected."""
    *_, results, nc = loop_run
    assert not nc["accepted"][0, 0]
    np.testing.assert_array_equal(nc["valid"], np.arange(len(nc["valid"])) != 0)
    assert np.isnan(nc["mean_rms_all_pairs"][0])
    assert np.isnan(nc["percent_under_250m"][0])
    assert np.isfinite(nc["rms_error_m"][0, 0])
    assert results[0]["aggregate_rms_error_m"] is None


@pytest.mark.extra
def test_loop_every_parameter_moves_the_geolocation(loop_run):
    """Each parameter's -/+ values move the re-geolocated subimage away from its nominal set (3k+1)."""
    _, sweep, *_, results, _nc = loop_run
    for k, param in enumerate(sweep.parameters):
        nominal = results[3 * k + 1]["geolocation"]
        for off in (3 * k, 3 * k + 2):
            moved = results[off]["geolocation"]
            shift_deg = np.hypot(moved.latitude - nominal.latitude, moved.longitude - nominal.longitude)
            assert float(shift_deg.mean()) * 111e3 > 20.0, f"{param.ptype.name} {param.config_file} set {off}"


@pytest.mark.extra
def test_loop_frames_outside_kernel_coverage_raise(loop_run):
    setup, sweep, output, sets, work, *_ = loop_run
    obs = xr.load_dataset(sets[0][1])
    obs["ugps"] = obs["ugps"] + 86_400_000_000
    late = work / "observation_next_day.nc"
    obs.to_netcdf(late)
    with pytest.raises(ValueError, match="could not be intersected with the ellipsoid"):
        loop(setup, sweep, work, [(sets[0][0], str(late), sets[0][2])], output=output)


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

        with caplog.at_level(logging.WARNING, logger="curryer.correction.verification"):
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
