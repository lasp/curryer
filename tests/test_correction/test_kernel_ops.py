"""Tests for ``curryer.correction.kernel_ops``.

Covers:
- ``apply_offset`` – all parameter types and unit-conversion paths
- ``_create_parameter_kernels`` – one CK per CONSTANT_KERNEL frame
- ``_load_calibration_data``
- ``_create_dynamic_kernels`` (``@pytest.mark.extra``, requires ``mkspk``)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from clarreo_config import create_clarreo_setup_sweep
from clarreo_data_loaders import load_clarreo_telemetry

from curryer import meta
from curryer import spicierpy as sp
from curryer.correction.config import CalibrationFiles, ParameterConfig, ParameterType
from curryer.correction.kernel_ops import (
    _UGPS_EPOCH_END,
    _create_dynamic_kernels,
    _create_parameter_kernels,
    apply_offset,
)
from curryer.correction.pipeline import _load_calibration_data
from curryer.kernels import create

# ── shared sample data ────────────────────────────────────────────────────────

_TLM = pd.DataFrame(
    {
        "frame": range(5),
        "hps.az_ang_nonlin": [1.14252] * 5,
        "hps.el_ang_nonlin": [-0.55009] * 5,
        "hps.resolver_tms": [1168477154.0 + i for i in range(5)],
        "ert": [1431903180.58 + i for i in range(5)],
    }
)

_FRAME_UGPS = np.array([1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000], dtype=np.int64)


@pytest.fixture(scope="module")
def clarreo_cfg(root_dir):
    setup, sweep, output = create_clarreo_setup_sweep(
        root_dir / "tests" / "data" / "clarreo" / "gcs",
        root_dir / "data" / "generic",
    )
    return setup, sweep, output


# ── apply_offset tests ────────────────────────────────────────────────────────


def test_apply_offset_kernel_adds_radians():
    """OFFSET_KERNEL adds the value as given: load_param_sets has already converted it to radians."""
    p = ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("cprs_az.json"),
        spec=dict(field="hps.az_ang_nonlin", units="arcseconds"),
    )
    original = _TLM["hps.az_ang_nonlin"].mean()
    offset_rad = np.deg2rad(100.0 / 3600.0)
    modified = apply_offset(p, offset_rad, _TLM)
    assert modified["hps.az_ang_nonlin"].mean() - original == pytest.approx(offset_rad, rel=1e-9)
    assert isinstance(modified, pd.DataFrame)


def test_apply_offset_kernel_negative():
    p = ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("cprs_el.json"),
        spec=dict(field="hps.el_ang_nonlin", units="arcseconds"),
    )
    original = _TLM["hps.el_ang_nonlin"].mean()
    modified = apply_offset(p, -2.5e-4, _TLM)
    assert modified["hps.el_ang_nonlin"].mean() - original == pytest.approx(-2.5e-4, rel=1e-9)


def test_apply_offset_kernel_missing_field_raises():
    """A field that is not a telemetry column raises rather than writing an unperturbed kernel."""
    p = ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("dummy.json"),
        spec=dict(field="nonexistent_field", units="arcseconds"),
    )
    with pytest.raises(KeyError, match="nonexistent_field"):
        apply_offset(p, 10.0, _TLM)


def test_apply_offset_time_shifts_frame_times():
    """OFFSET_TIME: seconds added to the uGPS frame times."""
    p = ParameterConfig(ptype=ParameterType.OFFSET_TIME, spec=dict(field="corrected_timestamp", units="milliseconds"))
    modified = apply_offset(p, 10.0 / 1000.0, _FRAME_UGPS)
    np.testing.assert_allclose(modified - _FRAME_UGPS, 10_000.0)


def test_apply_offset_time_negative():
    p = ParameterConfig(ptype=ParameterType.OFFSET_TIME, spec=dict(field="corrected_timestamp", units="milliseconds"))
    modified = apply_offset(p, -5.5 / 1000.0, _FRAME_UGPS)
    np.testing.assert_allclose(modified - _FRAME_UGPS, -5500.0)


def test_apply_offset_time_on_dataframe_raises():
    p = ParameterConfig(ptype=ParameterType.OFFSET_TIME, spec=dict(field="corrected_timestamp"))
    with pytest.raises(TypeError, match="ndarray of uGPS frame times"):
        apply_offset(p, 0.01, pd.DataFrame({"corrected_timestamp": _FRAME_UGPS}))


def test_apply_offset_constant_kernel_raises():
    """CONSTANT_KERNEL angles are written to a kernel, not applied to data."""
    p = ParameterConfig(ptype=ParameterType.CONSTANT_KERNEL, config_file=Path("base.json"), spec=dict(field="angle_x"))
    with pytest.raises(NotImplementedError):
        apply_offset(p, 0.001, _TLM)


# ── _create_parameter_kernels tests ───────────────────────────────────────────


def _axis_params(config_file: str) -> list[ParameterConfig]:
    return [
        ParameterConfig(ptype=ParameterType.CONSTANT_KERNEL, config_file=Path(config_file), spec=dict(field=axis))
        for axis in ("angle_z", "angle_x", "angle_y")
    ]


def test_create_parameter_kernels_writes_one_ck_per_frame(tmp_path):
    """A frame's three axis parameters become one two-row CK spanning the mission."""
    creator = MagicMock()
    creator.write_from_json.side_effect = lambda config_file, **kwargs: tmp_path / f"{config_file.stem}.bc"
    time = ParameterConfig(ptype=ParameterType.OFFSET_TIME, spec=dict(field="t"))
    params = [
        *zip(_axis_params("base.json"), [3.0, 1.0, 2.0]),
        (time, 0.5),
        *zip(_axis_params("yoke.json"), [6.0, 4.0, 5.0]),
    ]

    kernels, frame_ugps = _create_parameter_kernels(params, tmp_path, _TLM, _FRAME_UGPS, creator)

    assert kernels == [tmp_path / "base.bc", tmp_path / "yoke.bc"]
    np.testing.assert_array_equal(frame_ugps, _FRAME_UGPS + 500_000)
    for call, angles in zip(creator.write_from_json.call_args_list, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]):
        expected = pd.DataFrame(
            {"ugps": [0, _UGPS_EPOCH_END], **{a: [v, v] for a, v in zip(("angle_x", "angle_y", "angle_z"), angles)}}
        )
        pd.testing.assert_frame_equal(call.kwargs["input_data"], expected)
        assert call.kwargs["overrides"] == {"input_gap_threshold": None}


def test_apply_offset_no_units():
    """OFFSET_KERNEL without units: offset applied in raw (radian) units."""
    p = ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("test.json"),
        spec=dict(field="hps.az_ang_nonlin"),
    )
    original = _TLM["hps.az_ang_nonlin"].mean()
    modified = apply_offset(p, 0.001, _TLM)
    assert modified["hps.az_ang_nonlin"].mean() - original == pytest.approx(0.001, rel=1e-6)


def test_apply_offset_not_inplace():
    """Original DataFrame is not mutated."""
    p = ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("cprs_az.json"),
        spec=dict(field="hps.az_ang_nonlin", units="arcseconds"),
    )
    original = _TLM.copy()
    apply_offset(p, 1e-3, _TLM)
    pd.testing.assert_frame_equal(_TLM, original)


def test_apply_offset_preserves_columns():
    """All columns are present in the returned DataFrame."""
    p = ParameterConfig(
        ptype=ParameterType.OFFSET_KERNEL,
        config_file=Path("cprs_az.json"),
        spec=dict(field="hps.az_ang_nonlin", units="arcseconds"),
    )
    modified = apply_offset(p, 1e-3, _TLM)
    assert set(modified.columns) == set(_TLM.columns)
    assert modified["frame"].equals(_TLM["frame"])
    assert not modified["hps.az_ang_nonlin"].equals(_TLM["hps.az_ang_nonlin"])


# ── _load_calibration_data ────────────────────────────────────────────────────


def test_load_calibration_data_no_dir(clarreo_cfg):
    """When no direct calibration paths are set, returned data contains no vectors."""
    setup, _sweep, _output = clarreo_cfg
    setup = setup.model_copy(deep=True)
    setup.calibration = CalibrationFiles(los_vectors_file=None, psf_file=None)
    cal = _load_calibration_data(setup)
    assert cal.los_vectors is None
    assert cal.optical_psfs is None


def test_load_calibration_data_direct_los_missing(clarreo_cfg, tmp_path):
    """FileNotFoundError when los_vectors_file points to a non-existent file."""
    setup, _sweep, _output = clarreo_cfg
    setup = setup.model_copy(deep=True)
    setup.calibration = CalibrationFiles(los_vectors_file=tmp_path / "nonexistent_los.mat", psf_file=None)
    with pytest.raises(FileNotFoundError, match="LOS vectors"):
        _load_calibration_data(setup)


def test_load_calibration_data_direct_psf_missing(clarreo_cfg, tmp_path):
    """FileNotFoundError when psf_file points to a non-existent file."""
    from unittest.mock import patch

    setup, _sweep, _output = clarreo_cfg
    setup = setup.model_copy(deep=True)
    # Provide a fake LOS file so the LOS loading succeeds
    fake_los = tmp_path / "los.mat"
    fake_los.touch()
    setup.calibration = CalibrationFiles(los_vectors_file=fake_los, psf_file=tmp_path / "nonexistent_psf.mat")
    # Mock the actual loader so we don't need a real .mat file
    with patch("curryer.correction.pipeline.load_los_vectors", return_value=[[0.0, 0.0, 1.0]]):
        with pytest.raises(FileNotFoundError, match="PSF"):
            _load_calibration_data(setup)


# ── _create_dynamic_kernels ───────────────────────────────────────────────────


@pytest.mark.extra
def test_create_dynamic_kernels(root_dir, clarreo_cfg, tmp_path):
    """_create_dynamic_kernels builds kernel files. Needs ``mkspk`` – ``--run-extra``."""
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
    work = tmp_path / "kernels"
    work.mkdir()
    setup, _sweep, _output = clarreo_cfg
    tlm = load_clarreo_telemetry(data_dir)
    creator = create.KernelCreator(overwrite=True, append=False)
    mkrn = meta.MetaKernel.from_json(setup.geo.meta_kernel_file, relative=True, sds_dir=setup.geo.generic_kernel_dir)
    with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels]):
        dynamic_kernels = _create_dynamic_kernels(setup, work, tlm, creator)
    assert isinstance(dynamic_kernels, list)
