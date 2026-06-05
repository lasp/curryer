"""Tests for ``curryer.correction.kernel_ops``.

Covers:
- ``apply_offset`` – all parameter types and unit-conversion paths
- ``_load_calibration_data``
- ``_create_dynamic_kernels`` (``@pytest.mark.extra``, requires ``mkspk``)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from clarreo_config import create_clarreo_correction_config
from clarreo_data_loaders import load_clarreo_telemetry

from curryer import meta
from curryer import spicierpy as sp
from curryer.correction import correction
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

_SCI = pd.DataFrame(
    {
        "corrected_timestamp": [1_000_000.0, 2_000_000.0, 3_000_000.0, 4_000_000.0, 5_000_000.0],
        "measurement": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)


@pytest.fixture(scope="module")
def clarreo_cfg(root_dir):
    return create_clarreo_correction_config(
        root_dir / "tests" / "data" / "clarreo" / "gcs",
        root_dir / "data" / "generic",
    )


# ── apply_offset tests ────────────────────────────────────────────────────────


def test_apply_offset_kernel_arcseconds():
    """OFFSET_KERNEL converts arcseconds to radians and adds the offset."""
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.OFFSET_KERNEL,
        config_file=Path("cprs_az.json"),
        spec=dict(field="hps.az_ang_nonlin", units="arcseconds"),
    )
    original = _TLM["hps.az_ang_nonlin"].mean()
    modified = correction.apply_offset(p, 100.0, _TLM)
    assert modified["hps.az_ang_nonlin"].mean() - original == pytest.approx(np.deg2rad(100.0 / 3600.0), rel=1e-6)
    assert isinstance(modified, pd.DataFrame)


def test_apply_offset_kernel_negative():
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.OFFSET_KERNEL,
        config_file=Path("cprs_el.json"),
        spec=dict(field="hps.el_ang_nonlin", units="arcseconds"),
    )
    original = _TLM["hps.el_ang_nonlin"].mean()
    modified = correction.apply_offset(p, -50.0, _TLM)
    assert modified["hps.el_ang_nonlin"].mean() - original == pytest.approx(np.deg2rad(-50.0 / 3600.0), rel=1e-6)


def test_apply_offset_kernel_missing_field():
    """Non-existent field: returns original DataFrame unchanged."""
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.OFFSET_KERNEL,
        config_file=Path("dummy.json"),
        spec=dict(field="nonexistent_field", units="arcseconds"),
    )
    modified = correction.apply_offset(p, 10.0, _TLM)
    pd.testing.assert_frame_equal(modified, _TLM)


def test_apply_offset_time_milliseconds():
    """OFFSET_TIME: seconds input → microsecond output on timestamp column."""
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.OFFSET_TIME,
        config_file=None,
        spec=dict(field="corrected_timestamp", units="milliseconds"),
    )
    original = _SCI["corrected_timestamp"].mean()
    modified = correction.apply_offset(p, 10.0 / 1000.0, _SCI)  # 10 ms in seconds
    assert modified["corrected_timestamp"].mean() - original == pytest.approx(10_000.0, rel=1e-6)


def test_apply_offset_time_negative():
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.OFFSET_TIME,
        config_file=None,
        spec=dict(field="corrected_timestamp", units="milliseconds"),
    )
    original = _SCI["corrected_timestamp"].mean()
    modified = correction.apply_offset(p, -5.5 / 1000.0, _SCI)
    assert modified["corrected_timestamp"].mean() - original == pytest.approx(-5500.0, rel=1e-6)


def test_apply_offset_constant_kernel_passthrough():
    """CONSTANT_KERNEL: data is returned unchanged."""
    kernel_data = pd.DataFrame({"ugps": [1_000_000], "angle_x": [0.001], "angle_y": [0.002], "angle_z": [0.003]})
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.CONSTANT_KERNEL,
        config_file=Path("base.json"),
        spec=dict(field="base"),
    )
    modified = correction.apply_offset(p, kernel_data, pd.DataFrame())
    pd.testing.assert_frame_equal(modified, kernel_data)


def test_apply_offset_no_units():
    """OFFSET_KERNEL without units: offset applied in raw (radian) units."""
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.OFFSET_KERNEL,
        config_file=Path("test.json"),
        spec=dict(field="hps.az_ang_nonlin"),
    )
    original = _TLM["hps.az_ang_nonlin"].mean()
    modified = correction.apply_offset(p, 0.001, _TLM)
    assert modified["hps.az_ang_nonlin"].mean() - original == pytest.approx(0.001, rel=1e-6)


def test_apply_offset_not_inplace():
    """Original DataFrame is not mutated."""
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.OFFSET_KERNEL,
        config_file=Path("cprs_az.json"),
        spec=dict(field="hps.az_ang_nonlin", units="arcseconds"),
    )
    original = _TLM.copy()
    correction.apply_offset(p, 100.0, _TLM)
    pd.testing.assert_frame_equal(_TLM, original)


def test_apply_offset_preserves_columns():
    """All columns are present in the returned DataFrame."""
    p = correction.ParameterConfig(
        ptype=correction.ParameterType.OFFSET_KERNEL,
        config_file=Path("cprs_az.json"),
        spec=dict(field="hps.az_ang_nonlin", units="arcseconds"),
    )
    modified = correction.apply_offset(p, 50.0, _TLM)
    assert set(modified.columns) == set(_TLM.columns)
    assert modified["frame"].equals(_TLM["frame"])
    assert not modified["hps.az_ang_nonlin"].equals(_TLM["hps.az_ang_nonlin"])


# ── _load_calibration_data ────────────────────────────────────────────────────


def test_load_calibration_data_no_dir(clarreo_cfg):
    """When calibration_dir is None and no direct paths, returned data contains no vectors."""
    cfg = clarreo_cfg.model_copy(deep=True)
    cfg.calibration_dir = None
    cfg.los_vectors_file = None
    cfg.psf_file = None
    cal = correction._load_calibration_data(cfg)
    assert cal.los_vectors is None
    assert cal.optical_psfs is None


def test_load_calibration_data_direct_los_missing(clarreo_cfg, tmp_path):
    """FileNotFoundError when los_vectors_file points to a non-existent file."""
    cfg = clarreo_cfg.model_copy(deep=True)
    cfg.calibration_dir = None
    cfg.los_vectors_file = tmp_path / "nonexistent_los.mat"
    cfg.psf_file = None
    with pytest.raises(FileNotFoundError, match="LOS vectors"):
        correction._load_calibration_data(cfg)


def test_load_calibration_data_direct_psf_missing(clarreo_cfg, tmp_path):
    """FileNotFoundError when psf_file points to a non-existent file."""
    from unittest.mock import patch

    cfg = clarreo_cfg.model_copy(deep=True)
    cfg.calibration_dir = None
    # Provide a fake LOS file so the LOS loading succeeds
    fake_los = tmp_path / "los.mat"
    fake_los.touch()
    cfg.los_vectors_file = fake_los
    cfg.psf_file = tmp_path / "nonexistent_psf.mat"
    # Mock the actual loader so we don't need a real .mat file
    with patch("curryer.correction.pipeline.load_los_vectors", return_value=[[0.0, 0.0, 1.0]]):
        with pytest.raises(FileNotFoundError, match="PSF"):
            correction._load_calibration_data(cfg)


# ── _create_dynamic_kernels ───────────────────────────────────────────────────


@pytest.mark.extra
def test_create_dynamic_kernels(root_dir, clarreo_cfg, tmp_path):
    """_create_dynamic_kernels builds kernel files. Needs ``mkspk`` – ``--run-extra``."""
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
    work = tmp_path / "kernels"
    work.mkdir()
    tlm = load_clarreo_telemetry(data_dir)
    creator = create.KernelCreator(overwrite=True, append=False)
    mkrn = meta.MetaKernel.from_json(
        clarreo_cfg.geo.meta_kernel_file, relative=True, sds_dir=clarreo_cfg.geo.generic_kernel_dir
    )
    with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels]):
        dynamic_kernels = correction._create_dynamic_kernels(clarreo_cfg, work, tlm, creator)
    assert isinstance(dynamic_kernels, list)
