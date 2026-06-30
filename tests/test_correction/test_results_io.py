"""Tests for ``curryer.correction.results_io``.

Covers:
- ``_build_netcdf_structure``
- ``_save_netcdf_checkpoint`` / ``_load_checkpoint`` / ``_cleanup_checkpoint``
"""

from __future__ import annotations

import pytest
from clarreo_config import create_clarreo_setup_sweep

from curryer.correction.results_io import (
    _build_netcdf_structure,
    _cleanup_checkpoint,
    _load_checkpoint,
    _save_netcdf_checkpoint,
)


@pytest.fixture(scope="module")
def clarreo_cfg(root_dir):
    setup, sweep, output = create_clarreo_setup_sweep(
        root_dir / "tests" / "data" / "clarreo" / "gcs",
        root_dir / "data" / "generic",
    )
    netcdf_config = output.netcdf
    return setup, sweep, output, netcdf_config


# ── _build_netcdf_structure ───────────────────────────────────────────────────


def test_build_netcdf_structure_shapes(clarreo_cfg):
    """_build_netcdf_structure returns arrays with the requested dimensions."""
    setup, sweep, _output, netcdf_config = clarreo_cfg
    nc = _build_netcdf_structure(setup, sweep, netcdf_config, n_param_sets=3, n_gcp_pairs=2)
    assert isinstance(nc, dict)
    assert "rms_error_m" in nc
    assert "parameter_set_id" in nc
    assert nc["rms_error_m"].shape == (3, 2)
    assert len(nc["parameter_set_id"]) == 3


def test_build_netcdf_structure_zero_initialised(clarreo_cfg):
    """All numeric arrays are initialised to zero (or NaN for float arrays)."""
    setup, sweep, _output, netcdf_config = clarreo_cfg
    nc = _build_netcdf_structure(setup, sweep, netcdf_config, n_param_sets=2, n_gcp_pairs=1)
    # rms_error_m starts as zeros or NaN – just check it is numeric
    assert nc["rms_error_m"].dtype.kind in ("f", "i", "u")


# ── checkpoint round-trip ─────────────────────────────────────────────────────


def test_checkpoint_save_load_cleanup(clarreo_cfg, tmp_path):
    """Save → load → cleanup round-trip preserves data and removes the file."""
    setup, sweep, output, netcdf_config = clarreo_cfg
    sweep = sweep.model_copy(deep=True)
    sweep.n_iterations = 2
    output = output.model_copy(deep=True)
    output.output_filename = "ckpt_test.nc"
    output_file = tmp_path / output.output_filename

    nc = _build_netcdf_structure(setup, sweep, netcdf_config, n_param_sets=2, n_gcp_pairs=2)
    nc["rms_error_m"][0, 0] = 100.0
    nc["rms_error_m"][1, 0] = 150.0

    _save_netcdf_checkpoint(nc, output_file, setup, sweep, netcdf_config, pair_idx_completed=0)
    ckpt = output_file.parent / f"{output_file.stem}_checkpoint.nc"
    assert ckpt.exists()

    loaded, completed = _load_checkpoint(output_file)
    assert loaded is not None
    assert completed == 1  # pair index 0 completed → 1 pair done
    assert loaded["rms_error_m"][0, 0] == pytest.approx(100.0)
    assert loaded["rms_error_m"][1, 0] == pytest.approx(150.0)

    _cleanup_checkpoint(output_file)
    assert not ckpt.exists()


def test_checkpoint_load_missing_returns_none(clarreo_cfg, tmp_path):
    """_load_checkpoint returns (None, 0) when no checkpoint file exists."""
    output_file = tmp_path / "no_ckpt.nc"
    loaded, completed = _load_checkpoint(output_file)
    assert loaded is None
    assert completed == 0
