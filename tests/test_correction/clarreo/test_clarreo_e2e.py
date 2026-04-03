"""CLARREO end-to-end integration tests.

Exercises the upstream (kernel creation + geolocation) and downstream
(GCP pairing + image matching + error statistics) pipelines using
CLARREO test data.

Extra tests (require GMTED data or SPICE binaries): ``pytest --run-extra``
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr
from _image_match_helpers import (
    apply_error_variation_for_testing,
    discover_test_image_match_cases,
    run_image_matching_with_applied_errors,
)
from _pipeline_helpers import run_downstream_pipeline, run_upstream_pipeline
from _synthetic_helpers import (
    _generate_nadir_aligned_transforms,
    _generate_spherical_positions,
    _generate_synthetic_boresights,
    synthetic_gcp_pairing,
)
from clarreo_config import create_clarreo_correction_config

from curryer.correction.config import DataConfig

logger = logging.getLogger(__name__)


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


def test_upstream_configuration(clarreo_gcs_data_dir, clarreo_generic_dir):
    """Upstream configuration loads and validates correctly."""
    config = create_clarreo_correction_config(clarreo_gcs_data_dir, clarreo_generic_dir)
    config.data = DataConfig(file_format="csv", time_scale_factor=1e6)
    config.validate()
    assert config.geo.instrument_name == "CPRS_HYSICS"
    assert len(config.parameters) > 0
    assert config.seed == 42
    assert config.data is not None


def test_downstream_test_case_discovery(clarreo_image_match_data_dir):
    """Downstream test cases can be discovered from the data directory."""
    test_cases = discover_test_image_match_cases(clarreo_image_match_data_dir)
    assert len(test_cases) > 0
    for tc in test_cases:
        assert "case_id" in tc
        assert "subimage_file" in tc
        assert "gcp_file" in tc
        assert tc["subimage_file"].exists()
        assert tc["gcp_file"].exists()


def test_downstream_image_matching(clarreo_image_match_data_dir):
    """Downstream image matching runs successfully on test case 1."""
    test_cases = discover_test_image_match_cases(clarreo_image_match_data_dir, test_cases=["1"])
    assert len(test_cases) > 0
    result = run_image_matching_with_applied_errors(
        test_cases[0], param_idx=0, randomize_errors=False, cache_results=True
    )
    assert isinstance(result, xr.Dataset)
    assert "lat_error_km" in result.attrs
    assert "lon_error_km" in result.attrs


@pytest.mark.extra
def test_upstream_quick(clarreo_gcs_data_dir, clarreo_generic_dir, work_dir):
    """Quick upstream pipeline (2 iterations). Requires GMTED – ``--run-extra``."""
    results_list, results_dict, output_file = run_upstream_pipeline(n_iterations=2, work_dir=work_dir)
    assert results_dict["status"] == "complete"
    assert results_dict["iterations"] == 2
    assert results_dict["mode"] == "upstream"
    assert results_dict["parameter_sets"] > 0
    assert output_file.exists()


def test_downstream_quick(work_dir, clarreo_image_match_data_dir):
    """Quick downstream pipeline (2 iterations, test case 1)."""
    results_list, results_dict, output_file = run_downstream_pipeline(
        n_iterations=2, test_cases=["1"], work_dir=work_dir
    )
    assert results_dict["status"] == "complete"
    assert results_dict["iterations"] == 2
    assert output_file.exists()


def test_downstream_helpers_basic(clarreo_image_match_data_dir):
    """Downstream helper functions work correctly."""
    test_cases = discover_test_image_match_cases(clarreo_image_match_data_dir, test_cases=["1"])
    assert len(test_cases) > 0
    assert "case_id" in test_cases[0]

    base = xr.Dataset(
        {"lat_error_deg": (["m"], [0.001]), "lon_error_deg": (["m"], [0.002])},
        attrs={"lat_error_km": 0.1, "lon_error_km": 0.2, "correlation_ccv": 0.95},
    )
    varied = apply_error_variation_for_testing(base, param_idx=1, error_variation_percent=3.0)
    assert isinstance(varied, xr.Dataset)
    assert varied.attrs["lat_error_km"] != base.attrs["lat_error_km"]


def test_synthetic_helpers_basic():
    """Generic synthetic helper functions produce correctly-shaped outputs."""
    pairs = synthetic_gcp_pairing(["science_1.nc", "science_2.nc"])
    assert len(pairs) == 2

    boresights = _generate_synthetic_boresights(5, max_off_nadir_rad=0.07)
    assert boresights.shape == (5, 3)
    assert np.all(np.abs(boresights[:, 0]) < 0.01)

    positions = _generate_spherical_positions(5, 6.78e6, 4e3)
    assert positions.shape == (5, 3)
    assert np.all(np.linalg.norm(positions, axis=1) > 6.7e6)

    transforms = _generate_nadir_aligned_transforms(5, positions, boresights)
    assert transforms.shape == (5, 3, 3)
    assert abs(abs(np.linalg.det(transforms[0])) - 1.0) < 0.2
