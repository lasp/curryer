"""Tests for ``curryer.correction.search.im_search`` bounds and peak distinctness."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from curryer.correction.config import SearchConfig
from curryer.correction.grid_types import ImageGrid
from curryer.correction.search import im_search

LAT0, LON0 = 26.0, -102.0
STEP_DEG = 0.0009  # ~100 m


def _chip(texture: np.ndarray) -> ImageGrid:
    n_lat, n_lon = texture.shape
    lat = LAT0 + STEP_DEG * (n_lat // 2 - np.arange(n_lat))
    lon = LON0 + STEP_DEG * (np.arange(n_lon) - n_lon // 2)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return ImageGrid(data=texture, lat=lat2d, lon=lon2d)


def _subimage(chip: ImageGrid, half: int) -> ImageGrid:
    c_lat, c_lon = chip.lat.shape[0] // 2, chip.lat.shape[1] // 2
    s = (slice(c_lat - half, c_lat + half + 1), slice(c_lon - half, c_lon + half + 1))
    return ImageGrid(data=chip.data[s].copy(), lat=chip.lat[s].copy(), lon=chip.lon[s].copy())


def test_max_shift_km_bounds_the_search():
    config = SearchConfig(
        grid_size=11, grid_span_km=2.0, reduction_factor=0.5, spacing_limit_m=50.0, peak_exclusion_km=0.5
    )
    assert config.max_shift_km() == pytest.approx(5 * 0.2 / 0.5)


def test_distinct_feature_gives_secondary_well_below_peak():
    rng = np.random.default_rng(0)
    chip = _chip(rng.normal(size=(121, 121)))
    config = SearchConfig(
        grid_size=11, grid_span_km=2.0, reduction_factor=0.5, spacing_limit_m=50.0, peak_exclusion_km=0.5
    )
    sub = _subimage(chip, 30)  # 3 km inside each side, beyond max_shift_km() = 2 km
    lat_err, lon_err, ccv, ccv_secondary, *_ = im_search(chip, sub, config)
    assert abs(lat_err) < 0.06
    assert abs(lon_err) < 0.06
    assert ccv > 0.99
    assert ccv - ccv_secondary > 0.5


def test_featureless_scene_gives_secondary_close_to_peak():
    lat_wave = np.sin(np.arange(121) * 2 * np.pi / 121)[:, None] * np.ones((1, 121))  # varies only N-S
    chip = _chip(lat_wave + 1e-3 * np.random.default_rng(1).normal(size=(121, 121)))
    config = SearchConfig(
        grid_size=11, grid_span_km=2.0, reduction_factor=0.5, spacing_limit_m=50.0, peak_exclusion_km=0.5
    )
    _, _, ccv, ccv_secondary, *_ = im_search(chip, _subimage(chip, 30), config)
    assert ccv - ccv_secondary < 0.01  # an E-W shift is indistinguishable


def test_search_leaving_the_chip_raises():
    chip = _chip(np.random.default_rng(2).normal(size=(61, 61)))
    config = SearchConfig(
        grid_size=11, grid_span_km=4.0, reduction_factor=0.5, spacing_limit_m=50.0, peak_exclusion_km=0.5
    )
    with pytest.raises(ValueError, match="outside the GCP chip"):
        im_search(chip, _subimage(chip, 25), config)  # 0.5 km inside, first grid reaches 2 km


def _small_config(**changes) -> SearchConfig:
    return SearchConfig(
        **{
            "grid_size": 11,
            "grid_span_km": 2.0,
            "reduction_factor": 0.5,
            "spacing_limit_m": 50.0,
            "peak_exclusion_km": 0.5,
            **changes,
        }
    )


def test_non_finite_subimage_raises():
    chip = _chip(np.random.default_rng(3).normal(size=(121, 121)))
    sub = _subimage(chip, 30)
    sub.data[5, 7] = np.nan
    with pytest.raises(ValueError, match="1 non-finite value"):
        im_search(chip, sub, _small_config())


def test_uniform_subimage_raises():
    chip = _chip(np.random.default_rng(3).normal(size=(121, 121)))
    sub = _subimage(chip, 30)
    sub.data[:] = 7.0
    with pytest.raises(ValueError, match="Subimage data are uniform"):
        im_search(chip, sub, _small_config())


def test_search_config_is_frozen():
    config = _small_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.spacing_limit_m = 500.0


def test_search_with_no_finite_correlation_raises():
    chip = _chip(np.full((121, 121), np.nan))
    sub = _subimage(_chip(np.random.default_rng(4).normal(size=(121, 121))), 30)
    with pytest.raises(ValueError, match="No shift in the search grid has a finite correlation"):
        im_search(chip, sub, _small_config())


def test_secondary_ignores_shifts_over_missing_chip_data():
    rng = np.random.default_rng(5)
    texture = rng.normal(size=(121, 121))
    texture[:12, :] = np.nan  # reached only by the northernmost shifts of the first grid
    chip = _chip(texture)
    _, _, ccv, ccv_secondary, *_ = im_search(chip, _subimage(chip, 30), _small_config(peak_exclusion_km=0.5))
    assert np.isfinite(ccv)
    assert np.isfinite(ccv_secondary)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"grid_size": 1}, "grid_size must be an integer >= 2"),
        ({"grid_size": 11.0}, "grid_size must be an integer >= 2"),
        ({"grid_span_km": 0.0}, "grid_span_km must be finite and positive"),
        ({"spacing_limit_m": float("nan")}, "spacing_limit_m must be finite and positive"),
        ({"peak_exclusion_km": -1.0}, "peak_exclusion_km must be finite and positive"),
        ({"reduction_factor": 1.0}, r"reduction_factor must be in \(0, 1\)"),
        ({"reduction_factor": 0.0}, r"reduction_factor must be in \(0, 1\)"),
        ({"spacing_limit_m": 200.0}, "must be smaller than the first grid step"),
        ({"peak_exclusion_km": 1.0}, r"peak_exclusion_km \(1.0 km\) must be less than grid_span_km / 2"),
    ],
    ids=[
        "one-point",
        "float-size",
        "zero-span",
        "nan-limit",
        "negative-exclusion",
        "no-reduction",
        "zero-reduction",
        "search-never-runs",
        "exclusion-beyond-grid",
    ],
)
def test_invalid_search_config_raises(changes, message):
    with pytest.raises(ValueError, match=message):
        _small_config(**changes)


@pytest.mark.parametrize("value", [0.0, -1e-5, float("inf")])
def test_invalid_psf_sampling_raises(value):
    from curryer.correction.config import PSFSamplingConfig

    with pytest.raises(ValueError, match="psf_lon_sample_dist_deg must be finite and positive"):
        PSFSamplingConfig(psf_lon_sample_dist_deg=value)
