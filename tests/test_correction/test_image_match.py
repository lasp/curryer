"""Unit tests for ``curryer.correction.image_match``.

``integrated_image_match`` as a whole is tested on a synthetic scene in
``tests/integration/test_image_match_synthetic.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from curryer.compute import constants
from curryer.correction.config import SearchConfig
from curryer.correction.grid_types import ImageGrid


def test_chip_image_dataset_emulated_image_reproduces_match_correlation():
    from types import SimpleNamespace

    from test_search import _chip, _subimage

    from curryer.correction.image_match import chip_image_dataset
    from curryer.correction.search import ccv2d, emulate_image, im_search

    chip = _chip(np.random.default_rng(4).normal(size=(121, 121)))
    sub = _subimage(chip, 30)
    sub = ImageGrid(data=sub.data, lat=sub.lat + 0.004, lon=sub.lon - 0.003)  # geolocated ~0.4 km off
    config = SearchConfig(
        grid_size=11, grid_span_km=2.0, reduction_factor=0.5, spacing_limit_m=50.0, peak_exclusion_km=0.5
    )
    lat_err, lon_err, ccv, ccv_secondary, *_ = im_search(chip, sub, config)
    result = SimpleNamespace(
        lat_error_km=lat_err, lon_error_km=lon_err, ccv_final=ccv, ccv_secondary=ccv_secondary, convolved_gcp=chip
    )

    images = chip_image_dataset(sub, chip, result, config)

    assert images["observed"].dims == ("row", "col")
    assert images["reference"].dims == ("gcp_row", "gcp_col")
    np.testing.assert_array_equal(images["latitude"], sub.lat)
    assert ccv2d(images["emulated"].values, images["observed"].values) == pytest.approx(ccv, abs=1e-12)
    assert images.attrs["correlation"] == ccv

    # The saved reference is cropped to the search's reach and still holds every sample the emulated image used.
    assert images["reference"].dtype == np.float32
    assert images["reference"].size < chip.data.size
    saved = ImageGrid(
        data=images["reference_convolved"].values.astype(float),
        lat=images["reference_latitude"].values,
        lon=images["reference_longitude"].values,
    )
    radius_km = constants.WGS84_SEMI_MAJOR_AXIS_KM
    reach_deg = np.rad2deg(config.max_shift_km() / radius_km)
    assert saved.lat.min() <= sub.lat.min() - reach_deg
    assert saved.lat.max() >= sub.lat.max() + reach_deg
    mid_lat = sub.lat[sub.data.shape[0] // 2, sub.data.shape[1] // 2]
    lat_shift = np.rad2deg(lat_err / radius_km)
    lon_shift = np.rad2deg(lon_err / (radius_km * np.cos(np.deg2rad(mid_lat))))
    np.testing.assert_allclose(
        emulate_image(sub.lon - lon_shift, sub.lat - lat_shift, saved), images["emulated"].values, rtol=1e-6, atol=1e-6
    )


def test_chip_image_dataset_with_no_reachable_chip_raises():
    from types import SimpleNamespace

    from test_search import _chip, _subimage

    from curryer.correction.image_match import chip_image_dataset

    chip = _chip(np.random.default_rng(4).normal(size=(121, 121)))
    sub = _subimage(chip, 30)
    far = ImageGrid(data=sub.data, lat=sub.lat + 5.0, lon=sub.lon)
    config = SearchConfig(
        grid_size=11, grid_span_km=2.0, reduction_factor=0.5, spacing_limit_m=50.0, peak_exclusion_km=0.5
    )
    result = SimpleNamespace(lat_error_km=0.0, lon_error_km=0.0, ccv_final=0.9, ccv_secondary=0.1, convolved_gcp=chip)
    with pytest.raises(ValueError, match="No GCP chip row or column"):
        chip_image_dataset(far, chip, result, config)
