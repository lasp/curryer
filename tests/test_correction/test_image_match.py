"""Unit tests for ``curryer.correction.image_match``.

``integrated_image_match`` as a whole is tested on a synthetic scene in
``tests/integration/test_image_match_synthetic.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from curryer.correction.config import SearchConfig
from curryer.correction.grid_types import ImageGrid


def test_chip_image_dataset_emulated_image_reproduces_match_correlation():
    from types import SimpleNamespace

    from test_search import _chip, _subimage

    from curryer.correction.image_match import chip_image_dataset
    from curryer.correction.search import ccv2d, im_search

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

    images = chip_image_dataset(sub, chip, result)

    assert images["observed"].dims == ("row", "col")
    assert images["reference"].dims == ("gcp_row", "gcp_col")
    np.testing.assert_array_equal(images["latitude"], sub.lat)
    assert ccv2d(images["emulated"].values, images["observed"].values) == pytest.approx(ccv, abs=1e-12)
    assert images.attrs["correlation"] == ccv
