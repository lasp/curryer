"""
Tests for pairing.py module

This module tests the GCP (Ground Control Point) pairing functionality:
- Spatial pairing of L1A science data with GCP reference imagery
- File discovery and matching
- Geographic overlap detection
- Pairing validation

Running Tests:
-------------
# Via pytest (recommended)
pytest tests/test_correction/test_pairing.py -v

# Run specific test
pytest tests/test_correction/test_pairing.py::PairingTestCase::test_find_l1a_gcp_pairs -v

# Standalone execution
python tests/test_correction/test_pairing.py

Requirements:
-----------------
These tests validate that GCP pairing correctly identifies which reference
images overlap with science data, ensuring accurate geolocation validation.
"""

from __future__ import annotations

import logging
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from curryer import utils
from curryer.correction.data_structures import NamedImageGrid
from curryer.correction.pairing import find_l1a_gcp_pairs

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.DEBUG, extra_loggers=[__name__])


L1A_FILES = [
    ("1/TestCase1a_subimage.mat", "subimage"),
    ("1/TestCase1b_subimage.mat", "subimage"),
    ("2/TestCase2a_subimage.mat", "subimage"),
    ("2/TestCase2b_subimage.mat", "subimage"),
    ("3/TestCase3a_subimage.mat", "subimage"),
    ("3/TestCase3b_subimage.mat", "subimage"),
    ("4/TestCase4a_subimage.mat", "subimage"),
    ("4/TestCase4b_subimage.mat", "subimage"),
    ("5/TestCase5a_subimage.mat", "subimage"),
    ("5/TestCase5b_subimage.mat", "subimage"),
]

GCP_FILES = [
    ("1/GCP12055Dili_resampled.mat", "GCP"),
    ("2/GCP10121Maracaibo_resampled.mat", "GCP"),
    ("3/GCP10665SantaRosa_resampled.mat", "GCP"),
    ("4/GCP20484Morocco_resampled.mat", "GCP"),
    ("5/GCP10087Titicaca_resampled.mat", "GCP"),
]

EXPECTED_MATCHES = {
    "1/TestCase1a_subimage.mat": "1/GCP12055Dili_resampled.mat",
    "1/TestCase1b_subimage.mat": "1/GCP12055Dili_resampled.mat",
    "2/TestCase2a_subimage.mat": "2/GCP10121Maracaibo_resampled.mat",
    "2/TestCase2b_subimage.mat": "2/GCP10121Maracaibo_resampled.mat",
    "3/TestCase3a_subimage.mat": "3/GCP10665SantaRosa_resampled.mat",
    "3/TestCase3b_subimage.mat": "3/GCP10665SantaRosa_resampled.mat",
    "4/TestCase4a_subimage.mat": "4/GCP20484Morocco_resampled.mat",
    "4/TestCase4b_subimage.mat": "4/GCP20484Morocco_resampled.mat",
    "5/TestCase5a_subimage.mat": "5/GCP10087Titicaca_resampled.mat",
    "5/TestCase5b_subimage.mat": "5/GCP10087Titicaca_resampled.mat",
}


class PairingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parent.parent.parent
        self.test_dir = root_dir / "tests" / "data" / "clarreo" / "image_match"
        self.assertTrue(self.test_dir.is_dir(), self.test_dir)

    def _load_image_grid(self, relative_path: str, key: str) -> NamedImageGrid:
        mat_path = self.test_dir / relative_path
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)[key]
        h = getattr(mat, "h", None)
        return NamedImageGrid(
            data=np.asarray(mat.data),
            lat=np.asarray(mat.lat),
            lon=np.asarray(mat.lon),
            h=np.asarray(h) if h is not None else None,
            name=relative_path,
        )

    def _prepare_inputs(self, file_list):
        for rel_path, key in file_list:
            yield self._load_image_grid(rel_path, key)

    def test_find_l1a_gcp_pairs(self):
        l1a_inputs = list(self._prepare_inputs(L1A_FILES))
        gcp_inputs = list(self._prepare_inputs(GCP_FILES))

        result = find_l1a_gcp_pairs(l1a_inputs, gcp_inputs, max_distance_m=0.0)

        matches_by_l1a = {}
        for match in result.matches:
            l1a_name = result.l1a_images[match.l1a_index].name
            gcp_name = result.gcp_images[match.gcp_index].name
            distance = match.distance_m
            if l1a_name not in matches_by_l1a or distance < matches_by_l1a[l1a_name][1]:
                matches_by_l1a[l1a_name] = (gcp_name, distance)

        assert set(matches_by_l1a.keys()) == set(expected for expected in EXPECTED_MATCHES)

        for l1a_name, expected_gcp in EXPECTED_MATCHES.items():
            assert l1a_name in matches_by_l1a, f"Missing match for {l1a_name}"
            gcp_name, distance = matches_by_l1a[l1a_name]
            assert gcp_name == expected_gcp, f"Unexpected match for {l1a_name}: {gcp_name}"
            assert distance >= 0.0, f"Expected non-negative margin for {l1a_name}: {distance}"

    @staticmethod
    def _make_rect_image(name: str, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> NamedImageGrid:
        lat = np.array([[lat_max, lat_max], [lat_min, lat_min]], dtype=float)
        lon = np.array([[lon_min, lon_max], [lon_min, lon_max]], dtype=float)
        data = np.zeros_like(lat)
        return NamedImageGrid(data=data, lat=lat, lon=lon, name=name)

    @staticmethod
    def _make_point_gcp(name: str, lon: float, lat: float) -> NamedImageGrid:
        data = np.array([[1.0]])
        lat_arr = np.array([[lat]])
        lon_arr = np.array([[lon]])
        return NamedImageGrid(data=data, lat=lat_arr, lon=lon_arr, name=name)

    def test_synthetic_pairing_no_overlap(self):
        l1a = [self._make_rect_image("L1A", lon_min=1.0, lon_max=2.0, lat_min=-1.0, lat_max=1.0)]
        gcp = [self._make_point_gcp("GCP", lon=0.0, lat=0.0)]

        result = find_l1a_gcp_pairs(l1a, gcp, max_distance_m=100_000.0)
        assert result.matches == []

    def test_synthetic_pairing_partial_less_than_threshold(self):
        l1a = [self._make_rect_image("L1A", lon_min=0.0, lon_max=1.0, lat_min=-1.0, lat_max=1.0)]
        gcp = [self._make_point_gcp("GCP", lon=0.0, lat=0.0)]

        result = find_l1a_gcp_pairs(l1a, gcp, max_distance_m=100_000.0)
        assert result.matches == []

    def test_synthetic_pairing_complete_above_threshold(self):
        l1a = [self._make_rect_image("L1A", lon_min=-1.0, lon_max=1.0, lat_min=-1.0, lat_max=1.0)]
        gcp = [self._make_point_gcp("GCP", lon=0.0, lat=0.0)]

        result = find_l1a_gcp_pairs(l1a, gcp, max_distance_m=100_000.0)
        assert len(result.matches) == 1
        match = result.matches[0]
        assert match.l1a_index == 0
        assert match.gcp_index == 0
        assert match.distance_m >= 100_000.0

    def test_synthetic_pairing_partial_threshold_not_met(self):
        l1a = [self._make_rect_image("L1A", lon_min=-1.0, lon_max=1.0, lat_min=-1.0, lat_max=1.0)]
        gcp = [self._make_point_gcp("GCP", lon=0.0, lat=0.0)]

        result = find_l1a_gcp_pairs(l1a, gcp, max_distance_m=200_000.0)
        assert result.matches == []


if __name__ == "__main__":
    unittest.main()
