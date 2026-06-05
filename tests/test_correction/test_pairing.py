"""Tests for ``curryer.correction.pairing`` (GCP spatial pairing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from curryer.correction.grid_types import NamedImageGrid
from curryer.correction.pairing import find_l1a_gcp_pairs

# ── test-case metadata ────────────────────────────────────────────────────────

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

# ── fixtures / helpers ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def image_match_dir():
    root = Path(__file__).parent.parent.parent
    d = root / "tests" / "data" / "clarreo" / "image_match"
    assert d.is_dir(), str(d)
    return d


def _load(path: Path, key: str, name: str) -> NamedImageGrid:
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)[key]
    h = getattr(mat, "h", None)
    return NamedImageGrid(
        data=np.asarray(mat.data),
        lat=np.asarray(mat.lat),
        lon=np.asarray(mat.lon),
        h=np.asarray(h) if h is not None else None,
        name=name,
    )


def _rect(name, lon_min, lon_max, lat_min, lat_max) -> NamedImageGrid:
    lat = np.array([[lat_max, lat_max], [lat_min, lat_min]], dtype=float)
    lon = np.array([[lon_min, lon_max], [lon_min, lon_max]], dtype=float)
    return NamedImageGrid(data=np.zeros_like(lat), lat=lat, lon=lon, name=name)


def _point(name, lon, lat) -> NamedImageGrid:
    return NamedImageGrid(data=np.array([[1.0]]), lat=np.array([[lat]]), lon=np.array([[lon]]), name=name)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_find_l1a_gcp_pairs(image_match_dir):
    l1a = [_load(image_match_dir / rel, key, rel) for rel, key in L1A_FILES]
    gcp = [_load(image_match_dir / rel, key, rel) for rel, key in GCP_FILES]
    result = find_l1a_gcp_pairs(l1a, gcp, max_distance_m=0.0)
    by_l1a = {}
    for m in result.matches:
        name = result.l1a_images[m.l1a_index].name
        gname = result.gcp_images[m.gcp_index].name
        if name not in by_l1a or m.distance_m < by_l1a[name][1]:
            by_l1a[name] = (gname, m.distance_m)
    assert set(by_l1a) == set(EXPECTED_MATCHES)
    for l1a_name, expected_gcp in EXPECTED_MATCHES.items():
        gname, dist = by_l1a[l1a_name]
        assert gname == expected_gcp
        assert dist >= 0.0


def test_synthetic_pairing_no_overlap():
    result = find_l1a_gcp_pairs(
        [_rect("L1A", 1.0, 2.0, -1.0, 1.0)], [_point("GCP", 0.0, 0.0)], max_distance_m=100_000.0
    )
    assert result.matches == []


def test_synthetic_pairing_partial_less_than_threshold():
    result = find_l1a_gcp_pairs(
        [_rect("L1A", 0.0, 1.0, -1.0, 1.0)], [_point("GCP", 0.0, 0.0)], max_distance_m=100_000.0
    )
    assert result.matches == []


def test_synthetic_pairing_complete_above_threshold():
    result = find_l1a_gcp_pairs(
        [_rect("L1A", -1.0, 1.0, -1.0, 1.0)], [_point("GCP", 0.0, 0.0)], max_distance_m=100_000.0
    )
    assert len(result.matches) == 1
    m = result.matches[0]
    assert m.l1a_index == 0
    assert m.gcp_index == 0
    assert m.distance_m >= 100_000.0


def test_synthetic_pairing_partial_threshold_not_met():
    result = find_l1a_gcp_pairs(
        [_rect("L1A", -1.0, 1.0, -1.0, 1.0)], [_point("GCP", 0.0, 0.0)], max_distance_m=200_000.0
    )
    assert result.matches == []
