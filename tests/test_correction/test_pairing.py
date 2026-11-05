"""Tests for ``curryer.correction.pairing`` (GCP spatial pairing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.io import loadmat

from curryer.compute.footprint import EARTH_MEAN_RADIUS_KM, footprint_margin_m
from curryer.correction.grid_types import NamedImageGrid
from curryer.correction.pairing import find_l1a_gcp_pairs, pair_files, pair_geolocated_dataset_with_gcp_files

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


def _rect(name, lon_min, lon_max, lat_min, lat_max, n=201) -> NamedImageGrid:
    """Dense grid of pixel centers filling the box, so its footprint is the box to within a pixel."""
    lat, lon = np.meshgrid(np.linspace(lat_max, lat_min, n), np.linspace(lon_min, lon_max, n), indexing="ij")
    return NamedImageGrid(data=np.zeros_like(lat), lat=lat, lon=lon, name=name)


def _curved_swath(n_frames=3000, n_pixels=60, pixel_km=0.46, drift_deg_per_frame=0.001):
    """Swath along a great circle from (30 N, 100 W) heading 50 deg, each frame rotated about the polar axis
    by ``drift_deg_per_frame`` times its index, which curves the swath as Earth rotation curves a ground track."""
    lat0, lon0, az = np.radians(30.0), np.radians(-100.0), np.radians(50.0)
    start = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    east = np.cross([0.0, 0.0, 1.0], start)
    east /= np.linalg.norm(east)
    north = np.cross(start, east)
    heading = np.cos(az) * north + np.sin(az) * east
    s = np.arange(n_frames)[:, None, None] * pixel_km / EARTH_MEAN_RADIUS_KM
    t = (np.arange(n_pixels) - (n_pixels - 1) / 2)[None, :, None] * pixel_km / EARTH_MEAN_RADIUS_KM
    xyz = np.cos(t) * (np.cos(s) * start + np.sin(s) * heading) + np.sin(t) * np.cross(start, heading)
    rot = np.radians(drift_deg_per_frame) * np.arange(n_frames)[:, None]
    x = xyz[..., 0] * np.cos(rot) - xyz[..., 1] * np.sin(rot)
    y = xyz[..., 0] * np.sin(rot) + xyz[..., 1] * np.cos(rot)
    return np.degrees(np.arcsin(xyz[..., 2])), np.degrees(np.arctan2(y, x))


def _beyond(lat, lon, inner, outer, distance_km):
    """(lat, lon) ``distance_km`` past pixel ``outer``, continuing the great circle from pixel ``inner``."""

    def unit(i):
        la, lo = np.radians(lat[i]), np.radians(lon[i])
        return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

    p0, p1 = unit(outer), unit(inner)
    direction = p0 - p1 - ((p0 - p1) @ p0) * p0
    direction /= np.linalg.norm(direction)
    d = distance_km / EARTH_MEAN_RADIUS_KM
    q = np.cos(d) * p0 + np.sin(d) * direction
    return float(np.degrees(np.arcsin(q[2]))), float(np.degrees(np.arctan2(q[1], q[0])))


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


# ── pair_files ────────────────────────────────────────────────────────────────


@pytest.fixture
def maracaibo_gcp_dir(image_match_dir, tmp_path):
    gcp_dir = tmp_path / "gcp"
    gcp_dir.mkdir()
    (gcp_dir / "GCP10121Maracaibo_resampled.mat").symlink_to(image_match_dir / "2/GCP10121Maracaibo_resampled.mat")
    return gcp_dir


def test_pair_files_returns_the_overlapping_observation(image_match_dir, maracaibo_gcp_dir):
    dili = image_match_dir / "1/TestCase1a_subimage.mat"
    maracaibo = image_match_dir / "2/TestCase2a_subimage.mat"
    pairs = pair_files([dili, maracaibo], maracaibo_gcp_dir)
    assert pairs == [(maracaibo, maracaibo_gcp_dir / "GCP10121Maracaibo_resampled.mat")]


def test_pair_files_with_no_pairs_says_to_lower_max_distance(image_match_dir, maracaibo_gcp_dir, caplog):
    with caplog.at_level("WARNING", logger="curryer.correction.pairing"):
        pairs = pair_files([image_match_dir / "1/TestCase1a_subimage.mat"], maracaibo_gcp_dir)
    assert pairs == []
    assert "lower it (negative allows centers outside)" in caplog.text


def test_pair_files_raises_on_unreadable_l1a(image_match_dir, maracaibo_gcp_dir, tmp_path):
    bad = tmp_path / "bad_subimage.mat"
    bad.write_bytes(b"not a MATLAB file")
    maracaibo = image_match_dir / "2/TestCase2a_subimage.mat"
    with pytest.raises(ValueError, match="Failed to load L1A file .*bad_subimage.mat"):
        pair_files([bad, maracaibo], maracaibo_gcp_dir)


def test_pair_files_raises_on_unreadable_gcp(image_match_dir, maracaibo_gcp_dir):
    (maracaibo_gcp_dir / "GCP00000Bad_resampled.mat").write_bytes(b"not a MATLAB file")
    maracaibo = image_match_dir / "2/TestCase2a_subimage.mat"
    with pytest.raises(ValueError, match="Failed to load GCP file .*GCP00000Bad_resampled.mat"):
        pair_files([maracaibo], maracaibo_gcp_dir)


def test_pair_files_raises_on_empty_l1a_list(maracaibo_gcp_dir):
    with pytest.raises(ValueError, match="No L1A files provided"):
        pair_files([], maracaibo_gcp_dir)


def test_pair_files_missing_l1a_chains_file_not_found(maracaibo_gcp_dir, tmp_path):
    with pytest.raises(ValueError, match="Failed to load L1A file .*missing_subimage.mat") as excinfo:
        pair_files([tmp_path / "missing_subimage.mat"], maracaibo_gcp_dir)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


# ── pair_geolocated_dataset_with_gcp_files ────────────────────────────────────


def _geolocated_over(grid: NamedImageGrid) -> xr.Dataset:
    return xr.Dataset(
        {
            "latitude": (["frame", "pixel"], grid.lat),
            "longitude": (["frame", "pixel"], grid.lon),
        },
        coords={"frame": np.arange(grid.lat.shape[0], dtype=float)},
    )


def test_pair_geolocated_returns_overlapping_chip(image_match_dir):
    maracaibo = _load(image_match_dir / "2/TestCase2a_subimage.mat", "subimage", "obs")
    chips = [image_match_dir / "1/GCP12055Dili_resampled.mat", image_match_dir / "2/GCP10121Maracaibo_resampled.mat"]
    matched = pair_geolocated_dataset_with_gcp_files(_geolocated_over(maracaibo), chips)
    assert matched == [chips[1]]


def test_pair_geolocated_raises_on_unreadable_chip(image_match_dir, tmp_path):
    bad = tmp_path / "GCP00000Bad_resampled.mat"
    bad.write_bytes(b"not a MATLAB file")
    maracaibo = _load(image_match_dir / "2/TestCase2a_subimage.mat", "subimage", "obs")
    chips = [bad, image_match_dir / "2/GCP10121Maracaibo_resampled.mat"]
    with pytest.raises(ValueError, match="Failed to load GCP file .*GCP00000Bad_resampled.mat"):
        pair_geolocated_dataset_with_gcp_files(_geolocated_over(maracaibo), chips)


# ── footprint ─────────────────────────────────────────────────────────────────


def test_curved_swath_pairs_by_its_traced_edge():
    lat, lon = _curved_swath()
    mid = lat.shape[0] // 2
    # Two pixels in from each edge of the middle frame (inside), and 3 km beyond each edge (outside).
    inside = [(float(lat[mid, 2]), float(lon[mid, 2])), (float(lat[mid, -3]), float(lon[mid, -3]))]
    outside = [_beyond(lat, lon, (mid, 1), (mid, 0), 3.0), _beyond(lat, lon, (mid, -2), (mid, -1), 3.0)]
    points = inside + outside

    # The swath bows about 11 km over its 1380 km, as a ground track does over a several-minute granule.
    # The polygon through the four corner pixels gets both sides wrong: it cuts inside the swath on the
    # concave side and reaches past it on the convex side.
    corners = np.array([[lon[i, j], lat[i, j]] for i, j in ((0, 0), (-1, 0), (-1, -1), (0, -1), (0, 0))])
    corner_margin = footprint_margin_m(corners, np.array([p[0] for p in points]), np.array([p[1] for p in points]))
    assert (corner_margin[:2] < 0).sum() == 1
    assert (corner_margin[2:] >= 0).sum() == 1

    gcps = [_point(f"GCP{k}", p_lon, p_lat) for k, (p_lat, p_lon) in enumerate(points)]
    swath = NamedImageGrid(data=np.zeros_like(lat), lat=lat, lon=lon, name="L1A")
    result = find_l1a_gcp_pairs([swath], gcps, max_distance_m=0.0)
    assert sorted(m.gcp_index for m in result.matches) == [0, 1]
    assert result.l1a_images[0].footprint.shape[1] == 2


def test_footprint_tolerance_is_passed_through():
    l1a = _rect("L1A", -1.0, 1.0, -1.0, 1.0)
    default = find_l1a_gcp_pairs([l1a], [_point("GCP", 0.0, 0.0)], max_distance_m=0.0)
    coarse = find_l1a_gcp_pairs([l1a], [_point("GCP", 0.0, 0.0)], max_distance_m=0.0, footprint_tolerance_km=5.0)
    assert len(default.l1a_images[0].footprint) >= 5
    assert not np.array_equal(default.l1a_images[0].footprint, coarse.l1a_images[0].footprint)


def test_l1a_with_non_nan_fill_raises_naming_the_image():
    l1a = _rect("granule_7", -1.0, 1.0, -1.0, 1.0)
    l1a.lat[0, 0] = -9999.0
    with pytest.raises(ValueError, match="Footprint of L1A image 'granule_7': latitude values must be within"):
        find_l1a_gcp_pairs([l1a], [_point("GCP", 0.0, 0.0)], max_distance_m=0.0)


def test_nan_fill_at_the_edge_is_excluded_from_the_footprint():
    l1a = _rect("L1A", -1.0, 1.0, -1.0, 1.0)
    l1a.lat[:, :100] = np.nan
    l1a.lon[:, :100] = np.nan
    result = find_l1a_gcp_pairs([l1a], [_point("west", -0.5, 0.0), _point("east", 0.5, 0.0)], max_distance_m=0.0)
    assert [result.gcp_images[m.gcp_index].name for m in result.matches] == ["east"]


def test_no_gcp_images_gives_no_matches():
    result = find_l1a_gcp_pairs([_rect("L1A", -1.0, 1.0, -1.0, 1.0)], [], max_distance_m=0.0)
    assert result.matches == []
    assert len(result.l1a_images) == 1


def test_l1a_that_is_not_2d_raises_naming_the_image():
    l1a = NamedImageGrid(data=np.zeros(5), lat=np.linspace(0, 1, 5), lon=np.linspace(0, 1, 5), name="granule_1d")
    with pytest.raises(ValueError, match="Footprint of L1A image 'granule_1d': latitude must be a 2-D numpy array"):
        find_l1a_gcp_pairs([l1a], [_point("GCP", 0.0, 0.0)], max_distance_m=0.0)


def test_invalid_footprint_tolerance_raises_naming_the_image():
    with pytest.raises(ValueError, match="Footprint of L1A image 'L1A': tolerance_km must be"):
        find_l1a_gcp_pairs(
            [_rect("L1A", -1.0, 1.0, -1.0, 1.0)],
            [_point("GCP", 0.0, 0.0)],
            max_distance_m=0.0,
            footprint_tolerance_km=0.0,
        )


def test_untraceable_footprint_falls_back_with_a_warning_naming_the_image(caplog):
    folded = _rect("folded", -1.0, 1.0, -1.0, 1.0)
    folded.lon[-50:] = folded.lon[-50:, ::-1].copy()
    with caplog.at_level("WARNING", logger="curryer.correction.pairing"):
        result = find_l1a_gcp_pairs([folded], [_point("GCP", 0.0, 0.0)], max_distance_m=0.0)
    assert "'folded'" in caplog.text
    assert "convex hull" in caplog.text
    assert len(result.matches) == 1


def test_single_row_l1a_raises_naming_the_image():
    row = _rect("one_row", -1.0, 1.0, -1.0, 1.0)
    row = NamedImageGrid(data=row.data[:1], lat=row.lat[:1].copy(), lon=row.lon[:1].copy(), name="one_row")
    with pytest.raises(ValueError, match="Footprint of L1A image 'one_row': no two adjacent valid pixels along grid"):
        find_l1a_gcp_pairs([row], [_point("GCP", 0.0, 1.0)], max_distance_m=0.0)


@pytest.mark.parametrize("max_distance_m", [float("nan"), float("inf"), True, "0", np.float32(0.0)])
def test_invalid_max_distance_raises(max_distance_m):
    with pytest.raises(ValueError, match="max_distance_m must be a finite float or int"):
        find_l1a_gcp_pairs([_rect("L1A", -1.0, 1.0, -1.0, 1.0)], [_point("GCP", 0.0, 0.0)], max_distance_m)


@pytest.mark.parametrize(
    ("max_distance_m", "paired"),
    [
        (0.0, ["inside"]),
        (-1_000.0, ["inside", "near"]),
        (-6_000.0, ["inside", "near"]),
        (5_000.0, ["inside"]),
        (8_000.0, []),
    ],
)
def test_max_distance_is_a_signed_margin(max_distance_m, paired):
    # Pixels every 0.01 deg (1.11 km); with a 10 m tolerance the edge lies half a pixel plus 10 m beyond the edge
    # pixel centres at lon +-1. "inside" is 5.56 km inside the east pixels (margin about +6.1 km), "near" 0.9 pixel
    # beyond them (about -0.43 km), "far" 5.56 km beyond them (about -5.0 km): far meets -6 km but is more than a
    # pixel spacing from its nearest pixel, so it is never paired.
    gcps = [_point("inside", 0.95, 0.0), _point("near", 1.009, 0.0), _point("far", 1.05, 0.0)]
    result = find_l1a_gcp_pairs(
        [_rect("L1A", -1.0, 1.0, -1.0, 1.0)], gcps, max_distance_m=max_distance_m, footprint_tolerance_km=0.01
    )
    assert [result.gcp_images[m.gcp_index].name for m in result.matches] == paired


def test_footprint_edge_is_about_one_pixel_beyond_the_edge_pixels():
    result = find_l1a_gcp_pairs([_rect("L1A", -1.0, 1.0, -1.0, 1.0)], [_point("edge", 1.0, 0.0)], max_distance_m=0.0)
    pixel_m = np.radians(0.01) * EARTH_MEAN_RADIUS_KM * 1e3
    assert 0.5 * pixel_m < result.matches[0].distance_m < 1.5 * pixel_m


def _grid_at_40n() -> NamedImageGrid:
    # 41 x 41 pixels every 0.01 deg, rows from 40.2 N south, columns from 100.2 W east.
    return _rect("L1A", -100.2, -99.8, 39.8, 40.2, n=41)


def test_corner_chip_inside_the_footprint_but_beyond_a_pixel_is_not_paired(caplog):
    # 0.8 pixel north and 0.9 pixel east of the north-east corner pixel: 1.06 north-south spacings from it, farther
    # than its adjacent pixels, yet about 90 m inside the footprint, which reaches past the corner diagonally.
    grid = _grid_at_40n()
    corner = _point("corner", -99.8 + 0.009, 40.2 + 0.008)
    with caplog.at_level("INFO", logger="curryer.correction.pairing"):
        result = find_l1a_gcp_pairs([grid], [corner], max_distance_m=0.0)
    margin = footprint_margin_m(result.l1a_images[0].footprint, np.array([40.208]), np.array([-99.791]))
    assert margin[0] >= 0.0
    assert result.matches == []
    assert any(
        "No-pair [L1A] to [corner]" in r.message and "outside the observation grid" in r.message for r in caplog.records
    )


def test_side_chip_within_a_pixel_pairs_and_gets_a_line_of_sight():
    from curryer.compute.spatial import geodetic_to_ecef
    from curryer.correction.psf import resolve_spacecraft_ecef

    grid = _grid_at_40n()
    side = _point("side", -99.8 + 0.009, 40.0)
    result = find_l1a_gcp_pairs([grid], [side], max_distance_m=0.0)
    assert [m.gcp_index for m in result.matches] == [0]
    r = geodetic_to_ecef(np.array([-100.0, 40.0, 400_000.0]), meters=True, degrees=True)
    _, boresight, _ = resolve_spacecraft_ecef(grid, r, 40.0, -99.8 + 0.009)
    edge = geodetic_to_ecef(np.array([-99.8, 40.0, 0.0]), meters=True, degrees=True)
    np.testing.assert_allclose(boresight, (edge - r) / np.linalg.norm(edge - r), atol=1e-12)


def test_pairing_across_the_antimeridian():
    lat, lon = np.meshgrid(np.linspace(1.0, -1.0, 201), np.linspace(179.0, 181.0, 201), indexing="ij")
    lon = (lon + 180.0) % 360.0 - 180.0
    l1a = NamedImageGrid(data=np.zeros_like(lat), lat=lat, lon=lon, name="L1A")
    gcps = [
        _point("180", 180.0, 0.0),
        _point("-180", -180.0, 0.0),
        _point("-179.5", -179.5, 0.5),
        _point("178.5", 178.5, 0.0),
    ]
    result = find_l1a_gcp_pairs([l1a], gcps, max_distance_m=0.0)
    assert [result.gcp_images[m.gcp_index].name for m in result.matches] == ["180", "-180", "-179.5"]


@pytest.mark.parametrize(("lat", "lon"), [(np.nan, 0.0), (0.0, np.nan), (0.0, 200.0)])
def test_gcp_with_invalid_center_raises_naming_the_chip(lat, lon):
    with pytest.raises(ValueError, match="GCP chip 'bad_chip': center pixel"):
        find_l1a_gcp_pairs([_rect("L1A", -1.0, 1.0, -1.0, 1.0)], [_point("bad_chip", lon, lat)], max_distance_m=0.0)
