"""Synthetic correction-loop inputs on the CLARREO open-loop test kernels.

Builds an observation subimage, a GCP chip and the telemetry for
:func:`~curryer.correction.pipeline.loop` from one synthetic scene, so the loop
runs with real kernel creation, SPICE geolocation, terrain correction and image
matching.  The observation's radiance is the scene where the instrument looked
at its frame times plus a known time offset, so a parameter set with that
OFFSET_TIME re-geolocates it onto the truth.

**Test infrastructure helper – not a pytest test.**  Geolocation needs the
GMTED elevation data, so tests using it are ``extra``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from clarreo_config import create_clarreo_setup_sweep
from clarreo_data_loaders import load_clarreo_science, load_clarreo_telemetry
from scipy.io import savemat

from curryer import meta
from curryer import spicierpy as sp
from curryer.compute import elevation
from curryer.correction.config import CalibrationFiles, DataConfig, SearchConfig, SearchStrategy
from curryer.correction.kernel_ops import _create_dynamic_kernels, _create_parameter_kernels
from curryer.correction.parameters import _get_nominal_value
from curryer.correction.pipeline import LoopObservation, _geolocate_observation
from curryer.kernels import create

# Frames 40.. of the open-loop science times view New Zealand's South Island,
# about 65 degrees off nadir (1.5 km pixels).
FIRST_FRAME = 40
N_FRAMES = 21
N_COLUMNS = 21
FIXED_OFFSET_BODIES = ("pede", "base", "az", "yoke", "el", "hysics")


def _scene(lat0: float, lon0: float, rng: np.random.Generator):
    """Return f(lat, lon): 500 Gaussian blobs, 1.5-5 km wide, within 70 km of (lat0, lon0)."""
    blobs = [(*rng.uniform(-70, 70, 2), rng.uniform(1.5, 5.0), rng.uniform(0.5, 2.0)) for _ in range(500)]

    def scene(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        north_km = (lat - lat0) * 111.0
        east_km = (lon - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
        out = np.zeros_like(lat)
        for cx, cy, sigma, amplitude in blobs:
            out += amplitude * np.exp(-((east_km - cx) ** 2 + (north_km - cy) ** 2) / (2 * sigma**2))
        return out

    return scene


def build_loop_scene(root_dir: Path, work: Path, time_offset_s: float = 0.2):
    """Write the loop inputs for a scene seen *time_offset_s* after its frame times.

    The sweep is SINGLE_OFFSET with 3 values per parameter: each CLARREO kernel
    parameter at -20, 0 and +20 arcseconds, and OFFSET_TIME at -, 0 and
    +*time_offset_s*.  Parameter set ``3 * k + 1`` is nominal for parameter ``k``.

    Returns
    -------
    setup, sweep, output, tlm_sci_gcp_sets
    """
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
    setup, sweep, output = create_clarreo_setup_sweep(data_dir, root_dir / "data" / "generic")

    # The open-loop kernels position HySICS on the ISS through prebuilt fixed-offset SPKs.
    mk_config = json.loads((data_dir / "cprs_v01.kernels.tm.json").read_text())
    mk_config["mission_kernels"] = [str((data_dir / k).resolve()) for k in mk_config["mission_kernels"]] + [
        str((root_dir / "tests" / "data" / "clarreo" / f"cprs_{body}_v01.fixed_offset.spk.bsp").resolve())
        for body in FIXED_OFFSET_BODIES
    ]
    (work / "meta_kernel.json").write_text(json.dumps(mk_config))
    setup.geo.meta_kernel_file = work / "meta_kernel.json"
    setup.geo.minimum_correlation = 0.5

    # A cross-track fan of detector pixels in the HySICS frame.
    y = np.linspace(-0.015, 0.015, N_COLUMNS)
    los = np.column_stack([np.zeros(N_COLUMNS), y, np.ones(N_COLUMNS)])
    los /= np.linalg.norm(los, axis=1)[:, None]
    savemat(work / "los.mat", {"b_HS": los})
    setup.calibration = CalibrationFiles(
        los_vectors_file=work / "los.mat",
        psf_file=root_dir
        / "tests"
        / "data"
        / "clarreo"
        / "image_match"
        / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
    )
    setup.data_config = DataConfig(file_format="csv")
    setup.search = SearchConfig(grid_size=11, grid_span_km=4.0)

    for param in sweep.parameters:
        param.spec.bounds = (
            [-time_offset_s * 1e3, time_offset_s * 1e3] if param.ptype.name == "OFFSET_TIME" else [-20.0, 20.0]
        )
    sweep.search_strategy = SearchStrategy.SINGLE_OFFSET
    sweep.n_iterations = 3

    tlm = load_clarreo_telemetry(data_dir)
    tlm_csv = work / "telemetry.csv"
    tlm.to_csv(tlm_csv)
    science_ugps = (load_clarreo_science(data_dir)[setup.geo.time_field].values * 1e6).astype(np.int64)
    frame_ugps = science_ugps[FIRST_FRAME : FIRST_FRAME + N_FRAMES]

    # Where the pixels looked: nominal kernels at the frame times plus the offset.
    kernel_dir = work / "truth_kernels"
    kernel_dir.mkdir()
    mkrn = meta.MetaKernel.from_json(setup.geo.meta_kernel_file, relative=True, sds_dir=setup.geo.generic_kernel_dir)
    creator = create.KernelCreator(overwrite=True, append=False)
    dynamic = _create_dynamic_kernels(setup, kernel_dir, tlm, creator)
    nominal = [(param, _get_nominal_value(param)) for param in sweep.parameters]
    param_kernels, _ = _create_parameter_kernels(nominal, kernel_dir, tlm, frame_ugps, creator)
    blank = LoopObservation(radiance=np.zeros((N_FRAMES, N_COLUMNS)), frame_ugps=frame_ugps, los_vectors=los)
    with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, dynamic, param_kernels]):
        truth, _ = _geolocate_observation(
            setup.geo.instrument_name,
            frame_ugps + int(round(time_offset_s * 1e6)),
            blank,
            elevation.Elevation(setup.geo.dem_data_dir, meters=False, degrees=False),
        )

    scene = _scene(float(truth.lat.mean()), float(truth.lon.mean()), np.random.default_rng(3))
    xr.Dataset(
        {
            "band_data": (["frame", "pixel"], scene(truth.lat, truth.lon)),
            "ugps": (["frame"], frame_ugps),
            "detector_pixel": (["pixel"], np.arange(N_COLUMNS)),
        }
    ).to_netcdf(work / "observation.nc")

    # The chip covers the footprint, the largest parameter-set displacement and the search range.
    margin_deg = 0.3
    chip_lat = np.linspace(truth.lat.max() + margin_deg, truth.lat.min() - margin_deg, 400)
    chip_lon = np.linspace(truth.lon.min() - margin_deg, truth.lon.max() + margin_deg, 500)
    lon_grid, lat_grid = np.meshgrid(chip_lon, chip_lat)
    xr.Dataset(
        {
            "band_data": (["y", "x"], scene(lat_grid, lon_grid)),
            "lat": (["y", "x"], lat_grid),
            "lon": (["y", "x"], lon_grid),
        }
    ).to_netcdf(work / "chip.nc")

    return setup, sweep, output, [(str(tlm_csv), str(work / "observation.nc"), str(work / "chip.nc"))]
