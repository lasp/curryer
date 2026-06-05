"""
Pipeline runner helpers for CLARREO integration tests.

Provides ``run_upstream_pipeline`` and ``run_downstream_pipeline``,
which exercise the upstream (kernel creation + geolocation) and
downstream (GCP pairing + image matching + error statistics) halves
of the Correction pipeline respectively.

These are *test infrastructure helpers*, not pytest tests themselves.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from curryer.correction import correction
from curryer.correction.config import DataConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (avoid hard-coding sys.path assumptions at module import time)
# ---------------------------------------------------------------------------


def _load_clarreo_loaders():
    from _image_match_helpers import discover_test_image_match_cases, run_image_matching_with_applied_errors
    from clarreo_config import create_clarreo_correction_config
    from clarreo_data_loaders import load_clarreo_science, load_clarreo_telemetry

    return (
        create_clarreo_correction_config,
        load_clarreo_telemetry,
        load_clarreo_science,
        discover_test_image_match_cases,
        run_image_matching_with_applied_errors,
    )


# ---------------------------------------------------------------------------
# Upstream pipeline
# ---------------------------------------------------------------------------


def run_upstream_pipeline(
    n_iterations: int = 5,
    work_dir: Path | None = None,
) -> tuple[list, dict, Path]:
    """Test the upstream segment: parameter generation → kernel creation → geolocation.

    Uses ``synthetic_image_matching`` so it does NOT require valid GCP pairs.

    Returns
    -------
    (results_list, results_summary_dict, output_file_path)
    """
    from _synthetic_helpers import synthetic_image_matching

    (
        create_clarreo_correction_config,
        load_clarreo_telemetry,
        load_clarreo_science,
        _,
        _,
    ) = _load_clarreo_loaders()

    logger.info("=== UPSTREAM PIPELINE TEST ===")

    root_dir = Path(__file__).parents[3]
    generic_dir = root_dir / "data" / "generic"
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"

    if work_dir is None:
        _tmp = tempfile.mkdtemp(prefix="curryer_upstream_")
        work_dir = Path(_tmp)
        atexit.register(shutil.rmtree, work_dir, True)
        logger.info("Temporary work dir: %s", work_dir)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    config = create_clarreo_correction_config(data_dir, generic_dir)
    config.n_iterations = n_iterations
    config.output_filename = "upstream_results.nc"

    tlm_df = load_clarreo_telemetry(data_dir)
    sci_df = load_clarreo_science(data_dir)

    tlm_csv = work_dir / "clarreo_telemetry.csv"
    sci_csv = work_dir / "clarreo_science.csv"
    tlm_df.to_csv(tlm_csv)
    sci_df.to_csv(sci_csv)

    config.data_config = DataConfig(file_format="csv", time_scale_factor=1e6)
    config._image_matching_override = synthetic_image_matching

    tlm_sci_gcp_sets = [(str(tlm_csv), str(sci_csv), "synthetic_gcp.mat")]

    logger.info("Executing Correction upstream workflow (%d iterations)…", n_iterations)
    results, netcdf_data = correction.loop(config, work_dir, tlm_sci_gcp_sets)

    output_file = work_dir / config.get_output_filename()
    logger.info("Upstream pipeline complete. Output: %s", output_file)

    summary = {
        "mode": "upstream",
        "iterations": n_iterations,
        "parameter_sets": len(netcdf_data["parameter_set_id"]),
        "status": "complete",
    }
    return results, summary, output_file


# ---------------------------------------------------------------------------
# Downstream pipeline
# ---------------------------------------------------------------------------


def run_downstream_pipeline(
    n_iterations: int = 5,
    test_cases: list[str] | None = None,
    work_dir: Path | None = None,
) -> tuple[list, dict, Path]:
    """Test the downstream segment: GCP pairing → image matching → error statistics.

    Uses pre-geolocated test data (.mat files) with known errors applied.
    Does NOT test kernel creation or SPICE geolocation.

    Returns
    -------
    (results_list, results_summary_dict, output_file_path)
    """
    from _image_match_helpers import discover_test_image_match_cases, run_image_matching_with_applied_errors
    from clarreo_config import create_clarreo_correction_config

    from curryer.correction.grid_types import NamedImageGrid
    from curryer.correction.image_io import load_image_grid
    from curryer.correction.pairing import find_l1a_gcp_pairs

    logger.info("=== DOWNSTREAM PIPELINE TEST ===")

    root_dir = Path(__file__).parents[3]
    generic_dir = root_dir / "data" / "generic"
    data_dir = root_dir / "tests" / "data" / "clarreo" / "gcs"
    test_data_dir = root_dir / "tests" / "data" / "clarreo" / "image_match"

    if work_dir is None:
        _tmp = tempfile.mkdtemp(prefix="curryer_downstream_")
        work_dir = Path(_tmp)
        atexit.register(shutil.rmtree, work_dir, True)
        logger.info("Temporary work dir: %s", work_dir)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    # --- STEP 1: discover test cases ---
    discovered_cases = discover_test_image_match_cases(test_data_dir, test_cases)
    if not discovered_cases:
        raise RuntimeError(f"No test cases found in {test_data_dir} (test_cases={test_cases})")

    # --- STEP 2: GCP spatial pairing ---
    l1a_images = []
    l1a_to_testcase: dict[str, dict] = {}
    for tc in discovered_cases:
        _g = load_image_grid(tc["subimage_file"], mat_key="subimage")
        img = NamedImageGrid(
            data=_g.data,
            lat=_g.lat,
            lon=_g.lon,
            h=_g.h,
            name=str(tc["subimage_file"].relative_to(test_data_dir)),
        )
        l1a_images.append(img)
        l1a_to_testcase[img.name] = tc

    gcp_files_seen: set = set()
    gcp_images = []
    for tc in discovered_cases:
        if tc["gcp_file"] not in gcp_files_seen:
            _g = load_image_grid(tc["gcp_file"], mat_key="GCP")
            gcp_img = NamedImageGrid(
                data=_g.data,
                lat=_g.lat,
                lon=_g.lon,
                h=_g.h,
                name=str(tc["gcp_file"].relative_to(test_data_dir)),
            )
            gcp_images.append(gcp_img)
            gcp_files_seen.add(tc["gcp_file"])

    pairing_result = find_l1a_gcp_pairs(l1a_images, gcp_images, max_distance_m=0.0)
    paired_test_cases = [l1a_to_testcase[pairing_result.l1a_images[m.l1a_index].name] for m in pairing_result.matches]
    n_gcp_pairs = len(paired_test_cases)
    logger.info("Pairing complete: %d valid pairs", n_gcp_pairs)

    # --- STEP 3: build config ---
    base_config = create_clarreo_correction_config(data_dir, generic_dir)
    config = correction.CorrectionConfig(
        seed=42,
        n_iterations=n_iterations,
        parameters=[
            correction.ParameterConfig(
                ptype=correction.ParameterType.CONSTANT_KERNEL,
                config_file=data_dir / "cprs_hysics_v01.attitude.ck.json",
                spec={
                    "current_value": [0.0, 0.0, 0.0],
                    "sigma": 0.0,
                    "units": "arcseconds",
                    "transformation_type": "dcm_rotation",
                    "coordinate_frames": ["HYSICS_SLIT", "CRADLE_ELEVATION"],
                },
            )
        ],
        geo=base_config.geo,
        performance_threshold_m=base_config.performance_threshold_m,
        performance_spec_percent=base_config.performance_spec_percent,
        netcdf=base_config.netcdf,
        calibration_file_names=base_config.calibration_file_names,
        spacecraft_position_name=base_config.spacecraft_position_name,
        boresight_name=base_config.boresight_name,
        transformation_matrix_name=base_config.transformation_matrix_name,
    )
    config.data_config = DataConfig(file_format="csv", time_scale_factor=1e6)

    # --- STEP 4: iterate ---
    netcdf_data = correction._build_netcdf_structure(config, n_iterations, n_gcp_pairs)
    threshold_metric = config.netcdf.threshold_metric_name
    image_match_cache: dict[str, xr.Dataset] = {}

    for param_idx in range(n_iterations):
        logger.info("Iteration %d/%d", param_idx + 1, n_iterations)
        pair_errors = []

        for pair_idx, tc in enumerate(paired_test_cases):
            cache_key = f"{tc['case_id']}_{tc.get('subcase_name', '')}"
            cached = image_match_cache.get(cache_key)

            output = run_image_matching_with_applied_errors(
                tc,
                param_idx,
                randomize_errors=True,
                error_variation_percent=3.0,
                cache_results=True,
                cached_result=cached,
            )
            if cache_key not in image_match_cache:
                image_match_cache[cache_key] = output

            rms_m = np.sqrt((output.attrs["lat_error_km"] * 1000) ** 2 + (output.attrs["lon_error_km"] * 1000) ** 2)
            pair_errors.append(rms_m)

            netcdf_data["rms_error_m"][param_idx, pair_idx] = rms_m
            netcdf_data["mean_error_m"][param_idx, pair_idx] = rms_m
            netcdf_data["max_error_m"][param_idx, pair_idx] = rms_m
            netcdf_data["n_measurements"][param_idx, pair_idx] = 1
            netcdf_data["im_lat_error_km"][param_idx, pair_idx] = output.attrs["lat_error_km"]
            netcdf_data["im_lon_error_km"][param_idx, pair_idx] = output.attrs["lon_error_km"]
            netcdf_data["im_ccv"][param_idx, pair_idx] = output.attrs["correlation_ccv"]
            netcdf_data["im_grid_step_m"][param_idx, pair_idx] = output.attrs["final_grid_step_m"]

        valid = np.array([e for e in pair_errors if not np.isnan(e)])
        if len(valid) > 0:
            pct = (valid < config.performance_threshold_m).sum() / len(valid) * 100
            netcdf_data[threshold_metric][param_idx] = pct
            netcdf_data["mean_rms_all_pairs"][param_idx] = np.mean(valid)
            netcdf_data["best_pair_rms"][param_idx] = np.min(valid)
            netcdf_data["worst_pair_rms"][param_idx] = np.max(valid)

    # --- STEP 5: error statistics ---
    image_matching_results = list(image_match_cache.values())
    correction.call_error_stats_module(image_matching_results, correction_config=config)

    # --- STEP 6: save ---
    output_file = work_dir / "downstream_results.nc"
    correction._save_netcdf_results(netcdf_data, output_file, config)

    logger.info("Downstream pipeline complete. Output: %s", output_file)
    summary = {"mode": "downstream", "iterations": n_iterations, "test_pairs": n_gcp_pairs, "status": "complete"}
    return [], summary, output_file
