# Correction Package User Guide

The `curryer.correction` package provides two main workflows:

**Verification** checks whether the current SPICE kernels and alignment parameters meet mission
geolocation accuracy requirements without modifying anything. It accepts pre-computed
image-matching results and evaluates them against configurable thresholds.

**Correction** runs a parameter sweep (Monte Carlo, grid search, or single-offset) to find kernel
adjustment values that minimise geolocation error. At each iteration it regenerates SPICE kernels
with the trial parameters, runs image matching against ground-control points (GCPs), and computes
error statistics.

Both workflows share a single `CorrectionConfig` object and the same image-matching infrastructure.

---

## New Mission Integration Checklist

To adapt the correction system to a new mission, provide values for the following in your
`CorrectionConfig` (or JSON config file):

1. **SPICE kernels** — set `GeolocationConfig.meta_kernel_file`, `generic_kernel_dir`, and
   `dynamic_kernels` to the paths for your mission's kernel JSON files.
2. **Instrument name** — set `GeolocationConfig.instrument_name` to the NAIF instrument name
   defined in your Instrument Kernel (IK), e.g. `"CPRS_HYSICS"`.
3. **Parameters to vary** — define one `ParameterConfig` per adjustable frame offset or timing
   correction. Each parameter points to a SPICE kernel JSON template (`config_file`) and specifies
   bounds, sigma, and units.
4. **Telemetry field names** — set `GeolocationConfig.time_field` to the column name in your
   telemetry DataFrame that holds uGPS timestamps (or GPS seconds; set `DataConfig.time_scale_factor`
   accordingly). For `OFFSET_KERNEL` and `OFFSET_TIME` parameters, also set `data.field` to the
   telemetry column to perturb.
5. **Dataset variable names** — set `spacecraft_position_name`, `boresight_name`, and
   `transformation_matrix_name` on `CorrectionConfig` to match the variable names used in your
   image-matching output `xr.Dataset`.
6. **Mission requirements** — set `performance_threshold_m` (per-measurement error limit in metres)
   and `performance_spec_percent` (minimum percentage of measurements that must pass).

A fully annotated example is in `examples/correction/clarreo_config.py` and
`examples/correction/clarreo_config.json`.
A generic template with comments is in `examples/correction/example_config.json`.

---

## Key Concepts

| Concept              | Description                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `CorrectionConfig`   | Top-level configuration; passed to `run_correction()`, `loop()`, and `verify()`              |
| `GeolocationConfig`  | SPICE kernel paths and instrument identity                                                   |
| `ParameterConfig`    | One parameter to vary (kernel offset or timing correction)                                   |
| `ParameterType`      | `CONSTANT_KERNEL`, `OFFSET_KERNEL`, or `OFFSET_TIME`                                         |
| `SearchStrategy`     | `RANDOM`, `GRID_SEARCH`, or `SINGLE_OFFSET`                                                  |
| `RequirementsConfig` | Verification thresholds (`performance_threshold_m`, `performance_spec_percent`)              |
| `CorrectionInput`    | Typed alternative to a raw `(telemetry, science, gcp)` tuple; accepted by `run_correction()` |
| `DataConfig`         | Declarative file-loading specification (format, time scale factor)                           |
| `VerificationResult` | Output of `verify()`: pass/fail flag, per-GCP errors, summary table                          |
| `GCPError`           | Per-measurement error detail (lat/lon error, nadir-equivalent, pass/fail)                    |

---

## Quick Start: Verification

The recommended input mode is `image_matching_results` — a list of `xr.Dataset` objects,
one per GCP pair, produced by your image-matching pipeline. This is the path used by weekly
automated checks.

```python
import xarray as xr
from pathlib import Path
from curryer.correction import (
    CorrectionConfig, GeolocationConfig, ParameterConfig,
    ParameterType, verify,
)

# 1. Build config (or load from JSON — see below)
config = CorrectionConfig(
    n_iterations=1,
    parameters=[
        ParameterConfig(
            ptype=ParameterType.CONSTANT_KERNEL,
            config_file=Path("path/to/frame_a.attitude.ck.json"),
            data={"current_value": [0.0, 0.0, 0.0], "bounds": [-300.0, 300.0]},
        )
    ],
    geo=GeolocationConfig(
        meta_kernel_file=Path("path/to/mission.kernels.tm.json"),
        generic_kernel_dir=Path("data/generic"),
        instrument_name="YOUR_INSTRUMENT",
        time_field="corrected_timestamp",
    ),
    performance_threshold_m=250.0,          # mission requirement: per-measurement limit (m)
    performance_spec_percent=39.0,          # mission requirement: % of measurements that must pass
    spacecraft_position_name="sc_position", # variable name in your image-matching xr.Dataset
    boresight_name="boresight",
    transformation_matrix_name="t_inst2ref",
)

# 2. Provide pre-computed image-matching results (one xr.Dataset per GCP pair).
image_matching_results = [xr.open_dataset("matching_result_001.nc")]

# 3. Run verification — no SPICE kernel loading required for this path
result = verify(config, image_matching_results=image_matching_results)

# 4. Check result
print(result.summary_table)
print("Passed:", result.passed)
print("Percent within threshold:", result.percent_within_threshold)
```

### `verify()` Input Modes

The first provided argument wins:

| Mode                         | Argument                                | Notes                                                                                                        |
| ---------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Pre-computed image matching  | `image_matching_results=`               | Recommended for all production and automated checks                                                          |
| Run image matching on demand | `geolocated_data=`                      | Calls `config._image_matching_override(geolocated_data)` — you must attach the callable; no built-in matcher |
| Explicit file-path pairs     | `gcp_pairs=`                            | Not yet implemented                                                                                          |
| Auto-paired paths            | `observation_paths=` + `gcp_directory=` | Not yet implemented                                                                                          |

The `geolocated_data` path does **not** include a built-in image-matching algorithm. `verify()` simply
calls whatever callable you attach to `config._image_matching_override`. If that attribute is not set
the call raises `ValueError`. This path is intended for test injection or custom production pipelines
that own their own image matcher. Use `image_matching_results=` everywhere else.

A runnable example using real test data is in `examples/correction/example_verification.py`.

---

## Quick Start: Correction Loop

`run_correction()` is the preferred entry point. It returns a structured `CorrectionResult` with
best parameters, pass/fail verdict, recommendation, and a summary table.

```python
import pathlib
from curryer.correction import (
    CorrectionInput, load_config_from_json,
    SearchStrategy, run_correction,
)

# Option A: Load config from JSON (recommended for production)
config = load_config_from_json("examples/correction/clarreo_config.json")

# Option B: Override settings at runtime without modifying the JSON
config = config.model_copy(update={"search_strategy": SearchStrategy.RANDOM, "n_iterations": 50})

# Define input file sets — one entry per (telemetry, science, GCP) triplet.
# CorrectionInput uses named fields; raw (str, str, str) tuples are also accepted.
# S3 URIs ("s3://...") are accepted when boto3 is installed.
inputs = [
    CorrectionInput(
        telemetry_file=pathlib.Path("data/tlm_20240317.csv"),
        science_file=pathlib.Path("data/sci_20240317.csv"),
        gcp_file=pathlib.Path("data/gcp/GCP_scene_001_resampled.mat"),
    )
]

work_dir = pathlib.Path("workdir_correction")
result = run_correction(config, work_dir, inputs)

# Structured result
print(result.summary_table)       # human-readable ASCII table
print("Passed:", result.passed)
print(result.recommendation)

# Raw results list (one dict per iteration) and NetCDF output are also available
best = min(result.results, key=lambda r: r["rms_error_m"])
print(f"Best RMS: {best['rms_error_m']:.2f} m  (parameters: {best['parameters']})")
```

### Low-level alternative: `loop()`

`loop()` returns the raw `(results, netcdf_data)` tuple. It only accepts plain
`(str, str, str)` tuples — not `CorrectionInput`:

```python
from curryer.correction import loop

inputs = [("data/tlm.csv", "data/sci.csv", "data/gcp.mat")]
results, netcdf_data = loop(config, work_dir, inputs)
```

A workflow template with annotated file paths is in
`examples/correction/example_run_correction.py`.

---

## Loading Config from JSON

```python
from curryer.correction import load_config_from_json
config = load_config_from_json("examples/correction/clarreo_config.json")
```

The JSON file must contain three top-level sections: `mission_config`, `correction`, and
`geolocation`. A missing section raises a `KeyError`.

**Minimal schema:**

```json
{
  "mission_config": {
    "mission_name": "YOUR_MISSION",
    "instrument_name": "YOUR_INSTRUMENT",
    "kernel_mappings": {
      "constant_kernel": { "frame_a": "path/to/frame_a.attitude.ck.json" },
      "offset_kernel": { "sensor_az": "path/to/sensor_az.attitude.ck.json" }
    }
  },
  "correction": {
    "n_iterations": 50,
    "seed": 42,
    "performance_threshold_m": 250.0,
    "performance_spec_percent": 39.0,
    "parameters": [
      {
        "name": "frame_a_roll",
        "parameter_type": "CONSTANT_KERNEL",
        "initial_value": 0.0,
        "bounds": [-300.0, 300.0],
        "sigma": 50.0,
        "units": "arcseconds"
      }
    ]
  },
  "geolocation": {
    "meta_kernel_file": "path/to/mission.kernels.tm.json",
    "generic_kernel_dir": "data/generic",
    "instrument_name": "YOUR_INSTRUMENT",
    "time_field": "corrected_timestamp",
    "dynamic_kernels": []
  }
}
```

A fully populated mission example is `examples/correction/clarreo_config.json`.
A generic annotated template is `examples/correction/example_config.json`.

---

## Configuration Reference

### `CorrectionConfig`

| Field                        | Type                    | Notes                                                                |
| ---------------------------- | ----------------------- | -------------------------------------------------------------------- |
| `n_iterations`               | `int`                   | Number of parameter sets to evaluate                                 |
| `seed`                       | `int \| None`           | Random seed for reproducibility                                      |
| `parameters`                 | `list[ParameterConfig]` | Parameters to vary                                                   |
| `search_strategy`            | `SearchStrategy`        | `RANDOM` / `GRID_SEARCH` / `SINGLE_OFFSET`                           |
| `grid_points_per_param`      | `int`                   | Points per parameter when `GRID_SEARCH` is used                      |
| `max_grid_sets`              | `int`                   | Upper bound on total grid-search parameter sets                      |
| `geo`                        | `GeolocationConfig`     | SPICE kernel paths and instrument identity                           |
| `performance_threshold_m`    | `float`                 | **Required.** Per-measurement nadir-error limit in metres            |
| `performance_spec_percent`   | `float`                 | **Required.** Minimum percentage of measurements that must pass      |
| `data`                       | `DataConfig \| None`    | File-loading specification (format, time scale factor)               |
| `netcdf`                     | `NetCDFConfig \| None`  | NetCDF output metadata                                               |
| `spacecraft_position_name`   | `str`                   | Variable name in the image-matching `xr.Dataset` for SC position     |
| `boresight_name`             | `str`                   | Variable name in the image-matching `xr.Dataset` for boresight       |
| `transformation_matrix_name` | `str`                   | Variable name in the image-matching `xr.Dataset` for rotation matrix |

### `GeolocationConfig`

| Field                 | Type            | Notes                                                              |
| --------------------- | --------------- | ------------------------------------------------------------------ |
| `meta_kernel_file`    | `Path`          | Path to the mission meta-kernel JSON file                          |
| `generic_kernel_dir`  | `Path`          | Directory containing generic shared SPICE kernels                  |
| `dynamic_kernels`     | `list[Path]`    | Kernel JSONs regenerated from telemetry each iteration             |
| `instrument_name`     | `str`           | SPICE instrument name as defined in the IK (e.g. `"CPRS_HYSICS"`)  |
| `time_field`          | `str`           | Column in the science DataFrame holding uGPS timestamps            |
| `minimum_correlation` | `float \| None` | Image-matching quality filter threshold (0.0–1.0); `None` disables |

### `ParameterConfig`

| Field                | Type                   | Notes                                                               |
| -------------------- | ---------------------- | ------------------------------------------------------------------- |
| `ptype`              | `ParameterType`        | `CONSTANT_KERNEL`, `OFFSET_KERNEL`, or `OFFSET_TIME`                |
| `config_file`        | `Path \| None`         | Path to the SPICE kernel JSON template; required for kernel types   |
| `data.current_value` | `float \| list[float]` | Baseline value(s). For `CONSTANT_KERNEL`: `[roll, pitch, yaw]`      |
| `data.bounds`        | `[min, max]`           | Offset limits in the same units as `sigma`                          |
| `data.sigma`         | `float \| None`        | Sampling standard deviation for the `RANDOM` strategy               |
| `data.units`         | `str \| None`          | Physical units string, e.g. `"arcseconds"` or `"milliseconds"`      |
| `data.field`         | `str \| None`          | Telemetry column name; required for `OFFSET_KERNEL` / `OFFSET_TIME` |

### `ParameterType`

| Value             | Description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `CONSTANT_KERNEL` | Fixed attitude rotation applied to an instrument frame (roll/pitch/yaw offset) |
| `OFFSET_KERNEL`   | Dynamic bias added to a telemetry angle field to regenerate a CK kernel        |
| `OFFSET_TIME`     | Timing offset applied to science frame timestamps                              |

### `SearchStrategy`

| Value           | Description                                                                        |
| --------------- | ---------------------------------------------------------------------------------- |
| `RANDOM`        | Monte Carlo: draws from a normal distribution at each iteration (default)          |
| `GRID_SEARCH`   | Cartesian product of evenly spaced grid points across all parameter bounds         |
| `SINGLE_OFFSET` | Each parameter swept independently while all others remain at their nominal values |

### `CorrectionInput`

| Field            | Description                                   |
| ---------------- | --------------------------------------------- |
| `telemetry_file` | Path to preprocessed telemetry CSV/NetCDF     |
| `science_file`   | Path to science timing CSV/NetCDF             |
| `gcp_file`       | Path to GCP reference image (`.mat` or `.nc`) |

### `DataConfig`

| Field               | Description                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| `file_format`       | `"csv"`, `"netcdf"`, or `"hdf5"`                                                       |
| `time_scale_factor` | Multiply science timestamps by this factor to obtain uGPS (e.g. `1e6` for GPS seconds) |
| `position_columns`  | Optional list of spacecraft-position column names in the telemetry DataFrame           |

---

## Image-Matching Dataset Format

The `image_matching_results` passed to `verify()` must be a list of `xr.Dataset` objects, each
with a `measurement` dimension and the following variables. The spacecraft-state variable names
are configurable via `CorrectionConfig`.

| Variable                       | Dimension(s)                      | Description                                                        |
| ------------------------------ | --------------------------------- | ------------------------------------------------------------------ |
| `lat_error_deg`                | `[measurement]`                   | Latitude error from image matching (degrees, positive = northward) |
| `lon_error_deg`                | `[measurement]`                   | Longitude error from image matching (degrees, positive = eastward) |
| `gcp_lat_deg`                  | `[measurement]`                   | GCP centre latitude (degrees)                                      |
| `gcp_lon_deg`                  | `[measurement]`                   | GCP centre longitude (degrees)                                     |
| `gcp_alt`                      | `[measurement]`                   | GCP altitude (metres; typically `0.0`)                             |
| `<spacecraft_position_name>`   | `[measurement, xyz]`              | Spacecraft position in CTRS/ITRF93 frame, metres                   |
| `<boresight_name>`             | `[measurement, xyz]`              | Instrument boresight unit vector in the instrument frame           |
| `<transformation_matrix_name>` | `[measurement, xyz_from, xyz_to]` | Rotation matrix from instrument frame to CTRS                      |

The `<spacecraft_position_name>` placeholder corresponds to `CorrectionConfig.spacecraft_position_name`
(and similarly for the other two). For CLARREO these are `"riss_ctrs"`, `"bhat_hs"`, and
`"t_hs2ctrs"` respectively.

When spacecraft-state variables are unavailable (e.g. during testing without loaded SPICE kernels),
set the boresight to the nadir unit vector `(-r_sc / |r_sc|)` and the rotation matrix to the
identity. This produces nadir-equivalent scaling factors of 1.0 and passes raw errors through
unchanged — the correct conservative default when real pointing data is unavailable.

---

## Interpreting Results

### Verification

```python
result = verify(config, image_matching_results=datasets)

# Human-readable ASCII table with per-GCP pass/fail
print(result.summary_table)

# Overall pass/fail
print("Passed:", result.passed)
print(f"Percent within threshold: {result.percent_within_threshold:.1f}%")

# Per-measurement detail
for err in result.per_gcp_errors:
    print(f"GCP {err.gcp_index}: nadir_error={err.nadir_equiv_error_m:.1f} m  passed={err.passed}")

# Serialise to JSON (xr.Dataset field must be excluded)
json_str = result.model_dump_json(exclude={"aggregate_stats"})
result.aggregate_stats.to_netcdf("verification_stats.nc")
```

### Comparing Before and After Correction

```python
from curryer.correction import compare_results

before = verify(config, image_matching_results=pre_correction_datasets)
after  = verify(config, image_matching_results=post_correction_datasets)

print(compare_results(before, after))
```

### Correction Loop

```python
result = run_correction(config, work_dir, inputs)

print(result.summary_table)
print("Passed:", result.passed)
print(result.recommendation)

# Find the parameter set with the lowest RMS error
best = min(result.results, key=lambda r: r["rms_error_m"])
print(f"Best RMS: {best['rms_error_m']:.2f} m  (parameters: {best['parameters']})")
```

---

## AWS / S3 Data Access

For missions that store image-matching results in S3, use the helpers in
`curryer.correction.dataio` (requires `boto3`):

```python
import datetime
from curryer.correction.dataio import S3Configuration, find_netcdf_objects, download_netcdf_objects

# S3Configuration takes (bucket, base_prefix) — no region argument;
# configure region via AWS_DEFAULT_REGION env var or IAM role.
s3_config = S3Configuration(
    bucket="my-mission-bucket",
    base_prefix="image_match",           # date-partitioned subdirs expected: base_prefix/YYYYMMDD/
)

# find_netcdf_objects requires an inclusive date range
object_keys = find_netcdf_objects(
    s3_config,
    start_date=datetime.date(2024, 3, 17),
    end_date=datetime.date(2024, 3, 17),
)

# download to a local directory; returns list of local Paths
local_paths = download_netcdf_objects(s3_config, object_keys, destination="/tmp/downloads")
```

Configure credentials via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
or an IAM role (EC2/ECS). S3 support is optional; the core correction API works with local
`Path` objects only.

---

## Troubleshooting

**`KeyError: Missing required 'correction' section`**
The JSON config is missing one of the three required top-level sections. Verify that
`mission_config`, `correction`, and `geolocation` are all present.

**`KeyError: Missing required 'performance_threshold_m'`**
The `correction` section must contain `performance_threshold_m` and `performance_spec_percent`.
These cannot be omitted; they encode the mission's geolocation requirement.

**`ValidationError` on `CorrectionConfig` construction**
Pydantic will identify the offending field. Common causes: wrong types (`sigma` must be `float`,
not a string) or missing required fields (`n_iterations` is required).

**`SPICE(PATHTOOLONG)` kernel path error**
SPICE enforces an 80-character kernel path limit. Curryer works around this automatically using
symlinks or copies to `/tmp`. If the default location is unavailable, override it:

```bash
export CURRYER_TEMP_DIR=/tmp
```

**`NotImplementedError` from `verify()`**
The `gcp_pairs=` and `observation_paths=` input modes are not yet implemented. Use
`image_matching_results=` instead.

**`geolocated_data was provided but config._image_matching_override is not set`**
`verify()` does not include a built-in image matcher. When `geolocated_data=` is used, it expects
a callable attached at `config._image_matching_override` and calls
`config._image_matching_override(geolocated_data)`. Either attach the callable before calling
`verify()`, or pre-compute your image-matching results and pass them via `image_matching_results=`
instead.

**NaN values in `nadir_equiv_total_error_m`**
The nadir-equivalent conversion requires valid spacecraft geometry. A negative discriminant
(logged as `Suspicious geometry: discriminant < 0`) means the computed off-nadir angle exceeds
the Earth-limb limit — typically caused by incorrect or missing spacecraft-state variables.
Verify that `spacecraft_position_name`, `boresight_name`, and `transformation_matrix_name`
resolve to the correct variables and that the spacecraft position vector is in metres in the
CTRS (Earth-fixed) frame. If real pointing data is unavailable, see the fallback guidance in
the Image-Matching Dataset Format section above.

---

## Reference Examples

| File                                            | Description                                                     |
| ----------------------------------------------- | --------------------------------------------------------------- |
| `examples/correction/example_verification.py`   | Runnable verification demo; uses committed test data            |
| `examples/correction/example_run_correction.py` | Correction loop template; dry-runs when SPICE tools are missing |
| `examples/correction/clarreo_config.py`         | Mission config factory — use as a template for new missions     |
| `examples/correction/clarreo_config.json`       | Fully populated JSON config for the CLARREO mission             |
| `examples/correction/example_config.json`       | Annotated generic JSON config template                          |

Run from the repository root:

```bash
python examples/correction/example_verification.py
```
