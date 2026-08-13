# Correction & Verification User Guide

`curryer.correction` answers two practical questions about a mission's
geolocation. Each is driven by **one configuration file** and a short,
copy-paste call — there are no Python classes to write.

- **Verification** — _"Does our current geolocation meet the accuracy
  requirement?"_ Returns a pass/fail verdict with per-point error.
- **Correction** — _"What adjustment makes it more accurate?"_ Tries a range
  of pointing/timing adjustments and reports the best one.

Correction is verification run in a loop: for each candidate adjustment it
re-geolocates the observations, then evaluates the result with verification.

```
correction loop
│
├── [per candidate adjustment]
│   ├── generate trial SPICE kernels
│   ├── geolocate observations (SPICE → lat/lon/alt)
│   └── verification ──► GCP pairing
│                    ──► image matching
│                    ──► error statistics
│                    ──► pass/fail verdict
│
└── aggregate results → best adjustment + verdict
```

## Quickstart

Both workflows read the same mission config file. Copy
`examples/correction/example_config.json`, edit the values for your mission,
and run.

**Check accuracy** — verification:

```python
from curryer.correction import load_setup_from_json, verify

setup = load_setup_from_json("mission.json")
result = verify(setup, image_matching_results=[...])   # one dataset per GCP comparison

print(result.summary_table)        # per-point pass/fail table
print("Passed:", result.passed)
```

**Find a correction** — correction loop:

```python
from curryer.correction import load_config_files, run_correction, CorrectionInput

setup, sweep, output = load_config_files("mission.json")
inputs = [CorrectionInput(telemetry_file="obs.nc", science_file="obs.nc", gcp_file="chip.nc")]

result = run_correction(setup, sweep, inputs, work_dir="out", output=output)

print(result.summary_table)
print(result.recommendation)
```

> The data you process (observation and GCP files) is passed at run time via
> `inputs=`, **not** stored in the config file — it's this-run data, not a
> property of the mission.

## How a mission is configured

Everything a mission needs lives in **one JSON file** with up to three
sections. You edit values — you don't write code.

| Section  | Answers                                                            | How often it changes |
| -------- | ------------------------------------------------------------------ | -------------------- |
| `setup`  | Where are the kernels? Which instrument? What's the accuracy spec? | Once per mission     |
| `sweep`  | Which adjustments to try, and how to search them?                  | Per experiment       |
| `output` | What to name the result file (optional)                            | Rarely               |

```json
{
  "setup": {
    "geo": {
      "meta_kernel_file": "path/to/mission.kernels.tm.json",
      "generic_kernel_dir": "data/generic",
      "instrument_name": "YOUR_INSTRUMENT",
      "time_field": "corrected_timestamp"
    },
    "requirements": {
      "performance_threshold_m": 250.0,
      "performance_spec_percent": 39.0
    }
  },
  "sweep": {
    "search_strategy": "random",
    "n_iterations": 50,
    "seed": 42,
    "parameters": [
      {
        "ptype": "CONSTANT_KERNEL",
        "config_file": "path/to/frame_a.attitude.ck.json",
        "spec": {
          "current_value": [0.0, 0.0, 0.0],
          "bounds": [-300.0, 300.0],
          "sigma": 50.0,
          "units": "arcseconds"
        }
      }
    ]
  }
}
```

> **Templates to copy:** `examples/correction/example_config.json` (annotated,
> minimal) or `examples/correction/clarreo_config.json` (a complete real
> mission). Every field is specified in [Configuration Reference](#configuration-reference).

### Reuse the setup, try many adjustments

The `setup` is fixed for a mission; the `sweep` is the cheap, swappable part.
Keep one `mission.json` and either point at per-experiment sweep files
(`load_setup_from_json` + `load_sweep_from_json` read the `setup` and `sweep`
sections independently) or adjust in code:

```python
grid  = sweep.with_strategy("grid", grid_points_per_param=5)         # deterministic grid search
wider = sweep.update_param("hps.az_ang_nonlin", bounds=[-100, 100])  # widen one parameter
run_correction(setup, wider, inputs, work_dir="out")
```

Both return validated copies, so a typo or out-of-bounds value is caught
immediately. The `update_param` selector is the parameter's position, its
`spec.field` name, or its `config_file` filename stem.

---

## What to provide for a new mission

Copy `examples/correction/clarreo_config.json` and fill in these values. The
JSON path (in parentheses) is where each one goes.

1. **SPICE kernels** (`setup.geo.meta_kernel_file`, `generic_kernel_dir`,
   `dynamic_kernels`) — your mission's kernel files. `dynamic_kernels` are
   regenerated from telemetry on each run.
2. **Instrument** (`setup.geo.instrument_name`) — the NAIF instrument name from
   your Instrument Kernel, e.g. `"CPRS_HYSICS"`.
3. **Timestamps** (`setup.geo.time_field`) — the column holding uGPS times. If
   your times need scaling (e.g. GPS seconds → uGPS), set
   `setup.data_config.time_scale_factor` to `1e6`.
4. **Accuracy requirement** (`setup.requirements.performance_threshold_m` and
   `performance_spec_percent`) — the per-point error limit in metres, and the
   minimum percentage of points that must fall within it.
5. **Adjustments to try** (`sweep.parameters`, correction only) — one entry per
   pointing offset or timing correction (see [Configuration Reference](#configuration-reference)).
6. **Calibration** (`setup.calibration.los_vectors_file` / `psf_file`) — needed
   for the built-in image matcher (interim; will become SPICE-derived).

That's the full setup. The sections below are field-by-field reference and
runnable examples.

---

## Verification

Verification scores geolocation results against the mission requirement. The
recommended input is `image_matching_results` — a list of `xr.Dataset` objects
(one per GCP comparison) from your image-matching pipeline; this is the path
weekly automated checks use.

```python
import xarray as xr
from curryer.correction import load_setup_from_json, verify

setup = load_setup_from_json("mission.json")
results = [xr.open_dataset("matching_result_001.nc")]   # one per GCP comparison

result = verify(setup, image_matching_results=results)  # no SPICE kernels needed for this path

print(result.summary_table)                 # per-point pass/fail table
print("Passed:", result.passed)
print(f"Within threshold: {result.percent_within_threshold:.1f}%")
```

### Input modes

`verify()` takes `setup` first; everything else is keyword-only (the first
input you supply, in the order below, wins):

| Mode                         | Argument                                 | Notes                                                                                                              |
| ---------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Pre-computed image matching  | `image_matching_results=`                | **Recommended** for production and automated checks                                                                |
| Run image matching on demand | `geolocated_data=`                       | Custom `setup.image_matching_func`, **or** built-in pairing+matching with `gcp_directory=`/`los_file=`/`psf_file=` |
| Explicit file-path pairs     | `gcp_pairs=` (+ `los_file=`/`psf_file=`) | Supported                                                                                                          |
| Auto-paired paths            | `observation_paths=` + `gcp_directory=`  | Supported (+ `los_file=`/`psf_file=`)                                                                              |

The `geolocated_data` path matches in one of two ways. If you attach a callable
to `setup.image_matching_func`, `verify()` calls it. Otherwise, supply
`gcp_directory=`, `los_file=`, and `psf_file=` and `verify()` runs built-in
spatial pairing + image matching. If neither the override nor that trio is
provided, the call raises `ValueError`.

A runnable example using real test data: `examples/correction/example_verification.py`

---

## Correction

`run_correction()` is the entry point. It returns the best parameters, a
pass/fail verdict, a recommendation, and a summary table.

```python
import pathlib
from curryer.correction import (
    CorrectionInput, load_config_files, run_correction,
)

# Load setup + sweep (+ output) from one JSON file (recommended for production)
setup, sweep, output = load_config_files("examples/correction/clarreo_config.json")

# Each CorrectionInput maps one telemetry + science + GCP triplet.
# Raw (str, str, str) tuples are also accepted.
# S3 URIs ("s3://...") are accepted when boto3 is installed.
inputs = [
    CorrectionInput(
        telemetry_file=pathlib.Path("data/obs_20240317.nc"),
        science_file=pathlib.Path("data/obs_20240317.nc"),
        gcp_file=pathlib.Path("data/gcp/landsat_chip_001.nc"),
    )
]

work_dir = pathlib.Path("workdir_correction")
# NOTE: inputs come before work_dir; output is optional.
result = run_correction(setup, sweep, inputs, work_dir, output=output)

print(result.summary_table)
print("Passed:", result.passed)
print(result.recommendation)

best = min(result.results, key=lambda r: r["rms_error_m"])
print(f"Best RMS: {best['rms_error_m']:.2f} m  (parameters: {best['parameters']})")
```

(To reuse one setup across many sweeps, see
[Reuse the setup, try many adjustments](#reuse-the-setup-try-many-adjustments) above.)

### Low-level alternative: `loop()`

`loop()` returns the raw `(results, netcdf_data)` tuple and only accepts
plain `(str, str, str)` tuples (not `CorrectionInput`). Note its argument
order: `work_dir` comes **before** the inputs list.

```python
from curryer.correction import loop

inputs = [("data/obs.nc", "data/obs.nc", "data/gcp.nc")]
results, netcdf_data = loop(setup, sweep, work_dir, inputs)
```

A workflow template: `examples/correction/example_run_correction.py`

---

## Loading config from JSON

```python
from curryer.correction import load_config_files
setup, sweep, output = load_config_files("mission.json")
```

`load_config_files()` reads the `setup`, `sweep`, and optional `output`
sections. A missing `setup` or `sweep` raises a clear `KeyError`; `output`
defaults to empty. Load a single section with `load_setup_from_json()` or
`load_sweep_from_json()`.

A complete file, including the spacecraft variable names and `output` section
the [Quickstart](#quickstart) template leaves out:

```json
{
  "setup": {
    "geo": {
      "meta_kernel_file": "path/to/mission.kernels.tm.json",
      "generic_kernel_dir": "data/generic",
      "instrument_name": "YOUR_INSTRUMENT",
      "time_field": "corrected_timestamp",
      "dynamic_kernels": []
    },
    "requirements": {
      "performance_threshold_m": 250.0,
      "performance_spec_percent": 39.0
    },
    "spacecraft_position_name": "sc_position",
    "boresight_name": "boresight",
    "transformation_matrix_name": "t_inst2ref"
  },
  "sweep": {
    "search_strategy": "random",
    "n_iterations": 50,
    "seed": 42,
    "parameters": [
      {
        "ptype": "CONSTANT_KERNEL",
        "config_file": "path/to/frame_a.attitude.ck.json",
        "spec": {
          "current_value": [0.0, 0.0, 0.0],
          "bounds": [-300.0, 300.0],
          "sigma": 50.0,
          "units": "arcseconds"
        }
      }
    ]
  },
  "output": {
    "output_filename": "correction_results.nc"
  }
}
```

A frame rotation is authored as a single `CONSTANT_KERNEL` parameter whose
`spec.current_value` is the `[roll, pitch, yaw]` triplet.

A fully populated mission example: `examples/correction/clarreo_config.json`
A generic annotated template: `examples/correction/example_config.json`

---

## Configuration Reference

### Setup — `setup`

The durable, mission-specific configuration (built once, reused across sweeps).

| Field                        | Type                       | Notes                                                                |
| ---------------------------- | -------------------------- | -------------------------------------------------------------------- |
| `geo`                        | `GeolocationConfig`        | **Required.** SPICE kernel paths and instrument identity             |
| `requirements`               | `RequirementsConfig`       | **Required.** Pass/fail thresholds                                   |
| `data_config`                | `DataConfig \| None`       | File-loading specification; `None` uses CSV defaults                 |
| `calibration`                | `CalibrationFiles \| None` | Optional direct LOS/PSF calibration file paths (interim)             |
| `spacecraft_position_name`   | `str`                      | Variable name in the image-matching `xr.Dataset` for SC position     |
| `boresight_name`             | `str`                      | Variable name in the image-matching `xr.Dataset` for boresight       |
| `transformation_matrix_name` | `str`                      | Variable name in the image-matching `xr.Dataset` for rotation matrix |
| `image_matching_func`        | `Callable \| None`         | Optional custom image-matching callable; excluded from JSON          |

### Sweep — `sweep`

The parameter-variation experiment, varied between runs. Use
`sweep.with_strategy(strategy, **changes)` and
`sweep.update_param(selector, **spec_changes)` for cheap, re-validated copies.

| Field                   | Type                    | Notes                                           |
| ----------------------- | ----------------------- | ----------------------------------------------- |
| `parameters`            | `list[ParameterConfig]` | **Required** (at least one). Parameters to vary |
| `search_strategy`       | `SearchStrategy`        | `RANDOM` / `GRID_SEARCH` / `SINGLE_OFFSET`      |
| `n_iterations`          | `int`                   | Iterations (RANDOM / values-per-param SINGLE)   |
| `seed`                  | `int \| None`           | Random seed for reproducible `RANDOM` sweeps    |
| `grid_points_per_param` | `int`                   | Points per parameter when `GRID_SEARCH` is used |
| `max_grid_sets`         | `int`                   | Safety cap on total grid-search parameter sets  |

### Output — `output`

| Field             | Type                   | Notes                                                     |
| ----------------- | ---------------------- | --------------------------------------------------------- |
| `netcdf`          | `NetCDFConfig \| None` | NetCDF metadata; `None` auto-populated from the threshold |
| `output_filename` | `str \| None`          | Output NetCDF filename; `None` uses the default           |

### Requirements — `setup.requirements`

| Field                      | Type    | Notes                                                     |
| -------------------------- | ------- | --------------------------------------------------------- |
| `performance_threshold_m`  | `float` | **Required.** Per-measurement nadir-error limit in metres |
| `performance_spec_percent` | `float` | **Required.** Minimum % of measurements that must pass    |

### Kernels & instrument — `setup.geo`

| Field                 | Type            | Notes                                                             |
| --------------------- | --------------- | ----------------------------------------------------------------- |
| `meta_kernel_file`    | `Path`          | Path to the mission meta-kernel JSON file                         |
| `generic_kernel_dir`  | `Path`          | Directory containing generic shared SPICE kernels                 |
| `dynamic_kernels`     | `list[Path]`    | Kernel JSONs regenerated from telemetry each iteration            |
| `instrument_name`     | `str`           | SPICE instrument name as defined in the IK (e.g. `"CPRS_HYSICS"`) |
| `time_field`          | `str`           | Column in the science DataFrame holding uGPS timestamps           |
| `minimum_correlation` | `float \| None` | Image-matching quality filter (0.0–1.0); `None` disables          |

### Parameters — `sweep.parameters[]`

| Field                | Type                   | Notes                                                               |
| -------------------- | ---------------------- | ------------------------------------------------------------------- |
| `ptype`              | `ParameterType`        | `CONSTANT_KERNEL`, `OFFSET_KERNEL`, or `OFFSET_TIME`                |
| `config_file`        | `Path \| None`         | Path to the SPICE kernel JSON template; required for kernel types   |
| `spec`               | `ParameterSpec`        | Sampling specification (see below)                                  |
| `spec.current_value` | `float \| list[float]` | Baseline value(s). For `CONSTANT_KERNEL`: `[roll, pitch, yaw]`      |
| `spec.bounds`        | `[min, max]`           | Offset limits in the same units as `sigma`                          |
| `spec.sigma`         | `float \| None`        | Sampling standard deviation for the `RANDOM` strategy               |
| `spec.units`         | `str \| None`          | Physical units string, e.g. `"arcseconds"` or `"milliseconds"`      |
| `spec.field`         | `str \| None`          | Telemetry column name; required for `OFFSET_KERNEL` / `OFFSET_TIME` |

### Parameter types — `ptype`

| Value             | Description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `CONSTANT_KERNEL` | Fixed attitude rotation applied to an instrument frame (roll/pitch/yaw offset) |
| `OFFSET_KERNEL`   | Dynamic bias added to a telemetry angle field to regenerate a CK kernel        |
| `OFFSET_TIME`     | Timing offset applied to science frame timestamps                              |

### Search strategies — `search_strategy`

| Value           | Description                                                                  |
| --------------- | ---------------------------------------------------------------------------- |
| `RANDOM`        | Monte Carlo: draws from a normal distribution at each iteration (default)    |
| `GRID_SEARCH`   | Cartesian product of evenly spaced grid points across all parameter bounds   |
| `SINGLE_OFFSET` | Each parameter swept independently while all others remain at nominal values |

### Inputs — `inputs=`

Each input is format-neutral: every field is just a path, and the reader is
chosen by `setup.data_config.file_format`. The first-class real-data path is
a **NetCDF image observation** (radiance as the science variable) that carries
telemetry, metadata, and science times — enough for curryer/SPICE to compute
the geometry — so the same file commonly serves as both the telemetry and
science input. See [Inputs & Data Formats](#inputs--data-formats) below.

| Field            | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `telemetry_file` | Telemetry observation file (NetCDF first-class; CSV/HDF5 read) |
| `science_file`   | Science/timing observation file (NetCDF first-class; CSV/HDF5) |
| `gcp_file`       | GCP reference-image file (NetCDF; `.mat` interim)              |

### Calibration — `setup.calibration`

Direct calibration file paths. Both fields are optional
and **interim** — real line-of-sight vectors and spacecraft geometry will be
SPICE-derived from telemetry, so nothing in the pipeline requires these.

| Field              | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `los_vectors_file` | Per-detector line-of-sight unit vectors (instrument) |
| `psf_file`         | Optical point-spread-function calibration            |

### Data loading — `setup.data_config`

| Field               | Description                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| `file_format`       | `"csv"`, `"netcdf"`, or `"hdf5"` — drives the reader for telemetry/science             |
| `time_scale_factor` | Multiply science timestamps by this factor to obtain uGPS (e.g. `1e6` for GPS seconds) |
| `position_columns`  | Optional list of spacecraft-position column names in the telemetry DataFrame           |

---

## Inputs & Data Formats

The config and API are **format-agnostic**: the internal contract is an
`ImageGrid` / `xr.Dataset`, and `DataConfig.file_format` selects the reader.
Two distinct input families exist, and it is worth being explicit about which
is which:

- **NetCDF image observations (first-class, intended real-data path).** A
  NetCDF observation carries the radiance as its science variable plus the
  telemetry, metadata, and science times needed for curryer/SPICE to compute
  the line-of-sight and spacecraft geometry. This is the direction the package
  is built toward, and the recommended format for new missions.
- **`.mat` / file-based LOS & PSF (interim test scaffolding).** The code was
  developed against interim `.mat` test fixtures — fake image arrays and
  standalone `.mat` line-of-sight / PSF files supplied via
  `CalibrationFiles.los_vectors_file` / `psf_file` (or `verify(..., los_file=,
psf_file=)`). These are convenience inputs for testing without SPICE-derived
  geometry; they are never required and will be superseded as LOS/spacecraft
  geometry becomes SPICE-derived.

Real-data NetCDF ingestion (deriving geometry from observation telemetry) is
the intended direction, not a claim that it is fully implemented today. Frame
new work around the NetCDF-observation path and treat `.mat`/file-based LOS/PSF
as interim.

---

## Image-Matching Dataset Format

The `image_matching_results` passed to `verify()` must be a list of
`xr.Dataset` objects with a `measurement` dimension. Spacecraft-state
variable names are configured on `GeolocationSetup`.

| Variable                       | Dimension(s)                      | Description                                              |
| ------------------------------ | --------------------------------- | -------------------------------------------------------- |
| `lat_error_deg`                | `[measurement]`                   | Latitude error (degrees, positive = northward)           |
| `lon_error_deg`                | `[measurement]`                   | Longitude error (degrees, positive = eastward)           |
| `gcp_lat_deg`                  | `[measurement]`                   | GCP centre latitude (degrees)                            |
| `gcp_lon_deg`                  | `[measurement]`                   | GCP centre longitude (degrees)                           |
| `gcp_alt`                      | `[measurement]`                   | GCP altitude (metres; typically `0.0`)                   |
| `<spacecraft_position_name>`   | `[measurement, xyz]`              | Spacecraft position in CTRS/ITRF93 frame, metres         |
| `<boresight_name>`             | `[measurement, xyz]`              | Instrument boresight unit vector in the instrument frame |
| `<transformation_matrix_name>` | `[measurement, xyz_from, xyz_to]` | Rotation matrix from instrument frame to CTRS            |

`<spacecraft_position_name>` corresponds to `GeolocationSetup.spacecraft_position_name`
(and similarly for the other two). For CLARREO these are `"riss_ctrs"`,
`"bhat_hs"`, and `"t_hs2ctrs"` respectively.

When spacecraft-state variables are unavailable (e.g. testing without SPICE
kernels), set the boresight to the nadir unit vector `(-r_sc / |r_sc|)` and
the rotation matrix to identity. This makes the nadir-equivalent scaling
factor 1.0 and passes raw errors through unchanged — the correct conservative
default.

---

## Interpreting Results

### Verification

```python
result = verify(setup, image_matching_results=datasets)

print(result.summary_table)         # ASCII table: per-GCP pass/fail
print("Passed:", result.passed)
print(f"Within threshold: {result.percent_within_threshold:.1f}%")

for err in result.per_gcp_errors:
    print(f"GCP {err.gcp_index}: nadir_error={err.nadir_equiv_error_m:.1f} m  passed={err.passed}")

# Serialise to JSON (xr.Dataset field must be excluded)
json_str = result.model_dump_json(exclude={"aggregate_stats"})
result.aggregate_stats.to_netcdf("verification_stats.nc")
```

### Comparing Before and After

```python
from curryer.correction import compare_results

before = verify(setup, image_matching_results=pre_datasets)
after  = verify(setup, image_matching_results=post_datasets)
print(compare_results(before, after))
```

### Correction Loop

```python
result = run_correction(setup, sweep, inputs, work_dir)

print(result.summary_table)
print("Passed:", result.passed)
print(result.recommendation)

best = min(result.results, key=lambda r: r["rms_error_m"])
print(f"Best RMS: {best['rms_error_m']:.2f} m  (parameters: {best['parameters']})")
```

---

## AWS / S3 Data Access

For missions storing image-matching results in S3 (requires `boto3`):

```python
import datetime
from curryer.correction.dataio import S3Configuration, find_netcdf_objects, download_netcdf_objects

s3_config = S3Configuration(
    bucket="my-mission-bucket",
    base_prefix="image_match",   # date-partitioned subdirs: base_prefix/YYYYMMDD/
)

object_keys = find_netcdf_objects(
    s3_config,
    start_date=datetime.date(2024, 3, 17),
    end_date=datetime.date(2024, 3, 17),
)

local_paths = download_netcdf_objects(s3_config, object_keys, destination="/tmp/downloads")
```

Configure credentials via `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
environment variables or an IAM role. S3 support is optional; the core API
works with local `Path` objects only.

---

## Troubleshooting

**`KeyError: Missing required 'sweep' section`**
The JSON config is missing a required top-level section. `load_config_files()`
requires `"setup"` and `"sweep"`; `"output"` is optional.

**`ValidationError` on `requirements`**
The setup's `requirements` block must contain `performance_threshold_m` and
`performance_spec_percent`. These encode the mission's geolocation requirement
and cannot be omitted.

**`ValidationError` when loading the config**
The error names the offending field. Common causes: a wrong type (`sigma` must
be a number, not a string), a missing required field (`setup.geo` and
`setup.requirements` are required), or an unknown field name in a parameter's
`spec` (unknown keys are rejected rather than silently ignored).

**`SPICE(PATHTOOLONG)` kernel path error**
SPICE enforces an 80-character kernel path limit. Curryer works around this
automatically. Override the temp directory if `/tmp` is unavailable:

```bash
export CURRYER_TEMP_DIR=/tmp
```

**`ValueError` when passing `geolocated_data=` to `verify()`**
The `geolocated_data` path needs a way to match. Either attach a callable to
`setup.image_matching_func`, or pass `gcp_directory=`, `los_file=`, and
`psf_file=` to use the built-in pairing + matching. (Or skip this path and pass
pre-computed `image_matching_results=` instead.)

**NaN values in `nadir_equiv_total_error_m`**
The nadir-equivalent conversion requires valid spacecraft geometry. A
negative discriminant (logged as `Suspicious geometry: discriminant < 0`)
means the off-nadir angle exceeds the Earth-limb limit — typically caused by
incorrect or missing spacecraft-state variables. Verify that
`spacecraft_position_name`, `boresight_name`, and
`transformation_matrix_name` resolve to the correct variables and that the
spacecraft position vector is in metres in the CTRS (Earth-fixed) frame.

---

## Reference Examples

| File                                            | Status   | Description                                                           |
| ----------------------------------------------- | -------- | --------------------------------------------------------------------- |
| `examples/correction/example_verification.py`   | Runnable | End-to-end verification demo; real CLARREO data or synthetic fallback |
| `examples/correction/example_run_correction.py` | Template | Correction loop template; exits cleanly when data/tools missing       |
| `examples/correction/clarreo_config.py`         | —        | Mission config factory — use as a template for new missions           |
| `examples/correction/clarreo_config.json`       | —        | Fully populated JSON config for CLARREO                               |
| `examples/correction/example_config.json`       | —        | Annotated generic JSON config template                                |

Run from the repository root:

```bash
python examples/correction/example_verification.py
python examples/correction/example_run_correction.py
```

---

## GCP Chip Regridding

Ground Control Point (GCP) reference imagery (e.g. Landsat imagery) is often in HDF format, in which each pixel's
position is stored as Earth-Centered, Earth-Fixed (ECEF) X/Y/Z coordinates on
an irregular geodetic grid. Before these chips can be matched against L1A
science images for verification and correction purposes, they must be resampled onto a regular latitude/longitude grid
that matches the spatial resolution of the mission's detector.

The `curryer.correction` package provides a complete pipeline for this:

```
HDF chip  →  ECEF → WGS84 geodetic  →  bilinear regrid  →  regular NetCDF grid
```

### Quick start — single file (Python)

```python
from pathlib import Path
from curryer.correction.config import RegridConfig
from curryer.correction.image_io import load_gcp_chip_from_hdf
from curryer.correction.regrid import regrid_gcp_chip

# 1. Load raw chip (returns band data + ECEF X/Y/Z arrays)
band, ecef_x, ecef_y, ecef_z = load_gcp_chip_from_hdf(
    Path("LT08CHP.20140803.p002r071.c01.v001.hdf")
)

# 2. Configure regridding (~100 m resolution for CLARREO)
config = RegridConfig(output_resolution_deg=(0.0009, 0.0009))

# 3. Regrid and save in one call
regridded = regrid_gcp_chip(
    band,
    (ecef_x, ecef_y, ecef_z),
    config,
    output_file="regridded_chip.nc",
    output_metadata={
        "source_file": "LT08CHP.20140803.p002r071.c01.v001.hdf",
        "mission": "CLARREO Pathfinder",
        "sensor": "Landsat-8",
        "band": "red",
    },
)

# 4. regridded is an ImageGrid — ready for image matching
print(regridded.data.shape)   # e.g. (421, 433)
print(regridded.lat[0, 0])    # top-left latitude
```

### Batch processing — command-line script

For a directory of 100 Landsat chips use the provided script directly:

```bash
# Regrid all *.hdf files in /data/landsat_gcps/ and write NetCDF to /data/regridded/
python examples/correction/regrid_gcp_chips.py /data/landsat_gcps/ /data/regridded/
```

Output:

```
Processing 100 file(s) → /data/regridded/

[ 1/100] START LT08CHP.20140101.p002r071.c01.v001.hdf
         ✓ LT08CHP.20140101.p002r071.c01.v001_regridded.nc  (1823 KB, 4.2s)
[ 2/100] START LT08CHP.20140116.p002r071.c01.v001.hdf
         ✓ LT08CHP.20140116.p002r071.c01.v001_regridded.nc  (1791 KB, 4.0s)
...
────────────────────────────────────────────────────────────
Finished: all 100 file(s) processed successfully.
```

#### Common options

| Flag                       | Default              | Description                                       |
| -------------------------- | -------------------- | ------------------------------------------------- |
| `--resolution DLAT DLON`   | `0.0009 0.0009`      | Output resolution in degrees (~100 m)             |
| `--pattern GLOB`           | `*.hdf`              | Filename pattern when source is a directory       |
| `--mission NAME`           | `CLARREO Pathfinder` | Written to NetCDF `mission` attribute             |
| `--band DATASET`           | `Band_1`             | HDF dataset name for the radiometric channel      |
| `--skip-existing`          | off                  | Skip files whose output `.nc` already exists      |
| `--dry-run`                | off                  | Print what would be done, write nothing           |
| `--no-conservative-bounds` | off                  | Use full ECEF extent (may include edge artefacts) |
| `-v` / `--verbose`         | off                  | Show per-row progress and DEBUG log output        |

Resume an interrupted run with `--skip-existing`; preview before committing with `--dry-run`.

### Batch processing — Python API

Use this when you need custom logic (filtering, parallel execution, etc.):

```python
from pathlib import Path
from curryer.correction.config import RegridConfig
from curryer.correction.image_io import load_gcp_chip_from_hdf
from curryer.correction.regrid import regrid_gcp_chip

input_dir  = Path("/data/landsat_gcps")
output_dir = Path("/data/regridded")
output_dir.mkdir(parents=True, exist_ok=True)

config = RegridConfig(output_resolution_deg=(0.0009, 0.0009))

hdf_files = sorted(input_dir.glob("LT08CHP.*.hdf"))
errors = {}
for hdf_file in hdf_files:
    nc_file = output_dir / f"{hdf_file.stem}_regridded.nc"
    try:
        band, ecef_x, ecef_y, ecef_z = load_gcp_chip_from_hdf(hdf_file)
        regrid_gcp_chip(
            band, (ecef_x, ecef_y, ecef_z), config,
            output_file=str(nc_file),
            output_metadata={"source_file": hdf_file.name, "mission": "CLARREO Pathfinder"},
        )
        print(f"  ✓ {hdf_file.name}")
    except Exception as exc:
        errors[hdf_file.name] = str(exc)
        print(f"  ✗ {hdf_file.name}: {exc}")
```

### Loading regridded output

```python
from pathlib import Path
from curryer.correction.image_io import load_image_grid

gcp = load_image_grid(Path("regridded_chip.nc"))
# gcp.data  — 2-D radiometric values
# gcp.lat   — 2-D latitude  (regular grid, decreasing from top to bottom)
# gcp.lon   — 2-D longitude (regular grid, increasing left to right)
# gcp.h     — 2-D height above WGS84 ellipsoid (metres), or None
```

### Configuration reference

```python
from curryer.correction.config import RegridConfig

config = RegridConfig(output_resolution_deg=(0.0009, 0.0009))   # resolution-based (most common)
config = RegridConfig(output_grid_size=(500, 500))               # fixed output size
config = RegridConfig(                                           # explicit extent + resolution
    output_bounds=(-116.5, -115.5, 38.0, 39.0),   # (minlon, maxlon, minlat, maxlat)
    output_resolution_deg=(0.001, 0.001),
)
```

| Parameter               | Type                               | Description                                                               |
| ----------------------- | ---------------------------------- | ------------------------------------------------------------------------- |
| `output_resolution_deg` | `(dlat, dlon)`                     | Grid spacing in degrees. Mutually exclusive with `output_grid_size`.      |
| `output_grid_size`      | `(nrows, ncols)`                   | Fixed output dimensions. Mutually exclusive with `output_resolution_deg`. |
| `output_bounds`         | `(minlon, maxlon, minlat, maxlat)` | Explicit geographic extent. Requires `output_resolution_deg`.             |
| `conservative_bounds`   | `bool` (default `True`)            | Shrink bounds to grid interior to avoid edge extrapolation.               |
| `interpolation_method`  | `"bilinear"` \| `"nearest"`        | Interpolation algorithm (default `"bilinear"`).                           |
| `fill_value`            | `float` (default `NaN`)            | Value assigned to output points outside the input footprint.              |

### Output NetCDF structure

```
dimensions:  y (rows), x (cols)
variables:
  band_data(y, x)   — radiometric values      [digital_number]
  lat(y, x)         — latitude                [degrees_north]
  lon(y, x)         — longitude               [degrees_east]
  h(y, x)           — height above WGS84      [metres]  (present when height is available)
global attributes:
  title, Conventions (CF-1.8), source_file, mission, band, processing_software, …
```

> **Coordinate convention:** row 0 is northernmost (latitude decreases down),
> column 0 is westernmost (longitude increases right) — consistent with the
> MATLAB `Chip_regrid2.m` output format.

**Notes:**

- **Ellipsoid:** ECEF → geodetic conversion always uses **WGS84** (`curryer.compute.spatial.ecef_to_geodetic`).
- **HDF format:** `load_gcp_chip_from_hdf` tries HDF4 (`pyhdf`) first, then falls back to HDF5 (`h5py`).
- **Memory:** a 1400 × 1400 Landsat chip uses ~30 MB RAM; 100 chips processed sequentially stay well within a 4 GB budget.
- **Performance:** each chip typically takes 3–6 seconds on a single CPU core.

```{toctree}
:maxdepth: 1

gcp_regridding.md
```
