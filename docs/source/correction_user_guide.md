# Correction & Verification User Guide

The `curryer.correction` package provides two workflows that share the same
configuration models and image-matching infrastructure.

## Architecture Overview

**Verification** is the inner step: it takes pre-computed image-matching
results (or raw geolocated data plus an image-matching callable) and
evaluates them against mission performance requirements, producing a
pass/fail verdict and per-measurement error details.

**Correction** is the outer loop: it tweaks SPICE kernel parameters, calls
the geolocation pipeline, then calls _verification_ to score each trial.
Because correction = geolocation + verification, every correction run
produces a `VerificationResult` as part of its output.

```
correction loop
│
├── [per parameter-set iteration]
│   ├── generate trial SPICE kernels
│   ├── geolocate observations (SPICE → lat/lon/alt)
│   └── verification ──► GCP pairing
│                    ──► image matching
│                    ──► error statistics
│                    ──► pass/fail verdict
│
└── aggregate results → CorrectionResult
```

Both `run_correction()` and `verify()` take the same `GeolocationSetup` as
their first argument.

## The config surface: Setup / Sweep / Output

Configuration is split into three focused Pydantic models so that the durable,
mission-specific parts are built once and the parts you vary between runs stay
lightweight:

- **`GeolocationSetup`** — the durable setup, built **once** per mission and
  reused across many runs. Holds `geo` (a `GeolocationConfig`: SPICE kernels,
  instrument, `time_field`), `requirements` (a `RequirementsConfig`), optional
  `data_config` (a `DataConfig`), optional `calibration` (a `CalibrationFiles`
  with direct `los_vectors_file` / `psf_file` paths), the science-Dataset
  variable names (`spacecraft_position_name` / `boresight_name` /
  `transformation_matrix_name`), and an optional `image_matching_func` hook.
- **`Sweep`** — the lightweight experiment you **vary** between runs: the
  `parameters` to sweep, the `search_strategy`, `n_iterations`, `seed`, and
  grid settings. Cheap to copy; see `with_strategy()` / `update_param()` below.
- **`OutputConfig`** — output settings (`netcdf` and `output_filename`).
  Optional; `run_correction()` auto-populates NetCDF metadata from the setup's
  performance threshold when omitted.

A run is `run_correction(setup, sweep, inputs, work_dir, output=None)`.

### Where setup & sweep are specified

A mission specifies its setup and sweep in one of two equivalent forms — a
**JSON config** (preferred for production/reproducibility) or a **Python
factory** (handier for building configs dynamically or varying a `Sweep` in
code). Both produce the same `(GeolocationSetup, Sweep, OutputConfig)` triple.

| Form                | File                                              | How it's consumed                                                       |
| ------------------- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| JSON (full example) | `examples/correction/clarreo_config.json`         | `setup, sweep, output = load_config_files(path)`                        |
| JSON (template)     | `examples/correction/example_config.json`         | Minimal 3-parameter starting point in the same schema                   |
| Python factory      | `examples/correction/clarreo_config.py`           | `create_clarreo_config(data_dir, generic_dir) → (setup, sweep, output)` |
| Test fixture        | `tests/test_correction/clarreo/clarreo_config.py` | `create_clarreo_setup_sweep(...)` — test infra, not public API          |

The JSON file has three top-level sections — `"setup"`, `"sweep"`, and an
optional `"output"` — each validated directly against its model (see
[Loading Config from JSON](#loading-config-from-json)). To copy a config for a
new mission, start from `clarreo_config.json` (or `clarreo_config.py`) and edit
the values.

The models, validators, and loaders themselves are all defined in
**`curryer/correction/config.py`** (`GeolocationSetup`, `Sweep`, `OutputConfig`,
`RequirementsConfig`, `CalibrationFiles`, `ParameterConfig` / `ParameterSpec`,
and `load_config_files` / `load_setup_from_json` / `load_sweep_from_json`).

---

## New Mission Checklist

To use correction or verification on a new mission, provide these values when
building your `GeolocationSetup` / `Sweep` (or in the JSON config file):

1. **SPICE kernels** — set `GeolocationConfig.meta_kernel_file`,
   `generic_kernel_dir`, and `dynamic_kernels` to your mission's kernel
   JSON files (on `setup.geo`).
2. **Instrument name** — set `GeolocationConfig.instrument_name` to the
   NAIF instrument name defined in your Instrument Kernel (IK), e.g.
   `"CPRS_HYSICS"`.
3. **Parameters to vary** (correction only) — define one `ParameterConfig`
   per adjustable frame offset or timing correction in `sweep.parameters`.
   Each entry points to a SPICE kernel JSON template (`config_file`) and a
   `ParameterSpec` (`spec`) with bounds, sigma, and units.
4. **Telemetry field names** — set `GeolocationConfig.time_field` to the
   column holding uGPS timestamps. Set `DataConfig.time_scale_factor` if
   your timestamps need scaling (e.g. GPS seconds → uGPS: `1e6`).
5. **Spacecraft variable names** — set `spacecraft_position_name`,
   `boresight_name`, and `transformation_matrix_name` on the setup to match
   the variable names in your image-matching output `xr.Dataset`.
6. **Performance requirements** — set `RequirementsConfig.performance_threshold_m`
   (per-measurement nadir-error limit in metres) and
   `performance_spec_percent` (minimum % of measurements that must pass) on
   `setup.requirements`.
7. **Image-matching function** (verification with raw geolocated data only)
   — attach a callable to `setup.image_matching_func` that accepts an
   `xr.Dataset` and returns an `xr.Dataset` with `lat_error_deg`,
   `lon_error_deg`, spacecraft state variables, and GCP coordinates.

**Annotated examples:**
`examples/correction/clarreo_config.py` and
`examples/correction/clarreo_config.json`

**Generic template:** `examples/correction/example_config.json`

---

## Key Concepts

| Concept              | Description                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `GeolocationSetup`   | Durable, mission-specific setup; first argument to `run_correction()`, `loop()`, `verify()`  |
| `Sweep`              | The parameter-variation experiment you vary between runs                                     |
| `OutputConfig`       | Output settings (`netcdf` metadata + `output_filename`)                                      |
| `GeolocationConfig`  | SPICE kernel paths and instrument identity (`setup.geo`)                                     |
| `ParameterConfig`    | One parameter to vary (kernel offset or timing correction)                                   |
| `ParameterSpec`      | Sampling spec for one parameter (`current_value`, `bounds`, `sigma`, `units`, `field`)       |
| `ParameterType`      | `CONSTANT_KERNEL`, `OFFSET_KERNEL`, or `OFFSET_TIME`                                         |
| `SearchStrategy`     | `RANDOM`, `GRID_SEARCH`, or `SINGLE_OFFSET`                                                  |
| `RequirementsConfig` | Verification thresholds (`performance_threshold_m`, `performance_spec_percent`)              |
| `CalibrationFiles`   | Direct (interim) `los_vectors_file` / `psf_file` paths on `setup.calibration`                |
| `CorrectionInput`    | Typed alternative to a raw `(telemetry, science, gcp)` tuple; accepted by `run_correction()` |
| `DataConfig`         | Declarative file-loading specification (format, time scale factor)                           |
| `VerificationResult` | Output of `verify()`: pass/fail flag, per-GCP errors, summary table                          |
| `GCPError`           | Per-measurement error detail (lat/lon error, nadir-equivalent, pass/fail)                    |

---

## Quick Start: Verification

The recommended input mode is `image_matching_results` — a list of
`xr.Dataset` objects, one per GCP pair, produced by your image-matching
pipeline. This is the path used by weekly automated checks.

```python
import xarray as xr
from pathlib import Path
from curryer.correction import (
    GeolocationSetup, GeolocationConfig, RequirementsConfig, verify,
)

# 1. Build the mission setup (or load from JSON — see below)
setup = GeolocationSetup(
    geo=GeolocationConfig(
        meta_kernel_file=Path("path/to/mission.kernels.tm.json"),
        generic_kernel_dir=Path("data/generic"),
        instrument_name="YOUR_INSTRUMENT",
        time_field="corrected_timestamp",
    ),
    requirements=RequirementsConfig(
        performance_threshold_m=250.0,       # per-measurement error limit (m)
        performance_spec_percent=39.0,       # % of measurements required to pass
    ),
    spacecraft_position_name="sc_position",  # variable name in your xr.Dataset
    boresight_name="boresight",
    transformation_matrix_name="t_inst2ref",
)

# 2. Provide pre-computed image-matching results (one xr.Dataset per GCP pair)
image_matching_results = [xr.open_dataset("matching_result_001.nc")]

# 3. Run verification — no SPICE kernel loading required for this path
result = verify(setup, image_matching_results=image_matching_results)

# 4. Inspect result
print(result.summary_table)
print("Passed:", result.passed)
print(f"Within threshold: {result.percent_within_threshold:.1f}%")
```

### `verify()` Input Modes

`verify()` takes `setup` as its only positional argument; all input modes are
keyword-only (first match wins, in the order below):

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

## Quick Start: Correction Loop

`run_correction()` is the preferred entry point. It returns a structured
`CorrectionResult` with the best parameters, pass/fail verdict,
recommendation, and a summary table.

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

### Build setup once, vary the sweep rapidly

The setup is durable, so build it once and re-run with cheap, re-validated
sweep variations. `with_strategy()` and `update_param()` return copies and
validate eagerly, so typos or out-of-spec values raise immediately rather than
deep inside a run:

```python
# Same setup, switch to a deterministic grid search
grid = sweep.with_strategy("grid", grid_points_per_param=5)
result_grid = run_correction(setup, grid, inputs, work_dir)

# Same setup, widen one parameter's bounds (selector = index, spec.field, or
# config_file stem) and re-seed a reproducible random sweep
wider = sweep.update_param("hps.az_ang_nonlin", bounds=[-100.0, 100.0])
repro = wider.with_strategy("random", seed=7, n_iterations=200)
result_repro = run_correction(setup, repro, inputs, work_dir)
```

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

## Loading Config from JSON

```python
from curryer.correction import load_config_files
setup, sweep, output = load_config_files("examples/correction/clarreo_config.json")
```

`load_config_files()` returns `(GeolocationSetup, Sweep, OutputConfig)`. The
file has three top-level sections — `"setup"`, `"sweep"`, and an optional
`"output"`. Missing `"setup"` or `"sweep"` raises a `KeyError`; `"output"`
defaults to an empty `OutputConfig`. To load just one model, use
`load_setup_from_json()` or `load_sweep_from_json()`.

**Minimal schema:**

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

### `GeolocationSetup`

The durable, mission-specific setup (built once, reused across sweeps).

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

### `Sweep`

The lightweight parameter-variation experiment, varied between runs. Use
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

### `OutputConfig`

| Field             | Type                   | Notes                                                     |
| ----------------- | ---------------------- | --------------------------------------------------------- |
| `netcdf`          | `NetCDFConfig \| None` | NetCDF metadata; `None` auto-populated from the threshold |
| `output_filename` | `str \| None`          | Output NetCDF filename; `None` uses the default           |

### `RequirementsConfig`

| Field                      | Type    | Notes                                                     |
| -------------------------- | ------- | --------------------------------------------------------- |
| `performance_threshold_m`  | `float` | **Required.** Per-measurement nadir-error limit in metres |
| `performance_spec_percent` | `float` | **Required.** Minimum % of measurements that must pass    |

### `GeolocationConfig`

| Field                 | Type            | Notes                                                             |
| --------------------- | --------------- | ----------------------------------------------------------------- |
| `meta_kernel_file`    | `Path`          | Path to the mission meta-kernel JSON file                         |
| `generic_kernel_dir`  | `Path`          | Directory containing generic shared SPICE kernels                 |
| `dynamic_kernels`     | `list[Path]`    | Kernel JSONs regenerated from telemetry each iteration            |
| `instrument_name`     | `str`           | SPICE instrument name as defined in the IK (e.g. `"CPRS_HYSICS"`) |
| `time_field`          | `str`           | Column in the science DataFrame holding uGPS timestamps           |
| `minimum_correlation` | `float \| None` | Image-matching quality filter (0.0–1.0); `None` disables          |

### `ParameterConfig`

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

### `ParameterType`

| Value             | Description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `CONSTANT_KERNEL` | Fixed attitude rotation applied to an instrument frame (roll/pitch/yaw offset) |
| `OFFSET_KERNEL`   | Dynamic bias added to a telemetry angle field to regenerate a CK kernel        |
| `OFFSET_TIME`     | Timing offset applied to science frame timestamps                              |

### `SearchStrategy`

| Value           | Description                                                                  |
| --------------- | ---------------------------------------------------------------------------- |
| `RANDOM`        | Monte Carlo: draws from a normal distribution at each iteration (default)    |
| `GRID_SEARCH`   | Cartesian product of evenly spaced grid points across all parameter bounds   |
| `SINGLE_OFFSET` | Each parameter swept independently while all others remain at nominal values |

### `CorrectionInput`

A `CorrectionInput` is format-neutral: each field is just a path, and the
reader is chosen by `DataConfig.file_format`. The first-class real-data path is
a **NetCDF image observation** (radiance as the science variable) that carries
telemetry, metadata, and science times — enough for curryer/SPICE to compute
the geometry — so the same file commonly serves as both the telemetry and
science input. See [Inputs & Data Formats](#inputs--data-formats) below.

| Field            | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `telemetry_file` | Telemetry observation file (NetCDF first-class; CSV/HDF5 read) |
| `science_file`   | Science/timing observation file (NetCDF first-class; CSV/HDF5) |
| `gcp_file`       | GCP reference-image file (NetCDF; `.mat` interim)              |

### `CalibrationFiles`

Direct calibration file paths on `setup.calibration`. Both fields are optional
and **interim** — real line-of-sight vectors and spacecraft geometry will be
SPICE-derived from telemetry, so nothing in the pipeline requires these.

| Field              | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `los_vectors_file` | Per-detector line-of-sight unit vectors (instrument) |
| `psf_file`         | Optical point-spread-function calibration            |

### `DataConfig`

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

**`ValidationError` on `GeolocationSetup` / `Sweep` construction**
Pydantic will identify the offending field. Common causes: wrong types
(`sigma` must be `float`, not a string), missing required fields (`geo` and
`requirements` are required on the setup), or an out-of-spec value passed to
`Sweep.update_param()` (its `ParameterSpec` is `extra="forbid"`, so unknown
field names raise immediately).

**`SPICE(PATHTOOLONG)` kernel path error**
SPICE enforces an 80-character kernel path limit. Curryer works around this
automatically. Override the temp directory if `/tmp` is unavailable:

```bash
export CURRYER_TEMP_DIR=/tmp
```

**`geolocated_data was provided but setup.image_matching_func is not set`**
The `geolocated_data` path needs an image-matching callable. Attach one to
`setup.image_matching_func` before calling `verify()`, or pass pre-computed
results via `image_matching_results=` instead.

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
