# Correction & Verification User Guide

The `curryer.correction` package provides two workflows that share the same
configuration object and image-matching infrastructure.

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

Both `run_correction()` and `verify()` accept the same `CorrectionConfig`.

---

## New Mission Checklist

To use correction or verification on a new mission, provide these values in
your `CorrectionConfig` (or JSON config file):

1. **SPICE kernels** — set `GeolocationConfig.meta_kernel_file`,
   `generic_kernel_dir`, and `dynamic_kernels` to your mission's kernel
   JSON files.
2. **Instrument name** — set `GeolocationConfig.instrument_name` to the
   NAIF instrument name defined in your Instrument Kernel (IK), e.g.
   `"CPRS_HYSICS"`.
3. **Parameters to vary** (correction only) — define one `ParameterConfig`
   per adjustable frame offset or timing correction. Each entry points to a
   SPICE kernel JSON template (`config_file`) and specifies bounds, sigma,
   and units.
4. **Telemetry field names** — set `GeolocationConfig.time_field` to the
   column holding uGPS timestamps. Set `DataConfig.time_scale_factor` if
   your timestamps need scaling (e.g. GPS seconds → uGPS: `1e6`).
5. **Spacecraft variable names** — set `spacecraft_position_name`,
   `boresight_name`, and `transformation_matrix_name` to match the variable
   names in your image-matching output `xr.Dataset`.
6. **Performance requirements** — set `performance_threshold_m`
   (per-measurement nadir-error limit in metres) and
   `performance_spec_percent` (minimum % of measurements that must pass).
7. **Image-matching function** (verification with raw geolocated data only)
   — attach a callable to `config._image_matching_override` that accepts
   `xr.Dataset` and returns `xr.Dataset` with `lat_error_deg`,
   `lon_error_deg`, spacecraft state variables, and GCP coordinates.

**Annotated examples:**
`examples/correction/clarreo_config.py` and
`examples/correction/clarreo_config.json`

**Generic template:** `examples/correction/example_config.json`

---

## Key Concepts

| Concept              | Description                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `CorrectionConfig`   | Top-level config; passed to `run_correction()`, `loop()`, and `verify()`                     |
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

The recommended input mode is `image_matching_results` — a list of
`xr.Dataset` objects, one per GCP pair, produced by your image-matching
pipeline. This is the path used by weekly automated checks.

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
            data={"current_value": [0.0, 0.0, 0.0], "bounds": [-300.0, 300.0]},
        )
    ],
    geo=GeolocationConfig(
        meta_kernel_file=Path("path/to/mission.kernels.tm.json"),
        generic_kernel_dir=Path("data/generic"),
        instrument_name="YOUR_INSTRUMENT",
        time_field="corrected_timestamp",
    ),
    performance_threshold_m=250.0,           # per-measurement error limit (m)
    performance_spec_percent=39.0,           # % of measurements required to pass
    spacecraft_position_name="sc_position",  # variable name in your xr.Dataset
    boresight_name="boresight",
    transformation_matrix_name="t_inst2ref",
)

# 2. Provide pre-computed image-matching results (one xr.Dataset per GCP pair)
image_matching_results = [xr.open_dataset("matching_result_001.nc")]

# 3. Run verification — no SPICE kernel loading required for this path
result = verify(config, image_matching_results=image_matching_results)

# 4. Inspect result
print(result.summary_table)
print("Passed:", result.passed)
print(f"Within threshold: {result.percent_within_threshold:.1f}%")
```

### `verify()` Input Modes

| Mode                         | Argument                                | Notes                                                      |
| ---------------------------- | --------------------------------------- | ---------------------------------------------------------- |
| Pre-computed image matching  | `image_matching_results=`               | **Recommended** for production and automated checks        |
| Run image matching on demand | `geolocated_data=`                      | Calls `config._image_matching_override(data)`; must be set |
| Explicit file-path pairs     | `gcp_pairs=`                            | Supported                                                  |
| Auto-paired paths            | `observation_paths=` + `gcp_directory=` | Supported                                                  |

The `geolocated_data` path does **not** include a built-in image-matching
algorithm. `verify()` calls whatever callable you attach to
`config._image_matching_override`. If that attribute is not set, the call
raises `ValueError`. Use `image_matching_results=`, `gcp_pairs=`, or
`observation_paths=` + `gcp_directory=` for other supported cases.

A runnable example using real test data: `examples/correction/example_verification.py`

---

## Quick Start: Correction Loop

`run_correction()` is the preferred entry point. It returns a structured
`CorrectionResult` with the best parameters, pass/fail verdict,
recommendation, and a summary table.

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

# Each CorrectionInput maps one telemetry + science + GCP triplet.
# Raw (str, str, str) tuples are also accepted.
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

print(result.summary_table)
print("Passed:", result.passed)
print(result.recommendation)

best = min(result.results, key=lambda r: r["rms_error_m"])
print(f"Best RMS: {best['rms_error_m']:.2f} m  (parameters: {best['parameters']})")
```

### Low-level alternative: `loop()`

`loop()` returns the raw `(results, netcdf_data)` tuple and only accepts
plain `(str, str, str)` tuples (not `CorrectionInput`):

```python
from curryer.correction import loop

inputs = [("data/tlm.csv", "data/sci.csv", "data/gcp.mat")]
results, netcdf_data = loop(config, work_dir, inputs)
```

A workflow template: `examples/correction/example_run_correction.py`

---

## Loading Config from JSON

```python
from curryer.correction import load_config_from_json
config = load_config_from_json("examples/correction/clarreo_config.json")
```

The JSON file must contain three top-level sections: `mission_config`,
`correction`, and `geolocation`. A missing section raises a `KeyError`.

**Minimal schema:**

```json
{
  "mission_config": {
    "mission_name": "YOUR_MISSION",
    "instrument_name": "YOUR_INSTRUMENT",
    "kernel_mappings": {
      "constant_kernel": { "frame_a": "path/to/frame_a.attitude.ck.json" }
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

A fully populated mission example: `examples/correction/clarreo_config.json`
A generic annotated template: `examples/correction/example_config.json`

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

| Value           | Description                                                                  |
| --------------- | ---------------------------------------------------------------------------- |
| `RANDOM`        | Monte Carlo: draws from a normal distribution at each iteration (default)    |
| `GRID_SEARCH`   | Cartesian product of evenly spaced grid points across all parameter bounds   |
| `SINGLE_OFFSET` | Each parameter swept independently while all others remain at nominal values |

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

The `image_matching_results` passed to `verify()` must be a list of
`xr.Dataset` objects with a `measurement` dimension. Spacecraft-state
variable names are configured via `CorrectionConfig`.

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

`<spacecraft_position_name>` corresponds to `CorrectionConfig.spacecraft_position_name`
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
result = verify(config, image_matching_results=datasets)

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

before = verify(config, image_matching_results=pre_datasets)
after  = verify(config, image_matching_results=post_datasets)
print(compare_results(before, after))
```

### Correction Loop

```python
result = run_correction(config, work_dir, inputs)

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

**`KeyError: Missing required 'correction' section`**
The JSON config is missing one of the three required top-level sections
(`mission_config`, `correction`, `geolocation`).

**`KeyError: Missing required 'performance_threshold_m'`**
The `correction` section must contain `performance_threshold_m` and
`performance_spec_percent`. These encode the mission's geolocation requirement
and cannot be omitted.

**`ValidationError` on `CorrectionConfig` construction**
Pydantic will identify the offending field. Common causes: wrong types
(`sigma` must be `float`, not a string) or missing required fields
(`n_iterations` is required).

**`SPICE(PATHTOOLONG)` kernel path error**
SPICE enforces an 80-character kernel path limit. Curryer works around this
automatically. Override the temp directory if `/tmp` is unavailable:

```bash
export CURRYER_TEMP_DIR=/tmp
```

**`NotImplementedError` from `verify()`**
The `gcp_pairs=` and `observation_paths=` input modes are not yet
implemented. Use `image_matching_results=` instead.

**`geolocated_data was provided but config._image_matching_override is not set`**
`verify()` does not include a built-in image matcher. Attach a callable to
`config._image_matching_override` before calling `verify()`, or pass
pre-computed results via `image_matching_results=` instead.

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
from curryer.correction.data_structures import RegridConfig
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
from curryer.correction.data_structures import RegridConfig
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
from curryer.correction.data_structures import RegridConfig

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
