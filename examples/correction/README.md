# Correction Package Examples

Reference examples and configuration templates for the `curryer.correction`
package. These files are intended as a starting point when integrating the
geolocation correction and verification workflow into a new mission repository.

## Contents

| File                        | Status   | Description                                                                 |
| --------------------------- | -------- | --------------------------------------------------------------------------- |
| `README.md`                 | —        | This file                                                                   |
| `example_config.json`       | —        | Generic config template — copy and adapt for your mission                   |
| `clarreo_config.json`       | —        | CLARREO-specific config — loadable by `load_config_from_json()`             |
| `clarreo_config.py`         | —        | CLARREO config factory using the Python API; use as a reference             |
| `example_verification.py`   | Runnable | End-to-end verification demo (real CLARREO test data or synthetic fallback) |
| `example_run_correction.py` | Template | Correction loop template — exits cleanly if SPICE tools or data are missing |
| `regrid_gcp_chips.py`       | Runnable | Batch-regrid HDF GCP chips to regular lat/lon NetCDF (local files, no S3)   |

**Runnable** — executes immediately against committed test data; no additional setup required.
**Template** — documents the API pattern; requires SPICE tools and mission data not included in the repo.

## Prerequisites

Install `lasp-curryer` with test and development extras:

```bash
pip install -e ".[test,dev]"
```

All runtime dependencies (including `scipy`) are declared in `pyproject.toml`.

The verification example also requires the committed image-matching test data:

```
tests/data/clarreo/image_match/
```

## Quick Start

### Geolocation Verification

```bash
# From the repo root:
python examples/correction/example_verification.py
```

Loads real CLARREO image-matching test data from `tests/data/clarreo/image_match/`
when available, or falls back to synthetic data automatically. Demonstrates the
full verification API surface regardless of data availability.

### GCP Chip Regridding

```bash
# Regrid the committed Landsat test chip:
python examples/correction/regrid_gcp_chips.py \
    tests/data/clarreo/landsat_gcp/ \
    /tmp/regridded_chips/

# Preview without writing any files:
python examples/correction/regrid_gcp_chips.py \
    tests/data/clarreo/landsat_gcp/ /tmp/regridded_chips/ --dry-run
```

Converts raw HDF chips from ECEF coordinates to a regular lat/lon grid and
writes CF-1.8 NetCDF files. The output is loadable with
`load_image_grid(Path('...nc'))` and is directly compatible with the correction
and verification pipeline. See `docs/source/gcp_regridding.md` for a full
walkthrough.

### Correction Loop

```bash
python examples/correction/example_run_correction.py
# Or with a config file:
python examples/correction/example_run_correction.py \
    --config examples/correction/clarreo_config.json
```

The correction loop regenerates SPICE kernels at every iteration and therefore
requires:

- Preprocessed telemetry and science CSV files
- GCP `.mat` or `.nc` image-chip files
- Generic SPICE kernels in `data/generic/` (see `docs/source/users.md` for download links)
- SPICE kernel creation tools (`mkspk`, `msopck`) on `PATH` or in `bin/spice/<platform>/`

When any of these are absent the script exits in dry-run mode and prints the
API pattern without executing the loop.

## Adding Correction to a New Mission

### 1. Define a configuration

Copy `example_config.json` and fill in your mission's kernel paths, instrument
name, parameter definitions, and performance requirements. The Python factory
pattern in `clarreo_config.py` shows how to build the same config
programmatically when dynamic path resolution is needed.

```python
from curryer.correction import load_config_from_json

config = load_config_from_json("my_mission_config.json")
```

Required top-level JSON sections: `mission_config`, `correction`, `geolocation`.

### 2. Preprocess telemetry

Produce a merged telemetry CSV and a science-timing CSV using your mission's
equivalent of the CLARREO preprocessing step. The CSVs must be compatible with
`DataConfig` (see `curryer/correction/config.py` for the field contract).

### 3. Regrid GCP reference chips (if not already in NetCDF format)

```bash
python examples/correction/regrid_gcp_chips.py /path/to/hdf_chips/ /path/to/output/
```

### 4. Run the correction loop or verification

```python
from curryer.correction import load_config_from_json, run_correction, verify

config = load_config_from_json("my_mission_config.json")

# Correction parameter sweep
result = run_correction(config, work_dir, inputs)

# Standalone compliance check
verification_result = verify(config, work_dir=work_dir, image_matching_results=datasets)
print(verification_result.summary_table)
```

## Loading Regridded GCP Chips

```python
from pathlib import Path
from curryer.correction.image_io import load_image_grid

gcp = load_image_grid(Path("regridded_chip.nc"))
# gcp.data  — 2-D radiometric values
# gcp.lat   — 2-D latitude array  (degrees north)
# gcp.lon   — 2-D longitude array (degrees east)
# gcp.h     — 2-D height above WGS84 ellipsoid (metres), or None
```

## Further Reading

- **User Guide:** `docs/source/correction_user_guide.md`
- **GCP Regridding Guide:** `docs/source/gcp_regridding.md`
- **Verification Guide:** `docs/verification_guide.md`
- **API Reference:** `curryer/correction/__init__.py` (`__all__` list)
- **Config models:** `curryer/correction/config.py`
- **CLARREO test fixtures:** `tests/test_correction/clarreo/`
