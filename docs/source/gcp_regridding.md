# GCP Chip Regridding

Ground Control Point (GCP) reference imagery from missions such as CLARREO
Pathfinder arrives as raw HDF files (Landsat format) in which each pixel's
position is stored as Earth-Centered, Earth-Fixed (ECEF) X/Y/Z coordinates on
an irregular geodetic grid. Before these chips can be matched against L1A
science images, they must be resampled onto a regular latitude/longitude grid
that matches the spatial resolution of the mission's detector.

The `curryer.correction` package provides a complete pipeline for this:

```
HDF chip  →  ECEF → WGS84 geodetic  →  bilinear regrid  →  regular NetCDF grid
```

---

## Quick start — single file (Python)

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

---

## Batch processing — command-line script

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

### Common options

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

### Resume an interrupted run

```bash
# Already processed 60/100 — pick up from where it stopped
python examples/correction/regrid_gcp_chips.py /data/landsat_gcps/ /data/regridded/ --skip-existing
```

### Preview before committing

```bash
python examples/correction/regrid_gcp_chips.py /data/landsat_gcps/ /data/regridded/ --dry-run
```

### Non-standard band or file pattern

```bash
# Band_4 (near-IR), only files matching a date range
python scripts/regrid_gcp_chips.py /data/ /out/ \
    --pattern "LT08CHP.2016*.hdf" \
    --band Band_4 \
    --resolution 0.001 0.001
```

---

## Batch processing — Python API

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
print(f"Found {len(hdf_files)} chips")

errors = {}
for hdf_file in hdf_files:
    nc_file = output_dir / f"{hdf_file.stem}_regridded.nc"

    try:
        band, ecef_x, ecef_y, ecef_z = load_gcp_chip_from_hdf(hdf_file)
        regrid_gcp_chip(
            band,
            (ecef_x, ecef_y, ecef_z),
            config,
            output_file=str(nc_file),
            output_metadata={"source_file": hdf_file.name, "mission": "CLARREO Pathfinder"},
        )
        print(f"  ✓ {hdf_file.name}")
    except Exception as exc:
        errors[hdf_file.name] = str(exc)
        print(f"  ✗ {hdf_file.name}: {exc}")

if errors:
    print(f"\n{len(errors)} file(s) failed:", *errors, sep="\n  ")
```

### Loading the regridded output

The output NetCDF files are `ImageGrid`-compatible and plug directly into the
correction pipeline:

```python
from pathlib import Path
from curryer.correction.image_io import load_image_grid

gcp = load_image_grid(Path("regridded_chip.nc"))
# gcp.data  — 2-D radiometric values
# gcp.lat   — 2-D latitude  (regular grid, decreasing from top to bottom)
# gcp.lon   — 2-D longitude (regular grid, increasing left to right)
# gcp.h     — 2-D height above WGS84 ellipsoid (metres), or None
```

---

## Configuration reference

```python
from curryer.correction.data_structures import RegridConfig

# Resolution-based (most common)
config = RegridConfig(output_resolution_deg=(0.0009, 0.0009))

# Fixed output size
config = RegridConfig(output_grid_size=(500, 500))

# Explicit geographic extent + resolution
config = RegridConfig(
    output_bounds=(-116.5, -115.5, 38.0, 39.0),   # (minlon, maxlon, minlat, maxlat)
    output_resolution_deg=(0.001, 0.001),
)

# Disable conservative clipping (use full ECEF extent)
config = RegridConfig(
    output_resolution_deg=(0.0009, 0.0009),
    conservative_bounds=False,
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

---

## Output NetCDF structure

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

---

## Notes

- **Ellipsoid:** ECEF → geodetic conversion always uses **WGS84**
  (`curryer.compute.spatial.ecef_to_geodetic`). This is correct for
  Landsat-8/9 GCP chips.

- **HDF format:** `load_gcp_chip_from_hdf` tries HDF4 (`pyhdf`) first, then
  falls back to HDF5 (`h5py`). Both are handled transparently.

- **Memory:** a 1400 × 1400 Landsat chip uses ~30 MB RAM; 100 chips processed
  sequentially stay well within a 4 GB budget. If memory is tight, process
  files one at a time (the CLI script already does this).

- **Performance:** each chip typically takes 3–6 seconds on a single CPU core.
  100 chips ≈ 5–10 minutes.
