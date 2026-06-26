# User Documentation (Getting Started)

# Installation

### Install System Dependencies

- GDAL Libraries (e.g. `libgdal-dev` for apt)
- SPICE Toolkit

The SPICE toolkit can be installed from conda-forge for AMD or ARM:

```
conda config --add channels conda-forge
conda install cspice
```

or from NAIF: https://naif.jpl.nasa.gov/naif/toolkit_C.html

### Install Python Package

```shell
pip install lasp-curryer
```

# Basic Usage

```python
# Example code goes here
```

### Data / Binary Files

_NOTE: Data files and precompiled binaries are not currently automated and thus
require manual downloading. This will be addressed in the next major release._

Download from the Curryer repo:

- `data/generic` - Generic spice kernels (e.g., leapsecond kernel)
  - Download
- `data/<mission>` - Mission specific kernels and/or kernel definitions.

Define the top-level directory using the environment variable `CURRYER_DATA_DIR`
or pass the path to routines which require data files.

Download Third-party Files:

- SPICE Utilities: https://naif.jpl.nasa.gov/naif/utilities.html
  - At minimum: `mkspk`, `msopck`, `brief`, `ckbrief`
- SPICE Generic Kernels (large):
  - [de430.bsp](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp),
    place in `data/generic`.
- PyProj Data:
  - Data directory: `import pyproj; print(pyproj.datadir.get_user_data_dir())`
  - [EGM96 TIFF](https://cdn.proj.org/us_nga_egm96_15.tif)
- (OPTIONAL, USED FOR TESTING) `data/gmted` - Digital Elevation Model (DEMs) with global coverage at
  15-arc-second.
  - Alternatively, use the script [download_dem.py](bin/download_dem.py) to
    download different types and/or resolutions from the USGS.

## Examples

### SPICE Extensions

Time conversion:

```python
from curryer import spicetime

print(spicetime.adapt(0, from_='ugps', to='iso'))
# 1980-01-06 00:00:00.000000

print(spicetime.adapt('2024-11-13', 'iso'))
# 1415491218000000

print(spicetime.adapt(1415491218000000, to='et'))
# 784728069.1827033

import numpy as np

print(repr(spicetime.adapt(np.arange(4) * 60e6 + 1415491218000000, to='dt64')))
# array(['2024-11-13T00:00:00.000000', '2024-11-13T00:01:00.000000',
#        '2024-11-13T00:02:00.000000', '2024-11-13T00:03:00.000000'],
#       dtype='datetime64[us]')
```

Abstractions:

```python
from curryer import spicierpy

spicierpy.ext.infer_ids('ISS', 25544, from_norad=True)
# {'mission': 'ISS',
#  'spacecraft': -125544,
#  'clock': -125544,
#  'ephemeris': -125544,
#  'attitude': -125544000,
#  'instruments': {}}

earth = spicierpy.obj.Body('Earth')
print(earth, earth.id)
# Body(EARTH) 399

import curryer

mkrn = curryer.meta.MetaKernel.from_json(
    'data/tsis1/tsis_v01.kernels.tm.json', sds_dir='data/generic', relative=True
)
print(mkrn)
# MetaKernel(Spacecraft(ISS_SC), Body(ISS_ELC3), Body(ISS_EXPA35), Body(TSIS_TADS),
#   Body(TSIS_AZEL), Body(TSIS_TIM), Body(TSIS_TIM_GLINT))

with spicierpy.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels]):
    print(spicierpy.ext.instrument_boresight('TSIS_TIM'))
# [0. 0. 1.]

mkrn = curryer.meta.MetaKernel.from_json(
    'tests/data/clarreo/cprs_v01.kernels.tm.json', sds_dir='data/generic', relative=True
)
print(mkrn)
# MetaKernel(Spacecraft(ISS_SC), Body(CPRS_BASE), Body(CPRS_PEDE),
#   Body(CPRS_AZ), Body(CPRS_YOKE), Body(CPRS_EL), Body(CPRS_HYSICS))

with spicierpy.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels]):
    print(curryer.compute.spatial.get_instrument_kernel_pointing_vectors('CPRS_HYSICS'))
# (480,
#  array([[ 0.00173869, -0.08715574,  0.99619318],
#         [ 0.0017315 , -0.08679351,  0.99622482],
#         [ 0.00172431, -0.08643127,  0.99625632],
#         ...,
#         [-0.00171712,  0.08606901,  0.9962877 ],
#         [-0.00172431,  0.08643127,  0.99625632],
#         [-0.0017315 ,  0.08679351,  0.99622482]]))
```

### SPICE Kernel Creation

Create CLARREO Dynamic Kernels:

```python
import curryer

meta_kernel = 'tests/data/clarreo/cprs_v01.kernels.tm.json'
generic_dir = 'data/generic'
kernel_configs = [
    'data/clarreo/iss_sc_v01.ephemeris.spk.json',
    'data/clarreo/iss_sc_v01.attitude.ck.json',
    'data/clarreo/cprs_az_v01.attitude.ck.json',
    'data/clarreo/cprs_el_v01.attitude.ck.json',
]
output_dir = '/tmp'
input_file_or_obj = 'tests/data/demo/cprs_geolocation_tlm_20230101_20240430.nc'

# Load meta kernel details. Includes existing static kernels.
mkrn = curryer.meta.MetaKernel.from_json(meta_kernel, relative=True, sds_dir=generic_dir)

# Create the dynamic kernels from the JSONs alone. Note that they
# contain the reference to the input_data netcdf4 file to read.
generated_kernels = []
creator = curryer.kernels.create.KernelCreator(overwrite=False, append=False)

# Generate the kernels from the config and input data (file or object).
for kernel_config in kernel_configs:
    generated_kernels.append(creator.write_from_json(
        kernel_config, output_kernel=output_dir, input_data=input_file_or_obj,
    ))

```

### Level-1 Geospatial Processing

Geolocate CLARREO HYSICS Instrument:

```python
import pandas as pd
import curryer

meta_kernel = 'tests/data/clarreo/cprs_v01.kernels.tm.json'
generic_dir = 'data/generic'

time_range = ('2023-01-01', '2023-01-01T00:05:00')
ugps_times = curryer.spicetime.adapt(pd.date_range(*time_range, freq='67ms', inclusive='left'), 'iso')

# Load meta kernel details. Includes existing static kernels.
mkrn = curryer.meta.MetaKernel.from_json(meta_kernel, relative=True, sds_dir=generic_dir)

# Geolocate all the individual pixels and create the L1A data product!
with curryer.spicierpy.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels]):
    geoloc_inst = curryer.compute.spatial.Geolocate('CPRS_HYSICS')
    l1a_dataset = geoloc_inst(ugps_times)
    l1a_dataset.to_netcdf('cprs_geolocation_l1a_20230101.nc')

```

_Assumes dynamic kernels have been created and their file names defined within
the metakernel JSON file._

### Geometry Ancillary Fields

{py:class}`~curryer.compute.geometry.GeometryData` computes the geolocation/geometry
ancillary fields a Level-1 product needs — sub-satellite / sub-solar point, satellite
radius and altitude, Earth-Sun distance (and, with attitude kernels, viewing/solar
angles). You request any subset of fields by name and it queries the minimal set of
SPICE inputs, each exactly once. The {py:mod}`curryer.compute.geometry` API reference
documents every field, leaf function, and accessor.

```python
import pandas as pd
import curryer
from curryer.compute import geometry

meta_kernel = 'tests/data/clarreo/cprs_v01.kernels.tm.json'
generic_dir = 'data/generic'
mkrn = curryer.meta.MetaKernel.from_json(meta_kernel, relative=True, sds_dir=generic_dir)

time_range = ('2023-01-01', '2023-01-01T00:05:00')
ugps_times = curryer.spicetime.adapt(pd.date_range(*time_range, freq='1s', inclusive='left'), 'iso')

# Construct with the observing body (spacecraft or instrument). Kernels must be
# furnished for the requested times -- GeometryData does not load them itself.
geo = geometry.GeometryData('CPRS_HYSICS')
with curryer.spicierpy.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels]):
    df = geo.get_geometry(ugps_times, fields=['subsatellite', 'sc_radius', 'earth_sun_distance'])

print(df.columns.tolist())
# ['subsatellite_latitude', 'subsatellite_longitude', 'subsatellite_colatitude',
#  'spacecraft_radius', 'earth_sun_distance']
```

For typed, per-field arrays instead of a flat table, use
{py:meth}`~curryer.compute.geometry.GeometryData.get_vectors`:

```python
with curryer.spicierpy.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels]):
    vectors = geo.get_vectors(ugps_times, fields=['sc_position', 'subsatellite'])

vectors['sc_position'].shape   # (N, 3) ECEF position, km
vectors['subsatellite'].shape  # (N, 3) latitude / longitude / colatitude
```

Things worth knowing:

- **Times** are uGPS (`int64` microseconds since 1980-01-06); convert with
  `curryer.spicetime.adapt(...)`. The times you pass are evaluated exactly — no
  resampling or interpolation.
- **Observer** is the only mission input. Position-derived fields work for any body;
  attitude fields (e.g. `boresight`) need an instrument with a defined FOV.
- **Discover fields** with
  {py:meth}`~curryer.compute.geometry.GeometryData.available_fields`; the table below
  (and the {py:mod}`module docstring <curryer.compute.geometry>`) lists each field and
  the columns it expands to.
- **Default** — {py:meth}`~curryer.compute.geometry.GeometryData.get_geometry` with no
  `fields` returns the ephemeris-only set, valid for any observer (it skips the
  per-sample attitude loop).
- **Coverage gaps** surface as `NaN` (rows are never dropped); a provider that returns
  all-NaN logs a warning — usually an unfurnished kernel. Pass `allow_nans=False` to
  raise instead.
- **Typed access** — {py:meth}`~curryer.compute.geometry.GeometryData.get_vectors`
  returns `{field: (N, k) ndarray}` addressed by field name rather than column strings.

| Field                | Columns                                              | Meaning                                      |
| -------------------- | ---------------------------------------------------- | -------------------------------------------- |
| `subsatellite`       | `subsatellite_latitude`, `_longitude`, `_colatitude` | Ground point beneath the spacecraft          |
| `subsolar`           | `subsolar_latitude`, `_longitude`, `_colatitude`     | Ground point beneath the Sun                 |
| `sc_radius`          | `spacecraft_radius`                                  | Geocentric distance from Earth's center (km) |
| `sc_altitude`        | `spacecraft_altitude`                                | Geodetic height above the ellipsoid (km)     |
| `earth_sun_distance` | `earth_sun_distance`                                 | Earth-Sun distance (AU)                      |
| `sc_position`        | `spacecraft_position_x` / `_y` / `_z`                | Spacecraft position in ECEF (km)             |

**Composing custom fields.** The registry is built on pure, SPICE-free leaf functions
— {py:func}`~curryer.compute.geometry.subobserver_point`,
{py:func}`~curryer.compute.geometry.sc_radius`,
{py:func}`~curryer.compute.geometry.satellite_altitude`,
{py:func}`~curryer.compute.geometry.colatitude`, and
{py:func}`~curryer.compute.geometry.earth_sun_distance` — that take positions as
arguments. Call them directly to derive fields the registry does not expose yet.

Additional attitude-derived fields (instrument boresight, viewing/solar zenith and
azimuth, cone angle) are added by the later geometry field groups and require an
instrument FOV plus attitude (CK) coverage.

---

## SPICE Path Length Handling

Curryer automatically handles SPICE's 80-character path limit using a simple two-strategy approach:

1. **Symlink** (always tried first—zero overhead, no copying)
2. **File copy** to temp directory (bulletproof fallback if symlink fails)

No configuration needed for most users. Temp files are automatically cleaned up after kernel generation.

### Configuration Options

```bash
# Custom temp directory (default: /tmp on Unix, auto-detected on Windows)
export CURRYER_TEMP_DIR="/tmp"

# AWS/Cloud: Disable file copying to avoid storage costs
export CURRYER_DISABLE_COPY="true"
```

### How It Works

When kernel paths exceed 80 characters:

```
INFO: Path exceeds 80 chars (102 chars): /very/long/path.../naif0012.tls
INFO:   → Using symlink: /tmp/curryer_naif0012.tls
```

Or if symlinks fail:

```
INFO: Path exceeds 80 chars (102 chars): naif0012.tls
INFO:   → Using copy: /tmp/curryer_abc12345.tls
```

See [SPICE Path Handling Documentation](../spice_path_handling.md) for more details.
