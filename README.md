# Curryer

A library for SPICE extensions and geospatial data product generation.

- Github: https://github.com/lasp/curryer
- PyPi: https://pypi.org/project/lasp-curryer/

## Core Features

- Extensions and wrappers for SPICE routines and common data patterns.
- Automation of SPICE kernel creation from JSON definition files and modern data
  file formats and third-party data structures.
- Level-1 geospatial data processing routines (e.g., geolocation).

## Install

```shell
pip install lasp-curryer
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
  - At minimum: `mkspk`, `msopck`, `brief`, `ckbreif`
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
    print(curryer.compute.spatial.pixel_vectors('CPRS_HYSICS'))
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
