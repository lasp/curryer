# Data

Curryer includes two data directories: `data` and `tests/data`.

## Example Data (`data`)

This directory contains example data that is used in testing and example code.
These data files are generally representative of real mission data (though may be outdated)
and may be removed from the repository in the future.

### Generic Kernels

The data directory contains a subdirectory of generic data files such as SPICE kernels.
for example, it includes a modified version of the NAIF default ephemeris kernel `de430.bsp`
that has been reduced to a minimal size by limiting the time range to `[2015-01-01, 2035-01-01]` and removing
body references that we don't use often.

## Test Data (`tests/data`)

This directory contains data that is required for running unit and integration tests.
This data is not generally "real" data but contrived for the purposes of testing.
