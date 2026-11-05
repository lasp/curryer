# Correction & Verification User Guide

`curryer.correction` measures and improves a mission's geolocation against ground control
points (GCPs). It answers two questions:

- **Verification**: does the geolocation meet the accuracy requirement? It returns a pass/fail
  verdict, with the error at every GCP.
- **Correction**: which pointing or timing adjustment makes the geolocation more accurate? It
  tries a sweep of candidate adjustments and reports the one that fits the GCPs best.

Both are driven by one configuration file and a short call; a mission writes no classes. Both
measure error the same way: an observation subimage is matched to a GCP chip, and the shift that
aligns them is the error. Correction runs that measurement once per candidate, after
re-geolocating the observation with SPICE:

```
verification                          correction
                                      for each candidate adjustment:
                                          write the candidate's kernels
                                          re-geolocate each observation (SPICE + DEM)
match each observation to its chip        match each observation to its chip
error statistics -> verdict           compare candidates -> best adjustment + verdict
```

This guide is generic. A mission supplies its own data access and command lines on top of it,
in its own repository.

## Contents

1. [Concepts](#concepts)
2. [Quickstart](#quickstart)
3. [Configuration](#configuration)
4. [Input contracts](#input-contracts)
5. [Verification](#verification)
6. [Correction](#correction)
7. [Configuration reference](#configuration-reference)
8. [Adding a mission](#adding-a-mission)
9. [Troubleshooting](#troubleshooting)
10. [Reference examples](#reference-examples)

---

## Concepts

### GCP pairs

A **GCP chip** is a reference image of a patch of ground with known geolocation, such as a
Landsat chip regridded to latitude and longitude ([GCP chip regridding](gcp_regridding.md)).
An **observation subimage** is the mission's image of the same ground: rows are frames (time),
columns are cross-track detector pixels. A **GCP pair** is one subimage with the chip it lies in.

### Image matching

The chip is convolved with the instrument's point-spread function (PSF), so it looks as the
instrument would see it. A correlation search then shifts the subimage over the chip on a grid
that shrinks each iteration (`setup.search`), keeping the shift with the highest correlation.
The result is:

- the error in latitude and longitude: the subimage's geolocation minus the chip's, at the match;
- `correlation`: the correlation of the best match;
- `correlation_secondary`: the best first-grid correlation farther than
  `search.peak_exclusion_km` from the peak. The **peak margin** is `correlation` minus this.

The search samples only the part of the PSF-convolved chip where the PSF lies wholly inside the
chip, so half a PSF kernel at each edge is not searchable. A shift that would sample outside that
part raises rather than correlate against zero fill. Every search reaches the first grid's
half-width, `grid_size // 2` steps of `grid_span_km / (grid_size - 1)` (5.6 km with the defaults),
beyond the subimage, so a subimage needs that margin plus its expected error inside the chip.
`search.max_shift_km()` is the worst case, the best point on the grid's edge in every iteration
(28.1 km with the defaults); that margin always suffices but is rarely needed.

### Match-quality gates and statuses

A match can be meaningless: the chip may be clouded over, featureless, or changed since it was
imaged. Two gates on `setup.geo` decide whether a match is a measurement:

- `minimum_correlation`: rejects a match whose `correlation` is lower;
- `minimum_peak_margin`: rejects a match whose peak is not distinct, i.e. one whose `correlation`
  exceeds `correlation_secondary` by less than this.

Each matched GCP gets a status:

| Status     | Meaning                                                                       |
| ---------- | ----------------------------------------------------------------------------- |
| `pass`     | Passed the gates; nadir-equivalent error below `performance_threshold_m`      |
| `fail`     | Passed the gates; error at or above the threshold                             |
| `rejected` | Failed a gate; `rejection_reason` names the gate, the value and the threshold |

Rejected GCPs keep their errors so they can be reviewed, but they are not measurements: they are
left out of every statistic.

### Errors

Every error is **geolocated minus true**. It is reported three ways:

- **Latitude and longitude** (`lat_error_deg`, `lon_error_deg`): positive is north and east.
- **Along- and cross-track** (`along_track_error_m`, `cross_track_error_m`): along the ground
  track (the direction of increasing frame time at the GCP) and 90° clockwise from it. A
  positive along-track error means the geolocation is ahead of the GCP along the track. A timing
  error shows up along track; a roll error across track.
- **Nadir-equivalent** (`nadir_equiv_error_m`): the error scaled to what it would be at nadir,
  so GCPs viewed off nadir count fairly against one requirement. The error is split into its
  components in the view plane (containing the line of sight and nadir) and across it. Each is
  divided by a scale factor that grows with the off-nadir angle of the line of sight of the pixel
  that viewed the GCP. The test suite checks it against a MATLAB reference implementation, to
  within 2 mm.

The **requirement** is more than `performance_spec_percent` of accepted GCPs within
`performance_threshold_m`, nadir-equivalent. A run passes when the percentage of accepted GCPs
below the threshold exceeds the spec; a run exactly at the spec fails, so the spec must be below 100. With no accepted GCP nothing was measured, and the run fails with `percent_within_threshold`
`None`.

### Quality weighting

Each accepted GCP gets a `quality_weight`, the match signal-to-noise ratio `ρ² / (1 − ρ²)` of its
correlation `ρ` (`error_stats.match_snr_weight`). If the observed image is the emulated scene plus
independent noise, `ρ²` is the scene's share of the variance. The variance of a
correlation-registration error scales as noise over signal, so this is an inverse-variance
weight. `ρ` is capped at 0.99 (weight 49), so one near-perfect match cannot dominate; rejected
GCPs weigh 0.

| ρ      | 0.80 | 0.90 | 0.95 | 0.99 |
| ------ | ---- | ---- | ---- | ---- |
| weight | 1.8  | 4.3  | 9.3  | 49   |

Weighted statistics are reported next to the unweighted ones and never decide pass or fail.
`effective_measurements = (Σw)² / Σw²` is the number of equally good GCPs the weighted set is
worth: when it is much smaller than the accepted count, a few GCPs carry the weighted statistics.

### The correction loop

The **parameters** of a correction are the adjustments it may make: a rotation angle of a
fixed-attitude frame, a bias on a telemetry angle, or a time offset (see
[Parameter types](#parameter-types--ptype)). A **sweep** gives each parameter a nominal value
and bounds, and a **strategy** picks the candidate parameter sets: a grid, one parameter at a
time, or random draws (Monte Carlo).

For every candidate, the loop re-geolocates each subimage with SPICE and the DEM and matches it
to its chip. Candidates are compared on the same GCP pairs:

- A pair is **usable** when it passes the gates under at least one candidate.
- A candidate is **valid** when it passes the gates on every usable pair.
- The **best** candidate is the valid one with the lowest mean, over usable pairs, of each pair's
  RMS error.

One candidate that passes a hard pair, even on a spurious peak, makes that pair usable and so
invalidates every candidate that fails it; `n_valid_sets` shows when this has narrowed the
comparison to a few candidates.

The loop raises if no pair is usable or no candidate is valid. The best candidate is one of the
candidates tried; the loop does not interpolate between them, so its precision is the grid step
or the spread of the random draws.

Three consequences to plan a sweep around:

- **Degenerate parameters.** Parameters that move the image the same way cannot be separated by
  GCPs that see them the same way. For example, a pitch rotation and a time offset both move a
  near-nadir image along track, so the fit finds a valley of equal error rather than a point.
  Separating them needs GCPs at different viewing geometries: a rotation moves a pixel in
  proportion to slant range, a time offset does not.
- **Bounds limit the subimage.** The search-reach rule applies under every candidate, so a
  subimage must lie inside its chip by the search reach plus the largest shift the sweep can
  cause there. Wider bounds mean smaller subimages; a pair with no room must be dropped from
  the inputs, or the loop raises.
- **Fail loud.** A pixel the kernels cannot geolocate, a candidate that moves a subimage off its
  chip, or an `OFFSET_KERNEL` field missing from the telemetry stops the run; nothing is skipped
  silently.

---

## Quickstart

Both workflows read the same configuration file. Copy `examples/correction/example_config.json`,
edit the values for the mission, and run.

**Check accuracy:**

```python
from curryer.correction import load_setup_from_json, verify

setup = load_setup_from_json("mission.json")
result = verify(setup, gcp_pairs=[("obs_001.nc", "chip_001.nc")], los_file="los.mat", psf_file="psf.mat")

print(result.summary_table)  # one row per GCP, with the verdict
print("Passed:", result.passed)
```

**Find a correction:**

```python
from curryer.correction import CorrectionInput, load_config_files, run_correction

setup, sweep, output = load_config_files("mission.json")
inputs = [CorrectionInput(telemetry_file="telemetry.nc", science_file="obs_001.nc", gcp_file="chip_001.nc")]

result = run_correction(setup, sweep, inputs, work_dir="out", output=output)

print(result.summary_table)
print(result.best_parameter_set, result.best_rms_m)
print(result.recommendation)
```

The data a run processes (observations and GCP chips) is passed at call time, not stored in the
configuration: it belongs to the run, not the mission.

---

## Configuration

A mission's configuration is one JSON file with up to three sections. Every value is validated
when it loads: a wrong type, an unknown key or an out-of-range value raises, naming the field.

| Section  | Holds                                                                | Changes          |
| -------- | -------------------------------------------------------------------- | ---------------- |
| `setup`  | Kernels, instrument, requirement, gates, image matching, calibration | Once per mission |
| `sweep`  | The parameters to vary, their bounds, and the strategy               | Per experiment   |
| `output` | NetCDF output name and metadata (optional)                           | Rarely           |

```json
{
  "setup": {
    "geo": {
      "meta_kernel_file": "path/to/mission.kernels.tm.json",
      "generic_kernel_dir": "data/generic",
      "instrument_name": "YOUR_INSTRUMENT",
      "time_field": "corrected_timestamp",
      "dynamic_kernels": [],
      "minimum_correlation": 0.8
    },
    "requirements": {
      "performance_threshold_m": 250.0,
      "performance_spec_percent": 39.0
    },
    "search": {
      "grid_size": 11,
      "grid_span_km": 8.0,
      "reduction_factor": 0.5,
      "spacing_limit_m": 25.0
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
          "field": "angle_x",
          "current_value": 0.0,
          "bounds": [-300.0, 300.0],
          "sigma": 50.0,
          "units": "arcseconds"
        }
      },
      {
        "ptype": "CONSTANT_KERNEL",
        "config_file": "path/to/frame_a.attitude.ck.json",
        "spec": {
          "field": "angle_y",
          "current_value": 0.0,
          "bounds": [-300.0, 300.0],
          "sigma": 50.0,
          "units": "arcseconds"
        }
      },
      {
        "ptype": "CONSTANT_KERNEL",
        "config_file": "path/to/frame_a.attitude.ck.json",
        "spec": {
          "field": "angle_z",
          "current_value": 0.0,
          "bounds": [0.0, 0.0],
          "sigma": null,
          "units": "arcseconds"
        }
      },
      {
        "ptype": "OFFSET_TIME",
        "spec": {
          "field": "time_offset",
          "current_value": 0.0,
          "bounds": [-0.3, 0.3],
          "sigma": 0.1,
          "units": "seconds"
        }
      }
    ]
  },
  "output": { "output_filename": "correction_results.nc" }
}
```

`load_config_files("mission.json")` returns `(setup, sweep, output)`; a missing `setup` or `sweep`
raises `KeyError`, and `output` defaults to empty. `load_setup_from_json` and
`load_sweep_from_json` read one section each, so a mission can keep one setup file and many sweep
files.

**A frame rotation** is three `CONSTANT_KERNEL` parameters sharing a `config_file`, one per axis:
`angle_x` (roll), `angle_y` (pitch) and `angle_z` (yaw). The loop writes the frame's CK from the
three angles. A sweep missing an axis, or naming one twice, raises.

**Holding a parameter.** Zero-width bounds `[b, b]` hold a parameter at `current_value + b` in
every strategy. In the example above, yaw is held at its nominal value.

**Varying a sweep in code.** `sweep.with_strategy(...)` and `sweep.update_param(...)` return
validated copies:

```python
grid = sweep.with_strategy("grid", grid_points_per_param=5)
wider = sweep.update_param("angle_x", bounds=[-600.0, 600.0])
```

The `update_param` selector is the parameter's position, its `spec.field`, or its `config_file`
stem.

---

## Input contracts

The inputs are checked where they enter, and a violation raises naming the file and the problem.
Nothing is repaired or defaulted.

### Observation files for verification (`gcp_pairs`, `observation_paths`)

| Variable                        | Contract                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `band_data`                     | Radiance, (frame, pixel): rows are frames, the middle row is the mid-frame                                   |
| `lat`, `lon`                    | Per pixel, degrees (`latitude` / `longitude` also read); `h` (metres) optional                               |
| `position`                      | Spacecraft position at the mid-frame, ECEF, metres; a missing, non-finite or Earth-interior value raises     |
| `detector_pixel` (column crops) | One 0-based LOS-table row per column, strictly increasing; without it the LOS table needs one row per column |

`.mat` observations are also read, with the position in `R_ISS_midframe`.

The GCP chip's centre must lie inside the observation grid. The line of sight used for the
nadir-equivalent error is that of the mid-frame pixel in the column nearest the chip centre.

### Pre-computed matches (`image_matching_results`)

A list of `xr.Dataset` objects with a `measurement` dimension. The spacecraft-state variable names
are set on the setup.

| Variable                       | Dimension(s)                      | Description                                                                                          |
| ------------------------------ | --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `lat_error_deg`                | `[measurement]`                   | Latitude error, degrees, positive north                                                              |
| `lon_error_deg`                | `[measurement]`                   | Longitude error, degrees, positive east                                                              |
| `gcp_lat_deg`, `gcp_lon_deg`   | `[measurement]`                   | GCP centre, degrees                                                                                  |
| `gcp_alt`                      | `[measurement]`                   | GCP altitude, metres                                                                                 |
| `<spacecraft_position_name>`   | `[measurement, xyz]`              | Spacecraft position, ITRF93 (Earth-fixed), metres                                                    |
| `<boresight_name>`             | `[measurement, xyz]`              | Line of sight of the pixel that viewed the GCP, unit vector, instrument frame                        |
| `<transformation_matrix_name>` | `[measurement, xyz_from, xyz_to]` | Rotation from the instrument frame to ITRF93                                                         |
| `correlation`                  | `[measurement]`                   | Required when `minimum_correlation` or `minimum_peak_margin` is set (also read as `ccv` or `im_ccv`) |
| `correlation_secondary`        | `[measurement]`                   | Required when `minimum_peak_margin` is set                                                           |
| `track_azimuth_deg`            | `[measurement]`                   | Ground-track azimuth at the GCP, degrees; without it the along- and cross-track errors are `None`    |

### Correction inputs (`CorrectionInput`)

| Field            | Contract                                                                                |
| ---------------- | --------------------------------------------------------------------------------------- |
| `telemetry_file` | Telemetry the dynamic kernels are built from, read with `setup.data_config.file_format` |
| `science_file`   | Observation subimage, NetCDF (below)                                                    |
| `gcp_file`       | GCP chip, NetCDF (`.mat` accepted)                                                      |

The subimage holds:

- `band_data` (frame, pixel): finite radiance;
- `ugps` (frame): strictly increasing integer frame times, microseconds since the GPS epoch;
- `detector_pixel` (pixel): the LOS-table row of each column; without it the columns must be the
  table's rows in order.

The loop ignores any latitude and longitude the subimage carries and re-geolocates it. The
subimage must lie inside its chip by the search reach plus the largest shift the sweep can cause
([The correction loop](#the-correction-loop)). Paths may be `s3://` URIs when `boto3` is
installed.

### Calibration

The LOS table (`los_vectors_file`, per-detector line-of-sight unit vectors) and the PSF
(`psf_file`) may be `.mat` files. Verification takes them as `los_file=` / `psf_file=`; the
correction loop reads `setup.calibration`. The loop requires the LOS table, since it
re-geolocates each pixel with it, and the built-in matching requires the PSF.

---

## Verification

### Running it

`verify(setup, ...)` takes one input mode, keyword-only. If several are given, the first in this
order is used:

| Mode                 | Arguments                                                                                               | Use                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Pre-computed matches | `image_matching_results=`                                                                               | A mission's own matching pipeline; needs no SPICE                                                 |
| Geolocated dataset   | `geolocated_data=` with `setup.image_matching_func`, or with `gcp_directory=`, `los_file=`, `psf_file=` | Matching a whole geolocated dataset, cropped inside the chip (`detector_pixel` for a column crop) |
| Explicit file pairs  | `gcp_pairs=`, `los_file=`, `psf_file=`                                                                  | Observation subimages already paired with their chips                                             |
| Auto-paired files    | `observation_paths=`, `gcp_directory=`, `los_file=`, `psf_file=`                                        | Pairing by spatial overlap                                                                        |

With no mode, or a mode missing what it needs, `verify()` raises `ValueError`.

### The result

```python
result = verify(setup, gcp_pairs=pairs, los_file=los, psf_file=psf)

print(result.summary_table)
print(result.passed, result.percent_within_threshold, result.weighted_percent_within_threshold)
for err in result.per_gcp_errors:
    print(err.gcp_index, err.status, err.nadir_equiv_error_m, err.along_track_error_m, err.rejection_reason)
```

| Attribute                           | Contents                                                                                     |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
| `passed`                            | The verdict                                                                                  |
| `percent_within_threshold`          | Accepted GCPs within the threshold, %; `None` when nothing was measured                      |
| `weighted_percent_within_threshold` | The quality-weighted percentage; information only                                            |
| `per_gcp_errors`                    | One `GCPError` per matched GCP: errors, status, reason, correlation, off-nadir angle, weight |
| `aggregate_stats`                   | `xr.Dataset`: per-GCP errors with every intermediate term, the statistics as attributes      |
| `warnings`, `summary_table`         | Text for people                                                                              |

`aggregate_stats` holds `vp_error_m` and `xvp_error_m`, the scale factors, the nadir-equivalent
components and total, and `off_nadir_angle_deg`. Its attributes include `total_measurements`
(accepted GCPs), `n_matched`, `n_accepted`, `n_rejected`, and the mean, RMS, percentiles and
`percent_below_<N>m` for 100–1000 m, with their weighted versions.

### Saving, reviewing and re-applying

The gates are automatic, but a person should look at every match before a result is reported. A
cloud edge can correlate well, and a good match can narrowly miss a gate. In the file-pair modes,
`keep_images=True` keeps the images behind each match, and `save_verification` writes everything
a review needs:

```python
from curryer.correction import apply_review, load_verification, read_review_decisions, save_verification

result = verify(setup, gcp_pairs=pairs, los_file=los, psf_file=psf, keep_images=True)
save_verification(result, Path("verification/run1"))
```

| File                 | Contents                                                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `result.json`        | The `VerificationResult`, without `aggregate_stats` and the images; written last, so a directory without it is an incomplete save          |
| `summary.csv`        | One row per GCP: status, reason, errors, correlations, off-nadir angle, quality weight, and an empty `review` column                       |
| `aggregate_stats.nc` | `aggregate_stats`                                                                                                                          |
| `chips/gcp_NNNN.nc`  | Per GCP: `observed`, `emulated` (the PSF-blurred chip at the match), `reference`, `reference_convolved`, with their latitude and longitude |

`save_verification` refuses a directory that already holds a result. CSV and NetCDF read directly
in MATLAB (`readtable`, `ncread`; `ncread` returns 2-D arrays transposed) and Python.

A reviewer puts `accept` or `reject` in the `review` column; a blank cell keeps the automatic
status. Applying the decisions recomputes the statistics without re-matching:

```python
saved = load_verification(Path("verification/run1"))
reviewed = apply_review(saved, read_review_decisions(Path("verification/run1/summary.csv"), saved))
save_verification(reviewed, Path("verification/run1_reviewed"))
```

`accept` brings a GCP into the statistics (status `pass` or `fail` by its error); `reject` takes it
out (status `rejected`, reason `rejected in review`). Quality weights are recomputed, so a
low-correlation GCP accepted in review counts fully in the unweighted statistics but little in the
weighted ones. `read_review_decisions` raises on a value other than `accept`, `reject` or blank,
and on a duplicated `gcp_index`.

curryer has no plotting dependency; review figures belong to the mission. Figures that have
worked well:

- **Chip panel**, one per GCP: `observed`, the PSF-blurred chip sampled at the observation's own
  latitude and longitude (`search.emulate_image` on `reference_convolved`), and `emulated`. Side
  by side, they show the misregistration and, for a rejected GCP, why. Title it with the status
  and reason.
- **Map**, one per observation: a box around each chip coloured by status.
- **Error scatter**: along-track against cross-track error, rejected GCPs hollow, with a circle at
  the threshold.

### Before and after

`compare_results(before, after)` returns a text comparison of two verification results, such as
before and after a correction is applied.

---

## Correction

### Running it

`run_correction(setup, sweep, inputs, work_dir, output=None)` runs the sweep and returns a
`CorrectionResult`. Inputs are `CorrectionInput` objects or `(telemetry, science, gcp)` tuples.

For each input, the loop builds the dynamic kernels from the telemetry once. Then, for each
candidate parameter set, it:

1. writes the parameter kernels (`CONSTANT_KERNEL`, `OFFSET_KERNEL`) and shifts the frame times
   by any `OFFSET_TIME`;
2. re-geolocates the subimage's frames and detector pixels with SPICE, using the LOS-table rows
   named by `detector_pixel`, and terrain-corrects them with the DEM in `setup.geo.dem_data_dir`;
3. matches the subimage's radiance on that grid to its chip (`match_observation`, or
   `setup.observation_matching_func`).

### The result

| Attribute                                        | Contents                                                                                           |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `best_parameter_set`, `best_rms_m`, `best_index` | The best valid candidate, its mean per-pair RMS, and its index in the NetCDF                       |
| `met_threshold`                                  | Whether the best candidate meets the requirement, on the usable pairs                              |
| `n_parameter_sets`, `n_gcp_pairs`                | Candidates tried and pairs run                                                                     |
| `n_usable_pairs`, `n_valid_sets`                 | Pairs the comparison rests on, and candidates compared                                             |
| `all_parameter_sets`                             | Every candidate with its mean, best-pair and worst-pair RMS, sorted by mean RMS                    |
| `recommendation`, `summary_table`                | Text for people, including the basis of the verdict: how many pairs and candidates stood behind it |
| `netcdf_path`                                    | The saved NetCDF                                                                                   |

Read the best candidate against `n_usable_pairs` and `n_valid_sets`: a best set chosen on one pair,
or from a handful of valid candidates, is weak evidence.

The NetCDF (`output.output_filename` in `work_dir`) records, per candidate and pair, the errors
(`rms_error_m`, `mean_error_m`, `max_error_m`, `std_error_m`), the match (`im_lat_error_km`,
`im_lon_error_km`, `im_ccv`, `im_grid_step_m`) and `accepted`; per candidate, the parameter values,
`mean_rms_all_pairs`, `best_pair_rms`, `worst_pair_rms` and `valid`. An invalid candidate's
`mean_rms_all_pairs` is NaN. When the selection raises, the NetCDF is saved first, so a failed
sweep can still be inspected.

### Choosing a strategy

| Strategy        | Candidates                                                                                                                             | Use                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `GRID_SEARCH`   | Every combination of `grid_points_per_param` values across each parameter's bounds; more than `max_grid_sets` combinations raises      | Mapping the error surface, e.g. a degenerate valley |
| `SINGLE_OFFSET` | Each parameter varied alone over `n_iterations` values, the others at nominal                                                          | A first look at each parameter                      |
| `RANDOM`        | `n_iterations` draws, each offset normal about nominal with the parameter's `sigma`, clipped to its bounds; `seed` for reproducibility | Searching a wide space (Monte Carlo)                |

A sigma comparable to the bounds piles draws on the bounds. To refine, run again with narrower
bounds around the best candidate.

### Low-level: `loop()`

`loop(setup, sweep, work_dir, inputs)` returns the raw `(results, netcdf_data)`. It accepts only
`(str, str, str)` tuples, and `work_dir` comes before the inputs.

---

## Configuration reference

### Setup — `setup`

| Field                        | Type                       | Notes                                                            |
| ---------------------------- | -------------------------- | ---------------------------------------------------------------- |
| `geo`                        | `GeolocationConfig`        | **Required.** Kernels, instrument, gates                         |
| `requirements`               | `RequirementsConfig`       | **Required.** The accuracy requirement                           |
| `data_config`                | `DataConfig \| None`       | Telemetry reader; `None` uses CSV                                |
| `calibration`                | `CalibrationFiles \| None` | LOS table and PSF for the correction loop                        |
| `psf_sampling`               | `PSFSamplingConfig`        | PSF sample spacing; defaults suit 30 m Landsat chips             |
| `search`                     | `SearchConfig`             | Correlation search grid; defaults suit 30 m Landsat chips        |
| `spacecraft_position_name`   | `str`                      | Variable name of the spacecraft position in pre-computed matches |
| `boresight_name`             | `str`                      | Variable name of the line of sight in pre-computed matches       |
| `transformation_matrix_name` | `str`                      | Variable name of the rotation matrix in pre-computed matches     |
| `image_matching_func`        | `Callable \| None`         | Custom matching for `verify(geolocated_data=...)`; not in JSON   |
| `observation_matching_func`  | `Callable \| None`         | Replaces `match_observation` in the correction loop; not in JSON |

### Requirements — `setup.requirements`

| Field                      | Type    | Notes                                                                                          |
| -------------------------- | ------- | ---------------------------------------------------------------------------------------------- |
| `performance_threshold_m`  | `float` | **Required.** Nadir-equivalent error limit, metres                                             |
| `performance_spec_percent` | `float` | **Required.** Percentage of accepted GCPs within the limit that a run must exceed, in [0, 100) |

### Kernels, instrument and gates — `setup.geo`

| Field                 | Type            | Notes                                                                                              |
| --------------------- | --------------- | -------------------------------------------------------------------------------------------------- |
| `meta_kernel_file`    | `Path`          | The mission meta-kernel JSON                                                                       |
| `generic_kernel_dir`  | `Path`          | Generic SPICE kernels                                                                              |
| `dynamic_kernels`     | `list[Path]`    | Kernel JSONs built from each input's telemetry                                                     |
| `instrument_name`     | `str`           | NAIF instrument name from the IK                                                                   |
| `time_field`          | `str`           | Telemetry column of uGPS times                                                                     |
| `dem_data_dir`        | `Path \| None`  | Elevation data the correction loop terrain-corrects with; `None` uses curryer's standard location  |
| `minimum_correlation` | `float \| None` | Gate, 0–1; `None` disables. When set, matches must carry `correlation` or verification raises      |
| `minimum_peak_margin` | `float \| None` | Gate; `None` disables. When set, matches must carry `correlation_secondary` or verification raises |

### Image matching — `setup.search` and `setup.psf_sampling`

Set these for the instrument's ground sampling. Invalid values raise when the setup is built.

| Field                                  | Default    | Notes                                                                                                     |
| -------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------- |
| `search.grid_size`                     | `44`       | Grid points per axis, integer ≥ 2                                                                         |
| `search.grid_span_km`                  | `11.0`     | Full width of the first grid, km                                                                          |
| `search.reduction_factor`              | `0.8`      | Grid-step shrink per iteration, in (0, 1)                                                                 |
| `search.spacing_limit_m`               | `10.0`     | The search stops at this step, m; must be below the first step, `grid_span_km / (grid_size - 1)`          |
| `search.peak_exclusion_km`             | `2.0`      | Distance from the first-grid peak beyond which `correlation_secondary` is taken; below `grid_span_km / 2` |
| `psf_sampling.psf_lat_sample_dist_deg` | `2.44e-05` | PSF sample spacing in latitude, degrees                                                                   |
| `psf_sampling.psf_lon_sample_dist_deg` | `2.87e-05` | PSF sample spacing in longitude, degrees                                                                  |

### Calibration — `setup.calibration`

| Field              | Notes                                   |
| ------------------ | --------------------------------------- |
| `los_vectors_file` | Per-detector line-of-sight unit vectors |
| `psf_file`         | Optical point-spread function           |

### Data loading — `setup.data_config`

| Field              | Notes                                                       |
| ------------------ | ----------------------------------------------------------- |
| `file_format`      | `"csv"`, `"netcdf"` or `"hdf5"`: the telemetry reader       |
| `position_columns` | Spacecraft-position column names in the telemetry, optional |

### Sweep — `sweep`

| Field                   | Type                    | Notes                                                       |
| ----------------------- | ----------------------- | ----------------------------------------------------------- |
| `parameters`            | `list[ParameterConfig]` | **Required**, at least one                                  |
| `search_strategy`       | `SearchStrategy`        | `RANDOM` (default), `GRID_SEARCH` or `SINGLE_OFFSET`        |
| `n_iterations`          | `int`                   | Draws (`RANDOM`) or values per parameter (`SINGLE_OFFSET`)  |
| `seed`                  | `int \| None`           | Seed for `RANDOM`                                           |
| `grid_points_per_param` | `int`                   | Points per parameter for `GRID_SEARCH`                      |
| `max_grid_sets`         | `int`                   | Largest grid allowed (default 100 000); a larger one raises |

### Parameters — `sweep.parameters[]`

| Field                | Type            | Notes                                                                                                                                      |
| -------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `ptype`              | `ParameterType` | `CONSTANT_KERNEL`, `OFFSET_KERNEL` or `OFFSET_TIME`                                                                                        |
| `config_file`        | `Path \| None`  | The kernel JSON template; required for the kernel types                                                                                    |
| `spec.current_value` | `float`         | Nominal value; for `CONSTANT_KERNEL`, the nominal angle of `field`                                                                         |
| `spec.bounds`        | `[min, max]`    | Offsets from nominal, in `units`; zero width holds the parameter                                                                           |
| `spec.sigma`         | `float \| None` | Standard deviation for `RANDOM`, in `units`                                                                                                |
| `spec.units`         | `str \| None`   | e.g. `"arcseconds"`, `"radians"`, `"seconds"`, `"milliseconds"`                                                                            |
| `spec.field`         | `str \| None`   | Telemetry column for `OFFSET_KERNEL`; rotation axis (`angle_x`, `angle_y`, `angle_z`) for `CONSTANT_KERNEL`; output name for `OFFSET_TIME` |

### Parameter types — `ptype`

| Value             | Adjusts                                                                      |
| ----------------- | ---------------------------------------------------------------------------- |
| `CONSTANT_KERNEL` | One rotation angle (roll, pitch or yaw) of a fixed-attitude frame's CK       |
| `OFFSET_KERNEL`   | A bias added to a telemetry angle field before its CK is built               |
| `OFFSET_TIME`     | Seconds added to the frame times: true frame time = recorded `ugps` + offset |

Kernel parameters are applied in radians, so each `CONSTANT_KERNEL` and `OFFSET_KERNEL` template
must set `"input_angle_units": "RADIANS"` (the attitude writer reads degrees without it), and an
`OFFSET_KERNEL` telemetry field must be in radians. The loop raises before the first pair
otherwise.

### Output — `output`

| Field             | Type                   | Notes                                                 |
| ----------------- | ---------------------- | ----------------------------------------------------- |
| `netcdf`          | `NetCDFConfig \| None` | NetCDF metadata; `None` builds it from the threshold  |
| `output_filename` | `str \| None`          | NetCDF file name in `work_dir`; `None` uses a default |

---

## Adding a mission

A mission provides, in its configuration:

1. **SPICE kernels**: the meta-kernel, generic kernels, and the dynamic kernels built from
   telemetry (`setup.geo`).
2. **The instrument**: its NAIF name, the LOS table and the PSF.
3. **Timestamps**: frame times as integer uGPS in each subimage's `ugps`; `setup.geo.time_field`
   names the telemetry's uGPS column.
4. **The requirement**: threshold and spec.
5. **Image matching for its pixel size**: `setup.search` and `setup.psf_sampling` (the defaults
   suit 30 m Landsat chips), and the gates.
6. **The adjustments to try**, for correction: `sweep.parameters`.

And, in its own code:

- **GCP chips** regridded to latitude and longitude ([GCP chip regridding](gcp_regridding.md));
- **observation subimages** that meet the [input contracts](#input-contracts): cut around each
  chip, inside it by the search reach (plus the sweep's largest shift, for correction);
- **data access and command lines**: finding granules, telemetry and chips, and any figures.

`examples/correction/example_config.json` is an annotated template;
`examples/correction/clarreo_config.json` is a complete configuration for one mission.

### S3

Paths given to `verify()` and `run_correction()` may be `s3://` URIs when `boto3` is installed.
For image-matching results stored by date, `curryer.correction.dataio` provides
`S3Configuration`, `find_netcdf_objects` and `download_netcdf_objects`:

```python
import datetime

from curryer.correction.dataio import S3Configuration, download_netcdf_objects, find_netcdf_objects

s3_config = S3Configuration(bucket="my-mission-bucket", base_prefix="image_match")  # base_prefix/YYYYMMDD/
keys = find_netcdf_objects(s3_config, start_date=datetime.date(2024, 3, 17), end_date=datetime.date(2024, 3, 17))
paths = download_netcdf_objects(s3_config, keys, destination="/tmp/downloads")
```

Credentials come from the standard AWS sources (environment, profile or role).

---

## Troubleshooting

| Error                                                                             | Cause and fix                                                                                                                                                                                                        |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KeyError: Missing required 'sweep' section`                                      | `load_config_files()` needs `setup` and `sweep`                                                                                                                                                                      |
| `ValidationError` loading the configuration                                       | The message names the field: a wrong type, a missing required field, or an unknown key                                                                                                                               |
| `SPICE(PATHTOOLONG)`                                                              | SPICE's 80-character path limit. curryer works around it with a short temporary directory; set `CURRYER_TEMP_DIR` if `/tmp` is unavailable                                                                           |
| `ValueError` with `geolocated_data=`                                              | Attach `setup.image_matching_func`, or pass `gcp_directory=`, `los_file=` and `psf_file=`                                                                                                                            |
| `ValueError: Spacecraft position ...`                                             | File-pair verification needs the observation's ECEF `position` in metres; a value in kilometres lands inside the Earth                                                                                               |
| `ValueError: Point (lat ..., lon ...) is outside the observation grid`            | The chip's centre must lie inside the observation it is paired with                                                                                                                                                  |
| `'detector_pixel' ... must be ...` / `... columns but the LOS table has ... rows` | A column crop, file or `geolocated_data`, needs strictly increasing `detector_pixel`; an uncropped one needs one LOS row per column                                                                                  |
| `Search shift ... samples the subimage outside the GCP chip`                      | Crop the observation further inside the chip: the first grid's half-width plus the expected error (and the sweep's largest shift, for correction), or `setup.search.max_shift_km()` to be sure; or narrow the search |
| `Subimage has ... non-finite value(s)`                                            | The correlation is undefined over missing data; crop or mask before matching                                                                                                                                         |
| `No GCP pair passes the match-quality gates ... under any parameter set.`         | Every pair failed the gates under every candidate: cloud, or bounds that never reach the true offset                                                                                                                 |
| `No parameter set passes the match-quality gates on every usable GCP pair; ...`   | The usable pairs pass under different candidates. The message counts the passing candidates per pair: narrow the sweep, or drop the pair that rarely passes                                                          |
| NaN in `nadir_equiv_total_error_m` (`Suspicious geometry: discriminant < 0`)      | The off-nadir angle exceeds the Earth-limb limit: check the spacecraft-state variable names, and that the position is in metres, Earth-fixed                                                                         |

---

## Reference examples

| File                                            | Description                                                            |
| ----------------------------------------------- | ---------------------------------------------------------------------- |
| `examples/correction/example_verification.py`   | Runnable: verify, save, mark a review decision, apply it               |
| `examples/correction/example_run_correction.py` | Correction loop template; exits cleanly when data or tools are missing |
| `examples/correction/example_config.json`       | Annotated configuration template                                       |
| `examples/correction/clarreo_config.json`       | A complete configuration for one mission                               |
| `examples/correction/clarreo_config.py`         | The same configuration built in Python                                 |

```bash
python examples/correction/example_verification.py
python examples/correction/example_run_correction.py
```

GCP chip regridding has its own guide: [GCP chip regridding](gcp_regridding.md).
