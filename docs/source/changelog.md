# Changelog

## Unreleased

`verify()` gives correct nadir-equivalent statistics on real flight data: pairing fails loudly,
the error uses the viewing geometry of the pixel that saw the GCP, matching is configurable per
mission, and the correlation gates apply. Every matched GCP is reported with a status, and the
results can be saved, reviewed by a person and re-scored without re-matching. The correction
loop runs on observation subimages, with SPICE geolocation and image matching on the
observation's radiance.

### Breaking changes

- **Viewing geometry from the observing pixel** -- `resolve_spacecraft_ecef(grid, r_spacecraft_m,
target_lat_deg, target_lon_deg)` aims the boresight at the mid-frame pixel in the GCP's column
  and raises when the GCP is outside the grid; a spacecraft position is required and validated
  (`psf.validate_spacecraft_ecef_m`: shape, finite, metres, above the ellipsoid). Nothing falls
  back to a nadir geometry: `default_altitude_m` is removed from `verify()`,
  `_run_image_matching_for_pairs` and `resolve_spacecraft_ecef`, and the deprecated
  `infer_spacecraft_state` raises on a missing position. `image_matching` raises on a SPICE
  failure or a missing mid-frame time, and the per-pair loops no longer skip failed pairs.
- **Pairing raises** -- `pair_files` and `pair_geolocated_dataset_with_gcp_files` raise
  `ValueError` naming a file that fails to load (previously skipped, which misaligned every
  later pair).
- **Pairing uses the traced swath footprint** -- `find_l1a_gcp_pairs`, `pair_files` and
  `pair_geolocated_dataset_with_gcp_files` test each GCP center against the polygon traced around
  the L1A's geolocated pixels (`curryer.compute.footprint.footprint_polygon`), about one pixel
  outside the edge pixel centres at the default tolerance, instead of the polygon through its four corner pixels in a flat ENU
  plane. On a curved swath the corner polygon paired chips outside the swath and missed chips
  inside it (by up to about 18 km on 6-9 minute granules). Margins are great-circle
  distances on the mean-radius sphere. `PairingResult.l1a_images` holds `L1AMetadata`, which adds
  the `footprint`; an optional `footprint_tolerance_km` sets its tolerance (the edge's offset and
  simplification). An untraceable footprint (folded or overlapping grids) falls back to the
  buffered convex hull with a warning naming the image. An L1A whose geolocation has non-NaN fill,
  out-of-range values or a single row, a GCP chip whose center pixel is non-finite or out of range
  (previously never paired), and a non-finite `max_distance_m` raise `ValueError` naming the image
  or chip.
- **`im_search` returns seven values** -- `(lat_err_km, lon_err_km, ccv_max, ccv_secondary, row,
col, step_m)`, and `IntegratedImageMatchResult` gains `ccv_secondary` before its index fields.
  `im_search` raises when a shifted subimage samples outside the GCP chip (it correlated against
  zero fill), when the subimage has non-finite or uniform values, and when no shift in a grid has
  a finite correlation. `integrated_image_match` searches only the part of the convolved chip
  where the PSF lies wholly inside it (`psf.crop_to_full_psf_overlap`), so a subimage within half
  a PSF kernel of the chip edge raises instead of matching the convolution's zero padding.
- **Correlation gates apply** -- `minimum_correlation` filters on the `correlation` variable
  (also read as `ccv` / `im_ccv`); `_filter_by_correlation` and `process_from_netcdf` raise when a
  threshold is set and no correlation variable exists (they warned and skipped). New
  `GeolocationConfig.minimum_peak_margin` drops matches whose `correlation -
correlation_secondary` is below it, and raises where `correlation_secondary` is non-finite.
  `minimum_correlation` must be in [0, 1] and `minimum_peak_margin` positive. Neither gate
  applies to per-pair errors inside the correction loop.
- **Matching configuration on the setup** -- `GeolocationSetup.psf_sampling` and `.search`
  (defaults equal the previous hardcoded Landsat values); `SearchConfig` and `PSFSamplingConfig`
  are frozen and reject invalid values, including a `peak_exclusion_km` not below
  `grid_span_km / 2`. `PSFSamplingConfig.gcp_step_m` and `motion_convolution_step_m`, never
  read, are removed, which changes its positional field order.
- **The requirement is strict** -- a run passes when the percentage of accepted measurements
  within `performance_threshold_m` exceeds `performance_spec_percent`, as a "more than N %"
  requirement is written; a run exactly at the spec used to pass. This applies to
  `VerificationResult.passed` and `CorrectionResult.met_threshold`. `RequirementsConfig` rejects
  a spec of 100, which no run could exceed. The summary table reads `(req: > 39.0%)`.
- **Rejected matches are kept** -- `ErrorStatsProcessor.compute_nadir_equivalent_errors`,
  `process_geolocation_errors` and `process_from_netcdf` return every measurement with
  `accepted` and `rejection_reason`; statistics cover the accepted ones. A caller averaging
  `nadir_equiv_total_error_m` must mask by `accepted`. `verify()` no longer raises when the gates
  reject everything: `percent_within_threshold` is `None` (nothing was measured), `passed` is
  `False`, and the warning says the run was inconclusive.
- `GCPError.status` (`pass` / `fail` / `rejected`) is required and must agree with `passed`.
- Error-stats NetCDF attributes are strings or numbers: `processing_timestamp` is an ISO string
  and `correlation_filtering_applied` is `1`.
- An accepted measurement with a non-finite correlation raises (its quality weight is undefined).
- **`loop()` / `run_correction()` take an observation subimage as the science input** -- a
  NetCDF file with `band_data` (frame, pixel) radiance, `ugps` (frame) integer frame times and
  `detector_pixel` (pixel) LOS-table rows, cropped inside its GCP chip
  (`curryer.correction.load_loop_observation`). Each parameter set re-geolocates those frames
  and pixels with the LOS table (`compute_ellipsoid_intersection`), terrain-corrects them with
  `GeolocationConfig.dem_data_dir`, takes the spacecraft position from SPICE, and matches the
  radiance on that grid to the chip. The loop previously geolocated a whole granule and matched
  an all-ones image, so its errors did not depend on the parameters.
- **New `curryer.correction.match_observation`**, shared by `verify()` (observation files) and
  the loop. `GeolocationSetup.observation_matching_func` replaces it in the loop;
  `image_matching_func` now applies only to `verify(geolocated_data=...)`.
- **Parameter sets are compared on the same GCP pairs** -- a pair is used when some parameter
  set passes the match-quality gates on it, and a parameter set is selectable (`valid`) only
  when it passes them on every used pair. The output NetCDF gains `accepted` and `valid`.
- **`DataConfig.time_scale_factor` is removed** and `DataConfig` rejects unknown fields;
  `file_format` applies to the telemetry file.
- `OFFSET_TIME` shifts the observation's frame times (`apply_offset` takes a uGPS ndarray and
  returns int64 uGPS).
- `loop()` raises before the first pair when a `CONSTANT_KERNEL` or `OFFSET_KERNEL` template
  does not set `input_angle_units` to `RADIANS`; the loop applies radians, and a template
  reading degrees made every correction 57.3 times too small.
- Missing `OFFSET_KERNEL` telemetry fields, unknown parameter variables and SPICE coverage gaps
  raise instead of logging a warning; the 111 km/degree fallback statistics are removed.
- `ParameterConfig` raises on `spec.units` its type does not convert (previously any unknown
  string was treated as radians or seconds).
- `loop(resume_from_checkpoint=True)` raises `NotImplementedError` when the checkpoint holds
  completed pairs, whose image-matching results it cannot restore (CURRYER-100).
- `dataio.validate_science_output` is removed.
- **A `CONSTANT_KERNEL` parameter is one rotation axis** -- `spec.field` names it (`angle_x`,
  `angle_y` or `angle_z`) and `spec.current_value` is a float; a frame takes three parameters
  sharing its `config_file`, joined into one CK. `GRID_SEARCH` and `SINGLE_OFFSET` previously
  added one offset to all three axes at once, so pitch and roll could not be swept separately.
  The output variables keep their `<kernel>_roll` / `_pitch` / `_yaw` names and are stored in
  the configured units (radians when `units` is `None`). `Sweep.update_param` raises on a
  selector matching more than one parameter, and `NetCDFConfig.get_parameter_netcdf_metadata`
  no longer takes `angle_type`.
- **A parameter with zero-width `bounds` `[b, b]` is held at `current_value + b`** in every
  strategy (one grid point in `GRID_SEARCH`, not swept by `SINGLE_OFFSET`, not drawn by
  `RANDOM`); `SINGLE_OFFSET` raises when no parameter has non-zero-width bounds. A held
  `CONSTANT_KERNEL` axis no longer needs `sigma=None`.
- `CONSTANT_KERNEL` accepts `units="radians"`, as `OFFSET_KERNEL` does; with `units=None` its
  NetCDF `units` attribute is `radians` (it said `arcseconds`).
- `KernelContext` and `ImageMatchingContext` are removed from `curryer.correction.config`.
- The `results` dicts `loop()` returns carry the subimage's (frame, pixel) `geolocation`; a set
  that is not compared has `aggregate_rms_error_m=None` and no `aggregate_error_stats`.
- `CorrectionResult` gains `n_usable_pairs` and `n_valid_sets`, and its recommendation and
  summary table state the pairs and sets the verdict rests on.

### New

- `SearchConfig.peak_exclusion_km` and `SearchConfig.max_shift_km()` (the search's worst-case
  reach, a crop margin that always suffices).
- `image_io.observation_los_vectors`: a column-cropped NetCDF observation carries
  `detector_pixel`, its LOS-table row per column; without it the LOS table must have one row per
  column. `image_matching()` reads `detector_pixel` from a column-cropped `geolocated_data` the same
  way, so that dataset can be cropped inside the chip across track as well as along it.
- Per-GCP `along_track_error_m` / `cross_track_error_m` (geolocated minus true; from
  `track_azimuth_deg`, `psf.ground_track_azimuth_deg`), `off_nadir_angle_deg` and
  `correlation_secondary`.
- `verify(..., keep_images=True)` keeps each match's observed, reference and emulated images
  (`image_match.chip_image_dataset`); file-pair modes only.
- `save_verification` / `load_verification` (`result.json`, `summary.csv`, `aggregate_stats.nc`,
  `chips/`), `read_review_decisions` and `apply_review` re-score after a reviewer edits the
  `review` column of `summary.csv`. `read_review_decisions(summary_csv, result)` refuses a CSV
  whose GCP indices and keys do not match `result`, and `load_verification` refuses a `chips/`
  directory that does not hold one chip per GCP with matching keys.
- Match signal-to-noise weights `match_snr_weight` (rho^2 / (1 - rho^2), rho capped at 0.99),
  `weighted_statistics`, `weighted_percent_below` and
  `VerificationResult.weighted_percent_within_threshold`, reported alongside the unweighted
  statistics and never deciding `passed`. `error_stats.STATISTIC_ATTRS` and `without_statistics`
  name and clear the statistic attributes.

### Fixes

- Matched errors were converted to degrees with 111.0 km/degree and back with the WGS-84 radius,
  a 0.29% north-south error; both now use `WGS84_SEMI_MAJOR_AXIS_KM`.
- Per-GCP attribution in `verify()` follows the measurements that survive filtering.
- File-pair verification downloads an `s3://` observation once per pair (it was read twice), and
  so does the correction loop.
- The verification summary table's rows lined up with its borders only when the footer was not
  the widest line.
- `OFFSET_KERNEL` offsets were converted from arcseconds twice and applied about 2e5 times too
  small; they are applied in radians as `load_param_sets` returns them.
- `CONSTANT_KERNEL` attitude kernels covered only their two sentinel epochs because the
  template's gap threshold split them; gap chunking is disabled for them.
- `loop()` creates `work_dir` when it is missing; the kernel writers took the missing
  directory for a file name and wrote every kernel to that one path.
- `OFFSET_KERNEL` and `OFFSET_TIME` values are stored in the output in their configured units
  (they were stored in radians and seconds under arcsecond and millisecond labels).
- A failed parameter-set selection (no usable pair, or no set passing the gates on every usable
  pair) writes the per-pair results NetCDF before raising.
- Parameter values are stored under `NetCDFConfig.get_parameter_netcdf_metadata(...).variable_name`,
  so a custom `variable_name` in `parameter_metadata` works (it raised `KeyError`).
- `CorrectionResult.met_threshold` counts only the pairs the best parameter set passes the
  match-quality gates on; it counted the errors of rejected pairs too.
- The motion blur clipped the PSF. `convolve_psf_with_spacecraft_motion` returned the blurred
  PSF on the bounding box of its rotated frame, reused as a lat/lon box, so a PSF was cut off
  wherever the motion was not east-west (on a CLARREO flight chip with the motion 41 degrees from
  east, 0.13 km RMS across track instead of 0.26 km); it also filled outside the sampled PSF with
  the nearest edge value. The blurred PSF is now returned on its own footprint, zero outside the
  sampled PSF.
- The motion blur follows the frame step (rows of the subimage, along track), the direction the
  footprint moves during a frame; it followed the detector step (across track). It raises when
  the subimage has fewer than two frames or a mean frame step that is not finite or is zero.
- `resample_psf_to_gcp_resolution` centres its grid on the PSF centroid with an odd number of
  samples; the grid started at the PSF's edge, so the centroid fell up to a chip pixel off centre
  and the convolution shifted the chip by that much (12-81 m on 100 m chips). It raises on PSF
  weights that do not sum to a positive value.
- Matched errors and correlations change with these three fixes. The 20 MATLAB-simulated
  image-match cases, which reproduced the clipped and shifted PSF, are removed; image matching
  is tested on a synthetic scene simulated independently of the convolution
  (`tests/integration/test_image_match_synthetic.py`).

### Documentation

- The correction user guide is reorganized around concepts (pairs, matching, gates, errors,
  weighting, the correction loop), the input contracts and what raises, and the result objects.
  It links the GCP regridding guide, now in the docs index, instead of repeating it.

## Version 0.5.4 (unreleased)

Adds footprint polygons for 2-D geolocation grids.

### Highlights

- **New `curryer.compute.footprint`** -- `footprint_polygon(latitude, longitude, tolerance_km=None)`
  traces the outer edge of a grid of geolocated pixels into a closed, counter-clockwise polygon of
  great-circle arcs, so it follows curved and antimeridian-crossing swaths. The edge lies half a
  pixel plus `tolerance_km` outside the edge pixel centres, and simplification moves it by up to
  `tolerance_km` either way; `None` uses half the smaller median pixel spacing of the two grid
  axes. Where the edge cannot be traced (folded or overlapping grids) it logs a warning and returns
  a buffered spherical convex hull that contains every valid pixel centre with at least half the
  median pixel diagonal to spare. `swath_footprint` and `convex_footprint` are the two methods on
  their own. Invalid geolocation, including latitude and longitude fill at different pixels, raises
  `ValueError`, as does a grid axis with no two adjacent valid pixels (a single row, say), along
  which the pixel size is unknown; an untraceable edge raises `FootprintGeometryError` (a
  `ValueError`) from `swath_footprint`.
- **New `curryer.compute.footprint.footprint_margin_m`** -- signed great-circle distance, in
  metres, from points to a footprint's edge: positive inside, negative outside. A clockwise,
  open or self-crossing polygon raises `ValueError`.

## Version 0.5.3 (2026-10)

Adds a per-pixel geolocation and surface-angle path for large focal planes.

### Highlights

- **New `curryer.compute.spatial.pixel_geometry`** -- geolocates an instrument's pixel
  vectors and computes the solar and viewing zenith and azimuth, relative azimuth and
  per-pixel quality flags at each ellipsoid intersection. ndarrays in, a
  `{column: (n_times, n_pixels) ndarray}` dict out, no pandas product index, so it scales to
  multi-megapixel imagers. At most three SPICE calls per time; everything per pixel is the
  existing closed-form numpy (`ray_intersect_ellipsoid`, `calc_azimuth`, `calc_zenith`). The
  viewing angles reuse the spacecraft position the intersection queried, so no second
  ephemeris read is made.
- **New `curryer.compute.geometry_fields.PixelField`** -- the field registry for
  `pixel_geometry`, mirroring `GeometryField`: `pixel_geometry(fields=[...])` computes only
  the requested fields and only the work behind them, so a call asking for no solar field
  skips the Sun ephemeris query and a call asking for no angle skips the local-frame
  construction. The five surface-angle members share their selector and column names with the
  `GeometryField` members of the same name, so a boresight product and a per-pixel product
  address the same quantity by the same key.
  `spacecraft`/`sun` positions are not pixel fields; `GeometryData` already serves them.
- **New `curryer.compute.spatial.relative_azimuth`** -- the CERES BDS R3V4 relative azimuth
  (`mod(view - sun + 180, 360)`, Sun at 180) as a public vectorized helper. The
  `GeometryData` `relative_azimuth` field now calls it, so boresight and per-pixel products
  agree by construction.

- **`GeometryData.get_geometry` and `get_vectors` take `require_coverage=`** -- when True, a
  SPICE input the requested fields read that is NaN at every requested time (its kernels are
  not furnished or do not cover the span) raises `ValueError` naming the input and the fields
  that read it, instead of logging a warning. Per-sample gaps still come back as NaN, so a
  product keeps per-frame fill while a missing kernel stops the run. Default False is
  unchanged.

### Fixes

- `ray_intersect_ellipsoid` no longer emits a NumPy `RuntimeWarning` for rays that miss the
  ellipsoid (negative discriminant); misses were already documented as NaN. Degenerate
  zero-length vectors still warn.
- `calc_azimuth` and `calc_zenith` share one local-frame helper; results are unchanged for
  float64 inputs.
- `calc_zenith` clamps the cosine before `arccos`, so a target along the surface normal (zenith
  0 or 180) returns that angle instead of NaN. Rounding could put the dot product of the two
  unit vectors a few ulps past 1. `pixel_geometry` shares the clamp.
- `compute_ellipsoid_intersection` docstring: `ugps_times` are GPS microseconds, not seconds.

## Version 0.5.2 (2026-09)

Adds the building blocks of a mission-agnostic kernel-management layer:
pool introspection, kernel coverage queries and aggregation, NAIF
generic-kernel discovery, and a local kernel file cache. Also adds lunar
geometry fields and the instrument FOV wrapper they are measured against.

- **`curryer.spicierpy.ext`** – `loaded_kernels()` lists the furnished
  kernel pool (with a `KernelType` filter enum); `kernel_coverage` /
  `kernel_objects` gain binary-PCK support keyed on frame class IDs, with
  `frame_class_id()` resolving names, `Body`/`Frame` objects, and TK-frame
  aliases (e.g. `MOON_PA`).
- **New `curryer.kernels.coverage`** – run-level coverage aggregation:
  `valid_window` (simultaneous coverage across targets), `coverage_gaps`
  (uncovered sub-ranges, warning or raising), `coverage_rollup`, and
  `pairwise_overlap`. Targets that resolve nowhere are diagnosed instead of
  silently reporting as uncovered.
- **New `curryer.kernels.discovery`** – resolves the most recent NAIF
  generic kernels (LSK, DE, PCKs) by scraping the server's index listings,
  with mirror and flat-directory support.
- **New `curryer.kernels.cache`** – fetches kernel sources (local, HTTP,
  S3) into a per-version user cache with atomic writes, per-entry max ages,
  size revalidation, retry backoff, and offline stale-copy fallback.
- **New `curryer.kernels.path_utils`** – consolidated SPICE path-length
  handling (`validate_path_length`, shortening strategies for the
  80-character meta-kernel limit).
- **Leapsecond updates survive upgrades** – a leapsecond kernel ships
  inside the package, and `spicetime.leapsecond.update_file` caches updates
  in a version-independent directory.
- **New lunar geometry fields** – an opt-in `moon_geometry` provider
  reports `moon_direction` (a unit vector in the frame the instrument
  declares its FOV in), `moon_angular_radius` and `moon_distance`, from an
  apparent (LT+S) position of the Moon body center.
- **New `compute.spatial.boresight_offset_angles`** – decomposes a target
  direction into signed azimuth/elevation offsets about an instrument
  boresight, with `spicierpy.ext.instrument_fov` supplying the boresight,
  `FOV_REF_VECTOR` and half angle it is measured against.

## Version 0.5.0 (2026-07)

Expands `curryer.compute.geometry` from 6 to 24 computable fields and adds NumPy 2 support.

### Highlights

- **18 new geometry fields** on `GeometryData`, all routed through the existing
  selective-compute registry so a request costs only the SPICE providers it actually needs:
  - _Boresight and surface_ – `boresight`, `surface_colatitude`.
  - _Surface angles_ – `viewing_zenith`, `solar_zenith`, `viewing_azimuth`, `solar_azimuth`,
    `relative_azimuth`, `cone_angle`, `cone_angle_rate`. These are pure math over the boresight
    ellipsoid intersection and providers already in hand, so they add no new SPICE queries.
  - _State and attitude_ – `sc_velocity`, `satellite_attitude`.
  - _Inertial and orbital frame_ – `sc_position_inertial`, `sc_velocity_inertial`,
    `boresight_inertial`, `clock_angle`, `clock_angle_rate`, `along_track_angle`,
    `cross_track_angle`. The inertial fields are a second ephemeris read (a different reference
    frame), so they stay opt-in rather than joining the default field set.
- **New `curryer.compute.geometry_fields` module** exposing the `GeometryField` enum, so
  downstream consumers can import field identifiers without pulling in the broader constants
  module. `GeometryField` subclasses `str`, so members and plain strings
  (`GeometryField.SUBSATELLITE` vs `"subsatellite"`) remain interchangeable as field selectors
  and column keys. It is re-exported from `curryer.compute.constants` for backward compatibility.
- **Angle conventions documented once, in the module** – azimuths clockwise from geodetic North
  in `[0,360)`, zeniths geodetic from the surface normal, degrees. `relative_azimuth` uses the
  CERES BDS R3V4 origin (Sun at 180) and is kept in its lossless unfolded form; the `[0,180]`
  fold is a deliberate downstream step.
- **NumPy 2 support** – the `numpy` constraint widens to `>=1.23,<3`. `SpiceTime` now implements
  `__array_wrap__` with the NumPy 2 signature (NumPy 1 remains supported); `__array_prepare__`,
  removed in NumPy 2, is gone. Ruff's `NPY201` rule is enabled to catch removed APIs.
- **SpiceyPy unpinned on Python 3.11+** – SpiceyPy 8.1.0–8.1.2 ship a broken `__all__` that
  breaks wildcard imports; `curryer.spicierpy` now detects and works around it rather than
  capping the version (issue #147). Python 3.10 stays capped at `<8.1`, since SpiceyPy 8.1+
  publishes no cp310 wheels.
- **New SPICE error-classification helpers** in `curryer.spicierpy.ext`: `SpiceErrorInfo`,
  `classify_spice_error`, and `spice_error_message`.
- **Release and repository documentation** – new `repo_management.md` covering the release
  process, test workflows, dependency management, and linting/coverage tooling.

### Notes

- No breaking changes. `GeometryData.__init__` gains `inertial_frame` and `attitude_frame`
  keyword arguments, both appended with defaults.

---

## Version 0.4.0 (2026-06)

Major redesign of the `curryer.correction` package with a new modular architecture, Pydantic-based configuration, standalone verification workflows, GCP regridding support, and expanded documentation and examples.

### Highlights

- **Refactored `curryer.correction` into focused modules** – split the former monolithic implementation into dedicated modules: `config`, `pipeline`, `verification`, `parameters`, `kernel_ops`, `image_io`, `results_io`, `regrid`, and `error_stats`.
- **Pydantic-based configuration** – introduced structured config models (`GeolocationSetup`, `Sweep`, `OutputConfig`, `DataConfig`, `GeolocationConfig`, `NetCDFConfig`, `RequirementsConfig`, `RegridConfig`) with improved validation and serialization.
- **Split correction config into Setup / Sweep / Output** – replaced the monolithic `CorrectionConfig` god-object with three focused models: `GeolocationSetup` (durable, mission-specific setup built once), `Sweep` (the lightweight parameter experiment varied between runs), and `OutputConfig`. Added `Sweep.with_strategy()` / `Sweep.update_param()` for cheap, eagerly re-validated copies. Calibration collapsed to direct file paths (`CalibrationFiles.los_vectors_file` / `psf_file`). JSON loading is now `load_config_files()` (returns `(setup, sweep, output)` from `"setup"` / `"sweep"` / optional `"output"` sections), plus `load_setup_from_json()` / `load_sweep_from_json()`; `load_config_from_json()` was removed.
- **Standalone verification workflow** – new `verify()` entry point and result models (`VerificationResult`, `GCPError`) for checking geolocation performance without running a full correction sweep.
- **GCP chip regridding support** – new tooling to load HDF GCP chips, convert ECEF coordinates to geodetic coordinates, and regrid them to regular lat/lon grids; includes CLI examples for batch workflows.
- **Expanded correction workflow API** – clearer public entry points (`run_correction`, `run_image_matching`, `compute_error_stats`) and structured result types (`CorrectionResult`, `ParameterSetResult`).
- **Deterministic parameter search strategies** – support for random sampling, grid search, and single-parameter sweep via the `SearchStrategy` enum.
- **Improved image/data I/O** – unified path resolution with optional S3 support; format-agnostic helpers for MATLAB, NetCDF, HDF, and GeoTIFF workflows.
- **Expanded documentation and examples** – new correction user guide, regridding documentation, runnable examples, and JSON config templates.

### Breaking Changes

- `curryer.correction` has been significantly reorganized internally; loader protocols were removed.
- Configuration now uses Pydantic models; some module, class, and function names were renamed or relocated.
- Removed the monolithic `CorrectionConfig` (and `load_config_from_json`) in favor of `GeolocationSetup` / `Sweep` / `OutputConfig` and `load_config_files` / `load_setup_from_json` / `load_sweep_from_json`. Entry points now take `setup` first: `run_correction(setup, sweep, inputs, work_dir)` and `verify(setup, ...)`.
- This release should be treated as a breaking redesign, however there are minimal downstream users of correction.

---

## Version 0.3.3 (2026-06)

- Added the `curryer.compute.geometry` module: `GeometryData` computes geolocation/geometry
  ancillary fields (subsatellite and subsolar points, satellite radius, Earth-Sun distance,
  spacecraft position and geodetic altitude) through a selective-compute registry that queries
  each SPICE input once, with importable math-only leaf functions and a documented per-field NaN
  fill contract
- Added frame-to-frame rotation primitives to `curryer.compute.spatial`: `frame_to_frame_rotation`,
  `frame_to_frame_euler`, and `frame_to_frame_quaternion`

## Version 0.3.2 (2026-04)

- Pinned SpiceyPy below 8.1.0 for compatibility
- Refreshed the documentation

## Version 0.3.1 (2026-02)

- Added a citation file

## Version 0.3.0

- Add automatic SPICE file path shortening tools
- Corrected a bug within the apply_offsets function of Correction loop
- BREAKING CHANGE: Renamed MonteCarlo function to more generic "Correction"

## Version 0.2.1 (2025-12)

- Added support for custom pointing vectors in `compute.spatial.compute_ellipsoid_intersection`
- Clarified function and parameter naming and improved testing for associated functions
- Left deprecated spatial compute functions in place for backward compatibility with warnings
  - Both `pixel_vectors` and `instrument_intersect_ellipsoid` were deprecated
- Bug fix for montecarlo testing using pytest tmp_path

## Version 1.0.0 (unreleased)

- Add documentation building framework with Sphinx
