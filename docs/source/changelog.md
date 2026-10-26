# Changelog

## Version 0.5.5 (unreleased)

Terrain-correction fixes: one point's view geometry no longer fails the other points, and elevation
regions near the dateline read the right heights.

### Fixes

- **Terrain correction failed where every local height was below the ellipsoid, as over ocean with
  a negative geoid (#198).** The search for the terrain intersection starts at the local maximum
  height along the view vector. A maximum at or below zero started it at or below the ellipsoid,
  stepping away from the surface: `terrain_correct` flagged every point `CALC_TERRAIN_MAX_ITER` and
  `terrain_correct_single` raised `ValueError`. The search now starts at least
  `TERRAIN_MIN_START_KM` (10 m) up. Regions whose maximum is 10 m or more are unchanged: on real
  DEM regions, every point corrected before is bit-identical. Regions whose maximum is between 0
  and 10 m (open ocean with a small positive geoid) now start higher, which can move a result by
  the surface's height change within one step.
- **A terrain search that started exactly on the surface returned NaN.** Where the start fell on
  the region's highest pixel, the search never stepped and its interpolation divided 0/0:
  `terrain_correct` flagged the point `CALC_TERRAIN_NOT_FINITE`, and `terrain_correct_single`
  returned NaN or raised `ZeroDivisionError`. The search now steps at least once.
- **Exact-nadir views were flagged `CALC_TERRAIN_EXTREME_ZENITH` and returned NaN (#202).** Rounding
  could put the cosine of the view zenith just above 1, where `arccos` is NaN; on a test grid this
  hit 204 of 2409 exact-nadir views. The cosine is now clipped to [-1, 1], and the infinite step
  length at zero zenith no longer warns.
- **`Elevation.local_region` bounds past +180 or -180 loaded the wrong side of the dateline
  (#199).** Bounds such as 175 to 185, from padding a box near the dateline, built a region from
  -180 to -170 only, so lookups at 175 to 180 failed. Bounds are now wrapped into range first, and
  tiles that merge into an unevenly spaced grid raise `ValueError` instead of placing heights at
  the wrong pixels.
- **`ElevationRegion.query` failed on points outside the region with an `AssertionError` and a
  misleading longitude (#211).** It now raises `ValueError` giving the number of such points, the
  first one's index and longitude/latitude as passed, and the region's bounds; an `assert` also
  disappears under `python -O`. A point west of a region that does not cross the dateline is no
  longer shifted by 360 degrees before the check, which made its reported longitude wrong. New
  `ElevationRegion.contains` tests points without raising.
- **One point whose terrain search left the elevation region failed the whole terrain correction
  (#211).** Given an `ElevationRegion`, `terrain_correct` now sets such points to NaN with the new
  quality flag `SpatialQualityFlags.CALC_TERRAIN_OUTSIDE_DEM` (`0x8000`) and corrects the rest.
  Products that store the flags need at least an unsigned 16-bit or a signed 32-bit integer. This
  includes a search that starts outside the region, which an oblique view near the region's edge
  can; `Geolocate.correct_terrain`'s `pad_degrees` sets how far the region reaches.
- **A point that did not reach the surface in 1000 steps set every point to NaN.**
  `terrain_correct` now flags only the points still searching `CALC_TERRAIN_MAX_ITER` and returns
  the others.
- **`terrain_correct_single` placed the intersection along the view vector using the start
  height in place of the previous step's.** Every search longer than one step returned a shifted
  position, by about 90 m on a flat surface viewed at 45 degrees. `terrain_correct`, which
  `Geolocate` uses, was not affected.

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
- **Faster CK and SPK writes with `input_gap_threshold`** -- the attitude and ephemeris-state
  writers find input gaps on the input times instead of parsing back the UTC strings they
  wrote for msopck/mkspk, which took about a third of the write. On 1M samples, gap detection
  drops from 12.6 s to 4.1 s. UTC-string input still parses its strings once. Segment
  boundaries and kernel data are unchanged, except that a gap within about 2 µs of the
  threshold can split differently.

### Fixes

- **A gap threshold under 1 s or of a day or more left gaps in kernel coverage that the threshold
  should have bridged.** Only the seconds component of `input_gap_threshold` was used:
  `"500ms"` and `"1D"` were read as 0, which split the input after every sample, and `"25h"`
  split at every gap of 1 h or more. The full duration is now used. Thresholds in whole seconds under a day, such as `"30s"` and `"5min"`, were not
  affected. A threshold that is not a duration of at least 1 µs now raises `ValueError` when
  the kernel properties are created.

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
