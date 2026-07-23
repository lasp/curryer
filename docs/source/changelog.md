# Changelog

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
