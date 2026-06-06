# Changelog

## Version 0.4.0b1 (2026-05)

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
- This release should be treated as a breaking redesign milestone.

---

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
