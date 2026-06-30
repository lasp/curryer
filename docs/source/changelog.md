# Changelog

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
