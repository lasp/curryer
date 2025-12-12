# Changelog

## Version 1.0.0 (unreleased)

- Add documentation building framework with Sphinx

# Version 0.2.1 (2025-12)

- Added support for custom pointing vectors in `compute.spatial.compute_ellipsoid_intersection`
- Clarified function and parameter naming and improved testing for associated functions
- Left deprecated spatial compute functions in place for backward compatibility with warnings
  - Both `pixel_vectors` and `instrument_intersect_ellipsoid` were deprecated
- Bug fix for montecarlo testing using pytest tmp_path
