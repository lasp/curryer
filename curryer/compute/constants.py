"""SPICE related constants for TIM.

@author: Brandon Stone
"""

from enum import IntFlag, unique

from spiceypy.utils.exceptions import SpiceyError

# Time-interval between output samples (TIM/SIM missions only).
#   Units: Microseconds
EPHEMERIS_TIMESTEP_USEC = 60000000
ATTITUDE_TIMESTEP_USEC = 10000000

# Astronomical unit (2012 definition)
#   Units: KM / 1 AU
#   Source: http://www.iau.org/static/resolutions/IAU2012_English.pdf
#   Note: Java uses ###.66 instead of ###.700
KM_PER_ASTRONOMICAL_UNIT = 149597870.700

# Speed of light in a vacuum
#   Units: KM / SEC
SPEED_OF_LIGHT_KM_PER_S = 299792.458

# WGS84 Ellipsoid reference
#   Units: KM
#   Source: https://epsg.org/ellipsoid_7030/WGS-84.html
WGS84_SEMI_MAJOR_AXIS_KM = 6378.137
WGS84_SEMI_MINOR_AXIS_KM = 6356.752314245
WGS84_INVERSE_FLATTENING = 1 / 298.257223563
WGS84_ECCENTRICITY2 = 2 * WGS84_INVERSE_FLATTENING - WGS84_INVERSE_FLATTENING**2


@unique
class SpatialQualityFlags(IntFlag):
    GOOD = 0x0

    SPICE_ERR_NO_RESULTS_FOUND = 0x1  # E.g. no surface intersect.
    SPICE_ERR_MISSING_EPHEMERIS = 0x2
    SPICE_ERR_MISSING_ATTITUDE = 0x4
    SPICE_ERR_INVALID_LOCATION = 0x8  # E.g. inside planet.
    SPICE_ERR_UNKNOWN_NONSPECIFIC = 0x10

    CALC_UNEXPECTED_NOT_FINITE = 0x20

    CALC_KERNELS_UNDESIRED_TLM = 0x40
    CALC_KERNELS_FAILED = 0x80

    CALC_ELLIPS_INSUFF_DATA = 0x100
    CALC_ELLIPS_NO_INTERSECT = 0x200

    # TODO: CALC_TERRAIN_INSUFF_DATA ?
    CALC_TERRAIN_EXTREME_ZENITH = 0x400
    CALC_TERRAIN_MAX_ITER = 0x800
    CALC_TERRAIN_NOT_FINITE = 0x1000

    CALC_ANCIL_INSUFF_DATA = 0x2000
    CALC_ANCIL_NOT_FINITE = 0x4000

    @classmethod
    def from_spice_error(cls, error: SpiceyError):
        if error is None:
            return cls.GOOD
        if not isinstance(error, SpiceyError):
            raise TypeError(f"Method requires a SpiceyError, not: {type(error)}")
        # if isinstance(error, NotFoundError):
        #     return cls.SPICE_ERR_NO_LOOKUP_FOUND
        if "returns not found" in error.message:
            return cls.SPICE_ERR_NO_RESULTS_FOUND
        if "SPICE(SPKINSUFFDATA)" in error.short:
            return cls.SPICE_ERR_MISSING_EPHEMERIS
        if "SPICE(NOFRAMECONNECT)" in error.short:
            return cls.SPICE_ERR_MISSING_ATTITUDE
        if "SPICE(NOTDISJOINT)" in error.short:
            return cls.SPICE_ERR_INVALID_LOCATION
        return cls.SPICE_ERR_UNKNOWN_NONSPECIFIC
