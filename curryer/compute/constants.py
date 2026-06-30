"""SPICE related constants for TIM.

@author: Brandon Stone
"""

from enum import Enum, IntFlag, unique

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


class GeometryField(str, Enum):
    """Importable identifiers for the fields :class:`curryer.compute.geometry.GeometryData` computes.

    Each member's value is the selector passed to ``get_geometry(fields=...)``; :attr:`columns`
    lists the output columns it expands to and :attr:`description` is a one-line summary. Members
    are plain strings, so ``GeometryField.SUBSATELLITE`` and ``"subsatellite"`` are interchangeable
    as field selectors and as column keys.
    """

    SC_RADIUS = ("sc_radius", ("spacecraft_radius",), "Observer distance from Earth's center (km).")
    SUBSATELLITE = (
        "subsatellite",
        ("subsatellite_latitude", "subsatellite_longitude", "subsatellite_colatitude"),
        "Geodetic ground point beneath the spacecraft.",
    )
    SUBSOLAR = (
        "subsolar",
        ("subsolar_latitude", "subsolar_longitude", "subsolar_colatitude"),
        "Geodetic ground point beneath the Sun.",
    )
    EARTH_SUN_DISTANCE = ("earth_sun_distance", ("earth_sun_distance",), "Earth-Sun distance (AU).")
    SC_POSITION = (
        "sc_position",
        ("spacecraft_position_x", "spacecraft_position_y", "spacecraft_position_z"),
        "Spacecraft position (ECEF, km).",
    )
    SC_ALTITUDE = ("sc_altitude", ("spacecraft_altitude",), "Observer geodetic height above the ellipsoid (km).")
    BORESIGHT = ("boresight", ("boresight_x", "boresight_y", "boresight_z"), "Instrument boresight unit vector (ECEF).")
    SURFACE_COLATITUDE = (
        "surface_colatitude",
        ("surface_colatitude",),
        "Colatitude of the boresight ellipsoid intersection.",
    )
    VIEWING_ZENITH = (
        "viewing_zenith",
        ("viewing_zenith",),
        "Geodetic zenith of the satellite at the boresight intersection.",
    )
    SOLAR_ZENITH = (
        "solar_zenith",
        ("solar_zenith",),
        "Geodetic zenith of the Sun at the boresight intersection.",
    )
    VIEWING_AZIMUTH = (
        "viewing_azimuth",
        ("viewing_azimuth",),
        "Satellite azimuth (clockwise from North) at the boresight intersection.",
    )
    SOLAR_AZIMUTH = (
        "solar_azimuth",
        ("solar_azimuth",),
        "Solar azimuth (clockwise from North) at the boresight intersection.",
    )
    RELATIVE_AZIMUTH = (
        "relative_azimuth",
        ("relative_azimuth",),
        "Viewing azimuth relative to solar azimuth (CERES origin, unfolded).",
    )
    CONE_ANGLE = (
        "cone_angle",
        ("cone_angle",),
        "Boresight angle off the satellite-to-geocenter (nadir) vector.",
    )

    def __new__(cls, value, columns=(), description=""):
        member = str.__new__(cls, value)
        member._value_ = value
        member._columns = tuple(columns)
        member._description = description
        return member

    def __str__(self):
        return self._value_

    @property
    def columns(self) -> tuple[str, ...]:
        """Output column names this field expands to."""
        return self._columns

    @property
    def description(self) -> str:
        """One-line description of the field."""
        return self._description
