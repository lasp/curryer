"""The :class:`GeometryField` registry for :class:`curryer.compute.geometry.GeometryData`.

Kept in its own module so downstream consumers can import the geometry field identifiers
without pulling in the broader constants module.
"""

from enum import Enum


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
    CONE_ANGLE_RATE = (
        "cone_angle_rate",
        ("cone_angle_rate",),
        "Rate of change of the cone angle (deg/s), finite-differenced over the requested times.",
    )
    SC_VELOCITY = (
        "sc_velocity",
        ("spacecraft_velocity_x", "spacecraft_velocity_y", "spacecraft_velocity_z"),
        "Spacecraft velocity (ECEF, km/s).",
    )
    SATELLITE_ATTITUDE = (
        "satellite_attitude",
        ("attitude_q0", "attitude_q1", "attitude_q2", "attitude_q3"),
        "Spacecraft body attitude quaternion (body -> inertial, scalar-first).",
    )
    CLOCK_ANGLE = (
        "clock_angle",
        ("clock_angle",),
        "Boresight azimuth in the inertial-velocity orbital frame (CERES SCI-12), [0, 360).",
    )
    CLOCK_ANGLE_RATE = (
        "clock_angle_rate",
        ("clock_angle_rate",),
        "Rate of change of the clock angle (deg/s), unwrapped finite difference.",
    )
    ALONG_TRACK_ANGLE = (
        "along_track_angle",
        ("along_track_angle",),
        "Boresight look angle from nadir in the velocity-nadir plane (forward +).",
    )
    CROSS_TRACK_ANGLE = (
        "cross_track_angle",
        ("cross_track_angle",),
        "Boresight look angle from nadir in the cross-track-nadir plane.",
    )
    SC_POSITION_INERTIAL = (
        "sc_position_inertial",
        (
            "spacecraft_position_inertial_x",
            "spacecraft_position_inertial_y",
            "spacecraft_position_inertial_z",
        ),
        "Spacecraft position (inertial, km).",
    )
    SC_VELOCITY_INERTIAL = (
        "sc_velocity_inertial",
        (
            "spacecraft_velocity_inertial_x",
            "spacecraft_velocity_inertial_y",
            "spacecraft_velocity_inertial_z",
        ),
        "Spacecraft velocity (inertial, km/s).",
    )
    BORESIGHT_INERTIAL = (
        "boresight_inertial",
        ("boresight_inertial_x", "boresight_inertial_y", "boresight_inertial_z"),
        "Instrument boresight unit vector (inertial).",
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
