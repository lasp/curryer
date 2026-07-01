"""Selective geometric data-field computation.

This module computes the geolocation/geometry ancillary data fields that
geolocated products commonly need. It is organized in two layers, both
mission-generic:

- **Math-only leaf functions** (``sc_radius``, ``colatitude``,
  ``subobserver_point``, ``earth_sun_distance``, ``satellite_altitude``) -- pure,
  vectorized, SPICE-free.
  They take high-level inputs (positions) as arguments and can be called directly
  to compose custom fields; the coordinate/frame primitives they build on (e.g.
  ``ecef_to_geodetic``) stay in :mod:`curryer.compute.spatial`.
- **A selective-compute registry** -- a declarative map from each output field to
  the SPICE-derived inputs ("providers") it needs. A caller requests an arbitrary
  subset of fields and only the inputs that subset needs are queried, each
  provider exactly once, never per-field. This is the pre-built convenience over
  the leaf tools.

Why a registry rather than one method per field: the field-to-input mapping is
many-to-many and growing. The registry computes the minimal provider set for any
requested subset automatically; the naive alternative (each field querying its
own inputs) re-queries SPICE redundantly.

Two public accessors are exposed on :class:`GeometryData`:

- ``get_geometry(ugps_times, fields)`` -> :class:`pandas.DataFrame` -- the
  primary, tabular API. Vector fields expand to per-field-prefixed columns.
- ``get_vectors(ugps_times, fields)`` -> ``{field: (N, k) ndarray}`` -- the typed
  sibling, addressed by field name rather than by string-built column prefixes.

Mission genericity: the observing body is the only mission input (a constructor
argument). Only universal identifiers (``EARTH``, ``SUN``, ``ITRF93``) appear in
code -- no spacecraft/instrument names are hardcoded.

Frame contract: the Earth-fixed (ECEF) fields all share a single reference frame
-- the ``earth_frame`` configured on :class:`GeometryData` (``ITRF93`` by
default) -- so fields are never combined in mismatched reference frames.

Fill contract: SPICE coverage gaps (and off-Earth samples) surface as NaN. The
providers query with ``allow_nans`` (True by default), so an uncovered time
yields a NaN provider row; the math-only leaves are pure and propagate NaN
elementwise, so every field is NaN on exactly the rows its inputs are missing and
finite elsewhere -- per time, per field. Downstream maps those NaNs onto the
product ``_FillValue`` (e.g. -999); the rows are never dropped or back-filled.

Available fields: request any by name (``GeometryData.available_fields()`` lists
them). Each field expands to the columns below.

- ``sc_radius`` -> ``spacecraft_radius`` -- observer distance from Earth center.
- ``subsatellite`` -> ``subsatellite_latitude``, ``subsatellite_longitude``,
  ``subsatellite_colatitude`` -- ground point beneath the spacecraft.
- ``subsolar`` -> ``subsolar_latitude``, ``subsolar_longitude``,
  ``subsolar_colatitude`` -- ground point beneath the Sun.
- ``earth_sun_distance`` -> ``earth_sun_distance`` -- Earth-Sun distance (AU).
- ``sc_position`` -> ``spacecraft_position_x``, ``spacecraft_position_y``,
  ``spacecraft_position_z`` -- spacecraft position (ECEF).
- ``sc_altitude`` -> ``spacecraft_altitude`` -- observer geodetic height above the
  ellipsoid (km).
- ``boresight`` -> ``boresight_x``, ``boresight_y``, ``boresight_z`` --
  instrument boresight unit vector (ECEF).
- ``surface_colatitude`` -> ``surface_colatitude`` -- colatitude of the boresight
  ellipsoid intersection.
- ``viewing_zenith`` / ``solar_zenith`` -> same-named columns -- geodetic zenith of
  the satellite / Sun at the boresight ellipsoid intersection.
- ``viewing_azimuth`` / ``solar_azimuth`` -> same-named columns -- satellite / Sun
  azimuth at that intersection.
- ``relative_azimuth`` -> ``relative_azimuth`` -- viewing relative to solar azimuth.
- ``cone_angle`` -> ``cone_angle`` -- boresight angle off the satellite-to-geocenter
  vector.

Angle convention: the surface angles are in degrees, over the boresight ellipsoid
intersection. Azimuths (``viewing_azimuth``, ``solar_azimuth``) are clockwise from
geodetic North in [0, 360); zeniths (``viewing_zenith``, ``solar_zenith``) are
geodetic, from the local surface normal, in [0, 180]. ``relative_azimuth`` uses the
CERES BDS R3V4 origin -- ``mod(viewing_azimuth - solar_azimuth + 180, 360)``, so the
Sun sits at 180 -- and is the lossless unfolded value; the CERES [0, 180] *fold*
(``min(raa, 360 - raa)``) is a separate, lossy, downstream step. ``cone_angle`` is
in [0, 90] for Earth-disk views.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import spicierpy
from . import abstract, constants, spatial
from .constants import GeometryField

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math-only leaf functions (the "tools" layer): pure, vectorized, SPICE-free.
# Usable directly or composed by the registry below.
# ---------------------------------------------------------------------------
def sc_radius(observer_position: np.ndarray) -> np.ndarray:
    """Distance of the observer from the body center (vectorized).

    Implements the "radius of satellite from center of Earth" field.

    Parameters
    ----------
    observer_position : np.ndarray
        Observer (e.g., spacecraft) positions in rectangular coordinates, shape
        (..., 3). Units are arbitrary; the result is in the input's units.

    Returns
    -------
    np.ndarray
        Euclidean distance ``||position||`` per point, shape (...,).

    """
    observer_position = np.asarray(observer_position, dtype=float)
    if observer_position.shape[-1] != 3:
        raise ValueError("`observer_position` must have 3 values per point!")
    return np.linalg.norm(observer_position, axis=-1)


def colatitude(latitude: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Convert geodetic latitude to colatitude (vectorized).

    Colatitude is the complement of latitude (``90 - lat``), ranging from 0 at
    the north pole to 180 at the south pole. Implements the surface-point
    colatitude field; also reused by the sub-point fields.

    Parameters
    ----------
    latitude : np.ndarray
        Geodetic latitude.
    degrees : bool, optional
        If True (default), inputs and outputs are in degrees, otherwise radians.

    Returns
    -------
    np.ndarray
        Colatitude in the same units as the input.

    """
    quarter_turn = 90.0 if degrees else np.pi / 2
    return quarter_turn - np.asarray(latitude, dtype=float)


def subobserver_point(observer_position: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Sub-observer geodetic latitude, longitude, and colatitude (vectorized).

    The sub-observer point is the geodetic ground point directly beneath the
    observer; it shares the observer's geodetic latitude and longitude. Generic
    over the observing body: pass the spacecraft position for the sub-satellite
    point or the Sun position for the sub-solar point.

    Parameters
    ----------
    observer_position : np.ndarray
        Observer positions in ECEF rectangular coordinates (km), shape (..., 3).
    degrees : bool, optional
        If True (default), latitude/longitude/colatitude are in degrees,
        otherwise radians. Longitude follows ``ecef_to_geodetic`` (-180 to 180).

    Returns
    -------
    np.ndarray
        Stacked ``[latitude, longitude, colatitude]``, shape (..., 3).

    """
    lla = spatial.ecef_to_geodetic(observer_position, meters=False, degrees=degrees)
    lat = lla[..., 1]
    lon = lla[..., 0]
    colat = colatitude(lat, degrees=degrees)
    return np.stack([lat, lon, colat], axis=-1)


def earth_sun_distance(earth_sun_position: np.ndarray, au: bool = True) -> np.ndarray:
    """Distance between Earth and Sun (vectorized).

    Implements the Earth-Sun distance field. The input is the
    Earth-to-Sun position vector (any frame, since only the magnitude is used).

    Parameters
    ----------
    earth_sun_position : np.ndarray
        Earth-to-Sun position in rectangular coordinates (km), shape (..., 3).
    au : bool, optional
        If True (default), return astronomical units, otherwise kilometers.

    Returns
    -------
    np.ndarray
        Earth-Sun distance per point, shape (...,).

    """
    distance = np.linalg.norm(np.asarray(earth_sun_position, dtype=float), axis=-1)
    return distance / constants.KM_PER_ASTRONOMICAL_UNIT if au else distance


def satellite_altitude(observer_position: np.ndarray) -> np.ndarray:
    """Geodetic altitude of the observer above the ellipsoid (vectorized).

    The height-above-ellipsoid component of the observer's geodetic position -- the
    piece ``subobserver_point`` drops when it returns latitude/longitude/colatitude.
    Complements ``sc_radius`` (the geocentric distance from the body center).

    Parameters
    ----------
    observer_position : np.ndarray
        Observer (e.g., spacecraft) positions in ECEF rectangular coordinates (km),
        shape (..., 3).

    Returns
    -------
    np.ndarray
        Geodetic altitude in km, shape (...,).

    """
    lla = spatial.ecef_to_geodetic(observer_position, meters=False, degrees=True)
    return lla[..., 2]


@dataclass(frozen=True)
class _Field:
    """Registry entry for a single output field.

    Parameters
    ----------
    providers : frozenset of str
        Keys of the providers (high-level inputs) this field requires.
    columns : tuple of str
        Output column names. One column for scalar fields, three for vectors.
    evaluate : Callable
        Maps the gathered provider dict to an ``(N, len(columns))`` array.

    """

    providers: frozenset[str]
    columns: tuple[str, ...]
    evaluate: Callable[[dict], np.ndarray]


# ---------------------------------------------------------------------------
# Providers: each queries SPICE once over the ugps grid and returns a cached
# array. The observing body and frames come from the caller's ``GeometryData``.
# ---------------------------------------------------------------------------
def _provider_sc_position(ugps_times, ctx):
    """Observer (spacecraft) position in the configured Earth-fixed frame
    (``ctx.earth_frame``, ``ITRF93`` by default), shape (N, 3), km."""
    state = spicierpy.ext.query_ephemeris(
        ugps_times,
        target=ctx.observer,
        observer=ctx.earth,
        ref_frame=ctx.earth_frame,
        velocity=False,
        allow_nans=ctx.allow_nans,
    )
    return state[list(spicierpy.ext.POSITION_COLUMNS)].values


def _provider_sun_position(ugps_times, ctx):
    """Earth-to-Sun position in the configured Earth-fixed frame
    (``ctx.earth_frame``, ``ITRF93`` by default), shape (N, 3), km."""
    state = spicierpy.ext.query_ephemeris(
        ugps_times,
        target=ctx.sun,
        observer=ctx.earth,
        ref_frame=ctx.earth_frame,
        velocity=False,
        allow_nans=ctx.allow_nans,
    )
    return state[list(spicierpy.ext.POSITION_COLUMNS)].values


# The boresight is a pure attitude quantity: the IK boresight rotated from the
# instrument frame into ``earth_frame`` via ``spatial.frame_to_frame_rotation`` (the
# shared per-sample ``pxform`` primitive). It queries no ephemeris -- position comes
# from the ``sc_position`` provider -- so requesting the boresight beside the position
# fields costs one attitude pass plus one shared ephemeris pass, never a duplicate.
# Recoverable SPICE failures NaN-fill under ``allow_nans``: a missing FOV/frame is
# logged and fills every row, and per-sample attitude gaps come back as NaN rows from
# the rotation primitive.
def _provider_boresight(ugps_times, ctx):
    """Instrument boresight unit vector in the configured Earth-fixed frame.

    Resolves the IK boresight in the instrument frame, then rotates it into
    ``ctx.earth_frame`` (``ITRF93`` by default) with
    :func:`curryer.compute.spatial.frame_to_frame_rotation`. The one-time pointing
    lookup (instrument frame / IK boresight, which fails for a body with no defined
    FOV) is guarded: under ``ctx.allow_nans`` a failure is logged and NaN-fills, and
    without it the underlying SPICE error is raised. Per-sample attitude gaps surface
    as NaN rows from the rotation, honoring the module fill contract.

    Parameters
    ----------
    ugps_times : array_like of int
        Times in GPS microseconds at which to evaluate the boresight.
    ctx : GeometryData
        Supplies the observing body (``ctx.observer``), the target Earth-fixed
        frame (``ctx.earth_frame``), and the NaN-fill toggle (``ctx.allow_nans``).

    Returns
    -------
    numpy.ndarray
        Boresight unit vectors in ``ctx.earth_frame``, shape (N, 3). Rows where
        the attitude/FOV is unavailable are NaN (under ``allow_nans``).
    """
    ugps_times = np.atleast_1d(np.asarray(ugps_times))

    @spicierpy.ext.spice_error_to_val(err_value=None, err_flag=lambda err: err, disable=not ctx.allow_nans)
    def _resolve_pointing():
        from_frame = spicierpy.obj.Body(ctx.observer, frame=True).frame.name
        to_frame = spicierpy.obj.Frame(ctx.earth_frame).name
        boresight = spicierpy.ext.instrument_boresight(ctx.observer, norm=True)
        return from_frame, to_frame, boresight

    pointing, error = _resolve_pointing()
    if pointing is None:  # missing IK/FOV or frame -> surface the error and NaN-fill (allow_nans only).
        logger.warning("Boresight unavailable for observer %r: %s", ctx.observer, error)
        return np.full((ugps_times.size, 3), np.nan)
    from_frame, to_frame, boresight = pointing

    matrices = spatial.frame_to_frame_rotation(from_frame, to_frame, ugps_times, allow_nans=ctx.allow_nans)
    return np.einsum("nij,j->ni", matrices, boresight)


_PROVIDERS = {
    "sc_position": _provider_sc_position,
    "sun_position": _provider_sun_position,
    "boresight": _provider_boresight,
}


# ---------------------------------------------------------------------------
# Field-layer helpers: pure math over already-queried providers (no SPICE).
# These compose the spatial leaves the surface-angle fields share.
# ---------------------------------------------------------------------------
_INTERSECTION_KEY = "boresight_intersection"  # cache slot in the per-request providers dict


def _boresight_intersection(providers):
    """ECEF point where the instrument's boresight ray, cast from the S/C position,
    meets the ellipsoid -- the shared ground point for the surface angles.

    The ``boresight`` is the instrument's single IK boresight (populated by
    ``_provider_boresight``), not an arbitrary or per-pixel pointing vector: this
    server is boresight-only by design (see :class:`GeometryData` for the scope and
    the multi-pixel alternative). A boresight that misses the ellipsoid (off-disk /
    deep-space pointing) yields NaN -- ``ray_intersect_ellipsoid`` is a direct
    ray-ellipsoid intersection, not a nearest-point projection, so it returns no
    PNEAR/DIST.

    Takes the per-request ``providers`` dict (rather than the two arrays directly) so
    it can memoize: the ray-cast runs once on first use and the result is cached under
    ``_INTERSECTION_KEY`` for every other surface-angle field in the same request to
    reuse.

    Parameters
    ----------
    providers : dict
        Per-request provider results. Reads ``boresight`` (the IK boresight unit
        vectors, ECEF) and ``sc_position`` (the ray origin, ECEF); mutated to cache
        the intersection under ``_INTERSECTION_KEY``.

    Returns
    -------
    numpy.ndarray
        Intersection points in ECEF, shape (N, 3); NaN rows where the boresight
        misses the ellipsoid.
    """
    intersection = providers.get(_INTERSECTION_KEY)
    if intersection is None:
        intersection = spatial.ray_intersect_ellipsoid(providers["boresight"], providers["sc_position"])
        providers[_INTERSECTION_KEY] = intersection
    return intersection


def _relative_azimuth(providers):
    """Relative azimuth of the viewing direction about the solar plane, in [0, 360).

    ``mod(viewing_azimuth - solar_azimuth + 180, 360)`` (azimuths clockwise from
    geodetic North per ``spatial.calc_azimuth``), matching the CERES BDS R3V4 origin
    so the Sun sits at 180. This is the lossless full-range value -- it keeps which
    side of the principal plane the geometry is on. The CERES [0, 180] *fold*
    (``min(raa, 360 - raa)``) is a separate, lossy step applied downstream if a
    product wants it; curryer keeps the unfolded value, since a reference/wrap shift
    is reversible but a fold is not.

    Parameters
    ----------
    providers : dict
        Per-request provider results; uses ``boresight``, ``sc_position``, and
        ``sun_position`` (via the shared boresight intersection).

    Returns
    -------
    numpy.ndarray
        Relative azimuth in degrees, shape (N,), in [0, 360); NaN where the
        boresight misses the ellipsoid.
    """
    intersection = _boresight_intersection(providers)
    view_az = spatial.calc_azimuth(intersection, providers["sc_position"], degrees=True)
    sun_az = spatial.calc_azimuth(intersection, providers["sun_position"], degrees=True)
    return np.mod(view_az - sun_az + 180.0, 360.0)


def _cone_angle(providers):
    """Angle between the boresight and the satellite-to-geocenter direction.

    The cone angle (CERES BDS R3V4 SCI-18) is the off-nadir angle of the boresight
    measured from the vector pointing from the spacecraft to Earth's center
    (``-sc_position``). In [0, 90] for Earth-disk views; larger toward the limb.

    Parameters
    ----------
    providers : dict
        Per-request provider results; uses ``boresight`` (unit vectors, ECEF) and
        ``sc_position`` (ECEF).

    Returns
    -------
    numpy.ndarray
        Cone angle in degrees, shape (N,).
    """
    sc_position = providers["sc_position"]
    nadir = -sc_position / np.linalg.norm(sc_position, axis=-1, keepdims=True)
    cos_angle = np.sum(providers["boresight"] * nadir, axis=-1)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


# Field registry: each field names its providers and an evaluate() over the gathered
# provider dict. Column names live on ``GeometryField`` (the single source) and are
# mirrored onto each ``_Field`` for lookup.
_FIELD_SPECS = (
    (GeometryField.SC_RADIUS, {"sc_position"}, lambda p: sc_radius(p["sc_position"])[:, None]),
    (GeometryField.SUBSATELLITE, {"sc_position"}, lambda p: subobserver_point(p["sc_position"])),
    (GeometryField.SUBSOLAR, {"sun_position"}, lambda p: subobserver_point(p["sun_position"])),
    (GeometryField.EARTH_SUN_DISTANCE, {"sun_position"}, lambda p: earth_sun_distance(p["sun_position"])[:, None]),
    (GeometryField.SC_POSITION, {"sc_position"}, lambda p: p["sc_position"]),
    (GeometryField.SC_ALTITUDE, {"sc_position"}, lambda p: satellite_altitude(p["sc_position"])[:, None]),
    (GeometryField.BORESIGHT, {"boresight"}, lambda p: p["boresight"]),
    # The boresight intersection is where the boresight, cast from the S/C position, meets the
    # ellipsoid; ``ray_intersect_ellipsoid`` returns geodetic [lon, lat, alt], so column 1 is latitude.
    (
        GeometryField.SURFACE_COLATITUDE,
        {"boresight", "sc_position"},
        lambda p: colatitude(
            spatial.ray_intersect_ellipsoid(p["boresight"], p["sc_position"], geodetic=True, degrees=True)[:, 1]
        )[:, None],
    ),
    # Surface angles -- pure math over the boresight ellipsoid intersection and the shared
    # ephemeris providers; see the module "Angle convention" note. No new SPICE.
    (
        GeometryField.VIEWING_ZENITH,
        {"boresight", "sc_position"},
        lambda p: spatial.calc_zenith(_boresight_intersection(p), p["sc_position"], degrees=True)[:, None],
    ),
    (
        GeometryField.SOLAR_ZENITH,
        {"boresight", "sc_position", "sun_position"},
        lambda p: spatial.calc_zenith(_boresight_intersection(p), p["sun_position"], degrees=True)[:, None],
    ),
    (
        GeometryField.VIEWING_AZIMUTH,
        {"boresight", "sc_position"},
        lambda p: spatial.calc_azimuth(_boresight_intersection(p), p["sc_position"], degrees=True)[:, None],
    ),
    (
        GeometryField.SOLAR_AZIMUTH,
        {"boresight", "sc_position", "sun_position"},
        lambda p: spatial.calc_azimuth(_boresight_intersection(p), p["sun_position"], degrees=True)[:, None],
    ),
    (
        GeometryField.RELATIVE_AZIMUTH,
        {"boresight", "sc_position", "sun_position"},
        lambda p: _relative_azimuth(p)[:, None],
    ),
    (GeometryField.CONE_ANGLE, {"boresight", "sc_position"}, lambda p: _cone_angle(p)[:, None]),
)

_FIELDS = {
    field: _Field(providers=frozenset(providers), columns=field.columns, evaluate=evaluate)
    for field, providers, evaluate in _FIELD_SPECS
}


# Providers available for any observing body (queried from ephemeris alone).
_EPHEMERIS_PROVIDERS = frozenset({"sc_position", "sun_position"})

# Default field set for ``fields=None``: every field whose providers are a subset
# (``<=``) of the ephemeris providers -- i.e. computable from position (an SPK)
# alone, which every observer has. The attitude fields (boresight, and
# surface_colatitude via it) need more: the spacecraft attitude (a CK, to rotate
# into the Earth-fixed frame) plus an instrument FOV (IK boresight + FK frame). They
# are opt-in so the no-args call need not furnish a CK or use an instrument observer
# -- not because a spacecraft lacks attitude (it has its own CK), but because these
# fields require those extra inputs. Self-maintaining via the providers.
_DEFAULT_FIELDS = tuple(name for name, field in _FIELDS.items() if field.providers <= _EPHEMERIS_PROVIDERS)


class GeometryData(abstract.AbstractMissionData):
    """Selective geometric data-field server.

    Construct with the observing body, then request any subset of registered
    fields via :meth:`get_geometry` (DataFrame) or :meth:`get_vectors` (typed
    ``{field: ndarray}``). Only the SPICE inputs the requested subset needs are
    queried, each exactly once. Relevant kernels must already be loaded for the
    requested times, mirroring the other ``compute`` servers.

    The requested times are the independent variable: :meth:`get_geometry` and
    :meth:`get_vectors` evaluate every field at exactly the ``ugps_times`` passed
    -- an arbitrary, possibly non-uniform array -- with no resampling or
    interpolation. ``microsecond_cadence`` / :meth:`get_times` only offers an
    optional uniform grid for callers that want one; it never constrains which
    times are queried.

    Scope -- this server produces *per-observation* ancillary geometry: one row per
    requested time. Its fields are of two kinds: spacecraft-level quantities that are
    independent of pointing (subsatellite/subsolar points, radius, altitude, Earth-Sun
    distance) and angles referenced to the instrument's *single* boresight -- the IK
    boresight (viewing/solar zenith and azimuth, relative azimuth, cone angle, surface
    colatitude). It deliberately does *not* compute per-pixel geometry: an instrument
    with many pointing vectors (a camera focal plane, an offset pixel) is a
    ``(time, pixel)`` problem served by
    :func:`~curryer.compute.spatial.compute_ellipsoid_intersection`
    (``custom_pointing_vectors=``), whose vectorized ray-cast and per-pixel quality
    flags belong with the geolocated product rather than this ancillary table. A future
    mission needing a single *non-boresight* reference vector here (still one vector, not
    the array) is a natural additive extension, not a change to this contract.

    Parameters
    ----------
    observer : str or int or spicierpy.obj.Body
        Observing body whose position is queried -- typically the instrument or
        spacecraft.
    microsecond_cadence : int, optional
        Default cadence for :meth:`get_times`, in microseconds.
    earth, sun : str or int or spicierpy.obj.Body, optional
        Central body and solar body for the ephemeris queries. Default
        ``"EARTH"`` / ``"SUN"``; overridable so no body name is fixed in code.
    earth_frame : str or spicierpy.obj.Frame, optional
        Earth-fixed (ECEF) reference frame. Default ``spatial.EARTH_FRAME``
        (ITRF93).

    """

    DEFAULT_CADENCE = constants.EPHEMERIS_TIMESTEP_USEC

    def __init__(self, observer, microsecond_cadence=None, earth="EARTH", sun="SUN", earth_frame=None):
        microsecond_cadence = self.DEFAULT_CADENCE if microsecond_cadence is None else microsecond_cadence
        super().__init__(microsecond_cadence=microsecond_cadence)
        # Store raw names; SPICE objects are resolved lazily inside the providers
        # (within the caller's loaded-kernel context), matching the other servers.
        self.observer = observer
        self.earth = earth
        self.sun = sun
        self.earth_frame = spatial.EARTH_FRAME if earth_frame is None else earth_frame

    @classmethod
    def available_fields(cls):
        """Registered fields as :class:`~curryer.compute.constants.GeometryField` members.

        Members are plain strings, so the result is usable directly as
        ``fields=`` selectors and each member carries its ``columns`` and
        ``description``.
        """
        return tuple(_FIELDS)

    def _resolve_fields(self, fields):
        """Validate the requested fields, defaulting to the ephemeris-only set."""
        if fields is None:
            return list(_DEFAULT_FIELDS)
        unknown = [name for name in fields if name not in _FIELDS]
        if unknown:
            available = [str(field) for field in self.available_fields()]
            raise KeyError(f"Unknown geometry field(s): {unknown}. Available: {available}")
        return list(fields)

    def _gather_providers(self, fields, ugps_times):
        """Query the minimal set of providers for ``fields``, once each.

        Providers are evaluated in a stable (sorted) order so runs are
        reproducible. A provider that comes back entirely NaN almost always means
        the required kernels are not furnished (or do not cover the requested
        span) rather than a genuine per-sample gap, so it is logged as a warning to
        surface the likely misconfiguration. The all-NaN result still propagates
        per the fill contract; set the instance ``allow_nans`` attribute to False
        to raise on the underlying SPICE error instead.
        """
        needed = sorted(set().union(*(_FIELDS[name].providers for name in fields)))
        logger.debug("Querying providers %s for fields %s", needed, fields)
        providers = {}
        for key in needed:
            values = _PROVIDERS[key](ugps_times, self)
            if np.size(values) and np.all(np.isnan(np.asarray(values, dtype=float))):
                logger.warning(
                    "Provider %r returned all-NaN over %d time(s); check that the required kernels "
                    "are furnished and cover the requested span.",
                    key,
                    len(ugps_times),
                )
            providers[key] = values
        return providers

    @abstract.log_return()
    def get_geometry(self, ugps_times: np.ndarray, fields: list[str] | None = None) -> pd.DataFrame:
        """Compute the requested fields as a table.

        Parameters
        ----------
        ugps_times : array_like of int
            One or more times in GPS microseconds. Arbitrary and need not be
            uniformly spaced; each time is evaluated exactly (no interpolation).
        fields : list of str, optional
            Field names to compute. Default is the ephemeris-only set (valid for
            any observer); attitude/instrument fields (e.g. ``boresight``) must be
            requested explicitly. See :meth:`available_fields` for the full list.

        Returns
        -------
        pandas.DataFrame
            One row per time (index ``ugps``); vector fields expand to
            per-field-prefixed columns (e.g. ``spacecraft_position_x,
            spacecraft_position_y, spacecraft_position_z``). Times outside SPICE
            coverage are NaN across that field's columns (see the module Fill
            contract).

        """
        ugps_times = np.atleast_1d(np.asarray(ugps_times))
        fields = self._resolve_fields(fields)
        providers = self._gather_providers(fields, ugps_times)

        data = {}
        for name in fields:
            field = _FIELDS[name]
            values = np.asarray(field.evaluate(providers), dtype=float)
            for jth, column in enumerate(field.columns):
                data[column] = values[:, jth]
        return pd.DataFrame(data, index=pd.Index(ugps_times, name="ugps"))

    def get_vectors(self, ugps_times: np.ndarray, fields: list[str]) -> dict[str, np.ndarray]:
        """Compute the requested fields as typed arrays.

        The typed sibling of :meth:`get_geometry`, addressed by field name rather
        than by string-built column prefixes.

        Parameters
        ----------
        ugps_times : array_like of int
            One or more times in GPS microseconds. Arbitrary and need not be
            uniformly spaced; each time is evaluated exactly (no interpolation).
        fields : list of str
            Field names to compute.

        Returns
        -------
        dict of {str: numpy.ndarray}
            Maps each field name to its ``(N, k)`` array. Vector fields
            (e.g. ``sc_position``) are ``(N, 3)`` in the configured Earth-fixed
            frame (``ITRF93`` by default). Rows outside SPICE coverage are NaN
            (see the module Fill contract).

        """
        ugps_times = np.atleast_1d(np.asarray(ugps_times))
        fields = self._resolve_fields(fields)
        providers = self._gather_providers(fields, ugps_times)
        return {name: np.asarray(_FIELDS[name].evaluate(providers), dtype=float) for name in fields}
