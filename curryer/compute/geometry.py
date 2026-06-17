"""Selective geometric data-field computation.

This module computes the geolocation/geometry ancillary data fields the Level-1
products require. It is organized in two layers, both mission-generic:

- **Math-only leaf functions** (``sc_radius``, ``colatitude``,
  ``subobserver_point``, ``earth_sun_distance``) -- pure, vectorized, SPICE-free.
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

Angle convention: the surface angles share one documented convention, in degrees,
over the same boresight ellipsoid footprint as their surface point. Azimuths
(``viewing_azimuth``, and the viewing and solar azimuths from which
``relative_azimuth`` is derived) are measured clockwise from geodetic North in
[0, 360); zeniths (``viewing_zenith``, ``solar_zenith``) are geodetic, from the
local surface normal. ``relative_azimuth`` is the lossless ``viewing - solar``
difference wrapped to [0, 360). A mission needing a different wrap -- e.g. the CERES [0, 180] fold -- converts on its end;
curryer deliberately keeps the unfolded value, since reference/wrap remaps are
reversible but a fold is not, and only the unfolded form lets every consumer adapt.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import spicetime, spicierpy
from . import abstract, constants, spatial

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


def colatitude(latitude: np.ndarray, degrees=True) -> np.ndarray:
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


def subobserver_point(observer_position: np.ndarray, degrees=True) -> np.ndarray:
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


def earth_sun_distance(earth_sun_position: np.ndarray, au=True) -> np.ndarray:
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

    providers: frozenset
    columns: tuple
    evaluate: Callable


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
# instrument frame into ``earth_frame`` by ``pxform`` per sample (``pxform`` has no
# vectorized override, so it loops). It queries no ephemeris -- position comes from
# the ``sc_position`` provider -- so requesting the boresight beside the position
# fields costs one attitude pass plus one shared ephemeris pass, never a duplicate.
# The SPICE-error guard covers the lookup and the rotation (but no ephemeris), so a
# NaN row tracks an attitude gap alone, independent of position coverage.
def _provider_boresight(ugps_times, ctx):
    """Instrument boresight unit vector in the configured Earth-fixed frame
    (``ctx.earth_frame``, ``ITRF93`` by default), shape (N, 3).

    Both the one-time pointing lookup (instrument frame / IK boresight, which
    fails for a body with no defined FOV) and the per-sample rotation are guarded:
    a recoverable SPICE failure NaN-fills under ``allow_nans`` and raises without
    it, so the whole provider honors the module fill contract.
    """
    et_times = spicetime.adapt(ugps_times, to="et")

    @spicierpy.ext.spice_error_to_val(err_value=None, disable=not ctx.allow_nans)
    def _resolve_pointing():
        from_frame = spicierpy.obj.Body(ctx.observer, frame=True).frame.name
        to_frame = spicierpy.obj.Frame(ctx.earth_frame).name
        boresight = spicierpy.ext.instrument_boresight(ctx.observer, norm=True)
        return from_frame, to_frame, boresight

    pointing, _ = _resolve_pointing()
    if pointing is None:  # missing IK/FOV or frame -> NaN-fill (allow_nans only).
        return np.full((et_times.size, 3), np.nan)
    from_frame, to_frame, boresight = pointing

    @spicierpy.ext.spice_error_to_val(err_value=np.full(3, np.nan), disable=not ctx.allow_nans)
    def _rotate(sample_et):
        return spicierpy.pxform(from_frame, to_frame, sample_et) @ boresight

    return np.array([_rotate(sample_et)[0] for sample_et in et_times])


_PROVIDERS = {
    "sc_position": _provider_sc_position,
    "sun_position": _provider_sun_position,
    "boresight": _provider_boresight,
}


# ---------------------------------------------------------------------------
# Field-layer helpers: pure math over already-queried providers (no SPICE).
# These compose the spatial leaves the surface-angle fields share.
# ---------------------------------------------------------------------------
_FOOTPRINT_KEY = "footprint"  # cache slot in the per-request providers dict


def _footprint(providers):
    """ECEF point where the instrument boresight, cast from the S/C position,
    meets the ellipsoid -- the shared surface point for the surface angles.

    Memoized into the per-request ``providers`` dict: every surface-angle field in
    a request shares the same footprint, so the ray-cast runs once per request
    rather than once per field.
    """
    footprint = providers.get(_FOOTPRINT_KEY)
    if footprint is None:
        footprint = spatial.ray_intersect_ellipsoid(providers["boresight"], providers["sc_position"])
        providers[_FOOTPRINT_KEY] = footprint
    return footprint


def _relative_azimuth(providers):
    """Relative azimuth between the viewing and solar directions, in [0, 360).

    Defined as ``viewing_azimuth - solar_azimuth`` (both measured clockwise from
    geodetic North per ``spatial.calc_azimuth``) wrapped to [0, 360). This is the
    lossless full-range form -- it retains which side of the principal plane the
    geometry is on. The CERES BDS R3V4 convention folds this to [0, 180] via
    ``min(raa, 360 - raa)`` (and may apply a principal-plane origin offset); apply
    that downstream as the product requires. Curryer keeps the unfolded value so
    the fold (which is not reversible) is always available to do, never undone.
    """
    footprint = _footprint(providers)
    view_az = spatial.calc_azimuth(footprint, providers["sc_position"], degrees=True)
    sun_az = spatial.calc_azimuth(footprint, providers["sun_position"], degrees=True)
    return np.mod(view_az - sun_az, 360.0)


_FIELDS = {
    "sc_radius": _Field(
        providers=frozenset({"sc_position"}),
        columns=("scradius",),
        evaluate=lambda p: sc_radius(p["sc_position"])[:, None],
    ),
    "subsatellite": _Field(
        providers=frozenset({"sc_position"}),
        columns=("subsatlat", "subsatlon", "subsatcolat"),
        evaluate=lambda p: subobserver_point(p["sc_position"]),
    ),
    "subsolar": _Field(
        providers=frozenset({"sun_position"}),
        columns=("subsollat", "subsollon", "subsolcolat"),
        evaluate=lambda p: subobserver_point(p["sun_position"]),
    ),
    "earth_sun_distance": _Field(
        providers=frozenset({"sun_position"}),
        columns=("earthsundist",),
        evaluate=lambda p: earth_sun_distance(p["sun_position"])[:, None],
    ),
    "sc_position": _Field(
        providers=frozenset({"sc_position"}),
        columns=("scx", "scy", "scz"),
        evaluate=lambda p: p["sc_position"],
    ),
    "boresight": _Field(
        providers=frozenset({"boresight"}),
        columns=("boresightx", "boresighty", "boresightz"),
        evaluate=lambda p: p["boresight"],
    ),
    "surface_colatitude": _Field(
        # The footprint is where the boresight, cast from the S/C position, meets
        # the ellipsoid; ``ray_intersect_ellipsoid`` returns geodetic
        # [lon, lat, alt], so column 1 is the latitude.
        providers=frozenset({"boresight", "sc_position"}),
        columns=("surfcolat",),
        evaluate=lambda p: colatitude(
            spatial.ray_intersect_ellipsoid(p["boresight"], p["sc_position"], geodetic=True, degrees=True)[:, 1]
        )[:, None],
    ),
    # Surface angles -- pure math over the boresight footprint and the shared
    # ephemeris providers; see the module "Angle convention" note. No new SPICE.
    "viewing_zenith": _Field(
        providers=frozenset({"boresight", "sc_position"}),
        columns=("viewzen",),
        evaluate=lambda p: spatial.calc_zenith(_footprint(p), p["sc_position"], degrees=True)[:, None],
    ),
    "solar_zenith": _Field(
        providers=frozenset({"boresight", "sc_position", "sun_position"}),
        columns=("solzen",),
        evaluate=lambda p: spatial.calc_zenith(_footprint(p), p["sun_position"], degrees=True)[:, None],
    ),
    "viewing_azimuth": _Field(
        providers=frozenset({"boresight", "sc_position"}),
        columns=("viewaz",),
        evaluate=lambda p: spatial.calc_azimuth(_footprint(p), p["sc_position"], degrees=True)[:, None],
    ),
    "relative_azimuth": _Field(
        providers=frozenset({"boresight", "sc_position", "sun_position"}),
        columns=("relaz",),
        evaluate=lambda p: _relative_azimuth(p)[:, None],
    ),
}


# Providers available for any observing body (queried from ephemeris alone).
_EPHEMERIS_PROVIDERS = frozenset({"sc_position", "sun_position"})

# Default field set for ``fields=None``: the fields needing only ephemeris, so
# ``get_geometry()`` is valid for any observer and skips the per-sample attitude
# loop. Attitude/instrument fields (needing the boresight provider, hence an
# instrument FOV) are opt-in. Derived from the providers, so it self-maintains.
_DEFAULT_FIELDS = tuple(name for name, field in _FIELDS.items() if field.providers <= _EPHEMERIS_PROVIDERS)


class GeometryData(abstract.AbstractMissionData):
    """Selective geometric data-field server.

    Construct with the observing body, then request any subset of registered
    fields via :meth:`get_geometry` (DataFrame) or :meth:`get_vectors` (typed
    ``{field: ndarray}``). Only the SPICE inputs the requested subset needs are
    queried, each exactly once. Relevant kernels must already be loaded for the
    requested times, mirroring the other ``compute`` servers.

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

    DEFAULT_CADENCE = constants.ATTITUDE_TIMESTEP_USEC

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
        """Tuple of registered field names."""
        return tuple(_FIELDS)

    def _resolve_fields(self, fields):
        """Validate the requested fields, defaulting to the ephemeris-only set."""
        if fields is None:
            return list(_DEFAULT_FIELDS)
        unknown = [name for name in fields if name not in _FIELDS]
        if unknown:
            raise KeyError(f"Unknown geometry field(s): {unknown}. Available: {self.available_fields()}")
        return list(fields)

    def _gather_providers(self, fields, ugps_times):
        """Query the minimal set of providers for ``fields``, once each."""
        needed = set().union(*(_FIELDS[name].providers for name in fields))
        logger.debug("Querying providers %s for fields %s", sorted(needed), fields)
        return {key: _PROVIDERS[key](ugps_times, self) for key in needed}

    @abstract.log_return()
    def get_geometry(self, ugps_times, fields=None) -> pd.DataFrame:
        """Compute the requested fields as a table.

        Parameters
        ----------
        ugps_times : array_like of int
            One or more times in GPS microseconds.
        fields : list of str, optional
            Field names to compute. Default is the ephemeris-only set (valid for
            any observer); attitude/instrument fields (e.g. ``boresight``) must be
            requested explicitly. See :meth:`available_fields` for the full list.

        Returns
        -------
        pandas.DataFrame
            One row per time (index ``ugps``); vector fields expand to
            per-field-prefixed columns (e.g. ``scx, scy, scz``). Times outside
            SPICE coverage are NaN across that field's columns (see the module
            Fill contract).

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

    def get_vectors(self, ugps_times, fields) -> dict:
        """Compute the requested fields as typed arrays.

        The typed sibling of :meth:`get_geometry`, addressed by field name rather
        than by string-built column prefixes.

        Parameters
        ----------
        ugps_times : array_like of int
            One or more times in GPS microseconds.
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
