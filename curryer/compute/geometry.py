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
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import spicierpy
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


# The boresight provider reuses the SPICE-geometry primitive in ``spatial``
# (which resolves the instrument frame from the body and handles the data-gap ->
# NaN mapping). Unlike the ephemeris providers it loops internally, since
# ``pxform`` has no vectorized override; the result is still one query per request.
def _provider_boresight(ugps_times, ctx):
    """Instrument boresight unit vector in the configured Earth-fixed frame
    (``ctx.earth_frame``, ``ITRF93`` by default), shape (N, 3)."""
    hs_boresight = spicierpy.ext.instrument_boresight(ctx.observer)
    pointing, _, _ = spatial.instrument_pointing_state(
        ugps_times,
        ctx.observer,
        boresight_vector=hs_boresight,
        pointing_frame=ctx.earth_frame,
        allow_nans=ctx.allow_nans,
    )
    return pointing[["x", "y", "z"]].values


_PROVIDERS = {
    "sc_position": _provider_sc_position,
    "sun_position": _provider_sun_position,
    "boresight": _provider_boresight,
}


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
        # the ellipsoid -- both already queried for other fields, so the ray-cast
        # is a math-only leaf and adds no SPICE. ``ray_intersect_ellipsoid``
        # returns geodetic [lon, lat, alt]; column 1 is the latitude.
        providers=frozenset({"boresight", "sc_position"}),
        columns=("surfcolat",),
        evaluate=lambda p: colatitude(
            spatial.ray_intersect_ellipsoid(p["boresight"], p["sc_position"], geodetic=True, degrees=True)[:, 1]
        )[:, None],
    ),
}


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
        """Validate the requested fields, defaulting to all registered."""
        if fields is None:
            return list(_FIELDS)
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
            Field names to compute. Default is all registered fields.

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
