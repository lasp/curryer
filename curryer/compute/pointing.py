# pylint: disable=no-member
"""Pointing related calculations.

@author: Brandon Stone
"""

import logging
from enum import IntEnum
from types import MappingProxyType

import numpy as np
import pandas as pd
from spiceypy.utils.exceptions import SpiceyError

from .. import spicetime, spicierpy
from . import abstract, constants, spatial

logger = logging.getLogger(__name__)


def calc_cosine(target_state, boresight_vector):
    """Calculate the cosine angle between a boresight vector and one or more
    state vectors. Vectors will be normalized (inplace).

    Parameters
    ----------
    target_state : numpy.ndarray
        Array of state vectors [N, 3] or [3].
    boresight_vector : numpy.ndarray
        Boresight vector [3].

    Returns
    -------
    float or numpy.ndarray
        Cosine values. Return is a scalar if `target_state` was a 1D array.

    """
    if target_state.ndim == 1:
        target_state = np.expand_dims(target_state, axis=0)
        as_array = False
    else:
        as_array = True

    # Normalize the vectors.
    boresight_vector /= np.linalg.norm(boresight_vector, axis=0)
    ix = np.where(target_state.sum(1) != 0)  # TODO: FIX Not real, but if inside a planet, cosine would be zero...
    if len(ix[0]) > 0:
        target_state[ix] /= np.linalg.norm(target_state[ix], axis=1)[..., None]

    # Cosine angle between the boresight and target, in the instrument frame.
    instrument_dot_target = (boresight_vector @ target_state[..., None])[..., 0]
    return instrument_dot_target if as_array else instrument_dot_target[0]


# TODO: Update signature to be similar to other func (i.e., time first).
def boresight_dot_object(instrument, target, ugps_times):
    """Calculate the cosine between an instruments boresight and target.

    Parameters
    ----------
    instrument : str or sds_spice.spicierpy.obj.Instrument
        Instrument to calculate the cosine angles for. If an Instrument object,
        it must contain a "frame" and "spacecraft". Otherwise they are
        inferred from the instrument string.
    target : str or sds_spice.spicierpy.obj.Body
        Target ephemeris object (e.g., Earth, Sun).
    ugps_times : int or list of ints or numpy.ndarray
        One or more uGPS times to calculate values at.

    Returns
    -------
    float or numpy.ndarray
        Cosine angle between the boresight and target, in the `instrument` frame.

    Notes
    -----
    - The boresight is retrieve from the `instrument`'s kernel (IK).

    """
    # Infer the instrument's reference frame (defined in an FK).
    if not isinstance(instrument, spicierpy.obj.Body):
        instrument = spicierpy.obj.Body(instrument, frame=True)
    observer = instrument.spacecraft if isinstance(instrument, spicierpy.obj.Instrument) else instrument
    if instrument.frame is None:
        raise ValueError(f'Pre-defined Instrument objects must contain a "frame". Invalid: {instrument}')

    # Load ephemeris data for the target object, in the instrument's reference
    #   frame.
    target_state = spicierpy.ext.query_ephemeris(
        ugps_times=ugps_times,
        target=spicierpy.obj.Body(target).name,
        observer=observer,
        ref_frame=instrument.frame,
        allow_nans=False,
        velocity=False,
    ).values

    # Load the boresight vector from the instrument kernel.
    boresight_vector = spicierpy.ext.instrument_boresight(instrument)  # , norm=True)

    # Cosine angle between the boresight and target, in the instrument frame.
    return calc_cosine(target_state, boresight_vector)


def check_fov(ugps_times, instrument, target="MOON", observer=True, correction="LT+S", allow_nans=False):
    """Check if there was an FOV event at the specified time."""
    # Prepare the SPICE objects.
    if not isinstance(instrument, spicierpy.obj.Body):
        instrument = spicierpy.obj.Body(instrument, frame=True)
    if not isinstance(target, spicierpy.obj.Body):
        target = spicierpy.obj.Body(target, frame=True)
    observer = instrument.spacecraft if isinstance(instrument, spicierpy.obj.Instrument) else instrument
    name = f"Obs[{observer}]Instrument[{instrument.name}]Target[{target.name}]"

    # Check for FOV events at each time step!
    et_times = spicetime.adapt(ugps_times, to="et")

    # Function to handle insufficient data errors.
    # lacks_data = re.compile(r'^SPICE\((SPKINSUFFDATA|NOFRAMECONNECT)\)', re.MULTILINE)
    def _fov(sample_et):
        try:
            return spicierpy.fovtrg(
                et=sample_et,
                inst=instrument.name,
                target=target.name,
                tshape="ELLIPSOID",
                tframe=target.frame.name,
                abcorr=correction or "NONE",
                observer=observer.name,
            )
        except SpiceyError as e:
            if not allow_nans:
                raise
            if "SPICE(SPKINSUFFDATA)" in e.short:
                return -1  # Lacks ephemeris (generally).
            if "SPICE(NOFRAMECONNECT)" in e.short:
                return -2  # Lacks attitude (generally).
            if "SPICE(NOTDISJOINT)" in e.short:
                # Interp through an invalid ephemeris (generally a gap).
                return -3  # Viewpoint is inside target.
            raise e

            # if not lacks_data.search(e.value):
            #     raise
            # return -1

    logger.info("Checking for [%s] FOV events: %s", len(et_times), name)
    # vec_fovtrg = np.vectorize(_fov, otypes=[np.int8])
    # flags = vec_fovtrg(et_times)
    # flags = np.array(list(map(_fov, et_times)), dtype=np.int8)
    # flags = np.fromiter(map(_fov, et_times), dtype=np.int8, count=len(et_times))
    # return pd.Series(flags, index=ugps_times, name=name)
    results = pd.Series(map(_fov, et_times), index=ugps_times, name=name, dtype=np.int8)

    if len(ugps_times) > 0:
        cnts = results.value_counts().to_dict()
        logger.info(
            "Check FOV [%s] between [%s, %s] had nInFov=[%d], nOutFov=[%d],"
            " nMissEphem=[%d], nMissAtt=[%d], nDisjoint=[%d]",
            name,
            spicetime.adapt(ugps_times[0], to="utc"),
            spicetime.adapt(ugps_times[-1], to="utc"),
            cnts.get(1, 0),
            cnts.get(0, 0),
            cnts.get(-1, 0),
            cnts.get(-2, 0),
            cnts.get(-3, 0),
        )
    return results


class OccultType(IntEnum):
    """Types of planetary occultation.
    See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/occult_c.html
    """

    TOTAL_TARGET1_BY_TARGET2 = -3
    ANNULAR_TARGET1_BY_TARGET2 = -2
    PARTIAL_TARGET1_BY_TARGET2 = -1
    NO_OCCULT = 0
    PARTIAL_TARGET2_BY_TARGET1 = 1
    ANNULAR_TARGET2_BY_TARGET1 = 2
    TOTAL_TARGET2_BY_TARGET1 = 3


def check_occult(ugps_times, observer, target1="MOON", target2="SUN", correction="LT"):
    """Check if there was an FOV event at the specified time."""
    # Prepare the SPICE objects.
    if not isinstance(observer, spicierpy.obj.Body):
        observer = spicierpy.obj.Body(observer)
    if not isinstance(target1, spicierpy.obj.Body):
        target1 = spicierpy.obj.Body(target1, frame=True)
    if not isinstance(target2, spicierpy.obj.Body):
        target2 = spicierpy.obj.Body(target2, frame=True)
    name = f"Obs[{observer.name}]T1[{target1.name}]T2[{target2.name}]"

    # Check for occult events at each time step!
    et_times = spicetime.adapt(ugps_times, to="et")

    vec_occult = np.vectorize(
        spicierpy.occult,
        otypes=[np.int8],
        excluded={"target1", "shape1", "frame1", "target2", "shape2", "frame2", "abcorr", "observer"},
    )
    codes = vec_occult(
        et=et_times,
        target1=target1.name,
        shape1="ELLIPSOID",
        frame1=target1.frame.name,
        target2=target2.name,
        shape2="ELLIPSOID",
        frame2=target2.frame.name,
        abcorr=correction or "NONE",
        observer=observer.name,
    )
    return pd.Series(codes, index=ugps_times, name=name)


class PointingData(abstract.AbstractMissionData):
    """Class for working with TIM pointing data."""

    DEFAULT_CADENCE = constants.ATTITUDE_TIMESTEP_USEC
    DEFAULT_QF_MAP = MappingProxyType(
        dict(
            NOT_TRACKING_THE_SUN=0x10,
            EARTH_IN_FIELD_OF_VIEW=0x40,
            MOON_IN_FIELD_OF_VIEW=0x80,
            POINTING_GAP=0x200000,
            POINTING_OBSC_STATIC_STRUCT=0x8000,
            POINTING_OBSC_SOLAR_ARRAY=0x10000,
        )
    )

    def __init__(self, observer, qf_map=None, with_geolocate=False, microsecond_cadence=None):
        """Setup how the attitude data will be used.

        Parameters
        ----------
        observer : str or int or Body
            SPICE ID or name for the boresight instrument.
        qf_map : dict[str, int], optional
            Mapping of quality flag names to integer values.
        with_geolocate : bool
            Option to add Earth geolocation fields. Must match DB table!
        microsecond_cadence : int
            Cadence in microseconds to generate data at.

        """
        microsecond_cadence = self.DEFAULT_CADENCE if microsecond_cadence is None else microsecond_cadence
        super().__init__(microsecond_cadence=microsecond_cadence)

        # if not isinstance(instrument, spicierpy.obj.Instrument):
        #     instrument = spicierpy.obj.Instrument(instrument, frame=True, spacecraft=True)
        if not isinstance(observer, spicierpy.obj.Body):
            observer = spicierpy.obj.Body(observer, frame=True)
        self.observer = observer
        self.qf_map = qf_map if qf_map is not None else self.DEFAULT_QF_MAP
        self.with_geolocate = with_geolocate

        self.max_deltat = 0  # TODO: Make this easily configurable?

    @abstract.log_return()
    def get_pointing(self, ugps_times):
        """Determine pointing values.

        Parameters
        ----------
        ugps_times : int or iter of int
            Determine pointing at 1 or more times (uGPS).

        Returns
        -------
        pandas.DataFrame
            Table of [?] at each time.

        """
        logger.debug("Creating pointing table with up to [%i] rows", len(ugps_times))
        cosines = pd.DataFrame(
            index=pd.Index(ugps_times, name="microsecondssincegpsepoch"),
            columns=["timdotearth", "timdotmoon", "timdotsun"],
        )
        qualityflags = pd.Series(np.zeros(cosines.shape[0], dtype=np.uint32), index=cosines.index, name="qualityflags")

        # Search for FOV events.
        #   This includes every time, and will let us know when we are missing
        #   data, making the following calls faster than this.
        fov = check_fov(ugps_times, self.observer, target="MOON", allow_nans=self.allow_nans)

        # Times with complete SPICE data.  TODO: This doesn't work as intended. Still get excep lower!
        idata = fov >= 0
        ugps_good = fov[idata].index.values
        qualityflags[~idata] |= self.qf_map["POINTING_GAP"]

        # Return early if no kernel data exists.
        if ugps_good.size == 0:
            logger.warning(
                "[0/%i] times had compete ephemeris/attitude data; returning table with all cosine NaNs.",
                len(ugps_times),
            )
            return cosines.join(qualityflags)

        # Set Moon-in-FOV, but only if there is occultation of the sun.
        occult = check_occult(ugps_good, self.observer, "MOON", "SUN")
        fov = (fov > 0) & (occult > OccultType.NO_OCCULT.value)
        qualityflags[fov == 1] |= self.qf_map["MOON_IN_FIELD_OF_VIEW"]

        # Check Earth-in-FOV.
        fov = check_fov(ugps_good, self.observer, target="EARTH", allow_nans=self.allow_nans)
        qualityflags[idata & (fov == 1)] |= self.qf_map["EARTH_IN_FIELD_OF_VIEW"]

        # Check Sun-in-FOV (not used in production).
        # Note: Inverted QF b/c NOT tracking (i.e., not in FOV).
        fov = check_fov(ugps_good, self.observer, target="SUN", allow_nans=self.allow_nans)
        occult = check_occult(ugps_good, self.observer, "EARTH", "SUN")
        fov = (fov > 0) & (occult <= OccultType.NO_OCCULT.value)
        qualityflags[idata & (fov != 1)] |= self.qf_map["NOT_TRACKING_THE_SUN"]

        # Calculate cosines.
        cosines.loc[idata, "timdotearth"] = boresight_dot_object(self.observer, "EARTH", ugps_good)
        cosines.loc[idata, "timdotmoon"] = boresight_dot_object(self.observer, "MOON", ugps_good)
        cosines.loc[idata, "timdotsun"] = boresight_dot_object(self.observer, "SUN", ugps_good)
        cosines = cosines.astype(np.float64)

        # Option to geolocate.
        geoloc = None
        if self.with_geolocate:
            surf_xyz, sc_xyz, sqf = spatial.instrument_intersect_ellipsoid(ugps_times, self.observer)

            surf_norm = surf_xyz.values / np.linalg.norm(surf_xyz.values, axis=1)[..., None]
            vec_norm = sc_xyz.values - surf_xyz.values
            vec_norm /= np.linalg.norm(vec_norm, axis=1)[..., None]
            surf_dot = np.prod([surf_norm, vec_norm], axis=0).sum(axis=1)

            surf_lla = spatial.ecef_to_geodetic(surf_xyz.values, degrees=True)
            sc_alt = spatial.ecef_to_geodetic(sc_xyz.values)[:, 2]

            geoloc = pd.DataFrame(
                {
                    "surfacelon": np.nan_to_num(surf_lla[:, 0], nan=0),
                    "surfacelat": np.nan_to_num(surf_lla[:, 1], nan=0),
                    "surfacedot": np.nan_to_num(surf_dot, nan=-1),
                    "surfacealt": np.nan_to_num(sc_alt, nan=0),
                },
                index=pd.Index(ugps_times, name="microsecondssincegpsepoch"),
            )

            # Add QF for nadir pointing.
            # TODO: Use fovspec? The "corners" don't match expected value?!
            # fovspec = spicierpy.getfov(self.instrument.id, 1)
            # assert fovspec[0] == 'CIRCLE'
            if spicierpy.gcpool(f"INS{self.observer.frame.id}_FOV_SHAPE", 0, 1)[0] != "CIRCLE":
                raise ValueError(
                    f"Expected FOV_SHAPE to be CIRCLE, got {spicierpy.gcpool(f'INS{self.observer.frame.id}_FOV_SHAPE', 0, 1)[0]}"
                )
            if spicierpy.gcpool(f"INS{self.observer.frame.id}_FOV_ANGLE_UNITS", 0, 1)[0] != "DEGREES":
                raise ValueError(
                    f"Expected FOV_ANGLE_UNITS to be DEGREES, got {spicierpy.gcpool(f'INS{self.observer.frame.id}_FOV_ANGLE_UNITS', 0, 1)[0]}"
                )
            instr_fov_deg = spicierpy.gdpool(f"INS{self.observer.frame.id}_FOV_REF_ANGLE", 0, 1)[0]

            instr_fov_cos = np.cos(np.deg2rad(instr_fov_deg))  # Half-angle as a cosine.
            is_nadir = cosines["timdotearth"] >= instr_fov_cos
            qualityflags[is_nadir.index[is_nadir]] |= self.qf_map["POINTING_OBSC_STATIC_STRUCT"]  # TODO: New one?!!

            # Add QF for FOV entirely over Earth.
            minor_radii = constants.WGS84_SEMI_MINOR_AXIS_KM
            max_alt = 510  # TODO: Compute?
            horizon_ang = np.arcsin(minor_radii / (minor_radii + max_alt))
            threshold_cos = np.cos(horizon_ang - np.deg2rad(instr_fov_deg))
            is_full_earth = cosines["timdotearth"] >= threshold_cos
            qualityflags[is_full_earth.index[is_full_earth]] |= self.qf_map["POINTING_OBSC_SOLAR_ARRAY"]  # TODO: New?

        # Combine data.
        result = cosines.join(qualityflags)
        if geoloc is not None:
            result = result.join(geoloc)

        # Drop rows with a NaN in any column.
        sz_before = result.shape[0]
        result.dropna(inplace=True)
        sz_after = result.shape[0]
        if sz_before != sz_after:
            logger.info("Dropping [%d/%d] rows with NaNs", sz_before - sz_after, sz_before)
        return result

    # @abstract.log_return(max_rows=3)
    # def _get_static_table(self, ugps_times):
    #     """Generate a table of pointing metadata.
    #     """
    #     logger.debug('Creating static table with [%i] rows', len(ugps_times))
    #     table = pd.DataFrame(
    #         OrderedDict([('instrumentmodeid', self.instrumentmodeid),
    #                      ('version', self.version),
    #                      ('deltat', self.max_deltat)]),
    #         index=pd.Index(ugps_times, name='microsecondssincegpsepoch')
    #     )
    #     return table
    #
    # @abstract.log_return(max_rows=10)
    # def get_data(self, ugps_range):
    #     """Get the available TIM pointing data.
    #
    #     Parameters
    #     ----------
    #     ugps_range : two int
    #         uGPS start and stop values. Exclusive end. Default time step.
    #
    #     Returns
    #     -------
    #     pandas.DataFrame
    #         Table of "TimPointingData" values:
    #             timdotearth, timdotmoon, timdotsun : float64
    #             qualityflags : int32
    #             instrumentmodeid, version, deltat : int64
    #
    #     """
    #     logger.info('Getting available point data in range [%s]', ugps_range)
    #     ugps_times = self._get_times(ugps_range)
    #     table = self._get_pointing_table(ugps_times).dropna()
    #     idx_name = table.index.name
    #     table = table.join(self._get_static_table(ugps_times), how='inner')
    #     table.index.name = idx_name
    #     return table


# ----------------------------------------------------------------------------
# DEV NOTE: The following function is used to prove the _theory_ of the
#   pointing calculations (cosines), but it is not needed when using SPICE
#   kernels with frame definitions.
# ----------------------------------------------------------------------------
def legacy_instrument_dot_object(
    spacecraft_to_eci_rotation,
    spacecraft_in_eci_state,
    object_in_eci_state,
    boresight_in_instrument_vector,
    spacecraft_to_instrument_rotation=None,
):
    """Calculate the cosine angle between an instrument's boresight and an
    object (planet).

    Parameters
    ----------
    spacecraft_to_eci_rotation : numpy.ndarray
        Rotation matrix (3x3 or Nx3x3) from the spacecraft's reference frame to
        the ECI (earth centered inertial) reference frame.
    spacecraft_in_eci_state : numpy.ndarray
        State vector (3 or Nx3; x,y,z) of the spacecraft's location in the ECI
        reference frame.
    object_in_eci_state : numpy.ndarray
        State vector (3 or Nx3; x,y,z) of the object's location in the ECI
        reference frame. The object is typically a planet. For example, the
        vector would be all zeros if the object was Earth.
    boresight_in_instrument_vector : numpy.ndarray
        Pointing vector (3; x,y,z) of the instrument's boresight, in the
        instrument's reference frame (e.g., [1, 0, 0]).
    spacecraft_to_instrument_rotation : numpy.ndarray, optional
        Rotation matrix (3x3) from the spacecraft's reference frame to the
        instrument's reference frame. Default assumes they are the same.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Cosine angle based on the normalized dot product of the instrument
        boresight and the object's state vector(s). The return is a scalar
        numpy float if `spacecraft_to_eci_rotation` was a 2D array, otherwise
        it's a 1D array of floats.

    Notes
    -----
    - ECI and J2000 are considered the same spatial reference frame.
    - Based on the java implementation in tim_processing:
        src/java/processing/tim/pointing_data/TCTEPointingDataServer.java
    - Java implementation difference - A "Quat4d" & "Vector3d" transformation
    includes a matrix transposition; likely a quaternion format difference.

    """
    # Ensure each "sample" has a 2D rotation matrix and two 1D state vectors.
    #   Increase the dimensionality of single "sample" calculations. Remember
    #   whether we got a scalar-like data or array-like data; everything should
    #   be an array, but multi-sample inputs will be a higher order array.
    # pylint: disable=misplaced-comparison-constant
    if 2 == spacecraft_to_eci_rotation.ndim and 1 == spacecraft_in_eci_state.ndim == object_in_eci_state.ndim:
        spacecraft_to_eci_rotation = spacecraft_to_eci_rotation[None, ...]  # 2D -> 3D
        spacecraft_in_eci_state = spacecraft_in_eci_state[None, ...]  # 1D -> 2D
        object_in_eci_state = object_in_eci_state[None, ...]  # 1D -> 2D
        as_scalar = True
    elif 3 == spacecraft_to_eci_rotation.ndim and 2 == spacecraft_in_eci_state.ndim == object_in_eci_state.ndim:
        as_scalar = False
    else:
        raise ValueError(
            "Invalid dimensions for arguments. Rotation array must be 2 or 3 dim, and state arrays must have one less"
            f" (i.e., 1 or 2). Got ndim: {spacecraft_to_eci_rotation.ndim}, {spacecraft_in_eci_state.ndim}, {object_in_eci_state.ndim}"
        )

    # Define the rotation from the instrument (boresight) reference frame to
    #   the spacecraft reference frame. Default assumes they are equal.
    if spacecraft_to_instrument_rotation is None:
        instrument_to_spacecraft_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    elif 2 == spacecraft_to_instrument_rotation.ndim:
        instrument_to_spacecraft_rotation = spacecraft_to_instrument_rotation.T
    else:
        raise ValueError("The spacecraft to instrument rotation must be fixed (i.e., a 2D rotation matrix).")

    # Define the rotation from the ECI (earth centered inertial) reference
    #   frame to the spacecraft's reference frame. To support multi-sample
    #   calculations, don't transpose the the first dimension.
    eci_to_spacecraft_rotation = spacecraft_to_eci_rotation.transpose((0, 2, 1))

    # Rotate the instrument's boresight to the spacecraft's reference frame
    #   and normalize its magnitude.
    boresight_in_spacecraft_vector = instrument_to_spacecraft_rotation @ boresight_in_instrument_vector
    boresight_in_spacecraft_vector /= np.linalg.norm(boresight_in_spacecraft_vector, axis=0)

    # Translate and rotate the "object" to determine its normalized state in
    #   the spacecraft's reference frame.
    object_in_spacecraft_state = object_in_eci_state - spacecraft_in_eci_state
    object_in_spacecraft_state = (eci_to_spacecraft_rotation @ object_in_spacecraft_state[..., None])[..., 0]
    object_in_spacecraft_state /= np.linalg.norm(object_in_spacecraft_state, axis=1)[..., None]

    # The cosine angle between the boresight and the "object" is the dot
    #   product of the normalized instrument vector and object state-vector(s).
    instrument_to_object_cosine = (boresight_in_spacecraft_vector @ object_in_spacecraft_state[..., None])[..., 0]
    return instrument_to_object_cosine[0] if as_scalar else instrument_to_object_cosine
