"""Native (fast) time conversions.

@author: Brandon Stone
"""

import logging

import numpy as np

from . import constants, leapsecond
from .utils import InputAsArray

logger = logging.getLogger(__name__)


@InputAsArray(np.int64)
def ugps2datetime(times, ceil_leapseconds=True):
    """Convert GPS microseconds to datetime64[us].

    Fast conversion; very convenient for plotting.

    Parameters
    ----------
    times : int or list[int] or numpy.ndarray or pandas.Series
        One or more uGPS times to convert.
    ceil_leapseconds : bool, optional
        Ceil times within a leapsecond. Datetime64 does not support times within
        leapseconds. Default=True, otherwise throws a ValueError if encountered.

    Returns
    -------
    numpy.datetime64 or numpy.ndarray
        Converted times - output is a scalar if the input was, otherwise it's
        an array.

    """
    leapsecs = leapsecond.cache.get()

    # Convert uGPS to microseconds since linux epoch.
    idx = (times >= leapsecs["ugps"].values[..., None]).sum(axis=0) - 1
    usec = times + leapsecs["offset"].iloc[idx].values

    # Catch edges cases (mid-leapsecond).
    idx2 = times < leapsecs["ugps"][-1]
    if idx2.any():
        diff = (leapsecs["ugps"][idx[idx2] + 1] - times[idx2]).values
        idx3 = diff < 1000000
        if idx3.any():
            # Option to error since dt64 doesn't support leapseconds.
            if not ceil_leapseconds:
                raise ValueError(
                    "Datetime64 does not support times within leapseconds! Set `ceil_leapseconds=True`"
                    " to round up those times and prevent this error (default)."
                    f" Found [{idx3.sum()}] times: [{times[idx2 & idx3]}]"
                )

            # Otherwise round times within leapseconds.
            usec[idx2 & idx3] -= 1000000 - diff[idx3] % 1000000

    # Linux microseconds to datetime64, the former is the underlying data.
    return np.array(usec, dtype="M8[us]")


@InputAsArray(np.dtype("M8[us]"))
def datetime2ugps(times):
    """Convert datetime64[us] to GPS microseconds.

    Parameters
    ----------
    times : numpy.datetime64 or numpy.ndarray
        Scalar or array of datetime64 values with microsecond precision.

    Returns
    -------
    int or numpy.ndarray
        Converted uGPS times - output is a scalar if the input was, otherwise
        it's an array.

    """
    leapsecs = leapsecond.cache.get()

    # Convert from linux epoch to GPS epoch.
    usec = times.astype(int)
    idx = (usec >= leapsecs["unix"].values[..., None]).sum(axis=0) - 1
    return usec - leapsecs["offset"].iloc[idx].values


@InputAsArray(np.int64)
def ugps_to_gps_fraction(times):
    """Convert uGPS to GPS fractional days."""
    leapsecs = leapsecond.cache.get()

    # Convert uGPS to microseconds since _GPS_ epoch.
    idx = (times >= leapsecs["ugps"].values[..., None]).sum(axis=0) - 1
    # leap_micros_since_gps = (leapsecs['nsec'].iloc[idx].values - GPS_EPOCH_LEAPSEC) * SEC_TO_MICROSECOND
    leap_micros_since_gps = leapsecs["nsec"].iloc[idx].values - constants.EpochOffsetSeconds.GPS_LEAPSECONDS
    leap_micros_since_gps *= constants.TimeConstant.SEC_TO_USEC
    usec = times - leap_micros_since_gps

    # Catch edges cases (mid-leapsecond).
    idx2 = times < leapsecs["ugps"][-1]
    if idx2.any():
        diff = (leapsecs["ugps"][idx[idx2] + 1] - times[idx2]).values
        idx3 = diff < constants.TimeConstant.SEC_TO_USEC
        if idx3.any():
            usec[idx2 & idx3] -= constants.TimeConstant.SEC_TO_USEC - diff[idx3] % constants.TimeConstant.SEC_TO_USEC

    # Convert to fractional days.
    gps_days = usec / constants.TimeConstant.DAY_TO_USEC
    return gps_days


@InputAsArray(np.float64)
def gps_fraction_to_ugps(times):
    """Convert GPS fractional days to uGPS."""
    leapsecs = leapsecond.cache.get()

    # Convert from GPS fractional days epoch to GPS microseconds.
    usec = times * constants.TimeConstant.DAY_TO_USEC
    usec = np.round(usec).astype(np.int64)
    leap_micros_since_gps = leapsecs["nsec"].values - constants.EpochOffsetSeconds.GPS_LEAPSECONDS
    leap_micros_since_gps *= constants.TimeConstant.SEC_TO_USEC
    idx = (usec >= (leapsecs["ugps"] - leap_micros_since_gps).values[..., None]).sum(axis=0) - 1
    ugps = usec + leap_micros_since_gps[idx]
    return ugps


@InputAsArray(np.float64)
def gps_fraction_to_epoch_fraction(times, to_epoch):
    """Convert GPS fractional days to a different epoch."""
    if not isinstance(to_epoch, constants.EpochGpsDays):
        to_epoch = constants.EpochGpsDays[to_epoch.upper()]
    epoch_days = times - to_epoch.value
    return epoch_days


@InputAsArray(np.float64)
def epoch_fraction_to_gps_fraction(times, from_epoch):
    """Convert epoch fractional days to the GPS epoch."""
    if not isinstance(from_epoch, constants.EpochGpsDays):
        from_epoch = constants.EpochGpsDays[from_epoch.upper()]
    gps_days = times + from_epoch.value
    return gps_days
