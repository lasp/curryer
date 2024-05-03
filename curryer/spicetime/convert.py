# pylint: disable=attribute-defined-outside-init, protected-access
"""Convert between time formats (ttype).

Routine Listings
----------------
adapt
    Convert from one time format to another. Maintains container type (e.g.,
    scalar input returns a scalar, list input returns a list).
SpiceTime
    Subclass of numpy's ndarray that stores the data's time format `ttype`, and
    facilitates converting to other time formats.

Examples
--------
>>> utc = adapt([0, 1], from_='ugps', to='utc')
>>> print('Adapt API:\n\t{!r}\n\ttype={}'.format(utc, type(utc)))
Adapt API:
	['1980-01-06 00:00:00.000000', '1980-01-06 00:00:00.000001']
	type=<class 'list'>


>>> ugps = SpiceTime([0, 1], ttype='ugps')
>>> utc_arr = ugps.to_utc()
>>> print('SpiceTime API:\n\t{!r}'.format(utc_arr))
SpiceTime API:
	SpiceTime(['1980-01-06 00:00:00.000000', '1980-01-06 00:00:00.000001'],
      dtype='<U26', ttype='utc')

@author: Brandon Stone
"""
import logging
import re
from types import MappingProxyType

import numpy as np

from .. import spicierpy as sp
from . import constants, utils, native


# TODO: Support format type of a SPICE clock obj (i.e., convert to/from sclk ticks).
#   See methods `sce2c` and `sct2e`. Limited usage and requires kernels!
def adapt(dt_val, from_=None, to=None, **kwargs):
    """Convert between different date time formats.

    Time Formats
    ------------
    et : float64, seconds
        Ephemeris time, seconds since J2000 in ET. Also called Barycentric
        Dynamical Time (TDB) in SPICE documentation. Most times are converted
        to and from ET.
    ugps : int64, microseconds
        GPS microseconds since 1980-01-06T00:00:00.0 UTC. Supports native (no
        SPICE) conversion to/from `dt64` and `epoch` et al.
    gps : float64, seconds
        GPS seconds since 1980-01-06T00:00:00.0 UTC.
    tai : float64, seconds
        International Atomic Time, seconds since 1958-01-01T00:00:00.0 UTC.
        NOTE: Within the SPICE system "TAI" often represents the J2000 epoch,
        we deviate from that to support the standard 1958 epoch.
    utc : str, ISO
        Coordinated Universal Time (UTC), including leapseconds. The default
        input and output format is ISO. The output format can be specified
        with a Python-style datetime format string and the keyword
        `date_format` (e.g.,`date_format='%Y-%d-%m'`). Some format keys are not
        supported.
    iso : str, ISO
        Alias for utc.
    dt64 : np.datetime64, ns
        Numpy datetime64 type with nanosecond precision. Does *not* support
        leapseconds. Default is to round those times up to the nearest second.
        Set `ceil_leapseconds=False` to raise errors instead.
    epoch : float64, days
        Fractional days since an epoch without leapseconds. The epoch name is
        defined with `from_epoch` or `to_epoch`, depending on the direction.
        See `sds_spice.spicetime.constants.EpochGpsDays`. Typically results in
        a loss of microsecond precision. Leapseconds are accounted for when
        determining the day, but the fraction for days with leapseconds assumes
        8.64e4 seconds per day (no leapsecond). Times within leapseconds are
        rounded up.
    sorce : float64, days
        Fractional days since 2003-01-24. See `epoch` for details.
    tcte : float64, days
        Fractional days since 2013-11-18. See `epoch` for details.
    tsis : float64, days
        Fractional days since 2017-12-14. See `epoch` for details.
    tsis1 : float64, days
        Alias for `tsis`. See `tsis` for details.
    tsis2 : float64, days
        Fractional days since 2017-12-14 (placeholder; launch TBD).
        See `epoch` for details.
    ctim : float64, days
        Fractional days since 2022-07-02. See `epoch` for details.
    gpsd : float64, days
        Fractional days since 1980-01-06. See `epoch` for details.
    mjd : float64, days
        Fractional days since the Modified Julian epoch (1858-11-17). See
        `epoch` for details.
    jd : float64, days
        Fractional days since the Julian epoch. See `epoch` for details.
        Least precision of all time formats.

    Parameters
    ----------
    dt_val : scalar or list or tuple or numpy.ndarray of str or int or float
        One or more time values to convert from one time format to another.
    from_ : str, optional
        Time format (ttype) of the input data `dt_val`. Default='ugps'.
        Note: Either one or both of `from_` or `to` must be specified.
    to : str, optional
        Time format (ttype) to convert the input data `dt_val`, to.
        Default='ugps'. Note: Either one or both of `from_` or `to` must be
        specified.
    **kwargs
        Passes extra keywords that are not None to the conversion functions
        that support them (e.g., "date_format" when `to="utc"`).

    Returns
    -------
    scalar or list or tuple or numpy.ndarray of str or int or float
        The converted time(s). Each time value has the data type of output
        time format specified by `to` (e.g., str for "utc"). When converting
        multiple times, the output container matches the input `dt_val`.

    """
    # Developer note: Pylint gets confused here and thinks that any "ndarry"s
    #   are tuples, and then warns about missing methods and bad casts.
    # pylint: disable=no-member

    # Default to uGPS. Matches datetime_adapt, but note that spice uses ET
    # internally, and thus these conversions do as well.
    if from_ is to is None:
        raise ValueError('Keywords either `from_` or `to` must not be None.')

    from_ = 'ugps' if from_ is None else from_.lower()
    to = 'ugps' if to is None else to.lower()

    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(5):  # Trace-ish level.
        logger.log(5, 'Time conversion [%s] -> [%s] for [%d] values of type [%s]', from_, to,
                   1 if np.isscalar(dt_val) else len(dt_val), type(dt_val))

    # Determine how to convert between the formats and then apply it.
    out_val = utils.apply_conversions(utils.find_mapping_path(conversions, from_, to), dt_val, **kwargs)

    # Make output match input if it was a list or tuple.
    if isinstance(out_val, np.ndarray) and isinstance(dt_val, (list, tuple)):
        out_val = type(dt_val)(out_val.tolist())

    return out_val


#
# Convert from a time format to ET.
#
# Developer note: Within the SPICE system, "TAI" often represents the J2000
#   epoch, we deviate from that to support the standard 1958 epoch.
#
@utils.InputAsArray(np.float64)
def from_et(dt_val):
    """Convert times from Ephemeris Time (SPICE; float64).
    """
    return dt_val


@utils.InputAsArray(np.int64)
def from_ugps(dt_val):
    """Convert times from GPS microseconds.
    """
    return sp.unitim(dt_val / constants.TimeConstant.SEC_TO_USEC
                     - constants.EpochOffsetSeconds.GPS_TO_J2000ET, 'TAI', 'ET')


@utils.InputAsArray(np.float64)
def from_gps(dt_val):
    """Convert times from GPS seconds.
    """
    return sp.unitim(dt_val - constants.EpochOffsetSeconds.GPS_TO_J2000ET, 'TAI', 'ET')


@utils.InputAsArray(np.float64)
def from_tai(dt_val):
    """Convert times from International Atomic Time seconds (1958).
    """
    return sp.unitim(dt_val + constants.EpochOffsetSeconds.GPS_TO_TAI
                     - constants.EpochOffsetSeconds.GPS_TO_J2000ET, 'TAI', 'ET')


@utils.InputAsArray(np.str_)
def from_utc(dt_val):
    """Convert times from UTC string(s) (ISO format).
    """
    return sp.str2et(dt_val)


#
# Convert to a time format, from ET.
#
@utils.InputAsArray(np.float64)
def to_et(dt_val):
    """Convert times to Ephemeris Time (SPICE; float64).
    """
    return dt_val


@utils.InputAsArray(np.float64)
def to_ugps(dt_val):
    """Convert times to GPS microseconds (int64).
    """
    ugps = ((sp.unitim(dt_val, 'ET', 'TAI') + constants.EpochOffsetSeconds.GPS_TO_J2000ET)
            * constants.TimeConstant.SEC_TO_USEC)
    return np.round(ugps).astype(np.int64)


@utils.InputAsArray(np.float64)
def to_gps(dt_val):
    """Convert times to GPS seconds (float64).
    """
    return sp.unitim(dt_val, 'ET', 'TAI') + constants.EpochOffsetSeconds.GPS_TO_J2000ET


@utils.InputAsArray(np.float64)
def to_tai(dt_val):
    """Convert times to International Atomic Time seconds (1958; float64).
    """
    return (sp.unitim(dt_val, 'ET', 'TAI') + constants.EpochOffsetSeconds.GPS_TO_J2000ET
            - constants.EpochOffsetSeconds.GPS_TO_TAI)


@utils.InputAsArray(np.float64)
def to_utc(dt_val, date_format=None):
    """Convert times to UTC strings (ISO format).
    """
    # SPICE compatible format string, default to ISO.
    if date_format is None:
        date_format = 'YYYY-MM-DD HR:MN:SC.###### ::UTC ::RND'
    else:
        date_format = spice_strftime(date_format)

    # SPICE needs to know string length (plus 1).
    #   Ignore the "meta" commands.
    nchar = len(date_format.split('::', 1)[0].rstrip()) + 1

    return sp.timout(dt_val, date_format, nchar)


#
# Map every conversion to itself (mostly) and at least one other!
# The shortest conversion path is used, or if multiple are tied, the order they
# are defined in is used.
#
conversions = MappingProxyType(dict(
    et=dict(
        et=utils.noop_float64,
        ugps=to_ugps,
        gps=to_gps,
        tai=to_tai,
        utc=to_utc,
        iso=to_utc,
    ),
    ugps=dict(
        ugps=utils.noop_int64,
        et=from_ugps,
        dt64=native.ugps2datetime,
        gpsd=native.ugps_to_gps_fraction,
    ),
    gps=dict(
        gps=utils.noop_float64,
        et=from_gps,
    ),
    tai=dict(
        tai=utils.noop_float64,
        et=from_tai,
    ),
    # Note: ISO string formats only map to ET in order to force correct
    #   datetime string formatting. Mapping to self/alias breaks that.
    #   e.g. "2018-01-29" with utc -> utc and output date_format="%Y/%j"
    utc=dict(
        et=from_utc,
    ),
    iso=dict(
        et=from_utc,
    ),
    dt64=dict(
        dt64=utils.noop_dt64,
        ugps=native.datetime2ugps,
    ),
    gpsd=dict(
        gpsd=utils.noop_float64,
        ugps=native.gps_fraction_to_ugps,
        epoch=native.gps_fraction_to_epoch_fraction,
        ctim=utils.InputAsArray(np.float64, False, defaults=dict(to_epoch='CTIM'))(
            native.gps_fraction_to_epoch_fraction),
        tsis2=utils.InputAsArray(np.float64, False, defaults=dict(to_epoch='TSIS2'))(
            native.gps_fraction_to_epoch_fraction),
        tsis=utils.InputAsArray(np.float64, False, defaults=dict(to_epoch='TSIS'))(
            native.gps_fraction_to_epoch_fraction),
        tcte=utils.InputAsArray(np.float64, False, defaults=dict(to_epoch='TCTE'))(
            native.gps_fraction_to_epoch_fraction),
        sorce=utils.InputAsArray(np.float64, False, defaults=dict(to_epoch='SORCE'))(
            native.gps_fraction_to_epoch_fraction),
        jd=utils.InputAsArray(np.float64, False, defaults=dict(to_epoch='JD'))(
            native.gps_fraction_to_epoch_fraction),
        mjd=utils.InputAsArray(np.float64, False, defaults=dict(to_epoch='MJD'))(
            native.gps_fraction_to_epoch_fraction),
    ),
    epoch=dict(
        epoch=utils.noop_float64,
        gpsd=native.epoch_fraction_to_gps_fraction,
    ),
    ctim=dict(
        ctim=utils.noop_float64,
        gpsd=utils.InputAsArray(np.float64, False, defaults=dict(from_epoch='CTIM'))(
            native.epoch_fraction_to_gps_fraction),
    ),
    tsis2=dict(
        tsis2=utils.noop_float64,
        gpsd=utils.InputAsArray(np.float64, False, defaults=dict(from_epoch='TSIS2'))(
            native.epoch_fraction_to_gps_fraction),
    ),
    tsis1=dict(
        tsis1=utils.noop_float64,
        tsis=utils.noop_float64,
    ),
    tsis=dict(
        tsis=utils.noop_float64,
        tsis1=utils.noop_float64,
        gpsd=utils.InputAsArray(np.float64, False, defaults=dict(from_epoch='TSIS'))(
            native.epoch_fraction_to_gps_fraction),
    ),
    tcte=dict(
        tcte=utils.noop_float64,
        gpsd=utils.InputAsArray(np.float64, False, defaults=dict(from_epoch='TCTE'))(
            native.epoch_fraction_to_gps_fraction),
    ),
    sorce=dict(
        sorce=utils.noop_float64,
        gpsd=utils.InputAsArray(np.float64, False, defaults=dict(from_epoch='SORCE'))(
            native.epoch_fraction_to_gps_fraction),
    ),
    jd=dict(
        jd=utils.noop_float64,
        gpsd=utils.InputAsArray(np.float64, False, defaults=dict(from_epoch='JD'))(
            native.epoch_fraction_to_gps_fraction),
    ),
    mjd=dict(
        mjd=utils.noop_float64,
        gpsd=utils.InputAsArray(np.float64, False, defaults=dict(from_epoch='MJD'))(
            native.epoch_fraction_to_gps_fraction),
    ),
))

#
# Class to maintain `ttype` for arrays.
# Pylint has no clue how to inspect ndarray.
# pylint: disable=too-many-function-args,arguments-differ,no-member
#
TTYPE_TO_DTYPE = MappingProxyType({
    'et': np.float64,
    'ugps': np.int64,
    'gps': np.float64,
    'tai': np.float64,
    'utc': np.str_,
})


# TODO: Deprecate!
class SpiceTime(np.ndarray):
    """SPICE Date and Time Conversions.

    Time Formats
    ------------
    et : float64, seconds
        Ephemeris time, seconds since J2000 in ET. Also called Barycentric
        Dynamical Time (TDB) in SPICE documentation. All times are converted
        to and from ET.
    ugps : int64, microseconds
        GPS microseconds since 1980-01-06T00:00:00.0 UTC.
    gps : float64, seconds
        GPS seconds since 1980-01-06T00:00:00.0 UTC.
    tai : float64, seconds
        International Atomic Time, seconds since 1958-01-01T00:00:00.0 UTC.
        NOTE: Within the SPICE system "TAI" often represents the J2000 epoch,
        we deviate from that to support the standard 1958 epoch.
    utc : str, ISO
        Coordinated Universal Time (UTC), including leapseconds. The default
        input and output format is ISO. The output format can be specified
        with a Python-style datetime format string and the keyword
        `date_format` (e.g.,`date_format='%Y-%d-%m'`). Some format keys are not
        supported.

    Attributes
    ----------
    ttype : str
        Date time format (e.g., "ugps"). Read-only, unless None.

    Parameters
    ----------
    in_arr : float
        Scalar or array of times.
    ttype : str
        Supported format string specifying the input time format.

    Examples
    --------
    >>> st = SpiceTime('2016-05-09 14:49:17.371099', 'utc')
    >>> print(repr(st))
    SpiceTime('2016-05-09 14:49:17.371099',
          dtype='<U26', ttype='utc')

    >>> print('Ephemeris Time: {}'.format(st.to_et()))
    Ephemeris Time: 516077425.55644935

    >>> print('uGPS: {}'.format(st.to_ugps()))
    uGPS: 1146840574371099

    >>> gps = st.to_gps()
    >>> print(gps.ttype)
    gps

    >>> st = SpiceTime([0, 1], 'gps')
    >>> print(st, st.ttype)
    [ 0.  1.] gps

    >>> print(st.to_utc('%S.%f'))
    ['00.000000' '01.000000']


    Notes
    -----
    - Requires a leapsecond kernel to be loaded (automated by library).

    """

    def __new__(cls, in_arr, ttype=None):
        """Setup the new numpy.ndarray object as a float64 with a time format
        string attribute.
        """
        dtype = TTYPE_TO_DTYPE.get(ttype, np.float64)
        obj = np.asarray(in_arr, dtype=dtype).view(cls)

        # If ttype is None, it can later be set, but only once.
        #   Conversions cannot be done until it is set.
        obj._ttype = ttype
        return obj

    def __array_finalize__(self, obj):
        """Numpy's replacement for "__init__", supports construction, casting,
        and templates.

        Notes
        -----
        Three possible ways to create this subclass:
            1) Explicit construction:   arr = SpiceTime(np.arange(3), fmt='et')
                type(`obj`) is ndarray; `self` is built in `SpiceTime.__new__`
            2) View casting:            arr = np.arange(3).view(SpiceTime)
                type(`obj`) can be `SpiceTime` or another
            3) New-from-template:       arr = np.SpiceTime(arange(3))[1:]
                type(`obj`) is `SpiceTime`

        The optional method `__array_prepare__` is similar, but before the
        values are calculated.

        References
        ----------
        - https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

        """
        if obj is None:
            # Using `asarray(...).view(cls)` in `__new__`, makes this unlikely
            #   to be reached since `obj` is what the view is taken from.
            #   Example: `np.ndarray.__new__(SpiceTime, in_arr)` would create
            #   a new instance of SpiceTime (self), but without using a view.
            #   If it is reached, `ttype` can't get set, so error out!
            raise TypeError('New arrays created using explicit creation must use the `SpiceTime` class.')

        # Set the `ttype`. Could be None if casting from a non-SpiceTime obj.
        self._ttype = getattr(obj, 'ttype', None)

    def __array_prepare__(self, out_arr, context=None):
        """Numpy's way of supporting subclassing from ufuncs (e.g.,
        `np.min(et_arr)`). Allows viewing (ONLY) the output data structure and
        context (func & args) before the calculation.
        """
        # Prevent mixing with datetime64.
        if out_arr.dtype.type == np.datetime64:
            raise TypeError('Can not combine SpiceTime and numpy.datetime64 arrays. '
                            'Datetime64 does not support leapseconds!')
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        """Numpy's way of supporting subclassing from ufuncs (e.g.,
        `np.min(et_arr)`). Allows tweaking the output (`out_arr`), before final
        casting as `SpiceTime`.
        """
        # Cast timedelta output as the ttype's data type (e.g., int64 for ugps)
        #   but only if they use the same time units.
        if out_arr.dtype.type == np.timedelta64:
            allowed_timedelta64_units = {
                'us': ['ugps'],
                's': ['gps', 'tai'],
            }
            search_units = re.search(r'\w+\[(\w{1,2})\]', out_arr.dtype.str)
            units = None if search_units is None else search_units.group(1)

            if out_arr.ttype not in allowed_timedelta64_units.get(units, []):
                raise TypeError('Cannot combine ttype {!r} with timedelta units {!r} ({})'.format(
                    self.ttype, units, out_arr.dtype.str
                ))
            out_arr = out_arr.astype(TTYPE_TO_DTYPE[out_arr.ttype])

        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __repr__(self):
        """String representation of the array with the time format `ttype`.
        """
        s = np.ndarray.__repr__(self)
        return s[:-1] + ', ttype={!r})'.format(self.ttype)

    @property
    def ttype(self):
        """Time format.
        """
        return self._ttype

    @ttype.setter
    def ttype(self, value):
        """Set the time format, but only if undefined.
        """
        if self._ttype is None:
            self._ttype = value
        else:
            raise AttributeError('Cannot change the time format `ttype` once defined. '
                                 'Use the conversion functions to convert between formats '
                                 '(e.g., `self.adapt("utc")`')

    def adapt(self, to, **kwargs):
        """Convert from the current time format to another.
        """
        if self.ttype is None:
            raise AttributeError('Time format `ttype` cannot be None. It should be set at creation '
                                 '(e.g., `st = SpiceTime([0, 1], "ugps")`, or after (e.g., `st.ttype = "ugps").')
        dt_out = adapt(self, from_=self.ttype, to=to, **kwargs)

        # Maintain the SpiceTime class (lost in SPICE/C or dtype casting).
        #   Update the ttype based on the `to` conversion.
        if not hasattr(dt_out, 'ttype'):
            return self.__class__(dt_out, ttype=to)

        dt_out._ttype = to
        return dt_out

    def to_et(self):
        """Convert times to Ephemeris Time (SPICE; float64).
        """
        return self.adapt('et')

    def to_ugps(self):
        """Convert times to GPS microseconds (int64).
        """
        return self.adapt('ugps')

    def to_gps(self):
        """Convert times to GPS seconds (float64).
        """
        return self.adapt('gps')

    def to_tai(self):
        """Convert times to International Atomic Time seconds (1958; float64).
        """
        return self.adapt('tai')

    def to_utc(self, date_format=None):
        """Convert times to UTC strings (ISO format).
        """
        return self.adapt('utc', date_format=date_format)


# Convert Python date time format style to SPICE's alternative.
_py_to_spice_format = {
    '%a': 'Wkd',  # Weekday, abbrev
    '%A': 'Weekday',  # Weekday, full name
    '%w': None,  # Weekday, decimal number (0-6)
    '%d': 'DD',  # Day of month, zero-padded
    '%b': 'Mon',  # Month, abbrev
    '%B': 'Month',  # Month, full name
    '%m': 'MM',  # Month, zero-padded
    '%y': 'YR',  # Year w/o century, zero-padded
    '%Y': 'YYYY',  # Year, zero-padded
    '%H': 'HR',  # Hour (24), zero-padded
    '%I': 'AP',  # Hour (12), zero-padded
    '%p': 'AMPM',  # AM / PM (NOTE: Adds periods)
    '%M': 'MN',  # Minute, zero-padded
    '%S': 'SC',  # Second, zero-padded
    '%f': '.######',  # Microsecond, zero-padded (6 places)
    '%z': None,  # UTC offset (e.g., +0000)
    '%Z': None,  # Time zone name (e.g., MST)
    '%j': 'DOY',  # Day of year, zero-padded
    '%U': None,  # Week number (Sunday), zero-padded
    '%W': None,  # Week number (Monday), zero-padded
    '%c': None,  # Locale's date and time representation
    '%x': None,  # Locale's date representation
    '%X': None  # Locale's time representation
}
PY_TO_SPICE_FORMAT = MappingProxyType(_py_to_spice_format)


def spice_strftime(date_format):
    """Convert a Python datetime format string to an equivalent SPICE string.

    Parameters
    ----------
    date_format : str
        Python datetime format string (e.g., "%Y-%m-%d %H:%M:%S.%f").
        NOTE: Does not support every Python format code. Formats with a
        non-none return in `PY_TO_SPICE_FORMAT` are supported.

    Returns
    -------
    str
        SPICE UTC format string.

    Examples
    --------
    >>> py_format = '%Y-%m-%d %H:%M:%S.%f'
    >>> print(spice_strftime(py_format))
    'YYYY-MM-DD HR:MN:SC.######::RND ::UTC'

    References
    ----------
    Python's format string:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    SPICE's format string:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/timout_c.html#Particulars

    """
    dt_fmt = ''
    meta = '::RND ::UTC'
    ix_last = 0

    # Special case where microseconds must follow seconds.
    date_format = re.sub(
        r'(?<!%)%S\.%f',
        PY_TO_SPICE_FORMAT['%S'] + PY_TO_SPICE_FORMAT['%f'],
        date_format
    )

    for reg in re.finditer(r'(?<!%)%\w', date_format):
        fmt_code = PY_TO_SPICE_FORMAT.get(reg.group(), None)
        if fmt_code is None:
            raise ValueError('SPICE does not support the Python date format code: {!r}'.format(reg.group()))

        # Special case where microseconds must follow seconds.
        #   Already replaced the only valid configuration.
        if reg.group() == '%f':
            raise ValueError('Invalid use of `%f`. SPICE only supports `%f` if preceded by `%S.`.')

        dt_fmt += date_format[ix_last:reg.start()]
        dt_fmt += fmt_code
        ix_last = reg.end()

    dt_fmt += date_format[ix_last:]
    dt_fmt += meta
    return dt_fmt.replace('%%', '%')
