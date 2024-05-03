"""SPICE-based time conversions.

Module Listings
---------------
leapsecond
    Methods to handle finding, loading and updating leapsecond kernels.
    The included leapsecond kernel will be loaded on import of this package
    if there is not another one loaded.
convert
    Methods and classes to convert between time formats. Offers two APIs:
    `adapt` for converting from one format to another, and `SpiceTime` for
    tracking the time format in a numpy ndarray and converting to other time
    formats.

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

Examples
--------
>>> from curryer import spicetime
>>> print(spicetime.adapt(0, 'ugps', 'utc'))
'1980-01-06 00:00:00.000000'

>>> ugps = spicetime.SpiceTime([0, 1], 'ugps')
>>> print(repr(ugps))
SpiceTime([0, 1], ttype='ugps')

>>> print(ugps.to_utc('%S.%f'))
['00.000000' '00.000001']

@author: Brandon Stone
"""
from . import constants
from . import leapsecond
from . import native
from . import utils
from .convert import adapt
from .convert import SpiceTime
