"""Time constants.

@author: Brandon Stone
"""

from enum import Enum, IntEnum


class TimeConstant(IntEnum):
    """Time related constants."""

    SEC_TO_USEC = 1000000
    DAY_TO_USEC = int(8.64e10)


class EpochOffsetSeconds(IntEnum):
    """Seconds between epochs."""

    GPS_TO_J2000ET = 630763181  # GPS seconds to 2000-01-01T11:59:28.0 (J2000 ET, in UTC)
    GPS_TO_TAI = -694656019  # GPS seconds to 1958-01-01T00:00:00.0 (TAI epoch, in UTC)
    GPS_LEAPSECONDS = 19  # Leapseconds elapsed prior to 1980-01-06


class EpochGpsDays(float, Enum):
    """Epochs in fractional days since 1980-01-06."""

    GPS = 0  # 1980-01-06T00:00:00
    JD = -2444244.5
    MJD = -44244
    SORCE = 8419  # 2003-01-24T00:00:00
    TCTE = 12370  # 2013-11-18T00:00:00
    TSIS = 13857  # 2017-12-14T00:00:00
    TSIS1 = TSIS
    TSIS2 = 16049  # 2023-12-15T00:00:00 (estimated)
    CTIM = 15518  # 2022-07-02T00:00:00
