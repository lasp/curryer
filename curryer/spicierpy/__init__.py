# pylint: disable=wildcard-import
"""A Spicier version of SpiceyPy (vectorized + extensions).

@author: Brandon Stone
"""

#
# Import everything from SpiceyPy. Required to "replace" issue methods.
#
from spiceypy import *

#
# Lastly load in the custom modules.
#
from . import ext, obj

#
# Override non-vectorized routines.
#
from .vectorized import (  # Time-related  # Ephemeris
    ckgp,  # Attitude
    recgeo,  # Utilities
    sce2c,
    sct2e,
    spkezp,
    spkezr,
    str2et,
    timout,
    unitim,
)

__all__ = ["ext", "obj", "ckgp", "recgeo", "sce2c", "sct2e", "spkezp", "spkezr", "str2et", "timout", "unitim"]
