# pylint: disable=wildcard-import
"""A Spicier version of SpiceyPy (vectorized + extensions).

@author: Brandon Stone
"""
#
# Import everything from SpiceyPy. Required to "replace" issue methods.
#
from spiceypy import *

#
# Override non-vectorized routines.
#
from .vectorized import recgeo  # Utilities
from .vectorized import sce2c, sct2e, unitim, timout, str2et  # Time-related
from .vectorized import spkezp, spkezr  # Ephemeris
from .vectorized import ckgp  # Attitude

#
# Lastly load in the custom modules.
#
from . import obj
from . import ext
