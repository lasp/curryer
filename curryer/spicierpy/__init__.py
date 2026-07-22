# pylint: disable=wildcard-import
"""A Spicier version of SpiceyPy (vectorized + extensions).

@author: Brandon Stone
"""

#
# Import everything from SpiceyPy. Required to "replace" issue methods.
#
try:
    from spiceypy import *
except AttributeError:
    # SpiceyPy 8.1.0-8.1.2 have a broken `__all__` (a missing comma merges two
    # entries into the bogus name "exceptionsstypes"), which kills wildcard
    # imports. Fall back to copying the module's public namespace directly.
    import spiceypy as _spiceypy

    globals().update({_k: _v for _k, _v in vars(_spiceypy).items() if not _k.startswith("_")})
    del _spiceypy

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
