"""Vectorized SPICE functions which properly handle numpy arrays.

@author: Brandon Stone
"""
from functools import wraps

import numpy as np
import spiceypy


def _vectorize(pyfunc, **vec_kwargs):
    """Vectorize a func and supply it as the first arg to a wrapper.
    """
    vectorized_func = np.vectorize(pyfunc, **vec_kwargs)

    def outer_wrapper(wrapper_func):
        """Decorate a wrapper function. Maintains the docstring and func name!
        """

        @wraps(vectorized_func)
        def inner_wrapper(*args, **kwargs):
            """Call the wrapper with the vectorized func.
            """
            return wrapper_func(vectorized_func, *args, **kwargs)

        return inner_wrapper

    return outer_wrapper


@wraps(spiceypy.recgeo)  # Maintains the original docstring.
def recgeo(rectan, re, f, as_deg=False):
    """Convert rectangular coords to geodetic (lon/lat/alt) (supports 2D ndarrays).
    """
    if not isinstance(rectan, np.ndarray):
        rectan = np.asarray(rectan)
    if rectan.ndim == 1:
        geodetic = np.asarray(spiceypy.recgeo(rectan, re, f), dtype=np.float64)
        if as_deg:
            geodetic[[0, 1]] = np.rad2deg(geodetic[[0, 1]])
    else:
        geodetic = np.zeros(rectan.shape, dtype=np.float64)
        for i in range(rectan.shape[0]):
            geodetic[i, :] = spiceypy.recgeo(rectan[i, :], re, f)
        if as_deg:
            geodetic[:, [0, 1]] = np.rad2deg(geodetic[:, [0, 1]])
    return geodetic


@wraps(spiceypy.spkezr)
def spkezr(targ, et, ref, abcorr, obs):
    """Get position and velocity (supports ndarrays).
    """
    # Add support for target/observer codes (int).
    if isinstance(targ, int):
        targ = spiceypy.bodc2n(targ)
    if isinstance(obs, int):
        obs = spiceypy.bodc2n(obs)

    # Add support for arrays of time.
    if hasattr(et, '__iter__') and not (isinstance(et, np.ndarray) and et.ndim == 0):
        vlen = len(et)
        position_velocity = np.empty((vlen, 6), dtype=np.float64)
        light_times = np.empty(vlen, dtype=np.float64)
        for index, time in enumerate(et):
            position_velocity[index], light_times[index] = spiceypy.spkezr(targ, time, ref, abcorr, obs)
        return position_velocity, light_times

    return spiceypy.spkezr(targ, et, ref, abcorr, obs)


@wraps(spiceypy.spkezp)
def spkezp(targ, et, ref, abcorr, obs):
    """Get position (supports ndarrays).
    """
    # Add support for target/observer names (str).
    if isinstance(targ, str):
        targ = spiceypy.bodn2c(targ)
    if isinstance(obs, str):
        obs = spiceypy.bodn2c(obs)

    # Add support for arrays of time.
    if hasattr(et, '__iter__') and not (isinstance(et, np.ndarray) and et.ndim == 0):
        vlen = len(et)
        position_arr = np.empty((vlen, 3), dtype=np.float64)
        light_times = np.empty(vlen, dtype=np.float64)
        for index, time in enumerate(et):
            position_arr[index], light_times[index] = spiceypy.spkezp(targ, time, ref, abcorr, obs)
        return position_arr, light_times

    return spiceypy.spkezp(targ, et, ref, abcorr, obs)


@wraps(spiceypy.ckgp)
def ckgp(inst, sclkdp, tol, ref):
    """Get pointing (attitude) as a rotation matrix (supports ndarrays).
    """
    # Add support for object names (str).
    if isinstance(inst, str):
        inst = spiceypy.namfrm(inst)
    if isinstance(ref, int):
        ref = spiceypy.frmnam(ref)

    # Add support for arrays of time.
    if hasattr(sclkdp, '__iter__') and not (isinstance(sclkdp, np.ndarray) and sclkdp.ndim == 0):
        vlen = len(sclkdp)
        attitude_arr = np.empty((vlen, 3, 3), dtype=np.float64)
        sclk_times = np.empty(vlen, dtype=np.float64)
        # The SpiceyPy docs are wrong (copied C).
        # pylint: disable=unbalanced-tuple-unpacking
        for index, time in enumerate(sclkdp):
            attitude_arr[index], sclk_times[index] = spiceypy.ckgp(
                inst, time, tol, ref
            )
        return attitude_arr, sclk_times

    return spiceypy.ckgp(inst, sclkdp, tol, ref)


# TODO: Add support for sister function that gets angular velocity too (ckgpav). Need a special test kernel!


@_vectorize(spiceypy.sce2c, otypes=[np.float64], excluded='sc')
def sce2c(func, sc, et):
    """Convert ephemeris time (ET) to continuous encoded spacecraft clock
    "ticks" (supports ndarrays).
    """
    # Add support for object names (str).
    if isinstance(sc, str):
        sc = spiceypy.bodn2c(sc)
    return func(sc, et)


@_vectorize(spiceypy.sct2e, otypes=[np.float64], excluded='sc')
def sct2e(func, sc, sclkdp):
    """Convert continuous encoded spacecraft clock "ticks" to ephemeris time
    (ET; supports ndarrays).
    """
    # Add support for object names (str).
    if isinstance(sc, str):
        sc = spiceypy.bodn2c(sc)  # TODO: Correct?
    return func(sc, sclkdp)


@wraps(spiceypy.str2et)
def str2et(time, *args, **kwargs):
    """Convert strings to Ephemeris Time (ET) (supports ndarrays).
    """
    if isinstance(time, np.ndarray):
        return spiceypy.str2et(time.tolist(), *args, **kwargs)

    return spiceypy.str2et(time, *args, **kwargs)


@wraps(spiceypy.timout)
def timout(et, *args, **kwargs):
    """Convert Ephemeris Time (ET @J2000) to a string (supports scalar (0D) ndarrays).
    """
    input_shape = None
    if isinstance(et, np.ndarray):
        if et.ndim == 0:
            return spiceypy.timout(et.tolist(), *args, **kwargs)

        if et.ndim > 1:
            input_shape = et.shape
            et = et.ravel()

    out = spiceypy.timout(et, *args, **kwargs)
    if input_shape is not None:
        out = out.reshape(input_shape)
    return out


@_vectorize(spiceypy.unitim, excluded=['insys', 'outsys'], cache=True)
def unitim(func, epoch, *args, **kwargs):
    """Convert between uniform (count) time scales (supports ndarrays).
    """
    output = func(epoch, *args, **kwargs)
    if isinstance(output, np.ndarray) and output.ndim == 0:
        output = output.tolist()  # Converts it to a scalar (not a list).
    return output
