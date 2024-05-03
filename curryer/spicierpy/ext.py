"""Extensions to SPICE and the wrapper SpiceyPy.

@author: Brandon Stone
"""
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy
from spiceypy.utils.exceptions import SpiceyError

from . import vectorized
from .obj import Instrument, Body, Frame
from .. import spicetime
from ..utils import capture_subprocess


logger = logging.getLogger(__name__)


class load_kernel:
    """SPICE Kernel Context Manager.
    """

    def __init__(self, kernels):
        """SPICE Kernel Context Manager

        Parameters
        ----------
        kernels : str or iter of str or dict
            Kernel file(s) to load.

        Examples
        --------
        >>> kernel_fn = './kernels/tcte/tcte_meta_v01.tm'
        >>> with load_kernel(kernel_fn):
        >>>     print('Earth frame: %s' % center2frame('earth'))
        Earth's frame: ITRF93

        Notes
        -----
        - [Dev] See "contextlib.ContextDecorator" for python 3.2+

        """
        # TODO: Should this be a "set"? What happens if you load the same kernel twice and unload once?
        self._loaded = []
        self._iter_load(kernels)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload(clear=True)

    def __del__(self):
        self.unload(clear=True)

    @property
    def loaded(self):
        """List of kernels that have been loaded.
        """
        return self._loaded

    def _iter_load(self, kernels):
        """Load multiple kernels.
        """
        if isinstance(kernels, (str, Path)):
            self.load(str(kernels))
        elif isinstance(kernels, (list, tuple)):
            for k in kernels:
                self._iter_load(k)
        elif hasattr(kernels, 'keys'):
            if 'meta' in kernels:
                self._iter_load(kernels['meta'])
            for k in kernels:
                if k != 'meta':
                    self._iter_load(kernels[k])
        else:
            raise ValueError('Invalid `kernels`: {!r}'.format(kernels))

    def load(self, kernel):
        """Load a kernel file.
        """
        logger.debug('Loading kernel: %r', kernel)
        spiceypy.furnsh(kernel)
        self._loaded.append(kernel)

    def unload(self, kernel=None, clear=False):
        """Unload a kernel file.
        """
        if clear:
            # spiceypy.kclear()
            # self._loaded.clear()
            for k in reversed(self.loaded):
                self.unload(k)
        elif isinstance(kernel, str):
            logger.debug('Unloading kernel: %r', kernel)
            spiceypy.unload(kernel)
            self._loaded.remove(kernel)
        else:
            raise ValueError('Must specify `kernel` (str) or `all`=True.')


def object_frame(obj_name, as_id=False):
    """Retrieve frame name/id associated with an object name or id.

    Parameters
    ----------
    obj_name : str or int
        Object name or id to get the frame for.
    as_id : bool, default=False
        If True, change return to be the frame id instead of the name.

    Returns
    -------
    frame_name : str or int
        Associated frame name, or id if `as_id=True`.

    """
    if isinstance(obj_name, int):
        obj_name = spiceypy.bodc2n(obj_name)
    return spiceypy.cnmfrm(obj_name)[not as_id]


def kernel_coverage(filename, body, as_segments=False, to_fmt='ugps'):
    """Determine the coverage window for an entire kernel file.

    Parameters
    ----------
    filename : str or Path
        SPICE kernel file.
    body : int or str
        NAIF body code or name to check coverage of. A string is assumed to be
        a body name, while an int is assumed to be a body code.
        Note: Using a body name requires that the body definition is loaded
        into memory; integer codes will work regardless.
    as_segments : bool
        Option to return the coverage of each segment.
    to_fmt : str, optional
        Datetime format for the returned times. Default='ugps'.

    Returns
    -------
    tuple of two int or float or str
        Start and stop coverage times in `to_fmt` or GPS microseconds.

    Notes
    -----
    Kernel dependencies of `filename` should be loaded prior to the function
    call (i.e., the spacecraft clock kernel (SCLK) for an attitude kernel
    (CK)).

    """
    # Determine the kernel type (returns uppercase).
    filename = str(filename)
    _, ktype = spiceypy.getfat(filename)

    # TODO: Switch to using objs for `body`.

    # Ephemeris kernel.
    if ktype == 'SPK':
        if isinstance(body, Frame):
            body = body.body
        elif not isinstance(body, Body):
            body = Body(body)
        window = spiceypy.spkcov(
            filename,
            idcode=body.id,
            # cover=window
        )

    # Attitude (orientation, pointing) kernel.
    elif ktype == 'CK':
        if isinstance(body, Body):
            body = body.frame
        elif not isinstance(body, Frame):
            body = Frame(body)
        window = spiceypy.ckcov(
            filename,
            idcode=body.id,
            needav=False,
            level='SEGMENT',
            tol=0,
            timsys='TDB'
        )

    # Valid, but unsupported kernel types.
    elif ktype in ['PCK']:
        # TODO: Consider implementing using: pckcov
        raise NotImplementedError('For kernel type: {!r}'.format(ktype))

    # Invalid kernel types (i.e., no related "coverage" function).
    else:
        raise ValueError('Unknown or unexpected kernel type: {!r} from {!r}'.format(ktype, filename))

    # Option to do the overall range.
    window = tuple(window)
    if not as_segments and len(window) > 2:
        window = (window[0], window[-1])

    # Return times in uGPS.
    if len(window) == 0:
        contains_ids = 'UNKNOWN'
        try:
            contains_ids = kernel_objects(filename, as_id=True)
            if contains_ids:
                contains_ids = ', '.join(str(val) for val in contains_ids)
        except:
            logger.exception('Exception occurred while preparing another exception! Suppressing...')
        raise ValueError(f'No data for body [{body}] was found in the kernel containing IDs=[{contains_ids}],'
                         f' type=[{ktype}], file: {filename!r}')
    return spicetime.adapt(window, 'et', to_fmt)


def kernel_objects(filename, as_id=False):
    """Determine what objects (names or codes) are within a kernel file.

    Parameters
    ----------
    filename : str
        SPICE kernel file.
    as_id : bool, optional
        If False (default) return the NAIF body names, otherwise return the
        body codes.
        Note: Using a body name requires that the body definition is loaded
        into memory; integer codes will work regardless.

    Returns
    -------
    tuple of str or tuple of ints
        Collection of NAIF body names (default) or codes found within the
        kernel file (`filename`).

    """
    # Determine the kernel type (returns uppercase).
    filename = str(filename)
    _, ktype = spiceypy.getfat(filename)

    # Ephemeris kernel.
    if ktype == 'SPK':
        objs = tuple(spiceypy.spkobj(filename))
        if not as_id:
            objs = tuple(Body(v) for v in objs)

    # Attitude (orientation, pointing) kernel.
    elif ktype == 'CK':
        objs = tuple(spiceypy.ckobj(filename))
        if not as_id:
            objs = tuple(Frame(v) for v in objs)

    # Valid, but unsupported kernel types.
    elif ktype in ['DSK']:
        raise NotImplementedError('For kernel type: {!r}'.format(ktype))

    # Invalid kernel types (i.e., no related "obj code" function).
    else:
        raise ValueError('Unknown or unexpected kernel type: {!r} from {!r}'.format(ktype, filename))
    return objs


def infer_ids(spacecraft_name, spacecraft_id, instruments=None, from_dsn=False, from_norad=False):
    """Infer NAIF IDs based on the spacecraft ID.

    Useful when planning out the necessary IDs; not meant to reverse existing
    IDs since the following rules are not strictly enforced.

    Parameters
    ----------
    spacecraft_name : str
        Spacecraft or mission name.
    spacecraft_id : int
        Spacecraft ID; all other IDs are based on this.
        NOTE: It should be a negative number, unless `from_dsn` or `from_norad`
        is used.
    instruments : str or list of str, optional
        One or more instrument names to create IDs for.
    from_dsn : bool, optional
        Option to interpret `spacecraft_id` as a JPL Deep Space Network (DSN)
        ID. It will be converted to a SPICE-like spacecraft ID using the
        standard convention.
    from_norad : bool, optional
        Option to interpret `spacecraft_id` as a NORAD tracking ID. It will be
        converted to a SPICE-like spacecraft ID using the standard convention.

    Returns
    -------
    collections.dict
        Collection of the mission name and IDs (spacecraft, clock, ephemeris,
        attitude), and instrument names and IDs.

    """
    # TODO: Remove this now that there is the obj module?
    if isinstance(instruments, str):
        instruments = [instruments]
    elif instruments is None:
        instruments = []

    # Spacecraft ID.
    #   Should generally be a negative number.
    if from_dsn:
        spacecraft_id = spacecraft_id * -1
    elif from_norad:
        spacecraft_id = -100000 - spacecraft_id

    # Spacecraft clock ID.
    #   Typically the same as the SC ID.
    clock_id = spacecraft_id

    # Ephemeris ID.
    #   Always the spacecraft ID.
    ephemeris_id = spacecraft_id

    # Attitude ID (c-kernel ID; platform ID).
    #   Typically `SC ID * 1000`. Some methods will fail when assuming that and
    #   there is a remainder. Can think of this a s the zero-ith instrument.
    attitude_id = spacecraft_id * 1000

    # Instrument IDs.
    #   Typically `SC ID * 1000 - ordinal number`, starting at ordinal 1.
    instrument_ids = []
    for i, instrument_name in enumerate(instruments, 1):
        instrument_ids.append((instrument_name, spacecraft_id * 1000 - i))

    return dict([
        ('mission', spacecraft_name),
        ('spacecraft', spacecraft_id),
        ('clock', clock_id),
        ('ephemeris', ephemeris_id),
        ('attitude', attitude_id),
        ('instruments', dict(instrument_ids))
    ])


def instrument_boresight(instrument, n_vectors=1, norm=False):
    """Retrieve an instrument's boresight vector from the kernel pool.

    Parameters
    ----------
    instrument : str or int or sds_spice.spicierypy.obj.Instrument
        The instrument ID, name or object to retrieve the boresight of.
    n_vectors : int, optional
        Number of vectors to retrieve. Default=1
    norm : bool, optional
        Option to return normalized vectors. Default=False

    Returns
    -------
    numpy.ndarray

    """
    instrument = Instrument(instrument)
    _, _, boresight_vector, _, _ = spiceypy.getfov(instrument.id, n_vectors, 80, 80)
    if norm:
        boresight_vector /= np.linalg.norm(boresight_vector, axis=boresight_vector.ndim == 2)
    return boresight_vector


def brief(kernel_file, bin_=None):
    """Brief summary of a kernel file.
    """
    if bin_ is None:
        bin_ = shutil.which('brief')
        if bin_ is None:
            raise FileNotFoundError('Unable to find executable "brief" in system PATH.')
    cmd = [bin_, os.path.realpath(kernel_file)]
    return capture_subprocess(cmd, capture_output=True)


def spice_error_to_val(err_value=None, err_flag=None, pass_flag=None, disable=False):
    """Wrapper to catch spice errors and convert them to non-error values.

    Parameters
    ----------
    err_value : any
        Value to return when a SPICE error is encountered (e.g. nans).
    err_flag : any
        Value to return with the `err_value` to indicate that an error had
        occurred. Can be a callable, excepting the error object.
    pass_flag : any
        Value to return with the function's return to indicate that no error had
        occurred. Can be a callable, excepting the output value.
    disable : bool
        Option to disable to the error handling.

    Returns
    -------
    any
        Return from the wrapped function if no error was encountered, otherwise
        the err_value.
    any
        Flag indicating if an error occurred (`err_flag`) or not (`pass_flag`).

    """
    is_err_lookup = callable(err_flag)
    is_pass_lookup = callable(pass_flag)

    def wrapped_func(func):
        def wrapped_call(*args, **kwargs):
            try:
                out_value = func(*args, **kwargs)
                out_flag = pass_flag(out_value) if is_pass_lookup else pass_flag

            except SpiceyError as err:
                if disable:
                    raise

                out_value = err_value
                out_flag = err_flag(err) if is_err_lookup else err_flag

            return out_value, out_flag

        return wrapped_call

    return wrapped_func


POSITION_COLUMNS = ('x', 'y', 'z')
VELOCITY_COLUMNS = ('vx', 'vy', 'vz')


def query_ephemeris(ugps_times, target, observer, ref_frame='J2000', correction=None, velocity=False, allow_nans=False):
    """Query SPICE ephemeris data from pre-loaded kernels.

    Parameters
    ----------
    ugps_times : list of int
        One or more UGPS times to query data for.
    target : str or int or Body
        Name or ID of the target object.
    observer : str or int or Body
        Name or ID of the observing object.
    ref_frame : str or Frame, optional
        Reference frame of the ephemeris data. Default="J2000"
    correction : str, optional
        SPICE correction to apply to the data (e.g., "LT"). Default=None
    velocity : bool, optional
        Query position and velocity. Default is position only. Note that
        velocity requires angular velocity in all connected CK kernels.
    allow_nans : bool, optional
        Allow setting NaNs for times when insufficient SPICE kernel data
        would otherwise raise an exception. Note: Setting this flag
        significantly impacts read performance. Default=False.

    Returns
    -------
    pd.DataFrame

    """
    ref_frame = Frame(ref_frame)
    correction = correction or 'NONE'

    if not hasattr(ugps_times, '__iter__'):
        ugps_times = [ugps_times]
    et_times = spicetime.adapt(ugps_times, to='et')

    # Determine if we are reading position or position+velocity.
    #   This matters because if you request velocity, then any reference
    #   frame conversions require angular velcity in the CK kernel.
    if velocity:
        read_ephem = vectorized.spkezr
        default_columns = POSITION_COLUMNS + VELOCITY_COLUMNS

    else:
        read_ephem = vectorized.spkezp
        default_columns = POSITION_COLUMNS
    nan_output = np.array([np.NaN for _ in default_columns])

    # Load SPICE mappings.
    target = Body(target)
    observer = Body(observer)

    # Read the data from SPICE kernels.
    #   If time is a scalar, spice returns a 1d array of 6 values.
    #   If time is an iterable, spice returns a 2d array of N,6 values.
    logger.debug('Reading [%i] ephemeris values of [%s], from [%s], in frame [%s], with correction [%s]',
                 len(et_times), target.name, observer.name, ref_frame.name, correction)

    # Slowly iterate through the times. Catches insufficient data
    #   exceptions and returns NaNs instead.
    if allow_nans:
        def _checked_read_ephem(sample_et):
            """Read ephemeris data; converts insufficient data
            exceptions into NaNs.
            """
            # TODO: Update to use `spice_error_to_val`!
            try:
                sample_arr, _ = read_ephem(
                    target.id,
                    sample_et,
                    ref=ref_frame.name,
                    abcorr=correction,
                    obs=observer.id
                )
                return sample_arr
            except SpiceyError as e:
                if 'SPICE(SPKINSUFFDATA)' in e.short:  # Lacks ephemeris (generally).
                    return nan_output
                if 'SPICE(NOFRAMECONNECT)' in e.short:  # Lacks attitude (generally).
                    return nan_output
                if 'SPICE(NOTDISJOINT)' in e.short:  # Interp through an invalid ephemeris (generally a gap).
                    return nan_output  # Viewpoint is inside target.
                raise e

        arr = np.array(list(map(_checked_read_ephem, et_times)))

    # Can safely assume no NaNs (much faster).
    else:
        arr, _ = read_ephem(
            target.id,
            et_times,
            ref=ref_frame.name,
            abcorr=correction,
            obs=observer.id
        )

    # Organize the return data into a table.
    if arr.ndim == 1:
        ugps_times = [ugps_times]
        arr = arr[None]

    table = pd.DataFrame(arr, columns=default_columns, index=pd.Index(ugps_times, name='ugps'))
    table.columns.name = '{}.{}@{}'.format(target, observer, ref_frame)
    return table
