"""Leapsecond kernel methods.

Importing this module will load the default leapsecond kernel that is included
with this library. If the kernel pool is cleared, `load()` should be called to
reload the leapsecond kernel. The included kernel can be updated using
`update_file()`; if a new kernel is available, it will be downloaded to the
package's "data" directory.

The last leapsecond kernel that is loaded takes the highest precedence.

@author: Brandon Stone
"""
import datetime
import logging
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .. import spicierpy as sp


logger = logging.getLogger(__name__)

_LEAPSECOND_FILE_PATH = '../../data/generic'
_LEAPSECOND_FILE_GLOB = 'naif*.tls'
LEAPSECOND_BASE_URL = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/'

LEAPSECOND_USER_FILE_PATH = None


def find_default_file():
    """Find the library's default leapsecond kernel file.

    Returns
    -------
    pathlib.Path
        Path object for the default leapsecond kernel.

    """
    # Locate the latest kernel file, relative to this module file.
    #   Pylint thinks the `Path` class is really a `PurePath`.
    # pylint: disable=no-member
    if LEAPSECOND_USER_FILE_PATH is not None:
        path = Path(LEAPSECOND_USER_FILE_PATH)
    elif os.getenv('LEAPSECOND_FILE_ENV', None):
        path = Path(os.getenv('LEAPSECOND_FILE_ENV'))
    else:
        path = Path(__file__).parent
        path = path.joinpath(_LEAPSECOND_FILE_PATH).resolve()
    leapsecond_files = list(path.glob(_LEAPSECOND_FILE_GLOB))
    leapsecond_files.sort()

    if len(leapsecond_files) == 0:
        raise FileNotFoundError('Unable to find the default leapsecond kernel file. '
                                'Searched directory: {}'.format(path))

    file = leapsecond_files[-1]
    logger.debug('Found default leapsecond kernel: %s', file)

    # Warn if the file might be outdated (not modified in 3 yrs; units=sec).
    if (time.time() - file.stat().st_mtime) > int(63072000 / 2 * 3):
        # warnings.warn(
        logger.debug(
            'The leapsecond kernel is older than two years. Consider running `check_for_update` or `update_file`.'
        )

    return file


def are_loaded():
    """List the loaded leapsecond kernels (LSK), if any.

    Returns
    -------
    list of str
        List of the loaded leapsecond kernels. The last file has precedence.

    """
    files = []
    for i in range(sp.ktotal('text')):
        kernel_file = sp.kdata(i, kind='text')[0]
        _, ktype = sp.getfat(kernel_file)
        if ktype == 'LSK':
            files.append(kernel_file)
    return files


def load(filename=None):
    """Load a leapsecond kernel.

    Parameters
    ----------
    filename : str or pathlib.Path, optional
        Leapsecond kernel to load. If omitted, attempt to load the included
        kernel. Do not reload it if it has already been loaded.

    Returns
    -------
    None

    """
    loaded_files = are_loaded()

    # Load the supplied kernel file.
    if filename is not None:
        sp.furnsh(str(filename))

    # Skip if a file is already loaded.
    elif loaded_files:
        logger.debug('Leapsecond kernel(s) already loaded: %r', loaded_files)

    # Load the default kernel into the kernel pool.
    else:
        default_kernel = find_default_file()
        sp.furnsh(str(default_kernel))


def check_for_update():
    """Check for updated leapsecond kernels from NAIF.

    This should be run at least twice a year.

    Returns
    -------
    str or None
        If an update is available, returns the file name of the latest
        leapsecond kernel (e.g., "naif0012.tls"), otherwise None.

    """
    resp = requests.get(LEAPSECOND_BASE_URL, timeout=10)
    resp.raise_for_status()

    files = re.findall(r'href="(naif[0-9]{4}\.tls)"', resp.text)
    if len(files) == 0:
        raise ValueError('No files were found on the NAIF page: {!r}'.format(LEAPSECOND_BASE_URL))

    files.sort()
    logger.debug('Found files on NAIF page: %r', files)

    # Compare the file name to the default kernel.
    default_kernel = find_default_file()
    if default_kernel.name == files[-1]:
        logger.info('No update found. File name matches current file: %s', default_kernel)
        return None
    return files[-1]


def update_file():
    """Update the leapsecond kernel from NAIF.

    Returns
    -------
    pathlib.Path or None
        Path to the updated leapsecond file, or None if the current leapsecond
        file was already up-to-date. Note: If an update was found, then the
        module constant `LEAPSECOND_FILE` will be updated and the kernel will
        be loaded into memory (overriding any existing leapsecond kernels).

    """
    # Check if we need to update, and if so, the new filename.
    kernel_name = check_for_update()
    if kernel_name is None:
        return None

    # Form the URL to download and the destination filename.
    kernel_url = LEAPSECOND_BASE_URL + kernel_name
    default_file = find_default_file()
    kernel_file = default_file.with_name(kernel_name)
    if kernel_file.is_file():
        raise FileExistsError('New file already exists, but wasnt the default! File: {}'.format(kernel_file))

    # Download the latest leapsecond file.
    logger.debug('Downloading kernel file: %r', kernel_url)
    resp = requests.get(kernel_url, timeout=60)
    resp.raise_for_status()

    logger.debug('Saving kernel file: %r', kernel_file)
    kernel_file.write_text(resp.text)

    # Check the file header to ensure it's correct.
    _, ktype = sp.getfat(str(kernel_file))
    if ktype != 'LSK':
        logger.error('Downloaded file is not a leapsecond kernel (500):\n%s', kernel_file.read_text()[:500])
        kernel_file.unlink()
        raise AssertionError('Downloaded file does not appear to be a kernel file. Deleting...')

    # Load the new kernel file and update module constant.
    load(kernel_file)
    return kernel_file


# Find the library's default leapsecond file when imported.
#   Then load it into memory (SPICE kernel pool). If the kernel file is older
#   than 2 years (created/modified), warn that it might be outdated.
def _quiet_load():
    """Load the default leapsecond kernel, but suppress missing file errors.
    """
    try:
        load()
    except FileNotFoundError:
        logger.exception('An exception occurred while locating the leapsecond file. Suppressing:')
        warnings.warn('Unable to find the default leapsecond kernel file. No leapsecond kernel was loaded!')


_quiet_load()


def read_leapseconds(filename=None):
    """Determine the current leapseconds.

    Parameters
    ----------
    filename : str, optional
        Leapsecond file to read. Default=library leapsecond file.

    Returns
    -------
    pd.DataFrame
        Leapsecond data. Index is the time a leapsecond was added. Columns:
            nsec : int, cumulative number of leapseconds
            linux_offset : int, microsecond offset between unix time and uGPS.
            ugps : int, GPS microsecond time a leapsecond was added.

    """
    if filename is None:
        filename = str(find_default_file())
    with open(filename, 'r') as f:
        txt = f.read()

    microseconds_per_second = 1000000
    microseconds_per_day = int(8.64e10)

    entries = []
    for data in re.findall(r'\\begindata(.*?)\\begintext', txt, re.I | re.M | re.DOTALL):
        for delta_at in re.findall(r'DELTET/DELTA_AT\s*=\s*\((.*?)\)', data, re.I | re.M | re.DOTALL):
            entries.extend(
                re.findall(r'([0-9]+),\s*@([0-9]{4}-[A-Z]{3}-[0-9]+)', delta_at, re.I | re.M | re.DOTALL)
            )
    nleapsec, dates = zip(*entries)

    # Convert to date string to a numpy datetime64 array.
    utc_dates = np.array([datetime.datetime.strptime(dt, '%Y-%b-%d') for dt in dates], dtype='M8[us]')

    leapsec = pd.DataFrame({'nsec': np.array(nleapsec, dtype=int)},
                           index=pd.DatetimeIndex(utc_dates))

    # Pre-compute the static offset (uGPS to unix).
    # Adjust for the leapsecond offset with the GPS epoch.
    usec_leaps = (leapsec['nsec'] - 19) * microseconds_per_second

    # Adjust for unix epoch vs. GPS epoch.
    usec_shift = 3657 * microseconds_per_day

    # Calc epoch times for reference.
    ndays = (utc_dates - np.datetime64('1970-01-01', 'D')).astype('m8[D]').astype(int)
    leapsec['unix'] = ndays * microseconds_per_day
    leapsec['ugps'] = leapsec['unix'] - usec_shift + usec_leaps

    # Epoch shift without leapseconds.
    leapsec['offset'] = usec_shift - usec_leaps

    return leapsec


class LeapsecondCache:
    """Simply class for storing leapsecond data in-memory.
    """

    def __init__(self):
        """Create a new cache.
        """
        self._data = None

    def get(self):
        """Get the leapsecond information.

        The first call will trigger a read, otherwise a cached copy is used.

        Returns
        -------
        pandas.DataFrame
            Leapsecond data. See method: `read_leapseconds`

        """
        if self._data is None:
            self._data = read_leapseconds()
        return self._data


cache = LeapsecondCache()
