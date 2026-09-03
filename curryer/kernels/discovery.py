"""NAIF generic-kernel discovery.

Finds the current version of generic kernels on the NAIF server (or a
mirror) by scraping its index listings: leapsecond (LSK), planetary
ephemeris (DE SPK), planetary constants (text PCK), and Earth orientation
(binary PCK) files, each published under a versioned filename in a
type-specific subdirectory. Resolution returns URLs; fetching and caching
them is :mod:`curryer.kernels.cache`'s job.

@author: Matthew Maclay
"""

import logging
import re
from urllib.parse import urljoin

import requests

from . import cache

logger = logging.getLogger(__name__)

NAIF_GENERIC_KERNELS_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"

# Filename patterns for versioned NAIF generic kernels.
NAIF_LSK_REGEX = r"naif[0-9]{4}\.tls"
NAIF_DE_REGEX = r"de[0-9]{3}s\.bsp"
NAIF_PCK_REGEX = r"pck[0-9]{5}\.tpc"
NAIF_HIGH_PREC_PCK_REGEX = r"earth_[0-9]{6}_[0-9]{6}_[0-9]{6}\.bpc"
NAIF_EARTH_EXTENDED_PCK_REGEX = r"earth_[0-9]{4}_[0-9]{6}_[0-9]{4}_predict\.bpc"

# Index subdirectory for each kernel file extension, relative to the base URL.
NAIF_SUBDIR_BY_EXTENSION = {
    "tls": "lsk/",
    "tpc": "pck/",
    "bpc": "pck/",
    "tf": "fk/planets/",
    "bsp": "spk/planets/",
}

HTTP_ATTEMPTS = 3
HTTP_TIMEOUT_SEC = 30


def naif_index_url(kernel_file_regex: str, base_url: str | None = None, flat: bool = False) -> str:
    """Index page URL for the subdirectory that serves a kernel pattern.

    Parameters
    ----------
    kernel_file_regex : str
        Kernel filename pattern; its file extension selects the
        subdirectory (see ``NAIF_SUBDIR_BY_EXTENSION``).
    base_url : str, optional
        Server base URL. Default: the NAIF generic-kernels URL.
    flat : bool, optional
        Skip subdirectory routing — for test servers or mirrors that put
        every kernel in one directory. Default=False.

    Returns
    -------
    str
        URL of the index page to scrape.

    Raises
    ------
    ValueError
        If the pattern's extension has no known subdirectory.
    """
    base_url = NAIF_GENERIC_KERNELS_URL if base_url is None else base_url
    if not base_url.endswith("/"):
        # urljoin drops the last path component of a slash-less base.
        base_url += "/"
    if flat:
        return base_url
    extension = kernel_file_regex.rsplit(".", 1)[-1]
    if extension not in NAIF_SUBDIR_BY_EXTENSION:
        raise ValueError(
            f"Unknown NAIF kernel extension {extension!r} for pattern {kernel_file_regex!r};"
            f" expected one of {sorted(NAIF_SUBDIR_BY_EXTENSION)}"
        )
    return urljoin(base_url, NAIF_SUBDIR_BY_EXTENSION[extension])


def find_most_recent_naif_kernel(
    naif_base_url: str, kernel_file_regex: str, allowed_attempts: int = HTTP_ATTEMPTS
) -> str:
    """Find the most recent kernel matching a pattern on one index page.

    NAIF versions its generic kernels in the filename (e.g.
    ``naif0012.tls``), so the lexicographically last match is the most
    recent.

    Parameters
    ----------
    naif_base_url : str
        URL of the index page to scrape.
    kernel_file_regex : str
        Filename pattern to match against the page's links.
    allowed_attempts : int, optional
        Retries for fetching the index page. Default=``HTTP_ATTEMPTS``.

    Returns
    -------
    str
        Full URL of the most recent matching kernel.

    Raises
    ------
    requests.exceptions.RequestException
        If the index page fetch fails after all retries.
    ValueError
        If `allowed_attempts` is not positive, or no filenames on the page
        match the pattern.
    """
    if allowed_attempts < 1:
        raise ValueError(f"allowed_attempts must be >= 1; got {allowed_attempts}")
    kernel_link_regex = re.compile(f'href="({kernel_file_regex})"')

    try:
        resp = cache.get_with_retries(naif_base_url, timeout=HTTP_TIMEOUT_SEC, attempts=allowed_attempts)
    except requests.exceptions.RequestException as error:
        logger.error("Failed to fetch NAIF index after %d attempts: %s", allowed_attempts, error)
        raise

    file_names = kernel_link_regex.findall(resp.text)
    if len(file_names) == 0:
        raise ValueError(f"No files matching {kernel_file_regex!r} were found on the NAIF page: {naif_base_url}")

    file_names.sort()  # NAIF filenames sort properly.
    logger.debug("Found files on NAIF page: %r", file_names)

    return urljoin(naif_base_url if naif_base_url.endswith("/") else naif_base_url + "/", file_names[-1])


def find_latest_naif_kernel_url(
    kernel_file_regex: str,
    base_url: str | None = None,
    flat: bool = False,
    allowed_attempts: int = HTTP_ATTEMPTS,
) -> str:
    """Resolve a kernel pattern to the most recent kernel URL on the server.

    Combines subdirectory routing (:func:`naif_index_url`) with index
    scraping (:func:`find_most_recent_naif_kernel`). The result is a URL
    for :func:`curryer.kernels.cache.fetch` to consume.

    Parameters
    ----------
    kernel_file_regex : str
        Kernel filename pattern (see the module ``NAIF_*_REGEX`` constants).
    base_url : str, optional
        Server base URL. Default: the NAIF generic-kernels URL.
    flat : bool, optional
        Skip subdirectory routing for single-directory mirrors.
        Default=False.
    allowed_attempts : int, optional
        Retries for fetching the index page. Default=``HTTP_ATTEMPTS``.

    Returns
    -------
    str
        Full URL of the most recent matching kernel.
    """
    index = naif_index_url(kernel_file_regex, base_url=base_url, flat=flat)
    return find_most_recent_naif_kernel(index, kernel_file_regex, allowed_attempts=allowed_attempts)
