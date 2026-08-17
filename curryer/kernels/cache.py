"""Generic kernel file cache.

Resolves a kernel source — a local path, ``s3://`` URI, or ``http(s)://``
URL — to a local file under a per-version user cache directory, usable
standalone or through the kernel manager.

Cache entries are keyed by source **basename**: an entry younger than its
``max_age`` is reused without touching the network, while an older entry is
revalidated against the source's **file size** (local ``stat``, HTTP
``HEAD``, S3 ``head_object``) and re-downloaded only on a mismatch, since
rolling NAIF kernels (e.g. ``earth_latest_high_prec.bpc``) update under a
stable name. Distinct sources that share a basename share one entry (the
most recently fetched wins) — NAIF generic kernels version their filenames,
which makes the basename a sufficient key for its designed scope; sources
whose basenames collide across directories need distinct ``cache_dir``s.
Writes are atomic (temp file + rename), so an interrupted download never
leaves a partial entry. When the source is unreachable and a cached copy
exists, the stale copy is used with a warning rather than raising — offline
environments keep working on a warm cache.

The normal flow emits debug logs only, never warnings.

@author: Matthew Maclay
"""

import datetime
import logging
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..utils import get_local_cache_dir

logger = logging.getLogger(__name__)

HTTP_ATTEMPTS = 3
HTTP_TIMEOUT_SEC = 30
HTTP_BACKOFF_SEC = (1, 5, 20)


def get_with_retries(url: str, dest=None, timeout: int = HTTP_TIMEOUT_SEC, attempts: int = HTTP_ATTEMPTS):
    """GET a URL with backoff retries.

    Parameters
    ----------
    url : str
        URL to fetch.
    dest : binary file object, optional
        When given, the body is streamed into it (rewound and truncated on
        retry) and None is returned; otherwise the response is returned.
    timeout : int, optional
        Per-attempt timeout in seconds. Default=``HTTP_TIMEOUT_SEC``.
    attempts : int, optional
        Total attempts before the last error propagates.
        Default=``HTTP_ATTEMPTS``.

    Returns
    -------
    requests.Response or None
        The response when `dest` is None, otherwise None.

    Raises
    ------
    requests.exceptions.RequestException
        If every attempt fails.
    """
    for attempt in range(1, attempts + 1):
        try:
            if dest is None:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=8192):
                    dest.write(chunk)
            return None
        except requests.exceptions.RequestException as error:
            if attempt == attempts:
                raise
            if dest is not None:
                dest.seek(0)
                dest.truncate()
            delay = HTTP_BACKOFF_SEC[min(attempt - 1, len(HTTP_BACKOFF_SEC) - 1)]
            logger.debug("GET attempt %d/%d failed (%s); retrying in %ds: %s", attempt, attempts, error, delay, url)
            time.sleep(delay)
    return None


def clear_cache(cache_dir: Path | None = None) -> list[Path]:
    """Remove all cached files from a curryer cache directory.

    Refuses directories outside the curryer cache root (the parent of
    :func:`curryer.utils.get_local_cache_dir`, holding one subdirectory per
    package version) — this function deletes files indiscriminately, so it
    only ever operates on directories this package owns.

    Parameters
    ----------
    cache_dir : pathlib.Path, optional
        Directory to empty; must be the cache root or a descendant.
        Default: :func:`curryer.utils.get_local_cache_dir`.

    Returns
    -------
    list of pathlib.Path
        The removed files.

    Raises
    ------
    ValueError
        If `cache_dir` is not under the curryer cache root.
    """
    root = get_local_cache_dir().parent.resolve()
    cache_dir = get_local_cache_dir() if cache_dir is None else Path(cache_dir)
    resolved = cache_dir.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to clear {cache_dir}: not under the curryer cache root {root}")
    # Validation resolves symlinks; removal keeps the caller's path form so
    # returned entries compare equal to what fetch() handed out.
    removed = []
    if cache_dir.is_dir():
        for cached_file in cache_dir.iterdir():
            if cached_file.is_file():
                cached_file.unlink()
                removed.append(cached_file)
    return removed


def fetch(
    source: str | Path,
    max_age: datetime.timedelta | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a kernel source to a local cached file.

    Parameters
    ----------
    source : str or pathlib.Path
        Kernel source: an ``http(s)://`` URL, an ``s3://`` URI, or a local
        file path.
    max_age : datetime.timedelta, optional
        Maximum age before a cached entry is revalidated against the source
        (size check, then re-download on mismatch). Default: entries never
        go stale. Supply a short age for rolling kernels (e.g. Earth
        orientation) and a long or absent one for static kernels (DE, LSK).
    cache_dir : pathlib.Path, optional
        Cache directory. Default: :func:`get_local_cache_dir`.

    Returns
    -------
    pathlib.Path
        Path to the cached local file.

    Raises
    ------
    ValueError
        If the source has no file basename (e.g. ends with ``/``).
    FileNotFoundError
        If a local source does not exist and there is no cached copy.
    requests.exceptions.RequestException
        If an HTTP download fails after retries and there is no cached copy.
    botocore.exceptions.BotoCoreError or botocore.exceptions.ClientError
        If an S3 download fails and there is no cached copy.

    Warns
    -----
    UserWarning
        When the source is unreachable (or the refresh fails) and a stale
        cached copy is returned instead.

    Notes
    -----
    A refresh replaces the entry in place (atomic ``os.replace``). A SPICE
    handle furnished from the old file keeps reading the old inode; unload
    and re-furnish the returned path after a fetch that may have refreshed
    an already-furnished kernel.
    """
    source_str = str(source)
    cache_dir = get_local_cache_dir() if cache_dir is None else Path(cache_dir)
    entry = cache_dir / _basename(source_str)

    if entry.is_file():
        age_sec = time.time() - entry.stat().st_mtime
        if max_age is None or age_sec < max_age.total_seconds():
            logger.debug("Cache hit (age %.0fs): %s", age_sec, entry)
            return entry
        try:
            source_size = _source_size(source_str)
        except Exception as error:
            warnings.warn(f"Source {source_str} unreachable ({error}); using stale cached copy: {entry}")
            return entry
        if source_size is not None and source_size == entry.stat().st_size:
            entry.touch()
            logger.debug("Cache hit after size revalidation (%d bytes): %s", source_size, entry)
            return entry
        logger.debug("Cache entry stale (source size %s != cached %d): %s", source_size, entry.stat().st_size, entry)

    try:
        _materialize(source_str, entry)
    except Exception as error:
        if entry.is_file():
            warnings.warn(f"Failed to update {entry} from {source_str} ({error}); using stale cached copy")
            return entry
        raise
    logger.debug("Cached %s to %s", source_str, entry)
    return entry


def _basename(source: str) -> str:
    """Base filename of a source path, URL, or S3 URI.

    Raises
    ------
    ValueError
        If the source has no file basename (e.g. ends with ``/``), which
        would otherwise alias the cache directory itself.
    """
    if source.startswith(("http://", "https://", "s3://")):
        name = os.path.basename(urlparse(source).path)
    else:
        name = Path(source).name
    if not name:
        raise ValueError(f"Kernel source has no file basename: {source!r}")
    return name


def _source_size(source: str) -> int | None:
    """Size of the source in bytes, or None when the source does not report one."""
    if source.startswith(("http://", "https://")):
        resp = requests.head(source, timeout=HTTP_TIMEOUT_SEC, allow_redirects=True)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None
    if source.startswith("s3://"):
        import boto3

        bucket, key = _parse_s3_uri(source)
        return boto3.client("s3").head_object(Bucket=bucket, Key=key)["ContentLength"]
    return Path(source).expanduser().resolve().stat().st_size


def _materialize(source: str, entry: Path) -> None:
    """Download or copy the source into the cache entry atomically.

    The payload lands in a temp file inside the cache directory and is
    renamed into place, so a failed transfer never leaves a partial entry.
    """
    entry.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{entry.name}.", dir=entry.parent)
    try:
        with os.fdopen(temp_fd, "wb") as temp_file:
            if source.startswith(("http://", "https://")):
                get_with_retries(source, dest=temp_file)
            elif source.startswith("s3://"):
                _download_s3(source, temp_file)
            else:
                local = Path(source).expanduser().resolve()
                if not local.is_file():
                    raise FileNotFoundError(f"Local kernel file not found: {local}")
                with local.open("rb") as local_file:
                    shutil.copyfileobj(local_file, temp_file)
        os.replace(temp_name, entry)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _download_s3(uri: str, dest) -> None:
    """Stream an S3 object into an open binary file."""
    import boto3

    bucket, key = _parse_s3_uri(uri)
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]
    shutil.copyfileobj(body, dest)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into bucket and key."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")
    return bucket, key
