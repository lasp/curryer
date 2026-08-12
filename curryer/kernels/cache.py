"""Generic kernel file cache.

Resolves a kernel source — a local path, ``s3://`` URI, or ``http(s)://``
URL — to a local file under a per-version user cache directory, usable
standalone or through the kernel manager.

Cache identity is **basename + file size**, bounded by a per-entry maximum
age: an entry younger than ``max_age`` is reused without touching the
network; an older entry is revalidated against the source's size (local
``stat``, HTTP ``HEAD``, S3 ``head_object``) and re-downloaded only on a
mismatch, since rolling NAIF kernels (e.g. ``earth_latest_high_prec.bpc``)
update under a stable name. Writes are atomic (temp file + rename), so an
interrupted download never leaves a partial entry. When the source is
unreachable and a cached copy exists, the stale copy is used with a warning
rather than raising — offline environments keep working on a warm cache.

The normal flow emits debug logs only, never warnings.

@author: Matthew Maclay
"""

import datetime
import logging
import os
import shutil
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

HTTP_ATTEMPTS = 3
HTTP_TIMEOUT_SEC = 30

# The distribution registers under a different name than the import package.
DISTRIBUTION_NAME = "lasp-curryer"


def _package_version() -> str:
    """Installed version of this package, whichever name it registered under."""
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return version(__name__.split(".", 1)[0])


def get_local_cache_dir() -> Path:
    """Determine the user cache directory for this package version.

    The directory is keyed by installed package version, so a new release
    starts from an empty cache.

    Returns
    -------
    pathlib.Path
        Platform cache directory for the current curryer version. Not
        created by this function.
    """
    package_name = __name__.split(".", 1)[0]
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", "~/AppData/Local")).expanduser()
    elif os.uname().sysname == "Darwin":
        base = Path("~/Library/Caches").expanduser()
    else:
        base = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return base / package_name / version(package_name)


def clear_cache(cache_dir: Path | None = None) -> list[Path]:
    """Remove all cached files from a cache directory.

    Parameters
    ----------
    cache_dir : pathlib.Path, optional
        Directory to empty. Default: :func:`get_local_cache_dir`.

    Returns
    -------
    list of pathlib.Path
        The removed files.
    """
    cache_dir = get_local_cache_dir() if cache_dir is None else Path(cache_dir)
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
    FileNotFoundError
        If a local source does not exist and there is no cached copy.
    requests.exceptions.RequestException
        If an HTTP download fails after retries and there is no cached copy.
    botocore.exceptions.BotoCoreError or botocore.exceptions.ClientError
        If an S3 download fails and there is no cached copy.
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
            logger.warning("Source %s unreachable (%s); using stale cached copy: %s", source_str, error, entry)
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
            logger.warning("Failed to update %s from %s (%s); using stale cached copy", entry, source_str, error)
            return entry
        raise
    logger.debug("Cached %s to %s", source_str, entry)
    return entry


def _basename(source: str) -> str:
    """Base filename of a source path, URL, or S3 URI."""
    if source.startswith(("http://", "https://", "s3://")):
        return os.path.basename(urlparse(source).path)
    return Path(source).name


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
                _download_http(source, temp_file)
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


def _download_http(url: str, dest) -> None:
    """Stream an HTTP(S) URL into an open binary file, with retries."""
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SEC) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=8192):
                    dest.write(chunk)
            return
        except requests.exceptions.RequestException as error:
            if attempt == HTTP_ATTEMPTS:
                raise
            logger.debug("Download attempt %d/%d failed (%s); retrying: %s", attempt, HTTP_ATTEMPTS, error, url)
            dest.seek(0)
            dest.truncate()
            time.sleep(1)


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
