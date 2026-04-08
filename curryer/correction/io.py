"""Unified path resolution for the correction pipeline.

All file loading in the correction package should go through resolve_path()
to transparently handle local paths and S3 URIs.

S3 access follows the same pattern as :mod:`curryer.correction.dataio`:
callers may provide an explicit ``s3_client`` (useful for testing) or rely
on the default boto3 client.
"""

from __future__ import annotations

import atexit
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Registry of temp files created by S3 downloads. Cleaned up at process exit.
_temp_files: list[Path] = []


def _cleanup_temp_files() -> None:
    """Remove temporary files created by S3 downloads."""
    for path in _temp_files:
        try:
            if path.exists():
                path.unlink()
                logger.debug("Cleaned up temp file: %s", path)
        except OSError as exc:
            logger.warning("Failed to clean up temp file %s: %s", path, exc)
    _temp_files.clear()


atexit.register(_cleanup_temp_files)


def _require_client(client: object | None) -> object:
    """Return *client* if given, otherwise create a default boto3 S3 client.

    This is intentionally identical to :func:`curryer.correction.dataio._require_client`.

    Parameters
    ----------
    client : object or None
        An injected S3 client, or None to use the default.

    Returns
    -------
    object
        A boto3 S3 client.

    Raises
    ------
    ImportError
        If *client* is None and boto3 is not installed.
    """
    if client is not None:
        return client
    try:
        import boto3
    except ImportError:
        raise ImportError("S3 paths require boto3. Install with: pip install boto3")
    return boto3.client("s3")


def resolve_path(path: str | Path, *, s3_client=None) -> Path:
    """Resolve a file path, downloading from S3 if necessary.

    Parameters
    ----------
    path : str or Path
        Local file path or S3 URI (``s3://bucket/key``).
    s3_client : boto3 S3 client, optional
        Injected client for testing. If omitted and *path* is an S3 URI,
        a default client is created via boto3.

    Returns
    -------
    Path
        Local file path. For S3 URIs, a temporary local file that is
        cleaned up at process exit via :func:`atexit`.

    Raises
    ------
    ImportError
        If *path* is an S3 URI and boto3 is not installed.
    FileNotFoundError
        If *path* is a local path that does not exist.
    ValueError
        If *path* is an S3 URI with no object key (e.g. ``s3://bucket``).
    """
    path_str = str(path)
    if path_str.startswith("s3://"):
        return _download_from_s3(path_str, s3_client=s3_client)
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")
    return local_path


def _download_from_s3(s3_uri: str, *, s3_client=None) -> Path:
    """Download an S3 object to a local temporary file.

    The temp file is registered for cleanup at process exit via
    :func:`_cleanup_temp_files`.

    Parameters
    ----------
    s3_uri : str
        S3 URI in the form ``s3://bucket/key``.
    s3_client : boto3 S3 client, optional
        Injected client for testing.

    Returns
    -------
    Path
        Path to the local temporary file.

    Raises
    ------
    ImportError
        If boto3 is not installed and no client was injected.
    ValueError
        If the URI has no object key.
    """
    client = _require_client(s3_client)

    # Parse s3://bucket/key
    stripped = s3_uri.replace("s3://", "", 1)
    parts = stripped.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    if not key or not key.strip():
        raise ValueError(
            f"S3 URI must include an object key: {s3_uri!r}\nExpected format: s3://bucket/path/to/file.ext"
        )

    # Determine suffix from key for proper file handling
    suffix = Path(key).suffix or ""

    logger.info("Downloading from S3: %s", s3_uri)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        client.download_file(bucket, key, tmp.name)
    except Exception:
        # Clean up partial download on failure
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise
    tmp.close()

    tmp_path = Path(tmp.name)
    _temp_files.append(tmp_path)
    logger.info("  Downloaded to: %s", tmp_path)
    return tmp_path
