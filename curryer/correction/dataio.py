"""Validation helpers and S3 data-access utilities for the correction pipeline.

S3 access relies on the boto3 S3 client.  Callers may either provide an
explicit client instance (useful for testing) or rely on the default client, in
which case boto3 must be installed and AWS credentials are read from the
standard ``AWS_*`` environment variables.
"""

from __future__ import annotations

import datetime as _dt
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

# TODO: Remove if boto3 is made a required dependency!
try:  # pragma: no cover - exercised indirectly when boto3 is available
    import boto3
except Exception:  # pragma: no cover - protects environments without boto3
    boto3 = None  # type: ignore

if TYPE_CHECKING:
    import pandas as pd


# ============================================================================
# Data Validation Helpers
# ============================================================================


def validate_telemetry_output(df: pd.DataFrame, config) -> None:
    """
    Validate that telemetry loader output has expected structure.

    Args:
        df: DataFrame returned by telemetry loader
        config: GeolocationSetup object

    Raises:
        TypeError: If not a DataFrame
        ValueError: If DataFrame is empty

    Note:
        Specific column requirements depend on mission and kernel configs.
        This performs basic structure checks only.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Telemetry loader must return pd.DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError("Telemetry loader returned empty DataFrame")


def validate_science_output(df: pd.DataFrame, config) -> None:
    """
    Validate that science loader output has expected structure.

    Args:
        df: DataFrame returned by science loader
        config: GeolocationSetup object

    Raises:
        TypeError: If not a DataFrame
        ValueError: If DataFrame is empty or missing required time field

    Example:
        >>> sci_df = pd.read_csv("science.csv")
        >>> validate_science_output(sci_df, config)
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Science loader must return pd.DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError("Science loader returned empty DataFrame")

    time_field = config.geo.time_field
    if time_field not in df.columns:
        raise ValueError(
            f"Science loader must include time field '{time_field}'. Available columns: {list(df.columns)}"
        )


# ============================================================================
# S3 Data Access Utilities
# ============================================================================


class S3Configuration:
    """Configuration describing how data is organised within an S3 bucket."""

    def __init__(self, bucket: str, base_prefix: str) -> None:
        self.bucket = bucket
        self.base_prefix = base_prefix.rstrip("/")

    def date_prefix(self, date: _dt.date) -> str:
        """Return the S3 prefix for ``date``."""

        return f"{self.base_prefix}/{date:%Y%m%d}/"


def _require_client(client: object | None) -> object:
    if client is not None:
        return client
    if boto3 is None:
        raise RuntimeError("boto3 is not available. Install boto3 or provide an explicit s3_client.")
    return boto3.client("s3")


def _iter_dates(start: _dt.date, end: _dt.date) -> Iterable[_dt.date]:
    cur = start
    step = _dt.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += step


def find_netcdf_objects(
    config: S3Configuration,
    start_date: _dt.date,
    end_date: _dt.date,
    *,
    s3_client=None,
) -> list[str]:
    """Return S3 object keys for NetCDF files in the given date range.

    Parameters
    ----------
    config : S3Configuration
        Describes the bucket and prefix layout.
    start_date, end_date : datetime.date
        Inclusive date range to scan for NetCDF files.
    s3_client : boto3 S3 client, optional
        Client instance to use.  If omitted, a default client is created.
    """

    client = _require_client(s3_client)
    keys: list[str] = []
    for date in _iter_dates(start_date, end_date):
        prefix = config.date_prefix(date)
        continuation_token = None
        while True:
            params = {"Bucket": config.bucket, "Prefix": prefix}
            if continuation_token:
                params["ContinuationToken"] = continuation_token
            response = client.list_objects_v2(**params)
            for obj in response.get("Contents", []):
                key = obj.get("Key", "")
                if key.lower().endswith(".nc"):
                    keys.append(key)
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
    return keys


def download_netcdf_objects(
    config: S3Configuration,
    object_keys: Iterable[str],
    destination: os.PathLike[str] | str,
    *,
    s3_client=None,
) -> list[Path]:
    """Download the specified S3 objects to ``destination``.

    Parameters
    ----------
    config : S3Configuration
        Describes the bucket hosting the objects.
    object_keys : iterable of str
        S3 object keys to download.
    destination : path-like
        Directory where the files should be stored.  It is created if needed.
    s3_client : boto3 S3 client, optional
        Client instance to use.  If omitted, a default client is created.
    """

    client = _require_client(s3_client)
    dest_root = Path(destination)
    dest_root.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for key in object_keys:
        filename = Path(key).name
        local_path = dest_root / filename
        client.download_file(config.bucket, key, str(local_path))
        downloaded.append(local_path)
    return downloaded
