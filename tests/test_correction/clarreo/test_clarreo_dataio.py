"""CLARREO-specific data I/O integration tests (require AWS credentials)."""

from __future__ import annotations

import datetime as dt
import logging
import os

import pytest

from curryer.correction.dataio import S3Configuration, find_netcdf_objects

logger = logging.getLogger(__name__)

_NEEDS_AWS = pytest.mark.skipif(
    not ((os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")) or os.getenv("C9_USER")),
    reason="Requires AWS credentials or Cloud9 environment.",
)


# TODO: Migrate to moto mock (follow-up issue)
@_NEEDS_AWS
def test_clarreo_find_l0_objects(tmp_path):
    """Find L0 telemetry objects in CSDS S3 bucket."""
    config = S3Configuration("clarreo", "L0/telemetry/hps_navigation/")
    keys = find_netcdf_objects(config, start_date=dt.date(2017, 1, 15), end_date=dt.date(2017, 1, 15))
    assert keys == ["L0/telemetry/hps_navigation/20170115/CPF_TLM_L0.V00-000.hps_navigation-20170115-0.0.0.nc"]


@_NEEDS_AWS
def test_clarreo_find_l1a_objects(tmp_path):
    """Find L1a science objects in CSDS S3 bucket."""
    config = S3Configuration("clarreo", "L1a/nadir/")
    keys = find_netcdf_objects(config, start_date=dt.date(2022, 6, 3), end_date=dt.date(2022, 6, 3))
    assert len(keys) == 34
    assert "L1a/nadir/20220603/nadir-20220603T235952-step22-geolocation_creation-0.0.0.nc" in keys
