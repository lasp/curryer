#!/usr/bin/env python3
"""CLARREO-specific data preprocessing functions (compatibility shim).

.. deprecated::
    The Protocol-based loader pattern (``TelemetryLoader``, ``ScienceLoader``,
    ``GCPLoader``) has been removed.  Data loading is now config-driven via
    :class:`~curryer.correction.config.DataConfig`.

    This file is kept as a shim so existing imports continue to resolve.
    New code should call :mod:`scripts.clarreo_preprocess` directly and pass
    preprocessed CSV file paths to the pipeline.
"""

# ---------------------------------------------------------------------------
# Re-export from the canonical preprocessing module so old imports work.
# ---------------------------------------------------------------------------
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from clarreo_preprocess import (  # noqa: E402
    preprocess_clarreo_science as load_clarreo_science,
)
from clarreo_preprocess import (
    preprocess_clarreo_telemetry as load_clarreo_telemetry,
)

logger = logging.getLogger(__name__)


def load_clarreo_gcp(gcp_key: str, config=None):  # noqa: ANN001
    """Placeholder – GCPLoader protocol removed; pass GCP path in tlm_sci_gcp_sets."""
    logger.info("load_clarreo_gcp is a no-op placeholder (GCPLoader protocol removed).")
    return None


__all__ = ["load_clarreo_telemetry", "load_clarreo_science", "load_clarreo_gcp"]
