"""Structured result models for the correction pipeline.

Provides :class:`CorrectionResult` and :class:`ParameterSetResult`,
returned by :func:`~curryer.correction.pipeline.run_correction`.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from curryer.correction.config import CorrectionConfig


class ParameterSetResult(BaseModel):
    """Results for a single parameter set in a correction sweep.

    Attributes
    ----------
    index : int
        Zero-based index of this parameter set in the sweep.
    parameter_values : dict[str, float]
        Parameter name → sampled value for this set.
    mean_rms_m : float
        Mean RMS geolocation error across all GCP pairs (metres).
    best_pair_rms_m : float
        RMS error of the best-performing GCP pair (metres).
    worst_pair_rms_m : float
        RMS error of the worst-performing GCP pair (metres).
    """

    index: int
    parameter_values: dict[str, float]
    mean_rms_m: float
    best_pair_rms_m: float
    worst_pair_rms_m: float


class CorrectionResult(BaseModel):
    """Structured result from :func:`~curryer.correction.pipeline.run_correction`.

    Provides the instrument engineer with a clear answer: what is the best
    parameter set, does it meet requirements, and what should they do next.

    Serialisation
    -------------
    Most fields are JSON-serialisable via ``model_dump()`` / ``model_dump_json()``.
    The ``results`` and ``netcdf_data`` fields are excluded from serialisation
    by default because they contain non-JSON-serialisable types (xr.Dataset,
    numpy arrays).  Access them directly on the object when raw data is needed::

        result = run_correction(config, work_dir, inputs)
        result.best_parameter_set   # dict[str, float] — use this
        result.results              # raw per-iteration dicts — advanced use only
        result.netcdf_data          # raw numpy arrays — advanced use only

    Attributes
    ----------
    best_parameter_set : dict[str, float]
        Parameter values that produced the lowest aggregate RMS.
    best_rms_m : float
        Best aggregate RMS achieved across all parameter sets (metres).
    best_index : int
        Index of the best parameter set (for cross-referencing with NetCDF output).
    worst_rms_m : float
        Worst aggregate RMS across all parameter sets (metres).
    mean_rms_m : float
        Mean of all aggregate RMS values (metres).
    n_parameter_sets : int
        Number of parameter sets tested in the sweep.
    n_gcp_pairs : int
        Number of GCP pairs used.
    all_parameter_sets : list[ParameterSetResult]
        All tested parameter sets sorted by mean RMS (ascending).
    met_threshold : bool
        Whether the best parameter set met the mission performance requirements.
    recommendation : str
        Human-readable next-step guidance for the instrument engineer.
    summary_table : str
        Human-readable ASCII table of the top results.
    netcdf_path : Path or None
        Path to the saved NetCDF output file.
    config_snapshot : dict
        Key config values used, for reproducibility records.
    elapsed_time_s : float
        Total wall-clock processing time in seconds.
    timestamp : datetime
        UTC time when the run completed.
    results : list
        Raw per-iteration result dicts from :func:`~curryer.correction.pipeline.loop`.
        Excluded from JSON serialisation.
    netcdf_data : dict
        Raw NetCDF numpy arrays from :func:`~curryer.correction.pipeline.loop`.
        Excluded from JSON serialisation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_parameter_set: dict[str, float]
    best_rms_m: float
    best_index: int
    worst_rms_m: float
    mean_rms_m: float
    n_parameter_sets: int
    n_gcp_pairs: int
    all_parameter_sets: list[ParameterSetResult]
    met_threshold: bool
    recommendation: str
    summary_table: str
    netcdf_path: Path | None = None
    config_snapshot: dict = Field(default_factory=dict)
    elapsed_time_s: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Raw data — accessible on the result object, excluded from JSON serialisation.
    results: list = Field(default_factory=list, exclude=True)
    netcdf_data: dict = Field(default_factory=dict, exclude=True)


# ============================================================================
# Internal table formatting helpers
# ============================================================================


def _fmt_rms(value: float) -> str:
    """Format an RMS value for display; returns ``'N/A'`` for non-finite values."""
    return f"{value:.1f}m" if math.isfinite(value) else "N/A"


def _format_correction_summary_table(
    top_sets: list[ParameterSetResult],
    total_sets: int,
    n_gcp_pairs: int,
    met_threshold: bool,
) -> str:
    """Generate a human-readable correction sweep summary table.

    Uses the same box-drawing and ljust/rjust pattern as
    :func:`~curryer.correction.verification._format_summary_table`.

    Parameters
    ----------
    top_sets : list[ParameterSetResult]
        Top-ranked parameter sets to display (typically ``all_parameter_sets[:10]``).
    total_sets : int
        Total number of parameter sets evaluated in the sweep.
    n_gcp_pairs : int
        Number of GCP pairs used.
    met_threshold : bool
        Whether any parameter set met performance requirements.

    Returns
    -------
    str
        Multi-line box-drawn ASCII table.
    """
    # Column widths
    w_rank = 6
    w_mean = 14
    w_best = 14
    w_worst = 14
    w_idx = 8
    w_mark = 4
    # col_inner = sum of column widths + number of separators (one between each column)
    col_inner = w_rank + w_mean + w_best + w_worst + w_idx + w_mark + 5

    title = f" Correction Sweep Summary ({total_sets} sets × {n_gcp_pairs} pairs)"

    if top_sets:
        best_line = f" Best: Set #{top_sets[0].index} — {_fmt_rms(top_sets[0].mean_rms_m)} mean RMS"
    else:
        best_line = " No results available"

    verdict_text = "MET REQUIREMENTS ✓" if met_threshold else "DID NOT MEET REQUIREMENTS ✗"
    verdict_line = f" {verdict_text}"

    inner_width = max(col_inner, len(title) + 2, len(best_line) + 2, len(verdict_line) + 2)

    def _h_sep(mid: str, fill: str = "─") -> str:
        """Build a column separator row padded to inner_width."""
        core = (
            fill * w_rank
            + mid
            + fill * w_mean
            + mid
            + fill * w_best
            + mid
            + fill * w_worst
            + mid
            + fill * w_idx
            + mid
            + fill * w_mark
        )
        return core + fill * max(0, inner_width - len(core))

    lines: list[str] = []
    lines.append("┌" + "─" * inner_width + "┐")
    lines.append("│" + title.ljust(inner_width) + "│")
    lines.append("├" + _h_sep("┬") + "┤")

    header = (
        "Rank".center(w_rank)
        + "│"
        + "Mean RMS".center(w_mean)
        + "│"
        + "Best Pair".center(w_best)
        + "│"
        + "Worst Pair".center(w_worst)
        + "│"
        + "Index".center(w_idx)
        + "│"
        + "".center(w_mark)
    )
    lines.append("│" + header.ljust(inner_width) + "│")
    lines.append("├" + _h_sep("┼") + "┤")

    for rank, ps in enumerate(top_sets, 1):
        marker = " ★" if rank == 1 else "  "
        row = (
            str(rank).center(w_rank)
            + "│"
            + _fmt_rms(ps.mean_rms_m).center(w_mean)
            + "│"
            + _fmt_rms(ps.best_pair_rms_m).center(w_best)
            + "│"
            + _fmt_rms(ps.worst_pair_rms_m).center(w_worst)
            + "│"
            + str(ps.index).center(w_idx)
            + "│"
            + marker.center(w_mark)
        )
        lines.append("│" + row.ljust(inner_width) + "│")

    lines.append("├" + "─" * inner_width + "┤")
    lines.append("│" + best_line.ljust(inner_width) + "│")
    lines.append("│" + verdict_line.ljust(inner_width) + "│")
    lines.append("└" + "─" * inner_width + "┘")

    return "\n".join(lines)


# ============================================================================
# Public factory
# ============================================================================


def build_correction_result(
    config: CorrectionConfig,
    results: list,
    netcdf_data: dict,
    netcdf_path: Path | None,
    elapsed_time_s: float,
) -> CorrectionResult:
    """Build a :class:`CorrectionResult` from raw :func:`~curryer.correction.pipeline.loop` outputs.

    Parameters
    ----------
    config : CorrectionConfig
        The correction configuration used for the run.
    results : list
        Per-iteration result dicts from :func:`~curryer.correction.pipeline.loop`.
    netcdf_data : dict
        Raw NetCDF data dict from :func:`~curryer.correction.pipeline.loop`.
    netcdf_path : Path or None
        Path to the saved NetCDF file, if any.
    elapsed_time_s : float
        Total wall-clock time of the run in seconds.

    Returns
    -------
    CorrectionResult
    """
    all_mean_rms: np.ndarray = netcdf_data.get("mean_rms_all_pairs", np.array([]))
    rms_grid: np.ndarray = netcdf_data.get("rms_error_m", np.empty((0, 0)))
    best_pair_rms_arr: np.ndarray = netcdf_data.get("best_pair_rms", np.array([]))
    worst_pair_rms_arr: np.ndarray = netcdf_data.get("worst_pair_rms", np.array([]))

    n_param_sets = len(all_mean_rms)
    n_gcp_pairs = int(rms_grid.shape[1]) if rms_grid.ndim == 2 and rms_grid.shape[0] > 0 else 0

    # Best / worst parameter sets
    valid_mask = ~np.isnan(all_mean_rms) if n_param_sets > 0 else np.array([], dtype=bool)
    if np.any(valid_mask):
        best_idx = int(np.nanargmin(all_mean_rms))
        best_rms = float(all_mean_rms[best_idx])
        worst_rms = float(all_mean_rms[int(np.nanargmax(all_mean_rms))])
        mean_rms = float(np.nanmean(all_mean_rms))
    else:
        best_idx = 0
        best_rms = float("nan")
        worst_rms = float("nan")
        mean_rms = float("nan")

    # Extract parameter arrays using the same naming rules as results_io.
    from curryer.correction.config import ParameterType  # local import to avoid cycles

    param_keys: list[str] = []
    for p in config.parameters:
        if p.ptype == ParameterType.CONSTANT_KERNEL:
            for angle in ("roll", "pitch", "yaw"):
                param_keys.append(config.netcdf.get_parameter_netcdf_metadata(p, angle).variable_name)
        else:
            param_keys.append(config.netcdf.get_parameter_netcdf_metadata(p).variable_name)

    # Keep only keys that are present and are 1-D arrays in netcdf_data
    param_keys = [
        k
        for k in param_keys
        if isinstance(netcdf_data.get(k), np.ndarray) and netcdf_data[k].ndim == 1
    ]

    # Build per-set results
    all_sets: list[ParameterSetResult] = []
    for idx in range(n_param_sets):
        pvals = {k: float(netcdf_data[k][idx]) for k in param_keys}
        all_sets.append(
            ParameterSetResult(
                index=idx,
                parameter_values=pvals,
                mean_rms_m=float(all_mean_rms[idx]),
                best_pair_rms_m=float(best_pair_rms_arr[idx]) if idx < len(best_pair_rms_arr) else float("nan"),
                worst_pair_rms_m=float(worst_pair_rms_arr[idx]) if idx < len(worst_pair_rms_arr) else float("nan"),
            )
        )
    all_sets.sort(key=lambda s: (float("inf") if math.isnan(s.mean_rms_m) else s.mean_rms_m))

    best_params: dict[str, float] = (
        {k: float(netcdf_data[k][best_idx]) for k in param_keys} if param_keys and n_param_sets > 0 else {}
    )

    # Evaluate requirements using legacy performance_threshold_m / performance_spec_percent.
    # (The new-style Requirement.evaluate_all() path is tracked by TODO(#151).)
    met_threshold = False
    if n_gcp_pairs > 0 and math.isfinite(best_rms) and rms_grid.shape[0] > best_idx:
        pair_errors = [float(rms_grid[best_idx, pi]) for pi in range(n_gcp_pairs)]
        valid_errors = [e for e in pair_errors if math.isfinite(e)]
        if valid_errors:
            pct_below = sum(1 for e in valid_errors if e < config.performance_threshold_m) / len(valid_errors) * 100
            met_threshold = pct_below >= config.performance_spec_percent

    # Human-readable recommendation
    if met_threshold:
        recommendation = (
            f"Best parameter set (#{best_idx}) achieved {best_rms:.1f}m mean RMS "
            f"and meets performance requirements. "
            f"Update kernel files with these values."
        )
    else:
        rms_str = f"{best_rms:.1f}m" if math.isfinite(best_rms) else "N/A"
        recommendation = (
            f"No parameter set met performance requirements. "
            f"Best achieved: {rms_str} mean RMS (set #{best_idx}). "
            f"Consider widening parameter bounds or increasing iterations."
        )

    summary_table = _format_correction_summary_table(all_sets[:10], n_param_sets, n_gcp_pairs, met_threshold)

    search_strategy = config.search_strategy
    config_snapshot = {
        "seed": config.seed,
        "n_iterations": config.n_iterations,
        "search_strategy": search_strategy.value if hasattr(search_strategy, "value") else str(search_strategy),
        "performance_threshold_m": config.performance_threshold_m,
        "performance_spec_percent": config.performance_spec_percent,
    }

    return CorrectionResult(
        best_parameter_set=best_params,
        best_rms_m=best_rms,
        best_index=best_idx,
        worst_rms_m=worst_rms,
        mean_rms_m=mean_rms,
        n_parameter_sets=n_param_sets,
        n_gcp_pairs=n_gcp_pairs,
        all_parameter_sets=all_sets,
        met_threshold=met_threshold,
        recommendation=recommendation,
        summary_table=summary_table,
        netcdf_path=netcdf_path,
        config_snapshot=config_snapshot,
        elapsed_time_s=elapsed_time_s,
        results=results,
        netcdf_data=netcdf_data,
    )
