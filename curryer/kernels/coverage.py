"""Scoped, run-level SPICE kernel coverage aggregation.

Answers the coverage questions a processing run asks of the furnished kernel
pool (or an explicit set of kernel files): the window where every required
object is simultaneously covered (:func:`valid_window`), the sub-ranges of a
requested window that are not covered (:func:`coverage_gaps`), a per-kernel /
per-object roll-up (:func:`coverage_rollup`), and the overlap between two
kernels for one object (:func:`pairwise_overlap`).

Queries are scoped to the requested target objects: kernels that do not
contain a target contribute nothing, and unrelated furnished kernels never
affect a result. All window times are uGPS (int64 microseconds since the GPS
epoch); windows are closed ``(start, stop)`` intervals.

@author: Matthew Maclay
"""

import logging
import typing
import warnings

import spiceypy
from spiceypy.utils.exceptions import SpiceyError

from ..spicierpy import ext
from ..spicierpy.obj import Body, Frame

logger = logging.getLogger(__name__)

# Kernel types that carry time coverage.
COVERAGE_KERNEL_TYPES = ("SPK", "CK", "PCK")


class ObjectCoverage(typing.NamedTuple):
    """Union of coverage windows for one target object.

    Attributes
    ----------
    target : int or str or Body or Frame
        The requested object, as given.
    windows : tuple of tuple
        Merged, sorted, disjoint ``(start, stop)`` uGPS windows.
    kernels : tuple of str
        Kernel files that contributed coverage.
    """

    target: typing.Any
    windows: tuple
    kernels: tuple


class KernelObjectCoverage(typing.NamedTuple):
    """Overall coverage window of one object within one kernel file.

    Attributes
    ----------
    file : str
        Kernel file path.
    ktype : str
        SPICE kernel type (``"SPK"``, ``"CK"``, or ``"PCK"``).
    object_id : int
        NAIF object ID within the kernel's ID space (body ID for SPK, frame ID
        for CK, frame class ID for PCK).
    window : tuple
        Overall ``(start, stop)`` uGPS window.
    """

    file: str
    ktype: str
    object_id: int
    window: tuple


def _coverage_kernels(kernels=None):
    """Resolve the kernel files to consider as ``(filename, ktype)`` pairs.

    Defaults to the furnished pool; an explicit list is filtered to binary
    kernel types that carry time coverage.
    """
    if kernels is None:
        return [(rec.file, rec.ktype) for rec in ext.loaded_kernels() if rec.ktype in COVERAGE_KERNEL_TYPES]
    resolved = []
    for filename in kernels:
        filename = str(filename)
        arch, ktype = spiceypy.getfat(filename)
        if arch == "DAF" and ktype in COVERAGE_KERNEL_TYPES:
            resolved.append((filename, ktype))
    return resolved


def _target_id(target, ktype):
    """Map `target` into a kernel type's ID space, or None when not mappable.

    Mirrors the per-type coercion in :func:`curryer.spicierpy.ext.kernel_coverage`:
    SPK segments are keyed by body ID, CK by frame ID, and binary PCK by frame
    *class* ID (integer only — class IDs have no name lookup). A target whose
    name cannot be resolved (e.g., its definitions are not loaded) is treated
    as not mappable rather than an error, so scoping stays permissive.
    """
    try:
        if ktype == "SPK":
            obj = target.body if isinstance(target, Frame) else target if isinstance(target, Body) else Body(target)
            return obj.id
        if ktype == "CK":
            obj = target.frame if isinstance(target, Body) else target if isinstance(target, Frame) else Frame(target)
            return obj.id
        if ktype == "PCK":
            if isinstance(target, Body | Frame):
                return target.id
            return target if isinstance(target, int) else None
    except (SpiceyError, ValueError):
        return None
    return None


def _merge(windows):
    """Merge ``(start, stop)`` windows into a sorted, disjoint tuple."""
    merged = []
    for start, stop in sorted((int(start), int(stop)) for start, stop in windows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return tuple((start, stop) for start, stop in merged)


def _intersect(left, right):
    """Intersection of two merged (sorted, disjoint) window tuples."""
    result = []
    li = ri = 0
    while li < len(left) and ri < len(right):
        start = max(left[li][0], right[ri][0])
        stop = min(left[li][1], right[ri][1])
        if start <= stop:
            result.append((start, stop))
        if left[li][1] < right[ri][1]:
            li += 1
        else:
            ri += 1
    return tuple(result)


def _subtract(window, covered):
    """Sub-ranges of one ``(start, stop)`` window not in a merged window tuple."""
    start, stop = int(window[0]), int(window[1])
    gaps = []
    cursor = start
    for cstart, cstop in covered:
        if cstop < cursor:
            continue
        if cstart > stop:
            break
        if cstart > cursor:
            gaps.append((cursor, min(cstart, stop)))
        cursor = max(cursor, cstop)
        if cursor >= stop:
            break
    if cursor < stop:
        gaps.append((cursor, stop))
    return tuple(gap for gap in gaps if gap[0] < gap[1])


def object_coverage(target, kernels=None):
    """Union of coverage windows for one object across kernels.

    Parameters
    ----------
    target : int or str or Body or Frame
        Object to check coverage of. Integer IDs are interpreted within each
        kernel type's ID space (body ID for SPK, frame ID for CK, frame class
        ID for binary PCK); names require their definitions to be loaded.
    kernels : iterable of str or Path, optional
        Kernel files to consider. Default: every furnished kernel of a
        coverage-capable type.

    Returns
    -------
    ObjectCoverage
        The merged windows and the kernels that contributed. Kernels that do
        not contain the target contribute nothing.

    """
    windows = []
    contributors = []
    for filename, ktype in _coverage_kernels(kernels):
        target_id = _target_id(target, ktype)
        if target_id is None or target_id not in ext.kernel_objects(filename, as_id=True):
            continue
        flat = ext.kernel_coverage(filename, target_id, as_segments=True, to_fmt="ugps")
        windows.extend(zip(flat[0::2], flat[1::2], strict=True))
        contributors.append(filename)
    return ObjectCoverage(target=target, windows=_merge(windows), kernels=tuple(contributors))


def valid_window(targets, kernels=None):
    """Windows where every target is simultaneously covered.

    The intersection, across targets, of each target's union of coverage —
    i.e., "when is everything I need valid at once."

    Parameters
    ----------
    targets : iterable
        Objects that must all be covered (see :func:`object_coverage`).
    kernels : iterable of str or Path, optional
        Kernel files to consider. Default: the furnished pool.

    Returns
    -------
    tuple of tuple
        Merged, disjoint ``(start, stop)`` uGPS windows; empty when there is
        no common coverage.

    """
    windows = None
    for target in targets:
        target_windows = object_coverage(target, kernels=kernels).windows
        windows = target_windows if windows is None else _intersect(windows, target_windows)
        if not windows:
            return ()
    return windows if windows is not None else ()


def coverage_gaps(targets, start, stop, kernels=None, error=False):
    """Sub-ranges of a requested window not covered for every target.

    Emits a warning by default when gaps exist — SPICE otherwise fails
    opaquely deep inside a later computation, so surfacing the gap up front
    makes the failure traceable. Raising instead is caller opt-in.

    Parameters
    ----------
    targets : iterable
        Objects that must all be covered (see :func:`object_coverage`).
    start, stop : int
        Requested window in uGPS.
    kernels : iterable of str or Path, optional
        Kernel files to consider. Default: the furnished pool.
    error : bool, optional
        Raise a ``ValueError`` instead of warning when gaps exist.
        Default=False.

    Returns
    -------
    tuple of tuple
        Uncovered ``(start, stop)`` uGPS sub-ranges; empty when the window is
        fully covered.

    """
    gaps = _subtract((start, stop), valid_window(targets, kernels=kernels))
    if gaps:
        message = (
            f"Requested window [{start}, {stop}] (ugps) is not fully covered for targets"
            f" {[str(target) for target in targets]}: {len(gaps)} uncovered sub-range(s),"
            f" first gap [{gaps[0][0]}, {gaps[0][1]}]."
        )
        if error:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    return gaps


def coverage_rollup(kernels=None):
    """Per-kernel, per-object overall coverage windows (diagnostic).

    Parameters
    ----------
    kernels : iterable of str or Path, optional
        Kernel files to consider. Default: the furnished pool.

    Returns
    -------
    list of KernelObjectCoverage
        One record per object per kernel.

    """
    records = []
    for filename, ktype in _coverage_kernels(kernels):
        for object_id in ext.kernel_objects(filename, as_id=True):
            window = ext.kernel_coverage(filename, int(object_id), to_fmt="ugps")
            records.append(
                KernelObjectCoverage(
                    file=filename,
                    ktype=ktype,
                    object_id=int(object_id),
                    window=(int(window[0]), int(window[-1])),
                )
            )
    return records


def pairwise_overlap(kernel_a, kernel_b, target):
    """Overlap windows of two kernels for the same target (diagnostic).

    Parameters
    ----------
    kernel_a, kernel_b : str or Path
        Kernel files to compare.
    target : int or str or Body or Frame
        Object to compare coverage of (see :func:`object_coverage`).

    Returns
    -------
    tuple of tuple
        ``(start, stop)`` uGPS windows covered by both kernels; empty when
        they do not overlap for the target.

    """
    windows_a = object_coverage(target, kernels=[kernel_a]).windows
    windows_b = object_coverage(target, kernels=[kernel_b]).windows
    return _intersect(windows_a, windows_b)
