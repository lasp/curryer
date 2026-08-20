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
epoch); windows are ``(start, stop)`` bounds as reported by SPICE. Boundary
instants are shared rather than exclusive: a coverage window's stop is also
the start of any adjacent gap, so exactly-touching windows merge and a
single-instant overlap counts as an intersection.

@author: Matthew Maclay
"""

import logging
import typing
import warnings
from pathlib import Path

import spiceypy
from spiceypy.utils.exceptions import SpiceyError

from ..spicierpy import ext
from ..spicierpy.obj import Body, Frame

logger = logging.getLogger(__name__)

# Kernel types that carry time coverage (str-valued enum members).
COVERAGE_KERNEL_TYPES = (ext.KernelType.SPK, ext.KernelType.CK, ext.KernelType.PCK)


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

    target: int | str | Body | Frame
    windows: tuple[tuple[int, int], ...]
    kernels: tuple[str, ...]


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
    window: tuple[int, int]


def _coverage_kernels(kernels: typing.Iterable[str | Path] | None = None) -> list[tuple[str, str]]:
    """Resolve the kernel files to consider as ``(filename, ktype)`` pairs.

    Defaults to the furnished pool, silently scoped to coverage-capable types
    (the pool legitimately holds text kernels). An explicitly given file that
    carries no time coverage is dropped with a warning instead, since passing
    one is most likely a caller mistake.
    """
    if kernels is None:
        return [(rec.file, rec.ktype) for rec in ext.loaded_kernels() if rec.ktype in COVERAGE_KERNEL_TYPES]
    resolved = []
    for filename in kernels:
        filename = str(filename)
        arch, ktype = spiceypy.getfat(filename)
        if arch == "DAF" and ktype in COVERAGE_KERNEL_TYPES:
            resolved.append((filename, ktype))
        else:
            warnings.warn(f"Ignoring kernel without time coverage (arch={arch}, type={ktype}): {filename}")
    return resolved


def _kernel_catalog(kernels: typing.Iterable[str | Path] | None = None) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Resolve kernels once into ``{filename: (ktype, contained_ids)}``.

    Reading each kernel's object directory is file I/O; resolving the catalog
    up front avoids rescanning every kernel for every target.
    """
    return {
        filename: (ktype, tuple(int(code) for code in ext.kernel_objects(filename, as_id=True)))
        for filename, ktype in _coverage_kernels(kernels)
    }


def _target_id(target, ktype):
    """Map `target` into a kernel type's ID space, or None when not mappable.

    Mirrors the per-type coercion in :func:`curryer.spicierpy.ext.kernel_coverage`:
    SPK segments are keyed by body ID, CK by frame ID, and binary PCK by frame
    *class* ID (integers are used as-is — class IDs have no name lookup — while
    names and `Frame`/`Body` targets resolve via
    :func:`curryer.spicierpy.ext.frame_class_id`, which follows TK aliases
    such as ``MOON_PA`` and requires the frame definitions to be loaded).
    A target that cannot be resolved is treated as not mappable rather than an
    error, so scoping stays permissive; the aggregation entry points diagnose
    targets that resolve nowhere.
    """
    try:
        if ktype == "SPK":
            obj = target.body if isinstance(target, Frame) else target if isinstance(target, Body) else Body(target)
            return obj.id
        if ktype == "CK":
            obj = target.frame if isinstance(target, Body) else target if isinstance(target, Frame) else Frame(target)
            # Frame keeps the name when no loaded FK defines its ID.
            return obj.id if isinstance(obj.id, int) else None
        if ktype == "PCK":
            return target if isinstance(target, int) else ext.frame_class_id(target)
    except (SpiceyError, ValueError, TypeError):
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


def _object_coverage(target, catalog):
    """Coverage of one target against a resolved catalog.

    Returns the ObjectCoverage plus whether the target resolved into at least
    one kernel's ID space — needed to tell an unresolvable target apart from
    a resolvable one that no kernel contains.
    """
    windows = []
    contributors = []
    resolved = False
    for filename, (ktype, contained) in catalog.items():
        target_id = _target_id(target, ktype)
        if target_id is None:
            continue
        resolved = True
        if int(target_id) not in contained:
            continue
        flat = ext.kernel_coverage(filename, int(target_id), as_segments=True, to_fmt="ugps")
        windows.extend(zip(flat[0::2], flat[1::2], strict=True))
        contributors.append(filename)
    return ObjectCoverage(target=target, windows=_merge(windows), kernels=tuple(contributors)), resolved


def object_coverage(
    target: int | str | Body | Frame,
    kernels: typing.Iterable[str | Path] | None = None,
) -> ObjectCoverage:
    """Union of coverage windows for one object across kernels.

    Parameters
    ----------
    target : int or str or Body or Frame
        Object to check coverage of. Integer IDs are interpreted within each
        kernel type's ID space (body ID for SPK, frame ID for CK, frame class
        ID for binary PCK); names require their definitions to be loaded.
    kernels : iterable of str or Path, optional
        Kernel files to consider. Default: every currently loaded (furnished)
        kernel of a coverage-capable type.

    Returns
    -------
    ObjectCoverage
        The merged windows and the kernels that contributed. Kernels that do
        not contain the target contribute nothing.

    """
    result, _ = _object_coverage(target, _kernel_catalog(kernels))
    return result


def _resolve_windows(targets, kernels):
    """Intersect per-target coverage; collect targets with no contribution.

    Every target is checked (no early exit on an empty intersection) so the
    diagnostics name all problem targets, not just the first.
    """
    catalog = _kernel_catalog(kernels)
    windows = None
    missing = []
    for target in targets:
        cov, resolved = _object_coverage(target, catalog)
        if not cov.kernels:
            reason = "no kernel contains it" if resolved else "it could not be resolved to an ID"
            missing.append(f"{target} ({reason})")
        windows = cov.windows if windows is None else _intersect(windows, cov.windows)
    return windows if windows is not None else (), missing


def _missing_message(missing):
    return (
        f"Target(s) without coverage in any considered kernel: {'; '.join(missing)}."
        " A mistyped or undefined target is otherwise indistinguishable from missing data."
    )


def valid_window(
    targets: typing.Iterable[int | str | Body | Frame],
    kernels: typing.Iterable[str | Path] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Windows where every target is simultaneously covered.

    The intersection, across targets, of each target's union of coverage —
    i.e., "when is everything I need valid at once." A target that no
    considered kernel covers — or that cannot be resolved to an ID at all —
    triggers a warning, since it would otherwise be indistinguishable from a
    genuine lack of common coverage.

    Parameters
    ----------
    targets : iterable
        Objects that must all be covered (see :func:`object_coverage`).
    kernels : iterable of str or Path, optional
        Kernel files to consider. Default: every currently loaded (furnished)
        kernel of a coverage-capable type.

    Returns
    -------
    tuple of tuple
        Merged, disjoint ``(start, stop)`` uGPS windows; empty when there is
        no common coverage.

    """
    windows, missing = _resolve_windows(targets, kernels)
    if missing:
        warnings.warn(_missing_message(missing))
    return windows


def coverage_gaps(
    targets: typing.Iterable[int | str | Body | Frame],
    start: int,
    stop: int,
    kernels: typing.Iterable[str | Path] | None = None,
    error: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Sub-ranges of a requested window not covered for every target.

    Emits a warning by default when gaps exist — SPICE otherwise fails
    opaquely deep inside a later computation, so surfacing the gap up front
    makes the failure traceable. Raising instead is caller opt-in. A target
    that no considered kernel covers (or that cannot be resolved to an ID)
    is diagnosed separately from an ordinary gap, and raises first when
    `error` is set.

    Parameters
    ----------
    targets : iterable
        Objects that must all be covered (see :func:`object_coverage`).
    start, stop : int
        Requested window in uGPS.
    kernels : iterable of str or Path, optional
        Kernel files to consider. Default: every currently loaded (furnished)
        kernel of a coverage-capable type.
    error : bool, optional
        Raise a ``ValueError`` instead of warning when gaps exist or a target
        has no coverage anywhere. Default=False.

    Returns
    -------
    tuple of tuple
        Uncovered ``(start, stop)`` uGPS sub-ranges; empty when the window is
        fully covered.

    """
    if stop < start:
        raise ValueError(f"Requested window is inverted: stop ({stop}) < start ({start}) (ugps)")
    windows, missing = _resolve_windows(targets, kernels)
    if missing:
        message = _missing_message(missing)
        if error:
            raise ValueError(message)
        warnings.warn(message)
    gaps = _subtract((start, stop), windows)
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


def coverage_rollup(kernels: typing.Iterable[str | Path] | None = None) -> list[KernelObjectCoverage]:
    """Per-kernel, per-object overall coverage windows (diagnostic).

    Parameters
    ----------
    kernels : iterable of str or Path, optional
        Kernel files to consider. Default: every currently loaded (furnished)
        kernel of a coverage-capable type.

    Returns
    -------
    list of KernelObjectCoverage
        One record per object per kernel.

    """
    records = []
    for filename, (ktype, contained) in _kernel_catalog(kernels).items():
        for object_id in contained:
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


def pairwise_overlap(
    kernel_a: str | Path,
    kernel_b: str | Path,
    target: int | str | Body | Frame,
) -> tuple[tuple[int, int], ...]:
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
