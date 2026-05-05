#!/usr/bin/env python
"""Batch-regrid Landsat (or other HDF) GCP chips to regular lat/lon NetCDF files.

Usage
-----
Single file::

    python examples/correction/regrid_gcp_chips.py input.hdf output_dir/

Directory of HDF files::

    python examples/correction/regrid_gcp_chips.py /data/landsat_gcps/ /data/regridded/

Custom resolution and glob pattern::

    python examples/correction/regrid_gcp_chips.py /data/ /out/ \\
        --pattern "LT08CHP.*.hdf" \\
        --resolution 0.001 0.001

Resume an interrupted run (skip already-converted files)::

    python examples/correction/regrid_gcp_chips.py /data/ /out/ --skip-existing

Preview what would be processed without writing any files::

    python examples/correction/regrid_gcp_chips.py /data/ /out/ --dry-run

See all options::

    python examples/correction/regrid_gcp_chips.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core per-file processing
# ---------------------------------------------------------------------------


def regrid_one(
    input_file: Path,
    output_file: Path,
    resolution_deg: tuple[float, float],
    conservative_bounds: bool,
    mission: str,
    band_name: str,
) -> None:
    """Load one HDF chip, regrid it, and write the result to a NetCDF file.

    Parameters
    ----------
    input_file : Path
        Source HDF4 or HDF5 chip file.
    output_file : Path
        Destination NetCDF file.  Parent directory must already exist.
    resolution_deg : tuple[float, float]
        Output grid resolution as ``(dlat, dlon)`` in degrees.
    conservative_bounds : bool
        Shrink output bounds to the interior of the input grid when ``True``
        (avoids edge extrapolation artefacts).
    mission : str
        Mission label written to the ``mission`` global attribute.
    band_name : str
        HDF dataset name for the radiometric band (default ``"Band_1"``).
    """
    from curryer.correction.data_structures import RegridConfig
    from curryer.correction.image_io import load_gcp_chip_from_hdf
    from curryer.correction.regrid import regrid_gcp_chip

    band, ecef_x, ecef_y, ecef_z = load_gcp_chip_from_hdf(input_file, band_name=band_name)

    config = RegridConfig(
        output_resolution_deg=resolution_deg,
        conservative_bounds=conservative_bounds,
        interpolation_method="bilinear",
    )

    metadata = {
        "source_file": input_file.name,
        "mission": mission,
        "band": band_name,
        "resolution_deg": f"{resolution_deg[0]}, {resolution_deg[1]}",
        "processing_software": "curryer",
    }

    regrid_gcp_chip(
        band,
        (ecef_x, ecef_y, ecef_z),
        config,
        output_file=str(output_file),
        output_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def collect_inputs(source: Path, pattern: str) -> list[Path]:
    """Return a sorted list of HDF files to process.

    Parameters
    ----------
    source : Path
        Either a single HDF file or a directory to search.
    pattern : str
        Glob pattern applied when *source* is a directory.
    """
    if source.is_file():
        return [source]
    files = sorted(source.glob(pattern))
    if not files:
        logger.warning("No files matched pattern %r in %s", pattern, source)
    return files


def output_path_for(input_file: Path, output_dir: Path) -> Path:
    """Derive the output NetCDF path from an input HDF filename."""
    return output_dir / f"{input_file.stem}_regridded.nc"


def run_batch(
    input_files: list[Path],
    output_dir: Path,
    resolution_deg: tuple[float, float],
    conservative_bounds: bool,
    skip_existing: bool,
    dry_run: bool,
    mission: str,
    band_name: str,
) -> int:
    """Process *input_files* and return the number of failures."""
    total = len(input_files)
    failures: list[tuple[Path, str]] = []

    print(f"{'DRY RUN — ' if dry_run else ''}Processing {total} file(s) → {output_dir}\n")

    for idx, input_file in enumerate(input_files, start=1):
        output_file = output_path_for(input_file, output_dir)
        prefix = f"[{idx:>{len(str(total))}}/{total}]"

        if skip_existing and output_file.exists():
            print(f"{prefix} SKIP  {input_file.name}  (already exists)")
            continue

        print(f"{prefix} START {input_file.name}")

        if dry_run:
            print(f"        → {output_file}")
            continue

        t0 = time.monotonic()
        try:
            regrid_one(input_file, output_file, resolution_deg, conservative_bounds, mission, band_name)
            elapsed = time.monotonic() - t0
            size_kb = output_file.stat().st_size / 1024
            print(f"        ✓ {output_file.name}  ({size_kb:.0f} KB, {elapsed:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            msg = str(exc)
            failures.append((input_file, msg))
            print(f"        ✗ FAILED ({elapsed:.1f}s): {msg}")
            logger.debug("Traceback for %s:", input_file.name, exc_info=True)

    # Summary
    print(f"\n{'─' * 60}")
    if dry_run:
        print(f"Dry run complete — {total} file(s) would be processed.")
    elif failures:
        print(f"Finished: {total - len(failures)}/{total} succeeded, {len(failures)} failed.")
        print("\nFailed files:")
        for path, reason in failures:
            print(f"  {path.name}: {reason}")
    else:
        print(f"Finished: all {total} file(s) processed successfully.")

    return len(failures)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regrid_gcp_chips",
        description="Batch-regrid HDF GCP chips to regular lat/lon NetCDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Single HDF file or directory containing HDF files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where regridded NetCDF files are written (created if needed).",
    )
    parser.add_argument(
        "--pattern",
        default="*.hdf",
        metavar="GLOB",
        help="Glob pattern for HDF files when source is a directory (default: '*.hdf').",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=float,
        default=[0.0009, 0.0009],
        metavar=("DLAT", "DLON"),
        help="Output grid resolution in degrees (default: 0.0009 0.0009 ≈ 100 m).",
    )
    parser.add_argument(
        "--no-conservative-bounds",
        dest="conservative_bounds",
        action="store_false",
        default=True,
        help="Use full ECEF extent instead of shrinking bounds to avoid edge extrapolation.",
    )
    parser.add_argument(
        "--mission",
        default="CLARREO Pathfinder",
        help="Mission name written to NetCDF metadata (default: 'CLARREO Pathfinder').",
    )
    parser.add_argument(
        "--band",
        default="Band_1",
        metavar="DATASET",
        help="HDF dataset name for the radiometric band (default: 'Band_1').",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output NetCDF already exists (useful for resuming).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (shows per-row regridding progress).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    source: Path = args.source
    output_dir: Path = args.output_dir
    resolution_deg = (args.resolution[0], args.resolution[1])

    if not source.exists():
        parser.error(f"source does not exist: {source}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    input_files = collect_inputs(source, args.pattern)
    if not input_files:
        print(f"No files found. Check --pattern (currently {args.pattern!r}).")
        sys.exit(1)

    n_failures = run_batch(
        input_files=input_files,
        output_dir=output_dir,
        resolution_deg=resolution_deg,
        conservative_bounds=args.conservative_bounds,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        mission=args.mission,
        band_name=args.band,
    )

    sys.exit(1 if n_failures else 0)


if __name__ == "__main__":
    main()
