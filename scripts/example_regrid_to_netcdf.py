#!/usr/bin/env python
"""
Example: Regrid GCP chip and save to NetCDF

This script demonstrates the new NetCDF output functionality for GCP regridding.
It can be used as a template for batch processing GCP chips.
"""

from pathlib import Path

import numpy as np

from curryer.correction.image_io import load_gcp_chip_from_hdf
from curryer.correction.regrid import RegridConfig, regrid_gcp_chip


def regrid_and_save_gcp_chip(
    input_file: Path,
    output_file: Path,
    mission: str = "CLARREO Pathfinder",
    resolution_deg: tuple[float, float] = (0.0009, 0.0009),
) -> None:
    """
    Load, regrid, and save a GCP chip to NetCDF.

    Parameters
    ----------
    input_file : Path
        Input HDF file containing raw GCP chip.
    output_file : Path
        Output NetCDF file path.
    mission : str
        Mission name for metadata.
    resolution_deg : tuple[float, float]
        Output resolution (dlat, dlon) in degrees.
    """
    print(f"Processing: {input_file.name}")

    # Load raw GCP chip
    print("  Loading raw chip...")
    band, x, y, z = load_gcp_chip_from_hdf(input_file)
    print(f"    Input shape: {band.shape}")

    # Configure regridding
    config = RegridConfig(
        output_resolution_deg=resolution_deg,
        conservative_bounds=True,
        interpolation_method="bilinear",
    )

    # Prepare metadata
    metadata = {
        "source_file": input_file.name,
        "mission": mission,
        "sensor": "Landsat-8",
        "band": "red (Band 1)",
        "resolution_deg": f"{resolution_deg[0]}°, {resolution_deg[1]}°",
        "processing_software": "curryer",
    }

    # Regrid and save
    print("  Regridding...")
    regridded = regrid_gcp_chip(
        band,
        (x, y, z),
        config,
        output_file=str(output_file),
        output_metadata=metadata,
    )

    print(f"    Output shape: {regridded.data.shape}")
    print(f"    Valid pixels: {(~np.isnan(regridded.data)).sum()}/{regridded.data.size}")
    print(f"  Saved to: {output_file}")
    print(f"    File size: {output_file.stat().st_size / (1024**2):.2f} MB")


def main():
    """Example usage: Process a single GCP chip."""
    # Example file paths
    input_file = Path("tests/data/clarreo/landsat_gcp/LT08CHP.20140803.p002r071.c01.v001.hdf")
    output_dir = Path("output/regridded_gcps")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "LT08CHP.20140803.p002r071.c01.v001_regridded.nc"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("\nUpdate the script with your GCP chip path.")
        return

    # Process with CLARREO-specific resolution (~100m)
    regrid_and_save_gcp_chip(
        input_file,
        output_file,
        mission="CLARREO Pathfinder",
        resolution_deg=(0.0009, 0.0009),
    )

    print("\n✓ Processing complete!")


def batch_process_example():
    """Example: Batch process multiple GCP chips."""
    input_dir = Path("tests/data/clarreo/landsat_gcp")
    output_dir = Path("output/regridded_gcps")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all HDF files
    hdf_files = list(input_dir.glob("LT08CHP.*.hdf"))

    if not hdf_files:
        print(f"No HDF files found in {input_dir}")
        return

    print(f"Found {len(hdf_files)} GCP chips to process\n")

    for input_file in hdf_files:
        output_file = output_dir / f"{input_file.stem}_regridded.nc"

        try:
            regrid_and_save_gcp_chip(input_file, output_file)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print()

    print(f"✓ Batch processing complete! Processed {len(hdf_files)} chips.")


if __name__ == "__main__":
    # Run single file example
    main()

    # Uncomment to run batch processing
    # batch_process_example()
