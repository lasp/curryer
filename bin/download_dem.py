#!/usr/bin/env python3
"""Download DEMs from the USGS.

Examples
--------
Global 15 arc-min mean DEMs, except the South Pole where only 30 arc-min is
available (-56 latitude). Total size about 5.8G.
% ./download_dem.py -d . -r 15 -s mean --lat_min -50
% ./download_dem.py -d . -r 30 -s mean --lat_max -50

See Also
--------
https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-multi-resolution-terrain-elevation
https://pubs.usgs.gov/of/2011/1073/pdf/of2011-1073.pdf

@author: Brandon Stone

"""

import argparse
import time
from pathlib import Path

import requests

GMTED_BASE_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Global_tiles_GMTED/"
    "{res:03d}darcsec/{stat}/{lon1}/{lat}{lon2}_20101117_gmted_{stat}{res:03d}.tif"
)
GMTED_STATS = {
    "mean": "mea",
    "max": "max",
    "min": "min",
}
GMTED_RESOLUTIONS = {
    30: 300,
    15: 150,
    7.5: 75,
}
GMTED_LON_STEP = 30
GMTED_LAT_STEP = 20


def main(
    directory: Path = None,
    stat: str = "mean",
    arcsec: float = 15,
    overwrite=False,
    sleep=1.0,
    lon_min=-180,
    lon_max=180,
    lat_min=-90,
    lat_max=90,
):
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory).resolve()

    lon_min += (-180 - lon_min) % GMTED_LON_STEP
    lon_max += (-180 - lon_max) % GMTED_LON_STEP
    lat_min += (-90 - lat_min) % GMTED_LAT_STEP
    lat_max += (-90 - lat_max) % GMTED_LAT_STEP

    print(
        f"Starting download of stat=[{stat}], arcsec=[{arcsec}], lon=[{lon_min}, {lon_max}],"
        f" lat=[{lat_min}, {lat_max}], directory=[{directory}]"
    )
    files = []
    for lat in range(lat_min, lat_max, GMTED_LAT_STEP):
        for lon in range(lon_min, lon_max, GMTED_LON_STEP):
            url = GMTED_BASE_URL.format(
                res=GMTED_RESOLUTIONS[arcsec],  # Ex: "150" for 15-arcsec.
                stat=GMTED_STATS[stat],  # Ex: "mea" for mean.
                lon1=f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}",  # Ex: "W090" in URL path.
                lon2=f"{abs(lon):03d}{'E' if lon >= 0 else 'W'}",  # Ex: "090W" in file name.
                lat=f"{abs(lat):02d}{'N' if lat >= 0 else 'S'}",  # Ex: "30N" in file name.
            )
            name = url.rsplit("/", 1)[-1]
            filepath = directory / name

            if filepath.is_file() and not overwrite:
                print(f"Skipping existing file: {name}")
                continue

            print(f"Downloading: {url}")
            with requests.get(url, stream=True, timeout=30) as resp:
                if resp.status_code == 404 and lat == -90 and arcsec != 30:
                    print(f"Skipping [{lon}, {lat}] deg lon/lat (south pole only has 30 arc-sec data)!")
                    continue

                resp.raise_for_status()
                filepath.write_bytes(resp.content)

            print(f"Saved: {name}")
            files.append(filepath)
            time.sleep(sleep)  # Be nice to the server.

    print(
        f"Completed download of [{len(files)}] files for stat=[{stat}], arcsec=[{arcsec}],"
        f" lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}], directory=[{directory}]"
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DEM GeoTIFFs from the USGS.")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=Path.cwd(),
        help="Directory to save the files (default=current working directory).",
    )
    parser.add_argument(
        "-s",
        "--stat",
        type=str,
        default=next(iter(GMTED_STATS)),
        choices=list(GMTED_STATS),
        help="DEM statistic type (default=mean).",
    )
    parser.add_argument(
        "-r",
        "--arcsec",
        type=float,
        default=15,
        choices=list(GMTED_RESOLUTIONS),
        help="Resolution in arc-seconds (default=15).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Overwrite any existing files (default=False; skip existing).",
    )
    for ll in ["lon", "lat"]:
        for mm in ["min", "max"]:
            parser.add_argument(
                f"--{ll}_{mm}",
                type=int,
                default=argparse.SUPPRESS,
                help=f"Limit download of tiles to {mm.upper()} {ll.upper()} lower-left corner"
                " (default=global coverage).",
            )
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Number of seconds to sleep between downloads (default=1)."
    )
    kwargs = vars(parser.parse_args())
    main(**kwargs)
