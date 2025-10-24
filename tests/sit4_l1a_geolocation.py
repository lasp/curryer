"""SIT-4 L1A Geolocation Integration Test."""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from curryer import meta, spicetime, utils
from curryer import spicierpy as sp
from curryer.compute import spatial
from curryer.kernels import create

logger = logging.getLogger("curryer.script")


def main(meta_kernel, generic_dir, kernel_configs, output_dir, input_file_or_obj, time_range=None):
    generic_dir = Path(generic_dir)
    if not generic_dir.is_dir():
        raise NotADirectoryError(generic_dir)

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    # Load meta kernel details. Includes existing static kernels.
    mkrn = meta.MetaKernel.from_json(meta_kernel, relative=True, sds_dir=generic_dir)

    # Create the dynamic kernels from the JSONs alone. Note that they
    # contain the reference to the input_data netcdf4 file to read.
    generated_kernels = []
    creator = create.KernelCreator(overwrite=False, append=False)

    # Generate the kernels from the config and input data (file or object).
    with tempfile.TemporaryDirectory() as tmp_dir:
        for kernel_config in kernel_configs:
            generated_kernels.append(
                creator.write_from_json(
                    kernel_config,
                    output_kernel=tmp_dir,
                    input_data=input_file_or_obj,
                )
            )

        # Geolocate all the individual pixels and create the L1A data product!
        with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, generated_kernels]):
            ugps_times = spicetime.adapt(pd.date_range(*time_range, freq="67ms", inclusive="left"), "iso")
            geoloc_inst = spatial.Geolocate("CPRS_HYSICS")
            l1a_dataset = geoloc_inst(ugps_times)

    created = pd.Timestamp(time_range[0]).strftime("%Y%m%dT%H%M%S")
    out_file = output_dir / f"cprs_geolocation_l1a_{created}.nc"
    l1a_dataset.to_netcdf(out_file)
    logger.info("Script saved L1A geolocation:\n%s", out_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate L1A geolocation data product.")
    parser.add_argument("--meta_kernel", type=str, help="Meta kernel json file for the mission.")
    parser.add_argument("--generic_dir", type=str, help="Directory containing the generic SPICE kernels.")
    parser.add_argument(
        "--kernel_configs", type=str, nargs="+", help="One or more kernel configuration properties files (json)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Path.cwd(),
        help="Directory to output data products to (default is current working directory).",
    )
    parser.add_argument("--input_file_or_obj", type=str, help="Input telemetry file to create kernels from.")
    parser.add_argument(
        "-t",
        "--time_range",
        type=str,
        nargs=2,
        default=argparse.SUPPRESS,
        help="Time range (inclusive, exclusive; default start/stop of input telemetry).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        dest="log_level",
        default=argparse.SUPPRESS,
        help='Set log reporting level to "debug" (default="info").',
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Ignore all keywords but `output_dir`, and instead run with Clarreo demo values.",
    )
    kwargs = vars(parser.parse_args())
    orig_kwargs = str(kwargs)

    # Start logging to the console (stdout).
    log_level = kwargs.pop("log_level", logging.INFO)
    utils.enable_logging(log_level=log_level, extra_loggers=[__name__])
    xr.set_options(display_width=120, display_max_rows=30)
    np.set_printoptions(linewidth=120)

    # Execute the logic!!!
    logger.debug("Supplied arguments: %s", orig_kwargs)
    if "demo" in kwargs or kwargs.pop("demo", False):
        main(
            meta_kernel="tests/data/clarreo/cprs_v01.kernels.tm.json",
            generic_dir="data/generic",
            kernel_configs=[
                "data/clarreo/iss_sc_v01.ephemeris.spk.json",
                "data/clarreo/iss_sc_v01.attitude.ck.json",
                "data/clarreo/cprs_az_v01.attitude.ck.json",
                "data/clarreo/cprs_el_v01.attitude.ck.json",
            ],
            output_dir=kwargs.get("output_dir", "/tmp"),
            input_file_or_obj="tests/data/demo/cprs_geolocation_tlm_20230101_20240430.nc",
            time_range=("2023-01-01", "2023-01-01T00:05:00"),
        )
    else:
        main(**kwargs)
