"""Create a SPICE kernel.

Examples
--------
% ipython3 scripts/create_kernel.py -- kernels/spk/iss_sc_v01.ephemeris.spk.json \
    -t "2018-01-17T23:00:00" "2018-01-23T01:00:00"

% ipython35 scripts/create_kernel.py -- kernels/ck/iss_sc_v01.attitude.ck.json \
    -t "2018-01-17T23:00:00" "2018-01-23T01:00:00"

% ipython3 scripts/create_kernel.py -- kernels/ck/tsis_azel_v01.attitude.ck.json \
    -t "2018-01-17T23:00:00" "2018-01-23T01:00:00"


% ipython3 scripts/create_kernel.py -- kernels/spk/iss_sc_v01.ephemeris.spk.json \
    kernels/ck/iss_sc_v01.attitude.ck.json kernels/ck/tsis_azel_v01.attitude.ck.json \
    -t "2018-01-26T23:00:00" "2018-02-03T01:00:00" --append


Created on Aug 18, 2017

@author: Brandon Stone
"""

import argparse
import logging

from curryer.kernels import create
from curryer.utils import enable_logging


def cmd_line_call():
    """Method to process command line arguments (`python <file>.py --help`)."""
    parser = argparse.ArgumentParser(description="Create a SPICE kernel.")
    parser.add_argument(
        "kernel_configs", type=str, nargs="+", help="One or more kernel configuration properties files (json)."
    )
    parser.add_argument(
        "-o",
        "--output_kernels",
        type=str,
        nargs="+",
        help="Output kernels. Default is `kernel_configs` with the proper ext.",
    )
    parser.add_argument(
        "-t",
        "--time_range",
        type=str,
        nargs=2,
        default=argparse.SUPPRESS,
        help="Time range (inclusive, exclusive; default all available data).",
    )
    parser.add_argument("-f", "--time_format", type=str, default="utc", help="Time format (default=utc iso strings).")
    parser.add_argument(
        "-bh",
        "--buffer_hours",
        type=float,
        nargs=2,
        default=argparse.SUPPRESS,
        help="Option to buffer time_range, in hours (default=[0, 0]).",
    )
    parser.add_argument(
        "-ld",
        "--lag_days",
        type=int,
        default=argparse.SUPPRESS,
        help="Define the `time_range` to be a 24-hour window starting `lag_days`"
        " days prior to the start of the current day (UTC).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Option to overwrite an existing kernel (default=False).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Option to append to an existing kernel (default=False).",
    )
    parser.add_argument(
        "-l", "--log_dir", type=str, default=argparse.SUPPRESS, help="Directory to save logging output in."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const="debug",
        dest="log_level",
        default=argparse.SUPPRESS,
        help='Set log reporting level to "debug" (default="info").',
    )
    kwargs = vars(parser.parse_args())
    orig_kwargs = str(kwargs)

    # Start logging to the console (stdout).
    log_level = kwargs.pop("log_level", "info")
    logger = enable_logging(log_level=log_level)

    # TODO: Improve!
    # # Start logging to a file.
    # log_dir = kwargs.pop('log_dir', '.')
    # log_file = os.path.join(log_dir, 'create_kernel_{}.log'.format(
    #     datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    # ))
    # enable_logging.init(log_level='DEBUG', log_file=log_file)

    # Run the main method...
    logger.debug("Supplied arguments: %s", orig_kwargs)
    try:
        return create.batch_kernels(**kwargs)
    except Exception:
        logging.exception("An exception occurred:")
        parser.exit(status=1, message="Script failed with errors! Exiting early...\n")


if __name__ == "__main__":
    cmd_line_call()
