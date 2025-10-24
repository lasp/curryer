"""Abstract mission data class.

@author: Brandon Stone
"""

import functools
import logging
from abc import ABCMeta, abstractmethod
from io import StringIO

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def log_return(max_rows=5):
    """Log any output tables/arrays from a function."""

    def log_return_wrapper(func):
        """Store the function to log output from."""
        _logger = logging.getLogger(getattr(func, "__module__", __name__))

        @functools.wraps(func)
        def log_return_call(*args, **kwargs):
            """Call the wrapper function and log the output."""
            output = func(*args, **kwargs)

            if _logger.isEnabledFor(logging.DEBUG):
                output_name = output.__class__.__name__

                # Table.
                if isinstance(output, pd.DataFrame):
                    if pd.get_option("display.width") == 80:
                        pd.set_option("display.width", 140)
                    info_strm = StringIO()
                    _logger.debug("%s%s:\n %s", output_name, output.shape, output.to_string(max_rows=max_rows))
                    output.info(buf=info_strm)
                    _logger.debug("%s details:\n%s", output_name, info_strm.getvalue())

                # Numpy array.
                elif isinstance(output, np.ndarray):
                    if np.get_printoptions()["threshold"] == 1000:
                        np.set_printoptions(threshold=100)
                    _logger.debug("%s%s:\n %s", output_name, output.shape, output)  # Auto trims.

            return output

        return log_return_call

    return log_return_wrapper


class AbstractMissionData(metaclass=ABCMeta):
    """Abstract class to get/write mission data."""

    @abstractmethod
    def __init__(self, microsecond_cadence, *args, **kwargs):
        self.microsecond_cadence = microsecond_cadence
        self._loaded_kernels = None
        self.allow_nans = True

    def __del__(self):
        if not self._loaded_kernels:
            return
        try:
            self._loaded_kernels.unload(clear=True)
        except:  # noqa E722
            logging.exception("Error while unloading kernels:\n")

    @log_return()
    def get_times(self, ugps_range, cadence=None):
        """Array of evenly spaced values.

        Parameters
        ----------
        ugps_range : (int, int)
            Time range [start, end).
        cadence : int, optional
            Number of microseconds to step by. Default=`self.DEFAULT_CADENCE`

        Returns
        -------
        numpy.ndarray
            Array of time values (int64).

        """
        if cadence is None:
            cadence = self.microsecond_cadence
        if len(ugps_range) != 2:
            raise ValueError("Must specify two uGPS times: inclusive start, exclusive end.")

        logger.debug("Determining times in range [%s] with cadence [%s]", ugps_range, cadence)
        return np.arange(ugps_range[0], ugps_range[1], cadence)


def write_to_database(table, session, dbtable, auto_commit):
    """Insert table values using the SQLAlchemy bulk insert method.

    Parameters
    ----------
    table : DataFrame
    session : Session
    dbtable : Table
    auto_commit : bool

    Returns
    -------
    int

    """
    logger.info("Inserting [%s] rows into [%s] @ [%s]", table.shape[0], dbtable.__table__.fullname, session)

    if table.shape[0] == 0:
        logger.warning("Skipping insert for empty dataset [%s]!", dbtable.__table__.fullname)
        return 0

    query = session.query(
        dbtable,
    ).filter(
        dbtable.ugps >= int(table.index.min()),
        dbtable.ugps <= int(table.index.max()),
        *[getattr(dbtable, col) == table[col].iloc[0] for col in table.columns if col != "ugps"],
        # dbtable.instrumentmodeid == int(table['instrumentmodeid'].iloc[0]),
        # dbtable.version == int(table['version'].iloc[0])
    )
    if query.count() != 0:
        raise ValueError("Data already exists!")

    # Bulk insert a list of dicts (rows).
    #   Convert DataFrame to a Numpy records array to maintain
    #   the individual column data types. Convert to a list to change
    #   from Numpy scalar types to Python scalar types (e.g., int).
    # TODO: Add an iter to flush every (e.g., 1440 rows / day's worth)?
    col_names = (table.index.name, *table.columns)
    session.bulk_insert_mappings(dbtable, (dict(zip(col_names, row)) for row in table.to_records().tolist()))

    if query.count() != table.shape[0]:
        raise ValueError("Database is missing some data.")
    if auto_commit:
        session.commit()
    return table.shape[0]
