"""TLE related logic.

@author: Brandon Stone
"""

import logging
import time
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# TODO: Create an accessor to parse a (local) TLE file?


class TLERemoteAccessor:
    """Access TLE data from SpaceTrack.org."""

    URL_BASE = "https://www.space-track.org"
    URL_AUTH = f"{URL_BASE}/ajaxauth/login"
    MAIN_COLUMNS = ("object_name", "norad_cat_id", "epoch", "creation_date", "file", "tle_line1", "tle_line2")

    __prev_request_time = 0

    def __init__(self, user, pswd, keep_duplicates=False):
        """Construct a TLE accessor.

        Parameters
        ----------
        user : str
            Space-Track.org username.
        pswd : str
            Space-Track.org password.
        keep_duplicates : bool, optional
            Keep duplicate TLE entries.

        """
        self.__spacetrack_user = user
        self.__spacetrack_pswd = pswd
        self.keep_duplicates = keep_duplicates
        self.__prev_request_time = time.time()
        self.__cookies = None

    def limiter(self):
        """Check API limits and sleep if necessary.

        Must self limit API requests or risk server-side errors (to limit).
        Limits: 30 per minute, 300 per hour.

        Returns
        -------
        float
            Number of seconds that it paused for (0=not API limited).

        """
        wait = 0
        time_since = time.time() - self.__prev_request_time
        if time_since < 13:
            wait = 13 - time_since
            logger.debug("%s sleeping for [%.3f] seconds due to API request limits...", self, wait)
            time.sleep(wait)
        self.__prev_request_time = time.time()
        return wait

    def authenticate(self):
        """Authenticate, storing cookie for later queries."""
        logger.debug("Requesting authentication for [%s] from: %s", self.__spacetrack_user, self.URL_AUTH)
        auth_data = dict(
            identity=self.__spacetrack_user,
            password=self.__spacetrack_pswd,
        )
        self.limiter()
        with requests.post(self.URL_AUTH, data=auth_data, timeout=30) as auth_resp:
            auth_resp.raise_for_status()
            self.__cookies = auth_resp.cookies

    def is_authenticated(self):
        """Check if queries are authenticated."""
        if self.__cookies is None:
            return False

        self.__cookies.clear_expired_cookies()
        return len(self.__cookies.items()) != 0

    @staticmethod
    def _as_str(value):
        """Convert misc. data types to a string."""
        if isinstance(value, np.datetime64):
            value = pd.to_datetime(value)
        if isinstance(value, pd.Timestamp):
            value = value.isoformat(sep=" ")
        if value is None:
            value = "null-val"
        return str(value).replace(" ", "%20")

    def render_query(self, norad_cat_id, columns=None, query_args=None):
        """Render a query string.

        Parameters
        ----------
        norad_cat_id : int
            NORAD catalog ID.
        columns : list[str], optional
            Columns to request. Default is `MAIN_COLUMNS`.
        query_args : list[tuple[str, str, any]], optional
            List of query arguments (field, comparison, value). Default is
            to get latest TLE entry.

        Returns
        -------
        str
            URL to query data from.

        """
        if norad_cat_id is None:
            raise ValueError("Must specify a NORAD catalog ID!")

        query_url = f"{self.URL_BASE}/basicspacedata/query/class/gp_history/format/json"

        # https://www.space-track.org/documentation#api-restOperators
        if query_args:
            logger.debug("Creating TLE query for NORAD ID [%d]", norad_cat_id)
            for field, cmp_type, val in query_args:
                if cmp_type in ("le", "ge"):
                    # If both were present, they would have been converted to a range!
                    raise ValueError(f"{self} queries do not support >= or <= comparisons without the other!")

                if cmp_type == "in":
                    val = ",".join(self._as_str(subval) for subval in val)
                elif cmp_type == "range":
                    val = f"{self._as_str(val[0])}--{self._as_str(val[1])}"
                else:
                    val = self._as_str(val)

                # Convert to URL query string.
                if cmp_type in ("eq", "in", "range"):
                    query_url += f"/{field}/{val}"
                elif cmp_type == "ne":
                    query_url += f"/{field}/%3C%3E{val}"
                elif cmp_type == "lt":
                    query_url += f"/{field}/%3C{val}"
                elif cmp_type == "gt":
                    query_url += f"/{field}/%3E{val}"
                else:
                    raise ValueError(f"Invalid comparison type [{cmp_type}]! Developer mistake?")

            query_url += f"/norad_cat_id/{norad_cat_id}"

        else:
            logger.info("Defaulting TLE query to get latest TLE entry for NORAD ID [%d]", norad_cat_id)
            query_url += f"/norad_cat_id/{norad_cat_id}/limit/1/orderby/EPOCH%20asc"

        if columns:
            if not isinstance(columns, list):
                raise TypeError(f"Expected a list of column names, not: {columns}")
        else:
            columns = self.MAIN_COLUMNS
        query_url += f"/predicates/{','.join(columns)}"

        if "/orderby/" not in query_url:
            query_url += "/orderby/EPOCH%20desc"

        query_url += "/metadata/true/emptyresult/show/distinct/true"
        return query_url

    def query(self, query_url):
        """Query for TLE data using a pre-rendered URL.

        Parameters
        ----------
        query_url : str
            URL to query data from. Expected to be generated using
            `render_query`, otherwise it set JSON and metadata args.

        Returns
        -------
        dict
            Dictionary response. If empty, the dict will be:
                `{'request_metadata': {'ReturnedRows': 0}, 'data': []}`

        """
        logger.debug("Querying for TLE data using: %s", query_url)

        # Authenticate if necessary (i.e., already called auth and the cookie
        # hasn't expired).
        if not self.is_authenticated():
            self.authenticate()

        # Request the data (but wait if API usage is too fast)!
        self.limiter()
        with requests.get(query_url, allow_redirects=True, cookies=self.__cookies, timeout=60) as resp:
            resp.raise_for_status()
            try:
                result = resp.json()
            except JSONDecodeError:
                try:
                    logger.exception("TLE request did not return a JSON! Raw response text:\n%s", resp.text)
                finally:
                    raise

            # Returns an empty list of there was no data.
            if isinstance(result, list):
                if len(result) and isinstance(result[0], dict) and "error" in result[0]:
                    # Only known to be a violation of the rate limit.
                    # TODO: Check the message and retry once with a long sleep?
                    raise ValueError(f"SpaceTrack.org error: {result[0]['error']}")

                if result != []:
                    raise ValueError(f"Unexpected result format: {result}")
                result = {"request_metadata": {"ReturnedRows": 0}, "data": []}
        return result

    def read(self, norad_cat_id, columns=None, query_args=None, index_col=None):
        """Get TLE data and format it into a table.

        Parameters
        ----------
        norad_cat_id : int
            NORAD catalog ID.
        columns : list[str], optional
            Columns to request. Default is `MAIN_COLUMNS`.
        query_args : list[tuple], optional
            List of query arguments (field, comparison, value). Default is
            to get the latest TLE entry.
        index_col : str, optional
            Column to set as the table's index.

        Returns
        -------
        pd.DataFrame
            Table of data. Will always contain the columns, even if no data
            was returned.

        """
        requested_columns = list(columns if columns else self.MAIN_COLUMNS)

        # Check for columns that are required to drop duplicates (i.e. TLEs that
        # were updated). Not an issue if no columns were specified (site returns
        # all possible columns).
        later_drop_columns = []
        if not self.keep_duplicates and requested_columns:
            if "epoch" not in requested_columns:
                requested_columns.append("epoch")
                later_drop_columns.append("epoch")
            if "creation_date" not in requested_columns:
                requested_columns.append("creation_date")
                later_drop_columns.append("creation_date")

        query_url = self.render_query(norad_cat_id, columns=requested_columns, query_args=query_args)

        for col in later_drop_columns:
            requested_columns.remove(col)

        result = self.query(query_url)
        logger.info("Retrieved [%d] TLE items for NORAD=[%s]", result["request_metadata"]["ReturnedRows"], norad_cat_id)

        # Convert to a dataframe and drop duplicates TLEs (keeping latest).
        table = pd.DataFrame(result["data"])

        if table.size == 0:
            table = pd.DataFrame(columns=requested_columns if requested_columns else self.MAIN_COLUMNS)

        table.rename(columns=str.lower, inplace=True)

        if not self.keep_duplicates:
            table.sort_values(by=["epoch", "creation_date"], ascending=True, inplace=True)
            table.drop_duplicates(subset="epoch", keep="last", inplace=True)

        if "epoch" in table.columns:
            table["epoch"] = pd.to_datetime(table["epoch"])
        if "creation_date" in table.columns:
            table["creation_date"] = pd.to_datetime(table["creation_date"])

        # Drop any extra columns that were required and reorder them.
        if requested_columns:
            table = table[requested_columns]
        table.columns.name = f"NORAD({norad_cat_id})"

        if index_col is not None:
            return table.set_index(index_col)
        return table

    @staticmethod
    def write(tle_table, filename, overwrite=False, append=False, header=True):
        """Write TLE data to a file.

        Parameters
        ----------
        tle_table : pd.DataFrame
            TLE data in table form. Must contain the columns "tle_line1" and
            "tle_line2". An empty table will cause a warning log.
        filename : str or Path or func, optional
            File path to write to or function to send the text. If None,
            returns the text.
        overwrite : bool, optional
            Option to overwrite an existing file. Can not use with `append`.
        append : bool, optional
            Option to append to an existing file. It will create a new file if
            one does not already exist. Can not use with `overwrite`.
        header : bool, optional
            Option to include a header in the file as the `table` name.

        Returns
        -------
        str or None
            TLE file text if `filename` was None, otherwise None is returned.

        """
        if isinstance(filename, (str, Path)):
            filename = Path(filename)
            if filename.is_file() and not overwrite and not append:
                raise FileExistsError(filename)
            if overwrite and append:
                raise ValueError("Can not set both overwrite and append!")
            if filename.is_file() and overwrite:
                filename.unlink()
        elif filename is not None:
            raise TypeError(f"`filename` must be a str or Path or None, not: {type(filename)}")

        if "tle_line1" not in tle_table.columns or "tle_line2" not in tle_table.columns:
            raise ValueError('Invalid existing query! Must include "TLE_LINE1" and "TLE_LINE2" columns!')

        if tle_table.size == 0:
            logger.warning("No TLE items were read by the accessor! Write step returning early (file might not exist!)")
            return

        tle_txt = ""
        if header is True:
            tle_txt += f"{tle_table.columns.name}\n"
        elif header:
            tle_txt += f"{header}\n"
        tle_txt += "\n".join(row["tle_line1"] + "\n" + row["tle_line2"] for _, row in tle_table.iterrows())
        tle_txt += "\n"  # SPICE will error out without this.

        if filename is None:
            return tle_txt

        if append:
            # TODO: Consider trimming any duplicate overlap...
            tle_txt = filename.read_text() + "\n" + tle_txt
        filename.write_text(tle_txt)
