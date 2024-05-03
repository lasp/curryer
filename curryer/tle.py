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

from . import spicetime


logger = logging.getLogger(__name__)


# TODO: Create an accessor to parse a (local) TLE file?

class TLERemoteAccessor:
    """Access TLE data from SpaceTrack.org.
    """

    URL_BASE = 'https://www.space-track.org'
    URL_AUTH = f'{URL_BASE}/ajaxauth/login'
    MAIN_COLUMNS = ('object_name', 'norad_cat_id', 'epoch', 'creation_date', 'file', 'tle_line1', 'tle_line2')

    __prev_request_time = 0

    def __init__(self, user, pswd, keep_duplicates=False):  # , as_ugps=True):
        self.__spacetrack_user = user
        self.__spacetrack_pswd = pswd
        self.keep_duplicates = keep_duplicates
        # self.as_ugps = as_ugps
        self.__prev_request_time = time.time()

    @staticmethod
    def _as_str(value):
        if isinstance(value, np.datetime64):
            value = pd.to_datetime(value)
        if isinstance(value, pd.Timestamp):
            value = value.isoformat(sep=' ')
        if value is None:
            value = 'null-val'
        return str(value).replace(' ', '%20')

    def _render_query(self, norad_cat_id, columns=None, query_args=None):
        if norad_cat_id is None:
            raise ValueError('Must specify a NORAD catalog ID!')

        query_url = f'{self.URL_BASE}/basicspacedata/query/class/gp_history/format/json'

        # https://www.space-track.org/documentation#api-restOperators
        if query_args:
            for field, cmp_type, val in query_args:

                # # Special cases.
                # if field == 'ugps':  # Requires range conversion...
                #     if cmp_type == 'lt':
                #         val -= 1
                #         cmp_type = 'le'
                #     elif cmp_type == 'gt':
                #         val += 1
                #         cmp_type = 'ge'
                #     val = spicetime.adapt(val, 'ugps', 'dt64')

                if cmp_type in ('le', 'ge'):
                    # If both were present, they would have been converted to a range!
                    raise ValueError(f'{self} queries do not support >= or <= comparisons without the other!')

                if cmp_type == 'in':
                    val = ','.join(self._as_str(subval) for subval in val)
                elif cmp_type == 'range':
                    val = f'{self._as_str(val[0])}--{self._as_str(val[1])}'
                else:
                    val = self._as_str(val)

                # Convert to URL query string.
                if cmp_type in ('eq', 'in', 'range'):
                    query_url += f'/{field}/{val}'
                elif cmp_type == 'ne':
                    query_url += f'/{field}/%3C%3E{val}'
                elif cmp_type == 'lt':
                    query_url += f'/{field}/%3C{val}'
                elif cmp_type == 'gt':
                    query_url += f'/{field}/%3E{val}'
                else:
                    raise ValueError(f'Invalid comparison type [{cmp_type}]! Developer mistake?')

        if query_args is None:
            logger.info('Defaulting TLE query to get latest TLE entry for NORAD ID [%d]', norad_cat_id)
            query_url += f'/norad_cat_id/{norad_cat_id}/limit/1/orderby/EPOCH%20asc'
        else:
            logger.debug('Creating TLE query for NORAD ID [%d]', norad_cat_id)
            query_url += f'/norad_cat_id/{norad_cat_id}'

        if columns:
            if not isinstance(columns, list):
                raise TypeError(f'Expected a list of column names as `data`, not: {columns}')
        else:
            columns = self.MAIN_COLUMNS
        query_url += f'/predicates/{",".join(columns)}'

        if '/orderby/' not in query_url:
            query_url += f'/orderby/EPOCH%20desc'

        query_url += f'/metadata/true/emptyresult/show/distinct/true'
        return query_url

    def read(self, norad_cat_id, columns=None, query_args=None, index_col=None):
        requested_columns = list(columns if columns else self.MAIN_COLUMNS)

        # Check for columns that are required to drop duplicates (i.e. TLEs that
        # were updated). Not an issue if no columns were specified (site returns
        # all possible columns).
        later_drop_columns = []
        if not self.keep_duplicates and requested_columns:
            if 'epoch' not in requested_columns:
                requested_columns.append('epoch')
                later_drop_columns.append('epoch')
            if 'creation_date' not in requested_columns:
                requested_columns.append('creation_date')
                later_drop_columns.append('creation_date')

        query_url = self._render_query(norad_cat_id, columns=requested_columns, query_args=query_args)
        logger.debug('Querying for TLE data using: %s', query_url)

        for col in later_drop_columns:
            requested_columns.remove(col)

        # Authenticate and query at the same time by posting our auth details
        # along with the query arguments.
        post_data = dict(
            identity=self.__spacetrack_user,
            password=self.__spacetrack_pswd,
            query=query_url,
        )

        # Must self limit API requests or risk server-side errors (to limit).
        # Limits: 30 per minute, 300 per hour.
        time_since = time.time() - self.__prev_request_time
        if time_since < 2:
            wait = 2 - time_since
            logger.debug('%s sleeping for [%.3f] seconds due to API request limits...', self, wait)
            time.sleep(wait)
        self.__prev_request_time = time.time()

        # Request the data!
        with requests.post(self.URL_AUTH, data=post_data, allow_redirects=True) as resp:
            resp.raise_for_status()
            try:
                result = resp.json()
            except JSONDecodeError:
                try:
                    logger.exception('TLE request did not return a JSON! Raw response text:\n%s', resp.text)
                finally:
                    raise

            # Returns an empty list of there was no data.
            if isinstance(result, list):
                assert result == []
                result = {'request_metadata': {'ReturnedRows': 0}, 'data': []}

            logger.info('Retrieved [%d] TLE items for NORAD=[%s]', result['request_metadata']['ReturnedRows'],
                        norad_cat_id)

        # Convert to a dataframe and drop duplicates TLEs (keeping latest).
        table = pd.DataFrame(result['data'])

        if table.size == 0:
            table = pd.DataFrame(columns=requested_columns if requested_columns else self.MAIN_COLUMNS)

        table.rename(columns=str.lower, inplace=True)

        if not self.keep_duplicates:
            table.sort_values(by=['epoch', 'creation_date'], ascending=True, inplace=True)
            table.drop_duplicates(subset='epoch', keep='last', inplace=True)

        if 'epoch' in table.columns:
            table['epoch'] = pd.to_datetime(table['epoch'])
        if 'creation_date' in table.columns:
            table['creation_date'] = pd.to_datetime(table['creation_date'])

        # Drop any extra columns that were required and reorder them.
        if requested_columns:
            table = table[requested_columns]
        table.columns.name = f'NORAD({norad_cat_id})'

        if index_col is not None:
            return table.set_index(index_col)
        return table

    @staticmethod
    def write(tle_table, filename, overwrite=False, append=False, header=True):
        is_a_path = isinstance(filename, (str, Path))
        if is_a_path:
            filename = Path(filename)
            if filename.is_file() and not overwrite and not append:
                raise FileExistsError(filename)
            if overwrite and append:
                raise ValueError('Can not set both overwrite and append!')
            if filename.is_file() and overwrite:
                filename.unlink()

        if 'tle_line1' not in tle_table.columns or 'tle_line2' not in tle_table.columns:
            raise ValueError('Invalid existing query! Must include "TLE_LINE1" and "TLE_LINE2" columns!')

        if tle_table.size == 0:
            logger.warning('No TLE items were read by the accessor! Write step returning early (file might not exist!)')
            return

        tle_txt = ''
        if header is True:
            tle_txt += f'{tle_table.columns.name}\n'
        elif header:
            tle_txt += f'{header}\n'
        tle_txt += '\n'.join(row['tle_line1'] + '\n' + row['tle_line2'] for _, row in tle_table.iterrows())
        tle_txt += '\n'  # SPICE will error out without this.

        if append:
            # TODO: Consider trimming any duplicate overlap...
            tle_txt = filename.read_text() + '\n' + tle_txt

        if filename is None:
            return tle_txt
        if is_a_path:
            filename.write_text(tle_txt)
        else:
            filename.write(tle_txt)
