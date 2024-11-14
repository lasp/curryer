import logging
import os
import unittest

import numpy as np
import pandas as pd

from curryer import tle, utils


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class TLETestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ctim_norad_id = 52950
        self.__spacetrack_user = os.getenv('SPACETRACK_USER')
        self.__spacetrack_pswd = os.getenv('SPACETRACK_PSWD')

    @unittest.skip  # TODO: API Limits?
    def test_ctim_tle_read(self):
        accessor = tle.TLERemoteAccessor(self.__spacetrack_user, self.__spacetrack_pswd)
        table = accessor.read(self.ctim_norad_id, query_args=[
            ('epoch', 'range', ('2023-08-01', '2023-08-06'))], index_col='epoch')
        self.assertIsInstance(table, pd.DataFrame)
        self.assertTupleEqual(table.shape, (7, 6))
        self.assertTrue((table['object_name'] == 'CTIM').all())
        self.assertEqual(table.index[0], np.datetime64('2023-08-01 02:52:41.704032'))
        self.assertEqual(table.index[-1], np.datetime64('2023-08-05 21:28:55.890912'))

    def test_ctim_tle_write(self):
        table = pd.DataFrame(dict(
            tle_line1=['1 52950U 22074G   23213.11992713  .00078271  00000-0  14692-2 0  9999',
                       '1 52950U 22074G   23214.41074019  .00072198  00000-0  13476-2 0  9994',
                       '1 52950U 22074G   23215.05608240  .00087555  00000-0  16230-2 0  9998',
                       '1 52950U 22074G   23215.12061409  .00077375  00000-0  14365-2 0  9992',
                       '1 52950U 22074G   23216.08854086  .00068430  00000-0  12662-2 0  9999',
                       '1 52950U 22074G   23217.05638161  .00085666  00000-0  15720-2 0  9999',
                       '1 52950U 22074G   23217.89509133  .00091375  00000-0  16656-2 0  9990'],
            tle_line2=['2 52950  44.9973 107.3271 0013094 161.8408 198.2952 15.47650271 60555',
                       '2 52950  44.9974 100.0626 0013220 170.7779 189.3356 15.47843836 60753',
                       '2 52950  44.9972  96.4283 0013299 175.3139 184.7879 15.47958490 60858',
                       '2 52950  44.9972  96.0650 0013327 175.6137 184.4873 15.47967214 60861',
                       '2 52950  44.9969  90.6136 0013476 181.9313 178.1528 15.48111360 61016',
                       '2 52950  44.9967  85.1619 0013618 188.5025 171.5637 15.48253066 61164',
                       '2 52950  44.9972  80.4362 0013702 194.0738 165.9772 15.48414179 61295']),
        )
        table.columns.name = 'NORAD(52950)'

        expected_txt = '''NORAD(52950)
1 52950U 22074G   23213.11992713  .00078271  00000-0  14692-2 0  9999
2 52950  44.9973 107.3271 0013094 161.8408 198.2952 15.47650271 60555
1 52950U 22074G   23214.41074019  .00072198  00000-0  13476-2 0  9994
2 52950  44.9974 100.0626 0013220 170.7779 189.3356 15.47843836 60753
1 52950U 22074G   23215.05608240  .00087555  00000-0  16230-2 0  9998
2 52950  44.9972  96.4283 0013299 175.3139 184.7879 15.47958490 60858
1 52950U 22074G   23215.12061409  .00077375  00000-0  14365-2 0  9992
2 52950  44.9972  96.0650 0013327 175.6137 184.4873 15.47967214 60861
1 52950U 22074G   23216.08854086  .00068430  00000-0  12662-2 0  9999
2 52950  44.9969  90.6136 0013476 181.9313 178.1528 15.48111360 61016
1 52950U 22074G   23217.05638161  .00085666  00000-0  15720-2 0  9999
2 52950  44.9967  85.1619 0013618 188.5025 171.5637 15.48253066 61164
1 52950U 22074G   23217.89509133  .00091375  00000-0  16656-2 0  9990
2 52950  44.9972  80.4362 0013702 194.0738 165.9772 15.48414179 61295
'''
        tle_txt = tle.TLERemoteAccessor.write(table, None)
        self.assertEqual(tle_txt, expected_txt)


if __name__ == '__main__':
    unittest.main()
