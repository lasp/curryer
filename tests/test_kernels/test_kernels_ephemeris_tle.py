import logging
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from curryer import utils, spicierpy
from curryer.kernels import ephemeris


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class EphemerisTLETestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / 'data' / 'generic'
        self.assertTrue(generic_dir.is_dir())

        self.bin_dir = root_dir / 'bin' / 'spice' / (
            'macintel' if sys.platform == 'darwin' else 'linux')
        self.assertTrue(self.bin_dir.is_dir())

        self.ctim_tle_prop = {
            "input_tle": 52950,
            "input_body": "CTIM",
            "input_frame": "J2000",
            "input_center": "EARTH",
            "tle_start_pad": "12 hours",
            "tle_stop_pad": "12 hours",
            "planet_kernels": [
                generic_dir / "pck00010.tpc",
                generic_dir / "earth_200101_990628_predict.bpc",
                generic_dir / "earth_720101_070426.bpc",
                generic_dir / "earth_000101_221016_220723.bpc",
                generic_dir / "moon_pa_de421_1900-2050.bpc",
                generic_dir / "geophysical.ker"
            ]
        }
        spicierpy.boddef('CTIM', -152950)

    def test_ctim_tle_kernel(self):
        tle_input_data = pd.DataFrame(dict(
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
            # Not required since TLE lines contain time, but useful for validation.
            index=pd.DatetimeIndex(['2023-08-01 02:52:41.704032',
                                    '2023-08-02 09:51:27.952416',
                                    '2023-08-03 01:20:45.519360',
                                    '2023-08-03 02:53:41.057376',
                                    '2023-08-04 02:07:29.930304',
                                    '2023-08-05 01:21:11.371104',
                                    '2023-08-05 21:28:55.890912']),
        )

        tle_prop = ephemeris.EphemerisTLEProperties(**self.ctim_tle_prop)
        tle_writer = ephemeris.EphemerisTLEWriter(properties=tle_prop, bin_dir=self.bin_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            kfile = Path(tmpdir) / 'test_tle.spk'
            self.assertFalse(kfile.is_file())
            tle_writer(tle_input_data, kfile)

            self.assertTrue(kfile.is_file())

            # Check the kernel time span. Note how it's expanded by 12hrs on
            # each side, per our configuration. The times aren't exact due to
            # loss of precision during the TLE time encoding.
            span = spicierpy.ext.kernel_coverage(kfile, 'CTIM', to_fmt='utc')
            self.assertTupleEqual(span, ('2023-07-31 14:52:41.704019', '2023-08-06 09:28:55.890924'))

            span = spicierpy.ext.kernel_coverage(kfile, 'CTIM', to_fmt='dt64')
            self.assertLessEqual(np.abs(tle_input_data.index.min() - np.timedelta64(12, 'h') - span[0]),
                                 np.timedelta64(1, 'ms'))
            self.assertLessEqual(np.abs(tle_input_data.index.max() + np.timedelta64(12, 'h') - span[1]),
                                 np.timedelta64(1, 'ms'))


if __name__ == '__main__':
    unittest.main()
