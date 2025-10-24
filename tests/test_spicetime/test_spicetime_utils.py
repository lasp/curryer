"""utils - Unit test

@author: Brandon Stone
"""

import logging
import unittest

import numpy as np
import numpy.testing as npt

from curryer.spicetime import convert, native, utils
from curryer.utils import enable_logging

logger = logging.getLogger(__name__)
enable_logging(extra_loggers=[__name__])


class SpicetimeUtilsTestCase(unittest.TestCase):
    def test_find_mapping_path_fake(self):
        fmap = dict(
            et=dict(
                et="et2et",
                ugps="et2ugps",
                utc="et2utc",
            ),
            ugps=dict(
                et="ugps2et",
                ugps="ugps2ugps",
                gpsd="ugps2gpsd",
                dt64="ugps2dt64",
            ),
            utc=dict(
                et="utc2et",
                utc="utc2utc",
            ),
            gpsd=dict(
                ugps="gpsd2ugps",
                gpsd="gpsd2gpsd",
                epoch="gpsd2epoch",
            ),
            epoch=dict(
                gpsd="epoch2gpsd",
                epoch="epoch2epoch",
            ),
            dt64=dict(
                ugps="dt642ugps",
                dt64="dt642dt64",
            ),
        )

        self.assertListEqual(["ugps2ugps"], utils.find_mapping_path(fmap, "ugps", "ugps"))
        self.assertListEqual(["ugps2gpsd", "gpsd2epoch"], utils.find_mapping_path(fmap, "ugps", "epoch"))
        self.assertListEqual(["utc2et", "et2ugps", "ugps2dt64"], utils.find_mapping_path(fmap, "utc", "dt64"))
        with self.assertRaisesRegex(ValueError, "Unable to find"):
            fmap_bad = fmap.copy()
            fmap_bad["et"].pop("ugps")
            utils.find_mapping_path(fmap, "utc", "dt64")

    def test_find_mapping_path_realish(self):
        gpsd_func = utils.InputAsArray(np.float64, False, defaults=dict(to_epoch="GPS"))(
            native.gps_fraction_to_epoch_fraction
        )

        fmap = dict(
            et=dict(
                et="et2et",
                ugps=convert.to_ugps,
                utc="et2utc",
            ),
            ugps=dict(
                et="ugps2et",
                ugps="ugps2ugps",
                gpsd=native.ugps_to_gps_fraction,
                dt64="ugps2dt64",
            ),
            utc=dict(
                et=convert.from_utc,
                utc="utc2utc",
            ),
            gpsd=dict(
                ugps="gpsd2ugps",
                gpsd="gpsd2gpsd",
                # epoch=(native.gps_fraction_to_epoch_fraction, {'epoch_name'}),
                epoch=gpsd_func,
            ),
            epoch=dict(
                gpsd="epoch2gpsd",
                epoch="epoch2epoch",
            ),
            dt64=dict(
                ugps="dt642ugps",
                dt64="dt642dt64",
            ),
        )

        conversions = utils.find_mapping_path(fmap, "utc", "epoch")
        self.assertListEqual([convert.from_utc, convert.to_ugps, native.ugps_to_gps_fraction, gpsd_func], conversions)

        utc_times = np.array(["2020-01-01", "1980-01-06", "1981-01-06"])
        gpsd_times = utils.apply_conversions(conversions, utc_times)
        npt.assert_allclose(np.array([14605.0, 0.0, 366.0]), gpsd_times)

        self.assertEqual(
            "2020-10-01T18:44:57.167939",
            utils.apply_conversions(
                utils.find_mapping_path(convert.conversions, "ugps", "iso"),
                1285613115167939,
                date_format="%Y-%m-%dT%H:%M:%S.%f",
            ),
        )
        self.assertAlmostEqual(
            1022.7812172215162,
            utils.apply_conversions(utils.find_mapping_path(convert.conversions, "ugps", "tsis"), 1285613115167939),
            places=12,
        )
        self.assertAlmostEqual(
            2459124.2812172216,
            utils.apply_conversions(utils.find_mapping_path(convert.conversions, "ugps", "jd"), 1285613115167939),
            places=9,
        )
        self.assertAlmostEqual(
            59123.781217221513,
            utils.apply_conversions(utils.find_mapping_path(convert.conversions, "ugps", "mjd"), 1285613115167939),
            places=10,
        )

    def test_noop_decorated_funcs(self):
        arr = np.array([123.456], dtype=np.float64)
        out = utils.noop_float64(arr)
        self.assertIs(out, arr)

        arr = np.array([123.456, np.pi], dtype=np.float64)
        out = utils.noop_float64(arr)
        self.assertIs(out, arr)


if __name__ == "__main__":
    unittest.main()
