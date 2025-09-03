"""convert - Unit test

Dependencies
------------
- Leapsecond kernel (loaded in memory)
    - sds_spice/data/leapseconds/naif*.tls
    - Automated by the `leapsecond` module. If the tests fail because of that,
    either the library testing was incorrectly setup OR another test is
    modifying the kernel pool!

Notes
-----
- The Ephemeris Time test values are from SPICE (direct calls).
- Other test values were taken from the datetime IDL library.

@author: Brandon Stone
"""
import copy
import logging
import random
import time
import unittest

import numpy as np

from curryer.spicetime.leapsecond import are_loaded, load
from curryer.spicetime import convert, utils
from curryer.utils import enable_logging

logger = logging.getLogger(__name__)
enable_logging(extra_loggers=[__name__])


class ConvertFmtStrTestCase(unittest.TestCase):
    def test_iso_only(self):
        r = convert.spice_strftime('%Y-%m-%d %H:%M:%S.%f')
        self.assertEqual('YYYY-MM-DD HR:MN:SC.######::RND ::UTC', r)

    def test_iso_with_extra_char(self):
        r = convert.spice_strftime('ISO: %Y-%m-%d %H:%M:%S.%f, 0%% - 100%%, XYZ')
        self.assertEqual('ISO: YYYY-MM-DD HR:MN:SC.######, 0% - 100%, XYZ::RND ::UTC', r)

    def test_decimal_seconds(self):
        r = convert.spice_strftime('%S.%f')
        self.assertEqual('SC.######::RND ::UTC', r)

    def test_fails_if_microseconds_alone(self):
        # SPICE does not support microseconds without seconds.
        with self.assertRaises(ValueError):
            convert.spice_strftime('%f')

    def test_fails_if_uses_unsupported_py_format(self):
        with self.assertRaises(ValueError):
            convert.spice_strftime('%Y-%m-%d %Z')  # Timezone name


class ConvertTestCase(unittest.TestCase):
    def setUp(self):
        # Trigger lazy loading by calling load() explicitly for tests
        load()
        self.assertTrue(are_loaded(), 'No leapsecond kernel is loaded. Check the `leapsecond` tests.')

    def test_utc_to_et_at_19800106(self):
        et = convert.from_utc('1980-01-06 00:00:00.0')
        self.assertEqual(et, -630763148.8159368)

    def test_utc_to_ugps_at_19800106(self):
        et = convert.from_utc('1980-01-06 00:00:00.0')
        ugps = convert.to_ugps(et)
        self.assertEqual(ugps, 0)

    def test_utc_to_gps_at_19800106(self):
        et = convert.from_utc('1980-01-06 00:00:00.0')
        gps = convert.to_gps(et)
        self.assertEqual(gps, 0.0)

    def test_utc_to_tai_at_19800106(self):
        et = convert.from_utc('1980-01-06 00:00:00.0')
        tai = convert.to_tai(et)
        self.assertEqual(tai, 694656019.0)

    def test_ugps_to_utc_at_19800106(self):
        et = convert.from_ugps(0)
        utc = convert.to_utc(et)
        self.assertEqual(utc, '1980-01-06 00:00:00.000000')

    def test_ugps_to_jd_at_19800106(self):
        gpsd = convert.conversions['ugps']['gpsd'](0)
        jd = convert.conversions['gpsd']['jd'](gpsd)
        self.assertEqual(jd, 2444244.5)

    def test_ugps_to_mjd_at_19800106(self):
        gpsd = convert.conversions['ugps']['gpsd'](0)
        mjd = convert.conversions['gpsd']['mjd'](gpsd)
        self.assertEqual(mjd, 44244.0)

    def test_from_utc_handles_dt64_input(self):
        dt = np.datetime64('2000-01-01 12:00:00.0', 'us')
        r = convert.from_utc(dt)
        self.assertAlmostEqual(64.18392728473108, r, places=6)

    def test_every_format_is_convertible_to_everything(self):
        formats = list(convert.conversions)
        for src in formats:
            for dst in formats:
                methods = utils.find_mapping_path(convert.conversions, src, dst)
                self.assertGreater(len(methods), 0)


class AdaptTestCase(unittest.TestCase):
    def setUp(self):
        # Trigger lazy loading by calling load() explicitly for tests
        load()
        self.assertTrue(are_loaded(), 'No leapsecond kernel is loaded. Check the `leapsecond` tests.')

    def test_ugps_to_utc_scalar(self):
        r = convert.adapt(1, 'ugps', 'utc')
        self.assertIsInstance(r, str)
        self.assertEqual('1980-01-06 00:00:00.000001', r)

    def test_utc_to_utc_respects_date_format(self):
        r = convert.adapt('2018-01-11 11:59:59', 'utc', 'utc', date_format='%Y/%j')
        self.assertEqual('2018/011', r)
        r = convert.adapt('2018-01-11 12:00:00', 'utc', 'utc', date_format='%Y-%m-%d')
        self.assertEqual('2018-01-12', r)  # Rounded.

    def test_tsis_to_utc_handles_2d_array(self):
        fdays = np.arange(12).reshape((4, 3))
        r = convert.adapt(fdays, 'tsis', 'utc', date_format='%Y-%m-%d')
        self.assertIsInstance(r, np.ndarray)
        self.assertTupleEqual(fdays.shape, r.shape)
        self.assertListEqual([['2017-12-14', '2017-12-15', '2017-12-16'],
                              ['2017-12-17', '2017-12-18', '2017-12-19'],
                              ['2017-12-20', '2017-12-21', '2017-12-22'],
                              ['2017-12-23', '2017-12-24', '2017-12-25']], r.tolist())

    def test_default_from_ttype_is_ugps(self):
        r = convert.adapt(1, to='gps')
        self.assertAlmostEqual(0.000001, r, places=6)

    def test_default_to_ttype_is_ugps(self):
        r = convert.adapt(1, from_='gps')
        self.assertEqual(1000000, r)

    def test_fail_missing_from_and_to(self):
        with self.assertRaises(ValueError):
            convert.adapt(0)

    def test_fail_bad_from_value(self):
        with self.assertRaises(ValueError):
            convert.adapt(0, from_='invalid-value')

    def test_fail_bad_to_value(self):
        with self.assertRaises(ValueError):
            convert.adapt(0, to='invalid-value')


class AdaptIntegrationTestCase(unittest.TestCase):
    loggers = {'sds_utils.idl': 15}

    def setUp(self):
        # Trigger lazy loading by calling load() explicitly for tests
        load()
        self.assertTrue(are_loaded(), 'No leapsecond kernel is loaded. Check the `leapsecond` tests.')

        self.ttypes = list(convert.TTYPE_TO_DTYPE.keys())
        self.ttypes.sort()

        self.seed = int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)
        logger.debug('Random seed: %r', self.seed)

    def test_scalar(self):
        # Test all `ttypes` against the input value(s).
        #   Assert a round trip test of uGPS (ugps -> ttype -> ugps).
        #   Use random times between 1980-01-06 and 2043-05-23T03:33:02
        dt_in = random.randint(0, int(2e15))

        for ttype in self.ttypes:
            with self.subTest(ttype=ttype):
                dt_val = copy.deepcopy(dt_in)
                dt_middle = convert.adapt(dt_val, from_='ugps', to=ttype)
                dt_out = convert.adapt(dt_middle, from_=ttype, to='ugps')

                self.assertIsInstance(dt_out, np.int64)
                self.assertEqual(dt_in, dt_out, 'Failed round trip test of {!r}'.format(ttype))

    def test_list(self):
        dt_in = [random.randint(0, int(2e15)) for _ in range(100)]

        for ttype in self.ttypes:
            with self.subTest(ttype=ttype):
                dt_val = copy.deepcopy(dt_in)
                dt_middle = convert.adapt(dt_val, from_='ugps', to=ttype)
                dt_out = convert.adapt(dt_middle, from_=ttype, to='ugps')

                self.assertIsInstance(dt_out, list)
                self.assertEqual(len(dt_in), len(dt_out))
                self.assertListEqual(dt_in, dt_out, 'Failed round trip test of {!r}'.format(ttype))

    def test_tuple(self):
        dt_in = tuple(random.randint(0, int(2e15)) for _ in range(100))

        for ttype in self.ttypes:
            with self.subTest(ttype=ttype):
                dt_val = copy.deepcopy(dt_in)
                dt_middle = convert.adapt(dt_val, from_='ugps', to=ttype)
                dt_out = convert.adapt(dt_middle, from_=ttype, to='ugps')

                self.assertIsInstance(dt_out, tuple)
                self.assertEqual(len(dt_in), len(dt_out))
                self.assertTupleEqual(dt_in, dt_out, 'Failed round trip test of {!r}'.format(ttype))

    def test_ndarray(self):
        dt_in = np.random.randint(0, int(2e15), 100)

        for ttype in self.ttypes:
            with self.subTest(ttype=ttype):
                dt_val = copy.deepcopy(dt_in)
                dt_middle = convert.adapt(dt_val, from_='ugps', to=ttype)
                dt_out = convert.adapt(dt_middle, from_=ttype, to='ugps')

                self.assertIsInstance(dt_out, np.ndarray)
                self.assertEqual(len(dt_in), dt_out.size)
                self.assertFalse((dt_in != dt_out).any(), 'Failed round trip test of {!r}'.format(ttype))


class SpiceTimeClassTestCase(unittest.TestCase):
    def setUp(self):
        # Trigger lazy loading by calling load() explicitly for tests
        load()
        self.assertTrue(are_loaded(), 'No leapsecond kernel is loaded. Check the `leapsecond` tests.')

    def test_ndarray_explicit_creation(self):
        r = convert.SpiceTime(1, 'ugps')
        self.assertIsInstance(r, convert.SpiceTime)
        self.assertTrue(hasattr(r, 'ttype'))
        self.assertEqual('ugps', r.ttype)

    def test_ndarray_fail_explicit_creation_wrong_type(self):
        with self.assertRaises(TypeError):
            # For explicit creation, new objects must use SpiceTime's
            #   `__new__` method.
            np.ndarray.__new__(convert.SpiceTime, 1)

    def test_ndarray_view_casting(self):
        in_arr = np.arange(2)
        r = in_arr.view(convert.SpiceTime)
        self.assertIsInstance(r, convert.SpiceTime)
        self.assertTrue(hasattr(r, 'ttype'))
        self.assertIsNone(r.ttype)

    def test_ndarray_slicing(self):
        arr = convert.SpiceTime([0, 1, 2], 'ugps')
        r = arr[1:]
        self.assertIsInstance(r, convert.SpiceTime)
        self.assertTrue(hasattr(r, 'ttype'))
        self.assertEqual('ugps', r.ttype)
        self.assertEqual(2, len(r))

    def test_default_ttype_is_none(self):
        r = convert.SpiceTime(1)
        self.assertIsNone(r.ttype)

    def test_change_ttype_if_none(self):
        r = np.arange(2).view(convert.SpiceTime)
        self.assertIsNone(r.ttype)
        r.ttype = 'ugps'
        self.assertEqual('ugps', r.ttype)

    def test_change_ttype_fails_if_not_none(self):
        r = convert.SpiceTime([0, 1], 'ugps')
        self.assertEqual('ugps', r.ttype)
        with self.assertRaises(AttributeError):
            r.ttype = 'tai'

    def test_no_ttype_in_to_conversion_raises_excep(self):
        ugps = convert.SpiceTime(1, ttype='ugps')
        ugps._ttype = None
        self.assertIsNone(ugps.ttype)
        with self.assertRaises(AttributeError):
            ugps.to_gps()

    def test_update_ttype_after_adapt(self):
        ugps = convert.SpiceTime([0, 1], ttype='ugps')
        self.assertEqual('ugps', ugps.ttype)
        tai = ugps.adapt('tai')
        self.assertIsInstance(tai, convert.SpiceTime)
        self.assertEqual('tai', tai.ttype)

    def test_ugps_to_utc_array(self):
        ugps = convert.SpiceTime([0, 1], ttype='ugps')
        utc = ugps.to_utc()  # date_format='%Y-%m-%d %H:%M:%S.%f')

        self.assertEqual('utc', utc.ttype)
        self.assertListEqual(
            ['1980-01-06 00:00:00.000000', '1980-01-06 00:00:00.000001'],
            utc.tolist()
        )

    def test_ugps_to_utc_custom_date_format(self):
        ugps = convert.SpiceTime(1000001, ttype='ugps')
        utc = ugps.to_utc(date_format='%S.%f')
        self.assertEqual('01.000001', utc)

    def test_cls_to_et(self):
        ugps = convert.SpiceTime(1, ttype='ugps')
        r = ugps.to_et()
        self.assertEqual('et', r.ttype)
        self.assertEqual('float64', r.dtype)
        # For scalar arrays, the ndarray method `tolist` returns a scalar data
        #   (e.g., int, float or str). Required since `assertAlmostEqual`
        #   calls `round`, and thus `__round__`.
        self.assertAlmostEqual(-630763148.8159359, r.tolist(), places=6)

    def test_cls_to_ugps(self):
        gps = convert.SpiceTime(0.000001, ttype='gps')
        r = gps.to_ugps()
        self.assertEqual('ugps', r.ttype)
        self.assertEqual('int64', r.dtype)
        self.assertAlmostEqual(1, r.tolist(), places=0)

    def test_cls_to_gps(self):
        ugps = convert.SpiceTime(1, ttype='ugps')
        r = ugps.to_gps()
        self.assertEqual('gps', r.ttype)
        self.assertEqual('float64', r.dtype)
        self.assertAlmostEqual(0.000001, r.tolist(), places=6)

    def test_cls_to_tai(self):
        ugps = convert.SpiceTime(1, ttype='ugps')
        r = ugps.to_tai()
        self.assertEqual('tai', r.ttype)
        self.assertEqual('float64', r.dtype)
        self.assertAlmostEqual(694656019.000001, r.tolist(), places=6)

    def test_cls_to_utc(self):
        ugps = convert.SpiceTime(1, ttype='ugps')
        r = ugps.to_utc()
        self.assertEqual('utc', r.ttype)
        self.assertEqual('<U26', r.dtype)
        self.assertEqual('1980-01-06 00:00:00.000001', r)

    def test_add_timedelta64_us_with_ugps(self):
        times = convert.SpiceTime(np.arange(4), 'ugps')
        diff = (np.arange(4) * 10).astype('timedelta64[us]')
        self.assertEqual('ugps', times.ttype)
        self.assertEqual(np.timedelta64, diff.dtype.type)

        out = times + diff
        self.assertIsInstance(out, convert.SpiceTime)
        self.assertEqual('ugps', times.ttype)
        self.assertEqual(np.int64, out.dtype)

        self.assertEqual(0, out[0])
        self.assertEqual(33, out[3])

    def test_add_timedelta64_ms_with_ugps_fails(self):
        times = convert.SpiceTime(np.arange(4), 'ugps')
        diff = (np.arange(4) * 10).astype('timedelta64[ms]')
        self.assertEqual('ugps', times.ttype)
        self.assertEqual(np.timedelta64, diff.dtype.type)

        # Can't add milliseconds to microseconds.
        with self.assertRaises(TypeError):
            _ = times + diff

    def test_add_timedelta64_s_with_gps(self):
        # Must cast as an int because numpy wont allow adding floats to
        #   timedelta64 values.
        times = convert.SpiceTime(np.arange(4), 'gps')
        times = times.astype(np.int64)
        diff = (np.arange(4) * 10).astype('timedelta64[s]')
        self.assertEqual('gps', times.ttype)
        self.assertEqual(np.timedelta64, diff.dtype.type)

        out = times + diff
        self.assertIsInstance(out, convert.SpiceTime)
        self.assertEqual('gps', times.ttype)
        self.assertEqual(np.float64, out.dtype)

        self.assertEqual(0, out[0])
        self.assertEqual(33, out[3])

    def test_add_datetime64_fails(self):
        times = convert.SpiceTime(np.arange(4), 'ugps')
        dt64 = (np.arange(4) * 10).astype('datetime64[us]')

        # Dt64 has no support for leapseconds.
        with self.assertRaises(TypeError):
            _ = times + dt64


if __name__ == '__main__':
    unittest.main()
