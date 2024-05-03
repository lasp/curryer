"""native - Unit test

@author: Brandon Stone
"""
import time
import logging
import unittest

import numpy as np
import numpy.testing as npt

from curryer import utils
from curryer.spicetime import leapsecond, adapt, native, constants


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class NativeDT64TestCase(unittest.TestCase):
    def test_ugps2datetime_scalar(self):
        input_data = [
            ('1980-01-06', False),
            ('2015-06-30 23:59:59.999999', False),  # Before a leapsecond.
            ('2015-06-30 23:59:60.000001', True),  # During a leapsecond.
            ('2015-06-30 23:59:60.999999', True),  # During a leapsecond.
            ('2015-07-01', False),  # After a leapsecond.
        ]

        # Test scalar support and leapsecond handling.
        for utc_time, in_leap in input_data:
            ugps = adapt(utc_time, 'utc')
            dt64 = native.ugps2datetime(ugps)

            # Times during leapseconds are rounded up.
            if in_leap:
                exp_time = np.datetime64(utc_time.rsplit(':', 1)[0] + ':59', 'us') + np.timedelta64(1, 's')
            else:
                exp_time = np.datetime64(utc_time, 'us')
            self.assertEqual(exp_time, dt64)

    def test_ugps2datetime_iterable(self):
        ugps_times = [0, 1119787217000000]
        exp_times = np.array(['1980-01-06', '2015-07-01 12:00:00'], 'M8[us]')

        # List input.
        dt64 = native.ugps2datetime(ugps_times)
        npt.assert_array_equal(exp_times, dt64)

        # Array input of unexpected type.
        dt64 = native.ugps2datetime(np.array(ugps_times, dtype=np.float64))
        npt.assert_array_equal(exp_times, dt64)

    def test_ugps2datetime_ceil_leapseonds(self):
        utc_times = np.array([
            '2015-06-30 23:59:59.999999',  # Before a leapsecond.
            '2015-06-30 23:59:60.000001',  # During a leapsecond.
            '2015-06-30 23:59:60.999999',  # During a leapsecond.
            '2015-07-01 00:00:00.000000',  # After a leapsecond.
        ])
        exp_ceil_times = np.array([utc_times[0], utc_times[3], utc_times[3], utc_times[3]], dtype='M8[us]')
        self.assertEqual(2, np.unique(exp_ceil_times).size)

        ugps_times = adapt(utc_times, 'utc', 'ugps')
        self.assertEqual(4, np.unique(ugps_times).size)

        # Default is to ceil the times.
        out_times = native.ugps2datetime(ugps_times)
        npt.assert_allclose(exp_ceil_times.astype(int), out_times.astype(int))

        # Proof that dt64 does not support them.
        with self.assertRaisesRegex(ValueError, 'Seconds out of range'):
            np.array(utc_times, dtype='M8[us]')

        # Option to make them fatal.
        with self.assertRaisesRegex(ValueError, 'Datetime64 does not support times within leapseconds'):
            native.ugps2datetime(ugps_times, ceil_leapseconds=False)

        # Not fatal if the impacted times are removed.
        out_times2 = native.ugps2datetime(ugps_times[[0, 3]], ceil_leapseconds=False)
        npt.assert_allclose(exp_ceil_times[[0, 3]].astype(int), out_times2.astype(int))

    def test_datetime2ugps_scalar(self):
        dt64 = np.datetime64('1980-01-06', 'us')
        ugps = native.datetime2ugps(dt64)
        self.assertEqual(0, ugps)

    def test_datetime2ugps_iterable(self):
        dt64 = np.array(['1980-01-06', '2015-06-30 23:59:59.999999', '2015-07-01'], dtype='M8[us]')
        ugps = native.datetime2ugps(dt64)
        self.assertEqual([0, 1119744015999999, 1119744017000000], ugps.tolist())

    def test_datetime_round_trips_static(self):
        dt64 = native.ugps2datetime(1)
        self.assertEqual(np.datetime64('1980-01-06T00:00:00.000001'), dt64)
        ugps = native.datetime2ugps(dt64)
        self.assertEqual(1, ugps)

        dt64 = native.ugps2datetime((0, 1))
        npt.assert_array_equal(np.array(['1980-01-06', '1980-01-06T00:00:00.000001'], 'M8[us]'), dt64)
        ugps = native.datetime2ugps(dt64)
        npt.assert_array_equal(np.array([0, 1]), ugps)

        dt64 = native.ugps2datetime(np.array([0, 1]))
        npt.assert_array_equal(np.array(['1980-01-06', '1980-01-06T00:00:00.000001'], 'M8[us]'), dt64)
        ugps = native.datetime2ugps(dt64)
        npt.assert_array_equal(np.array([0, 1]), ugps)

        dt64_in = np.array(['1980-01-07T00:00:00.000001', '1981-06-30', '1981-07-01'], dtype='M8[us]')
        dt64_out = native.ugps2datetime(native.datetime2ugps(dt64_in))
        npt.assert_array_equal(dt64_in, dt64_out)

    def assert_datetime_round_trips(self, low, high, size=1000):
        # Random times.
        seed = int(time.time())
        np.random.seed(seed)
        logger.info('Random seed: %r', seed)
        ugps_in = np.random.randint(low, high, size=size)

        # uGPS -> datetime (leapseconds are lost).
        dt64 = native.ugps2datetime(ugps_in)

        # Identify which times were in leapseconds.
        #   Assumes all leapseconds are 1-sec long.
        half_error = 500000
        leapsecs = leapsecond.cache.get()
        idx_in_leap = np.abs(leapsecs['ugps'].values[..., None] - half_error - ugps_in).min(axis=0) < half_error

        # idx_next_leap = (ugps_in >= leapsecs['ugps'].values[..., None]).sum(axis=0)
        # idx_in_leap = (leapsecs[idx_next_leap < leapsecs.shape[0], 'ugps'].iloc[idx_next_leap] - ugps_in) < 1e6

        # Datetime -> uGPS (preserves the lose of leapseconds).
        ugps_out = native.datetime2ugps(dt64)

        # Exact matches outside leapseconds.
        npt.assert_array_equal(ugps_in[~idx_in_leap], ugps_out[~idx_in_leap])

        # Off by up to +1-sec during leapseconds.
        npt.assert_allclose(ugps_in[idx_in_leap], ugps_out[idx_in_leap] - half_error, rtol=0, atol=half_error)

    def test_datetime_round_trips_random_full(self):
        # Assert a round trip test of uGPS using random times between
        #   1980-01-06 and 2043-05-23T03:33:02
        self.assert_datetime_round_trips(0, int(2e15))

    def test_datetime_round_trips_random_leapseconds(self):
        # Random times around a leapsecond.
        ugps_leap = 1119744017000000
        self.assert_datetime_round_trips(ugps_leap - 2000000, ugps_leap + 1000000)


class NativeDayFractionTestCase(unittest.TestCase):
    def test_ugps2dayfraction_scalar(self):
        input_data = [
            ('1980-01-06 00:00:00.000000', 0, False),
            ('2015-06-30 23:59:59.999999', 12960 - 1 / 8.64e10, False),  # Before a leapsecond.
            ('2015-06-30 23:59:60.000001', 12960, True),  # During a leapsecond.
            ('2015-06-30 23:59:60.999999', 12960, True),  # During a leapsecond.
            ('2015-07-01 00:00:00.000000', 12960, False),  # After a leapsecond.
        ]

        # Test scalar support and leapsecond handling.
        for utc_time, exp_dayfrac, in_leap in input_data:
            ugps = adapt(utc_time, 'utc')
            out_dayfrac = native.ugps_to_gps_fraction(ugps)
            self.assertIsInstance(out_dayfrac, np.float64)
            self.assertEqual(exp_dayfrac, out_dayfrac, utc_time)

    def test_day_fraction_to_ugps_scalar(self):
        input_data = [
            ('1980-01-06 00:00:00.000000', 0, False),
            ('2015-06-30 23:59:59.999999', 12960 - 1 / 8.64e10, False),  # Before a leapsecond.
            ('2015-06-30 23:59:60.000001', 12960, True),  # During a leapsecond.
            ('2015-06-30 23:59:60.999999', 12960, True),  # During a leapsecond.
            ('2015-07-01 00:00:00.000000', 12960, False),  # After a leapsecond.
        ]

        # Test scalar support and leapsecond handling.
        for utc_time, in_dayfrac, in_leap in input_data:
            exp_ugps = adapt(utc_time, 'utc')
            if in_leap:
                exp_ugps += constants.TimeConstant.SEC_TO_USEC - exp_ugps % constants.TimeConstant.SEC_TO_USEC

            out_ugps = native.gps_fraction_to_ugps(in_dayfrac)
            self.assertIsInstance(out_ugps, np.int64)
            self.assertEqual(exp_ugps, out_ugps, utc_time)

    def test_day_fraction_to_epoch_fraction_scalar(self):
        # Validate the epochs which are days since the GPS epoch.
        for mission_epoch in constants.EpochGpsDays:
            out_epochfrac = native.gps_fraction_to_epoch_fraction(0.0, mission_epoch.name)
            self.assertEqual(-1 * mission_epoch, out_epochfrac, mission_epoch)

        # Validate random times.
        input_data = [
            (0, -13857, 'tsis'),  # 1980-01-06
            (12959.5, 589.5, 'tCTe'),  # 2015-06-30T12:00
        ]
        for in_dayfrac, exp_epochfrac, epoch_name in input_data:
            out_epochfrac = native.gps_fraction_to_epoch_fraction(in_dayfrac, epoch_name)
            self.assertIsInstance(out_epochfrac, np.float64)
            self.assertEqual(exp_epochfrac, out_epochfrac)

    def test_epoch_fraction_to_day_fraction_scalar(self):
        # Validate the epochs which are days since the GPS epoch.
        for mission_epoch in constants.EpochGpsDays:
            out_gpsfrac = native.epoch_fraction_to_gps_fraction(0.0, mission_epoch.name)
            self.assertEqual(mission_epoch, out_gpsfrac, mission_epoch)

        # Validate random times.
        input_data = [
            (-13857, 0, 'tsis'),  # 1980-01-06
            (589.5, 12959.5, 'tCTe'),  # 2015-06-30T12:00
        ]
        for in_epochfrac, exp_gpsfrac, epoch_name in input_data:
            out_gpsfrac = native.epoch_fraction_to_gps_fraction(in_epochfrac, epoch_name)
            self.assertIsInstance(out_gpsfrac, np.float64)
            self.assertEqual(exp_gpsfrac, out_gpsfrac)


if __name__ == '__main__':
    unittest.main()
