import logging
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
from spiceypy.utils.exceptions import SpiceNOFRAMECONNECT

from curryer import utils, spicetime, meta, spicierpy
from curryer.compute import pointing


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class PointingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parents[2]
        self.generic_dir = root_dir / 'data' / 'generic'
        self.data_dir = root_dir / 'data'
        self.test_dir = root_dir / 'tests' / 'data'
        self.assertTrue(self.generic_dir.is_dir())
        self.assertTrue(self.data_dir.is_dir())
        self.assertTrue(self.test_dir.is_dir())

        self.ctim_time_range = ('2023-04-20T05:00', '2023-04-20T07:00')
        self.tsis_time_range = ('2021-06-10T10:00', '2021-06-10T12:00')

    def test_pointing_ctim(self):
        expected = pd.read_csv(self.test_dir / 'ctim' / 'ctim_v01.pointingdata.20230420.csv')
        expected = expected.drop(columns=['version', 'instrumentmodeid', 'deltat']).set_index(
            'microsecondssincegpsepoch')

        mkrn = meta.MetaKernel.from_json(
            self.test_dir / 'ctim' / 'ctim_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )

        with mkrn.load():
            server = pointing.PointingData(
                observer='ctim_tim',
                microsecond_cadence=10000000,
                with_geolocate=True,
            )
            ugps_range = spicetime.adapt(self.ctim_time_range, 'utc')
            ugps_times = server.get_times(ugps_range)
            data = server.get_pointing(ugps_times)

        self.assertIsInstance(data, pd.DataFrame)
        self.assertTupleEqual(expected.shape, data.shape)
        for col in expected.columns:
            npt.assert_allclose(expected[col], data[col], rtol=3e-6 if 'surf' in col else 3e-12, err_msg=col)

    def test_pointing_tsis(self):
        expected = pd.read_csv(self.test_dir / 'tsis1' / 'tsis_v01.pointingdata.20210610.csv')
        expected = expected.drop(columns=['version', 'instrumentmodeid', 'deltat']).set_index(
            'microsecondssincegpsepoch')

        mkrn = meta.MetaKernel.from_json(
            self.test_dir / 'tsis1' / 'tsis_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )

        with mkrn.load():
            server = pointing.PointingData(
                observer='tsis_tim_glint',
                microsecond_cadence=10000000,
                with_geolocate=False,
            )
            ugps_range = spicetime.adapt(self.tsis_time_range, 'utc')
            ugps_times = server.get_times(ugps_range)
            data = server.get_pointing(ugps_times)

        self.assertIsInstance(data, pd.DataFrame)
        self.assertTupleEqual(expected.shape, data.shape)
        for col in expected.columns:
            npt.assert_allclose(expected[col], data[col], rtol=1e-12, err_msg=col)

    def test_sun_dot_earth_ctim(self):
        expected = pd.read_csv(self.test_dir / 'ctim' / 'ctim_v01.pointingdata.20230420.csv')
        expected = expected.drop(columns=['version', 'instrumentmodeid', 'deltat']).set_index(
            'microsecondssincegpsepoch')
        nrows = expected.shape[0]

        mkrn = meta.MetaKernel.from_json(
            self.test_dir / 'ctim' / 'ctim_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )

        usec_step = 10000000
        ugps_range = spicetime.adapt(self.ctim_time_range, 'utc')
        ugps_times = np.arange(ugps_range[0] + usec_step, ugps_range[1], usec_step)

        # Verify normal cosine calcs work with all kernels.
        with mkrn.load():
            timdotearth = pointing.boresight_dot_object('CTIM_TIM', 'EARTH', ugps_times)
            npt.assert_allclose(expected['timdotearth'], timdotearth, rtol=3e-12)

        # Remove the attitude kernel to simulate lack of pointing tlm.
        mkrn.mission_kernels = [fn for fn in mkrn.mission_kernels if 'attitude' not in fn.name]

        with mkrn.load():
            # Verify missing attitude for normal cosine methods.
            with self.assertRaisesRegex(SpiceNOFRAMECONNECT, r'insufficient information.*CTIM_COORD.*to.*J2000'):
                _ = pointing.boresight_dot_object('CTIM_TIM', 'EARTH', ugps_times)

            sun_state = spicierpy.ext.query_ephemeris(
                ugps_times=ugps_times,
                target='SUN',
                observer='CTIM',  # S/C, not instrument.
                ref_frame='J2000',
                allow_nans=False,
                velocity=False,
            ).values
            earth_state = spicierpy.ext.query_ephemeris(
                ugps_times=ugps_times,
                target='EARTH',
                observer='CTIM',  # S/C, not instrument.
                ref_frame='J2000',
                allow_nans=False,
                velocity=False,
            ).values

            sun_state /= np.linalg.norm(sun_state, axis=1)[..., None]
            earth_state /= np.linalg.norm(earth_state, axis=1)[..., None]

            sundotearth = np.prod([sun_state, earth_state], axis=0).sum(axis=1)
            self.assertTupleEqual(sundotearth.shape, (nrows,))

            # # TODO: New API added in latest version of SPICE!
            # _ = np.cos(spicierpy.trgsep(
            #     spicetime.adapt(ugps_times, to='et')[0],
            #     'SUN', 'POINT', 'NULL',
            #     'EARTH', 'POINT', 'NULL',
            #     'CTIM', 'NONE'
            # ))

            # Convert cosines to separation angles in degrees.
            sun_earth_ang = np.rad2deg(np.arccos(sundotearth))
            tim_earth_ang = np.rad2deg(np.arccos(expected['timdotearth'].values))
            tim_sun_ang = np.rad2deg(np.arccos(expected['timdotsun'].values))

            # Sanity check that when TIM is Sun pointing, the difference between
            # the TIM-Earth angle and Sun-Earth angle should be small.
            idx_sun = tim_sun_ang <= 1
            self.assertEqual(idx_sun.sum(), np.sum((sun_earth_ang[idx_sun] - tim_earth_ang[idx_sun]) <= 1))

            # The Sun-Earth angle should be less than the sum of the TIM-Sun
            # and TIM-Earth angles but greater than their difference.
            self.assertEqual(nrows, np.sum(sun_earth_ang <= (tim_earth_ang + tim_sun_ang)))
            self.assertEqual(nrows, np.sum(sun_earth_ang >= np.abs(tim_earth_ang - tim_sun_ang)))


if __name__ == '__main__':
    unittest.main()
