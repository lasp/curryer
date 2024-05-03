import logging
import tempfile
import unittest
from pathlib import Path

import numpy.testing as npt
import pandas as pd

from curryer import utils, spicetime, meta
from curryer import spicierpy as sp
from curryer.compute import pointing
from curryer.kernels import create


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class Tsis1IntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parents[1]
        self.generic_dir = root_dir / 'data' / 'generic'
        self.data_dir = root_dir / 'data' / 'tsis1'
        self.test_dir = root_dir / 'tests' / 'data' / 'tsis1'
        self.assertTrue(self.generic_dir.is_dir())
        self.assertTrue(self.data_dir.is_dir())
        self.assertTrue(self.test_dir.is_dir())

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

        self.mkrn = meta.MetaKernel.from_json(
            self.data_dir / 'tsis_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )
        for fn in self.mkrn.mission_kernels.copy():
            if fn.suffix in ('.bsp', '.bc'):
                self.mkrn.mission_kernels.remove(fn)

        self.fixed_kernel_configs = [
            self.data_dir / "iss_expa35_v01.fixed_offset.spk.json",
            self.data_dir / "tsis_tads_v01.fixed_offset.spk.json",
            self.data_dir / "tsis_azel_v01.fixed_offset.spk.json",
            self.data_dir / "tsis_tim_v01.fixed_offset.spk.json",
            self.data_dir / "tsis_tim_glint_v01.fixed_offset.spk.json",
        ]
        self.pos_kernel_config = self.test_dir / 'iss_sc_v01.ephemeris.spk.testcase.json'
        self.att_kernel_config = self.test_dir / 'iss_sc_v01.attitude.ck.testcase.json'
        self.azel_kernel_config = self.test_dir / 'tsis_azel_v01.attitude.ck.testcase.json'

        self.expected_results_file = self.test_dir / 'tsis_v01.pointingdata.20210610.csv'
        self.time_range = ('2021-06-10T10:00', '2021-06-10T12:00')

    def test_integration_tsis(self):
        expected = pd.read_csv(self.expected_results_file)
        expected = expected.drop(columns=['version', 'instrumentmodeid', 'deltat']).set_index(
            'microsecondssincegpsepoch')

        creator = create.KernelCreator(overwrite=False, append=False)

        # Static offsets.
        generated_kernels = []
        for kernel_config_file in self.fixed_kernel_configs:
            self.assertTrue(kernel_config_file.is_file(), kernel_config_file)
            generated_kernels.append(creator.write_from_json(
                kernel_config_file, output_kernel=self.tmp_dir
            ))

        # ISS Ephemeris in ECI.
        generated_kernels.append(creator.write_from_json(
            self.pos_kernel_config, output_kernel=self.tmp_dir,
            input_data=self.test_dir / 'iss_sc_v01.ephemeris.spk.20210610.csv',
        ))

        # ISS Attitude in ECI.
        generated_kernels.append(creator.write_from_json(
            self.att_kernel_config, output_kernel=self.tmp_dir,
            input_data=self.test_dir / 'iss_sc_v01.attitude.ck.20210610.csv',
        ))

        # TPS Azimuth & Elevation angles.
        generated_kernels.append(creator.write_from_json(
            self.azel_kernel_config, output_kernel=self.tmp_dir,
            input_data=self.test_dir / 'tsis_azel_v01.attitude.ck.20210610.csv',
        ))

        for fn in sorted(self.tmp_dir.glob('*')):
            logger.info(fn)

        with sp.ext.load_kernel([self.mkrn.sds_kernels, self.mkrn.mission_kernels, generated_kernels]):
            server = pointing.PointingData(
                observer='tsis_tim_glint',
                microsecond_cadence=10000000,
                with_geolocate=False,
            )
            ugps_range = spicetime.adapt(self.time_range, 'utc')
            ugps_times = server.get_times(ugps_range)
            data = server.get_pointing(ugps_times)

        self.assertIsInstance(data, pd.DataFrame)
        self.assertTupleEqual(expected.shape, data.shape)
        for col in expected.columns:
            npt.assert_allclose(expected[col], data[col], rtol=1e-12, err_msg=col)


if __name__ == '__main__':
    unittest.main()
