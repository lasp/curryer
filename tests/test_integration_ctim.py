import logging
import tempfile
import unittest
from pathlib import Path

import numpy.testing as npt
import pandas as pd

from curryer import meta, spicetime, utils
from curryer import spicierpy as sp
from curryer.compute import pointing
from curryer.kernels import create

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class CtimIntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parents[1]
        self.generic_dir = root_dir / "data" / "generic"
        self.data_dir = root_dir / "data" / "ctim"
        self.test_dir = root_dir / "tests" / "data" / "ctim"
        self.assertTrue(self.generic_dir.is_dir())
        self.assertTrue(self.data_dir.is_dir())
        self.assertTrue(self.test_dir.is_dir())

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

        self.mkrn = meta.MetaKernel.from_json(
            self.data_dir / "ctim_v01.kernels.tm.json",
            relative=True,
            sds_dir=self.generic_dir,
        )
        for fn in self.mkrn.mission_kernels.copy():
            if fn.suffix in (".bsp", ".bc"):
                self.mkrn.mission_kernels.remove(fn)

        self.fixed_kernel_configs = [
            self.data_dir / "ctim_tim_v01.fixed_offset.spk.json",
        ]
        self.pos_kernel_config = self.data_dir / "ctim_v01.tle.spk.json"
        self.att_kernel_config = self.test_dir / "ctim_v01.attitude_apid1.ck.testcase.json"

        self.expected_results_file = self.test_dir / "ctim_v01.pointingdata.20230420.csv"
        self.time_range = ("2023-04-20T05:00", "2023-04-20T07:00")

    def test_integration_ctim(self):
        expected_point = pd.read_csv(self.expected_results_file)
        expected_point = expected_point.drop(columns=["version", "instrumentmodeid", "deltat"]).set_index(
            "microsecondssincegpsepoch"
        )

        creator = create.KernelCreator(overwrite=False, append=False)

        # Static offsets.
        generated_kernels = []
        for kernel_config_file in self.fixed_kernel_configs:
            self.assertTrue(kernel_config_file.is_file(), kernel_config_file)
            generated_kernels.append(creator.write_from_json(kernel_config_file, output_kernel=self.tmp_dir))

        generated_kernels.append(
            creator.write_from_json(
                self.pos_kernel_config,
                output_kernel=self.tmp_dir,
                input_data=self.test_dir / "ctim_v01.tle.spk.20230420.csv",
            )
        )

        # NOTE: Normally CTIM's quats have to be sign-flipped, but these
        # were saved after that conversion had been applied!
        generated_kernels.append(
            creator.write_from_json(
                self.att_kernel_config,
                output_kernel=self.tmp_dir,
                input_data=self.test_dir / "ctim_v01.attitude_apid1.ck.20230420.csv",
            )
        )

        for fn in sorted(self.tmp_dir.glob("*")):
            logger.info(fn)

        with sp.ext.load_kernel([self.mkrn.sds_kernels, self.mkrn.mission_kernels, generated_kernels]):
            server = pointing.PointingData(
                observer="ctim_tim",
                microsecond_cadence=10000000,
                with_geolocate=True,
            )
            ugps_range = spicetime.adapt(self.time_range, "utc")
            ugps_times = server.get_times(ugps_range)
            data = server.get_pointing(ugps_times)

        self.assertIsInstance(data, pd.DataFrame)
        self.assertTupleEqual(expected_point.shape, data.shape)
        for col in expected_point.columns:
            # NOTE: Error increased from 1e-12 to 1e-5 due to minor differences
            # between the original (slow) SPICE impl and the vectorized impl.
            rtol = 3e-6 if "surface" in col else 1e-10
            npt.assert_allclose(expected_point[col], data[col], rtol=rtol, err_msg=col)


if __name__ == "__main__":
    unittest.main()
