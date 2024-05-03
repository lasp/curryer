import logging
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from curryer import utils, spicierpy
from curryer.kernels import ephemeris


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class EphemerisStateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parents[2]
        generic_dir = root_dir / 'data' / 'generic'
        self.assertTrue(generic_dir.is_dir())

        self.bin_dir = root_dir / 'bin' / 'spice' / (
            'macintel' if sys.platform == 'darwin' else 'linux')
        self.assertTrue(self.bin_dir.is_dir())

        self.data_dir = root_dir / 'tests' / 'data' / 'tsis1'
        self.assertTrue(self.data_dir.is_dir())

        self.tsis_state_prop = {
            "input_body": "ISS_SC",
            "input_frame": "J2000",
            "input_center": "EARTH",
            "input_data_units": {
                "angles": "radians",
                "distances": "feet"
            },
            "input_time_type": "utc",  # New due to CSV.
            "input_time_columns": ["UTC"],  # New due to CSV.
            "input_gap_threshold": "30s",
            "spk_type": "INTERP_HERMITE_UNEVEN",  # Was 13 (val instead of key)
            "polynom_degree": 3,
            "planet_kernels": [
                generic_dir / "pck00010.tpc",
                generic_dir / "earth_200101_990628_predict.bpc",
                generic_dir / "earth_720101_070426.bpc",
                generic_dir / "earth_000101_221016_220723.bpc",
                generic_dir / "moon_pa_de421_1900-2050.bpc",
                generic_dir / "geophysical.ker"
            ]
        }
        spicierpy.boddef('ISS_SC', -125544)

    def test_tsis_state_kernel(self):
        state_input_data = pd.read_csv(self.data_dir / 'iss_sc_v01.ephemeris.spk.20210610.csv')

        state_prop = ephemeris.EphemerisStateProperties(**self.tsis_state_prop)
        state_writer = ephemeris.EphemerisStateWriter(properties=state_prop, bin_dir=self.bin_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            kfile = Path(tmpdir) / 'test_state.spk'
            self.assertFalse(kfile.is_file())
            state_writer(state_input_data, kfile)

            self.assertTrue(kfile.is_file())

            # Check the kernel time span.
            span = spicierpy.ext.kernel_coverage(kfile, 'ISS_SC', to_fmt='utc')
            self.assertTupleEqual(span, ('2021-06-10 10:00:08.293690', '2021-06-10 11:59:58.166280'))

            span = spicierpy.ext.kernel_coverage(kfile, 'ISS_SC', to_fmt='dt64')
            input_times = pd.to_datetime(state_input_data['UTC'])
            self.assertEqual(input_times.min(), span[0])
            self.assertEqual(input_times.max(), span[1])


if __name__ == '__main__':
    unittest.main()
