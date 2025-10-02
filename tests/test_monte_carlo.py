import logging
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import xarray as xr

from curryer import utils, spicetime, meta
from curryer import spicierpy as sp
from curryer.compute.constants import SpatialQualityFlags as SQF
from curryer.correction import monte_carlo as mc
from curryer.kernels import create


logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.DEBUG, extra_loggers=[__name__])

xr.set_options(display_width=120, display_max_rows=30)
np.set_printoptions(linewidth=120)


def load_param_sets(config):
    output = []
    for ith in range(config.n_iterations):
        out_set = []
        for param in config.parameters:
            # TODO: Fake to just return the center. Should consider range and
            #   apply random factor...
            param_vals = param.data['center']

            if param.ptype is mc.ParameterType.CONSTANT_KERNEL and '.ck.' in param.config_file.name:
                param_vals = pd.DataFrame({
                    # "ugps": [0, 3155760018000000],  # FIXME: Why does SPICE CK function think a 2080 time is invalid?
                    "ugps": [0, 2209075218000000],
                    "angle_x": [param_vals[0], param_vals[0]],
                    "angle_y": [param_vals[1], param_vals[1]],
                    "angle_z": [param_vals[2], param_vals[2]],
                })

            out_set.append((param, param_vals))
        output.append(out_set)
    return output


def apply_offset(config, param_data, input_data):
    input_data = input_data.copy()

    # TODO: Fake to just modify by the center. Should consider range and
    #   apply random factor...
    input_data[config.data['field']] += config.data['center'] + param_data

    return input_data


class ClarreoMonteCarloTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parent.parent
        self.generic_dir = root_dir / 'data' / 'generic'
        self.data_dir = root_dir / 'tests' / 'data' / 'clarreo' / 'gcs'  # Uses 5a configs.
        self.assertTrue(self.generic_dir.is_dir(), self.generic_dir)
        self.assertTrue(self.data_dir.is_dir(), self.data_dir)

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

    def create_always_static_kernels(self):
        # While this obj is not used, it's required to load name and IDs for
        # the following kernel creation steps.
        mkrn = meta.MetaKernel.from_json(
            self.data_dir / 'cprs_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )

        fixed_kernel_configs = [
            self.data_dir / "cprs_st_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_base_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_pede_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_az_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_yoke_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_el_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_hysics_v01.fixed_offset.spk.json",
        ]

        generated_kernels = []
        creator = create.KernelCreator(overwrite=False, append=False)
        for kernel_config_file in fixed_kernel_configs:
            self.assertTrue(kernel_config_file.is_file(), kernel_config_file)
            generated_kernels.append(creator.write_from_json(
                kernel_config_file, output_kernel=self.tmp_dir
            ))
        return generated_kernels

    def load_telemetry(self):
        sc_spk_df = pd.read_csv(self.data_dir / "openloop_tlm_5a_sc_spk_20250521T225242.csv", index_col=0)
        sc_ck_df = pd.read_csv(self.data_dir / "openloop_tlm_5a_sc_ck_20250521T225242.csv", index_col=0)
        st_ck_df = pd.read_csv(self.data_dir / "openloop_tlm_5a_st_ck_20250521T225242.csv", index_col=0)
        azel_ck_df = pd.read_csv(self.data_dir / "openloop_tlm_5a_azel_ck_20250521T225242.csv", index_col=0)

        # Reverse the direction of the Azimuth element.
        azel_ck_df['hps.az_ang_nonlin'] = azel_ck_df['hps.az_ang_nonlin'] * -1

        # Convert star-tracker from an rot-mat to a quat.
        tlm_st_rot = np.vstack([st_ck_df['hps.dcm_base_iss_1_1'].values,
                                st_ck_df['hps.dcm_base_iss_1_2'].values,
                                st_ck_df['hps.dcm_base_iss_1_3'].values,
                                st_ck_df['hps.dcm_base_iss_2_1'].values,
                                st_ck_df['hps.dcm_base_iss_2_2'].values,
                                st_ck_df['hps.dcm_base_iss_2_3'].values,
                                st_ck_df['hps.dcm_base_iss_3_1'].values,
                                st_ck_df['hps.dcm_base_iss_3_2'].values,
                                st_ck_df['hps.dcm_base_iss_3_3'].values]).T
        tlm_st_rot = np.reshape(tlm_st_rot, (-1, 3, 3)).copy()
        tlm_st_rot_q = np.vstack([sp.m2q(tlm_st_rot[i, :, :]) for i in range(tlm_st_rot.shape[0])])
        st_ck_df['hps.dcm_base_iss_s'] = tlm_st_rot_q[:, 0]
        st_ck_df['hps.dcm_base_iss_i'] = tlm_st_rot_q[:, 1]
        st_ck_df['hps.dcm_base_iss_j'] = tlm_st_rot_q[:, 2]
        st_ck_df['hps.dcm_base_iss_k'] = tlm_st_rot_q[:, 3]

        left_df = sc_spk_df
        for right_df in [sc_ck_df, st_ck_df, azel_ck_df]:
            left_df = pd.merge(left_df, right_df, on='ert', how='outer')
        left_df = left_df.sort_values('ert')

        # Compute combined second and subsecond timetag.
        for col in list(left_df):
            if col in ('hps.bad_ps_tms', 'hps.corrected_tms', 'hps.resolver_tms', 'hps.st_quat_coi_tms'):
                assert col + 's' in left_df.columns, col

                if col == 'hps.bad_ps_tms':
                    left_df[col + '_tmss'] = left_df[col] + left_df[col + 's'] / 256
                elif col in ('hps.corrected_tms', 'hps.resolver_tms', 'hps.st_quat_coi_tms'):
                    left_df[col + '_tmss'] = left_df[col] + left_df[col + 's'] / 2 ** 32
                else:
                    raise ValueError('Missing if for expected column...')

        return left_df

    def load_science(self):
        sci_time_df = pd.read_csv(self.data_dir / "openloop_tlm_5a_sci_times_20250521T225242.csv", index_col=0)

        sci_time_df['corrected_timestamp'] *= 1e6  # Frame times are GPS seconds, geolocation expects uGPS.

        return sci_time_df

    @patch("curryer.correction.monte_carlo.load_param_sets", new=load_param_sets)
    @patch("curryer.correction.monte_carlo.apply_offset", new=apply_offset)
    def test_scenario_5a(self):
        # Define the Monte Carlo configuration...
        config = mc.MonteCarloConfig(
            seed=None,  # Not implemented.
            n_iterations=2,
            parameters=[
                mc.ParameterConfig(
                    ptype=mc.ParameterType.CONSTANT_KERNEL,
                    config_file=self.data_dir / "cprs_base_v01.attitude.ck.json",
                    data=dict(
                        # center=[0.00014637416712048754, 3.1964190810378915e-08, 0.00021837316449198788],  # 3,2,1
                        center=[0.0002183731668313877, 0.0, 0.0001463741706105875],  # 1,2,3
                        arange=[-1.0, 1.0],
                    ),
                ),
                mc.ParameterConfig(
                    ptype=mc.ParameterType.CONSTANT_KERNEL,
                    config_file=self.data_dir / "cprs_yoke_v01.attitude.ck.json",
                    data=dict(
                        # center=[-0.00044086911909579945, 0.025488902871536698, 0.008866582061788912],  # 3,2,1
                        center=[0.00885822324677249, 0.02549180848333736, -0.00021494961810004178],  # 1,2,3
                        arange=[-1.0, 1.0],
                    ),
                ),
                mc.ParameterConfig(
                    ptype=mc.ParameterType.CONSTANT_KERNEL,
                    config_file=self.data_dir / "cprs_hysics_v01.attitude.ck.json",
                    data=dict(
                        # center=[0.0007384287288093838, 0.00391102555703321, 0.0012154184503222237],  # 3,2,1
                        center=[0.0012183154394864839, 0.0039101240887754575, 0.0007431873840775429],  # 1,2,3
                        arange=[-1.0, 1.0],
                    ),
                ),
                mc.ParameterConfig(
                    ptype=mc.ParameterType.OFFSET_KERNEL,
                    config_file=self.data_dir / "cprs_az_v01.attitude.ck.json",
                    data=dict(
                        field="hps.az_ang_nonlin",
                        center=0.0,
                        arange=[-5.0, 5.0],
                    ),
                ),
                mc.ParameterConfig(
                    ptype=mc.ParameterType.OFFSET_KERNEL,
                    config_file=self.data_dir / "cprs_el_v01.attitude.ck.json",
                    data=dict(
                        field="hps.el_ang_nonlin",
                        center=0.0,
                        arange=[-5.0, 5.0],
                    ),
                ),
                mc.ParameterConfig(
                    ptype=mc.ParameterType.OFFSET_TIME,
                    config_file=None,
                    data=dict(
                        field="corrected_timestamp",
                        center=0.0,
                        arange=[-2.0, 2.0],
                    ),
                ),
            ],
            geo=mc.GeolocationConfig(
                meta_kernel_file=self.data_dir / 'cprs_v01.kernels.tm.json',
                generic_kernel_dir=self.generic_dir,
                dynamic_kernels=[
                    # Dynamic kernels that aren't altered (aka NOT az/el).
                    self.data_dir / "iss_sc_v01.ephemeris.spk.json",
                    self.data_dir / "iss_sc_v01.attitude.ck.json",
                    self.data_dir / "cprs_st_v01.attitude.ck.json",
                ],
                instrument_name='CPRS_HYSICS',
                time_field='corrected_timestamp',
            )
        )

        # Load the telemetry and sciene data that will be mocked into the test.
        tlm_dataset = self.load_telemetry()
        sci_dataset = self.load_science()
        # gpc_datgaset = self.load_gcp()  # Not implemented...

        # Nominally, these are created at the start of the mission, saved to S3
        # (or similar) and never changed. Created here in case their pre-launch
        # values change.
        static_kernels = self.create_always_static_kernels()

        # Since the static kernels are not currently in the meta-kernel, we
        # must manually load them now (if in the MK, they would get loaded).
        with sp.ext.load_kernel(static_kernels):

            # Patch the data loads...
            with patch('curryer.correction.monte_carlo.load_telemetry') as mock_tlm, patch(
                    'curryer.correction.monte_carlo.load_science') as mock_sci, patch(
                    'curryer.correction.monte_carlo.load_gcp') as mock_gcp:
                mock_tlm.return_value = tlm_dataset
                mock_sci.return_value = sci_dataset
                mock_gcp.return_value = None

                results = mc.loop(
                    config=config,
                    work_dir=self.tmp_dir,
                    tlm_sci_gcp_sets=[
                        # These are meaningless since patch always returns the same data.
                        ("tlm_fn1.fake", "sci_fn1.fake", "gcp_fn1.fake"),
                        ("tlm_fn1.fake", "sci_fn1.fake", "gcp_fn1.fake"),  # Fake a second set (same data)
                    ]
                )

        assert len(results) == 4  # 2 input sets x 2 param sets.

        # Simple validation...
        exp_data = pd.read_csv(self.data_dir / "geo_outvar_5a_spice2.csv")
        exp_data = exp_data.iloc[250:-250, :]  # Specific to 5a.

        out_data = results[0][1]
        lat_diff = exp_data['LLH_ter_pix2_lat'] - out_data['latitude'].sel(spatial_pixel=2).values
        lon_diff = exp_data['LLH_ter_pix2_lon'] - out_data['longitude'].sel(spatial_pixel=2).values
        assert np.abs(lat_diff).max() < 5.1e-5
        assert np.abs(lon_diff).max() < 2.0e-4
