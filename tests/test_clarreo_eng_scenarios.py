import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import xarray as xr

from curryer import utils, spicetime, meta
from curryer import spicierpy as sp
from curryer.compute import pointing, spatial, constants, elevation
from curryer.compute.constants import SpatialQualityFlags as SQF
from curryer.kernels import create


logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.DEBUG, extra_loggers=[__name__])

xr.set_options(display_width=120, display_max_rows=30)
np.set_printoptions(linewidth=120)


class ClarreoEngScenariosTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parent.parent
        self.generic_dir = root_dir / 'data' / 'generic'
        self.data_dir = root_dir / 'data' / 'clarreo'
        self.test_dir = root_dir / 'tests' / 'data' / 'clarreo'
        self.assertTrue(self.generic_dir.is_dir(), self.generic_dir)
        self.assertTrue(self.data_dir.is_dir(), self.data_dir)
        self.assertTrue(self.test_dir.is_dir(), self.test_dir)

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

        self.expected_results_file = self.test_dir / 'GeolocationTestCase1.xlsx'

    def test_case1_kernels_files_high_level(self):
        # Load meta kernel details. Includes existing static kernels.
        mkrn = meta.MetaKernel.from_json(
            self.test_dir / 'cprs_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )

        # Create the dynamic kernels from the JSONs alone. Note that they
        # contain the reference to the input_data netcdf4 file to read.
        generated_kernels = []
        creator = create.KernelCreator(overwrite=False, append=False)

        # ISS Ephemeris in ECF, instead of nominal ECI.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'iss_sc_v01.ephemeris.spk.testcase1.json', output_kernel=self.tmp_dir,
        ))

        # ISS Attitude in ECF, instead of nominal ECI.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'iss_sc_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # HPS Azimuth angle.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'cprs_az_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # HPS Elevation angle.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'cprs_el_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # Geolocate all the individual pixels and create the L1A data product!
        with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, generated_kernels]):
            ugps_times = spicetime.adapt(
                pd.date_range('2023-01-01', '2023-01-01T00:05:00', freq='67ms', inclusive='left'), 'iso')

            t0 = pd.Timestamp.utcnow()

            geoloc_inst = spatial.Geolocate('CPRS_HYSICS')
            l1a_dataset = geoloc_inst(ugps_times)

            t1 = pd.Timestamp.utcnow()
            logger.info('Geoloc+Dataset process time=[%s], count=[%d], granule_est_time=[%s]',
                        t1 - t0, ugps_times.size, (t1 - t0) / ugps_times.size * 4500)

            self.assertIsInstance(l1a_dataset, xr.Dataset)

            # Validate against Engineering expected value (single time).
            lonlats_ds = l1a_dataset.sel(frame=ugps_times[0] / 1e6)
            lonlats_arr = np.stack([lonlats_ds['longitude_ellipsoidal'].values,
                                   lonlats_ds['latitude_ellipsoidal'].values], axis=1)
            self.compare_to_expected(lonlats_arr, atols=(2e-5, 9e-7))

            # Note that terrain correction was NOT applied to the validation
            # dataset, so applying it is expected to increase the error.
            lonlats_arr = np.stack([lonlats_ds['longitude'].values, lonlats_ds['latitude'].values], axis=1)
            self.compare_to_expected(lonlats_arr, atols=(3e-4, 2e-4))

    def test_case1_kernels_files_tlm_input(self):
        # Load meta kernel details. Includes existing static kernels.
        mkrn = meta.MetaKernel.from_json(
            self.test_dir / 'cprs_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )

        # Create the dynamic kernels from the JSONs alone. Note that they
        # contain the reference to the input_data netcdf4 file to read.
        generated_kernels = []
        creator = create.KernelCreator(overwrite=False, append=False)

        # ISS Ephemeris in ECF, instead of nominal ECI.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'iss_sc_v01.ephemeris.spk.testcase1.json', output_kernel=self.tmp_dir,
        ))

        # ISS Attitude in ECF, instead of nominal ECI.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'iss_sc_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # HPS Azimuth angle.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'cprs_az_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # HPS Elevation angle.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'cprs_el_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # Load the kernels so that we can create a prod-like input file.
        with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, generated_kernels]):
            tlm_dataset = self.build_prod_tlm_file()
            # tlm_dataset.to_netcdf(self.test_dir.parent / 'cprs_geolocation_tlm_20230101_20240430.nc')

        # Delete the previously generated kernels.
        for tmp_kernel in generated_kernels:
            tmp_kernel.unlink()
        generated_kernels.clear()

        # Now recreate them from but using the prod kernel definitions and the
        # prod-like telemetry input file. Note the different config dir!
        generated_kernels.append(creator.write_from_json(
            self.data_dir / 'iss_sc_v01.ephemeris.spk.json', output_kernel=self.tmp_dir,
            input_data=tlm_dataset,
        ))
        generated_kernels.append(creator.write_from_json(
            self.data_dir / 'iss_sc_v01.attitude.ck.json', output_kernel=self.tmp_dir,
            input_data=tlm_dataset,
        ))
        generated_kernels.append(creator.write_from_json(
            self.data_dir / 'cprs_az_v01.attitude.ck.json', output_kernel=self.tmp_dir,
            input_data=tlm_dataset,
        ))
        generated_kernels.append(creator.write_from_json(
            self.data_dir / 'cprs_el_v01.attitude.ck.json', output_kernel=self.tmp_dir,
            input_data=tlm_dataset,
        ))

        # Geolocate all the individual pixels and create the L1A data product!
        with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, generated_kernels]):
            ugps_times = spicetime.adapt(
                pd.date_range('2023-01-01', '2023-01-01T00:05:00', freq='67ms', inclusive='left'), 'iso')

            t0 = pd.Timestamp.utcnow()

            geoloc_inst = spatial.Geolocate('CPRS_HYSICS')
            l1a_dataset = geoloc_inst(ugps_times)

            t1 = pd.Timestamp.utcnow()
            logger.info('Geoloc+Dataset process time=[%s], count=[%d], granule_est_time=[%s]',
                        t1 - t0, ugps_times.size, (t1 - t0) / ugps_times.size * 4500)

            self.assertIsInstance(l1a_dataset, xr.Dataset)

            # Validate against Engineering expected value (single time).
            lonlats = l1a_dataset.sel(frame=ugps_times[0] / 1e6)
            lonlats = np.stack([lonlats['longitude_ellipsoidal'].values,
                                lonlats['latitude_ellipsoidal'].values], axis=1)
            self.compare_to_expected(lonlats, atols=(2e-5, 9e-7))

            qf_counts = l1a_dataset['quality_flags'].to_dataframe().value_counts()
            self.assertEqual(qf_counts.size, 4)
            self.assertEqual(qf_counts[SQF.GOOD.value], 2137990)
            self.assertEqual(qf_counts[(SQF.CALC_ANCIL_NOT_FINITE | SQF.CALC_TERRAIN_EXTREME_ZENITH).value], 4091)
            self.assertEqual(qf_counts[(SQF.CALC_ANCIL_NOT_FINITE | SQF.CALC_ELLIPS_NO_INTERSECT).value], 159)
            self.assertEqual(qf_counts[(SQF.CALC_ANCIL_NOT_FINITE | SQF.CALC_ANCIL_INSUFF_DATA
                                        | SQF.CALC_ELLIPS_INSUFF_DATA | SQF.SPICE_ERR_MISSING_ATTITUDE).value], 7200)

    @staticmethod
    def build_prod_tlm_file():
        # Determine timestamps.
        ugps_start = spicetime.adapt('2023-01-01', 'iso', 'ugps')
        cadence = 1  # Hz.
        duration = 5 * 60
        ugps_times = np.arange(0, duration, cadence) * 1e6 + ugps_start
        et_times = spicetime.adapt(ugps_times, 'ugps', 'et')

        sc_body_id = -125544
        sc_frame_id = -125544000
        obs_id = sp.obj.Body('EARTH').id
        ref_frame = 'J2000'
        base_gim_angles = [9.7235, 6.322]  # az,el in degrees.

        position_velocity = []
        sc_attitude = []
        sc_ang_vel = []

        for sample_et in et_times:
            # Query ephemeris.
            pos_vel, _ = sp.spkezr(sc_body_id, sample_et, ref=ref_frame, abcorr='NONE', obs=obs_id)
            position_velocity.append(pos_vel)

            # Query attitude.
            sample_sct = sp.sce2c(sc_body_id, sample_et)  # Aka ugps.
            # att_mat, _ = sp.ckgp(sc_frame_id, sample_sct, 0.0, ref_frame)
            att_mat, att_av, _ = sp.ckgpav(sc_frame_id, sample_sct, 0.0, ref_frame)
            att_quat = sp.m2q(att_mat)
            sc_attitude.append(att_quat)
            sc_ang_vel.append(att_av)

        position_velocity = np.vstack(position_velocity)
        sc_attitude = np.vstack(sc_attitude)
        sc_ang_vel = np.vstack(sc_ang_vel)

        # Create fake gimbal angles.
        gim_shift = np.linspace(0.0, 45.0, ugps_times.size // 2)
        gim_angles = np.zeros((ugps_times.size, 2))
        for ith, ang in enumerate(base_gim_angles):
            gim_angles[:, ith] += ang
            gim_angles[:gim_shift.size, ith] += gim_shift * (-1 if ith else 1)
            gim_angles[-gim_shift.size:, ith] += gim_shift[::-1] * (-1 if ith else 1)

        # Convert KM to feet, deg to rad.
        position_velocity *= 1e3 / 0.3048
        gim_angles = np.deg2rad(gim_angles)

        # Create dataset.
        dataset = xr.Dataset(
            {
                'hps.bad_pos_eci_x': ('frame', position_velocity[:, 0]),
                'hps.bad_pos_eci_y': ('frame', position_velocity[:, 1]),
                'hps.bad_pos_eci_z': ('frame', position_velocity[:, 2]),

                'hps.bad_vel_eci_x': ('frame', position_velocity[:, 3]),
                'hps.bad_vel_eci_y': ('frame', position_velocity[:, 4]),
                'hps.bad_vel_eci_z': ('frame', position_velocity[:, 5]),

                'hps.bad_quat_iss_eci_s': ('frame', sc_attitude[:, 0]),
                'hps.bad_quat_iss_eci_i': ('frame', sc_attitude[:, 1]),
                'hps.bad_quat_iss_eci_j': ('frame', sc_attitude[:, 2]),
                'hps.bad_quat_iss_eci_k': ('frame', sc_attitude[:, 3]),

                'hps.bad_rate_iss_x': ('frame', sc_ang_vel[:, 0]),
                'hps.bad_rate_iss_y': ('frame', sc_ang_vel[:, 1]),
                'hps.bad_rate_iss_z': ('frame', sc_ang_vel[:, 2]),

                'hps.az_ang_nonlin': ('frame', gim_angles[:, 0]),
                'hps.el_ang_nonlin': ('frame', gim_angles[:, 1]),
            },
            coords={
                'corrected_timestamp': ('frame', ugps_times / 1e6),
            },
            attrs={
                'creation_date': pd.Timestamp.utcnow().isoformat(),
                'source': ('GeolocationTestCase1.xlsx, SPICE conversions from ECEF to ECI'
                           ', made-up gimbal angles between first and last time'),
                'units': 'time=seconds, distance=feet, angle=radians, quaternion=cxyz, epoch=GPS',
            }
        )
        logger.info('Created telemetry dataset:\n%s', dataset)
        return dataset

    def test_case1_kernels_files_low_level(self):
        # Load meta kernel details. Includes existing static kernels.
        mkrn = meta.MetaKernel.from_json(
            self.test_dir / 'cprs_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )

        # Create the dynamic kernels from the JSONs alone. Note that they
        # contain the reference to the input_data netcdf4 file to read.
        generated_kernels = []
        creator = create.KernelCreator(overwrite=False, append=False)

        # ISS Ephemeris in ECF, instead of nominal ECI.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'iss_sc_v01.ephemeris.spk.testcase1.json', output_kernel=self.tmp_dir,
        ))

        # ISS Attitude in ECF, instead of nominal ECI.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'iss_sc_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # HPS Azimuth angle.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'cprs_az_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # HPS Elevation angle.
        generated_kernels.append(creator.write_from_json(
            self.test_dir / 'cprs_el_v01.attitude.ck.testcase1.json', output_kernel=self.tmp_dir
        ))

        # Geolocate all the individual pixel!
        with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, generated_kernels]):
            # ugps_times = spicetime.adapt(['2023-01-01'], 'iso')
            ugps_times = spicetime.adapt(
                pd.date_range('2023-01-01', '2023-01-01T00:05:00', freq='67ms', inclusive='left'), 'iso')

            t0 = pd.Timestamp.utcnow()
            geoloc, sc_data, sqf = spatial.instrument_intersect_ellipsoid(
                ugps_times, 'CPRS_HYSICS', geodetic=True, degrees=True)

            t1 = pd.Timestamp.utcnow()
            logger.info('Geolocation process time=[%s], count=[%d], granule_est_time=[%s]',
                        t1 - t0, ugps_times.size, (t1 - t0) / ugps_times.size * 4500)

            self.assertIsInstance(geoloc, pd.DataFrame)
            self.assertTupleEqual(geoloc.shape, (480 * ugps_times.size, 3))
            self.assertTrue(np.isfinite(geoloc.values).all())

            lonlats = geoloc.loc[ugps_times[0], 'lon':'lat'].values
            self.compare_to_expected(lonlats, atols=(2e-5, 2e-6))
            self.compare_to_expected(geoloc.values[:, :2], atols=(2e-5, 2e-6))

            ec_srf_pos = spatial.geodetic_to_ecef(geoloc.values, degrees=True)
            ec_sat_pos = sc_data.values

            # Apply terrain correction!
            t2 = pd.Timestamp.utcnow()
            elev = elevation.Elevation(meters=False, degrees=False)
            elev_region = elev.local_region(*np.deg2rad((0, 5, 50, 55)))

            local_minmax = elev_region.local_minmax()
            corr_srf_lla, sqf = spatial.terrain_correct(
                elev=elev_region,
                ec_srf_pos=ec_srf_pos,
                ec_sat_pos=ec_sat_pos,
                local_minmax=local_minmax,
            )

            t3 = pd.Timestamp.utcnow()
            logger.info('Terrain-correction process time=[%s], shape=[%d], granule_est_time=[%s]',
                        t3 - t2, ugps_times.size, (t3 - t2) / ugps_times.size * 4500)
            self.assertIsInstance(corr_srf_lla, np.ndarray)
            self.assertTupleEqual(corr_srf_lla.shape, (480 * ugps_times.size, 3))
            self.assertTrue(np.isfinite(corr_srf_lla).all())

            # Note that terrain correction was NOT applied to the validation
            # dataset, so applying it is expected to increase the error.
            self.compare_to_expected(corr_srf_lla[:, :2], atols=(3e-4, 2e-4))

    def compare_to_expected(self, lonlats, atols):
        # Compare to the expected lon/lat intersections.
        expect = pd.read_excel(self.expected_results_file, sheet_name='Intersections')
        expect = expect[[expect.columns[2], expect.columns[1]]].iloc[1:, :].to_numpy(np.float64)
        self.assertTupleEqual(expect.shape, (480, 2))

        if lonlats.shape[0] > expect.shape[0]:
            error = expect - lonlats.reshape((-1,) + expect.shape)
            max_lonlat = np.max(np.abs(error), axis=(0, 1))
            npt.assert_allclose(error[..., 0], 0.0, atol=atols[0])
            npt.assert_allclose(error[..., 1], 0.0, atol=atols[1])
        else:
            error = expect - lonlats
            max_lonlat = np.max(np.abs(error), axis=0)
            npt.assert_allclose(expect[:, 0], lonlats[:, 0], atol=atols[0], rtol=0)
            npt.assert_allclose(expect[:, 1], lonlats[:, 1], atol=atols[1], rtol=0)
        logger.info('Max Lon/Lat error: %f, %f', max_lonlat[0], max_lonlat[1])

    def test_case1_kernels_manual(self):
        mkrn = meta.MetaKernel.from_json(
            self.data_dir / 'cprs_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )
        for fn in mkrn.mission_kernels:
            if fn.suffix in ('.bsp', '.bc'):
                mkrn.mission_kernels.remove(fn)

        fixed_kernel_configs = [
            self.data_dir / "cprs_base_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_pede_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_az_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_yoke_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_el_v01.fixed_offset.spk.json",
            self.data_dir / "cprs_hysics_v01.fixed_offset.spk.json",
        ]
        pos_kernel_config = self.test_dir / 'iss_sc_v01.ephemeris.spk.testcase1.json'
        att_kernel_config = self.test_dir / 'iss_sc_v01.attitude.ck.testcase1.json'
        az_kernel_config = self.test_dir / 'cprs_az_v01.attitude.ck.testcase1.json'
        el_kernel_config = self.test_dir / 'cprs_el_v01.attitude.ck.testcase1.json'

        creator = create.KernelCreator(overwrite=False, append=False)

        # Static offsets.
        generated_kernels = []
        for kernel_config_file in fixed_kernel_configs:
            self.assertTrue(kernel_config_file.is_file(), kernel_config_file)
            generated_kernels.append(creator.write_from_json(
                kernel_config_file, output_kernel=self.tmp_dir
            ))

        # ISS Ephemeris in ECF, instead of nominal ECI.
        ephem_data = pd.DataFrame({
            "ugps": [0, 3155760018000000],
            "position_x": [4206007.302, 4206007.302],
            "position_y": [149.929072, 149.929072],
            "position_z": [5308768.616, 5308768.616],
        })
        generated_kernels.append(creator.write_from_json(
            pos_kernel_config, output_kernel=self.tmp_dir, input_data=ephem_data,
        ))

        # ISS Attitude in ECF, instead of nominal ECI.
        att_iss_data = pd.DataFrame({
            "ugps": [1198800018000000, 2208643218000000],  # 2018 - 2050.
            "quaternion_c": [0.283828613, 0.283828613],  # sp.m2q((eci2ecf @ iss2eci).T)
            "quaternion_x": [0.643671875, 0.643671875],
            "quaternion_y": [0.688350639, 0.688350639],
            "quaternion_z": [-0.176921546, -0.176921546],
        })
        generated_kernels.append(creator.write_from_json(
            att_kernel_config, output_kernel=self.tmp_dir, input_data=att_iss_data,
        ))

        # HPS Azimuth angle.
        att_az_data = pd.DataFrame({
            "ugps": [1198800018000000, 2208643218000000],  # 2018 - 2050.
            "angle_y": [np.deg2rad(9.7235), np.deg2rad(9.7235)],
        })
        generated_kernels.append(creator.write_from_json(
            az_kernel_config, output_kernel=self.tmp_dir, input_data=att_az_data,
        ))

        # HPS Elevation angle.
        att_el_data = pd.DataFrame({
            "ugps": [1198800018000000, 2208643218000000],  # 2018 - 2050.
            "angle_x": [np.deg2rad(6.322), np.deg2rad(6.322)],
        })
        generated_kernels.append(creator.write_from_json(
            el_kernel_config, output_kernel=self.tmp_dir, input_data=att_el_data,
        ))

        for fn in sorted(self.tmp_dir.glob('*')):
            logger.info(fn)

        with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, generated_kernels]):
            ugps_times = spicetime.adapt(np.array(['2023-01-01']), 'iso')
            et_times = spicetime.adapt(ugps_times, 'ugps', 'et')

            # Throws an error if there is any disconnect.
            # Assert all have `sp.det` around 1 (aka valid rot matrix)
            t1 = sp.pxform('CPRS_HYSICS_COORD', 'CPRS_EL_COORD', et_times[0])
            npt.assert_allclose(sp.det(t1), 1)
            t2 = sp.pxform('CPRS_EL_COORD', 'CPRS_YOKE_COORD', et_times[0])
            npt.assert_allclose(sp.det(t2), 1)
            t3 = sp.pxform('CPRS_YOKE_COORD', 'CPRS_AZ_COORD', et_times[0])
            npt.assert_allclose(sp.det(t3), 1)
            t4 = sp.pxform('CPRS_AZ_COORD', 'CPRS_PEDE_COORD', et_times[0])
            npt.assert_allclose(sp.det(t4), 1)
            t5 = sp.pxform('CPRS_PEDE_COORD', 'CPRS_BASE_COORD', et_times[0])
            npt.assert_allclose(sp.det(t5), 1)
            t6 = sp.pxform('CPRS_BASE_COORD', 'ISS_ISSACS', et_times[0])
            npt.assert_allclose(sp.det(t6), 1)

            t16 = sp.pxform('CPRS_HYSICS_COORD', 'ISS_ISSACS', et_times[0])
            npt.assert_allclose(sp.det(t16), 1)
            t60 = sp.pxform('ISS_ISSACS', 'ITRF93', et_times[0])
            npt.assert_allclose(sp.det(t60), 1)

            # Earth is in the FOV.
            chk = pointing.check_fov(ugps_times, 'CPRS_HYSICS', 'EARTH')
            self.assertEqual(chk.iloc[0], 1)

            # Pointing at the earth (degrees from center).
            dot = pointing.boresight_dot_object('CPRS_HYSICS', 'EARTH', ugps_times)
            self.assertLessEqual(np.rad2deg(np.arccos(dot)), 18)

            # Attempt to geolocate!
            # geoloc = spatial.geolocate(ugps_times, 'CPRS_HYSICS')
            geoloc, sc_pos, sqf = spatial.instrument_intersect_ellipsoid(
                ugps_times, 'CPRS_HYSICS', geodetic=True, degrees=True)
            calc_lon = geoloc['lon'][ugps_times[0]]
            calc_lat = geoloc['lat'][ugps_times[0]]

            # Expected for b240 (row 243):
            expt_lon = 1.693510638
            expt_lat = 52.27591878
            logger.info('Expected range:\n\tLon: %10.6f, %10.6f\n\tLat: %10.6f, %10.6f',
                        1.659132592, 1.727539955, 51.94185669, 52.61839297)

            err_lon = expt_lon - calc_lon
            err_lat = expt_lat - calc_lat
            logger.info('Geolocation (expected, computed, error):'
                        '\n\tLon: %10.6f, %10.6f (%.6f)'
                        '\n\tLat: %10.6f, %10.6f (%.6f)',
                        expt_lon, calc_lon, err_lon,
                        expt_lat, calc_lat, err_lat)

            # Misc...
            # sp.ext.query_ephemeris(ugps_times, 'ISS_SC', 'EARTH', 'ITRF93')
            # sp.ext.query_ephemeris(ugps_times, 'CPRS_BASE', 'ISS_SC', 'ISS_ISSACS')

            npix = 480

            # Load the individual pixel vectors.
            vectors = pd.read_excel(self.expected_results_file, sheet_name='PixelVectors')
            vectors = vectors.iloc[:, 1:4].values.copy()  # Remove stride...
            self.assertTupleEqual(vectors.shape, (npix, 3))
            lonlats = np.zeros((npix, 2))

            # Geolocate each individual pixel.
            for i in range(npix):
                geoloc, _, _ = spatial.instrument_intersect_ellipsoid(
                    ugps_times, 'CPRS_HYSICS', boresight_vector=vectors[i, :], geodetic=True, degrees=True)
                lonlats[i, :] = geoloc.iloc[0, :2].values

            # Compare to the expected lon/lat intersections.
            expect = pd.read_excel(self.expected_results_file, sheet_name='Intersections')
            expect = expect[[expect.columns[2], expect.columns[1]]].iloc[1:, :].to_numpy(np.float64)
            self.assertTupleEqual(expect.shape, (npix, 2))

            error = expect - lonlats
            max_lonlat = np.max(np.abs(error), axis=0)
            logger.info('Max Lon/Lat error: %f, %f', max_lonlat[0], max_lonlat[1])
            self.assertLessEqual(max_lonlat[0], 2e-5)
            self.assertLessEqual(max_lonlat[1], 2e-6)

            # Estimate across-track GSD...
            delta = np.sqrt(((expect[1:, :] - expect[:-1, :]) ** 2).sum(axis=1))
            m_per_deg = 2 * np.pi * constants.WGS84_SEMI_MAJOR_AXIS_KM / 360 * 1e3
            self.assertTrue(160 <= delta.max() * m_per_deg <= 165)


if __name__ == '__main__':
    unittest.main()
