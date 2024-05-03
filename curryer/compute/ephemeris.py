"""Ephemeris related routines.

Routine Listings
----------------
SolarDistAndDoppler(instrument, mission=None, kernels=None)
    Class for working with TIM solar distance and doppler data.

Examples
--------
>>> ugps_range = (1135641617000000, 1135728017000000)  # 2016-01-01 & 2016-01-02
>>> sdad = SolarDistAndDoppler('tcte', mission='tcte')
>>> table = sdad.get_corrections(ugps_range)
>>> print(table.head())

@author: Brandon Stone
"""
import logging

import numpy as np
import pandas as pd

from . import constants, abstract
from .. import spicierpy


logger = logging.getLogger(__name__)


class SolarDistAndDoppler(abstract.AbstractMissionData):
    """Solar distance and Doppler server.
    """
    DEFAULT_CADENCE = constants.EPHEMERIS_TIMESTEP_USEC

    def __init__(self, observer, microsecond_cadence=None):
        """Define how to generate SDD data.

        Parameters
        ----------
        observer : str or int
            SPICE ID or name for the instrument or spacecraft.
        microsecond_cadence : int, optional
            Cadence of computed measurements in microseconds.

        """
        microsecond_cadence = self.DEFAULT_CADENCE if microsecond_cadence is None else microsecond_cadence
        super().__init__(microsecond_cadence=microsecond_cadence)
        self.observer = observer

        # Ephemeris columns used in calculations.
        self.position_columns = ['x', 'y', 'z']
        self.velocity_columns = ['vx', 'vy', 'vz']

    def _dist_correction(self, state):
        """TIM distance correction.

        Definition:
            (R_1au / R_actual)^2
        where
            R_1au = KM per 1AU (constant)
            R_actual = KM between target and observer

        Parameters
        ----------
        state : pandas.DataFrame
            Table of ephemeris state data. Columns: [x, y, z, vx, vy, vz].
            Units must be KM & KM/Sec.

        Returns
        -------
        numpy.ndarray
            Array of distance correction factors.

        """
        ax = int(state.ndim == 2)
        dist = np.linalg.norm(state[self.position_columns], axis=ax)
        return (constants.KM_PER_ASTRONOMICAL_UNIT / dist) ** 2

    def _doppler_correction(self, state):
        """Doppler correction.

        Definition:
            (1 - (R_vel / c))
        where
            R_vel = Radial velocity (KM/sec)
                dot(velocity, position) / norm(position)
            c = Speed of light (vacuum; constant; KM/sec)

        Parameters
        ----------
        state : pandas.DataFrame
            Table of ephemeris state data. Columns: [x, y, z, vx, vy, vz].
            Units must be KM & KM/Sec.

        Returns
        -------
        numpy.ndarray
            Array of doppler correction factors.

        """
        ax = int(state.ndim == 2)
        r_vel = (
                np.sum(state[self.velocity_columns].values * state[self.position_columns].values, axis=ax)
                / np.linalg.norm(state[self.position_columns], axis=ax)
        )
        return 1 - (r_vel / constants.SPEED_OF_LIGHT_KM_PER_S)

    @abstract.log_return()
    def get_corrections(self, ugps_times):
        """Generate a table of the SDD corrections.
        """
        logger.debug('Creating corrections table with [%i] rows', len(ugps_times))
        table = pd.DataFrame(
            index=pd.Index(ugps_times, name='microsecondssincegpsepoch'),
            columns=['sunobserverdopplerfactor', 'sunearthdopplerfactor',
                     'sunobserverdistancecorrection', 'sunearthdistancecorrection'],
        )

        # Query the ephemeris data for corrections for instrument to sun.
        sunobs = spicierpy.ext.query_ephemeris(
            ugps_times=ugps_times,
            target=spicierpy.obj.Body('SUN'),
            observer=spicierpy.obj.Body(self.observer),
            ref_frame='IAU_SUN',
            allow_nans=self.allow_nans,
            velocity=True,
        )

        # Return early if no kernel data exists.
        sunobs.dropna(inplace=True)
        if sunobs.size == 0:
            logger.warning('[0/%i] times had compete ephemeris data; returning table with all NaNs.',
                           len(ugps_times))
            return table

        table.loc[sunobs.index, 'sunobserverdistancecorrection'] = self._dist_correction(sunobs)
        table.loc[sunobs.index, 'sunobserverdopplerfactor'] = self._doppler_correction(sunobs)

        # Corrections for earth to sun.
        sunearth = spicierpy.ext.query_ephemeris(
            ugps_times=sunobs.index.values,  # Subset times with data.
            target=spicierpy.obj.Body('SUN'),
            observer=spicierpy.obj.Body('EARTH'),
            ref_frame='IAU_SUN',
            allow_nans=self.allow_nans,
            velocity=True,
        )
        sunearth.dropna(inplace=True)

        table.loc[sunearth.index, 'sunearthdistancecorrection'] = self._dist_correction(sunearth)
        table.loc[sunearth.index, 'sunearthdopplerfactor'] = self._doppler_correction(sunearth)

        return table

    # @abstract.log_return(max_rows=3)
    # def _get_static_table(self, ugps_times):
    #     """Generate a table of SDD metadata.
    #     """
    #     logger.debug('Creating static table with [%i] rows', len(ugps_times))
    #     table = pd.DataFrame(
    #         OrderedDict([('instrumentmodeid', self.instrumentmodeid),
    #                      ('version', self.version),
    #                      ('predictivedata', 0)]),
    #         index=pd.Index(ugps_times, name='microsecondssincegpsepoch')
    #     )
    #     return table
    #
    # @abstract.log_return(max_rows=10)
    # def get_data(self, ugps_range):
    #     logger.info('Getting available distance and doppler data in range [%s]', ugps_range)
    #     ugps_times = self._get_times(ugps_range)
    #     table = self.get_corrections(ugps_times).dropna()
    #     idx_name = table.index.name
    #     table = table.join(self._get_static_table(ugps_times), how='inner')
    #     table.index.name = idx_name
    #     return table
