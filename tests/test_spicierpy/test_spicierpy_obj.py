"""obj - Unit test

@author: Brandon Stone
"""
import re
import logging
import unittest
from unittest.mock import MagicMock
from pathlib import Path

from spiceypy.utils.exceptions import SpiceyError

from curryer import utils, spicierpy, meta
from curryer.spicierpy import obj


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class ObjTestCase(unittest.TestCase):
    def setUp(self):
        root_dir = Path(__file__).parents[2]
        self.generic_dir = root_dir / 'data' / 'generic'
        self.test_dir = root_dir / 'tests' / 'data' / 'tsis1'
        self.assertTrue(self.generic_dir.is_dir())
        self.assertTrue(self.test_dir.is_dir())

        self.mkrn = meta.MetaKernel.from_json(
            self.test_dir / 'tsis_v01.kernels.tm.json',
            relative=True, sds_dir=self.generic_dir,
        )
        self.kernels = [self.mkrn.sds_kernels, self.mkrn.mission_kernels]

        # Load the SPICE test kernels.
        self._loaded_kernels = spicierpy.ext.load_kernel([
            self.kernels
        ])
        self.addCleanup(self._loaded_kernels.unload, clear=True)

    def test_obj_body_of_a_planet(self):
        for id_or_name in [399, 'earth', 'EARTH', ' EartH   ']:
            with self.subTest(arg=id_or_name):
                body = obj.Body(id_or_name)
                self.assertEqual(399, body.id)
                self.assertEqual('EARTH', body.name)
                self.assertEqual('Body(EARTH)', str(body))

                with self.assertRaises(ValueError) as raised:
                    _ = body.frame
                self.assertIn('Body(EARTH) does not have an attached \'Frame\'', raised.exception.args[0])

    def test_obj_frame_of_a_planet(self):
        for id_or_name in [13000, 'itrf93']:
            with self.subTest(arg=id_or_name):
                frame = obj.Frame(id_or_name)
                self.assertEqual(13000, frame.id)
                self.assertEqual('ITRF93', frame.name)
                self.assertEqual('Frame(ITRF93)', str(frame))

    def test_obj_frame_define_not_defined_passes(self):
        frame = obj.Frame.define('fake', -999999)
        self.assertEqual(-999999, frame.id)
        self.assertEqual('fake', frame.name)
        self.assertEqual('Frame(fake)', str(frame))

    def test_obj_body_and_frame_of_a_planet(self):
        # Note: This uses the built-in default mappings.
        for id_or_name in [10, 'sun']:
            with self.subTest(arg=id_or_name):
                body = obj.Body(id_or_name, frame=True)
                self.assertEqual(10, body.id)
                self.assertEqual('SUN', body.name)

                frame = body.frame
                self.assertIsInstance(frame, obj.Frame)
                self.assertEqual(10010, frame.id)
                self.assertEqual('IAU_SUN', frame.name)
                self.assertIs(frame, body.frame)

                body_again = frame.body
                self.assertEqual(10, body_again.id)
                self.assertEqual('SUN', body_again.name)
                self.assertEqual(body, body_again, 'although different instances, they should equate')

    def test_obj_body_define_or_change_values(self):
        with self.assertRaisesRegex(SpiceyError, r'not found'):
            _ = spicierpy.gipool('NAIF_BODY_CODE', 0, 2000)
        with self.assertRaisesRegex(SpiceyError, r'not found'):
            _ = spicierpy.gcpool('NAIF_BODY_NAME', 0, 2000)

        # Define a new name <--> id mapping.
        body = obj.Body.define('fake_name', -9999)
        self.assertEqual(-9999, body.id)
        self.assertEqual('FAKE_NAME', body.name)

        # Now we can use the name without redefining the mapping.
        body2 = obj.Body(-9999)
        self.assertEqual(-9999, body2.id)
        self.assertEqual('FAKE_NAME', body2.name, 'Name should be this after we defined the mapping')

        # Can not change the ID or name values!
        with self.assertRaises(AttributeError):
            body.id = -9998
        with self.assertRaises(AttributeError):
            body.name = '-9997'

    def test_obj_body_and_frame_of_a_spacecraft(self):
        # Note: This uses the custom mappings defined in the ISS_SC kernels.
        for id_or_name in [-125544, 'iss_sc']:
            with self.subTest(arg=id_or_name):
                body = obj.Body(id_or_name, frame=True)
                self.assertEqual(-125544, body.id)
                self.assertEqual('ISS_SC', body.name)

                frame = body.frame
                self.assertIsInstance(frame, obj.Frame)
                self.assertEqual(-125544000, frame.id)
                self.assertEqual('ISS_ISSACS', frame.name)

                body_again = frame.body
                self.assertEqual(-125544, body_again.id)
                self.assertEqual('ISS_SC', body_again.name)

    def test_obj_spacecraft_specified_attr(self):
        frame = obj.Frame('iau_sun')
        clock = obj.Clock(-9999)  # Fake values...
        ephemeris = obj.Ephemeris(-9999)
        attitude = obj.Attitude(-9999000)
        instrument = obj.Instrument(-9999001)

        sc = obj.Spacecraft(-9999, frame=frame, clock=clock, ephemeris=ephemeris, attitude=attitude,
                            instruments=[instrument])

        self.assertIs(frame, sc.frame)
        self.assertIs(clock, sc.clock)
        self.assertIs(ephemeris, sc.ephemeris)
        self.assertIs(attitude, sc.attitude)
        self.assertIn(instrument.id, sc.instruments)
        self.assertIs(instrument, sc.instruments[instrument.id])

    def test_obj_instrument(self):
        for id_or_name in [-125544101, 'tsis_tim_glint']:
            with self.subTest(arg=id_or_name):
                instrument = obj.Instrument(id_or_name, frame=True, spacecraft=True)
                self.assertEqual(-125544101, instrument.id)
                self.assertEqual('TSIS_TIM_GLINT', instrument.name)

                # Instrument spacecraft.
                self.assertIsInstance(instrument.spacecraft, obj.Spacecraft)
                self.assertEqual(-125544, instrument.spacecraft.id)
                self.assertEqual('ISS_SC', instrument.spacecraft.name)
                self.assertIs(instrument, instrument.spacecraft.instruments[instrument.id])

                # Instrument frame.
                self.assertIsInstance(instrument.frame, obj.Frame)
                # Uses the same frame as the non-glint, hence the ID.
                self.assertEqual(-125544100, instrument.frame.id)
                self.assertEqual('TSIS_TIM_COORD', instrument.frame.name)

                # Pretty print.
                txt = instrument.to_string(verbose=True)
                self.assertIsNotNone(re.search(
                    r'^Instrument\(.+\)$\s+Frame\(.+\)$', txt, re.MULTILINE
                ))
                self.assertEqual('Instrument(TSIS_TIM_GLINT)', instrument.to_string(verbose=False))

    def test_obj_clock(self):
        for id_or_name in [-125544, 'iss_sc']:
            with self.subTest(arg=id_or_name):
                clock = obj.Clock(id_or_name, spacecraft=True)
                self.assertEqual(-125544, clock.id)
                self.assertEqual('ISS_SC', clock.name)

                self.assertIsInstance(clock.spacecraft, obj.Spacecraft)
                self.assertEqual(-125544, clock.spacecraft.id)
                self.assertEqual('ISS_SC', clock.spacecraft.name)
                self.assertIs(clock, clock.spacecraft.clock)

    def test_obj_ephemeris(self):
        for id_or_name in [-125544, 'iss_sc']:
            with self.subTest(arg=id_or_name):
                ephemeris = obj.Ephemeris(id_or_name, spacecraft=True)
                self.assertEqual(-125544, ephemeris.id)
                self.assertEqual('ISS_SC', ephemeris.name)

                self.assertIsInstance(ephemeris.spacecraft, obj.Spacecraft)
                self.assertEqual(-125544, ephemeris.spacecraft.id)
                self.assertEqual('ISS_SC', ephemeris.spacecraft.name)
                self.assertIs(ephemeris, ephemeris.spacecraft.ephemeris)

    def test_obj_attitude(self):
        # Note: The CK-ID (attitude), doesn't generally get named. Remember,
        #   it's not a frame id, that's a different system!
        for id_or_name in [-125544000, '-125544000']:
            with self.subTest(arg=id_or_name):
                attitude = obj.Attitude(id_or_name, spacecraft=True)
                self.assertEqual(-125544000, attitude.id)
                self.assertEqual('-125544000', attitude.name)

                self.assertIsInstance(attitude.spacecraft, obj.Spacecraft)
                self.assertEqual(-125544, attitude.spacecraft.id)
                self.assertEqual('ISS_SC', attitude.spacecraft.name)
                self.assertIs(attitude, attitude.spacecraft.attitude)

    def test_obj_fails_non_spacecraft_and_attitude(self):
        # This should be an instrument, not a spacecraft.
        bad_sc = obj.Spacecraft('tsis_tim_glint', frame=True)
        self._loaded_kernels.unload(clear=True)
        with self.assertRaisesRegex(SpiceyError, r'not found'):
            bad_sc.attitude = True  # Tell it to infer...

    def test_not_implemented_api(self):
        with self.assertRaises(NotImplementedError):
            obj.AbstractObj.define(None, None)
        with self.assertRaises(NotImplementedError):
            obj.AbstractObj._name2code(MagicMock(), None)
        with self.assertRaises(NotImplementedError):
            obj.AbstractObj._code2name(MagicMock(), None)

        with self.assertRaises(NotImplementedError):
            obj._SpacecraftItem._infer_spacecraft(MagicMock(), None)

    def test_init_with_existing_mapping_obj(self):
        sc = obj.Spacecraft('iss_sc')
        sc_instr = sc.get_instrument('tsis_tim_glint')
        new_instr = obj.Instrument(sc_instr)
        self.assertEqual(sc_instr.id, new_instr.id)
        self.assertEqual(sc_instr.name, new_instr.name)
        self.assertIsNot(sc_instr, new_instr, 'should be a new instance with matching id and name')
        with self.assertRaises(ValueError) as raised:
            _ = new_instr.spacecraft
        self.assertIn('Instrument(TSIS_TIM_GLINT) does not have an attached \'Spacecraft\'', raised.exception.args[0])

        # Fake example showing that specifying the spacecraft keyword will
        #   override an existing spacecraft reference.
        earth = obj.Body('earth')
        alt_instr = obj.Instrument(sc_instr, spacecraft=earth)
        self.assertIsNot(sc_instr, alt_instr)
        self.assertEqual(sc_instr.id, alt_instr.id)
        self.assertEqual(sc_instr.name, alt_instr.name)
        self.assertNotEqual(sc_instr.spacecraft, alt_instr.spacecraft, 'spacecraft should not match')
        self.assertEqual(earth, alt_instr.spacecraft)


if __name__ == '__main__':
    unittest.main()
