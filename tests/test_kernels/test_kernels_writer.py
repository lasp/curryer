"""writer - Unit test

@author: Brandon Stone
"""
import logging
import os
import unittest
from io import StringIO

from curryer.kernels import writer
from curryer import utils


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])


class WriterSpkTestCase(unittest.TestCase):
    def setUp(self):
        self.spk_configs = {
            'INPUT_DATA_TYPE': 'STATES',
            'OUTPUT_SPK_TYPE': 13,
            'OBJECT_NAME': 'test_ephem',
            'CENTER_NAME': 'Earth',
            'REF_FRAME_NAME': 'J2000',
            'PRODUCER_ID': 'LASP_SDC_TEAM',
            'APPEND_TO_OUTPUT': 'NO',
            'IGNORE_FIRST_LINE': 1,
            'LINES_PER_RECORD': 1,
            'DATA_DELIMITER': ',',
            'POLYNOM_DEGREE': 3,
            'TIME_WRAPPER': '# UTC',
            'DATA_ORDER': 'epoch x y z vx vy vz',
            'INPUT_DATA_UNITS': ['ANGLES=RADIANS', 'DISTANCES=METERS'],
            'LEAPSECONDS_FILE': '%(path)s/data/leapsecond_kernel.tls',
            'none_value_ignored': None,
            'PCK_FILE': [
                '%(path)s/data/pck00010.tpc',
                '%(path)s/data/earth_070425_370426_predict.bpc',
                '%(path)s/data/earth_720101_070426.bpc'
            ],
            'version': 1
        }
        self.naif_body = {
            'name': 'spk_test',
            'code': '-999',  # Forced to be an int
        }

    def test_writer_assume_templates_dir_exists(self):
        filename = os.path.realpath(os.path.join(writer.__file__, '..', 'templates'))
        self.assertTrue(os.path.isdir(filename))

    def test_write_setup_return_str(self):
        txt = writer.write_setup(
            setup_file=None,
            template='spk',
            mappings=self.naif_body,
            configs=self.spk_configs
        )
        self.assertIn('NAIF_BODY_CODE += ( -999 )', txt)
        self.assertIn("NAIF_BODY_NAME += ( 'spk_test' )", txt)

    def test_write_setup_write_to_obj(self):
        setup_file = StringIO()
        r = writer.write_setup(
            setup_file=setup_file,
            template='spk',
            mappings=self.naif_body,
            configs=self.spk_configs
        )
        setup_file.seek(0)
        txt = setup_file.read()

        self.assertIsNone(r)
        self.assertIn('NAIF_BODY_CODE += ( -999 )', txt)
        self.assertIn("NAIF_BODY_NAME += ( 'spk_test' )", txt)


class WriterCkTestCase(unittest.TestCase):
    def setUp(self):
        self.ck_configs = {
            'INSTRUMENT_ID': -999,
            'CK_TYPE': 3,
            'INPUT_DATA_TYPE': 'SPICE QUATERNIONS',
            'INPUT_TIME_TYPE': 'UTC',
            'REFERENCE_FRAME_NAME': 'J2000',
            'ANGULAR_RATE_PRESENT': 'NO',
            'LSK_FILE_NAME': '{root.properties.kernel_path}/lsk/naif0012.tls',
            'MAKE_FAKE_SCLK': '{root.properties.path}/orient_fake_clock.sclk.tsc',
            'PRODUCER_ID': 'LASP_SDS_TEAM'
        }
        self.naif_body = {
            'name': 'ck_test',
            'code': '-999',  # Forced to be an int
        }

    def test_write_setup_return_str(self):
        txt = writer.write_setup(
            setup_file=None,
            template='ck',
            mappings=self.naif_body,
            configs=self.ck_configs
        )
        self.assertIn('NAIF_BODY_CODE += ( -999 )', txt)
        self.assertIn("NAIF_BODY_NAME += ( 'ck_test' )", txt)


class ValidateTestCase(unittest.TestCase):
    """Test validation of a setup file.

    Dependencies
    ------------
    - Test data files:
        - ephem_setup.txt

    """

    def test_line_length_pass(self):
        txt = 'Line1\n' + ('X' * 132) + '\nLine3\n'
        self.assertTrue(writer.validate_text_kernel(StringIO(txt)))

    def test_line_length_fail(self):
        txt = 'Line1\n' + ('X' * 133) + '\nLine3\n'
        self.assertFalse(writer.validate_text_kernel(StringIO(txt)))

    def test_string_length_pass(self):
        txt = 'Line1\n' + 'VARIABLE = \'' + ('S' * 80) + '\'\n' + 'Line3\n'
        self.assertTrue(writer.validate_text_kernel(StringIO(txt)))

    def test_string_length_fail(self):
        txt = 'Line1\n' + 'VARIABLE = \'' + ('S' * 81) + '\'\n' + 'Line3\n'
        self.assertFalse(writer.validate_text_kernel(StringIO(txt)))

    def test_string_length_fail_with_escaped_quote(self):
        txt = 'Line1\n' + 'VARIABLE = \'' + ('S' * 41) + '\'\'' + ('s' * 41) + '\'\n' + 'Line3\n'
        self.assertFalse(writer.validate_text_kernel(StringIO(txt)))

    def test_no_double_quotes_pass(self):
        txt = '\\begindata\n' + 'VARIABLE = \'value\'\n' + '\\begintext\n'
        self.assertTrue(writer.validate_text_kernel(StringIO(txt)))

    def test_no_double_quotes_fail(self):
        txt = '\\begindata\n' + 'VARIABLE = "value"\n' + '\\begintext\n'
        self.assertFalse(writer.validate_text_kernel(StringIO(txt)))

    def test_no_newline_end_pass(self):
        txt = '\\begindata\nVARIABLE = 123\n\\begintext\n'
        self.assertTrue(writer.validate_text_kernel(StringIO(txt)))

    def test_no_newline_end_fail(self):
        txt = '\\begindata\nVARIABLE = 123\n\\begintext'
        self.assertFalse(writer.validate_text_kernel(StringIO(txt)))

    def test_no_tabs_pass(self):
        txt = '\\begindata\n    VARIABLE = 123\n\\begintext\n'
        self.assertTrue(writer.validate_text_kernel(StringIO(txt)))

    def test_no_tabs_fail(self):
        txt = '\\begindata\n\tVARIABLE = 123\n\\begintext\n'
        self.assertFalse(writer.validate_text_kernel(StringIO(txt)))

    def test_valid_char_pass(self):
        txt = '\\begindata\nVARIABLE = \'$123.00\'\n\\begintext\n'
        self.assertTrue(writer.validate_text_kernel(StringIO(txt)))

    def test_valid_char_fail(self):
        txt = '\\begindata\nVARIABLE = \'' + chr(8364) + '123.00\'\n\\begintext\n'
        self.assertFalse(writer.validate_text_kernel(StringIO(txt)))


if __name__ == '__main__':
    unittest.main()
