"""init - Unit test

@author: Brandon Stone
"""

import unittest

from curryer import spicierpy


class SpicierPyInitTestCase(unittest.TestCase):
    def test_spicierpy_init(self):
        # Loads the `spiceypy` namespace.
        #   These are merely examples to make the point.
        self.assertEqual("spiceypy.utils", spicierpy.utils.__name__)
        self.assertEqual("spiceypy.spiceypy", spicierpy.exists.__module__)

        # Adds in the extension module.
        self.assertTrue(hasattr(spicierpy, "ext"))

        # Replaces some `spiceypy` methods with vectorized versions.
        #   Hard to check since they mock the signature of the original.
        self.assertIn("vectorized.py", spicierpy.recgeo.__code__.co_filename)


if __name__ == "__main__":
    unittest.main()
