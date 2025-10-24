"""spicetime - Unit test

@author: Brandon Stone
"""

import unittest

from curryer import spicetime


class SpiceTimePkgTestCase(unittest.TestCase):
    def test_api_import(self):
        self.assertTrue(hasattr(spicetime, "adapt"))
        self.assertTrue(hasattr(spicetime, "SpiceTime"))


if __name__ == "__main__":
    unittest.main()
