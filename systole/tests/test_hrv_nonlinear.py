# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import unittest
from systole.hrv_nonlinear import nonlinear_domain
from systole import import_rr

rr = import_rr().rr.values


class TestHrvNonlinear(unittest.TestCase):

    def test_nonlinear_domain(self):
        """Test nonlinear_domain function"""
        stats = nonlinear_domain(rr)
        assert isinstance(stats, pd.DataFrame)
        self.assertEqual(stats.size, 4)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
