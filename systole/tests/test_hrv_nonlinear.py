# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import unittest
from unittest import TestCase
from systole.hrv_nonlinear import nonlinear_domain
from systole import import_rr

rr = import_rr().rr.values


class TestHrvNonlinear(TestCase):

    def test_nonlinear_domain(self):
        """Test nonlinear_domain function"""
        stats = nonlinear_domain(rr)
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 4


if __name__ == '__main__':
    unittest.main()
