# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import numpy as np
import unittest
from unittest import TestCase
from systole.interactive import plot_raw
from systole import import_ppg

ppg = import_ppg()
signal_df = pd.DataFrame({'time': np.arange(0, len(ppg[0]))/75,
                          'signal': ppg[0]})


class TestInteractive(TestCase):

    def test_plot_raw(self):
        """Test nnX function"""
        plot_raw(signal_df)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
