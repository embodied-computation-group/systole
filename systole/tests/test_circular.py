# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
import unittest
import matplotlib
from unittest import TestCase
from systole.circular import to_angles, circular, plot_circular
from systole import import_rr

# Create angular data
x = np.random.normal(np.pi, 0.5, 100)
y = np.random.uniform(0, np.pi*2, 100)
z = np.concatenate([np.random.normal(np.pi/2, 0.5, 50),
                    np.random.normal(np.pi + np.pi/2, 0.5, 50)])


class TestCircular(TestCase):

    def test_circular(self):
        """Tests _circular function"""
        ax = circular(x)
        assert isinstance(ax, matplotlib.axes.Axes)
        for dens in ['area', 'heigth', 'alpha']:
            ax = circular(x, density='alpha', offset=np.pi)
            assert isinstance(ax, matplotlib.axes.Axes)
        ax = circular(x, density='height', mean=True,
                      units='degree', color='r')
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_circular(self):
        """Test plot_circular function"""
        data = pd.DataFrame(data={'x': x, 'y': y, 'z': z}).melt()
        ax = plot_circular(data=data, y='value', hue='variable')
        assert isinstance(ax, matplotlib.axes.Axes)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
