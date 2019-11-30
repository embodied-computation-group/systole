# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
import pytest
from unittest import TestCase
from systole.circular import to_angles, circular, plot_circular
from systole import import_rr

# Create angular data
x = np.random.normal(np.pi, 0.5, 100)
y = np.random.uniform(0, np.pi*2, 100)
z = np.concatenate([np.random.normal(np.pi/2, 0.5, 50),
                    np.random.normal(np.pi + np.pi/2, 0.5, 50)])


class TestCircular(TestCase):

    def test_to_angle():
        """Test to_angles function"""
        rr = import_rr().rr.values
        # Create event vector
        events = rr + np.random.normal(500, 100, len(rr))
        to_angles(np.cumsum(rr), np.cumsum(events))

    def test_circular():
        """Tests _circular function"""
        circular(x)
        circular(x, density='alpha', offset=np.pi)
        circular(x, density='height', mean=True, units='degree', color='r')

    def test_plot_circular():
        """Test plot_circular function"""
        data = pd.DataFrame(data={'x': x, 'y': y, 'z': z}).melt()
        plot_circular(data=data, y='value', hue='variable')
