# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np

from systole import import_rr
from systole.correction import (
    correct_extra_rr,
    correct_missed_peaks,
    correct_missed_rr,
    correct_peaks,
    correct_rr,
    interpolate_rr,
)
from systole.detection import rr_artefacts
from systole.utils import simulate_rr


class TestDetection(TestCase):
    def test_correct_extra_rr(self):
        """Test correct_extra_rr function"""

        # When the artefact is the first RR interval
        rr = import_rr().rr.values  # Import RR time series
        rr[0] = 100
        clean_rr, _ = correct_extra_rr(rr, extra_idx=np.array([0]))
        assert clean_rr[0] == rr[0] + rr[1]
        assert clean_rr[1] != 100

        # When the artefact is the last RR interval
        rr = import_rr().rr.values  # Import RR time series
        rr[len(rr) - 1] = 100
        clean_rr, _ = correct_extra_rr(rr, extra_idx=np.array([len(rr) - 1]))
        assert clean_rr[-1] == rr[-2]

        # With many artefacts
        rr = import_rr().rr.values  # Import RR time series
        rr[10] = 100
        rr[20] = 100
        clean_rr, _ = correct_extra_rr(rr, extra_idx=np.array([10, 20]))
        assert clean_rr[10] == rr[11] + rr[10]
        assert clean_rr[19] == rr[21] + rr[20]
        assert len(clean_rr) == len(rr) - 2

        # With the artefact array provided
        rr = import_rr().rr.values  # Import RR time series
        rr[30] = 100
        rr[50] = 100
        artefacts = rr_artefacts(rr)

        # Create the array from dict
        artefacts = np.array(
            [
                artefacts["missed"],
                artefacts["extra"],
                artefacts["ectopic"],
                artefacts["short"],
                artefacts["long"],
            ],
            dtype=bool,
        )
        assert len(np.where(artefacts[1, :])[0]) == 2

        clean_rr, clean_artefacts = correct_extra_rr(
            rr=rr,
            extra_idx=np.where(artefacts[1, :])[0],
            artefacts=artefacts,
        )

        assert clean_rr[30] > 100
        assert clean_rr[49] > 100
        assert clean_artefacts.shape[0] == artefacts.shape[0]
        assert clean_artefacts.shape[1] == artefacts.shape[1] - 2
        assert len(clean_rr) == clean_artefacts.shape[1]

    def test_correct_missed_rr(self):
        """Test correct_missed_rr function"""

        # When the artefact is the first RR interval
        rr = import_rr().rr.values  # Import RR time series
        rr[0] = 1600
        clean_rr, _ = correct_missed_rr(rr, missed_idx=np.array([0]))
        assert clean_rr[0] == clean_rr[0] == rr[0] / 2

        # When the artefact is the last RR interval
        rr = import_rr().rr.values  # Import RR time series
        rr[len(rr) - 1] = 1600
        clean_rr, _ = correct_missed_rr(rr, missed_idx=np.array([len(rr) - 1]))
        assert clean_rr[-1] == clean_rr[-2] == rr[-1] / 2

        # With many artefacts
        rr = import_rr().rr.values  # Import RR time series
        rr[10] = 1600
        rr[20] = 1600
        clean_rr, _ = correct_missed_rr(rr, missed_idx=np.array([10, 20]))
        assert clean_rr[10] == clean_rr[10] == rr[10] / 2
        assert clean_rr[21] == clean_rr[22] == rr[20] / 2
        assert len(clean_rr) == len(rr) + 2

        # With the artefact array provided
        rr = import_rr().rr.values  # Import RR time series
        rr[100] = 1400
        artefacts = rr_artefacts(rr)

        # Create the array from dict
        artefacts = np.array(
            [
                artefacts["missed"],
                artefacts["extra"],
                artefacts["ectopic"],
                artefacts["short"],
                artefacts["long"],
            ],
            dtype=bool,
        )
        assert len(np.where(artefacts[0, :])[0]) == 1

        clean_rr, clean_artefacts = correct_missed_rr(
            rr=rr, artefacts=artefacts, missed_idx=np.where(artefacts[0, :])[0]
        )
        assert clean_rr[100] == clean_rr[101] == rr[100] / 2
        assert clean_artefacts.shape[0] == artefacts.shape[0]
        assert clean_artefacts.shape[1] == artefacts.shape[1] + 1
        assert len(clean_rr) == clean_artefacts.shape[1]

    def test_interpolate_rr(self):
        """Test interpolate_rr function"""

        # When the artefact is the first RR interval
        rr = import_rr().rr.values  # Import RR time series
        rr[0] = 1600
        clean_rr = interpolate_rr(rr, idx=np.array([0]))
        assert clean_rr[0] == clean_rr[1] == rr[1]

        # When the artefact is the last RR interval
        rr = import_rr().rr.values  # Import RR time series
        rr[len(rr) - 1] = 1600
        clean_rr = interpolate_rr(rr, idx=np.array([len(rr) - 1]))
        assert clean_rr[-1] == clean_rr[-2] == rr[-2]

        # With many artefacts
        rr = import_rr().rr.values  # Import RR time series
        rr[10] = 1600
        rr[20] = 1600
        clean_rr = interpolate_rr(rr, idx=np.array([10, 20]))
        assert clean_rr[10] != rr[10]
        assert clean_rr[20] != rr[20]
        assert np.sum(~(clean_rr == rr)) == 2

    def test_correct_rr(self):
        """Test correct_rr function"""

        rr = simulate_rr()  # Import RR time series
        corrected_rr, (nMissed, nExtra, nEctopic, nShort, nLong) = correct_rr(rr)

        assert len(corrected_rr) == 350
        assert nMissed == 1
        assert nExtra == 1
        assert nEctopic == 3
        assert nShort == 1
        assert nLong == 1

    def test_correct_peaks(self):
        """Test correct_peaks function"""
        peaks = simulate_rr(as_peaks=True)
        peaks_correction = correct_peaks(peaks)
        peaks_correction = correct_peaks(list(peaks))
        assert len(peaks_correction["clean_peaks"]) == 280154
        assert peaks_correction["missed"] == 1
        assert peaks_correction["extra"] == 1

    def test_correct_missed_peaks(self):
        """Test correct_missed_peaks function"""
        np.random.seed(123)
        rr = np.random.normal(1000, 200, 10).astype("int")
        peaks = np.zeros(10000)
        peaks[np.cumsum(rr)] = 1
        assert np.where(peaks)[0].sum() == 52029

        peaks[3735] = 0
        peaks = correct_missed_peaks(peaks, idx=4619)
        assert np.where(peaks)[0].sum() == 52122

        with self.assertRaises(ValueError):
            correct_missed_peaks(peaks, idx=4610)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
