# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pytest
from numba.core.errors import TypingError

from systole import import_dataset1, import_ppg, import_rr
from systole.detection import ppg_peaks
from systole.utils import (
    find_clipping,
    get_valid_segments,
    heart_rate,
    input_conversion,
    nan_cleaning,
    norm_bad_segments,
    norm_triggers,
    simulate_rr,
    time_shift,
    to_angles,
    to_epochs,
    to_neighbour,
)


class TestUtils(TestCase):
    def test_to_neighbour(sef):
        ppg = import_ppg().ppg.to_numpy()[:200]  # Import PPG recording
        peaks = np.zeros(len(ppg), dtype=bool)
        peaks[50] = True

        new_peaks = to_neighbour(signal=ppg, peaks=peaks)
        assert np.where(new_peaks)[0] == 51
        new_peaks = to_neighbour(signal=ppg, peaks=peaks, kind="min")
        assert np.where(new_peaks)[0] == 1

    def test_norm_triggers(self):
        ppg = import_ppg().ppg.to_numpy()  # Import PPG recording
        _, peaks = ppg_peaks(ppg, sfreq=75)
        peaks[np.where(peaks)[0] + 1] = 1
        peaks[np.where(peaks)[0] + 2] = 1
        peaks[-1:] = 1
        y = norm_triggers(peaks)
        assert sum(y) == 379
        peaks = -peaks.astype(int)
        y = norm_triggers(peaks, threshold=-1, direction="lower")
        assert sum(y) == 379
        with pytest.raises(TypingError):
            norm_triggers(None)
        with pytest.raises(ValueError):
            norm_triggers(peaks, direction="invalid")

    def test_heart_rate(self):
        """Test heart_rate function"""
        ppg = import_ppg().ppg.to_numpy()  # Import PPG recording
        _, peaks = ppg_peaks(ppg, sfreq=75)
        heartrate, time = heart_rate(peaks)
        assert len(heartrate) == len(time)
        np.testing.assert_almost_equal(np.nanmean(heartrate), 884.92526408453)
        heartrate, time = heart_rate(list(peaks))
        assert len(heartrate) == len(time)
        np.testing.assert_almost_equal(np.nanmean(heartrate), 884.92526408453)
        heartrate, time = heart_rate(peaks, unit="bpm", kind="cubic", sfreq=500)
        assert len(heartrate) == len(time)
        np.testing.assert_almost_equal(np.nanmean(heartrate), 34.34558271737578)
        with pytest.raises(ValueError):
            heartrate, time = heart_rate([1, 2, 3])
        heartrate, time = heart_rate(
            np.diff(np.where(peaks)), kind="cubic", input_type="rr_ms"
        )
        np.testing.assert_almost_equal(np.nanmean(heartrate), 884.9253824912565)
        heartrate, time = heart_rate(
            np.diff(np.where(peaks)) / 1000, kind="cubic", input_type="rr_s"
        )
        np.testing.assert_almost_equal(np.nanmean(heartrate), 884.92526408453)

    def test_time_shift(self):
        """Test time_shift function"""
        lag = time_shift([40, 50, 60], [45, 52])
        assert np.all(lag == [5, 2])

    def test_to_angle(self):
        """Test to_angles function"""
        rr = import_rr().rr.values
        # Create event vector
        events = rr + np.random.normal(500, 100, len(rr))
        ang = to_angles(list(np.cumsum(rr)), list(np.cumsum(events)))
        assert ~np.any(np.asarray(ang) < 0)
        assert ~np.any(np.asarray(ang) > np.pi * 2)
        ppg = import_ppg().ppg.to_numpy()  # Import PPG recording
        signal, peaks = ppg_peaks(ppg, sfreq=75)
        ang = to_angles(peaks, peaks)

    def test_to_epochs(self):
        """Test ppg_peaks function"""
        # Load dataset
        ecg_df = import_dataset1(modalities=["ECG", "Stim"])

        triggers_idx = [
            np.where(ecg_df.stim.to_numpy() == 2)[0],
            np.where(ecg_df.stim.to_numpy() == 1)[0],
        ]
        signal = ecg_df.ecg.to_numpy()

        # Using event idx
        epoch, rejected = to_epochs(signal=signal, triggers_idx=triggers_idx)
        assert len(epoch) == 2
        assert len(rejected) == 2
        np.testing.assert_almost_equal(epoch[0].mean(), 0.047150987567323624)
        assert rejected[0].mean() == 0.0

        # Using event triggers
        epoch, rejected = to_epochs(
            signal=signal,
            triggers=ecg_df.stim.to_numpy(),
            event_val=2,
            apply_baseline=(-1.0, 0.0),
        )
        np.testing.assert_almost_equal(epoch[0].mean(), 0.008389195914220333)
        assert rejected[0].mean() == 0.0

        # Using a rejection vector
        reject = np.zeros(len(signal))
        reject[768285:] = 1
        epoch, rejected = to_epochs(
            signal=signal,
            triggers=ecg_df.stim.to_numpy(),
            event_val=2,
            apply_baseline=(-1.0, 0.0),
            reject=reject,
        )
        assert len(epoch[0]) == 0
        assert rejected[0].mean() == 1

    def test_simulate_rr(self):
        """Test ppg_peaks function"""
        rr = simulate_rr(artefacts=True)
        assert isinstance(rr, np.ndarray)
        assert len(rr) == 350

    def test_input_conversion(self):
        """Test the input_conversion function"""
        # Load example PPG signal
        ppg = import_ppg().ppg.to_numpy()
        _, peaks = ppg_peaks(ppg, sfreq=75)

        # input_type = "peaks"
        rr_ms = input_conversion(peaks, input_type="peaks", output_type="rr_ms")
        rr_s = input_conversion(peaks, input_type="peaks", output_type="rr_s")
        peaks_idx = input_conversion(peaks, input_type="peaks", output_type="peaks_idx")
        assert rr_ms.mean() == rr_s.mean() * 1000
        assert rr_ms.mean() == np.diff(peaks_idx).mean()

        # input_type = "peaks_idx"
        pks_idx = np.where(peaks)[0]
        rr_ms = input_conversion(pks_idx, input_type="peaks_idx", output_type="rr_ms")
        rr_s = input_conversion(pks_idx, input_type="peaks_idx", output_type="rr_s")
        pks = input_conversion(pks_idx, input_type="peaks_idx", output_type="peaks")
        assert rr_ms.mean() == rr_s.mean() * 1000
        assert rr_ms.mean() == np.diff(np.where(pks)[0]).mean()

        # input_type = "rr_ms"
        rr_ms = np.diff(np.where(peaks)[0])
        pks = input_conversion(rr_ms, input_type="rr_ms", output_type="peaks")
        rr_s = input_conversion(rr_ms, input_type="rr_ms", output_type="rr_s")
        peaks_idx = input_conversion(rr_ms, input_type="rr_ms", output_type="peaks_idx")
        assert np.diff(np.where(pks)[0]).mean() == rr_s.mean() * 1000
        assert rr_s.mean() * 1000 == np.diff(peaks_idx).mean()

        # input_type = "rr_s"
        rr_s = np.diff(np.where(peaks)[0]) / 1000
        pks = input_conversion(rr_s, input_type="rr_s", output_type="peaks")
        rr_ms = input_conversion(rr_s, input_type="rr_s", output_type="rr_ms")
        peaks_idx = input_conversion(rr_s, input_type="rr_s", output_type="peaks_idx")
        assert np.diff(np.where(pks)[0]).mean() == rr_ms.mean()
        assert rr_ms.mean() == np.diff(peaks_idx).mean()

    def test_nan_cleaning(self):
        ppg = import_ppg().ppg.to_list()
        ppg[30] = np.nan
        nan_cleaning(signal=np.array(ppg), verbose=True)

    def test_find_clipping(self):
        ppg = import_ppg().ppg.to_numpy()

        lower, upper = find_clipping(signal=ppg)
        assert (lower, upper) == (0, 255)

        # Create lower and upper clipping artefacts
        ppg[ppg <= 50] = 50
        ppg[ppg >= 230] = 230

        lower, upper = find_clipping(signal=ppg)
        assert (lower, upper) == (50, 230)

        lower, upper = find_clipping(signal=ppg[:100])
        assert (lower, upper) == (None, None)

    def test_norm_bad_segments(self):
        # Overlapping intervals
        bad_segments = [(100, 200), (150, 250)]
        new_segments = norm_bad_segments(bad_segments)
        assert new_segments == [(100, 250)]

        # A boolean vector
        bool_bad_segments = np.zeros(100, dtype=bool)
        bool_bad_segments[10:20] = True
        bool_bad_segments[50:60] = True
        new_segments = norm_bad_segments(bool_bad_segments)
        assert new_segments == [(10, 20), (50, 60)]

    def test_get_valid_segments(self):
        signal = np.random.normal(size=1000)

        bad_segments = [(500, 550), (700, 800)]
        valids = get_valid_segments(signal=signal, bad_segments=bad_segments)

        assert len(valids[0]) == 500
        assert len(valids[1]) == 200
        assert len(valids[2]) == 150


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
