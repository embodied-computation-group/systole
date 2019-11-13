import numpy as np
from ecg.utils import heart_rate

# Test interpolate
peaks = np.array([15, 77, 132, 200, 285, 350])
heart_rate(peaks, sfreq=75)
heart_rate(peaks, sfreq=75, unit='bpm')
heart_rate(peaks, sfreq=75, method='interpolate')
heart_rate(peaks, sfreq=75, method='interpolate', unit='bpm')
heart_rate(peaks, sfreq=75, method='staircase')
heart_rate(peaks, sfreq=75, method='staircase', unit='bpm')
