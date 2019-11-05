import numpy as np
from pyExG.utils import heart_rate

# Test interpolate
peaks = np.array([15, 77, 132, 200, 285, 350])
heart_rate(peaks, sfreq=75)
heart_rate(peaks, sfreq=75, unit='bpm')
heart_rate(peaks, sfreq=75, method='interpolate')
heart_rate(peaks, sfreq=75, method='interpolate', unit='bpm')
heart_rate(peaks, sfreq=75, method='staircase')
heart_rate(peaks, sfreq=75, method='staircase', unit='bpm')





inter = (np.diff(peaks)/75) * 1000
inter = (60/inter) * 1000

sfreq=75

# From 0 to first peak
staircase = np.repeat((peaks[0]/sfreq) * 1000, peaks[0])
for i in range(len(peaks)-1):
    rr = peaks[i+1] - peaks[i]
    a = np.repeat((rr/sfreq) * 1000, rr)
    staircase = np.append(staircase, a)
# End of the recording
a = np.repeat(staircase[-1], len(peaks) - peaks[-1])
staircase = np.append(staircase, a)




np.cumsum(peaks).shape
peaks.shape
import matplotlib.pyplot as plt
from scipy import interpolate
x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y)

xnew = np.arange(0, 9, 0.1)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()
