import numpy as np
import matplotlib.pyplot as plt
from ecg.plotting import plot_oxi, plot_peaks
from ecg.raw import Raw
from ecg.detection import oxi_peaks

data_path = 'C:/Users/au646069/Google Drive/ECG_root/Code/PythonToolboxes/pyExG/pyExG/Data/'

data = np.load(data_path + 'oxi/B.npy')[0][:60*75]
peaks = oxi_peaks(data)

plot_peaks(peaks)
plot_peaks(peaks, frequency='bpm')
plot_peaks(peaks, kind='heatmap')
plot_peaks(peaks, kind='heatmap', frequency='bpm')
plot_peaks(peaks, kind='staircase')
plot_peaks(peaks, kind='staircase', frequency='bpm')


# plot_oxi(x=data, peaks=peaks)
# plot_oxi(x=data[0], peaks=peaks, fill=False)
# plot_oxi(x=data[0], peaks=peaks, frequency=False)
# plot_oxi(x=data[0], peaks=peaks, kind='bpm')
# plot_oxi(x=data[0], peaks=peaks, kind='invalid')
