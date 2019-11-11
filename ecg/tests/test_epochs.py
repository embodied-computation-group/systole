import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ecg.raw import Raw
from ecg.epochs import Epochs

data_path = 'C:/Users/au646069/Google Drive/ECG_root/Code/PythonToolboxes/pyExG/pyExG/Data/'

for file in range(12):
    data = np.load(data_path + 'oxi/Subject_1' + str(file) + '.npy')[0]
    events = np.load(data_path + 'oxi/Subject_1' + str(file) + '.npy')[1]

    raw = Raw(oxi=data, events=events, sfreq=75)
    raw.find_peaks(interpolation='staircase')
    if file != 0:
        final_raw = final_raw.append(raw)
    else:
        final_raw = raw

event_id = pd.read_csv(data_path + '/Oxi/Subject_Camile.txt')
event_id = event_id['Valence']
events = np.where(final_raw.events == 2)[0]
epochs = Epochs(final_raw, events, event_id=event_id, tmin=-0.5, tmax=4)
epochs.apply_baseline(baseline=(None, 0))

fig = epochs.plot(kind='bpm')

fig = epochs.plot_image(kind='rr', limits=250)
