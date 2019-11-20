import numpy as np
import pandas as pd
from ecg.detection import oxi_peaks
from ecg.utils import heart_rate


def parse_data(filename, sfreq=75, hr_metric="rr"):
    """Read the provided file(s) and return dictionnary of dataframes with raw
    data, peaks and heartrate.

    Parameters
    ----------
    filename : str, list
        The the file(s) to read.
    sfreq : int
        The sampling frequency, default is 75.

    Returns
    -------
    df : dict
        The dataframes plotted in the 'raw' window.

    Notes
    -----
    If a Numpy array is provided, the first dimension should contain the
    recording. If a second dimension is provided, it should contain the events
    (optional).
    """
    try:
        # Load data
        if 'npy' in filename:
            x = np.load(filename, allow_pickle=True)
            signal = x[0]
            if x[1]:
                events = x[1]
            else:
                events = signal.copy()
                events[:] = np.nan
        if 'txt' in filename:
            x = np.load(filename, allow_pickle=True)
        if 'csv' in filename:
            x = np.load(filename, allow_pickle=True)

    except Exception as e:
        print(e)

    # Find peaks
    new_sfreq = 750
    signal, peaks = oxi_peaks(signal, sfreq=sfreq, new_sfreq=new_sfreq,
                              resample=True)
    time = np.arange(0, len(signal)/new_sfreq, 1/new_sfreq)

    # Security checks
    if len(signal) != len(peaks):
        raise ValueError('Error in peak detection')
    if len(signal) != len(time):
        raise ValueError('Error in time array')

    # Create dataframe
    df = pd.DataFrame({'time': time,
                       'signal': signal,
                       'peaks': peaks})

    # Creat heartrate vector
    hr, time = heart_rate(peaks, sfreq=new_sfreq,
                          method='staircase', unit=hr_metric)

    df['hr'] = hr

    return df

# filename = 'C:/Users/au646069/Desktop/dash/Subject_1111398.npy'
# df = parse_data(filename)
