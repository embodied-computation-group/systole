# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from systole.detection import oxi_peaks
from systole.correction import correct_rr
from systole.utils import heart_rate


def plot_raw(signal_df):
    """Interactive visualization of PPG signal and peak detection.

    Parameter
    ---------
    signal_df : `pd.DataFrame` instance
        Dataframe of signal recording in the long format. Should contain at
        least the two following columns: ['time', 'signal'].
    """
    # Find peaks - Remove learning phase
    signal, peaks = oxi_peaks(signal_df.signal, noise_removal=False)
    time = np.arange(0, len(signal))/1000

    # Convert to RR time series
    rr = np.diff(np.where(peaks)[0])

    # Extract heart rate
    hr, time = heart_rate(peaks, sfreq=1000, unit='rr', kind='previous')

    #############
    # Upper panel
    #############

    # Signal
    ppg_trace = go.Scattergl(x=time, y=signal, mode='lines', name='PPG',
                             hoverinfo='skip', showlegend=False,
                             line=dict(width=1, color='#c44e52'))
    # Peaks
    peaks_trace = go.Scattergl(x=time[peaks], y=signal[peaks], mode='markers',
                               name='Peaks', hoverinfo='y', showlegend=False,
                               marker=dict(size=8, color='white',
                               line=dict(width=2, color='DarkSlateGrey')))

    #############
    # Lower panel
    #############

    # Instantaneous Heart Rate - Lines
    rr_trace = go.Scattergl(x=time, y=hr, mode='lines', name='R-R intervals',
                            hoverinfo='skip', showlegend=False,
                            line=dict(width=1, color='#4c72b0'))

    # Instantaneous Heart Rate - Peaks
    rr_peaks = go.Scattergl(x=time[peaks], y=hr[peaks], mode='markers',
                            name='R-R intervals', showlegend=False,
                            marker=dict(size=6, color='white',
                            line=dict(width=2, color='DarkSlateGrey')))

    raw = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=.05, row_titles=['Recording',
                                                          'Heart rate'])

    raw.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=5, r=5, b=5, t=5), autosize=True)

    raw.add_trace(ppg_trace, 1, 1)
    raw.add_trace(peaks_trace, 1, 1)
    raw.add_trace(rr_trace, 2, 1)
    raw.add_trace(rr_peaks, 2, 1)

    return raw
