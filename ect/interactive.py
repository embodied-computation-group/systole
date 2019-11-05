import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pyExG.utils import heart_rate
from scipy import signal


def peak_detection(x, peaks, sfreq):
    """Editing peak detection on time series data

    Parameters
    ----------
    x : array
        Time serie used for peak detection
    peaks : array | list
        Indexes of detected peaks
    sfreq : int
        Sampling frequency.

    Returns
    -------
    Plotly interactive graphs
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(0, len(x)/sfreq, 1/sfreq),
        y=x,
        mode='lines',
        name='Original Plot'
    ))

    fig.add_trace(go.Scatter(
        x=peaks/75,
        y=[x[j] for j in peaks],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            symbol='cross'
        ),
        name='Detected Peaks'
    ))

    # Set title
    fig.update_layout(
        title_text="Time series with range slider and selectors",
    )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=75*5,
                         label="5 seconds",
                         step="all",
                         visible=True,
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="linear"
        )
    )
    fig.show()


def hrv(peaks, sfreq):
    """Editing peak detection on time series data

    Parameters
    ----------
    x : array
        Time serie used for peak detection.
    peaks : array | list
        Indexes of detected peaks.
    sfreq : int
        Sampling frequency.

    Returns
    -------
    Plotly interactive graphs
    """
    hr_hist, time = heart_rate(peaks, sfreq=75, method=None)

    hr, time = heart_rate(peaks, sfreq=75, method='interpolate')

    sfreq = 75
    # Define window length
    nperseg = 256 * sfreq

    # Compute Power Spectral Density
    freq, psd = signal.welch(x=hr,
                             fs=sfreq,
                             nperseg=nperseg,
                             nfft=10 * nperseg)

    freqs = freq[freq < 0.5]
    psd = psd[freq < 0.5]/1000000

    fig = make_subplots(rows=2, cols=2,
                        specs=[[{"colspan": 2}, None],
                               [{}, {}]],
                        subplot_titles=("Heart rate",
                                        "R-R intervals",
                                        "Frequency domain (Welch)"))

    # Raw RR data
    fig.add_trace(go.Scatter(x=time, y=hr), row=1, col=1)

    # Histogram of R-R interval
    fig.add_trace(go.Histogram(x=hr_hist, opacity=0.75), row=2, col=1)

    # Frequency plot
    fig.add_trace(go.Scatter(x=freqs, y=psd), row=2, col=2)

    # High frequency
    fig.add_trace(go.Scatter(x=freqs[(freqs < 0.4) & (freqs >= 0.15)],
                             y=psd[(freqs < 0.4) & (freqs >= 0.15)],
                             fill='tozeroy'), row=2, col=2)

    # Low frequency
    fig.add_trace(go.Scatter(x=freqs[(freqs < 0.15) & (freqs > 0.04)],
                             y=psd[(freqs < 0.15) & (freqs > 0.04)],
                             fill='tozeroy'), row=2, col=2)

    # Very low frequency
    fig.add_trace(go.Scatter(x=freqs[(freqs < 0.04)],
                             y=psd[(freqs < 0.04)],
                             fill='tozeroy'), row=2, col=2)

    fig.update_layout(showlegend=False, title_text="Heart rate variability")
    fig.show()
