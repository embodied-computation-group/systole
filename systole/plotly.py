# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
import plotly_express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from systole.detection import oxi_peaks
from systole.correction import rr_artefacts
from systole.utils import heart_rate
from systole.plotting import plot_psd
from systole.hrv import time_domain, frequency_domain, nonlinear


def plot_raw(signal, sfreq=75):
    """Interactive visualization of PPG signal and beats detection.

    Parameters
    ----------
    signal : `pd.DataFrame` instance or 1d array-like
        Dataframe of signal recording in the long format. Should contain at
        least the two following columns: ['time', 'signal']. If an array is
        provided, will automatically create the DataFrame using th array as
        signal and *sfreq* as sampling frequency.
    sfreq : int
        Signal sampling frequency. Default is 75 Hz.
    """
    if isinstance(signal, pd.DataFrame):
        # Find peaks - Remove learning phase
        signal, peaks = oxi_peaks(signal.signal, noise_removal=False)
    else:
        signal, peaks = oxi_peaks(signal, noise_removal=False)
    time = np.arange(0, len(signal))/1000

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


def plot_ectopic(rr=None, artefacts=None):
    """Plot interactive ectobeats subspace.

    Parameters
    ----------
    rr : 1d array-like or None
        The RR time serie.
    artefacts : dict or None
        The artefacts detected using *systole.detection.rr_artefacts()*.

    Returns
    -------
    subspacesPlot : plotly Figure
        The interactive plot.

    Notes
    -----
    If both *rr* or *artefacts* are provided, will recompute *artefacts*
    given the current rr time-series.
    """
    c1, c2, xlim, ylim = 0.13, 0.17, 10, 10

    if artefacts is None:
        if rr is None:
            raise ValueError('rr or artefacts should be provided')
        artefacts = rr_artefacts(rr)

    outliers = (artefacts['ectopic'] | artefacts['short'] | artefacts['long']
                | artefacts['extra'] | artefacts['missed'])

    # All vlaues fit in the x and y lims
    for this_art in [artefacts['subspace1'], artefacts['subspace2']]:
        this_art[this_art > xlim] = xlim
        this_art[this_art < -xlim] = -xlim
        this_art[this_art > ylim] = ylim
        this_art[this_art < -ylim] = -ylim

    subspacesPlot = go.Figure()

    # Upper area
    def f1(x): return -c1*x + c2
    subspacesPlot.add_trace(go.Scatter(x=[-10, -10, -1, -1],
                                       y=[f1(-10), 10, 10, f1(-1)],
                                       fill='toself', mode='lines',
                                       opacity=0.2, showlegend=False,
                                       fillcolor='gray', hoverinfo='none',
                                       line_color='gray'))

    # Lower area
    def f2(x): return -c1*x - c2
    subspacesPlot.add_trace(go.Scatter(x=[1, 1, 10, 10],
                            y=[f2(1), -10, -10, f2(10)],
                            fill='toself', mode='lines', opacity=0.2,
                            showlegend=False, fillcolor='gray',
                            hoverinfo='none', line_color='gray',
                            text="Points only"))

    # Plot normal intervals
    subspacesPlot.add_trace(go.Scattergl(x=artefacts['subspace1'][~outliers],
                                         y=artefacts['subspace2'][~outliers],
                                         mode='markers', showlegend=False,
                                         name='Normal', marker=dict(size=8,
                                         color='#4c72b0', opacity=0.2,
                                         line=dict(width=2,
                                                   color='DarkSlateGrey'))))

    # Plot ectopic beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['ectopic']],
        y=artefacts['subspace2'][artefacts['ectopic']],
        mode='markers', name='Ectopic beats',
        showlegend=False, marker=dict(
            size=10, color='#c44e52',
            line=dict(width=2, color='DarkSlateGrey'))))

    # Plot missed beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['missed']],
        y=artefacts['subspace2'][artefacts['missed']],
        mode='markers', name='Missed beats',
        showlegend=False, marker=dict(
            size=10,
            color=px.colors.sequential.Greens[8],
            line=dict(width=2, color='DarkSlateGrey'))))

    # Plot long beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['long']],
        y=artefacts['subspace2'][artefacts['long']],
        mode='markers', name='Long beats', marker_symbol='square',
        showlegend=False, marker=dict(
            size=10, color=px.colors.sequential.Greens[6],
            line=dict(width=2, color='DarkSlateGrey'))))

    # Plot extra beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['extra']],
        y=artefacts['subspace2'][artefacts['extra']],
        mode='markers', name='Extra beats',
        showlegend=False, marker=dict(size=10,
                                      color=px.colors.sequential.Purples[8],
                                      line=dict(width=2,
                                                color='DarkSlateGrey'))))
    # Plot short beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['short']],
        y=artefacts['subspace2'][artefacts['short']],
        mode='markers', name='Short beats', marker_symbol='square',
        showlegend=False, marker=dict(size=10,
                                      color=px.colors.sequential.Purples[6],
                                      line=dict(width=2,
                                                color='DarkSlateGrey'))))

    subspacesPlot.update_layout(
        width=600, height=600, xaxis_title="Subspace $S_{11}$",
        yaxis_title="Subspace $S_{12}$", template='simple_white',
        title={'text': "Ectopic beats", 'x': 0.5, 'xanchor': 'center',
               'yanchor': 'top'})

    subspacesPlot.update_xaxes(showline=True, linewidth=2, linecolor='black',
                               range=[-xlim, xlim])
    subspacesPlot.update_yaxes(showline=True, linewidth=2, linecolor='black',
                               range=[-ylim, ylim])

    return subspacesPlot


def plot_shortLong(rr=None, artefacts=None):
    """Plot interactive short/long subspace.

    Parameters
    ----------
    rr : 1d array-like or None
        The RR time serie.
    artefacts : dict or None
        The artefacts detected using *systole.detection.rr_artefacts()*.

    Returns
    -------
    subspacesPlot : plotly Figure
        The interactive plot.

    Notes
    -----
    If both *rr* or *artefacts* are provided, will recompute *artefacts*
    given the current rr time-series.
    """
    xlim, ylim = 10, 10

    if artefacts is None:
        if rr is None:
            raise ValueError('rr or artefacts should be provided')
        artefacts = rr_artefacts(rr)

    outliers = (artefacts['ectopic'] | artefacts['short'] | artefacts['long']
                | artefacts['extra'] | artefacts['missed'])

    # All vlaues fit in the x and y lims
    for this_art in [artefacts['subspace1'], artefacts['subspace3']]:
        this_art[this_art > xlim] = xlim
        this_art[this_art < -xlim] = -xlim
        this_art[this_art > ylim] = ylim
        this_art[this_art < -ylim] = -ylim

    subspacesPlot = go.Figure()

    # Upper area
    subspacesPlot.add_trace(go.Scatter(x=[-10, -10, -1, -1],
                                       y=[1, 10, 10, 1],
                                       fill='toself', mode='lines',
                                       opacity=0.2, showlegend=False,
                                       fillcolor='gray', hoverinfo='none',
                                       line_color='gray'))

    # Lower area
    subspacesPlot.add_trace(go.Scatter(x=[1, 1, 10, 10],
                            y=[-1, -10, -10, -1],
                            fill='toself', mode='lines', opacity=0.2,
                            showlegend=False, fillcolor='gray',
                            hoverinfo='none', line_color='gray',
                            text="Points only"))

    # Plot normal intervals
    subspacesPlot.add_trace(go.Scattergl(x=artefacts['subspace1'][~outliers],
                                         y=artefacts['subspace3'][~outliers],
                                         mode='markers', showlegend=False,
                                         name='Normal', marker=dict(size=8,
                                         color='#4c72b0', opacity=0.2,
                                         line=dict(width=2,
                                                   color='DarkSlateGrey'))))

    # Plot ectopic beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['ectopic']],
        y=artefacts['subspace3'][artefacts['ectopic']],
        mode='markers', name='Ectopic beats',
        showlegend=False, marker=dict(
            size=10, color='#c44e52',
            line=dict(width=2, color='DarkSlateGrey'))))

    # Plot missed beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['missed']],
        y=artefacts['subspace3'][artefacts['missed']],
        mode='markers', name='Missed beats',
        showlegend=False, marker=dict(
            size=10,
            color=px.colors.sequential.Greens[8],
            line=dict(width=2, color='DarkSlateGrey'))))

    # Plot long beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['long']],
        y=artefacts['subspace3'][artefacts['long']],
        mode='markers', name='Long beats', marker_symbol='square',
        showlegend=False, marker=dict(
            size=10, color=px.colors.sequential.Greens[6],
            line=dict(width=2, color='DarkSlateGrey'))))

    # Plot extra beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['extra']],
        y=artefacts['subspace3'][artefacts['extra']],
        mode='markers', name='Extra beats',
        showlegend=False, marker=dict(size=10,
                                      color=px.colors.sequential.Purples[8],
                                      line=dict(width=2,
                                                color='DarkSlateGrey'))))
    # Plot short beats
    subspacesPlot.add_trace(go.Scattergl(
        x=artefacts['subspace1'][artefacts['short']],
        y=artefacts['subspace3'][artefacts['short']],
        mode='markers', name='Short beats', marker_symbol='square',
        showlegend=False, marker=dict(size=10,
                                      color=px.colors.sequential.Purples[6],
                                      line=dict(width=2,
                                                color='DarkSlateGrey'))))

    subspacesPlot.update_layout(
        width=600, height=600, xaxis_title="Subspace $S_{11}$",
        yaxis_title="Subspace $S_{12}$", template='simple_white',
        title={'text': "Short/longs beats", 'x': 0.5, 'xanchor': 'center',
               'yanchor': 'top'})

    subspacesPlot.update_xaxes(showline=True, linewidth=2, linecolor='black',
                               range=[-xlim, xlim])
    subspacesPlot.update_yaxes(showline=True, linewidth=2, linecolor='black',
                               range=[-ylim, ylim])

    return subspacesPlot


def plot_subspaces(rr):
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    rr : 1d array-like
        The dataframe containing the recording.

    Returns
    -------
    fig : `go.Figure`
        The figure.

    References
    ----------
    [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel beat
        classification. Journal of Medical Engineering & Technology, 43(3),
        173–181. https://doi.org/10.1080/03091902.2019.1640306
    """
    xlim, ylim = 10, 10
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        subplot_titles=("Ectopic", "Short/longs beats"))

    ectopic = plot_ectopic(rr.copy())
    sl = plot_shortLong(rr.copy())

    for traces in ectopic.data:
        fig.add_traces([traces], rows=[1], cols=[1])
    for traces in sl.data:
        fig.add_traces([traces], rows=[1], cols=[2])

    fig.update_layout(
        width=1200, height=600, xaxis_title="Subspace $S_{11}$",
        yaxis_title="Subspace $S_{12}$", xaxis2_title="Subspace $S_{21}$",
        yaxis2_title="Subspace $S_{22}$", template='simple_white')

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',
                     range=[-xlim, xlim], row=1, col=1)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                     range=[-ylim, ylim], row=1, col=1)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',
                     range=[-xlim, xlim], row=1, col=2)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                     range=[-ylim, ylim], row=1, col=2)

    return fig


def plot_frequency(rr):
    """Plot PSD and frequency domain metrics.

    Parameters
    ----------
    rr : 1d array-like
        Time series of R-R intervals.
    """
    df = frequency_domain(rr).round(2)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        specs=[[{"type": "scatter"}],
               [{"type": "table"}]],
    )

    fig.add_trace(go.Table(
      header=dict(
        values=['<b>Frequency band (HZ)</b>', '<b>Peak (Hz)</b>',
                '<b>Power (ms<sup>2</sup>)</b>',
                '<b>Power (%)</b>', '<b>Power (n.u.)</b>'], align='center'
      ),
      cells=dict(
        values=[['VLF \n (0-0.04 Hz)', 'LF \n (0.04 - 0.15 Hz)',
                 'HF \n (0.15 - 0.4 Hz)'],
                [df[df.Metric == 'vlf_peak'].Values,
                 df[df.Metric == 'lf_peak'].Values,
                 df[df.Metric == 'hf_peak'].Values],
                [df[df.Metric == 'vlf_power'].Values,
                 df[df.Metric == 'lf_power'].Values,
                 df[df.Metric == 'hf_power'].Values],
                ['-',
                 df[df.Metric == 'power_lf_nu'].Values,
                 df[df.Metric == 'power_hf_nu'].Values],
                ['-',
                 df[df.Metric == 'power_lf_per'].Values,
                 df[df.Metric == 'power_hf_per'].Values],
                ], align='center')), row=2, col=1)

    freq, psd = plot_psd(rr, show=False)

    fbands = {'vlf': ['Very low frequency', (0.003, 0.04), '#4c72b0'],
              'lf':	['Low frequency', (0.04, 0.15), '#55a868'],
              'hf':	['High frequency', (0.15, 0.4), '#c44e52']}

    for f in ['vlf', 'lf', 'hf']:
        mask = (freq >= fbands[f][1][0]) & (freq <= fbands[f][1][1])
        fig.add_trace(go.Scatter(
            x=freq[mask],
            y=psd[mask],
            fill='tozeroy',
            mode='lines',
            showlegend=False,
            line_color=fbands[f][2],
            line=dict(shape="spline",
                      smoothing=1,
                      width=1,
                      color="#fac1b7")), row=1, col=1)

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=5, r=5, b=5, t=5), autosize=True, width=600, height=600,
        xaxis_title='Frequencies (Hz)', yaxis_title='PSD',
        title={'text': "FFT Spectrum", 'x': 0.5, 'xanchor': 'center',
               'yanchor': 'top'})
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig


def plot_nonlinear(rr):
    """Plot nonlinear domain.

    Parameters
    ----------
    rr : 1d array-like
        Time sere of R-R intervals.
    """
    df = nonlinear(rr).round(2)

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "scatter"}],
               [{"type": "table"}]],
    )

    fig.add_trace(go.Table(
      header=dict(
        values=['<b>Pointcare Plot</b>', '<b>Value</b>'], align='center'
      ),
      cells=dict(
        values=[['SD1', 'SD2'],
                [df[df.Metric == 'SD1'].Values,
                 df[df.Metric == 'SD2'].Values]], align='center')),
                 row=2, col=1)

    ax_min = rr.min() - (rr.max() - rr.min())*.1
    ax_max = rr.max() + (rr.max() - rr.min())*.1

    fig.add_trace(go.Scattergl(
        x=rr[:-1],
        y=rr[1:],
        mode='markers',
        opacity=0.6,
        showlegend=False,
        marker=dict(size=8,
                    color='#4c72b0',
                    line=dict(width=2, color='DarkSlateGrey'))))

    fig.add_trace(go.Scatter(x=[0, 4000], y=[0, 4000], showlegend=False))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=5, r=5, b=5, t=5), autosize=True, width=500, height=800,
        xaxis_title='RR<sub>n</sub> (ms)', yaxis_title='RR<sub>n+1</sub> (ms)',
        title={'text': "Pointcare Plot", 'x': 0.5,
               'xanchor': 'center', 'yanchor': 'top'})
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',
                     range=[ax_min, ax_max])
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                     range=[ax_min, ax_max])

    return fig


def plot_timedomain(rr):
    """Plot time domain.

    Parameters
    ----------
    rr : 1d array-like
        Time sere of R-R intervals.
    """
    df = time_domain(rr).round(2)

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "scatter"}],
               [{"type": "table"}]],
    )

    fig.add_trace(go.Table(
      header=dict(
        values=['<b>Variable</b>', '<b>Unit</b>', '<b>Value</b>'],
        align='center'
      ),
      cells=dict(
        values=[['Mean RR', 'Mean BPM', 'SDNN', 'RMSSD', 'pnn50'],
                ['(ms)', '(1/min)', '(ms)', '(ms)', '(%)'],
                [df[df.Metric == 'MeanRR'].Values,
                 df[df.Metric == 'MeanBPM'].Values,
                 df[df.Metric == 'SDNN'].Values,
                 df[df.Metric == 'RMSSD'].Values,
                 df[df.Metric == 'pnn50'].Values],
                ], align='center')),
                 row=2, col=1)

    fig.add_trace(go.Histogram(x=rr, marker={'line': {'width': 2},
                                             'color': '#4c72b0'}))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=5, r=5, b=5, t=5), autosize=True, width=500, height=800,
        xaxis_title='RR intervals (ms)', yaxis_title='Counts',
        title={'text': "Distribution", 'x': 0.5,
               'xanchor': 'center', 'yanchor': 'top'})
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig
