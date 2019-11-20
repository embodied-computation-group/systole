import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from ecg.hrv_frequency import hrv_frequency


def plot_raw(df):
    """Plot raw data.
    """
    ppg_trace = go.Scatter(
        x=df.time,
        y=df.signal,
        mode='lines',
        name='PPG',
        line=dict(shape="spline", smoothing=1, width=1,
                  color="#fac1b7")
    )
    peaks_trace = go.Scatter(
        x=df[df.peaks == 1].time,
        y=df[df.peaks == 1].signal,
        mode='markers',
        name='Peaks',
        line=dict(shape="spline", smoothing=1, width=1, color="#92d8d8"),
        marker=dict(symbol="diamond-open")
    )
    rr_trace = go.Scatter(
        x=df.time,
        y=df.hr[df.peaks == 1].values[:-1],
        mode='lines+markers',
        name='R-R intervals',
        marker=dict(symbol="diamond-open"),
        line=dict(shape="spline", width=1,
                  color="#5698d9")
    )

    raw = make_subplots(rows=2, cols=1)

    raw['layout']['plot_bgcolor'] = "#F9F9F9"
    raw['layout']['paper_bgcolor'] = "#F9F9F9"
    raw['layout']['margin'] = dict(l=10, r=10, b=10, t=10)
    raw['layout']['legend'] = dict(font=dict(size=15), orientation="h")
    raw['layout']['xaxis']['rangeselector'] = dict(visible=True)

    raw.append_trace(ppg_trace, 1, 1)
    raw.append_trace(peaks_trace, 1, 1)
    raw.append_trace(rr_trace, 2, 1)

    return raw


def plot_hist(df):
    """Plot histogram of heart rate data.
    """
    his = px.histogram(df[df.peaks == 1].iloc[:-1], x='hr')
    his['layout']['plot_bgcolor'] = "#F9F9F9"
    his['layout']['paper_bgcolor'] = "#F9F9F9"
    his['layout']['margin'] = dict(l=10, r=10, b=10, t=10)
    his['autosize'] = True
    his['automargin'] = True

    return his


def plot_pointcarre(df):
    """Plot pointcarre.
    """
    rr = df.hr[df.peaks == 1].values[:-1]
    pointcarrePlot = go.Figure()
    pointcarrePlot.add_trace(go.Scatter(
        x=rr[:-1],
        y=rr[1:],
        mode='markers'))
    # pointcarrePlot['layout']['plot_bgcolor'] = "#F9F9F9"
    # pointcarrePlot['layout']['paper_bgcolor'] = "#F9F9F9"
    # pointcarrePlot['layout']['autosize'] = True
    # pointcarrePlot['layout']['margin'] = dict(l=10, r=10, b=10, t=10)
    # pointcarrePlot['layout']['legend'] = dict(font=dict(size=15),
    #                                           orientation="h")
    # pointcarrePlot['layout']['shapes'] = [{'type': 'line',
    #                                        'x0': rr.min(), 'x1': rr.max(),
    #                                        'y0': rr.min(), 'y1': rr.max()}]

    return pointcarrePlot


def plot_frequency(df):
    """Plot frequency.
    """
    freq, psd = hrv_frequency(df.hr[df.peaks == 1].values[:-1], show=False)

    fbands = {'vlf': ['Very low frequency', (0.003, 0.04), 'blue'],
              'lf':	['Low frequency', (0.04, 0.15), 'green'],
              'hf':	['High frequency', (0.15, 0.4), 'red']}

    frequencyPlot = go.Figure()
    for f in ['vlf', 'lf', 'hf']:
        mask = (freq >= fbands[f][1][0]) & (freq <= fbands[f][1][1])
        frequencyPlot.add_trace(go.Scatter(
            x=freq[mask],
            y=psd[mask],
            fill='tozeroy',
            mode='lines',
            line_color=fbands[f][2]))
    frequencyPlot['layout']['plot_bgcolor'] = "#F9F9F9"
    frequencyPlot['layout']['paper_bgcolor'] = "#F9F9F9"
    frequencyPlot['layout']['autosize'] = True
    frequencyPlot['layout']['margin'] = dict(l=10, r=10, b=10, t=10)
    frequencyPlot['layout']['legend'] = dict(font=dict(size=15),
                                             orientation="h")
    return frequencyPlot
