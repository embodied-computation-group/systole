# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from systole.plots.utils import get_plotting_function


def plot_events(
    triggers: Optional[Union[List, np.ndarray]] = None,
    triggers_idx: Optional[Union[List, np.ndarray]] = None,
    events_labels: Optional[Union[Dict[str, str], List, str]] = None,
    tmin: float = -1.0,
    tmax: float = 10.0,
    sfreq: int = 1000,
    behavior: Optional[Union[List, pd.DataFrame]] = None,
    figsize: Optional[Union[int, List[int], Tuple[int, int]]] = None,
    ax: Optional[Axes] = None,
    backend: str = "matplotlib",
    palette: Optional[List[str]] = None,
) -> Axes:
    """Plot events occurence across recording.

    Parameters
    ----------
    triggers : list | np.ndarray
        The events triggers. `0` indicates no events, `1` indicates the triger
        for one event. Different conditions should be provided separately as list
        of arrays.
    triggers_idx : list | np.ndarray
        Trigger indexes. Each value encode the sample where an event occured (see
        also `sfreq`). Different conditions should be provided separately as list of
        arrays (can have different lenght).
    events_labels : dict
        The events label. The key of the dictionary is the condition number (from 1 to
        n, as `str`), the value is the label (`str`). Default set to
        `{"1": "Event - 1"}` if one condition is provided, and generalize up to n
        conditions `{"n": "Event - n"}`.
    tmin, tmax : float
        Start and end time of the epochs in seconds, relative to the time-locked event.
        Defaults to -1.0 and 10.0, respectively.
    sfreq : int
        Signal sampling frequency. Default is set to 1000 Hz.
    behavior : list | py:class:`pandas.DataFrame`
        Additional information about trials that should appear when hovering on the
        trial (`bokeh` version only). A py:class:`pd.DataFrame` instance with length =
        n trials, or a list of py:class:`pd.DataFrame` (for multiple conditions) should
        be provided.
    figsize : tuple
        Figure size. Default is `(13, 5)`.
    ax : :class:`matplotlib.axes.Axes` | :class:`bokeh.plotting.figure.Figure` | None
        Where to draw the plot. Default is `None` (create a new figure).
    backend: str
        Select plotting backend (`"matplotlib"`, `"bokeh"`). Defaults to
        `"matplotlib"`.
    palette : list | None
        Color palette. Default sets to Seaborn `"deep"`.

    .. warning:: The `behavior` parameter will be implemented in a future release.

    Returns
    -------
    plot : :class:`matplotlib.axes.Axes` | :class:`bokeh.plotting.figure.Figure`
        The matplotlib axes, or the boken figure containing the plot.

    See also
    --------
    plot_rr, plot_raw

    Examples
    --------

    Plot events distributions using the Matplotlib backend.

    .. jupyter-execute::

       import numpy as np
       import seaborn as sns
       from systole.plots import plot_events
       from systole import import_dataset1

       ecg_df = import_dataset1(modalities=['ECG', "Stim"])

       # Get events triggers
       triggers_idx = [
            np.where(ecg_df.stim.to_numpy() == 2)[0],
            np.where(ecg_df.stim.to_numpy() == 1)[0]
       ]

       plot_events(
           triggers_idx=triggers_idx, events_labels=["Disgust", "Neutral"],
           tmin=-0.5, tmax=10.0, figsize=(13, 3),
           palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
        )

    Plot events distributions using the Bokeh backend and add RR time series.

    .. jupyter-execute::

       from systole.detection import ecg_peaks
       from systole.plots import plot_rr
       from bokeh.io import output_notebook
       from bokeh.plotting import show
       output_notebook()

       # Peak detection in the ECG signal using the Pan-Tompkins method
       signal, peaks = ecg_peaks(ecg_df.ecg, method='pan-tompkins', sfreq=1000)

       # First, we create a RR interval plot
       rr_plot = plot_rr(peaks, input_type='peaks', backend='bokeh', figsize=250)

       show(
           # Then we add events annotations to this plot using the plot_events function
           plot_events(triggers_idx=triggers_idx, backend="bokeh", events_labels=["Disgust", "Neutral"],
                       tmin=-0.5, tmax=10.0, palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
                       ax=rr_plot.children[0])
       )

    """
    # Define color palette
    if palette is None:
        this_palette = itertools.cycle(sns.color_palette("deep", as_cmap=True))
    elif isinstance(palette, list):
        this_palette = itertools.cycle(palette)
    else:
        raise ValueError("Invalid palette provided.")

    # Define figure size
    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 300

    # Create a list of triggers indexs with length n = number of conditions
    if triggers_idx is None:
        if triggers is None:
            raise ValueError("No triggers provided")
        else:
            if isinstance(triggers, np.ndarray):
                triggers = [triggers]
            triggers_idx = []
            for this_triggers in triggers:
                triggers_idx.append(np.where(this_triggers)[0])

    # Create the event dictionnary if not already provided
    if events_labels is None:
        events_labels = {}
        for i in range(len(triggers_idx)):
            events_labels[f"{i+1}"] = f"Event - {i+1}"
    elif isinstance(events_labels, str):
        event_str = events_labels
        events_labels = {}
        events_labels["1"] = event_str
    elif isinstance(events_labels, list):
        event_list = events_labels
        events_labels = {}
        for i, lab in enumerate(event_list):
            events_labels[f"{i+1}"] = lab
    else:
        raise ValueError("Invalid event label provided.")

    # Creating the events info df
    # DataFrame : (tmin, trigger, tmax, label, color, df)
    df = pd.DataFrame([])

    # Loop across conditions
    for i, this_trigger_idx in enumerate(triggers_idx):

        # Event color
        col = next(this_palette)

        # Loop across triggers
        for event in this_trigger_idx:
            this_tmin = (event / sfreq) + tmin
            this_trigger = event / sfreq
            this_tmax = (event / sfreq) + tmax
            df = df.append(
                pd.DataFrame(
                    {
                        "tmin": this_tmin,
                        "trigger": this_trigger,
                        "tmax": this_tmax,
                        "label": events_labels[str(i + 1)],
                        "color": [col],
                    }
                ),
                ignore_index=True,
            )

            # Add behaviors results when provided

    df["tmin"] = pd.to_datetime(df["tmin"], unit="s", origin="unix")
    df["trigger"] = pd.to_datetime(df["trigger"], unit="s", origin="unix")
    df["tmax"] = pd.to_datetime(df["tmax"], unit="s", origin="unix")

    plot_raw_args = {
        "df": df,
        "figsize": figsize,
        "ax": ax,
    }

    plotting_function = get_plotting_function("plot_events", "plot_events", backend)
    plot = plotting_function(**plot_raw_args)

    return plot
