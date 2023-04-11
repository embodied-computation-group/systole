# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.plotting._figure import figure
from matplotlib.axes import Axes

from systole.plots.utils import get_plotting_function


def plot_events(
    triggers: Optional[Union[List, np.ndarray]] = None,
    triggers_idx: Optional[Union[List, np.ndarray]] = None,
    labels: Optional[Union[Dict[str, str], List, str]] = None,
    tmin: float = -1.0,
    tmax: float = 10.0,
    sfreq: int = 1000,
    behavior: Optional[Union[List, pd.DataFrame]] = None,
    figsize: Optional[Union[int, List[int], Tuple[int, int]]] = None,
    ax: Optional[Axes] = None,
    backend: str = "matplotlib",
    palette: Optional[List[str]] = None,
) -> Union[Axes, figure]:
    """Visualize the occurence of events along the physiological recording.

    Parameters
    ----------
    triggers :
        The events triggers. `0` indicates no events, `1` indicates the triger
        for one event. Different conditions should be provided separately as list
        of arrays.
    triggers_idx :
        Trigger indexes. Each value encode the sample where an event occured (see
        also `sfreq`). Different conditions should be provided separately as list of
        arrays (can have different lenght).
    labels :
        The events label. The key of the dictionary is the condition number (from 1 to
        n, as `str`), the value is the label (`str`). Default set to
        `{"1": "Event - 1"}` if one condition is provided, and generalize up to n
        conditions `{"n": "Event - n"}`.
    tmin, tmax :
        Start and end time of the epochs in seconds, relative to the time-locked event.
        Defaults to `-1.0` and `10.0`, respectively.
    sfreq :
        Signal sampling frequency. Default is set to 1000 Hz.
    behavior :
        (Optional) Additional information about trials that will appear when hovering
        on the area (`bokeh` version only). A py:class:`pd.DataFrame` instance with
        length = n trials, or a list of py:class:`pd.DataFrame` (for multiple
        conditions) should be provided.
    figsize :
        Figure size. Default is `(13, 5)`.
    ax : :class:`matplotlib.axes.Axes` | :class:`bokeh.plotting.figure.Figure` | None
        Where to draw the plot. Default is `None` (create a new figure).
    backend :
        Select plotting backend (`"matplotlib"`, `"bokeh"`). Defaults to
        `"matplotlib"`.
    palette :
        Color palette. Default sets to Seaborn `"deep"`.

    Returns
    -------
    plot :
        The matplotlib axes, or the boken figure containing the plot.

    See also
    --------
    plot_rr, plot_raw

    Raises
    ------
    ValueError
        When no triggers or triggers indexes are provided.
        When both triggers and triggers indexes are provided.
        If the length of behavior optional data does not match with the provide triggers.
        If invalid event names are provided.

    Examples
    --------

    Plot events distributions using Matplotlib as plotting backend.

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
           triggers_idx=triggers_idx, labels=["Disgust", "Neutral"],
           tmin=-0.5, tmax=10.0, figsize=(13, 3),
           palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
        )

    Plot events distributions using Bokeh as plotting backend and add the RR time series.

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

       # Then we add events annotations to this plot using the plot_events function
       show(
            plot_events(
                triggers_idx=triggers_idx, labels=["Disgust", "Neutral"],
                tmin=-0.5, tmax=10.0, ax=rr_plot.children[0], backend="bokeh",
                palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["pale red"]],
            )
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

    # Create a list of triggers indexs (if provided) with length n = number of conditions
    if (triggers_idx is None) & (triggers is None):
        raise ValueError("No triggers provided")
    elif triggers_idx is None:
        if isinstance(triggers, np.ndarray):
            triggers = [triggers]

        # Transform the boolean vector into triggers indexes
        triggers_idx = []
        for this_triggers in triggers:  # type: ignore
            triggers_idx.append(np.where(this_triggers)[0])

    elif triggers is None:
        if isinstance(triggers_idx, np.ndarray):
            triggers_idx = [triggers_idx]
    else:
        raise ValueError("Both triggers and triggers indexes are provided.")

    # Check that the behaviors data (if provided) match with events length
    if behavior is not None:
        if isinstance(behavior, pd.DataFrame):
            behavior = [behavior]
        for tr, bh in zip(triggers_idx, behavior):
            if len(tr) != len(bh):
                raise ValueError(
                    "The length of triggers indexes and behavior data does not match"
                )

    # Create the event dictionary if not already provided
    if labels is None:
        labels = {}
        for i in range(len(triggers_idx)):
            labels[f"{i+1}"] = f"Event - {i+1}"
    elif isinstance(labels, str):
        event_str = labels
        labels = {}
        labels["1"] = event_str
    elif isinstance(labels, list):
        event_list = labels
        labels = {}
        for i, lab in enumerate(event_list):
            labels[f"{i+1}"] = lab
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
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "tmin": this_tmin,
                            "trigger": this_trigger,
                            "tmax": this_tmax,
                            "label": labels[str(i + 1)],
                            "color": [col],
                        }
                    ),
                ],
                ignore_index=True,
            )

    df["tmin"] = pd.to_datetime(df["tmin"], unit="s", origin="unix")
    df["trigger"] = pd.to_datetime(df["trigger"], unit="s", origin="unix")
    df["tmax"] = pd.to_datetime(df["tmax"], unit="s", origin="unix")

    plot_raw_args = {"df": df, "figsize": figsize, "ax": ax, "behavior": behavior}

    plotting_function = get_plotting_function("plot_events", "plot_events", backend)
    plot = plotting_function(**plot_raw_args)

    return plot
