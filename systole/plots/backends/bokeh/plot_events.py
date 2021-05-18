# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import itertools
from typing import Dict, Optional

import numpy as np
import seaborn as sns

from bokeh.plotting import figure
from bokeh.plotting.figure import Figure


def plot_events(
    events: np.ndarray,
    events_dict: Optional[Dict] = None,
    sfreq: int = 75,
    figsize: int = 300,
    **kwargs,
) -> Figure:
    """Plot events occurence across recording.

    Parameters
    ----------
    events : :py:class:`numpy.ndarray`
        The events.
    events_dict : dict
        A dictionary indexing the event names ({"1": "Event number 1"}).
    sfreq : int
        The sampling frequency.
    figsize : int
        Figure size. Default is `300`.

    Returns
    -------
    fig : :class:`bokeh.plotting.figure.Figure`
        The bokeh figure containing the plot.
    """
    unique_events = [e for e in np.unique(events) if e != 0]

    if events_dict is None:
        events_dict = {}
        for i, ev in enumerate(unique_events):
            events_dict[str(i)] = f"Event {i}"

    # Create a time vector
    time = np.datetime64(np.datetime64("now", "ms")) + (
        (np.arange(0, len(events)) / sfreq) * 1000
    ).astype("int")

    events_plot = figure(
        title="Events",
        x_axis_type="datetime",
        toolbar_location=None,
        sizing_mode="stretch_width",
        plot_height=figsize,
        x_axis_label="Time (s)",
        output_backend="webgl",
        x_range=(time[0], time[-1]),
    )

    palette = itertools.cycle(sns.color_palette().as_hex())
    for i, ev in enumerate(unique_events):
        event_idx = np.where(events == ev)[0]
        events_plot.circle(
            time[event_idx],
            i + 1,
            size=20,
            alpha=0.5,
            color=next(palette),
            legend_label=events_dict[str(i + 1)],
        )

    return events_plot
