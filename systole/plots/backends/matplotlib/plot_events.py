# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import itertools
from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from systole.recording import Oximeter


def plot_events(
    oximeter: "Oximeter",
    figsize: Tuple[float, float] = (13, 5),
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot events occurence across recording.

    Parameters
    ----------
    oximeter : `systole.recording.Oximeter`
        The recording instance, where additional channels track different
        events using boolean recording.
    figsize : tuple
        Figure size. Default is `(13, 5)`.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    palette = itertools.cycle(sns.color_palette("deep"))
    if oximeter.channels is not None:
        events = oximeter.channels.copy()
    else:
        raise ValueError("No event found")
    for i, ch in enumerate(events):
        ax.fill_between(
            x=oximeter.times,
            y1=i,
            y2=i + 0.5,
            color=next(palette),
            where=np.array(events[ch]) == 1,
        )

    # Add y ticks with channels names
    ax.set_yticks(np.arange(len(events)) + 0.5)
    ax.set_yticklabels([key for key in events])
    ax.set_xlabel("Time (s)")
    ax.set_title("Events", fontweight="bold")

    return ax
