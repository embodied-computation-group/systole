# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_events(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (13, 3),
    ax: Optional[Axes] = None,
    behavior=None,
) -> Axes:
    """Plot events to get a visual display of the paradigm (Matplotlib).

    Parameters
    ----------
    df :
        The events data frame (tmin, trigger, tmax, label, color, [behavior]).
    figsize :
        Figure size. Default is `(13, 5)`.
    ax :
        Where to draw the plot. Default is *None* (create a new figure).
    behavior :
        (Optional) Additional information about trials that will appear when hovering
        on the area (only relevant for `bokeh` backend). This parameter will be
        ignored.

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot first label only
    all_labels = []
    for i in range(len(df)):
        if df.loc[i, "label"] in all_labels:
            df.loc[i, "label"] = ""
        else:
            all_labels.append(df.loc[i, "label"])

    # Loop across events df
    for i, tmin, trigger, tmax, label, color in df.itertuples():
        # Plot time range
        ax.axvspan(xmin=tmin, xmax=tmax, color=color, alpha=0.2, label=label)

        # Plot trigger
        ax.axvline(x=trigger, color="gray", linestyle="--", linewidth=0.5)

    # Add y ticks with channels names
    ax.set_xlabel("Time")
    ax.legend()

    return ax
