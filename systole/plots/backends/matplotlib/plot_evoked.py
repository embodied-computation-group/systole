# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_evoked(
    epochs: List[np.ndarray],
    time: np.ndarray,
    figsize: Tuple[float, float],
    labels: List[str],
    unit: str,
    ax=None,
    **kwargs
) -> Axes:
    """Plot events occurence across recording.

    Parameters
    ----------
    epochs :
        A 2d (trial * time) numpy array containing the time series
        of the epoched signal.
    time :
        Start and end time of the epochs in seconds, relative to the
        time-locked event. Defaults to -1 and 10, respectively.
    figsize :
        The lines color.
    labels :
        The different condition/participants label/IDs.
    unit :
        The heart rate unit. Can be `'rr'` (R-R intervals, in ms) or `'bpm'` (beats
        per minutes). Default is `'bpm'`.
    ax :
        Figure size. Default is `(13, 5)`.
    kwargs :
        Other keyword arguments are passed down to py:`func:seaborn.lineplot()`.

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if isinstance(labels, str):
        labels = [labels]

    ax.axvline(x=0, linestyle="--", color="gray")
    ax.axhline(y=0, color="black", linewidth=1)

    # Loop across the many condition/participants provided and create a long data frame
    # that can be passed to Seaborn's lineplot() with custom args
    epoch_df = pd.DataFrame([])
    for ep, lab in zip(epochs, labels):
        # Create a dataframe for seaborn
        df = pd.DataFrame(ep.T)
        df["Time"] = time
        df = df.melt(id_vars=["Time"], var_name="subject", value_name="heart_rate")
        df["Label"] = lab
        epoch_df = pd.concat([epoch_df, df], ignore_index=True)

    # Use Seaborn lineplot with the transformed data
    sns.lineplot(data=epoch_df, x="Time", y="heart_rate", hue="Label", ax=ax, **kwargs)

    ax.set_xlabel("Time (s)")
    ylabel = "R-R interval change (ms)" if unit == "rr" else "Heart rate change (bpm)"
    ax.set_ylabel(ylabel)

    return ax
