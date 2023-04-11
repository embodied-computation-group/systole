# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_frequency(
    freq: np.ndarray,
    power: np.ndarray,
    figsize: Optional[Union[List[int], Tuple[int, int], int]] = None,
    fbands: Optional[Dict[str, Tuple[str, Tuple[float, float], str]]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot the frequency component of the heart rate variability.

    Parameters
    ----------
    freq :
        Frequencies.
    power :
        Power spectral density.
    figsize :
        Figure size. Default is `(8, 5)`.
    fbands :
        Dictionary containing the names of the frequency bands of interest
        (str), their range (tuples) and their color in the PSD plot.
        Default is:
        >>> {'vlf': ('Very low frequency', (0.003, 0.04), 'b'),
        >>> 'lf': ('Low frequency', (0.04, 0.15), 'g'),
        >>> 'hf': ('High frequency', (0.15, 0.4), 'r')}
    ax :
        Where to draw the plot. Default is `None` (create a new figure).

     Returns
     -------
     ax  :
        The matplotlib axes containing the plot.

    """
    if figsize is None:
        figsize = (8, 5)
    elif isinstance(figsize, int):
        figsize = (figsize, figsize)
    else:
        if len(figsize) == 1:
            figsize = (figsize[0], figsize[0])

    if fbands is None:
        fbands = {
            "vlf": ("Very low frequency", (0.003, 0.04), "#4c72b0"),
            "lf": ("Low frequency", (0.04, 0.15), "#55a868"),
            "hf": ("High frequency", (0.15, 0.4), "#c44e52"),
        }

    # Plot the PSD
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, power, "k")
    for f in ["vlf", "lf", "hf"]:
        mask = (freq >= fbands[f][1][0]) & (freq <= fbands[f][1][1])
        ax.fill_between(freq, power, where=mask, color=fbands[f][2], alpha=0.2)
        ax.axvline(x=fbands[f][1][0], linestyle="--", color="gray")
    ax.set_xlim(0.003, 0.4)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [$s^2$/Hz]")
    ax.set_title("Power Spectral Density", fontweight="bold")

    return ax
