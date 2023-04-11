# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from bokeh.plotting import figure


def plot_frequency(
    freq: np.ndarray,
    power: np.ndarray,
    figsize: Optional[Union[List[int], Tuple[int, int], int]] = None,
    fbands: Optional[Dict[str, Tuple[str, Tuple[float, float], str]]] = None,
    ax=None,
) -> figure:
    """Plot the frequency component of the heart rate variability.

    Parameters
    ----------
    freq :
        Frequencies.
    power :
        Power spectral density.
    fbands :
        Dictionary containing the names of the frequency bands of interest
        (str), their range (tuples) and their color in the PSD plot.
        Default is:
        >>> {'vlf': ('Very low frequency', (0.003, 0.04), 'b'),
        >>> 'lf': ('Low frequency', (0.04, 0.15), 'g'),
        >>> 'hf': ('High frequency', (0.15, 0.4), 'r')}
    figsize :
        Figure size. Default is `(13, 5)`.
    ax :
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    psd_plot :
        The boken figure containing the plot.

    """
    if isinstance(figsize, int):
        height, width = figsize, figsize
    elif figsize is None:
        figsize = (13, 5)
    elif isinstance(figsize, list) | isinstance(figsize, tuple):
        width, height = figsize

    psd_plot = figure(
        title="Power spectral density",
        height=height,
        width=width,
        x_axis_label="Frequency (Hz)",
        y_axis_label="PSD [s²/Hz]",
        output_backend="webgl",
    )

    if fbands is None:
        fbands = {
            "vlf": ("Very low frequency", (0.003, 0.04), "#4c72b0"),
            "lf": ("Low frequency", (0.04, 0.15), "#55a868"),
            "hf": ("High frequency", (0.15, 0.4), "#c44e52"),
        }

    for f in ["vlf", "lf", "hf"]:
        mask = (freq >= fbands[f][1][0]) & (freq <= fbands[f][1][1])

        # Line
        psd_plot.line(freq[mask], power[mask], color="grey")

        # Fill area
        psd_plot.varea(
            x=freq[mask], y1=0, y2=power[mask], color=fbands[f][2], alpha=0.2
        )

    return psd_plot
