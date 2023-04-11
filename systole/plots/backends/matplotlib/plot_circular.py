# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_circular(
    data: List[Union[float, List[float], np.ndarray]],
    palette: List[str],
    labels: List[str],
    ax: Axes,
    units: str = "radians",
    bins: int = 32,
    density: str = "area",
    norm: bool = True,
    mean: bool = False,
    offset: float = 0.0,
    **kwargs
) -> Axes:
    """Plot polar histogram.

    This function is an internal function used by:py:func`systole.plots.plot_circular`.

    Parameters
    ----------
    data :
        List of numpy arrays.
    palette :
        Color palette. Default sets to Seaborn `"deep"`.
    labels :
        The conditions labels.
    units :
        Unit of the angular values provided. Can be `"degree"` or `"radian"`.
        Default sets to `"radians"`.
    bins :
        Number of slices in the circle. Use even value to have a bin edge at zero.
    density :
        How to represent the density of the circular distribution. Can be one of the
        following:
        - `"area"`: use the area of the circular bins.
        - `"height"`: use the height of the circular bins.
        - `"alpha"`: change the transparency of the circular bins.
        Default set to `"area"`. This method should be prefered over `"height"` as
        increasing the height of the bars is increasin their visual importance (area)
        non linearly. The `"area"` method can control for this bias.
    norm :
        If `True` (default), normalize the distribution between 0 and 1.
    mean :
        If `True`, show the mean and 95% CI. Default set to `False`.
    offset :
        Where 0 will be placed on the circle, in radians. Default set to `0`.
    ax :
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    ax :
        The matplotlib axes containing the plot.

    """

    # Loop across conditions
    for angles, color, label in zip(data, palette, labels):
        angles = np.asarray(angles)

        # Bin data and count
        count, bin = np.histogram(angles, bins=bins, range=(0, np.pi * 2))

        # Compute width
        widths = np.diff(bin)[0]

        if density == "area":  # Default
            # Area to assign each bin
            area = count / angles.size
            # Calculate corresponding bin radius
            radius = (area / np.pi) ** 0.5
            alpha = (count * 0) + 1.0
        elif density == "height":  # Using height (can be misleading)
            radius = count / angles.size
            alpha = (count * 0) + 1.0
        elif density == "alpha":  # Using transparency
            radius = (count * 0) + 1.0
            # Alpha level to each bin
            alpha = count / angles.size
            alpha = alpha / alpha.max()
        else:
            raise ValueError("Invalid method")

        if norm is True:
            radius = radius / radius.max()

        # Plot data on ax
        for b, r, a in zip(bin[:-1], radius, alpha):
            ax.bar(
                b,
                r,
                align="edge",
                width=widths,
                edgecolor="k",
                linewidth=1,
                color=color,
                alpha=a,
            )

        # Plot mean and CI
        if mean:
            # Use pingouin.circ_mean() method
            alpha = np.array(angles)
            w = np.ones_like(alpha)
            circ_mean = np.angle(np.multiply(w, np.exp(1j * alpha)).sum(axis=0))
            ax.plot(circ_mean, radius.max(), "ko")

        # Set the direction of the zero angle
        ax.set_theta_offset(offset)

        # Remove ylabels
        ax.set_yticks([])

        if units == "radians":
            circle_label = [
                "$0$",
                r"$\pi/4$",
                r"$\pi/2$",
                r"$3\pi/4$",
                r"$\pi$",
                r"$5\pi/4$",
                r"$3\pi/2$",
                r"$7\pi/4$",
            ]
            ax.set_xticklabels(circle_label)
    plt.tight_layout()

    return ax
