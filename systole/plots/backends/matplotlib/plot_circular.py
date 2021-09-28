# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import itertools
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def circular(
    data: Union[List, np.ndarray],
    bins: int = 32,
    density: str = "area",
    offset: float = 0.0,
    mean: bool = False,
    norm: bool = True,
    units: str = "radians",
    color: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot polar histogram.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray` or list
        Angular values, in radians.
    bins : int
        Use even value to have a bin edge at zero.
    density : str
        Is the density represented via the height or the area of the bars.
        Default set to 'area' (avoid misleading representation).
    offset : float
        Where 0 will be placed on the circle, in radians. Default set to 0
        (right).
    mean : bool
        If `True`, show the mean and 95% CI. Default set to `False`
    norm : boolean
        Normalize the distribution between 0 and 1.
    units : str
        Unit of the angular representation. Can be `'degree'` or `'radian'`.
        Default set to 'radians'.
    color : str
        The color of the bars.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    Notes
    -----
    The density function can be represented using the area of the bars, the
    height or the transparency (alpha). The default behaviour will use the
    area. Using the heigth can visually biase the importance of the largest
    values. Adapted from [#]_.

    The circular mean was adapted from the implementation of the pingouin
    python package [#]_

    Examples
    --------
    Plot polar data.

    .. plot::

       import numpy as np
       from systole.plots import circular
       x = np.random.normal(np.pi, 0.5, 100)
       circular(x)

    References
    ----------
    .. [#] https://jwalton.info/Matplotlib-rose-plots/

    .. [#] https://pingouin-stats.org/_modules/pingouin/circular.html#circ_mean

    """
    data = np.asarray(data)

    if color is None:
        color = "#539dcc"

    if ax is None:
        ax = plt.subplot(111, polar=True)

    # Bin data and count
    count, bin = np.histogram(data, bins=bins, range=(0, np.pi * 2))

    # Compute width
    widths = np.diff(bin)[0]

    if density == "area":  # Default
        # Area to assign each bin
        area = count / data.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** 0.5
        alpha = (count * 0) + 1
    elif density == "height":  # Using height (can be misleading)
        radius = count / data.size
        alpha = (count * 0) + 1
    elif density == "alpha":  # Using transparency
        radius = (count * 0) + 1
        # Alpha level to each bin
        alpha = count / data.size
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
        alpha = np.array(data)
        w = np.ones_like(alpha)
        circ_mean = np.angle(np.multiply(w, np.exp(1j * alpha)).sum(axis=0))
        ax.plot(circ_mean, radius.max(), "ko")

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels
    ax.set_yticks([])

    if units == "radians":
        label = [
            "$0$",
            r"$\pi/4$",
            r"$\pi/2$",
            r"$3\pi/4$",
            r"$\pi$",
            r"$5\pi/4$",
            r"$3\pi/2$",
            r"$7\pi/4$",
        ]
        ax.set_xticklabels(label)
    plt.tight_layout()

    return ax


def plot_circular(
    data: pd.DataFrame,
    y: Union[str, List, None] = None,
    hue: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> Axes:
    """Plot polar histogram.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Angular data (rad.).
    y : str | list
        If data is a pandas instance, column containing the angular values.
    hue : str | list
        Columns in data encoding the different conditions.
    **kwargs : Additional `_circular()` arguments.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    """
    # Check data format
    if isinstance(data, pd.DataFrame):
        assert data.shape[0] > 0, "Data must have at least 1 row."

    palette = itertools.cycle(sns.color_palette("deep"))

    if hue is None:
        ax = circular(data[y].values, **kwargs)

    else:
        n_plot = data[hue].nunique()

        _, axs = plt.subplots(1, n_plot, subplot_kw=dict(projection="polar"))

        for i, cond in enumerate(data[hue].unique()):

            x = data[y][data[hue] == cond]

            ax = circular(x, color=next(palette), ax=axs[i], **kwargs)

    return ax
