# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Tuple

import numpy as np
from matplotlib.axes import Axes

from systole.plots import plot_ectopic, plot_shortLong


def plot_subspaces(
    artefacts: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 5),
) -> Tuple[Axes, Axes]:
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    artefacts : dict or None
        The artefacts detected using
        :py:func:`systole.detection.rr_artefacts()`.
    figsize : tuple, int or None
        Figure size. Defaults to `(10, 5)` when using matplotlib backend.

    Returns
    -------
    axs : tuple of :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    See also
    --------
    plot_events, plot_ectopic, plot_shortLong, plot_subspaces, plot_frequency,
    plot_timedomain, plot_nonlinear

    """
    ectopic = plot_ectopic(  # type: ignore
        artefacts=artefacts, figsize=figsize, input_type=None, backend="matplotlib"
    )
    shortLong = plot_shortLong(  # type: ignore
        artefacts=artefacts, figsize=figsize, input_type=None, backend="matplotlib"
    )

    return ectopic, shortLong
