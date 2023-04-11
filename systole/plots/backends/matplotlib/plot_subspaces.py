# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes

from systole.plots import plot_ectopic, plot_shortlong


def plot_subspaces(
    artefacts: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 5),
    ax: Optional[Union[Tuple, List]] = None,
) -> Tuple[Axes, Axes]:
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    artefacts :
        The artefacts detected using
        :py:func:`systole.detection.rr_artefacts()`.
    figsize :
        Figure size. Defaults to `(10, 5)` when using matplotlib backend.

    Returns
    -------
    axs :
        The matplotlib axes containing the plot.

    """
    if ax is None:
        ectopic_ax, short_long_ax = None, None
    else:
        ectopic_ax, short_long_ax = ax

    ectopic = plot_ectopic(  # type: ignore
        artefacts=artefacts,
        figsize=figsize,
        input_type=None,
        backend="matplotlib",
        ax=ectopic_ax,
    )
    shortLong = plot_shortlong(  # type: ignore
        artefacts=artefacts,
        figsize=figsize,
        input_type=None,
        backend="matplotlib",
        ax=short_long_ax,
    )

    return ectopic, shortLong
