# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict

import numpy as np
from bokeh.layouts import row
from bokeh.models.layouts import Row

from systole.plots import plot_ectopic, plot_shortlong


def plot_subspaces(
    artefacts: Dict[str, np.ndarray], figsize: int = 600, **kwargs
) -> Row:
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019) [#]_.

    Parameters
    ----------
    artefacts :
        The artefacts detected using
        :py:func:`systole.detection.rr_artefacts()`.
    figsize :
        Figure size. Defaults to `600` when using bokeh backend.

    Returns
    -------
    fig :
        The bokeh figure containing the two plots.

    """
    fig = row(
        plot_ectopic(  # type: ignore
            artefacts=artefacts, figsize=figsize, input_type=None, backend="bokeh"
        ),
        plot_shortlong(  # type: ignore
            artefacts=artefacts, figsize=figsize, input_type=None, backend="bokeh"
        ),
    )

    return fig
