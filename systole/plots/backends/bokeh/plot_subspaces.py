# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict

import numpy as np

from bokeh.layouts import row
from bokeh.models.layouts import Row
from systole.plots import plot_ectopic, plot_shortLong


def plot_subspaces(
    artefacts: Dict[str, np.ndarray],
    figsize: int = 600,
) -> Row:
    """Plot hrv subspace as described by Lipponen & Tarvainen (2019) [#]_.

    Parameters
    ----------
    artefacts : dict or None
        The artefacts detected using
        :py:func:`systole.detection.rr_artefacts()`.
    figsize : tuple, int or None
        Figure size. Defaults to `600` when using bokeh backend.

    Returns
    -------
    row : :class:`bokeh.models.layout.Row`
        The bokeh figure containing the two plots.

    See also
    --------
    plot_events, plot_ectopic, plot_shortLong, plot_subspaces, plot_frequency,
    plot_timedomain, plot_nonlinear
    """
    fig = row(
        plot_ectopic(  # type: ignore
            artefacts=artefacts, figsize=figsize, input_type=None, backend="bokeh"
        ),
        plot_shortLong(  # type: ignore
            artefacts=artefacts, figsize=figsize, input_type=None, backend="bokeh"
        ),
    )

    return fig
