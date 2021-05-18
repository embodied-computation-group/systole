# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


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
    y : str or list
        If data is a pandas instance, column containing the angular values.
    hue : str or list of strings
        Columns in data encoding the different conditions.
    **kwargs : Additional `_circular()` arguments.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    Examples
    --------
    .. plot::

       import numpy as np
       import pandas as pd
       from systole.plotting import plot_circular
       x = np.random.normal(np.pi, 0.5, 100)
       y = np.random.uniform(0, np.pi*2, 100)
       data = pd.DataFrame(data={'x': x, 'y': y}).melt()
       plot_circular(data=data, y='value', hue='variable')

    """

    _, axs = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))

    return axs
