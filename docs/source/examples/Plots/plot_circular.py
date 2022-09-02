"""

Plot circular
=============
The density function can be represented using the area of the bars, the height or
the transparency (alpha). The default behaviour will use the area. Using the heigth
can visually biase the importance of the largest values. Adapted from [#]_.

The circular mean was adapted from Pingouin's implementation [#]_.

.. [#] https://jwalton.info/Matplotlib-rose-plots/

.. [#] https://pingouin-stats.org/_modules/pingouin/circular.html#circ_mean

"""

# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
# Licence: GPL v3

#%%
# Using a numpy array of angular values as input
# ----------------------------------------------
import numpy as np
from systole.plots import plot_circular
x = np.random.normal(np.pi, 0.5, 100)
plot_circular(data=x)
#%%
# Using a data frame as input
# ---------------------------
import numpy as np
import pandas as pd
from systole.plots import plot_circular

# Create angular values (two conditions)
x = np.random.normal(np.pi, 0.5, 100)
y = np.random.uniform(0, np.pi*2, 100)
data = pd.DataFrame(data={'x': x, 'y': y}).melt()

plot_circular(data=data, y='value', hue='variable')