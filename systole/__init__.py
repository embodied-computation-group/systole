# Raise a warning if a newer version of Systole is available
from outdated import warn_if_outdated

from .correction import *
from .datasets import *
from .detection import *
from .hrv import *
from .plots import *
from .utils import *

__version__ = "0.2.0a"

warn_if_outdated("systole", __version__)
