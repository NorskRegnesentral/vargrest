import os
from vargrest.api import estimate_variogram_parameters


__version__ = open(os.path.join(os.path.dirname(__file__), 'VERSION.txt')).read()
__all__ = ['estimate_variogram_parameters', '__version__']
