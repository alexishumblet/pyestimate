"""

.. autosummary::
    :toctree: generated/
    
    pyestimate.sin_param_estimate
    pyestimate.sin2d_param_estimate
    pyestimate.multiple_sin_param_estimate
    pyestimate.pc_ar_estimator
    
"""

__version__ = "0.3.0"
__author__ = 'Alexis Humblet'
__credits__ = 'Alexis Humblet'
__all__ = ['sin2d_param_estimate', 'sin_param_estimate', 'multiple_sin_param_estimate', 'pc_ar_estimator', ]
from .estimators import *
