.. pyestimate documentation master file, created by
   sphinx-quickstart on Sun Jan 14 17:23:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

pyestimate documentation
========================

.. image:: images/pyestimate_logo.png
  :height: 192
  :alt: pyestimate logo

Installation
------------

Install pyestimate and its dependencies from PyPI using pip:

.. code-block:: python

    pip install pyestimate

Examples
--------

* `Frequency, amplitude and phase estimation of sinusoidal signal with noise <https://github.com/alexishumblet/pyestimate/blob/main/examples/sin_param_estimate_ex.ipynb>`_
* `Removal of 50Hz or 60Hz interferer without filtering <https://github.com/alexishumblet/pyestimate/blob/main/examples/60Hz_interference.ipynb>`_
* `Frequency, amplitude and phase estimation of 2D sinusoidal signal with noise <https://github.com/alexishumblet/pyestimate/blob/main/examples/sin2d_param_estimate_ex.ipynb>`_
* `Estimation of frequency, amplitude and phase for multiple sinusoids (less than 5) <https://github.com/alexishumblet/pyestimate/blob/main/examples/multiple_sin_param_estimate_ex.ipynb>`_
* `Principal components AR estimator for multiple sinusoids (faster/suboptimal algorithm) <https://github.com/alexishumblet/pyestimate/blob/main/examples/pc_ar_estimator_ex.ipynb>`_


Functions
---------

.. autosummary::
    pyestimate.sin_param_estimate
    pyestimate.sin2d_param_estimate
    pyestimate.multiple_sin_param_estimate
    pyestimate.pc_ar_estimator
    
.. automodule:: pyestimate
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
