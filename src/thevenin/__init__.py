"""
Summary
=======
The Thevenin equivalent circuit model is a common low-fidelity battery model
consisting of a single resistor in series with any number of RC pairs, i.e.,
parallel resistor-capacitor pairs. This Python package contains an API for
building and running experiments using Thevenin models.

Accessing the documentation
---------------------------
Documentation is accessible via Python's ``help()`` function which prints
docstrings from a package, module, function, class, etc. You can also access
the documentation by visiting the website, hosted through GitHub pages. The
website includes search functionality and more detailed examples.

"""

# core package
from ._ida_solver import IDASolver
from ._experiment import Experiment
from ._model import Model
from ._solutions import StepSolution, CycleSolution

# submodules
from . import loadfns
from . import plotutils

__version__ = '0.1.1'

__all__ = [
    'IDASolver',
    'Experiment',
    'Model',
    'StepSolution',
    'CycleSolution',
    'loadfns',
    'plotutils',
]
