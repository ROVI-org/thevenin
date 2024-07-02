"""
thevenin
========
The Thevenin equivalent circuit model is a common low-fidelity battery model
consisting of a single resistor in series with any number of RC pairs, i.e.,
parallel resistor-capacitor pairs. This Python package contains an API for
building and running experiments using Thevenin models.

How to use the documentation
----------------------------
Documentation is accessible via Python's ``help()`` function which prints
docstrings from a package, module, function, class, etc. In addition, you can
access the documentation by calling the built-in ``thevenin.docs()`` method to
open a local website. The website includes search functionality and examples,
beyond the code docstrings.

Viewing documentation using IPython
-----------------------------------
Start IPython and import ``thevenin``. To see what's available in ``thevenin``,
type ``thevenin.<TAB>`` (where ``<TAB>`` is the TAB key). To view type hints
and/or brief descriptions, type an open parenthesis ``(`` after any subpacakge,
module, class, function, etc. (e.g., ``thevenin.Model(``).

"""

from ._ida_solver import IDASolver
from ._experiment import Experiment
from ._model import Model
from ._solutions import StepSolution, CycleSolution

__version__ = '0.0.1'

__all__ = [
    'docs',
    'IDASolver',
    'Experiment',
    'Model',
    'StepSolution',
    'CycleSolution'
]
