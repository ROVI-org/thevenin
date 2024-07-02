from __future__ import annotations
from typing import TYPE_CHECKING

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from ._ida_solver import SolverReturn

if TYPE_CHECKING:  # pragma: no cover
    from ._model import Model


class BaseSolution(SolverReturn):
    """Base solution."""

    def __init__(self) -> None:
        """
        The base solution class is a parent class to both the StepSolution
        and CycleSolution classes. Inheriting from this class gives each
        solution instance a 'vars' dictionary, access to the 'plot' method,
        and ensures that the slicing of the solution vector into 'vars' is
        consistent between all solutions.

        """

        self.vars = {}

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return a readable repr string.

        Returns
        -------
        readable : str
            A console-readable instance representation.

        """
        return repr(self.vars.keys())

    def plot(self, x: str, y: str, **kwargs) -> None:
        """
        Plot any two variables in 'vars' against each other.

        Parameters
        ----------
        x : str
            A variable key in 'vars' to be used for the x-axis.
        y : str
            A variable key in 'vars' to be used for the y-axis.

        Returns
        -------
        None.

        """

        plt.figure()
        plt.plot(self.vars[x], self.vars[y], **kwargs)

        if '_' in x:
            variable, units = x.split('_')
            xlabel = variable.capitalize() + ' [' + units + ']'
        else:
            xlabel = x

        if '_' in y:
            variable, units = y.split('_')
            ylabel = variable.capitalize() + ' [' + units + ']'
        else:
            ylabel = y

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.show(block=False)

    def _to_dict(self) -> None:
        """
        Fills the 'vars' dictionary by slicing the SolverReturn solution
        states. Users should generally only access the solution via 'vars'
        since names are more intuitive than interpreting 'y' directly.

        Returns
        -------
        None.

        """

        model = self._model
        ptr = model._ptr

        time = self.t

        soc = self.y[:, ptr['soc']]
        T_cell = self.y[:, ptr['T_cell']]*model.T_inf
        eta_j = self.y[:, ptr['eta_j']]
        voltage = self.y[:, ptr['V_cell']]

        ocv = model.ocv(soc)
        R0 = model.R0(soc, T_cell)

        current = -(voltage - ocv + np.sum(eta_j, axis=1)) / R0
        capacity = cumulative_trapezoid(-current, x=time/3600., initial=0.)

        # stored time
        self.vars['time_s'] = time
        self.vars['time_min'] = time / 60.
        self.vars['time_h'] = time / 3600.

        # from state variables
        self.vars['soc'] = soc
        self.vars['temperature_K'] = T_cell
        self.vars['voltage_V'] = voltage

        # post-processed variables
        self.vars['current_A'] = current
        self.vars['power_W'] = current*voltage
        self.vars['capacity_Ah'] = soc[0]*model.capacity + capacity
        self.vars['eta0_V'] = current*R0

        for j, eta in enumerate(eta_j.T, start=1):
            self.vars['eta' + str(j) + '_V'] = eta


class StepSolution(BaseSolution):
    """Single-step solution."""

    __slots__ = ('vars', '_model', '_timer', '_success', '_message', '_t',
                 '_y', '_ydot', '_roots', '_tstop', '_events', '_errors',)

    def __init__(self, model: Model, idasol: SolverReturn,
                 timer: float) -> None:
        """
        A solution instance for a single experimental step.

        Parameters
        ----------
        model : Model
            The model instance that was run to produce the solution.
        idasol : SolverReturn
            The unformatted solution returned by IDASolver.
        timer : float
            Amount of time it took for IDASolver to perform the integration.

        """

        super().__init__()

        self._model = deepcopy(model)

        self._success = idasol.success
        self._message = idasol.message
        self._t = idasol.t
        self._y = idasol.y
        self._ydot = idasol.ydot
        self._roots = idasol.roots
        self._tstop = idasol.tstop
        self._errors = idasol.errors

        self._timer = timer

        self._to_dict()

    @property
    def solvetime(self) -> str:
        """
        Print a statement specifying how long IDASolver spent integrating.

        Returns
        -------
        solvetime : str
            An f-string with the solver integration time in seconds.

        """
        return f"Solve time: {self._timer:.3f} s"


class CycleSolution(BaseSolution):
    """All-step solution."""

    __slots__ = ('vars', '_sols', '_model', '_timers', '_success', '_message',
                 '_t', '_y', '_ydot', '_roots', '_tstop', '_events', '_errors',)

    def __init__(self, *sols: StepSolution) -> None:
        """
        A solution instance with all experiment steps stitch together into
        a single cycle.

        Parameters
        ----------
        *sols : StepSolution
            All unpacked StepSolution instances to stitch together. The given
            steps should be given in the same sequential order that they were
            run.

        """

        super().__init__()

        self._sols = sols
        self._model = sols[0]._model

        sv_size = self._model._sv0.size

        self._success = []
        self._message = []
        self._t = np.empty([0])
        self._y = np.empty([0, sv_size])
        self._ydot = np.empty([0, sv_size])
        self._roots = []
        self._tstop = []
        self._errors = []
        self._timers = []

        for sol in sols:
            if self._t.size > 0:
                shifted_times = self._t[-1] + sol.t
            else:
                shifted_times = sol.t

            self._success.append(sol.success)
            self._message.append(sol.message)
            self._t = np.hstack([self._t, shifted_times])
            self._y = np.vstack([self._y, sol.y])
            self._ydot = np.vstack([self._ydot, sol.ydot])
            self._roots.append(sol.roots)
            self._tstop.append(sol.tstop)
            self._errors.append(sol.errors)
            self._timers.append(sol._timer)

        self._to_dict()

    @property
    def solvetime(self) -> str:
        """
        Print a statement specifying how long IDASolver spent integrating.

        Returns
        -------
        solvetime : str
            An f-string with the total solver integration time in seconds.

        """
        return f"Solve time: {sum(self._timers):.3f} s"

    def get_steps(self, idx: int | tuple) -> StepSolution | CycleSolution:
        """
        Return a subset of the solution.

        Parameters
        ----------
        idx : int | tuple
            The step index (int) or first/last indices (tuple) to return.

        Returns
        -------
        sol : StepSolution | CycleSolution
            The returned solution subset. A StepSolution is returned if 'idx'
            is an int, and a CycleSolution will be returned for the range of
            requested steps when 'idx' is a tuple.

        """

        if isinstance(idx, int):
            return deepcopy(self._sols[idx])
        elif isinstance(idx, (tuple, list)):
            sols = self._sols[idx[0]:idx[1] + 1]
            return CycleSolution(*sols)
