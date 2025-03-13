from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import numpy as np

from thevenin._basemodel import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    from .solvers import CVODEResult


class TransientState:
    """
    The key/value pairs and units are given in the table below.

        ======== ==================== ==================
        Key      Value                *type*, units
        ======== ==================== ==================
        soc      state of charge      *float*, -
        T_cell   cell temperature     *float*, K
        hyst     hysteresis           *float*, V
        eta_j    RC overpotentials    *list[float]*, V
        ======== ==================== ==================

    """

    def __init__(self, soc: float, T_cell: float, hyst: float,
                 eta_j: np.ndarray) -> None:

        self.soc = soc
        self.T_cell = T_cell
        self.hyst = hyst
        self.eta_j = eta_j

        self._voltage = None

    @property
    def num_RC_pairs(self) -> int | None:
        if self.eta_j is None:
            return None
        else:
            return len(self.eta_j)

    @property
    def voltage(self) -> float | None:
        return self._voltage

    def _set_voltage(self, voltage: float) -> None:
        self._voltage = voltage


class Prediction(BaseModel):
    """
    Prediction model wrapper.

    This class is primarily intended to interface with prediction-correction
    algorithms, e.g., Kalman filters. The ``take_step`` method progresses the
    model forward by a single step, starting from a user-defined state.

    """

    def pre(self) -> None:

        self._check_RC_pairs()  # inherited from BaseModel

        ptr = {}
        ptr['soc'] = 0
        ptr['T_cell'] = 1
        ptr['hyst'] = 2
        ptr['eta_j'] = np.arange(3, 3 + self.num_RC_pairs)
        ptr['size'] = self.num_RC_pairs + 3

        self._ptr = ptr

        self.set_options()

    def set_options(self, **options) -> None:
        from .solvers import CVODESolver

        self._userdata = {}
        options = {'userdata': self._userdata, **options}
        self._solver = CVODESolver(self._svdot, **options)

    def take_step(self, state: TransientState, current: float | Callable,
                  delta_t: float) -> TransientState:
        """
        Take a step forward in time to predict the new state and voltage given
        a starting state, demand current, and time step.

        Parameters
        ----------
        state : TransientState
            Description of the starting state.
        current : float | Callable
            Demand current [A]. For a dynamic current, use a callable with a
            signature like ``def current(t: float) -> float``, where the input
            time is in seconds relative to the overall step.
        delta_t : float
            Magnitude of time step, in seconds.

        Returns
        -------
        :class:`~thevenin.TransientState`
            Predicted state at the end of the time step.

        """

        if callable(current):
            self._userdata['current'] = current
        else:
            self._userdata['current'] = lambda t: current

        sv0 = self._to_array(state)

        _ = self._solver.init_step(0., sv0)
        soln = self._solver.step(delta_t)

        # state prediction
        state = self._to_state(soln)

        # voltage prediction
        ocv = self.ocv(state.soc)
        R0 = self.R0(state.soc, state.T_cell)

        current = self._userdata['current'](soln.t)
        voltage = ocv + state.hyst - np.sum(state.eta_j) - current*R0

        state._set_voltage(voltage)

        return state

    def _to_state(self, soln: CVODEResult) -> dict:

        ptr = self._ptr

        state = {
            'soc': soln.y[ptr['soc']].item(),
            'T_cell': soln.y[ptr['T_cell']].item() * self.T_inf,
            'hyst': soln.y[ptr['hyst']].item(),
            'eta_j': soln.y[ptr['eta_j']],
        }

        return TransientState(**state)

    def _to_array(self, state: TransientState) -> np.ndarray:

        if state.num_RC_pairs != self.num_RC_pairs:
            raise ValueError(f"{state['eta_j']=} has an invalid length since"
                             f" num_RC_pairs={self.num_RC_pairs}.")

        ptr = self._ptr

        sv = np.zeros(self._ptr['size'])
        sv[ptr['soc']] = state.soc
        sv[ptr['T_cell']] = state.T_cell / self.T_inf
        sv[ptr['hyst']] = state.hyst
        sv[ptr['eta_j']] = state.eta_j

        return sv

    def _svdot(self, t: float, sv: np.ndarray, svdot: np.ndarray,
               userdata: dict) -> None:
        """
        Solver-structured right-hand-side.

        The CVODESolver requires a right-hand-side function in this form.
        Rather than outputting the derivatives, the function returns None,
        but fills the 'svdot' input array with the ODE expressions.

        Parameters
        ----------
        t : float
            Value of time [s].
        sv : 1D np.array
            State variables at time t.
        svdot : 1D np.array
            State variable time derivatives from ODEs, svdot = rhs(t, sv).
        userdata : dict
            Dictionary detailing an experimental step.

        Returns
        -------
        None.

        """

        svdot[:] = self._rhsfn(t, sv, userdata)
