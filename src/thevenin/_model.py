from __future__ import annotations
from typing import TYPE_CHECKING

import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sparse
from ruamel.yaml import YAML, add_constructor, SafeConstructor

if TYPE_CHECKING:  # pragma: no cover
    from ._experiment import Experiment
    from ._solutions import StepSolution, CycleSolution


class Model:
    """Circuit model."""

    def __init__(self, params: dict | str = 'params.yaml'):
        """
        A class to construct and run the model. Provide the parameters using
        either a dictionary or a '.yaml' file. Note that the number of Rj and
        Cj attributes must be consistent with the num_RC_pairs value. See the
        notes for more information on the callable parameters.

        Parameters
        ----------
        params : dict | str
            Mapping of model parameter names to their values. Can be either
            a dict or absolute/relateive file path to a yaml file (str). The
            keys/value pair descriptions are given below. The default uses a
            .yaml file. Use the templates() function to view this file.

            ============= =========================================
            Key           Value (*type*, units)
            ============= =========================================
            num_RC_pairs  number of RC pairs (*int*, -)
            soc0          initial state of charge (*float*, -)
            capacity      maximum battery capacity (*float*, Ah)
            mass          total battery mass (*float*, kg)
            isothermal    flag for isothermal model (*bool*, -)
            Cp            specific heat capacity (*float*, J/kg/K)
            T_inf         room/air temperature (*float*, K)
            h_therm       convective coefficient (*float*, W/m2/K)
            A_therm       heat loss area (*float*, m2)
            ocv           open circuit voltage (*callable*, V)
            R0            series resistance (*callable*, Ohm)
            Rj            resistance in RCj (*callable*, Ohm)
            Cj            capacity in RCj (*callable*, F)
            ============= =========================================

        Raises
        ------
        TypeError
            'params' must be type dict or str.
        ValueError
            'params' contains invalid and/or excess key/value pairs.

        Warnings
        --------
        A pre-processor runs at the end of the model initialization. If you
        modify any parameters after class instantiation, you will need to
        manually re-run the pre-processor (i.e., the pre() method) afterward.

        Notes
        -----
        The ocv property should have a signature like f(soc: float) -> float,
        where soc is the time-dependent state of charged solved for within
        the model. All R0, Rj, and Cj properties should have signatures like
        f(soc: float, T_cell: float) -> float, where T_cell is the temperature
        in K determined in the model.

        Rj and Cj are not true property names. These are just used generally
        in the documentation. If num_RC_pairs=1 then in addition to R0, you
        should define R1 and C1. If num_RC_pairs=2 then you should also give
        values for R2 and C2, etc. For the special case where num_RC_pairs=0,
        you should not provide any resistance or capacitance values besides
        the series resistance R0, which is always required.

        """

        if isinstance(params, dict):
            params = params.copy()
        elif isinstance(params, str):
            params = _yaml_reader(params)
        else:
            raise TypeError("'params' must be type dict or str.")

        self._repr_keys = [
            'num_RC_pairs',
            'soc0',
            'capacity',
            'mass',
            'isothermal',
            'Cp',
            'T_inf',
            'h_therm',
            'A_therm',
        ]

        self.num_RC_pairs = params.pop('num_RC_pairs')
        self.soc0 = params.pop('soc0')
        self.capacity = params.pop('capacity')
        self.mass = params.pop('mass')
        self.isothermal = params.pop('isothermal')
        self.Cp = params.pop('Cp')
        self.T_inf = params.pop('T_inf')
        self.h_therm = params.pop('h_therm')
        self.A_therm = params.pop('A_therm')

        self.ocv = params.pop('ocv')
        self.R0 = params.pop('R0')

        for i in range(self.num_RC_pairs):
            setattr(self, 'R' + str(i+1), params.pop('R' + str(i+1)))
            setattr(self, 'C' + str(i+1), params.pop('C' + str(i+1)))

        if len(params) != 0:
            extra_keys = list(params.keys())
            raise ValueError("'params' contains invalid and/or excess"
                             f" key/value pairs: {extra_keys=}")

        self.pre()

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return a readable repr string.

        Returns
        -------
        readable : str
            A console-readable instance representation.

        """

        keys = self._repr_keys
        values = [getattr(self, k) for k in keys]

        summary = "\n\t".join([f"{k}={v}," for k, v in zip(keys, values)])

        readable = f"Model(\n\t{summary}\n)"

        return readable

    def pre(self) -> None:
        """
        Pre-process and prepare the model for running experiments.

        This method builds solution pointers, registers algebraic variable
        indices, stores the mass matrix, and initializes the battery state.

        Returns
        -------
        None.

        Warnings
        --------
        This method runs the first time during the class initialization. It
        generally does not have to be run again unless you modify any model
        attributes. You should manually re-run the pre-processor if you alter
        any properties after initialization. Forgetting to manually re-run the
        pre-processor may cause inconsistencies between the updated properties
        and the model's pointers, state, etc.

        """

        self._t0 = 0.

        ptr = {}
        ptr['soc'] = 0
        ptr['T_cell'] = 1
        ptr['eta_j'] = np.arange(2, 2 + self.num_RC_pairs)
        ptr['V_cell'] = self.num_RC_pairs + 2
        ptr['size'] = self.num_RC_pairs + 3

        algidx = [ptr['V_cell']]

        mass_matrix = np.zeros(ptr['size'])
        mass_matrix[ptr['soc']] = 1.
        mass_matrix[ptr['T_cell']] = self.mass*self.Cp*self.T_inf
        mass_matrix[ptr['eta_j']] = 1.

        sv0 = np.zeros(ptr['size'])
        sv0[ptr['soc']] = self.soc0
        sv0[ptr['T_cell']] = 1.
        sv0[ptr['eta_j']] = 0.
        sv0[ptr['V_cell']] = self.ocv(self.soc0)

        svdot0 = np.zeros_like(sv0)

        self._ptr = ptr
        self._algidx = algidx
        self._mass_matrix = sparse.diags(mass_matrix)
        self._sv0 = sv0
        self._svdot0 = svdot0

    def rhs_funcs(self, t: float, sv: np.ndarray, inputs: dict) -> np.ndarray:
        """
        Right hand side functions.

        Returns the right hand side for the DAE system. For any differential
        variable i, rhs[i] must be equivalent to M[i, i]*y[i]. For algebraic
        variables rhs[i] must be an expression that equals zero.

        Parameters
        ----------
        t : float
            Value of time [s].
        sv : 1D np.array
            State variables at time t.
        inputs : dict
            Dictionary detailing an experimental step.

        Returns
        -------
        rhs : 1D np.array
            The right hand side values of the DAE system.

        """

        rhs = np.zeros(self._ptr['size'])

        # state
        soc = sv[self._ptr['soc']]
        T_cell = sv[self._ptr['T_cell']]*self.T_inf
        eta_j = sv[self._ptr['eta_j']]
        voltage = sv[self._ptr['V_cell']]

        # state-dependent properties
        ocv = self.ocv(soc)
        R0 = self.R0(soc, T_cell)

        # calculated current and power
        current = -(voltage - ocv + np.sum(eta_j)) / R0
        power = current*voltage

        # state of charge (differential)
        rhs[self._ptr['soc']] = -current / 3600. / self.capacity

        # temperature (differential)
        Q_gen = current*(ocv - voltage)
        Q_conv = self.h_therm*self.A_therm*(self.T_inf - T_cell)

        rhs[self._ptr['T_cell']] = (Q_gen + Q_conv) * (1 - self.isothermal)

        # RC overpotentials (differential)
        for j, ptr in enumerate(self._ptr['eta_j'], start=1):
            Rj = getattr(self, 'R' + str(j))(soc, T_cell)
            Cj = getattr(self, 'C' + str(j))(soc, T_cell)
            rhs[ptr] = -sv[ptr] / (Rj*Cj) + current / Cj

        # cell voltage (algebraic)
        if inputs['mode'] == 'current':
            rhs[self._ptr['V_cell']] = current - inputs['value'](t)
        elif inputs['mode'] == 'voltage':
            rhs[self._ptr['V_cell']] = voltage - inputs['value'](t)
        elif inputs['mode'] == 'power':
            rhs[self._ptr['V_cell']] = power - inputs['value'](t)

        # values for rootfns
        total_time = self._t0 + t

        roots = {
            'soc': soc,
            'temperature_K': T_cell,
            'current_A': current,
            'voltage_V': voltage,
            'power_W': power,
            'capacity_Ah': soc*self.capacity,
            'time_s': total_time,
            'time_min': total_time / 60.,
            'time_h': total_time / 3600.,
        }

        inputs['roots'] = roots

        return rhs

    def residuals(self, t: float, sv: np.ndarray, svdot: np.ndarray,
                  inputs: dict) -> np.ndarray:
        """
        Return the DAE residuals.

        The DAE residuals should be near zero at each time step. The solver
        requires the DAE to be written in terms of its residuals in order to
        minimize their values.

        Parameters
        ----------
        t : float
            Value of time [s].
        sv : 1D np.array
            State variables at time t.
        svdot : 1D np.array
            State variable time derivatives at time t.
        inputs : dict
            Dictionary detailing an experimental step.

        Returns
        -------
        res : 1D np.array
            DAE residuals, res = M*ydot - rhs(t, y).

        """
        return self._mass_matrix.dot(svdot) - self.rhs_funcs(t, sv, inputs)

    def run_step(self, exp: Experiment, stepidx: int) -> StepSolution:
        """
        Run a single experimental step.

        Parameters
        ----------
        exp : Experiment
            An experiment instance.
        stepidx : int
            Step index to run. The first step has index 0.

        Returns
        -------
        sol : StepSolution
            Solution to the experiment step.

        Warning
        -------
        The model's internal state is changed at the end of each experiment
        step. Consequently, you should not run steps out of order. You should
        always start with ``stepidx = 0`` and then progress to the subsequent
        steps afterward. After the last step, you should manually run the
        preprocessor ``pre()`` to reset the model before running additional
        experiments.

        See also
        --------
        Experiment : Build an experiment.
        StepSolution : Wrapper for a single-step solution.

        Notes
        -----
        Using the ``run()`` method will automatically run all steps in an
        experiment and will stitch the solutions together for you. You should
        only run step by step if you trying to fine tune solver options, or
        if you have a complex protocol and you can't set an experimental step
        until interpreting a previous step.

        """

        from ._ida_solver import IDASolver
        from ._solutions import StepSolution

        step = exp.steps[stepidx].copy()
        kwargs = exp._kwargs[stepidx].copy()

        if not callable(step['value']):
            value = step['value']
            step['value'] = lambda t: value

        kwargs['inputs'] = step
        kwargs['algidx'] = self._algidx

        if step['limit'] is not None:
            _setup_rootfn(step['limit'], kwargs)

        solver = IDASolver(self._residuals, **kwargs)

        start = time.time()
        idasol = solver.solve(step['tspan'], self._sv0, self._svdot0)
        timer = time.time() - start

        sol = StepSolution(self, idasol, timer)

        self._t0 = sol.t[-1]
        self._sv0 = sol.y[-1]
        self._svdot0 = sol.ydot[-1]

        return sol

    def run(self, exp: Experiment) -> CycleSolution:
        """
        Run an experiment.

        Parameters
        ----------
        exp : Experiment
            An experiment instance.

        Returns
        -------
        sol : CycleSolution
            A stitched solution will all experimental steps.

        See also
        --------
        Experiment : Build an experiment.
        CycleSolution : Wrapper for an all-steps solution.

        """

        from ._solutions import CycleSolution

        sv0 = self._sv0.copy()
        svdot0 = self._svdot0.copy()

        sols = []
        for i in range(exp.num_steps):
            sols.append(self.run_step(exp, i))

        sol = CycleSolution(*sols)

        self._t0 = 0.
        self._sv0 = sv0
        self._svdot0 = svdot0

        return sol

    def _residuals(self, t: float, sv: np.ndarray, svdot: np.ndarray,
                   res: np.ndarray, inputs: dict) -> None:
        """
        Solver-structured residuals.

        The IDASolver requires a residuals function in this exact form.
        Rather than outputting the residuals, the function returns None,
        but fills the 'res' input array with the DAE residuals.

        Parameters
        ----------
        t : float
            Value of time [s].
        sv : 1D np.array
            State variables at time t.
        svdot : 1D np.array
            State variable time derivatives at time t.
        res : 1D np.array
            DAE residuals, res = M*ydot - rhs(t, y).
        inputs : dict
            Dictionary detailing an experimental step.

        Returns
        -------
        None.

        """

        res[:] = self.residuals(t, sv, svdot, inputs)


class _RootFunction:
    """Root function callables."""

    __slots__ = ('key', 'value',)

    def __init__(self, key: str, value: float) -> None:
        """
        This class is a generalize root function callable. All possible root
        functions only have one event that can be triggered. Therefore, the
        'size' property is always unity, and is passed to the IDASolver as
        the 'nr_rootfns' argument.

        Parameters
        ----------
        key : str
            A valid limit type/units from an experiment step.
        value : float
            The value of the limit that triggers the solver to stop.

        """

        self.key = key
        self.value = value

    def __call__(self, t: float, sv: np.ndarray, svdot: np.ndarray,
                 events: np.ndarray, inputs: dict) -> None:
        """
        Solver-structured root function.

        The IDASolver requires a root function in this exact form. Rather
        than outputting the events array, the function returns None, but
        fills the 'events' input array with root functions. If any 'events'
        index equals zero, the solver will exit prior to 'tspan[-1]'.

        Parameters
        ----------
        t : float
            Value of time [s].
        sv : 1D np.array
            State variables at time t.
        svdot : 1D np.array
            State variable time derivatives at time t.
        events : 1D np.array
            An array of root/event functions. During integration, the solver
            will exit prior to 'tspan[-1]' if any 'events' index equals zero.
        inputs : dict
            Dictionary detailing an experimental step, with the 'roots' key
            added and filled within the `rhs_funcs()' method.

        Returns
        -------
        None.

        """

        events[0] = self.value - inputs['roots'][self.key]

    @property
    def size(self) -> int:
        """
        Size of the 'events' array.

        Returns
        -------
        size : int
            Number of indices available in the 1D 'events' array.

        """
        return 1


def _setup_rootfn(limit: tuple[str, float], kwargs: dict) -> None:
    """
    Set up a root function for the IDASolver.

    The IDASolver requires two keyword arguments to be set when using root
    functions. The first is 'rootfn' which requires a Callable. The second
    is 'nr_rootfns' with allocates memory to an array that stores the root
    function values.

    Parameters
    ----------
    limit : tuple[str, float]
        A tuple from the Experiment class that describes the type of limit,
        units, and value required to define the root function.
    kwargs : dict
        The IDASolver keyword argumnents dictionary. Both the 'rootfn' and
        'nr_rootfns' keyword arguments must be added to 'kwargs'.

    """

    rootfn = _RootFunction(*limit)

    kwargs['rootfn'] = rootfn
    kwargs['nr_rootfns'] = rootfn.size


def _yaml_reader(file: str) -> dict:
    """
    Read a yaml file.

    This yaml reader has a custom ``!eval`` constructor, which allows you to
    evaluate ``np`` expressions and ``lambda`` functions.

    Parameters
    ----------
    file : str
        An absolute or relative path to a '.yaml' file. The extension will
        be added if not included.

    Raises
    ------
    ValueError
        Invalid file. Only supports '.yaml' files.
    FileNotFoundError
        File does not exist.

    Returns
    -------
    data : dict
        Data dictionary corresponding to the input file.

    """

    _, extension = os.path.splitext(file)

    if extension == '':
        file += '.yaml'
    elif extension != '.yaml':
        raise ValueError("Invalid file. Only supports '.yaml' files.")

    here = os.path.dirname(__file__)
    templates = here + '/templates'

    if file in os.listdir(templates):
        warnings.warn(f"Using the default parameter file '{file}'.")
        file = templates + '/' + file

    def eval_constructor(loader, node):
        return eval(node.value)

    if not os.path.exists(file):
        raise FileNotFoundError(f"File '{file}' does not exist.")

    reader = YAML(typ='safe')

    add_constructor('!eval', eval_constructor, constructor=SafeConstructor)

    with open(file, 'r') as f:
        data = reader.load(f)

    return data


def _showwarning(message, category, filename, lineno, file=None, line=None):
    file = file or sys.stderr
    print(f"\n[thevenin {category.__name__}]: {message}\n", file=file)


warnings.showwarning = _showwarning
