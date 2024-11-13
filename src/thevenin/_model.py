from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING

import os
import re
import time
import warnings
from copy import deepcopy

import numpy as np
import scipy.sparse as sparse
from ruamel.yaml import YAML, add_constructor, SafeConstructor

if TYPE_CHECKING:  # pragma: no cover
    from ._experiment import Experiment
    from ._solutions import BaseSolution, StepSolution, CycleSolution

    Solution = TypeVar('Solution', bound='BaseSolution')


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

            ============= ========================== ================
            Key           Value                      *type*, units
            ============= ========================== ================
            num_RC_pairs  number of RC pairs         *int*, -
            soc0          initial state of charge    *float*, -
            capacity      maximum battery capacity   *float*, Ah
            mass          total battery mass         *float*, kg
            isothermal    flag for isothermal model  *bool*, -
            Cp            specific heat capacity     *float*, J/kg/K
            T_inf         room/air temperature       *float*, K
            h_therm       convective coefficient     *float*, W/m2/K
            A_therm       heat loss area             *float*, m2
            ocv           open circuit voltage       *callable*, V
            R0            series resistance          *callable*, Ohm
            Rj            resistance in RCj          *callable*, Ohm
            Cj            capacity in RCj            *callable*, F
            ============= ========================== ================

        Raises
        ------
        TypeError
            'params' must be type dict or str.
        ValueError
            'params' contains invalid and/or excess key/value pairs.

        Warning
        -------
        A pre-processor runs at the end of the model initialization. If you
        modify any parameters after class instantiation, you will need to
        re-run the pre-processor (i.e., the ``pre()`` method) afterward.

        Notes
        -----
        The 'ocv' property needs a signature like ``f(soc: float) -> float``,
        where 'soc' is the time-dependent state of charged solved for within
        the model. All R0, Rj, and Cj properties should have signatures like
        ``f(soc: float, T_cell: float) -> float``, where 'T_cell' is the cell
        temperature in K determined in the model.

        Rj and Cj are not real property names. These are used generally in the
        documentation. If ``num_RC_pairs=1`` then in addition to R0, you should
        define R1 and C1. If ``num_RC_pairs=2`` then you should also give R2
        and C2, etc. For the special case where ``num_RC_pairs=0``, you should
        not provide any resistance or capacitance values besides the series
        resistance R0, which is always required.

        """

        if isinstance(params, dict):
            params = params.copy()
        elif isinstance(params, str):
            params = _yaml_reader(params)

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

        for j in range(1, self.num_RC_pairs + 1):
            setattr(self, 'R' + str(j), params.pop('R' + str(j)))
            setattr(self, 'C' + str(j), params.pop('C' + str(j)))

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

        summary = "\n    ".join([f"{k}={v}," for k, v in zip(keys, values)])

        readable = f"Model(\n    {summary}\n)"

        return readable

    def pre(self, initial_state: bool | Solution = True) -> None:
        """
        Pre-process and prepare the model for running experiments.

        This method builds solution pointers, registers algebraic variable
        indices, stores the mass matrix, and initializes the battery state.

        Parameters
        ----------
        initial_state : bool | Solution
            Controls how the model state is initialized. If boolean it will set
            the state to a rested state at 'soc0' when True (default) or will
            bypass the state initialization update when False. Given a Solution
            object, the internal state will be set to the final state of the
            solution. See notes for more information.

        Returns
        -------
        None.

        Warning
        -------
        This method runs the first time during the class initialization. It
        generally does not have to be run again unless you modify any model
        attributes. You should manually re-run the pre-processor if you change
        any properties after initialization. Forgetting to manually re-run the
        pre-processor may cause inconsistencies between the updated properties
        and the model's pointers, state, etc.

        Notes
        -----
        Using ``initial_state=False`` will raise an error if you are changing
        the size of your circuit (e.g., changing from one to two RC pairs).
        The same logic applies when initializing based on a Solution instance.
        In other words, a 1RC-pair model cannot be initialized given a solution
        from a 2RC-pair circuit.

        """

        from ._solutions import BaseSolution

        missing_attrs = []
        for j in range(1, self.num_RC_pairs + 1):
            if not hasattr(self, 'R' + str(j)):
                missing_attrs.append('R' + str(j))
            if not hasattr(self, 'C' + str(j)):
                missing_attrs.append('C' + str(j))

        if missing_attrs:
            raise AttributeError(f"Model is missing attrs {missing_attrs} to"
                                 " be consistent with 'num_RC_pairs'.")

        extra_attrs = []
        pattern = re.compile(r"^[RC](\d+)")
        for attr in list(self.__dict__.keys()):

            matches = pattern.match(attr)
            if matches is None:
                pass
            elif int(matches.group(1)) > self.num_RC_pairs:
                extra_attrs.append(attr)

        if extra_attrs:
            short_warn(f"Extra RC attributes {extra_attrs} are present, beyond"
                       " what was expected based on 'num_RC_pairs'.")

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

        self._t0 = 0.
        if isinstance(initial_state, BaseSolution):
            soln = deepcopy(initial_state)
            if soln.y[-1].size != sv0.size:
                raise ValueError("Cannot initialize state based on Solution"
                                 " object given in 'initial_state'. The model"
                                 " and solution have incompatible sizes.")

            self._sv0 = soln.y[-1].copy()
            self._svdot0 = soln.yp[-1].copy()

        elif not initial_state:
            if self._sv0.size != sv0.size:
                raise ValueError("The pre-processor failed. The model state is"
                                 " changing sizes but 'initial_state=False'.")

        elif initial_state:
            self._sv0 = sv0
            self._svdot0 = svdot0

    def rhs_funcs(self, t: float, sv: np.ndarray, inputs: dict) -> np.ndarray:
        """
        Right hand side functions.

        Returns the right hand side for the DAE system. For any differential
        variable i, rhs[i] must be equivalent to M[i, i]*y[i] where M is the
        mass matrix and y is an array of states. For algebraic variables rhs[i]
        must be an expression that equals zero.

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
        mode = inputs['mode']
        units = inputs['units']
        value = inputs['value']

        if mode == 'current' and units == 'A':
            rhs[self._ptr['V_cell']] = current - value(t)
        elif mode == 'current' and units == 'C':
            rhs[self._ptr['V_cell']] = current - self.capacity*value(t)
        elif mode == 'voltage':
            rhs[self._ptr['V_cell']] = voltage - value(t)
        elif mode == 'power':
            rhs[self._ptr['V_cell']] = power - value(t)

        # values for rootfns
        total_time = self._t0 + t

        roots = {
            'soc': soc,
            'temperature_K': T_cell,
            'current_A': current,
            'current_C': current / self.capacity,
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
            DAE residuals, res = M*yp - rhs(t, y).

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
        :class:`~thevenin.StepSolution`
            Solution to the experiment step.

        Warning
        -------
        The model's internal state is changed at the end of each experiment
        step. Consequently, you should not run steps out of order. You should
        always start with ``stepidx = 0`` and then progress to the subsequent
        steps afterward. Run ``pre()`` after your last step to reset the state
        back to a rested state at 'soc0'. Otherwise the internal state will
        match the final state from the last step that was run.

        See also
        --------
        Experiment : Build an experiment.
        StepSolution : Wrapper for a single-step solution.

        Notes
        -----
        Using the ``run()`` method will automatically run all steps in an
        experiment and will stitch the solutions together for you. You should
        only run step by step if you are trying to fine tune solver options, or
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

        kwargs['userdata'] = step
        kwargs['calc_initcond'] = 'yp0'
        kwargs['algebraic_idx'] = self._algidx

        if step['limits'] is not None:
            _setup_rootfn(step['limits'], kwargs)

        solver = IDASolver(self._residuals, **kwargs)

        start = time.time()
        ida_soln = solver.solve(step['tspan'], self._sv0, self._svdot0)
        timer = time.time() - start

        soln = StepSolution(self, ida_soln, timer)

        self._t0 = soln.t[-1]
        self._sv0 = soln.y[-1].copy()
        self._svdot0 = soln.yp[-1].copy()

        return soln

    def run(self, exp: Experiment, reset_state: bool = True,
            t_shift: float = 1e-3) -> CycleSolution:
        """
        Run an experiment.

        Parameters
        ----------
        exp : Experiment
            An experiment instance.
        reset_state : bool
            If True (default), the internal state of the model will be reset
            back to a rested condition at 'soc0' at the end of the experiment.
            When False the state does not reset instead matches the final state
            of the last experimental step.
        t_shift : float
            Time (in seconds) to shift step solutions by when stitching them
            together. If zero the end time of each step overlaps the starting
            time of its following step. The default is 1e-3.

        Returns
        -------
        :class:`~thevenin.CycleSolution`
            A stitched solution will all experimental steps.

        Warning
        -------
        The default behavior resets the model's internal state back to a rested
        condition at 'soc0' by calling the ``pre()`` method. You can bypass this
        by using ``reset_state=False``, which will keep the state at the end of
        the final experimental step.

        See also
        --------
        Experiment : Build an experiment.
        CycleSolution : Wrapper for an all-steps solution.

        """

        from ._solutions import CycleSolution

        solns = []
        for i in range(exp.num_steps):
            solns.append(self.run_step(exp, i))

        soln = CycleSolution(*solns, t_shift=t_shift)

        self._t0 = 0.
        if reset_state:
            self.pre()

        return soln

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
            DAE residuals, res = M*yp - rhs(t, y).
        inputs : dict
            Dictionary detailing an experimental step.

        Returns
        -------
        None.

        """

        res[:] = self.residuals(t, sv, svdot, inputs)


class _RootFunction:
    """Root function callables."""

    def __init__(self, limits: tuple[str, float]) -> None:
        """
        This class is a generalize root function callable. All possible root
        functions only have one event that can be triggered. Therefore, the
        'size' property is always unity, and is passed to the IDASolver as
        the 'nr_rootfns' argument.

        Parameters
        ----------
        limits : tuple[str, float]
            A tuple of root function criteria arranged as ordered pairs of
            limit names and values, e.g., ('time_h', 10., 'voltage_V', 4.2).

        """

        self.keys = limits[0::2]
        self.values = limits[1::2]
        self.size = len(self.keys)

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

        for i, (key, value) in enumerate(zip(self.keys, self.values)):
            events[i] = inputs['roots'][key] - value


def _setup_rootfn(limits: tuple[str, float], kwargs: dict) -> None:
    """
    Set up a root function for the IDASolver.

    The IDASolver requires two keyword arguments to be set when using root
    functions. The first is 'rootfn' which requires a Callable. The second
    is 'nr_rootfns' with allocates memory to an array that stores the root
    function values.

    Parameters
    ----------
    limits : tuple[str, float]
        A tuple of root function criteria arranged as ordered pairs of limit
        names and values, e.g., ('time_h', 10., 'voltage_V', 4.2).
    kwargs : dict
        The IDASolver keyword argumnents dictionary. Both the 'rootfn' and
        'nr_rootfns' keyword arguments must be added to 'kwargs'.

    Returns
    -------
    None.

    """

    rootfn = _RootFunction(limits)

    kwargs['eventsfn'] = rootfn
    kwargs['num_events'] = rootfn.size


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
        short_warn(f"Using the default parameter file '{file}'.")
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


def formatwarning(message, category, filename, lineno, line=None):
    """Shortened warning format - used for parameter/pre warnings."""
    return f"\n[thevenin {category.__name__}] {message}\n\n"


def short_warn(message, category=None, stacklevel=1, source=None):
    """Print a warning with the short format from ``formatwarning``."""
    original_format = warnings.formatwarning

    warnings.formatwarning = formatwarning
    warnings.warn(message, category, stacklevel, source)

    warnings.formatwarning = original_format
