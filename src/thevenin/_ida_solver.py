from __future__ import annotations
from typing import Callable

import numpy as np
from scikits_odes_sundials import ida


class SolverReturn:
    """Solver return."""

    def __init__(self, solution: ida.SolverReturn) -> None:
        """
        A class to wrap the returned arrays and status of the IDASolver.
        Attributes are intentionally "hidden" and accessed via read-only
        properties to avoid unintentional overwrites.

        Parameters
        ----------
        solution : ida.SolverReturn
            The default SolverReturn class from scikits.odes.

        """

        self._success = solution.flag >= 0
        self._message = solution.message
        self._t = solution.values.t
        self._y = solution.values.y
        self._ydot = solution.values.ydot
        self._roots = solution.roots.t is not None
        self._tstop = solution.tstop.t is not None
        self._errors = solution.errors.t is not None

        labels = []
        times = np.array([])

        if self.roots:
            labels.append('roots')
            times = np.hstack([times, solution.roots.t])
        if self.tstop:
            labels.append('tstop')
            times = np.hstack([times, solution.tstop.t])
        if self.errors:
            labels.append('errors')
            times = np.hstack([times, solution.errors.t])

        if len(labels) != 0:
            sorted_times = sorted(set(times), reverse=True)
            mapping = {t: -i-1 for i, t in enumerate(sorted_times)}
            order = [mapping[t] for t in times]

            sorted_pairs = sorted(zip(order, labels))

            for i, label in sorted_pairs:
                setattr(self, '_' + label, (True, i))

                new_line = getattr(solution, label)
                if self.t[-1] < new_line.t:
                    self._t = np.hstack([self._t, new_line.t])
                    self._y = np.vstack([self._y, new_line.y])
                    self._ydot = np.vstack([self._ydot, new_line.ydot])

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return a readable repr string.

        Returns
        -------
        readable : str
            A console-readable instance representation.

        """

        keys = ['success', 'message', 'roots', 'tstop', 'errors']
        values = [getattr(self, k) for k in keys]

        summary = "\n\t".join([f"{k}={v!r}," for k, v in zip(keys, values)])

        readable = f"SolverReturn(\n\t{summary}\n)"

        return readable

    @property
    def success(self) -> bool:
        """
        Overall solver exit status.

        Returns
        -------
        success : bool
            True if no errors, False otherwise.

        """
        return self._success

    @property
    def message(self) -> str:
        """
        Readable solver exit message.

        Returns
        -------
        message : str
            Exit message from the IDASolver.

        """
        return self._message

    @property
    def t(self) -> np.ndarray:
        """
        Saved solution times.

        Returns
        -------
        t : 1D np.array
            Solution times [s].

        """
        return self._t

    @property
    def y(self) -> np.ndarray:
        """
        Solution variables [units]. Rows correspond to solution times and
        columns to state variables, in the same order as y0.

        Returns
        -------
        y : 2D np.array
            Solution variables [units].

        """
        return self._y

    @property
    def ydot(self) -> np.ndarray:
        """
        Solution variable time derivatives [units/s]. Rows and columns share
        the same organization as y.

        Returns
        -------
        ydot : 2D np.array
            Solution variable time derivatives [units/s].

        """
        return self._ydot

    @property
    def roots(self) -> bool | tuple:
        """
        Details regarding whether or not a rootfn stopped the solver.

        Returns
        -------
        roots : bool | tuple
            If a rootfn stopped the solver, this value will be a tuple. The
            first argument will be True, and the second argument will provide
            the index within t, y, and ydot that stores the values of time
            and solution when the root function was triggered. If a root did
            not stop the solver, this value will be False.

        """
        return self._roots

    @property
    def tstop(self) -> bool | tuple:
        """
        Details regarding whether or not the tstop option stopped the solver.

        Returns
        -------
        tstop : bool | tuple
            If the tstop option stopped the solver, this value is a tuple. The
            first argument will be True, and the second argument will provide
            the index within t, y, and ydot that stores the values of time
            and solution when the tstop function was triggered. If tstop did
            not stop the solver, this value will be False.

        """
        return self._tstop

    @property
    def errors(self) -> bool | tuple:
        """
        Details regarding whether or not an error stopped the solver.

        Returns
        -------
        errors : bool | tuple
            If an error stopped the solver, this value will be a tuple. The
            first argument will be True, and the second argument will provide
            the index within t, y, and ydot that stores the values of time
            and solution when the error was triggered. If an error did not
            stop the solver, this value will be False.

        """
        return self._errors


class IDASolver:
    """
    ODE/DAE solver.

    This solver supports first-order ODEs and DAEs. The solver requires the
    problem to be written in terms of a residual function, with a signature
    ``def residuals(t, y, yp, res, inputs) -> None``. Instead of a return
    value, the function fills ``res`` (a 1D array sized like ``y``) with
    expressions from the system of equations, ``res = M(y)*yp - f(t, y)``.
    Here, ``t`` is time, ``y`` is an array of dependent solution variables,
    and ``yp`` are time derivatives of ``y``. The ``inputs`` argument allows
    the user to pass any additional parameters to the residuals function.

    Parameters
    ----------
    residuals : Callable
        Function like ``def residuals(t, y, yp, res, inputs) -> None``.
    **kwargs : dict, optional
        Keywords, descriptions, and defaults given below.

        =========== =================================================
        Key         Description (*type* or {options}, default)
        =========== =================================================
        atol        absolute tolerance (*float*, 1e-6)
        rtol        relative tolerance (*float*, 1e-5)
        inputs      optional residual arguments (*tuple*, None)
        linsolver   linear solver ({'dense', 'band'}, 'dense')
        lband       residual function's lower bandwidth (*int*, 0)
        uband       residual function's upper bandwidth (*int*, 0)
        rootfn      root/event function (*Callable*, None)
        nr_rootfns  number of events in rootfn (*int*, 0)
        initcond    uncertain t0 values ({'y0', 'yp0', None}, 'yp0')
        algidx      algebraic variable indices (*list[int]*, None)
        max_dt      maximum allowable integration step (*float*, 0.)
        tstop       maximum integration time (*float*, None)
        =========== =================================================

    Notes
    -----
    * IDA stands for Implicit Differential Algebraic solver. The solver is
      accessed through `scikits-odes`_, a Python wrapper for `SUNDIALS`_.
    * Not setting ``algidx`` for DAEs will likely result in an instability.
    * For unrestricted integration steps, use ``max_dt = 0.``.
    * Root functions require a signature like ``def rootfn(t, y, yp, events,
      inputs) -> None``. Instead of a return value, the function fills the
      ``events`` argument (a 1D array with size equal to the number of events
      to track). If any ``events`` index equals zero during integration, the
      solver will exit.
    * If setting ``rootfn``, you also need to set ``nr_rootfns`` to allocate
      memory for the correct number of expressions (i.e., ``events.size``).

    .. _SUNDIALS: https://sundials.readthedocs.io/
    .. _scikits-odes: https://bmcage.github.io/odes/dev/

    Examples
    --------
    The following demonstrates solving a system of ODEs. For ODEs, derivative
    expressions ``yp`` can be written for each ``y``. Therefore, we can write
    each residual as ``res[i] = yp[i] - f(t, y)`` where ``f(t, y)`` is an
    expression for the derivative in terms of ``t`` and ``y``.

    Note that even though the solver requires knowing the initial derivatives,
    we set ``yp0 = np.zeros_like(y0)``, which are not true ``yp0`` values. The
    default option ``initcond='yp0'`` solves for the correct ``yp0`` values
    before starting the integration.

    .. code-block:: python

        import thevenin
        import numpy as np
        import matplotlib.pyplot as plt

        def residuals(t, y, yp, res):
            res[0] = yp[0] - y[1]
            res[1] = yp[1] - 1e3*(1. - y[0]**2)*y[1] + y[0]

        solver = thevenin.IDASolver(residuals)

        y0 = np.array([0.5, 0.5])
        yp0 = np.zeros_like(y0)
        tspan = np.linspace(0., 500., 200)

        solution = solver.solve(tspan, y0, yp0)

        plt.plot(solution.t, solution.y)
        plt.show()

    The next problem solves a DAE system. DAEs arise when systems of governing
    equations contain both ODEs and algebraic constraints.

    To solve a DAE, you should specify the ``y`` indices that store algebraic
    variables. In other words, for which ``y`` can you not write a ``yp``
    expression? In the example below, we have ``yp[0]`` and ``yp[1]`` filling
    the first two residual expressions. However, ``yp[2]`` does not appear in
    any of the residuals. Therefore, ``y[2]`` is an algebraic variable, and we
    tell this to the solver using the keyword argument ``algidx=[2]``. Even
    though we only have one algebraic variable, this option input must be a
    list of integers.

    As in the ODE example, we let the solver determine the ``yp0`` values
    that provide a consistent initial condition. Prior to plotting, ``y[1]``
    is scaled for visual purposes. You can see the same example provided by
    `MATLAB`_.

    .. code-block:: python

        import thevenin
        import numpy as np
        import matplotlib.pyplot as plt

        def residuals(t, y, yp, res):
            res[0] = yp[0] + 0.04*y[0] - 1e4*y[1]*y[2]
            res[1] = yp[1] - 0.04*y[0] + 1e4*y[1]*y[2] + 3e7*y[1]**2
            res[2] = y[0] + y[1] + y[2] - 1.

        solver = thevenin.IDASolver(residuals, algidx=[2])

        y0 = np.array([1., 0., 0.])
        yp0 = np.zeros_like(y0)
        tspan = np.hstack([0., 4.*np.logspace(-6, 6)])

        solution = solver.solve(tspan, y0, yp0)

        solution.y[:, 1] *= 1e4

        plt.semilogx(solution.t, solution.y)
        plt.show()

    .. _MATLAB:
        https://mathworks.com/help/matlab/math/
        solve-differential-algebraic-equations-daes.html

    """

    __slots__ = ('_integrator', '_kwargs',)

    def __init__(self, residuals: Callable, **kwargs) -> None:

        # Default kwargs
        kwargs.setdefault('atol', 1e-6)
        kwargs.setdefault('rtol', 1e-5)
        kwargs.setdefault('inputs', None)
        kwargs.setdefault('linsolver', 'dense')
        kwargs.setdefault('lband', 0)
        kwargs.setdefault('uband', 0)
        kwargs.setdefault('rootfn', None)
        kwargs.setdefault('nr_rootfns', 0)
        kwargs.setdefault('initcond', 'yp0')
        kwargs.setdefault('algidx', None)
        kwargs.setdefault('max_dt', 0.)
        kwargs.setdefault('tstop', None)

        self._kwargs = kwargs.copy()

        # Map renamed scikits.odes options, and force new api
        options = {
            'user_data': kwargs.pop('inputs'),
            'compute_initcond': kwargs.pop('initcond'),
            'algebraic_vars_idx': kwargs.pop('algidx'),
            'max_step_size': kwargs.pop('max_dt'),
            'old_api': False,
        }

        options = {**options, **kwargs}

        self._integrator = ida.IDA(residuals, **options)

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return a readable repr string.

        Returns
        -------
        readable : str
            A console-readable instance representation.

        """

        items = list(self._kwargs.items())

        summary = "\n\t".join([f"{k}={v!r}," for k, v in items])

        readable = f"IDASolver(\n\t{summary}\n)"

        return readable

    def init_step(self, t0: float, y0: np.ndarray,
                  ydot0: np.ndarray) -> SolverReturn:
        """
        Solve for a consistent initial condition.

        Parameters
        ----------
        t0 : float
            Initial time [s].
        y0 : 1D np.array
            State variables at t0.
        yp0 : 1D np.array
            State variable time derivatives at t0.

        Returns
        -------
        solution : SolverReturn
            Solution at time t0.

        """

        solution = self._integrator.init_step(t0, y0, ydot0)
        return SolverReturn(solution)

    def step(self, t: float) -> SolverReturn:
        """
        Solve for a successive time step.

        Before calling step() for the first time, call init_step() to
        initialize the solver at 't0'.

        Parameters
        ----------
        t : float
            Solution step time [s]. Can be higher or lower than the previous
            time, however, significantly lower values may return errors.

        Returns
        -------
        solution : SolverReturn
            Solution at time t.

        """

        solution = self._integrator.step(t)
        return SolverReturn(solution)

    def solve(self, tspan: np.ndarray, y0: np.ndarray,
              ydot0: np.ndarray) -> SolverReturn:
        """
        Solve the system over 'tspan'.

        Parameters
        ----------
        tspan : 1D np.array
            Times [s] to store the solution.
        y0 : 1D np.array
            State variables at tspan[0].
        yp0 : 1D np.array
            State variable time derivatives at tspan[0].

        Returns
        -------
        solution : SolverReturn
            Solution at times in tspan.

        """

        solution = self._integrator.solve(tspan, y0, ydot0)
        return SolverReturn(solution)
