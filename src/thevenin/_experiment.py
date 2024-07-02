from __future__ import annotations
from typing import Callable

import numpy as np


class Experiment:
    """Experiment builder."""

    __slots__ = ('_steps', '_kwargs', '_options',)

    def __init__(self, **kwargs) -> None:
        """
        A class to define an experimental protocol. Use the add_step() method
        to add a series of sequential steps. Each step defines a control mode,
        a constant or time-dependent load profile, a time span, and optional
        limiting criteria to stop the step early if a specified event/state is
        detected.

        Parameters
        ----------
        kwargs : dict, optional
            IDASolver keyword arguments that will span all steps.

        See also
        --------
        ~thevenin.IDASolver :
            The solver class, with documentation for most keyword arguments
            that you might want to adjust.

        """

        self._steps = []
        self._kwargs = []
        self._options = kwargs.copy()

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return a readable repr string.

        Returns
        -------
        readable : str
            A console-readable instance representation.

        """

        keys = ['num_steps', 'options']
        values = [self.num_steps, self._options]

        summary = "\n\t".join(f"{k}={v}," for k, v in zip(keys, values))

        readable = f"Experiment(\n\t{summary}\n)"

        return readable

    @property
    def steps(self) -> list[dict]:
        """
        Return steps list.

        Returns
        -------
        steps : list[dict]
            List of the step dictionaries.

        """
        return self._steps

    @property
    def num_steps(self) -> int:
        """
        Return number of steps.

        Returns
        -------
        num_steps : int
            Number of steps.

        """
        return len(self._steps)

    def print_steps(self) -> None:
        """
        Prints a formatted/readable list of steps.

        Returns
        -------
        None.

        """
        with np.printoptions(threshold=6, edgeitems=2):
            for i, step in enumerate(self.steps):
                print("\nstep:", i, "\n" + "-"*10)
                for key, value in step.items():
                    print(key, ":", f"{value!r}")

    def add_step(self, mode: str, value: float | Callable, tspan: tuple,
                 limit: tuple[str, float] = None, **kwargs) -> None:
        """
        Add a step to the experiment.

        Parameters
        ----------
        mode : str
            Control mode, from {'current_A', 'voltage_V', 'power_W'}.
        value : float | Callable
            Value of boundary contion in the appropriate units.
        tspan : tuple
            Relative times for recording solution [s]. Providing a tuple as
            (t_max: float, Nt: int) or (t_max: float, dt: float) constructs
            tspan using ``np.linspace`` or ``np.arange``, respectively.
        limit : tuple[str, float], optional
            Stopping criteria for the new step. The first index must be one
            of {'soc', 'temperature_K', 'current_A', 'voltage_V', 'power_W',
            'capacity_Ah', 'time_s', 'time_min', 'time_h'}. The second is the
            value of the stopping criteria in the appropriate units. All of
            the time limits represent the total experiment time. The default
            is None.
        **kwargs : dict, optional
            IDASolver keyword arguments specific to the new step only.

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            'mode' is invalid.
        ValueError
            'limit[0]' is invalid.
        ValueError
            'tspan' tuple must be length 2.
        TypeError
            'tspan[1]' must be type int or float.

        See also
        --------
        ~thevenin.IDASolver :
            The solver class, with documentation for most keyword arguments
            that you might want to adjust.

        Notes
        -----
        For time-dependent loads, use a Callable for 'value' with a function
        signature like def load(t: float) -> float, where t is the step's
        relative time, in seconds.

        The solution times array is constructed depending on the 'tspan'
        input types:

        * Given (float, int):
            ``tspan = np.linspace(0., tspan[0], tspan[1])``
        * Given (float, float):
            ``tspan = np.arange(0., tspan[0] + tspan[1], tspan[1])``

        """

        _check_mode(mode)
        _check_limit(limit)

        mode, units = mode.split('_')

        if not len(tspan) == 2:
            raise ValueError("'tspan' tuple must be length 2.")

        if isinstance(tspan[1], int):
            t_max, Nt = tspan
            tspan = np.linspace(0., t_max, Nt)
        elif isinstance(tspan[1], float):
            t_max, dt = tspan
            tspan = np.arange(0., t_max + dt, dt, dtype=float)
        else:
            raise TypeError("'tspan[1]' must be type int or float.")

        step = {}
        step['mode'] = mode
        step['value'] = value
        step['units'] = units
        step['tspan'] = tspan
        step['limit'] = limit

        self._steps.append(step)
        self._kwargs.append({**self._options, **kwargs})


def _check_mode(mode: str) -> None:
    """
    Check the operating mode.

    Parameters
    ----------
    mode : str
        Operating mode and units.

    Returns
    -------
    None.

    Raises
    ------
    ValueError
        'mode' is invalid.

    """

    valid = ['current_A', 'voltage_V', 'power_W']

    if mode not in valid:
        raise ValueError(f"{mode=} is invalid; valid values are {valid}.")


def _check_limit(limit: tuple[str, float]) -> None:
    """
    Check the limit criteria.

    Parameters
    ----------
    limit : tuple[str, float]
        Stopping criteria and limiting value.

    Returns
    -------
    None.

    Raises
    ------
    ValueError
        'limit[0]' is invalid.

    """

    valid = ['soc', 'temperature_K', 'current_A', 'voltage_V', 'power_W',
             'capacity_Ah', 'time_s', 'time_min', 'time_h']

    if limit is None:
        pass
    elif limit[0] not in valid:
        raise ValueError(f"{limit[0]=} is invalid; valid values are {valid}.")
