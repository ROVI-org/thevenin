from __future__ import annotations

import numpy as np


class StepFunction:
    """Piecewise step function."""

    __slots__ = ('_tp', '_yp', '_func',)

    def __init__(self, tp: np.ndarray, yp: np.ndarray, y0: float = 0.) -> None:
        """
        Construct a piecewise step function given the times at which step
        changes occur and the values for each time interval. For example,

        .. code-block:: python

            tp = np.array([0, 5])
            yp = np.array([-1, 1])

            y = HardSteps(tp, yp, np.nan)

        Corresponds to

        .. code-block:: python

            if t < 0:
                y = np.nan
            elif 0 <= t < 5:
                y = -1
            else:
                y = 1

        Parameters
        ----------
        tp : 1D np.array
            Times at which a step change occurs [s].
        yp : 1D np.array
            Constant values for each time interval.
        y0 : float
            Value to return when t < tp[0]. In addition to standard float
            values, np.nan and np.inf are supported. The default is 0.

        Raises
        ------
        ValueError
            tp and tp must both be 1D.
        ValueError
            tp and yp must be same size.
        TypeError
            y0 must be type float.
        ValueError
            tp must be strictly increasing.

        Examples
        --------
        >>> tp = np.array([0, 1, 5])
        >>> yp = np.array([-1, 0, 1])
        >>> func = HardSteps(tp, yp, np.nan)
        >>> print(func(np.array([-10, 0.5, 4, 10])))
        [nan  -1.  0.  1.]

        """

        if tp.ndim != 1 or yp.ndim != 1:
            raise ValueError("tp and yp must both be 1D.")

        if tp.size != yp.size:
            raise ValueError("tp and yp must be same size.")

        if not isinstance(y0, (int, float)):
            raise TypeError("y0 must be type float.")

        if any(np.diff(tp) <= 0.):
            raise ValueError("tp must be strictly increasing.")

        self._tp = tp
        self._yp = yp

        class _step:

            __slots__ = ('_y',)

            def __init__(self, y: float) -> None:
                self._y = y

            def __call__(self, t: float) -> float:
                return self._y

        funcs = [_step(np.nan), _step(y0)]
        for yp in self._yp:
            funcs.append(_step(yp))

        self._func = lambda t, conds: np.piecewise(t, conds, funcs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"StepFunction(num_steps={self._tp.size})"

    def __call__(self, t: np.ndarray) -> np.ndarray:

        if isinstance(t, np.ndarray):
            t = t.astype(float)

        conds = [np.isnan(t), (t < self._tp[0])]
        for i, tp in enumerate(self._tp[:-1]):
            conds.append((t >= tp) & (t < self._tp[i + 1]))

        conds.append(t >= self._tp[-1])

        return self._func(t, conds)


class RampedSteps:
    """Step function with ramps."""

    __slots__ = ('_tp', '_yp', '_func', '_t_ramp',)

    def __init__(self, tp: np.ndarray, yp: np.ndarray, t_ramp: float,
                 y0: float = 0.) -> None:
        """
        This class constructs a StepFunction with the same tp, yp, and y0,
        but "smooths" the transitions between steps using the t_ramp value
        Generally, this profile will be more stable to run compared to the
        hard steps in StepFunction.

        Parameters
        ----------
        tp : 1D np.array
            Times at which a step change occurs [seconds].
        yp : 1D np.array
            Constant values for each time interval.
        t_ramp : float
            Ramping time between step transitions [seconds].

        See also
        --------
        StepFunction : 
            Uses hard discontinuous steps rather than ramped steps. Generally
            non-ideal for simulations, but may be useful elsewhere.

        """

        steps = StepFunction(tp, yp, y0)

        self._tp = tp
        self._yp = yp
        self._t_ramp = t_ramp

        tp = np.arange(tp.min() - 2.*t_ramp, tp.max() + 2.*t_ramp, t_ramp)
        yp = steps(tp - t_ramp)

        self._func = lambda t: np.interp(t, tp, yp)

    def __repr__(self) -> str:  # pragma: no cover

        num_steps = self._tp.size
        t_ramp = self._t_ramp

        return f"RampedSteps(num_steps={num_steps}, t_ramp={t_ramp:.2e})"

    def __call__(self, t: float) -> float:
        return self._func(t)
