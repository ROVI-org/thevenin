thevenin.loadfns
================

.. py:module:: thevenin.loadfns

.. autoapi-nested-parse::

   Load Functions
   --------------
   This module contains classes to help construct time-varying load profiles.
   All of the classes are callable after construction and take in a value of
   time in seconds. Most load functions include a linear ramp that "smooths"
   transitions from rest to a constant load, or between constant steps. Using
   ramps helps the solver maintain stability when a boundary condition sharply
   changes from one value to another, e.g., jumping from rest into a high-rate
   charge or discharge. For example, in some cases the solver may crash for a
   high-rate discharge.



Classes
-------

.. autoapisummary::

   thevenin.loadfns.Ramp
   thevenin.loadfns.Ramp2Constant
   thevenin.loadfns.RampedSteps
   thevenin.loadfns.StepFunction


Package Contents
----------------

.. py:class:: Ramp(m, b = 0.0)

   Linearly ramping load.

   A load profile that continuously ramps with slope m.

   :param m: Slope [units/s].
   :type m: float
   :param b: Y-intercept [units]. The default is 0.
   :type b: float, optional


.. py:class:: Ramp2Constant(m, step, b = 0.0, sharpness = 100.0)

   Ramp to a constant load.

   A load profile that ramps with slope m unil the constant step value
   is reached, after which, the load is equal to the step constant. A
   sigmoid is used to smooth the transition between the two piecewise
   functions. Use a large 'sharpness' to reduce smoothing effects.

   :param m: Slope [units/s].
   :type m: float
   :param step: Constant step value [units].
   :type step: float
   :param b: Y-intercept [units]. The default is 0.
   :type b: float, optional
   :param sharpness: How sharp to make the transition between the ramp and step. Low
                     values will smooth the transition more. The default is 100.
   :type sharpness: float, optional

   :raises ValueError: m = 0. and m = inf are invalid slopes.
   :raises ValueError: Cannot reach step with m > 0. and b >= step.
   :raises ValueError: Cannot reach step with m < 0. and b <= step.
   :raises ValueError: 'sharpness' must be strictly positive.


.. py:class:: RampedSteps(tp, yp, t_ramp, y0 = 0.0)

   Step function with ramps.

   This class acts like StepFunction, with the same tp, yp, and y0, but
   step transitions include ramps with duration t_ramp. Generally, this
   profile will be more stable compared to a StepFunction profile.

   :param tp: Times at which a step change occurs [seconds].
   :type tp: 1D np.array
   :param yp: Constant values for each time interval.
   :type yp: 1D np.array
   :param t_ramp: Ramping time between step transitions [seconds].
   :type t_ramp: float
   :param y0: Value to return when t < tp[0]. In addition to standard float
              values, np.nan and np.inf are supported. The default is 0.
   :type y0: float

   :raises ValueError: tp and yp must both be 1D.
   :raises ValueError: tp and yp must be same size.
   :raises ValueError: t_ramp must be strictly positive.
   :raises ValueError: tp must be strictly increasing.

   .. seealso::

      :obj:`StepFunction`
          Uses hard discontinuous steps rather than ramped steps. Generally non-ideal for simulations, but may be useful elsewhere.


.. py:class:: StepFunction(tp, yp, y0 = 0.0, ignore_nan = False)

   Piecewise step function.

   Construct a piecewise step function given the times at which step
   changes occur and the values for each time interval. For example,

   .. code-block:: python

       tp = np.array([0, 5])
       yp = np.array([-1, 1])

       y = StepFunction(tp, yp, np.nan)

   Corresponds to

   .. code-block:: python

       if t < 0:
           y = np.nan
       elif 0 <= t < 5:
           y = -1
       else:
           y = 1

   :param tp: Times at which a step change occurs [s].
   :type tp: 1D np.array
   :param yp: Constant values for each time interval.
   :type yp: 1D np.array
   :param y0: Value to return when t < tp[0]. In addition to standard float
              values, np.nan and np.inf are supported. The default is 0.
   :type y0: float, optional
   :param ignore_nan: Whether or not to ignore NaN inputs. For NaN inputs, the callable
                      returns NaN when False (default) or yp[-1] when True.
   :type ignore_nan: bool, optional

   :raises ValueError: tp and yp must both be 1D.
   :raises ValueError: tp and yp must be same size.
   :raises ValueError: tp must be strictly increasing.

   .. rubric:: Examples

   >>> tp = np.array([0, 1, 5])
   >>> yp = np.array([-1, 0, 1])
   >>> func = StepFunction(tp, yp, np.nan)
   >>> print(func(np.array([-10, 0.5, 4, 10])))
   [nan  -1.  0.  1.]


