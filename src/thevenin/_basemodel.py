from __future__ import annotations

import re
import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
from ruamel.yaml import YAML, add_constructor, SafeConstructor


def calculated_current(voltage, ocv, hyst, eta_j, R0) -> float:
    if eta_j.ndim == 1:
        return -(voltage - ocv - hyst + np.sum(eta_j)) / R0
    elif eta_j.ndim == 2:
        return -(voltage - ocv - hyst + np.sum(eta_j, axis=1)) / R0
    else:
        raise ValueError("Dimension error in calculating voltage.")


def calculated_voltage(current, ocv, hyst, eta_j, R0) -> float:
    if eta_j.ndim == 1:
        return ocv + hyst - np.sum(eta_j) - current*R0
    elif eta_j.ndim == 2:
        return ocv + hyst - np.sum(eta_j, axis=1) - current*R0
    else:
        raise ValueError("Dimension error in calculating current.")


class BaseModel(ABC):
    """Abstract BaseModel for Simulation and Prediction."""

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
            keys/value pair descriptions are given below. The default uses an
            internal yaml file.

            ============= ========================== ================
            Key           Value                      *type*, units
            ============= ========================== ================
            num_RC_pairs  number of RC pairs         *int*, -
            soc0          initial state of charge    *float*, -
            capacity      maximum battery capacity   *float*, Ah
            ce            coulombic efficiency       *float*, -
            gamma         hysteresis approach rate   *float*, -
            mass          total battery mass         *float*, kg
            isothermal    flag for isothermal model  *bool*, -
            Cp            specific heat capacity     *float*, J/kg/K
            T_inf         room/air temperature       *float*, K
            h_therm       convective coefficient     *float*, W/m2/K
            A_therm       heat loss area             *float*, m2
            ocv           open circuit voltage       *callable*, V
            M_hyst        max hysteresis magnitude   *callable*, V
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
        The 'ocv' and 'M_hyst' properties need to be callables with signatures
        like ``f(soc: float) -> float``, where 'soc' is the state of charge.
        All other properties that require callables (e.g., R0, Rj, and Cj) need
        signatures like ``f(soc: float, T_cell: float) -> float``, with 'T_cell'
        being the cell temperature in K.

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
            'ce',
            'gamma',
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
        self.ce = params.pop('ce')
        self.gamma = params.pop('gamma')
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

        self.M_hyst = params.pop('M_hyst')

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
        readable = f"{self.classname}(\n    {summary}\n)"

        return readable

    @property
    def classname(self) -> str:
        """Return the name of the class."""
        return self.__class__.__name__

    @property
    def _classname(self) -> str:
        """Use until 'Model' is deprecated to control flags in _rhsfn."""
        if self.classname == 'Model':
            return 'Simulation'

        return self.classname

    @abstractmethod
    def pre(self, *args, **kwargs) -> None:
        """Preprocessor: ensure model setup is correct and ready to run."""
        pass

    def _check_RC_pairs(self) -> None:
        """Verify correct attributes are set based on num_RC_pairs."""

        missing_attrs = []
        for j in range(1, self.num_RC_pairs + 1):
            if not hasattr(self, 'R' + str(j)):
                missing_attrs.append('R' + str(j))
            if not hasattr(self, 'C' + str(j)):
                missing_attrs.append('C' + str(j))

        classname = self.classname
        if missing_attrs:
            raise AttributeError(f"'{classname}' missing attrs {missing_attrs}"
                                 " to be consistent with 'num_RC_pairs'.")

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

    def _rhsfn(self, t: float, sv: np.ndarray, userdata: dict) -> np.ndarray:
        """
        Right-hand-side functions.

        Given a DAE of the form ``M*yp = f(t, y)``, this function represents
        the right-hand-side ``f(t, y)``. For the 'Simulation' class, M has a
        single zero-element diagonal term for the algebraic expression used to
        solve for the cell voltage, making the system a DAE. When the system
        of equations is accessed via the 'Prediction' class, only the pure ODE
        is returned, allowing for faster solves via CVODE.

        Parameters
        ----------
        t : float
            Value of time [s].
        sv : 1D np.array
            State variables at time t.
        userdata : dict
            Dictionary detailing an experimental step.

        Returns
        -------
        rhs : 1D np.array
            The right-hand-side array: equivalent to yp for the 'Prediction'
            class, but contains an extra algebraic expression for 'Simulation'.

        """

        ptr = self._ptr
        rhs = np.zeros(ptr['size'])

        # state
        soc = sv[ptr['soc']]
        T_cell = sv[ptr['T_cell']]*self.T_inf
        hyst = sv[ptr['hyst']]
        eta_j = sv[ptr['eta_j']]

        # state-dependent properties
        ocv = self.ocv(soc)
        R0 = self.R0(soc, T_cell)

        # dependent parameters
        Q_inv = 1. / (3600. * self.capacity)
        alpha_inv = 1. / (self.mass * self.Cp * self.T_inf)

        # current, voltage, and power - different for Simulation/Prediction
        if self._classname == 'Simulation':
            voltage = sv[ptr['V_cell']]
            current = calculated_current(voltage, ocv, hyst, eta_j, R0)

        elif self._classname == 'Prediction':
            current = userdata['current'](t)
            voltage = calculated_voltage(current, ocv, hyst, eta_j, R0)

        power = current*voltage

        # state of charge (differential)
        ce = 1. if current >= 0. else self.ce
        rhs[ptr['soc']] = -ce*current*Q_inv

        # temperature (differential)
        Q_gen = current*(ocv - voltage)
        Q_conv = self.h_therm*self.A_therm*(self.T_inf - T_cell)

        rhs[ptr['T_cell']] = alpha_inv * (Q_gen + Q_conv) \
            * (1 - self.isothermal)

        # hysteresis (differential)
        direction = -np.sign(current)
        coeff = np.abs(ce*current*self.gamma*Q_inv)
        rhs[ptr['hyst']] = coeff*(direction*self.M_hyst(soc) - hyst)

        # RC overpotentials (differential)
        for j, pj in enumerate(ptr['eta_j'], start=1):
            Rj = getattr(self, 'R' + str(j))(soc, T_cell)
            Cj = getattr(self, 'C' + str(j))(soc, T_cell)
            rhs[pj] = -sv[pj] / (Rj*Cj) + current / Cj

        # cell voltage (algebraic) - only if using Simulation, not Prediction
        if self._classname == 'Simulation':
            mode = userdata['mode']
            units = userdata['units']
            value = userdata['value']

            if mode == 'current' and units == 'A':
                rhs[ptr['V_cell']] = current - value(t)
            elif mode == 'current' and units == 'C':
                rhs[ptr['V_cell']] = current - self.capacity*value(t)
            elif mode == 'voltage':
                rhs[ptr['V_cell']] = voltage - value(t)
            elif mode == 'power':
                rhs[ptr['V_cell']] = power - value(t)

            # values for eventsfn
            total_time = self._t0 + t

            events = {
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

            userdata['events'] = events

        return rhs


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


def short_warn(message, category=UserWarning, stacklevel=1, source=None):
    """Print a warning with the short format from ``formatwarning``."""
    original_format = warnings.formatwarning

    warnings.formatwarning = formatwarning
    warnings.warn_explicit(message, category, filename='None', lineno=0)

    warnings.formatwarning = original_format
