from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


def calculated_current(voltage, ocv, hyst, eta_j, R0) -> float:
    if eta_j.ndim == 1:
        return -(voltage - ocv - hyst + np.sum(eta_j)) / R0
    elif eta_j.ndim == 2:
        return -(voltage - ocv - hyst + np.sum(eta_j, axis=1)) / R0
    else:
        ValueError("Dimension error in calculating voltage.")


def calculated_voltage(current, ocv, hyst, eta_j, R0) -> float:
    if eta_j.ndim == 1:
        return ocv + hyst - np.sum(eta_j) - current*R0
    elif eta_j.ndim == 1:
        return ocv + hyst - np.sum(eta_j, axis=1) - current*R0
    else:
        ValueError("Dimension error in calculating current.")


class BaseModel(ABC):

    def __init__(self, params: dict | str = 'params.yaml') -> None:
        pass

    def _rhsfn(self, t, sv, userdata) -> np.ndarray:

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
        if self.classname == 'Simulation':
            voltage = sv[ptr['V_cell']]
            current = calculated_current(voltage, ocv, hyst, eta_j, R0)

        elif self.classname == 'Prediction':
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
        if self.classname == 'Simulation':
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

    @property
    @abstractmethod
    def classname(self) -> str:
        pass

    @abstractmethod
    def pre(self) -> None:
        pass
