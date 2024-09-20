import pytest
import thevenin
import numpy as np

from sksundae import ida
from thevenin._ida_solver import SolverReturn


@pytest.fixture(scope='function')
def dummy_soln():
    def residuals(t, y, yp, res):
        res[0] = yp[0]

    solver = ida.IDA(residuals)
    dummy_soln = solver.solve(np.linspace(0., 10., 11), [1.], [0.])

    return dummy_soln


@pytest.fixture(scope='function')
def events_soln():
    def eventsfn(t, y, yp, events):
        events[0] = y[0] - 0.5

    def residuals(t, y, yp, res):
        res[0] = yp[0] + 0.1

    solver = ida.IDA(residuals, eventsfn=eventsfn, num_events=1)
    events_soln = solver.solve(np.linspace(0., 10., 11), [1.], [0.])

    return events_soln


def test_solver_return(dummy_soln):
    solution = SolverReturn(dummy_soln)

    assert solution.success
    assert not solution.roots
    assert solution.tstop
    assert not solution.errors

    assert solution.message == dummy_soln.message
    assert np.allclose(solution.t, dummy_soln.t)
    assert np.allclose(solution.y, dummy_soln.y)
    assert np.allclose(solution.ydot, dummy_soln.yp)


def test_solver_return_roots(events_soln):
    solution = SolverReturn(events_soln)

    assert solution.success
    assert solution.roots
    assert np.allclose(solution.y[-1], events_soln.y_events)


def test_ida_solver_with_ode():
    def residuals(t, y, yp, res):
        res[0] = yp[0] - y[1]
        res[1] = yp[1] - 1e3*(1. - y[0]**2)*y[1] + y[0]

    solver = thevenin.IDASolver(residuals)

    y0 = np.array([0.5, 0.5])
    yp0 = np.zeros_like(y0)
    tspan = np.linspace(0., 500., 200)

    solution = solver.solve(tspan, y0, yp0)

    assert solution.success
    assert solution.t[-1] == 500.

    init_soln = solver.init_step(tspan[0], y0, yp0)
    step_soln = solver.step(tspan[1])

    assert np.allclose(init_soln.y, solution.y[0, :])
    assert np.allclose(step_soln.y, solution.y[1, :])


def test_ida_solver_with_dae():

    def residuals(t, y, yp, res):
        res[0] = yp[0] + 0.04*y[0] - 1e4*y[1]*y[2]
        res[1] = yp[1] - 0.04*y[0] + 1e4*y[1]*y[2] + 3e7*y[1]**2
        res[2] = y[0] + y[1] + y[2] - 1.

    solver = thevenin.IDASolver(residuals, max_dt=0.)

    y0 = np.array([1., 0., 0.])
    yp0 = np.zeros_like(y0)
    tspan = np.hstack([0., 4*np.logspace(-6, 6)])

    solution = solver.solve(tspan, y0, yp0)

    solution.y[:, 1] *= 1e4

    assert solution.success
    assert solution.t[-1] == 4e6
    assert np.allclose(solution.y[-1, :], np.array([0., 0., 1.]), atol=1e-3)

    init_soln = solver.init_step(tspan[0], y0, yp0)
    step_soln = solver.step(tspan[1])

    init_soln.y[1] *= 1e4
    step_soln.y[1] *= 1e4

    assert np.allclose(init_soln.y, solution.y[0, :])
    assert np.allclose(step_soln.y, solution.y[1, :])
