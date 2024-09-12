import pytest
import thevenin
import numpy as np

from scikits_odes_sundials import ida
from thevenin._ida_solver import SolverReturn


@pytest.fixture(scope='function')
def idasol():
    def residuals(t, y, yp, res):
        res[0] = yp[0]

    solver = ida.IDA(residuals)
    dummysol = solver.solve(np.linspace(0., 10., 11), [1.], [0.])

    return dummysol


@pytest.fixture(scope='function')
def roots():
    def rootfn(t, y, yp, events):
        events[0] = y[0] - 0.5

    def residuals(t, y, yp, res):
        res[0] = yp[0] + 0.1

    solver = ida.IDA(residuals, rootfn=rootfn, nr_rootfns=1)
    rootsol = solver.solve(np.linspace(0., 10., 11), [1.], [0.])

    return rootsol


@pytest.fixture(scope='function')
def tstop():
    def residuals(t, y, yp, res):
        res[0] = yp[0]

    solver = ida.IDA(residuals, tstop=4.5)
    tstopsol = solver.solve(np.linspace(0., 10., 11), [1.], [0.])

    return tstopsol


@pytest.fixture(scope='function')
def errors():
    def residuals(t, y, yp, res):
        res[0] = 0.

    solver = ida.IDA(residuals)
    errorsol = solver.solve(np.linspace(0., 10., 11), [1.], [0.])

    return errorsol


@pytest.fixture(scope='function')
def event_stack(tstop, roots):

    stackedsol = ida.SolverReturn(tstop.flag, tstop.values, tstop.errors,
                                  roots.roots, tstop.tstop, tstop.message)

    return stackedsol


def test_solver_return(idasol):
    solution = SolverReturn(idasol)

    assert solution.success
    assert not solution.roots
    assert not solution.tstop
    assert not solution.errors

    assert solution.message == idasol.message
    assert np.allclose(solution.t, idasol.values.t)
    assert np.allclose(solution.y, idasol.values.y)
    assert np.allclose(solution.ydot, idasol.values.ydot)


def test_solver_return_roots(roots):
    solution = SolverReturn(roots)

    assert solution.success
    assert solution.roots[0]
    assert np.allclose(solution.y[-1], roots.roots.y)


def test_solver_return_tstop(tstop):
    solution = SolverReturn(tstop)

    assert solution.success
    assert solution.tstop[0]
    assert np.allclose(solution.y[-1], tstop.tstop.y)


def test_solver_return_errors(errors):
    solution = SolverReturn(errors)

    assert not solution.success
    assert solution.errors[0]
    assert np.allclose(solution.y[-1], errors.errors.y)


def test_solver_return_event_stack(event_stack):
    solution = SolverReturn(event_stack)

    assert solution.tstop[0]
    assert solution.tstop[-1] == -2

    assert solution.roots[0]
    assert solution.roots[-1] == -1


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

    initsol = solver.init_step(tspan[0], y0, yp0)
    stepsol = solver.step(tspan[1])

    assert np.allclose(initsol.y, solution.y[0, :])
    assert np.allclose(stepsol.y, solution.y[1, :])


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

    initsol = solver.init_step(tspan[0], y0, yp0)
    stepsol = solver.step(tspan[1])

    initsol.y[1] *= 1e4
    stepsol.y[1] *= 1e4

    assert np.allclose(initsol.y, solution.y[0, :])
    assert np.allclose(stepsol.y, solution.y[1, :])
