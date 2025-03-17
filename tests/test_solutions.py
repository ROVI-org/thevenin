import warnings

import pytest
import numpy as np
import thevenin as thev
import matplotlib.pyplot as plt


@pytest.fixture(scope='function')  # cannot be module b/c appends
def soln():

    warnings.filterwarnings('ignore')

    sim = thev.Simulation()

    expr = thev.Experiment()
    expr.add_step('current_C', 1., (3600., 1.), limits=('voltage_V', 3.))
    expr.add_step('current_A', 0., (300., 1.))
    expr.add_step('current_C', -1., (3600., 1.), limits=('voltage_V', 4.3))
    expr.add_step('current_A', 0., (300., 1.))

    soln = sim.run(expr)

    return soln


@pytest.fixture(scope='function')
def rest():

    warnings.filterwarnings('ignore')

    sim = thev.Simulation()

    rest = thev.Experiment()
    rest.add_step('current_A', 0., (300., 1.))

    soln = sim.run(rest)

    return soln


def test_step_and_cycle_solutions(soln):

    # solvetime works
    step_soln = soln.get_steps(0)
    assert step_soln.solvetime

    # bad plot
    with pytest.raises(KeyError):
        step_soln.plot('fake', 'plot')

    # plots w/ and w/o units
    with plt.ioff():
        step_soln.plot('soc', 'soc')
        step_soln.plot('time_h', 'voltage_V')
        plt.close('all')

    # solvetime works and times stacked correctly
    cycle_soln = soln.get_steps((0, 1))
    assert cycle_soln.solvetime
    assert all(np.diff(cycle_soln.t) >= 0.)

    # bad plot
    with pytest.raises(KeyError):
        cycle_soln.plot('fake', 'plot')

    # plots w/ and w/o units
    with plt.ioff():
        cycle_soln.plot('soc', 'soc')
        cycle_soln.plot('time_h', 'voltage_V')
        plt.close('all')


def test_cycle_from_single_step(soln):

    step0 = soln.get_steps(0)
    assert isinstance(step0, thev.StepSolution)

    cycle = thev.CycleSolution(step0)
    assert isinstance(cycle, thev.CycleSolution)


def test_append_errors(soln):

    # append wrong size solution (different num_RC_pairs)
    with pytest.raises(ValueError):
        step0 = soln.get_steps(0)
        step0._sim.num_RC_pairs += 1
        soln.append_soln(step0)

    # only works with StepSolution and CycleSolution
    with pytest.raises(TypeError):
        sim = soln._sim
        soln.append_soln(sim)


def test_append_step_soln(soln):

    t_orig = soln.vars['time_s']
    V_orig = soln.vars['voltage_V']

    step0 = soln.get_steps(0)

    soln.append_soln(step0, t_shift=0.)
    assert len(soln.success) == 5

    t_new_times = t_orig[-1] + step0.vars['time_s']
    t_append = np.concatenate([t_orig, t_new_times])
    np.testing.assert_allclose(soln.vars['time_s'], t_append)

    V_append = np.concatenate([V_orig, step0.vars['voltage_V']])
    np.testing.assert_allclose(soln.vars['voltage_V'], V_append)


def test_append_cycle_soln(soln):

    t_orig = soln.vars['time_s']
    V_orig = soln.vars['voltage_V']

    soln.append_soln(soln, t_shift=0.)
    assert len(soln.success) == 8

    t_new_times = t_orig[-1] + t_orig
    t_append = np.concatenate([t_orig, t_new_times])
    np.testing.assert_allclose(soln.vars['time_s'], t_append)

    V_append = np.concatenate([V_orig, V_orig])
    np.testing.assert_allclose(soln.vars['voltage_V'], V_append)


def test_append_w_events(rest, soln):

    rest.append_soln(soln)
    assert len(rest.t_events) == 2
