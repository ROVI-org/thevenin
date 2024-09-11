import warnings

import pytest
import thevenin
import numpy as np
import matplotlib.pyplot as plt


@pytest.fixture(scope='function')
def sol():

    warnings.filterwarnings('ignore')

    model = thevenin.Model()

    exp = thevenin.Experiment()
    exp.add_step('current_A', 1., (3600., 1.), limits=('voltage_V', 3.))
    exp.add_step('current_A', 0., (3600., 1.))

    sol = model.run(exp)

    return sol


def test_step_and_cycle_solutions(sol):

    # solvetime works
    stepsol = sol.get_steps(0)
    assert stepsol.solvetime

    # bad plot
    with pytest.raises(KeyError):
        stepsol.plot('fake', 'plot')

    # plots w/ and w/o units
    stepsol.plot('soc', 'soc')
    stepsol.plot('time_h', 'voltage_V')
    plt.close('all')

    # solvetime works and times stacked correctly
    cyclesol = sol.get_steps((0, 1))
    assert cyclesol.solvetime
    assert all(np.diff(cyclesol.t) >= 0.)

    # bad plot
    with pytest.raises(KeyError):
        cyclesol.plot('fake', 'plot')

    # plots w/ and w/o units
    cyclesol.plot('soc', 'soc')
    cyclesol.plot('time_h', 'voltage_V')
    plt.close('all')
