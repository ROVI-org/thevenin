import warnings

import pytest
import thevenin
import numpy as np
import matplotlib.pyplot as plt


@pytest.fixture(scope='function')
def soln():

    warnings.filterwarnings('ignore')

    model = thevenin.Model()

    demand = thevenin.Experiment()
    demand.add_step('current_A', 1., (3600., 1.), limits=('voltage_V', 3.))
    demand.add_step('current_A', 0., (3600., 1.))

    soln = model.run(demand)

    return soln


def test_step_and_cycle_solutions(soln):

    # solvetime works
    step_soln = soln.get_steps(0)
    assert step_soln.solvetime

    # bad plot
    with pytest.raises(KeyError):
        step_soln.plot('fake', 'plot')

    # plots w/ and w/o units
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
    cycle_soln.plot('soc', 'soc')
    cycle_soln.plot('time_h', 'voltage_V')
    plt.close('all')
