import pytest
import thevenin
import numpy as np


@pytest.fixture(scope='function')
def demand():
    return thevenin.Experiment()


def test_initialization(demand):

    assert demand._steps == []
    assert demand._kwargs == []
    assert demand._options == {}

    assert demand.steps == []
    assert demand.num_steps == 0


def test_add_step(demand):

    # wrong mode
    with pytest.raises(ValueError):
        demand.add_step('current_C', 1., (3600., 1.))

    # wrong tspan length
    with pytest.raises(ValueError):
        demand.add_step('current_A', 1., (0., 3600., 150))

    # wrong tspan type
    with pytest.raises(TypeError):
        demand.add_step('current_A', 1., (3600., '1'))

    # bad limits name
    with pytest.raises(ValueError):
        demand.add_step('current_A', 1., (3600., 1.), limits=('fake', 3.))

    # bad limits length
    with pytest.raises(ValueError):
        demand.add_step('current_A', 1., (3600., 1.),
                        limits=('voltage_V', 3., 'soc'))

    # bad limits value type
    with pytest.raises(TypeError):
        demand.add_step('current_A', 1., (3600., 1.),
                        limits=('voltage_V', '3'))

    # test current and linspace construction
    demand.add_step('current_A', 1., (3600., 150))
    step = demand.steps[0]

    assert demand.num_steps == 1
    assert np.allclose(step['tspan'], np.linspace(0., 3600., 150))

    # test voltage and arange construction
    demand.add_step('voltage_V', 4., (3600., 1.))
    step = demand.steps[1]

    assert demand.num_steps == 2
    assert np.allclose(step['tspan'], np.arange(0., 3601., 1., dtype=float))

    # test power construction
    demand.add_step('power_W', 1., (3600., 1.))

    assert demand.num_steps == 3

    # test limit keyword
    demand.add_step('current_A', 1., (3600., 1.), limits=('voltage_V', 3.))

    assert demand.steps[-1]['limits'] == ('voltage_V', 3.)

    # test dynamic load
    demand.add_step('current_A', lambda t: 1., (3600., 1.))
    step = demand.steps[-1]

    assert callable(step['value'])


def test_print_steps(demand):

    # no error when empty
    demand.print_steps()
    assert True

    # also works with step(s)
    demand.add_step('current_A', 1., (3600., 1.))
    demand.print_steps()
    assert True
