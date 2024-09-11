import pytest
import thevenin
import numpy as np


@pytest.fixture(scope='function')
def exp():
    return thevenin.Experiment()


def test_initialization(exp):

    assert exp._steps == []
    assert exp._kwargs == []
    assert exp._options == {}

    assert exp.steps == []
    assert exp.num_steps == 0


def test_add_step(exp):

    # wrong mode
    with pytest.raises(ValueError):
        exp.add_step('current_C', 1., (3600., 1.))

    # wrong tspan length
    with pytest.raises(ValueError):
        exp.add_step('current_A', 1., (0., 3600., 150))

    # wrong tspan type
    with pytest.raises(TypeError):
        exp.add_step('current_A', 1., (3600., '1'))

    # bad limits name
    with pytest.raises(ValueError):
        exp.add_step('current_A', 1., (3600., 1.), limits=('fake', 3.))

    # bad limits length
    with pytest.raises(ValueError):
        exp.add_step('current_A', 1., (3600., 1.),
                     limits=('voltage_V', 3., 'soc'))

    # bad limits value type
    with pytest.raises(TypeError):
        exp.add_step('current_A', 1., (3600., 1.), limits=('voltage_V', '3'))

    # test current and linspace construction
    exp.add_step('current_A', 1., (3600., 150))
    step = exp.steps[0]

    assert exp.num_steps == 1
    assert np.allclose(step['tspan'], np.linspace(0., 3600., 150))

    # test voltage and arange construction
    exp.add_step('voltage_V', 4., (3600., 1.))
    step = exp.steps[1]

    assert exp.num_steps == 2
    assert np.allclose(step['tspan'], np.arange(0., 3601., 1., dtype=float))

    # test power construction
    exp.add_step('power_W', 1., (3600., 1.))

    assert exp.num_steps == 3

    # test limit keyword
    exp.add_step('current_A', 1., (3600., 1.), limits=('voltage_V', 3.))

    assert exp.steps[-1]['limits'] == ('voltage_V', 3.)

    # test dynamic load
    exp.add_step('current_A', lambda t: 1., (3600., 1.))
    step = exp.steps[-1]

    assert callable(step['value'])


def test_print_steps(exp):

    # no error when empty
    exp.print_steps()
    assert True

    # also works with step(s)
    exp.add_step('current_A', 1., (3600., 1.))
    exp.print_steps()
    assert True
