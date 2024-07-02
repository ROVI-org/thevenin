import io
import pytest
import warnings
import contextlib

import thevenin
import numpy as np


@pytest.fixture(scope='function')
def dict_params():

    coeffs = np.array([84.6, -348.6, 592.3, -534.3, 275., -80.3, 12.8, 2.8])

    params = {
        'num_RC_pairs': 0,
        'soc0': 0.5,
        'capacity': 1.,
        'mass': 0.5,
        'isothermal': False,
        'Cp': 1150.,
        'T_inf': 300.,
        'h_therm': 12.,
        'A_therm': 1.,
        'ocv': lambda soc: np.polyval(coeffs, soc),
        'R0': lambda soc, T_cell: 0.05 + 0.05*soc - T_cell/1e4,
    }

    return params


@pytest.fixture(scope='function')
def model_0RC(dict_params):
    return thevenin.Model(dict_params)


@pytest.fixture(scope='function')
def model_1RC(dict_params):
    model = thevenin.Model(dict_params)

    model.num_RC_pairs = 1
    model.R1 = lambda soc, T_cell: 0.01 + 0.01*soc - T_cell/3e4
    model.C1 = lambda soc, T_cell: 10. + 10.*soc - T_cell/3e1

    model.pre()

    return model


@pytest.fixture(scope='function')
def model_2RC(dict_params):
    model = thevenin.Model(dict_params)

    model.num_RC_pairs = 2
    model.R1 = lambda soc, T_cell: 0.01 + 0.01*soc - T_cell/3e4
    model.C1 = lambda soc, T_cell: 10. + 10.*soc - T_cell/3e1
    model.R2 = lambda soc, T_cell: 0.01 + 0.01*soc - T_cell/3e4
    model.C2 = lambda soc, T_cell: 10. + 10.*soc - T_cell/3e1

    model.pre()

    return model


@pytest.fixture(scope='function')
def constant_exp():
    exp = thevenin.Experiment()
    exp.add_step('current_A', 1., (3600., 1.), limit=('voltage_V', 3.))
    exp.add_step('current_A', 0., (600., 1.))
    exp.add_step('current_A', -1., (3600., 1.), limit=('voltage_V', 4.3))
    exp.add_step('voltage_V', 4.3, (600., 1.))
    exp.add_step('power_W', 1., (600., 1.), limit=('voltage_V', 3.))

    return exp


@pytest.fixture(scope='function')
def dynamic_current():
    load = lambda t: np.sin(2.*np.pi*t / 120.)

    exp = thevenin.Experiment()
    exp.add_step('current_A', load, (600., 1.))

    return exp


@pytest.fixture(scope='function')
def dynamic_voltage():
    load = lambda t: 3.8 + 10e-3*np.sin(2.*np.pi*t / 120.)

    exp = thevenin.Experiment()
    exp.add_step('voltage_V', load, (600., 1.))

    return exp


@pytest.fixture(scope='function')
def dynamic_power():
    load = lambda t: np.sin(2.*np.pi*t / 120.)

    exp = thevenin.Experiment()
    exp.add_step('power_W', load, (600., 1.))

    return exp


def test_bad_initialization(dict_params):

    # wrong params type
    with pytest.raises(TypeError):
        _ = thevenin.Model(['wrong_type'])

    # invalid/excess key/value pairs
    with pytest.raises(ValueError):
        dict_params['fake'] = 'parameter'
        _ = thevenin.Model(dict_params)


def test_model_w_yaml_input(constant_exp, dynamic_current, dynamic_voltage,
                            dynamic_power):

    # using default file
    with pytest.warns(UserWarning):
        model = thevenin.Model()

    # using default file by name
    with pytest.warns(UserWarning):
        model = thevenin.Model('params')

    # using default file by name w/ extension
    with pytest.warns(UserWarning):
        model = thevenin.Model('params.yaml')

    sol = model.run(constant_exp)
    assert sol.success
    assert any(sol.roots)

    sol = model.run(dynamic_current)
    assert sol.success

    sol = model.run(dynamic_voltage)
    assert sol.success

    sol = model.run(dynamic_power)
    assert sol.success


def test_bad_yaml_inputs():

    # only .yaml extensions
    with pytest.raises(ValueError):
        model = thevenin.Model('fake.fake')

    # file doesn't exist
    with pytest.raises(FileNotFoundError):
        model = thevenin.Model('fake')


def test_run_step(model_2RC, constant_exp):

    sv0 = model_2RC._sv0.copy()
    svdot0 = model_2RC._svdot0.copy()

    stepsol = model_2RC.run_step(constant_exp, 0)

    assert stepsol.success
    assert not np.allclose(sv0, model_2RC._sv0)
    assert not np.allclose(svdot0, model_2RC._svdot0)

    model_2RC.pre()

    assert np.allclose(sv0, model_2RC._sv0)
    assert np.allclose(svdot0, model_2RC._svdot0)


def test_model_w_multistep_exp(model_0RC, model_1RC, model_2RC,
                               constant_exp):

    sol = model_0RC.run(constant_exp)
    assert sol.success
    assert any(sol.roots)

    sol = model_1RC.run(constant_exp)
    assert sol.success
    assert any(sol.roots)

    sol = model_2RC.run(constant_exp)
    assert sol.success
    assert any(sol.roots)


def test_model_w_dynamic_current(model_0RC, model_1RC, model_2RC,
                                 dynamic_current):

    sol = model_0RC.run(dynamic_current)
    assert sol.success

    sol = model_1RC.run(dynamic_current)
    assert sol.success

    sol = model_2RC.run(dynamic_current)
    assert sol.success


def test_model_w_dynamic_voltage(model_0RC, model_1RC, model_2RC,
                                 dynamic_voltage):

    sol = model_0RC.run(dynamic_voltage)
    assert sol.success

    sol = model_1RC.run(dynamic_voltage)
    assert sol.success

    sol = model_2RC.run(dynamic_voltage)
    assert sol.success


def test_model_w_dynamic_power(model_0RC, model_1RC, model_2RC,
                               dynamic_power):

    sol = model_0RC.run(dynamic_power)
    assert sol.success

    sol = model_1RC.run(dynamic_power)
    assert sol.success

    sol = model_2RC.run(dynamic_power)
    assert sol.success


def test_resting_experiment(model_2RC):

    exp = thevenin.Experiment()
    exp.add_step('current_A', 0., (100., 1.))

    sol = model_2RC.run(exp)

    assert sol.success
    assert np.allclose(sol.vars['voltage_V'], sol.vars['voltage_V'][0])


def test_current_sign_convention(model_2RC, constant_exp):

    sol = model_2RC.run(constant_exp)

    discharge = sol.get_steps(0)
    assert all(np.diff(discharge.vars['voltage_V']) < 0.)

    charge = sol.get_steps(2)
    assert all(np.diff(charge.vars['voltage_V']) > 0.)


def test_constant_V_shift_w_constant_R0(model_0RC, constant_exp):

    model_0RC.R0 = lambda soc, T_cell: 1e-2
    model_0RC.pre()

    sol = model_0RC.run(constant_exp)

    discharge = sol.get_steps(0)
    ocv = model_0RC.ocv(discharge.vars['soc'])
    assert np.allclose(ocv - 1e-2, discharge.vars['voltage_V'])

    charge = sol.get_steps(2)
    ocv = model_0RC.ocv(charge.vars['soc'])
    assert np.allclose(ocv + 1e-2, charge.vars['voltage_V'])


def test_isothermal_flag(model_2RC, constant_exp):

    # with heat on
    model_2RC.isothermal = False
    model_2RC.pre()

    sol = model_2RC.run(constant_exp)
    assert sol.vars['temperature_K'].max() > model_2RC.T_inf
    assert all(sol.vars['temperature_K'] >= model_2RC.T_inf)

    # with heat off
    model_2RC.isothermal = True
    model_2RC.pre()

    sol = model_2RC.run(constant_exp)
    assert np.allclose(sol.vars['temperature_K'], model_2RC.T_inf)


def test_custom_showwarning():
    from thevenin._model import _showwarning

    warnings.simplefilter('always')

    # Redirect stderr to capture print statements
    captured_output = io.StringIO()
    with contextlib.redirect_stderr(captured_output):
        warnings.showwarning = _showwarning
        warnings.warn("This is a test warning.", UserWarning)

    # Get the captured output
    output = captured_output.getvalue().strip()

    assert output == "[thevenin UserWarning]: This is a test warning."
