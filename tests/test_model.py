import warnings

import pytest
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
def constant_steps():
    expr = thevenin.Experiment()
    expr.add_step('current_A', 1., (3600., 1.), limits=('voltage_V', 3.))
    expr.add_step('current_A', 0., (600., 1.))
    expr.add_step('current_A', -1., (3600., 1.), limits=('voltage_V', 4.3))
    expr.add_step('voltage_V', 4.3, (600., 1.))
    expr.add_step('power_W', 1., (600., 1.), limits=('voltage_V', 3.))

    return expr


@pytest.fixture(scope='function')
def dynamic_current():
    def load(t): return np.sin(2.*np.pi*t / 120.)

    expr = thevenin.Experiment()
    expr.add_step('current_A', load, (600., 1.))

    return expr


@pytest.fixture(scope='function')
def dynamic_voltage():
    def load(t): return 3.8 + 10e-3*np.sin(2.*np.pi*t / 120.)

    expr = thevenin.Experiment()
    expr.add_step('voltage_V', load, (600., 1.))

    return expr


@pytest.fixture(scope='function')
def dynamic_power():
    def load(t): return np.sin(2.*np.pi*t / 120.)

    expr = thevenin.Experiment()
    expr.add_step('power_W', load, (600., 1.))

    return expr


def test_bad_initialization(dict_params):

    # wrong params type
    with pytest.raises(TypeError):
        _ = thevenin.Model(['wrong_type'])

    # invalid/excess key/value pairs
    with pytest.raises(ValueError):
        dict_params['fake'] = 'parameter'
        _ = thevenin.Model(dict_params)


def test_model_w_yaml_input(constant_steps, dynamic_current, dynamic_voltage,
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

    soln = model.run(constant_steps)
    assert soln.success
    # assert any(soln.i_events)

    soln = model.run(dynamic_current)
    assert soln.success

    soln = model.run(dynamic_voltage)
    assert soln.success

    soln = model.run(dynamic_power)
    assert soln.success


def test_bad_yaml_inputs():

    # only .yaml extensions
    with pytest.raises(ValueError):
        _ = thevenin.Model('fake.fake')

    # file doesn't exist
    with pytest.raises(FileNotFoundError):
        _ = thevenin.Model('fake')


def test_run_step(model_2RC, constant_steps):

    sv0 = model_2RC._sv0.copy()
    svdot0 = model_2RC._svdot0.copy()

    stepsoln = model_2RC.run_step(constant_steps, 0)

    assert stepsoln.success
    assert not np.allclose(sv0, model_2RC._sv0)
    assert not np.allclose(svdot0, model_2RC._svdot0)

    model_2RC.pre()

    assert np.allclose(sv0, model_2RC._sv0)
    assert np.allclose(svdot0, model_2RC._svdot0)


def test_model_w_multistep_experiment(model_0RC, model_1RC, model_2RC,
                                      constant_steps):

    soln = model_0RC.run(constant_steps)
    assert soln.success
    # assert any(soln.i_events)

    soln = model_1RC.run(constant_steps)
    assert soln.success
    # assert any(soln.i_events)

    soln = model_2RC.run(constant_steps)
    assert soln.success
    # assert any(soln.i_events)


def test_model_w_dynamic_current(model_0RC, model_1RC, model_2RC,
                                 dynamic_current):

    soln = model_0RC.run(dynamic_current)
    assert soln.success

    soln = model_1RC.run(dynamic_current)
    assert soln.success

    soln = model_2RC.run(dynamic_current)
    assert soln.success


def test_model_w_dynamic_voltage(model_0RC, model_1RC, model_2RC,
                                 dynamic_voltage):

    soln = model_0RC.run(dynamic_voltage)
    assert soln.success

    soln = model_1RC.run(dynamic_voltage)
    assert soln.success

    soln = model_2RC.run(dynamic_voltage)
    assert soln.success


def test_model_w_dynamic_power(model_0RC, model_1RC, model_2RC,
                               dynamic_power):

    soln = model_0RC.run(dynamic_power)
    assert soln.success

    soln = model_1RC.run(dynamic_power)
    assert soln.success

    soln = model_2RC.run(dynamic_power)
    assert soln.success


def test_resting_experiment(model_2RC):

    expr = thevenin.Experiment()
    expr.add_step('current_A', 0., (100., 1.))

    soln = model_2RC.run(expr)

    assert soln.success
    assert np.allclose(soln.vars['voltage_V'], soln.vars['voltage_V'][0])


def test_current_sign_convention(model_2RC, constant_steps):

    soln = model_2RC.run(constant_steps)

    discharge = soln.get_steps(0)
    assert all(np.diff(discharge.vars['voltage_V']) < 0.)

    charge = soln.get_steps(2)
    assert all(np.diff(charge.vars['voltage_V']) > 0.)


def test_constant_V_shift_w_constant_R0(model_0RC, constant_steps):

    model_0RC.R0 = lambda soc, T_cell: 1e-2
    model_0RC.pre()

    soln = model_0RC.run(constant_steps)

    discharge = soln.get_steps(0)
    ocv = model_0RC.ocv(discharge.vars['soc'])
    assert np.allclose(ocv - 1e-2, discharge.vars['voltage_V'])

    charge = soln.get_steps(2)
    ocv = model_0RC.ocv(charge.vars['soc'])
    assert np.allclose(ocv + 1e-2, charge.vars['voltage_V'])


def test_isothermal_flag(model_2RC, constant_steps):

    # with heat on
    model_2RC.isothermal = False
    model_2RC.pre()

    soln = model_2RC.run(constant_steps)
    assert soln.vars['temperature_K'].max() > model_2RC.T_inf
    assert all(soln.vars['temperature_K'] >= model_2RC.T_inf)

    # with heat off
    model_2RC.isothermal = True
    model_2RC.pre()

    soln = model_2RC.run(constant_steps)
    assert np.allclose(soln.vars['temperature_K'], model_2RC.T_inf)


def test_mutable_warning():
    from thevenin._model import short_warn

    with warnings.catch_warnings(record=True) as report:
        warnings.simplefilter('ignore')
        short_warn("This is a test warning.")

    assert len(report) == 0


def test_detected_warning():
    from thevenin._model import short_warn

    with pytest.warns(UserWarning):
        short_warn("This is a test warning")


def test_custom_format():
    from thevenin._model import formatwarning, short_warn

    with warnings.catch_warnings(record=True) as report:
        warnings.simplefilter('always')
        short_warn("This is a test warning.", Warning)

    args = (
        report[0].message,
        report[0].category,
        report[0].filename,
        report[0].lineno,
        report[0].line,
    )

    # ensure the same inputs, remove  any \n, \t, etc.
    custom = formatwarning(*args).strip()
    original = warnings.formatwarning(*args).strip()

    # custom format works
    assert custom == "[thevenin Warning]: This is a test warning."

    # warnings from warnings.warn not impacted by custom format
    assert original != "[thevenin Warning]: This is a test warning."
