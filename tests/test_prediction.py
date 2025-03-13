import warnings

import numpy as np
import thevenin as thev


def test_prediction_steps_against_simulation():

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        pred = thev.Prediction()
        pred.gamma = 50.
        pred.M_hyst = lambda soc: 0.07
        pred.isothermal = False
        pred.pre()

        sim = thev.Simulation()
        sim.gamma = 50.
        sim.M_hyst = lambda soc: 0.07
        sim.isothermal = False
        sim.pre()

    # 1C simulation discharge to compare against
    expr = thev.Experiment()
    expr.add_step('current_C', 1., (3600., 1.))

    soln = sim.run(expr)
    assert soln.success

    # step-by-step prediction comparison
    state = thev.TransientState(
        soc=soln.vars['soc'][0],
        T_cell=soln.vars['temperature_K'][0],
        hyst=soln.vars['hysteresis_V'][0],
        eta_j=np.zeros(sim.num_RC_pairs),
    )

    time_s = soln.t
    current_A = sim.capacity*np.ones(time_s.size)

    pred_V = np.empty(time_s.size)
    pred_T = np.empty(time_s.size)
    pred_h = np.empty(time_s.size)

    pred_V[0] = soln.vars['voltage_V'][0]
    pred_T[0] = soln.vars['temperature_K'][0]
    pred_h[0] = soln.vars['hysteresis_V'][0]

    pred_etaj = np.empty((time_s.size, sim.num_RC_pairs))
    for j in range(sim.num_RC_pairs):
        pred_etaj[0][j] = soln.vars[f"eta{j+1}_V"][0]

    for i in range(3600):
        state = pred.take_step(state, current_A[i], 1.)

        pred_V[i+1] = state.voltage
        pred_T[i+1] = state.T_cell
        pred_h[i+1] = state.hyst

        for j in range(sim.num_RC_pairs):
            pred_etaj[i+1][j] = state.eta_j[j]

    np.testing.assert_allclose(pred_V, soln.vars['voltage_V'], rtol=1e-3)
    np.testing.assert_allclose(pred_T, soln.vars['temperature_K'], rtol=1e-3)

    np.testing.assert_allclose(
        pred_h, soln.vars['hysteresis_V'], rtol=1e-3, atol=1e-3,
    )

    for j in range(sim.num_RC_pairs):
        np.testing.assert_allclose(
            pred_etaj[:, j], soln.vars[f"eta{j+1}_V"], rtol=1e-3, atol=1e-5,
        )
