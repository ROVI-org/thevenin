import time
import warnings

import numpy as np
import thevenin as thev
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', UserWarning)


# Predictor example: ~8-10x faster
# -----------------------------------------------------------------------------
pred = thev.Prediction()
pred.gamma = 50.
pred.M_hyst = lambda soc: 0.07
pred.isothermal = True

kwargs = {'soc': 1., 'T_cell': 300., 'hyst': 0., 'eta_j': [0.]}
state = thev.TransientState(**kwargs)

time_s = np.linspace(0, 3600, 3601)
current_A = pred.capacity*np.ones(time_s.size)
y = np.empty(time_s.size)

y[0] = pred.ocv(state.soc)

start = time.time()
for i in range(3600):
    state = pred.take_step(state, current_A[i], 1.)
    y[i+1] = state.voltage

print(f"Prediction (1s steps): {time.time() - start:.5f}")

# Simulation example: original method, slower, but better for general
# simulations because more flexible boundary conditions
# -----------------------------------------------------------------------------
sim = thev.Simulation()
sim.gamma = 50.
sim.M_hyst = lambda soc: 0.07
sim.isothermal = True

sim.pre()

step = thev.Experiment()
step.add_step('current_C', 1., (10., 1))

start = time.time()
solns = []
for i in range(360):
    solns.append(sim.run_step(step, 0))

ida = thev.CycleSolution(*solns, t_shift=0.)
print(f"Simulation (10s steps): {time.time() - start:.5f}")

# Single-step 1C discharge to compare against
# -----------------------------------------------------------------------------
sim.pre()

expr = thev.Experiment()
expr.add_step('current_C', 1., (3600., 1.))

start = time.time()
soln = sim.run(expr)
print(f"Baseline: {time.time() - start:.5f}")

# Visualize and compare results
# -----------------------------------------------------------------------------
plt.figure()
plt.plot(
    time_s[::90], y[::90], 'o',
    ida.t, ida.vars['voltage_V'], '-k',
    soln.t, soln.vars['voltage_V'], '--',
)
plt.xlabel('Time [s]')
plt.ylabel('Volage [V]')
plt.legend(['Predictor', 'Model', 'Baseline'])
