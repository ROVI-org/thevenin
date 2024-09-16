<img alt='Logo' style='width: 75%; min-width: 250px; max-width: 500px;' 
 src='https://github.com/ROVI-org/thevenin/blob/main/images/dark.png?raw=true#gh-dark-mode-only'/>
<img alt='Logo' style='width: 75%; min-width: 250px; max-width: 500px;' 
 src='https://github.com/ROVI-org/thevenin/blob/main/images/light.png?raw=true#gh-light-mode-only'/>

 <br>

[![CI][ci-b]][ci-l] &nbsp;
![tests][test-b] &nbsp;
![coverage][cov-b] &nbsp;
[![pep8][pep-b]][pep-l]

[ci-b]: https://github.com/ROVI-org/thevenin/actions/workflows/ci.yaml/badge.svg
[ci-l]: https://github.com/ROVI-org/thevenin/actions/workflows/ci.yaml

[test-b]: https://raw.githubusercontent.com/ROVI-org/thevenin/1f2d5c206f2656163823352cbbf4459c4cb327ec/images/tests.svg
[cov-b]: https://raw.githubusercontent.com/ROVI-org/thevenin/1f2d5c206f2656163823352cbbf4459c4cb327ec/images/coverage.svg

[pep-b]: https://img.shields.io/badge/code%20style-pep8-orange.svg
[pep-l]: https://www.python.org/dev/peps/pep-0008

## Summary
This package is a wrapper for the well-known Thevenin equivalent circuit model. The model is comprised of a single series reistor followed by any number of parallel RC pairs. Figure 1 below illustrates a circuit with 2 RC pairs; however, the model can be run with as few as zero, and as many as $N$.

<p align="center">
  <img alt="2RC Thevenin circuit." style="width: 75%; min-width: 250px; max-width: 500px;" 
   src="https://github.com/ROVI-org/thevenin/blob/main/images/example_circuit.png?raw=true"/>
  </br>
  Figure 1: 2RC Thevenin circuit.
</p>

This system is governed by the evolution of the state of charge (soc, -), RC overpotentials ($V_j$, V), cell voltage ($V_{\rm cell}$, V), and temperature ($T_{\rm cell}$, K). soc and $V_j$ evolve in time as
```math
\begin{align}
  &\frac{d\rm soc}{dt} = \frac{-I}{3600 Q_{\rm max}}, \\
  &\frac{dV_j}{dt} = -\frac{V_j}{R_jC_j} + \frac{I}{C_j},
\end{align}
```
where $I$ is the load current (A), $Q_{\rm max}$ is the maximum nominal cell capacity (Ah), and $R_j$ and $C_j$ are the resistance (Ohm) and capacitance (F) of each RC pair $j$. Note that the sign convention for $I$ is chosen such that positive $I$ discharges the battery (reduces soc) and negative $I$ charges the battery (increases soc). This convention is consistent with common physics-based models, e.g., the single particle model or pseudo-2D model. While not explicitly included in the equations above, $R_j$ and $C_j$ are functions of soc and $T_{\rm cell}$. The temperature increases while the cell is active according to
```math
\begin{equation}
  mC_p\frac{dT_{\rm cell}}{dt} = \dot{Q}_{\rm gen} + \dot{Q}_{\rm conv},
\end{equation}
```
where $m$ is mass (kg), $C_p$ is specific heat capacity (J/kg/K), $Q_{\rm gen}$ is the heat generation (W), and $Q_{\rm conv}$ is the convective heat loss (W). Heat generation and convection are defined by
```math
\begin{align}
  &\dot{Q}_{\rm gen} = I \times (V_{\rm ocv}({\rm soc}) - V_{\rm cell}), \\
  &\dot{Q}_{\rm conv} = hA(T_{\infty} - T_{\rm cell}),
\end{align}
```
where $h$ is the convecitive heat transfer coefficient (W/m<sup>2</sup>/K), $A$ is heat loss area (m<sup>2</sup>), and $T_{\infty}$ is the air/room temperature (K). $V_{\rm ocv}$ is the open circuit voltage (V) and is a function of soc.

The overall cell voltage is
```math
\begin{equation}
  V_{\rm cell} = V_{\rm ocv}({\rm soc}) - \sum_j V_j - IR_0,
\end{equation}
```
where $R_0$ the lone series resistance (Ohm), as shown in Figure 1. Just like the other resistive elements, $R_0$ is a function of soc and $T_{\rm cell}$.

## Installation
We recommend using [Anaconda](https://anaconda.com) to install this package due to the [scikits-odes-sundials](https://scikits-odes.readthedocs.io) dependency, which is installed separately using `conda install` to avoid having to download and compile SUNDIALS using a local C compiler and a signicant number of C source files. Please refer to the linked `scikits-odes` documentation if you'd prefer to install their package without using `conda`.

After cloning the repository, or downloading the files, use your terminal (MacOS/Linux) or Anaconda Prompt (Windows) to navigate into the folder with the `pyproject.toml` file. Once in the correct folder, execute the following commands:

```command
conda create -n rovi python=3.12 scikits_odes_sundials -c conda-forge
conda activate rovi
pip install .
```

The first command will create a new Python environment named `rovi`. The environment will be set up using Python 3.12 and will install the `scikits-odes-sundials` dependency from the `conda-forge` channel. Feel free to use an alternate environment name as and/or to specify a different Python version >= 3.8. Although the package supports multiple Python versions, development and testing is primarily done using 3.12. Therefore, if you have issues with another version, you should revert to using 3.12. The last two commands activate your new environment and install `thevenin`.

If you plan to make changes to the package, you may also want to consider installing in "editable" mode using the `-e` flag, and including the optional developer dependencies, using `[dev]`, as shown below. If you plan to push any changes back into this repository, you should see the [contributing](#contributing) section first.

```command
pip install -e .[dev]
```

## Get Started
The API is organized around three main classes that allow you to construct the model, define an experiment, and interact with the solution. A basic example for a constant-current discharge is given below. To see the documentation for any of the classes or their methods, use Python's built in `help()` function. You can also access the documentation by visiting the [website](https://rovi-org.github.io/thevenin) hosted through GitHub pages. The website includes search functionality and more detailed examples compared to those included in the docstrings.

```python
import thevenin

model = thevenin.Model()

exp = thevenin.Experiment()
exp.add_step('current_A', 15., (3600., 1.), limits=('voltage_V', 3.))

sol = model.run(experiment)
sol.plot('capacity_Ah', 'voltage_V')
```

**Notes:**
* If you are new to Python, check out [Spyder IDE](https://www.spyder-ide.org/). Spyder is a powerful interactive development environment (IDE) that can make programming in Python more approachable to new users.

## Contributing
If you'd like to contribute to this package, please look through the existing [issues](https://github.com/ROVI-org/thevenin/issues). If the bug you've caught or the feature you'd like to add isn't already being worked on, please submit a new issue before getting started. You should also read through the [developer guidelines](https://rovi-org.github.io/thevenin/development).

## Acknowledgements
This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE). The views expressed in the repository do not necessarily represent the views of the DOE or the U.S. Government.

The motivation and funding for this project came from the Rapid Operational Validation Initiative (ROVI) sponsored by the Office of Electricity. The focus of ROVI is "to greatly reduce time required for emerging energy storage technologies to go from lab to market by developing new tools that will accelerate the testing and validation process needed to ensure commercial success." If interested, you can read more about ROVI [here](https://www.energy.gov/oe/rapid-operational-validation-initiative-rovi).
