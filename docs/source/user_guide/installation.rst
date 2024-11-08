Installation
============
Presently the Thevenin package is only available to install directly from the GitHub repository. It will eventually be hosted on both `PyPI <https://pypi.org>`_ and `Anaconda <https://anaconda.com>`_ once it is ready for a full v1.0.0 release. For users interested in early adoption, follow the directions below.

Setup Dependencies
------------------
Thevenin depends on `scikits-odes-sundials <https://pypi.org/project/scikits-odes-sundials/>`_ for its numerical solvers. Unfortunately, this package does not provide binary installations via PyPI, and installing from the source distribution is relatively complex and requires a significant number of steps. Therefore, for the time being, we recommend setting up a ``conda`` environment and installing ``scikits_odes_sundials`` via the ``conda-forge`` channel. Thevenin developers maintain this distribution to make installation more straightforward. There are also plans to replace this dependency in the near future, which will also help simplify installation.

If you don't already have Anaconda installed, you can download it for any operating system `here <https://www.anaconda.com/download/success>`_. After setup, run the following commands in the terminal (MacOS/Linux) or in Anaconda Prompt (Windows)::

    conda create -n rovi python=3.12 scikits_odes_sundials -c conda-forge 
    conda activate rovi 

The first command will set up Python 3.12 and the ``scikits_odes_sundials`` dependency in a new environment named "rovi". The next command activates the "rovi" environment prior to the next steps. If you'd prefer to use a different Python version and/or to name your environment something other than "rovi", you can change the number and/or name accordingly. Be aware, however, that Thevenin only supports Python versions 3.8 and higher. 

Download the Files
------------------
Once you have satisfied the ``scikits_odes_sundials`` requirement, you are ready to download and install Thevenin. If you are a ``git`` user, clone the GitHub repo using using the following command::

    git clone https://github.com/ROVI-org/thevenin.git

Alternatively, you can simply download the zipped repository files using the web interface.

Finish Installation
-------------------
Once the package files are available on your machine, use your terminal or Anaconda Prompt to navigate into the folder with the ``pyproject.toml`` file. Finally, run the following to install Thevenin::

    pip install . 

For users planning to make changes to the package, you should consider installing in editable mode and including the optional developer dependencies. In this case, your command should look like::

    pip install -e .[dev]

We also suggest reading through the :ref:`developer guidelines <development>` before modifying any source code if you plan for your changes to be merged into the main repository. These guidelines cover important tools and steps to ensure all developers maintain a consistent workflow, styling, etc. 
