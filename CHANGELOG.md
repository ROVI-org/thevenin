# thevenin Changelog

## [Unreleased](https://github.com/NREL/thevenin)

### New Features
- Added Coulombic efficiency (`ce`) as a parameter option ([#4](https://github.com/NREL/thevenin/pull/4))

### Optimizations

### Bug Fixes

### Breaking Changes
- New Coulombic efficiency option means users will need to update old `params` inputs to also include `ce`

## [v0.1.1](https://github.com/NREL/thevenin/tree/v0.1.1)

### Bug Fixes
- Corrected some docstrings

## [v0.1.0](https://github.com/NREL/thevenin/tree/v0.1.0)
This is the first official release of `thevenin`. Main features/capabilities are listed below.

### Features
- Support for any number of RC pairs
- Run constant or dynamic loads with current, voltage, or power control
- Parameters have temperature and state of charge dependence
- Experiment limits to trigger switching between steps
- Multi-limit support (end at voltage, time, etc. - whichever occurs first)

### Notes
- Implemented `pytest` with full package coverage
- Source/binary distributions available on [PyPI](https://pypi.org/project/thevenin)
- Documentation available on [Read the Docs](https://thevenin.readthedocs.io/)

