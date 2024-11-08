# thevenin Changelog

## [Version 0.1.1]()

### New Features

### Optimizations
* Add `options` to the `Experiment.print_steps()` report. This makes it easier to check solver options for each step.

### Bug Fixes
* Make the final value of `tspan` always match `t_max`. In cases where `dt` is used to construct the time array, the final `dt` may differ from the one given. Fixes [Issue #10](https://github.com/ROVI-org/thevenin/issues/10).

### Breaking Changes
* Drop support for Python 3.8 which reached end of support as of October 2024.
