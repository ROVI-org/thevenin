import warnings

import pytest


def test_mutable_warning():
    from thevenin._basemodel import short_warn

    with warnings.catch_warnings(record=True) as report:
        warnings.simplefilter('ignore')
        short_warn("This is a test warning.")

    assert len(report) == 0


def test_detected_warning():
    from thevenin._basemodel import short_warn

    with pytest.warns(UserWarning):
        short_warn("This is a test warning")


def test_custom_format():
    from thevenin._basemodel import formatwarning, short_warn

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
    assert custom == "[thevenin Warning] This is a test warning."

    # warnings from warnings.warn not impacted by custom format
    assert original != "[thevenin Warning] This is a test warning."
