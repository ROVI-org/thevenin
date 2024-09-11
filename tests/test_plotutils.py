import pytest
import thevenin as thev


def test_get_colors():

    # bad 'data' length
    with pytest.raises(ValueError):
        _ = thev.plotutils.get_colors(3, data=[1, 2, 3, 4])

    # bad 'norm' length
    with pytest.raises(ValueError):
        _ = thev.plotutils.get_colors(3, norm=[1, 2, 3])

    colors = thev.plotutils.get_colors(3)
    assert len(colors) == 3

    for color in colors:
        assert len(color) == 4
        assert all([0 <= x <= 1 for x in color])

    colors = thev.plotutils.get_colors(5, alpha=0.5)
    for color in colors:
        assert color[-1] == 0.5
