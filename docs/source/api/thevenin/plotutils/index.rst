thevenin.plotutils
==================

.. py:module:: thevenin.plotutils

.. autoapi-nested-parse::

   Plotting Utilities
   ------------------
   A module designed to enhance plotting with the matplotlib library. Helper
   functions include routines for simplifying color scheme management, formatting
   axis ticks, fonts, and more, making it easier to create polished and consistent
   visualizations.



Functions
---------

.. autoapisummary::

   thevenin.plotutils.get_colors


Package Contents
----------------

.. py:function:: get_colors(size, data = None, norm = None, alpha = 1.0, cmap = 'jet')

   Sample colors from 'cmap'.

   Return a list of colors from a specified colormap. Default options will
   provide evenly spaced colors across 'cmap'. Provide 'data' and/or 'norm'
   to control the ordering, spacing, and normalization.

   :param size: Number of colors to return.
   :type size: int
   :param data: A 1D array with length 'size' that controls the spacing and sorting of
                the output. By default, spacing is equal and sorting matches 'cmap'.
   :type data: array_like[float] or None, optional
   :param norm: An array-like (min, max) pair that normalizes the colormap to 'data'.
                By default (0, size) if 'data=None' or min/max of 'data' otherwise.
   :type norm: array_like[float] or None, optional
   :param alpha: Transparency to apply over the colormap. Must be in the range [0, 1].
                 The default is 1.
   :type alpha: float, optional
   :param cmap: A valid matplotlib colormap name. The default is 'jet'.
   :type cmap: str, optional

   :returns: **colors** (*list*) -- A list of (r, g, b, a) color codes.

   :raises ValueError: 'data' length must match 'size'.
   :raises ValueError: 'norm' length must equal 2.


