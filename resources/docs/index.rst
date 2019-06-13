.. _histlite:

histlite Reference
==================


.. automodule:: icecube.histlite


Creating Histograms
-------------------

.. autofunction:: hist

For example::

    h = histlite.hist ((energy, np.cos (zenith)), weights,
                       bins=(16, 20),
                       range=((1e3, 1e7), (-1,1)),
                       log=(True, False))


.. autofunction:: hist_like

For example::

    h2 = histlite.hist_like (h, (data_E, data_cz))

.. autofunction:: hist_like_indices

For example::

    indices = h.ravel_indices (energy, np.cos (zenith))
    h3 = histlite.hist_like_indices (h, indices, modified_weights)

.. autofunction:: hist_from_function

For example::

    f = lambda x, y: x**2 + y**2
    x = y = np.linspace (-2, 2)
    h = histlite.hist_from_function ([x, y], f)


Hist
----

.. autoclass:: Hist

    **Constructor:**

    .. automethod:: __init__

    **Properties:**

    .. autoattribute:: n_dim
    .. autoattribute:: bins
    .. autoattribute:: n_bins
    .. autoattribute:: range
    .. autoattribute:: values
    .. autoattribute:: errors
    .. autoattribute:: data
    .. autoattribute:: weights
    .. autoattribute:: log
    .. autoattribute:: centers
    .. autoattribute:: widths
    .. autoattribute:: volumes
    .. autoattribute:: T

    **Bin Data Access:**

    .. automethod:: index
    .. automethod:: indices
    .. automethod:: ravel_indices
    .. automethod:: get_value
    .. automethod:: get_values
    .. automethod:: get_error
    .. automethod:: get_errors
    .. automethod:: sample

    **Axis-wise Operations:**

    .. automethod:: sum
    .. automethod:: cumsum
    .. automethod:: integrate
    .. automethod:: project
    .. automethod:: contain
    .. automethod:: contain_project
    .. automethod:: median
    .. automethod:: normalize
    .. automethod:: rebin
    .. automethod:: get_slice

    **Operators:**

    .. automethod:: __add__
    .. automethod:: __radd__
    .. automethod:: __sub__
    .. automethod:: __rsub__
    .. automethod:: __mul__
    .. automethod:: __rmul__
    .. automethod:: __div__
    .. automethod:: __pow__
    .. automethod:: __rpow__
    .. automethod:: __neg__
    .. automethod:: __getitem__

    **Other Arithmetic Operations**

    .. automethod:: exp
    .. automethod:: log_base
    .. automethod:: ln
    .. automethod:: log2
    .. automethod:: log10
    .. automethod:: log1p

    **Other Methods:**

    .. automethod:: curve_fit
    .. automethod:: spline_fit
    .. automethod:: efficiency
    .. automethod:: matches
    .. automethod:: assert_match


Plotting
--------

.. autofunction:: plot1d

.. autofunction:: plot2d
.. autofunction:: label2d

.. autoclass:: LineStyle

    **Constructor:**

    .. automethod:: __init__

    **Properties:**

    .. autoattribute:: line
    .. autoattribute:: markers
    .. autoattribute:: errorbars
    .. autoattribute:: errorcaps
    .. autoattribute:: kwargs

    **Methods:**

    .. automethod:: update
    .. automethod:: copy


