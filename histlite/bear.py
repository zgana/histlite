# bear.py

from __future__ import print_function, division

__doc__  = """Calculate and plot distributions in DataFrames, easily."""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import histlite as hl


def atleast_gb(gb):
    if isinstance(gb, pd.DataFrame):
        gb = [('__all__', gb)]
    return gb


def _keys(gb, basetype=np.number):
    gb = atleast_gb(gb)
    for (a,d) in gb:
        if len(d):
            break
    return [key for key in d.columns
            if issubclass(d[key].dtype.type, basetype)]


def ranges(gb, keys=None, basetype=np.number):
    gb = atleast_gb(gb)
    if keys is None:
        keys = _keys(gb, basetype=basetype)
    else:
        keys = np.atleast_1d(keys)
    return {
        key: (min(d[key].values.min() for (a,d) in gb),
              max(d[key].values.max() for (a,d) in gb))
        for key in keys
    }


def hist1d(gb, keys=None, bins=10, range=None, kde=False, basetype=np.number, **kw):
    gb = atleast_gb(gb)
    if range is None:
        rangs = ranges(gb, keys)
    else:
        rangs = {key: range for key in keys}

    return rangs

    



