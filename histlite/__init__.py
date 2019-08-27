# histlite.py

from __future__ import print_function, division

__doc__  = """Calculate and plot histograms, easily."""

# Lame python 3 compat
try:
    xrange
except NameError:
    xrange = range

import copy
import datetime
import inspect
try:
    from itertools import izip
except ImportError:
    izip = zip
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate, ndimage, optimize, stats
import warnings

eps = np.finfo(float).eps

try:
    import heal
except:
    pass

def reindex (a, order):
    """Rearrange the axes of a multidimensional array.

    :type   a: ndarray
    :param  a: the input array

    :type   order: sequence of int
    :param  order: the axis that should wind up in each ordinal position

    :return: reindexed array

    Note: this is useful for implementing :meth:`Hist.sum`, etc., but you probably should
    prefer ``np.swapaxes``, possibly using multiple applications, instead.
    """
    cur = list(range (len (a.shape)))
    assert (sorted (order) == cur)
    for i_dest, i_source in enumerate (order):
        i_cur = cur.index (i_source)
        cur[i_dest], cur[i_cur] = cur[i_cur], cur[i_dest]
        a = np.swapaxes (a, i_dest, i_cur)
    return a

def unreindex (a, order):
    """Reverse the effects of :meth:`reindex`.

    :type   a: ndarray
    :param  a: the already reindexed array

    :type   order: sequence of int
    :param  order: order previously applied to :meth:`reindex`

    :return: unreindexed array

    Note: this is useful for implementing :meth:`Hist.sum`, etc., but you probably should
    prefer ``np.swapaxes``, possibly using multiple applications, instead.
    """
    cur = list(range (len (a.shape)))
    assert (sorted (order) == cur), '{} vs {}'.format (sorted (order), cur)
    for i_source, i_dest in enumerate (order):
        i_cur = cur.index (i_source)
        cur[i_dest], cur[i_cur] = cur[i_cur], cur[i_dest]
        a = np.swapaxes (a, i_dest, i_cur)
    return a


def breakout_1d (a):
    return np.r_[a[0], a, a[-1]]

def breakout_2d (a):
    z = np.zeros ((a.shape[0] + 2, a.shape[1] + 2))
    z[1:-1, 1:-1] = a
    z[0, 1:-1] = a[0]
    z[-1, 1:-1] = a[-1]
    z[1:-1, 0] = a[:,0]
    z[1:-1, -1] = a[:,-1]
    for (i, j) in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
        z[i, j] = a[i,j]
    return z


class Hist (object):

    """A histogram."""

    def __init__ (self, bins, values, errors=None, data=None, weights=None):
        """Construct a :class:`Hist`.

        :type   bins: 2D array-like
        :param  bins: the bin edges

        :type   values: n_dim array-like
        :param  values: the counts

        :type   errors: n_dim array-like
        :param  values: the per-bin errors if given; otherwise NaN is assumed.

        :type   data: tuple of array-like
        :param  data: the source data for the histogram: n_dim tuples of
            n_sample arrays

        :type   weights: array-like
        :param  weights: the weights of the source datapoints: an array of
            length n_sample

        """
        self._values = np.asarray (values)
        if len (self.values.shape) == 1:
            try:
                bins_array = np.asarray (bins)
                if len (bins_array) == self.values.shape[0] + 1:
                    bins = (bins,)
            except:
                pass
        self._bins = list(map (np.asarray, bins))
        if errors is not None:
            self._errors = np.asarray (errors)
        else:
            self._errors = None
            if len (self.values.shape):
                self._errors = np.empty (self.values.shape)
                self._errors[:] = np.nan
            else:
                self._errors = np.array (np.nan)
        self._data = data
        self._weights = weights
        self._n_dim = len (self.bins)
        self._range = [(b[0], b[-1]) for b in self.bins]
        # plan to lazy-compute log and centers
        self._log = None
        self._centers = None

    def copy (self):
        b, v, e = map (copy.deepcopy, (self._bins, self._values, self._errors))
        d, w = self._data, self._weights
        return Hist (b, v, e, data=d, weights=w)

    @property
    def zeronans (self):
        self = self.copy ()
        self._values[self._values == 0] = np.nan
        return self

    def to_finite (self, nan=0, inf=np.max, minf=np.min):
        self = self.copy ()
        masks = np.isnan (self.values), self.values == np.inf, self.values == -np.inf
        values = self.values
        for (mask, k) in izip (masks, (nan, inf, minf)):
            if callable (k):
                v = k (values[~mask])
            else:
                v = k
            values[mask] = v
        return self

    def _determine_log (self):
        log = []
        for dim in xrange (self.n_dim):
            diffs = np.diff (self.bins[dim])
            if 0 in self.bins[dim] \
                or isinstance (self.bins[dim][0], datetime.datetime) \
                or isinstance (self.bins[dim][0], np.datetime64):
                dim_log = False
            else:
                ratios = self.bins[dim][1:] / self.bins[dim][:-1]
                nlog = len (np.unique (ratios))
                nlin = len (np.unique (diffs))
                dim_log = (nlog < nlin)
            log.append (dim_log)
        self._log = np.array (log)

    def _determine_centers (self):
        log = self.log
        centers = []
        for dim in xrange (self.n_dim):
            if log[dim]:
                ratios = self.bins[dim][1:] / self.bins[dim][:-1]
                centers.append (self.bins[dim][:-1] * np.sqrt (ratios))
            else:
                diffs = np.diff (self.bins[dim])
                centers.append (self.bins[dim][:-1] + diffs / 2)
        self._centers = list(map (np.asarray, centers))

    # given construction properties of the Hist

    @property
    def n_dim (self):
        """The number of dimensions in the histogram."""
        return self._n_dim

    @property
    def bins (self):
        """A list of bin edge arrays, one for each dimension."""
        return self._bins

    @property
    def n_bins (self):
        """A list of the number of bins in each dimension."""
        return list(map (len, self.centers))

    @property
    def range (self):
        """A list of (min_value,max_value) tuples for each dimension."""
        return self._range

    @property
    def values (self):
        """An nD array of bin values (sum of weights in each bin)."""
        return self._values

    @property
    def errors (self):
        """An nD array of bin errors (sqrt(sum(squares of weights))) in each
        bin."""
        return self._errors

    @property
    def data (self):
        """The data used to construct the histogram (if given upon
        construction)."""
        return self._data

    @property
    def weights (self):
        """The weights used to construct the histogram (if given upon
        construction)."""
        return self._weights

    # inferred construction properties of the Hist

    @property
    def log (self):
        """A list of bools describing whether each dimenion is binned linearly
        or logarithmically."""
        if self._log is None:
            self._determine_log ()
        return self._log

    @property
    def centers (self):
        """A list of bin center arrays for each dimension."""
        if self._centers is None:
            self._determine_centers ()
        return self._centers

    @property
    def widths (self):
        """A list of bin width arrays for each dimension."""
        return list(map (np.diff, self.bins))

    @property
    def volumes (self):
        """An nD array of bin volumes (product of bin widths in
        each dimension)."""
        if self.n_dim == 0:
            return 0
        elif self.n_dim == 1:
            return np.diff (self.bins[0])
        else:
            return np.prod (np.meshgrid (*map (np.diff, self.bins)), axis=0).T

    # bin-data access

    def index (self, x, axis=0):
        """The bin index for value ``x`` in dimension ``axis``."""
        def digitize (b, a):
            return np.searchsorted (a, b, side='r')
        shape = np.shape (x)
        last_mask = x == self.range[axis][1]
        last_i = self.n_bins[axis] - 1
        if shape:
            out = (digitize (x.ravel(), self.bins[axis]) - 1).reshape (shape)
            out[last_mask] = last_i
            return out
        else:
            if last_mask:
                return last_i
            else:
                return int (digitize ([x], self.bins[axis]) - 1)

    def indices (self, *xs):
        """Get the indices for the specified coordinates."""
        if len (xs) != self.n_dim:
            raise TypeError ('`n_dim` coordinates required')
        return tuple (self.index (x, i) for (i,x) in enumerate (xs))

    def ravel_indices (self, *xs):
        """Get the indices into values.ravel() for the specified coordinates.
        
        Index of -1 indicates out-of-bounds
        """
        indices = self.indices (*xs)
        out_indices = np.zeros (len (indices[0]), dtype=int)
        step = 1
        bad_mask = np.zeros (len (indices[0]), dtype=bool)
        for axis in xrange (self.n_dim - 1, -1, -1):
            bad_mask |= ~ ( (self.range[axis][0] <= xs[axis])
                             & (xs[axis] < self.range[axis][1]))
            out_indices += step * indices[axis]
            step *= self.n_bins[axis]
        out_indices[bad_mask] = -1
        return out_indices

    def get_value (self, *xs):
        """Get the counts value at the specified coordinates."""
        return self.values[self.indices (*xs)]

    def get_values (self, *xs):
        """Get the counts values at the specified lists of coordinates."""
        indices = tuple ([self.index (x,i) for (i,x) in enumerate (xs)])
        return self.values[indices]

    def __call__ (self, *xs):
        return self.get_values (*xs)

    def eval (self, *a, **kw):
        kw.setdefault ('ndim', self.n_dim)
        kw.setdefault ('range', self.range)
        return hist_from_eval (self.get_values, *a, **kw)

    def get_error (self, *xs):
        """Get the error value at the specified coordinates."""
        return self.errors[self.indices (*xs)]

    def get_errors (self, *xs):
        """Get the error values at the specified lists of coordinates."""
        return np.array ([self.get_error (*x) for x in zip (*xs)])

    def transform_bins (self, f, axes=[-1]):
        bins = [
            f (self.bins[axis])
            if (axis in axes) or (axis - self.n_dim in axes)
            else self.bins[axis]
            for axis in xrange (self.n_dim)]
        return Hist (bins, self.values, self.errors)

    # sampling

    def sample (self, n_samples=1, *values, **kw):
        """Draw n samples from the data.

        :type   n_samples: int
        :param  n_samples: the number of samples

        Any given values select bins in the first len(values) dimensions, such
        that sampling is done only from the remaining dimensions.

        :return: tuple of arrays of length n_dim
        """
        seed = kw.get('seed', np.random.seed())
        random = kw.get('random', np.random.RandomState(seed))
        if values:
            return self[values].sample (n_samples)
        cdf = np.cumsum (self.values.ravel ()) / self.values.sum ()
        dice = random.uniform (0, 1, n_samples)
        dice_bins = np.searchsorted (cdf, dice)
        indices = np.unravel_index (dice_bins, self.values.shape)
        lefts = [self.bins[i][indices[i]] for i in xrange (self.n_dim)]
        rights = [self.bins[i][indices[i] + 1] for i in xrange (self.n_dim)]
        outs = [left * ((right / left) ** random.uniform (0, 1, n_samples))
                if self.log[i] else
                left + (right - left) * random.uniform (0, 1, n_samples)
                for (i, left,right)
                in izip (xrange (self.n_dim), lefts, rights)]
        return outs


    # axis-wise operations

    def cumsum (self, axes=[-1], normalize=False):
        """
        Calculate the cumulative sum along specified axes (in order).

        """
        values = self.values.copy ()
        axes = np.atleast_1d(axes)
        sum_axes = [self.n_dim + i if i < 0 else i for i in axes]
        for axis in sum_axes:
            values = values.cumsum (axis=axis)
        if normalize:
            values = values / self.sum(axes).values
        return Hist (self.bins, values)

    def sum (self, axes=None, integrate=False):
        """Project the histogram onto a subset of its dimensions by summing
        over ``axes``.

        :type   axes: sequence of int
        :param  axes: the axes along which to sum and thus the dimensions
            no longer present in the resulting Hist.

        :type   integrate: bool
        :param  integrate: if True, evaluate sum of (value * width) rather than
            just value.

        :return: :class:`Hist`
        """
        if axes is None:
            axes = list(range (self.n_dim))
        axes = np.atleast_1d(axes)

        # get sum and keep axes
        sum_axes = sorted ([self.n_dim + i if i < 0 else i for i in axes])
        keep_axes = [i for i in xrange (self.n_dim) if i not in sum_axes]

        bins = [self.bins[i] for i in xrange (self.n_dim) if i not in sum_axes]

        for axis in sum_axes:
            if axis not in xrange (self.n_dim):
                raise ValueError ('histogram has no axis {0}'.format (axis))

        # order new arrays so sum axes are last
        axes_order = keep_axes + sum_axes
        values = reindex (self.values.copy (), axes_order)
        if self.errors is not None:
            sq_errors = reindex (self.errors**2, axes_order)
        else:
            sq_errors = None

        # sum over last axes until done summing
        for axis in reversed (sum_axes):
            axis_bins = self.bins[axis]
            if integrate:
                values *= np.diff (axis_bins)
                if sq_errors is not None:
                    sq_errors *= np.diff (axis_bins)
            values = values.sum (axis=-1)
            if sq_errors is not None:
                sq_errors = sq_errors.sum (axis=-1)

        if self.errors is not None:
            errors = np.sqrt (sq_errors)
        else:
            errors = None
        return Hist (bins, values, errors)

    def integrate (self, axes=None):
        """Project the histogram onto a subset of its dimensions by integrating
        over ``axes``.

        :type   axes: sequence of int
        :param  axes: the axes along which to integrate and thus the dimensions
            no longer present in the resulting Hist.

        :return: :class:`Hist`
        """
        return self.sum (axes=axes, integrate=True)

    def project (self, axes=[-1], integrate=False):
        """Project the histogram onto a subset of its dimensions by summing
        over axes other than those listed in ``axes``.

        :type   axes: sequence of int
        :param  axes: the axes along which NOT to sum or integrate, and thus the
            dimensions no longer present in the resulting Hist.

        :return: :class:`Hist`
        """
        axes = [self.n_dim + i if i < 0 else i for i in axes]
        return self.sum ([i for i in xrange (self.n_dim) if i not in axes],
                         integrate=integrate)

    def contain (self, axis, frac=1 - 2 * stats.norm.sf (1)):
        """Project the histogram onto a subset of its dimensions by taking the
        containment interval along ``axis``.

        :type   axis: int
        :param  axis: the axis along which to measure containment and thus the
            dimensions no longer present in the resulting Hist.

        :type   frac: float
        :param  frac: the containment interval, measured from
            ``self.range[axis][0]`` and moving to the "right"

        :return: :class:`Hist`
        """
        if axis < 0:
            axis = len (self.bins) + axis

        def weighted_containment (a, w):
            as_ws = np.array (sorted (zip (a, w))).T
            if np.sum (w) == 0:
                m = np.nan
            else:
                m = as_ws[0][as_ws[1].cumsum () >= frac * w.sum ()][0]
            return m

        def apply_weighted_containment (w):
            return weighted_containment (self.centers[axis], w)

        values = np.apply_along_axis (apply_weighted_containment, axis, self.values)
        bins = [self.bins[i] for i in xrange (self.n_dim) if i != axis]
        return Hist (bins, values)

    def contain_project (self, axis,
                         frac=1 - 2 * stats.norm.sf (1),
                         n_sigma=None,
                         ):
        """
        Project the histogram taking median along one dimension, with errorbars
        reflecting the eliminated axis.

        :type   axis: int
        :param  axis: the axis along which to measure containment and thus the
            dimensions no longer present in the resulting Hist.

        :type   frac: float
        :param  frac: the containment fraction, which will be centered on the
            median value along the given axis

        :type   n_sigma: float
        :param  n_sigma: the containment fraction, specified as a number of
            sigmas

        :return: :class:`Hist`

        If given, ``n_sigma`` overrides ``frac``.
        """
        out = self.median (axis)
        f = (1 - frac) / 2. if n_sigma is None else stats.norm.sf (n_sigma)
        h_low = self.contain (axis, f)
        h_high = self.contain (axis, 1 - f)
        out._errors = np.vstack (((out - h_low).values, (h_high - out).values))
        return out

    def median (self, axis):
        """Project the histogram onto a subset of its dimensions by taking the
        median along ``axis``.

        :type   axis: int
        :param  axis: the axis along which to find the median and thus the
            dimensions no longer present in the resulting Hist.

        :return: :class:`Hist`
        """
        return self.contain (axis, .5)

    def normalize (self, axes=None, integrate=True, density=False):
        """Return a histogram normalized so the sums (or integrals) over the
        given axes are unity.

        :type   axes: sequence of int, optional
        :param  axes: the axes that will sum (or integrate) to unity for the
            normalized histogram

        :type   integrate: bool
        :param  integrate: if True, normalize so the integral is unity;
            otherwise, normalize so the sum is unity

        :type   density: bool
        :param  density: if True, normalize so the integral is unity, but
            *as though* the binning were linspaced, even if it is actually not.
            This option supersedes the ``integrate`` argument.

        :return: :class:`Hist`

        The norm is found by summing over all axes other than the ones
        specified, or by summing over all axes if ``axis`` is not given.  Note
        that setting ``density=True`` should obtain equivalent behavior to
        ``numpy.histogram(..., density=True)``.
        """
        if axes is None:
            axes = list(range (self.n_dim))
        axes = np.atleast_1d (axes)
        axes = sorted ([self.n_dim + i if i < 0 else i for i in axes])
        for axis in axes:
            if axis not in xrange (self.n_dim):
                raise ValueError ('histogram has no axis {0}'.format (axis))
        keep_axes = axes
        lose_axes = [i for i in xrange (self.n_dim) if i not in axes]

        if density:
            integrate = False
        hsum = 1. * self.sum (keep_axes, integrate=integrate)
        hother = self.sum (lose_axes)

        reorder = keep_axes + lose_axes

        if density:
            volumes = hother.volumes
        else:
            volumes = 1

        with warnings.catch_warnings ():
            warnings.filterwarnings ('ignore')
            values = unreindex (
                reindex (self.values, reorder) / hsum.values, reorder
            ) / volumes

        if self.errors is not None:
            with warnings.catch_warnings ():
                warnings.filterwarnings ('ignore')
                errors = unreindex (
                    reindex (self.errors, reorder) / hsum.values, reorder
                ) / volumes
        else:
            errors = None
        return Hist (self.bins, values, errors)

    def rebin (self, axis, bins, tol=1e-4):
        """Produce coarser binning along the given axis.

        :type   axis: int
        :param  axis: the axis along which to rebin

        :type   bins: sequence
        :param  bins: the new bin edges

        :type   tol: float
        :param  tol: the absolute error between the given ``bins`` and ones
            found in the original histogram

        :return: :class:`Hist`

        Each bin in bins should be contained in the existing bins, and the
        endpoints should match.  Tolerance for bin agreement is given as an
        absolute error by `tol`.
        """
        # validate bins
        oldbins = self.bins[axis]
        newbins = np.copy (np.sort (bins))
        for (i, b) in enumerate (newbins):
            misses = np.abs (b - oldbins)
            j = np.argmin (misses)
            closest = np.min (misses)
            if closest > tol:
                raise ValueError (
                        '{0} is not among current bin edges'.format (b))
            newbins[i] = oldbins[j]
        if newbins[0] != oldbins[0]:
            raise ValueError (
                    'binning startpoint should match')
        if newbins[-1] != oldbins[-1]:
            raise ValueError (
                    'binning endpoint should match')

        n_newbins = len (newbins) - 1
        newbin_indices = np.digitize (oldbins, newbins)[:-1] - 1

        def revalue_one (a):
            return [np.sum (a[newbin_indices==i])
                    for i in xrange (n_newbins)]
        def reerror_one (a):
            return [np.sqrt (np.sum (a[newbin_indices==i]**2))
                    for i in xrange (n_newbins)]

        values = np.apply_along_axis (revalue_one, axis, self.values)
        if self.errors is not None:
            errors = np.apply_along_axis (reerror_one, axis, self.errors)
        else:
            errors = None

        bins = [self.bins[i] if i != axis else newbins
                for i in xrange (self.n_dim)]

        return Hist (bins, values, errors)

    def __getitem__ (self, sli):

        def intypes (obj, types):
            for t in types:
                if isinstance (obj, t):
                    return True
            return False

        # handle non-tuple
        if not isinstance (sli, tuple):
            return self[sli,]

        # check against excess dimensions
        if len (sli) > self.n_dim:
            raise TypeError ('too many dimensions for this Hist')

        # TODO: do we really need a copy here? seems like we shouldn't
        #subh = copy.deepcopy (self)
        subh = self

        subsli = sli[0]

        n_reductions = 0

        for dim, subsli in enumerate (sli):
            # could be a number/sequence, or slice
            if intypes (subsli, (int, float, list, np.float32, np.ndarray)):
                # cannot subscript a 1D Hist
                if self.n_dim == 1:
                    raise TypeError ('cannot reduce dimensionality of 1D Hist')
                index = self.index (subsli, dim)
                subh = subh.get_slice (index, dim - n_reductions)
            else:
                if subsli.start is None:
                    start = 0
                else:
                    start = self.index (subsli.start, dim)
                if subsli.stop is None:
                    stop = len (self.bins[dim])
                else:
                    stop = self.index (subsli.stop, dim)
                    if subsli.stop not in self.bins[dim]:
                        stop += 1
                indsli = slice (start, stop)
                subh = subh.get_slice (indsli, dim - n_reductions)
            if subh.n_dim < self.n_dim:
                n_reductions += 1
        return subh

    def get_slice (self, index, axis=0):
        slices = []
        bins = []
        log = []
        for dim in xrange (self.n_dim):
            if dim == axis:
                slices.append (index)
                if not isinstance (index, int):
                    if isinstance (index, slice):
                        if index.stop is not None:
                            bin_index = slice (
                                    index.start, index.stop + 1, index.step)
                        else:
                            bin_index = index
                    bins.append (self.bins[axis][bin_index])
                    log.append (self.log[axis])
            else:
                n_axis_bins = len (self.bins[dim])
                slices.append (slice (0, n_axis_bins))
                bins.append (self.bins[dim])
                log.append (self.log[dim])
        values = self.values[tuple (slices)]
        if self.errors is not None:
            errors = self.errors[tuple (slices)]
        else:
            errors = None
        return Hist (bins, values, errors)


    @property
    def T (self):
        """A transposed version of the Hist."""
        bins = self.bins[::-1]
        values = self.values.T
        if self.errors is not None:
            errors = self.errors.T
        else:
            errors = None
        return Hist (bins, values, errors)

    # fitting

    def curve_fit (self, func, **kw):
        """
        Fit a function to the histogram.

        :type   func: function
        :param  func: model function as in scipy.optimize.curve_fit().

        :return: popt, pcov as in scipy.optimize.curve_fit()

        This function unravels the values and bin centers into ``n_dim`` 1D
        arrays which are then passed, along with any keyword arguments, to
        scipy.optimize.curve_fit().
        """
        ydata = np.ravel (self.values)
        errors = self.errors
        if errors is not None and not np.all (~np.isfinite (errors)):
            #sigma = np.ravel (self.errors)
            if len (errors.shape) == len (self.values.shape) + 1 \
               and errors.shape[0] == 2:
                errors = np.mean (errors, axis=0)
            sigma = np.ravel (errors)
        else:
            sigma = np.ones (ydata.shape)
        mask = (0 < sigma) & (sigma < np.inf)
        ydata = ydata[mask]
        sigma = sigma[mask]
        if self.n_dim >= 2:
            xdata = np.array (list(map (np.ravel, np.meshgrid (*self.centers, indexing='ij'))))
            xdata = np.array ([xd[mask] for xd in xdata])
        else:
            xdata = self.centers[0][mask]
        # special case for scipy rvs functions
        if hasattr(func, '__self__'):
            shapes = func.__self__.shapes
            if shapes:
                n_args = 2 + len(shapes.split(','))
            else:
                n_args = 2
            funcs = {
                1: (lambda x, a: func(x, a)),
                2: (lambda x, a,b: func(x, a,b)),
                3: (lambda x, a,b,c: func(x, a,b,c)),
                4: (lambda x, a,b,c,d: func(x, a,b,c,d)),
                5: (lambda x, a,b,c,d,e: func(x, a,b,c,d,e)),
            }
            f = funcs[n_args]
        else:
            f = func
        return optimize.curve_fit (f, xdata, ydata, sigma=sigma, **kw)

    def spline_fit (self, log=False, floor=None, *a, **kw):
        """
        Get a scipy spline fit to the histogram.

        :type   log: bool
        :param  log: whether to fit in log-value or linear-value

        :type   floor: float
        :param  floor: 10**floor is the arbitrary small number to stand in for
            zeros when ``log`` is true.  If not set, log10 of 0.1 times the
            smallest nonzero value will be used.

        :return: :class:`SplineFit`

        This method produces a spline fit to the histogram values for 1D or 2D
        histograms.  The domain of the fitted spline will be the same as that
        of the histogram.

        """
        errors = self.errors
        #if errors is not None and not np.all (~np.isfinite (errors)):
        #    errors = np.mean (np.atleast_2d (errors), axis=0)
        if log and floor is None:
            floor = np.log (0.1 * self.values[self.values > 0].min ())
        if self.n_dim == 1:
            x = np.r_[self.bins[0][0], self.centers[0], self.bins[0][-1]]
            if self.log[0]:
                x = np.log10 (x)
            y = breakout_1d (self.values)
            err = breakout_1d (errors)
            if np.all (err == 0) or not np.any (np.isfinite (err)):
                err = np.ones_like (err)
            if log:
                with warnings.catch_warnings ():
                    warnings.filterwarnings ('ignore')
                    logy = np.log (y)
                    err /= y
                logy[y == 0] = floor
                y = logy
            with warnings.catch_warnings ():
                warnings.filterwarnings ('ignore')
                mask = err > 0
            spl = interpolate.UnivariateSpline (
                x[mask], y[mask], 1 / err[mask], *a, **kw
            )
            return SplineFit (self, spl, self.log, log, floor)
        elif self.n_dim == 2:
            x = np.r_[self.bins[0][0], self.centers[0], self.bins[0][-1]]
            if self.log[0]:
                x = np.log10 (x)
            y = np.r_[self.bins[1][0], self.centers[1], self.bins[1][-1]]
            if self.log[1]:
                y = np.log10 (y)
            z = breakout_2d (self.values)
            if log:
                with warnings.catch_warnings ():
                    warnings.filterwarnings ('ignore')
                    logz = np.log (z)
                logz[z == 0] = floor
                z = logz
            s = interpolate.RectBivariateSpline (
                x, y, z, *a, **kw
            )
            return SplineFit (self, s, self.log, log, floor)
        else:
            args = [
                np.r_[self.bins[i][0], self.centers[i], self.bins[i][-1]]
                for i in xrange (self.n_dim)
            ]
            for i in xrange (self.n_dim):
                if self.log[i]:
                    args[i] = np.log10 (args[i])
            args = list (np.meshgrid (*args, indexing='ij'))
            V = np.pad (self.values, 1, mode='edge')
            if log:
                V = np.log (V)
            args.append (V)
            args += a
            s = interpolate.Rbf (*args, **kw)
            return SplineFit (self, s, self.log, log, floor)
        #else:
        #    raise TypeError (
        #        'cannot fit spline to {0} dimensional Hist'.format (
        #            self.n_dim
        #        )
        #    )

    # smoothing
    def gaussian_filter (self, *a, **kw):
        """
        Smooth both values and errors with ``scipy.ndimage.gaussian_filter()``.
        """
        values = ndimage.gaussian_filter (self.values, *a, **kw)
        errors = ndimage.gaussian_filter (self.errors, *a, **kw)
        return Hist (self.bins, values, errors)

    # smoothing
    def gaussian_filter1d (self, *a, **kw):
        """
        Smooth both values and errors with ``scipy.ndimage.gaussian_filter1d()``.
        """
        values = ndimage.gaussian_filter1d (self.values, *a, **kw)
        errors = ndimage.gaussian_filter1d (self.errors, *a, **kw)
        return Hist (self.bins, values, errors)

    # Hist-Hist operations

    def matches (self, other):
        """True if self and other have the same binning."""
        #return True
        for (sbins, obins) in izip (self.bins, other.bins):
            n = len (sbins)
            if len (obins) != n:
                return False
            deltas = np.abs (sbins - obins)
            sdiff, odiff = np.diff (sbins), np.diff (obins)

            diff = .5 * (sdiff + odiff)
            diff_deltas = np.abs (sdiff - odiff)
            ratios = np.maximum (deltas[:-1] / diff, deltas[1:] / diff)
            if np.max (ratios) > 1e-5:
                return False
        return True

    def assert_match (self, other):
        if not self.matches (other):
            raise ValueError ('histograms do not have matching binning')

    def __add__ (a, b):
        """
        Add two histograms with matching bins, or a scalar and a histogram.
        """
        if isinstance (b, Hist):
            a.assert_match (b)
            values = a.values + b.values
            if a.errors is not None and b.errors is not None:
                errors = np.sqrt (a.errors**2 + b.errors**2)
            else:
                errors = None
            return Hist (a.bins, values, errors)
        else:
            return Hist (a.bins, a.values + b, a.errors)

    def __radd__ (a, b):
        """Add a scalar and a histogram."""
        return a + b

    def __sub__ (a, b):
        """
        Subtract two histograms with matching bins, or a scalar from a
        histogram.
        """
        if isinstance (b, Hist):
            a.assert_match (b)
            values = a.values - b.values
            if a.errors is not None and b.errors is not None:
                errors = np.sqrt (a.errors**2 + b.errors**2)
            else:
                errors = None
            return Hist (a.bins, values, errors)
        else:
            return Hist (a.bins, a.values - b, a.errors)

    def __rsub__ (a, b):
        """
        Subtract a histogram from a scalar, returning a new histogram.
        """
        return (-a) + b

    def __mul__ (a, b):
        """
        Multiply two histograms with matching bins, or a scalar and a histogram.
        """
        if isinstance (b, Hist):
            a.assert_match (b)
            values = a.values * b.values
            if a.errors is not None and b.errors is not None:
                errors = values * np.sqrt (
                    (a.errors / a.values)**2 + (b.errors / b.values)**2)
            else:
                errors = None
            return Hist (a.bins, values, errors)
        else:
            return Hist (a.bins, b * a.values, abs (b) * a.errors)

    def __rmul__ (self, scalar):
        """
        Multiply a scalar and a histogram.
        """
        return self * scalar

    def __div__ (a, b):
        """
        Divide two histograms with matching bins, or histogram by a scalar.
        """
        if isinstance (b, Hist):
            a.assert_match (b)
            with warnings.catch_warnings ():
                warnings.filterwarnings ('ignore')
                values = a.values / b.values
                if a.errors is not None and b.errors is not None:
                    errors = values * np.sqrt (
                        (a.errors / a.values)**2 + (b.errors / b.values)**2)
                else:
                    errors = None
            return Hist (a.bins, values, errors)
        else:
            b = 1.0 * b
            return Hist (a.bins, a.values / b, a.errors / b)

    __truediv__ = __div__

    def __pow__ (a, b):
        """
        Raise to a histogram with matching bins, or a scalar power.
        """
        if isinstance (b, Hist):
            a.assert_match (b)
            values = a.values ** b.values
            if a.errors is not None and b.errors is not None:
                with warnings.catch_warnings ():
                    warnings.filterwarnings ('ignore')
                    errorsa = values * b.values * a.errors / a.values
                    errorsb = values * np.log (a.values) * b.errors
                errors = np.sqrt (errorsa**2 + errorsb**2)
            else:
                errors = None
            return Hist (a.bins, values, errors)
        else:
            values = a.values**b
            with warnings.catch_warnings ():
                warnings.filterwarnings ('ignore')
                errors = values * b * a.errors / a.values
            return Hist (a.bins, values, errors)

    def __rpow__ (self, scalar):
        """
        Raise a scalar to powers given by a histogram.
        """
        values = scalar**self.values
        errors = values * np.log (scalar) * self.errors
        return Hist (self.bins, values, errors)


    def efficiency (self, base_hist):
        """Get an efficiency Hist for this Hist divided by base_hist.

        :type   base_hist: :class:`Hist`
        :param  base_hist: The base histogram, of which this one should be a
            subset.

        This method differs from __div__ in the way that errors are propagated.

        """
        keep = self
        orig = base_hist
        rej = orig - keep
        use_errors = self.errors is not None and base_hist.errors is not None
        if use_errors:
            rej._errors = np.sqrt (base_hist.errors**2 - self.errors**2)

        eff = keep / orig
        nkeep = keep.values
        nrej = rej.values
        if use_errors:
            with warnings.catch_warnings ():
                warnings.filterwarnings ('ignore')
                eff._errors = np.sqrt (
                        (nrej / (nkeep+nrej)**2 * keep.errors)**2
                        + (nkeep / (nkeep+nrej)**2 * rej.errors)**2 )
        return eff

    # self operations

    def __neg__ (self):
        """
        Get a histogram with opposite-signed values.
        """
        return Hist (self.bins, -self.values, self.errors)

    def abs (self):
        return Hist (self.bins, np.abs (self.values), self.errors)

    def exp (self):
        values = np.exp (self.values)
        errors = values * self.errors
        return Hist (self.bins, values, errors)

    def log_base (self, base):
        mask = self.values > 0
        values = np.nan * np.ones_like (self.values)
        errors = np.nan * np.ones_like (self.values)
        lb = np.log (base)
        values[mask] = np.log (self.values[mask]) / lb
        errors[mask] = self.errors[mask] / (self.values[mask] * lb)
        return Hist (self.bins, values, errors)

    def ln (self):
        return self.log_base (np.e)

    def log2 (self):
        return self.log_base (2)

    def log10 (self):
        return self.log_base (10)

    def log1p (self):
        values = np.log1p (self.values) / lb
        # TODO: check error propagation here
        errors = self.errors / (self.values * lb)
        return Hist (self.bins, values, errors)

    def sqrt (self):
        return self ** 0.5

    # pretty

    def __repr__ (self):
        out = ['Hist(']
        # binning
        for dim in xrange (self.n_dim):
            out.append ('{} bins in [{},{}], '.format (
                self.n_bins[dim], self.range[dim][0], self.range[dim][1]
            ))
        out.append ('with {} {}'.format (
            'sum' if self.n_dim else 'value',
            self.values.sum()))
        if self.n_dim:
            out.append (', {} empty bins,'.format (np.sum (self.values == 0)))
            out.append (' and {} non-finite values'.format (
                np.sum (~np.isfinite (self.values))))
        out.append(')')
        return ''.join (out)


def _regularize_data (data):
    # validate data
    try:
        try:
            len (data[0])
        except:
            data = (data,)
        n_dims = len (data)
        n_samples = len (data[0])
        for i in xrange (1, n_dims):
            assert len (data[i]) == n_samples
    except:
        raise ValueError (
                'could not interpret `data` (check for matching array lengths)')
    return data, n_dims, n_samples

def hist (data, weights=None,
          bins=10, range=None, log=False,
          round_int_bins=False,
          keep_data=False):

    """Factory function for :class:`Hist`.

    :type   data: array-like or tuple of array-like
    :param  data: the source data for the histogram: a tuple of array-like
        each of length (number of samples), or just a single array of that
        length

    :type   weights: array-like
    :param  weights: the weights of the source datapoints

    :type   bins: sequence or int, optional
    :param  bins: a numpy.histogramdd() bins specification

    :type   range: sequence, optional
    :param  range: a numpy.histogramdd() range specification

    :type   log: sequence or bool
    :param  log: if `bins` gives bin counts rather than edges, log=True causes
        logarithmic bin edges to be chosen (can be given per-dimension)

    :type   keep_data: bool
    :param  keep_data: whether to keep the data and weights for the histogram

    :return: the :class:`Hist`
    """

    data, n_dims, n_samples = _regularize_data (data)

    data = tuple (np.asarray (a) for a in data)

    # regularize bins argument
    if bins is not None:
        if isinstance (bins, tuple):
            bins = list (bins)
        else:
            try:
                len (bins[0])
            except:
                bins = n_dims * [bins]

    # regularize range argument
    if range is None:
        range = []
        for b in bins:
            if hasattr (b, '__iter__'):
                range.append ((np.min (b), np.max (b)))
            else:
                range.append (None)
    if isinstance (range, tuple):
        if len (range) == 2 and not (
                isinstance (range[0], tuple) or isinstance (range[1], tuple)):
            range = n_dims * [range]

    # regularize log argument
    try:
        len (log)
    except:
        log = n_dims * [log]

    # ensure weights
    if weights is None:
        weights = np.ones (n_samples)

    # ignore non-finite values
    good_idx = np.isfinite (weights)
    for dim in xrange (n_dims):
        good_idx *= np.isfinite (data[dim])

    # ensure ranges
    if np.sum (good_idx) == 0:
        full_range = range
    else:
        full_range = [
                (a[good_idx].min(),
                    a[good_idx].max())
                for a in data]
        log_range = [
                (a[good_idx * (a > 0)].min(),
                    a[good_idx * (a > 0)].max())
                if np.sum (good_idx * (a > 0))
                else (np.nan, np.nan)
                for a in data]
    np_range = [r if r not in (None, (None,None)) else fr
            for (r, fr) in zip (range, full_range)]

    # get log bins if requested, but edges not specified
    for dim in xrange (n_dims):
        if log[dim]:
            try:
                len (bins[dim])
            except:
                if range and range[dim]:
                    a, b = list(map (np.log10, range[dim]))
                else:
                    a, b = np.log10 (log_range[dim])
                bins[dim] = np.logspace (a, b, bins[dim] + 1)
                np_range[dim] = (a,b)

    # validate ranges
    for dim in xrange (n_dims):
        if sum (~np.isfinite (np_range[dim])):
            raise ValueError (
                'NaN found in range for dimension {0}'.format (dim))


    data = tuple (a[good_idx] for a in data)
    weights = weights[good_idx]

    if bins:
        if len (data) != len (bins):
            raise ValueError (
                'data dimensions ({}) must match number of bin ({})'
                ' specifications'.format (len (data), len (bins)))
    if len (data) != len (range):
        raise ValueError (
            'data dimensions must match number of range specification')

    # deal with int-valued axes
    if round_int_bins:
        for dim in xrange (n_dims):
            if 'i' in data[dim].dtype.descr[0][1]:
                xmin, xmax = orig_xmin, orig_xmax = orig_range = np_range[dim]
                orig_n = bins[dim]
                n = min (orig_n, xmax - xmin)
                dx = lambda xmin, xmax, n: 1. * (xmax - xmin) / n
                orig_dx = dx (orig_xmin, orig_xmax, orig_n)
                while dx (xmin, xmax, n) % 1 > 1e-5:
                    xmax += 1
                np_range[dim] = xmin, xmax
                bins[dim] = int ((xmax - xmin) / dx (xmin, xmax, n))



    # build histogram
    values, edges = np.histogramdd (data, weights=weights,
            bins=bins, range=np_range)
    errors = np.sqrt (np.histogramdd (data, weights=weights**2,
            bins=bins, range=np_range)[0])

    others = dict (data=data, weights=weights) if keep_data else {}

    return Hist (edges, values, errors=errors, **others)

def hist_like (other, data, weights=None, keep_data=False):
    """Create a :class:`Hist` using the same binning as ``other``.

    :type   other: :class:`Hist`
    :param  other: the other Hist

    :type   data: array-like
    :param  data: the data to be histogrammed.  For multidimensional
        histograms, ``data`` will be transposed for input into
        ``np.histogramdd()``.

    :type   weights: array-like
    :param  weights: the weights

    :return: the :class:`Hist`
    """
    if len (np.shape (data)) > 1:
        data = np.transpose (data)
    bins = other.bins
    values, bins = np.histogramdd (data, bins=bins, weights=weights)
    if weights is not None:
        sq_errors, bins = np.histogramdd (data, bins=bins, weights=weights**2)
        errors = np.sqrt (sq_errors)
    else:
        errors = np.sqrt (values)
    others = dict (data=data, weights=weights) if keep_data else {}
    return Hist (bins, values, errors, **others)

def hist_like_indices (other, ravel_indices, weights=None):
    """
    Create a :class:`Hist` using pre-computed per-sample indices from ``other``.

    :type   other: :class:`Hist`
    :param  other: the pre-existing histogram which defines the binning

    :type   ravel_indices: array-like
    :param  ravel_indices: result of a :meth:`Hist.ravel_indices` call for the
        samples.

    :type   weights: array-like
    :param  weights: the weights of the source datapoints
    """
    bins = copy.deepcopy (other.bins)
    if weights is None:
        weights = np.ones (ravel_indices.size)
    mask = ravel_indices >= 0
    values = np.bincount (
        ravel_indices[mask], weights[mask], minlength=other.values.size)
    errors = np.sqrt (np.bincount (
        ravel_indices[mask], weights[mask]**2, minlength=other.values.size))
    shape = other.values.shape
    return Hist (bins, values.reshape (shape), errors.reshape (shape))


def hist_direct (data, weights=None, bins=None, range=None, keep_data=False):
    """Fast factory function for :class:`Hist`.

    :type   data: array-like
    :param  data: the data to be histogrammed.  For multidimensional
        histograms, ``data`` will be transposed for input into
        ``np.histogramdd()``.

    :type   weights: array-like
    :param  weights: the weights

    :type   bins: sequence or int, optional
    :param  bins: a ``np.histogramdd()`` bins specification

    :type   range: sequence, optional
    :param  bins: a ``np.histogramdd()`` range specification

    :type   keep_data: bool
    :param  keep_data: whether to keep the data and weights for the histogram

    :return: the :class:`Hist`

    This method creates a :class:`Hist` by calling ``np.histogramdd()`` as
    directly as possible.  Note that this requires a slightly more constrained
    format for ``data`` compared with :meth:`hist`.
    """
    if len (np.shape (data)) > 1:
        data = np.transpose (data)
    values, bins = np.histogramdd (
        data, bins=bins, range=range, weights=weights)
    if weights is not None:
        sq_errors, bins = np.histogramdd (
            data, bins=bins, range=range, weights=weights**2)
        errors = np.sqrt (sq_errors)
    else:
        errors = np.sqrt (values)
    others = dict (data=data, weights=weights) if keep_data else {}
    return Hist (bins, values, errors, **others)

def hist_from_function (bins, func, *args, **kwargs):
    """
    [Deprecated] Create a :class:`Hist` by evaluating ``func`` at the centers
    of specified ``bins``.

    :type   bins: list of np.array
    :param  bins: the bin edges

    :type   func: function
    :param  func: the function to evaluate. it should return floats and respect
        numpy broadcasting rules.  if ``splat`` is true, it should take
        ``len(bins)`` (that is, n_dim) arguments; otherwise, it should accept
        a single argument containing all values of all independent variables

    :type   err_func: function
    :param  err_func: if given, the function that returns the uncertainty at
        each part of the parameter space

    :type   splat: bool
    :param  splat: determines the signature of ``func`` as described above
        (default: True)

    :return: the :class:`Hist`

    Any additional positional or keyword arguments are passed to ``func``.

    Note: This function is now deprecated; use :meth:`hist_from_eval` instead.
    """
    warnings.warn (
        'this function is deprecated; use hist_from_eval() instead',
        DeprecationWarning)
    err_func = kwargs.pop ('err_func', None)
    splat = kwargs.pop ('splat', True)
    bins = copy.deepcopy (bins)
    shape = tuple (len (b) - 1 for b in bins)
    dummy_hist = Hist (bins, np.zeros (shape))
    if len (bins) > 1:
        mesh_centers = list (np.meshgrid (*dummy_hist.centers, indexing='ij'))
    else:
        mesh_centers = [dummy_hist.centers[0]]
    if splat:
        pos_args = mesh_centers + list (args)
    else:
        pos_args = [np.array ([mesh_centers])] + list (args)
    values = func (*pos_args, **kwargs)
    if err_func is not None:
        errors = err_func (*pos_args, **kwargs)
    else:
        errors = None
    return Hist (bins, values, errors)

def hist_slide (Ns, data, weights=None,
                *args, **kwargs):
    """
    Construct a "histogram" from ``N`` partially-overlapping Hist's.

    :type   Ns: int or sequence of int
    :param  Ns: number of sliding iterations, optionally per-axis

    :type   data: array-like or tuple of array-like
    :param  data: the source data for the histogram: a tuple of array-like
        each of length (number of samples), or just a single array of that
        length

    :type   weights: array-like
    :param  weights: the weights of the source datapoints

    :type   indices: sequence of array-like
    :param  indices: ravel_indices for each individual histogram, obtained from
        a previous hist_slide call

    :type   get_indices: bool
    :param  get_indices: if True, obtain ravel_indices values for use in later
        hist_slide calls
    """
    indices = kwargs.pop ('indices', None)
    get_indices = kwargs.pop ('get_indices', False)
    out_indices = []
    # get main ordinary hist
    data, n_dims, n_samples = _regularize_data (data)
    if indices is None:
        h_main = hist (data, weights, *args, **kwargs)
    else:
        empty = [[1] for i in xrange (n_dims)]
        h_main = hist (empty, *args, **kwargs)
        h_main = hist_like_indices (h_main, indices[0], weights=weights)
    if get_indices:
        out_indices.append (h_main.ravel_indices (*data))
    orig_bins = h_main.bins
    centers = [ [c[-1]] for c in h_main.centers ]
    # determine output shape
    if isinstance (Ns, int):
        Ns = [Ns if h_main.n_bins[axis] >= 2 else 1 for axis in xrange (n_dims)]
    out_shape = tuple (
        (h_main.values.shape[i] - 1) * N + 1 if N >= 2 else h_main.values.shape[i]
        for (i, N) in enumerate (Ns)
    )
    # store values from h_main
    values = np.zeros (out_shape)
    errors = np.zeros (out_shape)
    slices = tuple (
        slice (0, out_shape[i], N)
        for (i, N) in enumerate (Ns)
    )
    values[slices] = h_main.values
    errors[slices] = h_main.errors
    slide_ranges = [
        np.arange (0, N)
        for (i,N) in enumerate (Ns)
    ]
    # slide bins to fill in remaining values
    for (i_step, i_slides) in enumerate (itertools.product (*slide_ranges)):
        bins = copy.deepcopy (orig_bins)
        lasts = []
        for axis in xrange (n_dims):
            b = bins[axis]
            lasts.append (out_shape[axis] - 1)
            if Ns[axis] == 1:
                continue
            if i_slides[axis] == 0:
                bins[axis] = b
                lasts[-1] += 1
            elif h_main.log[axis]:
                bins[axis] = np.exp (
                    np.log (b[:-1])
                    + 1.0 * i_slides[axis] / Ns[axis] * np.diff (np.log (b))
                )
            else:
                bins[axis] = (
                    b[:-1]
                    + 1.0 * i_slides[axis] / Ns[axis] * np.diff (b)
                )
        if indices is None:
            h = hist_direct (data, weights=weights, bins=bins)
        else:
            empty = [[1] for i in xrange (n_dims)]
            h = hist (empty, bins=bins)
            h = hist_like_indices (h, indices[i_step+1], weights=weights)
        if get_indices:
            out_indices.append (h.ravel_indices (*data))
        for axis in xrange (n_dims):
            if len (centers[axis]) < out_shape[axis] - 1:
                centers[axis] = np.unique (np.r_[centers[axis], h.centers[axis]])
        slices = tuple (
            slice (i_slide, lasts[i] if Ns[i] >= 2 else out_shape[i], Ns[i])
            for (i, i_slide) in enumerate (i_slides)
        )
        values[slices] = h.values
        errors[slices] = h.errors
    # get non-overlapping binning
    bins = []
    for axis in xrange (n_dims):
        if not Ns[axis] >= 2:
            bins.append (orig_bins[axis])
        else:
            c = centers[axis]
            if h_main.log[axis]:
                log_c = np.log (c)
                log_delta = np.diff (log_c)
                log_delta = np.r_[log_delta, log_delta[-1]]
                b = np.exp (np.r_[
                    log_c - .5 * log_delta, log_c[-1] + .5 * log_delta[-1]
                ])
            else:
                delta = np.diff (c)
                delta = np.r_[delta, delta[-1]]
                b = np.r_[c - .5 * delta, c[-1] + .5 * delta[-1]]
            b[[0,-1]] = orig_bins[axis][[0,-1]]
            bins.append (b)
    if get_indices:
        return Hist (bins, values, errors), out_indices
    else:
        return Hist (bins, values, errors)

def hist_bootstrap (N,
                    data, weights=None,
                    *args,
                    **kwargs):
    """
    Like ``hist()``, but for N iterations of sampling from ``data`` with
    replacement.

    :type   stacked: bool
    :param  stacked: if True, output is an ndim+1 dimensional Hist where the
        first dimension has n_bins=``N`` and values are sorted along this first
        dimension. otherwise output is ndim dimensional

    :type   errors: str
    :param  errors: 'original' to obtain errorbars from standard
        (non-bootstrapped) Hist; 'bootstrap' to obtain errors from containment
        interval along the Hist that would have been returned if
        ``stacked=True``.  default is 'bootstrap'

    :type   frac: float in [0,1]
    :param  frac: fraction of samples to use in each iteration. default is 1.0

    :type   slide_Ns: float or sequence of float
    :param  slide_Ns: if given, use hist_slide() instead of hist(), with the
        given number(s) of slided binnings

    """
    data, n_dims, n_samples = _regularize_data (data)
    stacked = kwargs.pop ('stacked', False)
    error_type = kwargs.pop ('errors', 'original')
    slide_Ns = kwargs.pop ('slide_Ns', 1)
    frac = kwargs.pop ('frac', 1)
    hs = []
    all_i = np.arange (N)
    for i in all_i:
        mask = np.random.randint (0, n_samples, int (frac * n_samples))
        data_i = tuple (d[mask] for d in data)
        weights_i = None if weights is None else weights[mask]
        hs.append (hist_slide (slide_Ns, data_i, weights_i, *args, **kwargs))
    all_bins = [np.arange (N)] + hs[0].bins
    shape = (N,) + tuple ([len (b) - 1 for b in all_bins[1:]])
    all_values = np.vstack ([h.values for h in hs]).reshape (shape)
    #errors = np.vstack ([h.errors for h in hs])
    all_values = np.sort (all_values, axis=0)
    all_errors = None
    h_stacked = Hist (all_bins, all_values, all_errors)
    if stacked:
        return h_stacked
    ilow, imed, ihigh = np.array (N * stats.norm.cdf ([-1,0,1]), dtype=int)
    values = all_values[imed]
    errors = np.vstack ((values - all_values[ilow], all_values[ihigh] - values))
    h = Hist (all_bins[1:], values, errors)
    if error_type == 'original':
        h_orig = hist_slide (slide_Ns, data, weights, *args, **kwargs)
        h._errors = h_orig.errors
        return h
    elif error_type == 'bootstrap':
        return h
    else:
        raise ValueError ('`errors` must be one of "original" or "bootstrap"')

def hist_from_eval (f, vectorize=True, err_f=None, ndim=None, **kwargs):
    """
    Create a :class:`Hist` by evaluating a function.

    :type   f: callable
    :param  f: the function to evaluate

    :type   ndim: int
    :param  ndim: number of arguments to function

    :type   vectorize: bool
    :param  vectorize: whether ``numpy.vectorize`` is needed to evaluate ``f``
        over many sets of values

    :type   err_f: callable
    :param  err_f: if given, function to evaluate to obtain "errors"

    All other keyword arguments define the binning the same as for
    :meth:`hist`.

    This function supersedes :meth:`hist_from_function`.
    """
    # infer number of arguments
    try:
        spec = inspect.getargspec (f)
        ismethod = inspect.ismethod (f)
    except:
        spec = inspect.getargspec (f.__call__)
        ismethod = inspect.ismethod (f.__call__)
    if ndim is None:
        ndim = len (spec.args) - len (spec.defaults or [])
        if ismethod:
            ndim -= 1
    # get empty histogram
    if ndim <= 0:
        raise ValueError ('could not determine valid ndim; obtained {}'.format (ndim))
    empty = [[] for i in xrange (ndim)]
    h = hist (empty, **kwargs)
    # vectorize if needed
    if vectorize:
        F = np.vectorize (f)
        if err_f is not None:
            err_F = np.vectorize (err_f)
        else:
            err_F = None
    else:
        F = f
        err_F = err_f
    # obtain values and possibly errors
    errors = np.zeros (h.values.shape)
    if ndim == 1:
        X = h.centers[0]
        values = F (X)
        if err_F is not None:
            values = err_F (X)
    else:
        X = np.meshgrid (*h.centers, indexing='ij')
        values = F (*map (np.ravel, X)).reshape (X[0].shape)
        if err_F is not None:
            values = err_F (*map (np.ravel, X)).reshape (X[0].shape)
    h._values = values
    h._errors = errors
    return h

def kde (data, weights=None, kernel=.01, normalize=True,
         indices=None, get_indices=False, **kw):
    data, n_dim, n_samples = _regularize_data (data)
    if indices is None:
        kw.setdefault ('bins', int (np.ceil (1000 / n_dim)))
        h = hist (data, weights=weights, **kw)
    else:
        empty = [[1] for i in xrange (n_dim)]
        h = hist (empty, **kw)
        h = hist_like_indices (h, indices, weights=weights)
    orig_kernel = kernel
    if not np.shape (kernel):
        kernel = h.n_dim * [kernel]

    kernel_bins = [
        kernel[axis] * h.n_bins[axis]
        #kernel[axis] / np.mean (np.diff (
        #    np.log10 (h.bins[axis]) if h.log[axis] else h.bins[axis]))
        for axis in xrange (h.n_dim) ]
    h = h.gaussian_filter (kernel_bins)
    if normalize:
        h = h.normalize (axes=None if normalize is True else normalize)
    h.kernel, h.kernel_bins = orig_kernel, kernel_bins
    if get_indices:
        return h, h.ravel_indices (*data)
    else:
        return h

def profile (f, data, weights, **kw):
    # mean: weighted by value, divided by pure counting
    if f is kde:
        kw.setdefault ('normalize', False)
    h_count = f (data, **kw)
    h_weight = f (data, weights=weights, **kw)
    h_mean = h_weight / h_count
    # error: stderr = rms deviation
    dev = weights - h_mean (data)
    h_dev2 = f (data, dev**2, **kw)
    h_stddev = (h_dev2 / h_count**2).sqrt()
    h_mean._errors = h_stddev.values
    if f is kde:
        h_mean._errors /= np.asarray (h_count.kernel_bins)
    return h_mean


def save (h, filename):
    np.save (filename, (h.bins, h.values, h.errors))
    if len (filename) >= 5 and filename [-4:] != '.npy':
        os.rename (filename + '.npy', filename)

def load (filename):
    return Hist (*np.load (filename))


class SplineFit (object):

    """
    Wrapper for spline fits to histograms.
    """

    def __init__ (self, hist, spline, bin_logs, log, floor):
        self.bins = hist.bins
        self.range = hist.range
        self.spline = spline
        self.bin_logs = bin_logs
        self.log = log
        self.floor = floor

    @property
    def n_dim (self):
        return len (self.bins)

    def __call__ (self, *args):
        """
        Get the spline-fitted values.
        """
        if self.n_dim == 1:
            assert len (args) == 1
            X = np.atleast_1d (args[0])
            mask = (self.bins[0][0] <= X) & (X <= self.bins[0][-1])
            if self.bin_logs[0]:
                X = np.log10 (X)
            out = np.zeros_like (X, dtype=float)
            out[mask] = self.spline (X[mask])
            out[~mask] = np.nan
            if self.log:
                with warnings.catch_warnings ():
                    warnings.filterwarnings ('ignore')
                    zero_mask = out <= self.floor
                #out = 10**out
                out = np.exp (out)
                out[zero_mask] = 0
            return out
        elif self.n_dim == 2:
            assert len (args) == 2
            X = np.atleast_1d (args[0])
            Y = np.atleast_1d (args[1])
            if not X.shape == Y.shape:
                raise ValueError (
                    'shapes of X ({}) and Y ({}) do not match'.format (
                    X.shape, Y.shape))
            out = np.zeros_like (X, dtype=float)
            mask = ((self.bins[0][0] <= X) & (X <= self.bins[0][-1])
                    & (self.bins[1][0] <= Y) & (Y <= self.bins[1][-1]))
            if self.bin_logs[0]:
                X = np.log10 (X)
            if self.bin_logs[1]:
                Y = np.log10 (Y)
            if np.any (mask):
                out[mask] = self.spline.ev (X[mask], Y[mask])
            out[~mask] = np.nan
            if self.log:
                with warnings.catch_warnings ():
                    warnings.filterwarnings ('ignore')
                    zero_mask = out <= self.floor
                    #out = 10**out
                    out = np.exp (out)
                out[zero_mask] = 0
            return out
        else:
            assert len (args) == self.n_dim
            args = list(map (np.atleast_1d, args))
            assert len (set (map (np.shape, args))) == 1
            out = np.empty (len (args[0]), dtype=float)
            mask = np.product ([
                (self.bins[i][0] <= args[i]) & (args[i] <= self.bins[i][-1])
                for i in xrange (self.n_dim)
            ], dtype=bool)
            for i in xrange (self.n_dim):
                if self.bin_logs[i]:
                    args[i] = np.log10 (args[i])
            if np.any (mask):
                args_masked = [args[i][mask] for i in xrange (self.n_dim)]
                out[mask] = self.spline (*args_masked)
            out[~mask] = np.nan
            if self.log:
                with warnings.catch_warnings ():
                    warnings.filterwarnings ('ignore')
                    zero_mask = out <= self.floor
                    #out = 10**out
                    out = np.exp (out)
                out[zero_mask] = 0
            return out
            #raise ValueError ('>2D Hist spline fits are not thought to exist')

    def eval (self, *a, **kw):
        if not hasattr (self, 'range'):
            self.range = [(b[0], b[-1]) for b in self.bins]
        kw.setdefault ('ndim', self.n_dim)
        kw.setdefault ('range', self.range)
        return hist_from_eval (self, *a, **kw)


class LineStyle (object):

    """Style object for 1D :class:`Hist` s."""

    def __init__ (self,
                  line=True,
                  marker='None',
                  errorbars=False,
                  errorcaps=False,
                  errorbands=False,
                  xerrorbars=False,
                  crosses=False,
                  poisson=False,
                  **kwargs
            ):
        """Initialize a LineStyle.

        :type   line: bool
        :param  line: Whether to draw a line.

        :type   marker: str (optional)
        :param  marker: The marker specification

        :type   errorbars: bool
        :param  errorbars: Whether to draw errorbars.

        :type   errorcaps: bool
        :param  errorcaps: Whether to draw error bar caps.

        :type   errorbands: bool
        :param  errorbands: Whether to draw error bands.

        :type   xerrorbars: bool
        :param  xerrorbars: Whether to draw x errorbars.

        :type   crosses: bool
        :param  crosses: bool

        :type   poisson: bool
        :param  poisson: Whether to show Poisson errorbars.

        All other keyword args are saved to be passed to
        matplotlib.axes.Axes.errorbar(). Note that drawstyle should not be
        specified; if line == True, then drawstyle='steps-post' will be used.

        """
        self._kwargs = {}
        self.line = True
        self.marker = 'None'
        self.errorbars = False
        self.errorcaps = False
        self.errorbands = False
        self.xerrorbars = False
        self.poisson = False
        self.update (
            line=line, marker=marker,
            errorbars=errorbars, errorcaps=errorcaps,
            errorbands=errorbands, xerrorbars=xerrorbars,
            crosses=crosses, poisson=poisson,
            **kwargs)

    def update (self, **kwargs):
        """Update the keyword args with the given values."""
        crosses = kwargs.pop ('crosses', False)
        if crosses:
            kwargs['errorbars'] = True
            kwargs['xerrorbars'] = True
            kwargs['line'] = False
        self.line = kwargs.pop ('line', self.line)
        self.marker = kwargs.pop ('marker', self.marker)
        self.errorbars = kwargs.pop ('errorbars', self.errorbars)
        self.errorcaps = kwargs.pop ('errorcaps', self.errorcaps)
        self.errorbands = kwargs.pop ('errorbands', self.errorbands)
        self.xerrorbars = kwargs.pop ('xerrorbars', self.xerrorbars)
        self.poisson = kwargs.pop ('poisson', self.poisson)
        self._kwargs.update (copy.deepcopy (kwargs))

    def copy (self, **kwargs):
        """Get a copy of this LineStyle, updating the given keyword args.

        All arguments accepted by the :class:`LineStyle` constructor may be
        given, including line, markers, errorbars, errorcaps, and arbitrary
        matplotlib arguments.

        """
        out = copy.deepcopy (self)
        out.update (**kwargs)
        return out

    @property
    def line (self):
        """Whether to draw a line."""
        return self._line
    @line.setter
    def line (self, line):
        self._line = line

    @property
    def markers (self):
        """Whether to draw point markers."""
        return self.marker not in ('None',)

    @property
    def marker (self):
        """The marker to use."""
        return self._marker
    @marker.setter
    def marker (self, marker):
        self._marker = marker
        self._kwargs['marker'] = marker

    @property
    def errorbars (self):
        """Whether to draw error bars."""
        return self._errorbars
    @errorbars.setter
    def errorbars (self, errorbars):
        self._errorbars = errorbars

    @property
    def errorbands (self):
        """Whether to draw error bands."""
        return self._errorbands
    @errorbands.setter
    def errorbands (self, errorbands):
        self._errorbands = errorbands

    @property
    def errorcaps (self):
        """Whether to draw error bar caps."""
        return self._errorcaps
    @errorcaps.setter
    def errorcaps (self, errorcaps):
        self._errorcaps = errorcaps
        if not self.errorcaps:
            self._kwargs['capsize'] = 0
        else:
            self._kwargs.pop ('capsize', None)

    @property
    def xerrorbars (self):
        """Whether to draw error bars."""
        return self._xerrorbars
    @xerrorbars.setter
    def xerrorbars (self, xerrorbars):
        self._xerrorbars = xerrorbars

    @property
    def poisson (self):
        """Whether to draw Poisson error bars."""
        return self._poisson
    @poisson.setter
    def poisson (self, poisson):
        self._poisson = poisson

    @property
    def kwargs (self):
        """Keyword args for matplotlib.axes.Axes.errorbar()."""
        out = copy.deepcopy (self._kwargs)
        if (not self.errorbars) and (not self.xerrorbars):
            out['elinewidth'] = np.finfo (float).tiny
        else:
            out.pop ('elinewidth', None)
        return out


def plot1d (ax, h=None, style=None, **kwargs):
    """
    Plot 1D :class:`Hist` ``h`` on matplotlib Axes ``ax``.

    :type   ax: matplotlib Axes
    :param  ax: if given, the axes on which to plot

    :type   h: :class:`Hist`
    :param  h: the 1D histogram

    :type   style: :class:`LineStyle`
    :param  style: the line style

    Other keyword arguments are propagated to pyplot.errorbar() and
    pyplot.plot() as appropriate.
    """
    if h is None:
        h = ax
        ax = plt.gca ()
    if h.n_dim != 1:
        raise TypeError ('`h` must be a 1D Hist')
    if style is None:
        style = LineStyle ()
    style.update (**kwargs)
    kw = copy.deepcopy (style.kwargs)
    if 'markers' in kw:
        del kw['markers']
    legend_kw = dict (label=kw.get ('label', ''))
    # get errorband stuff out of the way early
    bkw = {}
    bkw['alpha'] = kw.pop ('ebandalpha', .5) * kw.get ('alpha', 1.0)
    ebandlabel = kw.pop ('ebandlabel', '')

    n = h.n_bins[0]
    x = h.centers[0]
    y = h.values
    err = h.errors

    if style.poisson:
        alpha = 2 * stats.norm.sf (1)
        # see 
        # https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
        low = np.where (y == 0, 0, stats.chi2.ppf (alpha / 2, 2 * y) / 2.)
        high = stats.chi2.ppf (1 - alpha / 2, 2 * y + 2) / 2.
        err = [y - low, high - y]

    color = kw.get ('color', None)
    legend_line = False
    if style.markers or style.errorbars or style.xerrorbars:
        ekw = copy.deepcopy (kw)
        ekw['linestyle'] = 'none'
        if 'ls' in ekw:
            del ekw['ls']
        if 'drawstyle' in ekw:
            del ekw['drawstyle']
        ekw['color'] = color
        ekw['label'] = ''
        if color is None:
            ekw.pop ('color')
        if style.xerrorbars:
            ekw['xerr'] = [x - h.bins[0][:-1], h.bins[0][1:] - x]
        else:
            ekw['xerr'] = None
        if style.errorbars:
            ekw['yerr'] = err
        else:
            ekw['yerr'] = np.zeros_like (y)
        if style.errorbars or style.xerrorbars:
            eline = ax.errorbar (x, y, **ekw)[0]
        else:
            if 'capsize' in ekw:
                del ekw['capsize']
            if 'elinewidth' in ekw:
                del ekw['elinewidth']
            eline = ax.plot (x, y, **ekw)[0]
        kw['color'] = color = eline.get_color ()
        legend_kw.update (kw)
        legend_line = True

    if style.line:
        lkw = {}
        def keep (*keys):
            for key in keys:
                if key in kw:
                    lkw[key] = kw[key]

        lkw['drawstyle'] = kw.get ('drawstyle', 'steps-post')
        steps = lkw['drawstyle'] == 'steps-post'
        keep ('alpha', 'color', 'linewidth', 'lw', 'linestyle', 'ls')
        line_x = h.bins[0] if steps else h.centers[0]
        line_y = np.r_[y, y[-1]] if steps else y
        lline = ax.plot (line_x, line_y, **lkw)[0]
        lkw['color'] = color = lline.get_color ()
        legend_kw.update (lkw)
        legend_line = True
    else:
        legend_kw['linestyle'] = 'none'

    # weird hack to make legend work properly without disrupting the plot
    if legend_line:
        legend_args = [[np.nan],
                       [np.nan]]
        if style.errorbars:
            legend_args.append ([0])
        if style.xerrorbars:
            legend_args.append ([0])
        out = ax.errorbar (*legend_args, **legend_kw)

    if style.errorbands:
        bkw['color'] = color
        bkw['lw'] = 0
        steps = 'steps-post' == kw.get ('drawstyle', 'steps-post')
        eshape = np.shape (err)
        if len (eshape) == 2 and eshape[0] == 2:
            merr = err[0]
            perr = err[1]
        else:
            perr = merr = err
        if steps:
            ix = np.sort (np.r_[:n, 1:n+1])
            band_x = h.bins[0][ix]
            iy = np.sort (np.r_[:n, :n])
            band_ymin = (y - merr)[iy]
            band_ymax = (y + perr)[iy]
            if style.line:
                res = ax.fill_between (band_x, band_ymax, band_ymin, **bkw)
            else:
                for ib in xrange (0, 2 * n, 2):
                    res = ax.fill_between (
                        band_x[ib:ib+2], band_ymax[ib:ib+2], band_ymin[ib:ib+2],
                        **bkw
                    )
        else:
            band_x = x
            band_ymin = y - merr
            band_ymax = y + perr
            res = ax.fill_between (band_x, band_ymax, band_ymin, **bkw)
        if ebandlabel:
            eband_kw = dict (
                color=res.get_facecolor()[0],
                linestyle='-', lw=8, label=ebandlabel
            )
            ax.plot ([np.nan, np.nan], [np.nan, np.nan],
                     **eband_kw)


def fill_between (ax, h1, h2,
                  interpolate=False,
                  drawstyle='steps',
                  **kw):
    """
    Fill the region between histograms h1 and h2.

    :type   ax: matplotlib Axes
    :param  ax: the axes on which to plot

    :type   h1: number or :class:`Hist`
    :param  h1: a number or the first histogram

    :type   h2: number or :class:`Hist`
    :param  h2: a number or the second histogram

    :param  where: see ``pyplot.fill_between()``.

    :param  interpolate: see ``pyplot.fill_between()``.

    :type   drawstyle: str
    :param  drawstyle: if 'line', plot smooth curves; otherwise, plot with
        histogram steps

    Other keyword arguments are passed to ``pyplot.fill_between()``.

    """
    if ax is None:
        fig, ax = plt.subplots ()
    if isinstance (h1, Hist) and isinstance (h2, Hist):
        h1.assert_match (h2)
        bins = h1.bins[0]
        centers = h1.centers[0]
        y1 = h1.values
        y2 = h2.values
        if h1.n_dim != 1 or h2.n_dim != 1:
            raise TypeError ('histogram arguments must be 1D Hists')
    elif isinstance (h1, Hist):
        bins = h1.bins[0]
        centers = h1.centers[0]
        y1 = h1.values
        y2 = h2 * np.ones (h1.values)
        if h1.n_dim != 1:
            raise TypeError ('histogram arguments must be 1D Hists')
    elif isinstance (h2, Hist):
        bins = h2.bins[0]
        centers = h2.centers[0]
        y1 = h1 * np.ones_like (h2.values)
        y2 = h2.values
        if h2.n_dim != 1:
            raise TypeError ('histogram arguments must be 1D Hists')
    else:
        raise TypeError ('one of `h1` or `h2` must be of type Hist')

    steps = drawstyle != 'line'
    n = len (centers)
    if steps:
        ix = np.sort (np.r_[:n, 1:n+1])
        band_x = bins[ix]
        iy = np.sort (np.r_[:n, :n])
        band_ymin = y1[iy]
        band_ymax = y2[iy]
    else:
        band_x = centers
        band_ymin = y1
        band_ymax = y2

    label = kw.pop ('label', '')

    res = ax.fill_between (band_x, band_ymax, band_ymin,
                             interpolate=interpolate, **kw)

    if label:
        kw['color'] = res.get_facecolor ()[0]
        ax.plot ([np.nan, np.nan], [np.nan, np.nan],
                 lw=8, label=label, **kw)

    return res


def stack1d (ax, hs,
             colors=None,
             labels=None,
             kws=None,
             interpolate=False,
             drawstyle='steps',
             ymin=0,
             **morekw):
    """
    Stack histograms ``hs`` using :func:`fill_between`.

    :type   ax: matplotlib Axes
    :param  ax: the axes on which to plot

    :type   hs: sequence of :class:`Hist`
    :param  hs: the histograms

    :type   colors: sequence
    :param  colors: the fill colors

    :type   labels: sequence of str
    :param  labels: the labels

    :type   kws: sequence of str to value mappings
    :param  kws: keyword arguments for individual fills

    :param  interpolate: see ``pyplot.fill_between()``

    :type   drawstyle: str
    :param  drawstyle: if 'line', plot smooth curves; otherwise, plot with
        histogram steps

    :type   ymin: number
    :param  ymin: minimum value (useful for semilogy plots)

    Other keyword arguments are passed to each :func:`fill_between` call.

    """
    if ax is None:
        fig, ax = plt.subplots ()
    if colors is None:
        default_cycle = 'bgrcmyk'
        colors = ''.join (default_cycle[i % len (default_cycle)]
                          for i in range (len (hs)))
    if labels is None:
        labels = ['' for h in hs]
    if kws is None:
        kws = [{} for h in hs]
    total = ymin
    outs = []
    for (h, kw, color, label) in izip (hs, kws, colors, labels):
        new_total = total + h
        kw.update (morekw)
        outs.append (fill_between (ax, total, new_total,
                                   interpolate=interpolate,
                                   drawstyle=drawstyle,
                                   color=color,
                                   label=label,
                                   **kw))
        total = new_total
    return outs


def plot2d (ax, h=None,
            log=False,
            cbar=False,
            levels=None,
            **kwargs):
    """ Plot 1D :class:`Hist` ``h`` on ``ax`` on a color scale.

    :type   ax: matplotlib Axes
    :param  ax: The main axes on which to plot.

    :type   h: :class:`Hist`
    :param  h: The 2D histogram to plot.

    :type   log: bool
    :param  log: Whether to use a log color scale

    :type   cbar: bool
    :param  cbar: If true, draw colorbar.

    :type   levels: int or array of float
    :param  levels: if given, plot with contourf rather than pcolormesh. if
        a number is given, automatically select that many levels between vmin
        and vmax.

    :type   zmin: float
    :param  zmin: Minimum value to plot with color; bins below the minimum
        value will be white.

    :return: If cbar, a dict containing a matplotlib.collection.QuadMesh and a
        matplotlib.colorbar.Colorbar as values; otherwise, just the QuadMesh.

    Other keyword arguments are passed to ax.pcolormesh().
    """
    if h is None:
        h = ax
        ax = plt.gca ()
    if h.n_dim != 2:
        raise TypeError ('`h` must be a 2D Hist')
    plotvalues = h.values.T
    zmin = kwargs.pop ('zmin', None)
    if log:
        if zmin is None:
            zmin = 0
        H = np.ma.array (plotvalues)
    else:
        H = np.ma.array (plotvalues)
    H.mask += ~np.isfinite (plotvalues)
    if zmin is not None:
        H.mask += (plotvalues <= zmin)
    xbins = h.bins[0]
    ybins = h.bins[1]
    Bx, By = np.meshgrid (xbins, ybins)
    vmin = kwargs.pop ('vmin', H.min ())
    vmax = kwargs.pop ('vmax', H.max ())
    clabel = kwargs.pop ('clabel', None)
    cticks = None
    if log:
        kwargs['norm'] = mpl.colors.LogNorm (vmin, vmax)
    if levels is None:
        if not log:
            kwargs['vmin'], kwargs['vmax'] = vmin, vmax
        pc = ax.pcolormesh (Bx, By, H, **kwargs)
    else:
        if isinstance (levels, int):
            if log:
                levels = np.logspace (np.log10 (vmin), np.log10 (vmax), levels + 1)
                cticks = 10**np.arange (
                    np.ceil (np.log10 (vmin)), np.floor (np.log10 (vmax)) + .1 )
            else:
                levels = np.linspace (vmin, vmax, levels + 1)
        fill = kwargs.pop ('fill', True)
        if fill:
            pc = ax.contourf (Bx[:-1,:-1], By[:-1,:-1], np.clip (H, vmin, vmax), levels, **kwargs)
        else:
            pc = ax.contour (Bx[:-1,:-1], By[:-1,:-1], np.clip (H, vmin, vmax), levels, **kwargs)
    if h.log[0]:
        ax.set_xscale ('log')
    if h.log[1]:
        ax.set_yscale ('log')
    ax.set_xlim (xbins.min (), xbins.max ())
    ax.set_ylim (ybins.min (), ybins.max ())
    if cbar:
        cb_kw = cbar if hasattr (cbar, 'keys') else {}
        if log:
            cb = ax.figure.colorbar (
                pc, ax=ax, format=mpl.ticker.LogFormatterMathtext (), **cb_kw)
        else:
            cb = ax.figure.colorbar (pc, ax=ax, **cb_kw)
        if cticks is not None:
            cb.set_ticks (cticks)
        if clabel:
            cb.set_label (clabel)
        return dict (colormesh=pc, colorbar=cb)
    else:
        return pc


def label2d (ax, h,
             fmt='',
             **kw):
    if ax is None:
        fig, ax = plt.subplots ()
    for i in xrange (h.n_bins[0]):
        for j in xrange (h.n_bins[1]):
            ax.text (h.centers[0][i], h.centers[1][j],
                       format (h.values[i,j], fmt),
                       ha='center', va='center',
                       fontdict=kw)
