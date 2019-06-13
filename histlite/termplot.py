# termplot.py
# This Python file uses the following encoding: utf-8


from __future__ import print_function

from copy import deepcopy
from collections import defaultdict
from itertools import izip
import numpy as np
try:
    import termcolor
except:
    termcolor = False



class Axes (object):

    def __init__ (self,
                  x=None, y=None, color=None, zorder=0, visible=True):
        self.x = x
        self.y = y
        self.color = color
        self.zorder = zorder
        self.visible = visible

    def i (self, fig):
        if None not in (self.x, self.y):
            return fig._i (self.x)
        else:
            return fig.xpad

    def j (self, fig):
        if None not in (self.x, self.y):
            return fig._j (self.y)
        else:
            return fig.ypad

    def raster (self, fig):
        raster = fig.blank
        i_xaxis = self.i (fig)
        j_yaxis = self.j (fig)
        II, JJ = fig.II, fig.JJ
        xaxis = np.where (II == i_xaxis)
        yaxis = np.where (JJ == j_yaxis)
        origin = np.where ((II == i_xaxis) & (JJ == j_yaxis))
        fig._set (raster, u'│', *xaxis, color=self.color)
        fig._set (raster, u'─', *yaxis, color=self.color)
        fig._set (raster, u'┼', *origin, color=self.color)
        return raster

class Ticks (object):

    def __init__ (self, axes, axis,
                  ticks=None, color=None, zorder=None, visible=None):
        assert axis in 'xy'
        self.axes = axes
        self.axis = axis
        self.ticks = ticks
        self.color = color
        self.zorder = zorder
        self.visible = visible

    def raster (self, fig):
        raster = fig.blank
        axis = self.axis
        axes = self.axes
        color = axes.color if self.color is None else self.color
        if axis == 'x':
            xmin, xmax = fig.xlim
            delta = xmax - xmin
            if fig.xlog:
                ticks = 10**np.arange (np.ceil (np.log10 (xmin)),
                                       np.floor (np.log10 (xmax)) + 1) \
                        if self.ticks is None else self.ticks
            else:
                ticks = np.linspace (xmin, xmax, 5) \
                        if self.ticks is None else self.ticks
            for xtick in ticks:
                j = axes.j (fig) - 1
                label = format (xtick if delta < 1 else np.round (xtick), 'g')
                L = len (label)
                i1 = fig._i (xtick) - L / 2
                i2 = i1 + L
                I = np.r_[i1:i2]
                J = np.repeat (j, L)
                fig._set (raster, list (label), I, J,
                          color=color, clip=False)
        elif axis == 'y':
            ymin, ymax = fig.ylim
            delta = ymax - ymin
            if fig.ylog:
                ticks = 10**np.arange (np.ceil (np.log10 (ymin)),
                                       np.floor (np.log10 (ymax)) + 1) \
                        if self.ticks is None else self.ticks
            else:
                ticks = np.linspace (ymin, ymax, 5) \
                        if self.ticks is None else self.ticks
            for ytick in ticks:
                j = fig._j (ytick)
                label = format (ytick if delta < 1 else np.round (ytick),
                                '{}g'.format (fig.xpad))
                L = len (label)
                i2 = axes.i (fig) - 1
                i1 = max (0, i2 - L)
                I = np.r_[i1:i1+L]
                J = np.repeat (j, L)
                fig._set (raster, list (label), I, J,
                          color=color, clip=False)
        return raster

class Scatter (object):

    def __init__ (self,
                  x, y,
                  marker=u'·',
                  color=None,
                  zorder=0,
                  visible=True,
                 ):
        """
        Draw a scatter plot.
        """
        self.x = x
        self.y = y
        self.marker = marker
        self.color = color
        self.zorder = zorder
        self.visible = visible

    def raster (self, fig):
        raster = fig.blank
        i, j = fig._i (self.x), fig._j (self.y)
        fig._set (raster, self.marker, i, j, color=self.color)
        return raster

class Hist1D (object):

    def __init__ (self, h, color=None, zorder=0, visible=True):
        assert (h.n_dim == 1)
        self.h = h
        self.color = color
        self.zorder = zorder
        self.visible = visible

    def raster (self, fig):
        raster = fig.blank
        i, j = [], []
        for (ix, x) in izip (fig._I, fig._X):
            if not self.h.range[0][0] <= x < self.h.range[0][-1]:
                continue
            v = self.h.get_value (x)
            i.append (ix), j.append (fig._j (v))
        fig._set (raster, u'─', i, j, color=self.color)
        return raster


class Figure (object):

    """
    Axes on which to plot.
    """

    dtype = 'U20'

    colors = {
        'r': 'red',
        'g': 'green',
        'y': 'yellow',
        'b': 'blue',
        'm': 'magenta',
        'c': 'cyan',
        'w': 'white',
        '.5': 'grey',
    }

    # Setup
    
    def __init__ (self,
                  xsize=73, ysize=23,
                  xpad=7, ypad=1,
                 ):
        # store dimensions
        self._xsize = xsize
        self._ysize = ysize
        self._xpad = xpad
        self._ypad = ypad
        big = float (np.finfo (np.float32).max)
        self._xlim = (big, -big)
        self._ylim = (big, -big)
        self._xlog = False
        self._ylog = False
        self.set_axes ()
        self.set_xticks ()
        self.set_yticks ()
        self._set_scale ()
        # prepare for drawing
        self.Xs = defaultdict (list)

    def _set_scale (self):
        self._Xsize = self.xsize + self.xpad
        self._Ysize = self.ysize + self.ypad
        Dx = np.diff (np.log10 (self.xlim))[0] \
                if self.xlog  else np.diff (self.xlim)[0]
        Dy = np.diff (np.log10 (self.ylim))[0] \
                if self.ylog  else np.diff (self.ylim)[0]
        self._dx = 1.0 * (self.xsize - 1) / Dx
        self._dy = 1.0 * (self.ysize - 1) / Dy
        self._I = np.arange (self.Xsize)
        self._J = np.arange (self.Ysize)
        self._II, self._JJ = np.meshgrid (self._I, self._J, indexing='ij')
        self._X = self._x (self._I)
        self._XX = self._x (self._II)
        self._Y = self._y (self._J)
        self._YY = self._y (self._JJ)

    def _get_lim (self, lim, log, *a):
        x = [np.ravel (value) for value in a if value is not None]
        if x:
            x = np.concatenate (x)
            i = np.isfinite (np.log10 (x)) if log else np.isfinite (x)
            xmin = np.min (x[i])
            xmax = np.max (x[i])
            if lim[0] is not None:
                xmin = min (lim[0], xmin - 5 * np.finfo (float).eps)
            if lim[1] is not None:
                xmax = max (lim[1], xmax + 5 * np.finfo (float).eps)
            return xmin, xmax
        else:
            return lim

    def _update_xlim (self, *a):
        self.set_xlim (*self._get_lim (self.xlim, self.xlog, *a))
    def _update_ylim (self, *a):
        self.set_ylim (*self._get_lim (self.ylim, self.ylog, *a))


    # Properties reference

    @property
    def xlim (self):
        return self._xlim
    @property
    def ylim (self):
        return self._ylim
    @property
    def xsize (self):
        return self._xsize
    @property
    def ysize (self):
        return self._ysize
    @property
    def xpad (self):
        return self._xpad
    @property
    def ypad (self):
        return self._ypad
    @property
    def Xsize (self):
        return self._Xsize
    @property
    def Ysize (self):
        return self._Ysize
    @property
    def xlog (self):
        return self._xlog
    @property
    def ylog (self):
        return self._ylog
    @property
    def dx (self):
        return self._dx
    @property
    def dy (self):
        return self._dy
    @property
    def I (self):
        return self._I
    @property
    def J (self):
        return self._J
    @property
    def II (self):
        return self._II
    @property
    def JJ (self):
        return self._JJ
    @property
    def X (self):
        return self._X
    @property
    def Y (self):
        return self._Y
    @property
    def XX (self):
        return self._XX
    @property
    def YY (self):
        return self._YY

    def set_xlim (self, xmin=None, xmax=None):
        xmin = self.xlim[0] if xmin is None else xmin
        xmax = self.xlim[1] if xmax is None else xmax
        self._xlim = (xmin, xmax)
        self._set_scale ()
    def set_ylim (self, ymin=None, ymax=None):
        ymin = self.ylim[0] if ymin is None else ymin
        ymax = self.ylim[1] if ymax is None else ymax
        self._ylim = (ymin, ymax)
        self._set_scale ()


    # Drawing building blocks

    @property
    def blank (self):
        a = np.ndarray ((self.Xsize, self.Ysize), dtype=self.dtype)
        a[:] = ' '
        return a
    def _i (self, x):
        if self.xlog:
            i0 = np.round (self.dx * (np.log10 (x) - np.log10 (self.xlim[0])))
        else:
            i0 = np.round (self.dx * (x - self.xlim[0]))
        return self.xpad + np.array (i0, dtype=int)
    def _j (self, y):
        if self.ylog:
            j0 = np.round (self.dy * (np.log10 (y) - np.log10 (self.ylim[0])))
        else:
            j0 = np.round (self.dy * (y - self.ylim[0]))
        return self.ypad + np.array (j0, dtype=int)
    def _x (self, i):
        if self.xlog:
            return 10**((i - self.xpad - .5) / self.dx + np.log10 (self.xlim[0]))
        else:
            return (i - self.xpad - .5) / self.dx + self.xlim[0]
    def _y (self, j):
        if self.ylog:
            return 10**((j - self.ypad - .5) / self.dy + np.log10 (self.ylim[0]))
        else:
            return (j - self.ypad - .5) / self.dy + self.ylim[0]
    def _set (self, raster, value, i, j, color=None, clip=True):
        i = np.atleast_1d (i)
        j = np.atleast_1d (j)
        if color and termcolor:
            value = np.atleast_1d (value)
            color = self.colors.get (color, color)
            value = [termcolor.colored (v, color) for v in value]
        if clip:
            i0, j0 = self.xpad, self.ypad
        else:
            i0 = j0 = 0
        mask = (i0 <= i) & (i < self.Xsize) & (j0 <= j) & (j < self.Ysize)
        if mask.sum ():
            raster[i[mask], j[mask]] = value

    # Plotting
    def set_axes (self, x=None, y=None, color='grey', zorder=0, visible=True):
        self.axes = Axes (x=x, y=y, color=color, zorder=zorder, visible=visible)

    def set_xticks (self, ticks=None, color=None, zorder=0, visible=True):
        self.xticks = Ticks (self.axes, 'x',
                             ticks=ticks, color=color, zorder=zorder,
                             visible=visible)
        self._update_xlim (ticks)

    def set_yticks (self, ticks=None, color=None, zorder=0, visible=True):
        self.yticks = Ticks (self.axes, 'y',
                             ticks=ticks, color=color, zorder=zorder,
                             visible=visible)
        self._update_ylim (ticks)

    def scatter (self, x, y, marker=u'·', color=None, zorder=0):
        s = Scatter (
            x, y, marker=marker, color=color, zorder=zorder
        )
        self.Xs[zorder].append (s)
        self._update_xlim (x)
        self._update_ylim (y)

    def plot (self, *a, **kw):
        x0, y0 = a[:2]
        a = a[2:]
        x = np.linspace (min (x0), max (x0), 10**3)
        y = np.interp (x, x0, y0)
        return self.scatter (x, y, *a, **kw)

    def semilogx (self, *a, **kw):
        self._xlog = True
        if a or kw:
            return self.plot (*a, **kw)

    def semilogy (self, *a, **kw):
        self._ylog = True
        if a or kw:
            return self.plot (*a, **kw)

    def loglog (self, *a, **kw):
        self._xlog = self._ylog = True
        if a or kw:
            return self.plot (*a, **kw)

    def hist1d (self, h, color=None, zorder=0):
        h1d = Hist1D (
            h, color=color, zorder=zorder
        )
        self.Xs[zorder].append (h1d)
        self._update_xlim (h.bins[0])
        self._update_ylim (h.values[np.isfinite (h.values)])

    # Display

    def __unicode__ (self):
        raster = self.blank
        Xs = deepcopy (self.Xs)
        for z in sorted (self.Xs):
            if self.axes.visible and self.axes.zorder == z:
                Xs[z].insert (0, self.axes)
            if self.xticks.visible and self.xticks.zorder == z:
                Xs[z].insert (0, self.xticks)
            if self.yticks.visible and self.yticks.zorder == z:
                Xs[z].insert (0, self.yticks)
            for X in Xs[z]:
                if not X.visible:
                    continue
                r = X.raster (self)
                i, j = np.where (r)
                raster[i,j] = r[i,j]
        out = u'\n'.join (map (u''.join, raster.T[::-1]))
        return out

    def __repr__ (self):
        out = 'Figure at 0x{:x}'.format (id (self))
        return out
