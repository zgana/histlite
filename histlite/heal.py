# heal.py

from __future__ import print_function

import healpy
import numpy as np
pi = np.pi
from scipy import special, stats


def _umap (x):
    if isinstance (x, HealHist):
        return x.map
    else:
        return x

class HealHist (object):

    """A HEALPix 'histogram'."""

    def __init__ (self, map):
        self.map = np.atleast_1d (map)
        self.npix = len (map)
        self.nside = healpy.npix2nside (self.npix)
        self.dOmega = healpy.nside2pixarea (self.nside)

    @property
    def _thetas_phis (self):
        return healpy.pix2ang (self.nside, np.r_[:self.npix])

    def sum (self):
        return np.sum (self.map)

    def integrate (self):
        return self.sum () * self.dOmega

    def normalize (self, integrate=True):
        """
        Return a normalized HealHist.
        """
        if integrate:
            return HealHist (self.map / self.integrate ())
        else:
            return HealHist (self.map / self.sum ())
        
    def ra_average (self):
        thetas, phis = self._thetas_phis
        amap = np.empty_like (self.map)
        for theta in np.unique (thetas):
            mask = (theta == thetas)
            amap[mask] = np.mean (self.map[mask])
        return HealHist (amap)

    def rotate (self, *a, **kw):
        kw['inv'] = kw.get ('inv', True)
        rot = healpy.Rotator (*a, **kw)
        thetas0, phis0 = self._thetas_phis
        thetas, phis = rot (thetas0, phis0)
        rmap = healpy.get_interp_val (self.map, thetas, phis)
        return HealHist (rmap)

    def ud_grade (self, *a, **kw):
        ud_map = healpy.ud_grade (self.map, *a, **kw)
        return HealHist (ud_map)

    def smoothing (self, *a, **kw):
        return HealHist (healpy.smoothing (self.map, *a, **kw))

    def __add__ (self, other):
        return HealHist (self.map + _umap (other))

    def __radd__ (self, other):
        return self + other

    def __sub__ (self, other):
        return HealHist (self.map - _umap (other))

    def __rsub__ (self, other):
        return HealHist (_umap (other) - self.map)

    def __mul__ (self, other):
        return HealHist (self.map * _umap (other))

    def __rmul__ (self, other):
        return self * other
        
    def __div__ (self, other):
        return HealHist (self.map / _umap (other))

    def __rdiv (self, other):
        return HealHist (_umap (other) / self.map)

    def __neg__ (self):
        return HealHist (-self.map)


def hist (nside, dec, ra, weights=None):
    dec = np.atleast_1d (dec)
    ra = np.atleast_1d (ra)
    if weights is None:
        weights = np.ones_like (dec)
    else:
        weights = np.atleast_1d (weights)
    if not (dec.shape == ra.shape == weights.shape):
        raise ValueError (
            'must have `dec.shape` == `ra.shape` == `weights.shape`'
        )
    import healpy
    npix = healpy.nside2npix (nside)
    pixels = healpy.ang2pix (nside, pi/2-dec, ra)
    map = np.bincount (pixels, weights=weights, minlength=npix)
    return HealHist (map)


def sph_harm (nside, m, n):
    npix = healpy.nside2npix (nside)
    theta, phi = healpy.pix2ang (nside, np.r_[:npix])
    map = special.sph_harm (m, n, phi, theta).real
    return HealHist (map)

