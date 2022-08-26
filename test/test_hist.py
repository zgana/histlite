#!/usr/bin/env python

import copy
import os
import tempfile

import unittest
import pytest

import numpy as np

import histlite as hl

eps = np.finfo(float).eps
xrange = range


class TestReindex(unittest.TestCase):
    """
    Test reindex().
    """
    def test_reindex(self):
        np.random.seed(0)
        # for a few tries at each of a few dimensionalities
        for n_dim in xrange(3, 6):
            for i in xrange(10):
                shape = tuple(np.random.randint(2, 6, n_dim))
                order = np.random.permutation(len(shape))
                n = np.prod(shape)
                a = np.arange(n).reshape(shape)
                self.assertEqual(a.shape, shape)
                # reorder
                ar = hl.reindex(np.copy(a), order)
                self.assertEqual(ar.shape, tuple(np.array(shape)[order]))
                # un-reorder
                arr = hl.unreindex(np.copy(ar), order)
                self.assertEqual(a.shape, arr.shape)


class TestHist(unittest.TestCase):

    def test_range(self):
        """
        Test basic hist() results.
        """
        x = np.random.uniform(0, 1, int(1e4))

        # default values
        h = hl.hist(x)
        self.assertFalse(h.log[0])
        dx0 = abs(h.bins[0][0] - np.min(x))
        dx1 = abs(h.bins[0][-1] - np.max(x))
        self.assertTrue(dx0 <= eps)
        self.assertTrue(dx1 <= eps)
        self.assertEqual(h.n_dim, 1)
        self.assertEqual(len(h.n_bins), 1)
        self.assertEqual(len(h.bins), 1)
        self.assertEqual(h.n_bins[0], len(h.centers[0]))
        self.assertEqual(h.n_bins[0], len(h.bins[0]) - 1)
        self.assertEqual(h.n_bins[0], np.sum(
            (h.bins[0][:-1] < h.centers[0])
            & (h.bins[0][1:] > h.centers[0])
        ))
        self.assertEqual(h.n_bins[0], len(h.values))
        self.assertTrue(eps >= np.abs(
            np.sum(h.widths) - (h.range[0][1] - h.range[0][0])
        ))
        self.assertTrue(np.max(np.abs(h.widths[0] - h.volumes) < eps))

        # log scale
        h = hl.hist(x, log=True)
        self.assertTrue(h.log[0])
        dx0 = abs(h.bins[0][0] - np.min(x))
        dx1 = abs(h.bins[0][-1] - np.max(x))
        self.assertTrue(dx0 <= eps)
        self.assertTrue(dx1 <= eps)
        self.assertEqual(h.n_bins[0], np.sum(
            (h.bins[0][:-1] < h.centers[0])
            & (h.bins[0][1:] > h.centers[0])
        ))
        self.assertTrue(eps >= np.abs(
            np.sum(h.widths) - (h.range[0][1] - h.range[0][0])
        ))

        # simple 3D
        x = np.random.uniform(0, 1, (3, int(1e3)))
        h = hl.hist(x, range=(0,1))
        self.assertTrue(len(h.bins) == len(h.centers) == h.n_dim)
        self.assertEqual(h.values.shape, tuple(h.n_bins))
        self.assertTrue(3 * eps >= np.abs(np.sum(h.volumes) - 1))

    def test_get(self):
        """
        Test Hist.get_index() and Hist.__slice__().
        """
        v = [[0., 1], [2, 3]]
        h = hl.Hist([[0, .5, 1], [1, 1.5, 2]], v, np.sqrt(v))
        self.assertEqual(h.sum().values, np.sum(v))
        self.assertTrue(eps >= np.max(np.abs(
            np.array(v) - h.values
        )))
        self.assertTrue(eps >= np.max(np.abs(
            np.sqrt(v) - h.errors
        )))
        self.assertEqual(h[.25].values[1], 1)
        self.assertEqual(h[:,1.25].values[1], 2)
        self.assertEqual(h[.25, 1.25].values, 0)
        self.assertEqual(h.get_value(.25, 1.25), 0)
        with self.assertRaises(TypeError):
            h.get_value(.25)
        self.assertTrue(np.all(
            h.get_values([.25, .75], [1.25, 1.75]) == np.array([0, 3])
        ))

    def test_sum(self):
        """
        Test Hist.sum(), Hist.cumsum(), and related methods.
        """
        np.random.seed(0)
        shape = 4, int(2e4)
        h = hl.hist(np.random.uniform(0, 1, shape),
                     bins=(2,4,5,10), range=(0,1))
        self.assertEqual(h.sum().values, shape[1])
        self.assertEqual(h.sum().volumes, 0)
        self.assertEqual(h.project([1, 2]).values.shape, (4, 5))
        with self.assertRaises(ValueError):
            h.sum([5])
        for i in xrange(h.n_dim):
            for j in xrange(h.n_dim):
                if i == j:
                    continue
                hn = h.normalize([i,j])
                self.assertEqual(np.shape(hn.values), np.shape(h.values))
                self.assertEqual(np.shape(hn.errors), np.shape(h.errors))
                self.assertTrue(
                    1e-6 > np.max(np.abs(hn.integrate([i,j]).values - 1))
                )
                hn = h.normalize([i,j], integrate=False)
                self.assertEqual(np.shape(hn.values), np.shape(h.values))
                self.assertEqual(np.shape(hn.errors), np.shape(h.errors))
                self.assertTrue(
                    1e-6 > np.max(np.abs(hn.sum([i,j]).values - 1))
                )
        for i in xrange(h.n_dim):
            hn = h.normalize([i], integrate=False)
            hncs = hn.cumsum([i])
            hncs_last = hncs.get_slice(-1, i)
            self.assertTrue(
                1e-6 > np.max(np.abs(hncs_last.values) - 1)
            )

        h = hl.hist(10**np.random.uniform(0, 1, int(1e4)),
                     bins=4, range=(1, 10), log=True)
        hn = h.normalize()
        hd = h.normalize(density=True)
        self.assertGreater(4 * eps, np.abs(hn.integrate().values - 1))
        self.assertGreater(4 * eps, np.abs(hd.integrate().values - 1))
        self.assertFalse(np.any(hn.values == hd.values))

    def test_err(self):
        """
        Test error propagation.
        """
        h1 = hl.hist([.25, .75], bins=1, range=(0,1))
        h2 = hl.hist([.25, .5, .75], bins=1, range=(0,1))
        self.assertGreaterEqual(
            eps,
            np.abs(h1.get_errors([.25]) - np.sqrt(2))
        )
        self.assertGreaterEqual(
            1e-6,
            np.max(h1.errors - np.sqrt(2))
        )
        self.assertGreaterEqual(
            1e-6,
            np.max(h2.errors - np.sqrt(3))
        )
        hadd = h1 + h2
        self.assertGreaterEqual(
            1e-6,
            np.max(hadd.errors - np.sqrt(2 + 3))
        )
        hadd = 1 + h1
        self.assertGreaterEqual(
            1e-6,
            np.max(hadd.errors - np.sqrt(2))
        )
        hsub = h2 - h1
        self.assertGreaterEqual(
            1e-6,
            np.max(hsub.errors - np.sqrt(2 + 3))
        )
        hsub = 1 - h1
        self.assertGreaterEqual(
            1e-6,
            np.max(hsub.errors - np.sqrt(2))
        )
        hmul = h1 * h2
        self.assertGreaterEqual(
            1e-6,
            np.max(hmul.errors - 6 * np.sqrt(1/np.sqrt(2) + 1/np.sqrt(3)))
        )
        hmul = 2 * h1
        self.assertGreaterEqual(
            1e-6,
            np.max(hmul.errors - 2 * np.sqrt(2))
        )
        hdiv = h2 / h1
        self.assertGreaterEqual(
            1e-6,
            np.max(hdiv.errors - 1.5 * np.sqrt(1/np.sqrt(2) + 1/np.sqrt(3)))
        )
        hdiv = h1 / 2.
        self.assertGreaterEqual(
            1e-6,
            np.max(hdiv.errors - 0.5 * np.sqrt(2))
        )

        x = np.random.uniform(0, 1, int(1e4))
        w = np.random.uniform(.2, .8, x.size)
        hall = hl.hist(x, w, bins=10, range=(0, 1))
        hcut = hl.hist(x[w > .5], w[w > .5], bins=10, range=(0, 1))
        div = hcut / hall
        eff = hcut.efficiency(hall)
        self.assertTrue(np.all(eff.errors < div.errors))

    def test_sample(self):
        """
        Test Hist.sample().
        """
        np.random.seed(0)
        h = hl.hist(np.random.uniform(0, 1, (2, int(1e3))),
                     bins=4, range=(0,1))

        # test sampling in both dimensions
        x, y = h.sample(int(1e5))
        h2 = hl.hist((x, y), bins=4, range=(0,1))
        deltas = np.abs((h2 / 1e2 - h).values)
        self.assertLess(np.max(deltas - h.errors), 0)
        self.assertGreater(np.min(x), h.bins[0][0])
        self.assertGreater(np.min(y), h.bins[1][0])
        self.assertLess(np.max(x), h.bins[0][-1])
        self.assertLess(np.max(y), h.bins[1][-1])

        # test sampling for a given x
        y = h.sample(int(1e5), x[0])
        h1 = hl.hist(y, bins=4, range=(0,1))
        deltas = np.abs((h1 / 1e2 / 4).values - h[x[0]].values)
        self.assertLess(np.max(deltas - h.errors), 0)

    def test_contain(self):
        """
        Test Hist.contain() and related methods.
        """
        x = np.random.uniform(0, 1, int(3e5))
        y = np.random.normal(0, 1, int(3e5))
        h = hl.hist((x, y), bins=(4,101), range=((0, 1), (-4, 4)))

        # median should be around 0
        hmed = h.median(1)
        self.assertGreater(1e-6, np.abs(np.mean(hmed.values)))

        # containment hist errors should reflect sigma=1
        hcont = h.contain_project(1, n_sigma=1)
        self.assertGreater(.05, np.abs(np.mean(hcont.errors) - 1))

    def test_mean_1d(self):
        """
        Test Hist.mean().
        """
        bins = [[-3.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 3.5]]
        # centers = -3, -2, -1, 0, +1, +2, +3
        values = [1, 2, 3, 4, 4, 4, 4]
        # mean -> (-3 - 4 - 3 + 0 + 4 + 8 + 12) / 22
        h = hl.Hist(bins, values)
        m = h.mean()
        self.assertAlmostEqual(
            m.values[()],
            (-3 - 4 - 3 + 0 + 4 + 8 + 12) / 22
        )
        # TODO: test_mean_2d for axis=0 and axis=1

    def test_rebin(self):
        """
        Test Hist.rebin().
        """
        # lin bins
        h = hl.hist(np.random.uniform(0, 1, int(1e4)), bins=4, range=(0, 1))
        hr = h.rebin(0, [0, .5, 1])
        self.assertEqual(h.values[:2].sum(), hr.values[0])
        self.assertEqual(h.values[2:4].sum(), hr.values[1])

        # log bins
        h = hl.hist(10**np.random.uniform(0, 2, int(1e4)),
                     bins=4, range=(1, 100), log=True)
        hr = h.rebin(0, [1, 10, 100])
        self.assertEqual(h.values[:2].sum(), hr.values[0])
        self.assertEqual(h.values[2:4].sum(), hr.values[1])

        with self.assertRaises(ValueError):
            h + hr

    def test_transpose(self):
        """
        Test Hist.T.
        """
        h = hl.hist(np.random.uniform(0, 1, (2, int(1e4))),
                     bins=(4, 10), range=(0, 1))
        hT = h.T
        self.assertEqual(h.bins[::-1], hT.bins)
        self.assertTrue(np.all(h.values.T == hT.values))
        self.assertTrue(np.all(h.errors.T == hT.errors))

    def test_fit_1D(self):
        """
        Test 1D Hist.curve_fit() and Hist.spline_fit().
        """
        np.random.seed(0)
        h = hl.hist(np.random.normal(0, 1, int(1e5)), bins=50, range=(-4, 4))
        hn = h.normalize()

        # curve_fit
        def gauss(x, sigma):
            return 1 / (np.sqrt(2*np.pi) * sigma) \
                * np.exp(-x**2 / (2 * sigma**2))
        sigma = hn.curve_fit(gauss)[0][0]
        self.assertGreater(.01, np.abs(sigma - 1))

        # spline_fit
        s = hn.spline_fit()
        self.assertTrue(np.all(
            1e-2 > np.abs(s(hn.centers) - hn.values)
        ))
        s = hn.spline_fit(log=True)
        self.assertTrue(np.all(
            1e-2 > np.abs(s(hn.centers) - hn.values)
        ))

    def test_fit_2D(self):
        """
        Test 2D Hist.curve_fit() and Hist.spline_fit().
        """
        np.random.seed(0)
        h = hl.hist(np.random.normal(0, 1, (2, int(1e5))),
                     bins=50, range=(-4, 4))
        hn = h.normalize()

        # curve_fit
        def gauss2(x, sigma):
            return 1 / (2*np.pi * sigma**2) \
                * np.exp(-np.sum(x**2, axis=0) / (2 * sigma**2))
        sigma = hn.curve_fit(gauss2)[0][0]
        self.assertGreater(.02, np.abs(sigma - 1))

        # spline_fit
        s = hn.spline_fit()
        deltas = np.abs(s(*np.meshgrid(*hn.centers)) - hn.values)
        self.assertTrue(np.all(
            3e-2 > np.abs(s(*np.meshgrid(*hn.centers)) - hn.values)
        ))
        s = hn.spline_fit(log=True)
        deltas = np.abs(s(*np.meshgrid(*hn.centers)) - hn.values)
        self.assertTrue(np.all(
            3e-2 > np.abs(s(*np.meshgrid(*hn.centers)) - hn.values)
        ))

    @pytest.mark.filterwarnings(r'ignore:.*use hist_from_eval\(\) instead')
    def test_from_function(self):
        """
        Test hist_from_function().
        """
        from scipy import stats
        bins = [np.linspace(-3, 3, 6)]
        h = hl.hist_from_function(bins, stats.norm.pdf)
        self.assertGreater(
            eps, np.abs(h.get_value(0) - stats.norm.pdf(0))
        )

        def gauss2(x, y, sigma):
            return 1 / (2*np.pi * sigma**2) \
                * np.exp(-(x**2 + y**2) / (2 * sigma**2))

        h = hl.hist_from_function(bins+bins, gauss2, 1)
        self.assertGreater(
            eps, np.abs(h.get_value(0, 0) - gauss2(0, 0, 1))
        )

    def test_other_create(self):
        """
        Test special-purpose Hist.hist_like() and Hist.hist_direct().
        """
        np.random.seed(0)
        x = np.random.uniform(0, 1, (2, int(1e4)))
        h = hl.hist(x, bins=10, range=(0, 1))
        hd = hl.hist_direct(x, bins=10, range=2 * [(0, 1)])
        self.assertTrue(np.all(h.values == hd.values))
        self.assertTrue(np.all(h.errors == hd.errors))
        h_like_h = hl.hist_like(h, x)
        h_like_hd = hl.hist_like(hd, x)
        self.assertTrue(np.all(h_like_h.values == h.values))
        self.assertTrue(np.all(h_like_hd.values == h.values))
        self.assertTrue(np.all(h_like_h.errors == h.errors))
        self.assertTrue(np.all(h_like_hd.errors == h.errors))



if __name__ == '__main__':
    unittest.main()

