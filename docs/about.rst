About this project
==================

Introduction
------------

**histlite** is a histogram calculation and plotting library that tries to be
"lite" on data structures but rich in statistics and visualization features.
So far, development has taken place during my (Mike Richman) time as a graduate
student and post-doctoral researcher in the field of particle astrophysics ---
specifically, working with the IceCube Neutrino Observatory.  Histlite is
intended both to facilitate high-paced exploratory data analysis as well as to
serve as a building block for potentially very complex maximum likelihood data
analysis implementations.

The core design considerations are:

* It must be trivial to work with and interchange between 1D, 2D, or ND histograms.
* It should be as simple as possible to perform bin-wise arithmetic
  operations on one or more histograms; to perform sums, integrals, etc. and
  thus normalizations along one or more axes simultaneously; and to perform
  spline or user-defined functional fits
* It should be as simple as possible to achieve publication-quality plots.

The primary histogramming functionality consists of a thin wrapper around
:meth:`numpy.histogramdd`.  Statistical tools leverage **scipy** but include
custom solutions for some use cases.  (Importantly, error propagation is
currently handled manually but may be migrated to the **uncertainties**
package in the future.)  Plotting is done using **matplotlib**.

Motivation
----------

Many problems in statistics involve understanding the behavior of
distributions of observables.  Histograms are one of the best-understood and
most commonly used tools for discerning the properties of those
distributions.  A key reason for preferring histograms over other tools such
as KDE is that per-bin error (or *uncertainty*) estimates are robust and
straightforward to reason about.

Numpy is a fantastic and an extremely general numerical library.  Scipy
provides many excellent statistical methods.  Matplotlib offers beautiful
and highly flexible visualizations suitable for publication-quality plots.
However, as far as I know, to date there is no standard, widely used
histogramming package for synthesizing functionality from these baseline
tools.

Numpy provides reasonable (if not particularly optimal) methods for
producing histograms in the form of (``values,edges``).  Matplotlib is
capable of plotting histograms, but that is about as far as it goes.  For
applications that require publication-quality plots only *after* non-trivial
histogram combinations or manipulations, matplotlib offers little support.
Scipy methods are happy to operate on bin centers and weights, but you'll
have to keep track of those yourself.

This package is my attempt to simplify histogram calculation and plotting
operations, primarily by bringing together functionality that is currently
spread across the scientific Python software stack.

Alternatives
------------

Alternatives include at least the following:

* **plain numpy/matplotlib/scipy**: You can get done what you want to get
  done, but you'll almost certainly violate the DRY (don't repeat yourself)
  guideline.
* **physt**: I only very recently became aware of this library which seeks
  to solve generally the same problems.  Perhaps someday we will join
  forces.  What I gather from the physt documentation is that it places much
  more emphasis on flexible (even optionally automatically-optimized)
  binning, but much less emphasis on working with histograms as probability
  density
  functions (PDFs) as required by much of my work.
* **ROOT**: Many particle physicists use this CERN product for data
  analysis, but... to be brief, even with its Python bindings it will remain
  unacceptable for serious work, especially but not only outside of particle
  physics, until it is rewritten or replaced in its entirety.


Features
--------

* **ND histograms**: all histograms are N-dimensional, which keeps things
  simple and generic in the case of operations that involve mixed
  dimensionality (the same as numpy broadcasting).
* **easy, flexible plotting**: we don't cover every use case, but you have
  sane defaults and fine-grained control over 1D and 2D histogram
  presentation.
* **histogram arithmetic**: histograms can be added, subtracted, multiplied,
  divided, log'd, sqrt'd, raised to powers, etc.
* **normalizations and projections**: histograms can be normalized to
  integrate to 1; integrate to 1 accounting for varying bin volume
  (``density=True``); or sum to (``integrate=False``).  Histograms can be
  projected to lower dimensionality by integrating or summing over one or
  more axes; or by computing quantiles along a specified axis; or by finding
  a specified containment interval along a specified axis, centered on the
  median and expressing the containment interval as errorbars.
* **error/uncertainty tracking** per-bin errors are tracked in all but a
  handful of cases where their meaning is unclear (e.g. uncertainty on a
  quantile computed from some oother histogram).  Errors are propagated in
  the standard way under histogram arithmetic.
* **simple** ("lite") **data model**: a :class:`histlite.Hist` is really
  just any curve with both values and errors defined spanning some specified
  bins.  You don't need to think about whether ``(h1 - h2) / (h1 + h2)`` is
  a histogram or some other thing; all the same methods, operators, and
  plotting functions will work the same.
* **fitting interface**: 1D and 2D (ND still a work in progress) splines can
  be computed easily, producing :class:`histlite.SplineFit` functors
  that know whether the underlying scipy splines were fitted in linear or
  log space on each axis.  For convenience, arbitrary functions can be fit
  analogously to :meth:`scipy.optimize.curve_fit`, using that function
  internally.
* **evaluation interface**: :class:`histlite.Hist`'s can be generated by
  evaluating arbitrary functions, or instances of :class:`histlite.Hist`
  or :class:`histlite.SplineFit`, on a specified grid.  The resulting
  :class:`histlite.Hist`'s can then of course be used in histogram
  arithmetic or plotting methds.
* **smoothing**: histograms can be smoothed using
  :meth:`histlite.Hist.gaussian_filter` or
  :meth:`histlite.Hist.gaussian_filter1d`.  Smooth "histograms" can be used
  to approximate Kernel Density Estimation (KDE) using :meth:`histlite.kde`.
