# histlite

See documentation on [ReadTheDocs](https://histlite.readthedocs.io/en/latest/index.html).

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
