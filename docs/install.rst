Installing histlite
===================

As of June, 2019, histlite is hosted on github, with releases accessible via
PyPI.  For a basic user installation, simply use:

.. code-block:: bash

    # install from PyPI
    pip install --user histlite

For a development install (i.e. you want to modify histlite yourself), navigate
to an appropriate development directory (e.g. ``$HOME/src``), and then:

.. code-block:: bash

    # using SSH
    git clone git@github.com:zgana/histlite.git
    # using HTTPS
    git clone https://github.com/zgana/histlite.git
    # finally, install
    pip install --user --editable ./histlite
