.. _install:

Install Guide
=============


1. Install `Anaconda`_.

2. Install ``cvxpy`_ with ``conda``. 

   ::

      conda install -c cvxgrp cvxpy

3. Install ``CVXPortfolio`` with ``pip`_. 

   ::

      pip install cvxportfolio 

4. Test the installation with ``nose``.

  ::

       conda install nose
       nosetests cvxportfolio

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _cvxpy: https://www.cvxpy.org/
.. _pip: https://pip.pypa.io/
