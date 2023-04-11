(install)=

# Install Guide

1. Install [Anaconda].

2. Install [cvxpy] with `conda`.

   ```
   conda install -c cvxgrp cvxpy
   ```

3. Install `CVXPortfolio` with [pip].

   ```
   pip install cvxportfolio
   ```

4. Test the installation with `nose`.

> ```
> conda install nose
> nosetests cvxportfolio
> ```

[anaconda]: https://store.continuum.io/cshop/anaconda/
[cvxpy]: https://www.cvxpy.org/
[pip]: https://pip.pypa.io/
