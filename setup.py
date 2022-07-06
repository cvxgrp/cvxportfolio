from setuptools import setup

setup(
    name='cvxportfolio',
    version='0.0.12',
    author='Enzo Busseti',
    author_email='enzo.busseti@gmail.com',
    packages=['cvxportfolio',
              'cvxportfolio.tests'],
    package_dir={'cvxportfolio': 'cvxportfolio'},
    package_data={'cvxportfolio': [
        'tests/returns.csv', 'tests/sigmas.csv', 'tests/volumes.csv']},
    url='http://github.com/cvxgrp/cvxportfolio/',
    license='Apache',
    zip_safe=False,
    description='ConVeX Portfolio Optimization and Simulation toolset.',
    install_requires=["cvxpy>=1.2.1",
                      "matplotlib",
                      "multiprocess",
                      "numpy",
                      "pandas_datareader"
                      "pandas"],
    extras_require={"dev": [
        "coveralls",
        "flake8",
        "numpydoc",
        "pytest",
        "pytest-cov",
        "sphinx",
        "twine",
    ]},
    use_2to3=True,
)
