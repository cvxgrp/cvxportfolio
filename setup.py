from setuptools import setup

setup(
    name='cvx_portfolio',
    version='0.0.1',
    author='Enzo Busseti and Steven Diamond',
    author_email='ebusseti@stanford.edu, stevend2@stanford.edu',
    packages=['cvx_portfolio',
              'cvx_portfolio.tests',
              'cvx_portfolio.utils'],
    package_dir={'cvx_portfolio': 'cvx_portfolio'},
    url='http://github.com/cvxgrp/cvx_portfolio/',
    license='Apache',
    zip_safe=False,
    description='A library for optimal portfolio construction and simulation.',
    install_requires=["pandas",
                      "pandas_datareader",
                      "matplotlib",
                      "cvxpy"],
    use_2to3=True,
)
