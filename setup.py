from setuptools import setup

setup(
    name='cvxportfolio',
    version='0.0.3',
    author='Enzo Busseti and Steven Diamond',
    author_email='ebusseti@stanford.edu, stevend2@stanford.edu',
    packages=['cvxportfolio',
              'cvxportfolio.tests',
              'cvxportfolio.utils'],
    package_dir={'cvxportfolio': 'cvxportfolio'},
    package_data={'cvxportfolio': ['tests/sample_data.pickle']},
    url='http://github.com/cvxgrp/cvxportfolio/',
    license='Apache',
    zip_safe=False,
    description='ConVeX Portfolio Optimization and Simulation toolset.',
    install_requires=["pandas",
                      "numpy",
                      "matplotlib",
                      "cvxpy"],
    use_2to3=True,
)
