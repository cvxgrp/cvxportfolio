from setuptools import setup

setup(
    name='cvxportfolio',
    version='0.3.3',
    author='Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.',
    maintainer='Enzo Busseti',
    author_email='enzo.busseti@gmail.com',
    packages=['cvxportfolio',
              'cvxportfolio.tests'],
    package_dir={'cvxportfolio': 'cvxportfolio'},
    package_data={'cvxportfolio': [
        'tests/returns.csv', 'tests/sigmas.csv', 'tests/volumes.csv']},
    url='https://cvxportfolio.readthedocs.io',
    license='Apache 2.0',
    description='Portfolio optimization.',
    install_requires=["pandas",
                      "numpy",
                      "matplotlib",
                      "yfinance",
                      "cvxpy>=1.0.6",
                      "multiprocess"],
)
