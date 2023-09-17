from setuptools import setup

with open('README.md') as f:
    long_descr = ''.join(f.readlines())

setup(
    name='cvxportfolio',
    version='0.4.10',
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
                      "scipy",
                      "matplotlib",
                      "yfinance",
                      "cvxpy",
                      "multiprocess"
                      ],
    long_description=long_descr,
    long_description_content_type='text/markdown',
)
