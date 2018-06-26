## How to upload to pip

- change the version number in setup.py
- `python setup.py bdist_wheel`
- `python setup.py sdist`, needed for conda
- `twine upload dist/*`
- steps taken from
  [https://packaging.python.org/tutorials/distributing-packages/](https://packaging.python.org/tutorials/distributing-packages/)
