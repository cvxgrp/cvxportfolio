# Copyright (C) 2023-2024 Enzo Busseti
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
"""Find __version__ in __init__.py in file tree (BFS) and upgrade.

Additionally, look for version string in any setup.py, pyproject.toml,
and conf.py and do the same. Version is in the format X.Y.Z,
where X, Y, and Z are integers. Take argument revision (Z -> Z+1),
minor (Y -> Y+1, Z -> 0), or major (X -> X+1, Y -> 0, Z -> 0).

Add the modifications to git staging, commit with version number and
editable message (opens git configured text editor), tag with version number,
push everything to origin.
"""

import subprocess
from ast import literal_eval
from pathlib import Path


def findversion(root='.'):
    """Find version number. Skip [env, venv, .*].

    We use the first __init__.py with a __version__ that we find.

    :param root: Root folder of the project.
    :type root: pathlib.Path or str

    :returns: Found version string.
    :rtype: str
    """
    p = Path(root)

    for fname in p.iterdir():

        if fname.name == '__init__.py':
            with open(fname,  encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if '__version__' in line:
                        return literal_eval(line.split('=')[1])
        if fname.is_dir():
            if not (fname.name in ['env', 'venv'] or fname.name[0] == '.'):
                result = findversion(fname)
                if result:
                    return result

def replaceversion(new_version, version, root='.'):
    """Replace version number. Skip [env, venv, .*].

    We replace in all __init__.py, conf.py, setup.py, and pyproject.toml

    :param new_version: New version.
    :type new_version: str
    :param version: Old version.
    :type version: str
    :param root: Root folder of the project.
    :type root: pathlib.Path or str
    """

    p = Path(root)

    for fname in p.iterdir():
        if fname.name in ['__init__.py', 'conf.py', 'setup.py',
            'pyproject.toml']:

            lines = []
            with open(fname, 'rt', encoding="utf-8") as fin:
                for line in fin:
                    lines.append(line.replace(version, new_version))

            with open(fname, "wt", encoding="utf-8") as fout:
                for line in lines:
                    fout.write(line)
            subprocess.run(['git', 'add', str(fname)], check=False)

        if fname.is_dir():
            if not (fname.name in ['env', 'venv'] or fname.name[0] == '.'):
                replaceversion(new_version, version, root=fname)


if __name__ == "__main__":

    while True:
        print('[revision/minor/major]')
        WHAT = input()
        if WHAT in ['revision', 'minor', 'major']:
            break

    VERSION = findversion()

    X, Y, Z = [int(el) for el in VERSION.split('.')]
    if WHAT == 'revision':
        Z += 1
    if WHAT == 'minor':
        Y += 1
        Z = 0
    if WHAT == 'major':
        X += 1
        Y = 0
        Z = 0
    NEW_VERSION = f"{X}.{Y}.{Z}"

    print(NEW_VERSION)

    replaceversion(NEW_VERSION, VERSION)
    subprocess.run(['git', 'commit', '--no-verify', '-em',
        f"version {NEW_VERSION}\n"], check=False)
    subprocess.run(['git', 'tag', NEW_VERSION], check=False)
    subprocess.run(
        ['git', 'push', '--no-verify', 'origin', NEW_VERSION], check=False)
