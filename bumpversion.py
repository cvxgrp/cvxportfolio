# Copyright 2023 Enzo Busseti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Find __version__ in __init__.py in file tree (BFS) and upgrade.

Additionally, look for version string in any setup.py, pyproject.toml,
and conf.py and do the same. Version is in the format X.Y.Z, 
where X, Y, and Z are integers. Take argument revision (Z -> Z+1),
minor (Y -> Y+1, Z -> 0), or major (X -> X+1, Y -> 0, Z -> 0).

Add the modifications to git staging, commit with version number and 
editable message (opens git configured text editor), tag with version number,
push everything to origin."""

from ast import literal_eval
import subprocess
from pathlib import Path


def findversion(root='.'):
    """Find version number. Skip [env, venv, .*].
    
    We use the first __init__.py with a __version__ that we find.

    :param root: Root folder of the project.
    :type root: Pathlib.Path or str

    :raises ValueError: No version found.

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

    raise ValueError('Not found any!')


def replaceversion(new_version, version, root='.'):
    """Replace version number. Skip [env, venv, .*].
    
    We replace in all __init__.py, conf.py, setup.py, and pyproject.toml
    """

    p = Path(root)

    for fname in p.iterdir():
        if fname.name in ['__init__.py', 'conf.py', 'setup.py',
            'pyproject.toml']:

            lines = []
            with open(fname, 'rt') as fin:
                for line in fin:
                    lines.append(line.replace(version, new_version))

            with open(fname, "wt") as fout:
                for line in lines:
                    fout.write(line)
            subprocess.run(['git', 'add', str(fname)])

        if fname.is_dir():
            if not (fname.name in ['env', 'venv'] or fname.name[0] == '.'):
                replaceversion(new_version, version, root=fname)


if __name__ == "__main__":

    while True:
        print('[revision/minor/major]')
        what = input()
        if what in ['revision', 'minor', 'major']:
            break

    version = findversion()

    x, y, z = [int(el) for el in version.split('.')]
    if what == 'revision':
        z += 1
    if what == 'minor':
        y += 1
        z = 0
    if what == 'major':
        x += 1
        y = 0
        z = 0
    new_version = f"{x}.{y}.{z}"

    print(new_version)

    replaceversion(new_version, version)
    subprocess.run(['git', 'commit', '--no-verify', '-em',
        f"version {new_version}\n"])
    subprocess.run(['git', 'tag', new_version])
    subprocess.run(['git', 'push', '--no-verify', 'origin', new_version])
