import os
import subprocess
import requests
import json
import cvxportfolio
from distutils.version import StrictVersion

# On Travis, you should set $CONDA_UPLOAD_TOKEN, $PYPI_USER, and $PYPI_PASSWORD

CHANNEL = "cvxgrp"
BUILDDIR = 'build'
ARCH = json.load(subprocess.Popen("conda info --json", shell=True,
                                  stdout=subprocess.PIPE).stdout)['platform']
LOCAL_VERSION = cvxportfolio.__version__


def pypi_version(test=False):
    if test:
        url = "https://test.pypi.org/pypi/cvxportfolio/json"
    else:
        url = "https://pypi.org/pypi/cvxportfolio/json"
    r = requests.get(url)
    data = r.json()
    versions = list(data["releases"].keys())
    versions.sort(key=StrictVersion)
    return versions[-1]


PYPI_VERSION = pypi_version()

print('local version:', LOCAL_VERSION)
print('pypi version:', PYPI_VERSION)

# if (LOCAL_VERSION == PYPI_VERSION):
#     print("Versions match, skipping build.")
#     exit(0)

subprocess.call(["conda", "config", "--add", "channels", "cvxgrp"])
subprocess.call(["conda", "config", "--add", "channels", "conda-forge"])

if not (subprocess.call(["conda", "build",
                         "--variant-config-files=conda-recipe/conda_build_config.yaml",
                         "--output-folder=build",
                         "conda-recipe/meta.yaml"]) == 0):
    print('build failed')
    exit(1)

LAST_BUILD_NAME_Py36 = [el for el in sorted(os.listdir(
    '%s/%s' % (BUILDDIR, ARCH))) if el[:5] == 'cvxpo' and el[-14:] == "py36_0.tar.bz2"][-1]

# LAST_BUILD_NAME_Py27 = [el for el in sorted(os.listdir(
#     '%s/%s' % (BUILDDIR, ARCH))) if el[:5] == 'cvxpo' and el[-14:] == "py27_0.tar.bz2"][-1]

for LAST_BUILD_NAME in [LAST_BUILD_NAME_Py36]:  # , LAST_BUILD_NAME_Py27]:
    for NEWARCH in ['osx-64', 'win-32', 'win-64', 'linux-32', 'linux-64']:
        subprocess.call(["conda", "convert",
                         "--platform=%s" % NEWARCH,
                         "%s/%s/%s" % (BUILDDIR, ARCH, LAST_BUILD_NAME),
                         "-o=%s" % BUILDDIR])
        subprocess.call(["anaconda", "upload", "--force",
                         "--user=%s" % CHANNEL] +
                        ["--token=$CONDA_UPLOAD_TOKEN"] +
                        ["%s/%s/%s" % (BUILDDIR, NEWARCH, LAST_BUILD_NAME)])

# pypi
subprocess.call(["python", "setup.py", "sdist"])
subprocess.call(["twine", "upload", "--skip-existing"] +
                ["-u $PYPI_USER"] +
                ["-p $PYPI_PASSWORD"] +
                ["dist/*"])
