"""Look for __version__ in any __init__.py recursively and upgrade.

Additionally, look for version string in any setup.py and (Sphinx) conf.py 
and do the same. Version is assumed to be in the format X.Y.Z, where X, Y, and Z
are integers. Take argument revision (Z -> Z+1), minor (Y -> Y+1, Z -> 0), or major
(X -> X+1, Y -> 0, Z -> 0). Return the updated version string."""

from pathlib import Path
import subprocess

def findversion(root='.'):
    """Find version number. Skip [env, venv, .*].
    
    We use the first __init__.py with a __version__ that we find.
    """
    p = Path(root)
    
    for fname in p.iterdir():

        if fname.name == '__init__.py':
            with open(fname) as f:
                lines = f.readlines()
                for line in lines:
                    if '__version__' in line:
                        return eval(line.split('=')[1])
        if fname.is_dir():
            if not (fname.name in ['env', 'venv'] or fname.name[0] == '.'):
                result = findversion(fname)
                if result:
                    return result
        

def replaceversion(new_version, version, root='.'):
    """Replace version number. Skip [env, venv, .*].
    
    We replace in all __init__.py, conf.py, and setup.py.
    """    

    p = Path(root)
    
    for fname in p.iterdir():
        if fname.name in ['__init__.py', 'conf.py', 'setup.py']:
            
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
    
    x,y,z = [int(el) for el in version.split('.')]
    if what == 'revision':
        z += 1
    if what == 'minor':
        y += 1
        z = 0
    if what == 'major':
        x += 1   
        y = 0
        z = 0
    new_version = f'{x}.{y}.{z}'

    print(new_version) 

    replaceversion(new_version, version)
    subprocess.run(['git', 'commit', '-em', f"version {new_version}\n"])
    subprocess.run(['git', 'tag', new_version])
    subprocess.run(['git', 'push', 'origin', new_version])
    