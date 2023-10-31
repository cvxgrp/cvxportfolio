"""Run some example scripts and produce plots for the documentation."""
import sys

import matplotlib.pyplot as plt

EXAMPLES = ['hello_world']

for example in EXAMPLES:
    with open(f'docs/_static/{example}_output.txt', 'w') as sys.stdout:
        exec(open(f"examples/{example}.py").read())
    plt.savefig(f'docs/_static/{example}.png')
