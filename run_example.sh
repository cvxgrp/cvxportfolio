#!/usr/bin/env bash
# Run it from the root of the development environment, for example
# ./run_example.sh hello_world

env CVXPORTFOLIO_SAVE_PLOTS=1 env/bin/python -m examples."$1" > docs/_static/"$1"_output.txt
mv *.png docs/_static/