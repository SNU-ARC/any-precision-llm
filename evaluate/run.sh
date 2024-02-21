#!/bin/bash

# No additional dependencies are required for this script.
# Uncomment line 17 in `run_eval.py` to evaluate on baseline models too.

# Results will be dumped to `results.json` in the current directory.

PYTHONPATH=.. CUDA_VISIBLE_DEVICES=0 python run_eval.py

