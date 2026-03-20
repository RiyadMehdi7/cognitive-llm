#!/usr/bin/env python
"""Wrapper that delegates to the root train.py with all CLI arguments."""

import os
import runpy
import sys

# Add project root to path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Run the root train.py as __main__
runpy.run_path(os.path.join(project_root, "train.py"), run_name="__main__")
