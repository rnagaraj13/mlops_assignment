"""
Root conftest.py – adds project root to sys.path so all imports resolve
regardless of where pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
