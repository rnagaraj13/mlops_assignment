"""
Root conftest.py – adds project root to sys.path so all imports resolve
regardless of where pytest is invoked from.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
