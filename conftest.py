"""Put the repo root on sys.path for pytest."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
