"""Repository paths shared by every experiment script.

Importing this module puts each experiment folder on ``sys.path``, so a script
can import a module of another folder (``import ltsf_nodes``), and it exposes
the repository root, under which data, graphs, configs and results live.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "cache")

_EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
for _d in sorted(os.listdir(_EXPERIMENTS)):
    _p = os.path.join(_EXPERIMENTS, _d)
    if os.path.isdir(_p) and not _d.startswith(("_", ".")) and _p not in sys.path:
        sys.path.insert(0, _p)
