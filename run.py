"""
One-Command Runner for Road Safety ML Greece (`run.py`)
=======================================================
Run simply with:
    python run.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from road_safety.train import main

if __name__ == "__main__":
    main()
