"""Root Streamlit Entry Point for The Five-Head Story Studio.

Usage:
------
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure 'src' is in Python module search path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from five_head_nlp.ui.application import main

if __name__ == "__main__":
    main()
