"""Dedicated console-script entry point for the Five-Head Story Studio Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch Streamlit web application via Streamlit's official CLI interface."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parent / "application.py"
    sys.argv = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
