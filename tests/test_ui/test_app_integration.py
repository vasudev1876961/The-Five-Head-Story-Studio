"""End-to-end integration test using Streamlit AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


class TestAppIntegration:
    """Verifies that app.py compiles, renders, and initializes headlessly without errors."""

    def test_app_loads_and_renders_cleanly(self):
        root_dir = Path(__file__).resolve().parents[2]
        app_path = root_dir / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run(timeout=20)

        assert not at.exception
        assert len(at.error) == 0

        # Verify page header rendered
        assert len(at.markdown) > 0
        header_rendered = any("The Five-Head Story Studio" in m.value for m in at.markdown)
        assert header_rendered is True

        # Verify navigation radio buttons rendered
        assert len(at.radio) > 0
