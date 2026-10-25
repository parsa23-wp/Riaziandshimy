"""UI tests need PyQt6 + pyqtgraph + pytest-qt; they are skipped otherwise.

The offscreen platform plugin lets them run without a display (CI, SSH).
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")
