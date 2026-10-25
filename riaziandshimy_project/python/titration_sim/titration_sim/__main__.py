"""``python -m titration_sim`` launcher (works without PATH setup on Windows).

A friendly launcher: if PyQt6 is missing it prints an install hint in
Persian and English instead of a bare traceback.
راه‌انداز ``python -m titration_sim``؛ اگر PyQt6 نصب نباشد پیام راهنما می‌دهد.
"""
from __future__ import annotations

import sys


def _run() -> int:
    try:
        from .ui.app import main
    except ModuleNotFoundError as exc:
        if exc.name not in ("PyQt6", "pyqtgraph"):
            raise
        sys.stderr.write(
            "PyQt6/pyqtgraph is not installed. Run:\n"
            "    python -m pip install -e \".[ui]\"\n"
            "PyQt6/pyqtgraph نصب نیست؛ دستور بالا را اجرا کنید.\n"
        )
        return 2
    return main()


if __name__ == "__main__":
    sys.exit(_run())
