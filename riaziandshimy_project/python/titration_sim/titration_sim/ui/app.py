"""Entry point: ``python -m titration_sim.ui.app`` or the ``titration-sim`` script."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Start the Qt event loop.

    Parameters
    ----------
    argv : list[str] | None
        Command line; ``--en`` starts in English.

    Returns
    -------
    int
        Process exit code.
    """
    from PyQt6.QtWidgets import QApplication

    from .main_window import MainWindow

    args = list(sys.argv if argv is None else argv)
    app = QApplication(args)
    win = MainWindow("en" if "--en" in args else "fa")
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
