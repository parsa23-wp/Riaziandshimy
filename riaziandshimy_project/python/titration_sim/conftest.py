"""Root conftest: skip the Qt modules during doctest collection when PyQt6/pyqtgraph cannot be imported.

A mere *spec* check is not enough: a wheel can be installed while its shared
libraries (libEGL etc.) are missing, which would crash collection.  Probe the
actual import instead.
"""
import importlib

collect_ignore_glob: list[str] = []


def _qt_importable() -> bool:
    try:
        importlib.import_module("PyQt6.QtWidgets")
        importlib.import_module("pyqtgraph")
        return True
    except Exception:  # noqa: BLE001 - any import failure means "no Qt here"
        return False


if not _qt_importable():
    collect_ignore_glob += [
        "titration_sim/ui/burette_widget.py",
        "titration_sim/ui/flask_widget.py",
        "titration_sim/ui/curve_widget.py",
        "titration_sim/ui/controls.py",
        "titration_sim/ui/main_window.py",
        "titration_sim/ui/app.py",
        "tests/ui/*",
    ]
