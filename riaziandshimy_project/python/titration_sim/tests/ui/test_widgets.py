"""pytest-qt tests of the real widgets.

NOT executed in the sandbox where the package was written (no PyQt6 there);
run them with ``pytest tests/ui`` and send me the output.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from titration_sim.ui.burette_widget import BuretteWidget
from titration_sim.ui.colors import WATER
from titration_sim.ui.controls import ControlsPanel
from titration_sim.ui.curve_widget import CurveWidget
from titration_sim.ui.flask_widget import FlaskWidget
from titration_sim.ui.main_window import MainWindow


def test_window_starts_with_report(qtbot) -> None:  # type: ignore[no-untyped-def]
    w = MainWindow("en")
    qtbot.addWidget(w)
    assert "pH" in w.results.toPlainText()
    assert w.windowTitle() == "Titration simulator"


def test_add_titrant_moves_burette_and_curve(qtbot) -> None:  # type: ignore[no-untyped-def]
    w = MainWindow("en")
    qtbot.addWidget(w)
    w.controls.add_requested.emit(10.0)
    assert abs(w.burette.reading_ml - 10.0) < 1e-9
    x, _y = w.curve._measured.getData()
    assert len(x) == 2


def test_persian_digits_in_custom_volume(qtbot) -> None:  # type: ignore[no-untyped-def]
    w = MainWindow("fa")
    qtbot.addWidget(w)
    w.controls.custom_dv.setText("۲٫۵")
    w.controls.custom_btn.click()
    assert abs(w.session.v_added_ml - 2.5) < 1e-9


def test_invalid_input_warns_and_keeps_session(qtbot, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    w = MainWindow("en")
    qtbot.addWidget(w)
    seen: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: seen.append(str(a[-1])))
    old = w.session
    w.controls.c_analyte.setText("abc")
    assert w.apply_inputs() is False
    assert w.session is old and seen


def test_auto_run_reaches_equivalence_and_stops(qtbot) -> None:  # type: ignore[no-untyped-def]
    w = MainWindow("en")
    qtbot.addWidget(w)
    w.controls.auto_btn.click()
    qtbot.waitUntil(lambda: not w._auto.isActive(), timeout=15000)
    a = w.session.analysis()
    assert a.v_eq_found is not None and abs(a.v_eq_found - 25.0) < 0.1


def test_indicator_turns_flask_pink(qtbot) -> None:  # type: ignore[no-untyped-def]
    w = MainWindow("en")
    qtbot.addWidget(w)
    idx = w.controls.indicator.findData("Phenolphthalein")
    w.controls.indicator.setCurrentIndex(idx)
    assert w.apply_inputs()
    before = w.flask.liquid_color
    w.controls.add_requested.emit(26.0)
    assert w.flask.liquid_color != before and w.flask.liquid_color != WATER


def test_language_toggle_switches_direction(qtbot) -> None:  # type: ignore[no-untyped-def]
    w = MainWindow("fa")
    qtbot.addWidget(w)
    assert w.layoutDirection() == Qt.LayoutDirection.RightToLeft
    w.toggle_language()
    assert w.layoutDirection() == Qt.LayoutDirection.LeftToRight


def test_burette_and_flask_validation(qtbot) -> None:  # type: ignore[no-untyped-def]
    b = BuretteWidget(50.0)
    f = FlaskWidget()
    qtbot.addWidget(b)
    qtbot.addWidget(f)
    b.set_volume_added(80.0)
    assert b.reading_ml == 50.0            # clipped to the scale
    try:
        b.set_capacity(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("capacity 0 accepted")
    f.set_content(500.0, "#ff0000")        # clipped, must not raise
    b.grab()
    f.grab()


def test_curve_widget_layers(qtbot) -> None:  # type: ignore[no-untyped-def]
    c = CurveWidget()
    qtbot.addWidget(c)
    v = np.linspace(0, 30, 31)
    c.set_measured(v, 7 + np.tanh(v - 15))
    c.set_derivative(v, np.gradient(7 + np.tanh(v - 15), v))
    c.set_equivalence(25.0, 24.9)
    c.set_indicator_band(8.2, 10.0)
    c.set_indicator_band(None, None)
    c.clear_all()
    x, _ = c._measured.getData()
    assert x is None or len(x) == 0


def test_controls_read_inputs_and_enable_rules(qtbot) -> None:  # type: ignore[no-untyped-def]
    p = ControlsPanel("en")
    qtbot.addWidget(p)
    assert not p.pka.isEnabled()
    p.kind.setCurrentIndex(1)
    assert p.pka.isEnabled()
    p.pka.setText("4.76")
    assert p.read_inputs().kind == "weak_acid" and p.read_inputs().pka == "4.76"
