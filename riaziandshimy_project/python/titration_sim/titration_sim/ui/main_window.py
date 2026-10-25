"""Main window: wires the controls, burette, flask, curve and the results panel.

All logic is in :mod:`titration_sim.ui.presenter`; this class only forwards
signals and repaints.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .burette_widget import BuretteWidget
from .colors import mix, named_color
from .controls import ControlsPanel
from .curve_widget import CurveWidget
from .flask_widget import FlaskWidget
from .presenter import AnySession, family_quantity_label
from .presenter import InputError, ModelInput, make_session
from .strings import LANGS, tr

AUTO_INTERVAL_MS = 60


class MainWindow(QMainWindow):
    """Titration simulator window.

    Parameters
    ----------
    lang : str
        ``'fa'`` (right-to-left) or ``'en'``.

    Example
    -------
    >>> app = QApplication([]); win = MainWindow(); win.show()   # doctest: +SKIP
    >>> win.close(); app.exec()                                  # doctest: +SKIP
    """

    def __init__(self, lang: str = "fa") -> None:
        super().__init__()
        self._lang = lang
        self.session: AnySession = make_session(ModelInput())
        self.controls = ControlsPanel(lang)
        self.burette = BuretteWidget(self.session.burette_capacity_ml)
        self.flask = FlaskWidget()
        self.curve = CurveWidget()
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        self._burette_title = QLabel()
        self._flask_title = QLabel()
        self._results_title = QLabel()
        self._auto = QTimer(self)
        self._auto.setInterval(AUTO_INTERVAL_MS)
        self._auto.timeout.connect(self._auto_step)
        self._build_layout()
        self._build_menu()
        self.controls.apply_requested.connect(self.apply_inputs)
        self.controls.add_requested.connect(self.add_titrant)
        self.controls.reset_requested.connect(self.restart)
        self.controls.auto_toggled.connect(self._set_auto)
        self.controls.theory_toggled.connect(lambda _on: self.refresh())
        self._apply_language()
        self._rebuild_theory()
        self.refresh()

    # ---- layout --------------------------------------------------------
    def _build_layout(self) -> None:
        mid = QWidget()
        ml = QHBoxLayout(mid)
        bcol = QVBoxLayout()
        bcol.addWidget(self._burette_title)
        bcol.addWidget(self.burette, 1)
        fcol = QVBoxLayout()
        fcol.addWidget(self._flask_title)
        fcol.addWidget(self.flask, 1)
        ml.addLayout(bcol)
        ml.addLayout(fcol, 1)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.curve, 3)
        rl.addWidget(self._results_title)
        rl.addWidget(self.results, 1)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.controls)
        split.addWidget(mid)
        split.addWidget(right)
        split.setStretchFactor(2, 3)
        self.setCentralWidget(split)
        self.resize(1250, 760)

    def _build_menu(self) -> None:
        self._lang_action = QAction(self)
        self._lang_action.triggered.connect(self.toggle_language)
        mb = self.menuBar()
        if mb is not None:
            mb.addAction(self._lang_action)

    # ---- language --------------------------------------------------------
    def _apply_language(self) -> None:
        lang = self._lang
        self.setWindowTitle(tr("title", lang))
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft if lang == "fa" else Qt.LayoutDirection.LeftToRight)
        self._burette_title.setText(tr("burette", lang))
        self._flask_title.setText(tr("flask", lang))
        self._results_title.setText(tr("results", lang))
        self._lang_action.setText(tr("lang", lang))
        self.controls.retranslate(lang)

    def toggle_language(self, *_args: object) -> None:
        """Switch between Persian (RTL) and English (LTR)."""
        self._lang = LANGS[(LANGS.index(self._lang) + 1) % len(LANGS)]
        self._apply_language()
        self.refresh()

    # ---- actions ---------------------------------------------------------
    def apply_inputs(self) -> bool:
        """Validate the text fields and start a new session.

        Returns
        -------
        bool
            False (and a message box) if an input is invalid; the old session is kept.
        """
        self._set_auto(False)
        try:
            session = make_session(self.controls.read_inputs())
        except InputError as exc:
            QMessageBox.warning(self, tr("input_error", self._lang), str(exc))
            return False
        self.session = session
        self.burette.set_capacity(session.burette_capacity_ml)
        self._rebuild_theory()
        self.refresh()
        return True

    def restart(self) -> None:
        """Refill the burette: same model, V = 0."""
        self._set_auto(False)
        self.session.reset()
        self.refresh()

    def add_titrant(self, d_v_ml: float) -> None:
        """Run titrant out of the burette (animates a drop for small additions)."""
        delivered = self.session.add(d_v_ml)
        if delivered == 0.0:
            self._set_auto(False)
        self.refresh(animate_drop=delivered > 0.0 and d_v_ml <= 0.2)

    def _set_auto(self, running: bool) -> None:
        if running and not self.session.burette_empty:
            self._auto.start()
        else:
            self._auto.stop()
            self.controls.stop_auto()

    def _auto_step(self) -> None:
        eq_last = self.session.v_eq_last()
        if self.session.v_added_ml >= 1.4 * eq_last or self.session.burette_empty:
            self._set_auto(False)
            return
        self.add_titrant(self.session.suggest_step_ml())

    # ---- drawing ---------------------------------------------------------
    def _rebuild_theory(self) -> None:
        v, y = self.session.theory_curve()
        self.curve.set_theory(v, y, self.controls.theory_chk.isChecked())
        self.curve.set_quantity(self._quantity_label())
        family = getattr(self.session, "family", "acid_base")
        if family == "back":
            self.curve.clear_all()
            return
        ind = getattr(self.session, "indicator", None)
        if ind is None:
            self.curve.set_indicator_band(None, None)
            return
        if hasattr(ind, "acid_color"):          # acid-base Indicator (pH band)
            from ..engine.indicators import Indicator as _Ind
            assert isinstance(ind, _Ind)
            col = mix(named_color(ind.acid_color), named_color(ind.base_color), 1.0)
            self.curve.set_indicator_band(ind.lo, ind.hi, col)
            return
        # phase-2 FamilyIndicator: narrow horizontal band at the transition
        half = {"redox": 0.02, "precip": 0.15, "edta": 0.15}.get(family, 0.1)
        from .presenter import FamilyIndicator as _FI
        assert isinstance(ind, _FI)
        col = mix(ind.below, ind.above, 0.5)
        self.curve.set_indicator_band(ind.transition - half, ind.transition + half, col)

    def _quantity_label(self) -> str:
        from .strings import tr
        family = getattr(self.session, "family", "acid_base")
        return tr(family_quantity_label(family), self._lang)

    def refresh(self, animate_drop: bool = False) -> None:
        """Repaint every view from the session."""
        s = self.session
        self.burette.set_volume_added(s.v_added_ml, animate_drop)
        self.flask.set_content(s.state.volume_ml, s.flask_color(), s.state.ph,
                               label=self._quantity_label())
        v, y = s.measured()
        self.curve.set_measured(v, y)
        dv, d = s.derivative()
        self.curve.set_derivative(dv, d)
        self.curve.set_theory_visible(self.controls.theory_chk.isChecked())
        self.curve.set_quantity(self._quantity_label())
        a = s.analysis()
        self.curve.set_equivalence(a.v_eq_theory, a.v_eq_found)
        self.results.setPlainText(s.report(self._lang))
