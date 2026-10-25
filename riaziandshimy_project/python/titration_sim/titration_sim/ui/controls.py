"""Controls panel: family, model inputs, burette buttons, indicator, options."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..engine.indicators import load_indicators
from .presenter import (
    FAMILY_INDICATORS, FAMILIES, ModelInput, parse_number,
)
from .strings import tr

_KINDS = ("strong_acid", "weak_acid", "strong_base", "weak_base")
_TITRANT = ("strong", "weak")

#: fields shown per family (besides c_analyte / v_analyte / c_titrant, common to all)
_FAMILY_FIELDS: dict[str, tuple[str, ...]] = {
    "acid_base": ("pka", "titrant_pka", "temperature", "pkw"),
    "redox": ("e1", "n1", "e2", "n2", "redox_ph", "m1", "m2", "ionic_strength"),
    "precip": ("ksp", "ionic_strength"),
    "edta": ("log_kf", "edta_ph", "aux_conc", "aux_metal", "ionic_strength"),
    "back": ("c_reagent", "v_reagent", "c_back_titrant", "v_back_titrant",
             "v_sample", "ratio_ra", "ratio_tr"),
}

_FAMILY_LABEL: dict[str, str] = {f: f"family_{f}" for f in FAMILIES}

#: phase-2 indicator combo keys (acid_base uses the Indicator table instead)
_GENERIC_NO_INDICATOR = "no_indicator_generic"


class ControlsPanel(QWidget):
    """Left-hand panel.  Emits signals; owns no chemistry.

    Signals
    -------
    apply_requested
        Text fields should be validated and a new session started.
    add_requested(float)
        Run this many mL out of the burette.
    reset_requested
    auto_toggled(bool)
    theory_toggled(bool)
    """

    apply_requested = pyqtSignal()
    add_requested = pyqtSignal(float)
    reset_requested = pyqtSignal()
    auto_toggled = pyqtSignal(bool)
    theory_toggled = pyqtSignal(bool)

    def __init__(self, lang: str = "fa", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lang = lang
        self._rows: dict[str, int] = {}
        self._build()
        self.retranslate(lang)

    # ---- construction --------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        self.model_box = QGroupBox()
        form = QFormLayout(self.model_box)
        self.family = QComboBox()
        self.kind = QComboBox()
        self.titrant = QComboBox()
        self.c_analyte = QLineEdit("0.1")
        self.v_analyte = QLineEdit("25")
        self.c_titrant = QLineEdit("0.1")
        self.pka = QLineEdit()
        self.titrant_pka = QLineEdit()
        self.temperature = QLineEdit("25")
        self.pkw = QLineEdit()
        self.indicator = QComboBox()
        # ---- phase-2 fields ------------------------------------------------
        self.e1 = QLineEdit()
        self.n1 = QLineEdit()
        self.e2 = QLineEdit()
        self.n2 = QLineEdit()
        self.redox_ph = QLineEdit()
        self.m1 = QLineEdit()
        self.m2 = QLineEdit()
        self.ionic_strength = QLineEdit()
        self.ksp = QLineEdit()
        self.log_kf = QLineEdit()
        self.edta_ph = QLineEdit()
        self.aux_conc = QLineEdit()
        self.aux_metal = QLineEdit()
        self.c_reagent = QLineEdit()
        self.v_reagent = QLineEdit()
        self.c_back_titrant = QLineEdit()
        self.v_back_titrant = QLineEdit()
        self.v_sample = QLineEdit()
        self.ratio_ra = QLineEdit("1")
        self.ratio_tr = QLineEdit("1")
        self.fields: dict[str, QLineEdit] = {
            "c_analyte": self.c_analyte, "v_analyte": self.v_analyte,
            "c_titrant": self.c_titrant, "pka": self.pka,
            "titrant_pka": self.titrant_pka, "temperature": self.temperature,
            "pkw": self.pkw, "e1": self.e1, "n1": self.n1, "e2": self.e2,
            "n2": self.n2, "redox_ph": self.redox_ph, "m1": self.m1,
            "m2": self.m2, "ionic_strength": self.ionic_strength,
            "ksp": self.ksp, "log_kf": self.log_kf, "edta_ph": self.edta_ph,
            "aux_conc": self.aux_conc, "aux_metal": self.aux_metal,
            "c_reagent": self.c_reagent, "v_reagent": self.v_reagent,
            "c_back_titrant": self.c_back_titrant,
            "v_back_titrant": self.v_back_titrant, "v_sample": self.v_sample,
            "ratio_ra": self.ratio_ra, "ratio_tr": self.ratio_tr,
        }
        self.labels: dict[str, QLabel] = {}
        rows: list[tuple[str, QWidget]] = [
            ("family", self.family), ("analyte", self.kind), ("titrant", self.titrant),
            ("c_analyte", self.c_analyte), ("v_analyte", self.v_analyte),
            ("c_titrant", self.c_titrant), ("pka", self.pka),
            ("titrant_pka", self.titrant_pka), ("temperature", self.temperature),
            ("pkw", self.pkw), ("indicator", self.indicator),
        ]
        rows += [(k, self.fields[k]) for k in
                 ("e1", "n1", "e2", "n2", "redox_ph", "m1", "m2", "ionic_strength",
                  "ksp", "log_kf", "edta_ph", "aux_conc", "aux_metal",
                  "c_reagent", "v_reagent", "c_back_titrant", "v_back_titrant",
                  "v_sample", "ratio_ra", "ratio_tr")]
        for key, widget in rows:
            lab = QLabel()
            self.labels[key] = lab
            form.addRow(lab, widget)
            self._rows[key] = form.rowCount() - 1
        self.family.currentIndexChanged.connect(self._on_family_changed)
        self.kind.currentIndexChanged.connect(self._sync_enabled)
        self.titrant.currentIndexChanged.connect(self._sync_enabled)
        self.apply_btn = QPushButton()
        self.apply_btn.clicked.connect(lambda *_a: self.apply_requested.emit())
        self.reset_btn = QPushButton()
        self.reset_btn.clicked.connect(lambda *_a: self.reset_requested.emit())
        root.addWidget(self.model_box)
        root.addWidget(self.apply_btn)

        self.burette_box = QGroupBox()
        bl = QVBoxLayout(self.burette_box)
        row = QHBoxLayout()
        self.step_buttons: list[tuple[QPushButton, float, str]] = []
        for key, dv in (("drop", 0.05), ("add_01", 0.1), ("add_1", 1.0), ("add_5", 5.0)):
            b = QPushButton()
            b.clicked.connect(lambda _checked=False, d=dv: self.add_requested.emit(d))
            row.addWidget(b)
            self.step_buttons.append((b, dv, key))
        bl.addLayout(row)
        crow = QHBoxLayout()
        self.custom_label = QLabel()
        self.custom_dv = QLineEdit("0.5")
        self.custom_btn = QPushButton()
        self.custom_btn.clicked.connect(self._emit_custom)
        crow.addWidget(self.custom_label)
        crow.addWidget(self.custom_dv)
        crow.addWidget(self.custom_btn)
        bl.addLayout(crow)
        self.auto_btn = QPushButton()
        self.auto_btn.setCheckable(True)
        self.auto_btn.toggled.connect(self._on_auto)
        bl.addWidget(self.auto_btn)
        bl.addWidget(self.reset_btn)
        self.theory_chk = QCheckBox()
        self.theory_chk.setChecked(True)
        self.theory_chk.toggled.connect(self.theory_toggled.emit)
        bl.addWidget(self.theory_chk)
        root.addWidget(self.burette_box)
        root.addStretch(1)
        self._sync_enabled()
        self._sync_family_rows()

    # ---- behaviour ----------------------------------------------------
    def _emit_custom(self, *_args: object) -> None:
        try:
            dv = parse_number(self.custom_dv.text())
        except ValueError:
            return
        if dv > 0.0:
            self.add_requested.emit(dv)

    def _on_auto(self, running: bool) -> None:
        self.auto_btn.setText(tr("auto_stop" if running else "auto_start", self._lang))
        self.auto_toggled.emit(running)

    def stop_auto(self) -> None:
        """Release the auto-run button without emitting a second signal."""
        if self.auto_btn.isChecked():
            self.auto_btn.blockSignals(True)
            self.auto_btn.setChecked(False)
            self.auto_btn.blockSignals(False)
            self.auto_btn.setText(tr("auto_start", self._lang))

    def _current_family(self) -> str:
        key = self.family.currentData()
        return str(key) if key else "acid_base"

    def _on_family_changed(self, *_args: object) -> None:
        self._sync_family_rows()
        self._repopulate_indicators(keep_current=False)
        self._sync_enabled()

    def _sync_family_rows(self) -> None:
        """Show only the rows of the selected family (+ common + indicator)."""
        form = self.model_box.layout()
        assert isinstance(form, QFormLayout)
        fam = self._current_family()
        shown = {"family", "c_analyte", "v_analyte", "c_titrant", "indicator"}
        shown |= set(_FAMILY_FIELDS.get(fam, ()))
        if fam == "acid_base":
            shown |= {"analyte", "titrant"}
        for key, row in self._rows.items():
            form.setRowVisible(row, key in shown)
        self.burette_box.setVisible(fam != "back")

    def _sync_enabled(self, *_args: object) -> None:
        weak_analyte = _KINDS[self.kind.currentIndex()] in ("weak_acid", "weak_base")
        self.pka.setEnabled(weak_analyte)
        self.titrant_pka.setEnabled(_TITRANT[self.titrant.currentIndex()] == "weak")

    def _repopulate_indicators(self, keep_current: bool = True) -> None:
        """Fill the indicator combo with the entries of the current family."""
        cur = self.indicator.currentData() if keep_current else None
        self.indicator.clear()
        fam = self._current_family()
        if fam == "acid_base":
            self.indicator.addItem(tr("no_indicator", self._lang), "")
            for ind in load_indicators():
                self.indicator.addItem(ind.name, ind.name)
        elif fam in FAMILY_INDICATORS:
            self.indicator.addItem(tr(_GENERIC_NO_INDICATOR, self._lang), "")
            for fi in FAMILY_INDICATORS[fam]:
                self.indicator.addItem(fi.name, fi.name)
        if cur is not None:
            i = self.indicator.findData(cur)
            if i >= 0:
                self.indicator.setCurrentIndex(i)

    def read_inputs(self) -> ModelInput:
        """Current text fields as a :class:`ModelInput` (validation happens in the presenter)."""
        ind = self.indicator.currentData()
        kwargs: dict[str, object] = {
            "family": self._current_family(),
            "c_analyte": self.c_analyte.text(),
            "v_analyte": self.v_analyte.text(),
            "c_titrant": self.c_titrant.text(),
            "indicator": str(ind) if ind else "",
            "burette_capacity": "50",
        }
        for key, field in self.fields.items():
            if key not in ("c_analyte", "v_analyte", "c_titrant"):
                kwargs[key] = field.text()
        if kwargs["family"] == "acid_base":
            kwargs["kind"] = _KINDS[self.kind.currentIndex()]
            kwargs["titrant_strength"] = _TITRANT[self.titrant.currentIndex()]
        return ModelInput(**kwargs)  # type: ignore[arg-type]

    def retranslate(self, lang: str) -> None:
        """Set all texts in ``lang`` ('fa' or 'en'), keeping the current selections."""
        self._lang = lang
        k, t = self.kind.currentIndex(), self.titrant.currentIndex()
        fam_i = self.family.currentIndex()
        self.family.clear()
        for key in FAMILIES:
            self.family.addItem(tr(_FAMILY_LABEL[key], lang), key)
        self.kind.clear()
        for key in _KINDS:
            self.kind.addItem(tr(f"kind_{key}", lang), key)
        self.titrant.clear()
        for key in _TITRANT:
            self.titrant.addItem(tr(f"titrant_{key}", lang), key)
        self.family.setCurrentIndex(max(0, fam_i))
        self.kind.setCurrentIndex(max(0, k))
        self.titrant.setCurrentIndex(max(0, t))
        self._repopulate_indicators(keep_current=True)
        for key, lab in self.labels.items():
            lab.setText(tr(key, lang))
        self.model_box.setTitle(tr("analyte", lang) + " / " + tr("titrant", lang))
        self.burette_box.setTitle(tr("burette", lang))
        self.apply_btn.setText(tr("apply", lang))
        self.reset_btn.setText(tr("reset", lang))
        for b, _dv, key in self.step_buttons:
            b.setText(tr(key, lang))
        self.custom_label.setText(tr("custom_dv", lang))
        self.custom_btn.setText(tr("add_custom", lang))
        self.auto_btn.setText(tr("auto_stop" if self.auto_btn.isChecked() else "auto_start", lang))
        self.theory_chk.setText(tr("show_theory", lang))
        self._sync_family_rows()
