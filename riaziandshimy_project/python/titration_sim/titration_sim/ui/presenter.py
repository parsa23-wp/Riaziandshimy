"""Qt-free presentation layer of the desktop UI.

All behaviour of the window that is *not* painting lives here, so it is
unit-tested without a display: parsing of (Persian) numbers, building the
model from text fields, the burette/flask state, the indicator colour, the
analysis of the points collected so far and the text report.

The session never keeps chemistry state of its own: every point is
``state_after = engine.titrate(model, state_before, dV)`` ([MATH.md section 0]).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

from ..engine.acid_base import (
    AcidBaseEngine,
    AcidBaseModel,
    AnalyteKind,
    TitrantStrength,
    FeasibilityReport,
    check_visual_endpoint,
    equivalence_volumes,
    generate_curve,
    ph_at,
)
from ..engine.analysis import find_equivalence, first_derivative
from ..engine.base import TitrationModel, TitrationState
from ..engine.indicators import Indicator, endpoint_ph, endpoint_volume, load_indicators
from ..engine.phase2 import (
    NH3_LOG_BETA, BackTitrationResult, EdtaModel, PrecipitationModel, RedoxModel,
    back_titration, edta_equivalence_volume, edta_pm, precip_equivalence_volume,
    precip_pag, redox_equivalence_potential, redox_equivalence_volume,
)
from ..engine.phase3 import quantity_of
from .colors import WATER, indicator_color
from .strings import tr

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


class InputError(ValueError):
    """Invalid user input; ``field`` names the offending field key."""

    def __init__(self, field_key: str, message: str) -> None:
        super().__init__(message)
        self.field_key = field_key


def parse_number(text: str) -> float:
    """Parse a number typed by the user.

    Accepts Persian/Arabic-Indic digits, the Persian decimal separator ``٫``,
    a decimal comma when there is no dot (``0,1``), and surrounding spaces.

    Raises
    ------
    ValueError
        If the text is empty, not a number, NaN or infinite.

    Example
    -------
    >>> parse_number("۰٫۱۲۵")
    0.125
    >>> parse_number("2,5")
    2.5
    """
    s = text.strip().translate(_DIGITS).replace("٫", ".").replace("−", "-").replace("٬", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    if not s:
        raise ValueError("empty number")
    x = float(s)
    if not math.isfinite(x):
        raise ValueError("number must be finite")
    return x


def parse_number_list(text: str) -> tuple[float, ...]:
    """Parse several numbers separated by spaces, ``;``, ``،`` or ``/``.

    Example
    -------
    >>> parse_number_list("2.15 ; 7.20 ۱۲٫۳۵")
    (2.15, 7.2, 12.35)
    """
    parts = [p for p in re.split(r"[;\s،/]+", text.strip()) if p]
    return tuple(parse_number(p) for p in parts)


@dataclass
class ModelInput:
    """Raw text fields of the controls panel.

    Example
    -------
    >>> ModelInput().kind
    'strong_acid'
    """

    kind: str = "strong_acid"
    titrant_strength: str = "strong"
    c_analyte: str = "0.1"
    v_analyte: str = "25"
    c_titrant: str = "0.1"
    pka: str = ""
    titrant_pka: str = ""
    temperature: str = "25"
    pkw: str = ""
    indicator: str = ""
    burette_capacity: str = "50"
    # ---- phase-2 families (v0.8) -----------------------------------------
    family: str = "acid_base"
    # redox
    e1: str = ""
    n1: str = ""
    e2: str = ""
    n2: str = ""
    redox_ph: str = ""
    m1: str = ""
    m2: str = ""
    ionic_strength: str = ""
    # precipitation
    ksp: str = ""
    # complexometry
    log_kf: str = ""
    edta_ph: str = ""
    aux_conc: str = ""
    aux_metal: str = ""
    # back titration (calculator)
    c_reagent: str = ""
    v_reagent: str = ""
    c_back_titrant: str = ""
    v_back_titrant: str = ""
    v_sample: str = ""
    ratio_ra: str = "1"
    ratio_tr: str = "1"


def _num(field_key: str, text: str, positive: bool = True) -> float:
    try:
        x = parse_number(text)
    except ValueError as exc:
        raise InputError(field_key, f"{field_key}: not a number ({text!r})") from exc
    if positive and x <= 0.0:
        raise InputError(field_key, f"{field_key}: must be positive")
    return x


def build_acid_base_model(inp: ModelInput) -> AcidBaseModel:
    """Build a validated :class:`AcidBaseModel` from text fields.

    Raises
    ------
    InputError
        For any invalid field (the engine's own ``ValueError`` is re-raised as
        ``InputError`` with field ``"model"``).

    Example
    -------
    >>> m = build_model(ModelInput(kind="weak_acid", pka="4.76"))
    >>> m.analyte_pka
    (4.76,)
    """
    try:
        kind = AnalyteKind(inp.kind)
    except ValueError as exc:
        raise InputError("kind", f"unknown analyte kind {inp.kind!r}") from exc
    try:
        strength = TitrantStrength(inp.titrant_strength)
    except ValueError as exc:
        raise InputError("titrant_strength", "unknown titrant strength") from exc
    c_a = _num("c_analyte", inp.c_analyte)
    v_a = _num("v_analyte", inp.v_analyte)
    c_t = _num("c_titrant", inp.c_titrant)
    temp = _num("temperature", inp.temperature, positive=False)
    pka: tuple[float, ...] = ()
    if kind in (AnalyteKind.WEAK_ACID, AnalyteKind.WEAK_BASE):
        try:
            pka = tuple(sorted(parse_number_list(inp.pka)))
        except ValueError as exc:
            raise InputError("pka", "pKa list is not valid") from exc
        if not pka:
            raise InputError("pka", "a weak analyte needs at least one pKa")
    t_pka: float | None = None
    if strength is TitrantStrength.WEAK:
        t_pka = _num("titrant_pka", inp.titrant_pka, positive=False)
    pkw: float | None = None
    if inp.pkw.strip():
        pkw = _num("pkw", inp.pkw)
    try:
        return AcidBaseModel(
            kind, c_a, v_a, c_t, pka, strength, t_pka, temperature_c=temp, pkw=pkw
        )
    except ValueError as exc:
        raise InputError("model", str(exc)) from exc


FAMILIES = ("acid_base", "redox", "precip", "edta", "back")
"""All titration families; ``family`` keys of :class:`ModelInput`."""

JUMP_OK = {"acid_base": 2.0, "redox": 0.3, "precip": 1.5, "edta": 2.0, "back": 0.0}
"""Minimum |Δy| over ±0.1 % of V_eq for a usable colour indicator, per family."""


def _opt(field_key: str, text: str) -> float | None:
    """Optional float field: empty/whitespace means ``None``."""
    if not text.strip():
        return None
    return _num(field_key, text, positive=False)


def build_phase2_model(inp: ModelInput) -> TitrationModel:
    """Build the phase-2 model selected by ``inp.family``.

    Raises
    ------
    InputError
        On any invalid field.

    Example
    -------
    >>> m = build_phase2_model(ModelInput(family="redox", e1="0.77", n1="1",
    ...                                   e2="1.44", n2="1"))
    >>> type(m).__name__
    'RedoxModel'
    """
    c_a = _num("c_analyte", inp.c_analyte)
    v_a = _num("v_analyte", inp.v_analyte)
    c_t = _num("c_titrant", inp.c_titrant)
    i_strength = _opt("ionic_strength", inp.ionic_strength)
    try:
        if inp.family == "redox":
            return RedoxModel(
                c_a, v_a, c_t,
                _num("e1", inp.e1, positive=False),
                _num("n1", inp.n1),
                _num("e2", inp.e2, positive=False),
                _num("n2", inp.n2),
                ph=_opt("redox_ph", inp.redox_ph),
                m1=_opt("m1", inp.m1) or 0.0,
                m2=_opt("m2", inp.m2) or 0.0,
                ionic_strength=i_strength,
            )
        if inp.family == "precip":
            return PrecipitationModel(
                c_a, v_a, c_t, _num("ksp", inp.ksp),
                ionic_strength=i_strength or 0.0,
            )
        if inp.family == "edta":
            metal = inp.aux_metal.strip()
            if metal and metal not in NH3_LOG_BETA:
                raise InputError("aux_metal", f"unknown metal key {metal!r}")
            return EdtaModel(
                c_a, v_a, c_t, _num("log_kf", inp.log_kf), _num("edta_ph", inp.edta_ph, positive=False),
                aux_ligand_conc=_opt("aux_conc", inp.aux_conc) or 0.0,
                aux_log_beta=NH3_LOG_BETA.get(metal, ()) if metal else (),
                ionic_strength=i_strength or 0.0,
            )
        raise InputError("family", f"unknown family {inp.family!r}")
    except InputError:
        raise
    except ValueError as exc:
        raise InputError("model", str(exc)) from exc


@dataclass(frozen=True)
class BackTitrationInput:
    """Validated inputs of the back-titration calculator.

    Example
    -------
    >>> b = build_back_input(ModelInput(family="back", c_reagent="0.1",
    ...     v_reagent="50", c_back_titrant="0.1", v_back_titrant="20",
    ...     v_sample="25"))
    >>> round(b.n_analyte_mmol(), 3), round(b.c_analyte(), 3)
    (3.0, 0.12)
    """

    c_reagent: float
    v_reagent_ml: float
    c_titrant: float
    v_titrant_ml: float
    v_sample_ml: float
    reagent_per_analyte: float = 1.0
    titrant_per_reagent: float = 1.0

    def n_analyte_mmol(self) -> float:
        return back_titration(self.c_reagent, self.v_reagent_ml, self.c_titrant,
                              self.v_titrant_ml, self.v_sample_ml,
                              self.reagent_per_analyte, self.titrant_per_reagent).n_analyte_mmol

    def c_analyte(self) -> float:
        return back_titration(self.c_reagent, self.v_reagent_ml, self.c_titrant,
                              self.v_titrant_ml, self.v_sample_ml,
                              self.reagent_per_analyte, self.titrant_per_reagent).c_analyte

    def consumed_mmol(self) -> float:
        return back_titration(self.c_reagent, self.v_reagent_ml, self.c_titrant,
                              self.v_titrant_ml, self.v_sample_ml,
                              self.reagent_per_analyte, self.titrant_per_reagent).n_reagent_consumed_mmol


def build_back_input(inp: ModelInput) -> BackTitrationInput:
    """Validate the back-titration calculator fields.

    Raises
    ------
    InputError
        On any invalid field.
    """
    try:
        b = BackTitrationInput(
            _num("c_reagent", inp.c_reagent),
            _num("v_reagent", inp.v_reagent),
            _num("c_back_titrant", inp.c_back_titrant),
            _num("v_back_titrant", inp.v_back_titrant),
            _num("v_sample", inp.v_sample),
            _num("ratio_ra", inp.ratio_ra),
            _num("ratio_tr", inp.ratio_tr),
        )
    except ValueError as exc:
        raise InputError("back", str(exc)) from exc
    try:
        b.n_analyte_mmol()  # consistency (negative consumption) check
    except ValueError as exc:
        raise InputError("back", str(exc)) from exc
    return b


def find_indicator(name: str) -> Indicator | None:
    """Indicator by name; empty string means none.

    Raises
    ------
    InputError
        For an unknown name.

    Example
    -------
    >>> find_indicator("") is None
    True
    >>> find_indicator("Methyl Red").pk_in
    5.1
    """
    if not name:
        return None
    for ind in load_indicators():
        if ind.name == name:
            return ind
    raise InputError("indicator", f"unknown indicator {name!r}")


# ---------------------------------------------------------------- families
@dataclass(frozen=True)
class FamilyIndicator:
    """A phase-2 indicator: colour flips when the measured quantity crosses ``transition``.

    ``below``/``above`` are CSS colours for y < transition and y > transition.
    Textbook transition values (APPROXIMATION: the real transition depends on
    the conditional constant of the metal-dye or dye-adsorption system).
    """

    name: str
    transition: float
    below: str
    above: str


FAMILY_INDICATORS: dict[str, tuple[FamilyIndicator, ...]] = {
    "redox": (
        FamilyIndicator("Ferroin", 1.06, "#c33c54", "#9db8c8"),
        FamilyIndicator("Diphenylamine sulfonate", 0.84, "#f2f0eb", "#8e44ad"),
        FamilyIndicator("Methylene blue", 0.53, "#3465c0", "#f2f0eb"),
    ),
    "precip": (
        FamilyIndicator("Mohr (K2CrO4)", 4.70, "#a2452e", "#e3c93f"),
        FamilyIndicator("Fajans (dichlorofluorescein)", 4.85, "#e0679a", "#f5f2ec"),
    ),
    "edta": (
        FamilyIndicator("Eriochrome Black T", 6.5, "#8e2a4a", "#2f5fa8"),
        FamilyIndicator("Calmagite", 6.1, "#8e2a4a", "#2f5fa8"),
        FamilyIndicator("Murexide", 5.0, "#b05a8e", "#6a3fa0"),
    ),
}


def find_family_indicator(family: str, name: str) -> FamilyIndicator | None:
    """Indicator of a phase-2 family by name; empty string means none.

    Raises
    ------
    InputError
        For an unknown name.

    Example
    -------
    >>> find_family_indicator("redox", "Ferroin").transition
    1.06
    >>> find_family_indicator("redox", "") is None
    True
    """
    if not name:
        return None
    for ind in FAMILY_INDICATORS.get(family, ()):
        if ind.name == name:
            return ind
    raise InputError("indicator", f"unknown {family} indicator {name!r}")


def family_quantity_label(family: str) -> str:
    """Axis/label key of the measured quantity of a family.

    Example
    -------
    >>> family_quantity_label("redox")
    'qty_E'
    """
    return {"acid_base": "qty_ph", "redox": "qty_E", "precip": "qty_pAg",
            "edta": "qty_pM", "back": "qty_ph"}[family]


@dataclass(frozen=True)
class SessionAnalysis:
    """What the analyst can conclude from the points collected so far.

    Attributes
    ----------
    v_eq_theory : float
        Theoretical first equivalence volume (mL).
    v_eq_found : float | None
        Equivalence volume from derivative analysis of the *collected points*
        (None until at least 7 points exist and the curve passed V_eq).
    c_found : float | None
        ``C_t V_eq,found / V_a`` (mol/L, 1:1 stoichiometry, first equivalence).
    rel_error_percent : float | None
        ``(c_found - c_true) / c_true * 100``.
    jump : float
        pH jump over +-0.1 % of V_eq (section 3.3).
    visual_ok : bool
    message : str
    indicator_changed : bool | None
        None when no indicator is used.
    indicator_error_percent : float | None
    """

    v_eq_theory: float
    v_eq_found: float | None
    c_found: float | None
    rel_error_percent: float | None
    jump: float
    visual_ok: bool
    message: str
    indicator_changed: bool | None
    indicator_error_percent: float | None


@dataclass
class TitrationSession:
    """One titration at the bench: model, history of states, burette, indicator.

    Parameters
    ----------
    model : AcidBaseModel
    indicator : Indicator | None
    burette_capacity_ml : float
        Burette volume (mL); the titrant cannot be added beyond it.

    Example
    -------
    >>> s = TitrationSession(AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0))
    >>> s.add(10.0)
    10.0
    >>> round(s.state.ph, 3)
    1.368
    """

    model: AcidBaseModel
    indicator: Indicator | None = None
    burette_capacity_ml: float = 50.0
    engine: AcidBaseEngine = field(default_factory=AcidBaseEngine)
    states: list[TitrationState] = field(init=False)
    _static: tuple[float, "FeasibilityReport", float | None] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.burette_capacity_ml <= 0.0:
            raise ValueError("burette capacity must be positive")
        self.states = [self.engine.initial_state(self.model)]

    # ---- state -----------------------------------------------------------
    @property
    def state(self) -> TitrationState:
        """Current vessel state."""
        return self.states[-1]

    @property
    def v_added_ml(self) -> float:
        """Titrant delivered so far (mL) = burette reading (zero at the top)."""
        return self.state.v_titrant_ml

    @property
    def burette_remaining_fraction(self) -> float:
        """Fraction of the burette still filled, 0 ... 1.

        Example
        -------
        >>> s = TitrationSession(AcidBaseModel.strong_acid(0.1, 25.0, 0.1), burette_capacity_ml=50.0)
        >>> _ = s.add(12.5)
        >>> s.burette_remaining_fraction
        0.75
        """
        return max(0.0, 1.0 - self.v_added_ml / self.burette_capacity_ml)

    def add(self, d_v_ml: float) -> float:
        """Run ``d_v_ml`` of titrant out of the burette.

        Returns
        -------
        float
            The volume actually delivered (smaller than requested if the
            burette runs empty; 0.0 if it is already empty).

        Raises
        ------
        ValueError
            If ``d_v_ml`` is negative or not finite.
        """
        if not math.isfinite(d_v_ml) or d_v_ml < 0.0:
            raise ValueError("dV must be a non-negative finite number")
        room = self.burette_capacity_ml - self.v_added_ml
        dv = min(d_v_ml, room)
        if dv <= 1e-12:
            return 0.0
        self.states.append(self.engine.titrate(self.model, self.state, dv))
        return dv

    def reset(self) -> None:
        """Back to V = 0 (fresh burette fill, same model)."""
        self.states = [self.engine.initial_state(self.model)]

    @property
    def burette_empty(self) -> bool:
        """True when no titrant is left."""
        return self.burette_capacity_ml - self.v_added_ml <= 1e-9

    # ---- data for the plots ------------------------------------------------
    def measured(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Collected points ``(V, pH)``."""
        return (
            np.array([s.v_titrant_ml for s in self.states]),
            np.array([s.ph for s in self.states]),
        )

    def derivative(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Non-uniform first derivative of the collected points ``(V, dpH/dV)``.

        Empty arrays if fewer than 3 points or duplicates make it undefined.
        """
        v, p = self.measured()
        if v.size < 3:
            return np.array([]), np.array([])
        return v, first_derivative(v, p)

    def theory_curve(self, v_max_ml: float | None = None) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Adaptive theoretical curve for the faint overlay.

        ``v_max_ml`` defaults to 1.5 times the last equivalence volume, capped
        by the burette capacity.
        """
        eq = equivalence_volumes(self.model)
        top = v_max_ml if v_max_ml is not None else min(self.burette_capacity_ml, 1.5 * eq[-1])
        return generate_curve(self.model, top)

    # ---- appearance --------------------------------------------------------
    def flask_color(self) -> str:
        """Colour of the flask content (indicator colour at the current pH)."""
        return indicator_color(self.indicator, self.state.ph) if self.indicator else WATER

    def indicator_changed(self) -> bool | None:
        """True once the pH passed the indicator's endpoint pH (None: no indicator)."""
        if self.indicator is None:
            return None
        target = endpoint_ph(self.indicator, self.model.analyte_is_acid)
        return self.state.ph >= target if self.model.analyte_is_acid else self.state.ph <= target

    def suggest_step_ml(self) -> float:
        """Step for the auto-run: small near the first equivalence point.

        The analyst knows the nominal concentrations, hence the nominal V_eq.
        """
        ve = equivalence_volumes(self.model)[0]
        rel = abs(self.v_added_ml - ve) / ve
        if rel < 0.01:
            return 0.04 * ve / 25.0  # 0.04 mL per 25 mL of V_eq
        if rel < 0.05:
            return 0.1 * ve / 25.0
        if rel < 0.2:
            return 0.4 * ve / 25.0
        return 1.0 * ve / 25.0

    # ---- analysis ----------------------------------------------------------
    def _static_part(self) -> tuple[float, FeasibilityReport, float | None]:
        """Quantities that depend only on model + indicator (computed once)."""
        if self._static is None:
            ve = equivalence_volumes(self.model)[0]
            ind_err: float | None = None
            if self.indicator is not None:
                vep = endpoint_volume(self.model, self.indicator)
                if vep is not None:
                    ind_err = (vep - ve) / ve * 100.0
            self._static = (ve, check_visual_endpoint(self.model), ind_err)
        return self._static

    def analysis(self) -> SessionAnalysis:
        """Analyse the collected points (see :class:`SessionAnalysis`)."""
        ve, feas, ind_err = self._static_part()
        v, p = self.measured()
        found: float | None = None
        c_found: float | None = None
        rel: float | None = None
        if v.size >= 7 and v[-1] > 1.02 * ve:
            try:
                found = find_equivalence(v, p, expected=ve)
            except ValueError:
                found = None
        if found is not None:
            c_found = self.model.c_titrant * found / self.model.v_analyte_ml
            rel = (c_found - self.model.c_analyte) / self.model.c_analyte * 100.0
        return SessionAnalysis(
            ve, found, c_found, rel, feas.jump, feas.visual_ok, feas.message,
            self.indicator_changed(), ind_err,
        )

    def v_eq_last(self) -> float:
        """Last theoretical equivalence volume (auto-run bound)."""
        return equivalence_volumes(self.model)[-1]

    def report(self, lang: str = "fa") -> str:
        """Plain-text report for the results panel.

        Example
        -------
        >>> s = TitrationSession(AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0))
        >>> "pH" in s.report("en")
        True
        """
        a = self.analysis()
        lines = [
            f"{tr('ph_now', lang)}: {self.state.ph:.3f}",
            f"{tr('v_added', lang)}: {self.v_added_ml:.2f} mL",
            f"{tr('v_eq_theory', lang)}: {a.v_eq_theory:.3f} mL",
            f"{tr('ph_jump', lang)}: {a.jump:.2f}",
            a.message,
        ]
        if a.v_eq_found is not None and a.c_found is not None and a.rel_error_percent is not None:
            lines += [
                f"{tr('v_eq_found', lang)}: {a.v_eq_found:.3f} mL",
                f"{tr('c_found', lang)}: {a.c_found:.5f} mol/L",
                f"{tr('rel_err', lang)}: {a.rel_error_percent:+.3f} %",
            ]
        else:
            lines.append(tr("need_points", lang))
        if a.indicator_changed is not None and self.indicator is not None:
            lines.append(f"{self.indicator.name}: " + (tr("ind_changed", lang) if a.indicator_changed else tr("ind_not_yet", lang)))
            if a.indicator_error_percent is not None:
                lines.append(f"{tr('ind_error', lang)}: {a.indicator_error_percent:+.3f} %")
        if self.burette_empty:
            lines.append(tr("burette_empty", lang))
        return "\n".join(lines)


class FamilySession:
    """Generic bench session for the phase-2 families (redox / precip / EDTA).

    Same duck-typed API as :class:`TitrationSession` so the window does not
    care which family is loaded; the measured quantity is
    ``y = quantity_of(model, V)`` (E in V, pAg or pM).
    """

    def __init__(self, model: TitrationModel, family: str,
                 indicator: FamilyIndicator | None = None,
                 burette_capacity_ml: float = 50.0) -> None:
        if burette_capacity_ml <= 0.0:
            raise ValueError("burette capacity must be positive")
        self.model = model
        self.family = family
        self.indicator = indicator
        self.burette_capacity_ml = burette_capacity_ml
        if family == "redox":
            self.v_eq = redox_equivalence_volume(model)  # type: ignore[arg-type]
        elif family == "precip":
            self.v_eq = precip_equivalence_volume(model)  # type: ignore[arg-type]
        elif family == "edta":
            self.v_eq = edta_equivalence_volume(model)  # type: ignore[arg-type]
        else:
            raise ValueError(f"no curve family {family!r}")
        self._v: list[float] = [0.0]
        self._y: list[float] = [float("nan")]

    # ---- bench API -------------------------------------------------------
    @property
    def state(self) -> TitrationState:
        """Vessel snapshot (ph carries the family quantity: E / pAg / pM)."""
        import math as _m
        vt = self._v[-1] + (self.model.v_analyte_ml if hasattr(self.model, "v_analyte_ml") else 0.0)
        return TitrationState(self._v[-1], vt, self._y[-1], float("nan"),
                              {}, temperature_c=25.0) if not _m.isnan(self._y[-1]) else \
            TitrationState(self._v[-1], vt, float("nan"), float("nan"), {}, 25.0)

    @property
    def v_added_ml(self) -> float:
        return self._v[-1]

    @property
    def burette_remaining_fraction(self) -> float:
        return max(0.0, 1.0 - self.v_added_ml / self.burette_capacity_ml)

    @property
    def burette_empty(self) -> bool:
        return self.burette_capacity_ml - self.v_added_ml <= 1e-9

    def add(self, d_v_ml: float) -> float:
        """Run ``d_v_ml`` out of the burette (clipped to the capacity)."""
        import math as _math
        if not _math.isfinite(d_v_ml) or d_v_ml < 0.0:
            raise ValueError("dV must be a non-negative finite number")
        dv = min(d_v_ml, self.burette_capacity_ml - self.v_added_ml)
        if dv <= 1e-12:
            return 0.0
        v_new = self.v_added_ml + dv
        self._v.append(v_new)
        self._y.append(quantity_of(self.model, v_new))
        return dv

    def reset(self) -> None:
        """Back to V = 0."""
        self._v = [0.0]
        self._y = [float("nan")]

    def current_y(self) -> float:
        """Measured quantity at the current volume."""
        return self._y[-1]

    # ---- plots -----------------------------------------------------------
    def measured(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return np.array(self._v), np.array(self._y)

    def derivative(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        v, y = self.measured()
        if v.size < 3:
            return np.array([]), np.array([])
        return v, first_derivative(v, y)

    def theory_curve(self, v_max_ml: float | None = None) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        top = v_max_ml if v_max_ml is not None else min(self.burette_capacity_ml, 1.6 * self.v_eq)
        v = np.linspace(top / 400, top, 400)
        return v, np.array([quantity_of(self.model, float(x)) for x in v])

    # ---- appearance --------------------------------------------------------
    def flask_color(self) -> str:
        """Indicator colour: ``below`` for y < transition, ``above`` after it.

        The ``below`` colour is the pre-end-point colour for every family:
        redox y = E *rises* through the transition (ferroin red -> pale blue),
        pAg *falls* through it (Mohr yellow -> red-brown) and pM falls
        (EBT wine-red -> blue).
        """
        if self.indicator is None:
            return WATER
        y = self.current_y()
        if math.isnan(y):
            return WATER
        return self.indicator.below if y < self.indicator.transition else self.indicator.above

    def indicator_changed(self) -> bool | None:
        """True once y crossed the transition (direction depends on the family).

        E and pM RISE through the end point; pAg FALLS through it.
        """
        if self.indicator is None:
            return None
        y = self.current_y()
        if math.isnan(y):
            return False
        if self.family == "precip":
            return y <= self.indicator.transition
        return y >= self.indicator.transition

    def suggest_step_ml(self) -> float:
        ve = self.v_eq
        rel = abs(self.v_added_ml - ve) / ve
        if rel < 0.01:
            return 0.04 * ve / 25.0
        if rel < 0.05:
            return 0.1 * ve / 25.0
        if rel < 0.2:
            return 0.4 * ve / 25.0
        return 1.0 * ve / 25.0

    def v_eq_last(self) -> float:
        return self.v_eq

    # ---- analysis ----------------------------------------------------------
    def analysis(self) -> SessionAnalysis:
        v, y = self.measured()
        finite = ~np.isnan(y)                       # V = 0 has no Ox yet (NaN)
        v, y = v[finite], y[finite]
        found: float | None = None
        c_found: float | None = None
        rel: float | None = None
        if v.size >= 7 and v[-1] > 1.02 * self.v_eq:
            try:
                found = find_equivalence(v, y, expected=self.v_eq)
            except ValueError:
                found = None
        if found is not None and not math.isfinite(found):
            found = None
        if found is not None:
            m = self.model
            if self.family == "redox":
                assert isinstance(m, RedoxModel)
                c_found = m.n2 * m.c_titrant * found / (m.n1 * m.v_analyte_ml)
                c_true = m.n2 * m.c_titrant * self.v_eq / (m.n1 * m.v_analyte_ml)
            else:
                c_found = m.c_titrant * found / m.v_analyte_ml  # type: ignore[attr-defined]
                c_true = m.c_titrant * self.v_eq / m.v_analyte_ml  # type: ignore[attr-defined]
            rel = (c_found - c_true) / c_true * 100.0
        jump = self._jump()
        visual_ok = jump >= JUMP_OK[self.family]
        ind_err: float | None = None
        if self.indicator is not None and not math.isnan(self._y_at(self.indicator.transition)):
            v_end = self._v_at(self.indicator.transition)
            if v_end is not None:
                ind_err = (v_end - self.v_eq) / self.v_eq * 100.0
        message = ""
        return SessionAnalysis(self.v_eq, found, c_found, rel, jump, visual_ok, message,
                               self.indicator_changed(), ind_err)

    # ---- helpers -----------------------------------------------------------
    def _jump(self) -> float:
        lo, hi = self.v_eq * 0.999, self.v_eq * 1.001
        y1, y2 = quantity_of(self.model, lo), quantity_of(self.model, hi)
        if math.isnan(y1) or math.isnan(y2):
            return 0.0
        return abs(y2 - y1)

    def _y_at(self, target: float) -> float:
        return target  # transitions are given in y-units directly

    def _v_at(self, y_target: float) -> float | None:
        """First collected volume whose y crosses ``y_target`` (linear interp)."""
        for i in range(1, len(self._y)):
            a, b = self._y[i - 1], self._y[i]
            if math.isnan(a) or math.isnan(b):
                continue
            if (a - y_target) * (b - y_target) <= 0.0 and a != b:
                t = (y_target - a) / (b - a)
                return self._v[i - 1] + t * (self._v[i] - self._v[i - 1])
        return None

    def report(self, lang: str = "fa") -> str:
        """Plain-text report (family-specific quantity names)."""
        qk = family_quantity_label(self.family)
        a = self.analysis()
        lines = [f"{tr(qk, lang)}: {self._fmt_y(self.current_y(), lang)}"]
        lines.append(f"{tr('v_added', lang)}: {self.v_added_ml:.2f} mL")
        lines.append(f"{tr('v_eq_theory', lang)}: {self.v_eq:.3f} mL")
        if self.family == "redox":
            lines.append(f"{tr('jump_redox', lang)}: {a.jump:.3f} V")
            assert isinstance(self.model, RedoxModel)
            lines.append(f"E_eq = {redox_equivalence_potential(self.model):.4f} V")
        else:
            lines.append(f"{tr('jump_pag_pm', lang)}: {a.jump:.2f}")
        lines.append(tr("redox_kinetics_note", lang))
        if a.v_eq_found is not None and a.c_found is not None and a.rel_error_percent is not None:
            lines += [
                f"{tr('v_eq_found', lang)}: {a.v_eq_found:.3f} mL",
                f"{tr('c_found', lang)}: {a.c_found:.5f} mol/L",
                f"{tr('rel_err', lang)}: {a.rel_error_percent:+.3f} %",
            ]
        else:
            lines.append(tr("need_points", lang))
        if a.indicator_changed is not None and self.indicator is not None:
            lines.append(f"{self.indicator.name}: " + (tr("ind_changed", lang) if a.indicator_changed else tr("ind_not_yet", lang)))
            if a.indicator_error_percent is not None:
                lines.append(f"{tr('ind_error', lang)}: {a.indicator_error_percent:+.3f} %")
        if self.burette_empty:
            lines.append(tr("burette_empty", lang))
        return "\n".join(lines)

    @staticmethod
    def _fmt_y(y: float, lang: str) -> str:
        if math.isnan(y):
            return "-" if lang == "en" else "—"
        return f"{y:.4f}"


AnySession = Union["TitrationSession", "FamilySession", "BackSession"]
"""Whatever :func:`make_session` can return (the window treats them alike)."""


def make_session(inp: ModelInput) -> AnySession:
    """Validate the text fields and start a session of the selected family.

    Raises
    ------
    InputError
        On any invalid field.

    Example
    -------
    >>> s = make_session(ModelInput(indicator="Phenolphthalein"))
    >>> s.indicator.name
    'Phenolphthalein'
    >>> s2 = make_session(ModelInput(family="redox", e1="0.77", n1="1", e2="1.44", n2="1"))
    >>> s2.v_eq
    25.0
    """
    cap = _num("burette_capacity", inp.burette_capacity)
    if inp.family == "back":
        build_back_input(inp)  # validation only; the report is produced on demand
        return BackSession(cap)
    if inp.family == "acid_base":
        ab_model = build_acid_base_model(inp)
        return TitrationSession(ab_model, find_indicator(inp.indicator), cap)
    p2_model = build_phase2_model(inp)
    return FamilySession(p2_model, inp.family, find_family_indicator(inp.family, inp.indicator), cap)


class BackSession:
    """Placeholder bench object for the back-titration calculator (no burette)."""

    def __init__(self, burette_capacity_ml: float = 50.0) -> None:
        self.burette_capacity_ml = burette_capacity_ml
        self.v_added_ml = 0.0
        self.input: BackTitrationInput | None = None
        self.error: str | None = None
        self.result: BackTitrationResult | None = None

    @property
    def state(self) -> TitrationState:
        """Empty vessel snapshot (calculator family)."""
        return TitrationState(0.0, 0.0, float("nan"), float("nan"), {}, 25.0)

    @property
    def burette_remaining_fraction(self) -> float:
        return 1.0

    @property
    def burette_empty(self) -> bool:
        return True

    def add(self, d_v_ml: float) -> float:  # noqa: ARG002 - API compat
        return 0.0

    def reset(self) -> None:
        pass

    def measured(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return np.array([]), np.array([])

    def derivative(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return np.array([]), np.array([])

    def theory_curve(self, v_max_ml: float | None = None) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return np.array([]), np.array([])

    def flask_color(self) -> str:
        return WATER

    def indicator_changed(self) -> bool | None:
        return None

    def suggest_step_ml(self) -> float:
        return 0.05

    def v_eq_last(self) -> float:
        return float("inf")

    def analysis(self) -> SessionAnalysis:
        return SessionAnalysis(float("inf"), None, None, None, 0.0, False, "", None, None)

    def report(self, lang: str = "fa") -> str:
        return tr("back_no_curve", lang)


__all__ = [
    "BackSession", "BackTitrationInput", "FAMILIES", "FAMILY_INDICATORS", "FamilyIndicator",
    "FamilySession", "InputError", "ModelInput", "SessionAnalysis", "TitrationSession",
    "build_acid_base_model", "build_back_input", "build_model", "build_phase2_model",
    "find_family_indicator", "find_indicator", "family_quantity_label", "make_session",
    "parse_number", "parse_number_list", "ph_at",
]


def build_model(inp: ModelInput) -> TitrationModel:
    """Dispatch model construction by ``inp.family``.

    ``acid_base`` (default) returns :class:`AcidBaseModel`; ``redox`` /
    ``precip`` / ``edta`` return the phase-2 models.

    Example
    -------
    >>> build_model(ModelInput(kind="weak_acid", pka="4.76")).analyte_pka   # acid_base default
    (4.76,)
    >>> type(build_model(ModelInput(family="precip", ksp="1.8e-10"))).__name__
    'PrecipitationModel'
    """
    if inp.family in ("", "acid_base"):
        return build_acid_base_model(inp)
    if inp.family == "back":
        return build_acid_base_model(inp)  # unused; calculator validated separately
    return build_phase2_model(inp)
