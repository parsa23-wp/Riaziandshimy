"""Acid-base titration engine (MVP phase 1).

Supported systems (any analyte/titrant combination):

* strong acid / strong base analyte,
* weak monoprotic or polyprotic acid, weak mono/poly base (given through the
  pKa values of the *protonated* form; use :meth:`AcidBaseModel.weak_base_from_pkb`
  for a Kb input),
* strong *or weak* titrant (weak/weak is the hardest case, second answer 3.5).

No region-by-region shortcut formulas are used.  Every point is the exact
solution of the proton balance (see :mod:`equilibrium`), which contains the
full cubic of a weak acid (section 2.4), the continuous strong/strong formula
with Kw (section 3.1) and all buffer/hydrolysis effects of section 3.2.

Temperature: pKw(T) (section 12.1), van't Hoff correction of pKa (12.2) and
optional thermal volume expansion (12.3).

Reference: [MATH.md sections 2, 3, 12]
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import BaseEngine, TitrationModel, TitrationState
from .equilibrium import (
    Ligand,
    pka_to_log_beta,
    solve_log_h,
    species_fractions,
)

R_GAS = 8.314462618  # J / (mol K)
T_REF_K = 298.15
BETA_WATER = 2.1e-4  # 1/K, volumetric expansion of water near 25 C (section 12.3)
_DATA = Path(__file__).resolve().parent.parent / "data"


class AnalyteKind(str, Enum):
    """Kind of analyte."""

    STRONG_ACID = "strong_acid"
    STRONG_BASE = "strong_base"
    WEAK_ACID = "weak_acid"
    WEAK_BASE = "weak_base"


class TitrantStrength(str, Enum):
    """Strength of the titrant."""

    STRONG = "strong"
    WEAK = "weak"


@lru_cache(maxsize=1)
def _pkw_fit() -> tuple[float, float, float, float]:
    with open(_DATA / "pKw_vs_T.json", encoding="utf-8") as fh:
        d = json.load(fh)["fit"]
    return float(d["a"]), float(d["b"]), float(d["c"]), float(d["d"])


def pkw_water(temperature_c: float) -> float:
    """pKw of pure water as a function of temperature.

    Uses ``pKw = a/T + b log10(T) + c T + d`` (T in K, section 12.1).
    Do NOT use a fixed 14 (section 4 rule); the value at 25 C is 13.995.

    Parameters
    ----------
    temperature_c : float
        Temperature in deg C (0 ... 100 recommended).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the temperature is outside -5 ... 150 C.

    Example
    -------
    >>> round(pkw_water(50.0), 2)
    13.26

    [MATH.md section 12.1]
    """
    if not -5.0 <= temperature_c <= 150.0:
        raise ValueError("temperature outside the validity range of the pKw fit")
    t = temperature_c + 273.15
    a, b, c, d = _pkw_fit()
    return a / t + b * math.log10(t) + c * t + d


def pka_at_temperature(pka_25: float, delta_h_kj_mol: float, temperature_c: float) -> float:
    """van't Hoff correction of a pKa.

    APPROXIMATION: dH of dissociation constant over the temperature interval
    (error grows with |T - 25 C| and with heat-capacity change; typically
    < 0.02 pKa for |dT| < 25 K and |dH| < 10 kJ/mol).

    ``pKa(T) = pKa25 + dH / (R ln10) * (1/T - 1/298.15)``

    Example
    -------
    >>> round(pka_at_temperature(4.76, -0.4, 50.0), 3)
    4.765

    [MATH.md section 12.2]
    """
    t = temperature_c + 273.15
    return pka_25 + (delta_h_kj_mol * 1000.0) / (R_GAS * math.log(10.0)) * (1.0 / t - 1.0 / T_REF_K)


def pka_from_pkb(pkb: float, pkw: float = 13.995) -> float:
    """pKa of the conjugate acid from the pKb of a base (pKa + pKb = pKw).

    Example
    -------
    >>> round(pka_from_pkb(4.75, 14.0), 2)
    9.25

    [MATH.md section 2.6]
    """
    return pkw - pkb


@dataclass(frozen=True)
class AcidBaseModel(TitrationModel):
    """Chemistry of one acid-base titration.

    Attributes
    ----------
    analyte_kind : AnalyteKind
    c_analyte : float
        Analytical concentration of the analyte before titration (mol/L).
    v_analyte_ml : float
        Volume of analyte solution (mL).
    c_titrant : float
        Titrant concentration (mol/L).
    analyte_pka : tuple[float, ...]
        pKa values (ascending) of the protonated form(s); empty for strong.
    titrant_strength : TitrantStrength
    titrant_pka : float | None
        For a weak titrant: pKa of HX (weak acid titrant) or of BH+ (weak
        base titrant).
    temperature_c : float
    pkw : float | None
        Override of pKw (e.g. 13.77 for a 0.1 M alkali chloride medium,
        Berto 2024).  ``None`` uses :func:`pkw_water`.
    delta_h_kj_mol : float
        One dH applied to all pKa values for the van't Hoff correction.
    thermal_volume_correction : bool
        Apply V(T) = V25 (1 + beta (T - 25)) (section 12.3).

    Raises
    ------
    ValueError
        On non-positive concentrations/volumes, missing/unsorted pKa values,
        or a weak titrant without ``titrant_pka``.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1)
    >>> m.analyte_kind.value
    'strong_acid'
    """

    analyte_kind: AnalyteKind
    c_analyte: float
    v_analyte_ml: float
    c_titrant: float
    analyte_pka: tuple[float, ...] = ()
    titrant_strength: TitrantStrength = TitrantStrength.STRONG
    titrant_pka: float | None = None
    temperature_c: float = 25.0
    pkw: float | None = None
    delta_h_kj_mol: float = 0.0
    thermal_volume_correction: bool = False

    def __post_init__(self) -> None:
        if not (self.c_analyte > 0 and self.v_analyte_ml > 0 and self.c_titrant > 0):
            raise ValueError("concentrations and volume must be positive")
        weak_analyte = self.analyte_kind in (AnalyteKind.WEAK_ACID, AnalyteKind.WEAK_BASE)
        if weak_analyte and len(self.analyte_pka) == 0:
            raise ValueError("a weak analyte needs at least one pKa")
        if not weak_analyte and len(self.analyte_pka) > 0:
            raise ValueError("a strong analyte has no pKa")
        if any(self.analyte_pka[i] > self.analyte_pka[i + 1] for i in range(len(self.analyte_pka) - 1)):
            raise ValueError("pKa values must be ascending")
        if self.titrant_strength is TitrantStrength.WEAK and self.titrant_pka is None:
            raise ValueError("a weak titrant needs titrant_pka")
        if self.pkw is not None and self.pkw <= 0:
            raise ValueError("pkw must be positive")

    # ---- constructors --------------------------------------------------
    @classmethod
    def strong_acid(cls, c: float, v_ml: float, c_titrant: float, **kw: object) -> "AcidBaseModel":
        """Strong acid analyte titrated with a strong base."""
        return cls(AnalyteKind.STRONG_ACID, c, v_ml, c_titrant, **kw)  # type: ignore[arg-type]

    @classmethod
    def strong_base(cls, c: float, v_ml: float, c_titrant: float, **kw: object) -> "AcidBaseModel":
        """Strong base analyte titrated with a strong acid."""
        return cls(AnalyteKind.STRONG_BASE, c, v_ml, c_titrant, **kw)  # type: ignore[arg-type]

    @classmethod
    def weak_acid(
        cls, c: float, v_ml: float, c_titrant: float, pka: Sequence[float] | np.ndarray[Any, Any], **kw: object
    ) -> "AcidBaseModel":
        """(Poly)protic weak acid titrated with a (strong) base."""
        return cls(AnalyteKind.WEAK_ACID, c, v_ml, c_titrant, tuple(pka), **kw)  # type: ignore[arg-type]

    @classmethod
    def weak_base(
        cls, c: float, v_ml: float, c_titrant: float, pka_conjugate: Sequence[float] | np.ndarray[Any, Any], **kw: object
    ) -> "AcidBaseModel":
        """Weak base given by the pKa of its conjugate acid(s)."""
        return cls(AnalyteKind.WEAK_BASE, c, v_ml, c_titrant, tuple(pka_conjugate), **kw)  # type: ignore[arg-type]

    @classmethod
    def weak_base_from_pkb(
        cls, c: float, v_ml: float, c_titrant: float, pkb: float, **kw: object
    ) -> "AcidBaseModel":
        """Monobasic weak base from pKb (pKa = pKw - pKb at the model's pKw)."""
        pkw = kw.get("pkw")
        pkw_v = float(pkw) if isinstance(pkw, (int, float)) else pkw_water(float(kw.get("temperature_c", 25.0)))  # type: ignore[arg-type]
        return cls.weak_base(c, v_ml, c_titrant, [pka_from_pkb(pkb, pkw_v)], **kw)

    # ---- derived -------------------------------------------------------
    @property
    def analyte_is_acid(self) -> bool:
        """True if the analyte is an acid (titrant is then a base)."""
        return self.analyte_kind in (AnalyteKind.STRONG_ACID, AnalyteKind.WEAK_ACID)

    @property
    def n_steps(self) -> int:
        """Number of proton steps of the analyte (1 for strong)."""
        return max(1, len(self.analyte_pka))

    @property
    def kw(self) -> float:
        """Ion product of water at the model's conditions."""
        pk = self.pkw if self.pkw is not None else pkw_water(self.temperature_c)
        return float(10.0 ** (-pk))

    def _pka_t(self, pka: float) -> float:
        if self.delta_h_kj_mol == 0.0:
            return pka
        return pka_at_temperature(pka, self.delta_h_kj_mol, self.temperature_c)

    @property
    def volume_factor(self) -> float:
        """Thermal expansion factor (1 if the correction is off)."""
        if not self.thermal_volume_correction:
            return 1.0
        return 1.0 + BETA_WATER * (self.temperature_c - 25.0)


@dataclass(frozen=True)
class FeasibilityReport:
    """Result of the pH-jump check around an equivalence point.

    Attributes
    ----------
    jump : float
        pH(Veq + 0.1 %) - pH(Veq - 0.1 %), absolute value.
    visual_ok : bool
        True if the jump is at least 2 pH units (section 3.3 criterion).
    message : str
    """

    jump: float
    visual_ok: bool
    message: str


def _balance_inputs(
    model: AcidBaseModel, v_ml: float
) -> tuple[float, list[Ligand], float, dict[str, tuple[Ligand, str]]]:
    """Build (total_h, ligands, total volume, named ligands) at titrant volume ``v_ml``."""
    f = model.volume_factor
    v0 = model.v_analyte_ml * f
    ct = model.c_titrant / f
    vt = v0 + v_ml
    ca = model.c_analyte * model.v_analyte_ml / vt
    cb = ct * v_ml / vt
    ligands: list[Ligand] = []
    named: dict[str, tuple[Ligand, str]] = {}
    total = 0.0
    kind = model.analyte_kind
    if kind is AnalyteKind.STRONG_ACID:
        total += ca
    elif kind is AnalyteKind.STRONG_BASE:
        total -= ca
    else:
        pkas = tuple(model._pka_t(p) for p in model.analyte_pka)
        lig = Ligand(ca, pka_to_log_beta(pkas))
        ligands.append(lig)
        named["analyte"] = (lig, "A")
        if kind is AnalyteKind.WEAK_ACID:
            total += len(pkas) * ca
    if model.titrant_strength is TitrantStrength.STRONG:
        total += -cb if model.analyte_is_acid else cb
    else:
        assert model.titrant_pka is not None
        pk_t = model._pka_t(model.titrant_pka)
        lig_t = Ligand(cb, (pk_t,))
        ligands.append(lig_t)
        named["titrant"] = (lig_t, "T")
        if not model.analyte_is_acid:
            total += cb  # weak acid titrant brings one proton per HX
    return total, ligands, vt, named


def ph_at(model: AcidBaseModel, v_titrant_ml: float) -> float:
    """pH after adding ``v_titrant_ml`` mL titrant (exact proton balance).

    Raises
    ------
    ValueError
        If the volume is negative.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> round(ph_at(m, 0.0), 6)
    1.0

    [MATH.md sections 3.1-3.5]
    """
    if v_titrant_ml < 0.0:
        raise ValueError("titrant volume must be non-negative")
    total, ligands, _, _ = _balance_inputs(model, v_titrant_ml)
    return -solve_log_h(total, ligands, model.kw)


def equivalence_volumes(model: AcidBaseModel) -> list[float]:
    """Theoretical equivalence volumes (mL), one per proton step.

    ``V_k = k * C_a V_a / C_t`` ([MATH.md section 3.1 / 1.2]).

    Example
    -------
    >>> equivalence_volumes(AcidBaseModel.strong_acid(0.1, 25.0, 0.1))
    [25.0]
    """
    unit = model.c_analyte * model.v_analyte_ml / model.c_titrant
    return [k * unit for k in range(1, model.n_steps + 1)]


class AcidBaseEngine(BaseEngine):
    """Stateless acid-base engine.

    Example
    -------
    >>> eng = AcidBaseEngine()
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> s = eng.titrate(m, eng.initial_state(m), 25.0)
    >>> round(s.ph, 3)
    7.0
    """

    def _state(self, model: AcidBaseModel, v_ml: float) -> TitrationState:
        total, ligands, vt, named = _balance_inputs(model, v_ml)
        kw = model.kw
        log_h = solve_log_h(total, ligands, kw)
        h = 10.0**log_h
        species: dict[str, float] = {"H+": h, "OH-": kw / h}
        if "analyte" in named:
            lig, _ = named["analyte"]
            alpha = species_fractions(log_h, lig.log_beta)
            for i, a in enumerate(alpha):
                species[f"H{i}A" if i else "A"] = float(a) * lig.conc
        if "titrant" in named:
            lig, _ = named["titrant"]
            alpha = species_fractions(log_h, lig.log_beta)
            species["T(deprotonated)"] = float(alpha[0]) * lig.conc
            species["T(protonated)"] = float(alpha[1]) * lig.conc
        return TitrationState(v_ml, vt, -log_h, h, species, model.temperature_c)

    def initial_state(self, model: TitrationModel) -> TitrationState:
        """State at V = 0.

        Raises
        ------
        TypeError
            If ``model`` is not an :class:`AcidBaseModel`.
        """
        if not isinstance(model, AcidBaseModel):
            raise TypeError("AcidBaseEngine needs an AcidBaseModel")
        return self._state(model, 0.0)

    def titrate(self, model: TitrationModel, state: TitrationState, d_v_ml: float) -> TitrationState:
        """Pure step ``state_after = titrate(model, state_before, dV)``.

        Raises
        ------
        ValueError
            If ``d_v_ml`` is negative.
        TypeError
            If ``model`` is not an :class:`AcidBaseModel`.

        [MATH.md section 0]
        """
        if not isinstance(model, AcidBaseModel):
            raise TypeError("AcidBaseEngine needs an AcidBaseModel")
        if d_v_ml < 0.0:
            raise ValueError("dV must be non-negative")
        return self._state(model, state.v_titrant_ml + d_v_ml)


def generate_curve(
    model: AcidBaseModel,
    v_max_ml: float,
    n_points: int = 401,
    adaptive: bool = True,
    dph_max: float = 0.05,
    max_points: int = 6000,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Titration curve pH vs V with optional adaptive refinement.

    Near the equivalence point the curve is steep, so intervals whose pH
    change exceeds ``dph_max`` are bisected (second answer 13.4, adaptive
    step).  The grid is therefore *non-uniform*; use
    :func:`analysis.first_derivative` (non-uniform formula) on it.

    Parameters
    ----------
    model : AcidBaseModel
    v_max_ml : float
        Final titrant volume (mL), > 0.
    n_points : int
        Initial number of uniformly spaced points (>= 2).
    adaptive : bool
    dph_max : float
        Maximum pH change allowed between neighbours.
    max_points : int
        Safety cap on the number of points.

    Returns
    -------
    (V, pH) : tuple of ndarray

    Raises
    ------
    ValueError
        If ``v_max_ml <= 0`` or ``n_points < 2``.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> v, ph = generate_curve(m, 30.0, 61)
    >>> bool(v[0] == 0.0 and len(v) >= 61)
    True

    [MATH.md section 13.4]
    """
    if v_max_ml <= 0.0 or n_points < 2:
        raise ValueError("v_max_ml must be positive and n_points >= 2")
    v: list[float] = [float(x) for x in np.linspace(0.0, v_max_ml, n_points)]
    p = [ph_at(model, x) for x in v]
    if adaptive:
        v_min_step = v_max_ml * 1e-7
        while len(v) < max_points:
            new_v: list[float] = []
            new_p: list[float] = []
            for i in range(len(v) - 1):
                if abs(p[i + 1] - p[i]) > dph_max and (v[i + 1] - v[i]) > v_min_step:
                    mid = 0.5 * (v[i] + v[i + 1])
                    new_v.append(mid)
                    new_p.append(ph_at(model, mid))
            if not new_v:
                break
            merged = sorted(zip(v + new_v, p + new_p))
            v = [float(a) for a, _ in merged]
            p = [float(b) for _, b in merged]
    return np.asarray(v), np.asarray(p)


def ph_jump(model: AcidBaseModel, k: int = 1, fraction: float = 1e-3) -> float:
    """Absolute pH change between V_k(1 - fraction) and V_k(1 + fraction).

    Raises
    ------
    ValueError
        If ``k`` is not a valid equivalence index.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> ph_jump(m) > 5
    True

    [MATH.md section 3.3]
    """
    eq = equivalence_volumes(model)
    if not 1 <= k <= len(eq):
        raise ValueError("invalid equivalence index")
    ve = eq[k - 1]
    return abs(ph_at(model, ve * (1.0 + fraction)) - ph_at(model, ve * (1.0 - fraction)))


def check_visual_endpoint(model: AcidBaseModel, k: int = 1) -> FeasibilityReport:
    """Warn if a colour indicator cannot resolve equivalence point ``k``.

    The criterion is the computed jump of at least 2 pH units over +-0.1 % of
    V_eq (section 3.3).  NOTE (contradiction report): the "second answer"
    also states ``Ka < 1e-10`` as a rule of thumb; that threshold is
    concentration dependent, so it is *not* used here.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> check_visual_endpoint(m).visual_ok
    True
    >>> m2 = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [8.0], pkw=14.0)
    >>> check_visual_endpoint(m2).visual_ok
    False
    """
    jump = ph_jump(model, k)
    ok = jump >= 2.0
    weak_weak = model.titrant_strength is TitrantStrength.WEAK and model.analyte_kind in (
        AnalyteKind.WEAK_ACID,
        AnalyteKind.WEAK_BASE,
    )
    if ok:
        msg = f"pH jump {jump:.2f} >= 2: a colour indicator can work."
    elif weak_weak:
        msg = (
            f"pH jump only {jump:.2f} (weak/weak): no ordinary indicator works; "
            "use a potentiometric titration."
        )
    else:
        msg = f"pH jump only {jump:.2f} < 2: use a potentiometric titration."
    return FeasibilityReport(jump, ok, msg)


def with_temperature(model: AcidBaseModel, temperature_c: float) -> AcidBaseModel:
    """Copy of ``model`` at another temperature.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1)
    >>> with_temperature(m, 50.0).temperature_c
    50.0
    """
    return replace(model, temperature_c=temperature_c)
