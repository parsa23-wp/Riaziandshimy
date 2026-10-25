"""Phase-2 titration families: redox, precipitation, complexometry (EDTA), back titration.

Every curve point is an exact closed-form or quadratic solution (no region
shortcuts).  Functions are pure: the result depends only on the arguments.

Scope (v0.8 — the earlier limitations are fixed)
-----------------------------------------------
* Redox: Nernst equation for arbitrary couples, now INCLUDING unequal
  (asymmetric) stoichiometry such as ``Cr2O7 2- + 14 H+ + 6 e -> 2 Cr3+``
  (``a``/``b`` exponents + general numeric solver, exact, no approximation),
  an optional pH shift and an optional ionic-strength (activity) correction
  of the formal potentials.  Redox *kinetics* remain out of scope (all
  electrode reactions are assumed Nernstian — stated on every report).
* Precipitation: 1:1 salt AgX with optional activity-corrected (conditional)
  Ksp; 1:1 complexes such as AgCl(aq)/AgCl2- provably do NOT change the free
  [Ag+]/[X-] curve (proof in MATH.md 16.6) and are provided for solubility
  reporting (:func:`precip_total_dissolved_silver`).
* EDTA: ``K'' = alpha_Y4- · alpha_M · (activity factor) · Kf`` — metal side
  reactions (auxiliary ligand such as NH3, hydrolysis) ARE now modelled
  through :func:`alpha_metal`.
* Constants shipped here are textbook values; check them against your table.
* Kinetics (redox) and non-1:1 salts (Ag2CrO4 as analyte salt) remain out of
  scope.

Reference: [MATH.md sections 16.1-16.8]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .activity import GammaModel, concentration_ksp, formal_potential_shift, gamma
from .base import TitrationModel

R_GAS = 8.314462618
FARADAY = 96485.33212


def nernst_slope(temperature_c: float = 25.0) -> float:
    """``2.303 R T / F`` in volts (0.05916 V at 25 C).

    Example
    -------
    >>> round(nernst_slope(25.0), 5)
    0.05916
    """
    return math.log(10.0) * R_GAS * (temperature_c + 273.15) / FARADAY


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-12) -> float:
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError("root not bracketed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
        if hi - lo < tol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- redox
@dataclass(frozen=True)
class RedoxModel(TitrationModel):
    """Reductant (analyte, couple 1) titrated with an oxidant (titrant, couple 2).

    Attributes
    ----------
    c_analyte, v_analyte_ml, c_titrant : float
    e1, n1 : float
        Standard (formal) potential in V and electrons of the analyte couple.
    e2, n2 : float
        Same for the titrant couple (``e2 > e1`` for a useful titration).
    temperature_c : float
    ph : float | None
        If given with ``m1``/``m2`` the formal potentials shift by
        ``-slope * m/n * pH`` (m = protons in the half reaction).
    m1, m2 : float
    a1, b1, a2, b2 : float
        Exponents of the Nernst ratio of each couple,
        ``E = E°' + (s/n) log([Ox]^a / [Red]^b)``.  Defaults 1/1 (symmetric
        couples).  Example — dichromate: ``a2=1, b2=2``;
        iodine/iodide: ``a2=1, b2=2``.
    atoms1_ox, atoms1_red, atoms2_ox, atoms2_red : float | None
        Metal atoms per formula unit of each form (for the mass balance).
        ``None`` derives them as ``(b, a)`` — correct for Cr2O7 2-/Cr3+,
        I2/I-, Fe3+/Fe2+ and most one-centre couples; set them explicitly
        for multi-atom couples such as S4O6 2- (4 S) / S2O3 2- (2 S).
    z1_ox, z1_red, z2_ox, z2_red : float | None
        Ionic charges; with ``ionic_strength`` they shift the formal
        potentials by the activity term ``E°' = E° + (s/n)(a log γOx - b log γRed)``.
    ionic_strength : float | None
        Mol/L; ``None`` = no activity correction (ideal).
    gamma_model : str
        ``"davies"`` | ``"edh"`` | ``"dh"`` (see :mod:`.activity`).

    Raises
    ------
    ValueError
        On non-positive amounts or ``e2 <= e1``.

    Example
    -------
    >>> m = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)   # Fe2+ / Ce4+
    >>> round(redox_equivalence_potential(m), 3)
    1.105
    """

    c_analyte: float
    v_analyte_ml: float
    c_titrant: float
    e1: float
    n1: float
    e2: float
    n2: float
    temperature_c: float = 25.0
    ph: float | None = None
    m1: float = 0.0
    m2: float = 0.0
    a1: float = 1.0
    b1: float = 1.0
    a2: float = 1.0
    b2: float = 1.0
    atoms1_ox: float | None = None
    atoms1_red: float | None = None
    atoms2_ox: float | None = None
    atoms2_red: float | None = None
    z1_ox: float | None = None
    z1_red: float | None = None
    z2_ox: float | None = None
    z2_red: float | None = None
    ionic_strength: float | None = None
    gamma_model: GammaModel = "davies"

    def __post_init__(self) -> None:
        if min(self.c_analyte, self.v_analyte_ml, self.c_titrant, self.n1, self.n2) <= 0:
            raise ValueError("amounts and electron numbers must be positive")
        if self.e2 <= self.e1:
            raise ValueError("titrant couple must be a stronger oxidant (e2 > e1)")
        if min(self.a1, self.b1, self.a2, self.b2) <= 0:
            raise ValueError("Nernst exponents must be positive")

    @property
    def asymmetric(self) -> bool:
        """True when any Nernst exponent differs from 1 (general solver needed).

        Example
        -------
        >>> RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1).asymmetric
        False
        """
        return any(x != 1.0 for x in (self.a1, self.b1, self.a2, self.b2))

    def _activity_shift(self, couple: int) -> float:
        """Activity part of the formal potential of couple 1 or 2 (V)."""
        if self.ionic_strength is None:
            return 0.0
        if couple == 1:
            z_ox, z_red, n, a, b = self.z1_ox, self.z1_red, self.n1, self.a1, self.b1
        else:
            z_ox, z_red, n, a, b = self.z2_ox, self.z2_red, self.n2, self.a2, self.b2
        if z_ox is None or z_red is None:
            return 0.0
        s = nernst_slope(self.temperature_c)
        return formal_potential_shift(0.0, n, s, z_ox, z_red, self.ionic_strength,
                                      a, b, self.gamma_model, self.temperature_c)

    def _e1f(self) -> float:
        s = nernst_slope(self.temperature_c)
        e = self.e1 - (s * self.m1 / self.n1 * self.ph if self.ph is not None else 0.0)
        return e + self._activity_shift(1)

    def _e2f(self) -> float:
        s = nernst_slope(self.temperature_c)
        e = self.e2 - (s * self.m2 / self.n2 * self.ph if self.ph is not None else 0.0)
        return e + self._activity_shift(2)


def redox_equivalence_volume(m: RedoxModel) -> float:
    """Equivalence volume (mL).

    Symmetric couples: ``V_eq = n1 C1 V1 / (n2 C2)``.  Asymmetric couples
    (or explicit atom counts): the exact stoichiometric form
    ``V_eq = n1 c_red1 b2 c_red2 C1 V1 / (a1 c_ox1 n2 c_ox2 C2)`` —
    electrons given by the analyte = electrons accepted by the titrant.

    Example
    -------
    >>> redox_equivalence_volume(RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1))
    50.0
    >>> m = RedoxModel(0.1, 50.0, 1.0 / 60.0, 0.771, 1, 1.33, 6, a2=1, b2=2)  # dichromate
    >>> round(redox_equivalence_volume(m), 6)
    50.0
    """
    if not (m.asymmetric or m.atoms1_ox is not None or m.atoms1_red is not None
            or m.atoms2_ox is not None or m.atoms2_red is not None):
        return m.n1 * m.c_analyte * m.v_analyte_ml / (m.n2 * m.c_titrant)
    c_red1 = m.atoms1_red if m.atoms1_red is not None else m.a1
    c_ox1 = m.atoms1_ox if m.atoms1_ox is not None else m.b1
    c_ox2 = m.atoms2_ox if m.atoms2_ox is not None else m.b2
    c_red2 = m.atoms2_red if m.atoms2_red is not None else m.a2
    return (m.n1 * c_red1 * m.b2 * c_red2 * m.c_analyte * m.v_analyte_ml
            / (m.a1 * c_ox1 * m.n2 * c_ox2 * m.c_titrant))


def redox_equivalence_potential(m: RedoxModel) -> float:
    """Equivalence potential (V).

    Symmetric couples (all exponents 1): the exact closed form
    ``E_eq = (n1 E1' + n2 E2')/(n1 + n2)`` (both couples soluble, no H+ term
    in the ratio).  Asymmetric couples (e.g. Cr2O7 2-/2 Cr3+): the exact
    value is obtained from the general electron-balance solver
    (:func:`redox_potential_general`), replacing the earlier approximation.

    Example
    -------
    >>> round(redox_equivalence_potential(RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)), 3)
    1.105
    """
    if m.asymmetric:
        return redox_potential_general(m, redox_equivalence_volume(m))
    return (m.n1 * m._e1f() + m.n2 * m._e2f()) / (m.n1 + m.n2)


def redox_potential(m: RedoxModel, v_ml: float) -> float:
    """Electrode potential (V) after ``v_ml`` mL titrant.

    Symmetric couples (all exponents 1): ``f = V/V_eq``; before eq
    ``E = E1' + (s/n1) log(f/(1-f))``, after ``E = E2' + (s/n2) log(f-1)``,
    at f = 1 the equivalence potential — all exact.
    Asymmetric couples: the exact general solver
    (:func:`redox_potential_general`) is used automatically.

    Raises
    ------
    ValueError
        If ``v_ml <= 0`` (E is -infinity before any titrant) or negative.

    Example
    -------
    >>> m = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)
    >>> round(redox_potential(m, 25.0), 3)      # half titrated: E = E1
    0.77
    >>> round(redox_potential(m, 100.0), 3)     # f = 2: E = E2
    1.44

    [MATH.md section 16.1]
    """
    if v_ml <= 0.0:
        raise ValueError("potential undefined at V <= 0 (no Ox form yet)")
    if m.asymmetric:
        return redox_potential_general(m, v_ml)
    s = nernst_slope(m.temperature_c)
    f = v_ml / redox_equivalence_volume(m)
    if abs(f - 1.0) < 1e-12:
        return redox_equivalence_potential(m)
    if f < 1.0:
        return m._e1f() + s / m.n1 * math.log10(f / (1.0 - f))
    return m._e2f() + s / m.n2 * math.log10(f - 1.0)


def redox_curve(m: RedoxModel, v_max_ml: float | None = None, n: int = 400) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``(V, E)`` arrays from the first point to ``v_max_ml`` (default 2 V_eq).

    Example
    -------
    >>> v, e = redox_curve(RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1), n=50)
    >>> bool(e[0] < e[-1])
    True
    """
    top = v_max_ml if v_max_ml is not None else 2.0 * redox_equivalence_volume(m)
    v = np.linspace(top / n, top, n)
    return v, np.array([redox_potential(m, float(x)) for x in v])


def _couple_split(t_atoms: float, log_r: float, a: float, b: float,
                  c_ox: float, c_red: float) -> tuple[float, float]:
    """Split a couple's metal-atom total into ``([Ox], [Red])`` at a trial E.

    Nernst ratio ``[Ox]^a / [Red]^b = r`` with ``log10 r = log_r`` and mass
    balance ``T = c_ox [Ox] + c_red [Red]`` (metal atoms).  Solved exactly by
    bisection on ``[Red]`` — monotone, single root, no region shortcuts.

    Raises
    ------
    ValueError
        If the mass balance is unsatisfiable (non-positive totals).
    """
    if t_atoms <= 0.0 or c_ox <= 0.0 or c_red <= 0.0:
        raise ValueError("couple total and atom counts must be positive")
    hi = t_atoms / c_red
    lo = 0.0
    for _ in range(200):
        red = 0.5 * (lo + hi)
        lg_ox = (log_r + b * math.log10(red)) / a if red > 0.0 else -math.inf
        lg_ox = min(max(lg_ox, -350.0), 350.0)
        g = c_ox * 10.0**lg_ox + c_red * red - t_atoms
        if g < 0.0:
            lo = red
        else:
            hi = red
        if hi - lo <= 1e-15 * max(hi, lo):
            break
    red = 0.5 * (lo + hi)
    lg_ox = (log_r + b * math.log10(red)) / a if red > 0.0 else -math.inf
    lg_ox = min(max(lg_ox, -350.0), 350.0)
    return 10.0**lg_ox, red


def redox_species(m: RedoxModel, v_ml: float) -> dict[str, float]:
    """Free concentrations (mol/L) ``{"Ox1","Red1","Ox2","Red2"}`` after ``v_ml`` mL.

    Exact for symmetric AND asymmetric couples (general electron balance
    ``n1 [Ox1]/a1 = n2 [Red2]/b2`` in metal-atom concentrations).  This is
    the building block for the instrumental layers (photometry, conductometry).

    Raises
    ------
    ValueError
        If ``v_ml < 0``.

    Example
    -------
    >>> m = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)
    >>> s = redox_species(m, 25.0)
    >>> round(s["Ox1"], 4), round(s["Red1"], 4)
    (0.0333, 0.0333)
    """
    if v_ml < 0.0:
        raise ValueError("volume must be non-negative")
    s = nernst_slope(m.temperature_c)
    vt = m.v_analyte_ml + v_ml
    c_ox1 = m.atoms1_ox if m.atoms1_ox is not None else m.b1
    c_red1 = m.atoms1_red if m.atoms1_red is not None else m.a1
    c_ox2 = m.atoms2_ox if m.atoms2_ox is not None else m.b2
    c_red2 = m.atoms2_red if m.atoms2_red is not None else m.a2
    t1 = c_red1 * m.c_analyte * m.v_analyte_ml / vt   # metal-atom totals
    t2 = c_ox2 * m.c_titrant * v_ml / vt
    if v_ml == 0.0 or t2 == 0.0:
        return {"Ox1": 0.0, "Red1": t1 / c_red1, "Ox2": 0.0, "Red2": 0.0}

    def balance(e: float) -> float:
        ox1, _ = _couple_split(t1, m.n1 * (e - m._e1f()) / s, m.a1, m.b1, c_ox1, c_red1)
        _, red2 = _couple_split(t2, m.n2 * (e - m._e2f()) / s, m.a2, m.b2, c_ox2, c_red2)
        return m.n1 * ox1 / m.a1 - m.n2 * red2 / m.b2

    e_lo = min(m._e1f(), m._e2f()) - 40.0 * s / min(m.n1, m.n2)
    e_hi = max(m._e1f(), m._e2f()) + 40.0 * s / min(m.n1, m.n2)
    e = _bisect(balance, e_lo, e_hi)
    ox1, red1 = _couple_split(t1, m.n1 * (e - m._e1f()) / s, m.a1, m.b1, c_ox1, c_red1)
    ox2, red2 = _couple_split(t2, m.n2 * (e - m._e2f()) / s, m.a2, m.b2, c_ox2, c_red2)
    return {"Ox1": ox1, "Red1": red1, "Ox2": ox2, "Red2": red2}


def redox_potential_general(m: RedoxModel, v_ml: float) -> float:
    """Potential (V) from the general electron-balance solver (exact, any stoichiometry).

    Solves BOTH couples' Nernst equations simultaneously with the electron
    balance ``n1 [Ox1]/a1 = n2 [Red2]/b2`` and the metal-atom mass balances —
    no region shortcuts.  For symmetric couples it reproduces the textbook
    closed form to < 3e-5 V in the steepest part of the jump (the small
    difference is the [Ox2] leak that the closed form neglects) and to
    < 1e-9 V elsewhere.  It is the slow path; :func:`redox_potential`
    dispatches to it only for asymmetric couples.

    Raises
    ------
    ValueError
        If ``v_ml <= 0``.

    Example
    -------
    >>> m = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)
    >>> round(redox_potential_general(m, 25.0), 6) == round(redox_potential(m, 25.0), 6)
    True
    """
    if v_ml <= 0.0:
        raise ValueError("potential undefined at V <= 0 (no Ox form yet)")
    s = nernst_slope(m.temperature_c)
    vt = m.v_analyte_ml + v_ml
    c_ox1 = m.atoms1_ox if m.atoms1_ox is not None else m.b1
    c_red1 = m.atoms1_red if m.atoms1_red is not None else m.a1
    c_ox2 = m.atoms2_ox if m.atoms2_ox is not None else m.b2
    c_red2 = m.atoms2_red if m.atoms2_red is not None else m.a2
    t1 = c_red1 * m.c_analyte * m.v_analyte_ml / vt
    t2 = c_ox2 * m.c_titrant * v_ml / vt

    def balance(e: float) -> float:
        ox1, _ = _couple_split(t1, m.n1 * (e - m._e1f()) / s, m.a1, m.b1, c_ox1, c_red1)
        _, red2 = _couple_split(t2, m.n2 * (e - m._e2f()) / s, m.a2, m.b2, c_ox2, c_red2)
        return m.n1 * ox1 / m.a1 - m.n2 * red2 / m.b2

    e_lo = min(m._e1f(), m._e2f()) - 40.0 * s / min(m.n1, m.n2)
    e_hi = max(m._e1f(), m._e2f()) + 40.0 * s / min(m.n1, m.n2)
    return _bisect(balance, e_lo, e_hi)


# ---------------------------------------------------------- precipitation
@dataclass(frozen=True)
class PrecipitationModel(TitrationModel):
    """Halide ``X-`` (analyte) titrated with ``Ag+`` (1:1 salt AgX).

    Attributes
    ----------
    c_analyte, v_analyte_ml, c_titrant : float
    ksp : float
        Solubility product of AgX (AgCl 1.8e-10, AgBr 5.0e-13, AgI 8.5e-17).
        Thermodynamic (I = 0) when ``ionic_strength`` is given; the
        conditional product ``Ksp' = Ksp°/(γAg γX)`` is applied automatically.
    ionic_strength : float
        Mol/L; 0 = ideal (no correction), matching the earlier behaviour.
    gamma_model, temperature_c :
        See :mod:`.activity`.

    Example
    -------
    >>> m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    >>> round(precip_pag(m, 50.0), 3)           # equivalence: pAg = pKsp/2
    4.872
    """

    c_analyte: float
    v_analyte_ml: float
    c_titrant: float
    ksp: float
    ionic_strength: float = 0.0
    gamma_model: GammaModel = "davies"
    temperature_c: float = 25.0

    def __post_init__(self) -> None:
        if min(self.c_analyte, self.v_analyte_ml, self.c_titrant, self.ksp) <= 0:
            raise ValueError("all parameters must be positive")
        if self.ionic_strength < 0:
            raise ValueError("ionic strength must be non-negative")

    @property
    def ksp_eff(self) -> float:
        """Conditional (concentration) Ksp at the model ionic strength.

        Example
        -------
        >>> m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10, ionic_strength=0.1)
        >>> m.ksp_eff > m.ksp
        True
        """
        if self.ionic_strength == 0.0:
            return self.ksp
        return concentration_ksp(self.ksp, self.ionic_strength, 1, -1,
                                 self.gamma_model, self.temperature_c)


def precip_equivalence_volume(m: PrecipitationModel) -> float:
    """``V_eq = C_x V_x / C_t`` (mL).

    Example
    -------
    >>> precip_equivalence_volume(PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10))
    50.0
    """
    return m.c_analyte * m.v_analyte_ml / m.c_titrant


def precip_ag(m: PrecipitationModel, v_ml: float) -> float:
    """Free ``[Ag+]`` (mol/L) with precipitate present (exact).

    With ``d = Ag_tot - X_tot``: ``[Ag+] = (d + sqrt(d^2 + 4 Ksp'))/2``
    (from ``[Ag+][X-] = Ksp'`` and ``[Ag+] - [X-] = d``).  ``Ksp'`` is the
    activity-corrected product when an ionic strength is set.  If the
    solubility product is not exceeded (``Ag_tot X_tot <= Ksp'``) there is no
    solid and ``[Ag+] = Ag_tot``.

    1:1 complexes (AgCl aq, AgCl2-) do NOT change this free concentration:
    they contribute one Ag and one X each, so they cancel in the difference
    balance ``[Ag+] - [X-] = Ag_tot - X_tot`` (proof in MATH.md 16.6); only
    the *total dissolved* silver changes — see
    :func:`precip_total_dissolved_silver`.

    Raises
    ------
    ValueError
        If ``v_ml`` is negative.

    [MATH.md section 16.2]
    """
    if v_ml < 0.0:
        raise ValueError("volume must be non-negative")
    vt = m.v_analyte_ml + v_ml
    ag_t = m.c_titrant * v_ml / vt
    x_t = m.c_analyte * m.v_analyte_ml / vt
    ksp = m.ksp_eff
    if ag_t * x_t <= ksp:
        return ag_t
    d = ag_t - x_t
    return 0.5 * (d + math.sqrt(d * d + 4.0 * ksp))


def precip_pag(m: PrecipitationModel, v_ml: float) -> float:
    """``pAg = -log10 [Ag+]``; at V = 0 there is no silver, so ``inf`` is returned.

    Example
    -------
    >>> m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    >>> precip_pag(m, 0.0)
    inf
    """
    ag = precip_ag(m, v_ml)
    return math.inf if ag <= 0.0 else -math.log10(ag)


def precip_px(m: PrecipitationModel, v_ml: float) -> float:
    """``pX = -log10 [X-]`` (``[X-] = Ksp'/[Ag+]`` with solid, else the total).

    Example
    -------
    >>> m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    >>> round(precip_px(m, 0.0), 3)
    1.0
    """
    vt = m.v_analyte_ml + v_ml
    x_t = m.c_analyte * m.v_analyte_ml / vt
    ag = precip_ag(m, v_ml)
    ag_t = m.c_titrant * v_ml / vt
    x = x_t if ag_t * x_t <= m.ksp_eff else m.ksp_eff / ag
    return -math.log10(x)


AGCL_LOG_BETA = (3.31, 5.25)
"""Cumulative log beta of AgCl(aq) and AgCl2- (25 C, I ~ 0); textbook values."""


def precip_total_dissolved_silver(m: PrecipitationModel, v_ml: float,
                                  log_beta: Sequence[float] = AGCL_LOG_BETA) -> float:
    """Total dissolved silver (free + 1:1 complexes) in mol/L at ``v_ml``.

    ``Ag_tot,diss = [Ag+] (1 + beta1 [X-] + beta2 [X-]^2 + ...)`` using the
    free concentrations from the exact quadratic.  With a precipitate present
    this exceeds the free ``[Ag+]`` (that is exactly why AgCl dissolves in
    concentrated chloride), although — being strictly 1:1 — the complexes do
    not shift the pAg curve itself (MATH.md 16.6).

    Example
    -------
    >>> m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    >>> precip_total_dissolved_silver(m, 25.0) > precip_ag(m, 25.0)
    True
    """
    if v_ml < 0.0:
        raise ValueError("volume must be non-negative")
    vt = m.v_analyte_ml + v_ml
    ag_t = m.c_titrant * v_ml / vt
    x_t = m.c_analyte * m.v_analyte_ml / vt
    ag = precip_ag(m, v_ml)
    x_free = m.ksp_eff / ag if ag_t * x_t > m.ksp_eff else x_t
    factor = 1.0
    for i, lb in enumerate(log_beta, start=1):
        factor += 10.0**lb * x_free**i
    return ag * factor


def mohr_endpoint_volume(m: PrecipitationModel, c_cro4: float = 0.0025, ksp_ag2cro4: float = 1.1e-12) -> float:
    """Volume (mL) at which red Ag2CrO4 appears (Mohr method).

    Ag2CrO4 precipitates when ``[Ag+] = sqrt(Ksp_chromate/[CrO4 2-])``;
    ``[CrO4]`` is taken constant (dilution of the indicator ignored:
    APPROXIMATION, error ~ V_eq-fraction of dilution).

    Raises
    ------
    ValueError
        If the endpoint ``[Ag+]`` is never reached before 3 V_eq.

    Example
    -------
    >>> m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    >>> 50.0 < mohr_endpoint_volume(m) < 50.2
    True

    [MATH.md section 16.2]
    """
    target = math.sqrt(ksp_ag2cro4 / c_cro4)
    ve = precip_equivalence_volume(m)
    return _bisect(lambda v: precip_ag(m, v) - target, 1e-9, 3.0 * ve)


def volhard_result(c_ag: float, v_ag_ml: float, c_scn: float, v_scn_ml: float, v_sample_ml: float) -> float:
    """Halide concentration (mol/L) by Volhard back titration.

    ``n_X = C_Ag V_Ag - C_SCN V_SCN`` (1:1:1).

    Raises
    ------
    ValueError
        If the thiocyanate consumed exceeds the silver added.

    Example
    -------
    >>> round(volhard_result(0.1, 50.0, 0.1, 20.0, 25.0), 4)
    0.12
    """
    n = c_ag * v_ag_ml - c_scn * v_scn_ml
    if n < 0:
        raise ValueError("more SCN- than Ag+: inconsistent data")
    return n / v_sample_ml


# ------------------------------------------------------------------ Fajans
FAJANS_DYES = {"fluorescein": 6.4, "dichlorofluorescein": 4.0, "eosin": 2.5}
"""pKa of the adsorption indicators (textbook approximations).

The anionic form adsorbs on the positively charged AgX surface just after
the equivalence point; the dye must therefore be deprotonated, i.e.
``pH > pKa`` (:func:`fajans_ph_ok`).
"""


def fajans_ph_ok(ph: float, dye: str = "dichlorofluorescein", margin: float = 0.0) -> bool:
    """Is the adsorption indicator usable at this pH?

    The dye only works in its anionic form (fraction > 50 %), i.e. the pH
    must exceed the dye pKa (plus an optional safety ``margin``).  Fluorescein
    therefore needs pH > ~6.5, dichlorofluorescein > ~4, eosin > ~2.5.

    Raises
    ------
    ValueError
        Unknown dye.

    Example
    -------
    >>> fajans_ph_ok(7.0, "fluorescein")
    True
    >>> fajans_ph_ok(5.0, "fluorescein")
    False
    >>> fajans_ph_ok(5.0, "dichlorofluorescein")
    True
    """
    if dye not in FAJANS_DYES:
        raise ValueError(f"unknown dye {dye!r}")
    return ph > FAJANS_DYES[dye] + margin


def fajans_endpoint_volume(m: PrecipitationModel, ag_threshold: float | None = None) -> float:
    """Volume (mL) at which the adsorption dye flips colour (Fajans method).

    The colour change occurs when the free ``[Ag+]`` first exceeds the
    adsorption threshold (the AgX surface turns positive).  ``ag_threshold``
    defaults to ``1.2 * sqrt(Ksp')`` — just past the equivalence value
    ``sqrt(Ksp')``, i.e. within a fraction of a drop of the stoichiometric
    point, which is why the Fajans error is tiny for realistic
    concentrations.  Pass a larger threshold to model a sluggish dye or an
    under-protected colloid.

    Raises
    ------
    ValueError
        If the threshold is never reached before 3 V_eq.

    Example
    -------
    >>> m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    >>> err = fajans_indicator_error_percent(m)
    >>> abs(err) < 0.2
    True

    [MATH.md section 16.7]
    """
    ksp = m.ksp_eff
    target = ag_threshold if ag_threshold is not None else 1.2 * math.sqrt(ksp)
    ve = precip_equivalence_volume(m)
    return _bisect(lambda v: precip_ag(m, v) - target, 1e-12, 3.0 * ve)


def fajans_indicator_error_percent(m: PrecipitationModel, ag_threshold: float | None = None) -> float:
    """Relative endpoint error of the Fajans method (percent).

    Example
    -------
    >>> m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    >>> -0.2 < fajans_indicator_error_percent(m) < 0.2
    True
    """
    ve = precip_equivalence_volume(m)
    return (fajans_endpoint_volume(m, ag_threshold) - ve) / ve * 100.0


# --------------------------------------------------------- complexometry
EDTA_PKA = (1.99, 2.67, 6.16, 10.26)
"""pKa1..pKa4 of H4Y: 1.99, 2.67, 6.16, 10.26 (K1..K4 = 1.02e-2, 2.14e-3, 6.92e-7, 5.50e-11, Skoog/West/Holler; I ~ 0.1 M).
Reproduces the tabulated alpha_Y4-: 2.5e-11 (pH 3), 5.4e-3 (pH 8), 0.35 (pH 10), 0.98 (pH 12).  pKa5/6 (1.5/0) irrelevant above pH 2."""

LOG_KF = {"Ca": 10.65, "Mg": 8.79, "Zn": 16.5, "Cu": 18.8, "Fe3": 25.1, "Pb": 18.0, "Ni": 18.6}
"""log Kf of MY (about 20 C, I = 0.1 M)."""


NH3_LOG_BETA = {
    "Ag": (3.31, 7.21),
    "Cu": (4.31, 7.98, 11.02, 13.32),
    "Zn": (2.37, 4.81, 7.31, 9.46),
    "Ni": (2.80, 5.04, 6.77, 7.96, 8.71, 8.74),
    "Cd": (2.65, 4.75, 6.19, 7.12, 6.80, 5.14),
    "Co2": (2.11, 3.74, 4.79, 5.55, 5.73, 5.69),
}
"""Cumulative log beta of the ammine complexes M(NH3)_n (25 C, I ~ 0.1).

Metal side reactions are now part of the model: pass ``aux_log_beta`` =
``NH3_LOG_BETA[metal]`` and ``aux_ligand_conc`` = free [NH3] to
:class:`EdtaModel`.  Textbook values — check against your table.
"""


def alpha_metal(aux_ligand_conc: float = 0.0, aux_log_beta: Sequence[float] = (),
                oh_log_beta: Sequence[float] = (), ph: float = 7.0,
                pkw: float = 14.0) -> float:
    """Side-reaction coefficient ``alpha_M = [M_free]/[M']`` (0 < alpha_M <= 1).

    ``[M']`` is the sum of all un-complexed-by-EDTA metal forms:

    ``[M'] = [M] (1 + sum_l beta_L,l [L]^l + sum_j beta_OH,j [OH-]^j)``

    ``[OH-] = 10^(ph - pkw)`` (25 C default pkw = 14).  For example Zn in a
    pH-10 ammoniacal buffer: ``alpha_M ~ 1e-5`` at 0.1 M NH3, which lowers
    the effective constant by 5 log units.

    Raises
    ------
    ValueError
        If the ligand concentration is negative.

    Example
    -------
    >>> round(math.log10(alpha_metal(0.1, NH3_LOG_BETA["Zn"], ph=10.0)), 2)
    -5.49
    >>> alpha_metal() == 1.0
    True
    """
    if aux_ligand_conc < 0.0:
        raise ValueError("ligand concentration must be non-negative")
    s = 1.0
    for l, lb in enumerate(aux_log_beta, start=1):
        s += 10.0**lb * aux_ligand_conc**l
    oh = 10.0 ** (ph - pkw)
    for j, lb in enumerate(oh_log_beta, start=1):
        s += 10.0**lb * oh**j
    return 1.0 / s


def alpha_y4(ph: float, pka: Sequence[float] = EDTA_PKA) -> float:
    """Fraction of EDTA present as Y4- at ``ph``.

    ``alpha = 1 / (1 + sum_i beta_i h^i)`` with cumulative protonation
    constants from the pKa values.

    Example
    -------
    >>> round(alpha_y4(10.0), 3)
    0.355
    >>> round(alpha_y4(12.0), 3)
    0.982
    """
    h = 10.0 ** (-ph)
    s = 1.0
    beta = 1.0
    for k, p in enumerate(reversed(pka), start=1):
        beta *= 10.0**p
        s += beta * h**k
    return 1.0 / s


@dataclass(frozen=True)
class EdtaModel(TitrationModel):
    """Metal ``M`` titrated with EDTA at fixed pH (1:1).

    Attributes
    ----------
    c_metal, v_metal_ml, c_edta : float
    log_kf : float
        log10 of the absolute formation constant (see :data:`LOG_KF`).
    ph : float
    aux_ligand_conc : float
        Free concentration of an auxiliary ligand (e.g. NH3 in a pH-10
        buffer); 0 = none.
    aux_log_beta : tuple
        Cumulative log beta of the metal's auxiliary-ligand complexes
        (e.g. :data:`NH3_LOG_BETA`).
    oh_log_beta : tuple
        Cumulative log beta of the metal hydroxo complexes (user-supplied;
        e.g. Fe3+ ~ (11.8, 21.2)).
    ionic_strength : float
        Mol/L; 0 = no activity correction.  With I > 0 the thermodynamic Kf
        is converted to a concentration constant through
        ``gamma_M gamma_Y4- / gamma_MY`` (charges ``z_metal``, -4, ``z_metal-4``).
    z_metal : float
    gamma_model, temperature_c :
        See :mod:`.activity`.

    Example
    -------
    >>> m = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Ca"], 10.0)
    >>> round(math.log10(conditional_kf(m)), 2)
    10.2
    """

    c_metal: float
    v_metal_ml: float
    c_edta: float
    log_kf: float
    ph: float
    aux_ligand_conc: float = 0.0
    aux_log_beta: tuple[float, ...] = ()
    oh_log_beta: tuple[float, ...] = ()
    ionic_strength: float = 0.0
    z_metal: float = 2.0
    gamma_model: GammaModel = "davies"
    temperature_c: float = 25.0

    def __post_init__(self) -> None:
        if min(self.c_metal, self.v_metal_ml, self.c_edta) <= 0:
            raise ValueError("amounts must be positive")
        if self.aux_ligand_conc < 0 or self.ionic_strength < 0:
            raise ValueError("ligand concentration and ionic strength must be non-negative")


def conditional_kf(m: EdtaModel) -> float:
    """Effective (conditional) formation constant of MY at the model conditions.

    ``K'' = Kf · alpha_Y4- · alpha_M · gamma_M gamma_Y4- / gamma_MY``

    The alpha factors cover the EDTA protonation (pH) and the metal side
    reactions (auxiliary ligand, hydrolysis); the gamma factor converts the
    thermodynamic constant to a concentration constant at the model ionic
    strength (omitted when ``ionic_strength == 0``).

    Example
    -------
    >>> m = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Ca"], 10.0)
    >>> round(math.log10(conditional_kf(m)), 2)
    10.2
    >>> m_zn = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Zn"], 10.0,
    ...                  aux_ligand_conc=0.1, aux_log_beta=NH3_LOG_BETA["Zn"])
    >>> round(math.log10(conditional_kf(m_zn)), 2)
    10.56
    """
    k = float(10.0**m.log_kf * alpha_y4(m.ph) * alpha_metal(
        m.aux_ligand_conc, m.aux_log_beta, m.oh_log_beta, m.ph))
    if m.ionic_strength > 0.0:
        k *= (gamma(m.z_metal, m.ionic_strength, m.gamma_model, m.temperature_c)
              * gamma(-4.0, m.ionic_strength, m.gamma_model, m.temperature_c)
              / max(gamma(m.z_metal - 4.0, m.ionic_strength, m.gamma_model,
                          m.temperature_c), 1e-300))
    return k


def edta_equivalence_volume(m: EdtaModel) -> float:
    """``V_eq = C_M V_M / C_Y`` (mL).

    Example
    -------
    >>> edta_equivalence_volume(EdtaModel(0.01, 50.0, 0.01, 10.65, 10.0))
    50.0
    """
    return m.c_metal * m.v_metal_ml / m.c_edta


def edta_pm(m: EdtaModel, v_ml: float) -> float:
    """``pM = -log10 [M]`` (free metal) after ``v_ml`` mL EDTA (exact).

    Mass balance with ``K' = [MY]/([M][Y'])``:
    ``K' M^2 + (1 + K'(Y_tot - M_tot)) M - M_tot = 0``; the positive root is
    taken in a cancellation-free form.

    Raises
    ------
    ValueError
        If ``v_ml`` is negative.

    Example
    -------
    >>> m = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Ca"], 10.0)
    >>> round(edta_pm(m, 0.0), 3)
    2.0
    >>> edta_pm(m, 50.0) > 5.5          # equivalence point
    True

    [MATH.md section 16.3]
    """
    if v_ml < 0.0:
        raise ValueError("volume must be non-negative")
    vt = m.v_metal_ml + v_ml
    mt = m.c_metal * m.v_metal_ml / vt
    yt = m.c_edta * v_ml / vt
    k = conditional_kf(m)
    b = 1.0 + k * (yt - mt)
    disc = math.sqrt(b * b + 4.0 * k * mt)
    # positive root of k M^2 + b M - mt = 0, stable for both signs of b
    root = 2.0 * mt / (b + disc) if b >= 0.0 else (-b + disc) / (2.0 * k)
    return -math.log10(root)


def hardness_mg_per_l_caco3(c_edta: float, v_edta_ml: float, v_sample_ml: float) -> float:
    """Total hardness as mg/L CaCO3 (M = 100.09 g/mol), 1:1.

    Example
    -------
    >>> round(hardness_mg_per_l_caco3(0.01, 12.5, 100.0), 1)
    125.1
    """
    return c_edta * v_edta_ml / v_sample_ml * 100.09 * 1000.0


# ------------------------------------------------------- back titration
@dataclass(frozen=True)
class BackTitrationResult:
    """Outcome of a back titration.

    Attributes
    ----------
    n_analyte_mmol, c_analyte : float
    n_reagent_consumed_mmol : float
    """

    n_analyte_mmol: float
    c_analyte: float
    n_reagent_consumed_mmol: float


def back_titration(
    c_reagent: float,
    v_reagent_ml: float,
    c_titrant: float,
    v_titrant_ml: float,
    v_sample_ml: float,
    reagent_per_analyte: float = 1.0,
    titrant_per_reagent: float = 1.0,
) -> BackTitrationResult:
    """Excess-reagent method.

    ``n_excess = C_t V_t / titrant_per_reagent``; ``n_reacted = n_reagent - n_excess``;
    ``n_analyte = n_reacted / reagent_per_analyte``.

    Raises
    ------
    ValueError
        If the titrant consumed exceeds the reagent added (negative consumption).

    Example
    -------
    >>> r = back_titration(0.1, 50.0, 0.1, 20.0, 25.0)         # 5 - 2 = 3 mmol
    >>> round(r.n_analyte_mmol, 3), round(r.c_analyte, 3)
    (3.0, 0.12)

    [MATH.md section 16.4]
    """
    n_excess = c_titrant * v_titrant_ml / titrant_per_reagent
    n_reacted = c_reagent * v_reagent_ml - n_excess
    if n_reacted < 0.0:
        raise ValueError("excess reagent exceeds the amount added: inconsistent data")
    n_a = n_reacted / reagent_per_analyte
    return BackTitrationResult(n_a, n_a / v_sample_ml, n_reacted)
