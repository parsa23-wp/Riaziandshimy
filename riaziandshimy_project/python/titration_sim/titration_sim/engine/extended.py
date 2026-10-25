"""Extended families (v1.0): non-aqueous, amperometry, thermometry, Karl Fischer,
coulometry, titrant standardization and a GUM uncertainty budget.

Every function is pure.  Honest limits are stated per section; constants are
textbook values and each table says whether it was cross-checked against a
source (``verified``) or is only a commonly quoted figure.

* Non-aqueous: the aqueous proton balance is reused with ``pKw -> pKs`` of the
  solvent.  Ion pairing / homoconjugation are NOT modelled, so the pKa values
  you give must be the *apparent* constants measured in that solvent.
* Amperometry: diffusion-limited currents proportional to concentration, with
  an optional dilution correction.  No migration, no capacity current.
* Thermometry: adiabatic cell, constant heat capacity, no heat loss.
* Karl Fischer / coulometry: Faraday's law and the 1:1 water:iodine rule.
* GUM: first-order propagation with numerical sensitivity coefficients
  (no Monte Carlo, no correlations).

Reference: [MATH.md section 18]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .acid_base import AcidBaseModel, ph_at

FARADAY = 96485.33212
M_WATER = 18.01528

# ----------------------------------------------------------- non-aqueous
SOLVENT_PKS: dict[str, tuple[float, bool]] = {
    "water": (14.0, True),          # 13.995 at 25 C; use pkw_water(T) for exact work
    "acetic_acid": (14.45, True),   # Kolthoff/Bruckenstein, JACS 78 (1956) 2974
    "methanol": (16.7, True),       # IUPAC compilation: 16.7
    "ethanol": (19.5, True),        # 19.4-19.5
    "acetonitrile": (33.3, False),  # commonly quoted, NOT cross-checked here
    "dmso": (33.3, False),          # commonly quoted, NOT cross-checked here
}
"""``solvent -> (pKs, verified)`` at 25 C; ``verified=False`` means: check it yourself."""


def solvent_pks(solvent: str) -> float:
    """pKs of ``solvent`` (key of :data:`SOLVENT_PKS`).

    >>> solvent_pks("acetic_acid")
    14.45

    Raises
    ------
    ValueError
        For an unknown solvent.
    """
    if solvent not in SOLVENT_PKS:
        raise ValueError(f"unknown solvent {solvent!r}; known: {sorted(SOLVENT_PKS)}")
    return SOLVENT_PKS[solvent][0]


def nonaqueous_weak_base(solvent: str, c: float, v_ml: float, c_titrant: float, pka_conjugate: float) -> AcidBaseModel:
    """Weak base titrated with a strong acid (e.g. HClO4) in ``solvent``.

    ``pka_conjugate`` is the apparent pKa of BH+ *in that solvent*.  The 'pH'
    of the curve is the solvent's own scale (``pAH2+``), spanning ``0..pKs``.

    >>> m = nonaqueous_weak_base("acetic_acid", 0.1, 25.0, 0.1, 6.1)
    >>> m.pkw
    14.45
    """
    return AcidBaseModel.weak_base(c, v_ml, c_titrant, [pka_conjugate], pkw=solvent_pks(solvent))


def nonaqueous_weak_acid(solvent: str, c: float, v_ml: float, c_titrant: float, pka: float) -> AcidBaseModel:
    """Weak acid titrated with a strong base (e.g. TBAOH / NaOMe) in ``solvent``.

    >>> m = nonaqueous_weak_acid("ethanol", 0.1, 25.0, 0.1, 10.0)
    >>> m.pkw
    19.5
    """
    return AcidBaseModel.weak_acid(c, v_ml, c_titrant, [pka], pkw=solvent_pks(solvent))


def nonaqueous_jump(model: AcidBaseModel, fraction: float = 0.001) -> float:
    """Jump of the solvent 'pH' between ``(1-f)`` and ``(1+f)`` of the equivalence volume.

    >>> m = nonaqueous_weak_acid("acetonitrile", 0.1, 25.0, 0.1, 20.0)
    >>> nonaqueous_jump(m) > 5.0       # impossible in water, easy here
    True
    """
    ve = model.c_analyte * model.v_analyte_ml / model.c_titrant
    return abs(ph_at(model, ve * (1.0 + fraction)) - ph_at(model, ve * (1.0 - fraction)))


# ------------------------------------------------- end point from a signal
def two_line_endpoint(v: Sequence[float], y: Sequence[float], expected: float, exclude: float = 0.1) -> float:
    """Intersection of the straight lines fitted before and after ``expected``.

    Points within ``+-exclude*expected`` of the break are ignored (curvature).

    Raises
    ------
    ValueError
        With fewer than 3 points on a side or parallel lines.

    >>> v = [1, 2, 3, 4, 7, 8, 9, 10.0]
    >>> y = [10 - x for x in v[:4]] + [5 + 2 * (x - 5) for x in v[4:]]
    >>> round(two_line_endpoint(v, y, 5.0, 0.1), 3)
    5.0
    """
    va, ya = np.asarray(v, dtype=float), np.asarray(y, dtype=float)
    lo, hi = va < expected * (1.0 - exclude), va > expected * (1.0 + exclude)
    if lo.sum() < 3 or hi.sum() < 3:
        raise ValueError("need at least 3 points on each side of the end point")
    s1, i1 = np.polyfit(va[lo], ya[lo], 1)
    s2, i2 = np.polyfit(va[hi], ya[hi], 1)
    if abs(s1 - s2) < 1e-12 * max(1.0, abs(s1), abs(s2)):
        raise ValueError("lines are parallel")
    return float((i2 - i1) / (s1 - s2))


# --------------------------------------------------------------- amperometry
@dataclass(frozen=True)
class AmperometricModel:
    """Single-indicator-electrode amperometric titration, ``a A + t T -> product``.

    Attributes
    ----------
    c_analyte, v_analyte_ml, c_titrant : float
        mol/L, mL, mol/L.
    ratio : float
        mol titrant per mol analyte.
    k_analyte, k_titrant, k_product : float
        Sensitivity (uA per mmol/L) of each species at the working electrode
        (0 = electro-inactive).  Pb2+ + CrO4 2-: ``k_analyte > 0, k_titrant > 0``
        gives the V-shaped curve; ``k_analyte > 0`` only gives the L-shaped one.
    residual_ua : float
        Residual current.

    >>> m = AmperometricModel(0.01, 50.0, 0.1, 1.0, k_analyte=10.0, k_titrant=0.0, k_product=0.0)
    >>> round(amperometric_current(m, 0.0), 3)
    100.0
    """

    c_analyte: float
    v_analyte_ml: float
    c_titrant: float
    ratio: float = 1.0
    k_analyte: float = 0.0
    k_titrant: float = 0.0
    k_product: float = 0.0
    residual_ua: float = 0.0

    def __post_init__(self) -> None:
        if min(self.c_analyte, self.v_analyte_ml, self.c_titrant, self.ratio) <= 0.0:
            raise ValueError("amounts and ratio must be positive")
        if min(self.k_analyte, self.k_titrant, self.k_product) < 0.0:
            raise ValueError("sensitivities must be >= 0")


def amperometric_equivalence_volume(m: AmperometricModel) -> float:
    """``V_eq = ratio C_a V_a / C_t`` (mL).

    >>> amperometric_equivalence_volume(AmperometricModel(0.01, 50.0, 0.1, 1.0, k_analyte=1.0))
    5.0
    """
    return m.ratio * m.c_analyte * m.v_analyte_ml / m.c_titrant


def amperometric_current(m: AmperometricModel, v_ml: float, dilution_corrected: bool = False) -> float:
    """Limiting current (uA) after ``v_ml`` mL titrant.

    With ``dilution_corrected`` the current is multiplied by ``(V0+V)/V0`` (the
    usual correction that makes the branches straight lines).

    Raises
    ------
    ValueError
        For a negative volume.
    """
    if v_ml < 0.0:
        raise ValueError("volume must be non-negative")
    n_a = m.c_analyte * m.v_analyte_ml
    n_t = m.c_titrant * v_ml
    extent = min(n_a, n_t / m.ratio)
    vt = m.v_analyte_ml + v_ml
    # n/V is in mol/L; sensitivities are per mmol/L
    i = 1000.0 * (m.k_analyte * (n_a - extent) + m.k_titrant * max(0.0, n_t - m.ratio * n_a) + m.k_product * extent) / vt
    i += m.residual_ua
    return i * vt / m.v_analyte_ml if dilution_corrected else i


def amperometric_curve(
    m: AmperometricModel, v_max_ml: float | None = None, n: int = 60, dilution_corrected: bool = False
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``(V, i)`` arrays from 0 to ``v_max_ml`` (default 1.5 V_eq).

    >>> m = AmperometricModel(0.01, 50.0, 0.1, 1.0, k_analyte=10.0, k_titrant=4.0)
    >>> v, i = amperometric_curve(m, n=30, dilution_corrected=True)
    >>> abs(two_line_endpoint(v, i, 5.0) - 5.0) < 1e-6
    True
    """
    top = v_max_ml if v_max_ml is not None else 1.5 * amperometric_equivalence_volume(m)
    v = np.linspace(0.0, top, n)
    return v, np.array([amperometric_current(m, float(x), dilution_corrected) for x in v])


def biamperometric_current(n_ox_mmol: float, n_red_mmol: float, v_total_ml: float, k: float = 1.0) -> float:
    """Dead-stop (two identical electrodes) current: limited by the scarcer form of a reversible couple.

    ``i = k * min([Ox], [Red])`` (uA, ``k`` in uA per mmol/L).

    >>> biamperometric_current(0.0, 1.0, 50.0)
    0.0
    >>> round(biamperometric_current(0.2, 1.0, 50.0, 100.0), 3)
    0.4
    """
    if v_total_ml <= 0.0 or min(n_ox_mmol, n_red_mmol) < 0.0:
        raise ValueError("amounts must be >= 0 and volume > 0")
    return k * min(n_ox_mmol, n_red_mmol) / v_total_ml


def deadstop_iodine_curve(
    c_thiosulfate: float, v_ml: float, c_iodine: float, volumes_ml: Sequence[float], k: float = 100.0
) -> np.ndarray[Any, Any]:
    """Thiosulfate titrated with I2 (``2 S2O3 2- + I2 -> S4O6 2- + 2 I-``), dead-stop detection.

    Before the equivalence point no I2 is left (current ~ 0); afterwards the
    couple I2/I- carries current.

    >>> y = deadstop_iodine_curve(0.1, 20.0, 0.05, [0.0, 10.0, 19.0, 30.0])
    >>> [round(float(x), 3) for x in y]
    [0.0, 0.0, 0.0, 1.0]
    """
    n_s = c_thiosulfate * v_ml
    out: list[float] = []
    for v in volumes_ml:
        n_i2 = c_iodine * v
        left = max(0.0, n_i2 - n_s / 2.0)
        iodide = min(n_i2, n_s / 2.0) * 2.0 + 0.0
        out.append(biamperometric_current(left, iodide, v_ml + v, k) if left > 0.0 else 0.0)
    return np.array(out)


# --------------------------------------------------------------- thermometry
@dataclass(frozen=True)
class ThermometricModel:
    """Adiabatic thermometric titration.

    Attributes
    ----------
    delta_h_kj_mol : float
        Reaction enthalpy (negative = exothermic), per mole of analyte.
    heat_capacity_j_ml_k : float
        Volumetric heat capacity of the solution (water 4.18).
    titrant_dt_k : float
        Titrant temperature minus cell temperature (mixing term).

    >>> m = ThermometricModel(0.1, 50.0, 0.1, -55.8)
    >>> round(thermometric_dt(m, 50.0), 3)       # 5 mmol * 55.8 J/mmol / (4.18 * 100)
    0.667
    """

    c_analyte: float
    v_analyte_ml: float
    c_titrant: float
    delta_h_kj_mol: float
    ratio: float = 1.0
    heat_capacity_j_ml_k: float = 4.18
    titrant_dt_k: float = 0.0

    def __post_init__(self) -> None:
        if min(self.c_analyte, self.v_analyte_ml, self.c_titrant, self.ratio, self.heat_capacity_j_ml_k) <= 0.0:
            raise ValueError("amounts, ratio and heat capacity must be positive")


def thermometric_equivalence_volume(m: ThermometricModel) -> float:
    """``V_eq = ratio C_a V_a / C_t`` (mL).

    >>> thermometric_equivalence_volume(ThermometricModel(0.1, 50.0, 0.1, -55.8))
    50.0
    """
    return m.ratio * m.c_analyte * m.v_analyte_ml / m.c_titrant


def thermometric_dt(m: ThermometricModel, v_ml: float, dilution_corrected: bool = False) -> float:
    """Temperature change (K) after ``v_ml`` mL titrant.

    ``dT = (-dH xi / Cp + dT_titrant V) / (V0 + V)`` with ``xi`` in mmol and
    ``dH`` in kJ/mol (so ``-dH xi`` is in J).  With ``dilution_corrected`` the
    result is multiplied by ``(V0 + V)/V0`` so that both branches are straight lines.

    Raises
    ------
    ValueError
        For a negative volume.
    """
    if v_ml < 0.0:
        raise ValueError("volume must be non-negative")
    extent = min(m.c_analyte * m.v_analyte_ml, m.c_titrant * v_ml / m.ratio)
    q = -m.delta_h_kj_mol * extent
    d_t = (q / m.heat_capacity_j_ml_k + m.titrant_dt_k * v_ml) / (m.v_analyte_ml + v_ml)
    return d_t * (m.v_analyte_ml + v_ml) / m.v_analyte_ml if dilution_corrected else d_t


def thermometric_curve(
    m: ThermometricModel, v_max_ml: float | None = None, n: int = 60, dilution_corrected: bool = False
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``(V, dT)``; the break is at the equivalence volume.

    >>> v, t = thermometric_curve(ThermometricModel(0.1, 50.0, 0.1, -55.8, titrant_dt_k=-2.0), n=40)
    >>> bool(t.max() > t[-1])
    True
    """
    top = v_max_ml if v_max_ml is not None else 1.6 * thermometric_equivalence_volume(m)
    v = np.linspace(0.0, top, n)
    return v, np.array([thermometric_dt(m, float(x), dilution_corrected) for x in v])


# --------------------------------------------------------------- Karl Fischer
KF_STANDARDS: dict[str, float] = {"water": 1.0, "sodium_tartrate_dihydrate": 0.1566}
"""Mass fraction of water in the standard (Na2C4H4O6.2H2O: 36.03 / 230.08 = 15.66 %)."""


def kf_titer_mg_per_ml(m_standard_mg: float, v_ml: float, standard: str = "water", v_blank_ml: float = 0.0) -> float:
    """Volumetric KF titer ``F`` (mg water per mL reagent).

    >>> round(kf_titer_mg_per_ml(100.0, 4.0, "sodium_tartrate_dihydrate"), 3)
    3.915
    """
    if standard not in KF_STANDARDS:
        raise ValueError(f"unknown standard {standard!r}")
    if v_ml - v_blank_ml <= 0.0 or m_standard_mg <= 0.0:
        raise ValueError("mass and net volume must be positive")
    return m_standard_mg * KF_STANDARDS[standard] / (v_ml - v_blank_ml)


def kf_water_percent(
    v_ml: float, titer_mg_ml: float, m_sample_mg: float, drift_ug_per_min: float = 0.0, time_min: float = 0.0,
    v_blank_ml: float = 0.0,
) -> float:
    """Water content (% w/w) with drift and blank correction.

    ``mg water = F (V - V_blank) - drift * t / 1000``.

    >>> round(kf_water_percent(3.0, 5.0, 1000.0), 3)
    1.5
    >>> round(kf_water_percent(3.0, 5.0, 1000.0, drift_ug_per_min=10.0, time_min=3.0), 3)
    1.497
    """
    if m_sample_mg <= 0.0:
        raise ValueError("sample mass must be positive")
    water = titer_mg_ml * (v_ml - v_blank_ml) - drift_ug_per_min * time_min / 1000.0
    return water / m_sample_mg * 100.0


def coulometric_kf_water_ug(charge_c: float) -> float:
    """Water (ug) from the charge: 1 mol H2O per 2 F (``10.71 C per mg``).

    >>> round(coulometric_kf_water_ug(10.711))
    1000
    """
    if charge_c < 0.0:
        raise ValueError("charge must be >= 0")
    return charge_c / (2.0 * FARADAY) * M_WATER * 1e6


# ----------------------------------------------------------------- coulometry
def coulometric_amount_mmol(i_ma: float, t_s: float, n_electrons: float, efficiency: float = 1.0) -> float:
    """Amount (mmol) generated/titrated: ``n = eta i t / (n_e F)`` (Faraday).

    >>> round(coulometric_amount_mmol(10.0, 965.0, 1.0), 4)
    0.1
    """
    if min(i_ma, t_s) < 0.0 or n_electrons <= 0.0 or not 0.0 < efficiency <= 1.0:
        raise ValueError("i, t >= 0, n_e > 0 and 0 < efficiency <= 1 required")
    return efficiency * i_ma * t_s / (n_electrons * FARADAY)


def coulometric_time_s(n_mmol: float, i_ma: float, n_electrons: float, efficiency: float = 1.0) -> float:
    """Generation time (s) for ``n_mmol`` at constant current.

    >>> round(coulometric_time_s(0.1, 10.0, 1.0), 1)
    964.9
    """
    if n_mmol < 0.0 or i_ma <= 0.0 or n_electrons <= 0.0 or not 0.0 < efficiency <= 1.0:
        raise ValueError("n >= 0, i > 0, n_e > 0, 0 < efficiency <= 1 required")
    return n_mmol * n_electrons * FARADAY / (efficiency * i_ma)


def coulometric_efficiency_bias_percent(efficiency: float) -> float:
    """Relative error (%) of the result when the true current efficiency is ``efficiency`` but 1 is assumed.

    >>> round(coulometric_efficiency_bias_percent(0.995), 2)
    0.5
    """
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("0 < efficiency <= 1 required")
    return (1.0 - efficiency) * 100.0


# ------------------------------------------------- titrant standardization
PRIMARY_STANDARDS: dict[str, tuple[float, float, str]] = {
    # key: (molar mass g/mol, mol primary standard per mol titrant, description)
    "khp_naoh": (204.22, 1.0, "KHP -> NaOH"),
    "na2co3_hcl": (105.99, 0.5, "Na2CO3 -> HCl (to H2CO3)"),
    "na2c2o4_kmno4": (134.00, 2.5, "Na2C2O4 -> KMnO4 (5:2)"),
    "kio3_thiosulfate": (214.00, 1.0 / 6.0, "KIO3 -> Na2S2O3 (1:6, via iodide)"),
    "caco3_edta": (100.09, 1.0, "CaCO3 -> EDTA"),
    "nacl_agno3": (58.44, 1.0, "NaCl -> AgNO3"),
}
"""``key -> (M, mol standard per mol titrant, label)``; stoichiometry is exact, M is textbook."""


def standardize_titrant(key: str, mass_g: float, v_titrant_ml: float, purity: float = 1.0) -> float:
    """Titrant concentration (mol/L) from a weighed primary standard.

    ``C = m purity / (M v) * (mol titrant per mol standard)``.

    >>> round(standardize_titrant("khp_naoh", 0.5106, 25.00), 4)
    0.1
    >>> round(standardize_titrant("na2co3_hcl", 0.1325, 25.0), 4)
    0.1
    """
    if key not in PRIMARY_STANDARDS:
        raise ValueError(f"unknown standard {key!r}; known: {sorted(PRIMARY_STANDARDS)}")
    if min(mass_g, v_titrant_ml) <= 0.0 or not 0.0 < purity <= 1.0:
        raise ValueError("mass and volume must be positive and 0 < purity <= 1")
    mw, per_titrant, _ = PRIMARY_STANDARDS[key]
    n_std_mmol = mass_g * purity / mw * 1000.0
    return n_std_mmol / per_titrant / v_titrant_ml


# ------------------------------------------------------------ GUM uncertainty
@dataclass(frozen=True)
class Quantity:
    """Input quantity: value, standard uncertainty ``u`` and degrees of freedom."""

    name: str
    value: float
    u: float
    dof: float = math.inf

    def __post_init__(self) -> None:
        if self.u < 0.0 or self.dof <= 0.0:
            raise ValueError("u must be >= 0 and dof > 0")


def u_rectangular(a: float) -> float:
    """Type B, rectangular half-width ``a``: ``a/sqrt(3)``.

    >>> round(u_rectangular(0.05), 4)
    0.0289
    """
    return a / math.sqrt(3.0)


def u_triangular(a: float) -> float:
    """Type B, triangular half-width ``a``: ``a/sqrt(6)``.

    >>> round(u_triangular(0.1), 4)
    0.0408
    """
    return a / math.sqrt(6.0)


def u_expanded_to_standard(big_u: float, k: float = 2.0) -> float:
    """Certificate value ``U`` with coverage factor ``k`` to a standard uncertainty.

    >>> u_expanded_to_standard(0.02, 2.0)
    0.01
    """
    return big_u / k


_T_95_45 = ((1, 13.97), (2, 4.53), (3, 3.31), (4, 2.87), (5, 2.65), (6, 2.52), (7, 2.43), (8, 2.37), (9, 2.32),
            (10, 2.28), (12, 2.23), (14, 2.20), (16, 2.17), (18, 2.15), (20, 2.13), (25, 2.11), (30, 2.09),
            (40, 2.06), (50, 2.05), (100, 2.025))


def coverage_factor(dof: float) -> float:
    """Coverage factor ``k`` for 95.45 % from the effective degrees of freedom.

    The tabulated value for the nearest *lower* tabulated dof is used
    (conservative).  Table typed from GUM Annex G.2 - check it before
    accrediting a method with it.

    >>> coverage_factor(math.inf)
    2.0
    >>> coverage_factor(4.9)
    2.87
    """
    if dof == math.inf or dof > 1000.0:
        return 2.0
    k = _T_95_45[0][1]
    for d, t in _T_95_45:
        if dof >= d:
            k = t
    return k


@dataclass(frozen=True)
class GumResult:
    """Output of :func:`gum_budget`.

    Attributes
    ----------
    value, u_c, u_expanded, k, dof_eff : float
    contributions : tuple[tuple[str, float, float, float], ...]
        ``(name, sensitivity c_i, contribution |c_i| u_i, share in % of u_c^2)``.
    """

    value: float
    u_c: float
    u_expanded: float
    k: float
    dof_eff: float
    contributions: tuple[tuple[str, float, float, float], ...]


def gum_budget(f: Callable[..., float], inputs: Sequence[Quantity], k: float | None = None) -> GumResult:
    """First-order GUM budget of ``y = f(x1, ..., xn)`` (uncorrelated inputs).

    ``c_i`` is obtained by a central difference; ``u_c^2 = sum (c_i u_i)^2``;
    ``dof_eff`` by Welch-Satterthwaite; ``k`` defaults to :func:`coverage_factor`.

    >>> q = [Quantity("c_t", 0.1, 0.0002), Quantity("v_t", 25.0, 0.03), Quantity("v_a", 25.0, 0.03)]
    >>> r = gum_budget(lambda c, vt, va: c * vt / va, q)
    >>> round(r.value, 4), round(r.u_c, 6)
    (0.1, 0.000262)
    >>> abs(sum(x[3] for x in r.contributions) - 100.0) < 1e-9
    True
    """
    if not inputs:
        raise ValueError("at least one input quantity is required")
    x = [q.value for q in inputs]
    y0 = f(*x)
    sens: list[float] = []
    for j, q in enumerate(inputs):
        h = max(abs(q.value), q.u, 1e-12) * 1e-6
        xp, xm = list(x), list(x)
        xp[j] += h
        xm[j] -= h
        sens.append((f(*xp) - f(*xm)) / (2.0 * h))
    parts = [(c * q.u) ** 2 for c, q in zip(sens, inputs)]
    var = sum(parts)
    u_c = math.sqrt(var)
    denom = sum(p * p / q.dof for p, q in zip(parts, inputs) if not math.isinf(q.dof))
    dof_eff = math.inf if denom == 0.0 else var * var / denom
    kk = coverage_factor(dof_eff) if k is None else k
    shares = [(100.0 * p / var if var > 0.0 else 0.0) for p in parts]
    contrib = tuple((q.name, c, abs(c) * q.u, s) for q, c, s in zip(inputs, sens, shares))
    return GumResult(y0, u_c, kk * u_c, kk, dof_eff, contrib)


def titration_gum(
    c_titrant: float, u_c_titrant: float, v_titrant_ml: float, u_v_titrant_ml: float,
    v_analyte_ml: float, u_v_analyte_ml: float, ratio: float = 1.0, u_endpoint_ml: float = 0.0,
) -> GumResult:
    """GUM budget of ``C_A = ratio C_T (V_T) / V_A`` with an extra end-point term.

    ``u_endpoint_ml`` (e.g. indicator end point vs equivalence) is added to the titrant volume as an
    independent input with unit sensitivity.

    >>> r = titration_gum(0.1, 0.0002, 25.0, 0.03, 25.0, 0.03, u_endpoint_ml=0.02)
    >>> r.u_c > titration_gum(0.1, 0.0002, 25.0, 0.03, 25.0, 0.03).u_c
    True
    """
    q = [Quantity("C_T", c_titrant, u_c_titrant), Quantity("V_T", v_titrant_ml, u_v_titrant_ml),
         Quantity("V_A", v_analyte_ml, u_v_analyte_ml), Quantity("end point", 0.0, u_endpoint_ml)]
    return gum_budget(lambda c, vt, va, ep: ratio * c * (vt + ep) / va, q)


# --------------------------------------------------------------------- report
def markdown_report(title: str, sections: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> str:
    """Plain Markdown report: ``sections = [(heading, [(label, value), ...]), ...]``.

    >>> print(markdown_report("T", [("A", [("x", "1")])]))
    # T
    <BLANKLINE>
    ## A
    <BLANKLINE>
    | quantity | value |
    |---|---|
    | x | 1 |
    <BLANKLINE>
    """
    out = [f"# {title}", ""]
    for heading, rows in sections:
        out += [f"## {heading}", "", "| quantity | value |", "|---|---|"]
        out += [f"| {a} | {b} |" for a, b in rows]
        out.append("")
    return "\n".join(out)
