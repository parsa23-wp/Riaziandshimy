"""Analytical quantities beyond the curve: titer, assay, factor, LOD/LOQ,
concordance, buffer capacity, half-equivalence pKa, non-aqueous workflow,
carbonate by double-endpoint Gran, and the Berto 2024 data-sufficiency check.

Everything here is pure calculation (no Qt, no UI): the same functions feed
the web calculators, the desktop presenter and the doctests.

مقادیر کمی آنالیزی: تیتر، عیار (Assay)، ضریب F، LOD/LOQ، نتایج هم‌سو،
ظرفیت بافر، pKa از نیم‌هم‌ارزی، جریان کار غیرآبی، کربنات با دو نقطهٔ پایان،
و آزمون کفایت دادهٔ Berto 2024.

[MATH.md section 18]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "titer_mg_per_ml",
    "assay_percent",
    "hclo4_from_khp",
    "nonaqueous_assay_percent",
    "resolvable_in_solvent",
    "factor_f",
    "ConcordanceResult",
    "concordant_results",
    "LodLoq",
    "lod_loq",
    "buffer_capacity",
    "pka_from_half_equivalence",
    "carbonate_from_two_endpoints",
    "CoverageReport",
    "species_coverage_check",
]

# potassium hydrogen phthalate, the classic primary standard for HClO4
KHP_MW = 204.22  # g/mol


def titer_mg_per_ml(c_titrant: float, mw_analyte: float, ratio: float = 1.0) -> float:
    """Titer: mg of analyte equivalent to 1 mL of titrant (section 83).

    ``ratio`` is ``a/b`` for ``aA + bB`` (moles of analyte per mole of titrant).

    >>> round(titer_mg_per_ml(0.1, 36.46), 4)
    3.646
    >>> round(titer_mg_per_ml(0.1, 100.09, ratio=0.5), 4)  # CaCO3 vs HCl
    5.0045
    """
    if c_titrant <= 0 or mw_analyte <= 0:
        raise ValueError("c_titrant and mw_analyte must be positive")
    return float(c_titrant * ratio * mw_analyte)  # mol/L * g/mol = mg/mL


def assay_percent(
    m_sample_g: float,
    c_titrant: float,
    v_titrant_l: float,
    mw_analyte: float,
    ratio: float = 1.0,
    v_flask_l: float = 1.0,
    v_aliquot_l: float = 1.0,
) -> float:
    """Assay: percent of the active compound in the sample (section 82).

    If the sample was dissolved in ``v_flask_l`` and only ``v_aliquot_l``
    of that flask was titrated, the result is scaled by ``v_flask/v_aliquot``.

    >>> round(assay_percent(0.15, 0.1, 0.025, 100.09, ratio=0.5), 2)
    83.41
    """
    if m_sample_g <= 0:
        raise ValueError("m_sample_g must be positive")
    if v_aliquot_l <= 0 or v_flask_l < v_aliquot_l:
        raise ValueError("need 0 < v_aliquot_l <= v_flask_l")
    n_analyte_mol = c_titrant * v_titrant_l * ratio * (v_flask_l / v_aliquot_l)
    return float(n_analyte_mol * mw_analyte / m_sample_g * 100.0)


def hclo4_from_khp(m_khp_g: float, v_hclo4_l: float, purity_khp: float = 1.0) -> float:
    """Standardize HClO4 in glacial acetic acid with KHP (sections 60-62).

    KHP is monoprotic, so ``C = m * purity / (MW * V)``.

    >>> round(hclo4_from_khp(0.5106, 0.025), 4)
    0.1
    """
    if v_hclo4_l <= 0 or purity_khp <= 0:
        raise ValueError("v_hclo4_l and purity_khp must be positive")
    return float(m_khp_g * purity_khp / KHP_MW / v_hclo4_l)


def nonaqueous_assay_percent(
    m_sample_g: float,
    v_titrant_l: float,
    c_titrant: float,
    mw_analyte: float,
    ratio: float = 1.0,
) -> float:
    """Assay by direct non-aqueous titration (section 53).

    Same stoichiometry as :func:`assay_percent` without the flask split;
    the titrant must be standardized in the SAME solvent (leveling!).
    Crystal violet (weak bases in AcOH) / thymol blue (acids) are the
    usual visual indicators.

    >>> round(nonaqueous_assay_percent(0.20, 0.020, 0.1, 122.12, ratio=0.5), 2)
    61.06
    """
    return assay_percent(m_sample_g, c_titrant, v_titrant_l, mw_analyte, ratio)


def resolvable_in_solvent(
    log_k_values: Sequence[float], min_log_gap: float = 4.0
) -> tuple[bool, float]:
    """Classic non-aqueous rule: a mixture of bases (or acids) can be titrated
    stepwise only if consecutive strengths differ by >= ``min_log_gap`` units
    (the usual textbook figure is 4 in log K, i.e. 10^4) — section 53.

    ``log_k_values`` must be sorted from strongest to weakest.

    >>> resolvable_in_solvent([10.0, 6.0, 2.0])
    (True, 4.0)
    >>> resolvable_in_solvent([10.0, 8.0])
    (False, 2.0)
    """
    if len(log_k_values) < 2:
        raise ValueError("need at least two constants")
    gaps = [a - b for a, b in zip(log_k_values, log_k_values[1:])]
    return (min(gaps) >= min_log_gap, float(min(gaps)))


def factor_f(actual: float, nominal: float) -> float:
    """Experimental titer factor ``F = C_actual / C_nominal`` (section 84).

    >>> round(factor_f(0.1023, 0.1), 4)
    1.023
    """
    if nominal == 0:
        raise ValueError("nominal must be non-zero")
    return float(actual / nominal)


@dataclass(frozen=True)
class ConcordanceResult:
    """Outcome of the concordance check (section 72)."""

    ok: bool
    spread: float  # max - min
    mean: float
    n: int
    threshold: float


def concordant_results(
    values: Sequence[float], tol_ml: float = 0.05, tol_rel: float = 0.002
) -> ConcordanceResult:
    """Are replicate titres concordant? Lab rule: spread <= max(tol_ml,
    tol_rel * |mean|) — section 72.

    >>> c = concordant_results([24.82, 24.79, 24.84])
    >>> (c.ok, round(c.spread, 2), round(c.threshold, 3))
    (True, 0.05, 0.05)
    >>> concordant_results([24.82, 25.30]).ok
    False
    """
    if len(values) < 2:
        raise ValueError("need at least two values")
    spread = float(max(values) - min(values))
    m = float(sum(values) / len(values))
    threshold = max(tol_ml, tol_rel * abs(m))
    # 1e-9 guard: binary floats make 24.84-24.79 == 0.05000000000000426
    return ConcordanceResult(spread <= threshold + 1e-9, spread, m, len(values), threshold)


@dataclass(frozen=True)
class LodLoq:
    """Detection / quantification limits (section 96)."""

    lod: float
    loq: float
    style: str  # "ich" (3.3 sigma) or "classic" (3 sigma)


def lod_loq(sigma_blank: float, slope: float, style: str = "ich") -> LodLoq:
    """LOD and LOQ from the blank standard deviation and the response slope.

    ICH style: ``LOD = 3.3 sigma / S``, ``LOQ = 10 sigma / S``;
    classic style uses ``3 sigma``. There is no single universal definition
    (section 96) — the style is reported in the result.

    >>> l = lod_loq(0.002, 1.5)
    >>> (round(l.lod, 5), round(l.loq, 5), l.style)
    (0.0044, 0.01333, 'ich')
    """
    if slope <= 0 or sigma_blank < 0:
        raise ValueError("slope must be positive, sigma_blank non-negative")
    k_lod = 3.3 if style == "ich" else 3.0 if style == "classic" else None
    if k_lod is None:
        raise ValueError("style must be 'ich' or 'classic'")
    lod = float(k_lod * sigma_blank / slope)
    loq = float(10.0 * sigma_blank / slope)
    return LodLoq(lod, loq, style)


def buffer_capacity(h: float, ka: float, c_total: float, kw: float = 1e-14) -> float:
    """Analytic buffer capacity ``beta = dC_b/dpH`` (mol/L per pH unit).

    ``beta = 2.303 * (c * ka*h / (ka+h)^2 + h + kw/h)`` — the acid term is
    maximal at ``h = ka`` (i.e. pH = pKa), which is why half-equivalence
    solutions resist pH change best.

    >>> round(buffer_capacity(10**-4.76, 10**-4.76, 0.1), 4)
    0.0576
    """
    if h <= 0 or ka <= 0 or c_total < 0:
        raise ValueError("h and ka must be positive, c_total non-negative")
    acid = c_total * ka * h / (ka + h) ** 2
    return float(2.302585 * (acid + h + kw / h))


def pka_from_half_equivalence(
    v_points: Sequence[float], ph_points: Sequence[float],
    v_eq_start: float, v_eq_end: float,
) -> float:
    """pKa estimate from the pH at the half-equivalence volume of one step
    (sections 16, 20, 28). ``v_eq_start`` is the previous equivalence volume
    (0 for the first step) and ``v_eq_end`` the one of this step; the pH is
    linearly interpolated at their midpoint. Valid for the usual
    weak-acid/strong-base case; for weak/weak it is an approximation.

    >>> pka_from_half_equivalence([0, 5, 10, 15], [3.0, 4.76, 6.0, 8.0], 0, 10)
    4.76
    """
    if len(v_points) != len(ph_points) or len(v_points) < 2:
        raise ValueError("v_points and ph_points must be equal length >= 2")
    v_half = 0.5 * (v_eq_start + v_eq_end)
    if not (min(v_points) <= v_half <= max(v_points)):
        raise ValueError("half-equivalence volume outside the measured range")
    for i in range(len(v_points) - 1):
        v0, v1 = v_points[i], v_points[i + 1]
        if (v0 - v_half) * (v1 - v_half) <= 0 and v1 != v0:
            t = (v_half - v0) / (v1 - v0)
            return float(ph_points[i] + t * (ph_points[i + 1] - ph_points[i]))
    return float(ph_points[0] if v_points[0] == v_half else ph_points[-1])


def carbonate_from_two_endpoints(
    v1_ml: float, v2_ml: float, c_acid: float, v_sample_ml: float
) -> tuple[float, float]:
    """Carbonate + hydroxide contents from a double-endpoint titration
    (Berto 2024 section 2.6(ii): the carbonate of a base titrant is measured
    by a Gran/double-endpoint titration).

    First end point consumes ``OH- + 1/2 CO3^2-`` (to HCO3-), the second the
    remaining half: ``n_CO3 = (V2-V1) c`` and ``n_OH = (2 V1 - V2) c``.

    Returns ``(c_co3_mol_l, c_oh_mol_l)``.

    >>> carbonate_from_two_endpoints(20.0, 30.0, 0.1, 50.0)
    (0.02, 0.02)
    """
    if v_sample_ml <= 0 or c_acid <= 0 or v2_ml < v1_ml:
        raise ValueError("need c_acid > 0, v_sample_ml > 0 and v2_ml >= v1_ml")
    n_co3 = (v2_ml - v1_ml) * c_acid / 1000.0
    n_oh = (2.0 * v1_ml - v2_ml) * c_acid / 1000.0
    if n_oh < 0:
        raise ValueError("2*V1 < V2: the data are not OH-/CO3^2- consistent")
    v_l = v_sample_ml / 1000.0
    return (float(n_co3 / v_l), float(n_oh / v_l))


@dataclass(frozen=True)
class CoverageReport:
    """Berto 2024 guideline: per-species point coverage of the 20-80 % band.

    Attributes
    ----------
    per_species : tuple[int, ...]
        Number of titration points whose alpha fraction of that species lies
        in ``[low, high]`` (index 0 = fully deprotonated form).
    adequate : bool
        True when every species has at least ``min_points`` such points.
    """

    per_species: tuple[int, ...]
    adequate: bool


def species_coverage_check(
    ph_points: Sequence[float],
    pka_list: Sequence[float],
    min_points: int = 10,
    low: float = 0.2,
    high: float = 0.8,
) -> CoverageReport:
    """How many points describe each species inside its 20-80 % abundance
    band (Berto 2024: 'at least 10-15 titration points that cover 20-80 % of
    the maximal abundance of each particular species should be analysed to
    provide reliable equilibrium constants').

    ``pka_list`` is ordered from the first deprotonation step (pKa1) onward.

    >>> r = species_coverage_check([4.76 + 0.05 * i for i in range(-10, 11)], [4.76])
    >>> r.adequate
    True
    """
    if not pka_list or len(ph_points) < 2:
        raise ValueError("need at least one pKa and two pH points")
    # cumulative PROTONATION constants of the fully deprotonated form:
    # log KH_n = pKa_n, log KH_{n+1} = pKa_n + pKa_{n-1}, ...
    rev = list(reversed(pka_list))
    log_beta: list[float] = []
    acc = 0.0
    for p in rev:
        acc += p
        log_beta.append(acc)
    counts = [0] * (len(pka_list) + 1)
    for ph in ph_points:
        h = 10.0 ** (-float(ph))
        terms = [1.0]
        hp = 1.0
        for lb in log_beta:
            hp *= h
            terms.append(10.0**lb * hp)
        total = sum(terms)
        alphas = (t / total for t in terms)
        for j, a in enumerate(alphas):
            if low <= a <= high:
                counts[j] += 1
    return CoverageReport(tuple(counts), all(c >= min_points for c in counts))


# ---------------------------------------------------------------------------
# EDTA masking / demasking and selectivity (sections 50-51)
# ---------------------------------------------------------------------------

# Textbook (25 C) cumulative formation constants with EDTA, log Kf.
EDTA_LOG_KF: dict[str, float] = {
    "Ca2+": 10.65, "Mg2+": 8.79, "Zn2+": 16.50, "Cd2+": 16.46,
    "Cu2+": 18.80, "Ni2+": 18.62, "Pb2+": 18.04, "Mn2+": 13.87,
    "Fe2+": 14.32, "Co2+": 16.31, "Hg2+": 21.70, "Fe3+": 25.10,
}

# Cyanide masking: only the tetracyano (or dicyano for Ag) complex matters,
# so the cumulative tuples carry unit betas for the lower steps.
CN_LOG_BETA: dict[str, tuple[float, ...]] = {
    "Zn2+": (0.0, 0.0, 0.0, 19.6),
    "Cd2+": (0.0, 0.0, 0.0, 18.9),
    "Cu2+": (0.0, 0.0, 0.0, 30.5),  # Cu(I) cyanide in practice
    "Ni2+": (0.0, 0.0, 0.0, 31.3),
    "Ag+": (0.0, 20.5),
    "Hg2+": (0.0, 0.0, 0.0, 41.5),
}

# pKa of HCN: at pH below it, part of the added CN- is protonated and the
# FREE cyanide concentration must be used in beta[L]^l (documented refinement).
HCN_PKA = 9.21


def edta_masking_selectivity(
    ph: float,
    ligand_conc: float = 0.0,
    ligand_log_beta: dict[str, tuple[float, ...]] | None = None,
    metals_log_kf: dict[str, float] | None = None,
    c_metal: float = 0.01,
    min_log_kc: float = 6.0,
    ligand_pka: float | None = HCN_PKA,
    pkw: float = 14.0,
) -> dict[str, tuple[float, float, bool]]:
    """Which metals can be titrated with EDTA in the presence of a masking
    ligand (usually KCN)?  For every metal:

    ``log K'' = log Kf + log alpha_Y4- + log alpha_M``

    and the metal is titratable when ``log(K'' * c_metal) >= min_log_kc``
    (the classical sharp-end-point rule ``K''C >= 10^6``).  A metal whose
    neighbours are titratable but that fails the test itself is *masked*;
    releasing it again is *demasking* (e.g. formaldehyde for Zn-CN).

    ``ligand_pka``: if set, the free-ligand fraction is corrected for the
    protonation of the ligand (CN- -> HCN), ``[L] = c * Ka/(Ka + [H+])``.

    Returns ``{metal: (log K'', log(K'' C), titratable)}``.

    Example: at pH 10 with 0.01 M CN-, Ca and Mg are titratable while
    Zn/Cd/Cu/Ni are masked:

    >>> r = edta_masking_selectivity(10.0, 0.01)
    >>> r["Mg2+"][2] and r["Ca2+"][2]
    True
    >>> (r["Zn2+"][2], r["Cd2+"][2], r["Cu2+"][2])
    (False, False, False)
    """
    from .phase2 import EDTA_PKA, alpha_metal, alpha_y4

    lb = ligand_log_beta if ligand_log_beta is not None else CN_LOG_BETA
    kf = metals_log_kf if metals_log_kf is not None else EDTA_LOG_KF
    l_free = ligand_conc
    if ligand_pka is not None and ligand_conc > 0:
        ka = 10.0 ** (-ligand_pka)
        h = 10.0 ** (-ph)
        l_free = ligand_conc * ka / (ka + h)
    log_ay = math.log10(alpha_y4(ph, EDTA_PKA))
    out: dict[str, tuple[float, float, bool]] = {}
    for metal, log_kf in kf.items():
        betas = lb.get(metal, ())
        a_m = alpha_metal(l_free, betas, ph=ph, pkw=pkw) if l_free > 0 else 1.0
        log_k2 = log_kf + log_ay + math.log10(a_m)
        log_kc = log_k2 + math.log10(c_metal)
        out[metal] = (log_k2, log_kc, log_kc >= min_log_kc)
    return out
