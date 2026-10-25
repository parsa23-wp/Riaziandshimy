"""Instrumental titration layers (phase 3): potentiometry, conductometry, photometry.

These are *measurement* layers on top of the chemistry engines (architecture
section 0: chemistry and instrument are separate).  Every function is a pure
mapping from chemistry (volume → concentrations/potential) to the signal an
instrument would report:

* :func:`emf_mv` / :func:`potentiometric_curve_redox` — indicator electrode
  versus a reference electrode (SCE, Ag/AgCl, ...), for redox titrations.
  The acid-base glass-electrode model (E0, slope, junction terms jA/jB)
  already lives in the web app and in :mod:`.potentiometry`.
* :func:`conductivity_us_cm` / :func:`conductometric_curve_acid_base` —
  conductometric titration with limiting ionic conductivities at 25 C and a
  linear temperature correction.  Documented APPROXIMATION: no Onsager
  interionic correction, so κ is a few percent high at 0.1 M; protonated
  forms of a polyprotic ion share the fully-deprotonated λ°.
* :func:`absorbance` / :func:`permanganate_curve` — photometric titration,
  Beer-Lambert ``A = ε · c · l``; the built-in convenience covers the classic
  permanganate end point (purple MnO4- excess after the equivalence point).

Reference: [MATH.md section 17]
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Mapping, Sequence

import numpy as np

from .acid_base import AcidBaseEngine, AcidBaseModel, ph_at, pkw_water
from .base import TitrationModel
from .phase2 import RedoxModel, redox_potential, redox_species

# ------------------------------------------------------------ potentiometry
REF_ELECTRODES = {"SCE": 0.244, "AgAgCl_satKCl": 0.199, "AgAgCl_3MKCl": 0.210, "NHE": 0.0}
"""Reference-electrode potentials vs NHE at 25 C (V). Textbook values."""


def emf_mv(e_indicator_v: float, reference: str = "SCE", junction_mv: float = 0.0) -> float:
    """Cell EMF in millivolts: ``EMF = E_indicator - E_ref + E_junction``.

    Raises
    ------
    ValueError
        Unknown reference electrode.

    Example
    -------
    >>> round(emf_mv(1.105, "SCE"), 1)
    861.0
    >>> round(emf_mv(0.443, "AgAgCl_satKCl"), 1)
    244.0
    """
    if reference not in REF_ELECTRODES:
        raise ValueError(f"unknown reference electrode {reference!r}")
    return (e_indicator_v - REF_ELECTRODES[reference]) * 1000.0 + junction_mv


def _gauss_stream(seed: int) -> Iterator[float]:
    """Deterministic standard-normal stream (mulberry32 + Box-Muller)."""
    state = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 4294967296.0

    while True:
        u = rnd()
        while u == 0.0:
            u = rnd()
        v = rnd()
        yield math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def potentiometric_curve_redox(m: RedoxModel, v_max_ml: float | None = None, n: int = 200,
                               reference: str = "SCE", junction_mv: float = 0.0,
                               noise_mv: float = 0.0, seed: int = 1) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``(V, EMF_mV)`` of a potentiometric redox titration.

    ``noise_mv > 0`` superimposes a deterministic Gaussian meter noise so
    students can practise derivative end-point seeking on realistic data.

    Raises
    ------
    ValueError
        Unknown reference electrode.

    Example
    -------
    >>> m = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)
    >>> v, e = potentiometric_curve_redox(m, n=50)
    >>> round(e[0]), round(e[-1])
    (444, 1196)
    >>> v2, e2 = potentiometric_curve_redox(m, n=50, noise_mv=0.5, seed=3)
    >>> bool(abs(e2[25] - e[25]) < 2.0)
    True
    """
    if reference not in REF_ELECTRODES:
        raise ValueError(f"unknown reference electrode {reference!r}")
    top = v_max_ml if v_max_ml is not None else 2.0 * redox_equivalence_volume_of(m)
    v = np.linspace(top / n, top, n)
    emf = np.array([emf_mv(redox_potential(m, float(x)), reference, junction_mv) for x in v])
    if noise_mv > 0.0:
        g = _gauss_stream(seed)
        emf = emf + np.array([noise_mv * next(g) for _ in range(n)])
    return v, emf


def redox_equivalence_volume_of(m: RedoxModel) -> float:
    """Re-exported :func:`phase2.redox_equivalence_volume` (import shim)."""
    from .phase2 import redox_equivalence_volume
    return redox_equivalence_volume(m)


# ----------------------------------------------------------- conductometry
LAMBDA_25 = {
    "H+": 349.65, "OH-": 198.0, "Na+": 50.10, "K+": 73.50, "Li+": 38.69,
    "NH4+": 73.5, "Ag+": 61.90, "Ca2+": 119.0, "Mg2+": 106.1, "Ba2+": 127.2,
    "Cl-": 76.35, "NO3-": 71.46, "CH3COO-": 40.9, "HCO3-": 44.5,
    "SO42-": 160.0, "CO32-": 138.6, "MnO4-": 61.3,
}
"""Limiting ionic conductivities λ° at 25 C (S·cm²/mol); textbook values."""


def conductivity_us_cm(conc: Mapping[str, float], temperature_c: float = 25.0,
                       custom_lambda: Mapping[str, float] | None = None) -> float:
    """Conductivity κ in µS/cm from ``{ion: mol/L}`` (limiting-law model).

    ``κ = 1000 · Σ λ°_i c_i`` (µS/cm when c is in mol/L and λ° in
    S·cm²/mol), then the linear temperature correction
    ``κ(T) = κ(25)·(1 + α (T - 25))`` with the usual ``α ≈ 0.019 /K`` for
    dilute aqueous solutions (valid ~15 ... 35 C).

    Raises
    ------
    ValueError
        On an unknown ion (pass ``custom_lambda``), negative concentration
        or out-of-range temperature.

    Example
    -------
    >>> round(conductivity_us_cm({"H+": 0.1, "Cl-": 0.1}))
    42600
    >>> round(conductivity_us_cm({"Na+": 0.1, "Cl-": 0.1}))
    12645
    >>> round(conductivity_us_cm({"H+": 0.1, "Cl-": 0.1}, 35.0))
    50694
    """
    lam = dict(LAMBDA_25)
    if custom_lambda:
        lam.update(custom_lambda)
    total = 0.0
    for ion, c in conc.items():
        if c < 0.0:
            raise ValueError(f"negative concentration for {ion!r}")
        if ion not in lam:
            raise ValueError(f"unknown ion {ion!r} (pass custom_lambda)")
        total += lam[ion] * c
    if not 0.0 <= temperature_c <= 60.0:
        raise ValueError("temperature outside 0 ... 60 C")
    return total * 1000.0 * (1.0 + 0.019 * (temperature_c - 25.0))


def conductometric_species_acid_base(model: AcidBaseModel, v_ml: float,
                                     analyte_anion_lambda: float = 40.9,
                                     analyte_cation_lambda: float = 73.5,
                                     strong_titrant_cation_lambda: float = 50.10,
                                     strong_titrant_anion_lambda: float = 76.35,
                                     weak_acid_titrant_anion_lambda: float = 40.9,
                                     weak_base_titrant_cation_lambda: float = 73.5,
                                     ) -> tuple[dict[str, float], dict[str, float]]:
    """Ions and their λ° of an acid-base vessel at ``v_ml``.

    Returns
    -------
    conc : dict[str, float]
        ``{ion: mol/L}`` — H+/OH- from the exact engine pH, the charged
        analyte forms (charge ``i - n`` for acid ``H_iA``, ``+i`` for the
        protonated base) and the spectator ions.
    lambdas : dict[str, float]
        λ° for every ion in ``conc``.

    Raises
    ------
    ValueError
        If ``v_ml`` is negative.

    Notes
    -----
    All charged forms of a polyprotic analyte share one λ° (documented
    APPROXIMATION); spectators are assumed Na+/Cl-/acetate-like/NH4+-like
    according to the titrant chemistry (parameters).

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1)
    >>> conc, lam = conductometric_species_acid_base(m, 0.0)
    >>> round(conc["H+"], 2), round(conc["Cl-"], 2), lam["Cl-"]
    (0.1, 0.1, 76.35)
    >>> m2 = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76])
    >>> conc2, lam2 = conductometric_species_acid_base(m2, 20.0)
    >>> "analyte(A)" in conc2 and conc2["Na+"] > 0
    True
    """
    if v_ml < 0.0:
        raise ValueError("volume must be non-negative")
    state = AcidBaseEngine()._state(model, v_ml)  # within-package use of the pure helper
    sp = state.species
    vt = model.v_analyte_ml + v_ml
    conc = {"H+": sp["H+"], "OH-": sp["OH-"]}
    lam = {"H+": LAMBDA_25["H+"], "OH-": LAMBDA_25["OH-"]}
    n_steps = len(model.analyte_pka)
    kind_v = getattr(model.analyte_kind, "value", model.analyte_kind)
    for name, c in sp.items():
        if name in ("H+", "OH-", "T(deprotonated)", "T(protonated)"):
            continue
        i = 0 if name == "A" else int(name[1])       # H_iA, i protons bound
        charge = (i - n_steps) if model.analyte_is_acid else i
        if c <= 0.0 or charge == 0:
            continue
        key = f"analyte({name})"
        conc[key] = c
        lam[key] = analyte_anion_lambda if charge < 0 else analyte_cation_lambda
    if kind_v == "strong_acid":
        conc["Cl-"] = model.c_analyte * model.v_analyte_ml / vt
        lam["Cl-"] = strong_titrant_anion_lambda
    elif kind_v == "strong_base":
        conc["Na+"] = model.c_analyte * model.v_analyte_ml / vt
        lam["Na+"] = strong_titrant_cation_lambda
    ct = model.c_titrant * v_ml / vt
    weak = getattr(model.titrant_strength, "value", model.titrant_strength) == "weak"
    if model.analyte_is_acid:
        if not weak:                                  # NaOH titrant
            conc["Na+"] = ct
            lam["Na+"] = strong_titrant_cation_lambda
        elif "T(deprotonated)" in sp:                 # weak acid titrant HX → X-
            conc["X-"] = sp["T(deprotonated)"]
            lam["X-"] = weak_acid_titrant_anion_lambda
    else:
        if not weak:                                  # HCl titrant
            conc["Cl-"] = ct
            lam["Cl-"] = strong_titrant_anion_lambda
        elif "T(protonated)" in sp:                   # weak base titrant B → BH+
            conc["BH+"] = sp["T(protonated)"]
            lam["BH+"] = weak_base_titrant_cation_lambda
    return conc, lam


def conductometric_curve_acid_base(model: AcidBaseModel, v_max_ml: float | None = None,
                                   n: int = 200,
                                   temperature_c: float | None = None) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``(V, κ_µS/cm)`` conductometric curve of an acid-base titration.

    Exact engine pH + limiting ionic conductivities + linear temperature
    correction.  Classic shapes: strong/strong gives a V, weak acid/strong
    base falls then rises after the end point.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1)
    >>> v, k = conductometric_curve_acid_base(m, n=50)
    >>> bool(k[0] > k[len(k)//2] < k[-1])     # V-shape
    True
    """
    t = model.temperature_c if temperature_c is None else temperature_c
    top = v_max_ml if v_max_ml is not None else 2.0 * model.n_steps * model.c_analyte * model.v_analyte_ml / model.c_titrant
    v = np.linspace(0.0, top, n)
    kappa = []
    for x in v:
        conc, lam = conductometric_species_acid_base(model, float(x))
        total = 0.0
        for ion, c in conc.items():
            if c > 0.0:
                total += lam[ion] * c
        kappa.append(total * 1000.0 * (1.0 + 0.019 * (t - 25.0)))
    return v, np.array(kappa)


# -------------------------------------------------------------- photometry
def absorbance(conc: Mapping[str, float], epsilon: Mapping[str, float],
               path_cm: float = 1.0) -> float:
    """Beer-Lambert absorbance ``A = l · Σ ε_i c_i`` (only absorbing species).

    Raises
    ------
    ValueError
        On negative concentrations or non-positive path length.

    Example
    -------
    >>> round(absorbance({"MnO4-": 1.0e-4}, {"MnO4-": 2400.0}), 3)
    0.24
    """
    if path_cm <= 0.0:
        raise ValueError("path length must be positive")
    total = 0.0
    for sp_name, c in conc.items():
        if c < 0.0:
            raise ValueError(f"negative concentration for {sp_name!r}")
        if sp_name in epsilon:
            total += epsilon[sp_name] * c
    return total * path_cm


EPSILON_MNO4_525 = 2400.0
"""Molar absorptivity of MnO4- at 525 nm (L·mol⁻¹·cm⁻¹); textbook value."""


def permanganate_curve(m: RedoxModel, v_max_ml: float | None = None, n: int = 200,
                       epsilon_mno4: float = EPSILON_MNO4_525, path_cm: float = 1.0,
                       wavelength_nm: int = 525) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``(V, A)`` photometric curve of a permanganate titration.

    ``MnO4-`` (the titrant = Ox2) is the only absorber (ε ≈ 2400 L·mol⁻¹·cm⁻¹
    at 525 nm); its free concentration is the exact post-equilibrium excess
    from the general redox solver, so A stays ≈ 0 before the end point and
    rises linearly afterwards — the classic photometric end point.

    Example
    -------
    >>> m = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)
    >>> v, a = permanganate_curve(m, n=50)
    >>> bool(a[10] < 1e-6 and a[-1] > 0.05)
    True
    """
    _ = wavelength_nm  # the built-in epsilon is the 525 nm value
    top = v_max_ml if v_max_ml is not None else 2.0 * redox_equivalence_volume_of(m)
    v = np.linspace(top / n, top, n)
    a = [absorbance({"MnO4-": redox_species(m, float(x))["Ox2"]},
                    {"MnO4-": epsilon_mno4}, path_cm) for x in v]
    return v, np.array(a)


def colored_species_curve(volumes: Sequence[float], concentrations: Sequence[Mapping[str, float]],
                          epsilon: Mapping[str, float], path_cm: float = 1.0) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Generic photometric curve from per-volume species maps (advanced use)."""
    a = [absorbance(c, epsilon, path_cm) for c in concentrations]
    return np.asarray(list(volumes)), np.asarray(a)


def quantity_of(model: TitrationModel, v_ml: float) -> float:
    """Dispatch the family's measured quantity: pH / E(V) / pAg / pM.

    Used by the generic session layer (desktop UI and web) so that the same
    plot, derivative and end-point machinery works for every family.

    Example
    -------
    >>> round(quantity_of(RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1), 25.0), 3)
    0.77
    """
    from .acid_base import AcidBaseModel as _ABM
    from .phase2 import EdtaModel, PrecipitationModel, edta_pm, precip_pag
    if isinstance(model, _ABM):
        return ph_at(model, v_ml)
    if isinstance(model, RedoxModel):
        return redox_potential(model, v_ml) if v_ml > 0.0 else float("nan")
    if isinstance(model, PrecipitationModel):
        return precip_pag(model, v_ml)
    if isinstance(model, EdtaModel):
        return edta_pm(model, v_ml)
    raise TypeError(f"no quantity defined for {type(model).__name__}")


__all__ = [
    "REF_ELECTRODES", "emf_mv", "potentiometric_curve_redox",
    "LAMBDA_25", "conductivity_us_cm", "conductometric_species_acid_base",
    "conductometric_curve_acid_base", "absorbance", "permanganate_curve",
    "colored_species_curve", "quantity_of", "EPSILON_MNO4_525",
]
