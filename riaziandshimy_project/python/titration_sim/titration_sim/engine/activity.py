"""Activity coefficients and ionic strength (phase-2 limitation fix).

The previous scope note of :mod:`titration_sim.engine.phase2` said "no ionic
strength correction".  This module removes that limitation for the families
where it matters most:

* redox formal potentials (``E°' = E° + (s/n) log(γ_Ox/γ_Red)``),
* precipitation solubility products (``Ksp' = Ksp° / (γ_+ γ_-)``),
* complexometry formation constants (``Kf' = Kf · γ_M γ_Y / γ_MY``).

Three models are provided, mirroring the choices of the web app
(``titration-lab.html``, Berto et al. 2024 comparison):

* ``"davies"`` — Davies equation, recommended default up to I ≈ 0.5 mol/L;
* ``"edh"``   — extended Debye-Hückel with the common ``B·a ≈ 1.5`` lumping
  (ion-size parameter folded into one constant);
* ``"dh"``    — plain (limiting law) Debye-Hückel, I < 0.01 mol/L only.

All functions are pure and exact to the model equations (no shortcuts).

Reference: [MATH.md section 16.5]
"""
from __future__ import annotations

import math
from typing import Literal

GammaModel = Literal["davies", "edh", "dh"]

#: Debye-Hückel constant A in water at 25 C (mol/L)^(-1/2); the small
#: temperature dependence (A = 0.4918 at 20 C, 0.5091 at 25 C, 0.533 at 40 C)
#: is folded into :func:`dh_A` with a linear fit (APPROXIMATION < 1 % for
#: 15 ... 35 C).
A_25 = 0.509


def dh_A(temperature_c: float = 25.0) -> float:
    """Debye-Hückel constant ``A`` at a given temperature.

    Linear interpolation of the tabulated values (0, 15, 20, 25, 30, 35, 40,
    50 C); the tabulated values themselves scatter by ±0.005 between sources
    (documented APPROXIMATION).

    Example
    -------
    >>> round(dh_A(25.0), 3)
    0.509
    >>> dh_A(5.0) < dh_A(25.0)
    True
    """
    table = ((0.0, 0.4883), (15.0, 0.500), (20.0, 0.496), (25.0, 0.509),
             (30.0, 0.515), (35.0, 0.521), (40.0, 0.527), (50.0, 0.539))
    t = temperature_c
    if t <= table[0][0]:
        return table[0][1]
    for (t0, a0), (t1, a1) in zip(table, table[1:]):
        if t <= t1:
            return a0 + (a1 - a0) * (t - t0) / (t1 - t0)
    return table[-1][1]


def ionic_strength(charges_concs: dict[float, float]) -> float:
    """Ionic strength ``I = 0.5 Σ c_i z_i²`` from a ``{charge: concentration}`` map.

    Concentrations of species with the same charge are simply added.

    Example
    -------
    >>> round(ionic_strength({1: 0.1, -1: 0.1}), 10)
    0.1
    >>> round(ionic_strength({2: 0.1, -1: 0.2}), 10)
    0.3
    """
    return 0.5 * sum(z * z * c for z, c in charges_concs.items() if c)


def log_gamma(z: float, ionic: float, model: GammaModel = "davies",
              temperature_c: float = 25.0) -> float:
    """``log10 γ`` of an ion of charge ``z`` at ionic strength ``ionic``.

    Davies:    ``log γ = -A z² (√I/(1+√I) - 0.3 I)``
    Ext. DH:   ``log γ = -A z² √I/(1 + 1.5 √I)``
    Limiting:  ``log γ = -A z² √I``

    Neutral species: γ = 1 by convention (log γ = 0; the salting-out term of
    the extended models is deliberately omitted — documented APPROXIMATION).

    Parameters
    ----------
    z : float
        Ion charge (signed).
    ionic : float
        Ionic strength (mol/L), ``0 <= I``.
    model : str
        ``"davies"`` (default), ``"edh"`` or ``"dh"``.
    temperature_c : float
        For the temperature-dependent constant ``A``.

    Raises
    ------
    ValueError
        If ``ionic`` is negative or the model name is unknown.

    Example
    -------
    >>> round(log_gamma(2, 0.1), 4)
    -0.4281
    >>> round(log_gamma(1, 0.1, "davies"), 4)
    -0.107
    >>> round(log_gamma(0, 0.5), 6)
    0.0
    """
    if ionic < 0.0:
        raise ValueError("ionic strength must be non-negative")
    if model not in ("davies", "edh", "dh"):
        raise ValueError(f"unknown gamma model {model!r}")
    if z == 0 or ionic == 0.0:
        return 0.0
    a = dh_A(temperature_c)
    s = math.sqrt(ionic)
    if model == "davies":
        return -a * z * z * (s / (1.0 + s) - 0.3 * ionic)
    if model == "edh":
        return -a * z * z * s / (1.0 + 1.5 * s)
    return -a * z * z * s


def gamma(z: float, ionic: float, model: GammaModel = "davies",
          temperature_c: float = 25.0) -> float:
    """``γ = 10^(log γ)`` — activity coefficient (see :func:`log_gamma`).

    Example
    -------
    >>> round(gamma(1, 0.01), 4)
    0.9021
    """
    return float(10.0 ** log_gamma(z, ionic, model, temperature_c))


def concentration_ksp(ksp_thermo: float, ionic: float, z_cation: float = 1,
                      z_anion: float = -1, model: GammaModel = "davies",
                      temperature_c: float = 25.0) -> float:
    """Conditional (concentration) solubility product at fixed ionic strength.

    ``Ksp° = a_+ a_- = γ_+ γ_- [M][X]  =>  Ksp' = Ksp° / (γ_+ γ_-)``

    For salts like ``Ag2CrO4`` pass ``z_cation=1, z_anion=-2`` and keep the
    *concentration form* ``Ksp' = [Ag+]²[CrO4]`` consistent in the caller.

    Example
    -------
    >>> c = concentration_ksp(1.8e-10, 0.1)
    >>> c > 1.8e-10
    True
    """
    return ksp_thermo / (gamma(z_cation, ionic, model, temperature_c)
                         * gamma(z_anion, ionic, model, temperature_c))


def formal_potential_shift(e_standard: float, n: float, slope_v: float,
                           z_ox: float, z_red: float, ionic: float,
                           coef_ox: float = 1.0, coef_red: float = 1.0,
                           model: GammaModel = "davies",
                           temperature_c: float = 25.0) -> float:
    """Formal potential ``E°'`` at ionic strength (redox activity correction).

    Writing the Nernst equation in concentrations,

    ``E = E° + (s/n) log( a_Ox^c_o / a_Red^c_r )``
      ``= E° + (s/n) log( (γ_Ox [Ox])^c_o / (γ_Red [Red])^c_r )``
      ``= E°' + (s/n) log( [Ox]^c_o / [Red]^c_r )``

    with ``E°' = E° + (s/n) (c_o log γ_Ox − c_r log γ_Red)`` — returned here.

    Example
    -------
    >>> round(formal_potential_shift(0.771, 1, 0.05916, 3, 2, 0.1), 5)
    0.73934
    """
    lg = (coef_ox * log_gamma(z_ox, ionic, model, temperature_c)
          - coef_red * log_gamma(z_red, ionic, model, temperature_c))
    return e_standard + slope_v / n * lg


def conditional_kf(kf_thermo: float, ionic: float, z_metal: float = 0.0,
                   z_ligand: float = 0.0, z_complex: float = 0.0,
                   model: GammaModel = "davies",
                   temperature_c: float = 25.0) -> float:
    """Concentration formation constant ``Kf' = Kf° γ_M γ_Y / γ_MY``.

    ``Kf° = a_MY/(a_M a_Y)  =>  Kf' = Kf° γ_M γ_Y / γ_MY``

    Example
    -------
    >>> kf = conditional_kf(10**10.65, 0.1, 2, -4)
    >>> kf < 10**10.65          # gamma_Y4- dominates: concentration Kf is smaller
    True
    """
    return (kf_thermo * gamma(z_metal, ionic, model, temperature_c)
            * gamma(z_ligand, ionic, model, temperature_c)
            / max(gamma(z_complex, ionic, model, temperature_c), 1e-300))


__all__ = [
    "GammaModel", "A_25", "dh_A", "ionic_strength", "log_gamma", "gamma",
    "concentration_ksp", "formal_potential_shift", "conditional_kf",
]
