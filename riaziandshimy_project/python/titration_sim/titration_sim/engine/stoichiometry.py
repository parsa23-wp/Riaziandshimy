"""Basic titration arithmetic (prompt-1 sections 1-12, 21-28, 42-50, 62, 83-90).

Everything here is exact algebra; no approximation is involved except where
stated.  Volumes are in litres or millilitres as indicated by the argument
name.

Reference: [MATH.md sections 1.1, 1.2, 10]
"""
from __future__ import annotations

import math


def moles_from_mass(mass_g: float, molar_mass: float) -> float:
    """n = m / M.

    Raises
    ------
    ValueError
        If the molar mass is not positive.

    Example
    -------
    >>> round(moles_from_mass(5.844, 58.44), 6)
    0.1
    """
    if molar_mass <= 0:
        raise ValueError("molar mass must be positive")
    return mass_g / molar_mass


def mass_from_moles(n_mol: float, molar_mass: float) -> float:
    """m = n * M.

    Example
    -------
    >>> mass_from_moles(0.1, 58.44)
    5.844
    """
    return n_mol * molar_mass


def molarity(n_mol: float, v_solution_l: float) -> float:
    """C = n / V (V is the volume of *solution* in litres).

    Raises
    ------
    ValueError
        If the volume is not positive.

    Example
    -------
    >>> molarity(0.025, 0.25)
    0.1
    """
    if v_solution_l <= 0:
        raise ValueError("volume must be positive")
    return n_mol / v_solution_l


def required_mass(c_mol_l: float, v_l: float, molar_mass: float) -> float:
    """m = C V M (Goal Seek: mass needed for a target molarity).

    Example
    -------
    >>> round(required_mass(0.1, 0.25, 58.44), 3)
    1.461
    """
    return c_mol_l * v_l * molar_mass


def dilution_volume(c_stock: float, c_target: float, v_target: float) -> float:
    """V1 = C2 V2 / C1 (dilution only, NOT titration stoichiometry).

    Raises
    ------
    ValueError
        If the target is more concentrated than the stock.

    Example
    -------
    >>> dilution_volume(1.0, 0.1, 250.0)
    25.0
    """
    if c_target > c_stock:
        raise ValueError("target must not exceed stock concentration")
    return c_target * v_target / c_stock


def analyte_concentration(
    c_titrant: float, v_titrant: float, v_analyte: float, a: float = 1.0, b: float = 1.0
) -> float:
    """C_A = a C_B V_B / (b V_A) for a A + b B -> products.

    ``C1 V1 = C2 V2`` is only the special case a = b = 1 (section 1.2).

    Raises
    ------
    ValueError
        If ``v_analyte`` or ``b`` is not positive.

    Example
    -------
    >>> analyte_concentration(0.1, 20.0, 10.0, a=1.0, b=2.0)
    0.1
    """
    if v_analyte <= 0 or b <= 0:
        raise ValueError("v_analyte and b must be positive")
    return a * c_titrant * v_titrant / (b * v_analyte)


def equivalents(c_mol_l: float, v_l: float, n_factor: float) -> float:
    """Eq = C V n-factor.

    Example
    -------
    >>> equivalents(0.5, 0.1, 2)
    0.1
    """
    return c_mol_l * v_l * n_factor


def normality(c_mol_l: float, n_factor: float) -> float:
    """N = C * n-factor (n-factor depends on the reaction, never assume N = M).

    Example
    -------
    >>> normality(0.5, 2)
    1.0
    """
    return c_mol_l * n_factor


def percent_by_mass(m_solute: float, m_solution: float) -> float:
    """w/w % = m_solute / m_solution * 100.

    Raises
    ------
    ValueError
        If the solution mass is not positive.

    Example
    -------
    >>> percent_by_mass(5.0, 100.0)
    5.0
    """
    if m_solution <= 0:
        raise ValueError("solution mass must be positive")
    return 100.0 * m_solute / m_solution


def molarity_from_percent(w_w_percent: float, density_g_ml: float, molar_mass: float) -> float:
    """M = (w/w % * rho * 10) / MW, rho in g/mL.

    Example
    -------
    >>> round(molarity_from_percent(37.0, 1.19, 36.46), 2)
    12.08
    """
    return w_w_percent * density_g_ml * 10.0 / molar_mass


def ppm_aqueous(mg_per_l: float) -> float:
    """ppm ~ mg/L for dilute aqueous solutions.

    APPROXIMATION: valid for density ~ 1 g/mL (error < 0.5 % below 5 g/L).

    Example
    -------
    >>> ppm_aqueous(12.5)
    12.5
    """
    return mg_per_l


def percent_error(experimental: float, true: float) -> float:
    """|exp - true| / true * 100.

    Raises
    ------
    ValueError
        If ``true`` is zero.

    Example
    -------
    >>> round(percent_error(9.9, 10.0), 9)
    1.0
    """
    if true == 0:
        raise ValueError("true value must be non-zero")
    return abs(experimental - true) / abs(true) * 100.0


def recovery_percent(measured: float, actual: float) -> float:
    """Recovery % = measured / actual * 100.

    Example
    -------
    >>> round(recovery_percent(9.8, 10.0), 9)
    98.0
    """
    if actual == 0:
        raise ValueError("actual value must be non-zero")
    return measured / actual * 100.0


def purity_percent(mass_pure: float, mass_sample: float) -> float:
    """Purity % = m_pure / m_sample * 100.

    Example
    -------
    >>> purity_percent(0.95, 1.0)
    95.0
    """
    if mass_sample <= 0:
        raise ValueError("sample mass must be positive")
    return 100.0 * mass_pure / mass_sample


def standardize_titrant(n_standard_mol: float, v_titrant_l: float, ratio: float = 1.0) -> float:
    """C_titrant = n_standard * ratio / V (primary-standard standardisation).

    ``ratio`` = mol titrant per mol standard in the reaction.

    Raises
    ------
    ValueError
        If the volume is not positive.

    Example
    -------
    >>> round(standardize_titrant(0.0025, 0.025), 6)
    0.1
    """
    if v_titrant_l <= 0:
        raise ValueError("volume must be positive")
    return n_standard_mol * ratio / v_titrant_l


def burette_used(v_initial: float, v_final: float) -> float:
    """V_used = V_f - V_i.

    Raises
    ------
    ValueError
        If the final reading is smaller than the initial one.

    Example
    -------
    >>> round(burette_used(0.05, 24.87), 2)
    24.82
    """
    if v_final < v_initial:
        raise ValueError("final reading below initial reading")
    return v_final - v_initial


def blank_corrected(v_sample: float, v_blank: float) -> float:
    """V_corr = V_sample - V_blank.

    Example
    -------
    >>> blank_corrected(25.1, 0.1)
    25.0
    """
    return v_sample - v_blank


def log10_safe(x: float) -> float:
    """log10 that raises a clear error for non-positive input.

    Example
    -------
    >>> log10_safe(100.0)
    2.0
    """
    if x <= 0:
        raise ValueError("argument must be positive")
    return math.log10(x)
