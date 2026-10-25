"""Equilibrium solver for proton (acid-base) systems.

The solver uses the *proton balance* formulation of the potentiometry
literature (Berto et al., Anal. Chim. Acta 1303 (2024) 342476, section 2.3 /
2.4; MATH.md section 2):

    T_H = [H+] - [OH-] + sum_l C_l * (sum_i i*beta_li*h^i) / (1 + sum_i beta_li*h^i)

where every ligand ``l`` is described by its *cumulative protonation
constants* ``beta_li`` relative to its fully deprotonated reference
species.  This is mathematically identical to the full cubic/polynomial of
the "second answer" (sections 2.4, 2.5) but is solved on the log10(h) axis
with a bracketed Brent method, which is robust for any number of protons
(including the hexaprotic acid of the paper) and has no spurious roots.

Reference: [MATH.md sections 2.1-2.6, 13.3]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

LOG_H_MIN = -20.0
LOG_H_MAX = 4.0


class EquilibriumError(ValueError):
    """Raised when an equilibrium problem is ill-posed or cannot be solved."""


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-13,
    max_iter: int = 200,
) -> float:
    """Find a root of ``f`` in ``[a, b]`` with Brent's method.

    Parameters
    ----------
    f : callable
        Continuous scalar function.
    a, b : float
        Bracket; ``f(a)`` and ``f(b)`` must have opposite signs (or one of
        them must be zero).
    tol : float
        Absolute tolerance on the root.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    float
        The root.

    Raises
    ------
    EquilibriumError
        If the interval does not bracket a sign change.

    Example
    -------
    >>> round(brent(lambda x: x * x - 2.0, 0.0, 2.0), 9)
    1.414213562
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise EquilibriumError("root is not bracketed")
    c, fc = b, fb
    d = e = b - a
    for _ in range(max_iter):
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * 2.2e-16 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                qq = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0))
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            m1 = 3.0 * xm * q - abs(tol1 * q)
            m2 = abs(e * q)
            if 2.0 * p < min(m1, m2):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm >= 0.0 else -tol1)
        fb = f(b)
    return b


@dataclass(frozen=True)
class Ligand:
    """A protonatable component.

    Attributes
    ----------
    conc : float
        Total analytical concentration in the vessel (mol/L).
    log_beta : tuple[float, ...]
        Cumulative protonation constants log10(beta_i), i = 1..n, relative
        to the fully deprotonated reference species.

    Example
    -------
    >>> Ligand(0.1, (4.76,)).log_beta
    (4.76,)
    """

    conc: float
    log_beta: tuple[float, ...]


def pka_to_log_beta(pka: Sequence[float] | np.ndarray[Any, Any]) -> tuple[float, ...]:
    """Convert dissociation pKa values of a protonated form to cumulative log(beta).

    ``pka`` are the pKa values of HnA in ascending order (pKa1 < pKa2 < ...).
    The first protonation constant of the fully deprotonated species is
    10**pKa_n (the *last* dissociation).

    Parameters
    ----------
    pka : sequence of float
        pKa1 ... pKan.

    Returns
    -------
    tuple[float, ...]
        log10(beta_1) ... log10(beta_n).

    Raises
    ------
    EquilibriumError
        If ``pka`` is empty or not sorted ascending.

    Example
    -------
    >>> pka_to_log_beta([2.0, 7.0])
    (7.0, 9.0)

    [MATH.md section 2.5]
    """
    if len(pka) == 0:
        raise EquilibriumError("at least one pKa is required")
    if any(pka[i] > pka[i + 1] for i in range(len(pka) - 1)):
        raise EquilibriumError("pKa values must be given in ascending order")
    n = len(pka)
    step = [pka[n - i] for i in range(1, n + 1)]
    return tuple(float(v) for v in np.cumsum(step))


def stepwise_to_cumulative(log_k: Sequence[float] | np.ndarray[Any, Any]) -> tuple[float, ...]:
    """Cumulative log(beta) from stepwise log(K).

    Example
    -------
    >>> stepwise_to_cumulative([10, 8])
    (10.0, 18.0)
    """
    return tuple(float(v) for v in np.cumsum(np.asarray(log_k, dtype=float)))


def cumulative_to_stepwise(log_beta: Sequence[float] | np.ndarray[Any, Any]) -> tuple[float, ...]:
    """Stepwise log(K) from cumulative log(beta).

    Example
    -------
    >>> cumulative_to_stepwise([10, 18])
    (10.0, 8.0)
    """
    b = np.asarray(log_beta, dtype=float)
    return tuple(float(v) for v in np.diff(np.concatenate(([0.0], b))))


def _ligand_term(h: float, lig: Ligand) -> float:
    """Mean number of bound protons per ligand at free proton conc. ``h``."""
    hp = 1.0
    den = 1.0
    num = 0.0
    for i, lb in enumerate(lig.log_beta, start=1):
        hp *= h
        t = 10.0**lb * hp
        den += t
        num += i * t
    return num / den


def proton_balance(
    log_h: float, total_h: float, ligands: Sequence[Ligand], kw: float
) -> float:
    """Residual of the proton balance (zero at equilibrium).

    Parameters
    ----------
    log_h : float
        log10 of the free proton concentration.
    total_h : float
        Analytical proton excess ``T_H`` relative to the reference species
        (mol/L); acid contributes positive, base negative.
    ligands : sequence of Ligand
        Protonatable components.
    kw : float
        Ion product of water (concentration scale).

    Returns
    -------
    float
        ``[H] - Kw/[H] + sum(...) - T_H``.

    [MATH.md section 2.5]
    """
    h = 10.0**log_h
    s = h - kw / h - total_h
    for lig in ligands:
        s += lig.conc * _ligand_term(h, lig)
    return float(s)


def solve_log_h(
    total_h: float,
    ligands: Sequence[Ligand],
    kw: float,
    lo: float = LOG_H_MIN,
    hi: float = LOG_H_MAX,
) -> float:
    """Solve the proton balance for log10([H+]).

    The balance is strictly increasing in [H+], so the root is unique.

    Parameters
    ----------
    total_h : float
        Net analytical proton excess (mol/L).
    ligands : sequence of Ligand
        Protonatable components.
    kw : float
        Ion product of water.
    lo, hi : float
        Search bracket on log10(h).

    Returns
    -------
    float
        log10([H+]).

    Raises
    ------
    EquilibriumError
        If ``kw`` is not positive or the root is outside the bracket.

    Example
    -------
    >>> round(-solve_log_h(0.01, [], 1e-14), 6)
    2.0

    [MATH.md sections 2.2, 2.4, 3.1]
    """
    if kw <= 0.0:
        raise EquilibriumError("Kw must be positive")
    # Non-aqueous solvents have pKs up to ~35: widen the lower bracket so that [H+] ~ Ks/[base] stays inside.
    lo = min(lo, math.log10(kw) - 4.0)
    return brent(lambda x: proton_balance(x, total_h, ligands, kw), lo, hi, tol=1e-13)


def species_fractions(log_h: float, log_beta: Sequence[float] | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Mole fractions alpha_0..alpha_n of a protonatable component.

    Index ``i`` is the number of protons bound to the reference species.

    Example
    -------
    >>> a = species_fractions(0.0, [2.0])
    >>> bool(abs(a.sum() - 1.0) < 1e-12)
    True

    [MATH.md section 2.5 (alpha fractions)]
    """
    h = 10.0**log_h
    terms = [1.0]
    hp = 1.0
    for lb in log_beta:
        hp *= h
        terms.append(10.0**lb * hp)
    arr = np.asarray(terms, dtype=float)
    return np.asarray(arr / arr.sum(), dtype=float)


def weak_acid_cubic_ph(ka: float, c: float, kw: float = 1e-14) -> float:
    """pH of a monoprotic weak acid from the *full cubic* (reference solution).

    Solves ``h^3 + Ka h^2 - (Ka C + Kw) h - Ka Kw = 0`` with numpy.roots and
    returns the physical (positive real) root.  Used as an independent check
    of :func:`solve_log_h`.

    Raises
    ------
    EquilibriumError
        If no positive real root exists.

    Example
    -------
    >>> round(weak_acid_cubic_ph(1.8e-5, 0.1), 3)
    2.875

    [MATH.md section 2.4]
    """
    roots = np.roots([1.0, ka, -(ka * c + kw), -ka * kw])
    pos = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0.0]
    if not pos:
        raise EquilibriumError("no positive real root")
    return -math.log10(max(pos))


def weak_acid_ph_approx(ka: float, c: float) -> tuple[float, float]:
    """Simple approximation pH = (pKa - log C) / 2 and its error.

    APPROXIMATION: valid when [H+] << C (error < 5 % in [H+] when
    Ka/C < ~0.003) and [H+] >> [OH-].  The returned second value is the
    error in pH units against the exact balance (water at Kw = 1e-14).

    Returns
    -------
    tuple[float, float]
        (approximate pH, approximate minus exact).

    Example
    -------
    >>> ph, err = weak_acid_ph_approx(1.8e-5, 0.1)
    >>> abs(err) < 0.01
    True

    [MATH.md section 2.4]
    """
    approx = 0.5 * (-math.log10(ka) - math.log10(c))
    exact = weak_acid_cubic_ph(ka, c)
    return approx, approx - exact


def henderson_hasselbalch(pka: float, n_base: float, n_acid: float) -> float:
    """Henderson-Hasselbalch pH from mole amounts.

    APPROXIMATION: valid when the acid and its conjugate base are the only
    significant proton donors/acceptors, i.e. [H+], [OH-] << both
    concentrations (typically within 0.01 pH between 10 % and 90 %
    neutralisation of 0.1 M acids with pKa 4-10).

    Raises
    ------
    EquilibriumError
        If either amount is not positive.

    Example
    -------
    >>> henderson_hasselbalch(4.76, 1.0, 1.0)
    4.76

    [MATH.md section 3.2 (buffer region)]
    """
    if n_base <= 0.0 or n_acid <= 0.0:
        raise EquilibriumError("both amounts must be positive")
    return pka + math.log10(n_base / n_acid)
