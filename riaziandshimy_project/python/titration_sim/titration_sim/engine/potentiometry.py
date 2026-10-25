"""Potentiometric refinement of protonation constants (Berto et al. 2024).

Implements, from "A tutorial on potentiometric data processing. Analysis of
software for optimization of protonation constants" (Anal. Chim. Acta 1303
(2024) 342476, CC BY):

* extended Nernst equation with liquid-junction terms (Eq. 10),
* weighted / unweighted non-linear least squares with the weights
  ``w = 1/s^2``, ``s^2 = s_E^2 + (dE/dV)^2 s_V^2`` (Eqs. 1-4),
* activity coefficients (Debye-Hueckel with A = 0.51, B = 1; Davies; extended
  Debye-Hueckel with B = 1.5) and stoichiometric <-> thermodynamic constants
  (Eqs. 7-9), variable-ionic-strength refinement,
* stepwise constants and their uncertainty including the covariance (Sec. 3.1),
* the simulated six-curve hexaprotic-acid benchmark (Sec. 2.4) and the four
  systematic-error scenarios of Sec. 2.6.

The Levenberg-Marquardt implementation is our own; no code of BSTAC,
HYPERQUAD, KEV, ReactLab or SUPERQUAD was used.

Reference: [MATH.md section 14 (paper)]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import local_slope_poly
from .equilibrium import cumulative_to_stepwise, stepwise_to_cumulative

# --------------------------------------------------------------------------
# activity coefficients
# --------------------------------------------------------------------------
DH_A = 0.51


def gamma_log(z: float, ionic_strength: float, model: str = "dh") -> float:
    """log10 of the activity coefficient of an ion of charge ``z``.

    ``dh``: -A z^2 sqrt(I)/(1+sqrt(I)) (A = 0.51, B = 1: the setting used in
    the paper's comparison, so that EDH-1 reduces to the DH of ReactLab);
    ``davies``: -A z^2 (sqrt(I)/(1+sqrt(I)) - 0.3 I);
    ``edh``: -A z^2 sqrt(I)/(1+1.5 sqrt(I)).

    APPROXIMATION: valid up to I ~ 0.001 (DH), ~0.1 (Davies, molar scale)
    mol/L (Berto 2024, section 2.3); errors grow beyond that.

    Raises
    ------
    ValueError
        For an unknown model name or negative ionic strength.

    Example
    -------
    >>> round(gamma_log(1, 0.1, "dh"), 4)
    -0.1225

    [MATH.md section 1.3 / paper Eq. 9]
    """
    if ionic_strength < 0:
        raise ValueError("ionic strength must be non-negative")
    s = math.sqrt(ionic_strength)
    z2 = z * z
    if model == "dh":
        return -DH_A * z2 * s / (1.0 + s)
    if model == "davies":
        return -DH_A * z2 * (s / (1.0 + s) - 0.3 * ionic_strength)
    if model == "edh":
        return -DH_A * z2 * s / (1.0 + 1.5 * s)
    raise ValueError(f"unknown activity model: {model}")


def _gamma_vec(z: float, ionic: np.ndarray[Any, Any], model: str) -> np.ndarray[Any, Any]:
    s = np.sqrt(np.maximum(ionic, 0.0))
    z2 = z * z
    if model == "dh":
        return np.asarray(-DH_A * z2 * s / (1.0 + s))
    if model == "davies":
        return np.asarray(-DH_A * z2 * (s / (1.0 + s) - 0.3 * ionic))
    if model == "edh":
        return np.asarray(-DH_A * z2 * s / (1.0 + 1.5 * s))
    raise ValueError(f"unknown activity model: {model}")


def to_stoichiometric(log_beta_thermo: Sequence[float] | np.ndarray[Any, Any], z_a: int, ionic_strength: float, model: str = "dh") -> list[float]:
    """Thermodynamic cumulative constants -> stoichiometric at ``ionic_strength``.

    ``log beta_i = log beta_i^T + log g_A + i log g_H - log g_(HiA)`` (Eq. 8).

    Example
    -------
    >>> out = to_stoichiometric([0.0], -1, 0.0)
    >>> out
    [0.0]
    """
    g_h = gamma_log(1, ionic_strength, model)
    g_a = gamma_log(z_a, ionic_strength, model)
    return [v + g_a + (k + 1) * g_h - gamma_log(z_a + k + 1, ionic_strength, model) for k, v in enumerate(log_beta_thermo)]


def to_thermodynamic(log_beta_stoich: Sequence[float] | np.ndarray[Any, Any], z_a: int, ionic_strength: float, model: str = "dh") -> list[float]:
    """Inverse of :func:`to_stoichiometric`.

    Example
    -------
    >>> a = to_stoichiometric([10.0, 18.0], -2, 0.1)
    >>> [round(v, 9) for v in to_thermodynamic(a, -2, 0.1)]
    [10.0, 18.0]
    """
    g_h = gamma_log(1, ionic_strength, model)
    g_a = gamma_log(z_a, ionic_strength, model)
    return [v - (g_a + (k + 1) * g_h - gamma_log(z_a + k + 1, ionic_strength, model)) for k, v in enumerate(log_beta_stoich)]


# --------------------------------------------------------------------------
# model definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtonationModel:
    """Description of the protonation equilibria of ``H_n A``.

    Attributes
    ----------
    n : int
        Number of protonation steps.
    z_a : int
        Charge of the fully deprotonated species A.
    n_init : int
        Protons per ligand initially present (n for the acid H_nA).
    sign : int
        -1 if the titrant is a base, +1 if it is an acid.
    mode : str
        'const' (constant ionic strength, stoichiometric constants) or
        'var' (variable ionic strength, thermodynamic constants).
    gmodel : str
        Activity model for 'var'.
    kw : float
        Kw (concentration scale for 'const'; thermodynamic for 'var').
    cs : float
        Concentration of background 1:1 salt (mol/L) used in 'var' mode.
    """

    n: int
    z_a: int
    n_init: int
    sign: int = -1
    mode: str = "const"
    gmodel: str = "dh"
    kw: float = 10.0**-13.77
    cs: float = 0.0


@dataclass
class FitCurve:
    """One titration curve.

    Attributes
    ----------
    name : str
    v0_ml : float
        Initial volume (mL).
    cl_mol_l : float
        Initial ligand concentration (mol/L).
    v_ml, y : ndarray
        Titrant volume (mL) and observable (emf in mV or pH).
    h0_mol_l : float
        Extra initial free proton excess (mol/L).
    f_h : float
        Multiplier of the initial proton amount (refinable "purity").
    """

    name: str
    v0_ml: float
    cl_mol_l: float
    v_ml: np.ndarray[Any, Any]
    y: np.ndarray[Any, Any]
    h0_mol_l: float = 0.0
    f_h: float = 1.0


@dataclass(frozen=True)
class Calibration:
    """Electrode parameters of Eq. 10: E = E0' + S log[H] + jA[H] + jB[OH].

    Example
    -------
    >>> Calibration(405.0, -59.16, -64.0, 0.0).e0
    405.0
    """

    e0: float = 405.0
    slope: float = -59.16
    ja: float = -64.0
    jb: float = 0.0


def _bound_protons(h: np.ndarray[Any, Any], log_beta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    hp = np.ones_like(h)
    den = np.ones_like(h)
    num = np.zeros_like(h)
    for i, lb in enumerate(log_beta, start=1):
        hp = hp * h
        t = 10.0**lb * hp
        den = den + t
        num = num + i * t
    return np.asarray(num / den)


def _solve_vec(total: np.ndarray[Any, Any], conc: np.ndarray[Any, Any], log_beta: np.ndarray[Any, Any], kw: np.ndarray[Any, Any] | float) -> np.ndarray[Any, Any]:
    """Vectorised bisection for log10 h of the proton balance."""
    lo = np.full_like(total, -20.0)
    hi = np.full_like(total, 4.0)
    for _ in range(62):
        mid = 0.5 * (lo + hi)
        h = 10.0**mid
        f = h - kw / h - total + conc * _bound_protons(h, log_beta)
        pos = f > 0.0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return np.asarray(0.5 * (lo + hi))


@dataclass(frozen=True)
class CurvePrediction:
    """Calculated values along one curve.

    Attributes
    ----------
    y : ndarray  (emf in mV, or pH)
    h : ndarray  free [H+] (mol/L)
    ph : ndarray activity pH (-log10 a_H)
    ionic : ndarray  ionic strength (NaN in 'const' mode)
    alpha : ndarray  species fractions (points x (n+1))
    """

    y: np.ndarray[Any, Any]
    h: np.ndarray[Any, Any]
    ph: np.ndarray[Any, Any]
    ionic: np.ndarray[Any, Any]
    alpha: np.ndarray[Any, Any]


def predict_curve(
    model: ProtonationModel,
    log_beta: Sequence[float] | np.ndarray[Any, Any],
    calib: Calibration,
    ct: float,
    curve: FitCurve,
    obs: str = "E",
) -> CurvePrediction:
    """Calculated observable for one curve (Eq. 10 and the proton balance).

    In 'var' mode ``log_beta`` are thermodynamic constants and the ionic
    strength of every titration point is obtained self-consistently
    (background salt + titrant counter-ion + all charged species), then used
    for the activity coefficients and for E0' (through log g_H).

    Raises
    ------
    ValueError
        If ``obs`` is not 'E' or 'pH'.

    [paper Eqs. 8-10]
    """
    if obs not in ("E", "pH"):
        raise ValueError("obs must be 'E' or 'pH'")
    v = np.asarray(curve.v_ml, dtype=float)
    lb = np.asarray(log_beta, dtype=float)
    n = model.n
    vt = curve.v0_ml + v
    total = ((model.n_init * curve.cl_mol_l + curve.h0_mol_l) * curve.f_h * curve.v0_ml + model.sign * ct * v) / vt
    conc = curve.cl_mol_l * curve.v0_ml / vt
    if model.mode != "var":
        kw: np.ndarray[Any, Any] | float = model.kw
        lg_h = np.zeros_like(v)
        ionic = np.full_like(v, np.nan)
        lbv = lb
        logh = _solve_vec(total, conc, lbv, kw)
    else:
        cion = (abs(curve.h0_mol_l * curve.f_h) * curve.v0_ml + ct * v) / vt
        ionic = model.cs + 0.5 * cion + 1e-3
        logh = np.zeros_like(v)
        lg_h = np.zeros_like(v)
        for _ in range(80):
            lg_h = _gamma_vec(1, ionic, model.gmodel)
            kw = model.kw / 10.0 ** (2.0 * lg_h)
            lg_a = _gamma_vec(model.z_a, ionic, model.gmodel)
            # per-point betas
            beta_cols = [lb[k] + lg_a + (k + 1) * lg_h - _gamma_vec(model.z_a + k + 1, ionic, model.gmodel) for k in range(n)]
            logh = _solve_var(total, conc, beta_cols, kw)
            h = 10.0**logh
            hp = np.ones_like(h)
            den = np.ones_like(h)
            zs = model.z_a * model.z_a * np.ones_like(h)
            num_z = zs.copy()
            for k in range(n):
                hp = hp * h
                t = 10.0 ** beta_cols[k] * hp
                den = den + t
                num_z = num_z + (model.z_a + k + 1) ** 2 * t
            ionic_new = model.cs + 0.5 * (cion + h + kw / h + conc * num_z / den)
            if float(np.max(np.abs(ionic_new - ionic))) < 1e-9:
                ionic = ionic_new
                break
            ionic = 0.5 * (ionic + ionic_new)
        lbv = np.asarray(beta_cols)  # shape (n, points)
    h = 10.0**logh
    if model.mode == "var":
        alpha = _alpha_var(h, lbv)
        kw_eff: np.ndarray[Any, Any] | float = kw
    else:
        alpha = _alpha_const(h, lbv)
        kw_eff = model.kw
    ph_act = -logh - lg_h
    if obs == "pH":
        y = ph_act
    else:
        y = calib.e0 + calib.slope * (logh + lg_h) + calib.ja * h + calib.jb * kw_eff / h
    return CurvePrediction(np.asarray(y), h, ph_act, ionic, alpha)


def _solve_var(total: np.ndarray[Any, Any], conc: np.ndarray[Any, Any], beta_cols: list[np.ndarray[Any, Any]], kw: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    lo = np.full_like(total, -20.0)
    hi = np.full_like(total, 4.0)
    for _ in range(62):
        mid = 0.5 * (lo + hi)
        h = 10.0**mid
        hp = np.ones_like(h)
        den = np.ones_like(h)
        num = np.zeros_like(h)
        for i, col in enumerate(beta_cols, start=1):
            hp = hp * h
            t = 10.0**col * hp
            den = den + t
            num = num + i * t
        f = h - kw / h - total + conc * num / den
        pos = f > 0.0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return np.asarray(0.5 * (lo + hi))


def _alpha_const(h: np.ndarray[Any, Any], lb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    cols = [np.ones_like(h)]
    hp = np.ones_like(h)
    for b in lb:
        hp = hp * h
        cols.append(10.0**b * hp)
    arr = np.stack(cols, axis=1)
    return np.asarray(arr / arr.sum(axis=1, keepdims=True))


def _alpha_var(h: np.ndarray[Any, Any], lb_cols: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    cols = [np.ones_like(h)]
    hp = np.ones_like(h)
    for k in range(lb_cols.shape[0]):
        hp = hp * h
        cols.append(10.0 ** lb_cols[k] * hp)
    arr = np.stack(cols, axis=1)
    return np.asarray(arr / arr.sum(axis=1, keepdims=True))


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


def weights_eq3(v_ml: Sequence[float] | np.ndarray[Any, Any], y: Sequence[float] | np.ndarray[Any, Any], s_y: float, s_v: float) -> np.ndarray[Any, Any]:
    """Weights ``w = 1/s^2`` with ``s^2 = s_E^2 + (dE/dV)^2 s_V^2`` (Eqs. 3, 4b).

    The slope is the 5-point local cubic first derivative (as SUPERQUAD and
    HYPERQUAD use).  Points near the equivalence point (large slope) get a
    small weight.

    Raises
    ------
    ValueError
        If ``s_y`` and ``s_v`` are both zero.

    Example
    -------
    >>> w = weights_eq3([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 0.1, 0.0)
    >>> bool(np.allclose(w, 100.0))
    True

    [paper Eqs. 3-4]
    """
    if s_y == 0.0 and s_v == 0.0:
        raise ValueError("s_y and s_v cannot both be zero")
    d = local_slope_poly(v_ml, y, 5, 3)
    return np.asarray(1.0 / (s_y * s_y + d * d * s_v * s_v))


# --------------------------------------------------------------------------
# refinement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """Outcome of :func:`fit_protonation`.

    Attributes
    ----------
    log_beta, log_k : ndarray
        Cumulative and stepwise (log10) constants.
    sd_log_beta, sd_log_k : ndarray
        Standard deviations; stepwise ones include the covariance:
        ``s^2(logK) = s^2(logb_n) + s^2(logb_{n-1}) - 2 cov``.
    calib : Calibration
    ct : float
    params : dict[str, tuple[float, float]]
        name -> (value, sd) of every refined parameter.
    u : float
        Uncertainty of the fit sqrt(SSR/(N-p)) (Eqs. 1, 2).
    ssr : float
    dof : int
    n_points : int
    converged : bool
    iterations : int
    cov : ndarray | None
    curves : list of CurvePrediction
    """

    log_beta: np.ndarray[Any, Any]
    log_k: np.ndarray[Any, Any]
    sd_log_beta: np.ndarray[Any, Any]
    sd_log_k: np.ndarray[Any, Any]
    calib: Calibration
    ct: float
    params: dict[str, tuple[float, float]]
    u: float
    ssr: float
    dof: int
    n_points: int
    converged: bool
    iterations: int
    cov: np.ndarray[Any, Any] | None
    curves: list[CurvePrediction] = field(default_factory=list)


def fit_protonation(
    curves: Sequence[FitCurve],
    model: ProtonationModel,
    log_beta0: Sequence[float] | np.ndarray[Any, Any],
    calib: Calibration,
    ct: float,
    refine: Sequence[str] = (),
    weighted: bool = True,
    s_y: float = 0.15,
    s_v: float = 0.005,
    obs: str = "E",
    max_iter: int = 120,
    fixed_log_beta: Sequence[bool] | None = None,
) -> FitResult:
    """Levenberg-Marquardt refinement of the cumulative protonation constants.

    Parameters
    ----------
    curves : sequence of FitCurve
    model : ProtonationModel
    log_beta0 : sequence of float
        Starting cumulative log(beta) (thermodynamic ones in 'var' mode).
    calib : Calibration
        Starting/fixed electrode parameters.
    ct : float
        Titrant concentration (mol/L).
    refine : sequence of str
        Extra refinable parameters among ``'e0', 'slope', 'ja', 'jb', 'ct',
        'cl'`` (ligand concentration of each curve, proton total
        = n C_L) and ``'fh'`` (total proton of each curve, C_L fixed).
    weighted : bool
        Weights of Eq. 3 (True) or unity (ReactLab style).
    s_y, s_v : float
        Standard deviations of the observable and of the volume.
    obs : str
        'E' or 'pH'.
    max_iter : int
    fixed_log_beta : sequence of bool, optional
        True entries keep that cumulative constant fixed.

    Returns
    -------
    FitResult

    Raises
    ------
    ValueError
        If there are fewer data points than parameters or an unknown
        parameter name is requested.

    [paper sections 2.2, 2.5, 3.1]
    """
    allowed = {"e0", "slope", "ja", "jb", "ct", "cl", "fh"}
    bad = set(refine) - allowed
    if bad:
        raise ValueError(f"unknown refinable parameter(s): {sorted(bad)}")
    cur = [FitCurve(c.name, c.v0_ml, c.cl_mol_l, np.asarray(c.v_ml, float), np.asarray(c.y, float), c.h0_mol_l, c.f_h) for c in curves]
    nb = model.n
    lb0 = np.asarray(log_beta0, dtype=float)
    fixed = list(fixed_log_beta) if fixed_log_beta is not None else [False] * nb
    # parameter bookkeeping --------------------------------------------------
    names: list[str] = []
    x0: list[float] = []
    steps: list[float] = []
    clips: list[float] = []
    for i in range(nb):
        if not fixed[i]:
            names.append(f"logb{i + 1}")
            x0.append(float(lb0[i]))
            steps.append(1e-5)
            clips.append(1.5)
    if obs == "E":
        for key, val, st, cl in (("e0", calib.e0, 1e-4, 30.0), ("slope", calib.slope, 1e-4, 5.0), ("ja", calib.ja, 1e-3, 200.0), ("jb", calib.jb, 1e-3, 200.0)):
            if key in refine:
                names.append(key)
                x0.append(float(val))
                steps.append(st)
                clips.append(cl)
    if "ct" in refine:
        names.append("ct")
        x0.append(ct)
        steps.append(ct * 1e-6)
        clips.append(ct * 0.2)
    for k, c in enumerate(cur):
        if "cl" in refine:
            names.append(f"cl{k + 1}")
            x0.append(c.cl_mol_l)
            steps.append(c.cl_mol_l * 1e-6)
            clips.append(c.cl_mol_l * 0.3)
        if "fh" in refine:
            names.append(f"fh{k + 1}")
            x0.append(c.f_h)
            steps.append(1e-6)
            clips.append(0.3)
    p = len(names)
    n_pts = sum(c.v_ml.size for c in cur)
    if n_pts - p <= 0:
        raise ValueError("fewer data points than parameters")
    if weighted:
        w_list = [weights_eq3(c.v_ml, c.y, s_y, s_v) for c in cur]
    else:
        w_list = [np.ones_like(c.v_ml) for c in cur]
    sw = np.concatenate([np.sqrt(w) for w in w_list])
    yexp = np.concatenate([c.y for c in cur])
    idx = {nm: j for j, nm in enumerate(names)}

    def unpack(theta: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], Calibration, float, list[FitCurve]]:
        lb = lb0.copy()
        for i in range(nb):
            j = idx.get(f"logb{i + 1}")
            if j is not None:
                lb[i] = theta[j]
        cal = Calibration(
            theta[idx["e0"]] if "e0" in idx else calib.e0,
            theta[idx["slope"]] if "slope" in idx else calib.slope,
            theta[idx["ja"]] if "ja" in idx else calib.ja,
            theta[idx["jb"]] if "jb" in idx else calib.jb,
        )
        ctv = theta[idx["ct"]] if "ct" in idx else ct
        cc: list[FitCurve] = []
        for k, c in enumerate(cur):
            cl = theta[idx[f"cl{k + 1}"]] if f"cl{k + 1}" in idx else c.cl_mol_l
            fh = theta[idx[f"fh{k + 1}"]] if f"fh{k + 1}" in idx else c.f_h
            cc.append(FitCurve(c.name, c.v0_ml, cl, c.v_ml, c.y, c.h0_mol_l, fh))
        return lb, cal, ctv, cc

    def calc(theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        lb, cal, ctv, cc = unpack(theta)
        return np.concatenate([predict_curve(model, lb, cal, ctv, c, obs).y for c in cc])

    def resid(theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray(sw * (yexp - calc(theta)))

    def jac(theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        cols = np.empty((n_pts, p))
        for j in range(p):
            t1 = theta.copy()
            t2 = theta.copy()
            t1[j] += steps[j]
            t2[j] -= steps[j]
            cols[:, j] = sw * (calc(t1) - calc(t2)) / (2.0 * steps[j])
        return cols

    theta = np.asarray(x0, dtype=float)
    r = resid(theta)
    ssr = float(r @ r)
    lam = 1e-3
    converged = False
    it = 0
    if p > 0:
        for it in range(max_iter):
            if ssr < 1e-24:
                converged = True
                break
            J = jac(theta)
            A = J.T @ J
            g = J.T @ r
            improved = False
            for _ in range(30):
                Ad = A + lam * np.diag(np.diag(A)) + 1e-300 * np.eye(p)
                try:
                    d = np.linalg.solve(Ad, g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                d = np.clip(d, -np.asarray(clips), np.asarray(clips))
                tn = theta + d
                rn = resid(tn)
                sn = float(rn @ rn)
                if np.isfinite(sn) and sn < ssr:
                    rel = (ssr - sn) / ssr
                    step_rel = float(np.max(np.abs(d) / (np.abs(theta) + 1e-3)))
                    theta, r, ssr = tn, rn, sn
                    lam = max(lam / 10.0, 1e-12)
                    improved = True
                    if rel < 1e-13 or step_rel < 1e-10:
                        converged = True
                    break
                lam *= 10.0
                if lam > 1e12:
                    break
            if not improved or converged:
                converged = True
                break
    else:
        converged = True
    dof = n_pts - p
    u2 = ssr / dof
    cov: np.ndarray[Any, Any] | None = None
    sd = np.full(p, np.nan)
    if p > 0:
        J = jac(theta)
        try:
            cov = np.linalg.inv(J.T @ J) * u2
            sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            cov = None
    lb, cal, ctv, cc = unpack(theta)
    sdb = np.zeros(nb)
    for i in range(nb):
        j = idx.get(f"logb{i + 1}")
        if j is not None and cov is not None:
            sdb[i] = sd[j]
    sdk = np.zeros(nb)
    for i in range(nb):
        if i == 0:
            sdk[i] = sdb[0]
            continue
        v = sdb[i] ** 2 + sdb[i - 1] ** 2
        ja_, jb_ = idx.get(f"logb{i + 1}"), idx.get(f"logb{i}")
        if cov is not None and ja_ is not None and jb_ is not None:
            v -= 2.0 * cov[ja_, jb_]
        sdk[i] = math.sqrt(max(v, 0.0))
    preds = [predict_curve(model, lb, cal, ctv, c, obs) for c in cc]
    params = {nm: (float(theta[j]), float(sd[j]) if cov is not None else float("nan")) for nm, j in idx.items()}
    return FitResult(
        lb, np.asarray(list(cumulative_to_stepwise(lb)), dtype=float), sdb, sdk, cal, ctv, params,
        math.sqrt(u2), ssr, dof, n_pts, converged, it + 1, cov, preds,
    )


# --------------------------------------------------------------------------
# benchmark dataset and error scenarios
# --------------------------------------------------------------------------

#: protonation constants of the hexaprotic benchmark (Berto 2024, section 2.4)
PAPER_LOG_K = (10.0, 8.0, 6.0, 4.0, 3.0, 2.0)
PAPER_PKW_I01 = 13.77
PAPER_CALIB = Calibration(405.0, -59.16, -64.0, 0.0)
PAPER_CL = (2e-3, 4e-3, 5e-3)

#: log K_n (BSTAC column) of Table 3 for the reference and perturbed fits
PAPER_TABLE3_BSTAC: dict[str, tuple[float, ...]] = {
    "reference": (10.010, 8.004, 6.003, 4.002, 3.003, 1.99),
    "wrong_formal_potential": (10.049, 8.018, 6.020, 4.020, 3.022, 2.06),
    "wrong_titrant": (10.303, 8.17, 6.14, 4.08, 3.01, 1.99),
    "wrong_acid": (9.979, 8.022, 6.061, 4.075, 3.058, 2.104),
}
#: Table 4: BSTAC at I = 0.1 constant and variable (DH) ionic strength
PAPER_TABLE4_BSTAC: dict[str, tuple[float, ...]] = {
    "constant": (10.005, 8.003, 6.002, 4.002, 3.012, 2.01),
    "variable": (10.219, 8.104, 6.052, 4.017, 3.02, 1.79),
}


def paper_delta(scenario: str) -> np.ndarray[Any, Any]:
    """Paper's (BSTAC) change of log K_n for a perturbation scenario.

    Scenarios: 'wrong_formal_potential', 'wrong_titrant', 'wrong_acid'
    (relative to Table 3 reference) and 'ionic_strength' (Table 4).

    Raises
    ------
    KeyError
        For an unknown scenario.

    Example
    -------
    >>> round(float(paper_delta('wrong_titrant')[0]), 3)
    0.293
    """
    if scenario == "ionic_strength":
        return np.asarray(np.asarray(PAPER_TABLE4_BSTAC["variable"]) - np.asarray(PAPER_TABLE4_BSTAC["constant"]))
    return np.asarray(np.asarray(PAPER_TABLE3_BSTAC[scenario]) - np.asarray(PAPER_TABLE3_BSTAC["reference"]))


def benchmark_curves(noise_sd: float = 0.05, seed: int = 7, n_points: int = 101) -> list[FitCurve]:
    """Six simulated titration curves of the hexaprotic benchmark.

    Three ligand concentrations (2, 4, 5 mM H6A, V0 = 25 mL, 0.1 M base),
    two replicates each, Gaussian noise on emf (mV).  Titration ends at
    111.5 % of the total equivalence volume.

    Raises
    ------
    ValueError
        If ``n_points < 10``.

    Example
    -------
    >>> len(benchmark_curves(0.0))
    6
    """
    if n_points < 10:
        raise ValueError("n_points must be >= 10")
    rng = np.random.default_rng(seed)
    model = ProtonationModel(6, -6, 6, -1, "const", "dh", 10.0**-PAPER_PKW_I01, 0.0)
    lb = stepwise_to_cumulative(PAPER_LOG_K)
    out: list[FitCurve] = []
    for cl in PAPER_CL:
        for rep in (1, 2):
            v_eq = 6.0 * cl * 25.0 / 0.1
            v = np.linspace(0.0, 1.115 * v_eq, n_points)
            c = FitCurve(f"C={cl * 1e3:g} mM #{rep}", 25.0, cl, v, np.zeros_like(v))
            y = predict_curve(model, lb, PAPER_CALIB, 0.1, c, "E").y
            if noise_sd > 0:
                y = y + rng.normal(0.0, noise_sd, size=y.size)
            c.y = y
            out.append(c)
    return out


def error_scenarios(
    noise_sd: float = 0.05,
    seed: int = 7,
    de0_mv: float = 1.0,
    titrant_error_pct: float = -2.0,
    acid_error_pct: float = -2.0,
    activity_model: str = "dh",
) -> dict[str, np.ndarray[Any, Any]]:
    """Change of the stepwise log K_n for the four perturbations of Sec. 2.6.

    All fits are weighted (s_E = 0.15 mV, s_V = 0.005 mL) and start from
    ``logK0 - 0.3``.  Returns differences ``perturbed - reference``.

    NOTE on sign: this function perturbs E0' by +``de0_mv`` and obtains
    *negative* differences.  Table 1/3 of the paper lists the 406 mV case with
    larger log K, Fig. 2 plots reference minus 406 mV; the magnitudes agree
    with ours, the paper's sign convention is ambiguous (see README).

    Returns
    -------
    dict
        keys 'wrong_formal_potential', 'wrong_titrant', 'wrong_acid',
        'ionic_strength'.

    [paper sections 2.6 and 3.3]
    """
    curves = benchmark_curves(noise_sd, seed)
    model = ProtonationModel(6, -6, 6, -1, "const", "dh", 10.0**-PAPER_PKW_I01, 0.1)
    start = stepwise_to_cumulative([v - 0.3 for v in PAPER_LOG_K])
    kw = dict(weighted=True, s_y=0.15, s_v=0.005, obs="E", max_iter=150)
    ref = fit_protonation(curves, model, start, PAPER_CALIB, 0.1, **kw)  # type: ignore[arg-type]
    out: dict[str, np.ndarray[Any, Any]] = {}
    f = fit_protonation(curves, model, start, Calibration(405.0 + de0_mv, -59.16, -64.0, 0.0), 0.1, **kw)  # type: ignore[arg-type]
    out["wrong_formal_potential"] = f.log_k - ref.log_k
    f = fit_protonation(curves, model, start, PAPER_CALIB, 0.1 * (1.0 + titrant_error_pct / 100.0), **kw)  # type: ignore[arg-type]
    out["wrong_titrant"] = f.log_k - ref.log_k
    wrong = [FitCurve(c.name, c.v0_ml, c.cl_mol_l * (1.0 + acid_error_pct / 100.0), c.v_ml, c.y, c.h0_mol_l, c.f_h) for c in curves]
    f = fit_protonation(wrong, model, start, PAPER_CALIB, 0.1, refine=("fh",), **kw)  # type: ignore[arg-type]
    out["wrong_acid"] = f.log_k - ref.log_k
    var_model = ProtonationModel(6, -6, 6, -1, "var", activity_model, 10.0**-13.997, 0.1)
    lbt = to_thermodynamic(ref.log_beta, -6, 0.1, activity_model)
    f = fit_protonation(curves, var_model, lbt, Calibration(410.0, -59.16, -64.0, 0.0), 0.1, refine=("e0",), **kw)  # type: ignore[arg-type]
    stoich = to_stoichiometric(f.log_beta, -6, 0.1, activity_model)
    out["ionic_strength"] = np.asarray(cumulative_to_stepwise(stoich)) - ref.log_k
    return out
