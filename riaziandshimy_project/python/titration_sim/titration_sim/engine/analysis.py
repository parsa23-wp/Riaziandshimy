"""Curve analysis and analytical statistics.

* derivatives on *non-uniform* grids (section 5.1 gives only the uniform
  formula; adaptive curves of section 13.4 are non-uniform, see the
  contradiction report),
* Savitzky-Golay type smoothing (local polynomial regression),
* equivalence-point detection by first derivative, second derivative and Gran
  plot, combined with weights (section 5.4),
* mean, SD, RSD, SE, Student-t confidence interval, Dixon Q and Grubbs tests
  (section 10).

Reference: [MATH.md sections 5 and 10]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------
# derivatives
# --------------------------------------------------------------------------


def first_derivative(x: Sequence[float] | np.ndarray[Any, Any], y: Sequence[float] | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """dy/dx on a possibly non-uniform grid (3-point formula).

    Interior: ``(h1^2 y[i+1] - h2^2 y[i-1] + (h2^2 - h1^2) y[i]) / (h1 h2 (h1+h2))``
    with ``h1 = x[i]-x[i-1]``, ``h2 = x[i+1]-x[i]``; on a uniform grid this is
    the central difference ``(y[i+1]-y[i-1])/(2 dV)`` of section 5.1.  The
    end points use one-sided differences.

    Raises
    ------
    ValueError
        If fewer than 3 points or ``x`` is not strictly increasing.

    Example
    -------
    >>> d = first_derivative([0, 1, 2, 3], [0, 1, 4, 9])
    >>> [round(float(v), 6) for v in d[1:3]]
    [2.0, 4.0]

    [MATH.md section 5.1]
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 3 or xa.size != ya.size:
        raise ValueError("need at least 3 points of equal length")
    if np.any(np.diff(xa) <= 0):
        raise ValueError("x must be strictly increasing")
    d = np.empty_like(ya)
    h1 = xa[1:-1] - xa[:-2]
    h2 = xa[2:] - xa[1:-1]
    d[1:-1] = (h1**2 * ya[2:] - h2**2 * ya[:-2] + (h2**2 - h1**2) * ya[1:-1]) / (h1 * h2 * (h1 + h2))
    d[0] = (ya[1] - ya[0]) / (xa[1] - xa[0])
    d[-1] = (ya[-1] - ya[-2]) / (xa[-1] - xa[-2])
    return d


def second_derivative(x: Sequence[float] | np.ndarray[Any, Any], y: Sequence[float] | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """d2y/dx2 on a non-uniform grid.

    ``2 [ (y[i+1]-y[i])/h2 - (y[i]-y[i-1])/h1 ] / (h1+h2)``; equals
    ``(y[i+1]-2y[i]+y[i-1])/dV^2`` on a uniform grid (section 5.2).  End
    values are copied from their neighbours.

    Raises
    ------
    ValueError
        As :func:`first_derivative`.

    Example
    -------
    >>> d2 = second_derivative([0, 1, 2, 3], [0, 1, 4, 9])
    >>> round(float(d2[1]), 6)
    2.0

    [MATH.md section 5.2]
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 3 or xa.size != ya.size:
        raise ValueError("need at least 3 points of equal length")
    if np.any(np.diff(xa) <= 0):
        raise ValueError("x must be strictly increasing")
    h1 = xa[1:-1] - xa[:-2]
    h2 = xa[2:] - xa[1:-1]
    d2 = np.empty_like(ya)
    d2[1:-1] = 2.0 * ((ya[2:] - ya[1:-1]) / h2 - (ya[1:-1] - ya[:-2]) / h1) / (h1 + h2)
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    return d2


def savgol_smooth(
    x: Sequence[float] | np.ndarray[Any, Any], y: Sequence[float] | np.ndarray[Any, Any], window: int = 7, polyorder: int = 2
) -> np.ndarray[Any, Any]:
    """Local polynomial (Savitzky-Golay type) smoothing for non-uniform x.

    Each point is replaced by the value at ``x[i]`` of a polynomial fitted by
    least squares to the ``window`` nearest-index points (shifted at the
    ends).

    Raises
    ------
    ValueError
        If ``window`` is even, smaller than ``polyorder + 2`` or larger than
        the data length.

    Example
    -------
    >>> out = savgol_smooth(list(range(10)), [2.0 * i for i in range(10)], 5, 2)
    >>> bool(np.allclose(out, [2.0 * i for i in range(10)]))
    True

    [MATH.md section 5.1 (noise warning), 5.4]
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = xa.size
    if window % 2 == 0 or window < polyorder + 2 or window > n:
        raise ValueError("window must be odd, >= polyorder+2 and <= len(data)")
    half = window // 2
    out = np.empty_like(ya)
    for i in range(n):
        a = min(max(0, i - half), n - window)
        sl = slice(a, a + window)
        xs = xa[sl] - xa[i]
        coef = np.polyfit(xs, ya[sl], polyorder)
        out[i] = coef[-1]
    return out


def local_slope_poly(x: Sequence[float] | np.ndarray[Any, Any], y: Sequence[float] | np.ndarray[Any, Any], window: int = 5, polyorder: int = 3) -> np.ndarray[Any, Any]:
    """First derivative from a local cubic over 5 points.

    This is the 5-point cubic first-derivative filter used by SUPERQUAD and
    HYPERQUAD for the weights (Berto 2024, section 2.2) generalised to
    non-uniform volumes.

    Example
    -------
    >>> d = local_slope_poly(list(range(8)), [3.0 * i for i in range(8)])
    >>> bool(np.allclose(d, 3.0))
    True
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = xa.size
    if n < window:
        raise ValueError("not enough points")
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        a = min(max(0, i - half), n - window)
        sl = slice(a, a + window)
        coef = np.polyfit(xa[sl] - xa[i], ya[sl], polyorder)
        out[i] = coef[-2]
    return out


# --------------------------------------------------------------------------
# equivalence detection
# --------------------------------------------------------------------------


def _parabola_peak(x3: np.ndarray[Any, Any], y3: np.ndarray[Any, Any]) -> float:
    """Abscissa of the extremum of a parabola through 3 points."""
    c = np.polyfit(x3 - x3[1], y3, 2)
    if abs(c[0]) < 1e-300:
        return float(x3[1])
    xm = -c[1] / (2.0 * c[0])
    if abs(xm) > max(abs(x3[2] - x3[1]), abs(x3[1] - x3[0])):
        return float(x3[1])
    return float(x3[1] + xm)


def equivalence_first_derivative(
    v: Sequence[float] | np.ndarray[Any, Any],
    ph: Sequence[float] | np.ndarray[Any, Any],
    expected: float | None = None,
    smooth: bool = True,
) -> float:
    """Equivalence volume from the maximum of |dpH/dV| (section 5.1).

    The maximum is refined with a parabola through three points.  If
    ``expected`` is given, only the neighbourhood (+-20 % of the range) of that
    volume is searched (useful for polyprotic curves).

    Raises
    ------
    ValueError
        If fewer than 7 points are supplied.

    Example
    -------
    >>> v = np.linspace(0, 20, 201); p = 7 + 3 * np.tanh(3 * (v - 10))
    >>> abs(equivalence_first_derivative(v, p) - 10.0) < 0.02
    True
    """
    va = np.asarray(v, dtype=float)
    pa = np.asarray(ph, dtype=float)
    if va.size < 7:
        raise ValueError("need at least 7 points")
    ps = savgol_smooth(va, pa, 7, 2) if smooth else pa
    d1 = np.abs(first_derivative(va, ps))
    mask = np.ones(va.size, dtype=bool)
    if expected is not None:
        span = 0.2 * (va[-1] - va[0])
        mask = np.abs(va - expected) <= span
    idx = int(np.argmax(np.where(mask, d1, -np.inf)))
    idx = min(max(idx, 1), va.size - 2)
    return _parabola_peak(va[idx - 1 : idx + 2], d1[idx - 1 : idx + 2])


def equivalence_second_derivative(
    v: Sequence[float] | np.ndarray[Any, Any],
    ph: Sequence[float] | np.ndarray[Any, Any],
    expected: float | None = None,
    smooth: bool = True,
) -> float:
    """Equivalence volume from the zero crossing of d2pH/dV2 (section 5.2).

    The crossing nearest to the first-derivative peak (or ``expected``) is
    located by linear interpolation.

    Raises
    ------
    ValueError
        If no sign change of the second derivative exists.

    Example
    -------
    >>> v = np.linspace(0, 20, 401); p = 7 + 3 * np.tanh(3 * (v - 10))
    >>> abs(equivalence_second_derivative(v, p) - 10.0) < 0.02
    True
    """
    va = np.asarray(v, dtype=float)
    pa = np.asarray(ph, dtype=float)
    ps = savgol_smooth(va, pa, 7, 2) if smooth else pa
    d1 = first_derivative(va, ps)
    d2 = second_derivative(va, ps)
    ref = expected if expected is not None else equivalence_first_derivative(va, pa, smooth=smooth)
    best: float | None = None
    for i in range(va.size - 1):
        if d2[i] == 0.0:
            cand = float(va[i])
        elif d2[i] * d2[i + 1] < 0.0:
            cand = float(va[i] - d2[i] * (va[i + 1] - va[i]) / (d2[i + 1] - d2[i]))
        else:
            continue
        if best is None or abs(cand - ref) < abs(best - ref):
            best = cand
    if best is None:
        raise ValueError("no sign change in the second derivative")
    _ = d1  # first derivative kept for symmetry/debugging
    return best


@dataclass(frozen=True)
class GranResult:
    """Result of a Gran-plot extrapolation.

    Attributes
    ----------
    v_eq : float
        Equivalence volume (mL) = -intercept/slope.
    slope, intercept : float
    constant : float | None
        Ka (weak acid) or Kb (weak base) = -slope, else None.
    n_points : int
    """

    v_eq: float
    slope: float
    intercept: float
    constant: float | None
    n_points: int


def gran_plot(
    v: Sequence[float] | np.ndarray[Any, Any],
    ph: Sequence[float] | np.ndarray[Any, Any],
    v_analyte_ml: float,
    kind: str,
    v_eq_guess: float,
    kw: float = 1e-14,
    lo: float = 0.2,
    hi: float = 0.9,
) -> GranResult:
    """Gran extrapolation using points before the equivalence point.

    ``kind``:

    * ``'strong_acid'``:  F = (V0 + V) [H+]       = Ca V0 - Ct V      (slope -Ct)
    * ``'strong_base'``:  F = (V0 + V) [OH-]      = Cb V0 - Ct V
    * ``'weak_acid'``:    F = V [H+]              = Ka (Veq - V)      (slope -Ka)
    * ``'weak_base'``:    F = V [OH-]             = Kb (Veq - V)      (slope -Kb)

    CORRECTION of the "second answer" section 5.3: its weak-acid expression
    carries an extra, undefined factor ``[HA]/[A-]``.  Since
    ``[HA]/[A-] = (Veq - V)/V`` the volumes cancel and the Gran function is
    exactly ``V [H+] = Ka (Veq - V)`` (with ``V`` the titrant volume).

    APPROXIMATION: weak-acid/base forms neglect [H+] and [OH-] against the
    buffer components (valid for the 20-90 % region when Ka C >> Kw and
    pH within pKa +- 2); concentration scale, activity coefficients set to 1.

    Parameters
    ----------
    v, ph : sequences
        Titrant volume (mL) and pH.
    v_analyte_ml : float
        Initial analyte volume (mL).
    kind : str
    v_eq_guess : float
        Rough V_eq (mL) fixing the 20-90 % window.
    kw : float
        Ion product of water (used for base forms).
    lo, hi : float
        Fraction window of ``v_eq_guess``.

    Returns
    -------
    GranResult

    Raises
    ------
    ValueError
        For an unknown ``kind`` or fewer than 3 points in the window.

    Example
    -------
    >>> v = np.linspace(1, 20, 40); h = (0.1 * 25 - 0.1 * v) / (25 + v)
    >>> g = gran_plot(v, -np.log10(h), 25.0, 'strong_acid', 25.0)
    >>> abs(g.v_eq - 25.0) < 1e-6
    True

    [MATH.md section 5.3]
    """
    va = np.asarray(v, dtype=float)
    pa = np.asarray(ph, dtype=float)
    mask = (va >= lo * v_eq_guess) & (va <= hi * v_eq_guess)
    if mask.sum() < 3:
        raise ValueError("fewer than 3 points in the Gran window")
    vv = va[mask]
    h = 10.0 ** (-pa[mask])
    if kind == "strong_acid":
        f = (v_analyte_ml + vv) * h
    elif kind == "strong_base":
        f = (v_analyte_ml + vv) * kw / h
    elif kind == "weak_acid":
        f = vv * h
    elif kind == "weak_base":
        f = vv * kw / h
    else:
        raise ValueError(f"unknown Gran kind: {kind}")
    slope, intercept = np.polyfit(vv, f, 1)
    const = -float(slope) if kind in ("weak_acid", "weak_base") else None
    return GranResult(float(-intercept / slope), float(slope), float(intercept), const, int(vv.size))


def combine_equivalence(
    estimates: Sequence[float] | np.ndarray[Any, Any], weights: Sequence[float] | np.ndarray[Any, Any] | None = None
) -> float:
    """Weighted average of several V_eq estimates (section 5.4 step 5).

    Raises
    ------
    ValueError
        On empty input, length mismatch or non-positive weight sum.

    Example
    -------
    >>> combine_equivalence([10.0, 10.2], [1, 1])
    10.1
    """
    if len(estimates) == 0:
        raise ValueError("no estimates")
    w = np.ones(len(estimates)) if weights is None else np.asarray(weights, dtype=float)
    if w.size != len(estimates) or w.sum() <= 0:
        raise ValueError("invalid weights")
    return float(np.dot(w, np.asarray(estimates, dtype=float)) / w.sum())


def find_equivalence(
    v: Sequence[float] | np.ndarray[Any, Any],
    ph: Sequence[float] | np.ndarray[Any, Any],
    expected: float | None = None,
    gran_kind: str | None = None,
    v_analyte_ml: float | None = None,
    kw: float = 1e-14,
) -> float:
    """Combined equivalence detection (first + second derivative [+ Gran]).

    Weights: 0.4 / 0.4 / 0.2.  The second-derivative estimate is skipped if
    it fails; Gran is used only if ``gran_kind`` and ``v_analyte_ml`` are given.

    Example
    -------
    >>> v = np.linspace(0, 20, 401); p = 7 + 3 * np.tanh(3 * (v - 10))
    >>> abs(find_equivalence(v, p) - 10.0) < 0.02
    True

    [MATH.md section 5.4]
    """
    est: list[float] = []
    w: list[float] = []
    e1 = equivalence_first_derivative(v, ph, expected)
    est.append(e1)
    w.append(0.4)
    try:
        est.append(equivalence_second_derivative(v, ph, expected))
        w.append(0.4)
    except ValueError:
        pass
    if gran_kind is not None and v_analyte_ml is not None:
        try:
            est.append(gran_plot(v, ph, v_analyte_ml, gran_kind, e1, kw).v_eq)
            w.append(0.2)
        except ValueError:
            pass
    return combine_equivalence(est, w)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def mean(x: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Arithmetic mean.

    Example
    -------
    >>> mean([24.82, 24.79, 24.84])
    24.816666666666666
    """
    if len(x) == 0:
        raise ValueError("empty sample")
    return float(np.mean(np.asarray(x, dtype=float)))


def median(x: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Sample median (robust central value, section 94 of the brief).

    >>> median([24.82, 24.79, 24.84])
    24.82
    >>> median([1.0, 2.0, 3.0, 4.0])
    2.5
    """
    if len(x) == 0:
        raise ValueError("empty sample")
    return float(np.median(np.asarray(x, dtype=float)))


def sd(x: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Sample standard deviation (n - 1 in the denominator).

    Raises
    ------
    ValueError
        If fewer than 2 values.

    Example
    -------
    >>> round(sd([1.0, 2.0, 3.0]), 6)
    1.0
    """
    if len(x) < 2:
        raise ValueError("need at least 2 values")
    return float(np.std(np.asarray(x, dtype=float), ddof=1))


def rsd_percent(x: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """RSD % = s / mean * 100.

    Example
    -------
    >>> round(rsd_percent([10.0, 10.2, 9.8]), 3)
    2.0
    """
    m = mean(x)
    if m == 0:
        raise ValueError("mean is zero")
    return sd(x) / abs(m) * 100.0


def sem(x: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Standard error of the mean, s / sqrt(n).

    Example
    -------
    >>> round(sem([1.0, 2.0, 3.0, 4.0]), 6)
    0.645497
    """
    return sd(x) / math.sqrt(len(x))


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln_bt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln_bt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Student-t cumulative distribution function.

    Example
    -------
    >>> round(t_cdf(0.0, 5), 6)
    0.5
    """
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Inverse Student-t CDF by bisection.

    Raises
    ------
    ValueError
        If ``p`` is not in (0, 1) or ``df`` is not positive.

    Example
    -------
    >>> round(t_quantile(0.975, 4), 2)
    2.78
    """
    if not 0.0 < p < 1.0 or df <= 0:
        raise ValueError("p must be in (0,1) and df positive")
    if p == 0.5:
        return 0.0
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Mean with a two-sided Student-t confidence interval.

    Attributes
    ----------
    mean, half_width, lower, upper, level, t : float
    """

    mean: float
    half_width: float
    lower: float
    upper: float
    level: float
    t: float


def confidence_interval(x: Sequence[float] | np.ndarray[Any, Any], level: float = 0.95) -> ConfidenceInterval:
    """mu = mean +- t(alpha/2, n-1) s / sqrt(n)  (section 10.4).

    Raises
    ------
    ValueError
        If fewer than 2 values or ``level`` not in (0, 1).

    Example
    -------
    >>> ci = confidence_interval([10.0, 10.2, 9.9, 10.1, 10.0])
    >>> round(ci.t, 2)
    2.78
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 values")
    t = t_quantile(0.5 + level / 2.0, n - 1)
    m = mean(x)
    hw = t * sem(x)
    return ConfidenceInterval(m, hw, m - hw, m + hw, level, t)


# Dixon Q critical values (95 % confidence), literature table (Rorabacher 1991)
# for the r10 statistic.  Values taken from the standard table, NOT derived.
_Q95 = {3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568, 8: 0.526, 9: 0.493, 10: 0.466}


@dataclass(frozen=True)
class OutlierResult:
    """Outcome of an outlier test.

    Attributes
    ----------
    index : int | None
        Index of the suspect value.
    statistic, critical : float
    is_outlier : bool
    """

    index: int | None
    statistic: float
    critical: float
    is_outlier: bool


def dixon_q_test(x: Sequence[float] | np.ndarray[Any, Any]) -> OutlierResult:
    """Dixon Q test (95 %), n = 3..10; Q = |x_suspect - x_nearest| / (x_max - x_min).

    Raises
    ------
    ValueError
        If n is outside 3..10 or all values are equal.

    Example
    -------
    >>> dixon_q_test([10.0, 10.1, 10.05, 10.9]).is_outlier
    True

    [MATH.md section 10.5]
    """
    n = len(x)
    if n not in _Q95:
        raise ValueError("Dixon Q table covers n = 3..10")
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr)
    s = arr[order]
    rng = s[-1] - s[0]
    if rng == 0:
        raise ValueError("all values are equal")
    q_low = (s[1] - s[0]) / rng
    q_high = (s[-1] - s[-2]) / rng
    if q_high >= q_low:
        q, idx = q_high, int(order[-1])
    else:
        q, idx = q_low, int(order[0])
    crit = _Q95[n]
    return OutlierResult(idx, float(q), crit, bool(q > crit))


def grubbs_test(x: Sequence[float] | np.ndarray[Any, Any], alpha: float = 0.05) -> OutlierResult:
    """Two-sided Grubbs test for one outlier, n >= 3.

    ``G = |x_suspect - mean| / s``;
    ``G_crit = (n-1)/sqrt(n) * sqrt(t^2 / (n - 2 + t^2))`` with
    ``t = t_quantile(1 - alpha/(2n), n - 2)``.

    Raises
    ------
    ValueError
        If n < 3 or the SD is zero.

    Example
    -------
    >>> grubbs_test([10.0, 10.1, 10.05, 10.9, 10.02]).index
    3

    [MATH.md section 10.5]
    """
    n = len(x)
    if n < 3:
        raise ValueError("Grubbs needs n >= 3")
    arr = np.asarray(x, dtype=float)
    s = sd(arr)
    if s == 0:
        raise ValueError("zero standard deviation")
    dev = np.abs(arr - arr.mean())
    idx = int(np.argmax(dev))
    g = float(dev[idx] / s)
    t = t_quantile(1.0 - alpha / (2.0 * n), n - 2)
    crit = (n - 1) / math.sqrt(n) * math.sqrt(t * t / (n - 2 + t * t))
    return OutlierResult(idx, g, crit, bool(g > crit))
