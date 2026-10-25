"""Error models: uncertainty propagation and simulated laboratory errors.

Section 10.6/10.7 (propagation, burette uncertainty) and section 11
(systematic, random, gross errors) of the "second answer".  Every error is a
toggle of :class:`ErrorConfig`; :func:`sensitivity_table` answers the
question "if the burette is not calibrated, how much does the result
change?".

Reference: [MATH.md sections 10.6, 10.7, 11]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .acid_base import AcidBaseModel, equivalence_volumes
from .analysis import ConfidenceInterval, OutlierResult, confidence_interval, grubbs_test, mean, rsd_percent, sd
from .indicators import Indicator, endpoint_volume


def burette_uncertainty(u_read: float = 0.01, u_cal: float = 0.02) -> float:
    """u_V = sqrt(2 u_read^2 + u_cal^2)  (two readings + calibration).

    Example
    -------
    >>> round(burette_uncertainty(0.01, 0.02), 4)
    0.0245

    [MATH.md section 10.7]
    """
    return math.sqrt(2.0 * u_read**2 + u_cal**2)


def propagate_sum(*u: float) -> float:
    """u_z for z = x + y (or x - y): sqrt(sum u_i^2).

    Example
    -------
    >>> propagate_sum(3.0, 4.0)
    5.0
    """
    return math.sqrt(sum(v * v for v in u))


def propagate_product(values: Sequence[float] | np.ndarray[Any, Any], uncertainties: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Relative uncertainty u_z/|z| for a product/quotient of independent factors.

    Raises
    ------
    ValueError
        On length mismatch or a zero value.

    Example
    -------
    >>> round(propagate_product([10.0, 20.0], [0.1, 0.4]), 6)
    0.022361
    """
    if len(values) != len(uncertainties) or any(v == 0 for v in values):
        raise ValueError("invalid inputs")
    return math.sqrt(sum((u / v) ** 2 for v, u in zip(values, uncertainties)))


def propagate_power(x: float, u_x: float, a: float) -> float:
    """Relative uncertainty of z = x**a: |a| u_x / |x|.

    Example
    -------
    >>> propagate_power(4.0, 0.2, 2.0)
    0.1
    """
    return abs(a) * u_x / abs(x)


def propagate_log10(x: float, u_x: float) -> float:
    """Absolute uncertainty of z = log10(x): u_x / (x ln 10).

    Example
    -------
    >>> round(propagate_log10(100.0, 1.0), 6)
    0.004343
    """
    return u_x / (x * math.log(10.0))


def titration_result_uncertainty(
    c_titrant: float,
    u_c_titrant: float,
    v_titrant: float,
    u_v_titrant: float,
    v_analyte: float,
    u_v_analyte: float,
    a: float = 1.0,
    b: float = 1.0,
) -> tuple[float, float]:
    """Analyte concentration and combined standard uncertainty.

    ``C_A = (a/b) C_B V_B / V_A`` and
    ``(u_C/C)^2 = (u_CB/CB)^2 + (u_VB/VB)^2 + (u_VA/VA)^2``; the
    stoichiometric coefficients are taken as exact.

    Example
    -------
    >>> c, u = titration_result_uncertainty(0.1, 0.0002, 25.0, 0.0245, 25.0, 0.03)
    >>> round(c, 4), round(u, 6)
    (0.1, 0.000253)

    [MATH.md section 10.6]
    """
    c = a / b * c_titrant * v_titrant / v_analyte
    rel = propagate_product([c_titrant, v_titrant, v_analyte], [u_c_titrant, u_v_titrant, u_v_analyte])
    return c, c * rel


class GrossError(str, Enum):
    """Gross-error scenarios ("student error simulation", section 11.3)."""

    NONE = "none"
    MISREAD_BURETTE = "misread_burette"
    WRONG_CONCENTRATION_ENTRY = "wrong_concentration_entry"
    SPILLED_SAMPLE = "spilled_sample"
    WRONG_INDICATOR = "wrong_indicator"


@dataclass(frozen=True)
class ErrorConfig:
    """Switches and sizes of all simulated errors (all default to "off").

    Attributes
    ----------
    burette_cal_fraction : float
        Burette delivers ``V_read = V_true (1 + f)``.
    titrant_factor : float
        Actual / nominal titrant concentration (F, section 11.1).
    co2_titrant_loss_fraction : float
        Relative loss of effective base titrant concentration through CO2
        uptake, ``C(t) = C0 (1 - fraction)``.
        APPROXIMATION: the effect of carbonate on the end point shape is not
        modelled (Berto 2024 also leaves it out), only the titer loss.
    blank_ml : float
        Extra titrant consumed by reagents; subtracted only if
        ``blank_corrected``.
    evaporation_ml : float
        Solvent lost before titration (moles conserved, concentration up).
    overshoot_ml : float
        Titrant added beyond the true end point (e.g. one drop = 0.05 mL).
    bubble_ml : float
        Air bubble at the burette tip that escapes during the run: it is
        counted in the reading but delivers no titrant, so the *read* volume
        is biased high by this fixed amount (sections 66-67 of the brief).
    sigma_parallax_ml : float
        Random meniscus-angle (parallax) reading error, independent for the
        initial and final readings, so the net effect carries a ``sqrt(2)``
        factor like ``sigma_read_ml`` (section 68 of the brief).
    sigma_read_ml, sigma_drop_ml, sigma_pipette_ml : float
        Random errors (meniscus reading, extra drop, pipette), normal.
    pipette_bias_ml : float
        Systematic pipette volume error.
    gross : GrossError
    gross_magnitude : float
        MISREAD: mL added to the reading; WRONG_CONCENTRATION_ENTRY:
        multiplier of the entered titrant concentration; SPILLED_SAMPLE:
        fraction of sample lost.
    wrong_indicator : Indicator | None
        Used by ``GrossError.WRONG_INDICATOR``.

    Example
    -------
    >>> ErrorConfig().titrant_factor
    1.0
    """

    burette_cal_fraction: float = 0.0
    titrant_factor: float = 1.0
    co2_titrant_loss_fraction: float = 0.0
    blank_ml: float = 0.0
    blank_corrected: bool = False
    evaporation_ml: float = 0.0
    overshoot_ml: float = 0.0
    bubble_ml: float = 0.0
    sigma_read_ml: float = 0.0
    sigma_parallax_ml: float = 0.0
    sigma_drop_ml: float = 0.0
    sigma_pipette_ml: float = 0.0
    pipette_bias_ml: float = 0.0
    gross: GrossError = GrossError.NONE
    gross_magnitude: float = 0.0
    wrong_indicator: Indicator | None = None


@dataclass(frozen=True)
class ReplicateResult:
    """One simulated replicate titration.

    Attributes
    ----------
    c_calc : float
        Analyte concentration computed by the analyst (mol/L).
    v_read_ml : float
        Corrected burette volume used in the calculation.
    v_end_ml : float
        True delivered volume at the detected end point.
    """

    c_calc: float
    v_read_ml: float
    v_end_ml: float


def simulate_replicate(
    model: AcidBaseModel,
    indicator: Indicator | None,
    config: ErrorConfig,
    rng: np.random.Generator,
    k: int = 1,
    alpha_visible: float = 0.5,
) -> ReplicateResult:
    """Simulate one titration with the errors of ``config``.

    Without an indicator the exact equivalence volume of the *actual*
    chemistry is used (ideal detection); with an indicator the end point is
    where the colour changes, so the indicator error is included.

    Raises
    ------
    ValueError
        If the indicator never changes colour near V_k.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> r = simulate_replicate(m, None, ErrorConfig(), np.random.default_rng(0))
    >>> round(r.c_calc, 6)
    0.1

    [MATH.md section 11]
    """
    cfg = config
    eff_ind = indicator
    if cfg.gross is GrossError.WRONG_INDICATOR and cfg.wrong_indicator is not None:
        eff_ind = cfg.wrong_indicator
    v_pip = model.v_analyte_ml + cfg.pipette_bias_ml
    if cfg.sigma_pipette_ml > 0:
        v_pip += float(rng.normal(0.0, cfg.sigma_pipette_ml))
    spill = cfg.gross_magnitude if cfg.gross is GrossError.SPILLED_SAMPLE else 0.0
    n_mol = model.c_analyte * v_pip * (1.0 - spill)
    v_act = v_pip - cfg.evaporation_ml
    if v_act <= 0:
        raise ValueError("evaporation larger than sample volume")
    c_act = n_mol / v_act
    ct_act = model.c_titrant * cfg.titrant_factor * (1.0 - cfg.co2_titrant_loss_fraction)
    actual = replace(model, c_analyte=c_act, v_analyte_ml=v_act, c_titrant=ct_act)
    if eff_ind is None:
        v_end = equivalence_volumes(actual)[k - 1]
    else:
        v_ep = endpoint_volume(actual, eff_ind, k, alpha_visible)
        if v_ep is None:
            raise ValueError("indicator does not change colour near the equivalence point")
        v_end = v_ep
    v_del = v_end + cfg.overshoot_ml + cfg.blank_ml + cfg.bubble_ml
    if cfg.sigma_drop_ml > 0:
        v_del += float(rng.normal(0.0, cfg.sigma_drop_ml))
    f = 1.0 + cfg.burette_cal_fraction
    v_read = v_del * f
    if cfg.sigma_read_ml > 0:
        v_read += math.sqrt(2.0) * float(rng.normal(0.0, cfg.sigma_read_ml))
    if cfg.sigma_parallax_ml > 0:
        v_read += math.sqrt(2.0) * float(rng.normal(0.0, cfg.sigma_parallax_ml))
    if cfg.blank_corrected:
        v_read -= cfg.blank_ml * f
    if cfg.gross is GrossError.MISREAD_BURETTE:
        v_read += cfg.gross_magnitude
    ct_entry = model.c_titrant * (cfg.gross_magnitude if cfg.gross is GrossError.WRONG_CONCENTRATION_ENTRY else 1.0)
    c_calc = ct_entry * v_read / (k * model.v_analyte_ml)
    return ReplicateResult(c_calc, v_read, v_end)


@dataclass(frozen=True)
class ReplicateSummary:
    """Statistics of a set of replicate results.

    Attributes
    ----------
    values : tuple[float, ...]
    mean, sd, rsd_percent : float | None
        ``sd``/``rsd_percent`` are None for a single replicate.
    true_value : float
    bias_percent : float
        (mean - true) / true * 100.
    recovery_percent : float
    ci : ConfidenceInterval | None
    outlier : OutlierResult | None
        Grubbs test (n >= 3, non-zero spread).
    """

    values: tuple[float, ...]
    mean: float
    sd: float | None
    rsd_percent: float | None
    true_value: float
    bias_percent: float
    recovery_percent: float
    ci: ConfidenceInterval | None
    outlier: OutlierResult | None


def run_replicates(
    model: AcidBaseModel,
    indicator: Indicator | None,
    config: ErrorConfig,
    n: int = 5,
    seed: int = 1,
    k: int = 1,
    alpha_visible: float = 0.5,
) -> ReplicateSummary:
    """Run ``n`` replicate titrations and summarise them.

    Raises
    ------
    ValueError
        If ``n < 1``.

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> s = run_replicates(m, None, ErrorConfig(sigma_read_ml=0.02), n=6, seed=3)
    >>> abs(s.bias_percent) < 1.0
    True

    [MATH.md sections 10, 11]
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    vals = [simulate_replicate(model, indicator, config, rng, k, alpha_visible).c_calc for n_ in range(n)]
    true = model.c_analyte * 1.0
    m = mean(vals)
    s = sd(vals) if n >= 2 else None
    rsd = rsd_percent(vals) if n >= 2 and m != 0 else None
    ci = confidence_interval(vals) if n >= 2 else None
    out: OutlierResult | None = None
    if n >= 3 and s is not None and s > 0:
        out = grubbs_test(vals)
    return ReplicateSummary(
        tuple(vals), m, s, rsd, true, (m - true) / true * 100.0, m / true * 100.0, ci, out
    )


def default_scenarios() -> dict[str, ErrorConfig]:
    """Ready-made single-error scenarios for the sensitivity table.

    Example
    -------
    >>> "Burette +0.5 %" in default_scenarios()
    True
    """
    return {
        "Burette +0.5 %": ErrorConfig(burette_cal_fraction=0.005),
        "Titrant factor 0.98": ErrorConfig(titrant_factor=0.98),
        "CO2 uptake 1 % titer loss": ErrorConfig(co2_titrant_loss_fraction=0.01),
        "Blank 0.10 mL (uncorrected)": ErrorConfig(blank_ml=0.10),
        "Overshoot 1 drop (0.05 mL)": ErrorConfig(overshoot_ml=0.05),
        "Air bubble 0.05 mL": ErrorConfig(bubble_ml=0.05),
        "Parallax sigma 0.02 mL": ErrorConfig(sigma_parallax_ml=0.02),
        "Evaporation 0.5 mL": ErrorConfig(evaporation_ml=0.5),
        "Pipette bias +0.03 mL": ErrorConfig(pipette_bias_ml=0.03),
        "Gross: burette misread by 1.00 mL": ErrorConfig(gross=GrossError.MISREAD_BURETTE, gross_magnitude=1.0),
        "Gross: wrong titrant entered (x1.1)": ErrorConfig(gross=GrossError.WRONG_CONCENTRATION_ENTRY, gross_magnitude=1.1),
        "Gross: 5 % of sample spilled": ErrorConfig(gross=GrossError.SPILLED_SAMPLE, gross_magnitude=0.05),
    }


def sensitivity_table(
    model: AcidBaseModel,
    indicator: Indicator | None,
    scenarios: Mapping[str, ErrorConfig] | None = None,
    k: int = 1,
    alpha_visible: float = 0.5,
) -> list[tuple[str, float, float]]:
    """Deterministic effect of each error alone on the computed concentration.

    Returns
    -------
    list of (name, computed concentration mol/L, shift % vs the error-free
    result of the same indicator)

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> tab = dict((n, s) for n, _, s in sensitivity_table(m, None))
    >>> round(tab["Burette +0.5 %"], 3)
    0.5
    """
    sc = dict(scenarios) if scenarios is not None else default_scenarios()
    rng = np.random.default_rng(0)
    base = simulate_replicate(model, indicator, ErrorConfig(), rng, k, alpha_visible).c_calc
    out: list[tuple[str, float, float]] = []
    for name, cfg in sc.items():
        c = simulate_replicate(model, indicator, cfg, rng, k, alpha_visible).c_calc
        out.append((name, c, (c - base) / base * 100.0))
    return out
