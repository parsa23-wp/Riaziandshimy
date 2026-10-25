"""Acid-base indicator model (second answer, section 4).

An indicator is a weak acid ``HIn <=> H+ + In-`` with constant ``K_In``:

    [In-]/[HIn] = 10**(pH - pK_In)

Default endpoint model ("visibility fraction"): the endpoint is seen when the
fraction of the *new* colour form reaches ``alpha_visible`` (0.5 gives
pH = pK_In).  The textbook rule "pK_In +- 1" is the special case where the
ratio 10:1 / 1:10 is perceptible.  An absorbance-threshold model is also
available (section 4) but only if the user supplies molar absorptivities,
because the source documents contain no reliable values.

Reference: [MATH.md section 4]
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .acid_base import AcidBaseModel, equivalence_volumes, ph_at
from .equilibrium import brent

_DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Indicator:
    """One indicator.

    Attributes
    ----------
    name : str
    pk_in : float
    lo, hi : float
        Textbook colour-transition range.
    acid_color, base_color : str
    fade_ph : float | None
        pH above which the base colour disappears again (phenolphthalein
        becomes colourless above ~13, section 4).

    Example
    -------
    >>> Indicator("X", 7.0, 6.0, 8.0, "a", "b").pk_in
    7.0
    """

    name: str
    pk_in: float
    lo: float
    hi: float
    acid_color: str
    base_color: str
    fade_ph: float | None = None


def load_indicators() -> list[Indicator]:
    """Load the indicator table from ``data/indicators.json``.

    Returns
    -------
    list[Indicator]
        At least ten indicators.

    Example
    -------
    >>> len(load_indicators()) >= 6
    True
    """
    with open(_DATA / "indicators.json", encoding="utf-8") as fh:
        raw = json.load(fh)["indicators"]
    return [
        Indicator(
            r["name"], float(r["pk_in"]), float(r["lo"]), float(r["hi"]),
            r["acid_color"], r["base_color"], r.get("fade_ph"),
        )
        for r in raw
    ]


def base_fraction(ind: Indicator, ph: float) -> float:
    """Fraction of the indicator in the basic form In-.

    Example
    -------
    >>> round(base_fraction(Indicator("X", 7.0, 6.0, 8.0, "a", "b"), 7.0), 3)
    0.5
    """
    return float(1.0 / (1.0 + 10.0 ** (ind.pk_in - ph)))


def base_color_visible(ind: Indicator, ph: float) -> bool:
    """False when the base colour has faded (pH >= ``fade_ph``).

    Example
    -------
    >>> pp = [i for i in load_indicators() if i.name == "Phenolphthalein"][0]
    >>> base_color_visible(pp, 14.0)
    False
    """
    return ind.fade_ph is None or ph < ind.fade_ph


def endpoint_ph(ind: Indicator, rising: bool = True, alpha_visible: float = 0.5) -> float:
    """pH at which the endpoint colour change is perceived.

    Parameters
    ----------
    ind : Indicator
    rising : bool
        True if pH rises during the titration (acid analyte).
    alpha_visible : float
        Fraction (0 < alpha < 1) of the new colour form needed for detection.

    Raises
    ------
    ValueError
        If ``alpha_visible`` is not in (0, 1).

    Example
    -------
    >>> ind = Indicator("X", 7.0, 6.0, 8.0, "a", "b")
    >>> endpoint_ph(ind, True, 0.5)
    7.0
    """
    if not 0.0 < alpha_visible < 1.0:
        raise ValueError("alpha_visible must be in (0, 1)")
    shift = math.log10(alpha_visible / (1.0 - alpha_visible))
    return ind.pk_in + shift if rising else ind.pk_in - shift


def absorbance_endpoint_ph(
    ind: Indicator,
    c_ind: float,
    eps_acid: float,
    eps_base: float,
    path_cm: float = 1.0,
    threshold: float = 0.02,
    rising: bool = True,
) -> float | None:
    """Endpoint pH from an absorbance-change threshold.

    ``A = b (eps_HIn [HIn] + eps_In [In-])``; the change is noticed when
    ``|dA| > threshold`` (section 4, about 0.02 AU).  Absorbance is that of one
    wavelength band, so the result is only as good as the supplied eps values.

    Returns
    -------
    float | None
        pH, or ``None`` if the threshold is never reached (too little
        indicator or too similar eps).

    Raises
    ------
    ValueError
        If concentration or path are not positive.

    Example
    -------
    >>> ind = Indicator("X", 7.0, 6.0, 8.0, "a", "b")
    >>> endpoint_ph(ind) == absorbance_endpoint_ph(ind, 1e-5, 0.0, 4e4, threshold=0.2)
    True
    """
    if c_ind <= 0 or path_cm <= 0:
        raise ValueError("c_ind and path must be positive")
    d_eps = abs(eps_base - eps_acid)
    full_change = c_ind * path_cm * d_eps
    if full_change <= threshold:
        return None
    alpha = threshold / full_change
    shift = math.log10(alpha / (1.0 - alpha))
    return ind.pk_in + shift if rising else ind.pk_in - shift


def endpoint_volume(
    model: AcidBaseModel,
    ind: Indicator,
    k: int = 1,
    alpha_visible: float = 0.5,
) -> float | None:
    """Titrant volume (mL) at which the indicator changes colour near V_k.

    The pH curve is scanned between the neighbouring equivalence points for
    crossings of the endpoint pH; the crossing nearest to ``V_k`` is returned.

    Returns
    -------
    float | None
        mL, or ``None`` if the indicator never reaches its endpoint pH there.

    Raises
    ------
    ValueError
        If ``k`` is not a valid equivalence index.

    Example
    -------
    >>> pp = [i for i in load_indicators() if i.name == "Phenolphthalein"][0]
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> abs(endpoint_volume(m, pp) - 25.0) < 0.05
    True

    [MATH.md section 4]
    """
    eqs = equivalence_volumes(model)
    if not 1 <= k <= len(eqs):
        raise ValueError("invalid equivalence index")
    rising = model.analyte_is_acid
    target = endpoint_ph(ind, rising, alpha_visible)
    ve = eqs[k - 1]
    lo = eqs[k - 2] if k >= 2 else 0.0
    hi = eqs[k] if k < len(eqs) else ve * 1.5
    n = 800
    grid = [lo + (hi - lo) * i / n for i in range(n + 1)]
    vals = [ph_at(model, v) - target for v in grid]
    best: float | None = None
    for i in range(n):
        if vals[i] == 0.0:
            cand = grid[i]
        elif vals[i] * vals[i + 1] < 0.0:
            cand = brent(lambda x: ph_at(model, x) - target, grid[i], grid[i + 1], 1e-12)
        else:
            continue
        if best is None or abs(cand - ve) < abs(best - ve):
            best = cand
    return best


def indicator_error_percent(
    model: AcidBaseModel, ind: Indicator, k: int = 1, alpha_visible: float = 0.5
) -> float | None:
    """Signed indicator error (V_EP - V_EQ) / V_EQ * 100.

    The absolute value is the textbook "Error %" of prompt-1 section 41.

    Example
    -------
    >>> pp = [i for i in load_indicators() if i.name == "Phenolphthalein"][0]
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> abs(indicator_error_percent(m, pp)) < 0.1
    True
    """
    ve = equivalence_volumes(model)[k - 1]
    vep = endpoint_volume(model, ind, k, alpha_visible)
    if vep is None:
        return None
    return (vep - ve) / ve * 100.0


def recommend_indicators(
    model: AcidBaseModel, k: int = 1, alpha_visible: float = 0.5
) -> list[tuple[Indicator, float | None, bool]]:
    """Rank all indicators for equivalence point ``k``.

    Returns
    -------
    list of (indicator, signed error %, suitable)
        Sorted by absolute error; ``suitable`` means the endpoint pH lies
        between pH(V_k - 0.1 %) and pH(V_k + 0.1 %) (so the error is < 0.1 %).

    Example
    -------
    >>> m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    >>> rec = recommend_indicators(m)
    >>> rec[0][2]
    True
    """
    ve = equivalence_volumes(model)[k - 1]
    p_lo = ph_at(model, ve * 0.999)
    p_hi = ph_at(model, ve * 1.001)
    lo_b, hi_b = min(p_lo, p_hi), max(p_lo, p_hi)
    rising = model.analyte_is_acid
    out: list[tuple[Indicator, float | None, bool]] = []
    for ind in load_indicators():
        err = indicator_error_percent(model, ind, k, alpha_visible)
        eph = endpoint_ph(ind, rising, alpha_visible)
        suitable = lo_b <= eph <= hi_b and base_color_visible(ind, hi_b)
        out.append((ind, err, suitable))
    out.sort(key=lambda t: (abs(t[1]) if t[1] is not None else float("inf")))
    return out
