"""Colour helpers for the flask: indicator colour as a function of pH.

Pure Python (no Qt) so that it can be unit-tested without a display.

Reference: [MATH.md section 4]  ([In-]/[HIn] = 10^(pH - pK_In)).
"""
from __future__ import annotations

from ..engine.indicators import Indicator, base_color_visible, base_fraction

WATER = "#d8edf4"
"""Colour of a clear aqueous solution (slightly blue-grey, still visibly 'empty')."""

NAMED: dict[str, str] = {
    "colorless": WATER,
    "colourless": WATER,
    "red": "#d62f2f",
    "yellow": "#f4d321",
    "orange": "#f08a1c",
    "pink": "#e6338f",
    "violet": "#7a2bb5",
    "blue": "#1f5bd1",
    "green": "#2f9e55",
}


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb``.

    Raises
    ------
    ValueError
        If the string is not a 6-digit hex colour.

    Example
    -------
    >>> hex_to_rgb("#ff8000")
    (255, 128, 0)
    """
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"not a #rrggbb colour: {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Format an RGB triple as ``#rrggbb``.

    Example
    -------
    >>> rgb_to_hex((255, 128, 0))
    '#ff8000'
    """
    r, g, b = (max(0, min(255, int(round(x)))) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def mix(c1: str, c2: str, t: float) -> str:
    """Linear mix ``(1 - t) c1 + t c2`` (t clipped to [0, 1]).

    Example
    -------
    >>> mix("#000000", "#ffffff", 0.5)
    '#808080'
    """
    t = max(0.0, min(1.0, t))
    a, b = hex_to_rgb(c1), hex_to_rgb(c2)
    return rgb_to_hex((int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t)), int(round(a[2] + (b[2] - a[2]) * t))))


def named_color(name: str) -> str:
    """Hex value of a colour name used in ``indicators.json``.

    Raises
    ------
    KeyError
        For an unknown colour name.

    Example
    -------
    >>> named_color("pink")
    '#e6338f'
    """
    return NAMED[name.lower()]


def indicator_color(ind: Indicator | None, ph: float) -> str:
    """Colour of a solution containing ``ind`` at ``ph``.

    The two forms are mixed with the base-form fraction
    ``1 / (1 + 10^(pK_In - pH))``.  APPROXIMATION: linear RGB mixing of the two
    forms (real colour perception is non-linear, equal path length and equal
    molar absorptivity of the forms are implied); the transition range comes
    out right (about pK_In +- 1) but intermediate hues are only illustrative.
    If the base colour has faded (phenolphthalein above pH ~13) the acid
    colour is returned.

    Example
    -------
    >>> from titration_sim.engine.indicators import load_indicators
    >>> pp = [i for i in load_indicators() if i.name == "Phenolphthalein"][0]
    >>> indicator_color(pp, 4.0) == WATER
    True
    """
    if ind is None:
        return WATER
    acid = named_color(ind.acid_color)
    base = named_color(ind.base_color)
    if not base_color_visible(ind, ph):
        return acid
    return mix(acid, base, base_fraction(ind, ph))
