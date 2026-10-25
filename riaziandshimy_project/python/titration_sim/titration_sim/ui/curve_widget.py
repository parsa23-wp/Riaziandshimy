"""Titration curve (pH vs V) and its first derivative, drawn with pyqtgraph."""
from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget


class CurveWidget(pg.GraphicsLayoutWidget):  # type: ignore[misc]
    """Two linked plots: pH(V) on top, dpH/dV below.

    Layers on the pH plot: faint theoretical curve (optional), measured
    points, vertical line at the theoretical equivalence volume, horizontal
    band of the indicator's colour-change range.

    Example
    -------
    >>> w = CurveWidget()                                   # doctest: +SKIP
    >>> w.set_measured(np.array([0.0, 1.0]), np.array([1.0, 1.1]))   # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setBackground("w")
        self.ph_plot = self.addPlot(row=0, col=0)
        self.d_plot = self.addPlot(row=1, col=0)
        self.d_plot.setXLink(self.ph_plot)
        self.ph_plot.setLabel("left", "pH")
        self.d_plot.setLabel("left", "dpH/dV")
        self.d_plot.setLabel("bottom", "V (mL)")
        self.ph_plot.showGrid(x=True, y=True, alpha=0.25)
        self.d_plot.showGrid(x=True, y=True, alpha=0.25)
        self._theory = self.ph_plot.plot(pen=pg.mkPen("#9bb3bd", width=1.5, style=Qt.PenStyle.DashLine))
        self._measured = self.ph_plot.plot(
            pen=pg.mkPen("#0e7c8c", width=2), symbol="o", symbolSize=5, symbolBrush="#0e7c8c"
        )
        self._deriv = self.d_plot.plot(pen=pg.mkPen("#b3364c", width=1.5))
        self._eq_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#c9861a", width=1.5), movable=False)
        self.ph_plot.addItem(self._eq_line)
        self._eq_line.setVisible(False)
        self._band: Any = None
        self._found_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#2f7d4f", width=1.5, style=Qt.PenStyle.DotLine))
        self.d_plot.addItem(self._found_line)
        self._found_line.setVisible(False)

    def set_quantity(self, name: str) -> None:
        """Axis label of the measured quantity (pH / E (V) / pAg / pM)."""
        self.ph_plot.setLabel("left", name)
        self.d_plot.setLabel("left", "d" + name + "/dV")

    def set_theory(self, v: np.ndarray[Any, Any], ph: np.ndarray[Any, Any], visible: bool = True) -> None:
        """Theoretical curve (faint, dashed)."""
        self._theory.setData(v, ph)
        self._theory.setVisible(visible)

    def set_theory_visible(self, visible: bool) -> None:
        """Show/hide the theoretical curve without recomputing it."""
        self._theory.setVisible(visible)

    def set_measured(self, v: np.ndarray[Any, Any], ph: np.ndarray[Any, Any]) -> None:
        """Collected points."""
        self._measured.setData(v, ph)

    def set_derivative(self, v: np.ndarray[Any, Any], d: np.ndarray[Any, Any]) -> None:
        """First derivative of the collected points (may be empty)."""
        self._deriv.setData(v, d)

    def set_equivalence(self, v_theory: float | None, v_found: float | None) -> None:
        """Vertical markers: theoretical V_eq (orange, pH plot), V_eq from the points (green, derivative plot)."""
        if v_theory is None:
            self._eq_line.setVisible(False)
        else:
            self._eq_line.setValue(v_theory)
            self._eq_line.setVisible(True)
        if v_found is None:
            self._found_line.setVisible(False)
        else:
            self._found_line.setValue(v_found)
            self._found_line.setVisible(True)

    def set_indicator_band(self, lo: float | None, hi: float | None, color: str = "#e6338f") -> None:
        """Horizontal band showing the indicator's colour-change range (None removes it)."""
        if self._band is not None:
            self.ph_plot.removeItem(self._band)
            self._band = None
        if lo is None or hi is None:
            return
        brush = pg.mkBrush(color + "33")
        self._band = pg.LinearRegionItem(values=(lo, hi), orientation="horizontal", movable=False, brush=brush)
        self.ph_plot.addItem(self._band)

    def clear_all(self) -> None:
        """Remove all data (keeps axes and styling)."""
        empty = np.array([])
        self._theory.setData(empty, empty)
        self._measured.setData(empty, empty)
        self._deriv.setData(empty, empty)
        self.set_equivalence(None, None)
        self.set_indicator_band(None, None)
