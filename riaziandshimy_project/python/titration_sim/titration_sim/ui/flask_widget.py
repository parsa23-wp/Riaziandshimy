"""Erlenmeyer flask with a liquid whose colour follows the indicator."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class FlaskWidget(QWidget):
    """Conical flask.

    Example
    -------
    >>> w = FlaskWidget()                          # doctest: +SKIP
    >>> w.set_content(40.0, "#e6338f", 7.0)        # doctest: +SKIP
    """

    def __init__(self, max_volume_ml: float = 150.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._max = max_volume_ml
        self._volume = 0.0
        self._color = QColor("#d8edf4")
        self._ph: float | None = None
        self._label = "pH"
        self.setMinimumSize(180, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_max_volume(self, max_volume_ml: float) -> None:
        """Flask capacity (mL, > 0)."""
        if max_volume_ml <= 0.0:
            raise ValueError("capacity must be positive")
        self._max = max_volume_ml
        self.update()

    def set_content(self, volume_ml: float, color: str, ph: float | None = None,
                    label: str = "pH") -> None:
        """Show ``volume_ml`` of a solution of the given colour.

        ``ph`` is the measured quantity (pH for acid-base; E, pAg or pM for
        the phase-2 families) and ``label`` names it on the flask.
        """
        self._volume = max(0.0, min(volume_ml, self._max))
        self._color = QColor(color)
        self._ph = ph
        self._label = label
        self.update()

    @property
    def liquid_color(self) -> str:
        """Current liquid colour as ``#rrggbb``."""
        return self._color.name()

    def paintEvent(self, event: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        cx = w / 2.0
        neck_w = min(w * 0.22, 46.0)
        base_w = min(w * 0.8, 220.0)
        y_top, y_neck, y_bot = 14.0, h * 0.30, h - 40.0
        outline = QPainterPath()
        outline.moveTo(QPointF(cx - neck_w / 2.0, y_top))
        outline.lineTo(QPointF(cx - neck_w / 2.0, y_neck))
        outline.lineTo(QPointF(cx - base_w / 2.0, y_bot))
        outline.lineTo(QPointF(cx + base_w / 2.0, y_bot))
        outline.lineTo(QPointF(cx + neck_w / 2.0, y_neck))
        outline.lineTo(QPointF(cx + neck_w / 2.0, y_top))
        # liquid: clip the outline to the part below the surface
        frac = self._volume / self._max if self._max > 0.0 else 0.0
        surface = y_bot - frac * (y_bot - y_top) * 0.9
        p.save()
        p.setClipPath(outline)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._color)
        p.drawRect(QRectF(0.0, surface, w, h - surface))
        p.restore()
        p.setPen(QPen(QColor("#53707c"), 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(outline)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor("#10303c"))
        import math as _m
        value = "" if self._ph is None or _m.isnan(self._ph) else \
            f"   {self._label} {self._ph:.3f}" if self._label != "pH" else f"   pH {self._ph:.2f}"
        label = f"{self._volume:.1f} mL" + value
        p.drawText(QRectF(0.0, h - 30.0, w, 26.0), int(Qt.AlignmentFlag.AlignCenter), label)
        p.end()
