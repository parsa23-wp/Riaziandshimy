"""Burette: graduated tube, liquid level, stopcock and a falling drop.

Painted with ``QPainter`` only (no images, no extra dependency).  The widget
holds no chemistry; it shows what :class:`TitrationSession` reports.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class BuretteWidget(QWidget):
    """Vertical burette.

    Parameters
    ----------
    capacity_ml : float
        Scale length (mL); the zero mark is at the top, as on a real burette.
    parent : QWidget | None

    Example
    -------
    >>> w = BuretteWidget(50.0)            # doctest: +SKIP
    >>> w.set_volume_added(12.5)           # doctest: +SKIP
    """

    def __init__(self, capacity_ml: float = 50.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._capacity = capacity_ml
        self._added = 0.0
        self._liquid = QColor("#cfe8f2")
        self._drop_y = -1.0
        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._tick)
        self.setMinimumSize(96, 320)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    # ---- API -----------------------------------------------------------
    def set_capacity(self, capacity_ml: float) -> None:
        """Change the scale length (mL, > 0)."""
        if capacity_ml <= 0.0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity_ml
        self.update()

    def set_volume_added(self, v_ml: float, animate_drop: bool = False) -> None:
        """Show ``v_ml`` mL as delivered (liquid level drops accordingly)."""
        self._added = max(0.0, min(v_ml, self._capacity))
        if animate_drop:
            self._drop_y = 0.0
            self._timer.start()
        self.update()

    def set_liquid_color(self, color: str) -> None:
        """Colour of the titrant (``#rrggbb``)."""
        self._liquid = QColor(color)
        self.update()

    @property
    def reading_ml(self) -> float:
        """Current burette reading (mL)."""
        return self._added

    # ---- internals -----------------------------------------------------
    def _tick(self) -> None:
        self._drop_y += 0.12
        if self._drop_y >= 1.0:
            self._drop_y = -1.0
            self._timer.stop()
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        tube_w = min(34.0, w * 0.35)
        x0 = w * 0.5 - tube_w / 2.0
        top, bottom = 24.0, h - 70.0
        scale_h = bottom - top
        tube = QRectF(x0, top, tube_w, scale_h)
        # liquid: from the current level down to the tip
        frac = self._added / self._capacity
        level_y = top + frac * scale_h
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._liquid)
        p.drawRect(QRectF(x0 + 2.0, level_y, tube_w - 4.0, bottom - level_y))
        # glass outline
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#53707c"), 2.0))
        p.drawRect(tube)
        # graduations: minor every 1 mL (or 0.1 of a scale tenth), major every 5 mL
        font = QFont()
        font.setPointSize(7)
        p.setFont(font)
        step = 1.0 if self._capacity <= 60.0 else self._capacity / 50.0
        k = 0
        while k * step <= self._capacity + 1e-9:
            y = top + (k * step / self._capacity) * scale_h
            major = abs((k * step) % 5.0) < 1e-9 or abs((k * step) % 5.0 - 5.0) < 1e-9
            tick = 12.0 if major else 6.0
            p.setPen(QPen(QColor("#2d4650"), 1.0))
            p.drawLine(int(x0 + tube_w), int(y), int(x0 + tube_w + tick), int(y))
            if major:
                p.drawText(int(x0 + tube_w + 14.0), int(y + 4.0), f"{k * step:g}")
            k += 1
        # stopcock and tip
        p.setPen(QPen(QColor("#53707c"), 2.0))
        p.setBrush(QColor("#9bb3bd"))
        p.drawRect(QRectF(w * 0.5 - 14.0, bottom + 4.0, 28.0, 8.0))
        p.drawLine(int(w * 0.5), int(bottom + 12.0), int(w * 0.5), int(bottom + 30.0))
        # drop
        if self._drop_y >= 0.0:
            y = bottom + 30.0 + self._drop_y * 28.0
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self._liquid)
            p.drawEllipse(QRectF(w * 0.5 - 3.0, y, 6.0, 8.0))
        # reading
        p.setPen(QColor("#10303c"))
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRectF(0.0, h - 28.0, w, 24.0), int(Qt.AlignmentFlag.AlignCenter), f"{self._added:.2f} mL")
        p.end()
