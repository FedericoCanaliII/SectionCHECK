"""
tool_rettangolo.py – Rettangolo/Foro.
Click1=V1, Click2=V2 → PENDING.
I vertici restano come piazzati dall'utente (no swap min/max in preview).
"""
import re
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui  import QColor, QPen, QFont, QBrush
from .base_tool   import BaseTool
from ._draw_helpers import draw_label, draw_hint, FONT_LBL


class _RettangoloBase(BaseTool):
    _TIPO = "rettangolo"
    _COL_PREV = QColor(80, 200, 80, 220)
    _COL_FILL = QColor(80, 200, 80, 30)
    _COL_PEND = QColor(255, 200, 50, 220)
    _COL_PFIL = QColor(255, 200, 50, 25)

    def __init__(self):
        self._p1 = None
        self._p2 = None
        self._phase = 0
        self._cursor = None

    def reset(self):
        self._p1 = None
        self._p2 = None
        self._phase = 0
        self._cursor = None

    @property
    def is_pending(self):
        return self._phase == 2

    def on_activate(self, w):
        w.setCursor(Qt.CrossCursor)
        w.setFocus()

    def on_deactivate(self, w):
        w.setCursor(Qt.ArrowCursor)
        self.reset()

    def on_mouse_press(self, w, e) -> bool:
        if e.button() == Qt.RightButton:
            self.reset()
            w.update()
            return True
        if e.button() == Qt.LeftButton:
            wx, wy = w.screen_to_world(e.x(), e.y())
            if self._phase == 0:
                self._p1 = (wx, wy)
                self._phase = 1
                return True
            elif self._phase == 1:
                self._p2 = (wx, wy)
                self._phase = 2
                w.update()
                return True
            elif self._phase == 2:
                self.confirm(w)
                return True
        return False

    def on_mouse_move(self, w, e) -> bool:
        wx, wy = w.screen_to_world(e.x(), e.y())
        self._cursor = (wx, wy)
        if self._phase == 1:
            self._p2 = (wx, wy)
        w.update()
        return self._phase > 0

    def confirm(self, w) -> bool:
        if self._phase < 2 or not self._p1 or not self._p2:
            return False
        # Solo al confirm facciamo il min/max per la geometria salvata
        x0 = min(self._p1[0], self._p2[0])
        y0 = min(self._p1[1], self._p2[1])
        x1 = max(self._p1[0], self._p2[0])
        y1 = max(self._p1[1], self._p2[1])
        if (x1 - x0) > 0.1 and (y1 - y0) > 0.1:
            w.aggiungi_elemento({
                "tipo": self._TIPO,
                "geometria": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "materiale": "",
            })
        self.reset()
        w.update()
        return True

    # --- Proprietà per label / lineEdit ---
    def get_properties_text(self):
        if not self._p1 or not self._p2:
            return ""
        return (f"V1 ({self._p1[0]:.1f},{self._p1[1]:.1f}); "
                f"V2 ({self._p2[0]:.1f},{self._p2[1]:.1f})")

    def apply_properties_text(self, text):
        m1 = re.search(r'V1\s*\(([\d.,-]+),([\d.,-]+)\)', text)
        m2 = re.search(r'V2\s*\(([\d.,-]+),([\d.,-]+)\)', text)
        if m1:
            try:
                self._p1 = (float(m1.group(1).replace(",", ".")),
                            float(m1.group(2).replace(",", ".")))
            except ValueError:
                pass
        if m2:
            try:
                self._p2 = (float(m2.group(1).replace(",", ".")),
                            float(m2.group(2).replace(",", ".")))
            except ValueError:
                pass

    # --- Rendering ---
    def draw_painter(self, w, painter):
        if self._phase == 0:
            draw_hint(painter, w, "Click = primo vertice")
            return
        p1 = self._p1
        p2 = self._p2 if self._p2 else self._cursor
        if not p1 or not p2:
            return
        sx1, sy1 = w.world_to_screen(*p1)
        sx2, sy2 = w.world_to_screen(*p2)
        pend = self._phase == 2
        col = self._COL_PEND if pend else self._COL_PREV
        cf = self._COL_PFIL if pend else self._COL_FILL

        # Rettangolo
        pen = QPen(col, 2 if pend else 1, Qt.SolidLine if pend else Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(cf))
        painter.drawRect(min(sx1, sx2), min(sy1, sy2), abs(sx2 - sx1), abs(sy2 - sy1))

        # Handle vertici
        for sx, sy in [(sx1, sy1), (sx2, sy2)]:
            painter.setBrush(QBrush(col))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(sx - 4, sy - 4, 8, 8)

        # Etichette singoli vertici
        painter.setFont(FONT_LBL)
        fm = painter.fontMetrics()
        draw_label(painter, sx1, sy1 + 16, f"V1({p1[0]:.1f}, {p1[1]:.1f})", fm)
        draw_label(painter, sx2, sy2 - 8, f"V2({p2[0]:.1f}, {p2[1]:.1f})", fm)

        if pend:
            draw_hint(painter, w, "INVIO/Click = conferma | ESC = annulla")
        else:
            draw_hint(painter, w, "Click = secondo vertice | ESC = annulla")


class ToolRettangolo(_RettangoloBase):
    _TIPO = "rettangolo"


class ToolForoRettangolo(_RettangoloBase):
    _TIPO = "foro_rettangolo"
    _COL_PREV = QColor(200, 80, 80, 220)
    _COL_FILL = QColor(200, 80, 80, 30)
    _COL_PEND = QColor(255, 120, 50, 220)
    _COL_PFIL = QColor(255, 120, 50, 25)
