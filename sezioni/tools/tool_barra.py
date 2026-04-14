"""tool_barra.py – Barra di armatura. Click → PENDING, etichetta unificata."""
import re
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui  import QColor, QPen, QFont, QBrush, QFontMetrics
from .base_tool   import BaseTool
from ._draw_helpers import draw_label, draw_hint, FONT_LBL


class ToolBarra(BaseTool):
    _COL_PREV = QColor(210, 80, 80, 200)
    _COL_FILL = QColor(210, 80, 80, 60)
    _COL_PEND = QColor(255, 200, 50, 220)
    _COL_PFIL = QColor(255, 200, 50, 80)

    def __init__(self, diametro=16.0):
        self.diametro = diametro
        self._pos = None
        self._cursor = None
        self._pending = False

    def reset(self):
        self._pos = None
        self._cursor = None
        self._pending = False

    @property
    def is_pending(self):
        return self._pending

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
            if self._pending:
                self.confirm(w)
                return True
            wx, wy = w.screen_to_world(e.x(), e.y())
            self._pos = (wx, wy)
            self._pending = True
            w.update()
            return True
        return False

    def on_mouse_move(self, w, e) -> bool:
        wx, wy = w.screen_to_world(e.x(), e.y())
        self._cursor = (wx, wy)
        w.update()
        return True

    def confirm(self, w) -> bool:
        if not self._pending or not self._pos:
            return False
        w.aggiungi_elemento({
            "tipo": "barra",
            "geometria": {"cx": self._pos[0], "cy": self._pos[1],
                          "r": self.diametro / 2.0},
            "materiale": "",
        })
        self.reset()
        w.update()
        return True

    def get_properties_text(self):
        if not self._pos:
            return ""
        return f"Ø={self.diametro:.1f}; C({self._pos[0]:.1f},{self._pos[1]:.1f})"

    def apply_properties_text(self, text):
        md = re.search(r'Ø\s*=\s*([\d.]+)', text)
        if md:
            try:
                self.diametro = max(1.0, float(md.group(1)))
            except ValueError:
                pass
        mc = re.search(r'C\s*\(([\d.,-]+),([\d.,-]+)\)', text)
        if mc and self._pos:
            try:
                self._pos = (float(mc.group(1).replace(",", ".")),
                             float(mc.group(2).replace(",", ".")))
            except ValueError:
                pass

    def draw_painter(self, w, painter):
        pos = self._pos if self._pending else self._cursor
        if not pos:
            draw_hint(painter, w, f"Click = piazza barra Ø{self.diametro:.0f}")
            return
        pend = self._pending
        col = self._COL_PEND if pend else self._COL_PREV
        cf = self._COL_PFIL if pend else self._COL_FILL
        cx_s, cy_s = w.world_to_screen(*pos)
        rx_s, _ = w.world_to_screen(pos[0] + self.diametro / 2, pos[1])
        r_px = max(3, abs(rx_s - cx_s))
        painter.setPen(QPen(col, 2 if pend else 1, Qt.SolidLine if pend else Qt.DashLine))
        painter.setBrush(QBrush(cf))
        painter.drawEllipse(cx_s - r_px, cy_s - r_px, r_px * 2, r_px * 2)
        if pend:
            painter.setFont(FONT_LBL)
            fm = painter.fontMetrics()
            draw_label(painter, cx_s, cy_s + r_px + 14,
                       f"Ø{self.diametro:.0f}  ({pos[0]:.1f}, {pos[1]:.1f})", fm)
            draw_hint(painter, w, "INVIO/Click = conferma | ESC = annulla")
        else:
            draw_hint(painter, w, f"Click = piazza barra Ø{self.diametro:.0f}")
