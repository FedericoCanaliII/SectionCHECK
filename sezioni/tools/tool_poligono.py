"""tool_poligono.py – Poligono libero / Foro poligonale."""
import re
from PyQt5.QtCore import Qt, QRectF, QPoint
from PyQt5.QtGui  import QColor, QPen, QFont, QBrush, QPolygon, QFontMetrics
from .base_tool   import BaseTool
from ._draw_helpers import draw_label, draw_hint, FONT_LBL


class _PoligonoBase(BaseTool):
    _TIPO = "poligono"
    _COL_LINE = QColor(80, 200, 80, 220)
    _COL_FILL = QColor(80, 200, 80, 25)
    _COL_PEND = QColor(255, 200, 50, 220)
    _COL_PFIL = QColor(255, 200, 50, 25)
    _COL_VTX  = QColor(80, 200, 80, 200)
    _COL_V1   = QColor(255, 255, 80, 220)

    def __init__(self):
        self._punti = []
        self._preview = None
        self._pending = False

    def reset(self):
        self._punti = []
        self._preview = None
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
            if len(self._punti) >= 3:
                if (abs(wx - self._punti[0][0]) < w.grid_spacing * 1.5 and
                    abs(wy - self._punti[0][1]) < w.grid_spacing * 1.5):
                    self._pending = True
                    w.update()
                    return True
            self._punti.append([wx, wy])
            w.update()
            return True
        return False

    def on_mouse_move(self, w, e) -> bool:
        if not self._pending:
            wx, wy = w.screen_to_world(e.x(), e.y())
            self._preview = (wx, wy)
            w.update()
        return len(self._punti) > 0

    def on_key_press(self, w, e) -> bool:
        key = e.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if len(self._punti) >= 3:
                self._pending = True
                return self.confirm(w)
            return False
        if key == Qt.Key_Escape:
            return self.cancel(w)
        if key == Qt.Key_Backspace and self._punti and not self._pending:
            self._punti.pop()
            w.update()
            return True
        return False

    def confirm(self, w) -> bool:
        if len(self._punti) < 3:
            return False
        w.aggiungi_elemento({
            "tipo": self._TIPO,
            "geometria": {"punti": [list(p) for p in self._punti]},
            "materiale": "",
        })
        self.reset()
        w.update()
        return True

    def get_properties_text(self):
        if not self._punti:
            return ""
        parts = []
        for i, p in enumerate(self._punti):
            parts.append(f"V{i+1} ({p[0]:.1f},{p[1]:.1f})")
        return "; ".join(parts)

    def apply_properties_text(self, text):
        matches = re.findall(r'V(\d+)\s*\(([\d.,-]+),([\d.,-]+)\)', text)
        for m in matches:
            idx = int(m[0]) - 1
            if 0 <= idx < len(self._punti):
                try:
                    self._punti[idx][0] = float(m[1].replace(",", "."))
                    self._punti[idx][1] = float(m[2].replace(",", "."))
                except ValueError:
                    pass

    def draw_painter(self, w, painter):
        if not self._punti:
            draw_hint(painter, w, "Click = vertice | Backspace = annulla ultimo")
            return
        pend = self._pending
        col = self._COL_PEND if pend else self._COL_LINE
        cf = self._COL_PFIL if pend else self._COL_FILL
        pts_s = [w.world_to_screen(*p) for p in self._punti]
        pts_draw = list(pts_s)
        if self._preview and not pend:
            pts_draw.append(w.world_to_screen(*self._preview))

        painter.setPen(QPen(col, 2 if pend else 1, Qt.SolidLine if pend else Qt.DashLine))
        painter.setBrush(QBrush(cf))
        if len(pts_draw) >= 3:
            painter.drawPolygon(QPolygon([QPoint(int(x), int(y)) for x, y in pts_draw]))
        elif len(pts_draw) == 2:
            painter.drawLine(pts_draw[0][0], pts_draw[0][1], pts_draw[1][0], pts_draw[1][1])

        for i, (sx, sy) in enumerate(pts_s):
            if i == 0 and len(pts_s) >= 3:
                painter.setBrush(QBrush(self._COL_V1))
                painter.setPen(QPen(self._COL_V1, 2))
                painter.drawEllipse(sx - 5, sy - 5, 10, 10)
            else:
                painter.setBrush(QBrush(self._COL_VTX))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(sx - 4, sy - 4, 8, 8)

        painter.setFont(FONT_LBL)
        fm = painter.fontMetrics()
        for i, (sx, sy) in enumerate(pts_s):
            p = self._punti[i]
            draw_label(painter, sx + 40, sy - 6, f"V{i+1}({p[0]:.1f},{p[1]:.1f})", fm)

        if pend:
            draw_hint(painter, w, f"{len(self._punti)} vertici | INVIO/Click = conferma | ESC = annulla")
        elif len(self._punti) >= 3:
            draw_hint(painter, w, "Click = vertice | Click V1 = chiudi | INVIO = conferma")
        else:
            draw_hint(painter, w, f"Click = vertice ({len(self._punti)}/3 min) | ESC/Backspace")


class ToolPoligono(_PoligonoBase):
    _TIPO = "poligono"

class ToolForoPoligono(_PoligonoBase):
    _TIPO = "foro_poligono"
    _COL_LINE = QColor(200, 80, 80, 220)
    _COL_FILL = QColor(200, 80, 80, 25)
    _COL_PEND = QColor(255, 120, 50, 220)
    _COL_PFIL = QColor(255, 120, 50, 25)
