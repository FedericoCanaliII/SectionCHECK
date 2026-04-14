"""tool_staffa.py – Staffa con punti + diametro. Etichetta unificata."""
import re
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui  import QColor, QPen, QFont, QBrush, QFontMetrics
from .base_tool   import BaseTool
from ._draw_helpers import draw_label, draw_hint, FONT_LBL


class ToolStaffa(BaseTool):
    _COL_LINE = QColor(200, 155, 60, 220)
    _COL_PEND = QColor(255, 200, 50, 220)
    _COL_VTX  = QColor(200, 155, 60, 200)
    _COL_V1   = QColor(255, 220, 80, 220)

    def __init__(self, diametro=8.0):
        self.diametro = diametro
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
            if len(self._punti) >= 2:
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
            if len(self._punti) >= 2:
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
        if len(self._punti) < 2:
            return False
        pts = [list(p) for p in self._punti]
        if pts[0] != pts[-1]:
            pts.append(list(pts[0]))
        w.aggiungi_elemento({
            "tipo": "staffa",
            "geometria": {"punti": pts, "r": self.diametro / 2.0},
            "materiale": "",
        })
        self.reset()
        w.update()
        return True

    def get_properties_text(self):
        if not self._punti:
            return ""
        parts = [f"Ø={self.diametro:.1f}"]
        for i, p in enumerate(self._punti):
            parts.append(f"P{i+1} ({p[0]:.1f},{p[1]:.1f})")
        return "; ".join(parts)

    def apply_properties_text(self, text):
        md = re.search(r'Ø\s*=\s*([\d.]+)', text)
        if md:
            try:
                self.diametro = max(1.0, float(md.group(1)))
            except ValueError:
                pass
        matches = re.findall(r'P(\d+)\s*\(([\d.,-]+),([\d.,-]+)\)', text)
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
            draw_hint(painter, w, f"Click = punto staffa Ø{self.diametro:.0f}")
            return
        pend = self._pending
        col = self._COL_PEND if pend else self._COL_LINE
        pts_s = [w.world_to_screen(*p) for p in self._punti]
        pts_draw = list(pts_s)
        if self._preview and not pend:
            pts_draw.append(w.world_to_screen(*self._preview))

        painter.setPen(QPen(col, 2, Qt.SolidLine if pend else Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        for i in range(len(pts_draw) - 1):
            painter.drawLine(pts_draw[i][0], pts_draw[i][1],
                             pts_draw[i + 1][0], pts_draw[i + 1][1])
        if pend and len(pts_draw) >= 2:
            painter.drawLine(pts_draw[-1][0], pts_draw[-1][1],
                             pts_draw[0][0], pts_draw[0][1])
        for i, (sx, sy) in enumerate(pts_s):
            if i == 0 and len(pts_s) >= 2:
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
            draw_label(painter, sx + 40, sy - 6, f"P{i+1}({p[0]:.1f},{p[1]:.1f})", fm)

        if pend:
            draw_hint(painter, w, f"Staffa Ø{self.diametro:.0f} | INVIO/Click = conferma | ESC = annulla")
        elif len(self._punti) >= 2:
            draw_hint(painter, w, "Click P1 = chiudi | INVIO = conferma | Backspace")
        else:
            draw_hint(painter, w, f"Click = punto ({len(self._punti)}/2 min)")
