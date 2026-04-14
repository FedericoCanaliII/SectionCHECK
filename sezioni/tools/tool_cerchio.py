"""
tool_cerchio.py – Cerchio / Ellisse / Foro.
Click1=centro, Click2=raggio → PENDING cerchio.
Click3=raggio2 → PENDING ellisse. INVIO = conferma.
"""
import math, re
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui  import QColor, QPen, QFont, QBrush, QFontMetrics
from .base_tool   import BaseTool
from ._draw_helpers import draw_label, draw_hint, FONT_LBL


class _CerchioBase(BaseTool):
    _TIPO = "cerchio"
    _COL_LINE = QColor(80, 200, 80, 220)
    _COL_FILL = QColor(80, 200, 80, 30)
    _COL_PEND = QColor(255, 200, 50, 220)
    _COL_PFIL = QColor(255, 200, 50, 25)

    def __init__(self):
        self._centro = None
        self._rx = 0.0
        self._ry = 0.0
        self._phase = 0
        self._cursor = None

    def reset(self):
        self._centro = None
        self._rx = 0.0
        self._ry = 0.0
        self._phase = 0
        self._cursor = None

    @property
    def is_pending(self):
        return self._phase >= 2

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
                self._centro = (wx, wy)
                self._phase = 1
                return True
            elif self._phase == 1:
                self._rx = math.hypot(wx - self._centro[0], wy - self._centro[1])
                self._ry = self._rx
                if self._rx > 0.5:
                    self._phase = 2
                w.update()
                return True
            elif self._phase == 2:
                self._ry = math.hypot(wx - self._centro[0], wy - self._centro[1])
                if self._ry > 0.5:
                    self._phase = 3
                w.update()
                return True
            elif self._phase == 3:
                self.confirm(w)
                return True
        return False

    def on_mouse_move(self, w, e) -> bool:
        wx, wy = w.screen_to_world(e.x(), e.y())
        self._cursor = (wx, wy)
        if self._phase == 1 and self._centro:
            r = math.hypot(wx - self._centro[0], wy - self._centro[1])
            self._rx = self._ry = r
        elif self._phase == 2 and self._centro:
            self._ry = math.hypot(wx - self._centro[0], wy - self._centro[1])
        w.update()
        return self._phase > 0

    def on_key_press(self, w, e) -> bool:
        key = e.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self._phase >= 2:
                return self.confirm(w)
            return False
        if key == Qt.Key_Escape:
            return self.cancel(w)
        return False

    def confirm(self, w) -> bool:
        if self._phase < 2 or not self._centro:
            return False
        rx = self._rx
        ry = self._ry if self._phase == 3 else self._rx
        if rx < 0.5:
            return False
        w.aggiungi_elemento({
            "tipo": self._TIPO,
            "geometria": {"cx": self._centro[0], "cy": self._centro[1],
                          "rx": rx, "ry": ry},
            "materiale": "",
        })
        self.reset()
        w.update()
        return True

    def get_properties_text(self):
        if not self._centro:
            return ""
        return (f"C({self._centro[0]:.1f},{self._centro[1]:.1f}); "
                f"rx={self._rx:.1f}; ry={self._ry:.1f}")

    def apply_properties_text(self, text):
        mc = re.search(r'C\s*\(([\d.,-]+),([\d.,-]+)\)', text)
        if mc and self._centro:
            try:
                self._centro = (float(mc.group(1).replace(",", ".")),
                                float(mc.group(2).replace(",", ".")))
            except ValueError:
                pass
        mrx = re.search(r'rx\s*=\s*([\d.]+)', text)
        if mrx:
            try:
                self._rx = float(mrx.group(1))
            except ValueError:
                pass
        mry = re.search(r'ry\s*=\s*([\d.]+)', text)
        if mry:
            try:
                self._ry = float(mry.group(1))
            except ValueError:
                pass

    def draw_painter(self, w, painter):
        if self._phase == 0:
            draw_hint(painter, w, "Click = piazza centro")
            return
        if not self._centro:
            return
        pend = self._phase >= 2
        col = self._COL_PEND if pend else self._COL_LINE
        cf = self._COL_PFIL if pend else self._COL_FILL
        cx_s, cy_s = w.world_to_screen(*self._centro)
        rx_s, _ = w.world_to_screen(self._centro[0] + self._rx, self._centro[1])
        _, ry_s = w.world_to_screen(self._centro[0], self._centro[1] + self._ry)
        rpx_x = abs(rx_s - cx_s)
        rpx_y = abs(ry_s - cy_s) if self._phase == 3 else rpx_x

        painter.setPen(QPen(col, 2 if pend else 1, Qt.SolidLine if pend else Qt.DashLine))
        painter.setBrush(QBrush(cf))
        painter.drawEllipse(QRectF(cx_s - rpx_x, cy_s - rpx_y, rpx_x * 2, rpx_y * 2))
        # Centro
        painter.setBrush(QBrush(col))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx_s - 4, cy_s - 4, 8, 8)
        # Linea raggio
        if self._cursor and self._phase in (1, 2):
            rx_e, ry_e = w.world_to_screen(*self._cursor)
            painter.setPen(QPen(col, 1, Qt.DotLine))
            painter.drawLine(cx_s, cy_s, rx_e, ry_e)

        painter.setFont(FONT_LBL)
        fm = painter.fontMetrics()
        draw_label(painter, cx_s, cy_s + rpx_y + 18,
                   f"C({self._centro[0]:.1f}, {self._centro[1]:.1f})", fm)
        if self._phase >= 2:
            if self._phase == 3:
                draw_label(painter, cx_s + rpx_x + 10, cy_s, f"Rx={self._rx:.1f}", fm)
                draw_label(painter, cx_s, cy_s - rpx_y - 10, f"Ry={self._ry:.1f}", fm)
            else:
                draw_label(painter, cx_s + rpx_x + 10, cy_s,
                           f"R={self._rx:.1f}  Ø={self._rx * 2:.1f}", fm)

        if self._phase == 3:
            draw_hint(painter, w, "Ellisse | INVIO/Click = conferma | ESC = annulla")
        elif self._phase == 2:
            draw_hint(painter, w, "Cerchio | INVIO = conferma | Click = Ry (ellisse)")
        elif self._phase == 1:
            draw_hint(painter, w, "Click = definisci raggio | ESC = annulla")


class ToolCerchio(_CerchioBase):
    _TIPO = "cerchio"


class ToolForoCerchio(_CerchioBase):
    _TIPO = "foro_cerchio"
    _COL_LINE = QColor(200, 80, 80, 220)
    _COL_FILL = QColor(200, 80, 80, 30)
    _COL_PEND = QColor(255, 120, 50, 220)
    _COL_PFIL = QColor(255, 120, 50, 25)
