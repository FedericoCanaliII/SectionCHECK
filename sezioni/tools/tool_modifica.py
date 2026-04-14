"""
tool_modifica.py – Selezione / Modifica con proprietà editabili.
Click = seleziona → mostra proprietà nel lineEdit esterno e overlay grafico.
"""
import math, copy, re
from PyQt5.QtCore import Qt, QRectF, QPoint
from PyQt5.QtGui  import QColor, QPen, QFont, QBrush, QPolygon, QFontMetrics
from .base_tool   import BaseTool
from ._draw_helpers import draw_label, draw_hint, FONT_LBL


def _pt_in_rect(px, py, x0, y0, x1, y1):
    return min(x0, x1) <= px <= max(x0, x1) and min(y0, y1) <= py <= max(y0, y1)

def _pt_in_poly(px, py, punti):
    n = len(punti)
    ins = False
    j = n - 1
    for i in range(n):
        xi, yi = punti[i]
        xj, yj = punti[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            ins = not ins
        j = i
    return ins

def _hit_test(el, wx, wy):
    tipo = el["tipo"]
    g = el["geometria"]
    tol = 3.0
    if tipo in ("rettangolo", "foro_rettangolo"):
        return _pt_in_rect(wx, wy, g["x0"] - tol, g["y0"] - tol,
                           g["x1"] + tol, g["y1"] + tol)
    if tipo in ("poligono", "foro_poligono"):
        return _pt_in_poly(wx, wy, g["punti"])
    if tipo in ("cerchio", "foro_cerchio", "barra"):
        rx = g.get("rx", g.get("r", 10))
        ry = g.get("ry", rx)
        dx = (wx - g["cx"]) / max(rx, 0.1)
        dy = (wy - g["cy"]) / max(ry, 0.1)
        return (dx * dx + dy * dy) <= (1 + tol / max(rx, 1)) ** 2
    if tipo == "staffa":
        pts = g["punti"]
        r = g.get("r", 4)
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx, dy = x2 - x1, y2 - y1
            l2 = dx * dx + dy * dy
            if l2 < 1e-10:
                continue
            t = max(0, min(1, ((wx - x1) * dx + (wy - y1) * dy) / l2))
            if math.hypot(wx - (x1 + t * dx), wy - (y1 + t * dy)) <= r + tol + 3:
                return True
    return False


class ToolModifica(BaseTool):
    _COL_SEL = QColor(255, 220, 50, 255)

    def __init__(self):
        self._dragging = False
        self._drag_start = None
        self._orig_geom = None

    def reset(self):
        self._dragging = False
        self._drag_start = None
        self._orig_geom = None

    def on_activate(self, w):
        w.setCursor(Qt.ArrowCursor)
        w.setFocus()

    def on_deactivate(self, w):
        w.set_selected(None)
        w.setCursor(Qt.ArrowCursor)
        self.reset()

    def on_mouse_press(self, w, e) -> bool:
        wx, wy = w.screen_to_world(e.x(), e.y())
        if e.button() == Qt.LeftButton:
            hit = None
            # Priority: barre > staffe > fori > carpenteria solida
            for cat in ("barre", "staffe"):
                for el in w.get_elementi(cat):
                    if _hit_test(el, wx, wy):
                        hit = el
                        break
                if hit:
                    break
            if not hit:
                # fori first, then solid carpenteria
                fori = []
                solidi = []
                for el in w.get_elementi("carpenteria"):
                    if el["tipo"].startswith("foro"):
                        fori.append(el)
                    else:
                        solidi.append(el)
                for el in fori:
                    if _hit_test(el, wx, wy):
                        hit = el
                        break
                if not hit:
                    for el in solidi:
                        if _hit_test(el, wx, wy):
                            hit = el
                            break
            w.set_selected(hit["id"] if hit else None)
            if hit:
                self._dragging = True
                self._drag_start = (wx, wy)
                self._orig_geom = copy.deepcopy(hit["geometria"])
                w.setCursor(Qt.SizeAllCursor)
            else:
                self._dragging = False
                self._drag_start = None
                self._orig_geom = None
            w.update()
            return True
        return False

    def on_mouse_move(self, w, e) -> bool:
        if not self._dragging or not w._selected_id:
            return False
        wx, wy = w.screen_to_world(e.x(), e.y())
        dx = wx - self._drag_start[0]
        dy = wy - self._drag_start[1]
        self._sposta_sel(w, dx, dy)
        w.update()
        return True

    def on_mouse_release(self, w, e) -> bool:
        if e.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._drag_start = None
            self._orig_geom = None
            w.setCursor(Qt.ArrowCursor)
            w.elementi_modificati.emit()
            return True
        return False

    def on_key_press(self, w, e) -> bool:
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if w._selected_id:
                w.rimuovi_elemento(w._selected_id)
                return True
        return False

    def _sposta_sel(self, w, dx, dy):
        sid = w._selected_id
        og = self._orig_geom
        if not sid or not og:
            return
        for cat in ("carpenteria", "barre", "staffe"):
            for el in w.get_elementi(cat):
                if el["id"] == sid:
                    g = el["geometria"]
                    tipo = el["tipo"]
                    if tipo in ("rettangolo", "foro_rettangolo"):
                        g["x0"] = og["x0"] + dx
                        g["y0"] = og["y0"] + dy
                        g["x1"] = og["x1"] + dx
                        g["y1"] = og["y1"] + dy
                    elif tipo in ("poligono", "foro_poligono", "staffa"):
                        g["punti"] = [[p[0] + dx, p[1] + dy] for p in og["punti"]]
                    elif tipo in ("cerchio", "foro_cerchio", "barra"):
                        g["cx"] = og["cx"] + dx
                        g["cy"] = og["cy"] + dy
                    return

    def _get_selected(self, w):
        if not w._selected_id:
            return None
        for cat in ("carpenteria", "barre", "staffe"):
            for el in w.get_elementi(cat):
                if el["id"] == w._selected_id:
                    return el
        return None

    def get_properties_text_for(self, el):
        if not el:
            return ""
        g = el["geometria"]
        tipo = el["tipo"]
        if tipo in ("rettangolo", "foro_rettangolo"):
            return (f"V1 ({g['x0']:.1f},{g['y0']:.1f}); "
                    f"V2 ({g['x1']:.1f},{g['y1']:.1f})")
        elif tipo in ("poligono", "foro_poligono"):
            pts = g["punti"]
            return "; ".join(f"V{i+1} ({p[0]:.1f},{p[1]:.1f})" for i, p in enumerate(pts))
        elif tipo in ("cerchio", "foro_cerchio"):
            rx = g.get("rx", g.get("r", 10))
            ry = g.get("ry", rx)
            return (f"C({g['cx']:.1f},{g['cy']:.1f}); "
                    f"rx={rx:.1f}; ry={ry:.1f}")
        elif tipo == "barra":
            return f"Ø={g['r'] * 2:.1f}; C({g['cx']:.1f},{g['cy']:.1f})"
        elif tipo == "staffa":
            pts = g["punti"]
            r = g.get("r", 4)
            parts = [f"Ø={r * 2:.1f}"]
            for i, p in enumerate(pts):
                if i == len(pts) - 1 and p == pts[0]:
                    continue
                parts.append(f"P{i+1} ({p[0]:.1f},{p[1]:.1f})")
            return "; ".join(parts)
        return ""

    def apply_properties_on(self, el, text):
        """Applica proprietà parsate dal lineEdit sull'elemento."""
        g = el["geometria"]
        tipo = el["tipo"]

        if tipo in ("rettangolo", "foro_rettangolo"):
            m1 = re.search(r'V1\s*\(([\d.,-]+),([\d.,-]+)\)', text)
            m2 = re.search(r'V2\s*\(([\d.,-]+),([\d.,-]+)\)', text)
            if m1:
                try:
                    g["x0"] = float(m1.group(1).replace(",", "."))
                    g["y0"] = float(m1.group(2).replace(",", "."))
                except ValueError:
                    pass
            if m2:
                try:
                    g["x1"] = float(m2.group(1).replace(",", "."))
                    g["y1"] = float(m2.group(2).replace(",", "."))
                except ValueError:
                    pass

        elif tipo in ("poligono", "foro_poligono"):
            matches = re.findall(r'V(\d+)\s*\(([\d.,-]+),([\d.,-]+)\)', text)
            for m in matches:
                idx = int(m[0]) - 1
                if 0 <= idx < len(g["punti"]):
                    try:
                        g["punti"][idx][0] = float(m[1].replace(",", "."))
                        g["punti"][idx][1] = float(m[2].replace(",", "."))
                    except ValueError:
                        pass

        elif tipo in ("cerchio", "foro_cerchio"):
            mc = re.search(r'C\s*\(([\d.,-]+),([\d.,-]+)\)', text)
            if mc:
                try:
                    g["cx"] = float(mc.group(1).replace(",", "."))
                    g["cy"] = float(mc.group(2).replace(",", "."))
                except ValueError:
                    pass
            mrx = re.search(r'rx\s*=\s*([\d.]+)', text)
            if mrx:
                try:
                    g["rx"] = float(mrx.group(1))
                except ValueError:
                    pass
            mry = re.search(r'ry\s*=\s*([\d.]+)', text)
            if mry:
                try:
                    g["ry"] = float(mry.group(1))
                except ValueError:
                    pass
            g.pop("r", None)

        elif tipo == "barra":
            md = re.search(r'Ø\s*=\s*([\d.]+)', text)
            if md:
                try:
                    g["r"] = max(0.5, float(md.group(1)) / 2)
                except ValueError:
                    pass
            mc = re.search(r'C\s*\(([\d.,-]+),([\d.,-]+)\)', text)
            if mc:
                try:
                    g["cx"] = float(mc.group(1).replace(",", "."))
                    g["cy"] = float(mc.group(2).replace(",", "."))
                except ValueError:
                    pass

        elif tipo == "staffa":
            md = re.search(r'Ø\s*=\s*([\d.]+)', text)
            if md:
                try:
                    g["r"] = max(0.5, float(md.group(1)) / 2)
                except ValueError:
                    pass
            matches = re.findall(r'P(\d+)\s*\(([\d.,-]+),([\d.,-]+)\)', text)
            for m in matches:
                idx = int(m[0]) - 1
                if 0 <= idx < len(g["punti"]):
                    try:
                        g["punti"][idx][0] = float(m[1].replace(",", "."))
                        g["punti"][idx][1] = float(m[2].replace(",", "."))
                    except ValueError:
                        pass

    def _draw_preview_overlay(self, w, painter, el):
        """Draw selected element in preview/pending style with labels."""
        if not el:
            return
        tipo = el["tipo"]
        g = el["geometria"]

        # Color scheme
        is_foro = tipo.startswith("foro")
        if tipo == "barra":
            col = QColor(255, 200, 50, 220)
            cf = QColor(255, 200, 50, 80)
        elif is_foro:
            col = QColor(255, 120, 50, 220)
            cf = QColor(255, 120, 50, 25)
        else:
            col = QColor(255, 200, 50, 220)
            cf = QColor(255, 200, 50, 25)

        painter.setFont(FONT_LBL)
        fm = painter.fontMetrics()

        if tipo in ("rettangolo", "foro_rettangolo"):
            sx0, sy0 = w.world_to_screen(g["x0"], g["y0"])
            sx1, sy1 = w.world_to_screen(g["x1"], g["y1"])
            pen = QPen(col, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(cf))
            painter.drawRect(min(sx0, sx1), min(sy0, sy1),
                             abs(sx1 - sx0), abs(sy1 - sy0))
            # Vertex dots
            for sx, sy in [(sx0, sy0), (sx1, sy1)]:
                painter.setBrush(QBrush(col))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(sx - 4, sy - 4, 8, 8)
            # Labels
            draw_label(painter, sx0, sy0 + 16,
                       f"V1({g['x0']:.1f},{g['y0']:.1f})", fm)
            draw_label(painter, sx1, sy1 - 8,
                       f"V2({g['x1']:.1f},{g['y1']:.1f})", fm)

        elif tipo in ("cerchio", "foro_cerchio"):
            rx = g.get("rx", g.get("r", 10))
            ry = g.get("ry", rx)
            cx_s, cy_s = w.world_to_screen(g["cx"], g["cy"])
            rx_s, _ = w.world_to_screen(g["cx"] + rx, g["cy"])
            _, ry_s = w.world_to_screen(g["cx"], g["cy"] + ry)
            rpx_x = abs(rx_s - cx_s)
            rpx_y = abs(ry_s - cy_s)
            pen = QPen(col, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(cf))
            painter.drawEllipse(QRectF(cx_s - rpx_x, cy_s - rpx_y,
                                       rpx_x * 2, rpx_y * 2))
            # Center dot
            painter.setBrush(QBrush(col))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(cx_s - 4, cy_s - 4, 8, 8)
            # Labels
            draw_label(painter, cx_s, cy_s + rpx_y + 18,
                       f"C({g['cx']:.1f},{g['cy']:.1f})", fm)
            draw_label(painter, cx_s + rpx_x + 10, cy_s,
                       f"rx={rx:.1f}", fm)
            draw_label(painter, cx_s, cy_s - rpx_y - 10,
                       f"ry={ry:.1f}", fm)

        elif tipo == "barra":
            r = g.get("r", 8)
            cx_s, cy_s = w.world_to_screen(g["cx"], g["cy"])
            rx_s, _ = w.world_to_screen(g["cx"] + r, g["cy"])
            r_px = max(3, abs(rx_s - cx_s))
            pen = QPen(col, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(cf))
            painter.drawEllipse(cx_s - r_px, cy_s - r_px, r_px * 2, r_px * 2)
            draw_label(painter, cx_s, cy_s + r_px + 14,
                       f"Ø{r * 2:.1f}  ({g['cx']:.1f},{g['cy']:.1f})", fm)

        elif tipo in ("poligono", "foro_poligono"):
            pts = g["punti"]
            pts_s = [w.world_to_screen(*p) for p in pts]
            pen = QPen(col, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(cf))
            if len(pts_s) >= 3:
                painter.drawPolygon(QPolygon([QPoint(int(x), int(y)) for x, y in pts_s]))
            elif len(pts_s) == 2:
                painter.drawLine(pts_s[0][0], pts_s[0][1],
                                 pts_s[1][0], pts_s[1][1])
            for i, (sx, sy) in enumerate(pts_s):
                painter.setBrush(QBrush(col))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(sx - 4, sy - 4, 8, 8)
                draw_label(painter, sx + 10, sy - 6,
                           f"V{i+1}({pts[i][0]:.1f},{pts[i][1]:.1f})", fm)

        elif tipo == "staffa":
            pts = g["punti"]
            pts_s = [w.world_to_screen(*p) for p in pts]
            pen = QPen(col, 3, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for i in range(len(pts_s) - 1):
                painter.drawLine(pts_s[i][0], pts_s[i][1],
                                 pts_s[i + 1][0], pts_s[i + 1][1])
            # Point dots and labels (skip last if == first)
            drawn = []
            for i, p in enumerate(pts):
                if i == len(pts) - 1 and p == pts[0]:
                    continue
                drawn.append(i)
            for i in drawn:
                sx, sy = pts_s[i]
                painter.setBrush(QBrush(col))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(sx - 4, sy - 4, 8, 8)
                draw_label(painter, sx + 10, sy - 6,
                           f"P{i+1}({pts[i][0]:.1f},{pts[i][1]:.1f})", fm)

    def draw_painter(self, w, painter):
        el = self._get_selected(w)
        if not el:
            draw_hint(painter, w, "Click = seleziona | Trascina = sposta | Canc = elimina")
            return

        # Draw preview overlay for selected element
        self._draw_preview_overlay(w, painter, el)

        # Info box top-right corner
        tipo = el["tipo"]
        g = el["geometria"]
        if tipo in ("rettangolo", "foro_rettangolo"):
            info = f"[{el['id']}] {tipo} {abs(g['x1'] - g['x0']):.1f}×{abs(g['y1'] - g['y0']):.1f}"
        elif tipo in ("cerchio", "foro_cerchio"):
            rx = g.get("rx", g.get("r", 10))
            ry = g.get("ry", rx)
            info = f"[{el['id']}] {tipo} Rx={rx:.1f} Ry={ry:.1f}"
        elif tipo == "barra":
            info = f"[{el['id']}] barra Ø{g['r'] * 2:.1f}"
        elif tipo in ("poligono", "foro_poligono"):
            info = f"[{el['id']}] {tipo} {len(g['punti'])} vert"
        elif tipo == "staffa":
            info = f"[{el['id']}] staffa Ø{g.get('r', 4) * 2:.1f}"
        else:
            info = f"[{el['id']}]"

        painter.setFont(FONT_LBL)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(info) + 12
        th = fm.height() + 6
        bg = QRectF(w.width() - tw - 10, 10, tw, th)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(30, 30, 32, 210)))
        painter.drawRoundedRect(bg, 4, 4)
        painter.setPen(QPen(self._COL_SEL))
        painter.drawText(bg, Qt.AlignCenter, info)

        draw_hint(painter, w, "Trascina = sposta | Canc = elimina | LineEdit = modifica")
