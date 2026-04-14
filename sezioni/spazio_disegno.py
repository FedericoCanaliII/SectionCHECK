"""
spazio_disegno.py – Widget OpenGL per il disegno interattivo delle sezioni.
- GL code correttamente formattato (no glEnd() dentro for loop)
- Fori: sfondo opaco che "buca" la carpenteria (no griglia sovrapposta)
- Middle mouse pan SEMPRE attivo
- Label editor: QLineEdit unificato, Enter conferma solo l'edit
- Layer order: carpenteria solida → fori → staffe → barre
"""

import math, copy
import numpy as np

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui     import (QPainter, QColor, QPixmap, QPen, QBrush,
                              QFont, QFontMetrics, QPolygon)
from PyQt5.QtCore    import Qt, QPoint, QRectF, QRect, pyqtSignal

from OpenGL.GL import *

_BG_RGB = (40, 40, 40)

# ---------------------------------------------------------------
#  Triangolazione ear-clipping per poligoni concavi
# ---------------------------------------------------------------
def _triangulate(pts):
    """Restituisce lista di triangoli [(a,b,c),...] da un poligono semplice."""
    n = len(pts)
    if n < 3: return []
    if n == 3: return [(pts[0], pts[1], pts[2])]

    def _cross2(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def _in_tri(p, a, b, c):
        d1 = _cross2(a, b, p); d2 = _cross2(b, c, p); d3 = _cross2(c, a, p)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    # Assicura ordine CCW
    area = sum(_cross2(pts[0], pts[i], pts[i+1]) for i in range(1, n-1))
    indices = list(range(n)) if area > 0 else list(reversed(range(n)))

    triangles = []
    limit = n * n + 10
    iters = 0
    while len(indices) > 3 and iters < limit:
        iters += 1
        m = len(indices)
        ear_found = False
        for i in range(m):
            ip = indices[(i-1) % m]; ic = indices[i]; ine = indices[(i+1) % m]
            a, b, c = pts[ip], pts[ic], pts[ine]
            if _cross2(a, b, c) <= 0: continue       # vertice reflex
            is_ear = True
            for j in range(m):
                if j in ((i-1)%m, i, (i+1)%m): continue
                if _in_tri(pts[indices[j]], a, b, c): is_ear = False; break
            if is_ear:
                triangles.append((a, b, c))
                indices.pop(i); ear_found = True; break
        if not ear_found: break
    if len(indices) == 3:
        triangles.append((pts[indices[0]], pts[indices[1]], pts[indices[2]]))
    return triangles

_COLORI = {
    # fill (RGBA) – alpha 178 = 70 % opacità sugli oggetti confermati
    # border (RGBA) – sempre 255 = bordo pieno e solido
    # fori: nessun fill (vengono disegnati col colore dello sfondo in _draw_foro)
    "rettangolo":      ((140, 148, 162, 130), (185, 190, 205, 255)),
    "poligono":        ((140, 148, 162, 130), (185, 190, 205, 255)),
    "cerchio":         ((140, 148, 162, 130), (185, 190, 205, 255)),
    "foro_rettangolo": (None, (120, 100, 100, 255)),
    "foro_poligono":   (None, (120, 100, 100, 255)),
    "foro_cerchio":    (None, (120, 100, 100, 255)),
    "barra":           ((210,  55,  55, 255), (235,  90,  90, 255)),
    "staffa":          (None, (200, 155,  55, 255)),
}
_SEL_BORDER = (255, 220, 50, 255)
_NSEG = 48


def _cat_for(tipo):
    if tipo == "barra":
        return "barre"
    if tipo == "staffa":
        return "staffe"
    return "carpenteria"


class SpazioDisegno(QOpenGLWidget):
    elementi_modificati = pyqtSignal()
    tool_preview_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_range_y = 300.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.last_mouse_pos = None
        self.cursor_pos = QPoint(0, 0)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.show_grid = True
        self.snap_to_grid = True
        self.grid_spacing = 10.0
        self.active_tool = None
        self._middle_pressed = False
        self._elementi = {"carpenteria": [], "barre": [], "staffe": []}
        self._id_counters = {}   # prefix → contatore per nuovi id
        self._selected_id = None
        # Richiedi stencil buffer per clipping fori
        from PyQt5.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        self.setFormat(fmt)

    # --- Coordinate ---
    def _world_bounds(self):
        w, h = max(1, self.width()), max(1, self.height())
        a = w / h
        hh = self.data_range_y * self.zoom
        hw = hh * a
        return (-hw + self.pan_x, hw + self.pan_x,
                -hh + self.pan_y, hh + self.pan_y)

    def screen_to_world(self, sx, sy):
        w, h = max(1, self.width()), max(1, self.height())
        mn_x, mx_x, mn_y, mx_y = self._world_bounds()
        wx = mn_x + (sx / w) * (mx_x - mn_x)
        wy = mn_y + (1 - sy / h) * (mx_y - mn_y)
        if self.snap_to_grid and self.grid_spacing > 0:
            wx = round(wx / self.grid_spacing) * self.grid_spacing
            wy = round(wy / self.grid_spacing) * self.grid_spacing
        return wx, wy

    def world_to_screen(self, wx, wy):
        w, h = max(1, self.width()), max(1, self.height())
        mn_x, mx_x, mn_y, mx_y = self._world_bounds()
        return (int((wx - mn_x) / (mx_x - mn_x or 1) * w),
                int((1 - (wy - mn_y) / (mx_y - mn_y or 1)) * h))

    # --- Elementi ---
    def _prefix_for(self, tipo):
        if tipo.startswith("foro"): return "foro"
        if tipo == "barra": return "barra"
        if tipo == "staffa": return "staffa"
        return "carp"

    def _next_id(self, tipo):
        prefix = self._prefix_for(tipo)
        # Trova il massimo numero esistente per questo prefix
        max_n = self._id_counters.get(prefix, 0)
        for cat in self._elementi.values():
            for el in cat:
                eid = el.get("id", "")
                if eid.startswith(prefix + "_"):
                    try: max_n = max(max_n, int(eid.rsplit("_", 1)[-1]))
                    except ValueError: pass
        n = max_n + 1
        self._id_counters[prefix] = n
        return f"{prefix}_{n}"

    def aggiungi_elemento(self, el):
        if "id" not in el or not el["id"]:
            el["id"] = self._next_id(el["tipo"])
        self._elementi[_cat_for(el["tipo"])].append(el)
        self.update()
        self.elementi_modificati.emit()

    def rimuovi_elemento(self, eid):
        for cat in self._elementi:
            self._elementi[cat] = [e for e in self._elementi[cat] if e["id"] != eid]
        if self._selected_id == eid:
            self._selected_id = None
        self.update()
        self.elementi_modificati.emit()

    def get_elementi(self, cat):
        return self._elementi.get(cat, [])

    def get_tutti_elementi(self):
        return copy.deepcopy(self._elementi)

    def carica_elementi(self, elementi):
        self._elementi = {
            "carpenteria": list(elementi.get("carpenteria", [])),
            "barre": list(elementi.get("barre", [])),
            "staffe": list(elementi.get("staffe", [])),
        }
        self._id_counters = {}
        self._selected_id = None
        self._auto_fit()
        self.update()

    def set_selected(self, eid):
        self._selected_id = eid
        self.update()

    def reset_elementi(self):
        self._elementi = {"carpenteria": [], "barre": [], "staffe": []}
        self._id_counters = {}
        self._selected_id = None
        self.update()

    def _auto_fit(self):
        bb = self._bounding_box()
        if not bb:
            self.pan_x = 0
            self.pan_y = 0
            self.zoom = 1
            return
        bx0, bx1, by0, by1 = bb
        self.pan_x = (bx0 + bx1) / 2
        self.pan_y = (by0 + by1) / 2
        rx = bx1 - bx0
        ry = by1 - by0
        m = 1.3
        w, h = max(1, self.width()), max(1, self.height())
        a = w / h
        zy = (ry * m / 2) / self.data_range_y if ry > 1e-6 else 0.5
        zx = (rx * m / 2) / (self.data_range_y * a) if rx > 1e-6 else 0.5
        self.zoom = max(zx, zy, 0.1)

    def _bounding_box(self):
        ax, ay = [], []
        for cat in self._elementi.values():
            for el in cat:
                g = el["geometria"]
                t = el["tipo"]
                if t in ("rettangolo", "foro_rettangolo"):
                    ax += [g["x0"], g["x1"]]
                    ay += [g["y0"], g["y1"]]
                elif t in ("poligono", "foro_poligono", "staffa"):
                    for p in g["punti"]:
                        ax.append(p[0])
                        ay.append(p[1])
                elif t in ("cerchio", "foro_cerchio", "barra"):
                    rx = g.get("rx", g.get("r", 0))
                    ry = g.get("ry", rx)
                    ax += [g["cx"] - rx, g["cx"] + rx]
                    ay += [g["cy"] - ry, g["cy"] + ry]
        if not ax:
            return None
        return (min(ax), max(ax), min(ay), max(ay))

    # ========================================
    #  GL
    # ========================================
    def initializeGL(self):
        glClearColor(_BG_RGB[0] / 255, _BG_RGB[1] / 255, _BG_RGB[2] / 255, 1.0)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def reset_view(self):
        self._auto_fit()
        self.update()

    def paintGL(self):
        # Qt5 resetta lo stato GL ogni frame quando QPainter viene usato nella
        # stessa paintGL → bisogna ri-abilitare il blend ad ogni chiamata.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        mn_x, mx_x, mn_y, mx_y = self._world_bounds()
        glOrtho(mn_x, mx_x, mn_y, mx_y, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 1. Griglia di fondo
        self._draw_grid()
        # 2. Carpenteria solida (rettangoli, poligoni, cerchi non-foro)
        self._draw_layer_solida()
        # 3. Fori: riempi con colore sfondo per "bucare"
        self._draw_layer_fori()
        # 4. Staffe
        self._draw_layer("staffe")
        # 5. Barre (sopra tutto)
        self._draw_layer("barre")

        # Tool GL overlay
        if self.active_tool:
            try:
                self.active_tool.draw_gl(self)
            except Exception:
                pass

        # QPainter overlays
        self._draw_labels_axes()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            self._draw_id_labels_painter(painter)
            if self.active_tool:
                try:
                    self.active_tool.draw_painter(self, painter)
                except Exception:
                    pass
        finally:
            painter.end()
        self._draw_tracker()

    # --- ID Labels ---
    def _draw_id_labels_painter(self, painter):
        """Disegna etichetta ID sotto il centro di ogni elemento confermato."""
        from .tools.tool_modifica import ToolModifica
        skip_id = self._selected_id if isinstance(self.active_tool, ToolModifica) else None
        painter.setFont(QFont("Consolas", 7))
        fm = painter.fontMetrics()
        for cat in ("carpenteria", "barre", "staffe"):
            for el in self._elementi[cat]:
                if el.get("id") == skip_id:
                    continue
                label = el["id"]
                cx_w, cy_w, ry_w = self._el_center_and_radius(el)
                cx_s, cy_s = self.world_to_screen(cx_w, cy_w)
                _, bot_s = self.world_to_screen(cx_w, cy_w - ry_w)
                offset_y = max(abs(bot_s - cy_s), 4) + 3
                tw = fm.horizontalAdvance(label) + 8
                th = fm.height() + 4
                rx = tw // 2
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(20, 20, 22, 180)))
                painter.drawRoundedRect(QRectF(cx_s - rx, cy_s + offset_y, tw, th), 2, 2)
                painter.setPen(QPen(QColor(160, 160, 190)))
                painter.drawText(QRectF(cx_s - rx, cy_s + offset_y, tw, th), Qt.AlignCenter, label)

    def _el_center_and_radius(self, el):
        """Ritorna (cx_world, cy_world, ry_world) per il posizionamento dell'ID label."""
        t = el["tipo"]
        g = el["geometria"]
        if t in ("rettangolo", "foro_rettangolo"):
            cx = (g["x0"] + g["x1"]) / 2
            cy = (g["y0"] + g["y1"]) / 2
            ry = abs(g["y1"] - g["y0"]) / 2
            return cx, cy, ry
        elif t in ("cerchio", "foro_cerchio", "barra"):
            ry = float(g.get("ry", g.get("rx", g.get("r", 10))))
            return float(g["cx"]), float(g["cy"]), ry
        elif t in ("poligono", "foro_poligono", "staffa"):
            pts = g["punti"]
            if not pts:
                return 0.0, 0.0, 10.0
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            ry = max(abs(p[1] - cy) for p in pts) if pts else 10.0
            return cx, cy, ry
        return 0.0, 0.0, 10.0

    # --- Grid ---
    def _draw_grid(self):
        if not self.show_grid or self.grid_spacing <= 0:
            return
        mn_x, mx_x, mn_y, mx_y = self._world_bounds()
        sp = self.grid_spacing
        glColor3f(0.19, 0.19, 0.19)
        glLineWidth(1.0)
        x = np.floor(mn_x / sp) * sp
        while x <= mx_x:
            glBegin(GL_LINES)
            glVertex2f(float(x), mn_y)
            glVertex2f(float(x), mx_y)
            glEnd()
            x += sp
        y = np.floor(mn_y / sp) * sp
        while y <= mx_y:
            glBegin(GL_LINES)
            glVertex2f(mn_x, float(y))
            glVertex2f(mx_x, float(y))
            glEnd()
            y += sp

    # --- Layer carpenteria solida ---
    def _draw_layer_solida(self):
        for el in self._elementi["carpenteria"]:
            if not el["tipo"].startswith("foro"):
                self._draw_el(el)

    # --- Layer fori (sfondo opaco = buca la carpenteria) ---
    def _draw_layer_fori(self):
        for el in self._elementi["carpenteria"]:
            if el["tipo"].startswith("foro"):
                self._draw_foro(el)

    def _draw_foro(self, el):
        from .tools.tool_modifica import ToolModifica
        g = el["geometria"]
        t = el["tipo"]
        is_mod = isinstance(self.active_tool, ToolModifica)
        sel = (el.get("id") == self._selected_id) and not is_mod
        bg = (_BG_RGB[0], _BG_RGB[1], _BG_RGB[2], 255)

        # Fill con sfondo per bucare
        if t == "foro_rettangolo":
            x0, y0, x1, y1 = g["x0"], g["y0"], g["x1"], g["y1"]
            glColor4ub(*bg)
            glBegin(GL_QUADS)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()
        elif t == "foro_poligono":
            pts = g["punti"]
            if len(pts) >= 3:
                glColor4ub(*bg)
                pts_f = [(float(p[0]), float(p[1])) for p in pts]
                for tri in _triangulate(pts_f):
                    glBegin(GL_TRIANGLES)
                    for v in tri:
                        glVertex2f(v[0], v[1])
                    glEnd()
        elif t == "foro_cerchio":
            cx, cy = float(g["cx"]), float(g["cy"])
            rx = float(g.get("rx", g.get("r", 10)))
            ry = float(g.get("ry", rx))
            glColor4ub(*bg)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(_NSEG + 1):
                a = 2 * math.pi * i / _NSEG
                glVertex2f(cx + rx * math.cos(a), cy + ry * math.sin(a))
            glEnd()

        # Ridisegna la griglia SOLO dentro l'area del foro usando clipping
        if self.show_grid and self.grid_spacing > 0:
            self._draw_grid_clipped_to_foro(el)

        # Bordo del foro
        bc = _SEL_BORDER if sel else _COLORI.get(t, (None, (170, 170, 185, 255)))[1]
        glColor4ub(*bc)
        glLineWidth(2.5 if sel else 1.5)
        if t == "foro_rettangolo":
            x0, y0, x1, y1 = g["x0"], g["y0"], g["x1"], g["y1"]
            glBegin(GL_LINE_LOOP)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()
        elif t == "foro_poligono":
            pts = g["punti"]
            if len(pts) >= 3:
                glBegin(GL_LINE_LOOP)
                for p in pts:
                    glVertex2f(float(p[0]), float(p[1]))
                glEnd()
        elif t == "foro_cerchio":
            cx, cy = float(g["cx"]), float(g["cy"])
            rx = float(g.get("rx", g.get("r", 10)))
            ry = float(g.get("ry", rx))
            glBegin(GL_LINE_LOOP)
            for i in range(_NSEG):
                a = 2 * math.pi * i / _NSEG
                glVertex2f(cx + rx * math.cos(a), cy + ry * math.sin(a))
            glEnd()

    def _draw_grid_clipped_to_foro(self, el):
        """Disegna la griglia solo dentro l'area esatta del foro usando stencil buffer."""
        if not self.show_grid or self.grid_spacing <= 0:
            return
        g = el["geometria"]
        t = el["tipo"]

        # 1. Scrivi la forma nel stencil buffer (disabilita color write)
        glEnable(GL_STENCIL_TEST)
        glStencilMask(0xFF)
        glClear(GL_STENCIL_BUFFER_BIT)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glStencilFunc(GL_ALWAYS, 1, 0xFF)
        glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE)

        if t == "foro_rettangolo":
            x0, y0, x1, y1 = g["x0"], g["y0"], g["x1"], g["y1"]
            glBegin(GL_QUADS)
            glVertex2f(x0, y0); glVertex2f(x1, y0)
            glVertex2f(x1, y1); glVertex2f(x0, y1)
            glEnd()
        elif t == "foro_poligono":
            pts = g["punti"]
            if len(pts) >= 3:
                pts_f = [(float(p[0]), float(p[1])) for p in pts]
                for tri in _triangulate(pts_f):
                    glBegin(GL_TRIANGLES)
                    for v in tri:
                        glVertex2f(v[0], v[1])
                    glEnd()
        elif t == "foro_cerchio":
            cx, cy = float(g["cx"]), float(g["cy"])
            rx = float(g.get("rx", g.get("r", 10)))
            ry = float(g.get("ry", rx))
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(_NSEG + 1):
                a = 2 * math.pi * i / _NSEG
                glVertex2f(cx + rx * math.cos(a), cy + ry * math.sin(a))
            glEnd()
        else:
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            glDisable(GL_STENCIL_TEST)
            return

        # 2. Disegna la griglia solo dove stencil == 1
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glStencilFunc(GL_EQUAL, 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        self._draw_grid()

        # 3. Pulizia stencil
        glStencilMask(0xFF)
        glClear(GL_STENCIL_BUFFER_BIT)
        glDisable(GL_STENCIL_TEST)

    # --- Layer generica ---
    def _draw_layer(self, cat):
        for el in self._elementi[cat]:
            self._draw_el(el)

    def _draw_el(self, el):
        from .tools.tool_modifica import ToolModifica
        t = el["tipo"]
        g = el["geometria"]
        # ToolModifica gestisce il highlight via QPainter overlay
        is_mod = isinstance(self.active_tool, ToolModifica)
        sel = (el.get("id") == self._selected_id) and not is_mod
        fc, bc = _COLORI.get(t, ((130, 130, 140, 130), (170, 170, 185, 255)))
        if t in ("rettangolo", "foro_rettangolo"):
            self._gl_rect(g, fc, bc, sel)
        elif t in ("poligono", "foro_poligono"):
            self._gl_poly(g["punti"], fc, bc, sel)
        elif t in ("cerchio", "foro_cerchio", "barra"):
            self._gl_ellipse(g, fc, bc, sel)
        elif t == "staffa":
            self._gl_staffa(g, bc, sel)

    def _gl_rect(self, g, fill, border, sel):
        x0, y0, x1, y1 = g["x0"], g["y0"], g["x1"], g["y1"]
        if fill:
            glColor4ub(*fill)
            glBegin(GL_QUADS)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()
        bc = _SEL_BORDER if sel else border
        glColor4ub(*bc)
        glLineWidth(2.5 if sel else 1.5)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x0, y0)
        glVertex2f(x1, y0)
        glVertex2f(x1, y1)
        glVertex2f(x0, y1)
        glEnd()

    def _gl_poly(self, punti, fill, border, sel):
        if len(punti) < 3:
            return
        if fill:
            glColor4ub(*fill)
            pts = [(float(p[0]), float(p[1])) for p in punti]
            for tri in _triangulate(pts):
                glBegin(GL_TRIANGLES)
                for v in tri:
                    glVertex2f(v[0], v[1])
                glEnd()
        bc = _SEL_BORDER if sel else border
        glColor4ub(*bc)
        glLineWidth(2.5 if sel else 1.5)
        glBegin(GL_LINE_LOOP)
        for p in punti:
            glVertex2f(float(p[0]), float(p[1]))
        glEnd()

    def _gl_ellipse(self, g, fill, border, sel):
        cx = float(g["cx"])
        cy = float(g["cy"])
        rx = float(g.get("rx", g.get("r", 10)))
        ry = float(g.get("ry", rx))
        if fill:
            glColor4ub(*fill)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(_NSEG + 1):
                a = 2 * math.pi * i / _NSEG
                glVertex2f(cx + rx * math.cos(a), cy + ry * math.sin(a))
            glEnd()
        bc = _SEL_BORDER if sel else border
        glColor4ub(*bc)
        glLineWidth(2.5 if sel else 1.5)
        glBegin(GL_LINE_LOOP)
        for i in range(_NSEG):
            a = 2 * math.pi * i / _NSEG
            glVertex2f(cx + rx * math.cos(a), cy + ry * math.sin(a))
        glEnd()

    def _gl_staffa(self, g, border, sel):
        pts = g["punti"]
        if len(pts) < 2:
            return
        r = float(g.get("r", 4))
        bc = _SEL_BORDER if sel else border
        glColor4ub(*bc)
        # Segmenti come quads spessi
        for i in range(len(pts) - 1):
            x1, y1 = float(pts[i][0]), float(pts[i][1])
            x2, y2 = float(pts[i + 1][0]), float(pts[i + 1][1])
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            px = -dy / length * r
            py = dx / length * r
            glBegin(GL_QUADS)
            glVertex2f(x1 + px, y1 + py)
            glVertex2f(x1 - px, y1 - py)
            glVertex2f(x2 - px, y2 - py)
            glVertex2f(x2 + px, y2 + py)
            glEnd()
        # Cerchi ai giunti per un aspetto continuo
        for p in pts:
            cx, cy = float(p[0]), float(p[1])
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(_NSEG + 1):
                a = 2 * math.pi * i / _NSEG
                glVertex2f(cx + r * math.cos(a), cy + r * math.sin(a))
            glEnd()

    # --- Labels / Axes ---
    def _draw_labels_axes(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        f = painter.font()
        f.setPointSize(8)
        painter.setFont(f)
        m = painter.fontMetrics()
        w, h = self.width(), self.height()
        mn_x, mx_x, mn_y, mx_y = self._world_bounds()

        def to_s(wx, wy):
            rng_x = mx_x - mn_x or 1
            rng_y = mx_y - mn_y or 1
            return (int((wx - mn_x) / rng_x * w),
                    int((1 - (wy - mn_y) / rng_y) * h))

        sp = self.grid_spacing
        if sp > 0:
            vx = mx_x - mn_x
            vy = mx_y - mn_y
            sx = sp
            sy = sp
            while vx / sx > 40:
                sx *= 2
            while vy / sy > 40:
                sy *= 2
            painter.setPen(QColor(255, 100, 100, 140))
            x = np.floor(mn_x / sx) * sx
            while x <= mx_x:
                if abs(x) > 1e-6:
                    px, oy = to_s(x, 0)
                    lbl = f"{x:.4g}"
                    tw = m.horizontalAdvance(lbl)
                    painter.drawLine(px, oy - 4, px, oy + 4)
                    painter.drawText(px - tw // 2, oy + m.height() + 2, lbl)
                x += sx
            painter.setPen(QColor(100, 255, 100, 140))
            y = np.floor(mn_y / sy) * sy
            while y <= mx_y:
                if abs(y) > 1e-6:
                    ox, py = to_s(0, y)
                    lbl = f"{y:.4g}"
                    tw = m.horizontalAdvance(lbl)
                    painter.drawLine(ox - 4, py, ox + 4, py)
                    painter.drawText(ox - tw - 6, py + m.height() // 2 - 2, lbl)
                y += sy

        # Assi
        rng_x = mx_x - mn_x or 1
        rng_y = mx_y - mn_y or 1
        ox = int((0 - mn_x) / rng_x * w)
        oy = int((1 - (0 - mn_y) / rng_y) * h)
        pen = QPen(QColor(255, 0, 0), 1)
        painter.setPen(pen)
        painter.drawLine(0, oy, w, oy)
        pen.setColor(QColor(0, 255, 0))
        painter.setPen(pen)
        painter.drawLine(ox, 0, ox, h)
        painter.end()

    def _draw_tracker(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = self.cursor_pos.x(), self.cursor_pos.y()
        mn_x, mx_x, mn_y, mx_y = self._world_bounds()
        rng_x = mx_x - mn_x or 1
        rng_y = mx_y - mn_y or 1
        nx = cx / max(1, w)
        ny = 1 - cy / max(1, h)
        wx = mn_x + nx * rng_x
        wy = mn_y + ny * rng_y
        if self.snap_to_grid and self.grid_spacing > 0:
            wx = round(wx / self.grid_spacing) * self.grid_spacing
            wy = round(wy / self.grid_spacing) * self.grid_spacing
        sx = int((wx - mn_x) / rng_x * w)
        sy = int((1 - (wy - mn_y) / rng_y) * h)
        pen = QPen(QColor(255, 255, 255, 255), 1)
        painter.setPen(pen)
        painter.drawLine(sx - 10, sy, sx + 10, sy)
        painter.drawLine(sx, sy - 10, sx, sy + 10)
        pen.setColor(QColor(100, 100, 100, 150))
        painter.setPen(pen)
        painter.drawLine(sx, 0, sx, h)
        painter.drawLine(0, sy, w, sy)
        pen.setColor(QColor(255, 255, 255, 155))
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(sx + 8, h - 8, f"X: {wx:.4g}")
        painter.drawText(8, sy - 8, f"Y: {wy:.4g}")
        painter.end()

    # --- API ---
    def set_show_grid(self, v):
        self.show_grid = v
        self.update()

    def set_grid_spacing(self, v):
        self.grid_spacing = v
        self.update()

    def set_snap_to_grid(self, v):
        self.snap_to_grid = v

    def set_active_tool(self, tool):
        if self.active_tool is tool:
            return
        if self.active_tool:
            try:
                self.active_tool.on_deactivate(self)
            except Exception:
                pass
        self.active_tool = tool
        if tool:
            try:
                tool.on_activate(self)
            except Exception:
                pass
        self.setFocus()
        self._emit_preview()
        self.update()

    def _emit_preview(self):
        if not self.active_tool:
            self.tool_preview_changed.emit("")
            return
        from .tools.tool_modifica import ToolModifica
        if isinstance(self.active_tool, ToolModifica):
            el = self.active_tool._get_selected(self)
            txt = self.active_tool.get_properties_text_for(el) if el else ""
        else:
            txt = self.active_tool.get_properties_text()
        self.tool_preview_changed.emit(txt)

    # ========================================
    #  MOUSE  –  middle pan SEMPRE prima
    # ========================================
    def mousePressEvent(self, e):
        self.cursor_pos = e.pos()
        # MIDDLE MOUSE PAN: sempre, prima di tutto
        if e.button() == Qt.MiddleButton:
            self._middle_pressed = True
            self.last_mouse_pos = e.pos()
            return
        self.setFocus()
        # Tool
        if self.active_tool:
            try:
                if self.active_tool.on_mouse_press(self, e):
                    self.update()
                    self._emit_preview()
                    return
            except Exception as ex:
                print(f"WARN tool press: {ex}")
        self.update()

    def mouseMoveEvent(self, e):
        self.cursor_pos = e.pos()
        if self._middle_pressed and self.last_mouse_pos:
            dx = e.x() - self.last_mouse_pos.x()
            dy = e.y() - self.last_mouse_pos.y()
            w, h = max(1, self.width()), max(1, self.height())
            mn_x, mx_x, mn_y, mx_y = self._world_bounds()
            self.pan_x -= dx * (mx_x - mn_x) / w
            self.pan_y += dy * (mx_y - mn_y) / h
            self.last_mouse_pos = e.pos()
            self.update()
            return
        if self.active_tool:
            try:
                if self.active_tool.on_mouse_move(self, e):
                    self.update()
                    return
            except Exception:
                pass
        self.update()

    def mouseReleaseEvent(self, e):
        self.cursor_pos = e.pos()
        if e.button() == Qt.MiddleButton:
            self._middle_pressed = False
            self.last_mouse_pos = None
            return
        if self.active_tool:
            try:
                if self.active_tool.on_mouse_release(self, e):
                    self.update()
                    self._emit_preview()
                    return
            except Exception:
                pass
        self.update()

    def wheelEvent(self, e):
        if self.active_tool:
            try:
                if self.active_tool.on_wheel(self, e):
                    self.update()
                    return
            except Exception:
                pass
        f = 1.15
        if e.angleDelta().y() > 0:
            self.zoom = self.zoom / f
        else:
            self.zoom = self.zoom * f
        self.update()

    def keyPressEvent(self, e):
        if self.active_tool:
            try:
                if self.active_tool.on_key_press(self, e):
                    self.update()
                    self._emit_preview()
                    return
            except Exception as ex:
                print(f"WARN key: {ex}")
        super().keyPressEvent(e)

    # --- Thumbnail ---
    def genera_thumbnail(self, width=180, height=65):
        px = QPixmap(width, height)
        px.fill(QColor(50,50,50))
        painter = QPainter(px)
        painter.setRenderHint(QPainter.Antialiasing)
        bb = self._bounding_box()
        if not bb:
            painter.end()
            return px
        bx0, bx1, by0, by1 = bb
        rx = bx1 - bx0 or 1
        ry = by1 - by0 or 1
        sc = min((width - 12) / rx, (height - 12) / ry)
        cx_w = (bx0 + bx1) / 2
        cy_w = (by0 + by1) / 2

        def tp(wx, wy):
            return (int((wx - cx_w) * sc + width / 2),
                    int(-(wy - cy_w) * sc + height / 2))

        _FORO_BG = QColor(50, 50, 50)

        # Passata 1: carpenteria solida
        for el in self._elementi["carpenteria"]:
            t = el["tipo"]
            if t.startswith("foro"):
                continue
            g = el["geometria"]
            fc, bc_rgba = _COLORI.get(t, ((130, 130, 140, 130), (170, 170, 185, 255)))
            bc = QColor(*bc_rgba)
            if t == "rettangolo":
                p0 = tp(g["x0"], g["y1"]); p1 = tp(g["x1"], g["y0"])
                r = QRect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1])
                painter.fillRect(r, QColor(*fc))
                painter.setPen(QPen(bc, 1)); painter.setBrush(Qt.NoBrush)
                painter.drawRect(r)
            elif t == "poligono":
                pts = [QPoint(*tp(*p)) for p in g["punti"]]
                painter.setBrush(QBrush(QColor(*fc)))
                painter.setPen(QPen(bc, 1))
                painter.drawPolygon(QPolygon(pts))
            elif t == "cerchio":
                sx, sy = tp(g["cx"], g["cy"])
                r_px = max(1, int(g.get("rx", g.get("r", 10)) * sc))
                painter.setBrush(QBrush(QColor(*fc)))
                painter.setPen(QPen(bc, 1))
                painter.drawEllipse(sx - r_px, sy - r_px, r_px * 2, r_px * 2)

        # Passata 2: fori (sfondo pieno per bucare la carpenteria)
        for el in self._elementi["carpenteria"]:
            t = el["tipo"]
            if not t.startswith("foro"):
                continue
            g = el["geometria"]
            _, bc_rgba = _COLORI.get(t, (None, (120, 100, 100, 255)))
            bc = QColor(*bc_rgba)
            if t == "foro_rettangolo":
                p0 = tp(g["x0"], g["y1"]); p1 = tp(g["x1"], g["y0"])
                r = QRect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1])
                painter.fillRect(r, _FORO_BG)
                painter.setPen(QPen(bc, 1)); painter.setBrush(Qt.NoBrush)
                painter.drawRect(r)
            elif t == "foro_poligono":
                pts = [QPoint(*tp(*p)) for p in g["punti"]]
                painter.setBrush(QBrush(_FORO_BG))
                painter.setPen(QPen(bc, 1))
                painter.drawPolygon(QPolygon(pts))
            elif t == "foro_cerchio":
                sx, sy = tp(g["cx"], g["cy"])
                r_px = max(1, int(g.get("rx", g.get("r", 10)) * sc))
                painter.setBrush(QBrush(_FORO_BG))
                painter.setPen(QPen(bc, 1))
                painter.drawEllipse(sx - r_px, sy - r_px, r_px * 2, r_px * 2)

        # Passata 3: staffe e barre
        for cat in ("staffe", "barre"):
            for el in self._elementi[cat]:
                t = el["tipo"]
                g = el["geometria"]
                fc, bc_rgba = _COLORI.get(t, ((130, 130, 140, 130), (170, 170, 185, 255)))
                bc = QColor(*bc_rgba)
                if t == "barra":
                    sx, sy = tp(g["cx"], g["cy"])
                    r_px = max(1, int(g.get("r", 8) * sc))
                    painter.setBrush(QBrush(QColor(*fc)))
                    painter.setPen(QPen(bc, 1))
                    painter.drawEllipse(sx - r_px, sy - r_px, r_px * 2, r_px * 2)
                elif t == "staffa":
                    painter.setPen(QPen(QColor(*bc_rgba), 1))
                    painter.setBrush(Qt.NoBrush)
                    pts = [tp(*p) for p in g["punti"]]
                    for i in range(len(pts) - 1):
                        painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

        painter.end()
        return px
