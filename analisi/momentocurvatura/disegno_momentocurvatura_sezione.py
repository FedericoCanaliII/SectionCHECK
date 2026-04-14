"""
disegno_momentocurvatura_sezione.py  –  analisi/momentocurvatura/
==================================================================
Widget OpenGL 2D per la visualizzazione della geometria della sezione
nel pannello Momento-Curvatura (anteprima sezione selezionata).

Identico per funzionalità al widget disegno_dominio_sezione del modulo
dominio_nm, specializzato qui per evitare dipendenze incrociate.

Funzionalità:
  - Rendering di forme carpenteria (rettangolo, poligono, cerchio/ellisse)
  - Fori disegnati con il colore dello sfondo (effetto buco)
  - Barre disegnate come cerchi pieni rossi
  - Pan con tasto sinistro, zoom con rotella
  - Griglia e assi sovrapposti con QPainter
  - Auto-fit automatico alla selezione di una sezione
  - reset_view() per centrare/rieseguire il fit
"""
from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore    import Qt, QPoint
from PyQt5.QtGui     import QColor, QFont, QPainter, QPen, QSurfaceFormat
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    GL_ALWAYS, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_TEST,
    GL_EQUAL, GL_FALSE, GL_KEEP, GL_LINE_LOOP, GL_LINE_SMOOTH, GL_LINE_STRIP,
    GL_LINES, GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_POINT_SMOOTH,
    GL_PROJECTION, GL_REPLACE, GL_SRC_ALPHA, GL_STENCIL_BUFFER_BIT,
    GL_STENCIL_TEST, GL_TEXTURE_2D, GL_TRIANGLE_FAN, GL_TRIANGLES, GL_TRUE,
    glBegin, glBlendFunc, glClear, glClearColor, glColor3f, glColor4f,
    glColorMask, glDisable, glEnable, glEnd, glLineWidth, glLoadIdentity,
    glMatrixMode, glOrtho, glStencilFunc, glStencilMask, glStencilOp,
    glVertex2f, glViewport,
)


# ==============================================================================
# COSTANTI STILE
# ==============================================================================
_BG_COLOR     = (40 / 255, 40 / 255, 40 / 255, 1.0)
_GRID_COLOR   = (0.19, 0.19, 0.19)

_CLS_FILL     = (140 / 255, 148 / 255, 162 / 255, 130 / 255)
_CLS_OUTLINE  = (185 / 255, 190 / 255, 205 / 255, 1.0)
_CLS_LINE_W   = 1.5

_FORO_FILL    = (40 / 255, 40 / 255, 40 / 255, 1.0)
_FORO_OUTLINE = (120 / 255, 100 / 255, 100 / 255, 1.0)
_FORO_LINE_W  = 1.5

_BAR_FILL     = (210 / 255,  55 / 255,  55 / 255, 1.0)
_BAR_OUTLINE  = (235 / 255,  90 / 255,  90 / 255, 1.0)
_BAR_LINE_W   = 1.5
_BAR_SEG      = 48

_STAFFA_OUTLINE = (200 / 255, 155 / 255, 55 / 255, 1.0)
_STAFFA_LINE_W  = 1.5

_AX_X   = QColor(220, 60,  60)
_AX_Y   = QColor(60,  200, 60)
_TXT    = QColor(200, 200, 200)
_CURSOR = QColor(255, 255, 255, 180)
_FONT   = QFont("Arial", 9)


# ==============================================================================
# HELPER TRIANGOLAZIONE
# ==============================================================================

def _triangola_poligono(pts: list) -> list:
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(pts[0], pts[1], pts[2])]

    def cross2(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def in_tri(p, a, b, c):
        d1 = cross2(a, b, p)
        d2 = cross2(b, c, p)
        d3 = cross2(c, a, p)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    area_s  = sum(cross2(pts[0], pts[i], pts[i + 1]) for i in range(1, n - 1))
    indices = list(range(n)) if area_s > 0 else list(reversed(range(n)))

    tris  = []
    limit = n * n + 10
    iters = 0
    while len(indices) > 3 and iters < limit:
        iters += 1
        m = len(indices)
        found = False
        for i in range(m):
            ip = indices[(i - 1) % m]
            ic = indices[i]
            ne = indices[(i + 1) % m]
            a, b, c = pts[ip], pts[ic], pts[ne]
            if cross2(a, b, c) <= 0:
                continue
            ear = all(
                not in_tri(pts[indices[j]], a, b, c)
                for j in range(m)
                if j not in ((i - 1) % m, i, (i + 1) % m)
            )
            if ear:
                tris.append((a, b, c))
                indices.pop(i)
                found = True
                break
        if not found:
            break
    if len(indices) == 3:
        tris.append((pts[indices[0]], pts[indices[1]], pts[indices[2]]))
    return tris


# ==============================================================================
# WIDGET
# ==============================================================================

class MomentoCurvaturaSezioneWidget(QOpenGLWidget):
    """Widget OpenGL per la visualizzazione della sezione nel pannello M-χ."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._shapes:  list = []
        self._bars:    list = []
        self._staffe:  list = []

        self.pan_x  = 0.0
        self.pan_y  = 0.0
        self.zoom   = 100.0

        self._last_pos = None
        self._cursor   = QPoint(0, 0)

        self.wx_min = self.wx_max = 0.0
        self.wy_min = self.wy_max = 0.0

        self._section_data_cache: dict | None = None
        self.setMouseTracking(True)

        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        self.setFormat(fmt)

    # ------------------------------------------------------------------
    # API PUBBLICA
    # ------------------------------------------------------------------

    def set_section_data(self, section_data: dict | None) -> None:
        self._section_data_cache = section_data
        self._shapes.clear()
        self._bars.clear()
        self._staffe.clear()

        if not section_data or 'elementi' not in section_data:
            self.zoom = 100.0
            self.pan_x = self.pan_y = 0.0
            self.update()
            return

        bb_pts: list = []
        elementi    = section_data['elementi']
        carpenteria = elementi.get('carpenteria', [])
        barre       = elementi.get('barre', [])
        staffe      = elementi.get('staffe', [])

        for elem in carpenteria:
            tipo    = elem.get('tipo', '')
            geom    = elem.get('geometria', {})
            is_foro = tipo.startswith('foro_')
            base    = tipo.replace('foro_', '')
            fill    = _FORO_FILL    if is_foro else _CLS_FILL
            outline = _FORO_OUTLINE if is_foro else _CLS_OUTLINE
            lw      = _FORO_LINE_W  if is_foro else _CLS_LINE_W

            if base == 'rettangolo':
                x0, y0 = geom['x0'], geom['y0']
                x1, y1 = geom['x1'], geom['y1']
                pts = [(min(x0, x1), min(y0, y1)),
                       (max(x0, x1), min(y0, y1)),
                       (max(x0, x1), max(y0, y1)),
                       (min(x0, x1), max(y0, y1))]
                tris = _triangola_poligono(pts)
                self._shapes.append({
                    'type': 'poly', 'pts': pts, 'tris': tris,
                    'fill': fill, 'outline': outline, 'lw': lw,
                    'is_foro': is_foro,
                })
                bb_pts.extend(pts)

            elif base == 'poligono':
                raw = geom.get('punti', [])
                pts = [(float(p[0]), float(p[1])) for p in raw]
                if len(pts) >= 3:
                    tris = _triangola_poligono(pts)
                    self._shapes.append({
                        'type': 'poly', 'pts': pts, 'tris': tris,
                        'fill': fill, 'outline': outline, 'lw': lw,
                        'is_foro': is_foro,
                    })
                    bb_pts.extend(pts)

            elif base == 'cerchio':
                cx = float(geom.get('cx', 0.0))
                cy = float(geom.get('cy', 0.0))
                rx = float(geom.get('rx', geom.get('r', 1.0)))
                ry = float(geom.get('ry', geom.get('r', 1.0)))
                self._shapes.append({
                    'type': 'ellipse', 'cx': cx, 'cy': cy,
                    'rx': rx, 'ry': ry,
                    'fill': fill, 'outline': outline, 'lw': lw,
                    'is_foro': is_foro,
                })
                bb_pts += [(cx - rx, cy - ry), (cx + rx, cy + ry)]

        for elem in barre:
            if elem.get('tipo') != 'barra':
                continue
            g  = elem.get('geometria', {})
            cx = float(g.get('cx', 0.0))
            cy = float(g.get('cy', 0.0))
            r  = float(g.get('r',  0.0))
            if r > 0:
                self._bars.append({'cx': cx, 'cy': cy, 'r': r})
                bb_pts += [(cx - r, cy - r), (cx + r, cy + r)]

        for elem in staffe:
            if elem.get('tipo') != 'staffa':
                continue
            g   = elem.get('geometria', {})
            pts = [(float(p[0]), float(p[1])) for p in g.get('punti', [])]
            if len(pts) >= 2:
                self._staffe.append({'punti': pts})
                bb_pts.extend(pts)

        # Auto-fit
        if bb_pts:
            xs = [p[0] for p in bb_pts]
            ys = [p[1] for p in bb_pts]
            w  = max(xs) - min(xs)
            h  = max(ys) - min(ys)
            self.pan_x = (min(xs) + max(xs)) / 2.0
            self.pan_y = (min(ys) + max(ys)) / 2.0
            self.zoom  = max(w, h) / 2.0 * 1.3 if max(w, h) > 0 else 100.0
        else:
            self.pan_x = self.pan_y = 0.0
            self.zoom  = 100.0

        self.update()

    def reset_view(self) -> None:
        self.set_section_data(self._section_data_cache)

    # ------------------------------------------------------------------
    # OPENGL
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        glClearColor(*_BG_COLOR)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

        glClear(GL_COLOR_BUFFER_BIT)

        asp = self.width() / self.height() if self.height() > 0 else 1.0
        hw  = self.zoom * asp
        hh  = self.zoom

        self.wx_min = self.pan_x - hw
        self.wx_max = self.pan_x + hw
        self.wy_min = self.pan_y - hh
        self.wy_max = self.pan_y + hh

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(self.wx_min, self.wx_max, self.wy_min, self.wy_max, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self._gl_grid()
        self._gl_shapes()
        self._gl_staffe()
        self._gl_bars()
        self._gl_overlay()

    # ------------------------------------------------------------------
    # DISEGNO ELEMENTI
    # ------------------------------------------------------------------

    def _gl_grid(self) -> None:
        dx = self.wx_max - self.wx_min
        dy = self.wy_max - self.wy_min
        tx = self._tick(dx)
        ty = self._tick(dy)

        glColor3f(*_GRID_COLOR)
        glLineWidth(0.6)
        glBegin(GL_LINES)
        x = math.floor(self.wx_min / tx) * tx
        while x <= self.wx_max + tx:
            glVertex2f(x, self.wy_min)
            glVertex2f(x, self.wy_max)
            x += tx
        y = math.floor(self.wy_min / ty) * ty
        while y <= self.wy_max + ty:
            glVertex2f(self.wx_min, y)
            glVertex2f(self.wx_max, y)
            y += ty
        glEnd()

    def _gl_shapes(self) -> None:
        for s in self._shapes:
            if s['type'] == 'poly':
                glColor4f(*s['fill'])
                for tri in s.get('tris', []):
                    glBegin(GL_TRIANGLES)
                    for pt in tri:
                        glVertex2f(pt[0], pt[1])
                    glEnd()
                if s.get('is_foro'):
                    self._gl_grid_in_hole(s)
                glColor4f(*s['outline'])
                glLineWidth(s['lw'])
                glBegin(GL_LINE_LOOP)
                for pt in s['pts']:
                    glVertex2f(pt[0], pt[1])
                glEnd()

            elif s['type'] == 'ellipse':
                self._gl_ellipse(s['cx'], s['cy'], s['rx'], s['ry'],
                                  s['fill'], s['outline'], s['lw'])
                if s.get('is_foro'):
                    self._gl_grid_in_hole(s)
                    glColor4f(*s['outline'])
                    glLineWidth(s['lw'])
                    ts = np.linspace(0, 2 * math.pi, _BAR_SEG, endpoint=False)
                    glBegin(GL_LINE_LOOP)
                    for t in ts:
                        glVertex2f(s['cx'] + s['rx'] * math.cos(t),
                                   s['cy'] + s['ry'] * math.sin(t))
                    glEnd()

    def _gl_grid_in_hole(self, shape: dict) -> None:
        glEnable(GL_STENCIL_TEST)
        glStencilMask(0xFF)
        glClear(GL_STENCIL_BUFFER_BIT)

        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glStencilFunc(GL_ALWAYS, 1, 0xFF)
        glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE)

        if shape['type'] == 'poly':
            for tri in shape.get('tris', []):
                glBegin(GL_TRIANGLES)
                for pt in tri:
                    glVertex2f(pt[0], pt[1])
                glEnd()
        elif shape['type'] == 'ellipse':
            ts = np.linspace(0, 2 * math.pi, _BAR_SEG, endpoint=True)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(shape['cx'], shape['cy'])
            for t in ts:
                glVertex2f(shape['cx'] + shape['rx'] * math.cos(t),
                           shape['cy'] + shape['ry'] * math.sin(t))
            glEnd()
        else:
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            glDisable(GL_STENCIL_TEST)
            return

        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glStencilFunc(GL_EQUAL, 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        self._gl_grid()

        glStencilMask(0xFF)
        glClear(GL_STENCIL_BUFFER_BIT)
        glDisable(GL_STENCIL_TEST)

    def _gl_staffe(self) -> None:
        glColor4f(*_STAFFA_OUTLINE)
        glLineWidth(_STAFFA_LINE_W)
        for s in self._staffe:
            pts = s['punti']
            glBegin(GL_LINE_STRIP)
            for pt in pts:
                glVertex2f(pt[0], pt[1])
            glEnd()

    def _gl_bars(self) -> None:
        for b in self._bars:
            self._gl_circle(b['cx'], b['cy'], b['r'],
                             _BAR_FILL, _BAR_OUTLINE, _BAR_LINE_W)

    def _gl_circle(self, cx, cy, r, fill, outline, lw) -> None:
        ts = np.linspace(0, 2 * math.pi, _BAR_SEG, endpoint=True)
        glColor4f(*fill)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for t in ts:
            glVertex2f(cx + r * math.cos(t), cy + r * math.sin(t))
        glEnd()
        glColor4f(*outline)
        glLineWidth(lw)
        glBegin(GL_LINE_STRIP)
        for t in ts:
            glVertex2f(cx + r * math.cos(t), cy + r * math.sin(t))
        glEnd()

    def _gl_ellipse(self, cx, cy, rx, ry, fill, outline, lw) -> None:
        ts = np.linspace(0, 2 * math.pi, _BAR_SEG, endpoint=True)
        glColor4f(*fill)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for t in ts:
            glVertex2f(cx + rx * math.cos(t), cy + ry * math.sin(t))
        glEnd()
        glColor4f(*outline)
        glLineWidth(lw)
        glBegin(GL_LINE_STRIP)
        for t in ts:
            glVertex2f(cx + rx * math.cos(t), cy + ry * math.sin(t))
        glEnd()

    def _gl_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(_FONT)
        w, h = self.width(), self.height()

        def ts(wx, wy):
            nx = (wx - self.wx_min) / (self.wx_max - self.wx_min)
            ny = (wy - self.wy_min) / (self.wy_max - self.wy_min)
            return int(nx * w), int((1 - ny) * h)

        sx0, sy0 = ts(0, 0)

        painter.setPen(QPen(_AX_X, 1))
        if 0 <= sy0 <= h:
            painter.drawLine(0, sy0, w, sy0)
            painter.drawText(w - 18, sy0 - 3, "X")

        painter.setPen(QPen(_AX_Y, 1))
        if 0 <= sx0 <= w:
            painter.drawLine(sx0, 0, sx0, h)
            painter.drawText(sx0 + 3, 13, "Y")

        mx, my = self._cursor.x(), self._cursor.y()
        wx, wy = self._s2w(mx, my)
        painter.setPen(QPen(_CURSOR, 1))
        painter.drawLine(mx - 8, my, mx + 8, my)
        painter.drawLine(mx, my - 8, mx, my + 8)
        painter.setPen(QPen(_TXT))
        painter.drawText(6, h - 6, f"X: {wx:.1f}  Y: {wy:.1f} mm")
        painter.end()

    # ------------------------------------------------------------------
    # MOUSE
    # ------------------------------------------------------------------

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._last_pos = e.pos()
        self._cursor = e.pos()
        self.update()

    def mouseMoveEvent(self, e) -> None:
        self._cursor = e.pos()
        if self._last_pos is not None:
            dx = e.x() - self._last_pos.x()
            dy = e.y() - self._last_pos.y()
            rx = self.wx_max - self.wx_min
            ry = self.wy_max - self.wy_min
            self.pan_x -= dx * rx / max(self.width(), 1)
            self.pan_y += dy * ry / max(self.height(), 1)
            self._last_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._last_pos = None
        self.update()

    def wheelEvent(self, e) -> None:
        delta = e.angleDelta().y()
        if delta == 0:
            return
        sx, sy   = e.pos().x(), e.pos().y()
        wx0, wy0 = self._s2w(sx, sy)
        factor   = 1.0 - math.copysign(0.10, delta)
        self.zoom = max(0.5, min(self.zoom * factor, 1e6))

        asp = self.width() / self.height() if self.height() > 0 else 1.0
        self.wx_min = self.pan_x - self.zoom * asp
        self.wx_max = self.pan_x + self.zoom * asp
        self.wy_min = self.pan_y - self.zoom
        self.wy_max = self.pan_y + self.zoom
        wx1, wy1 = self._s2w(sx, sy)
        self.pan_x += wx0 - wx1
        self.pan_y += wy0 - wy1
        self.update()

    # ------------------------------------------------------------------
    # UTILITÀ
    # ------------------------------------------------------------------

    def _s2w(self, sx: int, sy: int):
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return 0.0, 0.0
        nx = sx / w
        ny = 1.0 - sy / h
        return (self.wx_min + nx * (self.wx_max - self.wx_min),
                self.wy_min + ny * (self.wy_max - self.wy_min))

    @staticmethod
    def _tick(rng: float) -> float:
        if rng <= 0:
            return 1.0
        rough = rng / 8.0
        mag   = 10 ** math.floor(math.log10(max(rough, 1e-12)))
        ratio = rough / mag
        if   ratio >= 5: return 5 * mag
        elif ratio >= 2: return 2 * mag
        return mag
