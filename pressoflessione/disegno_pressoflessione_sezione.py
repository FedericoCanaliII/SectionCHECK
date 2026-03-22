"""
disegno_pressoflessione_sezione.py
===================================
Widget OpenGL 2D per la visualizzazione della geometria della sezione
nel pannello Pressoflessione.

Funzionalmente identico a disegno_output_sezione.OpenGLOutputSezioneWidget;
rinominato per mantenere i moduli separati e permettere estensioni future
(es. evidenziare l'asse neutro direttamente sul viewer geometria).
"""
from __future__ import annotations

import math

from PyQt5.QtCore  import Qt, QPoint
from PyQt5.QtGui   import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import (
    GL_BLEND, GL_CULL_FACE, GL_DEPTH_TEST, GL_LINE_LOOP, GL_LINE_SMOOTH,
    GL_LINE_STRIP, GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_POINT_SMOOTH,
    GL_POLYGON, GL_PROJECTION, GL_SRC_ALPHA, GL_TEXTURE_2D,
    GL_TRIANGLE_FAN, GL_LINES, GL_COLOR_BUFFER_BIT,
    glBegin, glBlendFunc, glClear, glClearColor, glColor3f, glColor4f,
    glDisable, glEnable, glEnd, glLineWidth, glLoadIdentity, glMatrixMode,
    glOrtho, glVertex2f, glViewport,
)
import numpy as np


# ==============================================================================
# CONFIGURAZIONE STILE
# ==============================================================================
STYLE_CONFIG = {
    # Sfondo
    "BACKGROUND_COLOR" : (40/255, 40/255, 40/255, 1.0),
    "GRID_COLOR"       : (0.22, 0.22, 0.22),

    # Assi
    "AXIS_X_COLOR"     : QColor(220,  60,  60),
    "AXIS_Y_COLOR"     : QColor( 60, 200,  60),

    # Calcestruzzo
    "CONCRETE_FILL"    : (0.78, 0.78, 0.78, 0.22),
    "CONCRETE_OUTLINE" : (1.0,  1.0,  1.0,  0.90),
    "CONCRETE_LINE_W"  : 1.2,

    # Armature longitudinali
    "STEEL_FILL"       : (1.0, 0.18, 0.18, 0.55),
    "STEEL_OUTLINE"    : (1.0, 0.18, 0.18, 1.00),
    "STEEL_LINE_W"     : 1.0,
    "BAR_SEGMENTS"     : 48,

    # Testo / cursore
    "TEXT_COLOR"       : QColor(200, 200, 200),
    "CURSOR_COLOR"     : QColor(255, 255, 255, 210),
    "FONT_SIZE"        : 9,
    "FONT_FAMILY"      : "Arial",
}


# ==============================================================================
# WIDGET
# ==============================================================================
class OpenGLPressoflessioneSezioneWidget(QOpenGLWidget):
    """
    Visualizzatore 2D (OpenGL) della sezione per il pannello Pressoflessione.

    Accetta i dati tramite set_section_data(section_data) dove section_data
    ha la stessa struttura restituita da Verifica.get_tutte_matrici_sezioni().
    """

    def __init__(self, ui, parent=None) -> None:
        super().__init__(parent)
        self.ui = ui

        # Geometria da renderizzare
        self.render_shapes: list = []
        self.render_bars:   list = []

        # Vista 2D
        self.pan_2d       : list  = [0.0, 0.0]
        self.zoom_2d      : float = 1.0
        self.last_mouse_pos       = None
        self.cursor_pos           = QPoint(0, 0)

        # Estremi del mondo (per info cursore)
        self.wx_min = self.wx_max = 0.0
        self.wy_min = self.wy_max = 0.0

        self.setMouseTracking(True)
        self.font_obj = QFont(STYLE_CONFIG["FONT_FAMILY"], STYLE_CONFIG["FONT_SIZE"])

    # --------------------------------------------------------------------------
    # DATI
    # --------------------------------------------------------------------------
    def set_section_data(self, section_data) -> None:
        """
        Analizza section_data (dict con chiave 'elementi') e aggiorna la scena.
        Formato elementi: stessa struttura di Verifica.get_tutte_matrici_sezioni().
        """
        self.render_shapes = []
        self.render_bars   = []

        if not section_data or 'elementi' not in section_data:
            self.zoom_2d = 100.0
            self.pan_2d  = [0.0, 0.0]
            self.update()
            return

        min_x = min_y =  float('inf')
        max_x = max_y = -float('inf')
        has_data = False

        def _aggiorna_bb(px, py):
            nonlocal min_x, max_x, min_y, max_y, has_data
            min_x, max_x = min(min_x, px), max(max_x, px)
            min_y, max_y = min(min_y, py), max(max_y, py)
            has_data = True

        try:
            for elem in section_data['elementi']:
                tipo = elem[0]

                if tipo == 'shape':
                    _, shape_type, _, _, *params = elem

                    if shape_type == 'rect':
                        p1, p2 = params[0], params[1]
                        x1, y1 = float(p1[0]), float(p1[1])
                        x2, y2 = float(p2[0]), float(p2[1])
                        pts = [
                            (min(x1,x2), min(y1,y2)),
                            (max(x1,x2), min(y1,y2)),
                            (max(x1,x2), max(y1,y2)),
                            (min(x1,x2), max(y1,y2)),
                        ]
                        self.render_shapes.append({'type': 'poly', 'points': pts})
                        for px, py in pts:
                            _aggiorna_bb(px, py)

                    elif shape_type == 'poly':
                        pts = [(float(p[0]), float(p[1])) for p in params[0]]
                        self.render_shapes.append({'type': 'poly', 'points': pts})
                        for px, py in pts:
                            _aggiorna_bb(px, py)

                    elif shape_type == 'circle':
                        cx, cy = float(params[0][0]), float(params[0][1])
                        r = float(params[1])
                        self.render_shapes.append({'type': 'circle', 'center': (cx, cy), 'radius': r})
                        _aggiorna_bb(cx - r, cy - r)
                        _aggiorna_bb(cx + r, cy + r)

                elif tipo == 'bar':
                    _, _, _, diam, center = elem
                    cx, cy = float(center[0]), float(center[1])
                    r = float(diam) / 2.0
                    self.render_bars.append({'center': (cx, cy), 'radius': r})
                    _aggiorna_bb(cx - r, cy - r)
                    _aggiorna_bb(cx + r, cy + r)

        except Exception as e:
            print(f"[disegno_pressoflessione_sezione] Errore parsing: {e}")

        if has_data:
            cx    = (min_x + max_x) / 2.0
            cy    = (min_y + max_y) / 2.0
            w     = max_x - min_x
            h     = max_y - min_y
            self.pan_2d  = [cx, cy]
            self.zoom_2d = max(w, h) / 2.0 * 1.25 if max(w, h) > 0 else 100.0
        else:
            self.pan_2d  = [0.0, 0.0]
            self.zoom_2d = 100.0

        self.update()

    # --------------------------------------------------------------------------
    # OPENGL
    # --------------------------------------------------------------------------
    def initializeGL(self) -> None:
        bc = STYLE_CONFIG["BACKGROUND_COLOR"]
        glClearColor(*bc)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        # Reset stato (QPainter da frame precedente può sporcare lo stato)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

        glClear(GL_COLOR_BUFFER_BIT)

        # Proiezione ortografica
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        hw     = self.zoom_2d * aspect
        hh     = self.zoom_2d

        self.wx_min = self.pan_2d[0] - hw
        self.wx_max = self.pan_2d[0] + hw
        self.wy_min = self.pan_2d[1] - hh
        self.wy_max = self.pan_2d[1] + hh

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(self.wx_min, self.wx_max, self.wy_min, self.wy_max, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self._draw_grid()
        self._draw_shapes()
        self._draw_bars()
        self._draw_overlay()

    def _draw_grid(self) -> None:
        dx = self.wx_max - self.wx_min
        dy = self.wy_max - self.wy_min
        tx = self._tick(dx)
        ty = self._tick(dy)

        glColor3f(*STYLE_CONFIG["GRID_COLOR"])
        glLineWidth(0.8)

        glBegin(GL_LINES)
        x = np.floor(self.wx_min / tx) * tx
        while x <= self.wx_max + tx:
            glVertex2f(x, self.wy_min)
            glVertex2f(x, self.wy_max)
            x += tx
        y = np.floor(self.wy_min / ty) * ty
        while y <= self.wy_max + ty:
            glVertex2f(self.wx_min, y)
            glVertex2f(self.wx_max, y)
            y += ty
        glEnd()

    def _draw_shapes(self) -> None:
        for shape in self.render_shapes:
            if shape['type'] == 'poly':
                pts = shape['points']
                glColor4f(*STYLE_CONFIG["CONCRETE_FILL"])
                glBegin(GL_POLYGON)
                for p in pts:
                    glVertex2f(p[0], p[1])
                glEnd()
                glColor4f(*STYLE_CONFIG["CONCRETE_OUTLINE"])
                glLineWidth(STYLE_CONFIG["CONCRETE_LINE_W"])
                glBegin(GL_LINE_LOOP)
                for p in pts:
                    glVertex2f(p[0], p[1])
                glEnd()

            elif shape['type'] == 'circle':
                self._draw_circle(shape['center'], shape['radius'],
                                  STYLE_CONFIG["CONCRETE_FILL"],
                                  STYLE_CONFIG["CONCRETE_OUTLINE"],
                                  STYLE_CONFIG["CONCRETE_LINE_W"])

    def _draw_bars(self) -> None:
        for bar in self.render_bars:
            self._draw_circle(bar['center'], bar['radius'],
                              STYLE_CONFIG["STEEL_FILL"],
                              STYLE_CONFIG["STEEL_OUTLINE"],
                              STYLE_CONFIG["STEEL_LINE_W"])

    def _draw_circle(self, center, radius, fill, outline, line_w) -> None:
        n   = STYLE_CONFIG["BAR_SEGMENTS"]
        cx, cy = center
        ts  = np.linspace(0, 2 * math.pi, n, endpoint=True)

        glColor4f(*fill)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for t in ts:
            glVertex2f(cx + radius * math.cos(t), cy + radius * math.sin(t))
        glEnd()

        glColor4f(*outline)
        glLineWidth(line_w)
        glBegin(GL_LINE_STRIP)
        for t in ts:
            glVertex2f(cx + radius * math.cos(t), cy + radius * math.sin(t))
        glEnd()

    def _draw_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font_obj)
        w, h = self.width(), self.height()

        def to_screen(wx, wy):
            nx = (wx - self.wx_min) / (self.wx_max - self.wx_min)
            ny = (wy - self.wy_min) / (self.wy_max - self.wy_min)
            return int(nx * w), int((1 - ny) * h)

        sx0, sy0 = to_screen(0, 0)

        # Asse X
        pen = QPen(STYLE_CONFIG["AXIS_X_COLOR"], 1)
        painter.setPen(pen)
        if 0 <= sy0 <= h:
            painter.drawLine(0, sy0, w, sy0)
            painter.drawText(w - 20, sy0 - 4, "X")

        # Asse Y
        pen.setColor(STYLE_CONFIG["AXIS_Y_COLOR"])
        painter.setPen(pen)
        if 0 <= sx0 <= w:
            painter.drawLine(sx0, 0, sx0, h)
            painter.drawText(sx0 + 4, 14, "Y")

        # Cursore + coordinate
        mx, my = self.cursor_pos.x(), self.cursor_pos.y()
        wx, wy = self._screen_to_world(mx, my)
        painter.setPen(QPen(STYLE_CONFIG["CURSOR_COLOR"]))
        painter.drawLine(mx - 10, my, mx + 10, my)
        painter.drawLine(mx, my - 10, mx, my + 10)
        painter.drawText(10, h - 10, f"X: {wx:.1f}  Y: {wy:.1f}")

        painter.end()

    # --------------------------------------------------------------------------
    # UTILITÀ
    # --------------------------------------------------------------------------
    @staticmethod
    def _tick(rng: float) -> float:
        rough = rng / 8.0
        if rough <= 0:
            return 1.0
        mag = 10 ** np.floor(np.log10(rough))
        res = rough / mag
        if   res >= 5: return 5   * mag
        elif res >= 2: return 2   * mag
        return mag

    def _screen_to_world(self, sx: int, sy: int):
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return 0.0, 0.0
        nx = sx / w
        ny = 1.0 - sy / h
        wx = self.wx_min + nx * (self.wx_max - self.wx_min)
        wy = self.wy_min + ny * (self.wy_max - self.wy_min)
        return wx, wy

    # --------------------------------------------------------------------------
    # MOUSE
    # --------------------------------------------------------------------------
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.last_mouse_pos = e.pos()
        self.cursor_pos = e.pos()
        self.update()

    def mouseMoveEvent(self, e) -> None:
        self.cursor_pos = e.pos()
        if self.last_mouse_pos is not None:
            dx = e.x() - self.last_mouse_pos.x()
            dy = e.y() - self.last_mouse_pos.y()
            rx = self.wx_max - self.wx_min
            ry = self.wy_max - self.wy_min
            self.pan_2d[0] -= dx * rx / self.width()
            self.pan_2d[1] += dy * ry / self.height()
            self.last_mouse_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        self.last_mouse_pos = None
        self.update()

    def wheelEvent(self, e) -> None:
        delta = e.angleDelta().y()
        if delta == 0:
            return
        sx, sy = e.pos().x(), e.pos().y()
        wx0, wy0 = self._screen_to_world(sx, sy)
        factor = 1.0 - np.sign(delta) * 0.10
        self.zoom_2d = max(0.5, min(self.zoom_2d * factor, 1e5))
        # Ricalcola wx_min/max/wy_min/max con il nuovo zoom
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        self.wx_min = self.pan_2d[0] - self.zoom_2d * aspect
        self.wx_max = self.pan_2d[0] + self.zoom_2d * aspect
        self.wy_min = self.pan_2d[1] - self.zoom_2d
        self.wy_max = self.pan_2d[1] + self.zoom_2d
        wx1, wy1 = self._screen_to_world(sx, sy)
        self.pan_2d[0] += wx0 - wx1
        self.pan_2d[1] += wy0 - wy1
        self.update()
