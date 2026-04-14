"""
disegno_momentocurvatura_2d.py  –  analisi/momentocurvatura/
=============================================================
Widget OpenGL 2D per la visualizzazione della sezione piana del
diagramma Momento-Curvatura (piano χ-M ruotante attorno all'asse verticale).

Funzionalità:
  - Mostra la curva M(χ) per l'angolo selezionato dallo slider.
  - Riempimento semitrasparente sotto la curva.
  - Punto di verifica M_Ed: verde (dentro) / rosso (fuori).
  - Griglia, assi, tracker con QPainter.
  - Pan (tasto sinistro) e zoom (rotella mouse) centrati.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore    import Qt, QPoint, pyqtSignal
from PyQt5.QtGui     import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_LINE_SMOOTH, GL_LINE_STRIP,
    GL_LINES, GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SMOOTH, GL_POINTS, GL_PROJECTION, GL_SRC_ALPHA,
    GL_TRIANGLE_STRIP,
    glBegin, glBlendFunc, glClear, glClearColor,
    glColor3f, glColor4f, glDisable, glEnable, glEnd,
    glLineWidth, glLoadIdentity, glMatrixMode, glOrtho,
    glPointSize, glVertex2f, glViewport,
)


# ==============================================================================
# WIDGET 2D MOMENTO-CURVATURA
# ==============================================================================

class MomentoCurvaturaWidget2D(QOpenGLWidget):
    """Visualizza la sezione piana χ-M del diagramma momento-curvatura."""

    verifica_cambiata = pyqtSignal(bool)

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Dati
        self.points_matrix:      np.ndarray | None = None
        self.current_curve:      np.ndarray | None = None   # (N, 2) → [χ, M]
        self.current_angle_deg = 0.0

        # Vista
        self._pan   = [0.0, 0.0]
        self._zoom  = 1.0
        self._last_mouse = None

        self._range_x = 0.1     # χ
        self._range_y = 100.0   # M

        # Punto verifica M_Ed
        self._M_Ed      = 0.0
        self._is_inside = True
        self._verifica_color = QColor(0, 200, 80)

        # Cursore
        self._cursor = QPoint(0, 0)
        self.font    = QFont("Arial", 9)
        self.setMouseTracking(True)

        # Colori assi
        self._bg = (40 / 255, 40 / 255, 40 / 255, 1.0)
        self._ax_x_color = QColor(255, 80, 80)    # χ rosso
        self._ax_y_color = QColor(0, 200, 255)    # M ciano

        # Limiti mondo (aggiornati in paintGL)
        self.wx_min = self.wx_max = 0.0
        self.wy_min = self.wy_max = 0.0

    # ------------------------------------------------------------------
    # API PUBBLICA
    # ------------------------------------------------------------------

    def set_points(self, points_matrix: np.ndarray | None) -> None:
        if points_matrix is None or points_matrix.size == 0:
            self.points_matrix  = None
            self.current_curve  = None
            self.update()
            return

        self.points_matrix = points_matrix

        # Range globali
        max_chi = float(np.max(np.abs(points_matrix[:, :, 1])))
        max_m   = float(np.max(np.abs(points_matrix[:, :, 0])))

        self._range_x = max_chi * 1.2 if max_chi > 1e-5 else 0.1
        self._range_y = max_m   * 1.2 if max_m   > 1e-5 else 100.0

        self._pan  = [self._range_x * 0.4, self._range_y * 0.4]
        self._zoom = 1.0

        self._aggiorna_slice()

    def set_angle_deg(self, angle_deg: float) -> None:
        """Aggiorna l'angolo e ricalcola la curva."""
        self.current_angle_deg = angle_deg
        self._aggiorna_slice()

    def set_M_Ed(self, M_Ed: float) -> bool | None:
        """Aggiorna M_Ed e il punto di verifica."""
        self._M_Ed = M_Ed
        self._check_inside()
        self.update()
        return self._is_inside if self.current_curve is not None else None

    def reset_view(self) -> None:
        if self.points_matrix is not None:
            self._pan  = [self._range_x * 0.4, self._range_y * 0.4]
        else:
            self._pan = [0.0, 0.0]
        self._zoom = 1.0
        self.update()

    # ------------------------------------------------------------------
    # SLICE E VERIFICA
    # ------------------------------------------------------------------

    def _aggiorna_slice(self) -> None:
        """Estrae la curva 2D per l'angolo corrente."""
        if self.points_matrix is None:
            self.current_curve = None
            self.update()
            return

        target_rad = math.radians(self.current_angle_deg)
        thetas = self.points_matrix[:, 0, 2]
        diff = np.abs(thetas - target_rad)
        diff = np.minimum(diff, 2 * np.pi - diff)
        idx = int(np.argmin(diff))

        row = self.points_matrix[idx]
        chi_vals = row[:, 1]   # Curvatura
        M_vals   = row[:, 0]   # Momento

        # Filtra punti "morti": M≈0 con χ>0 (escluso il punto origine)
        valid = np.ones(len(M_vals), dtype=bool)
        for i in range(1, len(M_vals)):
            if abs(M_vals[i]) < 1e-6 and chi_vals[i] > 1e-6:
                valid[i:] = False
                break

        self.current_curve = np.column_stack((chi_vals[valid], M_vals[valid]))
        self._check_inside()
        self.update()

    def _check_inside(self) -> None:
        if self.current_curve is None:
            self._is_inside = False
            self._verifica_color = QColor(220, 60, 60)
            return

        M_max = float(np.max(self.current_curve[:, 1]))
        inside = (self._M_Ed <= M_max) if M_max > 1e-9 else (self._M_Ed <= 0)

        if inside != self._is_inside:
            self._is_inside = inside
            self.verifica_cambiata.emit(inside)
        else:
            self._is_inside = inside

        self._verifica_color = QColor(0, 200, 80) if inside else QColor(220, 60, 60)

    # ------------------------------------------------------------------
    # OPENGL
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        glClearColor(*self._bg)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        self.wx_min = -self._range_x * self._zoom + self._pan[0]
        self.wx_max =  self._range_x * self._zoom + self._pan[0]
        self.wy_min = -self._range_y * self._zoom + self._pan[1]
        self.wy_max =  self._range_y * self._zoom + self._pan[1]

        glOrtho(self.wx_min, self.wx_max, self.wy_min, self.wy_max, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self._draw_grid()
        self._draw_curve()
        self._draw_M_Ed_point()
        self._draw_overlay()
        self._draw_tracker()

    # ------------------------------------------------------------------
    # GRIGLIA
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        tx = self._tick(self.wx_max - self.wx_min)
        ty = self._tick(self.wy_max - self.wy_min)

        glColor3f(0.2, 0.2, 0.2)
        glLineWidth(1)
        glBegin(GL_LINES)

        x = np.floor(self.wx_min / tx) * tx
        while x <= self.wx_max + 1e-12:
            glVertex2f(x, self.wy_min)
            glVertex2f(x, self.wy_max)
            x += tx

        y = np.floor(self.wy_min / ty) * ty
        while y <= self.wy_max + 1e-12:
            glVertex2f(self.wx_min, y)
            glVertex2f(self.wx_max, y)
            y += ty

        glEnd()

    # ------------------------------------------------------------------
    # CURVA M-χ
    # ------------------------------------------------------------------

    def _draw_curve(self) -> None:
        if self.current_curve is None:
            return

        # Riempimento sotto la curva
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(1.0, 1.0, 1.0, 0.1)

        glBegin(GL_TRIANGLE_STRIP)
        for p in self.current_curve:
            glVertex2f(p[0], 0)
            glVertex2f(p[0], p[1])
        glEnd()

        # Linea curva
        glDisable(GL_BLEND)
        glLineWidth(1.2)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_LINE_STRIP)
        for p in self.current_curve:
            glVertex2f(p[0], p[1])
        glEnd()

        # Punto finale (rottura)
        if len(self.current_curve) > 0:
            last = self.current_curve[-1]
            glPointSize(4.0)
            glColor3f(1.0, 0.2, 0.2)
            glBegin(GL_POINTS)
            glVertex2f(last[0], last[1])
            glEnd()

    # ------------------------------------------------------------------
    # PUNTO VERIFICA M_Ed
    # ------------------------------------------------------------------

    def _draw_M_Ed_point(self) -> None:
        if self.current_curve is None:
            return

        M_Ed = self._M_Ed
        col  = self._verifica_color

        if self._is_inside:
            # Interpola χ a M_Ed
            M_arr   = self.current_curve[:, 1]
            chi_arr = self.current_curve[:, 0]
            chi_at_M = float(np.interp(M_Ed, M_arr, chi_arr))
            px, py = chi_at_M, M_Ed
        else:
            # Fuori: posiziona comunque sulla linea M_Ed
            chi_max = float(self.current_curve[-1, 0])
            px, py = chi_max * 0.5, M_Ed

        # Linea orizzontale tratteggiata a M_Ed
        glLineWidth(1.0)
        glColor4f(col.redF(), col.greenF(), col.blueF(), 0.5)
        glBegin(GL_LINES)
        glVertex2f(self.wx_min, M_Ed)
        glVertex2f(self.wx_max, M_Ed)
        glEnd()

        # Punto
        glPointSize(10.0)
        glColor3f(col.redF(), col.greenF(), col.blueF())
        glBegin(GL_POINTS)
        glVertex2f(px, py)
        glEnd()

    # ------------------------------------------------------------------
    # OVERLAY (Assi, etichette, info angolo)
    # ------------------------------------------------------------------

    def _draw_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        fm = painter.fontMetrics()
        w, h = self.width(), self.height()

        c_x = self._ax_x_color
        c_y = self._ax_y_color

        def to_screen(wx, wy):
            nx = (wx - self.wx_min) / (self.wx_max - self.wx_min)
            ny = (wy - self.wy_min) / (self.wy_max - self.wy_min)
            return int(nx * w), int((1 - ny) * h)

        sx0, sy0 = to_screen(0, 0)

        # Asse X (χ)
        pen = QPen(c_x, 1)
        painter.setPen(pen)
        if 0 <= sy0 <= h:
            painter.drawLine(0, sy0, w, sy0)

        # Asse Y (M)
        pen.setColor(c_y)
        painter.setPen(pen)
        if 0 <= sx0 <= w:
            painter.drawLine(sx0, 0, sx0, h)

        # Ticks X
        tx = self._tick(self.wx_max - self.wx_min)
        painter.setPen(c_x)
        x = np.floor(self.wx_min / tx) * tx
        while x <= self.wx_max:
            if abs(x) > 1e-9:
                sx, sy = to_screen(x, 0)
                painter.drawLine(sx, sy - 4, sx, sy + 4)
                lbl = f"{x:.3g}"
                tw  = fm.horizontalAdvance(lbl)
                txt_y = min(max(sy + fm.height() + 2, 14), h - 4)
                painter.drawText(sx - tw // 2, txt_y, lbl)
            x += tx

        # Ticks Y
        ty = self._tick(self.wy_max - self.wy_min)
        painter.setPen(c_y)
        y = np.floor(self.wy_min / ty) * ty
        while y <= self.wy_max:
            if abs(y) > 1e-9:
                sx, sy = to_screen(0, y)
                painter.drawLine(sx - 4, sy, sx + 4, sy)
                lbl = f"{y:.3g}"
                tw  = fm.horizontalAdvance(lbl)
                txt_x = min(max(sx - tw - 6, 4), w - tw - 4)
                painter.drawText(txt_x, sy + fm.height() // 2 - 2, lbl)
            y += ty

        # Info angolo
        info = f"θ = {self.current_angle_deg:.1f}°"
        painter.setPen(Qt.white)
        painter.drawText(w - fm.horizontalAdvance(info) - 10, 20, info)

        # Titoli assi
        lab_x = "Curvatura χ [1/m]"
        lab_y = "Momento M [kNm]"
        painter.setPen(c_x)
        painter.drawText(w - fm.horizontalAdvance(lab_x) - 10, h - 10, lab_x)
        painter.setPen(c_y)
        painter.save()
        painter.translate(20, 150)
        painter.rotate(-90)
        painter.drawText(0, 0, lab_y)
        painter.restore()

        painter.end()

    # ------------------------------------------------------------------
    # TRACKER
    # ------------------------------------------------------------------

    def _draw_tracker(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        w, h = self.width(), self.height()
        cx, cy = self._cursor.x(), self._cursor.y()

        # Mirino
        painter.setPen(QPen(QColor(150, 150, 150, 100), 1))
        painter.drawLine(cx, 0, cx, h)
        painter.drawLine(0, cy, w, cy)

        painter.setPen(QPen(QColor(255, 255, 255, 255), 1))
        sz = 10
        painter.drawLine(cx - sz, cy, cx + sz, cy)
        painter.drawLine(cx, cy - sz, cx, cy + sz)

        # Coordinate
        wx, wy = self._screen_to_world(cx, cy)
        painter.setPen(QColor(255, 255, 255, 180))
        painter.drawText(cx + 10, h - 10, f"{wx:.4g}")
        painter.drawText(10, cy - 10, f"{wy:.4g}")

        painter.end()

    # ------------------------------------------------------------------
    # MOUSE
    # ------------------------------------------------------------------

    def mousePressEvent(self, e) -> None:
        if e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._last_mouse = e.pos()
        self._cursor = e.pos()
        self.update()

    def mouseMoveEvent(self, e) -> None:
        self._cursor = e.pos()
        if self._last_mouse is not None and (
                e.buttons() & (Qt.LeftButton | Qt.MiddleButton)):
            dx = e.x() - self._last_mouse.x()
            dy = e.y() - self._last_mouse.y()
            rx = self.wx_max - self.wx_min
            ry = self.wy_max - self.wy_min
            self._pan[0] -= dx * rx / self.width()
            self._pan[1] += dy * ry / self.height()
            self._last_mouse = e.pos()
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._last_mouse = None
        self.update()

    def wheelEvent(self, e) -> None:
        delta = e.angleDelta().y()
        if delta == 0:
            return
        sx, sy = e.pos().x(), e.pos().y()
        wx0, wy0 = self._screen_to_world(sx, sy)

        factor = 1.0 - np.sign(delta) * 0.1
        self._zoom *= factor
        self._zoom = max(0.0001, min(10000.0, self._zoom))

        # Ricalcola per compensare zoom
        self.wx_min = -self._range_x * self._zoom + self._pan[0]
        self.wx_max =  self._range_x * self._zoom + self._pan[0]
        self.wy_min = -self._range_y * self._zoom + self._pan[1]
        self.wy_max =  self._range_y * self._zoom + self._pan[1]

        wx1, wy1 = self._screen_to_world(sx, sy)
        self._pan[0] += (wx0 - wx1)
        self._pan[1] += (wy0 - wy1)

        self.update()

    # ------------------------------------------------------------------
    # UTILITÀ
    # ------------------------------------------------------------------

    def _screen_to_world(self, sx, sy):
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return 0.0, 0.0
        nx = sx / w
        ny = 1.0 - (sy / h)
        return (self.wx_min + nx * (self.wx_max - self.wx_min),
                self.wy_min + ny * (self.wy_max - self.wy_min))

    @staticmethod
    def _tick(rng):
        rough = rng / 10.0
        if rough <= 0:
            return 1.0
        mag = 10 ** np.floor(np.log10(max(rough, 1e-12)))
        res = rough / mag
        if res >= 5:
            return float(5 * mag)
        elif res >= 2:
            return float(2 * mag)
        return float(mag)
