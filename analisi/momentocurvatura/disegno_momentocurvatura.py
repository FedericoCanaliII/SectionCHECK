"""
disegno_momentocurvatura.py  –  analisi/momentocurvatura/
==========================================================
Widget OpenGL 3D per la visualizzazione del diagramma Momento-Curvatura.

Caratteristiche:
  - Superficie M-χ in coordinate cilindriche (χ·cosθ, M, -χ·sinθ).
  - Mesh grigia semitrasparente + wireframe bianco.
  - Rendering GPU tramite Vertex Arrays (NumPy → GPU).
  - Assi: X = χ·cosθ (rosso), Y = M (ciano), Z = -χ·sinθ (verde).
  - Anello di verifica M_Ed: filo chiuso verde (dentro) / rosso (fuori).
"""
from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore    import Qt, QPoint, pyqtSignal
from PyQt5.QtGui     import QColor, QFont, QPainter
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_LINES, GL_LINE_LOOP, GL_LINE_STRIP,
    GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SMOOTH, GL_POINTS, GL_PROJECTION, GL_QUADS, GL_SRC_ALPHA,
    glBegin, glBlendFunc, glClear, glClearColor,
    glColor3f, glColor4f, glCullFace, glDepthMask,
    glDisable, glEnable, glEnd, glGetDoublev, glGetIntegerv,
    glLineWidth, glLoadIdentity, glMatrixMode, glPointSize,
    glRotatef, glScalef, glShadeModel, glTranslatef, glVertex3f,
    glViewport,
    GL_BACK, GL_FRONT, GL_SMOOTH,
    GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT,
    GL_FALSE, GL_TRUE,
    glEnableClientState, glDisableClientState, glVertexPointer,
    glDrawArrays, GL_VERTEX_ARRAY, GL_FLOAT,
)
from OpenGL.GLU import gluPerspective, gluProject


# ==============================================================================
# WIDGET 3D MOMENTO-CURVATURA
# ==============================================================================

class MomentoCurvaturaWidget3D(QOpenGLWidget):
    """Visualizza il diagramma Momento-Curvatura 3D."""

    verifica_cambiata = pyqtSignal(bool)  # True = dentro

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Dati diagramma
        self.points_matrix:     np.ndarray | None = None   # (n_ang, n_pts, 3) originale
        self.normalized_matrix: np.ndarray | None = None   # (n_ang, n_pts, 3) OpenGL
        self.normalization_factors = np.ones(3)             # [max_χ, max_M, max_χ]

        # Vertex arrays GPU
        self._vbo_quads  = np.array([], dtype=np.float32)
        self._vbo_lines  = np.array([], dtype=np.float32)
        self._n_quads  = 0
        self._n_lines  = 0

        # Anello di verifica M_Ed
        self._M_Ed       = 0.0
        self._ring_pts:  np.ndarray | None = None   # (n_ang, 3) punti 3D normalizzati
        self._ring_inside = True
        self._verifica_attiva = False

        # Vista
        self.rotation    = [30.0, -45.0, 0.0]
        self.translation = [0.0,  0.0,  -3.5]
        self.scale       = 1.0
        self.last_pos    = QPoint()

        self.axis_length = 1.0
        self.axis_labels = 10
        self.font        = QFont("Arial", 9)

    # ------------------------------------------------------------------
    # API PUBBLICA
    # ------------------------------------------------------------------

    def set_points(self, points_matrix: np.ndarray | None) -> None:
        """
        Riceve matrice (n_angoli, n_punti, 3) con colonne [M, χ, θ].
        """
        if points_matrix is None or points_matrix.size == 0:
            self.points_matrix     = None
            self.normalized_matrix = None
            self._verifica_attiva  = False
            self._n_quads = self._n_lines = 0
            self.update()
            return

        # Filtra i punti "morti": M≈0 con χ>0 (escluso il punto origine)
        cleaned = points_matrix.copy()
        rows, cols, _ = cleaned.shape
        for r in range(rows):
            for c in range(1, cols):
                if abs(cleaned[r, c, 0]) < 1e-6 and cleaned[r, c, 1] > 1e-6:
                    # Tronca: copia l'ultimo punto valido (c-1) in avanti
                    cleaned[r, c:] = cleaned[r, c - 1]
                    break

        self.points_matrix = cleaned

        # Fattori di normalizzazione
        all_M   = np.abs(cleaned[:, :, 0])
        all_chi = np.abs(cleaned[:, :, 1])

        max_M   = float(np.max(all_M))   if np.max(all_M)   > 1e-9 else 1.0
        max_chi = float(np.max(all_chi))  if np.max(all_chi)  > 1e-9 else 1.0

        self.normalization_factors = np.array([max_chi, max_M, max_chi])

        # Trasformazione cilindriche → cartesiane OpenGL
        norm = np.zeros((rows, cols, 3), dtype=np.float64)

        for r in range(rows):
            for c in range(cols):
                M     = cleaned[r, c, 0]
                chi   = cleaned[r, c, 1]
                theta = cleaned[r, c, 2]

                norm[r, c, 0] = (chi * math.cos(theta)) / max_chi    # X
                norm[r, c, 1] = M / max_M                             # Y (su)
                norm[r, c, 2] = -(chi * math.sin(theta)) / max_chi   # Z

        self.normalized_matrix = norm
        self._build_vertex_arrays()
        self._verifica_attiva = True
        self._aggiorna_ring()
        self.update()

    def set_M_Ed(self, M_Ed: float) -> bool | None:
        """Aggiorna il valore di M_Ed e l'anello di verifica."""
        self._M_Ed = M_Ed
        return self._aggiorna_ring()

    def reset_view(self) -> None:
        self.rotation    = [30.0, -45.0, 0.0]
        self.translation = [0.0,  0.0,  -3.5]
        self.scale       = 1.0
        self.update()

    # ------------------------------------------------------------------
    # VERTEX ARRAYS
    # ------------------------------------------------------------------

    def _build_vertex_arrays(self) -> None:
        mat = self.normalized_matrix
        if mat is None:
            return
        R, C, _ = mat.shape

        quads = []
        lines = []

        for i in range(R):
            ni = (i + 1) % R
            for j in range(C - 1):
                quads.append(mat[i,  j])
                quads.append(mat[ni, j])
                quads.append(mat[ni, j + 1])
                quads.append(mat[i,  j + 1])

            # Wireframe: meridiani
            for j in range(C - 1):
                lines.append(mat[i, j])
                lines.append(mat[i, j + 1])
                lines.append(mat[i, j])
                lines.append(mat[ni, j])
            lines.append(mat[i, C - 1])
            lines.append(mat[ni, C - 1])

        self._vbo_quads = np.array(quads, dtype=np.float32)
        self._n_quads   = self._vbo_quads.shape[0]
        self._vbo_lines = np.array(lines, dtype=np.float32)
        self._n_lines   = self._vbo_lines.shape[0]

    # ------------------------------------------------------------------
    # ANELLO DI VERIFICA M_Ed
    # ------------------------------------------------------------------

    def _aggiorna_ring(self) -> bool | None:
        """
        Calcola l'anello di intersezione tra la superficie e il piano M = M_Ed.
        Per ogni angolo θ, interpola la curva M(χ) per trovare il χ a M_Ed.
        """
        if not self._verifica_attiva or self.points_matrix is None:
            self._ring_pts = None
            return None

        M_Ed = self._M_Ed
        pts = self.points_matrix
        rows, cols, _ = pts.shape
        max_chi = self.normalization_factors[0]
        max_M   = self.normalization_factors[1]

        ring = np.zeros((rows, 3), dtype=np.float64)
        all_inside = True

        for r in range(rows):
            M_arr   = pts[r, :, 0]   # Momenti del ramo
            chi_arr = pts[r, :, 1]   # Curvature del ramo
            theta   = pts[r, 0, 2]   # Angolo del ramo

            M_max = float(np.max(M_arr))

            if M_Ed <= M_max and M_max > 1e-9:
                # Interpolazione: trova χ dove M = M_Ed
                chi_interp = float(np.interp(M_Ed, M_arr, chi_arr))
                ring[r, 0] = (chi_interp * math.cos(theta)) / max_chi
                ring[r, 1] = M_Ed / max_M
                ring[r, 2] = -(chi_interp * math.sin(theta)) / max_chi
            else:
                # M_Ed supera il massimo → fuori dominio per questo angolo
                all_inside = False
                # Posiziona il punto sull'ultimo punto del ramo
                last_chi = float(chi_arr[-1])
                ring[r, 0] = (last_chi * math.cos(theta)) / max_chi
                ring[r, 1] = M_Ed / max_M
                ring[r, 2] = -(last_chi * math.sin(theta)) / max_chi

        self._ring_pts    = ring.astype(np.float32)
        self._ring_inside = all_inside

        if all_inside != getattr(self, '_last_inside', None):
            self._last_inside = all_inside
            self.verifica_cambiata.emit(all_inside)

        self.update()
        return all_inside

    # ------------------------------------------------------------------
    # OPENGL – INIT / RESIZE / PAINT
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        glClearColor(0.157, 0.157, 0.157, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION);  glLoadIdentity()
        asp = w / float(h) if h > 0 else 1.0
        gluPerspective(45.0, asp, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(*self.translation)
        glRotatef(self.rotation[0], 1.0, 0.0, 0.0)
        glRotatef(self.rotation[1], 0.0, 1.0, 0.0)
        glRotatef(self.rotation[2], 0.0, 0.0, 1.0)
        glScalef(self.scale, self.scale, self.scale)

        self._draw_grid()
        self._draw_axes()

        if self._n_quads > 0:
            self._draw_surface()
            self._draw_ring()

    # ------------------------------------------------------------------
    # GRIGLIA E ASSI
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        glColor3f(0.30, 0.30, 0.30)
        glLineWidth(0.8)
        al = self.axis_length
        step = 2 * al / self.axis_labels
        glBegin(GL_LINES)
        for i in range(self.axis_labels + 1):
            z = -al + i * step
            if abs(z) < 1e-5:
                continue
            glVertex3f(-al, 0.0, z)
            glVertex3f( al, 0.0, z)
        for i in range(self.axis_labels + 1):
            x = -al + i * step
            if abs(x) < 1e-5:
                continue
            glVertex3f(x, 0.0, -al)
            glVertex3f(x, 0.0,  al)
        glEnd()

    def _draw_axes(self) -> None:
        al = self.axis_length
        glLineWidth(1.5)
        glBegin(GL_LINES)
        # X – χ·cosθ (rosso)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-al - 0.1, 0.0, 0.0)
        glVertex3f( al + 0.1, 0.0, 0.0)
        # Y – M (ciano)
        glColor3f(0.0, 0.8, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, al + 0.1, 0.0)
        # Z – -χ·sinθ (verde)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, -al - 0.1)
        glVertex3f(0.0, 0.0,  al + 0.1)
        glEnd()
        self._draw_axis_labels()

    def _draw_axis_labels(self) -> None:
        glDisable(GL_DEPTH_TEST)
        painter = QPainter(self)
        painter.setFont(self.font)
        w, h = self.width(), self.height()

        mv  = glGetDoublev(GL_MODELVIEW_MATRIX)
        prj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp  = glGetIntegerv(GL_VIEWPORT)

        def project(x, y, z, text, pen, dx=0, dy=0):
            try:
                sc = gluProject(x, y, z, mv, prj, vp)
            except Exception:
                return
            if not sc:
                return
            sx, sy = int(sc[0]), int(h - sc[1])
            if not (0 <= sx <= w and 0 <= sy <= h):
                return
            painter.setPen(pen)
            painter.drawText(sx + dx, sy + dy, text)

        al = self.axis_length
        steps = 5
        for i in range(1, steps + 1):
            t   = i / float(steps)
            v   = al * t

            val_chi = v * self.normalization_factors[0]
            val_M   = v * self.normalization_factors[1]

            project(v, 0.0, 0.0, f"{val_chi:.3f}",
                    QColor(255, 100, 100), dx=-10, dy=20)
            project(0.0, v, 0.0, f"{val_M:.1f}",
                    QColor(100, 200, 255), dx=-30, dy=5)
            project(0.0, 0.0, v, f"{val_chi:.3f}",
                    QColor(100, 255, 100), dx=10, dy=-5)

        off = al * 1.15
        project( off, 0.0, 0.0, "χ·cos θ [1/m]", QColor(255, 50, 50))
        project(0.0,  off, 0.0, "M [kNm]",        QColor(50, 200, 255))
        project(0.0, 0.0,  off, "χ·sin θ [1/m]",  QColor(50, 255, 50))

        painter.end()
        glEnable(GL_DEPTH_TEST)

    # ------------------------------------------------------------------
    # SUPERFICIE
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # SUPERFICIE
    # ------------------------------------------------------------------

    def _draw_surface(self) -> None:
        """Disegna il dominio M-χ come un volume di vetro con reticolo marcato."""
        glEnableClientState(GL_VERTEX_ARRAY)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glDepthMask(GL_FALSE)

        # --- 1. SUPERFICIE VETRO (GL_QUADS) ---
        # Opacità uniforme al 30% senza accumulo sui bordi
        glColor4f(0.70, 0.70, 0.70, 0.30)
        glVertexPointer(3, GL_FLOAT, 0, self._vbo_quads)
        glDrawArrays(GL_QUADS, 0, self._n_quads)

        # --- 2. WIREFRAME (GL_LINES) - Più opaco e definito ---
        glLineWidth(1.2)
        glColor4f(1.0, 1.0, 1.0, 0.40)
        glVertexPointer(3, GL_FLOAT, 0, self._vbo_lines)
        glDrawArrays(GL_LINES, 0, self._n_lines)

        # Ripristiniamo la scrittura della profondità
        glDepthMask(GL_TRUE)
        glDisableClientState(GL_VERTEX_ARRAY)

        # --- 3. BORDO ESTERNO DI ROTTURA (Specifico di questo widget) ---
        # Lo manteniamo perché è un'informazione visiva importante, 
        # disegnato con la profondità riattivata così si posiziona correttamente
        if self.normalized_matrix is not None:
            R, C, _ = self.normalized_matrix.shape
            glLineWidth(1.5)
            glBegin(GL_LINE_LOOP)
            glColor4f(1.0, 0.2, 0.2, 0.8)  # Rosso acceso semitrasparente
            for i in range(R):
                p = self.normalized_matrix[i, C - 1]
                glVertex3f(p[0], p[1], p[2])
            glEnd()

    # ------------------------------------------------------------------
    # ANELLO M_Ed
    # ------------------------------------------------------------------

    def _draw_ring(self) -> None:
        if self._ring_pts is None or not self._verifica_attiva:
            return

        inside = self._ring_inside

        if inside:
            # Verde — anello chiuso sulla superficie
            glLineWidth(2.5)
            glBegin(GL_LINE_LOOP)
            glColor4f(0.0, 0.85, 0.35, 1.0)
            for p in self._ring_pts:
                glVertex3f(p[0], p[1], p[2])
            glEnd()
        else:
            # Rosso — cerchio/anello fuori dominio
            glLineWidth(2.5)
            glBegin(GL_LINE_LOOP)
            glColor4f(1.0, 0.15, 0.15, 1.0)
            for p in self._ring_pts:
                glVertex3f(p[0], p[1], p[2])
            glEnd()

    # ------------------------------------------------------------------
    # MOUSE
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event) -> None:
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        btns = event.buttons()
        if btns & Qt.MiddleButton or btns & Qt.LeftButton:
            # Rotazione (coerente con spazio 3D elementi)
            self.rotation[1] += dx * 0.40
            self.rotation[0] += dy * 0.40
        elif btns & Qt.RightButton:
            # Traslazione (coerente con spazio 3D elementi)
            speed = self.scale * 0.003
            self.translation[0] += dx * speed
            self.translation[1] -= dy * speed
        self.last_pos = event.pos()
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 0.88
        self.scale = max(0.01, min(self.scale * factor, 200.0))
        self.update()
