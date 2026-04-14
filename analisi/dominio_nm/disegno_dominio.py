"""
disegno_dominio.py  –  analisi/dominio_nm/  (vOttimizzata VBO)
==============================================================
Widget OpenGL 3D per la visualizzazione del dominio di interazione N-Mx-My.

Caratteristiche e Ottimizzazioni:
  - Superficie semitrasparente (mesh grigia tecnica) + wireframe bianco.
  - Rendering tramite NumPy Arrays (Vertex Arrays) per massime prestazioni GPU.
  - Assi coordinati colorati: X=Mx (rosso), Y=N (ciano), Z=-My (verde).
  - Etichette degli assi graduate con QPainter sovrapposto.
  - Punto di verifica (N_Ed, Mx_Ed, My_Ed): verde=dentro / rosso=fuori.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore    import Qt, QPoint, pyqtSignal
from PyQt5.QtGui     import QColor, QFont, QPainter
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_LINES, GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, 
    GL_POINT_SMOOTH, GL_POINTS, GL_PROJECTION, GL_QUADS, GL_SRC_ALPHA,
    glBegin, glBlendFunc, glClear, glClearColor,
    glColor3f, glColor4f, glCullFace, glDepthMask,
    glDisable, glEnable, glEnd, glGetDoublev, glGetIntegerv,
    glLineWidth, glLoadIdentity, glMatrixMode, glPointSize,
    glRotatef, glScalef, glShadeModel, glTranslatef, glVertex3f,
    glViewport,
    GL_BACK, GL_FRONT, GL_FRONT_AND_BACK, GL_SMOOTH,
    GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT,
    GL_FALSE, GL_TRUE,
    glEnableClientState, glDisableClientState, glVertexPointer, 
    glDrawArrays, GL_VERTEX_ARRAY, GL_FLOAT,
    # --- NUOVI IMPORT AGGIUNTI QUI ---
    glColorMask, glDepthFunc, GL_LEQUAL, GL_LESS
)
from OpenGL.GLU import gluPerspective, gluProject

try:
    from scipy.spatial import Delaunay as _Delaunay
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ==============================================================================
# WIDGET 3D
# ==============================================================================

class DominioWidget3D(QOpenGLWidget):
    """
    Visualizza il dominio di interazione N-Mx-My in 3D sfruttando Vertex Arrays.
    """

    verifica_cambiata = pyqtSignal(bool)   # True = dentro il dominio

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Dati dominio
        self.points_matrix:     np.ndarray | None = None
        self.normalized_matrix: np.ndarray | None = None
        self._delaunay                            = None
        self.normalization_factors = np.ones(3)

        # Buffer Array per la GPU (Prestazioni)
        self._vbo_quads  = np.array([], dtype=np.float32)
        self._vbo_lines  = np.array([], dtype=np.float32)
        self._vbo_points = np.array([], dtype=np.float32)
        self._n_quads  = 0
        self._n_lines  = 0
        self._n_points = 0

        # Punto di verifica (valori reali, non normalizzati)
        self._N  = 0.0
        self._Mx = 0.0
        self._My = 0.0
        self._is_inside       = False
        self._verifica_attiva = False   # True solo se il dominio è presente

        # Stato vista
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
        if points_matrix is None or points_matrix.size == 0:
            self.points_matrix     = None
            self.normalized_matrix = None
            self._delaunay         = None
            self._verifica_attiva  = False
            self._n_quads = self._n_lines = self._n_points = 0
            self.update()
            return

        self.points_matrix = points_matrix

        # Calcolo fattori di normalizzazione [Max_Mx, Max_My, Max_N]
        flat_pts = points_matrix.reshape(-1, 3)
        max_vals = np.max(np.abs(flat_pts), axis=0)
        self.normalization_factors = np.where(max_vals > 1e-9, max_vals, 1.0)

        # Normalizzazione e riorganizzazione per OpenGL (X=Mx, Y=N, Z=-My)
        norm_flat = flat_pts / self.normalization_factors
        transformed = np.column_stack((
            norm_flat[:, 0],    # X = Mx
            norm_flat[:, 2],    # Y = N   (verticale)
            -norm_flat[:, 1],   # Z = -My (profondità)
        ))

        rows, cols, _ = points_matrix.shape
        self.normalized_matrix = transformed.reshape(rows, cols, 3)

        # --- PREPARA I DATI PER LA GPU ---
        self._build_vertex_arrays()

        # Triangolazione Delaunay per verifica dentro/fuori
        if _HAS_SCIPY:
            try:
                self._delaunay = _Delaunay(transformed)
            except Exception as e:
                print(f"WARN  dominio 3D: Delaunay fallito ({e})")
                self._delaunay = None
        else:
            self._delaunay = None

        self._verifica_attiva = True
        self._aggiorna_verifica()
        self.update()

    def set_verification_point(self, N: float, Mx: float, My: float) -> bool | None:
        self._N  = N
        self._Mx = Mx
        self._My = My
        return self._aggiorna_verifica()

    def reset_view(self) -> None:
        self.rotation    = [30.0, -45.0, 0.0]
        self.translation = [0.0,  0.0,  -3.5]
        self.scale       = 1.0
        self.update()

    # ------------------------------------------------------------------
    # MOTORE VBO E VERIFICA
    # ------------------------------------------------------------------

    def _build_vertex_arrays(self) -> None:
        """Pre-calcola gli array per la GPU, raggruppando facce e linee."""
        mat = self.normalized_matrix
        if mat is None:
            return
        
        R, C, _ = mat.shape
        
        quads = []
        lines = []

        for i in range(R):
            ni = (i + 1) % R
            
            # --- Array Facce (GL_QUADS) ---
            for j in range(C - 1):
                quads.append(mat[i,  j])
                quads.append(mat[ni, j])
                quads.append(mat[ni, j + 1])
                quads.append(mat[i,  j + 1])
            
            # --- Array Wireframe (GL_LINES) ---
            for j in range(C - 1):
                # Meridiano: dal punto corrente al prossimo parallelo
                lines.append(mat[i, j])
                lines.append(mat[i, j + 1])
                # Parallelo: dal punto corrente al prossimo meridiano
                lines.append(mat[i, j])
                lines.append(mat[ni, j])
            
            # Chiudiamo l'ultimo anello orizzontale (parallelo estremo)
            lines.append(mat[i, C - 1])
            lines.append(mat[ni, C - 1])

        # Array Numpy finali
        self._vbo_quads = np.array(quads, dtype=np.float32)
        self._n_quads = self._vbo_quads.shape[0]

        self._vbo_lines = np.array(lines, dtype=np.float32)
        self._n_lines = self._vbo_lines.shape[0]

        # Array Punti (GL_POINTS)
        self._vbo_points = np.ascontiguousarray(mat, dtype=np.float32).reshape(-1, 3)
        self._n_points = self._vbo_points.shape[0]

    def _aggiorna_verifica(self) -> bool | None:
        if not self._verifica_attiva or self._delaunay is None:
            return None

        x = self._Mx / self.normalization_factors[0]
        y = self._N  / self.normalization_factors[2]
        z = -self._My / self.normalization_factors[1]

        simplex = self._delaunay.find_simplex(np.array([x, y, z]))
        inside  = bool(simplex >= 0)

        if inside != self._is_inside:
            self._is_inside = inside
            self.verifica_cambiata.emit(inside)
        else:
            self._is_inside = inside

        self.update()
        return inside

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
            self._draw_structured_mesh()
            self._draw_verification_point()

    # ------------------------------------------------------------------
    # DISEGNO GRIGLIA E ASSI
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        glColor3f(0.30, 0.30, 0.30)
        glLineWidth(0.8)
        step = 2 * self.axis_length / self.axis_labels
        glBegin(GL_LINES)
        for i in range(self.axis_labels + 1):
            z = -self.axis_length + i * step
            if abs(z) < 1e-5: continue
            glVertex3f(-self.axis_length, 0.0, z)
            glVertex3f( self.axis_length, 0.0, z)
        for i in range(self.axis_labels + 1):
            x = -self.axis_length + i * step
            if abs(x) < 1e-5: continue
            glVertex3f(x, 0.0, -self.axis_length)
            glVertex3f(x, 0.0,  self.axis_length)
        glEnd()

    def _draw_axes(self) -> None:
        al = self.axis_length
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)                                              # X – Mx rosso
        glVertex3f(-al-0.1, 0.0, 0.0); glVertex3f(al+0.1, 0.0, 0.0)
        glColor3f(0.0, 0.8, 1.0)                                              # Y – N  ciano
        glVertex3f(0.0, -al-0.1, 0.0); glVertex3f(0.0, al+0.1, 0.0)
        glColor3f(0.0, 1.0, 0.0)                                              # Z – My verde
        glVertex3f(0.0, 0.0, -al-0.1); glVertex3f(0.0, 0.0, al+0.1)
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
            sx = int(sc[0]);  sy = int(h - sc[1])
            if not (0 <= sx <= w and 0 <= sy <= h):
                return
            painter.setPen(pen)
            painter.drawText(sx + dx, sy + dy, text)

        al = self.axis_length
        for i in range(self.axis_labels + 1):
            t = i / float(self.axis_labels)
            v = -al + 2 * al * t

            val_mx = v * self.normalization_factors[0]
            project(v, 0.0, 0.0, f"{val_mx:.4g}", QColor(255, 100, 100), dx=-15, dy=20)

            val_n = v * self.normalization_factors[2]
            project(0.0, v, 0.0, f"{val_n:.4g}", QColor(100, 200, 255), dx=-28, dy=5)

            val_my = -v * self.normalization_factors[1]
            project(0.0, 0.0, v, f"{val_my:.4g}", QColor(100, 255, 100), dx=8, dy=-10)

        off = al * 1.15
        project( off, 0.0, 0.0, "Mx [kNm]", QColor(255, 50,  50))
        project(0.0,  off, 0.0, "N [kN]",   QColor(50, 200, 255))
        project(0.0, 0.0,  off, "My [kNm]", QColor(50, 255,  50))

        painter.end()
        glEnable(GL_DEPTH_TEST)

    # ------------------------------------------------------------------
    # DISEGNO DOMINIO (GPU OTTIMIZZATO)
    # ------------------------------------------------------------------

    def _draw_structured_mesh(self) -> None:
        """Disegna il dominio come un volume di vetro con reticolo marcato (senza punti)."""
        glEnableClientState(GL_VERTEX_ARRAY)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glDepthMask(GL_FALSE)

        # --- 1. SUPERFICIE VETRO (GL_QUADS) ---
        glColor4f(0.70, 0.70, 0.70, 0.30)
        glVertexPointer(3, GL_FLOAT, 0, self._vbo_quads)
        glDrawArrays(GL_QUADS, 0, self._n_quads)

        # --- 2. WIREFRAME (GL_LINES) ---
        glLineWidth(1.2)
        glColor4f(1.0, 1.0, 1.0, 0.40)
        glVertexPointer(3, GL_FLOAT, 0, self._vbo_lines)
        glDrawArrays(GL_LINES, 0, self._n_lines)

        # Il blocco "3. PUNTI VERTICE (GL_POINTS)" è stato rimosso.

        glDepthMask(GL_TRUE)
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_verification_point(self) -> None:
        if not self._verifica_attiva:
            return

        nf = self.normalization_factors
        x  = self._Mx / nf[0] if nf[0] > 1e-12 else 0.0
        y  = self._N  / nf[2] if nf[2] > 1e-12 else 0.0
        z  = -self._My / nf[1] if nf[1] > 1e-12 else 0.0

        col = (0.0, 1.0, 0.0) if self._is_inside else (1.0, 0.0, 0.0)

        glPointSize(12.0)
        glBegin(GL_POINTS)
        glColor3f(*col)
        glVertex3f(x, y, z)
        glEnd()

        glLineWidth(1.2)
        glBegin(GL_LINES)
        glColor3f(col[0] * 0.5, col[1] * 0.5, col[2] * 0.5)
        glVertex3f(x, 0.0, z)
        glVertex3f(x, y,   z)
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