"""
disegno_dominio_2d.py  –  analisi/dominio_nm/  (vOttimizzata VBO)
=================================================================
Widget OpenGL 2D per la visualizzazione delle sezioni piane del dominio
di interazione N-Mx-My.

Ottimizzazioni:
  - Triangolazione di Delaunay rimossa dal ciclo di rendering (paintGL) 
    e spostata nella fase di pre-calcolo.
  - Rendering del dominio tramite Vertex Arrays (NumPy -> GPU) per 
    eliminare il bottleneck della CPU.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore    import Qt, QPoint, pyqtSignal
from PyQt5.QtGui     import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_LINE_LOOP, GL_LINE_SMOOTH,
    GL_LINES, GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_POINT_SMOOTH,
    GL_POINTS, GL_PROJECTION, GL_SRC_ALPHA, GL_TRIANGLE_FAN, GL_TRIANGLES,
    glBegin, glBlendFunc, glClear, glClearColor, glColor3f, glColor4f,
    glDisable, glEnable, glEnd, glLineWidth, glLoadIdentity,
    glMatrixMode, glOrtho, glPointSize, glVertex2f, glViewport,
    # --- NUOVI IMPORT PER LE PRESTAZIONI ---
    glEnableClientState, glDisableClientState, glVertexPointer, 
    glDrawArrays, GL_VERTEX_ARRAY, GL_FLOAT
)

try:
    from scipy.spatial import Delaunay as _Delaunay
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ==============================================================================
# WIDGET 2D
# ==============================================================================

class DominioWidget2D(QOpenGLWidget):
    """
    Visualizza la sezione 2D del dominio di interazione nel piano scelto.
    """

    verifica_cambiata = pyqtSignal(bool)

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Dati dominio
        self._points_3d:          np.ndarray | None = None
        self._structured_pts:     np.ndarray | None = None   # (rows, cols, 3)
        self._structured:         bool              = False
        self._rows:               int | None        = None
        self._cols:               int | None        = None

        # Poligono/i di sezione calcolati
        self._slice_polygon:  np.ndarray | None = None
        self._slice_polygons: list              = []

        # Buffer Array per la GPU (Prestazioni)
        self._vbo_tri_verts     = np.array([], dtype=np.float32)
        self._vbo_border_verts  = np.array([], dtype=np.float32)
        self._vbo_sec_borders   = []
        self._n_tri_verts       = 0
        self._n_border_verts    = 0

        # Vista
        self.view_mode    = "N_Mx"
        self._pan_x       = 0.0
        self._pan_y       = 0.0
        self._zoom        = 1.0
        self._last_mouse  = None

        # Range dati per la proiezione ortografica
        self._range_x = 10.0
        self._range_y = 10.0
        self._global_ranges: dict = {'Mx': 10.0, 'My': 10.0, 'N': 100.0}

        # Punto di verifica
        self._N  = 0.0
        self._Mx = 0.0
        self._My = 0.0
        self._is_inside      = False
        self._verifica_color = QColor(255, 0, 0)

        self._cursor = QPoint(0, 0)
        self.font    = QFont("Arial", 9)

        self.setMouseTracking(True)

        # Colori assi
        self._ax_colors = {
            'N':  QColor(0,  200, 255),
            'Mx': QColor(255, 50,  50),
            'My': QColor(50,  255, 50),
        }

    # ------------------------------------------------------------------
    # API PUBBLICA
    # ------------------------------------------------------------------

    def set_points(self, points: np.ndarray | None) -> None:
        if points is None or (hasattr(points, '__len__') and len(points) == 0):
            self._points_3d      = None
            self._structured     = False
            self._structured_pts = None
            self._slice_polygon  = None
            self._slice_polygons = []
            self._build_vertex_arrays() # Azzera la GPU
            self.update()
            return

        pts = np.asarray(points, dtype=np.float64)

        if pts.ndim == 3 and pts.shape[2] == 3:
            self._rows, self._cols = pts.shape[0], pts.shape[1]
            self._points_3d        = pts.reshape(-1, 3)
            self._structured_pts   = pts
            self._structured       = True
        elif pts.ndim == 2 and pts.shape[1] == 3:
            self._points_3d        = pts
            self._structured       = False
            self._structured_pts   = None
            self._rows = self._cols = None
        else:
            return

        if len(self._points_3d) > 0:
            self._global_ranges = {
                'Mx': float(np.max(np.abs(self._points_3d[:, 0]))) or 10.0,
                'My': float(np.max(np.abs(self._points_3d[:, 1]))) or 10.0,
                'N':  float(np.max(np.abs(self._points_3d[:, 2]))) or 100.0,
            }

        self._aggiorna_range_per_vista()
        self._reset_zoom_pan()
        self._calcola_sezione()
        self.update()

    def set_view_mode(self, mode: str) -> None:
        if mode not in ('N_Mx', 'N_My', 'Mx_My'):
            return
        self.view_mode = mode
        self._aggiorna_range_per_vista()
        self._reset_zoom_pan()
        self._calcola_sezione()
        self.update()

    def set_verification_point(self, N: float, Mx: float, My: float) -> bool | None:
        old_cut = self._get_cut_value()
        self._N  = N
        self._Mx = Mx
        self._My = My
        new_cut = self._get_cut_value()

        if abs(new_cut - old_cut) > 1e-9 or self._slice_polygon is None:
            self._calcola_sezione()

        self._check_inside()
        self.update()
        return self._is_inside if self._slice_polygon is not None else None

    def reset_view(self) -> None:
        self._aggiorna_range_per_vista()
        self._reset_zoom_pan()
        self.update()

    # ------------------------------------------------------------------
    # GEOMETRIA INTERNA E VBO
    # ------------------------------------------------------------------

    def _get_cut_value(self) -> float:
        if self.view_mode == 'N_Mx':   return self._My
        if self.view_mode == 'N_My':   return self._Mx
        return self._N   # Mx_My

    def _aggiorna_range_per_vista(self) -> None:
        gr = self._global_ranges
        if self.view_mode == 'N_Mx':
            self._range_x = gr['Mx'] * 1.25
            self._range_y = gr['N']  * 1.25
        elif self.view_mode == 'N_My':
            self._range_x = gr['My'] * 1.25
            self._range_y = gr['N']  * 1.25
        else:
            self._range_x = gr['Mx'] * 1.25
            self._range_y = gr['My'] * 1.25

    def _reset_zoom_pan(self) -> None:
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._zoom  = 1.0

    def _calcola_sezione(self) -> None:
        """Calcola il poligono 2D e costruisce le memorie GPU."""
        self._slice_polygon  = None
        self._slice_polygons = []

        if self._points_3d is None or len(self._points_3d) < 4:
            self._build_vertex_arrays()
            return

        if self.view_mode == 'N_Mx':   ix, iy, icut = 0, 2, 1
        elif self.view_mode == 'N_My': ix, iy, icut = 1, 2, 0
        else:                          ix, iy, icut = 0, 1, 2

        cut_val = self._get_cut_value()

        pts = self._points_3d
        triangles = []

        if self._structured and self._structured_pts is not None:
            S = self._structured_pts
            rows, cols = S.shape[0], S.shape[1]
            for i in range(rows):
                ni = (i + 1) % rows
                for j in range(cols - 1):
                    a = i  * cols + j
                    b = ni * cols + j
                    c = ni * cols + j + 1
                    d = i  * cols + j + 1
                    triangles.append((a, b, c))
                    triangles.append((a, c, d))
        elif _HAS_SCIPY:
            try:
                proj2d = pts[:, [ix, iy]]
                tri    = _Delaunay(proj2d)
                triangles = [tuple(s) for s in tri.simplices]
            except Exception:
                pass

        if not triangles:
            self._build_vertex_arrays()
            return

        segments = []
        for i0, i1, i2 in triangles:
            vs    = pts[[i0, i1, i2]]
            dists = vs[:, icut] - cut_val

            if np.all(dists > 0) or np.all(dists < 0):
                continue

            inter_pts = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                da, db = dists[a], dists[b]
                if abs(da) < 1e-12:
                    inter_pts.append([vs[a, ix], vs[a, iy]])
                if (da > 0) != (db > 0) and abs(da - db) > 1e-15:
                    t = da / (da - db)
                    p = vs[a] + t * (vs[b] - vs[a])
                    inter_pts.append([float(p[ix]), float(p[iy])])

            if len(inter_pts) >= 2:
                arr = np.array(inter_pts)
                if len(arr) > 2:
                    dists_mat = np.linalg.norm(
                        arr[:, None] - arr[None, :], axis=2
                    )
                    i_a, i_b = np.unravel_index(np.argmax(dists_mat), dists_mat.shape)
                    arr = arr[[i_a, i_b]]
                segments.append((arr[0].tolist(), arr[1].tolist()))

        if not segments:
            self._build_vertex_arrays()
            return

        all_c  = np.vstack([np.array(s[0]) for s in segments] +
                            [np.array(s[1]) for s in segments])
        rng    = np.ptp(all_c, axis=0)
        tol    = max(1e-8, 1e-5 * float(np.max(rng + 1e-12)))
        dec    = max(0, int(-np.floor(np.log10(tol))) if tol < 1 else 0)

        def qkey(p):
            return (round(float(p[0]), dec), round(float(p[1]), dec))

        key_to_coord: dict = {}
        adj: dict          = {}
        for sa, sb in segments:
            ka, kb = qkey(sa), qkey(sb)
            key_to_coord.setdefault(ka, (float(sa[0]), float(sa[1])))
            key_to_coord.setdefault(kb, (float(sb[0]), float(sb[1])))
            adj.setdefault(ka, []).append(kb)
            adj.setdefault(kb, []).append(ka)

        used   = set()
        loops  = []

        def eid(u, v):
            return (u, v) if u <= v else (v, u)

        for start in list(adj.keys()):
            for nbr in list(adj.get(start, [])):
                e = eid(start, nbr)
                if e in used:
                    continue
                path = [start, nbr]
                used.add(e)
                curr, prev = nbr, start
                while True:
                    neighs     = adj.get(curr, [])
                    candidates = [n for n in neighs if n != prev]
                    next_node  = None
                    for cand in candidates:
                        if eid(curr, cand) not in used:
                            next_node = cand
                            break
                    if next_node is None:
                        break
                    e2 = eid(curr, next_node)
                    if e2 in used:
                        if next_node == path[0]:
                            path.append(next_node)
                        break
                    used.add(e2)
                    path.append(next_node)
                    prev, curr = curr, next_node
                    if path[-1] == path[0]:
                        break

                if len(path) >= 3 and path[0] == path[-1]:
                    coords = [key_to_coord[k] for k in path[:-1]]
                    loops.append(np.array(coords))

        if not loops:
            self._build_vertex_arrays()
            return

        def poly_area(poly):
            x, y = poly[:, 0], poly[:, 1]
            return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) -
                                   np.dot(y, np.roll(x, -1))))

        areas    = [poly_area(lp) for lp in loops]
        best     = loops[int(np.argmax(areas))]

        if np.cross(best[1] - best[0], best[2] - best[1]) < 0:
            best = best[::-1]

        self._slice_polygons = loops
        self._slice_polygon  = best
        self._check_inside()
        
        # ORA CALCOLA I VERTEX ARRAYS (Una volta sola!)
        self._build_vertex_arrays()

    def _build_vertex_arrays(self) -> None:
        """Sposta la pesantissima triangolazione di Delaunay e la creazione dei poligoni qui, fuori dal ciclo di paintGL."""
        poly = self._slice_polygon
        if poly is None or len(poly) < 3:
            self._n_tri_verts = 0
            self._n_border_verts = 0
            self._vbo_sec_borders = []
            return

        # 1. Triangoli per l'area di riempimento (GL_TRIANGLES)
        tri_verts = []
        if _HAS_SCIPY:
            try:
                tri = _Delaunay(poly)
                for t in tri.simplices:
                    p0, p1, p2 = poly[t[0]], poly[t[1]], poly[t[2]]
                    
                    # Filtra concavità
                    cx = float(p0[0] + p1[0] + p2[0]) / 3.0
                    cy = float(p0[1] + p1[1] + p2[1]) / 3.0
                    if self._point_in_polygon(cx, cy, poly):
                        tri_verts.extend([p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]])
            except Exception:
                tri_verts = self._build_fan_fallback(poly)
        else:
            tri_verts = self._build_fan_fallback(poly)

        self._vbo_tri_verts = np.array(tri_verts, dtype=np.float32)
        self._n_tri_verts = len(tri_verts) // 2

        # 2. Bordo principale (GL_LINE_LOOP)
        self._vbo_border_verts = np.array(poly, dtype=np.float32)
        self._n_border_verts = len(poly)

        # 3. Bordi secondari
        self._vbo_sec_borders = []
        for other in self._slice_polygons:
            if other.shape == poly.shape and np.allclose(other, poly, atol=1e-6):
                continue
            self._vbo_sec_borders.append(np.array(other, dtype=np.float32))

    def _build_fan_fallback(self, poly: np.ndarray) -> list:
        # Trasforma la logica FAN in triangoli semplici per usare un unico VBO
        tri_verts = []
        cx = float(np.mean(poly[:, 0]))
        cy = float(np.mean(poly[:, 1]))
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            tri_verts.extend([cx, cy, p1[0], p1[1], p2[0], p2[1]])
        return tri_verts

    def _point_in_polygon(self, x: float, y: float,
                           poly: np.ndarray) -> bool:
        n      = len(poly)
        inside = False
        j      = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    def _check_inside(self) -> None:
        if self._slice_polygon is None:
            self._is_inside      = False
            self._verifica_color = QColor(255, 0, 0)
            return

        if self.view_mode == 'N_Mx':   px, py = self._Mx, self._N
        elif self.view_mode == 'N_My': px, py = self._My, self._N
        else:                          px, py = self._Mx, self._My

        inside = self._point_in_polygon(px, py, self._slice_polygon)

        if inside != self._is_inside:
            self._is_inside = inside
            self.verifica_cambiata.emit(inside)
        else:
            self._is_inside = inside

        self._verifica_color = QColor(0, 200, 80) if inside else QColor(220, 60, 60)

    # ------------------------------------------------------------------
    # OPENGL – INIT / RESIZE / PAINT
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        glClearColor(40/255, 40/255, 40/255, 1.0)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        rx = self._range_x * self._zoom
        ry = self._range_y * self._zoom
        glOrtho(-rx + self._pan_x,  rx + self._pan_x,
                -ry + self._pan_y,  ry + self._pan_y,
                -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        self._gl_grid()
        self._gl_domain()
        self._gl_verification_point()
        self._gl_axes_and_labels()
        self._gl_tracker()

    # ------------------------------------------------------------------
    # DISEGNO GL
    # ------------------------------------------------------------------

    def _gl_grid(self) -> None:
        # La griglia rimane in immediate mode: sono pochissime linee, calcolarle dinamicamente
        # col VBO ad ogni spostamento non darebbe vantaggi percettibili.
        rx = self._range_x * self._zoom
        ry = self._range_y * self._zoom
        wx_min = -rx + self._pan_x;  wx_max = rx + self._pan_x
        wy_min = -ry + self._pan_y;  wy_max = ry + self._pan_y

        tx = self._tick(wx_max - wx_min)
        ty = self._tick(wy_max - wy_min)

        glColor3f(0.20, 0.20, 0.20)
        glLineWidth(0.8)
        glBegin(GL_LINES)
        x = np.floor(wx_min / tx) * tx
        while x <= wx_max + 1e-12:
            glVertex2f(x, wy_min); glVertex2f(x, wy_max); x += tx
        y = np.floor(wy_min / ty) * ty
        while y <= wy_max + 1e-12:
            glVertex2f(wx_min, y); glVertex2f(wx_max, y); y += ty
        glEnd()

    def _gl_domain(self) -> None:
        """Disegna il dominio passando la memoria VBO pre-calcolata."""
        if self._n_tri_verts == 0 and self._n_border_verts == 0:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnableClientState(GL_VERTEX_ARRAY)
        
        # 1. Area Interna (GL_TRIANGLES)
        if self._n_tri_verts > 0:
            glColor4f(0.5, 0.5, 0.5, 0.4)
            glVertexPointer(2, GL_FLOAT, 0, self._vbo_tri_verts)
            glDrawArrays(GL_TRIANGLES, 0, self._n_tri_verts)

        # 2. Bordo Principale (GL_LINE_LOOP)
        if self._n_border_verts > 0:
            glLineWidth(1.8)
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glVertexPointer(2, GL_FLOAT, 0, self._vbo_border_verts)
            glDrawArrays(GL_LINE_LOOP, 0, self._n_border_verts)

        # 3. Bordi Secondari (Buchi o altre sezioni)
        if self._vbo_sec_borders:
            glLineWidth(0.8)
            glColor4f(1.0, 1.0, 1.0, 0.45)
            for sec_border in self._vbo_sec_borders:
                glVertexPointer(2, GL_FLOAT, 0, sec_border)
                glDrawArrays(GL_LINE_LOOP, 0, len(sec_border))

        glDisableClientState(GL_VERTEX_ARRAY)

    def _gl_verification_point(self) -> None:
        if self.view_mode == 'N_Mx':   px, py = self._Mx, self._N
        elif self.view_mode == 'N_My': px, py = self._My, self._N
        else:                          px, py = self._Mx, self._My

        col = self._verifica_color
        glColor3f(col.redF(), col.greenF(), col.blueF())
        glPointSize(11.0)
        glBegin(GL_POINTS)
        glVertex2f(px, py)
        glEnd()

    def _gl_axes_and_labels(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        fm = painter.fontMetrics()
        W, H = self.width(), self.height()

        rx = self._range_x * self._zoom
        ry = self._range_y * self._zoom
        wx_min = -rx + self._pan_x;  wx_max = rx + self._pan_x
        wy_min = -ry + self._pan_y;  wy_max = ry + self._pan_y

        def to_screen(wx, wy):
            nx = (wx - wx_min) / (wx_max - wx_min)
            ny = (wy - wy_min) / (wy_max - wy_min)
            return int(nx * W), int((1 - ny) * H)

        if self.view_mode == 'N_Mx':
            col_x, lab_x = self._ax_colors['Mx'], "Mx [kNm]"
            col_y, lab_y = self._ax_colors['N'],  "N [kN]"
        elif self.view_mode == 'N_My':
            col_x, lab_x = self._ax_colors['My'], "My [kNm]"
            col_y, lab_y = self._ax_colors['N'],  "N [kN]"
        else:
            col_x, lab_x = self._ax_colors['Mx'], "Mx [kNm]"
            col_y, lab_y = self._ax_colors['My'], "My [kNm]"

        sx0, sy0 = to_screen(0.0, 0.0)

        pen = QPen(col_x, 1)
        painter.setPen(pen)
        painter.drawLine(0, sy0, W, sy0)

        pen.setColor(col_y)
        painter.setPen(pen)
        painter.drawLine(sx0, 0, sx0, H)

        tx = self._tick(wx_max - wx_min)
        painter.setPen(col_x)
        x = np.floor(wx_min / tx) * tx
        while x <= wx_max:
            if abs(x) > 1e-9:
                sx, sy = to_screen(x, 0.0)
                painter.drawLine(sx, sy - 4, sx, sy + 4)
                lbl = f"{x:.4g}"
                tw  = fm.horizontalAdvance(lbl)
                txt_y = min(max(sy + fm.height() + 2, 14), H - 4)
                painter.drawText(sx - tw // 2, txt_y, lbl)
            x += tx

        ty = self._tick(wy_max - wy_min)
        painter.setPen(col_y)
        y = np.floor(wy_min / ty) * ty
        while y <= wy_max:
            if abs(y) > 1e-9:
                sx, sy = to_screen(0.0, y)
                painter.drawLine(sx - 4, sy, sx + 4, sy)
                lbl = f"{y:.4g}"
                tw  = fm.horizontalAdvance(lbl)
                txt_x = min(max(sx - tw - 6, 4), W - tw - 4)
                painter.drawText(txt_x, sy + fm.height() // 2 - 2, lbl)
            y += ty

        painter.setPen(col_x)
        tw = fm.horizontalAdvance(lab_x)
        painter.drawText(W - tw - 8, H - 8, lab_x)
        painter.setPen(col_y)
        painter.save()
        painter.translate(18, min(110, H - 20))
        painter.rotate(-90)
        painter.drawText(0, 0, lab_y)
        painter.restore()

        painter.end()

    def _gl_tracker(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        W, H = self.width(), self.height()
        cx, cy = self._cursor.x(), self._cursor.y()

        painter.setPen(QPen(QColor(150, 150, 150, 80), 1))
        painter.drawLine(cx, 0, cx, H)
        painter.drawLine(0, cy, W, cy)

        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        sz = 10
        painter.drawLine(cx - sz, cy, cx + sz, cy)
        painter.drawLine(cx, cy - sz, cx, cy + sz)

        wx, wy = self._s2w(cx, cy)
        painter.setPen(QColor(220, 220, 220, 200))
        painter.drawText(cx + 12, H - 10, f"{wx:.4g}")
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
            rx = self._range_x * self._zoom
            ry = self._range_y * self._zoom
            self._pan_x -= dx * (2 * rx) / max(self.width(),  1)
            self._pan_y += dy * (2 * ry) / max(self.height(), 1)
            self._last_mouse = e.pos()
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._last_mouse = None
        self.update()

    def wheelEvent(self, e) -> None:
        delta = e.angleDelta().y()
        if delta == 0:
            return
        sx, sy   = e.pos().x(), e.pos().y()
        wx0, wy0 = self._s2w(sx, sy)
        factor   = 1.0 - np.sign(delta) * 0.10
        self._zoom *= factor
        self._zoom  = max(0.001, min(self._zoom, 200.0))
        wx1, wy1 = self._s2w(sx, sy)
        self._pan_x += wx0 - wx1
        self._pan_y += wy0 - wy1
        self.update()

    # ------------------------------------------------------------------
    # UTILITÀ
    # ------------------------------------------------------------------

    def _s2w(self, sx: int, sy: int):
        W, H = self.width(), self.height()
        if W == 0 or H == 0:
            return 0.0, 0.0
        rx = self._range_x * self._zoom
        ry = self._range_y * self._zoom
        nx = sx / W
        ny = 1.0 - sy / H
        return (-rx + self._pan_x + nx * 2 * rx,
                -ry + self._pan_y + ny * 2 * ry)

    @staticmethod
    def _tick(rng: float) -> float:
        rough = rng / 10.0
        if rough <= 0:
            return 1.0
        mag   = 10 ** np.floor(np.log10(max(rough, 1e-12)))
        ratio = rough / mag
        if   ratio >= 5: return float(5 * mag)
        elif ratio >= 2: return float(2 * mag)
        return float(mag)