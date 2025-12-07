from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QPoint
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

class Domain2DWidget(QOpenGLWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        # 3D data
        self.points = None
        # 2D-projection data (must exist before any view-mode logic)
        self.points_2d = None
        self.hull_2d   = None
        self.view_mode = "3D"
        # 2D view settings
        self.pan_2d = [0.0, 0.0]
        self.zoom_2d = 1.0
        self.last_mouse_pos = None
        self.data_range_x = 10.0
        self.data_range_y = 10.0
        self.cursor_pos = QPoint(0, 0)
        self.setMouseTracking(True)
        self.font = QFont("Arial", 10)

        # Axis labels
        self.axis_labels = {
            "N_Mx": ("Mx [kNm]", "N [kN]"),
            "N_My": ("My [kNm]", "N [kN]"),
            "Mx_My": ("Mx [kNm]", "My [kNm]")
        }

        self.axis_colors = {
            "N": QColor(0, 200, 255),   # Cyan/blue for N
            "Mx": QColor(255, 0, 0),     # Red for Mx
            "My": QColor(0, 255, 0)      # Green for My
        }

        # Color settings
        self.min_N = 0.0
        self.max_N = 0.0

        # Verification point
        self.verification_point = [0.0, 0.0, 0.0]  # Mx, My, N
        self.verification_color = QColor(255, 0, 0)  # Rosso di default

        # Plane intersection results:
        # plane_intersection_2d: Nx2 array with coordinates in the 2D view (e.g. Mx,N)
        # plane_intersection_n: N-array with the corresponding N values (useful per-vertex color)
        # plane_intersection_3d: Nx3 array with original 3D points of intersection
        self.plane_intersection_2d = None
        self.plane_intersection_n = None
        self.plane_intersection_3d = None

        # UI connections
        try:
            self.ui.out_Mx.textChanged.connect(self.update_verification_point)
            self.ui.out_My.textChanged.connect(self.update_verification_point)
            self.ui.out_N.textChanged.connect(self.update_verification_point)
        except Exception:
            pass

        try:
            self.ui.btn_out_verifica.clicked.connect(self.update_verification_point)
        except Exception:
            pass

    def update_verification_point(self):
        """Aggiorna il punto di verifica in base ai valori nell'UI"""
        try:
            Mx = float(self.ui.out_Mx.text() or "0")
            My = float(self.ui.out_My.text() or "0")
            N = float(self.ui.out_N.text() or "0")
            self.verification_point = [Mx, My, N]

            # Ricalcola l'intersezione piano/dominio in funzione del nuovo punto di verifica
            self._compute_plane_intersection()

            # Determina se il punto è interno al dominio (usa l'intersezione 2D)
            self._update_verification_status()

        except ValueError:
            # Valori non validi
            self.verification_color = QColor(255, 0, 0)
            try:
                self.ui.out_testo_punto.setText("Valori non validi")
            except Exception:
                pass

        self.update()

    def _update_verification_status(self):
        """Determina se il punto di verifica è interno al dominio"""
        Mx, My, N = self.verification_point

        if self.plane_intersection_2d is None or len(self.plane_intersection_2d) < 3:
            self.verification_color = QColor(255, 0, 0)
            try:
                self.ui.out_testo_punto.setText("Dominio non definito")
            except Exception:
                pass
            return

        # Proietta il punto nella vista corrente
        if self.view_mode == "N_Mx":
            point_2d = [Mx, N]
        elif self.view_mode == "N_My":
            point_2d = [My, N]
        elif self.view_mode == "Mx_My":
            point_2d = [Mx, My]
        else:
            return

        # Controlla se il punto è dentro l'inviluppo convesso dell'intersezione
        try:
            tri = Delaunay(self.plane_intersection_2d)
            inside = tri.find_simplex([point_2d]) >= 0

            if inside:
                self.verification_color = QColor(0, 255, 0)  # Verde
                try:
                    self.ui.out_testo_punto.setText("Verifica soddisfatta")
                except Exception:
                    pass
            else:
                self.verification_color = QColor(255, 0, 0)  # Rosso
                try:
                    self.ui.out_testo_punto.setText("Verifica non soddisfatta")
                except Exception:
                    pass

        except Exception as e:
            self.verification_color = QColor(255, 0, 0)
            try:
                self.ui.out_testo_punto.setText(f"Errore: {str(e)}")
            except Exception:
                pass

    def set_view_mode(self, mode):
        self.view_mode = mode
        if mode != "3D":
            self._update_2d_data()
            # Recompute plane intersection because projection axes changed
            self._compute_plane_intersection()
            self._update_verification_status()
        self.update()

    def set_points(self, points):
        """points: numpy array shape (N,3) with columns [Mx, My, N]"""
        self.points = points
        if points is not None and len(points) > 0:
            self.min_N = float(np.min(points[:, 2]))
            self.max_N = float(np.max(points[:, 2]))
        if self.view_mode != "3D":
            self._update_2d_data()
        # ricomputiamo l'intersezione piano/dominio
        self._compute_plane_intersection()
        self.update()

    def _update_2d_data(self):
        if self.points is None:
            self.points_2d = None
            self.hull_2d = None
            return

        # Project 3D points to 2D based on view mode
        if self.view_mode == "N_Mx":
            points_2d = self.points[:, [0, 2]]  # Mx orizzontale, N verticale
        elif self.view_mode == "N_My":
            points_2d = self.points[:, [1, 2]]  # My orizzontale, N verticale
        elif self.view_mode == "Mx_My":
            points_2d = self.points[:, [0, 1]]  # Mx orizzontale, My verticale
        else:
            return

        self.points_2d = points_2d

        # Calculate data range for 2D view
        if len(points_2d) > 0:
            min_vals = np.min(points_2d, axis=0)
            max_vals = np.max(points_2d, axis=0)
            # Use symmetric range around 0 to keep view centered
            self.data_range_x = max(abs(min_vals[0]), abs(max_vals[0])) or 1.0
            self.data_range_y = max(abs(min_vals[1]), abs(max_vals[1])) or 1.0

            # Reset view to fit all points
            self.pan_2d = [0.0, 0.0]
            self.zoom_2d = 1.0

            # Calculate convex hull for projected points (optional)
            try:
                self.hull_2d = ConvexHull(points_2d)
            except Exception:
                self.hull_2d = None

    def _compute_plane_intersection(self):
        """Compute intersection polygon (in world 2D coords of current view) between 3D convex hull and plane
        defined by verification_point. Stores:
            - self.plane_intersection_3d : (M,3) intersection points in 3D
            - self.plane_intersection_2d : (M,2) projected to current 2D view coords
            - self.plane_intersection_n  : (M,) corresponding N-values for each vertex
        """
        # reset
        self.plane_intersection_2d = None
        self.plane_intersection_3d = None
        self.plane_intersection_n = None

        if self.points is None or len(self.points) < 2:
            return

        # Choose plane coordinate (index) and projection axes according to view_mode
        if self.view_mode == "N_Mx":
            plane_idx = 1      # My = const
            plane_val = float(self.verification_point[1])
            proj_idx = (0, 2)  # project to (Mx, N)
            n_idx = 2
        elif self.view_mode == "N_My":
            plane_idx = 0      # Mx = const
            plane_val = float(self.verification_point[0])
            proj_idx = (1, 2)  # project to (My, N)
            n_idx = 2
        elif self.view_mode == "Mx_My":
            plane_idx = 2      # N = const
            plane_val = float(self.verification_point[2])
            proj_idx = (0, 1)  # project to (Mx, My)
            n_idx = 2
        else:
            return

        pts = self.points
        try:
            hull3 = ConvexHull(pts)  # may raise if degenerate
        except Exception:
            return

        eps = 1e-9
        inter_pts = []

        # For each triangular facet of the 3D hull intersect its edges with the plane
        for simplex in hull3.simplices:
            tri_idx = simplex  # three indices
            tri = pts[tri_idx]  # shape (3,3)
            # edges: (0,1), (1,2), (2,0)
            for a, b in ((0,1),(1,2),(2,0)):
                p1 = tri[a]
                p2 = tri[b]
                v1 = p1[plane_idx] - plane_val
                v2 = p2[plane_idx] - plane_val

                # both on plane -> add both endpoints (they will be deduped later)
                if abs(v1) < eps and abs(v2) < eps:
                    inter_pts.append(p1.copy()); inter_pts.append(p2.copy())
                    continue
                # one endpoint on plane
                if abs(v1) < eps:
                    inter_pts.append(p1.copy()); continue
                if abs(v2) < eps:
                    inter_pts.append(p2.copy()); continue
                # opposite signs -> interior intersection
                if v1 * v2 < 0:
                    t = (plane_val - p1[plane_idx]) / (p2[plane_idx] - p1[plane_idx])
                    p = p1 + t * (p2 - p1)
                    inter_pts.append(p)

        if len(inter_pts) == 0:
            return

        inter_pts = np.array(inter_pts)  # (M,3)

        # Project to view 2D coordinates
        pts2d = inter_pts[:, proj_idx]  # (M,2)
        n_vals = inter_pts[:, n_idx]    # corresponding N values

        # Deduplicate approximately identical points: round to tolerance then unique rows
        if pts2d.shape[0] > 0:
            pts2d_rounded = np.round(pts2d, decimals=9)
            uniq_idx = np.unique(pts2d_rounded, axis=0, return_index=True)[1]
            uniq_pts2d = pts2d[sorted(uniq_idx)]
            uniq_n = n_vals[sorted(uniq_idx)]
        else:
            return

        # If we have >=3 points compute 2D convex hull to get ordered polygon
        if len(uniq_pts2d) >= 3:
            try:
                hull2 = ConvexHull(uniq_pts2d)
                poly2d = uniq_pts2d[hull2.vertices]
                poly_n = uniq_n[hull2.vertices]
            except Exception:
                # fallback: sort points by angle around centroid
                centroid = np.mean(uniq_pts2d, axis=0)
                angles = np.arctan2(uniq_pts2d[:,1] - centroid[1], uniq_pts2d[:,0] - centroid[0])
                order = np.argsort(angles)
                poly2d = uniq_pts2d[order]
                poly_n = uniq_n[order]
        else:
            poly2d = uniq_pts2d
            poly_n = uniq_n

        # store
        self.plane_intersection_2d = np.array(poly2d)
        self.plane_intersection_n = np.array(poly_n)
        # reconstruct 3d points for completeness (only if needed)
        # We try to reconstruct 3D intersection points by solving for the missing coordinate:
        # use plane index to set plane_val.
        pts3d = []
        for p2, nval in zip(poly2d, poly_n):
            if self.view_mode == "N_Mx":
                # p2 = (Mx, N), plane: My = plane_val
                pts3d.append([p2[0], plane_val, p2[1]])
            elif self.view_mode == "N_My":
                # p2 = (My, N), plane: Mx = plane_val
                pts3d.append([plane_val, p2[0], p2[1]])
            elif self.view_mode == "Mx_My":
                # p2 = (Mx, My), plane: N = plane_val
                pts3d.append([p2[0], p2[1], plane_val])
        self.plane_intersection_3d = np.array(pts3d) if len(pts3d) > 0 else None

    def initializeGL(self):
        glClearColor(40/255, 40/255, 40/255, 1.0)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if self.view_mode == "3D":
            return
        # set projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(
            -self.data_range_x * self.zoom_2d + self.pan_2d[0],
             self.data_range_x * self.zoom_2d + self.pan_2d[0],
            -self.data_range_y * self.zoom_2d + self.pan_2d[1],
             self.data_range_y * self.zoom_2d + self.pan_2d[1],
            -1.0, 1.0
        )
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self._draw_grid()
        self._draw_domain()            # ora disegna fill trasparente + bordo bianco
        self._draw_verification_point()
        # Draw world axes and ticks on screen
        self._draw_screen_axes_and_labels()
        self._draw_tracker_with_coords()

    def _draw_verification_point(self):
        """Disegna un punto di verifica come quadrato nella vista corrente"""
        Mx, My, N = self.verification_point

        # Proiezione del punto in base alla vista selezionata
        if self.view_mode == "N_Mx":
            point_2d = [Mx, N]
        elif self.view_mode == "N_My":
            point_2d = [My, N]
        elif self.view_mode == "Mx_My":
            point_2d = [Mx, My]
        else:
            return  # Vista non valida

        # Conversione in coordinate schermo
        sx, sy = self._world_to_screen(point_2d[0], point_2d[1])

        # Disegno del quadrato con QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.verification_color)
        painter.setPen(self.verification_color)

        size = 10  # Dimensione del quadrato (lato)
        half = size // 2
        painter.drawRect(sx - half, sy - half, size, size)

        painter.end()

    def _draw_grid(self):
        # Calculate world coordinates
        world_min_x = -self.data_range_x * self.zoom_2d + self.pan_2d[0]
        world_max_x = self.data_range_x * self.zoom_2d + self.pan_2d[0]
        world_min_y = -self.data_range_y * self.zoom_2d + self.pan_2d[1]
        world_max_y = self.data_range_y * self.zoom_2d + self.pan_2d[1]

        # Calculate tick sizes
        tx = self._tick(world_max_x - world_min_x)
        ty = self._tick(world_max_y - world_min_y)

        # Draw grid lines
        glColor3f(0.2, 0.2, 0.2)
        glLineWidth(1)

        # Vertical lines
        x = np.floor(world_min_x / tx) * tx
        while x <= world_max_x + 1e-12:
            glBegin(GL_LINES)
            glVertex2f(x, world_min_y)
            glVertex2f(x, world_max_y)
            glEnd()
            x += tx

        # Horizontal lines
        y = np.floor(world_min_y / ty) * ty
        while y <= world_max_y + 1e-12:
            glBegin(GL_LINES)
            glVertex2f(world_min_x, y)
            glVertex2f(world_max_x, y)
            glEnd()
            y += ty

    def _draw_screen_axes_and_labels(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Arial", 10)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        w, h = self.width(), self.height()

        # Convert world->screen
        def to_screen(wx, wy):
            nx = (wx - (-self.data_range_x * self.zoom_2d + self.pan_2d[0])) / (2 * self.data_range_x * self.zoom_2d)
            ny = (wy - (-self.data_range_y * self.zoom_2d + self.pan_2d[1])) / (2 * self.data_range_y * self.zoom_2d)
            return int(nx * w), int((1 - ny) * h)

        # Determine axis colors based on view mode
        if self.view_mode == "N_Mx":
            x_color = self.axis_colors["Mx"]  # Red
            y_color = self.axis_colors["N"]   # Cyan
        elif self.view_mode == "N_My":
            x_color = self.axis_colors["My"]  # Green
            y_color = self.axis_colors["N"]   # Cyan
        elif self.view_mode == "Mx_My":
            x_color = self.axis_colors["Mx"]  # Red
            y_color = self.axis_colors["My"]  # Green

        # Draw horizontal axis (X)
        pen = painter.pen()
        pen.setColor(x_color)
        pen.setWidth(1)
        painter.setPen(pen)
        # world origin y=0 line across
        ox0, oy = to_screen(0, 0)
        painter.drawLine(0, oy, w, oy)

        # Draw vertical axis (Y)
        pen.setColor(y_color)
        painter.setPen(pen)
        ox, oy0 = to_screen(0, 0)
        painter.drawLine(ox, 0, ox, h)

        # Tick calculation
        world_min_x = -self.data_range_x * self.zoom_2d + self.pan_2d[0]
        world_max_x = self.data_range_x * self.zoom_2d + self.pan_2d[0]
        world_min_y = -self.data_range_y * self.zoom_2d + self.pan_2d[1]
        world_max_y = self.data_range_y * self.zoom_2d + self.pan_2d[1]
        tx = self._tick(world_max_x - world_min_x)
        ty = self._tick(world_max_y - world_min_y)

        # X ticks and labels (using X-axis color)
        painter.setPen(x_color)
        x = np.floor(world_min_x / tx) * tx
        while x <= world_max_x:
            if abs(x) > 1e-8:
                sx, sy = to_screen(x, 0)
                painter.drawLine(sx, sy-4, sx, sy+4)
                label = f"{x:.1f}"
                tw = metrics.horizontalAdvance(label)
                painter.drawText(sx - tw//2, sy + metrics.height() + 2, label)
            x += tx

        # Y ticks and labels (using Y-axis color)
        painter.setPen(y_color)
        y = np.floor(world_min_y / ty) * ty
        while y <= world_max_y:
            if abs(y) > 1e-8:
                sx, sy = to_screen(0, y)
                painter.drawLine(sx-4, sy, sx+4, sy)
                label = f"{y:.1f}"
                tw = metrics.horizontalAdvance(label)
                painter.drawText(sx - tw - 6, sy + metrics.height()//2 - 2, label)
            y += ty

        # Axis titles (using corresponding axis colors)
        x_label, y_label = self.axis_labels.get(self.view_mode, ("X", "Y"))

        # X-axis title
        painter.setPen(x_color)
        tw = metrics.horizontalAdvance(x_label)
        painter.drawText(w - tw - 10, h - 10, x_label)

        # Y-axis title
        painter.setPen(y_color)
        painter.save()
        painter.translate(20, 110)
        painter.rotate(-90)
        painter.drawText(0, 0, y_label)
        painter.restore()

        painter.end()

    def _draw_domain(self):
        """Disegna il riempimento (trasparente) e il bordo bianco del poligono risultante dall'intersezione."""
        if self.plane_intersection_2d is None:
            return

        pts = self.plane_intersection_2d
        n_vals = self.plane_intersection_n if self.plane_intersection_n is not None else np.full(len(pts), self.verification_point[2])

        if pts.shape[0] == 1:
            # disegnare un piccolo marker (punto) come crocetta
            x, y = pts[0]
            glColor3f(1.0, 1.0, 1.0)
            glPointSize(6.0)
            glBegin(GL_POINTS)
            glVertex2f(x, y)
            glEnd()
            return
        elif pts.shape[0] == 2:
            # disegnare segmento (linea)
            glColor3f(1.0, 1.0, 1.0)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex2f(pts[0,0], pts[0,1])
            glVertex2f(pts[1,0], pts[1,1])
            glEnd()
            return

        # DRAW FILL
        # For N_Mx and N_My: color varies per-vertex according to N value
        # For Mx_My: homogeneous color based on verification N
        alpha_fill = 0.40  # forte trasparenza

        if self.view_mode in ["N_Mx", "N_My"]:
            # Draw filled polygon with per-vertex colors (interpolated by OpenGL)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBegin(GL_POLYGON)
            for (x, y), n_val in zip(pts, n_vals):
                # Normalize n_val
                if self.max_N != self.min_N:
                    t = (n_val - self.min_N) / (self.max_N - self.min_N)
                else:
                    t = 0.5
                t = np.clip(t, 0.0, 1.0)
                # palette mapping (same formula usata prima)
                r = 1.0 - 0.5 * t
                g = 1.0 - t
                b = 0.5 + 0.5 * t
                glColor4f(r, g, b, alpha_fill)
                glVertex2f(x, y)
            glEnd()
            glDisable(GL_BLEND)

        elif self.view_mode == "Mx_My":
            # homogeneous color based on verification N
            Nval = float(self.verification_point[2])
            if self.max_N != self.min_N:
                t = (Nval - self.min_N) / (self.max_N - self.min_N)
            else:
                t = 0.5
            t = np.clip(t, 0.0, 1.0)
            r = 1.0 - 0.5 * t
            g = 1.0 - t
            b = 0.5 + 0.5 * t

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(r, g, b, alpha_fill)
            glBegin(GL_POLYGON)
            for x, y in pts:
                glVertex2f(x, y)
            glEnd()
            glDisable(GL_BLEND)

        # Draw border in white
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for x, y in pts:
            glVertex2f(x, y)
        glEnd()

    def _draw_tracker_with_coords(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        metrics = painter.fontMetrics()
        w, h = self.width(), self.height()
        x, y = self.cursor_pos.x(), self.cursor_pos.y()
        # Croce bianca
        pen = painter.pen(); pen.setColor(QColor(255,255,255)); painter.setPen(pen)
        size = 10
        painter.drawLine(x-size, y, x+size, y)
        painter.drawLine(x, y-size, x, y+size)
        # Linee guida scure
        pen.setColor(QColor(100,100,100,150)); painter.setPen(pen)
        painter.drawLine(x,0,x,h)
        painter.drawLine(0,y,w,y)
        # Coordinate in world
        wx, wy = self._screen_to_world(x, y)
        # Testo ai margini fissi
        pen.setColor(QColor(255,255,255,200)); painter.setPen(pen)
        # X: sempre in basso, centrato a cursor x
        text_x = f"{wx:.1f}"
        tw = metrics.horizontalAdvance(text_x)
        painter.drawText(x - tw//2+20, h - 5, text_x)
        # Y: sempre a sinistra, centrato a cursor y
        text_y = f"{wy:.1f}"
        th = metrics.height()
        painter.drawText(5, y + th//2-20, text_y)
        painter.end()

    def _world_to_screen(self, wx, wy):
        """Convert world coordinates to screen coordinates"""
        w, h = self.width(), self.height()
        nx = (wx - (-self.data_range_x * self.zoom_2d + self.pan_2d[0])) / (2 * self.data_range_x * self.zoom_2d)
        ny = (wy - (-self.data_range_y * self.zoom_2d + self.pan_2d[1])) / (2 * self.data_range_y * self.zoom_2d)
        return int(nx * w), int(h - ny * h)

    def _screen_to_world(self, sx, sy):
        """Convert screen coordinates to world coordinates"""
        w, h = self.width(), self.height()
        nx = sx / w
        ny = 1.0 - (sy / h)
        wx = (-self.data_range_x * self.zoom_2d + self.pan_2d[0]) + nx * (2 * self.data_range_x * self.zoom_2d)
        wy = (-self.data_range_y * self.zoom_2d + self.pan_2d[1]) + ny * (2 * self.data_range_y * self.zoom_2d)
        return wx, wy

    def _tick(self, rng):
        """Calculate appropriate tick size for grid"""
        rough = rng / 10
        if rough <= 0:
            return 1.0
        mag = 10 ** np.floor(np.log10(rough))
        res = rough / mag
        return 5 * mag if res >= 5 else 2 * mag if res >= 2 else mag

    # Mouse interaction for 2D view
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.last_mouse_pos = e.pos()
        self.cursor_pos = e.pos()
        self.update()

    def mouseMoveEvent(self, e):
        self.cursor_pos = e.pos()
        if self.last_mouse_pos:
            dx = e.x() - self.last_mouse_pos.x()
            dy = e.y() - self.last_mouse_pos.y()
            self.pan_2d[0] -= dx * (2 * self.data_range_x * self.zoom_2d) / self.width()
            self.pan_2d[1] += dy * (2 * self.data_range_y * self.zoom_2d) / self.height()
            self.last_mouse_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self.last_mouse_pos = None
        self.cursor_pos = e.pos()
        self.update()

    def wheelEvent(self, e):
        """Zoom centered on cursor"""
        # delta for PyQt5 QWheelEvent: angleDelta().y()
        delta = 0
        try:
            delta = e.angleDelta().y()
        except Exception:
            try:
                delta = e.delta()
            except Exception:
                delta = 0
        if delta == 0:
            return
        # zoom factor
        factor = 1.0 - np.sign(delta) * 0.1
        # limit zoom
        new_zoom = self.zoom_2d * factor
        new_zoom = max(0.05, min(10.0, new_zoom))

        # To zoom around cursor, adjust pan accordingly
        sx, sy = e.pos().x(), e.pos().y()
        wx_before, wy_before = self._screen_to_world(sx, sy)
        self.zoom_2d = new_zoom
        wx_after, wy_after = self._screen_to_world(sx, sy)
        # shift pan to keep same world point under cursor
        self.pan_2d[0] += (wx_before - wx_after)
        self.pan_2d[1] += (wy_before - wy_after)
        self.update()

        delta = e.angleDelta().y() / 120
        factor = 1.15
        if delta > 0:
            self.zoom_2d /= factor
        else:
            self.zoom_2d *= factor
        self.zoom_2d = max(0.1, min(self.zoom_2d, 10.0))
        self.update()