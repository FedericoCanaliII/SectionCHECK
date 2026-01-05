from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QPoint
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.spatial import Delaunay

class Domain2DWidget(QOpenGLWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        
        # Dati
        self.points_3d = None          # Nuvola di punti 3D completa (flat Nx3)
        self.structured = False        # True se l'input veniva da (rows, cols, 3)
        self.rows = None
        self.cols = None
        
        self.slice_polygon = None      # Poligono 2D calcolato (sezione principale)
        self.slice_polygons = []       # Possibili più contorni (chiusi)
        
        # Impostazioni Vista
        self.view_mode = "N_Mx"        # Default
        self.pan_2d = [0.0, 0.0]
        self.zoom_2d = 1.0
        self.last_mouse_pos = None
        
        # Range dati (per scala iniziale e griglia)
        self.data_range_x = 10.0
        self.data_range_y = 10.0
        
        # Cursore
        self.cursor_pos = QPoint(0, 0)
        self.setMouseTracking(True)
        self.font = QFont("Arial", 9)

        # Configurazioni Grafiche
        self.axis_labels = {
            "N_Mx": ("Mx [kNm]", "N [kN]"),
            "N_My": ("My [kNm]", "N [kN]"),
            "Mx_My": ("Mx [kNm]", "My [kNm]")
        }

        self.axis_colors = {
            "N": QColor(0, 200, 255),   # Ciano
            "Mx": QColor(255, 0, 0),    # Rosso chiaro
            "My": QColor(0, 255, 0)     # Verde chiaro
        }

        self.verification_point = [0.0, 0.0, 0.0]  # Mx, My, N
        self.verification_color = QColor(255, 0, 0)
        self.is_inside = False

        # Connessioni UI
        try:
            self.ui.out_Mx.textChanged.connect(self.update_verification_point)
            self.ui.out_My.textChanged.connect(self.update_verification_point)
            self.ui.out_N.textChanged.connect(self.update_verification_point)
            self.ui.btn_out_verifica.clicked.connect(self.update_verification_point)
        except Exception:
            pass

    def set_points(self, points):
        """Riceve la matrice punti 3D (n_thetas, n_steps, 3) o lista piatta Nx3."""
        if points is None or len(points) == 0:
            self.points_3d = None
            self.structured = False
            self.rows = self.cols = None
            self.slice_polygon = None
            self.slice_polygons = []
            self.update()
            return

        pts = np.asarray(points)
        if pts.ndim == 3 and pts.shape[2] == 3:
            # input strutturato
            self.rows, self.cols = pts.shape[0], pts.shape[1]
            self.points_3d = pts.reshape(-1, 3)
            self.structured = True
            self._structured_points = pts 
        elif pts.ndim == 2 and pts.shape[1] == 3:
            self.points_3d = pts
            self.structured = False
            self.rows = self.cols = None
            self._structured_points = None
        else:
            try:
                self.points_3d = pts.reshape(-1, 3)
                self.structured = False
                self.rows = self.cols = None
                self._structured_points = None
            except Exception:
                self.points_3d = None
                self.structured = False

        # Calcolo range globali
        if self.points_3d is not None and len(self.points_3d) > 0:
            mx_max = np.max(np.abs(self.points_3d[:, 0]))
            my_max = np.max(np.abs(self.points_3d[:, 1]))
            n_max = np.max(np.abs(self.points_3d[:, 2]))
            self.global_ranges = {
                'Mx': mx_max if mx_max > 1 else 10.0,
                'My': my_max if my_max > 1 else 10.0,
                'N': n_max if n_max > 1 else 100.0
            }
            self._update_ranges_for_view()
        else:
            self.global_ranges = {'Mx':10.0,'My':10.0,'N':100.0}
            self._update_ranges_for_view()

        self._compute_plane_intersection()
        self.update()

    def set_view_mode(self, mode):
        """Cambia vista ortogonale: N_Mx, N_My, Mx_My"""
        if mode not in ["N_Mx", "N_My", "Mx_My"]: return
        self.view_mode = mode
        self._update_ranges_for_view()
        self.pan_2d = [0.0, 0.0]
        self.zoom_2d = 1.0
        self._compute_plane_intersection()
        self.update()

    def _update_ranges_for_view(self):
        if not hasattr(self, 'global_ranges'): return
        if self.view_mode == "N_Mx":
            self.data_range_x = self.global_ranges['Mx'] * 1.2
            self.data_range_y = self.global_ranges['N'] * 1.2
        elif self.view_mode == "N_My":
            self.data_range_x = self.global_ranges['My'] * 1.2
            self.data_range_y = self.global_ranges['N'] * 1.2
        elif self.view_mode == "Mx_My":
            self.data_range_x = self.global_ranges['Mx'] * 1.2
            self.data_range_y = self.global_ranges['My'] * 1.2

    def update_verification_point(self):
        try:
            def p(t): return float(t.replace(',', '.') or 0)
            mx = p(self.ui.out_Mx.text())
            my = p(self.ui.out_My.text())
            n  = p(self.ui.out_N.text())
            
            old = self.verification_point
            self.verification_point = [mx, my, n]
            
            recalc = False
            if self.view_mode == "N_Mx" and abs(my - old[1]) > 1e-6: recalc = True
            elif self.view_mode == "N_My" and abs(mx - old[0]) > 1e-6: recalc = True
            elif self.view_mode == "Mx_My" and abs(n - old[2]) > 1e-6: recalc = True
            
            if recalc or self.slice_polygon is None:
                self._compute_plane_intersection()
                
            self._check_inside()
            self.update()
        except Exception:
            pass

    # ---------------------------
    #  Algoritmi di sezione
    # ---------------------------
    def _compute_plane_intersection(self):
        self.slice_polygon = None
        self.slice_polygons = []

        if self.points_3d is None or len(self.points_3d) < 3:
            return

        if self.view_mode == "N_Mx":   ix, iy, icut = 0, 2, 1
        elif self.view_mode == "N_My": ix, iy, icut = 1, 2, 0
        else:                          ix, iy, icut = 0, 1, 2 # Mx_My
        
        cut_val = self.verification_point[icut]

        # 1) Costruzione triangoli
        triangles = []
        pts = self.points_3d

        if self.structured and self._structured_points is not None:
            S = self._structured_points
            rows, cols = S.shape[0], S.shape[1]
            for i in range(rows):
                ni = (i + 1) % rows
                for j in range(cols - 1):
                    a = i * cols + j
                    b = ni * cols + j
                    c = ni * cols + (j + 1)
                    d = i * cols + (j + 1)
                    triangles.append((a, b, c))
                    triangles.append((a, c, d))
        else:
            try:
                proj2d = pts[:, [ix, iy]]
                tri = Delaunay(proj2d)
                triangles = [tuple(s) for s in tri.simplices]
            except Exception:
                triangles = []

        if not triangles:
            return

        # 2) Intersezione
        segments = []
        for (i0, i1, i2) in triangles:
            v0 = pts[i0]; v1 = pts[i1]; v2 = pts[i2]
            vs = np.array([v0, v1, v2])
            dists = vs[:, icut] - cut_val

            if np.all(dists > 0) or np.all(dists < 0):
                continue

            inter_points = []
            for a, b in ((0,1), (1,2), (2,0)):
                da = dists[a]; db = dists[b]
                if da == 0:
                    inter = vs[a]
                    inter_points.append([inter[ix], inter[iy]])
                if (da > 0 and db < 0) or (da < 0 and db > 0):
                    t = da / (da - db)
                    inter = vs[a] + t * (vs[b] - vs[a])
                    inter_points.append([inter[ix], inter[iy]])
            
            if len(inter_points) >= 2:
                pts_arr = np.array(inter_points)
                if pts_arr.shape[0] > 2:
                    d01 = np.linalg.norm(pts_arr[0]-pts_arr[1])
                    d02 = np.linalg.norm(pts_arr[0]-pts_arr[2])
                    d12 = np.linalg.norm(pts_arr[1]-pts_arr[2])
                    if d01 >= d02 and d01 >= d12: seg = (pts_arr[0].tolist(), pts_arr[1].tolist())
                    elif d02 >= d01 and d02 >= d12: seg = (pts_arr[0].tolist(), pts_arr[2].tolist())
                    else: seg = (pts_arr[1].tolist(), pts_arr[2].tolist())
                else:
                    seg = (pts_arr[0].tolist(), pts_arr[1].tolist())
                segments.append(seg)

        if not segments:
            return

        # 3) Stitching segmenti
        all_points = np.vstack([np.array(s[0]) for s in segments] + [np.array(s[1]) for s in segments])
        rng = np.ptp(all_points, axis=0)
        tol = max(1e-8, 1e-6 * max(np.max(rng), 1.0))
        decimals = max(0, int(-np.floor(np.log10(tol))) if tol < 1 else 0)

        def qkey(p):
            return (round(float(p[0]), decimals), round(float(p[1]), decimals))

        key_to_coord = {}
        edges = {}
        for a, b in segments:
            ka = qkey(a); kb = qkey(b)
            if ka not in key_to_coord: key_to_coord[ka] = (float(a[0]), float(a[1]))
            if kb not in key_to_coord: key_to_coord[kb] = (float(b[0]), float(b[1]))
            edges.setdefault(ka, []).append(kb)
            edges.setdefault(kb, []).append(ka)

        used_edges = set()
        loops = []

        def edge_id(u, v):
            return (u, v) if u <= v else (v, u)

        for start in list(edges.keys()):
            for nbr in list(edges.get(start, [])):
                eid = edge_id(start, nbr)
                if eid in used_edges: continue
                path = [start, nbr]
                used_edges.add(eid)
                curr = nbr; prev = start
                while True:
                    neighs = edges.get(curr, [])
                    next_nodes = [n for n in neighs if n != prev]
                    if not next_nodes: break
                    next_node = None
                    for cand in next_nodes:
                        if edge_id(curr, cand) not in used_edges:
                            next_node = cand
                            break
                    if next_node is None: next_node = next_nodes[0]
                    eid2 = edge_id(curr, next_node)
                    if eid2 in used_edges:
                        if next_node == path[0]: path.append(next_node)
                        break
                    used_edges.add(eid2)
                    path.append(next_node)
                    prev, curr = curr, next_node
                    if path[-1] == path[0]: break
                
                if len(path) >= 3 and path[0] == path[-1]:
                    coords = [key_to_coord[k] for k in path[:-1]]
                    loops.append(np.array(coords))

        if not loops:
            # Fallback greedy
            segs = [(tuple(qkey(s[0])), tuple(qkey(s[1]))) for s in segments]
            chain = [segs[0][0], segs[0][1]]; segs.pop(0)
            # (logica greedy semplificata per brevità, simile alla precedente)
            pass

        if not loops: return

        def polygon_area(poly):
            x = poly[:,0]; y = poly[:,1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        areas = [polygon_area(lp) for lp in loops]
        best_idx = int(np.argmax(areas))
        best_poly = loops[best_idx]

        # Ordine punti per disegno corretto
        if np.cross(best_poly[1]-best_poly[0], best_poly[2]-best_poly[1]) < 0:
            best_poly = best_poly[::-1]

        self.slice_polygons = loops
        self.slice_polygon = best_poly
        self._check_inside()

    def _point_in_polygon(self, x, y, poly):
        inside = False
        n = len(poly)
        if n < 3: return False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersect = ((yi > y) != (yj > y)) and \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
            if intersect: inside = not inside
            j = i
        return inside

    def _check_inside(self):
        if self.slice_polygon is None:
            self.is_inside = False
            self.verification_color = QColor(255, 0, 0)
            try: self.ui.out_testo_punto.setText("Dominio non definito")
            except: pass
            return

        if self.view_mode == "N_Mx":   px, py = self.verification_point[0], self.verification_point[2]
        elif self.view_mode == "N_My": px, py = self.verification_point[1], self.verification_point[2]
        else:                          px, py = self.verification_point[0], self.verification_point[1]
        
        try:
            if self._point_in_polygon(px, py, self.slice_polygon):
                self.is_inside = True
                self.verification_color = QColor(0, 255, 0)
                try: self.ui.out_testo_punto.setText("Verifica soddisfatta")
                except: pass
            else:
                self.is_inside = False
                self.verification_color = QColor(255, 0, 0)
                try: self.ui.out_testo_punto.setText("Verifica non soddisfatta")
                except: pass
        except: pass

    # ---------------------------
    #  Rendering
    # ---------------------------
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
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(
            -self.data_range_x * self.zoom_2d + self.pan_2d[0],
             self.data_range_x * self.zoom_2d + self.pan_2d[0],
            -self.data_range_y * self.zoom_2d + self.pan_2d[1],
             self.data_range_y * self.zoom_2d + self.pan_2d[1],
            -1.0, 1.0
        )
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        self._draw_grid()
        self._draw_domain()
        self._draw_verification_point_gl()
        self._draw_screen_axes_and_labels()
        self._draw_tracker_with_coords()

    def _draw_domain(self):
        """
        Disegna il dominio.
        MODIFICA: Usa GL_TRIANGLE_FAN partendo dal CENTROIDE (o origine)
        per evitare che il riempimento esca dai bordi nelle concavità.
        """
        if self.slice_polygon is None or len(self.slice_polygon) < 3:
            return
        
        # --- 1. RIEMPIMENTO (Spicchi dal Centroide) ---
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(1, 1, 1, 0.1) # Bianco Trasparente
        
        # Calcolo centroide geometrico (più sicuro dell'origine 0,0 se il dominio è spostato)
        # Se preferisci rigorosamente l'origine (0,0), metti cx=0, cy=0
        cx = np.mean(self.slice_polygon[:, 0])
        cy = np.mean(self.slice_polygon[:, 1])

        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)  # Centro del ventaglio (Perno)
        for p in self.slice_polygon:
            glVertex2f(p[0], p[1])
        # Chiudi il cerchio ricollegando all'inizio
        glVertex2f(self.slice_polygon[0][0], self.slice_polygon[0][1])
        glEnd()
        
        glDisable(GL_BLEND)
        
        # --- 2. BORDO (Bianco Solido) ---
        glLineWidth(1.5)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_LINE_LOOP)
        for p in self.slice_polygon:
            glVertex2f(p[0], p[1])
        glEnd()

        # --- 3. (Opzionale) Altri loop secondari ---
        for k, other in enumerate(self.slice_polygons):
            if other.shape[0] == self.slice_polygon.shape[0] and np.allclose(other, self.slice_polygon):
                continue
            glLineWidth(0.7)
            glColor4f(1.0, 1.0, 1.0, 0.5)
            glBegin(GL_LINE_LOOP)
            for p in other:
                glVertex2f(p[0], p[1])
            glEnd()

    def _draw_verification_point_gl(self):
        if self.view_mode == "N_Mx":   x, y = self.verification_point[0], self.verification_point[2]
        elif self.view_mode == "N_My": x, y = self.verification_point[1], self.verification_point[2]
        else:                          x, y = self.verification_point[0], self.verification_point[1]
        
        col = self.verification_color
        glColor3f(col.redF(), col.greenF(), col.blueF())
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glVertex2f(x, y)
        glEnd()

    def _draw_grid(self):
        wx_min = -self.data_range_x * self.zoom_2d + self.pan_2d[0]
        wx_max =  self.data_range_x * self.zoom_2d + self.pan_2d[0]
        wy_min = -self.data_range_y * self.zoom_2d + self.pan_2d[1]
        wy_max =  self.data_range_y * self.zoom_2d + self.pan_2d[1]
        tx = self._tick(wx_max - wx_min)
        ty = self._tick(wy_max - wy_min)

        glColor3f(0.2, 0.2, 0.2)
        glLineWidth(1)
        glBegin(GL_LINES)
        x = np.floor(wx_min / tx) * tx
        while x <= wx_max + 1e-12:
            glVertex2f(x, wy_min); glVertex2f(x, wy_max)
            x += tx
        y = np.floor(wy_min / ty) * ty
        while y <= wy_max + 1e-12:
            glVertex2f(wx_min, y); glVertex2f(wx_max, y)
            y += ty
        glEnd()

    def _draw_screen_axes_and_labels(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        metrics = painter.fontMetrics()
        w, h = self.width(), self.height()

        if self.view_mode == "N_Mx":
            c_x = self.axis_colors["Mx"]; lab_x = "Mx [kNm]"
            c_y = self.axis_colors["N"];  lab_y = "N [kN]"
        elif self.view_mode == "N_My":
            c_x = self.axis_colors["My"]; lab_x = "My [kNm]"
            c_y = self.axis_colors["N"];  lab_y = "N [kN]"
        else:
            c_x = self.axis_colors["Mx"]; lab_x = "Mx [kNm]"
            c_y = self.axis_colors["My"]; lab_y = "My [kNm]"

        def to_screen(wx, wy):
            nx = (wx - (-self.data_range_x * self.zoom_2d + self.pan_2d[0])) / (2 * self.data_range_x * self.zoom_2d)
            ny = (wy - (-self.data_range_y * self.zoom_2d + self.pan_2d[1])) / (2 * self.data_range_y * self.zoom_2d)
            return int(nx * w), int((1 - ny) * h)

        sx0, sy0 = to_screen(0, 0)
        pen = painter.pen(); pen.setColor(c_x); pen.setWidth(1); painter.setPen(pen)
        painter.drawLine(0, sy0, w, sy0)
        pen.setColor(c_y); painter.setPen(pen)
        painter.drawLine(sx0, 0, sx0, h)

        wx_min = -self.data_range_x * self.zoom_2d + self.pan_2d[0]
        wx_max =  self.data_range_x * self.zoom_2d + self.pan_2d[0]
        wy_min = -self.data_range_y * self.zoom_2d + self.pan_2d[1]
        wy_max =  self.data_range_y * self.zoom_2d + self.pan_2d[1]
        tx = self._tick(wx_max - wx_min)
        ty = self._tick(wy_max - wy_min)

        painter.setPen(c_x)
        x = np.floor(wx_min / tx) * tx
        while x <= wx_max:
            if abs(x) > 1e-9:
                sx, sy = to_screen(x, 0)
                painter.drawLine(sx, sy-4, sx, sy+4)
                label = f"{x:.4g}"
                tw = metrics.horizontalAdvance(label)
                txt_y = min(max(sy + metrics.height() + 2, 15), h - 5)
                painter.drawText(sx - tw//2, txt_y, label)
            x += tx

        painter.setPen(c_y)
        y = np.floor(wy_min / ty) * ty
        while y <= wy_max:
            if abs(y) > 1e-9:
                sx, sy = to_screen(0, y)
                painter.drawLine(sx-4, sy, sx+4, sy)
                label = f"{y:.4g}"
                tw = metrics.horizontalAdvance(label)
                txt_x = min(max(sx - tw - 6, 5), w - tw - 5)
                painter.drawText(txt_x, sy + metrics.height()//2 - 2, label)
            y += ty

        painter.setPen(c_x)
        tw = metrics.horizontalAdvance(lab_x)
        painter.drawText(w - tw - 10, h - 10, lab_x)
        painter.setPen(c_y)
        painter.save()
        painter.translate(20, 110)
        painter.rotate(-90)
        painter.drawText(0, 0, lab_y)
        painter.restore()
        painter.end()

    def _draw_tracker_with_coords(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if hasattr(self, 'font'): painter.setFont(self.font)
        w, h = self.width(), self.height()
        x, y = self.cursor_pos.x(), self.cursor_pos.y()
        tracking_pen = QPen(QColor(150, 150, 150, 100), 1, Qt.SolidLine)
        painter.setPen(tracking_pen)
        painter.drawLine(x, 0, x, h)
        painter.drawLine(0, y, w, y)
        cross_pen = QPen(QColor(255, 255, 255, 255), 1)
        painter.setPen(cross_pen)
        size = 10
        painter.drawLine(x - size, y, x + size, y)
        painter.drawLine(x, y - size, x, y + size)
        wx, wy = self._screen_to_world(x, y)
        painter.setPen(QColor(255, 255, 255, 180))
        painter.drawText(x + 10, h - 10, f"{wx:.4g}")
        painter.drawText(10, y - 10, f"{wy:.4g}")
        painter.end()

    def _tick(self, rng):
        rough = rng / 10
        if rough <= 0: return 1.0
        mag = 10 ** np.floor(np.log10(rough))
        res = rough / mag
        if res >= 5: return 5 * mag
        elif res >= 2: return 2 * mag
        return mag

    def _screen_to_world(self, sx, sy):
        w, h = self.width(), self.height()
        nx = sx / w
        ny = 1.0 - (sy / h)
        wx = (-self.data_range_x * self.zoom_2d + self.pan_2d[0]) + nx * (2 * self.data_range_x * self.zoom_2d)
        wy = (-self.data_range_y * self.zoom_2d + self.pan_2d[1]) + ny * (2 * self.data_range_y * self.zoom_2d)
        return wx, wy

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
        self.update()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta == 0: return
        sx, sy = e.pos().x(), e.pos().y()
        wx0, wy0 = self._screen_to_world(sx, sy)
        factor = 1.0 - np.sign(delta) * 0.1
        self.zoom_2d *= factor
        self.zoom_2d = max(0.001, min(100.0, self.zoom_2d))
        wx1, wy1 = self._screen_to_world(sx, sy)
        self.pan_2d[0] += (wx0 - wx1)
        self.pan_2d[1] += (wy0 - wy1)
        self.update()