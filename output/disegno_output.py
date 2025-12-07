from PyQt5.QtWidgets import QOpenGLWidget, QApplication
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QPainter, QFont
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

class OpenGLDomainWidget(QOpenGLWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui=ui
        self.points = None
        self.hull = None
        self.rotation = [30, -45, 0]  # Vista isometrica iniziale
        self.translation = [0, 0, -3.2]  # Vista più vicina
        self.scale = 1.0
        self.last_pos = QPoint()
        self.axis_length = 1.0
        self.normalized_points = None
        self.normalization_factors = [1.0, 1.0, 1.0]
        self.font = QFont("Arial", 10)
        self.min_N = 0.0
        self.max_N = 0.0
        self.axis_labels = 10
        try:
            self.ui.out_Mx.textChanged.connect(self.update)
            self.ui.out_My.textChanged.connect(self.update)
            self.ui.out_N.textChanged.connect(self.update)
        except Exception:
            pass
        
    def set_points(self, points):
        if points is None or points.size == 0:
            self.points = None
            self.normalized_points = None
            self.hull = None
            self.update()
            return
            
        self.points = points.copy()
        
        # Calcola i fattori di normalizzazione
        max_vals = np.max(np.abs(self.points), axis=0)
        self.normalization_factors = np.where(max_vals > 0, max_vals, 1.0)
        
        # Calcola min e max per l'asse N
        self.min_N = np.min(self.points[:, 2])
        self.max_N = np.max(self.points[:, 2])
        
        # Normalizza e riorganizza gli assi (Mx, My, N) -> (Mx, N, -My)
        normalized_original = self.points / self.normalization_factors
        self.normalized_points = np.column_stack((
            normalized_original[:, 0],  # Mx
            normalized_original[:, 2],  # N (asse verticale)
            -normalized_original[:, 1]  # -My (profondità)
        ))
        
        # Aggiorna i fattori di normalizzazione per i nuovi assi
        self.normalization_factors = [
            self.normalization_factors[0],  # Mx
            self.normalization_factors[2],  # N
            self.normalization_factors[1]   # My
        ]
        
        # Calcola l'inviluppo convesso
        try:
            self.hull = ConvexHull(self.normalized_points)
        except:
            self.hull = None
            
        self.update()

    def initializeGL(self):
        glClearColor(0.157, 0.157, 0.157, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height) if height > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Applica trasformazioni (invariato)
        glTranslatef(*self.translation)
        glRotatef(self.rotation[0], 1.0, 0.0, 0.0)
        glRotatef(self.rotation[1], 0.0, 1.0, 0.0)
        glRotatef(self.rotation[2], 0.0, 0.0, 1.0)
        glScalef(self.scale, self.scale, self.scale)
        
        # Disegna gli assi e la griglia (invariato)
        self.draw_axes()
        self.draw_grid()
        
        if self.normalized_points is None:
            self.draw_verification_point()
            return

        # ------------------------------------------------------------------
        # NUOVA: disegna tutti i punti (TONDINI) con gradiente basato su N
        # ------------------------------------------------------------------
        self.draw_all_points()
        # ------------------------------------------------------------------
        
        # Calcola il vettore di vista per l'ordinamento
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        view_vector = np.array([modelview[0][2], modelview[1][2], modelview[2][2]])
        
        # Ordina i triangoli per profondità (davanti -> dietro)
        if self.hull is not None:
            centroids = np.mean(self.normalized_points[self.hull.simplices], axis=1)
            depths = np.dot(centroids, view_vector)
            sorted_indices = np.argsort(depths)  # Ordine crescente: prima i più lontani
            sorted_simplices = self.hull.simplices[sorted_indices]
            
            # Calcola la profondità minima e massima per normalizzare
            min_depth = np.min(depths)
            max_depth = np.max(depths)
            depth_range = max_depth - min_depth if max_depth != min_depth else 1.0
        else:
            sorted_simplices = None
            min_depth = max_depth = depth_range = 0.0

        # 2. PASSAGGIO: Bordi opachi
        if sorted_simplices is not None:
            glLineWidth(1)
            
            for simplex in sorted_simplices:
                glBegin(GL_LINE_LOOP)
                # Calcola il colore medio del bordo in base ai due vertici
                # (oppure lo fai per-vertice se vuoi gradienti sui segmenti)
                for vertex in simplex:
                    # valore N del vertice
                    n_val = self.points[vertex][2]
                    # normalizza tra 0 e 1
                    t = (n_val - self.min_N) / (self.max_N - self.min_N) if self.max_N != self.min_N else 0.5
                    # colore “cipollotto” di base
                    r = 1.0 - 0.5 * t
                    g = 1.0 - t
                    b = 0.5 + 0.5 * t
                    # scurisci: moltiplichi per un fattore <1
                    dark_factor = 0.5
                    r_dark = r * dark_factor
                    g_dark = g * dark_factor
                    b_dark = b * dark_factor
                    # bordo totalmente opaco
                    glColor3f(r_dark, g_dark, b_dark)

                    # disegna il vertice
                    p = self.normalized_points[vertex]
                    glVertex3f(p[0], p[1], p[2])
                glEnd()
        
        # 1. PASSAGGIO: Superficie trasparente con effetto cipolla
        if sorted_simplices is not None:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)  # Disabilita scrittura depth buffer
            
            for simplex in sorted_simplices:
                centroid = np.mean(self.normalized_points[simplex], axis=0)
                depth = np.dot(centroid, view_vector)
                
                # Calcola trasparenza basata sulla profondità (più lontano = più trasparente)
                t_depth = (depth - min_depth) / depth_range
                alpha = 0.2 + 0.5 * (1.0 - t_depth)  # Range: 0.2-0.7
                
                glBegin(GL_TRIANGLES)
                for vertex in simplex:
                    # Colore basato sul valore N (invariato)
                    n_val = self.points[vertex][2]
                    t = (n_val - self.min_N) / (self.max_N - self.min_N) if self.max_N != self.min_N else 0.5
                    r = 1.0 - 0.5 * t
                    g = 1.0 - t
                    b = 0.5 + 0.5 * t
                    
                    # Applica trasparenza
                    glColor4f(r, g, b, alpha)
                    
                    p = self.normalized_points[vertex]
                    glVertex3f(p[0], p[1], p[2])
                glEnd()
            
            self.draw_verification_point()

            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)
    
    def draw_grid(self):
        """Disegna una griglia sul piano Mx-My (piano orizzontale), escludendo le linee sugli assi"""
        glColor3f(0.3, 0.3, 0.3)  # Grigio chiaro
        glLineWidth(0.8)

        step = 2 * self.axis_length / self.axis_labels

        # Linee parallele all'asse X (Z costante)
        for i in range(self.axis_labels + 1):
            z = -self.axis_length + i * step
            if abs(z) < 1e-5:  # Salta la linea a z = 0
                continue
            glBegin(GL_LINES)
            glVertex3f(-self.axis_length, 0, z)
            glVertex3f(self.axis_length, 0, z)
            glEnd()

        # Linee parallele all'asse Z (X costante)
        for i in range(self.axis_labels + 1):
            x = -self.axis_length + i * step
            if abs(x) < 1e-5:  # Salta la linea a x = 0
                continue
            glBegin(GL_LINES)
            glVertex3f(x, 0, -self.axis_length)
            glVertex3f(x, 0, self.axis_length)
            glEnd()

    def draw_axes(self):
        glBegin(GL_LINES)
        
        # Asse X (rosso) - Mx
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-self.axis_length-0.1, 0.0, 0.0)
        glVertex3f(self.axis_length+0.1, 0.0, 0.0)
        
        # Asse Y (verde) - N
        glColor3f(0.0, 0.8, 1.0)
        glVertex3f(0.0, -self.axis_length-0.1, 0.0)
        glVertex3f(0.0, self.axis_length+0.1, 0.0)
        
        # Asse Z (blu) - My
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, -self.axis_length-0.1)
        glVertex3f(0.0, 0.0, self.axis_length+0.1)
        glEnd()
        
        self.draw_axis_labels()

    def draw_axis_labels(self):
        glDisable(GL_DEPTH_TEST)
        painter = QPainter(self)
        painter.setFont(self.font)

        width, height = self.width(), self.height()

        def project_and_draw(x, y, z, text, pen, dx=0, dy=0):
            try:
                sc = gluProject(x, y, z,
                                glGetDoublev(GL_MODELVIEW_MATRIX),
                                glGetDoublev(GL_PROJECTION_MATRIX),
                                glGetIntegerv(GL_VIEWPORT))
            except ValueError:
                return
            if not sc:
                return
            sx, sy, _ = sc
            screen_x = int(sx)
            screen_y = int(height - sy)
            # Salta se fuori schermo o coordinate smisurate
            if screen_x < 0 or screen_x > width or screen_y < 0 or screen_y > height:
                return
            painter.setPen(pen)
            painter.drawText(screen_x + dx, screen_y + dy, text)

        # Etichette per l'asse X (Mx)
        for i in range(self.axis_labels + 1):
            t = i / float(self.axis_labels)
            x = -self.axis_length + 2 * self.axis_length * t
            val = x * self.normalization_factors[0]
            project_and_draw(x, 0.0, 0.0,
                             f"{val:.1f}",
                             QColor(255, 100, 100),
                             dx=-15, dy=20)

        # Etichette per l'asse Y (N)
        for i in range(self.axis_labels + 1):
            t = i / float(self.axis_labels)
            y = -self.axis_length + 2 * self.axis_length * t
            val = y * self.normalization_factors[1]
            project_and_draw(0.0, y, 0.0,
                             f"{val:.1f}",
                             QColor(100, 200, 255),
                             dx=-25, dy=5)

        # Etichette per l'asse Z (My)
        for i in range(self.axis_labels + 1):
            t = i / float(self.axis_labels)
            z = -self.axis_length + 2 * self.axis_length * t
            val = -z * self.normalization_factors[2]
            project_and_draw(0.0, 0.0, z,
                             f"{val:.1f}",
                             QColor(100, 255, 100),
                             dx=10, dy=-10)

        # Nomi degli assi
        axes = [
            (self.axis_length * 1.15, 0, 0, "Mx [kNm]", QColor(255, 0, 0), 0, 0),
            (0, self.axis_length * 1.15, 0, "N [kN]",  QColor(0,200,255), 0, 0),
            (0, 0, self.axis_length * 1.15, "My [kNm]", QColor(0,255,0),   0, 0)
        ]
        for x, y, z, label, col, dx, dy in axes:
            project_and_draw(x, y, z, label, col, dx, dy)

        painter.end()
        glEnable(GL_DEPTH_TEST)

    def draw_verification_point(self):
        # Read current UI values
        try:
            Mx = float(self.ui.out_Mx.text() or "0")
            My = float(self.ui.out_My.text() or "0")
            N  = float(self.ui.out_N.text()  or "0")
        except ValueError:
            return

        # Normalize coordinates
        v_norm = np.array([Mx, N, -My], dtype=float) / np.array(self.normalization_factors)

        # Determine point membership and feedback
        if self.hull is None:
            # No domain defined
            color = (1.0, 0.0, 0.0)
            self.ui.out_testo_punto.setText("Dominio non definito")
        else:
            # Check if point lies inside the convex hull
            tri = Delaunay(self.normalized_points)
            if tri.find_simplex(v_norm) >= 0:
                color = (0.0, 1.0, 0.0)
                self.ui.out_testo_punto.setText("Verifica soddisfatta")
            else:
                color = (1.0, 0.0, 0.0)
                self.ui.out_testo_punto.setText("Verifica non soddisfatta")

        # Render the point in OpenGL
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor3f(*color)
        glVertex3f(v_norm[0], v_norm[1], v_norm[2])
        glEnd()

    def draw_all_points(self):
        """Disegna TUTTI i punti (normalizzati) come tondini con gradiente basato su N.
        Include anche i punti non usati nella mesh.
        """
        if self.normalized_points is None or self.points is None:
            return

        # Assicuriamoci che min_N e max_N siano sensati
        n_min = self.min_N
        n_max = self.max_N
        n_range = n_max - n_min if n_max != n_min else 1.0

        # Abilita smoothing per punti (tondini) - nota: dipende dal profilo GL
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Imposta la dimensione dei punti (modificabile)
        glPointSize(6.0)

        glBegin(GL_POINTS)
        for i, p in enumerate(self.normalized_points):
            n_val = self.points[i][2]
            t = (n_val - n_min) / n_range
            # stesso gradiente usato altrove
            r = 1.0 - 0.5 * t
            g = 1.0 - t
            b = 0.5 + 0.5 * t
            glColor3f(r, g, b)
            glVertex3f(p[0], p[1], p[2])
        glEnd()

        # Ripristina stati
        glDisable(GL_POINT_SMOOTH)
        glDisable(GL_BLEND)

    # Resto del codice (gestione eventi mouse) rimane invariato
    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        if event.buttons() & Qt.LeftButton:
            self.rotation[0] += dy * 0.5
            self.rotation[1] += dx * 0.5
            self.update()
        elif event.buttons() & Qt.RightButton:
            self.translation[2] += dy * 0.05
            self.update()
        elif event.buttons() & Qt.MiddleButton:
            self.translation[0] += dx * 0.01
            self.translation[1] -= dy * 0.01
            self.update()

        self.last_pos = event.pos()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        self.scale *= 1.0 + delta * 0.1
        self.scale = max(0.1, min(self.scale, 10.0))
        self.update()
