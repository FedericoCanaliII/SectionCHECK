from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QPainter, QFont
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

class OpenGLMomentocurvaturaWidget(QOpenGLWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        
        # Dati strutturati: matrice (Angoli x Step x 3)
        # Input atteso: [Momento, Curvatura, Theta_rad]
        self.points_matrix = None 
        self.normalized_matrix = None
        
        # Variabili di vista (Camera)
        self.rotation = [30, -45, 0]
        self.translation = [0, 0, -3.5]
        self.scale = 1.0
        self.last_pos = QPoint()
        
        # Configurazione assi e griglia (stile richiesto invariato)
        self.axis_length = 1.0
        # [Max_Curvatura, Max_Momento, Max_Curvatura] -> X, Y, Z
        self.normalization_factors = [1.0, 1.0, 1.0] 
        self.font = QFont("Arial", 10)
        self.axis_labels = 10 
        
        # Tentativo di connessione segnali UI (se esistono i campi nella UI)
        try:
            # Se hai campi per visualizzare un punto specifico (es. Curvatura di calcolo)
            if hasattr(self.ui, 'out_max_curvatura'):
                self.ui.out_max_curvatura.textChanged.connect(self.update)
        except Exception:
            pass
        
    def set_points(self, points_matrix):
        """
        Riceve matrice numpy shape (n_angles, n_steps, 3).
        INPUT DATA PER RIGA: [Momento, Curvatura, Theta_rad]
        """
        if points_matrix is None or points_matrix.size == 0:
            self.points_matrix = None
            self.normalized_matrix = None
            self.update()
            return
            
        self.points_matrix = points_matrix
        
        # 1. Trova i massimi per la normalizzazione
        # Colonna 0: Momento
        # Colonna 1: Curvatura
        all_moments = points_matrix[:, :, 0]
        all_curvatures = points_matrix[:, :, 1]
        
        max_M = np.max(np.abs(all_moments))
        max_Chi = np.max(np.abs(all_curvatures))
        
        if max_M == 0: max_M = 1.0
        if max_Chi == 0: max_Chi = 1.0
        
        # Normalizzazione: X=Curvatura, Y=Momento, Z=Curvatura
        self.normalization_factors = [max_Chi, max_M, max_Chi]
        
        rows, cols, _ = points_matrix.shape
        self.normalized_matrix = np.zeros((rows, cols, 3))
        
        # 2. Trasformazione da Cilindriche (M, Chi, Theta) a Cartesiane OpenGL (x, y, z)
        # OpenGL Y = Up (Momento)
        # OpenGL X, Z = Piano Curvatura
        
        for r in range(rows):
            for c in range(cols):
                M = points_matrix[r, c, 0]
                Chi = points_matrix[r, c, 1]
                Theta = points_matrix[r, c, 2]
                
                # Conversione coordinate
                # x = Chi * cos(theta)
                # z = -Chi * sin(theta) (meno per profondità OpenGL)
                # y = Momento
                
                norm_x = (Chi * math.cos(Theta)) / self.normalization_factors[0]
                norm_y = M / self.normalization_factors[1]
                norm_z = -(Chi * math.sin(Theta)) / self.normalization_factors[2]
                
                self.normalized_matrix[r, c] = [norm_x, norm_y, norm_z]
                
        self.update()

    def initializeGL(self):
        # Sfondo scuro (Stile richiesto)
        glClearColor(0.157, 0.157, 0.157, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_POINT_SMOOTH)

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
        
        # Trasformazioni camera
        glTranslatef(*self.translation)
        glRotatef(self.rotation[0], 1.0, 0.0, 0.0)
        glRotatef(self.rotation[1], 0.0, 1.0, 0.0)
        glRotatef(self.rotation[2], 0.0, 0.0, 1.0)
        glScalef(self.scale, self.scale, self.scale)
        
        # Disegno ambiente
        self.draw_grid()
        self.draw_axes()
        
        # Disegno il diagramma (Superficie)
        if self.normalized_matrix is not None:
            self.draw_surface_mesh()
            # Opzionale: disegna cursore se ci sono input attivi
            # self.draw_cursor_point()

    def draw_grid(self):
        """Disegna griglia sul piano base (Curvatura = 0, Momento = 0 non ha senso, 
        quindi facciamo la griglia sul piano X-Z che rappresenta il piano delle curvature)"""
        glColor3f(0.3, 0.3, 0.3) 
        glLineWidth(0.8)

        step = 2 * self.axis_length / self.axis_labels

        glBegin(GL_LINES)
        # Linee lungo X
        for i in range(self.axis_labels + 1):
            z = -self.axis_length + i * step
            if abs(z) < 1e-5: continue 
            glVertex3f(-self.axis_length, 0, z)
            glVertex3f(self.axis_length, 0, z)

        # Linee lungo Z
        for i in range(self.axis_labels + 1):
            x = -self.axis_length + i * step
            if abs(x) < 1e-5: continue 
            glVertex3f(x, 0, -self.axis_length)
            glVertex3f(x, 0, self.axis_length)
        glEnd()

    def draw_axes(self):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        
        # Asse X (rosso) - Mx
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-self.axis_length-0.1, 0.0, 0.0)
        glVertex3f(self.axis_length+0.1, 0.0, 0.0)
        
        # Asse Y (Ciano/Blu) - N (SOLO POSITIVO)
        glColor3f(0.0, 0.8, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, self.axis_length+0.1, 0.0)
        
        # Asse Z (Verde) - My (o -My)
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
            except ValueError: return
            if not sc: return
            sx, sy, _ = sc
            screen_x = int(sx)
            screen_y = int(height - sy)
            
            # Clipping semplice
            if screen_x < -50 or screen_x > width+50 or screen_y < -50 or screen_y > height+50: return
            
            painter.setPen(pen)
            painter.drawText(screen_x + dx, screen_y + dy, text)

        # Etichette graduate
        # Curvatura (X e Z) e Momento (Y)
        steps = 5 # Meno etichette per pulizia
        for i in range(1, steps + 1):
            t = i / float(steps)
            dist = self.axis_length * t
            
            # Valore Reale Curvatura
            val_chi = dist * self.normalization_factors[0]
            # Valore Reale Momento
            val_M = dist * self.normalization_factors[1]
            
            # X Label (Curvatura X)
            project_and_draw(dist, 0.0, 0.0, f"{val_chi:.3f}", QColor(255, 100, 100), dx=-10, dy=20)
            
            # Y Label (Momento)
            project_and_draw(0.0, dist, 0.0, f"{val_M:.1f}", QColor(100, 200, 255), dx=-30, dy=5)
            
            # Z Label (Curvatura Y)
            project_and_draw(0.0, 0.0, dist, f"{val_chi:.3f}", QColor(100, 255, 100), dx=10, dy=-5)

        # Nomi assi
        axes = [
            (self.axis_length * 1.15, 0, 0, "Chi X [1/m]", QColor(255, 0, 0), 0, 0),
            (0, self.axis_length * 1.15, 0, "Momento [kNm]",  QColor(0,200,255), 0, 0),
            (0, 0, self.axis_length * 1.15, "Chi Y [1/m]", QColor(0,255,0),   0, 0)
        ]
        for x, y, z, label, col, dx, dy in axes:
            project_and_draw(x, y, z, label, col, dx, dy)

        painter.end()
        glEnable(GL_DEPTH_TEST)

    def draw_surface_mesh(self):
        """
        Disegna la superficie M-Chi.
        """
        rows, cols, _ = self.normalized_matrix.shape
        
        # --- 1. SUPERFICIE (Mesh Grigio Tecnico Trasparente) ---
        glEnable(GL_BLEND)
        
        # Disattiviamo il culling perché vogliamo vedere la superficie anche da "sotto" o "dentro"
        glDisable(GL_CULL_FACE) 
        glDepthMask(GL_FALSE) 
        
        glBegin(GL_QUADS)
        # Colore base grigio, semi-trasparente
        glColor4f(0.5, 0.5, 0.5, 0.25) # Leggermente più opaco del dominio per vedere la forma
        
        for i in range(rows):
            curr_i = i
            # Gestione chiusura angolare (l'ultimo angolo si collega al primo)
            next_i = (i + 1) % rows
            
            for j in range(cols - 1):
                # j è lo step di curvatura. Non va chiuso loop (parte da 0 va a max)
                p1 = self.normalized_matrix[curr_i, j]
                p2 = self.normalized_matrix[next_i, j]
                p3 = self.normalized_matrix[next_i, j+1]
                p4 = self.normalized_matrix[curr_i, j+1]
                
                # Gradiente di colore basato sull'altezza (Momento)?
                # Per ora mantengo stile flat grigio come richiesto
                glVertex3fv(p1)
                glVertex3fv(p2)
                glVertex3fv(p3)
                glVertex3fv(p4)
        glEnd()
        
        glDepthMask(GL_TRUE)
        
        # --- 2. WIREFRAME (Linee Bianche) ---
        glLineWidth(0.8)
        
        # A. Linee radiali (dall'origine verso l'esterno per ogni angolo)
        for i in range(rows):
            glBegin(GL_LINE_STRIP)
            glColor4f(1.0, 1.0, 1.0, 0.4)
            for j in range(cols):
                glVertex3fv(self.normalized_matrix[i, j])
            glEnd()
            
        # B. Curve di livello (Anelli a curvatura costante)
        # Disegnamo un anello ogni tot step per non affollare la vista
        step_draw = max(1, cols // 10) 
        for j in range(0, cols, step_draw):
            glBegin(GL_LINE_LOOP)
            glColor4f(1.0, 1.0, 1.0, 0.2) # Più tenue
            for i in range(rows):
                glVertex3fv(self.normalized_matrix[i, j])
            glEnd()
        
        # Disegna sempre l'ultimo anello (bordo esterno/rottura) più marcato
        glLineWidth(1.5)
        glBegin(GL_LINE_LOOP)
        glColor4f(1.0, 0.2, 0.2, 0.8) # Bordo rosso leggero per indicare fine calcolo
        for i in range(rows):
            glVertex3fv(self.normalized_matrix[i, cols-1])
        glEnd()

    # --- Mouse Interaction (Invariata) ---
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
        self.scale = max(0.01, min(self.scale, 100.0))
        self.update()