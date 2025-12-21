from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QPainter, QFont
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.spatial import Delaunay

class OpenGLDomainWidget(QOpenGLWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        
        # Dati strutturati: matrice (Angoli x Step x 3)
        self.points_matrix = None 
        self.normalized_matrix = None
        
        # Oggetto per il calcolo geometrico (Verifica dentro/fuori)
        self.delaunay = None
        
        # Variabili di vista
        self.rotation = [30, -45, 0]
        self.translation = [0, 0, -3.5]
        self.scale = 1.0
        self.last_pos = QPoint()
        
        # Configurazione assi e griglia (stile richiesto)
        self.axis_length = 1.0
        self.normalization_factors = [1.0, 1.0, 1.0] # [Max_Mx, Max_My, Max_N]
        self.font = QFont("Arial", 10)
        self.min_N = 0.0
        self.max_N = 0.0
        self.axis_labels = 10 # 5 per lato -> 10 totali
        
        # Connessioni UI se possibile
        try:
            self.ui.out_Mx.textChanged.connect(self.update)
            self.ui.out_My.textChanged.connect(self.update)
            self.ui.out_N.textChanged.connect(self.update)
        except Exception:
            pass
        
    def set_points(self, points_matrix):
        """
        Riceve una matrice numpy di shape (n_thetas, n_steps, 3).
        INPUT ATTESO: [Mx, My, N]
        """
        if points_matrix is None or points_matrix.size == 0:
            self.points_matrix = None
            self.normalized_matrix = None
            self.delaunay = None
            self.update()
            return
            
        self.points_matrix = points_matrix
        
        # Flatten per calcolare min/max globali
        flat_points = points_matrix.reshape(-1, 3)
        max_vals = np.max(np.abs(flat_points), axis=0)
        
        # Fattori di normalizzazione [Max_Mx, Max_My, Max_N]
        self.normalization_factors = np.where(max_vals > 1e-9, max_vals, 1.0)
        
        # Creiamo la matrice normalizzata
        norm_flat = flat_points / self.normalization_factors
        
        # Riorganizziamo in (Mx, N, -My) per OpenGL
        # INPUT:  0=Mx, 1=My, 2=N
        # OPENGL: X=Mx, Y=N,  Z=-My
        transformed_flat = np.column_stack((
            norm_flat[:, 0],      # X_gl = Mx
            norm_flat[:, 2],      # Y_gl = N (Mettiamo N sull'asse verticale Y)
            -norm_flat[:, 1]      # Z_gl = -My (Mettiamo My sull'asse profondità Z)
        ))
        
        # Ricostruiamo la forma matriciale
        rows, cols, _ = points_matrix.shape
        self.normalized_matrix = transformed_flat.reshape(rows, cols, 3)
        
        # --- CREAZIONE DELAUNAY PER VERIFICA ---
        try:
            self.delaunay = Delaunay(transformed_flat)
        except Exception as e:
            print(f"Errore generazione Delaunay: {e}")
            self.delaunay = None
        
        self.update()

    def initializeGL(self):
        # Sfondo scuro come nel codice esempio
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
        
        # Disegno ambiente (Griglia e Assi Stile Example)
        self.draw_grid()
        self.draw_axes()
        
        # Disegno il dominio (Stile Professionale Mesh/Wireframe)
        if self.normalized_matrix is not None:
            self.draw_structured_mesh()
            self.draw_verification_point()

    def draw_grid(self):
        """Disegna una griglia sul piano Mx-My (piano orizzontale)"""
        glColor3f(0.3, 0.3, 0.3)  # Grigio scuro
        glLineWidth(0.8)

        step = 2 * self.axis_length / self.axis_labels

        glBegin(GL_LINES)
        # Linee parallele all'asse X (Z costante)
        for i in range(self.axis_labels + 1):
            z = -self.axis_length + i * step
            if abs(z) < 1e-5: continue # Salta l'asse centrale
            glVertex3f(-self.axis_length, 0, z)
            glVertex3f(self.axis_length, 0, z)

        # Linee parallele all'asse Z (X costante)
        for i in range(self.axis_labels + 1):
            x = -self.axis_length + i * step
            if abs(x) < 1e-5: continue # Salta l'asse centrale
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
        
        # Asse Y (Ciano/Blu) - N
        glColor3f(0.0, 0.8, 1.0)
        glVertex3f(0.0, -self.axis_length-0.1, 0.0)
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
            if screen_x < 0 or screen_x > width or screen_y < 0 or screen_y > height: return
            painter.setPen(pen)
            painter.drawText(screen_x + dx, screen_y + dy, text)

        # Etichette assi graduate
        for i in range(self.axis_labels + 1):
            t = i / float(self.axis_labels)
            
            # X (Mx) --> Index 0
            x = -self.axis_length + 2 * self.axis_length * t
            val = x * self.normalization_factors[0] 
            project_and_draw(x, 0.0, 0.0, f"{val:.1f}", QColor(255, 100, 100), dx=-15, dy=20)
            
            # Y (N) --> Index 2 (Verticale nel plot)
            y = -self.axis_length + 2 * self.axis_length * t
            val = y * self.normalization_factors[2]
            project_and_draw(0.0, y, 0.0, f"{val:.1f}", QColor(100, 200, 255), dx=-25, dy=5)
            
            # Z (-My) --> Index 1 (Profondità nel plot)
            z = -self.axis_length + 2 * self.axis_length * t
            val = -z * self.normalization_factors[1] 
            project_and_draw(0.0, 0.0, z, f"{val:.1f}", QColor(100, 255, 100), dx=10, dy=-10)

        # Nomi assi alle estremità
        axes = [
            (self.axis_length * 1.15, 0, 0, "Mx [kNm]", QColor(255, 0, 0), 0, 0),
            (0, self.axis_length * 1.15, 0, "N [kN]",  QColor(0,200,255), 0, 0),
            (0, 0, self.axis_length * 1.15, "My [kNm]", QColor(0,255,0),   0, 0)
        ]
        for x, y, z, label, col, dx, dy in axes:
            project_and_draw(x, y, z, label, col, dx, dy)

        painter.end()
        glEnable(GL_DEPTH_TEST)

    def draw_structured_mesh(self):
        """
        Disegna il dominio con lo stile professionale (Mesh Grigia + Linee Bianche).
        """
        rows, cols, _ = self.normalized_matrix.shape
        
        # --- 1. SUPERFICIE (Mesh Grigio Tecnico Trasparente) ---
        glEnable(GL_BLEND)
        glEnable(GL_CULL_FACE) 
        glCullFace(GL_BACK) 
        glDepthMask(GL_FALSE) 
        
        glBegin(GL_QUADS)
        # Grigio medio (0.5) con alpha basso (0.15)
        glColor4f(0.5, 0.5, 0.5, 0.15)
        
        for i in range(rows):
            curr_i = i
            next_i = (i + 1) % rows
            
            for j in range(cols - 1):
                p1 = self.normalized_matrix[curr_i, j]
                p2 = self.normalized_matrix[next_i, j]
                p3 = self.normalized_matrix[next_i, j+1]
                p4 = self.normalized_matrix[curr_i, j+1]
                
                glVertex3fv(p1)
                glVertex3fv(p2)
                glVertex3fv(p3)
                glVertex3fv(p4)
        glEnd()
        
        glDepthMask(GL_TRUE)
        glDisable(GL_CULL_FACE)
        
        # --- 2. WIREFRAME (Linee Bianche) ---
        glLineWidth(0.8)
        
        # A. Meridiani (Linee verticali)
        for i in range(rows):
            glBegin(GL_LINE_STRIP)
            for j in range(cols):
                p = self.normalized_matrix[i, j]
                glColor4f(1.0, 1.0, 1.0, 0.3)
                glVertex3fv(p)
            glEnd()
            
        # B. Paralleli (Anelli orizzontali)
        for j in range(cols):
            # [MODIFICA] Rimosso il filtro "if cols > 20..." -> Ora disegna TUTTI gli anelli
            glBegin(GL_LINE_LOOP)
            for i in range(rows):
                p = self.normalized_matrix[i, j]
                glColor4f(1.0, 1.0, 1.0, 0.3)
                glVertex3fv(p)
            glEnd()

        # --- 3. PUNTI (Dots sui vertici) ---
        # Disegna i punti solo se non sono eccessivi (>10000 inizia a pesare)
        if rows * cols < 10000:
            glPointSize(2.0)
            glBegin(GL_POINTS)
            glColor4f(1.0, 1.0, 1.0, 0.6)
            for i in range(rows):
                for j in range(cols):
                    glVertex3fv(self.normalized_matrix[i, j])
            glEnd()

    def draw_verification_point(self):
        try:
            Mx = float(self.ui.out_Mx.text() or "0")
            My = float(self.ui.out_My.text() or "0")
            N  = float(self.ui.out_N.text()  or "0")
        except ValueError:
            return

        # --- 1. NORMALIZZAZIONE ---
        # Mx -> Index 0 -> Asse X
        x = Mx / self.normalization_factors[0] if self.normalization_factors[0] else 0
        
        # N -> Index 2 -> Asse Y (Verticale)
        y = N / self.normalization_factors[2] if self.normalization_factors[2] else 0
        
        # My -> Index 1 -> Asse Z (Profondità, invertito)
        z = -My / self.normalization_factors[1] if self.normalization_factors[1] else 0
        
        point_check = np.array([x, y, z])

        # --- 2. VERIFICA MATEMATICA ---
        is_inside = False
        if self.delaunay is not None:
            simplex = self.delaunay.find_simplex(point_check)
            is_inside = (simplex >= 0)

        # --- 3. UI FEEDBACK ---
        
        # Definisco lo stile base originale (senza il colore bianco fisso, 
        # oppure lascio che le regole successive lo sovrascrivano per "cascata")
        base_style = (
            "background-color: rgb(60,60,60); "
            "font: 10pt 'Segoe UI'; "
            "padding-left: 5px; "
            "border-radius: 4px; "
        )

        if self.delaunay is None:
            self.ui.out_testo_punto.setText("Dominio non definito")
            # Grigio, bordo grigio
            self.ui.out_testo_punto.setStyleSheet(
                base_style + "border: 1px solid gray; color: gray;"
            )
            col_rgb = (0.5, 0.5, 0.5)

        elif is_inside:
            self.ui.out_testo_punto.setText("Verifica soddisfatta")
            # Verde acceso, Bordo Verde 1px, Testo in grassetto
            self.ui.out_testo_punto.setStyleSheet(
                base_style + "border: 1px solid #00FF00; color: #00FF00; font-weight: bold;"
            )
            col_rgb = (0.0, 1.0, 0.0) # Verde OpenGL

        else:
            self.ui.out_testo_punto.setText("Verifica non soddisfatta")
            # Rosso acceso, Bordo Rosso 1px, Testo in grassetto
            self.ui.out_testo_punto.setStyleSheet(
                base_style + "border: 1px solid #FF0000; color: #FF0000; font-weight: bold;"
            )
            col_rgb = (1.0, 0.0, 0.0) # Rosso OpenGL
       
        # --- 4. DISEGNO PUNTO ---
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor3f(*col_rgb)
        glVertex3f(x, y, z)
        glEnd()
        
        # Drop line
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor3f(col_rgb[0]*0.5, col_rgb[1]*0.5, col_rgb[2]*0.5)
        glVertex3f(x, 0, z)
        glVertex3f(x, y, z)
        glEnd()

    # --- Mouse (Invariato) ---
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