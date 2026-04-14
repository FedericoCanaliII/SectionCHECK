from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QPoint
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

class Momentocurvatura2DWidget(QOpenGLWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        
        # --- Dati Funzionali (Logica Originale Mantenuta) ---
        # Matrice originale (Angoli x Step x 3) -> [Momento, Curvatura, Theta]
        self.points_matrix = None 
        # I punti della curva corrente da disegnare (N x 2) -> [Curvatura, Momento]
        self.current_curve_points = None
        self.current_angle_deg = 0.0

        # --- Impostazioni Vista (Stile Reference) ---
        self.pan_2d = [0.0, 0.0]
        self.zoom_2d = 1.0
        self.last_mouse_pos = None
        
        # Range dati (base per lo zoom 1.0)
        self.data_range_x = 0.1  # Curvatura default
        self.data_range_y = 100.0 # Momento default
        
        # Cursore
        self.cursor_pos = QPoint(0, 0)
        self.setMouseTracking(True)
        self.font = QFont("Arial", 9)

        # --- Colori e Stile (Adattati al Reference) ---
        self.bg_color = (40/255, 40/255, 40/255, 1.0) # Grigio scuro Reference

        # Mapping colori: 
        # Curvatura (X) -> Rosso (Simile a Mx del reference)
        # Momento (Y)   -> Ciano (Simile a N del reference)
        self.axis_colors = {
            "X": QColor(255, 80, 80),   # Rosso chiaro
            "Y": QColor(0, 200, 255)    # Ciano
        }

        # --- Connessione Slider (Logica Originale) ---
        try:
            slider = getattr(self.ui, 'horizontalSlider_momentocurvatura', None)
            if slider is not None:
                slider.valueChanged.connect(self.update_slice_from_slider)
                slider.setMinimum(0)
                slider.setMaximum(360)
        except Exception:
            pass

    def set_points(self, points_matrix):
        """
        Riceve la matrice numpy completa.
        Calcola i range e resetta la vista.
        """
        if points_matrix is None or points_matrix.size == 0:
            self.points_matrix = None
            self.current_curve_points = None
            self.update()
            return

        self.points_matrix = points_matrix
        
        # Calcolo range globali per il primo adattamento vista (Logica Originale)
        max_chi = np.max(np.abs(points_matrix[:, :, 1])) # Col 1 = Curvatura
        max_m = np.max(np.abs(points_matrix[:, :, 0]))   # Col 0 = Momento
        
        # Imposta i range base per la vista
        self.data_range_x = max_chi if max_chi > 1e-5 else 0.1
        self.data_range_y = max_m if max_m > 1e-5 else 100.0
        
        # Margine iniziale (zoom out leggero)
        self.data_range_x *= 1.2
        self.data_range_y *= 1.2
        
        # Reset Vista
        self.pan_2d = [self.data_range_x * 0.4, self.data_range_y * 0.4] # Centro leggermente spostato positivo
        self.zoom_2d = 1.0
        
        # Aggiorna la fetta corrente
        self.update_slice_from_slider()

    def update_slice_from_slider(self):
        """Logica Originale: estrae la curva 2D in base all'angolo slider."""
        if self.points_matrix is None:
            return

        slider = getattr(self.ui, 'horizontalSlider_momentocurvatura', None)
        val = slider.value() if slider else 0
        slider_max = slider.maximum() if slider else 360
        if slider_max == 0: slider_max = 360
        
        self.current_angle_deg = (val / float(slider_max)) * 360.0
        target_rad = math.radians(self.current_angle_deg)
        
        thetas = self.points_matrix[:, 0, 2] 
        diff = np.abs(thetas - target_rad)
        diff = np.minimum(diff, 2*np.pi - diff)
        idx = np.argmin(diff)
        
        data_row = self.points_matrix[idx]
        x_vals = data_row[:, 1] # Curvatura
        y_vals = data_row[:, 0] # Momento
        
        self.current_curve_points = np.column_stack((x_vals, y_vals))
        self.update()

    # --- RENDERING OPENGL (Stile Reference) ---

    def initializeGL(self):
        glClearColor(*self.bg_color)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        # 1. Setup Proiezione Ortogonale (Logica Reference: Centrata su Pan/Zoom)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        
        # Calcolo limiti mondo
        self.wx_min = -self.data_range_x * self.zoom_2d + self.pan_2d[0]
        self.wx_max =  self.data_range_x * self.zoom_2d + self.pan_2d[0]
        self.wy_min = -self.data_range_y * self.zoom_2d + self.pan_2d[1]
        self.wy_max =  self.data_range_y * self.zoom_2d + self.pan_2d[1]

        glOrtho(self.wx_min, self.wx_max, self.wy_min, self.wy_max, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        # 2. Griglia (Sfondo - Stile Reference)
        self._draw_grid()
        
        # 3. Curva (Logica Originale ma disegnata nel nuovo spazio)
        self._draw_curve_gl()
        
        # 4. Assi e Testi (Overlay QPainter - Stile Reference)
        self._draw_screen_axes_and_labels()
        self._draw_tracker_with_coords()

    def _draw_grid(self):
        """Disegna la griglia di sfondo (Stile Reference)"""
        tx = self._tick(self.wx_max - self.wx_min)
        ty = self._tick(self.wy_max - self.wy_min)

        glColor3f(0.2, 0.2, 0.2) # Grigio scuro Reference
        glLineWidth(1)

        glBegin(GL_LINES)
        # Verticali
        x = np.floor(self.wx_min / tx) * tx
        while x <= self.wx_max + 1e-12:
            glVertex2f(x, self.wy_min); glVertex2f(x, self.wy_max)
            x += tx
        # Orizzontali
        y = np.floor(self.wy_min / ty) * ty
        while y <= self.wy_max + 1e-12:
            glVertex2f(self.wx_min, y); glVertex2f(self.wx_max, y)
            y += ty
        glEnd()

    def _draw_curve_gl(self):
        """Disegna la curva con riempimento BIANCO TRASPARENTE"""
        if self.current_curve_points is None: return
        
        # --- A. Riempimento Sotto la Curva (BIANCO TRASPARENTE) ---
        glEnable(GL_BLEND)
        # Forza la modalità di trasparenza standard
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # BIANCO (1, 1, 1) con Alpha 0.20 (20% visibile, 80% trasparente)
        glColor4f(1.0, 1.0, 1.0, 0.1) 
        
        glBegin(GL_TRIANGLE_STRIP)
        for p in self.current_curve_points:
            glVertex2f(p[0], 0)    # Terra (Asse X)
            glVertex2f(p[0], p[1]) # Curva
        glEnd()
        
        # Disabilita il blend per far sì che la linea successiva sia perfettamente nitida
        glDisable(GL_BLEND) 
        
        # --- B. Linea della Curva (BIANCO SOLIDO) ---
        glLineWidth(1) 
        glColor4f(1.0, 1.0, 1.0, 1.0) # Bianco Pieno
        
        glBegin(GL_LINE_STRIP)
        for p in self.current_curve_points:
            glVertex2f(p[0], p[1])
        glEnd()
        
        # --- C. Punto finale (ROSSO) ---
        if len(self.current_curve_points) > 0:
            last = self.current_curve_points[-1]
            glPointSize(3.0)
            glColor3f(1.0, 0.2, 0.2) # Rosso per il punto finale
            glBegin(GL_POINTS)
            glVertex2f(last[0], last[1])
            glEnd()

    def _draw_screen_axes_and_labels(self):
        """Disegna assi, ticks e etichette usando QPainter (Stile Reference)"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)
        metrics = painter.fontMetrics()
        w, h = self.width(), self.height()

        # Definizioni Assi
        c_x = self.axis_colors["X"]; lab_x = "Curvatura [1/m]"
        c_y = self.axis_colors["Y"]; lab_y = "Momento [kNm]"

        # Funzione helper world->screen (ricalcolata qui per precisione)
        def to_screen(wx, wy):
            nx = (wx - self.wx_min) / (self.wx_max - self.wx_min)
            ny = (wy - self.wy_min) / (self.wy_max - self.wy_min)
            return int(nx * w), int((1 - ny) * h)

        # 1. Disegna Linee Assi Principali (passanti per 0,0)
        sx0, sy0 = to_screen(0, 0)
        
        # Asse X (y=0)
        pen = painter.pen(); pen.setColor(c_x); pen.setWidth(1); painter.setPen(pen)
        painter.drawLine(0, sy0, w, sy0)
        
        # Asse Y (x=0)
        pen.setColor(c_y); painter.setPen(pen)
        painter.drawLine(sx0, 0, sx0, h)

        # 2. Ticks e Valori
        tx = self._tick(self.wx_max - self.wx_min)
        ty = self._tick(self.wy_max - self.wy_min)

        # X Ticks
        painter.setPen(c_x)
        x = np.floor(self.wx_min / tx) * tx
        while x <= self.wx_max:
            if abs(x) > 1e-9: # Salta lo zero per non sovrapporsi
                sx, sy = to_screen(x, 0)
                # Disegna tick
                painter.drawLine(sx, sy-4, sx, sy+4)
                
                # Etichetta
                label = f"{x:.3g}"
                tw = metrics.horizontalAdvance(label)
                
                # Clamp verticale se l'asse X è fuori schermo
                txt_y = sy + metrics.height() + 2
                if txt_y > h - 5: txt_y = h - 5
                if txt_y < 15: txt_y = 15
                
                painter.drawText(sx - tw//2, txt_y, label)
            x += tx

        # Y Ticks
        painter.setPen(c_y)
        y = np.floor(self.wy_min / ty) * ty
        while y <= self.wy_max:
            if abs(y) > 1e-9:
                sx, sy = to_screen(0, y)
                # Disegna tick
                painter.drawLine(sx-4, sy, sx+4, sy)
                
                # Etichetta
                label = f"{y:.3g}"
                tw = metrics.horizontalAdvance(label)
                
                # Clamp orizzontale se l'asse Y è fuori schermo
                txt_x = sx - tw - 6
                if txt_x < 5: txt_x = 5
                if txt_x > w - tw - 5: txt_x = w - tw - 5
                
                painter.drawText(txt_x, sy + metrics.height()//2 - 2, label)
            y += ty

        # 3. Info Angolo (In alto a destra, mantenuto dall'originale)
        info_rect = f"Angolo: {self.current_angle_deg:.1f}°"
        painter.setPen(Qt.white)
        painter.drawText(w - metrics.horizontalAdvance(info_rect) - 10, 20, info_rect)

        # 4. Titoli Assi
        # X Label
        painter.setPen(c_x)
        tw = metrics.horizontalAdvance(lab_x)
        painter.drawText(w - tw - 10, h - 10, lab_x)

        # Y Label
        painter.setPen(c_y)
        painter.save()
        painter.translate(20, 150) # Posizione arbitraria a sinistra
        painter.rotate(-90)
        painter.drawText(0, 0, lab_y)
        painter.restore()

        painter.end()

    def _draw_tracker_with_coords(self):
        """Disegna cursore, linee di tracciamento e coordinate (Stile Reference)"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Usa il font impostato nel widget
        if hasattr(self, 'font_obj'):
            painter.setFont(self.font_obj)
        elif hasattr(self, 'font'):
            painter.setFont(self.font)

        w, h = self.width(), self.height()
        x, y = self.cursor_pos.x(), self.cursor_pos.y()

        # --- 1. DISEGNO LINEE MIRINO (FULL SCREEN) ---
        # Impostiamo un colore grigio discreto e semitrasparente
        tracking_pen = QPen(QColor(150, 150, 150, 100), 1, Qt.SolidLine)
        painter.setPen(tracking_pen)
        
        # Linea Verticale (da cima a fondo)
        painter.drawLine(x, 0, x, h)
        # Linea Orizzontale (da sinistra a destra)
        painter.drawLine(0, y, w, y)

        # --- 2. DISEGNO CROCE CENTRALE (PICCOLA) ---
        # Bianca e solida per indicare il punto esatto
        cross_pen = QPen(QColor(255, 255, 255, 255), 1)
        painter.setPen(cross_pen)
        size = 10
        painter.drawLine(x - size, y, x + size, y)
        painter.drawLine(x, y - size, x, y + size)

        # --- 3. CALCOLO E DISEGNO COORDINATE MONDO ---
        # Trasformiamo la posizione pixel in valori fisici
        wx, wy = self._screen_to_world(x, y)
        
        painter.setPen(QColor(255, 255, 255, 180)) # Testo bianco morbido
        lbl_x = f"{wx:.4g}"
        lbl_y = f"{wy:.4g}"

        # Posizionamento etichette (Stile professionale ai bordi)
        # X vicino al bordo inferiore, Y vicino al bordo sinistro
        painter.drawText(x + 10, h - 10, lbl_x)
        painter.drawText(10, y - 10, lbl_y)

        painter.end()

    # --- Utils Math (Stile Reference) ---
    def _tick(self, rng):
        """Calcola passo griglia adattivo"""
        rough = rng / 10.0
        if rough <= 0: return 1.0
        mag = 10 ** np.floor(np.log10(rough))
        res = rough / mag
        if res >= 5: return 5 * mag
        elif res >= 2: return 2 * mag
        return mag

    def _screen_to_world(self, sx, sy):
        """Converte coordinate schermo (pixel) in coordinate mondo"""
        w, h = self.width(), self.height()
        if w == 0 or h == 0: return 0,0
        nx = sx / w
        ny = 1.0 - (sy / h)
        
        range_x = self.wx_max - self.wx_min
        range_y = self.wy_max - self.wy_min
        
        wx = self.wx_min + nx * range_x
        wy = self.wy_min + ny * range_y
        return wx, wy

    # --- Interazione Mouse (Pan & Zoom Stile Reference) ---
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
            
            # Pan in world units
            range_x = self.wx_max - self.wx_min
            range_y = self.wy_max - self.wy_min
            
            self.pan_2d[0] -= dx * range_x / self.width()
            self.pan_2d[1] += dy * range_y / self.height()
            
            self.last_mouse_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self.last_mouse_pos = None
        self.update()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta == 0: return
        
        # Zoom verso il cursore (mouse-centric zoom)
        sx, sy = e.pos().x(), e.pos().y()
        wx0, wy0 = self._screen_to_world(sx, sy)
        
        # Fattore zoom
        factor = 1.0 - np.sign(delta) * 0.1
        self.zoom_2d *= factor
        self.zoom_2d = max(0.0001, min(10000.0, self.zoom_2d))
        
        # Compensazione Pan per mantenere il punto sotto il mouse fermo
        # Ricalcoliamo dove sarebbe il mondo dopo lo zoom senza pan
        # Ma è più facile usare la logica differenziale:
        # Nuovi limiti con nuovo zoom
        new_range_x = self.data_range_x * self.zoom_2d * 2 # Approssimazione per il calcolo delta
        # Nota: La logica di proiezione usa: +/- range * zoom + pan
        
        # Ricalcolo coordinate mondo post-zoom (virtuale)
        # Per semplicità, usiamo la logica inversa:
        # wx = (pan + coord_base*zoom)
        # Vogliamo che wx resti uguale.
        
        # Metodo più robusto implementato nel Reference:
        # 1. Calcola mondo prima dello zoom (wx0, wy0)
        # 2. Applica Zoom
        # 3. Calcola mondo dopo lo zoom nello stesso pixel (wx1, wy1)
        # 4. Sposta pan della differenza
        
        # Aggiorna variabili di stato temporanee per calcolare wx1
        self.wx_min = -self.data_range_x * self.zoom_2d + self.pan_2d[0]
        self.wx_max =  self.data_range_x * self.zoom_2d + self.pan_2d[0]
        self.wy_min = -self.data_range_y * self.zoom_2d + self.pan_2d[1]
        self.wy_max =  self.data_range_y * self.zoom_2d + self.pan_2d[1]
        
        wx1, wy1 = self._screen_to_world(sx, sy)
        
        self.pan_2d[0] += (wx0 - wx1)
        self.pan_2d[1] += (wy0 - wy1)
        
        self.update()