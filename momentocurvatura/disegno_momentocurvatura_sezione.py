from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PyQt5.QtCore import Qt, QPoint, QRectF
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# ... (MANTIENI LA TUA STYLE_CONFIG INVARIATA QUI) ...
# ==============================================================================
# CONFIGURAZIONE STILE VISUALIZZAZIONE
# ==============================================================================
STYLE_CONFIG = {
    # --- SFONDO E GRIGLIA ---
    "BACKGROUND_COLOR": (40/255, 40/255, 40/255, 1.0),
    "GRID_COLOR":       (0.23, 0.23, 0.23),
    "AXIS_X_COLOR":     QColor(255, 0, 0),
    "AXIS_Y_COLOR":     QColor(0, 255, 0),
    
    # --- CALCESTRUZZO (SHAPE) ---
    "CONCRETE_FILL":    (0.8, 0.8, 0.8, 0.20),
    "CONCRETE_OUTLINE": (1.0, 1.0, 1.0, 0.9),
    "CONCRETE_LINE_WIDTH": 1,

    # --- ARMATURE (BARS) ---
    "STEEL_FILL":       (1, 0, 0, 0.47),
    "STEEL_OUTLINE":    (1, 0, 0, 1.0),
    "STEEL_LINE_WIDTH": 1.0,
    "BAR_SEGMENTS":     40,

    # --- RINFORZI GENERICI ---
    "REINF_FILL":       (0.2, 0.5, 0.8, 0.4),
    "REINF_OUTLINE":    (0.4, 0.7, 1.0, 1.0),
    
    # --- TESTI E CURSORE ---
    "TEXT_COLOR":       QColor(200, 200, 200),
    "CURSOR_COLOR":     QColor(255, 255, 255, 200),
    "FONT_SIZE":        9,
    "FONT_FAMILY":      "Arial"
}

class OpenGLMomentocurvaturaSezioneWidget(QOpenGLWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        
        # --- Dati Geometrici ---
        self.render_shapes = [] 
        self.render_bars = []
        
        # --- Parametri Vista ---
        self.pan_2d = [0.0, 0.0]
        self.zoom_2d = 1.0
        self.last_mouse_pos = None
        self.cursor_pos = QPoint(0, 0)
        
        # Limiti del mondo
        self.world_bounds = {'min_x': -100, 'max_x': 100, 'min_y': -100, 'max_y': 100}
        
        self.setMouseTracking(True)
        self.font_obj = QFont(STYLE_CONFIG["FONT_FAMILY"], STYLE_CONFIG["FONT_SIZE"])

    # ==========================================================================
    # GESTIONE DATI
    # ==========================================================================
    def set_section_data(self, section_data):
        self.render_shapes = []
        self.render_bars = []
        
        if not section_data or 'elementi' not in section_data:
            self.update()
            return

        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        has_data = False

        try:
            for elem in section_data['elementi']:
                tipo = elem[0]
                
                if tipo == 'shape':
                    _, shape_type, _, _, *params = elem
                    
                    if shape_type == 'rect':
                        if len(params) == 2 and isinstance(params[0], (list, tuple)):
                            p1, p2 = params[0], params[1]
                            x1, y1 = p1
                            x2, y2 = p2
                            pts = [(min(x1,x2), min(y1,y2)), 
                                   (max(x1,x2), min(y1,y2)), 
                                   (max(x1,x2), max(y1,y2)), 
                                   (min(x1,x2), max(y1,y2))]
                            self.render_shapes.append({'type': 'poly', 'points': pts})
                            for px, py in pts:
                                min_x, max_x = min(min_x, px), max(max_x, px)
                                min_y, max_y = min(min_y, py), max(max_y, py)
                                has_data = True

                    elif shape_type == 'poly':
                        pts = params[0]
                        self.render_shapes.append({'type': 'poly', 'points': pts})
                        for px, py in pts:
                            min_x, max_x = min(min_x, px), max(max_x, px)
                            min_y, max_y = min(min_y, py), max(max_y, py)
                            has_data = True
                            
                    elif shape_type == 'circle':
                        center = params[0]
                        radius = params[1]
                        self.render_shapes.append({'type': 'circle', 'center': center, 'radius': radius})
                        min_x, max_x = min(min_x, center[0]-radius), max(max_x, center[0]+radius)
                        min_y, max_y = min(min_y, center[1]-radius), max(max_y, center[1]+radius)
                        has_data = True

                elif tipo == 'bar':
                    _, _, _, diam, center = elem
                    radius = diam / 2.0
                    self.render_bars.append({'center': center, 'radius': radius})
                    min_x, max_x = min(min_x, center[0]-radius), max(max_x, center[0]+radius)
                    min_y, max_y = min(min_y, center[1]-radius), max(max_y, center[1]+radius)
                    has_data = True

        except Exception as e:
            print(f"Errore parsing sezione per disegno: {e}")

        if has_data:
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            w = max_x - min_x
            h = max_y - min_y
            w = w * 1.2 if w > 0 else 100
            h = h * 1.2 if h > 0 else 100
            self.pan_2d = [cx, cy]
            self.zoom_2d = max(w, h) / 2.0
            self.world_bounds = {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}
        else:
            self.zoom_2d = 100.0
            self.pan_2d = [0,0]

        self.update()

    # ==========================================================================
    # OPENGL RENDERING
    # ==========================================================================
    def initializeGL(self):
        bgcolor = STYLE_CONFIG["BACKGROUND_COLOR"]
        glClearColor(*bgcolor)
        # Queste impostazioni iniziali sono utili, ma paintGL deve riaffermarle
        # perché QPainter potrebbe disabilitarle.
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        # 1. RESET DELLO STATO OPENGL (CRUCIALE PER EVITARE ARTEFATTI DOPO QPAINTER) 
        # QPainter "sporca" lo stato (disabilita blend, texture, ecc). Lo resettiamo qui.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_TEXTURE_2D) 
        glDisable(GL_DEPTH_TEST) # Siamo in 2D
        
        # 2. CLEAR
        glClear(GL_COLOR_BUFFER_BIT)
        
        # 3. SETUP PROIEZIONE
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        hw = self.zoom_2d * aspect
        hh = self.zoom_2d
        
        self.wx_min = self.pan_2d[0] - hw
        self.wx_max = self.pan_2d[0] + hw
        self.wy_min = self.pan_2d[1] - hh
        self.wy_max = self.pan_2d[1] + hh
        
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(self.wx_min, self.wx_max, self.wy_min, self.wy_max, -1, 1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        # 4. DISEGNO RAW OPENGL
        self._draw_grid()
        self._draw_concrete_shapes()
        self._draw_steel_bars()
        
        # 5. OVERLAY QPAINTER
        # Nota: QPainter qui modificherà lo stato OpenGL, ma noi lo abbiamo
        # resettato all'inizio di questa funzione per il prossimo frame.
        self._draw_overlay()

    def _draw_grid(self):
        dx = self.wx_max - self.wx_min
        dy = self.wy_max - self.wy_min
        tx = self._tick(dx)
        ty = self._tick(dy)
        
        glColor3f(*STYLE_CONFIG["GRID_COLOR"])
        glLineWidth(1)
        
        glBegin(GL_LINES)
        x = np.floor(self.wx_min / tx) * tx
        while x <= self.wx_max:
            glVertex2f(x, self.wy_min); glVertex2f(x, self.wy_max)
            x += tx
        y = np.floor(self.wy_min / ty) * ty
        while y <= self.wy_max:
            glVertex2f(self.wx_min, y); glVertex2f(self.wx_max, y)
            y += ty
        glEnd()

    def _draw_concrete_shapes(self):
        for shape in self.render_shapes:
            if shape['type'] == 'poly':
                pts = shape['points']
                # Fill
                glColor4f(*STYLE_CONFIG["CONCRETE_FILL"])
                glBegin(GL_POLYGON)
                for p in pts: glVertex2f(p[0], p[1])
                glEnd()
                # Outline
                glColor4f(*STYLE_CONFIG["CONCRETE_OUTLINE"])
                glLineWidth(STYLE_CONFIG["CONCRETE_LINE_WIDTH"])
                glBegin(GL_LINE_LOOP)
                for p in pts: glVertex2f(p[0], p[1])
                glEnd()
            elif shape['type'] == 'circle':
                self._draw_gl_circle(shape['center'], shape['radius'], 
                                     STYLE_CONFIG["CONCRETE_FILL"], 
                                     STYLE_CONFIG["CONCRETE_OUTLINE"], 
                                     STYLE_CONFIG["CONCRETE_LINE_WIDTH"])

    def _draw_steel_bars(self):
        for bar in self.render_bars:
            self._draw_gl_circle(bar['center'], bar['radius'], 
                                 STYLE_CONFIG["STEEL_FILL"], 
                                 STYLE_CONFIG["STEEL_OUTLINE"], 
                                 STYLE_CONFIG["STEEL_LINE_WIDTH"])

    def _draw_gl_circle(self, center, radius, fill_col, line_col, width):
        segments = STYLE_CONFIG["BAR_SEGMENTS"]
        thetas = np.linspace(0, 2*math.pi, segments, endpoint=True)
        cx, cy = center
        
        # Fill
        glColor4f(*fill_col)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for t in thetas:
            glVertex2f(cx + radius*math.cos(t), cy + radius*math.sin(t))
        glEnd()
        
        # Outline
        glColor4f(*line_col)
        glLineWidth(width)
        glBegin(GL_LINE_STRIP)
        for t in thetas:
            glVertex2f(cx + radius*math.cos(t), cy + radius*math.sin(t))
        glEnd()

    def _draw_overlay(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font_obj)
        metrics = painter.fontMetrics()
        w, h = self.width(), self.height()

        def to_screen(wx, wy):
            nx = (wx - self.wx_min) / (self.wx_max - self.wx_min)
            ny = (wy - self.wy_min) / (self.wy_max - self.wy_min)
            return int(nx * w), int((1 - ny) * h)

        sx0, sy0 = to_screen(0, 0)
        
        pen = QPen(STYLE_CONFIG["AXIS_X_COLOR"], 1)
        painter.setPen(pen)
        if 0 <= sy0 <= h:
            painter.drawLine(0, sy0, w, sy0)
            painter.drawText(w - 20, sy0 - 5, "X")
            
        pen.setColor(STYLE_CONFIG["AXIS_Y_COLOR"])
        painter.setPen(pen)
        if 0 <= sx0 <= w:
            painter.drawLine(sx0, 0, sx0, h)
            painter.drawText(sx0 + 5, 15, "Y")

        mx, my = self.cursor_pos.x(), self.cursor_pos.y()
        wx, wy = self._screen_to_world(mx, my)
        
        painter.setPen(STYLE_CONFIG["CURSOR_COLOR"])
        label = f"X: {wx:.1f}  Y: {wy:.1f}"
        painter.drawText(10, h - 10, label)
        
        painter.drawLine(mx - 10, my, mx + 10, my)
        painter.drawLine(mx, my - 10, mx, my + 10)

        painter.end()

    # ==========================================================================
    # UTILITIES & MATEMATICA
    # ==========================================================================
    def _tick(self, rng):
        rough = rng / 8.0
        if rough <= 0: return 1.0
        mag = 10 ** np.floor(np.log10(rough))
        res = rough / mag
        if res >= 5: return 5 * mag
        elif res >= 2: return 2 * mag
        return mag

    def _screen_to_world(self, sx, sy):
        w, h = self.width(), self.height()
        if w == 0 or h == 0: return 0,0
        nx = sx / w
        ny = 1.0 - (sy / h)
        range_x = self.wx_max - self.wx_min
        range_y = self.wy_max - self.wy_min
        wx = self.wx_min + nx * range_x
        wy = self.wy_min + ny * range_y
        return wx, wy

    # ==========================================================================
    # INPUT UTENTE (Mouse)
    # ==========================================================================
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
        sx, sy = e.pos().x(), e.pos().y()
        wx0, wy0 = self._screen_to_world(sx, sy)
        factor = 1.0 - np.sign(delta) * 0.1
        self.zoom_2d *= factor
        self.zoom_2d = max(0.1, min(10000.0, self.zoom_2d))
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        hw = self.zoom_2d * aspect
        hh = self.zoom_2d
        self.wx_min = self.pan_2d[0] - hw
        self.wx_max = self.pan_2d[0] + hw
        self.wy_min = self.pan_2d[1] - hh
        self.wy_max = self.pan_2d[1] + hh
        wx1, wy1 = self._screen_to_world(sx, sy)
        self.pan_2d[0] += (wx0 - wx1)
        self.pan_2d[1] += (wy0 - wy1)
        self.update()