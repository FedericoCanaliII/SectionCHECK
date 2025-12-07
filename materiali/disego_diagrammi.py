from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QPoint
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class OpenGLGraphWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.curves = []
        self.input_data = []
        # Base data bounds
        self.data_range_x = 10.0
        self.data_range_y = 10.0
        # Interaction
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.last_mouse_pos = None
        # Tracker
        self.cursor_pos = QPoint(0, 0)
        # Enable continuous mouse tracking
        self.setMouseTracking(True)

    def set_data(self, data_matrix):
        self.curves.clear()
        self.input_data = data_matrix
        all_x, all_y = [], []
        for formula, xs, xe in data_matrix:
            try:
                x = np.linspace(xs, xe, 300)
                y = eval(formula, {"x": x, "np": np, "__builtins__": {}}, {})
                if not isinstance(y, np.ndarray):
                    y = np.full_like(x, y)
                self.curves.append((x, y))
                all_x.append(x)
                all_y.append(y)
            except Exception as e:
                print(f"Errore formula {formula}: {e}")
        if all_x:
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            mx = max(abs(all_x.min()), abs(all_x.max()))
            my = max(abs(all_y.min()), abs(all_y.max()))
            self.data_range_x = mx or 1.0
            self.data_range_y = my or 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.update()

    def initializeGL(self):
        glClearColor(40/255, 40/255, 40/255, 1.0)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def reset_view(self):
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.update()

    def recalculate(self):
        self.set_data(self.input_data)
        self.update()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        # Aggiorna la proiezione con pan e zoom
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(
            -self.data_range_x * self.zoom + self.pan_x,
             self.data_range_x * self.zoom + self.pan_x,
            -self.data_range_y * self.zoom + self.pan_y,
             self.data_range_y * self.zoom + self.pan_y,
            -1.0, 1.0
        )
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self._draw_grid()
        self._draw_curves()
        self._draw_guides()
        self._draw_labels()
        self._draw_screen_axes()
        self._draw_tracker()

    def _draw_grid(self):
        half_wx = (self.data_range_x * 2 * self.zoom) / 2
        half_hy = (self.data_range_y * 2 * self.zoom) / 2
        world_min_x = -half_wx + self.pan_x
        world_max_x = half_wx + self.pan_x
        world_min_y = -half_hy + self.pan_y
        world_max_y = half_hy + self.pan_y
        tx = self._tick(world_max_x - world_min_x)
        ty = self._tick(world_max_y - world_min_y)
        glColor3f(0.2, 0.2, 0.2)
        glLineWidth(1)
        x = np.floor(world_min_x / tx) * tx
        while x <= world_max_x:
            glBegin(GL_LINES)
            glVertex2f(x, world_min_y)
            glVertex2f(x, world_max_y)
            glEnd()
            x += tx
        y = np.floor(world_min_y / ty) * ty
        while y <= world_max_y:
            glBegin(GL_LINES)
            glVertex2f(world_min_x, y)
            glVertex2f(world_max_x, y)
            glEnd()
            y += ty

    def _draw_curves(self):
        glColor3f(1, 1, 1)
        glLineWidth(2)
        for x, y in self.curves:
            glBegin(GL_LINE_STRIP)
            for xi, yi in zip(x, y):
                glVertex2f(xi, yi)
            glEnd()

    def _draw_guides(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(10, 0xAAAA)
        glColor3f(0.7, 0.7, 0.7)
        for expr, xs, xe in self.input_data:
            for xv in (xs, xe):
                if abs(xv) < 1e-8:
                    continue
                try:
                    y0 = eval(expr, {"__builtins__": None}, {"x": xv})
                except Exception:
                    continue
                glBegin(GL_LINES)
                glVertex2f(xv, y0)
                glVertex2f(xv, 0.0)
                glEnd()
        glDisable(GL_LINE_STIPPLE)

    def _draw_labels(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        w, h = self.width(), self.height()
        def to_screen(wx, wy):
            nx = (wx - (-self.data_range_x * self.zoom + self.pan_x)) / (self.data_range_x * 2 * self.zoom)
            ny = (wy - (-self.data_range_y * self.zoom + self.pan_y)) / (self.data_range_y * 2 * self.zoom)
            return int(nx * w), int((1 - ny) * h)
        painter.setPen(QColor(255, 100, 100))
        world_min_x = -self.data_range_x * self.zoom + self.pan_x
        world_max_x = self.data_range_x * self.zoom + self.pan_x
        tx = self._tick(world_max_x - world_min_x)
        x = np.floor(world_min_x / tx) * tx
        while x <= world_max_x:
            if abs(x) > 1e-8:
                sx, oy = to_screen(x, 0)
                painter.drawLine(sx, oy-4, sx, oy+4)
                lbl = f"{x:.2g}"; tw = metrics.horizontalAdvance(lbl)
                painter.drawText(sx-tw//2, oy+metrics.height()+2, lbl)
            x += tx
        painter.setPen(QColor(100, 255, 100))
        world_min_y = -self.data_range_y * self.zoom + self.pan_y
        world_max_y = self.data_range_y * self.zoom + self.pan_y
        ty = self._tick(world_max_y - world_min_y)
        y = np.floor(world_min_y / ty) * ty
        while y <= world_max_y:
            if abs(y) > 1e-8:
                ox, sy = to_screen(0, y)
                painter.drawLine(ox-4, sy, ox+4, sy)
                lbl = f"{y:.0f}"; tw = metrics.horizontalAdvance(lbl)
                painter.drawText(ox-tw-6, sy+metrics.height()//2-2, lbl)
            y += ty
        title_x = "Deformazione [ε]"
        painter.setPen(QColor(255, 0, 0))
        txw = metrics.horizontalAdvance(title_x)
        painter.drawText(w - txw - 10, h - 10, title_x)
        painter.setPen(QColor(0, 255, 0))
        painter.save()
        painter.translate(20, 110)
        painter.rotate(-90)
        painter.drawText(0, 0, "Tensione [σ]")
        painter.restore()

    def _draw_screen_axes(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        def to_screen(wx, wy):
            nx = (wx - (-self.data_range_x * self.zoom + self.pan_x)) / (self.data_range_x * 2 * self.zoom)
            ny = (wy - (-self.data_range_y * self.zoom + self.pan_y)) / (self.data_range_y * 2 * self.zoom)
            return int(nx * w), int((1 - ny) * h)
        ox, oy = to_screen(0, 0)
        pen = painter.pen()
        pen.setColor(QColor(255, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, oy, w, oy)
        pen.setColor(QColor(0, 255, 0))
        painter.setPen(pen)
        painter.drawLine(ox, 0, ox, h)
        painter.end()

    def _draw_tracker(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        x, y = self.cursor_pos.x(), self.cursor_pos.y()
        # Small light cross aligned with axes
        pen = painter.pen()
        pen.setColor(QColor(255, 255, 255, 255))
        pen.setWidth(1)
        painter.setPen(pen)
        size = 10
        painter.drawLine(x-size, y, x+size, y)
        painter.drawLine(x, y-size, x, y+size)
        # Dark full axes
        pen.setColor(QColor(100, 100, 100, 150))
        painter.setPen(pen)
        painter.drawLine(x, 0, x, h)
        painter.drawLine(0, y, w, y)
        # Compute world coordinates
        nx = x / w
        ny = 1 - (y / h)
        world_min_x = -self.data_range_x * self.zoom + self.pan_x
        world_max_x = self.data_range_x * self.zoom + self.pan_x
        world_min_y = -self.data_range_y * self.zoom + self.pan_y
        world_max_y = self.data_range_y * self.zoom + self.pan_y
        wx = world_min_x + nx * (world_max_x - world_min_x)
        wy = world_min_y + ny * (world_max_y - world_min_y)
        # Set color for coordinate labels
        pen.setColor(QColor(255, 255, 255, 150))  # white semi-opaque
        painter.setPen(pen)
        # Draw coordinates at margins
        painter.drawText(x + 8, h - 8, f"X: {wx:.3g}")
        painter.drawText(8, y - 8, f"Y: {wy:.3g}")
        painter.end()

    def _tick(self, rng):
        rough = rng / 10
        mag = 10**np.floor(np.log10(rough))
        res = rough / mag
        return 5 * mag if res >= 5 else 2 * mag if res >= 2 else mag

    # Interaction
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.last_mouse_pos = e.pos()
        elif e.button() == Qt.MiddleButton:
            self.middle_mouse_pressed = True
            self.last_mouse_pos = e.pos()
        self.cursor_pos = e.pos()
        self.update()

    def mouseMoveEvent(self, e):
        self.cursor_pos = e.pos()
        if self.last_mouse_pos:
            dx = e.x() - self.last_mouse_pos.x()
            dy = e.y() - self.last_mouse_pos.y()
            self.pan_x -= dx * (2 * self.data_range_x * self.zoom) / self.width()
            self.pan_y += dy * (2 * self.data_range_y * self.zoom) / self.height()
            self.last_mouse_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self.last_mouse_pos = None
        self.cursor_pos = e.pos()
        self.update()

    def wheelEvent(self, e):
        delta = e.angleDelta().y() / 120
        factor = 1.15
        if delta > 0:
            self.zoom /= factor
        else:
            self.zoom *= factor
        self.update()
