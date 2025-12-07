from PyQt5.QtWidgets import QOpenGLWidget, QLineEdit
from PyQt5.QtGui import QPainter, QColor, QFontMetrics
from PyQt5.QtCore import Qt, QPoint
from OpenGL.GL import *
import numpy as np

# ---------------- OpenGLSectionWidget ----------------
class OpenGLSectionWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # valori originali forniti
        self.data_range_x = 152.0
        self.data_range_y = 100.0

        # view transform
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0

        # input / cursor
        self.last_mouse_pos = None
        self.cursor_pos = QPoint(0, 0)
        self.setMouseTracking(True)

        # griglia
        self.show_grid = True
        self.snap_to_grid = True
        # grid_spacing deve essere impostato esternamente prima dell'uso,
        # oppure puoi inizializzarlo qui con un valore sensato
        self.grid_spacing = 1.0

        # tool management
        self.active_tool = None
        self._tools = []

        # track for internal middle-button panning
        self._middle_pressed = False

        # tracker coordinates (world)
        self.tracker_x = 0.0
        self.tracker_y = 0.0

    # ---- helper: compute world bounds that enforce 1:1 pixel ratio ----
    def _world_bounds(self):
        """
        Returns (world_min_x, world_max_x, world_min_y, world_max_y)
        such that the projection preserves 1 world-unit == 1 (same) pixel scale on X and Y.
        We base vertical half-range on data_range_y and adjust horizontal half-range
        to match widget aspect ratio.
        """
        w, h = max(1, self.width()), max(1, self.height())
        aspect = w / h

        # choose half-height from data_range_y * zoom
        half_h = self.data_range_y * self.zoom
        # derive half-width so that unit-to-pixel scale is equal:
        # half_w / w == half_h / h  => half_w = half_h * (w/h) = half_h * aspect
        half_w = half_h * aspect

        world_min_x = -half_w + self.pan_x
        world_max_x = half_w + self.pan_x
        world_min_y = -half_h + self.pan_y
        world_max_y = half_h + self.pan_y

        return world_min_x, world_max_x, world_min_y, world_max_y

    # ---- utility conversion methods (use world bounds) ----
    def screen_to_world(self, sx, sy):
        w, h = max(1, self.width()), max(1, self.height())
        nx = sx / w
        ny = 1 - (sy / h)
        world_min_x, world_max_x, world_min_y, world_max_y = self._world_bounds()
        wx = world_min_x + nx * (world_max_x - world_min_x)
        wy = world_min_y + ny * (world_max_y - world_min_y)
        if self.snap_to_grid:
            wx = round(wx / self.grid_spacing) * self.grid_spacing
            wy = round(wy / self.grid_spacing) * self.grid_spacing
        return wx, wy

    def world_to_screen(self, wx, wy):
        w, h = max(1, self.width()), max(1, self.height())
        world_min_x, world_max_x, world_min_y, world_max_y = self._world_bounds()
        sx = int((wx - world_min_x) / (world_max_x - world_min_x) * w)
        sy = int((1 - (wy - world_min_y) / (world_max_y - world_min_y)) * h)
        return sx, sy

    # ---- OpenGL lifecycle ----
    def initializeGL(self):
        glClearColor(40 / 255, 40 / 255, 40 / 255, 1.0)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        # viewport must match widget size
        glViewport(0, 0, w, h)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Non cambiamo qui data_range_x/data_range_y: la logica di adattamento
        # è delegata a _world_bounds() e quindi la vista rimane 1:1

    def reset_view(self):
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # use computed bounds that guarantee 1:1 unit-to-pixel ratio
        world_min_x, world_max_x, world_min_y, world_max_y = self._world_bounds()
        glOrtho(
            world_min_x, world_max_x,
            world_min_y, world_max_y,
            -1.0, 1.0
        )

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # draw grid and any world-space geometry produced by tools (OpenGL drawing)
        self._draw_grid()

        # let each registered tool draw its persistent world geometry using OpenGL
        for t in self._tools:
            draw_gl = getattr(t, 'draw_gl', None)
            if callable(draw_gl):
                try:
                    draw_gl(self)
                except Exception:
                    pass

        # after OpenGL drawing, we use QPainter to draw screen overlays/labels/axes/tracker
        self._draw_labels()
        self._draw_screen_axes()

        # painter-based dynamic previews (tools can implement draw_painter(widget, painter))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            for t in self._tools:
                draw_painter = getattr(t, 'draw_painter', None)
                if callable(draw_painter):
                    try:
                        draw_painter(self, painter)
                    except Exception:
                        pass
            # Also give active tool a chance to draw dynamic preview (if not already drawn)
            if self.active_tool is not None:
                draw_painter = getattr(self.active_tool, 'draw_painter', None)
                if callable(draw_painter):
                    try:
                        draw_painter(self, painter)
                    except Exception:
                        pass
        finally:
            painter.end()

        # draw tracker last using painter in its own methods (reuse existing impl)
        self._draw_tracker()

    # ---- grid / labels / axes / tracker ----
    def _draw_grid(self):
        if not self.show_grid:
            return

        world_min_x, world_max_x, world_min_y, world_max_y = self._world_bounds()

        tx = ty = self.grid_spacing

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

    def _draw_labels(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        w, h = self.width(), self.height()

        world_min_x, world_max_x, world_min_y, world_max_y = self._world_bounds()

        def to_screen(wx, wy):
            nx = (wx - world_min_x) / (world_max_x - world_min_x)
            ny = (wy - world_min_y) / (world_max_y - world_min_y)
            return int(nx * w), int((1 - ny) * h)

        painter.setPen(QColor(255, 100, 100, 150))
        tx = self.grid_spacing
        x = np.floor(world_min_x / tx) * tx
        while x <= world_max_x:
            if abs(x) > 1e-8:
                sx, oy = to_screen(x, 0)
                painter.drawLine(sx, oy - 4, sx, oy + 4)
                lbl = f"{x:.4g}"
                tw = metrics.horizontalAdvance(lbl)
                painter.drawText(sx - tw // 2, oy + metrics.height() + 2, lbl)
            x += tx

        painter.setPen(QColor(100, 255, 100, 150))
        ty = self.grid_spacing
        y = np.floor(world_min_y / ty) * ty
        while y <= world_max_y:
            if abs(y) > 1e-8:
                ox, sy = to_screen(0, y)
                painter.drawLine(ox - 4, sy, ox + 4, sy)
                lbl = f"{y:.4g}"
                tw = metrics.horizontalAdvance(lbl)
                painter.drawText(ox - tw - 6, sy + metrics.height() // 2 - 2, lbl)
            y += ty
        painter.end()

    def set_show_grid(self, show: bool):
        self.show_grid = show
        self.update()

    def set_grid_spacing(self, spacing: float):
        self.grid_spacing = spacing
        self.update()

    def set_snap_to_grid(self, enabled: bool):
        self.snap_to_grid = enabled

    def update_tracker_position(self, x, y):
        if self.snap_to_grid:
            x = round(x / self.grid_spacing) * self.grid_spacing
            y = round(y / self.grid_spacing) * self.grid_spacing
        self.tracker_x = x
        self.tracker_y = y
        self.update()

    def _draw_screen_axes(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        world_min_x, world_max_x, world_min_y, world_max_y = self._world_bounds()

        def to_screen(wx, wy):
            nx = (wx - world_min_x) / (world_max_x - world_min_x)
            ny = (wy - world_min_y) / (world_max_y - world_min_y)
            return int(nx * w), int((1 - ny) * h)

        ox, oy = to_screen(0, 0)
        pen = painter.pen()
        pen.setColor(QColor(255, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, oy, self.width(), oy)
        pen.setColor(QColor(0, 255, 0))
        painter.setPen(pen)
        painter.drawLine(ox, 0, ox, self.height())
        painter.end()

    def _draw_tracker(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        x, y = self.cursor_pos.x(), self.cursor_pos.y()

        world_min_x, world_max_x, world_min_y, world_max_y = self._world_bounds()

        # normalized screen coords -> world
        nx = x / max(1, w)
        ny = 1 - (y / max(1, h))
        wx = world_min_x + nx * (world_max_x - world_min_x)
        wy = world_min_y + ny * (world_max_y - world_min_y)

        # apply snap if needed
        if self.snap_to_grid:
            wx = round(wx / self.grid_spacing) * self.grid_spacing
            wy = round(wy / self.grid_spacing) * self.grid_spacing

        # reconvert to screen for accurate crosshair placement
        sx = int((wx - world_min_x) / (world_max_x - world_min_x) * w)
        sy = int((1 - (wy - world_min_y) / (world_max_y - world_min_y)) * h)

        pen = painter.pen()
        pen.setColor(QColor(255, 255, 255, 255))
        pen.setWidth(1)
        painter.setPen(pen)
        size = 10
        painter.drawLine(sx - size, sy, sx + size, sy)
        painter.drawLine(sx, sy - size, sx, sy + size)

        pen.setColor(QColor(100, 100, 100, 150))
        painter.setPen(pen)
        painter.drawLine(sx, 0, sx, h)
        painter.drawLine(0, sy, self.width(), sy)

        pen.setColor(QColor(255, 255, 255, 150))
        painter.setPen(pen)
        painter.drawText(sx + 8, self.height() - 8, f"X: {wx:.4g}")
        painter.drawText(8, sy - 8, f"Y: {wy:.4g}")
        painter.end()

    # ---- tool integration ----
    def set_active_tool(self, tool):
        # deactivate current
        if self.active_tool is tool:
            return
        if self.active_tool is not None:
            try:
                self.active_tool.on_deactivate(self)
            except Exception:
                pass
        self.active_tool = tool
        if tool is not None:
            # register tool if not present (so its persistent drawing appears)
            if tool not in self._tools:
                self._tools.append(tool)
            try:
                tool.on_activate(self)
            except Exception:
                pass
        self.update()

    # ---- event handling: delegate to active tool first ----
    def mousePressEvent(self, e):
        self.cursor_pos = e.pos()
        if self.active_tool is not None:
            try:
                consumed = self.active_tool.on_mouse_press(self, e)
                if consumed:
                    self.update()
                    return
            except Exception:
                pass

        # fallback: middle-button panning
        if e.button() == Qt.MiddleButton:
            self._middle_pressed = True
            self.last_mouse_pos = e.pos()
        self.update()

    def mouseMoveEvent(self, e):
        self.cursor_pos = e.pos()
        if self.active_tool is not None:
            try:
                consumed = self.active_tool.on_mouse_move(self, e)
                if consumed:
                    self.update()
                    return
            except Exception:
                pass

        # fallback middle-button pan handling
        if self._middle_pressed and self.last_mouse_pos:
            dx = e.x() - self.last_mouse_pos.x()
            dy = e.y() - self.last_mouse_pos.y()

            # convert pixel delta to world delta using current world bounds
            w, h = max(1, self.width()), max(1, self.height())
            world_min_x, world_max_x, world_min_y, world_max_y = self._world_bounds()

            span_x = (world_max_x - world_min_x)
            span_y = (world_max_y - world_min_y)

            # note: moving right (dx > 0) should pan left in world coordinates (decrease pan_x)
            self.pan_x -= dx * (span_x / w)
            # moving down (dy > 0) should pan up in world coordinates (increase pan_y)
            self.pan_y += dy * (span_y / h)

            self.last_mouse_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self.cursor_pos = e.pos()
        if self.active_tool is not None:
            try:
                consumed = self.active_tool.on_mouse_release(self, e)
                if consumed:
                    self.update()
                    return
            except Exception:
                pass

        if e.button() == Qt.MiddleButton:
            self._middle_pressed = False
            self.last_mouse_pos = None
        self.update()

    def wheelEvent(self, e):
        # give tool first chance
        if self.active_tool is not None:
            try:
                consumed = self.active_tool.on_wheel(self, e)
                if consumed:
                    self.update()
                    return
            except Exception:
                pass

        delta = e.angleDelta().y() / 120
        factor = 1.15
        if delta > 0:
            self.zoom /= factor
        else:
            self.zoom *= factor
        self.update()
