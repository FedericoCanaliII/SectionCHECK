from typing import List, Dict, Any, Optional

# PyQt imports (usati dal tool)
from PyQt5.QtWidgets import QLineEdit, QShortcut
from PyQt5.QtGui import QPainter, QColor, QKeySequence
from PyQt5.QtCore import Qt, QRect

# OpenGL imports per il draw_gl (se usi OpenGL nel widget)
try:
    from OpenGL.GL import *
    _HAS_GL = True
except Exception:
    _HAS_GL = False

class RectangleTool:
    """Tool rettangolo che supporta storage esterno per i rettangoli confermati.

    - set_confirmed_list(list_ref): collega la lista esterna (pass-by-ref)
    - _get_confirmed_list(): ritorna la lista attiva (esterna o interna)
    - il resto del comportamento e' simile al tuo tool: draft_vertices, preview,
      conferma, editing, ecc.
    """
    class CoordLineEdit(QLineEdit):
        def __init__(self, parent, on_focus_lost):
            super().__init__(parent)
            self._on_focus_lost = on_focus_lost

        def focusOutEvent(self, ev):
            try:
                if callable(self._on_focus_lost):
                    self._on_focus_lost()
            except Exception:
                pass
            super().focusOutEvent(ev)

    def __init__(self, line_width: float = 2.0, line_color: QColor = QColor(255,255,255,255),
                 fill_color: QColor = QColor(0,0,0,60), preview_line_color: QColor = QColor(150,200,255,200),
                 preview_fill_color: QColor = QColor(135,206,235,120), snap_to_grid: bool = False,
                 grid_spacing: float = 1.0):
        self.draft_vertices = []
        self._internal_confirmed = []
        self._external_confirmed_ref = None
        self._counter = 0

        self._active_editor = None
        self._editing_index = None
        self._preview_second = None
        self._finalizing = False

        self.line_width = float(line_width)
        self.line_color = line_color
        self.fill_color = fill_color
        self.preview_line_color = preview_line_color
        self.preview_fill_color = preview_fill_color

        self.snap_to_grid = bool(snap_to_grid)
        self.grid_spacing = float(grid_spacing)

        # precalc OpenGL colors
        def _qcolor_to_gl(c: QColor):
            try:
                return c.getRgbF()
            except Exception:
                return (1.0,1.0,1.0,1.0)
        self._gl_fill_color = _qcolor_to_gl(self.fill_color)
        self._gl_line_color = _qcolor_to_gl(self.line_color)

        # shortcuts
        self._shortcuts = []

    # --- external storage API ---
    def set_confirmed_list(self, list_ref: Optional[List[Dict[str,Any]]]):
        """Passa una list reference per memorizzare i rettangoli confermati.
        Se None, usera' lo storage interno."""
        if list_ref is None:
            self._external_confirmed_ref = None
        else:
            self._external_confirmed_ref = list_ref

    def _get_confirmed_list(self) -> List[Dict[str,Any]]:
        return self._external_confirmed_ref if self._external_confirmed_ref is not None else self._internal_confirmed

    # --- activation/deactivation ---
    def on_activate(self, widget):
        try:
            widget.setCursor(Qt.CrossCursor)
        except Exception:
            pass
        # shortcuts Enter/Return
        try:
            ks_return = QShortcut(QKeySequence(Qt.Key_Return), widget)
            ks_return.activated.connect(lambda: self._on_enter_pressed(widget))
            self._shortcuts.append(ks_return)
            ks_enter = QShortcut(QKeySequence(Qt.Key_Enter), widget)
            ks_enter.activated.connect(lambda: self._on_enter_pressed(widget))
            self._shortcuts.append(ks_enter)
        except Exception:
            self._shortcuts = []

    def on_deactivate(self, widget):
        if self._active_editor is not None:
            try:
                self._finalize_editor(widget)
            except Exception:
                pass
        try:
            for sc in getattr(self, '_shortcuts', []):
                try:
                    sc.activated.disconnect()
                except Exception:
                    pass
                try:
                    sc.setParent(None)
                    sc.deleteLater()
                except Exception:
                    pass
            self._shortcuts = []
        except Exception:
            pass
        try:
            widget.unsetCursor()
        except Exception:
            pass

    def _on_enter_pressed(self, widget):
        try:
            if self._active_editor is not None:
                self._finalize_editor(widget)
                try:
                    widget.setFocus()
                except Exception:
                    pass
                widget.update()
                return
            if len(self.draft_vertices) == 2:
                self._confirm_current_rectangle(widget)
                widget.update()
        except Exception:
            pass

    # --- events ---
    def on_mouse_press(self, widget, event):
        if event.button() != Qt.LeftButton:
            return False
        px, py = event.x(), event.y()
        # editor handling
        if self._active_editor is not None:
            ed_geom = self._active_editor.geometry()
            if ed_geom.contains(px, py):
                return False
            else:
                self._finalize_editor(widget)
                widget.update()
                event.accept()
                return True
        # label click
        if len(self.draft_vertices) > 0:
            fm = widget.fontMetrics()
            for idx, (wx, wy) in enumerate(self.draft_vertices):
                sx, sy = widget.world_to_screen(wx, wy)
                label = f"({wx:.3f}, {wy:.3f})"
                tw = fm.horizontalAdvance(label)
                th = fm.height()
                label_rect = QRect(sx + 8, sy - th // 2, tw, th)
                if label_rect.contains(px, py):
                    self._open_editor_for_vertex(widget, idx, label_rect)
                    event.accept()
                    return True
        # add draft vertex
        if len(self.draft_vertices) < 2:
            wx, wy = widget.screen_to_world(px, py)
            self.draft_vertices.append((wx, wy))
            self._preview_second = None
            widget.update()
            event.accept()
            return True
        return False

    def on_mouse_move(self, widget, event):
        if len(self.draft_vertices) == 1:
            px, py = event.x(), event.y()
            wx, wy = widget.screen_to_world(px, py)
            self._preview_second = (wx, wy)
            widget.update()
            return True
        return False

    def on_mouse_release(self, widget, event):
        return False

    def on_wheel(self, widget, event):
        return False

    # --- drawing ---
    def draw_gl(self, widget):
        if not _HAS_GL:
            return
        confirmed = self._get_confirmed_list()
        for rect in confirmed:
            v1 = rect['v1']; v2 = rect['v2']
            left, right = min(v1[0], v2[0]), max(v1[0], v2[0])
            bottom, top = min(v1[1], v2[1]), max(v1[1], v2[1])
            corners = [(left, bottom),(left, top),(right, top),(right, bottom)]
            fr, fg, fb, fa = self._gl_fill_color
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(fr, fg, fb, fa)
            glBegin(GL_QUADS)
            for x,y in corners:
                glVertex2f(x,y)
            glEnd()
            lr, lg, lb, la = self._gl_line_color
            glColor4f(lr, lg, lb, 1.0)
            glLineWidth(self.line_width)
            glBegin(GL_LINE_LOOP)
            for x,y in corners:
                glVertex2f(x,y)
            glEnd()

    def draw_painter(self, widget, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)
        fm = painter.fontMetrics()
        # preview
        if len(self.draft_vertices) >= 1:
            v1 = self.draft_vertices[0]
            if len(self.draft_vertices) == 2:
                v2 = self.draft_vertices[1]
            else:
                v2 = self._preview_second if self._preview_second is not None else v1
            x1, y1 = v1; x2, y2 = v2
            left, right = min(x1,x2), max(x1,x2)
            bottom, top = min(y1,y2), max(y1,y2)
            sx1, sy1 = widget.world_to_screen(left, bottom)
            sx2, sy2 = widget.world_to_screen(right, top)
            rx, ry = sx1, sy2
            rw, rh = sx2 - sx1, sy1 - sy2
            painter.setBrush(self.preview_fill_color)
            pen = painter.pen()
            pen.setWidth(1)
            pen.setColor(self.preview_line_color)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(rx, ry, rw, rh)
            for idx, (wx, wy) in enumerate(self.draft_vertices):
                sx, sy = widget.world_to_screen(wx, wy)
                painter.setBrush(QColor(150,220,255,255))
                pen = painter.pen()
                pen.setColor(QColor(40,100,140))
                painter.setPen(pen)
                radius = 6
                painter.drawEllipse(sx - radius, sy - radius, radius*2, radius*2)
                lbl = f"({wx:.3f}, {wy:.3f})"
                tw = fm.horizontalAdvance(lbl)
                th = fm.height()
                painter.setPen(QColor(255,255,255))
                painter.drawText(sx + 8, sy + th // 2 - 2, lbl)
        # confirmed names
        painter.setPen(QColor(255,255,255))
        confirmed = self._get_confirmed_list()
        for rect in confirmed:
            x1, y1 = rect['v1']; x2, y2 = rect['v2']
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            sx, sy = widget.world_to_screen(cx, cy)
            name = rect.get('name', '')
            tw = fm.horizontalAdvance(name)
            painter.drawText(sx - tw // 2, sy + fm.height() // 2, name)

    # --- editor helpers ---
    def _open_editor_for_vertex(self, widget, vertex_index, label_rect: QRect):
        if vertex_index >= len(self.draft_vertices):
            return
        if self._active_editor is not None:
            self._finalize_editor(widget)
        wx, wy = self.draft_vertices[vertex_index]
        text = f"{wx:.6g}, {wy:.6g}"
        editor = RectangleTool.CoordLineEdit(widget, on_focus_lost=lambda: self._finalize_editor(widget))
        editor.setText(text)
        extra_w = 10
        editor.setGeometry(label_rect.x(), label_rect.y(), label_rect.width() + extra_w, label_rect.height() + 4)
        editor.show()
        editor.setFocus()
        editor.setAttribute(Qt.WA_DeleteOnClose, True)
        self._active_editor = editor
        self._editing_index = vertex_index
        def _on_return():
            self._finalize_editor(widget)
            try:
                widget.setFocus()
            except Exception:
                pass
        editor.returnPressed.connect(_on_return)
        editor.editingFinished.connect(lambda: self._finalize_editor(widget))

    def _finalize_editor(self, widget):
        if self._finalizing:
            return
        self._finalizing = True
        try:
            if self._active_editor is None or self._editing_index is None:
                return
            editor = self._active_editor
            idx = self._editing_index
            txt = ''
            try:
                txt = editor.text().strip()
            except Exception:
                txt = ''
            try:
                editor.returnPressed.disconnect()
            except Exception:
                pass
            try:
                editor.editingFinished.disconnect()
            except Exception:
                pass
            try:
                editor.hide()
                editor.close()
            except Exception:
                pass
            try:
                editor.deleteLater()
            except Exception:
                pass
            self._active_editor = None
            self._editing_index = None
            parts = txt.replace(';', ' ').replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    x_new = float(parts[0])
                    y_new = float(parts[1])
                    snap = getattr(widget, 'snap_to_grid', self.snap_to_grid)
                    if snap:
                        spacing = getattr(widget, 'grid_spacing', self.grid_spacing)
                        try:
                            if spacing is None or float(spacing) == 0.0:
                                spacing = self.grid_spacing
                        except Exception:
                            spacing = self.grid_spacing
                        x_new = round(x_new / spacing) * spacing
                        y_new = round(y_new / spacing) * spacing
                    if 0 <= idx < len(self.draft_vertices):
                        self.draft_vertices[idx] = (x_new, y_new)
                except Exception:
                    pass
        finally:
            try:
                widget.update()
                widget.setFocus()
            except Exception:
                pass
            self._finalizing = False

    def _confirm_current_rectangle(self, widget):
        if len(self.draft_vertices) != 2:
            return
        v1 = self.draft_vertices[0]
        v2 = self.draft_vertices[1]
        self._counter += 1
        name = f"RETTANGOLO {self._counter}"
        rect_entry = {'v1': v1, 'v2': v2, 'name': name}
        self._get_confirmed_list().append(rect_entry)
        self.draft_vertices = []
        self._preview_second = None
        try:
            widget.update()
        except Exception:
            pass
