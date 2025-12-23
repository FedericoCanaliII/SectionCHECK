from PyQt5.QtWidgets import QLineEdit, QShortcut
from PyQt5.QtGui import QPainter, QColor, QKeySequence
from PyQt5.QtCore import Qt, QRect
import numpy as np

# OpenGL import opzionale (se manca, draw_gl non fa nulla)
try:
    from OpenGL.GL import *
    _HAS_GL = True
except Exception:
    _HAS_GL = False


class CircleTool:
    class CoordLineEdit(QLineEdit):
        """Sottoclasse che intercetta focusOutEvent per assicurare la
        chiamata al callback quando l'utente clicca fuori."""
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

    def __init__(self,
                 line_width: float = 2.0,
                 line_color: QColor = QColor(255, 255, 255, 255),
                 fill_color: QColor = QColor(255,255,255, 26),
                 preview_line_color: QColor = QColor(150, 200, 255, 200),
                 preview_fill_color: QColor = QColor(135, 206, 235, 120),
                 snap_to_grid: bool = False,
                 grid_spacing: float = 1.0):
        """
        CircleTool compatibile con SectionManager:
         - usa set_confirmed_list(list_ref) per collegare storage esterno
         - draft_vertices: [center, radius_point] (max 2)
        """
        # draft internals
        self.draft_vertices = []
        # internal/external storage for confirmed circles
        self._internal_confirmed = []
        self._external_confirmed_ref = None
        self._counter = 0

        self._active_editor = None
        self._editing_index = None
        self._preview_second = None
        self._finalizing = False

        # styles
        self.line_width = float(line_width)
        self.line_color = line_color
        self.fill_color = fill_color
        self.preview_line_color = preview_line_color
        self.preview_fill_color = preview_fill_color

        # snapping
        self.snap_to_grid = bool(snap_to_grid)
        self.grid_spacing = float(grid_spacing)

        # precalc GL colors
        def _qcolor_to_gl(c: QColor):
            try:
                return c.getRgbF()
            except Exception:
                return (1.0, 1.0, 1.0, 1.0)
        self._gl_fill_color = _qcolor_to_gl(self.fill_color)
        self._gl_line_color = _qcolor_to_gl(self.line_color)

        # shortcuts container
        self._shortcuts = []

    # --- external storage API ---
    def set_confirmed_list(self, list_ref):
        """Passa una reference ad una lista che conterrà i cerchi confermati.
        Se None, si userà lo storage interno."""
        if list_ref is None:
            self._external_confirmed_ref = None
        else:
            self._external_confirmed_ref = list_ref

    def _get_confirmed_list(self):
        return self._external_confirmed_ref if self._external_confirmed_ref is not None else self._internal_confirmed

    # optional: permettere persistenza dei draft per-sezione (get/set)
    def set_draft(self, draft):
        """Accetta una lista di 0..2 vertici per ripristinare lo stato di draft."""
        try:
            self.draft_vertices = list(draft) if draft is not None else []
        except Exception:
            self.draft_vertices = []

    def get_draft(self):
        return list(self.draft_vertices)

    # ---- activation / deactivation ----
    def on_activate(self, widget):
        try:
            widget.setCursor(Qt.CrossCursor)
        except Exception:
            pass

        # crea shortcuts Enter/Return per finalizzare editor o confermare
        try:
            self._shortcuts = []
            ks_return = QShortcut(QKeySequence(Qt.Key_Return), widget)
            ks_return.activated.connect(lambda: self._on_enter_pressed(widget))
            self._shortcuts.append(ks_return)

            ks_enter = QShortcut(QKeySequence(Qt.Key_Enter), widget)
            ks_enter.activated.connect(lambda: self._on_enter_pressed(widget))
            self._shortcuts.append(ks_enter)
        except Exception:
            self._shortcuts = []

    def on_deactivate(self, widget):
        # finalizza editor se presente
        if self._active_editor is not None:
            try:
                self._finalize_editor(widget)
            except Exception:
                pass

        # rimuovi shortcuts
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
                self._confirm_current_circle(widget)
                widget.update()
        except Exception:
            pass

    # ---- input events ----
    def on_mouse_press(self, widget, event):
        if event.button() != Qt.LeftButton:
            return False
        px, py = event.x(), event.y()

        # se editor attivo
        if self._active_editor is not None:
            ed_geom = self._active_editor.geometry()
            if ed_geom.contains(px, py):
                return False
            else:
                self._finalize_editor(widget)
                widget.update()
                event.accept()
                return True

        # click su label/vertice -> apri editor
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

        # piazza vertice (center o radius point)
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

    def on_key_press(self, widget, event):
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self._active_editor is not None:
                self._finalize_editor(widget)
                widget.update()
                return True
            else:
                if len(self.draft_vertices) == 2:
                    self._confirm_current_circle(widget)
                    widget.update()
                    return True
                return False

        if key == Qt.Key_Escape:
            if self._active_editor is not None:
                try:
                    self._active_editor.deleteLater()
                except Exception:
                    pass
                self._active_editor = None
                self._editing_index = None
            if len(self.draft_vertices) > 0:
                self.draft_vertices = []
                self._preview_second = None
                widget.update()
                return True
            return False

        return False

    # ---- drawing OpenGL ----
    def draw_gl(self, widget):
        if not _HAS_GL:
            return
        SEGMENTS = 64
        confirmed = self._get_confirmed_list()
        for circ in confirmed:
            cx, cy = circ['center']
            px, py = circ['radius_point']
            dx = px - cx
            dy = py - cy
            r = float(np.hypot(dx, dy))

            # fill
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            fr, fg, fb, fa = self._gl_fill_color
            glColor4f(fr, fg, fb, fa)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(SEGMENTS + 1):
                ang = 2.0 * np.pi * (i / SEGMENTS)
                x = cx + np.cos(ang) * r
                y = cy + np.sin(ang) * r
                glVertex2f(x, y)
            glEnd()

            # outline
            lr, lg, lb, la = self._gl_line_color
            glColor4f(lr, lg, lb, 1.0)
            glLineWidth(self.line_width)
            glBegin(GL_LINE_LOOP)
            for i in range(SEGMENTS):
                ang = 2.0 * np.pi * (i / SEGMENTS)
                x = cx + np.cos(ang) * r
                y = cy + np.sin(ang) * r
                glVertex2f(x, y)
            glEnd()

    # ---- drawing QPainter ----
    def draw_painter(self, widget, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)
        fm = painter.fontMetrics()

        # preview
        if len(self.draft_vertices) >= 1:
            center = self.draft_vertices[0]
            if len(self.draft_vertices) == 2:
                rp = self.draft_vertices[1]
            else:
                rp = self._preview_second if self._preview_second is not None else center

            cx, cy = center
            px, py = rp
            r_world = float(np.hypot(px - cx, py - cy))

            sx_c, sy_c = widget.world_to_screen(cx, cy)
            sx_r, sy_r = widget.world_to_screen(cx + r_world, cy)
            screen_r = abs(sx_r - sx_c)

            painter.setBrush(self.preview_fill_color)
            pen = painter.pen()
            pen.setWidth(1)
            pen.setColor(self.preview_line_color)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawEllipse(int(sx_c - screen_r), int(sy_c - screen_r), int(screen_r * 2), int(screen_r * 2))

            # markers + labels
            for idx, (wx, wy) in enumerate(self.draft_vertices):
                sx, sy = widget.world_to_screen(wx, wy)
                painter.setBrush(QColor(150, 220, 255, 255))
                pen = painter.pen()
                pen.setColor(QColor(40, 100, 140))
                painter.setPen(pen)
                radius = 6
                painter.drawEllipse(sx - radius, sy - radius, radius*2, radius*2)
                lbl = f"({wx:.3f}, {wy:.3f})"
                tw = fm.horizontalAdvance(lbl)
                th = fm.height()
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(sx + 8, sy + th // 2 - 2, lbl)

        # confirmed names
        painter.setPen(QColor(255, 255, 255))
        confirmed = self._get_confirmed_list()
        for circ in confirmed:
            cx, cy = circ['center']
            sx, sy = widget.world_to_screen(cx, cy)
            name = circ.get('name', '')
            tw = fm.horizontalAdvance(name)
            painter.drawText(sx - tw // 2, sy + fm.height() // 2, name)

    # ---- editor helpers ----
    def _open_editor_for_vertex(self, widget, vertex_index, label_rect: QRect):
        if vertex_index >= len(self.draft_vertices):
            return

        if self._active_editor is not None:
            self._finalize_editor(widget)

        wx, wy = self.draft_vertices[vertex_index]
        text = f"{wx:.6g}, {wy:.6g}"

        editor = CircleTool.CoordLineEdit(widget, on_focus_lost=lambda: self._finalize_editor(widget))
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

            # parse coordinates
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

    # ---- confirm ----
    def _confirm_current_circle(self, widget):
        if len(self.draft_vertices) != 2:
            return

        center = self.draft_vertices[0]
        rp = self.draft_vertices[1]
        self._counter += 1
        name = f"CERCHIO {self._counter}"

        circ_entry = {'center': center, 'radius_point': rp, 'name': name}
        # append alla lista attiva (esterna se presente)
        self._get_confirmed_list().append(circ_entry)

        # reset draft
        self.draft_vertices = []
        self._preview_second = None
        try:
            widget.update()
        except Exception:
            pass
