from PyQt5.QtWidgets import QLineEdit, QMessageBox, QShortcut
from PyQt5.QtGui import QPainter, QColor, QKeySequence
from PyQt5.QtCore import Qt, QRect, QPoint
import numpy as np

# OpenGL import opzionale
try:
    from OpenGL.GL import *
    _HAS_GL = True
except Exception:
    _HAS_GL = False


class PolygonTool:
    class CoordLineEdit(QLineEdit):
        """Sottoclasse per intercettare focusOutEvent e garantire la chiamata
        al callback di finalizzazione quando l'utente clicca fuori."""
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
                 fill_color: QColor = QColor(255, 255, 255, 26),
                 preview_line_color: QColor = QColor(150, 200, 255, 200),
                 preview_fill_color: QColor = QColor(135, 206, 235, 120),
                 snap_to_grid: bool = False,
                 grid_spacing: float = 1.0):
        """
        PolygonTool compatibile con SectionManager:
         - set_confirmed_list(list_ref) per collegare storage esterno
         - get_draft/set_draft per persistere draft per-sezione (opzionale)
        """
        # draft internals
        self.draft_vertices = []          # [(x,y), ...]
        self._internal_confirmed = []     # storage interna
        self._external_confirmed_ref = None
        self._counter = 0

        self._active_editor = None
        self._editing_index = None
        self._preview_vertex = None
        self._polygon_closed = False
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

        # shortcuts
        self._shortcuts = []

    # ---------- external storage API ----------
    def set_confirmed_list(self, list_ref):
        """Collega la lista esterna (pass-by-ref). Se None usa storage interno."""
        if list_ref is None:
            self._external_confirmed_ref = None
        else:
            self._external_confirmed_ref = list_ref

    def _get_confirmed_list(self):
        return self._external_confirmed_ref if self._external_confirmed_ref is not None else self._internal_confirmed

    # draft persistence helpers
    def get_draft(self):
        return list(self.draft_vertices)

    def set_draft(self, draft):
        try:
            self.draft_vertices = list(draft) if draft is not None else []
        except Exception:
            self.draft_vertices = []

    # ---------- activation / deactivation ----------
    def on_activate(self, widget):
        try:
            widget.setCursor(Qt.CrossCursor)
        except Exception:
            pass
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

        # se ci sono vertici aperti e non chiusi, prova a confermare se >=3
        if len(self.draft_vertices) >= 3 and not self._polygon_closed:
            try:
                self._confirm_current_polygon(widget)
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
            fw = None
            try:
                fw = widget.focusWidget()
            except Exception:
                fw = None

            # se il focus è su un QLineEdit e non è il nostro editor -> non interferire
            if isinstance(fw, QLineEdit):
                if fw is self._active_editor:
                    self._finalize_editor(widget)
                    try:
                        widget.setFocus()
                    except Exception:
                        pass
                    widget.update()
                return

            # se c'è editor finalizzalo
            if self._active_editor is not None:
                self._finalize_editor(widget)
                try:
                    widget.setFocus()
                except Exception:
                    pass
                widget.update()
                return

            # gestione chiusura/conferma poligono
            if not self._polygon_closed and len(self.draft_vertices) >= 3:
                self._polygon_closed = True
                widget.update()
                return
            elif self._polygon_closed:
                self._confirm_current_polygon(widget)
                widget.update()
                return
        except Exception:
            pass

    # ---------- input events ----------
    def on_mouse_press(self, widget, event):
        if event.button() == Qt.LeftButton:
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

            # click su vertice per aprire editor
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

            # aggiungi vertice se poligono non chiuso
            if not self._polygon_closed:
                wx, wy = widget.screen_to_world(px, py)
                self.draft_vertices.append((wx, wy))
                self._preview_vertex = None
                widget.update()
                event.accept()
                return True

        elif event.button() == Qt.RightButton and not self._polygon_closed:
            # click destro: se >=3 chiudi, se <3 pop, altrimenti ignora
            if len(self.draft_vertices) >= 3:
                self._polygon_closed = True
                widget.update()
                event.accept()
                return True
            elif len(self.draft_vertices) > 0:
                self.draft_vertices.pop()
                widget.update()
                event.accept()
                return True

        return False

    def on_mouse_move(self, widget, event):
        if not self._polygon_closed and len(self.draft_vertices) > 0:
            px, py = event.x(), event.y()
            wx, wy = widget.screen_to_world(px, py)
            self._preview_vertex = (wx, wy)
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
                if not self._polygon_closed and len(self.draft_vertices) >= 3:
                    self._polygon_closed = True
                    widget.update()
                    return True
                elif self._polygon_closed:
                    self._confirm_current_polygon(widget)
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
                widget.update()
                return True
            elif self._polygon_closed:
                self._polygon_closed = False
                widget.update()
                return True
            elif len(self.draft_vertices) > 0:
                self.draft_vertices = []
                self._preview_vertex = None
                widget.update()
                return True
            return False

        return False

    # ---------- drawing OpenGL ----------
    def draw_gl(self, widget):
        if not _HAS_GL:
            return
        confirmed = self._get_confirmed_list()
        for poly in confirmed:
            vertices = poly.get('vertices', [])
            if len(vertices) < 3:
                continue
            # triangolazione
            tris = self._triangulate_polygon(vertices)
            if not tris and len(vertices) == 3:
                tris = [(vertices[0], vertices[1], vertices[2])]
            if tris:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                fr, fg, fb, fa = self._gl_fill_color
                glColor4f(fr, fg, fb, fa)
                glBegin(GL_TRIANGLES)
                for tri in tris:
                    for x, y in tri:
                        glVertex2f(x, y)
                glEnd()
            lr, lg, lb, la = self._gl_line_color
            glColor4f(lr, lg, lb, 1.0)
            glLineWidth(self.line_width)
            glBegin(GL_LINE_LOOP)
            for x, y in vertices:
                glVertex2f(x, y)
            glEnd()

    # ---------- drawing QPainter ----------
    def draw_painter(self, widget, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)
        fm = painter.fontMetrics()

        # disegno draft/poligono in costruzione
        if len(self.draft_vertices) > 0:
            screen_vertices = [QPoint(*widget.world_to_screen(x, y)) for x, y in self.draft_vertices]

            pen = painter.pen()
            pen.setWidth(1)

            if self._polygon_closed:
                pen.setColor(self.preview_line_color)
                pen.setStyle(Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(self.preview_fill_color)
                # drawPolygon accetta QList<QPoint> o iterable di QPoint
                painter.drawPolygon(*screen_vertices)
            else:
                pen.setColor(self.preview_line_color)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                if len(screen_vertices) > 1:
                    for i in range(len(screen_vertices) - 1):
                        painter.drawLine(screen_vertices[i], screen_vertices[i + 1])
                if self._preview_vertex is not None:
                    preview_point = QPoint(*widget.world_to_screen(*self._preview_vertex))
                    if len(screen_vertices) > 0:
                        painter.drawLine(screen_vertices[-1], preview_point)
                    if len(screen_vertices) >= 3:
                        painter.drawLine(preview_point, screen_vertices[0])

            # marker e label per i vertici
            for idx, (wx, wy) in enumerate(self.draft_vertices):
                sx, sy = widget.world_to_screen(wx, wy)
                painter.setBrush(QColor(150, 220, 255, 255))
                pen = painter.pen()
                pen.setColor(QColor(40, 100, 140))
                painter.setPen(pen)
                radius = 6
                painter.drawEllipse(sx - radius, sy - radius, radius*2, radius*2)
                lbl = f"({wx:.2f}, {wy:.2f})"
                tw = fm.horizontalAdvance(lbl)
                th = fm.height()
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(sx + 8, sy + th // 2 - 2, lbl)

        # disegna i nomi dei poligoni confermati
        painter.setPen(QColor(255, 255, 255))
        confirmed = self._get_confirmed_list()
        for poly in confirmed:
            vertices = poly.get('vertices', [])
            if not vertices:
                continue
            cx = sum(v[0] for v in vertices) / len(vertices)
            cy = sum(v[1] for v in vertices) / len(vertices)
            sx, sy = widget.world_to_screen(cx, cy)
            name = poly.get('name', '')
            tw = fm.horizontalAdvance(name)
            painter.drawText(sx - tw // 2, sy + fm.height() // 2, name)

    # ---------- editor helpers ----------
    def _open_editor_for_vertex(self, widget, vertex_index, label_rect: QRect):
        if vertex_index >= len(self.draft_vertices):
            return
        if self._active_editor is not None:
            self._finalize_editor(widget)
        wx, wy = self.draft_vertices[vertex_index]
        text = f"{wx:.6g}, {wy:.6g}"
        editor = PolygonTool.CoordLineEdit(widget, on_focus_lost=lambda: self._finalize_editor(widget))
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

    # ---------- confirm ----------
    def _confirm_current_polygon(self, widget):
        if len(self.draft_vertices) < 3:
            try:
                QMessageBox.warning(widget, "Poligono non valido", "Un poligono deve avere almeno 3 vertici.")
            except Exception:
                pass
            return
        # controllo autointersezione
        if self._is_self_intersecting(self.draft_vertices):
            try:
                QMessageBox.warning(widget, "Poligono non valido", "Il poligono non può avere lati che si intersecano.")
            except Exception:
                pass
            return
        self._counter += 1
        name = f"POLIGONO {self._counter}"
        poly_entry = {'vertices': self.draft_vertices.copy(), 'name': name}
        self._get_confirmed_list().append(poly_entry)
        # reset stato
        self.draft_vertices = []
        self._preview_vertex = None
        self._polygon_closed = False
        try:
            widget.update()
        except Exception:
            pass

    # ---------- geometry helpers ----------
    def _is_self_intersecting(self, vertices):
        n = len(vertices)
        if n < 4:
            return False
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                p1, p2 = vertices[i], vertices[(i + 1) % n]
                p3, p4 = vertices[j], vertices[(j + 1) % n]
                if self._segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def _segments_intersect(self, a, b, c, d):
        def orient(p, q, r):
            return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
        def on_segment(p, q, r):
            return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))
        p1, p2, p3, p4 = a, b, c, d
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True
        return (o1 * o2 < 0) and (o3 * o4 < 0)

    def _polygon_area(self, vertices):
        a = 0.0
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            a += x1 * y2 - x2 * y1
        return a * 0.5

    def _is_convex(self, prev, curr, nextp):
        return ((curr[0] - prev[0]) * (nextp[1] - curr[1]) -
                (curr[1] - prev[1]) * (nextp[0] - curr[0])) > 0

    def _point_in_triangle(self, pt, a, b, c):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(pt, a, b)
        d2 = sign(pt, b, c)
        d3 = sign(pt, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def _triangulate_polygon(self, vertices):
        verts = vertices.copy()
        n = len(verts)
        if n < 3:
            return []
        # pulizia duplicati consecutivi
        cleaned = []
        for v in verts:
            if not cleaned or (abs(cleaned[-1][0] - v[0]) > 1e-12 or abs(cleaned[-1][1] - v[1]) > 1e-12):
                cleaned.append(v)
        verts = cleaned
        n = len(verts)
        if n < 3:
            return []
        # assicurati CCW
        if self._polygon_area(verts) < 0:
            verts.reverse()
        indices = list(range(len(verts)))
        triangles = []
        guard = 0
        while len(indices) > 3 and guard < 10000:
            guard += 1
            ear_found = False
            m = len(indices)
            for i in range(m):
                i_prev = indices[(i - 1) % m]
                i_curr = indices[i]
                i_next = indices[(i + 1) % m]
                p_prev = verts[i_prev]
                p_curr = verts[i_curr]
                p_next = verts[i_next]
                if not self._is_convex(p_prev, p_curr, p_next):
                    continue
                is_ear = True
                for j in indices:
                    if j in (i_prev, i_curr, i_next):
                        continue
                    if self._point_in_triangle(verts[j], p_prev, p_curr, p_next):
                        is_ear = False
                        break
                if not is_ear:
                    continue
                triangles.append((p_prev, p_curr, p_next))
                indices.pop(i)
                ear_found = True
                break
            if not ear_found:
                break
        if len(indices) == 3:
            a, b, c = indices
            triangles.append((verts[a], verts[b], verts[c]))
        return triangles
