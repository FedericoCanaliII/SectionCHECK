from PyQt5.QtWidgets import QLineEdit, QShortcut
from PyQt5.QtGui import QPainter, QColor, QKeySequence
from PyQt5.QtCore import Qt, QRect
import math

# OpenGL import opzionale (se non disponibile draw_gl è no-op)
try:
    from OpenGL.GL import *
    _HAS_GL = True
except Exception:
    _HAS_GL = False


class BarTool:
    class CoordLineEdit(QLineEdit):
        """Sottoclasse per intercettare focusOutEvent e garantire il callback."""
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
                 diameter: float = 16.0,
                 line_width: float = 2.0,
                 line_color: QColor = QColor(255, 0, 0, 255),
                 fill_color: QColor = QColor(255, 0, 0, 120),
                 preview_line_color: QColor = QColor(200, 0, 0, 200),
                 preview_fill_color: QColor = QColor(200, 0, 0, 120),
                 snap_to_grid: bool = False,
                 grid_spacing: float = 1.0):
        # draft / storage
        self.draft_bars = []         # [{'center':(x,y), 'diam':d, 'name': 'B1'}, ...]
        self._internal_confirmed = []  # internal storage if external not provided
        self._external_confirmed_ref = None
        self._counter = 0

        self._active_editor = None
        self._editing_index = None  # (idx, was_confirmed_bool)

        # guard per evitare rientri
        self._finalizing = False

        # stili
        self.default_diameter = float(diameter)
        self.line_width = float(line_width)
        self.line_color = line_color
        self.fill_color = fill_color
        self.preview_line_color = preview_line_color
        self.preview_fill_color = preview_fill_color

        # snap/grid
        self.snap_to_grid = bool(snap_to_grid)
        self.grid_spacing = float(grid_spacing)

        # GL precalc
        def _qcolor_to_gl(c: QColor):
            try:
                return c.getRgbF()
            except Exception:
                return (1.0, 1.0, 1.0, 1.0)
        self._gl_fill_color = _qcolor_to_gl(self.fill_color)
        self._gl_line_color = _qcolor_to_gl(self.line_color)

        # shortcuts
        self._shortcuts = []

    # --------- external storage API ----------
    def set_confirmed_list(self, list_ref):
        """Collega la lista esterna (pass-by-ref). Se None si usa lo storage interno."""
        if list_ref is None:
            self._external_confirmed_ref = None
        else:
            self._external_confirmed_ref = list_ref

    def _get_confirmed_list(self):
        return self._external_confirmed_ref if self._external_confirmed_ref is not None else self._internal_confirmed

    # draft persistence helpers
    def get_draft(self):
        return list(self.draft_bars)

    def set_draft(self, draft):
        try:
            self.draft_bars = list(draft) if draft is not None else []
        except Exception:
            self.draft_bars = []

    # --------- activation / deactivation ----------
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
            # nessun editor: conferma tutti i draft
            if len(self.draft_bars) > 0:
                self._confirm_all_drafts(widget)
                widget.update()
        except Exception:
            pass

    # --------- input events ----------
    def on_mouse_press(self, widget, event):
        if event.button() != Qt.LeftButton:
            return False
        px, py = event.x(), event.y()

        # se editor aperto
        if self._active_editor is not None:
            ed_geom = self._active_editor.geometry()
            if ed_geom.contains(px, py):
                return False
            else:
                self._finalize_editor(widget)
                widget.update()
                event.accept()
                return True

        # controllo click su label (draft + confirmed)
        fm = widget.fontMetrics()
        confirmed = self._get_confirmed_list()
        combined = self.draft_bars + confirmed
        for idx, bar in enumerate(combined):
            cx, cy = bar['center']
            sx, sy = widget.world_to_screen(cx, cy)
            name = bar.get('name', '')
            diam = bar.get('diam', self.default_diameter)
            label = f"{name} ({cx:.3f}, {cy:.3f}) φ: {diam:.3f}"
            tw = fm.horizontalAdvance(label)
            th = fm.height()
            label_rect = QRect(sx + 8, sy - th // 2, tw, th)
            if label_rect.contains(px, py):
                is_confirmed = (idx >= len(self.draft_bars))
                if is_confirmed:
                    real_idx = idx - len(self.draft_bars)
                    self._open_editor_for_bar(widget, real_idx, label_rect, confirmed=True)
                else:
                    self._open_editor_for_bar(widget, idx, label_rect, confirmed=False)
                event.accept()
                return True

        # piazza nuovo bar draft
        wx, wy = widget.screen_to_world(px, py)
        diam = self.default_diameter
        # snap to grid se attivo
        snap = getattr(widget, 'snap_to_grid', self.snap_to_grid)
        if snap:
            spacing = getattr(widget, 'grid_spacing', self.grid_spacing)
            try:
                if spacing is None or float(spacing) == 0.0:
                    spacing = self.grid_spacing
            except Exception:
                spacing = self.grid_spacing
            wx = round(wx / spacing) * spacing
            wy = round(wy / spacing) * spacing

        provisional_index = len(self.draft_bars) + 1
        name = f"B{provisional_index}"
        self.draft_bars.append({'center': (wx, wy), 'diam': diam, 'name': name})
        widget.update()
        event.accept()
        return True

    def on_mouse_move(self, widget, event):
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
                if len(self.draft_bars) > 0:
                    self._confirm_all_drafts(widget)
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
            if len(self.draft_bars) > 0:
                self.draft_bars = []
                widget.update()
                return True
            return False
        return False

    # --------- drawing ----------
    def draw_gl(self, widget):
        if not _HAS_GL:
            return
        confirmed = self._get_confirmed_list()
        for bar in confirmed:
            cx, cy = bar['center']
            diam = bar['diam']
            r = diam / 2.0
            # filled circle approx
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            fr, fg, fb, fa = self._gl_fill_color
            glColor4f(fr, fg, fb, fa)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            steps = 40
            for i in range(steps + 1):
                theta = 2.0 * math.pi * (i / steps)
                x = cx + r * math.cos(theta)
                y = cy + r * math.sin(theta)
                glVertex2f(x, y)
            glEnd()
            # outline
            lr, lg, lb, la = self._gl_line_color
            glColor4f(lr, lg, lb, 1.0)
            glLineWidth(self.line_width)
            glBegin(GL_LINE_LOOP)
            for i in range(steps):
                theta = 2.0 * math.pi * (i / steps)
                x = cx + r * math.cos(theta)
                y = cy + r * math.sin(theta)
                glVertex2f(x, y)
            glEnd()

    def draw_painter(self, widget, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)
        fm = painter.fontMetrics()

        # draw draft bars with preview style
        for idx, bar in enumerate(self.draft_bars):
            cx, cy = bar['center']
            diam = bar['diam']
            sx, sy = widget.world_to_screen(cx, cy)
            sx_r, _ = widget.world_to_screen(cx + diam / 2.0, cy)
            r_screen = abs(sx_r - sx)
            painter.setBrush(self.preview_fill_color)
            pen = painter.pen()
            pen.setWidth(1)
            pen.setStyle(Qt.DashLine)
            pen.setColor(self.preview_line_color)
            painter.setPen(pen)
            painter.drawEllipse(int(sx - r_screen), int(sy - r_screen), int(r_screen * 2), int(r_screen * 2))
            # marker center
            painter.setBrush(QColor(255, 0, 0, 255))
            pen = painter.pen()
            pen.setColor(QColor(0, 0, 0))
            painter.setPen(pen)
            radius = 6
            painter.drawEllipse(sx - radius, sy - radius, radius*2, radius*2)
            # label
            name = bar.get('name', f'B{idx+1}')
            label = f"{name} ({cx}, {cy}) φ: {diam}"
            painter.setPen(QColor(255,255,255))
            tw = fm.horizontalAdvance(label)
            th = fm.height()
            painter.drawText(sx + 8, sy + th // 2 - 2, label)

        # confirmed are drawn by draw_gl only to avoid duplication (but you can draw names here)
        confirmed = self._get_confirmed_list()
        for bar in confirmed:
            cx, cy = bar['center']
            name = bar.get('name', '')
            sx, sy = widget.world_to_screen(cx, cy)
            label = f"{name}"
            painter.setPen(QColor(255,255,255))
            tw = fm.horizontalAdvance(label)
            painter.drawText(sx - tw // 2, sy + fm.height() // 2 - 2, label)

    # --------- editor helpers ----------
    def _open_editor_for_bar(self, widget, bar_index, label_rect: QRect, confirmed: bool = False):
        arr = self._get_confirmed_list() if confirmed else self.draft_bars
        if bar_index >= len(arr):
            return
        if self._active_editor is not None:
            self._finalize_editor(widget)
        cx, cy = arr[bar_index]['center']
        diam = arr[bar_index]['diam']
        text = f"{arr[bar_index].get('name','B?')} ({cx:.6g}, {cy:.6g}) φ: {diam:.6g}"
        editor = BarTool.CoordLineEdit(widget, on_focus_lost=lambda: self._finalize_editor(widget))
        editor.setText(text)
        extra_w = 10
        editor.setGeometry(label_rect.x(), label_rect.y(), label_rect.width() + extra_w, label_rect.height() + 4)
        editor.show()
        editor.setFocus()
        editor.setAttribute(Qt.WA_DeleteOnClose, True)
        self._active_editor = editor
        self._editing_index = (bar_index, confirmed)
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
            idx, was_confirmed = self._editing_index
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

            # parse robusto: (x, y) and diam (φ or last numeric)
            import re
            x_new = y_new = d_new = None
            m = re.search(r'\(([^)]*)\)', txt)
            if m:
                inside = m.group(1)
                parts = re.split(r'[;,]\s*|\s+', inside.strip())
                nums = []
                for p in parts:
                    if not p:
                        continue
                    try:
                        nums.append(float(p))
                    except Exception:
                        try:
                            nums.append(float(p.replace(',', '.')))
                        except Exception:
                            pass
                if len(nums) >= 2:
                    x_new = nums[0]
                    y_new = nums[1]
            m2 = re.search(r'φ\s*[:=]?\s*([-\d\.,eE]+)', txt)
            if m2:
                ds = m2.group(1).strip()
                try:
                    d_new = float(ds)
                except Exception:
                    try:
                        d_new = float(ds.replace(',', '.'))
                    except Exception:
                        d_new = None
            else:
                all_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
                if len(all_nums) >= 1:
                    try:
                        d_new = float(all_nums[-1])
                    except Exception:
                        try:
                            d_new = float(all_nums[-1].replace(',', '.'))
                        except Exception:
                            d_new = None

            arr = self._get_confirmed_list() if was_confirmed else self.draft_bars
            if 0 <= idx < len(arr):
                if (x_new is not None) and (y_new is not None):
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
                    arr[idx]['center'] = (x_new, y_new)
                if d_new is not None:
                    try:
                        d_stepped = round(float(d_new))
                    except Exception:
                        try:
                            d_stepped = round(float(str(d_new).replace(',', '.')))
                        except Exception:
                            d_stepped = None
                    if d_stepped is not None:
                        arr[idx]['diam'] = max(0.0001, d_stepped)
        finally:
            try:
                widget.update()
                widget.setFocus()
            except Exception:
                pass
            self._finalizing = False

    # --------- confirm ----------
    def _confirm_all_drafts(self, widget):
        if len(self.draft_bars) == 0:
            return
        confirmed = self._get_confirmed_list()
        for b in self.draft_bars:
            self._counter += 1
            name = f"B{self._counter}"
            entry = {'center': b['center'], 'diam': b['diam'], 'name': name}
            confirmed.append(entry)
        self.draft_bars = []
        try:
            widget.update()
        except Exception:
            pass
