from PyQt5.QtWidgets import QLineEdit, QShortcut
from PyQt5.QtGui import QPainter, QColor, QKeySequence
from PyQt5.QtCore import Qt, QRect, QPointF
import math
import re

# OpenGL import opzionale (se non presente draw_gl è no-op)
try:
    from OpenGL.GL import *
    _HAS_GL = True
except Exception:
    _HAS_GL = False


class StaffaTool:
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
                 default_diam: float = 8.0,
                 thin_preview_mm: float = 2.0,
                 line_color: QColor = QColor(255, 255, 0, 100),
                 preview_line_color: QColor = QColor(200, 200, 100, 100),
                 snap_to_grid: bool = False,
                 grid_spacing: float = 1.0,
                 merge_vertex_px: int = 6):
        """
        default_diam : diametro di default (unità world)
        thin_preview_mm : spessore fisso (mm) della linea di preview indipendente dallo zoom
        merge_vertex_px : soglia in pixel per considerare vertici sovrapposti
        """
        # draft / storage
        # draft_staffe: [{'points':[(x,y),...], 'fixed':[bool,...], 'diam':d, 'name': 'S1', 'in_progress': True}]
        self.draft_staffe = []
        self._internal_confirmed = []  # internal storage if external not provided
        self._external_confirmed_ref = None
        self._counter = 0

        # editor state
        self._active_editor = None
        # struttura _editing_info: dict {'type': 'vertex'|'general', 'staffa_idx': int, 'vertex_idx': int or None, 'confirmed': bool}
        self._editing_info = None

        # guard
        self._finalizing = False

        # styles
        self.default_diam = float(default_diam)
        self.line_color = line_color
        self.preview_line_color = preview_line_color
        self.thin_preview_mm = float(thin_preview_mm)

        # snap/grid
        self.snap_to_grid = bool(snap_to_grid)
        self.grid_spacing = float(grid_spacing)

        # merge threshold
        self.merge_vertex_px = int(merge_vertex_px)

        # GL precalc
        def _qcolor_to_gl(c: QColor):
            try:
                return c.getRgbF()
            except Exception:
                return (1.0, 1.0, 1.0, 1.0)
        self._gl_line_color = _qcolor_to_gl(self.line_color)

        # shortcuts
        self._shortcuts = []

    # ---------------- API esterna ----------------
    def set_confirmed_list(self, list_ref):
        if list_ref is None:
            self._external_confirmed_ref = None
        else:
            self._external_confirmed_ref = list_ref

    def _get_confirmed_list(self):
        return self._external_confirmed_ref if self._external_confirmed_ref is not None else self._internal_confirmed

    def get_draft(self):
        return list(self.draft_staffe)

    def set_draft(self, draft):
        try:
            self.draft_staffe = list(draft) if draft is not None else []
        except Exception:
            self.draft_staffe = []

    # ---------------- attivazione / deattivazione ----------------
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
            # se editor aperto -> finalize editor (invio sull'editor)
            if self._active_editor is not None:
                self._finalize_editor(widget)
                try:
                    widget.setFocus()
                except Exception:
                    pass
                widget.update()
                return

            # 1) se esiste draft in progress => primo invio: conferma draft e mostra etichette vertici
            if len(self.draft_staffe) > 0:
                self._confirm_active_draft(widget)
                widget.update()
                return

            # 2) se non ci sono draft, trova ultima confermata in 'vertices_shown' o 'preview_thick' e gestisci stati:
            #    - vertices_shown -> preview_thick (dashed ma con spessore reale, etichette ancora visibili)
            #    - preview_thick -> final (linea solida, etichetta finale mostra solo il nome)
            confirmed = self._get_confirmed_list()
            idx_to_act = None
            for i in range(len(confirmed)-1, -1, -1):
                st = confirmed[i].get('state')
                if st in ('vertices_shown', 'preview_thick'):
                    idx_to_act = i
                    break
            if idx_to_act is not None:
                cur_state = confirmed[idx_to_act].get('state')
                if cur_state == 'vertices_shown':
                    confirmed[idx_to_act]['state'] = 'preview_thick'
                    # non aprire editor automaticamente
                    widget.update()
                    return
                elif cur_state == 'preview_thick':
                    confirmed[idx_to_act]['state'] = 'final'
                    # quando diventa final vogliamo che l'etichetta venga mostrata solo con il nome:
                    # (la logica di rendering mostra solo il nome per lo stato 'final')
                    widget.update()
                    return

            # altrimenti non c'è nulla da fare
        except Exception:
            pass

    # ---------------- eventi input ----------------
    def on_mouse_press(self, widget, event):
        if event.button() != Qt.LeftButton:
            return False
        px, py = event.x(), event.y()

        # se editor aperto e click dentro editor -> ignore
        if self._active_editor is not None:
            ed_geom = self._active_editor.geometry()
            if ed_geom.contains(px, py):
                return False
            else:
                # finalize editor su click esterno
                self._finalize_editor(widget)
                widget.update()
                event.accept()
                return True

        # controllo click su etichette dei vertici (solo staffe confermate con state 'vertices_shown' o 'preview_thick')
        fm = widget.fontMetrics()
        confirmed = self._get_confirmed_list()
        for s_idx, staffa in enumerate(confirmed):
            if staffa.get('state') not in ('vertices_shown', 'preview_thick'):
                continue
            pts = staffa.get('points', [])
            for v_idx, (vx, vy) in enumerate(pts):
                sx_label, sy_label = self._vertex_label_screen_pos(widget, staffa, v_idx)
                label = f"({vx:.6g}, {vy:.6g})"
                tw = fm.horizontalAdvance(label)
                th = fm.height()
                rect = QRect(int(sx_label - tw // 2 - 3), int(sy_label - th // 2 - 2), tw + 6, th + 4)
                if rect.contains(px, py):
                    # apro editor per quel vertice
                    self._open_editor_for_vertex(widget, s_idx, v_idx, rect)
                    event.accept()
                    return True

        # controllo click su etichetta generale (solo staffe in 'final' o in 'preview_thick'/'vertices_shown' se si vuole editare)
        for idx, staffa in enumerate(confirmed):
            # consentiamo apertura generale anche in preview_thick per modificare name/diam se desiderato
            if staffa.get('state') not in ('final', 'preview_thick', 'vertices_shown'):
                continue
            labx, laby = self._compute_label_screen_pos(widget, staffa)
            name = staffa.get('name', '')
            # testo mostrato per rilevamento click: se final -> solo name, altrimenti name+φ
            if staffa.get('state') == 'final':
                label_text = f"{name}"
            else:
                label_text = f"{name}  φ: {staffa.get('diam', self.default_diam)}"
            fm = widget.fontMetrics()
            tw = fm.horizontalAdvance(label_text)
            th = fm.height()
            label_rect = QRect(int(labx - tw // 2), int(laby - th // 2), tw + 6, th + 4)
            if label_rect.contains(px, py):
                # open general editor on demand
                confirmed_flag = True
                self._open_editor_for_staffa(widget, idx, label_rect, confirmed=confirmed_flag)
                event.accept()
                return True

        # altrimenti gestisco click per add/fissa vertice su draft in progress o avvio nuovo draft
        wx, wy = widget.screen_to_world(px, py)
        # snap
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

        # se draft in corso: fissa ultimo provvisorio e aggiungi nuovo provvisorio
        if len(self.draft_staffe) > 0 and self.draft_staffe[-1].get('in_progress', False):
            last = self.draft_staffe[-1]
            pts = last['points']
            fixed = last['fixed']
            if len(pts) >= 1:
                pts[-1] = (wx, wy)
                fixed[-1] = True
                pts.append((wx, wy))
                fixed.append(False)
            else:
                pts.extend([(wx, wy), (wx, wy)])
                fixed.extend([True, False])
            widget.update()
            event.accept()
            return True

        # altrimenti inizio nuova staffa in preview
        provisional_index = len(self.draft_staffe) + 1
        name = f"S{provisional_index}"
        new = {'points': [(wx, wy), (wx, wy)], 'fixed': [True, False],
               'diam': self.default_diam, 'name': name, 'in_progress': True}
        self.draft_staffe.append(new)
        widget.update()
        event.accept()
        return True

    def on_mouse_move(self, widget, event):
        # aggiorna l'ultimo punto provvisorio della draft
        if len(self.draft_staffe) > 0 and self.draft_staffe[-1].get('in_progress', False):
            px, py = event.x(), event.y()
            wx, wy = widget.screen_to_world(px, py)
            # snap
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
            try:
                last = self.draft_staffe[-1]
                if len(last['points']) >= 1:
                    last['points'][-1] = (wx, wy)
                else:
                    last['points'] = [(wx, wy)]
            except Exception:
                pass
            try:
                widget.update()
            except Exception:
                pass
            return True
        return False

    def on_mouse_release(self, widget, event):
        return False

    def on_key_press(self, widget, event):
        # gestito dalla _on_enter_pressed via shortcut, ma manteniamo comportamento per keyPressEvent
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            return self._on_enter_pressed(widget)
        if key == Qt.Key_Escape:
            if self._active_editor is not None:
                try:
                    self._active_editor.deleteLater()
                except Exception:
                    pass
                self._active_editor = None
                self._editing_info = None
            if len(self.draft_staffe) > 0:
                if self.draft_staffe[-1].get('in_progress', False):
                    self.draft_staffe.pop(-1)
                else:
                    self.draft_staffe = []
                widget.update()
                return True
            return False
        return False

    # ---------------- disegno ----------------
    def _mm_to_pixels(self, widget, mm):
        """Converte mm in pixel usando DPI logico (fallback 96)."""
        try:
            dpi = widget.logicalDpiX()
            if dpi is None or dpi <= 0:
                dpi = 96.0
        except Exception:
            dpi = 96.0
        return max(1.0, (mm * dpi) / 25.4)

    def draw_gl(self, widget):
        if not _HAS_GL:
            return
        confirmed = self._get_confirmed_list()
        for staffa in confirmed:
            pts = staffa.get('points', [])
            if len(pts) < 2:
                continue
            diam = staffa.get('diam', self.default_diam)
            state = staffa.get('state', 'final')
            lr, lg, lb, la = self._gl_line_color
            glColor4f(lr, lg, lb, 1.0)
            if state == 'final':
                try:
                    glLineWidth(max(1.0, float(diam)))
                except Exception:
                    glLineWidth(1.0)
            elif state == 'preview_thick':
                try:
                    glLineWidth(max(1.0, float(diam)))
                except Exception:
                    glLineWidth(1.0)
            else:
                glLineWidth(1.0)
            glBegin(GL_LINE_STRIP)
            for (x, y) in pts:
                glVertex2f(x, y)
            glEnd()

    def draw_painter(self, widget, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)
        fm = painter.fontMetrics()

        # 1) draft (in progress) drawn as thin constant (2 mm) dashed, senza etichette
        for staffa in self.draft_staffe:
            pts = staffa.get('points', [])
            if len(pts) < 2:
                continue
            pen_w = self._mm_to_pixels(widget, self.thin_preview_mm)
            pen = painter.pen()
            pen.setWidthF(pen_w)
            pen.setStyle(Qt.DashLine)
            pen.setColor(self.preview_line_color)
            painter.setPen(pen)
            from PyQt5.QtGui import QPolygonF
            poly = QPolygonF()
            for (wx, wy) in pts:
                sx, sy = widget.world_to_screen(wx, wy)
                poly.append(QPointF(sx, sy))
            painter.drawPolyline(poly)
            # draw small vertex markers
            painter.setBrush(QColor(255, 255, 0, 255))
            pen2 = painter.pen()
            pen2.setStyle(Qt.SolidLine)
            pen2.setWidth(1)
            pen2.setColor(QColor(0, 0, 0))
            painter.setPen(pen2)
            for (wx, wy) in pts:
                sx, sy = widget.world_to_screen(wx, wy)
                r = 3
                painter.drawEllipse(int(sx - r), int(sy - r), r * 2, r * 2)

        # 2) confirmed items
        confirmed = self._get_confirmed_list()
        for s_idx, staffa in enumerate(confirmed):
            pts = staffa.get('points', [])
            if len(pts) < 2:
                continue
            state = staffa.get('state', 'final')
            diam = staffa.get('diam', self.default_diam)

            # decide style based on state
            if state == 'final':
                # final: use real diameter converted from world to screen and draw SOLID (spesso)
                try:
                    sx0, _ = widget.world_to_screen(0.0, 0.0)
                    sx1, _ = widget.world_to_screen(diam, 0.0)
                    pen_w = abs(sx1 - sx0)
                    if pen_w < 1:
                        pen_w = 1.0
                except Exception:
                    pen_w = 2.0
                pen = painter.pen()
                pen.setWidthF(pen_w)
                pen.setStyle(Qt.SolidLine)
                pen.setColor(self.line_color)
                painter.setPen(pen)
            elif state == 'preview_thick':
                # dashed but with actual thickness (diam)
                try:
                    sx0, _ = widget.world_to_screen(0.0, 0.0)
                    sx1, _ = widget.world_to_screen(diam, 0.0)
                    pen_w = abs(sx1 - sx0)
                    if pen_w < 1:
                        pen_w = 1.0
                except Exception:
                    pen_w = 2.0
                pen = painter.pen()
                pen.setWidthF(pen_w)
                pen.setStyle(Qt.DashLine)
                pen.setColor(self.preview_line_color)
                painter.setPen(pen)
            else:
                # vertices_shown (o altri stati non-final): thin dashed preview (same as draft)
                pen_w = self._mm_to_pixels(widget, self.thin_preview_mm)
                pen = painter.pen()
                pen.setWidthF(pen_w)
                pen.setStyle(Qt.DashLine)
                pen.setColor(self.preview_line_color)
                painter.setPen(pen)

            # draw polyline
            from PyQt5.QtGui import QPolygonF
            poly = QPolygonF()
            for (wx, wy) in pts:
                sx, sy = widget.world_to_screen(wx, wy)
                poly.append(QPointF(sx, sy))
            painter.drawPolyline(poly)

            # draw vertex labels if state == 'vertices_shown' or 'preview_thick' (manteniamo visibili i vertici fino al final)
            if state in ('vertices_shown', 'preview_thick'):
                painter.setPen(QColor(255, 255, 255))
                painter.setBrush(QColor(255, 255, 0, 255))
                for v_idx, (vx, vy) in enumerate(pts):
                    sx_label, sy_label = self._vertex_label_screen_pos(widget, staffa, v_idx)
                    # small marker at vertex (center)
                    sx_v, sy_v = widget.world_to_screen(vx, vy)
                    r = 3
                    painter.setPen(QColor(0, 0, 0))
                    painter.setBrush(QColor(255, 255, 0, 255))
                    painter.drawEllipse(int(sx_v - r), int(sy_v - r), r*2, r*2)
                    # label a qualche pixel esterno
                    label = f"({vx:.6g}, {vy:.6g})"
                    painter.setPen(QColor(255, 255, 255))
                    tw = fm.horizontalAdvance(label)
                    th = fm.height()
                    painter.drawText(int(sx_label - tw // 2), int(sy_label + th // 2 - 2), label)

            # draw general label:
            # - if state == 'final' -> show only name
            # - else -> show name  φ: diam
            labx, laby = self._compute_label_screen_pos(widget, staffa)
            if state == 'final':
                label_text = f"{staffa.get('name','S?')}"
            else:
                label_text = f"{staffa.get('name','S?')}  φ: {staffa.get('diam', self.default_diam)}"
            painter.setPen(QColor(255, 255, 255))
            tw = fm.horizontalAdvance(label_text)
            painter.drawText(int(labx - tw // 2), int(laby + fm.height() // 2 - 2), label_text)

    # ---------------- utilità etichette ----------------
    def _compute_label_screen_pos(self, widget, staffa):
        """
        Etichetta generale posizionata a metà del primo segmento, verso l'esterno.
        """
        pts = staffa.get('points', [])
        if len(pts) >= 2:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            midx = 0.5 * (x0 + x1)
            midy = 0.5 * (y0 + y1)
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len == 0:
                perp_x, perp_y = 0.0, -1.0
            else:
                perp_x = -dy / seg_len
                perp_y = dx / seg_len
            diam = staffa.get('diam', self.default_diam)
            offs = max(diam * 0.5, 0.1 * seg_len)
            labx = midx + perp_x * offs
            laby = midy + perp_y * offs
            return widget.world_to_screen(labx, laby)
        elif len(pts) == 1:
            return widget.world_to_screen(pts[0][0], pts[0][1])
        else:
            return 0, 0

    def _vertex_label_screen_pos(self, widget, staffa, v_idx, offset_px: int = 10):
        """
        Posiziona l'etichetta di un vertice "un po' esternamente".
        Logica: calcola il centro medio (centroid) in pixel della staffa e sposta l'etichetta del vertice
        in direzione dal centro verso il vertice per offset_px pixel.
        """
        pts = staffa.get('points', [])
        if v_idx < 0 or v_idx >= len(pts):
            return 0, 0
        # centroid in screen coords
        sx_sum = 0.0
        sy_sum = 0.0
        n = 0
        for (wx, wy) in pts:
            sx, sy = widget.world_to_screen(wx, wy)
            sx_sum += sx
            sy_sum += sy
            n += 1
        if n == 0:
            return 0, 0
        cx = sx_sum / n
        cy = sy_sum / n
        vx_w, vy_w = pts[v_idx]
        vx, vy = widget.world_to_screen(vx_w, vy_w)
        dx = vx - cx
        dy = vy - cy
        d = math.hypot(dx, dy)
        if d == 0:
            ux, uy = 0.0, -1.0
        else:
            ux, uy = dx / d, dy / d
        sx_label = vx + ux * offset_px
        sy_label = vy + uy * offset_px
        return sx_label, sy_label

    # ---------------- editor per vertice / generale ----------------
    def _open_editor_for_vertex(self, widget, staffa_idx, vertex_idx, label_rect: QRect):
        confirmed = self._get_confirmed_list()
        if staffa_idx < 0 or staffa_idx >= len(confirmed):
            return
        staffa = confirmed[staffa_idx]
        pts = staffa.get('points', [])
        if vertex_idx < 0 or vertex_idx >= len(pts):
            return
        x, y = pts[vertex_idx]
        text = f"({x:.6g}, {y:.6g})"
        # chiudi eventuale editor aperto
        if self._active_editor is not None:
            self._finalize_editor(widget)
        editor = StaffaTool.CoordLineEdit(widget, on_focus_lost=lambda: self._finalize_editor(widget))
        editor.setText(text)
        extra_w = 10
        editor.setGeometry(label_rect.x(), label_rect.y(), label_rect.width() + extra_w, label_rect.height() + 4)
        editor.show()
        editor.setFocus()
        editor.setAttribute(Qt.WA_DeleteOnClose, True)
        self._active_editor = editor
        self._editing_info = {'type': 'vertex', 'staffa_idx': staffa_idx, 'vertex_idx': vertex_idx, 'confirmed': True}
        def _on_return():
            self._finalize_editor(widget)
            try:
                widget.setFocus()
            except Exception:
                pass
        editor.returnPressed.connect(_on_return)
        editor.editingFinished.connect(lambda: self._finalize_editor(widget))

    def _open_editor_for_staffa(self, widget, staffa_index, label_rect: QRect, confirmed: bool = False):
        arr = self._get_confirmed_list() if confirmed else self.draft_staffe
        if staffa_index >= len(arr):
            return
        if self._active_editor is not None:
            self._finalize_editor(widget)
        staffa = arr[staffa_index]
        diam = staffa.get('diam', self.default_diam)
        text = f"{staffa.get('name','S?')}  φ: {diam}"
        editor = StaffaTool.CoordLineEdit(widget, on_focus_lost=lambda: self._finalize_editor(widget))
        editor.setText(text)
        extra_w = 10
        editor.setGeometry(label_rect.x(), label_rect.y(), label_rect.width() + extra_w, label_rect.height() + 4)
        editor.show()
        editor.setFocus()
        editor.setAttribute(Qt.WA_DeleteOnClose, True)
        self._active_editor = editor
        self._editing_info = {'type': 'general', 'staffa_idx': staffa_index, 'vertex_idx': None, 'confirmed': confirmed}
        def _on_return():
            self._finalize_editor(widget)
            try:
                widget.setFocus()
            except Exception:
                pass
        editor.returnPressed.connect(_on_return)
        editor.editingFinished.connect(lambda: self._finalize_editor(widget))

    def _finalize_editor(self, widget):
        """
        Finalizza sia l'editor del vertice che quello generale.
        - vertex: aggiorna la coordinate del vertice.
        - general: aggiorna name e diam; non cambia automaticamente lo stato (lo fai tu con ENTER).
        """
        if self._finalizing:
            return
        self._finalizing = True
        try:
            if self._active_editor is None or self._editing_info is None:
                return
            editor = self._active_editor
            info = self._editing_info
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
            self._editing_info = None

            if info['type'] == 'vertex':
                # parsing delle coordinate: può essere "(x, y)" oppure "x y" o "x,y"
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
                if len(nums) >= 2:
                    try:
                        x_new = float(nums[0])
                        y_new = float(nums[1])
                    except Exception:
                        try:
                            x_new = float(nums[0].replace(',', '.'))
                            y_new = float(nums[1].replace(',', '.'))
                        except Exception:
                            x_new = None
                            y_new = None
                    if x_new is not None and y_new is not None:
                        # applica snap se attivo
                        confirmed = self._get_confirmed_list()
                        sidx = info['staffa_idx']
                        if 0 <= sidx < len(confirmed):
                            staffa = confirmed[sidx]
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
                            staffa['points'][info['vertex_idx']] = (x_new, y_new)
                            # dopo spostamento dei vertici potremmo voler rimuovere duplicati
                            staffa['points'] = self._merge_close_vertices_widget(widget, staffa['points'])
            elif info['type'] == 'general':
                # aggiorna name e diam (ma non cambia stato)
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
                name_new = None
                mname = re.match(r'^\s*([A-Za-z]\w*\d*|\w+)', txt)
                if mname:
                    name_new = mname.group(1)
                diam_new = None
                mphi = re.search(r'φ\s*[:=]?\s*([-\d\.,eE]+)', txt)
                if not mphi:
                    mphi = re.search(r'(?:diam|dia|d|phi)\s*[:=]?\s*([-\d\.,eE]+)', txt, flags=re.IGNORECASE)
                if mphi:
                    ds = mphi.group(1).strip()
                    try:
                        diam_new = float(ds)
                    except Exception:
                        try:
                            diam_new = float(ds.replace(',', '.'))
                        except Exception:
                            diam_new = None
                elif len(nums) >= 1:
                    try:
                        diam_new = float(nums[-1])
                    except Exception:
                        try:
                            diam_new = float(nums[-1].replace(',', '.'))
                        except Exception:
                            diam_new = None
                confirmed = self._get_confirmed_list()
                sidx = info['staffa_idx']
                if 0 <= sidx < len(confirmed):
                    staffa = confirmed[sidx]
                    if name_new:
                        staffa['name'] = name_new
                    if diam_new is not None:
                        try:
                            staffa['diam'] = max(0.0001, round(float(diam_new)))
                        except Exception:
                            pass

        finally:
            try:
                widget.update()
                widget.setFocus()
            except Exception:
                pass
            self._finalizing = False

    # ---------------- conferme/flusso ----------------
    def _confirm_active_draft(self, widget):
        """
        Primo Invio: conferma l'ultima draft in progress e passa lo stato
        'vertices_shown' (mostra etichette sui vertici). Non apre l'editor generale.
        Unisce vertici molto vicini.
        """
        if len(self.draft_staffe) == 0:
            return
        last = self.draft_staffe[-1]
        if not last.get('in_progress', False):
            return

        pts = last.get('points', [])
        fixed_flags = last.get('fixed', [])
        fixed_points = []
        for (p, f) in zip(pts, fixed_flags):
            if f:
                fixed_points.append(p)
            else:
                break

        # richiediamo almeno 2 punti fissati
        if len(fixed_points) < 2:
            # scarta la draft
            self.draft_staffe.pop(-1)
            try:
                widget.update()
            except Exception:
                pass
            return

        # merge vertici prossimi (soglia in pixel)
        merged = self._merge_close_vertices_widget(widget, fixed_points)

        # confermo: setto state 'vertices_shown' (vertex labels visibili)
        self._counter += 1
        name = f"S{self._counter}"
        entry = {'points': merged, 'diam': last.get('diam', self.default_diam), 'name': name,
                 'state': 'vertices_shown'}
        confirmed = self._get_confirmed_list()
        confirmed.append(entry)
        self.draft_staffe.pop(-1)

        try:
            widget.update()
        except Exception:
            pass

    def _confirm_all_drafts(self, widget):
        """Conferma tutte le draft (comportamento di massima: li porta tutti a 'vertices_shown')."""
        if len(self.draft_staffe) == 0:
            return
        confirmed = self._get_confirmed_list()
        for last in list(self.draft_staffe):
            pts = last.get('points', [])
            fixed_flags = last.get('fixed', [])
            fixed_points = []
            for (p, f) in zip(pts, fixed_flags):
                if f:
                    fixed_points.append(p)
                else:
                    break
            if len(fixed_points) >= 2:
                merged = self._merge_close_vertices_widget(None, fixed_points)  # widget None -> no merging by pixels
                self._counter += 1
                name = f"S{self._counter}"
                entry = {'points': merged, 'diam': last.get('diam', self.default_diam), 'name': name,
                         'state': 'vertices_shown'}
                confirmed.append(entry)
        self.draft_staffe = []
        try:
            widget.update()
        except Exception:
            pass

    # ---------------- utilità fusione vertici ----------------
    def _merge_close_vertices_widget(self, widget, points):
        """
        Unisce vertici molto vicini: se widget è fornito usa distanza in pixel,
        altrimenti usa distanza world molto piccola.
        """
        if not points:
            return []
        if widget is None:
            # fallback: merge exact duplicates only
            out = []
            for p in points:
                if not out or p != out[-1]:
                    out.append(p)
            return out

        out = []
        for p in points:
            if not out:
                out.append(p)
                continue
            sx1, sy1 = widget.world_to_screen(p[0], p[1])
            sx0, sy0 = widget.world_to_screen(out[-1][0], out[-1][1])
            dist = math.hypot(sx1 - sx0, sy1 - sy0)
            if dist <= self.merge_vertex_px:
                # unisci: sostituisci con media tra i due (world coords)
                ax, ay = out[-1]
                bx, by = p
                merged = ((ax + bx) / 2.0, (ay + by) / 2.0)
                out[-1] = merged
            else:
                out.append(p)
        return out
