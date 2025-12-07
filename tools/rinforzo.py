# ReinforcementTool - versione corretta per etichette cerchio a destra e rendering confermati solo via OpenGL
from typing import List, Dict, Any, Optional, Tuple
from PyQt5.QtWidgets import QLineEdit, QShortcut
from PyQt5.QtGui import QPainter, QColor, QKeySequence, QPainterPath
from PyQt5.QtCore import Qt, QRect, QPoint
import numpy as np

try:
    from OpenGL.GL import *
    _HAS_GL = True
except Exception:
    _HAS_GL = False


class ReinforcementTool:
    class OffsetEditor(QLineEdit):
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
                 default_offset: float = 2.0,
                 hover_color: QColor = QColor(255, 0, 0, 200),
                 hover_line_width: float = 3.0,
                 # preview changed to red dashed + transparent red fill
                 preview_line_color: QColor = QColor(255, 0, 0, 200),
                 preview_fill_color: QColor = QColor(255, 0, 0, 80),
                 # confirmed fill changed to transparent red by default
                 confirmed_fill_color: QColor = QColor(255, 0, 0, 120),
                 hover_threshold_px: int = 8,
                 label_gap_px: int = 8):
        # shape refs
        self._rects_ref: Optional[List[Dict[str, Any]]] = None
        self._polys_ref: Optional[List[Dict[str, Any]]] = None
        self._circs_ref: Optional[List[Dict[str, Any]]] = None

        # confirmed reinforcements (external or internal)
        self._internal_confirmed: List[Dict[str, Any]] = []
        self._external_confirmed_ref: Optional[List[Dict[str, Any]]] = None

        # styles / params
        self.default_offset = float(default_offset)
        self.hover_color = hover_color
        self.hover_line_width = float(hover_line_width)
        self.preview_line_color = preview_line_color
        self.preview_fill_color = preview_fill_color
        self.confirmed_fill_color = confirmed_fill_color
        self.hover_threshold_px = int(hover_threshold_px)
        self.label_gap_px = int(label_gap_px)

        # state
        self._hover_target: Optional[Dict[str, Any]] = None
        self._hover_label_rect: Optional[QRect] = None
        self._drafts: List[Dict[str, Any]] = []
        self._counter = 0

        # duplicates prevention
        self._taken_keys = set()

        # editor state
        self._active_editor = None
        self._editing_index = None
        self._finalizing = False

        # shortcuts
        self._shortcuts = []

        # precompute GL colors
        def _q_to_gl(c: QColor):
            try:
                # QColor.getRgbF() returns floats (r,g,b,a)
                return c.getRgbF()
            except Exception:
                return (1.0, 1.0, 1.0, 1.0)
        # confirmed fill and confirmed line (border) GL colors
        self._gl_confirmed_fill = _q_to_gl(self.confirmed_fill_color)
        # border of confirmed shapes: solid red (use hover_color if red else make solid from confirmed color)
        # prefer a fully opaque border color (alpha=1.0)
        try:
            border_q = QColor(self.confirmed_fill_color.red(), self.confirmed_fill_color.green(), self.confirmed_fill_color.blue(), 255)
            self._gl_confirmed_line = _q_to_gl(border_q)
        except Exception:
            self._gl_confirmed_line = (1.0, 0.0, 0.0, 1.0)

        # store preview GL as well if needed in future (not used for painter)
        self._gl_preview_fill = _q_to_gl(self.preview_fill_color)
        try:
            plq = QColor(self.preview_line_color.red(), self.preview_line_color.green(), self.preview_line_color.blue(), 255)
            self._gl_preview_line = _q_to_gl(plq)
        except Exception:
            self._gl_preview_line = (1.0, 0.0, 0.0, 1.0)

    # ---------------- binding API ----------------
    def set_shape_lists(self, rects: Optional[List[Dict[str, Any]]],
                        polys: Optional[List[Dict[str, Any]]],
                        circs: Optional[List[Dict[str, Any]]]):
        self._rects_ref = rects
        self._polys_ref = polys
        self._circs_ref = circs

    def set_confirmed_list(self, list_ref: Optional[List[Dict[str, Any]]]):
        if list_ref is None:
            self._external_confirmed_ref = None
        else:
            self._external_confirmed_ref = list_ref

    def _get_confirmed_list(self) -> List[Dict[str, Any]]:
        return self._external_confirmed_ref if self._external_confirmed_ref is not None else self._internal_confirmed

    # ---------------- activation ----------------
    def on_activate(self, widget):
        try:
            widget.setCursor(Qt.CrossCursor)
        except Exception:
            pass
        try:
            self._shortcuts = []
            ks = QShortcut(QKeySequence(Qt.Key_Return), widget)
            ks.activated.connect(lambda: self._on_enter_pressed(widget))
            self._shortcuts.append(ks)
            ks2 = QShortcut(QKeySequence(Qt.Key_Enter), widget)
            ks2.activated.connect(lambda: self._on_enter_pressed(widget))
            self._shortcuts.append(ks2)
        except Exception:
            self._shortcuts = []

    def on_deactivate(self, widget):
        if self._active_editor is not None:
            try:
                self._finalize_editor(widget)
            except Exception:
                pass
        self._hover_target = None
        self._hover_label_rect = None
        self._drafts = []
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
                widget.update()
                return
            if self._drafts:
                self.confirm_all(widget)
                widget.update()
        except Exception:
            pass

    # ---------------- geometry gathering ----------------
    def _gather_geometry(self, widget) -> List[Dict[str, Any]]:
        elems = []

        def add_segments(pts, parent=None):
            n = len(pts)
            if n < 2:
                return
            for i in range(n):
                a = tuple(pts[i]); b = tuple(pts[(i + 1) % n])
                elems.append({'type': 'segment', 'p1': a, 'p2': b, 'parent': parent, 'edge_index': i})

        if self._rects_ref:
            for r in self._rects_ref:
                try:
                    v1 = r['v1']; v2 = r['v2']
                    left, right = min(v1[0], v2[0]), max(v1[0], v2[0])
                    bottom, top = min(v1[1], v2[1]), max(v1[1], v2[1])
                    corners = [(left, bottom), (left, top), (right, top), (right, bottom)]
                    add_segments(corners, parent=r)
                except Exception:
                    pass

        if self._polys_ref:
            for p in self._polys_ref:
                verts = p.get('vertices') or p.get('pts') or p.get('points')
                if verts and len(verts) >= 2:
                    add_segments(verts, parent=p)

        if self._circs_ref:
            for c in self._circs_ref:
                try:
                    center = tuple(c['center']); rp = tuple(c['radius_point'])
                    r = float(np.hypot(rp[0] - center[0], rp[1] - center[1]))
                    elems.append({'type': 'circle', 'center': center, 'radius': r, 'parent': c})
                except Exception:
                    pass

        # fallback to widget attributes lightly (backwards compat)
        if not elems:
            for attr in ('confirmed_rects', 'rects'):
                col = getattr(widget, attr, None)
                if col:
                    for r in col:
                        try:
                            v1 = r['v1']; v2 = r['v2']
                            corners = [(v1[0], v1[1]), (v1[0], v2[1]), (v2[0], v2[1]), (v2[0], v1[1])]
                            add_segments(corners, parent=r)
                        except Exception:
                            pass
            for attr in ('confirmed_polygons', 'polygons', 'polys'):
                col = getattr(widget, attr, None)
                if col:
                    for p in col:
                        verts = p.get('vertices') or p.get('pts') or p.get('points')
                        if verts and len(verts) >= 2:
                            add_segments(verts, parent=p)
            for attr in ('confirmed_circles', 'circles'):
                col = getattr(widget, attr, None)
                if col:
                    for c in col:
                        try:
                            center = tuple(c['center']); rp = tuple(c['radius_point'])
                            r = float(np.hypot(rp[0] - center[0], rp[1] - center[1]))
                            elems.append({'type': 'circle', 'center': center, 'radius': r, 'parent': c})
                        except Exception:
                            pass

        return elems

    # ---------------- math helpers ----------------
    def _world_to_screen(self, widget, x, y) -> Tuple[float, float]:
        if not hasattr(widget, 'world_to_screen'):
            raise RuntimeError("widget missing method world_to_screen(x,y)")
        return widget.world_to_screen(x, y)

    def _dist_point_to_segment_screen(self, px, py, x1, y1, x2, y2):
        vx = x2 - x1; vy = y2 - y1
        wx = px - x1; wy = py - y1
        seg2 = vx * vx + vy * vy
        if seg2 == 0:
            return np.hypot(wx, wy), (x1, y1), 0.0
        t = (wx * vx + wy * vy) / seg2
        tc = max(0.0, min(1.0, t))
        sx = x1 + tc * vx; sy = y1 + tc * vy
        d = np.hypot(px - sx, py - sy)
        return d, (sx, sy), tc

    def _polygon_area(self, verts: List[Tuple[float, float]]) -> float:
        a = 0.0; n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]; x2, y2 = verts[(i + 1) % n]
            a += x1 * y2 - x2 * y1
        return 0.5 * a

    def _centroid(self, verts: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not verts:
            return (0.0, 0.0)
        return (sum(v[0] for v in verts) / len(verts), sum(v[1] for v in verts) / len(verts))

    def _compute_outward_normal(self, p1, p2, parent) -> Tuple[float, float]:
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        L = np.hypot(dx, dy)
        if L == 0:
            return (0.0, 0.0)
        left = (-dy / L, dx / L); right = (dy / L, -dx / L)
        try:
            if parent is not None:
                if 'vertices' in parent:
                    c = self._centroid(parent['vertices'])
                elif 'v1' in parent and 'v2' in parent:
                    v1 = parent['v1']; v2 = parent['v2']
                    c = ((v1[0] + v2[0]) / 2.0, (v1[1] + v2[1]) / 2.0)
                elif 'center' in parent:
                    c = tuple(parent['center'])
                else:
                    c = None
                if c is not None:
                    mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
                    vx, vy = mid[0] - c[0], mid[1] - c[1]
                    dot_left = vx * left[0] + vy * left[1]
                    return left if dot_left > 0 else right
        except Exception:
            pass
        return right

    def _make_segment_key(self, parent, edge_index, p1, p2):
        if parent is not None:
            try:
                return ('segment', id(parent), int(edge_index) if edge_index is not None else None)
            except Exception:
                pass
        a = (round(p1[0], 12), round(p1[1], 12)); b = (round(p2[0], 12), round(p2[1], 12))
        return ('segment_coords', a, b) if a <= b else ('segment_coords', b, a)

    def _make_circle_key(self, parent, center):
        if parent is not None:
            try:
                return ('circle', id(parent))
            except Exception:
                pass
        a = (round(center[0], 12), round(center[1], 12))
        return ('circle_coords', a)

    # ---------------- input handlers ----------------
    def on_mouse_move(self, widget, event) -> bool:
        px, py = event.x(), event.y()
        try:
            elems = self._gather_geometry(widget)
            best = None; bestd = float('inf'); thr = self.hover_threshold_px
            for el in elems:
                if el['type'] == 'segment':
                    sx1, sy1 = self._world_to_screen(widget, *el['p1'])
                    sx2, sy2 = self._world_to_screen(widget, *el['p2'])
                    d, proj, t = self._dist_point_to_segment_screen(px, py, sx1, sy1, sx2, sy2)
                    if d < bestd:
                        bestd = d
                        best = {'type': 'segment', 'p1': el['p1'], 'p2': el['p2'],
                                's1': (sx1, sy1), 's2': (sx2, sy2), 'proj': proj, 't': t,
                                'edge_index': el.get('edge_index'), 'parent': el.get('parent')}
                else:
                    scx, scy = self._world_to_screen(widget, *el['center'])
                    screen_r = abs(self._world_to_screen(widget, el['center'][0] + el['radius'], el['center'][1])[0] - scx)
                    dcenter = np.hypot(px - scx, py - scy)
                    d = abs(dcenter - screen_r)
                    if d < bestd:
                        bestd = d
                        best = {'type': 'circle', 'center': el['center'], 'radius': el['radius'],
                                'scenter': (scx, scy), 'sradius': screen_r, 'parent': el.get('parent')}
            if best is not None and bestd <= thr:
                # label to the right of element
                if best['type'] == 'segment':
                    lx = int(best['proj'][0] + self.label_gap_px); ly = int(best['proj'][1] - self.label_gap_px)
                else:
                    lx = int(best['scenter'][0] + best['sradius'] + self.label_gap_px); ly = int(best['scenter'][1] - self.label_gap_px)
                fmw = widget.fontMetrics().horizontalAdvance(f"RINFORZO {self._counter+1} ({self.default_offset:.3g})")
                fmh = widget.fontMetrics().height()
                self._hover_label_rect = QRect(lx, ly - fmh // 2, fmw + 8, fmh + 4)
                self._hover_target = best
                widget.update()
                return True
            else:
                changed = (self._hover_target is not None)
                self._hover_target = None
                self._hover_label_rect = None
                if changed:
                    widget.update()
                return False
        except Exception:
            self._hover_target = None
            self._hover_label_rect = None
            return False

    def _draft_label_hit(self, widget, px, py) -> Optional[int]:
        fm = widget.fontMetrics()
        for i, d in enumerate(self._drafts):
            r = d.get('label_rect')
            if r and r.contains(px, py):
                return i
            # compute on the fly
            if d['type'] == 'segment':
                xs = [self._world_to_screen(widget, x, y)[0] for x, y in d['poly']]
                ys = [self._world_to_screen(widget, x, y)[1] for x, y in d['poly']]
                mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
                txt = f"{d['name']} ({d['offset']:.3g})"
                w = fm.horizontalAdvance(txt); h = fm.height()
                rect = QRect(int(mx - w / 2), int(my - h / 2), int(w) + 6, int(h) + 4)
                d['label_rect'] = rect
                if rect.contains(px, py):
                    return i
            else:
                scx, scy = self._world_to_screen(widget, *d['center'])
                txt = f"{d['name']} ({d['offset']:.3g})"
                w = fm.horizontalAdvance(txt); h = fm.height()
                # place to the right of circle
                outer_screen_r = abs(self._world_to_screen(widget, d['center'][0] + d['radius'], d['center'][1])[0] - scx)
                rect = QRect(int(scx + outer_screen_r + self.label_gap_px), int(scy - h / 2), int(w) + 6, int(h) + 4)
                d['label_rect'] = rect
                if rect.contains(px, py):
                    return i
        return None

    def on_mouse_press(self, widget, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        px, py = event.x(), event.y()

        if self._active_editor is not None:
            geom = self._active_editor.geometry()
            if geom.contains(px, py):
                return False
            else:
                self._finalize_editor(widget)
                widget.update()
                event.accept()
                return True

        d_idx = self._draft_label_hit(widget, px, py)
        if d_idx is not None:
            self._open_editor_for_draft(widget, d_idx)
            event.accept()
            return True

        if self._hover_label_rect is not None and self._hover_label_rect.contains(px, py) and self._hover_target is not None:
            tgt = self._hover_target
            if tgt['type'] == 'segment':
                key = self._make_segment_key(tgt.get('parent'), tgt.get('edge_index'), tgt['p1'], tgt['p2'])
            else:
                key = self._make_circle_key(tgt.get('parent'), tgt.get('center'))
            if key in self._taken_keys:
                return True
            idx = self._create_draft_from_target(widget, tgt, self.default_offset)
            self._taken_keys.add(key)
            self._open_editor_for_draft(widget, idx)
            widget.update()
            event.accept()
            return True

        if self._hover_target is not None:
            tgt = self._hover_target
            if tgt['type'] == 'segment':
                key = self._make_segment_key(tgt.get('parent'), tgt.get('edge_index'), tgt['p1'], tgt['p2'])
            else:
                key = self._make_circle_key(tgt.get('parent'), tgt.get('center'))
            if key in self._taken_keys:
                return True
            self._create_draft_from_target(widget, tgt, self.default_offset)
            self._taken_keys.add(key)
            widget.update()
            event.accept()
            return True

        return False

    def on_key_press(self, widget, event) -> bool:
        k = event.key()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            if self._active_editor is not None:
                self._finalize_editor(widget)
                widget.update()
                return True
            if self._drafts:
                self.confirm_all(widget)
                widget.update()
                return True
            return False
        if k == Qt.Key_Escape:
            if self._active_editor is not None:
                try:
                    self._active_editor.deleteLater()
                except Exception:
                    pass
                self._active_editor = None
                self._editing_index = None
                widget.update()
                return True
            if self._drafts:
                for d in self._drafts:
                    if d.get('key') in self._taken_keys:
                        self._taken_keys.discard(d['key'])
                self._drafts = []
                widget.update()
                return True
            return False
        return False

    # ---------------- drafts / editing ----------------
    def _create_draft_from_target(self, widget, tgt: Dict[str, Any], offset: float) -> int:
        self._counter += 1
        name = f"RINFORZO {self._counter}"
        if tgt['type'] == 'segment':
            p1 = tuple(tgt['p1']); p2 = tuple(tgt['p2']); parent = tgt.get('parent'); ei = tgt.get('edge_index')
            nx, ny = self._compute_outward_normal(p1, p2, parent)
            op1 = (p1[0] + nx * offset, p1[1] + ny * offset)
            op2 = (p2[0] + nx * offset, p2[1] + ny * offset)
            poly = [p1, p2, op2, op1]
            key = self._make_segment_key(parent, ei, p1, p2)
            draft = {'type': 'segment', 'poly': poly, 'offset': float(offset), 'name': name,
                     'parent': parent, 'edge_index': ei, 'base_p1': p1, 'base_p2': p2, 'key': key, 'label_rect': None}
            self._drafts.append(draft)
            return len(self._drafts) - 1
        else:
            center = tuple(tgt['center']); r = float(tgt['radius']); parent = tgt.get('parent')
            new_r = float(r + offset)
            key = self._make_circle_key(parent, center)
            draft = {'type': 'circle', 'center': center, 'base_radius': r, 'radius': new_r,
                     'offset': float(offset), 'name': name, 'parent': parent, 'key': key, 'label_rect': None}
            self._drafts.append(draft)
            return len(self._drafts) - 1

    def _open_editor_for_draft(self, widget, idx: int):
        if not (0 <= idx < len(self._drafts)):
            return
        if self._active_editor is not None:
            self._finalize_editor(widget)
        d = self._drafts[idx]
        txt = f"{d['offset']:.6g}"
        rect = d.get('label_rect') or self._hover_label_rect or QRect(10, 10, 120, 20)
        editor = ReinforcementTool.OffsetEditor(widget, on_focus_lost=lambda: self._finalize_editor(widget))
        editor.setText(txt)
        editor.setGeometry(rect.x(), rect.y(), rect.width() + 8, rect.height() + 4)
        editor.show(); editor.setFocus(); editor.setAttribute(Qt.WA_DeleteOnClose, True)
        self._active_editor = editor
        self._editing_index = idx
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
            if self._active_editor is None:
                return
            editor = self._active_editor
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
                editor.hide(); editor.close()
            except Exception:
                pass
            try:
                editor.deleteLater()
            except Exception:
                pass
            idx = self._editing_index
            self._active_editor = None
            self._editing_index = None

            parsed_offset = None; parsed_name = None
            if '(' in txt and txt.endswith(')'):
                try:
                    left, right = txt.rsplit('(', 1)
                    parsed_name = left.strip()
                    parsed_offset = float(right[:-1].strip()) if right[:-1].strip() != '' else None
                except Exception:
                    parsed_name = txt
            else:
                try:
                    parsed_offset = float(txt)
                except Exception:
                    parsed_name = txt if txt else None

            if idx is None:
                if parsed_offset is not None:
                    self.default_offset = parsed_offset
            else:
                if 0 <= idx < len(self._drafts):
                    d = self._drafts[idx]
                    if parsed_name:
                        d['name'] = parsed_name
                    if parsed_offset is not None:
                        d['offset'] = float(parsed_offset)
                        if d['type'] == 'segment':
                            p1 = d['base_p1']; p2 = d['base_p2']; parent = d.get('parent')
                            nx, ny = self._compute_outward_normal(p1, p2, parent)
                            d['poly'] = [p1, p2, (p2[0] + nx * d['offset'], p2[1] + ny * d['offset']),
                                         (p1[0] + nx * d['offset'], p1[1] + ny * d['offset'])]
                        else:
                            d['radius'] = float(d['base_radius'] + d['offset'])
            try:
                widget.update()
            except Exception:
                pass
        finally:
            self._finalizing = False

    # ---------------- confirm / persistence ----------------
    def confirm_all(self, widget):
        out = self._get_confirmed_list()
        for d in self._drafts:
            # append shallow copy; label pos computed dynamically at paint-time
            out.append(dict(d))
            if d.get('key'):
                self._taken_keys.add(d['key'])
        self._drafts = []
        try:
            widget.update()
        except Exception:
            pass

    # ---------------- drawing painter (fixed label pos for circles) ----------------
    def draw_painter(self, widget, painter: QPainter):
        """
        Painter draws:
         - hover preview (segment or circle) with dashed outline and transparent fill (red, per defaults)
         - drafts (dashed) with preview fill
         - labels (text) for drafts
        Painter DOES NOT draw confirmed shapes fills/borders (to avoid overlap with OpenGL).
        Painter may still draw textual labels for confirmed items (no filled shapes).
        """
        painter.setRenderHint(QPainter.Antialiasing)
        fm = widget.fontMetrics()

        # hover preview
        if self._hover_target is not None:
            pen = painter.pen()
            pen.setWidth(int(self.hover_line_width))
            pen.setColor(self.hover_color)
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            if self._hover_target['type'] == 'segment':
                # draw base segment in hover color
                sx1, sy1 = self._hover_target['s1']; sx2, sy2 = self._hover_target['s2']
                painter.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))
                # compute outward quad for default offset
                p1 = self._hover_target['p1']; p2 = self._hover_target['p2']; parent = self._hover_target.get('parent')
                nx, ny = self._compute_outward_normal(p1, p2, parent)
                op1 = (p1[0] + nx * self.default_offset, p1[1] + ny * self.default_offset)
                op2 = (p2[0] + nx * self.default_offset, p2[1] + ny * self.default_offset)
                pts = [QPoint(*self._world_to_screen(widget, x, y)) for x, y in (p1, p2, op2, op1)]
                dash_pen = painter.pen()
                dash_pen.setStyle(Qt.DashLine)
                dash_pen.setWidth(1)
                # preview_line_color is expected to be QColor
                dash_pen.setColor(self.preview_line_color)
                painter.setPen(dash_pen)
                painter.setBrush(self.preview_fill_color)
                painter.drawPolygon(*pts)
                painter.setBrush(Qt.NoBrush)
            else:
                # circle preview: dashed outer ring and transparent red fill
                scx, scy = self._hover_target['scenter']; sr = int(self._hover_target['sradius'])
                painter.drawEllipse(int(scx - sr), int(scy - sr), int(sr * 2), int(sr * 2))
                cx, cy = self._hover_target['center']; r = self._hover_target['radius']
                outer_r = r + self.default_offset
                scx, scy = self._world_to_screen(widget, cx, cy)
                screen_or = abs(self._world_to_screen(widget, cx + outer_r, cy)[0] - scx)
                screen_ir = abs(self._world_to_screen(widget, cx + r, cy)[0] - scx)
                outer_rect = QRect(int(scx - screen_or), int(scy - screen_or), int(screen_or * 2), int(screen_or * 2))
                inner_rect = QRect(int(scx - screen_ir), int(scy - screen_ir), int(screen_ir * 2), int(screen_ir * 2))
                path = QPainterPath(); path.addEllipse(outer_rect)
                inner = QPainterPath(); inner.addEllipse(inner_rect)
                path.addPath(inner); path.setFillRule(Qt.OddEvenFill)
                dash_pen = painter.pen()
                dash_pen.setStyle(Qt.DashLine)
                dash_pen.setWidth(1)
                dash_pen.setColor(self.preview_line_color)
                painter.setPen(dash_pen)
                painter.setBrush(self.preview_fill_color)
                painter.drawPath(path)
                painter.setBrush(Qt.NoBrush)

            # draw hover label if present
            if self._hover_label_rect is not None:
                txt = f"RINFORZO {self._counter + 1} ({self.default_offset:.3g})"
                painter.setPen(QColor(255, 255, 255))
                painter.setBrush(QColor(0, 0, 0, 170))
                painter.drawRect(self._hover_label_rect)
                painter.drawText(self._hover_label_rect.x() + 4,
                                 self._hover_label_rect.y() + fm.ascent() + 2,
                                 txt)

        # draft reinforcements (dashed)
        dash_pen = painter.pen()
        dash_pen.setStyle(Qt.DashLine)
        dash_pen.setWidth(1)
        dash_pen.setColor(self.preview_line_color)
        painter.setPen(dash_pen)
        painter.setBrush(self.preview_fill_color)
        for d in self._drafts:
            if d['type'] == 'segment':
                pts = [QPoint(*self._world_to_screen(widget, x, y)) for x, y in d['poly']]
                painter.drawPolygon(*pts)
                xs = [self._world_to_screen(widget, x, y)[0] for x, y in d['poly']]
                ys = [self._world_to_screen(widget, x, y)[1] for x, y in d['poly']]
                mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
                txt = f"{d['name']} ({d['offset']:.3g})"
                # draw text with white color for readability
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(int(mx - fm.horizontalAdvance(txt) / 2), int(my + fm.height() / 2), txt)
                w = fm.horizontalAdvance(txt); h = fm.height()
                d['label_rect'] = QRect(int(mx - w / 2), int(my - h / 2), int(w) + 6, int(h) + 4)
                painter.setPen(dash_pen)
            else:
                cx, cy = d['center']; outer_r = d['radius']; inner_r = d['base_radius']
                scx, scy = self._world_to_screen(widget, cx, cy)
                screen_or = abs(self._world_to_screen(widget, cx + outer_r, cy)[0] - scx)
                txt = f"{d['name']} ({d['offset']:.3g})"
                w = fm.horizontalAdvance(txt); h = fm.height()
                # label to the right of circle
                rect = QRect(int(scx + screen_or + self.label_gap_px), int(scy - h / 2), int(w) + 6, int(h) + 4)
                d['label_rect'] = rect
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(rect.x() + 4, rect.y() + fm.ascent() + 2, txt)
                painter.setPen(dash_pen)
        painter.setBrush(Qt.NoBrush)

        # confirmed reinforcements: DO NOT DRAW FILLS/BORDERS via painter.
        # Only draw textual labels (no shapes) so OpenGL can handle the visuals exclusively.
        pen_text = painter.pen()
        pen_text.setStyle(Qt.SolidLine)
        pen_text.setWidth(1)
        pen_text.setColor(QColor(220, 220, 230))
        painter.setPen(pen_text)
        confirmed_list = self._get_confirmed_list()
        for r in confirmed_list:
            if r['type'] == 'segment':
                xs = [self._world_to_screen(widget, x, y)[0] for x, y in r['poly']]
                ys = [self._world_to_screen(widget, x, y)[1] for x, y in r['poly']]
                mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
                name = r.get('name', '')
                if name:
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(int(mx - fm.horizontalAdvance(name) / 2), int(my + fm.height() / 2), name)
                    painter.setPen(pen_text)
            else:
                cx, cy = r['center']; outer_r = r['radius']; inner_r = r.get('base_radius', 0.0)
                scx, scy = self._world_to_screen(widget, cx, cy)
                screen_or = abs(self._world_to_screen(widget, cx + outer_r, cy)[0] - scx)
                txt = r.get('name', '')
                if txt:
                    w = fm.horizontalAdvance(txt); h = fm.height()
                    rect = QRect(int(scx + screen_or + self.label_gap_px), int(scy - h / 2), int(w) + 6, int(h) + 4)
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(rect.x() + 4, rect.y() + fm.ascent() + 2, txt)
                    painter.setPen(pen_text)
        painter.setBrush(Qt.NoBrush)

    # ---------------- OpenGL draw ----------------
    def draw_gl(self, widget):
        """
        Draw confirmed reinforcements via OpenGL only (fills + borders).
        Painter intentionally does not render confirmed shapes to avoid double-drawing.
        """
        if not _HAS_GL:
            return
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        confirmed_list = self._get_confirmed_list()
        for r in confirmed_list:
            if r['type'] == 'segment':
                poly = r['poly']
                fr, fg, fb, fa = self._gl_confirmed_fill
                # fill (quad)
                glColor4f(fr, fg, fb, fa)
                glBegin(GL_QUADS)
                for x, y in poly:
                    glVertex2f(x, y)
                glEnd()
                # border: opaque color (use confirmed line color), width 2
                lr, lg, lb, la = self._gl_confirmed_line
                glColor4f(lr, lg, lb, 1.0)
                glLineWidth(2.0)
                glBegin(GL_LINE_LOOP)
                for x, y in poly:
                    glVertex2f(x, y)
                glEnd()
            else:
                cx, cy = r['center']; r_in = r.get('base_radius', 0.0); r_out = r['radius']
                SEG = 64
                fr, fg, fb, fa = self._gl_confirmed_fill
                # draw ring (triangle strip) between r_in and r_out
                glColor4f(fr, fg, fb, fa)
                glBegin(GL_TRIANGLE_STRIP)
                for i in range(SEG + 1):
                    ang = 2.0 * np.pi * (i / SEG)
                    x_out = cx + np.cos(ang) * r_out; y_out = cy + np.sin(ang) * r_out
                    x_in = cx + np.cos(ang) * r_in; y_in = cy + np.sin(ang) * r_in
                    glVertex2f(x_out, y_out); glVertex2f(x_in, y_in)
                glEnd()
                # border outer circle
                lr, lg, lb, la = self._gl_confirmed_line
                glColor4f(lr, lg, lb, 1.0)
                glLineWidth(2.0)
                glBegin(GL_LINE_LOOP)
                for i in range(SEG):
                    ang = 2.0 * np.pi * (i / SEG)
                    x = cx + np.cos(ang) * r_out; y = cy + np.sin(ang) * r_out
                    glVertex2f(x, y)
                glEnd()
        try:
            glDisable(GL_BLEND)
        except Exception:
            pass

    # ---------------- utility ----------------
    def list_confirmed(self) -> List[Dict[str, Any]]:
        return self._get_confirmed_list()

    def clear_all(self):
        self._drafts = []
        if self._external_confirmed_ref is None:
            self._internal_confirmed = []
        else:
            try:
                self._external_confirmed_ref.clear()
            except Exception:
                pass
        self._taken_keys.clear()
