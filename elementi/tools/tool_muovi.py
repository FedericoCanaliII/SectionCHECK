"""
tool_muovi.py – Translation tool with a 3-axis gizmo.

  • Hover over an axis arrow → arrow brightens
  • Left-click + drag on an arrow → translate along that axis
  • Left-click on an object body → select it
  • Clicking empty space → keep current selection
"""

import math
from PyQt5.QtCore import Qt
from OpenGL.GL import (
    glBegin, glEnd, glVertex3f, glColor4f, glLineWidth,
    glPointSize, glDepthFunc, GL_ALWAYS, GL_LEQUAL,
    GL_LINES, GL_POINTS,
)

from .base_tool_3d import BaseTool3D

_GIZMO_LEN  = 1.8
_HIT_RADIUS = 14     # screen pixels for click detection
_HOV_RADIUS = 16     # screen pixels for hover detection

_AXIS_DEF = [
    ("X", 1, 0, 0, 0.90, 0.18, 0.18, 1.0),
    ("Y", 0, 1, 0, 0.18, 0.80, 0.18, 1.0),
    ("Z", 0, 0, 1, 0.18, 0.45, 0.90, 1.0),
]


class ToolMuovi(BaseTool3D):
    name = "muovi"

    def __init__(self):
        self._active_axis      = None
        self._hover_axis       = None
        self._drag_start       = None
        self._obj_pos_at_start = None
        self._ref_at_start     = None   # reference vertex world pos at drag start

    # ------------------------------------------------------------------
    def on_activate(self, spazio):
        spazio.setCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------------
    def on_hover(self, spazio, event):
        if self._active_axis is not None:
            return   # mid-drag: don't update hover
        sel_id = spazio.get_id_selezionato()
        if sel_id == -1:
            if self._hover_axis is not None:
                self._hover_axis = None
                spazio.update()
            return
        new_hov = self._hit_axis(event.x(), event.y(), sel_id, spazio,
                                  radius=_HOV_RADIUS)
        if new_hov != self._hover_axis:
            self._hover_axis = new_hov
            spazio.update()

    # ------------------------------------------------------------------
    def on_mouse_press(self, spazio, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False

        mx, my = event.x(), event.y()
        sel_id = spazio.get_id_selezionato()

        # Try gizmo axis first
        if sel_id != -1:
            axis = self._hit_axis(mx, my, sel_id, spazio, radius=_HIT_RADIUS)
            if axis is not None:
                self._active_axis = axis
                self._drag_start  = event.pos()
                obj = spazio.get_oggetto(sel_id)
                if obj:
                    self._obj_pos_at_start = list(obj.posizione)
                    self._ref_at_start     = list(obj.get_vertex_ref_world())
                return True

        # Try to pick a new object (no deselect on miss)
        picked = spazio._pick_at(mx, my)
        if picked != -1:
            spazio.set_id_selezionato(picked)
        return True

    def on_mouse_move(self, spazio, event) -> bool:
        if self._active_axis is None or self._drag_start is None:
            return False

        dx = event.x() - self._drag_start.x()
        dy = event.y() - self._drag_start.y()

        sel_id = spazio.get_id_selezionato()
        obj    = spazio.get_oggetto(sel_id)
        if obj is None:
            return False

        # Use the ref from drag start to avoid feedback loop
        ref = self._ref_at_start
        axis_map = {"X": (1,0,0), "Y": (0,1,0), "Z": (0,0,1)}
        ax, ay, az = axis_map[self._active_axis]
        delta = self._axis_drag_delta(dx, dy, ax, ay, az, *ref, spazio)

        base = self._obj_pos_at_start
        obj.posizione = [base[0] + ax*delta,
                         base[1] + ay*delta,
                         base[2] + az*delta]
        spazio._emit_modificato(sel_id)
        return True

    def on_mouse_release(self, spazio, event) -> bool:
        if event.button() == Qt.LeftButton and self._active_axis is not None:
            self._active_axis      = None
            self._drag_start       = None
            self._obj_pos_at_start = None
            self._ref_at_start     = None
            return True
        return False

    # ------------------------------------------------------------------
    def draw_overlay(self, spazio):
        sel_id = spazio.get_id_selezionato()
        if sel_id == -1:
            return
        obj = spazio.get_oggetto(sel_id)
        if obj is None:
            return

        rx, ry, rz = obj.get_vertex_ref_world()
        L = _GIZMO_LEN

        glDepthFunc(GL_ALWAYS)

        for name, dx, dy, dz, r, g, b, a in _AXIS_DEF:
            is_active = (self._active_axis == name)
            is_hover  = (self._hover_axis  == name)

            if is_active:
                r2, g2, b2, lw = r, g, b, 4.5
            elif is_hover:
                r2, g2, b2, lw = min(r*1.3,1), min(g*1.3,1), min(b*1.3,1), 3.5
            else:
                r2, g2, b2, lw = r*0.75, g*0.75, b*0.75, 2.5

            glLineWidth(lw)
            glColor4f(r2, g2, b2, a)
            glBegin(GL_LINES)
            glVertex3f(rx, ry, rz)
            glVertex3f(rx + dx*L, ry + dy*L, rz + dz*L)
            glEnd()

            ps = 12.0 if (is_active or is_hover) else 9.0
            glPointSize(ps)
            glBegin(GL_POINTS)
            glVertex3f(rx + dx*L, ry + dy*L, rz + dz*L)
            glEnd()

        glDepthFunc(GL_LEQUAL)
        glLineWidth(1.0)
        glPointSize(1.0)

    # ------------------------------------------------------------------
    def _hit_axis(self, mx, my, sel_id, spazio, radius) -> str | None:
        obj = spazio.get_oggetto(sel_id)
        if obj is None:
            return None
        rx, ry, rz = obj.get_vertex_ref_world()
        L = _GIZMO_LEN
        # Sample along the axis from 25% to 100% of its length for better hit area
        samples = [0.25, 0.5, 0.75, 1.0]
        best_name = None; best_dist = radius
        for name, dx, dy, dz, *_ in _AXIS_DEF:
            for t in samples:
                sx, sy, _ = self._world_to_screen(
                    rx + dx*L*t, ry + dy*L*t, rz + dz*L*t, spazio)
                if sx is None:
                    continue
                d = math.hypot(mx-sx, my-sy)
                if d < best_dist:
                    best_dist = d; best_name = name
        return best_name

    def reset(self):
        self._active_axis      = None
        self._hover_axis       = None
        self._drag_start       = None
        self._obj_pos_at_start = None
        self._ref_at_start     = None
