"""
tool_ruota.py – Rotation tool with a 3-ring gizmo.

  • Hover over a ring → ring brightens
  • Left-click + drag on a ring → rotate around that axis
"""

import math
from PyQt5.QtCore import Qt
from OpenGL.GL import (
    glBegin, glEnd, glVertex3f, glColor4f, glLineWidth,
    glDepthFunc, GL_LINE_LOOP, GL_ALWAYS, GL_LEQUAL,
)

from .base_tool_3d import BaseTool3D

_RING_R     = 1.6
_RING_SEG   = 48
_HIT_RADIUS = 12
_HOV_RADIUS = 16
_ROT_SPEED  = 0.5   # degrees per pixel

_RING_DEF = [
    ("X", (1.00, 0.18, 0.18, 0.95)),
    ("Y", (0.18, 0.80, 0.18, 0.95)),
    ("Z", (0.18, 0.45, 0.90, 0.95)),
]


class ToolRuota(BaseTool3D):
    name = "ruota"

    def __init__(self):
        self._active_ring  = None
        self._hover_ring   = None
        self._drag_start_x = None
        self._drag_start_y = None
        self._rot_at_start = None
        self._center_screen = None  # screen-space center for angular drag

    # ------------------------------------------------------------------
    def on_activate(self, spazio):
        spazio.setCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------------
    def on_hover(self, spazio, event):
        if self._active_ring is not None:
            return
        sel_id = spazio.get_id_selezionato()
        if sel_id == -1:
            if self._hover_ring is not None:
                self._hover_ring = None; spazio.update()
            return
        new_hov = self._hit_ring(event.x(), event.y(), sel_id, spazio,
                                  radius=_HOV_RADIUS)
        if new_hov != self._hover_ring:
            self._hover_ring = new_hov; spazio.update()

    # ------------------------------------------------------------------
    def on_mouse_press(self, spazio, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False

        mx, my = event.x(), event.y()
        sel_id = spazio.get_id_selezionato()

        if sel_id != -1:
            ring = self._hit_ring(mx, my, sel_id, spazio, radius=_HIT_RADIUS)
            if ring is not None:
                self._active_ring  = ring
                self._drag_start_x = mx
                self._drag_start_y = my
                obj = spazio.get_oggetto(sel_id)
                if obj:
                    self._rot_at_start = list(obj.rotazione)
                    # Cache screen-space center for angular drag
                    ref = obj.get_vertex_ref_world()
                    sx, sy, _ = self._world_to_screen(*ref, spazio)
                    self._center_screen = (sx, sy) if sx is not None else (mx, my)
                return True

        picked = spazio._pick_at(mx, my)
        if picked != -1:
            spazio.set_id_selezionato(picked)
        return True

    def on_mouse_move(self, spazio, event) -> bool:
        if self._active_ring is None or self._drag_start_x is None:
            return False

        sel_id = spazio.get_id_selezionato()
        obj    = spazio.get_oggetto(sel_id)
        if obj is None:
            return False

        cx, cy = self._center_screen or (0, 0)
        # Compute angular delta (Blender-style: angle between start
        # vector and current vector relative to ring centre)
        a_start = math.atan2(self._drag_start_y - cy, self._drag_start_x - cx)
        a_now   = math.atan2(event.y() - cy, event.x() - cx)
        delta_deg = math.degrees(a_now - a_start)

        idx = {"X": 0, "Y": 1, "Z": 2}[self._active_ring]
        new_rot = list(self._rot_at_start)
        new_rot[idx] = self._rot_at_start[idx] + delta_deg
        obj.rotazione = new_rot
        spazio._emit_modificato(sel_id)
        return True

    def on_mouse_release(self, spazio, event) -> bool:
        if event.button() == Qt.LeftButton and self._active_ring is not None:
            self._active_ring   = None
            self._drag_start_x  = None
            self._drag_start_y  = None
            self._rot_at_start  = None
            self._center_screen = None
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

        cx, cy, cz = obj.get_vertex_ref_world()
        R = _RING_R; N = _RING_SEG

        glDepthFunc(GL_ALWAYS)

        for ring_name, (r, g, b, a) in _RING_DEF:
            is_active = (self._active_ring == ring_name)
            is_hover  = (self._hover_ring  == ring_name)

            if is_active:
                rc, gc, bc, lw = r, g, b, 3.5
            elif is_hover:
                rc, gc, bc, lw = min(r*1.35,1), min(g*1.35,1), min(b*1.35,1), 2.8
            else:
                rc, gc, bc, lw = r*0.72, g*0.72, b*0.72, 2.0

            glLineWidth(lw)
            glColor4f(rc, gc, bc, a)
            glBegin(GL_LINE_LOOP)
            for i in range(N):
                t = 2 * math.pi * i / N
                if ring_name == "X":
                    glVertex3f(cx, cy + R*math.cos(t), cz + R*math.sin(t))
                elif ring_name == "Y":
                    glVertex3f(cx + R*math.cos(t), cy, cz + R*math.sin(t))
                else:
                    glVertex3f(cx + R*math.cos(t), cy + R*math.sin(t), cz)
            glEnd()

        glDepthFunc(GL_LEQUAL)
        glLineWidth(1.0)

    # ------------------------------------------------------------------
    def _hit_ring(self, mx, my, sel_id, spazio, radius) -> str | None:
        obj = spazio.get_oggetto(sel_id)
        if obj is None:
            return None
        cx, cy, cz = obj.get_vertex_ref_world()
        R = _RING_R; N = _RING_SEG
        best_name = None; best_dist = radius

        for ring_name, _ in _RING_DEF:
            for i in range(N):
                t = 2 * math.pi * i / N
                if ring_name == "X":
                    wx, wy, wz = cx, cy + R*math.cos(t), cz + R*math.sin(t)
                elif ring_name == "Y":
                    wx, wy, wz = cx + R*math.cos(t), cy, cz + R*math.sin(t)
                else:
                    wx, wy, wz = cx + R*math.cos(t), cy + R*math.sin(t), cz
                sx, sy, _ = self._world_to_screen(wx, wy, wz, spazio)
                if sx is None:
                    continue
                d = math.hypot(mx-sx, my-sy)
                if d < best_dist:
                    best_dist = d; best_name = ring_name

        return best_name

    def reset(self):
        self._active_ring   = None
        self._hover_ring    = None
        self._drag_start_x  = None
        self._drag_start_y  = None
        self._rot_at_start  = None
        self._center_screen = None
