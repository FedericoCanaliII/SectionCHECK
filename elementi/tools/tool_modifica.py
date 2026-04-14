"""
tool_modifica.py – Vertex-editing tool.

Renders each vertex of the selected object as a coloured handle:
  • reference vertex  → red
  • selected vertex   → yellow
  • others            → white/light

Left-click vertex  → select it
Left-drag          → move vertex in 3D (depth-constrained screen-plane drag)
Ctrl+Left on empty → add waypoint (barra/staffa only)
Right-click vertex → remove waypoint (barra/staffa only, min 2 or 3 pts)

Vertex drag deforms the object shape for all structural objects too.
"""

import math
from PyQt5.QtCore import Qt
from PyQt5.QtGui  import QPainter, QColor, QFont
from OpenGL.GL    import (
    glBegin, glEnd, glVertex3f, glColor4f, glPointSize,
    glDisable, glEnable, GL_DEPTH_TEST, GL_POINTS
)

from .base_tool_3d import BaseTool3D
from ..modello_3d import trasforma_punto, detrasforma_punto

_HIT_RADIUS = 10   # pixel di tolleranza per il click
_VNORM_COL  = (0.85, 0.85, 0.85, 1.0) # Grigio chiaro per tutti i pallini


class ToolModifica(BaseTool3D):
    name = "modifica"

    def __init__(self):
        self._sel_v_idx   = None
        self._hover_v     = None
        self._drag_active = False
        self._drag_depth  = None
        self._drag_start_pos = None  # Salva la pos. iniziale per evitare i micro-click involontari

    # ------------------------------------------------------------------
    def on_activate(self, spazio):
        spazio.setCursor(Qt.CrossCursor)
        self._sel_v_idx = None

    def on_hover(self, spazio, event):
        if self._drag_active:
            return
        sel_id = spazio.get_id_selezionato()
        obj    = spazio.get_oggetto(sel_id)
        if obj is None:
            if self._hover_v is not None:
                self._hover_v = None; spazio.update()
            return
        new_hov = self._hit_vertex(event.x(), event.y(), obj, spazio)
        if new_hov != self._hover_v:
            self._hover_v = new_hov; spazio.update()

    # ------------------------------------------------------------------
    def on_mouse_press(self, spazio, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False

        mx, my = event.x(), event.y()
        sel_id = spazio.get_id_selezionato()
        obj    = spazio.get_oggetto(sel_id)

        if obj is None:
            picked = spazio._pick_at(mx, my)
            if picked != -1:
                spazio.set_id_selezionato(picked)
            return True

        hit_idx = self._hit_vertex(mx, my, obj, spazio)
        if hit_idx is not None:
            self._sel_v_idx   = hit_idx
            self._drag_active = True
            self._drag_start_pos = event.pos() # Registra dove abbiamo cliccato
            
            # Calcola la profondità per il drag 3D
            world_verts = obj.get_vertices_world()
            if hit_idx < len(world_verts):
                _, _, sz = self._world_to_screen(*world_verts[hit_idx], spazio)
                self._drag_depth = sz if sz is not None else 0.5
            spazio._notify_vertex_selected(sel_id, hit_idx)
            return True

        # Ctrl+click → aggiunge un punto (solo per armature)
        if (event.modifiers() & Qt.ControlModifier) and obj.tipo in ("barra", "staffa"):
            self._add_waypoint(obj)
            spazio._emit_modificato(sel_id)
            return True

        # Click nel vuoto → seleziona un altro oggetto (se presente) o deseleziona
        picked = spazio._pick_at(mx, my)
        spazio.set_id_selezionato(picked)
        self._sel_v_idx = None
        return True

    def on_mouse_move(self, spazio, event) -> bool:
        if not self._drag_active or self._sel_v_idx is None:
            return False
        if self._drag_depth is None:
            return False

        # ZONA MORTA: Ignora i micro-movimenti (jitter) minori di 5 pixel per non spaccare le facce 3D
        if self._drag_start_pos is not None:
            dist = (event.pos() - self._drag_start_pos).manhattanLength()
            if dist < 5:
                return False

        sel_id = spazio.get_id_selezionato()
        obj    = spazio.get_oggetto(sel_id)
        if obj is None:
            return False

        try:
            from OpenGL.GLU import gluUnProject
            dpr = spazio.devicePixelRatioF()
            mx  = event.x() * dpr
            my  = (spazio.height() * dpr) - (event.y() * dpr)
            new_world = gluUnProject(
                mx, my, self._drag_depth,
                spazio._gl_model_mat,
                spazio._gl_proj_mat,
                spazio._gl_viewport,
            )
        except Exception:
            return False

        local = detrasforma_punto(list(new_world), obj.posizione, obj.rotazione)
        verts = obj.get_vertices_local()
        idx   = self._sel_v_idx
        if idx >= len(verts):
            return False

        verts[idx] = local
        self._apply_verts(obj, verts)

        spazio._emit_modificato(sel_id)
        spazio._notify_vertex_selected(sel_id, idx)
        return True

    def on_mouse_release(self, spazio, event) -> bool:
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            self._drag_depth  = None
            self._drag_start_pos = None
            return self._sel_v_idx is not None
        return False

    def on_mouse_press_right(self, spazio, event) -> bool:
        mx, my = event.x(), event.y()
        sel_id = spazio.get_id_selezionato()
        obj    = spazio.get_oggetto(sel_id)
        if obj is None or obj.tipo not in ("barra", "staffa"):
            return False
        hit_idx = self._hit_vertex(mx, my, obj, spazio)
        if hit_idx is None:
            return False
        pts     = obj.geometria.get("punti", [])
        min_pts = 3 if obj.tipo == "staffa" else 2
        if len(pts) <= min_pts:
            return False
        pts.pop(hit_idx)
        obj.geometria["punti"] = pts
        if self._sel_v_idx == hit_idx:
            self._sel_v_idx = None
        spazio._emit_modificato(sel_id)
        return True

    # ------------------------------------------------------------------
    def _apply_verts(self, obj, verts):
        tipo = obj.tipo
        if tipo in ("barra", "staffa"):
            obj.geometria["punti"] = [list(v) for v in verts]
        else:
            obj.set_vertices_custom(verts)

    # ------------------------------------------------------------------
    def draw_overlay(self, spazio):
        sel_id = spazio.get_id_selezionato()
        if sel_id == -1:
            return
        obj = spazio.get_oggetto(sel_id)
        if obj is None:
            return

        verts_local = obj.get_vertices_local()
        
        # Effetto Raggi-X: disabilita la profondità così i pallini sono visibili anche attraverso i corpi
        glDisable(GL_DEPTH_TEST) 
        
        glColor4f(*_VNORM_COL)
        glPointSize(8.0)  # Tutti i pallini della stessa dimensione
        
        glBegin(GL_POINTS)
        for v in verts_local:
            wv = trasforma_punto(v, obj.posizione, obj.rotazione)
            glVertex3f(*wv)
        glEnd()
        
        glPointSize(1.0)
        glEnable(GL_DEPTH_TEST) # Riattiva per non disturbare il rendering globale

    def draw_labels(self, spazio, painter: QPainter):
        sel_id = spazio.get_id_selezionato()
        if sel_id == -1:
            return
        obj = spazio.get_oggetto(sel_id)
        if obj is None:
            return

        verts_local = obj.get_vertices_local()
        
        # Etichette pulite e minimaliste in grigio chiaro
        painter.setFont(QFont("Consolas", 9, QFont.Bold))
        painter.setPen(QColor(210, 210, 210)) 

        for i, v in enumerate(verts_local):
            wv = trasforma_punto(v, obj.posizione, obj.rotazione)
            sx, sy, _ = self._world_to_screen(*wv, spazio)
            if sx is None:
                continue
            painter.drawText(int(sx) + 8, int(sy) - 4, f"v{i+1}")

    # ------------------------------------------------------------------
    def _hit_vertex(self, mx, my, obj, spazio) -> int | None:
        best_idx = None; best_dist = _HIT_RADIUS
        for i, v in enumerate(obj.get_vertices_local()):
            wv = trasforma_punto(v, obj.posizione, obj.rotazione)
            sx, sy, _ = self._world_to_screen(*wv, spazio)
            if sx is None:
                continue
            d = math.hypot(mx-sx, my-sy)
            if d < best_dist:
                best_dist = d; best_idx = i
        return best_idx

    def _add_waypoint(self, obj):
        pts = obj.geometria.get("punti", [])
        last = pts[-1] if pts else [0, 0, 0]
        pts.append([last[0] + 1.0, last[1], last[2]])
        obj.geometria["punti"] = pts

    def reset(self):
        self._sel_v_idx   = None
        self._hover_v     = None
        self._drag_active = False
        self._drag_depth  = None
        self._drag_start_pos = None