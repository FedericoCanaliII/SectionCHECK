"""
elementi_spazio_3d.py – OpenGL 3D workspace for the Elements module.

Blender-style dark theme. Camera controls:
  Middle-button drag  → orbit
  Right-button drag   → pan
  Scroll wheel        → zoom
  Left click/drag     → tool interaction (select / gizmo)
"""

import math
import numpy as np

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore    import Qt, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui     import QPainter, QFont, QColor

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL, GL_ALWAYS,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
    GL_LINE_SMOOTH, GL_POINT_SMOOTH,
    GL_LINE_SMOOTH_HINT, GL_POINT_SMOOTH_HINT, GL_NICEST,
    GL_MODELVIEW, GL_PROJECTION,
    GL_LINE_LOOP, GL_LINES, GL_LINE_STRIP,
    GL_TRIANGLE_FAN, GL_QUADS, GL_QUAD_STRIP,
    GL_POINTS,
    GL_FALSE, GL_TRUE,
    GL_PROJECTION_MATRIX, GL_MODELVIEW_MATRIX, GL_VIEWPORT,
    glClearColor, glClear, glEnable, glDisable,
    glBlendFunc, glDepthFunc, glDepthMask,
    glLineWidth, glPointSize,
    glBegin, glEnd, glVertex3f, glColor3f, glColor4f,
    glMatrixMode, glLoadIdentity, glViewport, glOrtho,
    glTranslatef, glRotatef,
    glGetFloatv, glGetIntegerv,
    glPushMatrix, glPopMatrix,
    glHint,
)
from OpenGL.GLU import gluPerspective

try:
    from PIL import Image, ImageDraw, ImageFont as PILFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ── Palette Aggiornata ──────────────────────────────────────────────────────
_BG_R, _BG_G, _BG_B = 40/255, 40/255, 40/255
_PREV_BG_R, _PREV_BG_G, _PREV_BG_B = 50/255, 50/255, 50/255

_GF_R, _GF_G, _GF_B  = 0.20, 0.20, 0.20   # fine grid
_GC_R, _GC_G, _GC_B  = 0.27, 0.27, 0.27   # coarse grid

_AX_X = (0.862, 0.200, 0.200)
_AX_Y = (0.310, 0.620, 0.165)
_AX_Z = (0.161, 0.408, 0.784)

# Corpi molto più trasparenti (Alpha a 0.12 invece di 0.38)
_STRUCT_FILL = (140/255, 148/255, 162/255, 0.12) 
_STRUCT_EDGE = (185/255, 190/255, 205/255, 0.50) # Spigoli base

# Nuovi Colori Selezione e Armature
_GLOW_COL    = (1.00, 0.55, 0.00, 0.25)          # Arancione (riempimento selezione)
_SEL_EDGE    = (1.00, 0.60, 0.00, 1.00)          # Arancione acceso (linee selezione)
_BAR_COL     = (0.85, 0.25, 0.25, 1.00)          # Rosso (barre)
_STIR_COL    = (1.00, 0.90, 0.10, 1.00)          # Giallo (staffe - Richiesto!)
_REFV_COL    = (1.00, 0.12, 0.12, 1.00)

# Sphere / cylinder tessellation
_CYL_N = 24
_SPH_NL, _SPH_NM = 10, 20


class ElementiSpazio3D(QOpenGLWidget):
    """3-D OpenGL workspace for element geometry authoring."""

    selection_changed  = pyqtSignal(int)
    oggetto_modificato = pyqtSignal(int)

    # ------------------------------------------------------------------ init

    def __init__(self, parent=None):
        super().__init__(parent)

        self._oggetti: list       = []
        self._id_selezionato: int = -1
        self._active_tool         = None

        # Camera
        self.cam_dist: float = 12.0
        self.rot_x:    float = 25.0
        self.rot_y:    float = -40.0
        self.pan_x:    float = 0.0
        self.pan_y:    float = 0.0
        self._ortho:   bool  = False

        self._last_pos  = QPoint()
        self._mouse_btn = None

        # GL matrix cache (saved every paintGL for ray-casting / gizmos)
        self._gl_model_mat = None
        self._gl_proj_mat  = None
        self._gl_viewport  = None
        self._gl_ready     = False

        self._text_cache: dict = {}
        self._pil_font         = None
        self._preview_mode: bool = False   # when True: skip grid/axes (used by preview capture)

        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------ public API

    def aggiorna_oggetti(self, lista: list):
        self._oggetti = lista
        self._text_cache.clear()
        self.update()

    def get_oggetto(self, obj_id: int):
        for o in self._oggetti:
            if o.id == obj_id:
                return o
        return None

    def get_id_selezionato(self) -> int:
        return self._id_selezionato

    def set_id_selezionato(self, obj_id: int):
        self._id_selezionato = obj_id
        self.selection_changed.emit(obj_id)
        self.update()

    def _emit_modificato(self, obj_id: int):
        self.oggetto_modificato.emit(obj_id)
        self.update()

    def _notify_vertex_selected(self, obj_id: int, v_idx: int):
        self.oggetto_modificato.emit(obj_id)
        self.update()

    def set_active_tool(self, tool):
        if self._active_tool is not None:
            self._active_tool.on_deactivate(self)
        self._active_tool = tool
        if tool is not None:
            tool.on_activate(self)
        self.update()

    @property
    def active_tool(self):
        return self._active_tool

    def imposta_vista(self, preset: str):
        presets = {
            "3d": (25.0, -40.0, False),
            "x":  ( 0.0, -90.0, True),
            "y":  ( 0.0,   0.0, True),
            "z":  (90.0,   0.0, True),
        }
        if preset in presets:
            self.rot_x, self.rot_y, self._ortho = presets[preset]
            if self._ortho:
                self.pan_x = self.pan_y = 0.0
        self.update()

    def centra_vista(self):
        visible = [o for o in self._oggetti if o.visibile]
        if not visible:
            self.cam_dist = 12.0; self.pan_x = self.pan_y = 0.0
            self.update(); return
        pts = []
        for o in visible:
            pts.extend(o.get_vertices_world())
        if not pts:
            return
        pts  = np.array(pts, dtype=float)
        bbox = pts.max(axis=0) - pts.min(axis=0)
        self.cam_dist = max(float(np.max(bbox)) * 2.0, 2.0)
        ctr  = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
        self.pan_x = -float(ctr[0]); self.pan_y = -float(ctr[2])
        self.update()

    def reset_view(self):
        self.rot_x = 25.0; self.rot_y = -40.0
        self.pan_x = 0.0;  self.pan_y = 0.0
        self.cam_dist = 12.0; self._ortho = False
        self.update()

    # ------------------------------------------------------------------ GL init

    def initializeGL(self):
        glClearColor(_BG_R, _BG_G, _BG_B, 1.0)
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH); glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        if _PIL_OK:
            try:
                self._pil_font = PILFont.truetype("arial.ttf", 12)
            except Exception:
                self._pil_font = PILFont.load_default()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

    # ------------------------------------------------------------------ paint

    def paintGL(self):
        # --- AGGIUNGI QUESTE 4 RIGHE FONDAMENTALI ---
        # QPainter disabilita questi stati per disegnare il testo, 
        # dobbiamo forzare la loro riattivazione ad ogni frame!
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        # Reset critical GL state to known defaults every frame.
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
        glPointSize(1.0)

        # ---> MODIFICA QUI: Imposta il colore di sfondo in base al preview_mode <---
        if self._preview_mode:
            glClearColor(_PREV_BG_R, _PREV_BG_G, _PREV_BG_B, 1.0)
        else:
            glClearColor(_BG_R, _BG_G, _BG_B, 1.0)
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w = self.width() or 1; h = self.height() or 1
        aspect = w / h

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w = self.width() or 1; h = self.height() or 1
        aspect = w / h

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        if self._ortho:
            hh = self.cam_dist * 0.5; hw = hh * aspect
            glOrtho(-hw - self.pan_x,  hw - self.pan_x,
                    -hh - self.pan_y,  hh - self.pan_y,
                    -5000.0, 5000.0)
        else:
            gluPerspective(45.0, aspect, 0.05, 5000.0)

        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        if not self._ortho:
            glTranslatef(self.pan_x, self.pan_y, -self.cam_dist)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glRotatef(-90, 1, 0, 0)   # Z up

        # Cache GL state for picking / gizmo projection
        self._gl_model_mat = glGetFloatv(GL_MODELVIEW_MATRIX)
        self._gl_proj_mat  = glGetFloatv(GL_PROJECTION_MATRIX)
        self._gl_viewport  = glGetIntegerv(GL_VIEWPORT)
        self._gl_ready     = True

        if not self._preview_mode:
            self._disegna_griglia()
            self._disegna_assi()
        self._disegna_oggetti()

        if self._active_tool is not None:
            self._active_tool.draw_overlay(self)

        # Vertex labels (QPainter overlay) – must happen inside paintGL
        # to avoid GL state corruption from a separate paintEvent QPainter.
        from .tools.tool_modifica import ToolModifica
        if isinstance(self._active_tool, ToolModifica):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            self._active_tool.draw_labels(self, painter)
            painter.end()

    # ------------------------------------------------------------------ grid (identical to prendi_spunto.py)

    def _disegna_griglia(self):
        DIM   = 100
        FADE  = 40
        SEG_L = 2

        glLineWidth(1.0)
        glDepthMask(GL_FALSE)

        def get_alpha(x, y):
            dist = math.hypot(x, y) / FADE
            return max(0.0, 1.0 - dist * dist * 0.55)

        for step, cr, cg, cb in (
            (1, _GF_R, _GF_G, _GF_B),
            (5, _GC_R, _GC_G, _GC_B),
        ):
            glBegin(GL_LINES)
            for i in range(-DIM, DIM + 1, step):
                if step == 1 and (i % 5 == 0):
                    continue
                for j in range(-DIM, DIM, SEG_L):
                    # Vertical lines (x = i, y = j)
                    y1 = j; y2 = j + SEG_L
                    a1v = get_alpha(i, y1); a2v = get_alpha(i, y2)
                    if a1v > 0.01 or a2v > 0.01:
                        glColor4f(cr, cg, cb, a1v); glVertex3f(i, y1, 0)
                        glColor4f(cr, cg, cb, a2v); glVertex3f(i, y2, 0)
                    # Horizontal lines (y = i, x = j)
                    x1 = j; x2 = j + SEG_L
                    a1h = get_alpha(x1, i); a2h = get_alpha(x2, i)
                    if a1h > 0.01 or a2h > 0.01:
                        glColor4f(cr, cg, cb, a1h); glVertex3f(x1, i, 0)
                        glColor4f(cr, cg, cb, a2h); glVertex3f(x2, i, 0)
            glEnd()

        glDepthMask(GL_TRUE)

    # ------------------------------------------------------------------ axes

    def _disegna_assi(self):
        EXT = 40; NEG = -40
        glDepthMask(GL_FALSE)
        glLineWidth(1.8)
        glBegin(GL_LINES)
        glColor4f(*_AX_X, 1.0); glVertex3f(0,0,0); glVertex3f(EXT, 0,  0)
        glColor4f(*_AX_Y, 1.0); glVertex3f(0,0,0); glVertex3f(0,  EXT, 0)
        glColor4f(*_AX_Z, 1.0); glVertex3f(0,0,0); glVertex3f(0,  0,  EXT)
        glEnd()
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor4f(*_AX_X, 0.28); glVertex3f(0,0,0); glVertex3f(NEG, 0,   0)
        glColor4f(*_AX_Y, 0.28); glVertex3f(0,0,0); glVertex3f(0,  NEG,  0)
        glColor4f(*_AX_Z, 0.28); glVertex3f(0,0,0); glVertex3f(0,   0, NEG)
        glEnd()
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glColor4f(0.9, 0.9, 0.9, 0.85); glVertex3f(0, 0, 0)
        glEnd()
        glDepthMask(GL_TRUE)

    # ------------------------------------------------------------------ objects

    def _disegna_oggetti(self):
        sel = self._id_selezionato

        for obj in self._oggetti:
            if obj.visibile:
                self._disegna_oggetto(obj, obj.id == sel, glow_pass=False)

        # Selection highlight pass — standard alpha blend (no additive)
        # to keep the rendering style consistent regardless of active tool.
        if sel != -1:
            obj = self.get_oggetto(sel)
            if obj and obj.visibile:
                # Standard alpha blend; no GL_ONE additive to avoid wash-out
                self._disegna_oggetto(obj, True, glow_pass=True)

        # Ensure blend mode is always correct after object drawing
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _disegna_oggetto(self, obj, is_sel: bool, glow_pass: bool):
        glPushMatrix()
        glTranslatef(*obj.posizione)
        glRotatef(obj.rotazione[0], 1, 0, 0)
        glRotatef(obj.rotazione[1], 0, 1, 0)
        glRotatef(obj.rotazione[2], 0, 0, 1)

        if obj.tipo in ("parallelepipedo", "cilindro", "sfera"):
            self._draw_structural(obj, is_sel, glow_pass)
        elif obj.tipo == "barra":
            self._draw_barra(obj, is_sel, glow_pass)
        elif obj.tipo == "staffa":
            self._draw_staffa(obj, is_sel, glow_pass)

        glPopMatrix()

        # Reference vertex (always in world space, never in glow pass)
        if not glow_pass:
            rv = obj.get_vertex_ref_world()
            glPointSize(8.0)
            glBegin(GL_POINTS)
            glColor4f(*_REFV_COL)
            glVertex3f(*rv)
            glEnd()
            glPointSize(1.0)

    # ── structural ───────────────────────────────────────────────────

    def _draw_structural(self, obj, is_sel, glow_pass):
        # Assegnazione Colori
        alpha = _GLOW_COL[3] if glow_pass else _STRUCT_FILL[3]
        col   = (*_GLOW_COL[:3], alpha) if glow_pass else (*_STRUCT_FILL[:3], alpha)

        edge_col = _SEL_EDGE if is_sel else _STRUCT_EDGE
        line_w = 3.0 if is_sel else 2.0  # Spessore 2 di base, 3 se selezionato

        # 1. DISEGNO CORPO TRASPARENTE
        # glDepthMask(GL_FALSE) impedisce ai corpi di coprirsi tra loro in modo anomalo
        glDepthMask(GL_FALSE) 
        if obj.tipo == "parallelepipedo":
            self._draw_box_fill(obj, col)
        elif obj.tipo == "cilindro":
            self._draw_cyl_fill(obj, col)
        elif obj.tipo == "sfera":
            self._draw_sph_fill(obj, col)
        glDepthMask(GL_TRUE)

        # 2. DISEGNO LINEE DI COSTRUZIONE (Effetto Raggi X)
        if not glow_pass:
            # Spengiamo il Depth Test: le linee "bucheranno" lo schermo rendendo visibile l'interno!
            glDisable(GL_DEPTH_TEST) 
            
            glColor4f(*edge_col)
            glLineWidth(line_w)
            
            if obj.tipo == "parallelepipedo":
                self._draw_box_edges(obj)
            elif obj.tipo == "cilindro":
                self._draw_cyl_edges(obj)
            elif obj.tipo == "sfera":
                self._draw_sph_edges(obj)
                
            glLineWidth(1.0)
            # Riaccendiamo il Depth Test per gli altri oggetti
            glEnable(GL_DEPTH_TEST)

    # ── box ──────────────────────────────────────────────────────────

    def _get_box_verts(self, obj):
        """Always use the live vertex list (parametric or custom)."""
        v = obj.get_vertices_local()
        if len(v) < 8:
            from .modello_3d import calcola_vertici
            v = calcola_vertici("parallelepipedo", obj.geometria)
        return v

    def _draw_box_fill(self, obj, col):
        v = self._get_box_verts(obj)
        faces = [
            [0,1,2,3], [4,5,6,7],
            [0,1,5,4], [3,2,6,7],
            [0,3,7,4], [1,2,6,5],
        ]
        glColor4f(*col)
        glBegin(GL_QUADS)
        for face in faces:
            for idx in face:
                glVertex3f(*v[idx])
        glEnd()

    def _draw_box_edges(self, obj):
        v = self._get_box_verts(obj)
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]
        glBegin(GL_LINES)
        for a, b in edges:
            glVertex3f(*v[a]); glVertex3f(*v[b])
        glEnd()

    # ── cylinder ─────────────────────────────────────────────────────

    def _get_cyl_verts(self, obj):
        v = obj.get_vertices_local()
        N = _CYL_N
        needed = N * 2 + 3   # bottom_center + N_bottom + top_center + N_top
        if len(v) < needed:
            from .modello_3d import calcola_vertici
            v = calcola_vertici("cilindro", obj.geometria)
        return v

    def _draw_cyl_fill(self, obj, col):
        v = self._get_cyl_verts(obj)
        N = _CYL_N
        bc   = v[0]
        brim = v[1:N+1]
        tc   = v[N+1]
        trim = v[N+2:2*N+2]

        glColor4f(*col)
        # Bottom disc
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(*bc)
        for p in brim: glVertex3f(*p)
        glVertex3f(*brim[0])
        glEnd()
        # Top disc
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(*tc)
        for p in trim: glVertex3f(*p)
        glVertex3f(*trim[0])
        glEnd()
        # Side
        glBegin(GL_QUAD_STRIP)
        for i in range(N):
            glVertex3f(*brim[i]); glVertex3f(*trim[i])
        glVertex3f(*brim[0]); glVertex3f(*trim[0])
        glEnd()

    def _draw_cyl_edges(self, obj):
        v = self._get_cyl_verts(obj)
        N = _CYL_N
        brim = v[1:N+1]
        trim = v[N+2:2*N+2]

        glBegin(GL_LINE_LOOP)
        for p in brim: glVertex3f(*p)
        glEnd()
        glBegin(GL_LINE_LOOP)
        for p in trim: glVertex3f(*p)
        glEnd()
        # 4 vertical lines evenly spaced
        glBegin(GL_LINES)
        for i in (0, N//4, N//2, 3*N//4):
            glVertex3f(*brim[i]); glVertex3f(*trim[i])
        glEnd()

    # ── sphere ───────────────────────────────────────────────────────

    def _get_sph_verts(self, obj):
        v = obj.get_vertices_local()
        NL, NM = _SPH_NL, _SPH_NM
        needed = 1 + (NL+1) * NM
        if len(v) < needed:
            from .modello_3d import calcola_vertici
            # calcola_vertici uses NL=8,NM=12; for rendering use live parametric
            v = self._sph_verts_parametric(obj)
        return v

    def _sph_verts_parametric(self, obj):
        """Build sphere vertices with _SPH_NL/_SPH_NM tessellation."""
        R  = float(obj.geometria.get("raggio", 1.5))
        NL, NM = _SPH_NL, _SPH_NM
        verts = [[0.0, 0.0, 0.0]]
        for il in range(NL + 1):
            phi = math.pi * (il / NL - 0.5)
            for im in range(NM):
                theta = 2 * math.pi * im / NM
                verts.append([
                    R * math.cos(phi) * math.cos(theta),
                    R * math.cos(phi) * math.sin(theta),
                    R * math.sin(phi),
                ])
        return verts

    def _draw_sph_fill(self, obj, col):
        v  = self._sph_verts_parametric(obj) if not obj.custom_geometry else self._get_sph_verts(obj)
        NL = _SPH_NL; NM = _SPH_NM

        glColor4f(*col)
        for il in range(NL):
            glBegin(GL_QUAD_STRIP)
            for im in range(NM + 1):
                i0 = 1 + il       * NM + (im % NM)
                i1 = 1 + (il + 1) * NM + (im % NM)
                glVertex3f(*v[i0])
                glVertex3f(*v[i1])
            glEnd()

    def _draw_sph_edges(self, obj):
        v  = self._sph_verts_parametric(obj) if not obj.custom_geometry else self._get_sph_verts(obj)
        NL = _SPH_NL; NM = _SPH_NM

        # Latitude circles (every 2 rings)
        for il in range(0, NL + 1, 2):
            glBegin(GL_LINE_LOOP)
            for im in range(NM):
                glVertex3f(*v[1 + il * NM + im])
            glEnd()

        # Longitude lines (every 4)
        step = max(1, NM // 8)
        for im in range(0, NM, step):
            glBegin(GL_LINE_STRIP)
            for il in range(NL + 1):
                glVertex3f(*v[1 + il * NM + im])
            glEnd()

    # ── reinforcement ────────────────────────────────────────────────

    def _draw_barra(self, obj, is_sel, glow_pass):
        col = (*_GLOW_COL[:3], 0.7) if glow_pass else _BAR_COL
        pts = obj.geometria.get("punti", [])
        if len(pts) < 2:
            return
        glColor4f(*col)
        glLineWidth(3.0 if is_sel else 2.2)
        glBegin(GL_LINE_STRIP)
        for p in pts: glVertex3f(*p)
        glEnd()
        glLineWidth(1.0)

    def _draw_staffa(self, obj, is_sel, glow_pass):
        col = (*_GLOW_COL[:3], 0.7) if glow_pass else _STIR_COL
        pts = obj.geometria.get("punti", [])
        if len(pts) < 3:
            return
        glColor4f(*col)
        glLineWidth(3.0 if is_sel else 2.2)
        glBegin(GL_LINE_LOOP)
        for p in pts: glVertex3f(*p)
        glEnd()
        glLineWidth(1.0)

    # ------------------------------------------------------------------ picking (ray cast)

    def _pick_at(self, mx: int, my: int) -> int:
        if not self._gl_ready:
            return -1
        
        try:
            import numpy as np
            
            w = self.width() or 1
            h = self.height() or 1
            vp = self._gl_viewport
            
            gmx = float(mx) * (float(vp[2]) / float(w))
            gmy = float(vp[3]) - (float(my) * (float(vp[3]) / float(h)))

            mm = np.array(self._gl_model_mat, dtype=float).reshape(4, 4).T
            pm = np.array(self._gl_proj_mat, dtype=float).reshape(4, 4).T
            
            inv_matrix = np.linalg.inv(pm @ mm)
            
            def unproject(win_x, win_y, win_z):
                ndc_x = (win_x - vp[0]) / vp[2] * 2.0 - 1.0
                ndc_y = (win_y - vp[1]) / vp[3] * 2.0 - 1.0
                ndc_z = 2.0 * win_z - 1.0
                point_ndc = np.array([ndc_x, ndc_y, ndc_z, 1.0])
                point_world = inv_matrix @ point_ndc
                if point_world[3] == 0.0: return point_world[:3]
                return point_world[:3] / point_world[3]

            near = unproject(gmx, gmy, 0.0)
            far  = unproject(gmx, gmy, 1.0)
            
            ray_d = far - near
            n = np.linalg.norm(ray_d)
            if n < 1e-10:
                return -1
            ray_d /= n

            # =========================================================
            # MATEMATICA DI PRECISIONE: Distanza tra Raggio e Segmento
            # =========================================================
            def dist_ray_segment(p0, d, q0, q1):
                """Calcola la distanza minima tra una retta (raggio) e un segmento 3D."""
                v = q1 - q0
                w0 = p0 - q0
                a = 1.0 # La dir del raggio è già normalizzata
                b = np.dot(d, v)
                c = np.dot(v, v)
                d_dot = np.dot(d, w0)
                e = np.dot(v, w0)
                
                D = a * c - b * b
                
                if D < 1e-8: # Linee quasi parallele
                    tc = 0.0 if b > c else (e / b if b != 0 else 0.0)
                    tc = np.clip(tc, 0.0, 1.0)
                    sc = b * tc - d_dot
                else:
                    sc = (b * e - c * d_dot) / D
                    tc = (a * e - b * d_dot) / D
                    
                    if tc < 0.0:
                        tc = 0.0
                        sc = -d_dot
                    elif tc > 1.0:
                        tc = 1.0
                        sc = b - d_dot
                        
                if sc < 0.0: # Il raggio non può andare all'indietro
                    sc = 0.0
                    tc = e / c if c > 0 else 0.0
                    tc = np.clip(tc, 0.0, 1.0)
                    
                punto_su_raggio = p0 + sc * d
                punto_su_segmento = q0 + tc * v
                dist = np.linalg.norm(punto_su_raggio - punto_su_segmento)
                return dist, sc

            # =========================================================
            # FUNZIONI DI VERIFICA (Armature vs Corpi)
            # =========================================================
            def hit_armatura(obj, vw):
                best_t = 1e10
                is_hit = False
                
                # Tolleranza: il raggio del "cilindro invisibile" di selezione
                # 0.15 = 15cm nello spazio. Modificalo per rendere la barra più o meno facile da cliccare!
                tolleranza = 0.02
                
                # Controlla ogni singolo segmento della barra/staffa
                for i in range(len(vw) - 1):
                    dist, t = dist_ray_segment(near, ray_d, vw[i], vw[i+1])
                    if dist < tolleranza and t < best_t:
                        best_t = t
                        is_hit = True
                        
                # Se è una staffa chiusa ad anello, controlla l'ultimo segmento che si chiude col primo
                if obj.tipo == "staffa" and len(vw) > 2:
                    dist, t = dist_ray_segment(near, ray_d, vw[-1], vw[0])
                    if dist < tolleranza and t < best_t:
                        best_t = t
                        is_hit = True
                        
                return is_hit, best_t

            def hit_corpo(obj, vw):
                ray_d_safe = np.where(np.abs(ray_d) < 1e-8, 1e-8, ray_d)
                # Piccolissimo margine per i corpi strutturali (3cm)
                min_b = vw.min(axis=0) - 0.003
                max_b = vw.max(axis=0) + 0.003
                
                t1 = (min_b - near) / ray_d_safe
                t2 = (max_b - near) / ray_d_safe
                
                t_min_asse = np.minimum(t1, t2)
                t_max_asse = np.maximum(t1, t2)
                
                t_near = np.max(t_min_asse) 
                t_far  = np.min(t_max_asse) 
                
                if t_near > t_far or t_far < 0:
                    return False, 1e10
                    
                return True, (t_near if t_near >= 0 else t_far)

            # =========================================================
            # ESECUZIONE DELLA RICERCA
            # =========================================================
            best_id_armatura = -1
            min_t_armatura = 1e10
            
            best_id_corpo = -1
            min_t_corpo = 1e10
            
            for obj in self._oggetti:
                if not obj.visibile or not obj.selezionabile:
                    continue
                vw = np.array(obj.get_vertices_world(), dtype=float)
                if len(vw) == 0:
                    continue
                    
                if obj.tipo in ("barra", "staffa"):
                    colpito, t = hit_armatura(obj, vw)
                    if colpito and t < min_t_armatura:
                        min_t_armatura = t
                        best_id_armatura = obj.id
                else:
                    colpito, t = hit_corpo(obj, vw)
                    if colpito and t < min_t_corpo:
                        min_t_corpo = t
                        best_id_corpo = obj.id

            # Priorità assoluta alle armature!
            if best_id_armatura != -1:
                return best_id_armatura
                
            return best_id_corpo
            
        except Exception as e:
            print(f"\n--- ERRORE NEL RAYCASTING ---\n{e}\n")
            return -1

    # ------------------------------------------------------------------ mouse / keyboard

    def mousePressEvent(self, event):
        self._last_pos  = event.pos()
        self._mouse_btn = event.button()

        if event.button() == Qt.LeftButton:
            # 1. Offer to active tool first (gizmo / vertex interaction)
            if self._active_tool is not None:
                if self._active_tool.on_mouse_press(self, event):
                    return
            # 2. No tool (or tool didn't consume): pick object on click
            picked = self._pick_at(event.x(), event.y())
            if picked != -1:
                self.set_id_selezionato(picked)
            # Empty click: keep current selection (do nothing)

        elif event.button() == Qt.RightButton:
            # Right-click in vertex-edit mode for waypoint removal
            from .tools.tool_modifica import ToolModifica
            if isinstance(self._active_tool, ToolModifica):
                self._active_tool.on_mouse_press_right(self, event)

    def mouseMoveEvent(self, event):
        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()

        # Tool drag (left button only)
        if self._mouse_btn == Qt.LeftButton and self._active_tool is not None:
            if self._active_tool.on_mouse_move(self, event):
                self._last_pos = event.pos()
                return

        # Hover feedback (no button pressed or not dragging through tool)
        if self._active_tool is not None and hasattr(self._active_tool, "on_hover"):
            self._active_tool.on_hover(self, event)

        # Camera orbit (middle button)
        if self._mouse_btn == Qt.MiddleButton:
            self.rot_y += dx * 0.40
            self.rot_x += dy * 0.40
            self._last_pos = event.pos()
            self.update()
            return

        # Pan (right button)
        if self._mouse_btn == Qt.RightButton:
            speed = self.cam_dist * 0.0018
            self.pan_x += dx * speed
            self.pan_y -= dy * speed
            self._last_pos = event.pos()
            self.update()
            return

        self._last_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if self._active_tool is not None:
            self._active_tool.on_mouse_release(self, event)
        self._mouse_btn = None

    def wheelEvent(self, event):
        if self._active_tool is not None and self._active_tool.on_wheel(self, event):
            return
        delta  = event.angleDelta().y()
        factor = 0.88 if delta > 0 else 1.12
        self.cam_dist = max(0.5, self.cam_dist * factor)
        self.update()
