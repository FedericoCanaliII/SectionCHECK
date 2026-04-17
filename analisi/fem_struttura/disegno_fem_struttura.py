"""
disegno_fem_struttura.py
------------------------
Widget OpenGL per la visualizzazione dei risultati dell'analisi FEM
della struttura a telaio.

Conforme al sistema di riferimento di struttura_spazio_3d.py:
  - Z verticale (up) nel sistema strutturale
  - glRotatef(-90, 1, 0, 0) converte Z-up → Y-up per OpenGL
  - Griglia sul piano XY (appare orizzontale dopo la rotazione)
  - Pan in prospettiva applicato PRIMA delle rotazioni
  - Pan in ortho integrato nella glOrtho

Modalita' di visualizzazione:
  indeformata | N | Vy | Vz | My | Mz | deformata (+tensioni)

Controlli camera (stile Blender):
  Middle-drag  → orbit
  Right-drag   → pan
  Scroll       → zoom
"""
from __future__ import annotations

import math
import numpy as np
from ctypes import c_void_p

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import (QPainter, QFont, QColor, QPen, QBrush,
                          QLinearGradient)

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINE_SMOOTH, GL_POINT_SMOOTH,
    GL_LINE_SMOOTH_HINT, GL_POINT_SMOOTH_HINT, GL_NICEST,
    GL_MODELVIEW, GL_PROJECTION,
    GL_LINES, GL_TRIANGLES, GL_POINTS,
    GL_FALSE, GL_TRUE, GL_FLOAT,
    glClearColor, glClear, glEnable, glDisable,
    glBlendFunc, glDepthFunc, glDepthMask,
    glLineWidth, glPointSize,
    glMatrixMode, glLoadIdentity, glViewport, glOrtho,
    glTranslatef, glRotatef,
    glHint,
    glGenBuffers, glDeleteBuffers, glBindBuffer, glBufferData,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW,
    glEnableClientState, glDisableClientState,
    GL_VERTEX_ARRAY, GL_COLOR_ARRAY,
    glVertexPointer, glColorPointer,
    glDrawArrays,
)
from OpenGL.GLU import gluPerspective

_VBO_0 = c_void_p(0)

# ── Sfondo ──────────────────────────────────────────────────────────
_BG_R, _BG_G, _BG_B = 40/255, 40/255, 40/255

# ── Griglia ─────────────────────────────────────────────────────────
_GF_R, _GF_G, _GF_B = 0.20, 0.20, 0.20   # fine
_GC_R, _GC_G, _GC_B = 0.27, 0.27, 0.27   # grossa

# ── Assi globali ────────────────────────────────────────────────────
_AX_X = (0.862, 0.200, 0.200)
_AX_Y = (0.310, 0.620, 0.165)
_AX_Z = (0.161, 0.408, 0.784)

# ── Oggetti strutturali ─────────────────────────────────────────────
_COL_NODO        = (0.39, 0.71, 1.00, 0.90)
_COL_NODO_VINC   = (0.40, 0.75, 0.90, 1.00)
_COL_BEAM        = (0.80, 0.80, 0.80, 1.00)
_COL_SHELL_FILL  = (0.50, 0.70, 0.50, 0.18)
_COL_SHELL_EDGE  = (0.55, 0.75, 0.55, 0.70)
_COL_VINCOLO     = (0.30, 0.85, 0.40, 0.70)

# ── Risultati ───────────────────────────────────────────────────────
_COL_DEFORMATA   = (0.30, 0.75, 1.00, 1.00)
_COL_N_POS       = (0.20, 0.60, 1.00, 0.40)   # blu  – trazione
_COL_N_NEG       = (1.00, 0.35, 0.20, 0.40)   # rosso – compressione
_COL_V           = (0.20, 0.85, 0.35, 0.40)   # verde – taglio
_COL_M           = (0.90, 0.55, 0.15, 0.40)   # arancio – momento

# ── Tensioni (gradiente blu→verde→rosso) ────────────────────────────
_STRESS_LOW  = np.array([0.10, 0.30, 0.85], dtype=np.float32)
_STRESS_MID  = np.array([0.20, 0.80, 0.20], dtype=np.float32)
_STRESS_HIGH = np.array([0.90, 0.20, 0.15], dtype=np.float32)

# ── Dimensioni ──────────────────────────────────────────────────────
_NODE_SIZE   = 7.0
_BEAM_WIDTH  = 2.5
_VINCOLO_SZ  = 0.30


# ==============================================================================
#  WIDGET
# ==============================================================================

class FEMStrutturaSpazio3D(QOpenGLWidget):
    """Viewer 3D per i risultati dell'analisi FEM della struttura.

    Sistema di coordinate: Z verticale (up), identico a struttura_spazio_3d.py.
    La rotazione -90° attorno a X in paintGL converte al sistema OpenGL (Y-up).
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Camera ────────────────────────────────────────────────
        self.cam_dist: float = 20.0
        self.rot_x:    float = 25.0
        self.rot_y:    float = -40.0
        self.pan_x:    float = 0.0
        self.pan_y:    float = 0.0
        self._ortho:   bool  = False

        self._last_pos  = QPoint()
        self._mouse_btn = None

        # ── Dati ──────────────────────────────────────────────────
        self._dati      = None   # dict parse_struttura
        self._mesh      = None   # MeshStruttura
        self._risultati = None   # RisultatiFEMStruttura

        # ── Modalita' ─────────────────────────────────────────────
        self._modo:               str   = "indeformata"
        self._mostra_tensioni:    bool  = False
        self._scala_deformazione: float = 1.0

        # ── GPU buffers ───────────────────────────────────────────
        self._gpu: dict[str, tuple] = {}
        self._dirty_static    = True
        self._dirty_scena     = True
        self._dirty_risultati = True
        self._gl_ready        = False

        # ── Legenda tensioni ──────────────────────────────────────
        self._sigma_max_vis: float = 1.0

        self.setFocusPolicy(Qt.StrongFocus)

    # ================================================================
    #  INTERFACCIA PUBBLICA
    # ================================================================

    def set_dati(self, dati):
        self._dati = dati
        self._dirty_scena = True
        self._dirty_risultati = True
        self.update()

    def set_mesh(self, mesh):
        self._mesh = mesh
        self._dirty_risultati = True
        self.update()

    def set_risultati(self, risultati):
        self._risultati = risultati
        self._dirty_risultati = True
        self.update()

    def set_modo(self, modo: str):
        self._modo = modo
        self._dirty_risultati = True
        self.update()

    def set_mostra_tensioni(self, attivo: bool):
        self._mostra_tensioni = attivo
        self._dirty_risultati = True
        self.update()

    def set_scala_deformazione(self, scala: float):
        self._scala_deformazione = max(0.0, scala)
        self._dirty_risultati = True
        self.update()

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
        """Centra la vista sulla struttura, tenendo conto della rotazione -90°."""
        if not self._dati:
            self.cam_dist = 20.0
            self.pan_x = self.pan_y = 0.0
            self.update()
            return
        nodi = self._dati.get("nodi", {})
        if not nodi:
            self.cam_dist = 20.0
            self.pan_x = self.pan_y = 0.0
            self.update()
            return

        pts = np.array(list(nodi.values()), dtype=float)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        ctr  = (mn + mx) / 2.0
        diag = float(np.linalg.norm(mx - mn))

        # Calcola il centro in eye-space considerando le 3 rotazioni:
        #   Rx(rot_x) @ Ry(rot_y) @ Rx(-90)
        def Rx(deg):
            a = math.radians(deg)
            c, s = math.cos(a), math.sin(a)
            return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)
        def Ry(deg):
            a = math.radians(deg)
            c, s = math.cos(a), math.sin(a)
            return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

        R = Rx(self.rot_x) @ Ry(self.rot_y) @ Rx(-90.0)
        ctr_eye = R @ ctr
        self.pan_x    = float(-ctr_eye[0])
        self.pan_y    = float(-ctr_eye[1])
        self.cam_dist = max(diag / (2.0 * math.tan(math.radians(22.5))) * 1.25, 2.0)
        self.update()

    # ================================================================
    #  GPU BUFFER MANAGEMENT
    # ================================================================

    def _gpu_upload(self, name: str, verts_f32: np.ndarray,
                    colors_f32: np.ndarray) -> None:
        self._gpu_release(name)
        if verts_f32 is None or verts_f32.size == 0:
            return
        verts_f32  = np.ascontiguousarray(verts_f32,  dtype=np.float32)
        colors_f32 = np.ascontiguousarray(colors_f32, dtype=np.float32)
        n = verts_f32.size // 3
        if n == 0:
            return
        vbo_v = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo_v)
        glBufferData(GL_ARRAY_BUFFER, verts_f32.nbytes, verts_f32, GL_STATIC_DRAW)
        vbo_c = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo_c)
        glBufferData(GL_ARRAY_BUFFER, colors_f32.nbytes, colors_f32, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._gpu[name] = (vbo_v, vbo_c, n)

    def _gpu_draw(self, name: str, mode: int) -> None:
        buf = self._gpu.get(name)
        if not buf or buf[2] == 0:
            return
        vbo_v, vbo_c, n = buf
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_v)
        glVertexPointer(3, GL_FLOAT, 0, _VBO_0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_c)
        glColorPointer(4, GL_FLOAT, 0, _VBO_0)
        glDrawArrays(mode, 0, n)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _gpu_release(self, name: str) -> None:
        buf = self._gpu.pop(name, None)
        if buf:
            try:
                glDeleteBuffers(2, [buf[0], buf[1]])
            except Exception:
                pass

    def _gpu_release_group(self, prefix: str) -> None:
        for k in [n for n in self._gpu if n.startswith(prefix)]:
            self._gpu_release(k)

    # ================================================================
    #  OPENGL LIFECYCLE
    # ================================================================

    def initializeGL(self):
        glClearColor(_BG_R, _BG_G, _BG_B, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        self._gl_ready = True

    def resizeGL(self, w, h):
        glViewport(0, 0, max(w, 1), max(h, 1))

    def paintGL(self):
        if not self._gl_ready:
            return

        glClearColor(_BG_R, _BG_G, _BG_B, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
        glPointSize(1.0)

        w = self.width()  or 1
        h = self.height() or 1
        aspect = w / h

        # ── Proiezione ────────────────────────────────────────────
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self._ortho:
            hh = self.cam_dist * 0.5
            hw = hh * aspect
            # Pan integrato in glOrtho (identico all'originale)
            glOrtho(-hw - self.pan_x,  hw - self.pan_x,
                    -hh - self.pan_y,  hh - self.pan_y,
                    -5000.0, 5000.0)
        else:
            gluPerspective(45.0, aspect, 0.05, 5000.0)

        # ── Camera (identico a struttura_spazio_3d.py) ────────────
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if not self._ortho:
            # Pan PRIMA delle rotazioni (eye space)
            glTranslatef(self.pan_x, self.pan_y, -self.cam_dist)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        # Fondamentale: converte Z-up → Y-up per OpenGL
        glRotatef(-90.0, 1, 0, 0)

        # ── Rebuild VBO se dirty ──────────────────────────────────
        if self._dirty_static:
            self._build_griglia()
            self._build_assi()
            self._dirty_static = False

        if self._dirty_scena:
            self._build_scena()
            self._dirty_scena = False

        if self._dirty_risultati:
            self._build_risultati()
            self._dirty_risultati = False

        # ── Draw: griglia ─────────────────────────────────────────
        glDepthMask(GL_FALSE)
        glLineWidth(1.0)
        self._gpu_draw('griglia', GL_LINES)
        glDepthMask(GL_TRUE)

        # ── Draw: assi ────────────────────────────────────────────
        glLineWidth(1.8)
        self._gpu_draw('assi_pos', GL_LINES)
        glLineWidth(1.0)
        self._gpu_draw('assi_neg', GL_LINES)
        glPointSize(5.0)
        self._gpu_draw('assi_orig', GL_POINTS)
        glPointSize(1.0)

        # ── Draw: struttura base ──────────────────────────────────
        modo_corrente = self._modo
        in_diagramma = modo_corrente in ("N", "Vy", "Vz", "My", "Mz")
        in_deformata = modo_corrente == "deformata"
        tensioni_attive = (self._mostra_tensioni and in_deformata
                           and self._risultati and self._risultati.successo)

        # Shell base (verdi): nascoste nei diagrammi (non pertinenti alle aste),
        # quando le tensioni sono attive (sostituite dall'overlay colorato),
        # e in modo deformata (sostituite dalle shell deformate verdoline)
        disegna_shell_base = ((not in_diagramma) and (not tensioni_attive)
                              and (not in_deformata))
        if disegna_shell_base:
            glDepthMask(GL_FALSE)
            self._gpu_draw('scena_shell_fill', GL_TRIANGLES)
            glDepthMask(GL_TRUE)
            glLineWidth(1.5)
            self._gpu_draw('scena_shell_edge', GL_LINES)
            glLineWidth(1.0)

        # Aste: nascoste quando la deformata e' attiva (sostituita dai beam
        # deformati, in colore tensione o blu)
        if not in_deformata:
            glLineWidth(_BEAM_WIDTH)
            self._gpu_draw('scena_beams', GL_LINES)
            glLineWidth(1.0)

        # Vincoli (fill + bordi)
        glDepthMask(GL_FALSE)
        self._gpu_draw('scena_vincoli_fill', GL_TRIANGLES)
        glDepthMask(GL_TRUE)
        glLineWidth(1.5)
        self._gpu_draw('scena_vincoli', GL_LINES)
        glLineWidth(1.0)

        # Nodi originali (sopra tutto): nascosti in modo deformata,
        # dove sono sostituiti dai nodi deformati (blu)
        if not in_deformata:
            glPointSize(_NODE_SIZE)
            self._gpu_draw('scena_nodi_free', GL_POINTS)
            self._gpu_draw('scena_nodi_vinc', GL_POINTS)
            glPointSize(1.0)

        # ── Draw: risultati ───────────────────────────────────────
        if self._risultati and self._risultati.successo:
            if in_diagramma:
                glDepthMask(GL_FALSE)
                self._gpu_draw('diagramma_fill', GL_TRIANGLES)
                glDepthMask(GL_TRUE)
                glLineWidth(1.8)
                self._gpu_draw('diagramma_line', GL_LINES)
                glLineWidth(1.0)
            elif in_deformata:
                # Shell deformate (verdolino semitrasparente): disegnate prima
                # dei beam quando le tensioni non sono attive. In caso contrario
                # sono sostituite dall'overlay colorato (tensioni_fill/edge).
                if not tensioni_attive:
                    glDepthMask(GL_FALSE)
                    self._gpu_draw('deformata_shell_fill', GL_TRIANGLES)
                    glDepthMask(GL_TRUE)
                    glLineWidth(1.5)
                    self._gpu_draw('deformata_shell_edge', GL_LINES)
                    glLineWidth(1.0)

                # Beam deformati: colorati per tensione se attivo, altrimenti blu
                glLineWidth(_BEAM_WIDTH)
                if tensioni_attive:
                    self._gpu_draw('deformata_beam_tens', GL_LINES)
                else:
                    self._gpu_draw('deformata_beam', GL_LINES)
                glLineWidth(1.0)
                glPointSize(5.0)
                self._gpu_draw('deformata_nodi', GL_POINTS)
                glPointSize(1.0)

            # Tensioni shell (overlay): solo in modo deformata
            if tensioni_attive:
                glDepthMask(GL_FALSE)
                self._gpu_draw('tensioni_fill', GL_TRIANGLES)
                glDepthMask(GL_TRUE)
                glLineWidth(1.0)
                self._gpu_draw('tensioni_edge', GL_LINES)
                glLineWidth(1.0)

        # ── Overlay 2D: legenda tensioni (solo in deformata) ──────
        if tensioni_attive:
            glDisable(GL_DEPTH_TEST)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            try:
                self._draw_legend(painter, w, h)
            finally:
                painter.end()

    # ================================================================
    #  BUILD: GRIGLIA (piano XY, Z=0, come nell'originale)
    # ================================================================

    def _build_griglia(self):
        """Griglia sul piano XY — la rotazione -90° la porta orizzontale."""
        DIM, FADE, SEG_L = 100, 40, 2
        parts_v, parts_c = [], []

        for step, cr, cg, cb in ((1, _GF_R, _GF_G, _GF_B),
                                  (5, _GC_R, _GC_G, _GC_B)):
            ii = np.arange(-DIM, DIM + 1, step, dtype=np.float32)
            if step == 1:
                ii = ii[ii.astype(np.int32) % 5 != 0]
            jj  = np.arange(-DIM, DIM, SEG_L, dtype=np.float32)
            jj2 = jj + SEG_L
            I,  J  = np.meshgrid(ii, jj,  indexing='ij')
            _,  J2 = np.meshgrid(ii, jj2, indexing='ij')

            # Linee lungo Y (x=I costante)
            d1 = np.hypot(I, J ) / FADE
            d2 = np.hypot(I, J2) / FADE
            a1 = np.maximum(0.0, 1.0 - d1*d1*0.55).astype(np.float32)
            a2 = np.maximum(0.0, 1.0 - d2*d2*0.55).astype(np.float32)
            mask = (a1 > 0.01) | (a2 > 0.01)
            n = int(mask.sum())
            if n > 0:
                v = np.zeros((n * 2, 3), dtype=np.float32)
                v[0::2, 0] = I[mask];  v[0::2, 1] = J[mask];  v[0::2, 2] = 0
                v[1::2, 0] = I[mask];  v[1::2, 1] = J2[mask]; v[1::2, 2] = 0
                c = np.empty((n * 2, 4), dtype=np.float32)
                c[0::2] = [cr, cg, cb, 0]; c[0::2, 3] = a1[mask]
                c[1::2] = [cr, cg, cb, 0]; c[1::2, 3] = a2[mask]
                parts_v.append(v.ravel()); parts_c.append(c.ravel())

            # Linee lungo X (y=I costante)
            d1 = np.hypot(J,  I) / FADE
            d2 = np.hypot(J2, I) / FADE
            a1 = np.maximum(0.0, 1.0 - d1*d1*0.55).astype(np.float32)
            a2 = np.maximum(0.0, 1.0 - d2*d2*0.55).astype(np.float32)
            mask = (a1 > 0.01) | (a2 > 0.01)
            n = int(mask.sum())
            if n > 0:
                v = np.zeros((n * 2, 3), dtype=np.float32)
                v[0::2, 0] = J[mask];  v[0::2, 1] = I[mask];  v[0::2, 2] = 0
                v[1::2, 0] = J2[mask]; v[1::2, 1] = I[mask];  v[1::2, 2] = 0
                c = np.empty((n * 2, 4), dtype=np.float32)
                c[0::2] = [cr, cg, cb, 0]; c[0::2, 3] = a1[mask]
                c[1::2] = [cr, cg, cb, 0]; c[1::2, 3] = a2[mask]
                parts_v.append(v.ravel()); parts_c.append(c.ravel())

        if parts_v:
            self._gpu_upload('griglia',
                             np.concatenate(parts_v),
                             np.concatenate(parts_c))

    # ================================================================
    #  BUILD: ASSI (±40, identici all'originale)
    # ================================================================

    def _build_assi(self):
        EXT, NEG = 40.0, -40.0

        v_pos = np.array([
            0, 0, 0, EXT,  0,  0,
            0, 0, 0,  0,  EXT, 0,
            0, 0, 0,  0,   0, EXT,
        ], dtype=np.float32)
        c_pos = np.array([
            *_AX_X, 1, *_AX_X, 1,
            *_AX_Y, 1, *_AX_Y, 1,
            *_AX_Z, 1, *_AX_Z, 1,
        ], dtype=np.float32)
        self._gpu_upload('assi_pos', v_pos, c_pos)

        v_neg = np.array([
            0, 0, 0, NEG,  0,   0,
            0, 0, 0,  0,  NEG,  0,
            0, 0, 0,  0,   0,  NEG,
        ], dtype=np.float32)
        c_neg = np.array([
            *_AX_X, 0.28, *_AX_X, 0.28,
            *_AX_Y, 0.28, *_AX_Y, 0.28,
            *_AX_Z, 0.28, *_AX_Z, 0.28,
        ], dtype=np.float32)
        self._gpu_upload('assi_neg', v_neg, c_neg)

        v_o = np.array([0, 0, 0], dtype=np.float32)
        c_o = np.array([0.9, 0.9, 0.9, 0.85], dtype=np.float32)
        self._gpu_upload('assi_orig', v_o, c_o)

    # ================================================================
    #  BUILD: SCENA (struttura non deformata)
    # ================================================================

    def _build_scena(self):
        self._gpu_release_group('scena_')

        if not self._dati:
            return
        nodi    = self._dati.get("nodi", {})
        aste    = self._dati.get("aste", {})
        shells  = self._dati.get("shell", {})
        vincoli = self._dati.get("vincoli", {})

        if not nodi:
            return

        # ── Nodi liberi / vincolati ────────────────────────────────
        nids   = list(nodi.keys())
        coords = np.array([nodi[nid] for nid in nids], dtype=np.float32)
        is_vinc = np.array([nid in vincoli for nid in nids], dtype=bool)

        if np.any(~is_vinc):
            pts = coords[~is_vinc]
            col = np.tile(np.array(_COL_NODO, dtype=np.float32), (len(pts), 1))
            self._gpu_upload('scena_nodi_free', pts, col)
        if np.any(is_vinc):
            pts = coords[is_vinc]
            col = np.tile(np.array(_COL_NODO_VINC, dtype=np.float32), (len(pts), 1))
            self._gpu_upload('scena_nodi_vinc', pts, col)

        # ── Aste ──────────────────────────────────────────────────
        if aste:
            bv = []
            for a in aste.values():
                ni, nj = a["nodo_i"], a["nodo_j"]
                if ni in nodi and nj in nodi:
                    bv.append(nodi[ni]); bv.append(nodi[nj])
            if bv:
                V = np.array(bv, dtype=np.float32)
                C = np.tile(np.array(_COL_BEAM, dtype=np.float32), (len(V), 1))
                self._gpu_upload('scena_beams', V, C)

        # ── Shell ─────────────────────────────────────────────────
        if shells:
            tri_v, edge_v = [], []
            for sh in shells.values():
                pts = [nodi[n] for n in sh["nodi"] if n in nodi]
                if len(pts) < 3:
                    continue
                # Triangolazione identica a struttura_spazio_3d
                for (a, b, c) in self._triangoli_shell(pts):
                    tri_v.append(a); tri_v.append(b); tri_v.append(c)
                # Bordo come segmenti di LINE (coppie consecutive)
                k = len(pts)
                for i in range(k):
                    edge_v.append(pts[i])
                    edge_v.append(pts[(i + 1) % k])
            if tri_v:
                V = np.array(tri_v, dtype=np.float32)
                C = np.tile(np.array(_COL_SHELL_FILL, dtype=np.float32), (len(V), 1))
                self._gpu_upload('scena_shell_fill', V, C)
            if edge_v:
                V = np.array(edge_v, dtype=np.float32)
                C = np.tile(np.array(_COL_SHELL_EDGE, dtype=np.float32), (len(V), 1))
                self._gpu_upload('scena_shell_edge', V, C)

        # ── Vincoli (stile originale struttura_spazio_3d) ─────────
        self._build_vincoli(nodi, vincoli)

    def _build_vincoli(self, nodi: dict, vincoli: dict):
        """Vincoli grafici identici a struttura_spazio_3d.py.

        Incastro: quadrato pieno + hatching a 5 trattini
        Cerniera: triangolo pieno + cerchietto
        Parziale: rombo
        """
        if not vincoli:
            return

        s = _VINCOLO_SZ
        edge_verts = []   # GL_LINES  (bordi + hatch) — coppie di vertici
        fill_verts = []   # GL_TRIANGLES (riempimento pieno)
        col_edge = _COL_VINCOLO
        col_fill = (_COL_VINCOLO[0], _COL_VINCOLO[1], _COL_VINCOLO[2], 0.35)

        for nid, vals in vincoli.items():
            if nid not in nodi:
                continue
            x, y, z = nodi[nid]
            is_incastro = all(v == 1 for v in vals[:6])
            is_cerniera = (all(v == 1 for v in vals[:3]) and
                           all(v == 0 for v in vals[3:6]))

            if is_incastro:
                # Quadrato sotto il nodo (piano XY a z - s*0.5)
                p00 = (x - s, y - s, z - s * 0.5)
                p10 = (x + s, y - s, z - s * 0.5)
                p11 = (x + s, y + s, z - s * 0.5)
                p01 = (x - s, y + s, z - s * 0.5)
                # Bordo (4 lati = 4 coppie di vertici)
                edge_verts += [p00, p10, p10, p11, p11, p01, p01, p00]
                # Hatching (5 trattini diagonali nel piano XZ)
                for i in range(5):
                    t = -s + i * s * 0.5
                    edge_verts.append((x + t,          y, z - s * 0.5))
                    edge_verts.append((x + t - s*0.3,  y, z - s * 0.8))
                # Fill: 2 triangoli
                fill_verts += [p00, p10, p11, p00, p11, p01]

            elif is_cerniera:
                # Triangolo (apice sul nodo)
                t0 = (x,     y, z)
                t1 = (x - s, y, z - s)
                t2 = (x + s, y, z - s)
                # Bordo (3 lati = 3 coppie)
                edge_verts += [t0, t1, t1, t2, t2, t0]
                # Cerchietto in (x, y, z - s), piano XZ, raggio s*0.15
                N = 12
                r = s * 0.15
                circ = []
                for i in range(N):
                    ang = 2.0 * math.pi * i / N
                    circ.append((x + r * math.cos(ang), y,
                                 z - s + r * math.sin(ang)))
                for i in range(N):
                    edge_verts.append(circ[i])
                    edge_verts.append(circ[(i + 1) % N])
                # Fill triangolo
                fill_verts += [t0, t1, t2]

            else:
                # Rombo (parziale)
                r0 = (x,           y, z - s)
                r1 = (x - s * 0.5, y, z - s * 0.5)
                r2 = (x,           y, z)
                r3 = (x + s * 0.5, y, z - s * 0.5)
                # Bordo (4 lati = 4 coppie)
                edge_verts += [r0, r1, r1, r2, r2, r3, r3, r0]
                # Fill: 2 triangoli
                fill_verts += [r0, r1, r2, r0, r2, r3]

        if edge_verts:
            V = np.array(edge_verts, dtype=np.float32)
            C = np.tile(np.array(col_edge, dtype=np.float32), (len(V), 1))
            self._gpu_upload('scena_vincoli', V, C)
        if fill_verts:
            V = np.array(fill_verts, dtype=np.float32)
            C = np.tile(np.array(col_fill, dtype=np.float32), (len(V), 1))
            self._gpu_upload('scena_vincoli_fill', V, C)

    # ================================================================
    #  BUILD: RISULTATI
    # ================================================================

    def _build_risultati(self):
        self._gpu_release_group('diagramma')
        self._gpu_release_group('deformata')
        self._gpu_release_group('tensioni')

        if not self._risultati or not self._risultati.successo:
            return
        if not self._dati:
            return

        modo = self._modo
        if modo in ("N", "Vy", "Vz", "My", "Mz"):
            self._build_diagramma(modo)
        elif modo == "deformata":
            self._build_deformata()
            # Tensioni: overlay attivo solo in modo deformata
            if self._mostra_tensioni:
                # Range comune (aste + shell) per normalizzazione coerente
                sm_beam  = self._risultati.max_sigma_eq_beam()
                sm_shell = self._risultati.max_sigma_vm_shell()
                self._sigma_max_vis = max(sm_beam, sm_shell, 1e-10)
                self._build_deformata_beam_tensioni()
                self._build_tensioni_shell(usa_deformata=True)

    # ── Diagrammi sforzi ──────────────────────────────────────────────

    def _build_diagramma(self, componente: str):
        nodi     = self._dati.get("nodi", {})
        aste     = self._dati.get("aste", {})
        risultati = self._risultati

        max_val = risultati.max_sforzo(componente)
        if max_val < 1e-10:
            return   # nessuna forza → niente da disegnare

        span  = self._calc_span()
        scala = span * 0.15 / max_val

        # Colori per componente
        if componente == "N":
            col_pos, col_neg = _COL_N_POS, _COL_N_NEG
        elif componente.startswith("V"):
            col_pos = col_neg = _COL_V
        else:
            col_pos = col_neg = _COL_M

        verts_fill, cols_fill = [], []
        verts_line, cols_line = [], []

        for bid, asta in aste.items():
            sf = risultati.sforzi_aste.get(bid)
            if sf is None:
                continue
            ni, nj = asta["nodo_i"], asta["nodo_j"]
            if ni not in nodi or nj not in nodi:
                continue

            pi = np.array(nodi[ni], dtype=float)
            pj = np.array(nodi[nj], dtype=float)
            L  = np.linalg.norm(pj - pi)
            if L < 1e-10:
                continue
            e_x = (pj - pi) / L

            # Direzione normale al diagramma, convenzione:
            #   Vy → forza in direzione y locale     → disegnato lungo e_y
            #   Vz → forza in direzione z locale     → disegnato lungo e_z
            #   Mz → momento intorno a z (flessione nel piano x-y)
            #        → il diagramma giace nel piano di flessione
            #        → disegnato lungo e_y
            #   My → momento intorno a y (flessione nel piano x-z)
            #        → disegnato lungo e_z
            #   N  → Z globale (verticale, ortogonalizzato all'asta)
            #
            # Assi locali (coerenti con _vettore_trasformazione):
            #   e_y = ref × e_x,   e_z = e_x × e_y
            # con ref = Z globale (o X globale per aste verticali).
            if componente == "N":
                e_perp = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(e_x, e_perp)) > 0.95:
                    e_perp = np.array([1.0, 0.0, 0.0])
                # Ortogonalizza rispetto all'asse asta
                e_perp -= np.dot(e_perp, e_x) * e_x
                nn = np.linalg.norm(e_perp)
                e_perp = e_perp / nn if nn > 1e-12 else np.array([0.0, 1.0, 0.0])
            else:
                # Calcola terna locale (e_y, e_z)
                ref = np.array([0.0, 0.0, 1.0])
                if abs(e_x[2]) > 0.95:
                    ref = np.array([1.0, 0.0, 0.0])
                e_y_loc = np.cross(ref, e_x)
                nn = np.linalg.norm(e_y_loc)
                e_y_loc = e_y_loc / nn if nn > 1e-12 else np.array([0.0, 1.0, 0.0])
                e_z_loc = np.cross(e_x, e_y_loc)

                if componente in ("Vy", "Mz"):
                    # forza in y, oppure flessione nel piano x-y
                    e_perp = e_y_loc
                else:   # Vz, My
                    # forza in z, oppure flessione nel piano x-z
                    e_perp = e_z_loc

            vals      = getattr(sf, componente, [])
            positions = sf.posizioni
            if len(vals) < 2 or len(positions) < 2:
                continue

            # Costruzione diagramma segmento per segmento
            for k in range(len(vals) - 1):
                t1 = positions[k]     / L if L > 0 else 0.0
                t2 = positions[k + 1] / L if L > 0 else 1.0
                t1 = max(0.0, min(t1, 1.0))
                t2 = max(0.0, min(t2, 1.0))
                p1b = pi + t1 * (pj - pi)
                p2b = pi + t2 * (pj - pi)
                v1 = vals[k]     * scala
                v2 = vals[k + 1] * scala
                p1t = p1b + v1 * e_perp
                p2t = p2b + v2 * e_perp

                c1 = col_pos if vals[k]     >= 0 else col_neg
                c2 = col_pos if vals[k + 1] >= 0 else col_neg

                # Fill (due triangoli per ogni segmento)
                verts_fill.extend([*p1b, *p1t, *p2t, *p1b, *p2t, *p2b])
                cols_fill.extend([*c1, *c1, *c2, *c1, *c2, *c2])

                # Linea del diagramma
                cl = (*c1[:3], 0.95)
                verts_line.extend([*p1t, *p2t])
                cols_line.extend([*cl, *cl])

            # Segmenti verticali chiusura inizio/fine
            p0t = pi + vals[0]    * scala * e_perp
            pLt = pj + vals[-1]   * scala * e_perp
            cl0 = (*col_pos[:3], 0.95)
            verts_line.extend([*pi, *p0t, *pj, *pLt])
            cols_line.extend([*cl0, *cl0, *cl0, *cl0])

        if verts_fill:
            self._gpu_upload('diagramma_fill',
                             np.array(verts_fill, dtype=np.float32),
                             np.array(cols_fill,  dtype=np.float32))
        if verts_line:
            self._gpu_upload('diagramma_line',
                             np.array(verts_line, dtype=np.float32),
                             np.array(cols_line,  dtype=np.float32))

    # ── Deformata ─────────────────────────────────────────────────────

    def _build_deformata(self):
        if not self._mesh or not self._risultati:
            return

        mesh      = self._mesh
        risultati = self._risultati
        scala     = self._scala_deformazione

        def pos_def(tag):
            n  = mesh.nodi.get(tag)
            if n is None:
                return (0.0, 0.0, 0.0)
            sp = risultati.spostamenti.get(tag)
            if sp is None:
                return (n.x, n.y, n.z)
            return (n.x + sp.dx * scala,
                    n.y + sp.dy * scala,
                    n.z + sp.dz * scala)

        # Beam deformati
        bv, bc = [], []
        for elem in mesh.elementi_beam:
            bv.extend([*pos_def(elem.nodo_i), *pos_def(elem.nodo_j)])
            bc.extend([*_COL_DEFORMATA, *_COL_DEFORMATA])
        if bv:
            self._gpu_upload('deformata_beam',
                             np.array(bv, dtype=np.float32),
                             np.array(bc, dtype=np.float32))

        # Nodi deformati
        nv, nc = [], []
        for tag in mesh.nodi:
            nv.extend(pos_def(tag))
            nc.extend([*_COL_DEFORMATA])
        if nv:
            self._gpu_upload('deformata_nodi',
                             np.array(nv, dtype=np.float32),
                             np.array(nc, dtype=np.float32))

        # Shell deformate (verdolino uniforme semitrasparente):
        # triangolazione sugli elementi shell della mesh, seguendo gli
        # spostamenti dei nodi. Usate quando si visualizza la deformata
        # senza l'overlay tensioni.
        if mesh.elementi_shell:
            tri_v, edge_v = [], []
            for elem_shell in mesh.elementi_shell:
                pts = [pos_def(t) for t in elem_shell.nodi]
                if len(pts) < 3:
                    continue
                for (a, b, c) in self._triangoli_shell(pts):
                    tri_v.append(a); tri_v.append(b); tri_v.append(c)
                k = len(pts)
                for i in range(k):
                    edge_v.append(pts[i])
                    edge_v.append(pts[(i + 1) % k])
            if tri_v:
                V = np.array(tri_v, dtype=np.float32)
                C = np.tile(np.array(_COL_SHELL_FILL, dtype=np.float32),
                            (len(V), 1))
                self._gpu_upload('deformata_shell_fill', V, C)
            if edge_v:
                V = np.array(edge_v, dtype=np.float32)
                C = np.tile(np.array(_COL_SHELL_EDGE, dtype=np.float32),
                            (len(V), 1))
                self._gpu_upload('deformata_shell_edge', V, C)

    # ── Beam deformati colorati per tensione ──────────────────────────

    def _build_deformata_beam_tensioni(self):
        """Ricostruisce le aste deformate colorate per tensione equivalente."""
        if not self._mesh or not self._risultati or not self._dati:
            return

        mesh      = self._mesh
        risultati = self._risultati
        scala     = self._scala_deformazione
        aste_orig = self._dati.get("aste", {})

        def pos_def(tag):
            n = mesh.nodi.get(tag)
            if n is None:
                return (0.0, 0.0, 0.0)
            sp = risultati.spostamenti.get(tag)
            if sp is None:
                return (n.x, n.y, n.z)
            return (n.x + sp.dx * scala,
                    n.y + sp.dy * scala,
                    n.z + sp.dz * scala)

        verts, cols = [], []
        # Interpola sigma_eq lungo ogni sotto-elemento dell'asta originale
        for bid in aste_orig:
            sf = risultati.sforzi_aste.get(bid)
            nodi_asta = mesh.mappa_nodi_aste.get(bid, [])
            elem_tags = mesh.mappa_aste.get(bid, [])
            if not elem_tags or not nodi_asta or len(nodi_asta) < 2:
                continue
            sigmas = sf.sigma_eq if sf else []
            # Se non c'e' una sigma per nodo, fallback a 0
            if not sigmas or len(sigmas) != len(nodi_asta):
                sigmas = [0.0] * len(nodi_asta)

            # Un tag per ogni coppia (nodo_i, nodo_j) del sotto-elemento
            # nodi_asta e' ordinato da i a j con len = n_elem + 1
            for k in range(len(nodi_asta) - 1):
                ni = nodi_asta[k]
                nj = nodi_asta[k + 1]
                pi = pos_def(ni)
                pj = pos_def(nj)
                ci = self._stress_color(abs(sigmas[k]))
                cj = self._stress_color(abs(sigmas[k + 1]))
                verts.extend([*pi, *pj])
                cols.extend([*ci, *cj])

        if verts:
            self._gpu_upload('deformata_beam_tens',
                             np.array(verts, dtype=np.float32),
                             np.array(cols,  dtype=np.float32))

    # ── Tensioni shell ────────────────────────────────────────────────

    def _stress_color(self, vm: float) -> tuple:
        """Gradiente blu->verde->rosso, normalizzato su _sigma_max_vis."""
        sm = self._sigma_max_vis
        if sm < 1e-10:
            return (float(_STRESS_LOW[0]), float(_STRESS_LOW[1]),
                    float(_STRESS_LOW[2]), 1.0)
        t = min(max(vm, 0.0) / sm, 1.0)
        if t < 0.5:
            t2 = t * 2.0
            c = _STRESS_LOW * (1 - t2) + _STRESS_MID * t2
        else:
            t2 = (t - 0.5) * 2.0
            c = _STRESS_MID * (1 - t2) + _STRESS_HIGH * t2
        return (float(c[0]), float(c[1]), float(c[2]), 1.0)

    def _build_tensioni_shell(self, usa_deformata: bool = False):
        if not self._mesh or not self._risultati:
            return
        if not self._mesh.elementi_shell:
            return

        mesh      = self._mesh
        risultati = self._risultati
        scala     = self._scala_deformazione if usa_deformata else 0.0

        # Mappa shell_tag -> sigma_vm dell'elemento
        # (l'estrattore assegna la stessa sigma a tutti i nodi dell'elemento,
        #  quindi ogni shell ha la propria tinta uniforme di Von Mises)
        shell_vm: dict[int, float] = {}
        for ts in risultati.tensioni_shell:
            if ts.tensioni_nodi:
                vm_first = next(iter(ts.tensioni_nodi.values()))[3]
                shell_vm[ts.shell_tag] = abs(vm_first)

        # Fallback: usa comunque il dato per nodo (max) nel caso
        # l'estrattore abbia popolato solo alcuni nodi
        nodo_vm: dict[int, float] = {}
        for ts in risultati.tensioni_shell:
            for ntag, tens in ts.tensioni_nodi.items():
                vm = abs(tens[3])
                nodo_vm[ntag] = max(nodo_vm.get(ntag, 0.0), vm)

        def pos_def(tag):
            n  = mesh.nodi.get(tag)
            if n is None:
                return np.array([0.0, 0.0, 0.0])
            if scala == 0.0:
                return np.array([n.x, n.y, n.z])
            sp = risultati.spostamenti.get(tag)
            if sp is None:
                return np.array([n.x, n.y, n.z])
            return np.array([n.x + sp.dx * scala,
                             n.y + sp.dy * scala,
                             n.z + sp.dz * scala])

        def stress_col(vm: float) -> tuple:
            r, g, b, _ = self._stress_color(vm)
            return (r, g, b, 0.88)

        vf, cf = [], []
        ve, ce = [], []

        for elem_shell in mesh.elementi_shell:
            tags  = elem_shell.nodi
            pts   = [pos_def(t) for t in tags]
            n_pts = len(pts)
            if n_pts < 3:
                continue

            # Colore dell'elemento: prima per shell (uniforme), poi fallback
            # sul dato nodale se l'elemento non è stato estratto
            vm_el = shell_vm.get(elem_shell.tag)
            if vm_el is None:
                vm_nodi = [nodo_vm.get(t, 0.0) for t in tags]
                vm_el = max(vm_nodi) if vm_nodi else 0.0
            col_el = stress_col(vm_el)

            # Fan triangolazione, tutti i vertici con lo stesso colore
            for i in range(1, n_pts - 1):
                for idx in (0, i, i + 1):
                    vf.extend(pts[idx])
                    cf.extend(col_el)

            # Bordo più marcato per distinguere gli elementi
            ec = (0.10, 0.10, 0.10, 0.55)
            for i in range(n_pts):
                ve.extend([*pts[i], *pts[(i + 1) % n_pts]])
                ce.extend([*ec, *ec])

        if vf:
            self._gpu_upload('tensioni_fill',
                             np.array(vf, dtype=np.float32),
                             np.array(cf, dtype=np.float32))
        if ve:
            self._gpu_upload('tensioni_edge',
                             np.array(ve, dtype=np.float32),
                             np.array(ce, dtype=np.float32))

    # ================================================================
    #  LEGENDA TENSIONI (identica a fem_elemento)
    # ================================================================

    def _draw_legend(self, painter: QPainter, W: int, H: int):
        _FF, _FS = "Arial", 8
        bar_w  = 22
        bar_h  = min(442, H - 60)
        mrg_r  = 86
        x_bar  = W - mrg_r - bar_w
        y_bar  = (H - bar_h) // 2

        bg_x = x_bar - 10;  bg_y = y_bar - 52
        bg_w = bar_w + mrg_r - 12;  bg_h = bar_h + 100
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 18, 18, 185))
        painter.drawRoundedRect(bg_x, bg_y, bg_w, bg_h, 5, 5)

        f_t = QFont(_FF, _FS);  f_t.setBold(True)
        painter.setFont(f_t)
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(bg_x + 4, bg_y + 18, "σ_vm [MPa]")

        grad = QLinearGradient(x_bar, y_bar + bar_h, x_bar, y_bar)
        grad.setColorAt(0.00, QColor(25,  77,  217))
        grad.setColorAt(0.50, QColor(51,  204, 51))
        grad.setColorAt(1.00, QColor(230, 51,  38))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawRect(x_bar, y_bar, bar_w, bar_h)

        ma = self._sigma_max_vis if self._sigma_max_vis > 1e-10 else 1.0
        ticks = [0.0, ma / 4, ma / 2, 3 * ma / 4, ma]
        f_tk = QFont(_FF, _FS - 1)
        painter.setFont(f_tk)
        for val in ticks:
            t_n    = val / ma
            y_tick = int(y_bar + bar_h - t_n * bar_h)
            painter.setPen(QPen(QColor(155, 155, 155), 1))
            painter.drawLine(x_bar + bar_w, y_tick, x_bar + bar_w + 5, y_tick)
            txt = "0" if val == 0 else (f"{val:.1e}" if abs(val) > 999 else f"{val:.1f}")
            painter.setPen(QPen(QColor(210, 210, 210)))
            painter.drawText(x_bar + bar_w + 7, y_tick + 4, txt)

        painter.setPen(QColor(165, 165, 165))
        painter.drawText(bg_x + 3, y_bar + bar_h + 24, "Min: 0")
        painter.drawText(bg_x + 3, y_bar - 12, f"Max: {ma:.1f}")

    # ================================================================
    #  UTILITY
    # ================================================================

    def _triangoli_shell(self, pts):
        """Suddivide i punti di una shell (3 o 4) in triangoli.
        Replica struttura_spazio_3d.py per identica resa visiva."""
        if len(pts) == 3:
            return [(pts[0], pts[1], pts[2])]
        if len(pts) == 4:
            return [(pts[0], pts[1], pts[2]),
                    (pts[0], pts[2], pts[3])]
        return [(pts[0], pts[i], pts[i + 1]) for i in range(1, len(pts) - 1)]

    def _calc_span(self) -> float:
        if not self._dati:
            return 10.0
        nodi = self._dati.get("nodi", {})
        if not nodi:
            return 10.0
        pts = np.array(list(nodi.values()), dtype=float)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        return max(float(np.linalg.norm(mx - mn)), 1.0)

    # ================================================================
    #  MOUSE EVENTS (identici all'originale)
    # ================================================================

    def mousePressEvent(self, e):
        self._last_pos  = e.pos()
        self._mouse_btn = e.button()

    def mouseMoveEvent(self, e):
        dx = e.x() - self._last_pos.x()
        dy = e.y() - self._last_pos.y()
        self._last_pos = e.pos()

        if self._mouse_btn == Qt.MiddleButton:
            self.rot_x += dy * 0.5
            self.rot_y += dx * 0.5
            self.update()
        elif self._mouse_btn == Qt.RightButton:
            factor = self.cam_dist * 0.002
            self.pan_x += dx * factor
            self.pan_y -= dy * factor
            self.update()

    def mouseReleaseEvent(self, e):
        self._mouse_btn = None

    def wheelEvent(self, e):
        delta  = e.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.cam_dist = max(0.5, self.cam_dist * factor)
        self.update()
