"""
disegno_fem_elemento.py – OpenGL 3D workspace for FEM element visualization.

Rendering ottimizzato con VBO (Vertex Buffer Objects) per massimo sfruttamento GPU.
I dati dei vertici vengono pre-calcolati con numpy (vettorizzato, multi-core) e
caricati in memoria GPU una sola volta. Il re-upload avviene solo quando i dati cambiano.

Visualizza:
  - Oggetti 3D dell'elemento (carpenteria, barre, staffe) in trasparenza
  - Mesh esaedrica generata (wireframe griglia per carpenteria)
  - Mesh 1D per barre (rossa) e staffe (gialla)
  - Nodi vincolati (verde) e caricati (viola)
  - Deformata con scala regolabile
  - Colorazione tensioni (gradiente blu->rosso, nero per collasso materiale)
  - Animazione incrementale per analisi nonlineare

Camera controls: Middle=orbit, Right=pan, Scroll=zoom.
"""

import math
import numpy as np
from ctypes import c_void_p

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QFont, QColor, QPen, QBrush, QLinearGradient

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINE_SMOOTH, GL_POINT_SMOOTH,
    GL_LINE_SMOOTH_HINT, GL_POINT_SMOOTH_HINT, GL_NICEST,
    GL_MODELVIEW, GL_PROJECTION,
    GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP,
    GL_QUADS, GL_POINTS, GL_TRIANGLES,
    GL_TRIANGLE_FAN, GL_QUAD_STRIP,
    GL_FALSE, GL_TRUE, GL_FLOAT,
    glClearColor, glClear, glEnable, glDisable,
    glBlendFunc, glDepthFunc, glDepthMask,
    glLineWidth, glPointSize,
    glBegin, glEnd, glVertex3f, glColor3f, glColor4f,
    glMatrixMode, glLoadIdentity, glViewport, glOrtho,
    glTranslatef, glRotatef,
    glHint,
    glPushMatrix, glPopMatrix,
    # VBO
    glGenBuffers, glDeleteBuffers, glBindBuffer, glBufferData,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW,
    glEnableClientState, glDisableClientState,
    GL_VERTEX_ARRAY, GL_COLOR_ARRAY,
    glVertexPointer, glColorPointer,
    glDrawArrays,
)
from OpenGL.GLU import gluPerspective


# ── Palette (coerente con elementi_spazio_3d.py) ──────────────────
_BG_R, _BG_G, _BG_B = 40/255, 40/255, 40/255

_GF_R, _GF_G, _GF_B = 0.20, 0.20, 0.20
_GC_R, _GC_G, _GC_B = 0.27, 0.27, 0.27

_AX_X = (0.862, 0.200, 0.200)
_AX_Y = (0.310, 0.620, 0.165)
_AX_Z = (0.161, 0.408, 0.784)

_STRUCT_FILL = (140/255, 148/255, 162/255, 0.08)
_STRUCT_EDGE = (185/255, 190/255, 205/255, 0.30)

_MESH_EDGE_CARPENTERIA = (0.75, 0.75, 0.75, 0.60)
_MESH_EDGE_BARRE       = (0.85, 0.25, 0.25, 1.00)
_MESH_EDGE_STAFFE      = (1.00, 0.90, 0.10, 1.00)

_NODO_VINCOLATO = (0.20, 0.85, 0.30, 1.00)
_NODO_CARICATO  = (0.65, 0.20, 0.85, 1.00)

# Colori per gradiente tensioni (blu = basso, rosso = alto, nero = collasso)
_STRESS_LOW  = np.array([0.10, 0.30, 0.85], dtype=np.float32)
_STRESS_MID  = np.array([0.20, 0.80, 0.20], dtype=np.float32)
_STRESS_HIGH = np.array([0.90, 0.20, 0.15], dtype=np.float32)
_STRESS_FAIL = np.array([0.05, 0.05, 0.05], dtype=np.float32)

_CYL_N = 24
_SPH_NL, _SPH_NM = 10, 20

# Triangulation delle 6 facce di un C3D8 (indici locali 0-7)
_HEX_TRI_IDX = np.array([
    0, 1, 2,  0, 2, 3,   # face bottom
    4, 5, 6,  4, 6, 7,   # face top
    0, 1, 5,  0, 5, 4,   # face front
    3, 2, 6,  3, 6, 7,   # face back
    0, 3, 7,  0, 7, 4,   # face left
    1, 2, 6,  1, 6, 5,   # face right
], dtype=np.int32)  # 36 indices per hex

# Edges di un C3D8
_HEX_EDGE_IDX = np.array([
    0, 1, 1, 2, 2, 3, 3, 0,
    4, 5, 5, 6, 6, 7, 7, 4,
    0, 4, 1, 5, 2, 6, 3, 7,
], dtype=np.int32)  # 24 indices (12 edges × 2 verts)

# Offset nullo per VBO
_VBO_0 = c_void_p(0)


def _stress_color(sigma_vm: float, sigma_max: float, sigma_ult: float) -> tuple:
    if sigma_ult > 0 and sigma_vm >= sigma_ult:
        return (*_STRESS_FAIL, 1.0)
    if sigma_max < 1e-10:
        return (*_STRESS_LOW, 0.85)
    t = min(sigma_vm / sigma_max, 1.0)
    if t < 0.5:
        t2 = t * 2.0
        c = _STRESS_LOW * (1 - t2) + _STRESS_MID * t2
    else:
        t2 = (t - 0.5) * 2.0
        c = _STRESS_MID * (1 - t2) + _STRESS_HIGH * t2
    return (float(c[0]), float(c[1]), float(c[2]), 0.85)


def _stress_colors_batch(sv: np.ndarray, sigma_max: float,
                         su: np.ndarray) -> np.ndarray:
    """Calcola colori tensione per N vertici in batch (numpy vettorizzato)."""
    n = len(sv)
    colors = np.zeros((n, 4), dtype=np.float32)
    colors[:, 3] = 0.85

    if sigma_max < 1e-10:
        colors[:, 0] = _STRESS_LOW[0]
        colors[:, 1] = _STRESS_LOW[1]
        colors[:, 2] = _STRESS_LOW[2]
    else:
        t = np.minimum(sv / sigma_max, 1.0)
        mask_lo = t < 0.5
        mask_hi = ~mask_lo

        # t < 0.5: interpolate LOW -> MID
        t2_lo = t[mask_lo] * 2.0
        colors[mask_lo, :3] = (_STRESS_LOW[None, :] * (1 - t2_lo[:, None]) +
                                _STRESS_MID[None, :] * t2_lo[:, None])

        # t >= 0.5: interpolate MID -> HIGH
        t2_hi = (t[mask_hi] - 0.5) * 2.0
        colors[mask_hi, :3] = (_STRESS_MID[None, :] * (1 - t2_hi[:, None]) +
                                _STRESS_HIGH[None, :] * t2_hi[:, None])

    # Collasso: sigma_vm >= sigma_ult
    fail_mask = (su > 0) & (sv >= su)
    if np.any(fail_mask):
        colors[fail_mask, 0] = _STRESS_FAIL[0]
        colors[fail_mask, 1] = _STRESS_FAIL[1]
        colors[fail_mask, 2] = _STRESS_FAIL[2]
        colors[fail_mask, 3] = 1.0

    return colors


class FEMSpazio3D(QOpenGLWidget):
    """Widget OpenGL per visualizzazione FEM dell'elemento.
    Rendering GPU-accelerato con VBO e preparazione dati numpy vettorizzata."""

    animazione_step_changed = pyqtSignal(int, int)   # (current_step, total_steps)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Camera
        self.cam_dist: float = 12.0
        self.rot_x:    float = 25.0
        self.rot_y:    float = -40.0
        self.pan_x:    float = 0.0
        self.pan_y:    float = 0.0
        self._ortho:   bool  = False

        self._last_pos  = QPoint()
        self._mouse_btn = None

        # Dati mesh
        self._oggetti: list = []
        self._mesh = None                  # RisultatoMesh

        # Risultati analisi
        self._risultati_lineare = None     # RisultatiFRD
        self._risultati_nonlineare = None  # RisultatiFRD
        self._modo_vista = "mesh"          # "mesh" | "lineare" | "nonlineare"

        # Resistenze ultime per materiale: {nome_mat: sigma_ult}
        self._sigma_ult: dict[str, float] = {}

        # Scala deformazione
        self._scala_deformazione: float = 1.0

        # Visibilita'
        self._mostra_carpenteria = True
        self._mostra_barre = True
        self._mostra_staffe = True
        self._mostra_vincoli = True
        self._mostra_carichi = True
        self._mostra_crack   = True

        # Animazione nonlineare
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)
        self._anim_playing = False
        self._anim_step = 0
        self._anim_total = 1
        self._anim_durata_sec = 5.0

        # Deformata cache
        self._nodi_deformati = None

        # Sigma max visibile (per legenda)
        self._sigma_max_vis: float = 0.0

        # ── GPU buffers ──
        self._gpu: dict[str, tuple] = {}   # name -> (vbo_v, vbo_c, n_verts)
        self._dirty_griglia = True
        self._dirty_mesh = True
        self._dirty_risultati = True
        self._gl_ready = False

    # ================================================================
    # GPU BUFFER MANAGEMENT
    # ================================================================

    def _gpu_upload(self, name: str, verts_f32: np.ndarray,
                    colors_f32: np.ndarray) -> None:
        """Carica vertici e colori in VBO sulla GPU."""
        self._gpu_release(name)
        n = len(verts_f32) // 3
        if n == 0:
            return
        verts_f32 = np.ascontiguousarray(verts_f32, dtype=np.float32)
        colors_f32 = np.ascontiguousarray(colors_f32, dtype=np.float32)

        vbo_v = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo_v)
        glBufferData(GL_ARRAY_BUFFER, verts_f32.nbytes, verts_f32, GL_STATIC_DRAW)

        vbo_c = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo_c)
        glBufferData(GL_ARRAY_BUFFER, colors_f32.nbytes, colors_f32, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._gpu[name] = (vbo_v, vbo_c, n)

    def _gpu_draw(self, name: str, mode: int = GL_LINES) -> None:
        """Disegna un buffer VBO caricato sulla GPU."""
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
        for name in [k for k in self._gpu if k.startswith(prefix)]:
            self._gpu_release(name)

    # ================================================================
    # PUBLIC API
    # ================================================================

    def set_oggetti(self, oggetti):
        self._oggetti = list(oggetti) if oggetti else []
        self.update()

    def set_mesh(self, mesh):
        self._mesh = mesh
        self._dirty_mesh = True
        self._dirty_risultati = True
        self._aggiorna_deformata()
        self.update()

    def set_risultati(self, risultati_lineare=None, risultati_nonlineare=None,
                      sigma_ult: dict = None):
        if risultati_lineare is not None:
            self._risultati_lineare = risultati_lineare
        if risultati_nonlineare is not None:
            self._risultati_nonlineare = risultati_nonlineare
        if sigma_ult is not None:
            self._sigma_ult = sigma_ult
        self._anim_step = 0
        self._dirty_risultati = True
        self._aggiorna_deformata()
        self.update()

    def set_modo_vista(self, modo: str):
        if modo != self._modo_vista:
            self._modo_vista = modo
            self._anim_step = 0
            self._dirty_risultati = True
            self._aggiorna_deformata()
            self.update()

    def set_scala_deformazione(self, scala: float):
        self._scala_deformazione = scala
        self._dirty_risultati = True
        self._aggiorna_deformata()
        self.update()

    def set_visibilita(self, carpenteria=None, barre=None, staffe=None):
        changed = False
        if carpenteria is not None and carpenteria != self._mostra_carpenteria:
            self._mostra_carpenteria = carpenteria; changed = True
        if barre is not None and barre != self._mostra_barre:
            self._mostra_barre = barre; changed = True
        if staffe is not None and staffe != self._mostra_staffe:
            self._mostra_staffe = staffe; changed = True
        if changed:
            # Ricalcola colori: sigma_max dipende da cosa e' visibile
            self._dirty_risultati = True
            self.update()

    def set_visibilita_nodi(self, vincoli=None, carichi=None, crack=None):
        if vincoli is not None:
            self._mostra_vincoli = vincoli
        if carichi is not None:
            self._mostra_carichi = carichi
        if crack is not None:
            self._mostra_crack = crack
        self.update()

    def imposta_vista(self, nome: str):
        presets = {
            "3d": (25.0, -40.0, False),
            "x":  ( 0.0, -90.0, True),
            "y":  ( 0.0,   0.0, True),
            "z":  (90.0,   0.0, True),
        }
        if nome in presets:
            self.rot_x, self.rot_y, self._ortho = presets[nome]
            if self._ortho:
                self.pan_x = self.pan_y = 0.0
        self.update()

    def centra_vista(self):
        self.pan_x = 0.0; self.pan_y = 0.0
        self.cam_dist = 12.0
        self.update()

    # ================================================================
    # ANIMAZIONE NONLINEARE
    # ================================================================

    def anim_play_pause(self):
        if self._anim_playing:
            self._anim_timer.stop()
            self._anim_playing = False
        else:
            if self._modo_vista == "nonlineare":
                res = self._risultati_nonlineare
            elif self._modo_vista == "lineare":
                res = self._risultati_lineare
            else:
                return
            if not res or res.n_steps < 2:
                return
            self._anim_total = res.n_steps
            if self._anim_step >= self._anim_total - 1:
                self._anim_step = 0
            interval_ms = max(16, int(self._anim_durata_sec * 1000 / self._anim_total))
            self._anim_timer.start(interval_ms)
            self._anim_playing = True

    def anim_replay(self):
        self._anim_step = 0
        self._aggiorna_deformata()
        self.animazione_step_changed.emit(0, self._anim_total)
        self.update()
        if not self._anim_playing:
            self.anim_play_pause()

    def anim_set_durata(self, secondi: float):
        self._anim_durata_sec = max(1.0, secondi)
        if self._anim_playing and self._anim_total > 0:
            interval_ms = max(16, int(self._anim_durata_sec * 1000 / self._anim_total))
            self._anim_timer.setInterval(interval_ms)

    def anim_stop(self):
        self._anim_timer.stop()
        self._anim_playing = False

    @property
    def anim_is_playing(self) -> bool:
        return self._anim_playing

    def _anim_tick(self):
        self._anim_step += 1
        if self._anim_step >= self._anim_total:
            self._anim_step = self._anim_total - 1
            self._anim_timer.stop()
            self._anim_playing = False
        self._dirty_risultati = True
        self._aggiorna_deformata()
        self.animazione_step_changed.emit(self._anim_step, self._anim_total)
        self.update()

    # ================================================================
    # DEFORMATA
    # ================================================================

    def _aggiorna_deformata(self):
        if self._mesh is None:
            self._nodi_deformati = None
            return

        risultati = None
        step_idx = -1

        if self._modo_vista == "lineare" and self._risultati_lineare:
            risultati = self._risultati_lineare
            step_idx = self._anim_step if self._anim_step < risultati.n_steps else -1
        elif self._modo_vista == "nonlineare" and self._risultati_nonlineare:
            risultati = self._risultati_nonlineare
            step_idx = self._anim_step if self._anim_step < risultati.n_steps else -1
        else:
            self._nodi_deformati = None
            return

        step = risultati.get_step(step_idx)
        if step is None or not step.spostamenti:
            self._nodi_deformati = None
            return

        # Calcolo vettorizzato con numpy
        scala = self._scala_deformazione
        nids = sorted(self._mesh.nodi.keys())
        pos = np.array([self._mesh.nodi[nid] for nid in nids], dtype=np.float64)
        disp = np.zeros_like(pos)
        spost = step.spostamenti
        for i, nid in enumerate(nids):
            if nid in spost:
                disp[i] = spost[nid]
        deformed = pos + disp * scala
        self._nodi_deformati = {nid: deformed[i].tolist() for i, nid in enumerate(nids)}

    def _get_nodi_render(self) -> dict:
        if self._nodi_deformati is not None:
            return self._nodi_deformati
        if self._mesh is not None:
            return self._mesh.nodi
        return {}

    def _get_step_corrente(self):
        if self._modo_vista == "lineare" and self._risultati_lineare:
            idx = self._anim_step if self._anim_step < self._risultati_lineare.n_steps else -1
            return self._risultati_lineare.get_step(idx)
        elif self._modo_vista == "nonlineare" and self._risultati_nonlineare:
            idx = self._anim_step if self._anim_step < self._risultati_nonlineare.n_steps else -1
            return self._risultati_nonlineare.get_step(idx)
        return None

    def _get_sigma_ult_per_nodo(self, nid: int) -> float:
        if not self._mesh or not self._sigma_ult:
            return 0.0
        for obj_id, nodi_set in self._mesh.nodi_per_oggetto.items():
            if nid in nodi_set:
                mat = self._mesh.materiale_oggetto.get(obj_id, "")
                return self._sigma_ult.get(mat, 0.0)
        return 0.0

    # ================================================================
    # BUILD: GRIGLIA (statica, costruita una sola volta)
    # ================================================================

    def _build_griglia(self):
        DIM, FADE, SEG_L = 100, 40, 2
        parts_v = []
        parts_c = []

        for step, cr, cg, cb in ((1, _GF_R, _GF_G, _GF_B),
                                  (5, _GC_R, _GC_G, _GC_B)):
            ii = np.arange(-DIM, DIM + 1, step, dtype=np.float32)
            if step == 1:
                ii = ii[ii.astype(np.int32) % 5 != 0]
            jj = np.arange(-DIM, DIM, SEG_L, dtype=np.float32)
            jj2 = jj + SEG_L

            I, J = np.meshgrid(ii, jj, indexing='ij')
            _, J2 = np.meshgrid(ii, jj2, indexing='ij')

            # X-direction lines: (I, J, 0) -> (I, J2, 0)
            d1 = np.hypot(I, J) / FADE
            d2 = np.hypot(I, J2) / FADE
            a1 = np.maximum(0.0, 1.0 - d1 * d1 * 0.55).astype(np.float32)
            a2 = np.maximum(0.0, 1.0 - d2 * d2 * 0.55).astype(np.float32)
            mask = (a1 > 0.01) | (a2 > 0.01)
            n = int(mask.sum())
            if n > 0:
                v = np.zeros((n * 2, 3), dtype=np.float32)
                v[0::2, 0] = I[mask]; v[0::2, 1] = J[mask]
                v[1::2, 0] = I[mask]; v[1::2, 1] = J2[mask]
                c = np.empty((n * 2, 4), dtype=np.float32)
                c[0::2, 0] = cr; c[0::2, 1] = cg; c[0::2, 2] = cb; c[0::2, 3] = a1[mask]
                c[1::2, 0] = cr; c[1::2, 1] = cg; c[1::2, 2] = cb; c[1::2, 3] = a2[mask]
                parts_v.append(v.ravel()); parts_c.append(c.ravel())

            # Y-direction lines: (J, I, 0) -> (J2, I, 0)
            d1 = np.hypot(J, I) / FADE
            d2 = np.hypot(J2, I) / FADE
            a1 = np.maximum(0.0, 1.0 - d1 * d1 * 0.55).astype(np.float32)
            a2 = np.maximum(0.0, 1.0 - d2 * d2 * 0.55).astype(np.float32)
            mask = (a1 > 0.01) | (a2 > 0.01)
            n = int(mask.sum())
            if n > 0:
                v = np.zeros((n * 2, 3), dtype=np.float32)
                v[0::2, 0] = J[mask]; v[0::2, 1] = I[mask]
                v[1::2, 0] = J2[mask]; v[1::2, 1] = I[mask]
                c = np.empty((n * 2, 4), dtype=np.float32)
                c[0::2, 0] = cr; c[0::2, 1] = cg; c[0::2, 2] = cb; c[0::2, 3] = a1[mask]
                c[1::2, 0] = cr; c[1::2, 1] = cg; c[1::2, 2] = cb; c[1::2, 3] = a2[mask]
                parts_v.append(v.ravel()); parts_c.append(c.ravel())

        if parts_v:
            self._gpu_upload('griglia',
                             np.concatenate(parts_v),
                             np.concatenate(parts_c))

    # ================================================================
    # BUILD: ASSI (statici)
    # ================================================================

    def _build_assi(self):
        EXT, NEG = 40.0, -40.0

        # Assi positivi
        v_pos = np.array([
            0, 0, 0,  EXT, 0, 0,
            0, 0, 0,  0, EXT, 0,
            0, 0, 0,  0, 0, EXT,
        ], dtype=np.float32)
        c_pos = np.array([
            *_AX_X, 1, *_AX_X, 1,
            *_AX_Y, 1, *_AX_Y, 1,
            *_AX_Z, 1, *_AX_Z, 1,
        ], dtype=np.float32)
        self._gpu_upload('assi_pos', v_pos, c_pos)

        # Assi negativi
        v_neg = np.array([
            0, 0, 0,  NEG, 0, 0,
            0, 0, 0,  0, NEG, 0,
            0, 0, 0,  0, 0, NEG,
        ], dtype=np.float32)
        c_neg = np.array([
            *_AX_X, 0.28, *_AX_X, 0.28,
            *_AX_Y, 0.28, *_AX_Y, 0.28,
            *_AX_Z, 0.28, *_AX_Z, 0.28,
        ], dtype=np.float32)
        self._gpu_upload('assi_neg', v_neg, c_neg)

        # Origine
        v_o = np.array([0, 0, 0], dtype=np.float32)
        c_o = np.array([0.9, 0.9, 0.9, 0.85], dtype=np.float32)
        self._gpu_upload('assi_orig', v_o, c_o)

    # ================================================================
    # BUILD: MESH WIREFRAME (quando mesh cambia)
    # ================================================================

    def _build_mesh_buffers(self):
        self._gpu_release_group('mesh_')
        if self._mesh is None:
            return

        nodi = self._mesh.nodi
        elem_tipo = {}
        for obj_id, es in self._mesh.elementi_per_oggetto.items():
            t = self._mesh.tipo_oggetto.get(obj_id, "")
            for eid in es:
                elem_tipo[eid] = t

        # ── Hex wireframe (carpenteria) ──
        hex_verts = []
        for eid, hn in self._mesh.elementi_hex.items():
            if elem_tipo.get(eid) != "carpenteria":
                continue
            for i in range(0, len(_HEX_EDGE_IDX), 2):
                a, b = _HEX_EDGE_IDX[i], _HEX_EDGE_IDX[i + 1]
                hex_verts.extend(nodi[hn[a]])
                hex_verts.extend(nodi[hn[b]])

        if hex_verts:
            v = np.array(hex_verts, dtype=np.float32)
            n = len(v) // 3
            c = np.tile(np.array(_MESH_EDGE_CARPENTERIA, dtype=np.float32), n)
            self._gpu_upload('mesh_hex', v, c)

        # ── Barre wireframe ──
        barre_v = []
        for eid, bn in self._mesh.elementi_beam.items():
            if elem_tipo.get(eid) == "barra":
                barre_v.extend(nodi[bn[0]])
                barre_v.extend(nodi[bn[1]])
        if barre_v:
            v = np.array(barre_v, dtype=np.float32)
            n = len(v) // 3
            c = np.tile(np.array(_MESH_EDGE_BARRE, dtype=np.float32), n)
            self._gpu_upload('mesh_barre', v, c)

        # ── Staffe wireframe ──
        staffe_v = []
        for eid, bn in self._mesh.elementi_beam.items():
            if elem_tipo.get(eid) == "staffa":
                staffe_v.extend(nodi[bn[0]])
                staffe_v.extend(nodi[bn[1]])
        if staffe_v:
            v = np.array(staffe_v, dtype=np.float32)
            n = len(v) // 3
            c = np.tile(np.array(_MESH_EDGE_STAFFE, dtype=np.float32), n)
            self._gpu_upload('mesh_staffe', v, c)

        # ── Nodi speciali ──
        self._build_nodi_speciali('mesh_nodi_vinc', 'mesh_nodi_caric', nodi)

    def _build_nodi_speciali(self, name_vinc, name_caric, nodi):
        if self._mesh and self._mesh.nodi_vincolati:
            pts = [nodi[nid] for nid in self._mesh.nodi_vincolati if nid in nodi]
            if pts:
                v = np.array(pts, dtype=np.float32).ravel()
                c = np.tile(np.array(_NODO_VINCOLATO, dtype=np.float32), len(pts))
                self._gpu_upload(name_vinc, v, c)

        if self._mesh and self._mesh.nodi_caricati:
            pts = [nodi[nid] for nid in self._mesh.nodi_caricati if nid in nodi]
            if pts:
                v = np.array(pts, dtype=np.float32).ravel()
                c = np.tile(np.array(_NODO_CARICATO, dtype=np.float32), len(pts))
                self._gpu_upload(name_caric, v, c)

    # ================================================================
    # BUILD: RISULTATI (quando step/deformazione cambiano)
    # ================================================================

    def _build_risultati_buffers(self):
        self._gpu_release_group('ris_')
        if self._mesh is None:
            return

        nodi = self._get_nodi_render()
        step = self._get_step_corrente()
        if not step:
            return

        stress_vm = step.stress_vm if step else {}

        # Lookup tipo per elemento e nodo -> tipo
        elem_tipo = {}
        nodo_tipo = {}  # nid -> "carpenteria" | "barra" | "staffa"
        for obj_id, es in self._mesh.elementi_per_oggetto.items():
            t = self._mesh.tipo_oggetto.get(obj_id, "")
            for eid in es:
                elem_tipo[eid] = t
            for nid in self._mesh.nodi_per_oggetto.get(obj_id, set()):
                nodo_tipo[nid] = t

        # Calcola sigma_max separato per categoria visibile
        # Cosi' i colori sono rapportati a cio' che e' visibile a schermo
        nodi_visibili = set()
        if self._mostra_carpenteria:
            for nid, t in nodo_tipo.items():
                if t == "carpenteria":
                    nodi_visibili.add(nid)
        if self._mostra_barre:
            for nid, t in nodo_tipo.items():
                if t == "barra":
                    nodi_visibili.add(nid)
        if self._mostra_staffe:
            for nid, t in nodo_tipo.items():
                if t == "staffa":
                    nodi_visibili.add(nid)

        sigma_max_vis = 0.0
        for nid in nodi_visibili:
            sv = stress_vm.get(nid, 0.0)
            if sv > sigma_max_vis:
                sigma_max_vis = sv
        self._sigma_max_vis = sigma_max_vis

        # Pre-build sigma_ult lookup per nodo
        sigma_ult_map = {}
        if self._sigma_ult:
            for obj_id, nodi_set in self._mesh.nodi_per_oggetto.items():
                mat = self._mesh.materiale_oggetto.get(obj_id, "")
                su = self._sigma_ult.get(mat, 0.0)
                for nid in nodi_set:
                    sigma_ult_map[nid] = su

        # ── Facce colorate (triangolate) per carpenteria ──
        if self._mesh.elementi_hex:
            hex_list = [(eid, hn) for eid, hn in self._mesh.elementi_hex.items()
                        if elem_tipo.get(eid) == "carpenteria"]
            n_hex = len(hex_list)
            if n_hex > 0:
                face_verts = []
                face_sv = []
                face_su = []
                for eid, hn in hex_list:
                    for ti in range(0, len(_HEX_TRI_IDX), 3):
                        for k in range(3):
                            vi = _HEX_TRI_IDX[ti + k]
                            nid = hn[vi]
                            if nid in nodi:
                                face_verts.extend(nodi[nid])
                            else:
                                face_verts.extend([0, 0, 0])
                            face_sv.append(stress_vm.get(nid, 0.0))
                            face_su.append(sigma_ult_map.get(nid, 0.0))

                v = np.array(face_verts, dtype=np.float32)
                sv_arr = np.array(face_sv, dtype=np.float32)
                su_arr = np.array(face_su, dtype=np.float32)
                c = _stress_colors_batch(sv_arr, sigma_max_vis, su_arr).ravel()
                self._gpu_upload('ris_faces', v, c)

                # Wireframe sovrapposto
                wire_v = []
                for eid, hn in hex_list:
                    for i in range(0, len(_HEX_EDGE_IDX), 2):
                        a, b = _HEX_EDGE_IDX[i], _HEX_EDGE_IDX[i + 1]
                        if hn[a] in nodi and hn[b] in nodi:
                            wire_v.extend(nodi[hn[a]])
                            wire_v.extend(nodi[hn[b]])
                if wire_v:
                    wv = np.array(wire_v, dtype=np.float32)
                    n_wv = len(wv) // 3
                    wc = np.tile(np.array([0.15, 0.15, 0.15, 0.50], dtype=np.float32), n_wv)
                    self._gpu_upload('ris_wire', wv, wc)

        # ── Barre deformate con colori stress ──
        barre_v = []
        barre_sv = []
        barre_su = []
        for eid, bn in self._mesh.elementi_beam.items():
            if elem_tipo.get(eid) == "barra" and bn[0] in nodi and bn[1] in nodi:
                barre_v.extend(nodi[bn[0]])
                barre_v.extend(nodi[bn[1]])
                barre_sv.append(stress_vm.get(bn[0], 0.0))
                barre_sv.append(stress_vm.get(bn[1], 0.0))
                barre_su.append(sigma_ult_map.get(bn[0], 0.0))
                barre_su.append(sigma_ult_map.get(bn[1], 0.0))
        if barre_v:
            v = np.array(barre_v, dtype=np.float32)
            sv_arr = np.array(barre_sv, dtype=np.float32)
            su_arr = np.array(barre_su, dtype=np.float32)
            c = _stress_colors_batch(sv_arr, sigma_max_vis, su_arr)
            c[:, 3] = 1.0  # barre completamente opache
            self._gpu_upload('ris_barre', v, c.ravel())

        # ── Staffe deformate con colori stress ──
        staffe_v = []
        staffe_sv = []
        staffe_su = []
        for eid, bn in self._mesh.elementi_beam.items():
            if elem_tipo.get(eid) == "staffa" and bn[0] in nodi and bn[1] in nodi:
                staffe_v.extend(nodi[bn[0]])
                staffe_v.extend(nodi[bn[1]])
                staffe_sv.append(stress_vm.get(bn[0], 0.0))
                staffe_sv.append(stress_vm.get(bn[1], 0.0))
                staffe_su.append(sigma_ult_map.get(bn[0], 0.0))
                staffe_su.append(sigma_ult_map.get(bn[1], 0.0))
        if staffe_v:
            v = np.array(staffe_v, dtype=np.float32)
            sv_arr = np.array(staffe_sv, dtype=np.float32)
            su_arr = np.array(staffe_su, dtype=np.float32)
            c = _stress_colors_batch(sv_arr, sigma_max_vis, su_arr)
            c[:, 3] = 1.0  # staffe completamente opache
            self._gpu_upload('ris_staffe', v, c.ravel())

        # ── Nodi collassati (nero) ──
        if stress_vm and sigma_ult_map:
            collapse = []
            for nid, sv in stress_vm.items():
                su = sigma_ult_map.get(nid, 0.0)
                if su > 0 and sv >= su and nid in nodi:
                    collapse.extend(nodi[nid])
            if collapse:
                v = np.array(collapse, dtype=np.float32)
                c = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                            len(v) // 3)
                self._gpu_upload('ris_collapse', v, c)

        # ── Nodi speciali (posizioni deformate) ──
        self._build_nodi_speciali('ris_nodi_vinc', 'ris_nodi_caric', nodi)

    # ================================================================
    # GL INIT / RESIZE / PAINT
    # ================================================================

    def initializeGL(self):
        glClearColor(_BG_R, _BG_G, _BG_B, 1.0)
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH); glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        self._gl_ready = True
        self._dirty_griglia = True

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        if not self._gl_ready:
            return

        glEnable(GL_BLEND); glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH); glEnable(GL_POINT_SMOOTH)
        glDepthFunc(GL_LEQUAL); glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0); glPointSize(1.0)

        glClearColor(_BG_R, _BG_G, _BG_B, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w = self.width() or 1; h = self.height() or 1
        aspect = w / h

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        if self._ortho:
            hh = self.cam_dist * 0.5; hw = hh * aspect
            glOrtho(-hw - self.pan_x, hw - self.pan_x,
                    -hh - self.pan_y, hh - self.pan_y, -5000, 5000)
        else:
            gluPerspective(45.0, aspect, 0.05, 5000.0)

        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        if not self._ortho:
            glTranslatef(self.pan_x, self.pan_y, -self.cam_dist)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glRotatef(-90, 1, 0, 0)

        # ── Rebuild VBO se dirty ──
        if self._dirty_griglia:
            self._build_griglia()
            self._build_assi()
            self._dirty_griglia = False

        if self._dirty_mesh:
            self._build_mesh_buffers()
            self._dirty_mesh = False

        mostra_risultati = self._modo_vista in ("lineare", "nonlineare")

        if self._dirty_risultati and mostra_risultati:
            self._build_risultati_buffers()
            self._dirty_risultati = False

        # ── Draw griglia ──
        glLineWidth(1.0); glDepthMask(GL_FALSE)
        self._gpu_draw('griglia', GL_LINES)
        glDepthMask(GL_TRUE)

        # ── Draw assi ──
        glDepthMask(GL_FALSE)
        glLineWidth(1.8)
        self._gpu_draw('assi_pos', GL_LINES)
        glLineWidth(1.0)
        self._gpu_draw('assi_neg', GL_LINES)
        glPointSize(5.0)
        self._gpu_draw('assi_orig', GL_POINTS)
        glDepthMask(GL_TRUE)

        # ── Draw scene ──
        if mostra_risultati and self._mesh and self._get_step_corrente():
            # Facce colorate
            if self._mostra_carpenteria:
                glDepthMask(GL_TRUE)
                self._gpu_draw('ris_faces', GL_TRIANGLES)
                glLineWidth(0.8)
                self._gpu_draw('ris_wire', GL_LINES)
                glLineWidth(1.0)

            # Barre/staffe deformate
            if self._mostra_barre:
                glLineWidth(2.5)
                self._gpu_draw('ris_barre', GL_LINES)
            if self._mostra_staffe:
                glLineWidth(2.5)
                self._gpu_draw('ris_staffe', GL_LINES)
            glLineWidth(1.0)

            # Nodi collassati
            if self._mostra_crack:
                glPointSize(5.0)
                self._gpu_draw('ris_collapse', GL_POINTS)

            # Nodi speciali
            glPointSize(6.0)
            if self._mostra_vincoli:
                self._gpu_draw('ris_nodi_vinc', GL_POINTS)
            if self._mostra_carichi:
                self._gpu_draw('ris_nodi_caric', GL_POINTS)
            glPointSize(1.0)
        else:
            # Ghost objects (immediate mode - pochi oggetti)
            self._disegna_oggetti_trasparenti()

            # Mesh wireframe da VBO
            if self._mostra_carpenteria:
                glLineWidth(1.2)
                self._gpu_draw('mesh_hex', GL_LINES)
            if self._mostra_barre:
                glLineWidth(2.5)
                self._gpu_draw('mesh_barre', GL_LINES)
            if self._mostra_staffe:
                glLineWidth(2.5)
                self._gpu_draw('mesh_staffe', GL_LINES)
            glLineWidth(1.0)

            # Nodi speciali
            glPointSize(6.0)
            if self._mostra_vincoli:
                self._gpu_draw('mesh_nodi_vinc', GL_POINTS)
            if self._mostra_carichi:
                self._gpu_draw('mesh_nodi_caric', GL_POINTS)
            glPointSize(1.0)

        # ── Overlay 2D (legenda) ──
        if mostra_risultati and self._sigma_max_vis > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            self._draw_legend(painter, w, h)
            painter.end()

    # ================================================================
    # LEGENDA COLORMAP
    # ================================================================

    def _draw_legend(self, painter: QPainter, W: int, H: int) -> None:
        """Disegna la scala cromatica delle tensioni (stile pressoflessione)."""
        _FF = "Arial"
        _FS = 8

        bar_w  = 22
        bar_h  = min(442, H - 60)
        mrg_r  = 86
        x_bar  = W - mrg_r - bar_w
        y_bar  = (H - bar_h) // 2

        # Background
        bg_x = x_bar - 10
        bg_y = y_bar - 52
        bg_w = bar_w + mrg_r - 12
        bg_h = bar_h + 100
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 18, 18, 185))
        painter.drawRoundedRect(bg_x, bg_y, bg_w, bg_h, 5, 5)

        # Titolo
        f_t = QFont(_FF, _FS); f_t.setBold(True)
        painter.setFont(f_t)
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(bg_x + 4, bg_y + 18, "σ_vm [MPa]")

        # Gradiente: blu (basso) -> verde (medio) -> rosso (alto)
        grad = QLinearGradient(x_bar, y_bar + bar_h, x_bar, y_bar)
        grad.setColorAt(0.00, QColor(25,  77,  217))   # _STRESS_LOW
        grad.setColorAt(0.50, QColor(51,  204, 51))    # _STRESS_MID
        grad.setColorAt(1.00, QColor(230, 51,  38))    # _STRESS_HIGH
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawRect(x_bar, y_bar, bar_w, bar_h)

        # Ticks
        ma = self._sigma_max_vis / 1e6  # Pa -> MPa
        if ma < 1e-10:
            ma = 1.0
        ticks = [0.0, ma / 4, ma / 2, 3 * ma / 4, ma]
        f_tk = QFont(_FF, _FS - 1)
        painter.setFont(f_tk)
        for val in ticks:
            t_n    = val / ma  # 0..1
            y_tick = int(y_bar + bar_h - t_n * bar_h)
            painter.setPen(QPen(QColor(155, 155, 155), 1))
            painter.drawLine(x_bar + bar_w, y_tick, x_bar + bar_w + 5, y_tick)
            txt = "0" if val == 0 else (f"{val:.1e}" if abs(val) > 999 else f"{val:.1f}")
            painter.setPen(QPen(QColor(210, 210, 210)))
            painter.drawText(x_bar + bar_w + 7, y_tick + 4, txt)

        # Etichette basso/alto
        painter.setPen(QColor(165, 165, 165))
        painter.drawText(bg_x + 3, y_bar + bar_h + 24, "Min: 0")
        painter.drawText(bg_x + 3, y_bar - 12,         f"Max: {ma:.1f}")

    # ================================================================
    # OGGETTI TRASPARENTI (GHOST) - immediate mode (pochi oggetti)
    # ================================================================

    def _disegna_oggetti_trasparenti(self):
        for obj in self._oggetti:
            if not obj.visibile:
                continue
            tipo = self._tipo_categoria(obj)
            if tipo == "carpenteria" and not self._mostra_carpenteria:
                continue
            if tipo == "barra" and not self._mostra_barre:
                continue
            if tipo == "staffa" and not self._mostra_staffe:
                continue
            glPushMatrix()
            glTranslatef(*obj.posizione)
            glRotatef(obj.rotazione[0], 1, 0, 0)
            glRotatef(obj.rotazione[1], 0, 1, 0)
            glRotatef(obj.rotazione[2], 0, 0, 1)
            if obj.tipo in ("parallelepipedo", "cilindro", "sfera"):
                self._draw_struct_ghost(obj)
            elif obj.tipo == "barra":
                self._draw_line_ghost(obj, (0.85, 0.25, 0.25, 0.30))
            elif obj.tipo == "staffa":
                self._draw_line_ghost(obj, (1.0, 0.9, 0.1, 0.30), loop=True)
            glPopMatrix()

    def _draw_struct_ghost(self, obj):
        glDepthMask(GL_FALSE)
        glColor4f(*_STRUCT_FILL)
        if obj.tipo == "parallelepipedo":
            self._draw_box_fill(obj)
        elif obj.tipo == "cilindro":
            self._draw_cyl_fill(obj)
        elif obj.tipo == "sfera":
            self._draw_sph_fill(obj)
        glDepthMask(GL_TRUE)
        glDisable(GL_DEPTH_TEST)
        glColor4f(*_STRUCT_EDGE); glLineWidth(1.5)
        if obj.tipo == "parallelepipedo":
            self._draw_box_edges(obj)
        elif obj.tipo == "cilindro":
            self._draw_cyl_edges(obj)
        elif obj.tipo == "sfera":
            self._draw_sph_edges(obj)
        glLineWidth(1.0); glEnable(GL_DEPTH_TEST)

    def _draw_line_ghost(self, obj, col, loop=False):
        pts = obj.geometria.get("punti", [])
        if len(pts) < 2:
            return
        glColor4f(*col); glLineWidth(2.0)
        glBegin(GL_LINE_LOOP if loop else GL_LINE_STRIP)
        for p in pts:
            glVertex3f(*p)
        glEnd(); glLineWidth(1.0)

    # ── Primitive helpers ────────────────────────────────────────

    def _get_box_verts(self, obj):
        v = obj.get_vertices_local()
        if len(v) < 8:
            from elementi.modello_3d import calcola_vertici
            v = calcola_vertici("parallelepipedo", obj.geometria)
        return v

    def _draw_box_fill(self, obj):
        v = self._get_box_verts(obj)
        glBegin(GL_QUADS)
        for face in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                     [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]):
            for idx in face:
                glVertex3f(*v[idx])
        glEnd()

    def _draw_box_edges(self, obj):
        v = self._get_box_verts(obj)
        glBegin(GL_LINES)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                     (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            glVertex3f(*v[a]); glVertex3f(*v[b])
        glEnd()

    def _draw_cyl_fill(self, obj):
        A = float(obj.geometria.get("altezza", 3.0))
        R = float(obj.geometria.get("raggio", 0.2))
        N = _CYL_N
        for z in (0, A):
            glBegin(GL_TRIANGLE_FAN); glVertex3f(0, 0, z)
            for i in range(N + 1):
                t = 2 * math.pi * (i % N) / N
                glVertex3f(R * math.cos(t), R * math.sin(t), z)
            glEnd()
        glBegin(GL_QUAD_STRIP)
        for i in range(N + 1):
            t = 2 * math.pi * (i % N) / N
            x, y = R * math.cos(t), R * math.sin(t)
            glVertex3f(x, y, 0); glVertex3f(x, y, A)
        glEnd()

    def _draw_cyl_edges(self, obj):
        A = float(obj.geometria.get("altezza", 3.0))
        R = float(obj.geometria.get("raggio", 0.2))
        N = _CYL_N
        for z in (0, A):
            glBegin(GL_LINE_LOOP)
            for i in range(N):
                t = 2 * math.pi * i / N
                glVertex3f(R * math.cos(t), R * math.sin(t), z)
            glEnd()
        glBegin(GL_LINES)
        for i in (0, N // 4, N // 2, 3 * N // 4):
            t = 2 * math.pi * i / N
            x, y = R * math.cos(t), R * math.sin(t)
            glVertex3f(x, y, 0); glVertex3f(x, y, A)
        glEnd()

    def _draw_sph_fill(self, obj):
        R = float(obj.geometria.get("raggio", 0.2))
        NL, NM = _SPH_NL, _SPH_NM
        for il in range(NL):
            phi0 = math.pi * (il / NL - 0.5)
            phi1 = math.pi * ((il + 1) / NL - 0.5)
            glBegin(GL_QUAD_STRIP)
            for im in range(NM + 1):
                theta = 2 * math.pi * (im % NM) / NM
                for phi in (phi0, phi1):
                    glVertex3f(R * math.cos(phi) * math.cos(theta),
                               R * math.cos(phi) * math.sin(theta),
                               R * math.sin(phi))
            glEnd()

    def _draw_sph_edges(self, obj):
        R = float(obj.geometria.get("raggio", 0.2))
        NL, NM = _SPH_NL, _SPH_NM
        for il in range(0, NL + 1, 2):
            phi = math.pi * (il / NL - 0.5)
            glBegin(GL_LINE_LOOP)
            for im in range(NM):
                t = 2 * math.pi * im / NM
                glVertex3f(R * math.cos(phi) * math.cos(t),
                           R * math.cos(phi) * math.sin(t),
                           R * math.sin(phi))
            glEnd()
        step = max(1, NM // 8)
        for im in range(0, NM, step):
            t = 2 * math.pi * im / NM
            glBegin(GL_LINE_STRIP)
            for il in range(NL + 1):
                phi = math.pi * (il / NL - 0.5)
                glVertex3f(R * math.cos(phi) * math.cos(t),
                           R * math.cos(phi) * math.sin(t),
                           R * math.sin(phi))
            glEnd()

    # ================================================================
    # UTILITY
    # ================================================================

    @staticmethod
    def _tipo_categoria(obj) -> str:
        if obj.tipo in ("parallelepipedo", "cilindro", "sfera"):
            return "carpenteria"
        if obj.tipo == "barra":
            return "barra"
        if obj.tipo == "staffa":
            return "staffa"
        return "unknown"

    # ================================================================
    # MOUSE / KEYBOARD
    # ================================================================

    def mousePressEvent(self, event):
        self._last_pos = event.pos()
        self._mouse_btn = event.button()

    def mouseMoveEvent(self, event):
        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()
        if self._mouse_btn == Qt.MiddleButton:
            self.rot_y += dx * 0.40; self.rot_x += dy * 0.40
            self._last_pos = event.pos(); self.update(); return
        if self._mouse_btn == Qt.RightButton:
            s = self.cam_dist * 0.0018
            self.pan_x += dx * s; self.pan_y -= dy * s
            self._last_pos = event.pos(); self.update(); return
        self._last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        self._mouse_btn = None

    def wheelEvent(self, event):
        d = event.angleDelta().y()
        self.cam_dist = max(0.5, self.cam_dist * (0.88 if d > 0 else 1.12))
        self.update()
