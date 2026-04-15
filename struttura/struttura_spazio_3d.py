"""
struttura_spazio_3d.py – OpenGL 3D viewer per la visualizzazione della struttura.

Renderizza nodi, aste (beam), shell, vincoli e carichi a partire dai dati
parsati dal testo strutturale.

Controlli camera (stile Blender):
  Middle-drag  → orbit
  Right-drag   → pan
  Scroll       → zoom
"""

import math
import numpy as np

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QFont, QColor

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINE_SMOOTH, GL_POINT_SMOOTH,
    GL_LINE_SMOOTH_HINT, GL_POINT_SMOOTH_HINT, GL_NICEST,
    GL_MODELVIEW, GL_PROJECTION,
    GL_LINES, GL_LINE_LOOP, GL_LINE_STRIP,
    GL_TRIANGLE_FAN, GL_QUADS,
    GL_POINTS,
    GL_FALSE, GL_TRUE,
    GL_PROJECTION_MATRIX, GL_MODELVIEW_MATRIX, GL_VIEWPORT,
    glClearColor, glClear, glEnable, glDisable,
    glBlendFunc, glDepthFunc, glDepthMask,
    glLineWidth, glPointSize,
    glBegin, glEnd, glVertex3f, glColor3f, glColor4f,
    glMatrixMode, glLoadIdentity, glViewport, glOrtho,
    glTranslatef, glRotatef,
    glGetDoublev, glGetIntegerv,
    glPushMatrix, glPopMatrix,
    glHint,
)
from OpenGL.GLU import gluPerspective, gluProject


# ── Palette ──────────────────────────────────────────────────────
_BG_R, _BG_G, _BG_B = 40/255, 40/255, 40/255
_PREV_BG_R, _PREV_BG_G, _PREV_BG_B = 50/255, 50/255, 50/255

_GF_R, _GF_G, _GF_B = 0.20, 0.20, 0.20   # griglia fine
_GC_R, _GC_G, _GC_B = 0.27, 0.27, 0.27   # griglia grossa

_AX_X = (0.862, 0.200, 0.200)   # rosso
_AX_Y = (0.310, 0.620, 0.165)   # verde
_AX_Z = (0.161, 0.408, 0.784)   # blu

# Colori strutturali (Aggiornati)
_COL_NODO        = (0.39, 0.71, 1.00, 0.90)   # azzurrino
_COL_NODO_VINC   = (0.40, 0.75, 0.90, 1.00)   # azzurrino vincolato
_COL_BEAM        = (0.80, 0.80, 0.80, 1.00)   # grigio chiaro
_COL_SHELL_FILL  = (0.50, 0.70, 0.50, 0.18)   # verde trasparente
_COL_SHELL_EDGE  = (0.55, 0.75, 0.55, 0.70)
_COL_VINCOLO     = (0.30, 0.85, 0.40, 0.7)   # verde
_COL_CARICO_N    = (0.95, 0.30, 0.30, 0.90)   # rosso
_COL_CARICO_D    = (0.95, 0.65, 0.15, 0.90)   # arancione
_COL_CARICO_S    = (0.95, 0.85, 0.25, 0.90)   # giallo

# Colori selezione
_COL_SEL_EDGE = (1.00, 0.60, 0.00, 1.00)   # arancio selezione
_COL_SEL_FILL = (1.00, 0.55, 0.00, 0.22)   # arancio trasparente

# Colori assi locali (direzione elementi)
_COL_LOC_X = (0.95, 0.35, 0.35)
_COL_LOC_Y = (0.45, 0.90, 0.40)
_COL_LOC_Z = (0.40, 0.65, 1.00)

# Dimensioni rendering
_NODE_SIZE   = 7.0
_BEAM_WIDTH  = 2.5
_ARROW_LEN   = 0.75
_VINCOLO_SIZE = 0.25
_LOC_AXIS_LEN = 0.45


class StrutturaSpazio3D(QOpenGLWidget):
    """Viewer 3D per la visualizzazione di strutture definite via testo."""

    # Emette (tipo, id) quando l'utente clicca un oggetto del modello.
    # tipo ∈ {"nodo", "beam", "shell", ""} (stringa vuota → click a vuoto)
    oggetto_selezionato = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Dati strutturali (dal parser)
        self._dati = None

        # Camera
        self.cam_dist: float = 20.0
        self.rot_x:    float = 25.0
        self.rot_y:    float = -40.0
        self.pan_x:    float = 0.0
        self.pan_y:    float = 0.0
        self._ortho:   bool  = False

        self._last_pos  = QPoint()
        self._mouse_btn = None
        self._preview_mode: bool = False

        # Flag di visibilità (gestiti dai bottoni toolbar)
        self.show_nodi: bool    = True
        self.show_beams: bool   = True
        self.show_shells: bool  = True
        self.show_vincoli: bool = True
        self.show_carichi: bool = True
        self.show_labels: bool  = False
        self.show_direzioni: bool = False

        # Cache matrici GL per proiezione label
        self._gl_model = None
        self._gl_proj  = None
        self._gl_vp    = None

        # Selezione corrente: ("nodo"|"beam"|"shell", id) oppure (None, -1)
        self._sel_kind: str | None = None
        self._sel_id: int = -1

        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------ public

    def aggiorna_dati(self, dati: dict):
        """Imposta i dati strutturali parsati e ridisegna."""
        self._dati = dati
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
        if not self._dati or not self._dati.get("nodi"):
            self.cam_dist = 20.0
            self.pan_x = self.pan_y = 0.0
            self.update()
            return
        pts = np.array(list(self._dati["nodi"].values()), dtype=float)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        ctr = (mn + mx) / 2.0
        diag = float(np.linalg.norm(mx - mn))

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
        self.pan_x = float(-ctr_eye[0])
        self.pan_y = float(-ctr_eye[1])
        self.cam_dist = max(diag / (2.0 * math.tan(math.radians(22.5))) * 1.25, 2.0)
        self.update()

    def set_visibilita(self, **flags):
        """Aggiorna flag di visibilità in un colpo solo.
        Esempio: set_visibilita(show_nodi=True, show_labels=False)."""
        valide = {"show_nodi", "show_beams", "show_shells", "show_vincoli",
                  "show_carichi", "show_labels", "show_direzioni"}
        for k, v in flags.items():
            if k in valide:
                setattr(self, k, bool(v))
        self.update()

    def set_selezione(self, kind: str | None, oid: int = -1):
        """Imposta l'oggetto selezionato (halo arancio) senza emettere segnale."""
        self._sel_kind = kind if kind else None
        self._sel_id = int(oid) if kind else -1
        self.update()

    def clear_selezione(self):
        self.set_selezione(None, -1)

    def reset_view(self):
        self.rot_x = 25.0; self.rot_y = -40.0
        self.pan_x = 0.0; self.pan_y = 0.0
        self.cam_dist = 20.0; self._ortho = False
        self.update()

    # ------------------------------------------------------------------ helper logica carichi

    def _get_max_magnitudes(self):
        """Calcola la magnitudo massima per carichi nodali, distribuiti e shell."""
        max_n = 1e-12
        for (_, fx, fy, fz) in self._dati.get("carichi_nodali", []):
            mag = math.sqrt(fx*fx + fy*fy + fz*fz)
            if mag > max_n: max_n = mag

        max_d = 1e-12
        for (_, wx, wy, wz) in self._dati.get("carichi_distribuiti", []):
            mag = math.sqrt(wx*wx + wy*wy + wz*wz)
            if mag > max_d: max_d = mag

        max_s = 1e-12
        for (_, qx, qy, qz) in self._dati.get("carichi_shell", []):
            mag = math.sqrt(qx*qx + qy*qy + qz*qz)
            if mag > max_s: max_s = mag

        return max_n, max_d, max_s

    def _calc_load_scale(self, mag, max_mag, base_len=_ARROW_LEN):
        """Calcola la scala dinamica in base all'intensità del carico (dal 30% al 100% di base_len)."""
        if mag < 1e-12: return 0.0
        arrow_len = base_len * (0.3 + 0.7 * (mag / max_mag))
        return arrow_len / mag

    # ------------------------------------------------------------------ GL

    def initializeGL(self):
        glClearColor(_BG_R, _BG_G, _BG_B, 1.0)
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH); glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.beginNativePainting()

        if self._preview_mode:
            glClearColor(_PREV_BG_R, _PREV_BG_G, _PREV_BG_B, 1.0)
        else:
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

        w = self.width() or 1
        h = self.height() or 1
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
        glRotatef(-90, 1, 0, 0)

        self._gl_model = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._gl_proj  = glGetDoublev(GL_PROJECTION_MATRIX)
        self._gl_vp    = glGetIntegerv(GL_VIEWPORT)

        if not self._preview_mode:
            self._disegna_griglia()
            self._disegna_assi()

        if self._dati:
            if self._preview_mode or self.show_shells:
                self._disegna_shell()
            if self._preview_mode or self.show_beams:
                self._disegna_aste()
            if self._preview_mode or self.show_vincoli:
                self._disegna_vincoli()
            if self._preview_mode or self.show_carichi:
                self._disegna_carichi_distribuiti()
                self._disegna_carichi_shell()
                self._disegna_carichi_nodali()
            if self._preview_mode or self.show_nodi:
                self._disegna_nodi()
            if (not self._preview_mode) and self.show_direzioni:
                self._disegna_direzioni_locali()

            if (not self._preview_mode) and self._sel_kind is not None:
                self._disegna_halo_selezione()

        glDisable(GL_DEPTH_TEST)
        painter.endNativePainting()

        if self._dati and not self._preview_mode and self.show_labels:
            self._disegna_tutte_labels(painter)
            
        painter.end()

    # ------------------------------------------------------------------ grid

    def _disegna_griglia(self):
        DIM = 100; FADE = 40; SEG_L = 2
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
                if step == 1 and (i % 5 == 0): continue
                for j in range(-DIM, DIM, SEG_L):
                    y1, y2 = j, j + SEG_L
                    a1v = get_alpha(i, y1); a2v = get_alpha(i, y2)
                    if a1v > 0.01 or a2v > 0.01:
                        glColor4f(cr, cg, cb, a1v); glVertex3f(i, y1, 0)
                        glColor4f(cr, cg, cb, a2v); glVertex3f(i, y2, 0)
                    x1, x2 = j, j + SEG_L
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

    # ------------------------------------------------------------------ nodi

    def _disegna_nodi(self):
        nodi = self._dati.get("nodi", {})
        vincoli = self._dati.get("vincoli", {})
        glPointSize(_NODE_SIZE)
        glBegin(GL_POINTS)
        for nid, (x, y, z) in nodi.items():
            if nid in vincoli:
                glColor4f(*_COL_NODO_VINC)
            else:
                glColor4f(*_COL_NODO)
            glVertex3f(x, y, z)
        glEnd()
        glPointSize(1.0)

    def _project(self, x, y, z):
        """Proietta un punto 3D in coordinate schermo (sx, sy)."""
        if self._gl_model is None or self._gl_proj is None or self._gl_vp is None:
            return -1000, -1000
            
        try:
            sx, sy, sz = gluProject(float(x), float(y), float(z),
                                    self._gl_model, self._gl_proj, self._gl_vp)
            if sz < 0.0 or sz > 1.0:
                return -1000, -1000
                
            return int(sx), int(self.height() - sy)
        except Exception:
            return -1000, -1000

    def _disegna_tutte_labels(self, painter: QPainter):
        if self._gl_model is None:
            return

        font_label = QFont("Consolas", 9, QFont.Bold)
        painter.setFont(font_label)

        nodi = self._dati.get("nodi", {})
        vincoli = self._dati.get("vincoli", {})
        aste = self._dati.get("aste", {})

        # ── Nodi: N<id> ──
        if self.show_nodi:
            painter.setPen(QColor(100, 180, 255, 230))
            for nid, (x, y, z) in nodi.items():
                sx, sy = self._project(x, y, z)
                if sx == -1000: continue
                painter.drawText(sx + 8, sy - 8, f"N{nid}")

        # ── Aste: B<id> al punto medio ──
        if self.show_beams:
            painter.setPen(QColor(220, 220, 220, 240))
            for bid, asta in aste.items():
                ni, nj = asta["nodo_i"], asta["nodo_j"]
                if ni in nodi and nj in nodi:
                    pi, pj = nodi[ni], nodi[nj]
                    mx = (pi[0] + pj[0]) / 2.0
                    my = (pi[1] + pj[1]) / 2.0
                    mz = (pi[2] + pj[2]) / 2.0
                    sx, sy = self._project(mx, my, mz)
                    if sx != -1000:
                        painter.drawText(sx + 6, sy - 6, f"B{bid}")

        # ── Shell: S<id> al baricentro ──
        if self.show_shells:
            painter.setPen(QColor(150, 210, 150, 210))
            for sid, sh in self._dati.get("shell", {}).items():
                pts = [nodi[n] for n in sh["nodi"] if n in nodi]
                if len(pts) >= 3:
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    cz = sum(p[2] for p in pts) / len(pts)
                    sx, sy = self._project(cx, cy, cz)
                    if sx != -1000:
                        painter.drawText(sx + 4, sy - 4, f"S{sid}")

        # ── Vincoli: V<id> ──
        if self.show_vincoli:
            painter.setPen(QColor(100, 230, 130, 240))
            for nid in vincoli.keys():
                if nid not in nodi: continue
                x, y, z = nodi[nid]
                sx, sy = self._project(x, y, z - _VINCOLO_SIZE * 0.8)
                if sx != -1000:
                    painter.drawText(sx + 10, sy + 14, f"V{nid}")

        if not self.show_carichi:
            return

        # Otteniamo i massimi per calcolare lo scaling
        max_n, max_d, max_s = self._get_max_magnitudes()

        # ── Carichi nodali: F<id> ──
        painter.setPen(QColor(255, 120, 120, 240))
        for idx, (nid, fx, fy, fz) in enumerate(self._dati.get("carichi_nodali", []), 1):
            if nid not in nodi: continue
            x, y, z = nodi[nid]
            mag = math.sqrt(fx*fx + fy*fy + fz*fz)
            if mag < 1e-12: continue
            scale = self._calc_load_scale(mag, max_n)
            # Punta della freccia (x + fx*scale, ...)
            sx, sy = self._project(x + fx * scale, y + fy * scale, z + fz * scale)
            if sx != -1000:
                painter.drawText(sx + 6, sy - 6, f"F{idx}")

        # ── Carichi distribuiti: Q<id> ──
        painter.setPen(QColor(255, 190, 80, 240))
        for idx, (bid, wx, wy, wz) in enumerate(self._dati.get("carichi_distribuiti", []), 1):
            if bid not in aste: continue
            asta = aste[bid]
            ni, nj = asta["nodo_i"], asta["nodo_j"]
            if ni not in nodi or nj not in nodi: continue

            pi, pj = nodi[ni], nodi[nj]
            mx = pi[0] + (pj[0] - pi[0]) * 0.5
            my = pi[1] + (pj[1] - pi[1]) * 0.5
            mz = pi[2] + (pj[2] - pi[2]) * 0.5

            mag = math.sqrt(wx*wx + wy*wy + wz*wz)
            if mag < 1e-12: continue

            # La scala è moltiplicata per 0.8 nei distribuiti per tenerli proporzionalmente più corti
            scale = self._calc_load_scale(mag, max_d, _ARROW_LEN * 0.8)
            
            # Coda del carico (origine da cui parte verso l'asta)
            tail_x = mx + wx * scale
            tail_y = my + wy * scale
            tail_z = mz + wz * scale
            
            sx, sy = self._project(tail_x, tail_y, tail_z)
            if sx != -1000:
                painter.drawText(sx + 6, sy + 6, f"Q{idx}")

        # ── Carichi shell: Qs<id> al baricentro della shell ──
        shells = self._dati.get("shell", {})
        painter.setPen(QColor(240, 220, 90, 240))
        for idx, (sid, qx, qy, qz) in enumerate(self._dati.get("carichi_shell", []), 1):
            if sid not in shells: continue
            pts = [nodi[n] for n in shells[sid]["nodi"] if n in nodi]
            if len(pts) < 3: continue

            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cz = sum(p[2] for p in pts) / len(pts)

            mag = math.sqrt(qx*qx + qy*qy + qz*qz)
            if mag < 1e-12: continue
            scale = self._calc_load_scale(mag, max_s, _ARROW_LEN * 0.85)
            dx = qx * scale; dy = qy * scale; dz = qz * scale
            # Coerente con _disegna_carichi_shell: la freccia sta sopra la shell
            if qz < 0:
                # tail sopra, head sulla shell
                lx, ly, lz = cx - dx, cy - dy, cz - dz
            else:
                lx, ly, lz = cx + dx, cy + dy, cz + dz
            sx, sy = self._project(lx, ly, lz)
            if sx != -1000:
                painter.drawText(sx + 6, sy - 6, f"Qs{idx}")

    # ------------------------------------------------------------------ aste

    def _disegna_aste(self):
        nodi = self._dati.get("nodi", {})
        aste = self._dati.get("aste", {})
        if not aste: return

        glLineWidth(_BEAM_WIDTH)
        glBegin(GL_LINES)
        for bid, asta in aste.items():
            ni, nj = asta["nodo_i"], asta["nodo_j"]
            if ni in nodi and nj in nodi:
                pi, pj = nodi[ni], nodi[nj]
                glColor4f(*_COL_BEAM)
                glVertex3f(*pi)
                glVertex3f(*pj)
        glEnd()
        glLineWidth(1.0)

    # ------------------------------------------------------------------ shell

    def _disegna_shell(self):
        nodi = self._dati.get("nodi", {})
        shells = self._dati.get("shell", {})
        if not shells: return

        glDepthMask(GL_FALSE)
        for sid, sh in shells.items():
            pts = [nodi[n] for n in sh["nodi"] if n in nodi]
            if len(pts) < 3: continue
            glColor4f(*_COL_SHELL_FILL)
            glBegin(GL_TRIANGLE_FAN)
            for p in pts: glVertex3f(*p)
            glEnd()
        glDepthMask(GL_TRUE)

        glLineWidth(1.5)
        for sid, sh in shells.items():
            pts = [nodi[n] for n in sh["nodi"] if n in nodi]
            if len(pts) < 3: continue
            glColor4f(*_COL_SHELL_EDGE)
            glBegin(GL_LINE_LOOP)
            for p in pts: glVertex3f(*p)
            glEnd()
        glLineWidth(1.0)

    # ------------------------------------------------------------------ vincoli

    def _disegna_vincoli(self):
        nodi = self._dati.get("nodi", {})
        vincoli = self._dati.get("vincoli", {})
        if not vincoli: return

        s = _VINCOLO_SIZE
        for nid, vals in vincoli.items():
            if nid not in nodi: continue
            x, y, z = nodi[nid]
            is_incastro = all(v == 1 for v in vals[:6])
            is_cerniera = all(v == 1 for v in vals[:3]) and all(v == 0 for v in vals[3:6])

            glColor4f(*_COL_VINCOLO)

            if is_incastro:
                glBegin(GL_QUADS)
                glVertex3f(x - s, y - s, z - s * 0.5)
                glVertex3f(x + s, y - s, z - s * 0.5)
                glVertex3f(x + s, y + s, z - s * 0.5)
                glVertex3f(x - s, y + s, z - s * 0.5)
                glEnd()
                glLineWidth(1.5)
                glBegin(GL_LINES)
                for i in range(5):
                    t = -s + i * s * 0.5
                    glVertex3f(x + t, y, z - s * 0.5)
                    glVertex3f(x + t - s * 0.3, y, z - s * 0.8)
                glEnd()
                glLineWidth(1.0)

            elif is_cerniera:
                glBegin(GL_LINE_LOOP)
                glVertex3f(x, y, z)
                glVertex3f(x - s, y, z - s)
                glVertex3f(x + s, y, z - s)
                glEnd()
                N = 12
                r = s * 0.15
                glBegin(GL_LINE_LOOP)
                for i in range(N):
                    ang = 2.0 * math.pi * i / N
                    glVertex3f(x + r * math.cos(ang), y, z - s + r * math.sin(ang))
                glEnd()

            else:
                glBegin(GL_LINE_LOOP)
                glVertex3f(x, y, z - s)
                glVertex3f(x - s * 0.5, y, z - s * 0.5)
                glVertex3f(x, y, z)
                glVertex3f(x + s * 0.5, y, z - s * 0.5)
                glEnd()

    # ------------------------------------------------------------------ carichi

    def _disegna_carichi_nodali(self):
        nodi = self._dati.get("nodi", {})
        carichi = self._dati.get("carichi_nodali", [])
        if not carichi: return

        max_n, _, _ = self._get_max_magnitudes()

        glColor4f(*_COL_CARICO_N)
        for (nid, fx, fy, fz) in carichi:
            if nid not in nodi: continue
            x, y, z = nodi[nid]
            mag = math.sqrt(fx*fx + fy*fy + fz*fz)
            if mag < 1e-12: continue
            
            scale = self._calc_load_scale(mag, max_n)
            self._disegna_freccia(x, y, z, fx * scale, fy * scale, fz * scale)

    def _disegna_carichi_distribuiti(self):
        nodi = self._dati.get("nodi", {})
        aste = self._dati.get("aste", {})
        carichi = self._dati.get("carichi_distribuiti", [])
        if not carichi: return

        _, max_d, _ = self._get_max_magnitudes()

        glColor4f(*_COL_CARICO_D)
        for (bid, wx, wy, wz) in carichi:
            if bid not in aste: continue
            asta = aste[bid]
            ni, nj = asta["nodo_i"], asta["nodo_j"]
            if ni not in nodi or nj not in nodi: continue
            
            pi = np.array(nodi[ni], dtype=float)
            pj = np.array(nodi[nj], dtype=float)
            L = float(np.linalg.norm(pj - pi))
            if L < 1e-6: continue
            
            n_arrows = max(3, int(L / 0.8))
            mag = math.sqrt(wx*wx + wy*wy + wz*wz)
            if mag < 1e-12: continue
            
            scale = self._calc_load_scale(mag, max_d, _ARROW_LEN * 0.8)
            
            for i in range(n_arrows + 1):
                t = i / n_arrows
                pt = pi + t * (pj - pi)
                self._disegna_freccia(float(pt[0]), float(pt[1]), float(pt[2]),
                                      wx * scale, wy * scale, wz * scale)
            
            glLineWidth(1.2)
            glBegin(GL_LINE_STRIP)
            for i in range(n_arrows + 1):
                t = i / n_arrows
                pt = pi + t * (pj - pi)
                glVertex3f(float(pt[0] + wx * scale),
                           float(pt[1] + wy * scale),
                           float(pt[2] + wz * scale))
            glEnd()
            glLineWidth(1.0)

    def _disegna_freccia(self, x, y, z, dx, dy, dz):
        """Freccia da (x,y,z) (coda) verso (x+dx,y+dy,z+dz) (punta).
        La direzione della punta segue quindi quella del vettore (dx,dy,dz)."""
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x, y, z)                  # coda (applicazione)
        glVertex3f(x + dx, y + dy, z + dz)   # punta
        glEnd()

        mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        if mag < 1e-12: return

        head = 0.15
        ndx, ndy, ndz = dx/mag, dy/mag, dz/mag

        if abs(ndz) < 0.9:
            px, py, pz = -ndy, ndx, 0.0
        else:
            px, py, pz = 0.0, -ndz, ndy

        pm = math.sqrt(px*px + py*py + pz*pz)
        if pm > 1e-12:
            px /= pm; py /= pm; pz /= pm

        # Punta della freccia
        tipx = x + dx
        tipy = y + dy
        tipz = z + dz
        # Base delle alette: indietreggia di ~25% dalla punta
        bx = tipx - ndx * (mag * 0.25)
        by = tipy - ndy * (mag * 0.25)
        bz = tipz - ndz * (mag * 0.25)

        glBegin(GL_LINES)
        glVertex3f(tipx, tipy, tipz)
        glVertex3f(bx + px * head, by + py * head, bz + pz * head)
        glVertex3f(tipx, tipy, tipz)
        glVertex3f(bx - px * head, by - py * head, bz - pz * head)
        glEnd()
        glLineWidth(1.0)

    # ------------------------------------------------------------------ shell loads

    def _triangoli_shell(self, pts):
        """Suddivide i punti di una shell (3 o 4) in una lista di triangoli."""
        if len(pts) == 3:
            return [(pts[0], pts[1], pts[2])]
        if len(pts) == 4:
            return [(pts[0], pts[1], pts[2]),
                    (pts[0], pts[2], pts[3])]
        # fallback fan
        return [(pts[0], pts[i], pts[i+1]) for i in range(1, len(pts)-1)]

    def _campiona_punti_shell(self, pts, n=4):
        """Restituisce un reticolo di punti interni/bordo alla shell (4 o 3 nodi).
        Usato per distribuire le frecce di un carico di superficie."""
        out = []
        if len(pts) == 4:
            p0 = np.array(pts[0], dtype=float)
            p1 = np.array(pts[1], dtype=float)
            p2 = np.array(pts[2], dtype=float)
            p3 = np.array(pts[3], dtype=float)
            for i in range(n + 1):
                u = i / n
                for j in range(n + 1):
                    v = j / n
                    # Bilineare
                    p = (1-u)*(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3
                    out.append(p)
        elif len(pts) == 3:
            p0 = np.array(pts[0], dtype=float)
            p1 = np.array(pts[1], dtype=float)
            p2 = np.array(pts[2], dtype=float)
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    k = n - i - j
                    p = (i*p0 + j*p1 + k*p2) / n
                    out.append(p)
        return out

    def _disegna_carichi_shell(self):
        """Disegna un reticolo di frecce sulla parte SUPERIORE della shell.
        Convenzione segno: carico con qz<0 → punta della freccia sulla shell
        (freccia sopra, testa in basso); qz>=0 → base della freccia sulla
        shell (freccia sopra, testa in alto)."""
        nodi = self._dati.get("nodi", {})
        shells = self._dati.get("shell", {})
        carichi = self._dati.get("carichi_shell", [])
        if not carichi or not shells: return

        _, _, max_s = self._get_max_magnitudes()

        glColor4f(*_COL_CARICO_S)
        for (sid, qx, qy, qz) in carichi:
            if sid not in shells: continue
            sh = shells[sid]
            pts = [nodi[n] for n in sh["nodi"] if n in nodi]
            if len(pts) < 3: continue

            mag = math.sqrt(qx*qx + qy*qy + qz*qz)
            if mag < 1e-12: continue

            # Scala dinamica
            scale = self._calc_load_scale(mag, max_s, _ARROW_LEN * 0.85)
            dx = qx * scale
            dy = qy * scale
            dz = qz * scale

            # Le frecce stanno sempre sulla parte "superiore" della faccia:
            # se qz < 0 la punta tocca la shell (tail = p - vec, head = p)
            # se qz ≥ 0 la base tocca la shell (tail = p,       head = p + vec)
            tail_offset = (-dx, -dy, -dz) if qz < 0 else (0.0, 0.0, 0.0)

            glColor4f(*_COL_CARICO_S)
            n_div = 3 if len(pts) == 4 else 4
            sample_pts = self._campiona_punti_shell(pts, n=n_div)
            for p in sample_pts:
                tx = float(p[0]) + tail_offset[0]
                ty = float(p[1]) + tail_offset[1]
                tz = float(p[2]) + tail_offset[2]
                self._disegna_freccia(tx, ty, tz, dx, dy, dz)

    # ------------------------------------------------------------------ direzioni locali

    @staticmethod
    def _beam_local_axes(pi, pj):
        """Calcola gli assi locali di un'asta.
        x = direzione asta, z = verticale globale proiettata, y = z × x.
        Se asta verticale, usa X globale come riferimento."""
        pi = np.asarray(pi, dtype=float)
        pj = np.asarray(pj, dtype=float)
        ex = pj - pi
        L = float(np.linalg.norm(ex))
        if L < 1e-12:
            return None
        ex /= L
        gz = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ex, gz))) > 0.99:
            # asta verticale → usa X globale
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = gz
        ey = np.cross(ref, ex)
        ny = float(np.linalg.norm(ey))
        if ny < 1e-12:
            return None
        ey /= ny
        ez = np.cross(ex, ey)
        return ex, ey, ez

    @staticmethod
    def _shell_local_axes(pts):
        """Calcola gli assi locali di una shell.
        x = lungo il primo lato, z = normale, y = z × x."""
        if len(pts) < 3:
            return None
        p0 = np.array(pts[0], dtype=float)
        p1 = np.array(pts[1], dtype=float)
        p2 = np.array(pts[2], dtype=float)
        ex = p1 - p0
        nx = float(np.linalg.norm(ex))
        if nx < 1e-12: return None
        ex /= nx
        v = p2 - p0
        ez = np.cross(ex, v)
        nz = float(np.linalg.norm(ez))
        if nz < 1e-12: return None
        ez /= nz
        ey = np.cross(ez, ex)
        return ex, ey, ez

    def _disegna_terna(self, cx, cy, cz, ex, ey, ez, length=_LOC_AXIS_LEN):
        """Disegna tre frecce (rosso/verde/blu) al centro (cx,cy,cz)."""
        for d, col in ((ex, _COL_LOC_X),
                       (ey, _COL_LOC_Y),
                       (ez, _COL_LOC_Z)):
            glColor4f(col[0], col[1], col[2], 0.95)
            self._disegna_freccia(cx, cy, cz,
                                  float(d[0] * length),
                                  float(d[1] * length),
                                  float(d[2] * length))

    def _disegna_direzioni_locali(self):
        """Disegna la terna locale (x,y,z) al centro di ogni beam/shell visibile."""
        nodi = self._dati.get("nodi", {})

        if self.show_beams:
            aste = self._dati.get("aste", {})
            for bid, asta in aste.items():
                ni, nj = asta["nodo_i"], asta["nodo_j"]
                if ni not in nodi or nj not in nodi: continue
                pi, pj = nodi[ni], nodi[nj]
                axes = self._beam_local_axes(pi, pj)
                if axes is None: continue
                cx = (pi[0] + pj[0]) * 0.5
                cy = (pi[1] + pj[1]) * 0.5
                cz = (pi[2] + pj[2]) * 0.5
                self._disegna_terna(cx, cy, cz, *axes)

        if self.show_shells:
            shells = self._dati.get("shell", {})
            for sid, sh in shells.items():
                pts = [nodi[n] for n in sh["nodi"] if n in nodi]
                if len(pts) < 3: continue
                axes = self._shell_local_axes(pts)
                if axes is None: continue
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                cz = sum(p[2] for p in pts) / len(pts)
                self._disegna_terna(cx, cy, cz, *axes)

    # ------------------------------------------------------------------ halo selezione

    def _disegna_halo_selezione(self):
        """Disegna un evidenziatore arancio per l'oggetto correntemente
        selezionato (nodo / beam / shell)."""
        if self._sel_kind is None or self._dati is None:
            return
        nodi = self._dati.get("nodi", {})

        if self._sel_kind == "nodo":
            if self._sel_id not in nodi:
                return
            x, y, z = nodi[self._sel_id]
            glPointSize(_NODE_SIZE + 8.0)
            glBegin(GL_POINTS)
            glColor4f(*_COL_SEL_EDGE)
            glVertex3f(x, y, z)
            glEnd()
            glPointSize(1.0)

        elif self._sel_kind == "beam":
            aste = self._dati.get("aste", {})
            asta = aste.get(self._sel_id)
            if asta is None: return
            ni, nj = asta["nodo_i"], asta["nodo_j"]
            if ni not in nodi or nj not in nodi: return
            pi, pj = nodi[ni], nodi[nj]
            glLineWidth(_BEAM_WIDTH + 3.5)
            glBegin(GL_LINES)
            glColor4f(*_COL_SEL_EDGE)
            glVertex3f(*pi); glVertex3f(*pj)
            glEnd()
            glLineWidth(1.0)

        elif self._sel_kind == "shell":
            shells = self._dati.get("shell", {})
            sh = shells.get(self._sel_id)
            if sh is None: return
            pts = [nodi[n] for n in sh["nodi"] if n in nodi]
            if len(pts) < 3: return
            glDepthMask(GL_FALSE)
            glColor4f(*_COL_SEL_FILL)
            glBegin(GL_TRIANGLE_FAN)
            for p in pts: glVertex3f(*p)
            glEnd()
            glDepthMask(GL_TRUE)
            glLineWidth(3.0)
            glColor4f(*_COL_SEL_EDGE)
            glBegin(GL_LINE_LOOP)
            for p in pts: glVertex3f(*p)
            glEnd()
            glLineWidth(1.0)

    # ------------------------------------------------------------------ picking

    def _pick_at(self, mx: int, my: int) -> tuple[str | None, int]:
        """Raycasting di selezione. Ritorna (tipo, id) del primo hit.
        Priorità: nodi → beams → shells."""
        if self._gl_model is None or self._dati is None:
            return (None, -1)
        try:
            vp = self._gl_vp
            w = self.width() or 1
            h = self.height() or 1
            gmx = float(mx) * (float(vp[2]) / float(w))
            gmy = float(vp[3]) - (float(my) * (float(vp[3]) / float(h)))

            mm = np.array(self._gl_model, dtype=float).reshape(4, 4).T
            pm = np.array(self._gl_proj,  dtype=float).reshape(4, 4).T
            inv = np.linalg.inv(pm @ mm)

            def unproject(wx, wy, wz):
                nx = (wx - vp[0]) / vp[2] * 2.0 - 1.0
                ny = (wy - vp[1]) / vp[3] * 2.0 - 1.0
                nz = 2.0 * wz - 1.0
                v  = inv @ np.array([nx, ny, nz, 1.0])
                return v[:3] / v[3] if v[3] != 0 else v[:3]

            near = unproject(gmx, gmy, 0.0)
            far  = unproject(gmx, gmy, 1.0)
            ray_d = far - near
            n = np.linalg.norm(ray_d)
            if n < 1e-10:
                return (None, -1)
            ray_d /= n

            # Tolleranze in unità mondo; scale in base a cam_dist per rimanere
            # cliccabili anche in zoom out.
            tol_node = max(0.02, self.cam_dist * 0.006)
            tol_beam = max(0.015, self.cam_dist * 0.003)

            def ray_point_dist(p):
                w0 = p - near
                t = float(np.dot(w0, ray_d))
                if t < 0: return 1e10, 0.0
                proj = near + t * ray_d
                return float(np.linalg.norm(p - proj)), t

            def ray_segment_dist(q0, q1):
                v = q1 - q0
                w0 = near - q0
                a = 1.0
                b = float(np.dot(ray_d, v))
                c = float(np.dot(v, v))
                dd = float(np.dot(ray_d, w0))
                e = float(np.dot(v, w0))
                D = a * c - b * b
                if D < 1e-8:
                    tc = 0.0; sc = -dd
                else:
                    sc = (b * e - c * dd) / D
                    tc = (a * e - b * dd) / D
                    if tc < 0.0: tc, sc = 0.0, -dd
                    elif tc > 1.0: tc, sc = 1.0, b - dd
                if sc < 0: return 1e10, 0.0
                pr = near + sc * ray_d
                ps = q0 + tc * v
                return float(np.linalg.norm(pr - ps)), sc

            def ray_triangle(v0, v1, v2):
                e1 = v1 - v0
                e2 = v2 - v0
                pvec = np.cross(ray_d, e2)
                det = float(np.dot(e1, pvec))
                if abs(det) < 1e-10: return None
                inv_det = 1.0 / det
                tvec = near - v0
                u = float(np.dot(tvec, pvec)) * inv_det
                if u < 0.0 or u > 1.0: return None
                qvec = np.cross(tvec, e1)
                v = float(np.dot(ray_d, qvec)) * inv_det
                if v < 0.0 or u + v > 1.0: return None
                t = float(np.dot(e2, qvec)) * inv_det
                return t if t >= 0 else None

            nodi = self._dati.get("nodi", {})

            # --- Nodi ---
            if self.show_nodi:
                best_t, best_id = 1e10, -1
                for nid, (x, y, z) in nodi.items():
                    p = np.array([x, y, z], dtype=float)
                    d, t = ray_point_dist(p)
                    if d < tol_node and t < best_t:
                        best_t, best_id = t, nid
                if best_id != -1:
                    return ("nodo", best_id)

            # --- Aste ---
            if self.show_beams:
                best_t, best_id = 1e10, -1
                for bid, asta in self._dati.get("aste", {}).items():
                    ni, nj = asta["nodo_i"], asta["nodo_j"]
                    if ni not in nodi or nj not in nodi: continue
                    q0 = np.array(nodi[ni], dtype=float)
                    q1 = np.array(nodi[nj], dtype=float)
                    d, t = ray_segment_dist(q0, q1)
                    if d < tol_beam and t < best_t:
                        best_t, best_id = t, bid
                if best_id != -1:
                    return ("beam", best_id)

            # --- Shell ---
            if self.show_shells:
                best_t, best_id = 1e10, -1
                for sid, sh in self._dati.get("shell", {}).items():
                    pts = [nodi[n] for n in sh["nodi"] if n in nodi]
                    if len(pts) < 3: continue
                    tris = self._triangoli_shell(pts)
                    for (a, b, c) in tris:
                        t = ray_triangle(np.array(a, dtype=float),
                                         np.array(b, dtype=float),
                                         np.array(c, dtype=float))
                        if t is not None and t < best_t:
                            best_t, best_id = t, sid
                if best_id != -1:
                    return ("shell", best_id)

            return (None, -1)

        except Exception as err:
            print(f"WARN picking struttura: {err}")
            return (None, -1)

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, e):
        self._last_pos = e.pos()
        self._mouse_btn = e.button()

        if e.button() == Qt.LeftButton:
            kind, oid = self._pick_at(e.x(), e.y())
            if kind is not None:
                self._sel_kind = kind
                self._sel_id = oid
                self.oggetto_selezionato.emit(kind, oid)
            else:
                # Click a vuoto → deseleziona
                self._sel_kind = None
                self._sel_id = -1
                self.oggetto_selezionato.emit("", -1)
            self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._sel_kind = None
            self._sel_id = -1
            self.oggetto_selezionato.emit("", -1)
            self.update()
            e.accept()
            return
        super().keyPressEvent(e)

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
        delta = e.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.cam_dist = max(0.5, self.cam_dist * factor)
        self.update()