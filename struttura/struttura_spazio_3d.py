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
from PyQt5.QtCore import Qt, QPoint
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

# Dimensioni rendering
_NODE_SIZE   = 7.0
_BEAM_WIDTH  = 2.5
_ARROW_LEN   = 0.75 
_VINCOLO_SIZE = 0.25


class StrutturaSpazio3D(QOpenGLWidget):
    """Viewer 3D per la visualizzazione di strutture definite via testo."""

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

        # Cache matrici GL per proiezione label
        self._gl_model = None
        self._gl_proj  = None
        self._gl_vp    = None

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

    def reset_view(self):
        self.rot_x = 25.0; self.rot_y = -40.0
        self.pan_x = 0.0; self.pan_y = 0.0
        self.cam_dist = 20.0; self._ortho = False
        self.update()

    # ------------------------------------------------------------------ helper logica carichi

    def _get_max_magnitudes(self):
        """Calcola la magnitudo massima per carichi nodali e distribuiti."""
        max_n = 1e-12
        for (_, fx, fy, fz) in self._dati.get("carichi_nodali", []):
            mag = math.sqrt(fx*fx + fy*fy + fz*fz)
            if mag > max_n: max_n = mag
            
        max_d = 1e-12
        for (_, wx, wy, wz) in self._dati.get("carichi_distribuiti", []):
            mag = math.sqrt(wx*wx + wy*wy + wz*wz)
            if mag > max_d: max_d = mag
            
        return max_n, max_d

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
            self._disegna_shell()
            self._disegna_aste()
            self._disegna_vincoli()
            self._disegna_carichi_distribuiti()
            self._disegna_carichi_nodali()
            self._disegna_nodi()

        glDisable(GL_DEPTH_TEST)
        painter.endNativePainting()

        if self._dati and not self._preview_mode:
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
        painter.setPen(QColor(100, 180, 255, 230))
        for nid, (x, y, z) in nodi.items():
            sx, sy = self._project(x, y, z)
            if sx == -1000: continue
            
            painter.drawText(sx + 8, sy - 8, f"N{nid}")

        # ── Aste: B<id> al punto medio ──
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
        painter.setPen(QColor(100, 230, 130, 240)) # Verde
        for nid in vincoli.keys():
            if nid not in nodi: continue
            x, y, z = nodi[nid]
            sx, sy = self._project(x, y, z - _VINCOLO_SIZE * 0.8)
            if sx != -1000:
                painter.drawText(sx + 10, sy + 14, f"V{nid}")

        # Otteniamo i massimi per calcolare lo scaling
        max_n, max_d = self._get_max_magnitudes()

        # ── Carichi nodali: F<id> posizionato all'origine della freccia (coda) ──
        painter.setPen(QColor(255, 120, 120, 240))
        for idx, (nid, fx, fy, fz) in enumerate(self._dati.get("carichi_nodali", []), 1):
            if nid not in nodi: continue
            x, y, z = nodi[nid]
            
            mag = math.sqrt(fx*fx + fy*fy + fz*fz)
            if mag < 1e-12: continue
            
            scale = self._calc_load_scale(mag, max_n)
            
            # (x + fx*scale) = coda della freccia
            sx, sy = self._project(x + fx * scale, y + fy * scale, z + fz * scale)
            if sx != -1000:
                painter.drawText(sx + 6, sy - 6, f"F{idx}")

        # ── Carichi distribuiti: Q<id> posizionato all'origine della freccia centrale ──
        painter.setPen(QColor(255, 190, 80, 240))
        for idx, (bid, wx, wy, wz) in enumerate(self._dati.get("carichi_distribuiti", []), 1):
            if bid not in aste: continue
            asta = aste[bid]
            ni, nj = asta["nodo_i"], asta["nodo_j"]
            if ni not in nodi or nj not in nodi: continue
            
            pi, pj = nodi[ni], nodi[nj]
            # Troviamo il punto medio dell'asta per posizionare l'etichetta
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

        max_n, _ = self._get_max_magnitudes()

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

        _, max_d = self._get_max_magnitudes()

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
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x + dx, y + dy, z + dz) # coda
        glVertex3f(x, y, z)                # punta al nodo
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

        hx = x + dx * 0.7
        hy = y + dy * 0.7
        hz = z + dz * 0.7
        glBegin(GL_LINES)
        glVertex3f(x, y, z)
        glVertex3f(hx + px * head, hy + py * head, hz + pz * head)
        glVertex3f(x, y, z)
        glVertex3f(hx - px * head, hy - py * head, hz - pz * head)
        glEnd()
        glLineWidth(1.0)

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, e):
        self._last_pos = e.pos()
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
        delta = e.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.cam_dist = max(0.5, self.cam_dist * factor)
        self.update()