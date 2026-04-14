"""
disegno_pressoflessione.py  –  v3 (Super Ottimizzata con Vertex Arrays)
=======================================================================
Widget OpenGL 2D per la visualizzazione dei risultati dell'analisi
a pressoflessione.

Layout a tre pannelli condivisi sulla stessa vista ortografica:

  [ Sezione discretizzata ] | [ Diagramma ε ] | [ Diagramma σ ]

Modalità colore:
  - NORMALE:   cls grigio chiaro/scuro, barre rosso/blu per trazione/compressione
  - GRADIENTE: colormap Jet centrata su σ=0 (blu=compressione, rosso=trazione)
               con scala cromatica sul lato destro.

Ottimizzazioni:
  - Rendering della sezione tramite NumPy Array e Pointers OpenGL 
    (glDrawArrays) per eliminare il bottleneck della CPU.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore    import Qt, QPoint
from PyQt5.QtGui     import (QColor, QFont, QLinearGradient,
                              QPainter, QPen, QBrush)
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_TEST,
    GL_LINE_LOOP, GL_LINE_SMOOTH, GL_LINES, GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA, GL_POINT_SMOOTH, GL_PROJECTION,
    GL_QUADS, GL_SRC_ALPHA, GL_TEXTURE_2D, GL_TRIANGLE_FAN,
    glBegin, glBlendFunc, glClear, glClearColor, glColor3f, glColor4f,
    glDisable, glEnable, glEnd, glLineWidth, glLoadIdentity,
    glMatrixMode, glOrtho, glVertex2f, glViewport,
    # --- NUOVI IMPORT PER LE PRESTAZIONI ---
    glEnableClientState, glDisableClientState, glVertexPointer, glColorPointer,
    glDrawArrays, GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_FLOAT, GL_TRIANGLES
)


# ==============================================================================
# COLORMAP JET  (Blue → Cyan → Green → Yellow → Red)
# ==============================================================================
_JET_STOPS: List[Tuple[float, Tuple]] = [
    (0.00, (0.0,  0.0,  1.0)),
    (0.25, (0.0,  1.0,  1.0)),
    (0.50, (0.0,  1.0,  0.0)),
    (0.75, (1.0,  1.0,  0.0)),
    (1.00, (1.0,  0.0,  0.0)),
]


def _jet(t: float, alpha: float = 0.88) -> Tuple[float, float, float, float]:
    """Interpolazione lineare sulla colormap Jet, t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    for i in range(len(_JET_STOPS) - 1):
        t0, c0 = _JET_STOPS[i]
        t1, c1 = _JET_STOPS[i + 1]
        if t <= t1 + 1e-9:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (c0[0] + f * (c1[0] - c0[0]),
                    c0[1] + f * (c1[1] - c0[1]),
                    c0[2] + f * (c1[2] - c0[2]),
                    alpha)
    return (*_JET_STOPS[-1][1], alpha)


# ==============================================================================
# COSTANTI STILE
# ==============================================================================
_BG       = (40 / 255, 40 / 255, 40 / 255, 1.0)
_GRD      = (0.18, 0.18, 0.18)
_SEP      = (0.38, 0.38, 0.38)
_NA_CLR   = (1.00, 0.88, 0.08, 0.95)     # giallo-oro: asse neutro

# Sezione – modalità NORMALE
_CLS_REACT  = (0.78, 0.78, 0.78, 0.90)   # calcestruzzo reagisce
_CLS_INERT  = (0.26, 0.26, 0.26, 0.75)   # cls fessurato / non reagisce
_BAR_TRAZ   = (1.00, 0.18, 0.08, 0.92)   # barra in trazione
_BAR_COMP   = (0.15, 0.42, 1.00, 0.92)   # barra in compressione
_BAR_ZERO   = (0.65, 0.65, 0.65, 0.85)   # barra quasi scarica

# Diagrammi – modalità NORMALE
_COMP_FILL  = (0.10, 0.35, 0.90, 0.30)
_COMP_LINE  = (0.20, 0.55, 1.00, 0.95)
_TRAZ_FILL  = (0.90, 0.18, 0.06, 0.25)
_TRAZ_LINE  = (1.00, 0.28, 0.08, 0.95)

_FF  = "Arial"
_FS  = 8
_FM  = 9


# ==============================================================================
# WIDGET
# ==============================================================================

class PressoflessioneWidget(QOpenGLWidget):
    """
    Visualizzatore 2D a tre pannelli per l'analisi a pressoflessione.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._results: Optional[dict] = None
        self._mode    = 'normale'   # 'normale' | 'gradiente'
        self._tipo    = 'SLU'
        self._theta_deg = 0.0

        # Dati pre-elaborati per il rendering
        self._fc: list = []   # fibre calcestruzzo: (xr, yr, gs, eps, sig, fessurata)
        self._fa: list = []   # fibre acciaio/barre: (xr, yr, r, eps, sig)
        self._gs       = 10.0  # grid_step

        # Buffer Array per la GPU (Prestazioni)
        self._vbo_cls_verts = np.array([], dtype=np.float32)
        self._vbo_cls_colors = np.array([], dtype=np.float32)
        self._n_cls_verts = 0
        
        self._vbo_acc_verts = np.array([], dtype=np.float32)
        self._vbo_acc_colors = np.array([], dtype=np.float32)
        self._n_acc_verts = 0

        # Profili
        self._sp: list = []   # profilo deformazioni [(d, eps), ...]
        self._tp: list = []   # profilo tensioni cls  [(d, sig_avg), ...]
        self._bs: list = []   # tensioni barre        [(yr, sig, xr), ...]

        # Range sigma per colormap
        self._sig_mabs = 1.0

        # Bounds sezione (coordinate ruotate)
        self._sx0 = self._sx1 = -100.0
        self._sy0 = self._sy1 =  100.0

        # Layout pannelli
        self._xsc = 200.0    # centro x pannello deformazioni
        self._xtc = 400.0    # centro x pannello tensioni
        self._dhw = 80.0     # semi-larghezza pannelli diagramma
        self._ssc = 1.0      # scala strain
        self._tsc = 1.0      # scala stress

        # Asse neutro (coordinate ruotate)
        self._d_na = 0.0

        # Vista
        self.pan_x  = 0.0
        self.pan_y  = 0.0
        self.zoom   = 200.0
        self._last_pos = None
        self._cursor   = QPoint(0, 0)
        self.wx_min = self.wx_max = 0.0
        self.wy_min = self.wy_max = 0.0

        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # API PUBBLICA
    # ------------------------------------------------------------------

    def set_display_mode(self, mode: str) -> None:
        """'normale' o 'gradiente'."""
        self._mode = mode
        if self._results:
            self._build_vertex_arrays() # Ricalcola i colori della GPU!
        self.update()

    def set_results(self, results: Optional[dict]) -> None:
        """Aggiorna i risultati visualizzati e ridisegna."""
        if results is None:
            self._results = None
            self.update()
            return

        self._results = results
        self._elabora(results)
        self._auto_fit()
        self.update()

    def reset_view(self) -> None:
        """Reimposta la vista al centro automatico."""
        if self._results:
            self._auto_fit()
        else:
            self.pan_x = self.pan_y = 0.0
            self.zoom  = 200.0
        self.update()

    # ------------------------------------------------------------------
    # PRE-ELABORAZIONE DATI E MOTORE VBO
    # ------------------------------------------------------------------

    def _elabora(self, res: dict) -> None:
        self._gs    = res.get('grid_step', 10.0)
        nv          = res.get('n_vect', (0.0, 1.0))
        nx, ny      = float(nv[0]), float(nv[1])
        gs          = self._gs
        area_gs     = gs * gs
        self._tipo  = res.get('tipo', 'SLU')
        self._theta_deg = res.get('theta_deg', 0.0)

        self._d_na  = res.get('d_na', 0.0)

        # ── Ruota le coordinate ──────────────────────────────────────
        self._fc.clear()
        self._fa.clear()
        all_sig: list = []

        for x, y, area, eps, sig in res.get('fibre', []):
            all_sig.append(sig)
            xr = ny * x - nx * y
            yr = nx * x + ny * y

            # Discrimina barre (area ≠ gs²) da celle carpenteria
            if abs(area - area_gs) > area_gs * 0.30:
                r = math.sqrt(area / math.pi)
                self._fa.append((xr, yr, r, eps, sig))
            else:
                fessurata = (self._tipo == 'SLE') and (eps > 1e-8) and (sig == 0.0)
                self._fc.append((xr, yr, gs, eps, sig, fessurata))

        # Range sigma per colormap simmetrica centrata a 0
        if all_sig:
            smin = min(all_sig);  smax = max(all_sig)
            self._sig_mabs = max(abs(smin), abs(smax), 1e-3)
        else:
            self._sig_mabs = 1.0

        # ── Bounds sezione in coord. ruotate ─────────────────────────
        all_r = [(xr, yr) for xr, yr, *_ in self._fc]
        all_r += [(xr, yr) for xr, yr, *_ in self._fa]
        if all_r:
            xs = [c[0] for c in all_r];  ys = [c[1] for c in all_r]
            mg = gs * 0.6
            self._sx0, self._sx1 = min(xs) - mg, max(xs) + mg
            self._sy0, self._sy1 = min(ys) - mg, max(ys) + mg
        else:
            self._sx0 = self._sx1 = -100.0
            self._sy0 = self._sy1 =  100.0

        # ── Profilo deformazioni (2 punti, lineare) ───────────────────
        self._sp = [
            (res.get('d_min', self._sy0), res.get('eps_bot', 0.0)),
            (res.get('d_max', self._sy1), res.get('eps_top', 0.0)),
        ]

        # ── Profilo tensioni cls (media per banda y) ─────────────────
        ymap: dict = defaultdict(list)
        for xr, yr, _g, eps, sig, _f in self._fc:
            key = round(yr / gs) * gs
            ymap[key].append(sig)
        self._tp = sorted(
            [(yk, float(np.mean(vs))) for yk, vs in ymap.items()],
            key=lambda t: t[0]
        )

        # ── Tensioni barre ────────────────────────────────────────────
        self._bs = [(yr, sig, xr) for xr, yr, _r, _e, sig in self._fa]

        # ── Layout pannelli ──────────────────────────────────────────
        self._recalc_layout()
        
        # COSTRUISCI I VERTEX ARRAYS PER LA GPU (IL MOTORE VELOCE!)
        self._build_vertex_arrays()

    def _build_vertex_arrays(self) -> None:
        """Pre-calcola gli array NumPy per la GPU una volta sola."""
        # 1. ARRAY PER IL CALCESTRUZZO (GL_QUADS)
        cls_verts = []
        cls_colors = []
        h = self._gs * 0.90
        half = h / 2.0
        
        for xr, yr, _g, eps, sig, fessurata in self._fc:
            col = (self._jet_color(sig) if self._mode == 'gradiente' 
                   else self._color_fibra_conc(sig, fessurata))
            
            # Aggiungiamo i 4 vertici del quad
            cls_verts.extend([
                xr - half, yr - half,
                xr + half, yr - half,
                xr + half, yr + half,
                xr - half, yr + half
            ])
            # Aggiungiamo 4 volte lo stesso colore (uno per vertice)
            cls_colors.extend(col * 4)

        self._vbo_cls_verts = np.array(cls_verts, dtype=np.float32)
        self._vbo_cls_colors = np.array(cls_colors, dtype=np.float32)
        self._n_cls_verts = len(cls_verts) // 2

        # 2. ARRAY PER LE BARRE DI ACCIAIO (GL_TRIANGLES)
        acc_verts = []
        acc_colors = []
        N = 16  # 16 segmenti bastano per un cerchio fluido
        
        for xr, yr, r, eps, sig in self._fa:
            col = (self._jet_color(sig) if self._mode == 'gradiente' 
                   else self._color_fibra_acc(sig))
            
            for k in range(N):
                t1 = 2 * math.pi * k / N
                t2 = 2 * math.pi * (k + 1) / N
                
                acc_verts.extend([
                    xr, yr,  # Centro
                    xr + r * math.cos(t1), yr + r * math.sin(t1),
                    xr + r * math.cos(t2), yr + r * math.sin(t2)
                ])
                acc_colors.extend(col * 3)
                
        self._vbo_acc_verts = np.array(acc_verts, dtype=np.float32)
        self._vbo_acc_colors = np.array(acc_colors, dtype=np.float32)
        self._n_acc_verts = len(acc_verts) // 2

    def _recalc_layout(self) -> None:
        sw        = max(self._sx1 - self._sx0, 10.0)
        self._dhw = sw * 0.62
        gap       = sw * 0.28
        self._xsc = self._sx1 + gap + self._dhw
        self._xtc = self._xsc + 2 * self._dhw + gap

        eps_max = max(abs(self._sp[0][1]), abs(self._sp[-1][1]), 1e-9) if self._sp else 1e-9
        self._ssc = self._dhw * 0.87 / eps_max
        self._tsc = self._dhw * 0.87 / max(self._sig_mabs, 1e-6)

    def _auto_fit(self) -> None:
        """Centra e fa il fit dell'intera vista con tutti e tre i pannelli."""
        x_end  = self._xtc + self._dhw * 1.1
        x_beg  = self._sx0 - self._dhw * 0.1
        cx     = (x_beg + x_end) / 2.0
        cy     = (self._sy0 + self._sy1) / 2.0
        self.pan_x = cx
        self.pan_y = cy
        self.zoom  = max(x_end - x_beg, self._sy1 - self._sy0) / 2.0 * 1.2

    # ------------------------------------------------------------------
    # OPENGL – SETUP
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        glClearColor(*_BG)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glClear(GL_COLOR_BUFFER_BIT)

        asp = self.width() / self.height() if self.height() > 0 else 1.0
        hw  = self.zoom * asp;  hh = self.zoom

        self.wx_min = self.pan_x - hw;  self.wx_max = self.pan_x + hw
        self.wy_min = self.pan_y - hh;  self.wy_max = self.pan_y + hh

        glMatrixMode(GL_PROJECTION);  glLoadIdentity()
        glOrtho(self.wx_min, self.wx_max, self.wy_min, self.wy_max, -1, 1)
        glMatrixMode(GL_MODELVIEW);   glLoadIdentity()

        self._gl_grid()

        if self._results is None:
            self._gl_overlay_empty()
            return

        self._gl_separatori()
        self._gl_section()
        self._gl_axis_neutro()
        self._gl_strain_diagram()
        self._gl_stress_diagram()
        self._gl_overlay()

    # ------------------------------------------------------------------
    # OPENGL – GRIGLIA
    # ------------------------------------------------------------------

    def _gl_grid(self) -> None:
        dx = self.wx_max - self.wx_min;  dy = self.wy_max - self.wy_min
        tx = self._tick(dx);             ty = self._tick(dy)
        glColor3f(*_GRD)
        glLineWidth(0.6)
        glBegin(GL_LINES)
        x = math.floor(self.wx_min / tx) * tx
        while x <= self.wx_max + tx:
            glVertex2f(x, self.wy_min);  glVertex2f(x, self.wy_max);  x += tx
        y = math.floor(self.wy_min / ty) * ty
        while y <= self.wy_max + ty:
            glVertex2f(self.wx_min, y);  glVertex2f(self.wx_max, y);  y += ty
        glEnd()

    # ------------------------------------------------------------------
    # OPENGL – SEPARATORI TRA PANNELLI
    # ------------------------------------------------------------------

    def _gl_separatori(self) -> None:
        glColor3f(*_SEP)
        glLineWidth(0.8)
        xs_sep = [self._sx1 + (self._xsc - self._dhw - self._sx1) * 0.5,
                  self._xsc + self._dhw + (self._xtc - self._dhw - self._xsc - self._dhw) * 0.5]
        glBegin(GL_LINES)
        for xs in xs_sep:
            glVertex2f(xs, self.wy_min)
            glVertex2f(xs, self.wy_max)
        glEnd()

    # ------------------------------------------------------------------
    # OPENGL – SEZIONE DISCRETIZZATA (OTTIMIZZATA GPU)
    # ------------------------------------------------------------------

    def _jet_color(self, sig: float, alpha: float = 0.88) -> tuple:
        ma = self._sig_mabs or 1.0
        t  = max(0.0, min(1.0, (sig + ma) / (2.0 * ma)))
        return _jet(t, alpha)

    def _color_fibra_conc(self, sig: float, fessurata: bool) -> tuple:
        if fessurata or abs(sig) < 0.3:
            return _CLS_INERT
        return _CLS_REACT

    def _color_fibra_acc(self, sig: float) -> tuple:
        if sig > 1.0:
            return _BAR_TRAZ
        if sig < -1.0:
            return _BAR_COMP
        return _BAR_ZERO

    def _gl_section(self) -> None:
        """Disegna la sezione inviando i vertex array direttamente alla GPU."""
        if self._n_cls_verts == 0 and self._n_acc_verts == 0:
            return

        # Abilita la lettura diretta degli array
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # --- DISEGNA IL CALCESTRUZZO IN UN SOLO COMANDO ---
        if self._n_cls_verts > 0:
            glVertexPointer(2, GL_FLOAT, 0, self._vbo_cls_verts)
            glColorPointer(4, GL_FLOAT, 0, self._vbo_cls_colors)
            glDrawArrays(GL_QUADS, 0, self._n_cls_verts)

        # --- DISEGNA L'ACCIAIO IN UN SOLO COMANDO ---
        if self._n_acc_verts > 0:
            glVertexPointer(2, GL_FLOAT, 0, self._vbo_acc_verts)
            glColorPointer(4, GL_FLOAT, 0, self._vbo_acc_colors)
            glDrawArrays(GL_TRIANGLES, 0, self._n_acc_verts)

        # Disabilita gli array per non interferire col resto del disegno
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        # --- CONTORNI BARRE ACCIAIO (Pochi elementi, va bene Immediate Mode) ---
        if self._fa:
            glLineWidth(1.2)
            for xr, yr, r, eps, sig in self._fa:
                col_f = (self._jet_color(sig) if self._mode == 'gradiente' 
                         else self._color_fibra_acc(sig))
                col_o = (col_f[0] * 0.62, col_f[1] * 0.62, col_f[2] * 0.62, 1.0)
                glColor4f(*col_o)
                glBegin(GL_LINE_LOOP)
                N = 16
                for k in range(N):
                    t = 2 * math.pi * k / N
                    glVertex2f(xr + r * math.cos(t), yr + r * math.sin(t))
                glEnd()

    # ------------------------------------------------------------------
    # OPENGL – ASSE NEUTRO
    # ------------------------------------------------------------------

    def _gl_axis_neutro(self) -> None:
        d   = self._d_na
        x0  = self._sx0 - self._dhw * 0.05
        x1  = self._xtc + self._dhw
        L   = x1 - x0
        ns  = max(int(L / 10), 4)
        glColor4f(*_NA_CLR); glLineWidth(1.8)
        glBegin(GL_LINES)
        for k in range(ns):
            ta = k / ns; tb = min((k + 0.55) / ns, 1.0)
            glVertex2f(x0 + ta * L, d); glVertex2f(x0 + tb * L, d)
        glEnd()

    # ------------------------------------------------------------------
    # OPENGL – DIAGRAMMA DEFORMAZIONI
    # ------------------------------------------------------------------

    def _gl_strain_diagram(self) -> None:
        if len(self._sp) < 2:
            return

        xc  = self._xsc
        sc  = self._ssc
        d0, e0 = self._sp[0]
        d1, e1 = self._sp[-1]

        glColor4f(0.55, 0.55, 0.55, 0.55); glLineWidth(0.9)
        glBegin(GL_LINES)
        glVertex2f(xc, d0 - 4); glVertex2f(xc, d1 + 4)
        glEnd()

        hw_t = self._dhw * 0.15
        glColor4f(0.45, 0.45, 0.45, 0.5); glLineWidth(0.8)
        glBegin(GL_LINES)
        glVertex2f(xc - hw_t, d0); glVertex2f(xc + hw_t, d0)
        glVertex2f(xc - hw_t, d1); glVertex2f(xc + hw_t, d1)
        glEnd()

        dz = None
        if e0 * e1 < 0 and abs(e1 - e0) > 1e-15:
            dz = d0 + (-e0) / (e1 - e0) * (d1 - d0)

        segs = []
        if dz is None:
            segs = [(d0, e0, d1, e1)]
        else:
            if abs(e0) > 1e-12: segs.append((d0, e0, dz, 0.0))
            if abs(e1) > 1e-12: segs.append((dz, 0.0, d1, e1))

        for (da, va, db, vb) in segs:
            if abs(va) < 1e-12 and abs(vb) < 1e-12:
                continue
            mid_v = (va + vb) / 2.0
            fill  = _COMP_FILL if mid_v <= 0 else _TRAZ_FILL
            line  = _COMP_LINE if mid_v <= 0 else _TRAZ_LINE

            glColor4f(*fill)
            glBegin(GL_QUADS)
            glVertex2f(xc,          da)
            glVertex2f(xc + va * sc, da)
            glVertex2f(xc + vb * sc, db)
            glVertex2f(xc,          db)
            glEnd()

            glColor4f(*line); glLineWidth(1.9)
            glBegin(GL_LINES)
            glVertex2f(xc,          da); glVertex2f(xc + va * sc, da)
            glVertex2f(xc + va * sc, da); glVertex2f(xc + vb * sc, db)
            glVertex2f(xc + vb * sc, db); glVertex2f(xc,          db)
            glEnd()

    # ------------------------------------------------------------------
    # OPENGL – DIAGRAMMA TENSIONI
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_zero_crossings(
            profile: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for i, (d, s) in enumerate(profile):
            if i > 0:
                dp, sp = profile[i - 1]
                if sp * s < 0 and abs(s - sp) > 1e-15:
                    dz = dp + (-sp) / (s - sp) * (d - dp)
                    out.append((dz, 0.0))
            out.append((d, s))
        return out

    def _gl_stress_diagram(self) -> None:
        xc   = self._xtc
        sc   = self._tsc
        y0   = self._sy0
        y1   = self._sy1
        mode = self._mode

        glColor4f(0.55, 0.55, 0.55, 0.55); glLineWidth(0.9)
        glBegin(GL_LINES)
        glVertex2f(xc, y0 - 4); glVertex2f(xc, y1 + 4)
        glEnd()

        hw_t = self._dhw * 0.15
        glColor4f(0.45, 0.45, 0.45, 0.5); glLineWidth(0.8)
        glBegin(GL_LINES)
        glVertex2f(xc - hw_t, y0); glVertex2f(xc + hw_t, y0)
        glVertex2f(xc - hw_t, y1); glVertex2f(xc + hw_t, y1)
        glEnd()

        if len(self._tp) >= 2:
            prof = self._insert_zero_crossings(self._tp)
            n    = len(prof)

            glBegin(GL_QUADS)
            for i in range(n - 1):
                d0, s0 = prof[i]
                d1, s1 = prof[i + 1]
                mid = (s0 + s1) / 2.0
                if abs(mid) < 0.05:
                    continue

                if mode == 'gradiente':
                    col_f = self._jet_color(mid, alpha=0.42)
                else:
                    col_f = _COMP_FILL if mid < 0 else _TRAZ_FILL

                glColor4f(*col_f)
                glVertex2f(xc,          d0)
                glVertex2f(xc + s0 * sc, d0)
                glVertex2f(xc + s1 * sc, d1)
                glVertex2f(xc,          d1)
            glEnd()

            glLineWidth(1.9)
            glBegin(GL_LINES)
            for i in range(n - 1):
                d0, s0 = prof[i]
                d1, s1 = prof[i + 1]
                mid = (s0 + s1) / 2.0
                if abs(mid) < 0.05:
                    continue

                if mode == 'gradiente':
                    col_l = self._jet_color(mid, alpha=0.96)
                else:
                    col_l = _COMP_LINE if mid < 0 else _TRAZ_LINE

                glColor4f(*col_l)
                glVertex2f(xc + s0 * sc, d0)
                glVertex2f(xc + s1 * sc, d1)
            glEnd()

            d_bot, s_bot = prof[0]
            d_top, s_top = prof[-1]
            glLineWidth(1.9)
            glBegin(GL_LINES)
            if abs(s_bot) > 0.05:
                col = (self._jet_color(s_bot, 0.96) if mode == 'gradiente'
                       else (_COMP_LINE if s_bot < 0 else _TRAZ_LINE))
                glColor4f(*col)
                glVertex2f(xc,           d_bot)
                glVertex2f(xc + s_bot * sc, d_bot)
            if abs(s_top) > 0.05:
                col = (self._jet_color(s_top, 0.96) if mode == 'gradiente'
                       else (_COMP_LINE if s_top < 0 else _TRAZ_LINE))
                glColor4f(*col)
                glVertex2f(xc + s_top * sc, d_top)
                glVertex2f(xc,              d_top)
            glEnd()

        for (yr, sig, _xr) in self._bs:
            if mode == 'gradiente':
                col = self._jet_color(sig, 0.96)
            else:
                col = _BAR_TRAZ if sig > 1.0 else (_BAR_COMP if sig < -1.0 else _BAR_ZERO)

            x_sig = xc + sig * sc

            glColor4f(col[0], col[1], col[2], 0.65); glLineWidth(1.5)
            glBegin(GL_LINES)
            glVertex2f(xc, yr); glVertex2f(x_sig, yr)
            glEnd()

            r = max(self._gs * 0.18, 2.5)
            glColor4f(*col)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(x_sig, yr)
            for k in range(17):
                t = 2 * math.pi * k / 16
                glVertex2f(x_sig + r * math.cos(t), yr + r * math.sin(t))
            glEnd()

    # ------------------------------------------------------------------
    # OVERLAY TESTO (QPainter)
    # ------------------------------------------------------------------

    def _gl_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        def ws(wx, wy):
            nx = (wx - self.wx_min) / max(self.wx_max - self.wx_min, 1e-6)
            ny = (wy - self.wy_min) / max(self.wy_max - self.wy_min, 1e-6)
            return int(nx * W), int((1 - ny) * H)

        def lbl(wx, wy, text, col=QColor(200, 200, 200),
                bold=False, size=_FS, anc='left'):
            sx, sy = ws(wx, wy)
            f = QFont(_FF, size); f.setBold(bold)
            painter.setFont(f)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(text)
            if   anc == 'right':  sx -= tw
            elif anc == 'center': sx -= tw // 2
            if not (-80 <= sx <= W + 80 and -25 <= sy <= H + 25):
                return
            painter.setPen(QPen(col))
            painter.drawText(sx, sy, text)

        th_s  = f"  [θ={self._theta_deg:.1f}°]" if abs(self._theta_deg) > 0.5 else ""
        y_tit = self._sy1 + (self.wy_max - self._sy1) * 0.38
        cx_s  = (self._sx0 + self._sx1) / 2.0
        lbl(cx_s, y_tit, f"Sezione{th_s}  [{self._tipo}]",
            QColor(210, 210, 210), bold=True, size=_FM, anc='center')
        lbl(self._xsc, y_tit, "Deformazioni  ε",
            QColor(110, 165, 255), bold=True, size=_FM, anc='center')
        lbl(self._xtc, y_tit, "Tensioni  σ  [MPa]",
            QColor(255, 155, 65), bold=True, size=_FM, anc='center')

        lbl(self._sx0 + 3, self._d_na + 3, "A.N.", QColor(255, 220, 50), bold=True)

        if len(self._sp) == 2:
            d0, e0 = self._sp[0];  d1, e1 = self._sp[-1]
            xcs = self._xsc;  scs = self._ssc;  off = 4
            if e0 * e1 < 0 and abs(e1 - e0) > 1e-15:
                dz = d0 + (-e0) / (e1 - e0) * (d1 - d0)
                lbl(xcs + 3, dz, "ε=0", QColor(255, 220, 50))
            for d_v, e_v in [(d0, e0), (d1, e1)]:
                if abs(e_v) < 1e-12:
                    continue
                a   = 'left' if e_v >= 0 else 'right'
                col = QColor(140, 190, 255) if e_v < 0 else QColor(255, 155, 90)
                lbl(xcs + e_v * scs + (off if e_v >= 0 else -off), d_v,
                    f"{e_v * 1000:.2f}‰", col, anc=a)

        if self._tp:
            xct = self._xtc;  sct = self._tsc;  off = 4
            reported: set = set()
            cands = [self._tp[0], self._tp[-1]]
            mc = min(self._tp, key=lambda t: t[1])
            mt = max(self._tp, key=lambda t: t[1])
            if mc not in cands: cands.append(mc)
            if mt not in cands: cands.append(mt)
            for d_v, s_v in cands:
                if abs(s_v) < 0.3:
                    continue
                k = round(d_v / max(self._gs, 1))
                if k in reported:
                    continue
                reported.add(k)
                a   = 'left' if s_v >= 0 else 'right'
                col = QColor(140, 190, 255) if s_v < 0 else QColor(255, 155, 90)
                lbl(xct + s_v * sct + (off if s_v >= 0 else -off), d_v,
                    f"{s_v:.1f}", col, anc=a)

        xct = self._xtc;  sct = self._tsc;  off = 4
        seen_y: set = set()
        for yr, sig, _ in self._bs:
            k = round(yr / max(self._gs, 1))
            if k in seen_y:
                continue
            seen_y.add(k)
            if abs(sig) < 0.1:
                continue
            a   = 'left' if sig >= 0 else 'right'
            col = QColor(255, 100, 80) if sig > 0 else QColor(90, 160, 255)
            lbl(xct + sig * sct + (off if sig >= 0 else -off), yr,
                f"{sig:.1f}", col, anc=a)

        if self._mode == 'gradiente':
            self._draw_legend(painter, W, H)

        self._ov_box(painter, W, H)

        mx, my = self._cursor.x(), self._cursor.y()
        wxc, wyc = self._s2w(mx, my)
        painter.setFont(QFont(_FF, _FS))
        painter.setPen(QPen(QColor(200, 200, 200, 155)))
        painter.drawLine(mx - 7, my, mx + 7, my)
        painter.drawLine(mx, my - 7, mx, my + 7)
        trk = f"X: {wxc:.1f}   Y: {wyc:.1f}"
        tw  = painter.fontMetrics().horizontalAdvance(trk)
        painter.drawText(W - tw - 10, H - 9, trk)

        painter.end()

    def _gl_overlay_empty(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont(_FF, 11))
        painter.setPen(QPen(QColor(110, 110, 110)))
        msg = "Seleziona una sezione e avvia l'analisi"
        fm  = painter.fontMetrics()
        painter.drawText(
            self.width() // 2 - fm.horizontalAdvance(msg) // 2,
            self.height() // 2,
            msg
        )
        painter.end()

    # ------------------------------------------------------------------
    # LEGENDA COLORMAP
    # ------------------------------------------------------------------

    def _draw_legend(self, painter: QPainter, W: int, H: int) -> None:
        bar_w  = 22
        bar_h  = min(442, H - 60)
        mrg_r  = 86
        x_bar  = W - mrg_r - bar_w
        y_bar  = (H - bar_h) // 2

        bg_x = x_bar - 10
        bg_y = y_bar - 52
        bg_w = bar_w + mrg_r - 12
        bg_h = bar_h + 100
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 18, 18, 185))
        painter.drawRoundedRect(bg_x, bg_y, bg_w, bg_h, 5, 5)

        f_t = QFont(_FF, _FS); f_t.setBold(True)
        painter.setFont(f_t)
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(bg_x + 4, bg_y + 18, "σ [MPa]")

        grad = QLinearGradient(x_bar, y_bar + bar_h, x_bar, y_bar)
        grad.setColorAt(0.00, QColor(0,   0,   255))
        grad.setColorAt(0.25, QColor(0,   255, 255))
        grad.setColorAt(0.50, QColor(0,   255, 0))
        grad.setColorAt(0.75, QColor(255, 255, 0))
        grad.setColorAt(1.00, QColor(255, 0,   0))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawRect(x_bar, y_bar, bar_w, bar_h)

        ma    = self._sig_mabs
        ticks = [-ma, -ma / 2, 0.0, ma / 2, ma]
        f_tk  = QFont(_FF, _FS - 1)
        painter.setFont(f_tk)
        for val in ticks:
            t_n    = (val + ma) / (2.0 * ma) 
            y_tick = int(y_bar + bar_h - t_n * bar_h)
            painter.setPen(QPen(QColor(155, 155, 155), 1))
            painter.drawLine(x_bar + bar_w, y_tick, x_bar + bar_w + 5, y_tick)
            txt = "0" if val == 0 else (f"{val:.1e}" if abs(val) > 999 else f"{val:.1f}")
            painter.setPen(QPen(QColor(210, 210, 210)))
            painter.drawText(x_bar + bar_w + 7, y_tick + 4, txt)

        painter.setPen(QColor(165, 165, 165))
        painter.drawText(bg_x + 3, y_bar + bar_h + 24, f"Comp: {-ma:.1f}")
        painter.drawText(bg_x + 3, y_bar - 12,          f"Traz: {ma:.1f}")

    # ------------------------------------------------------------------
    # BOX VERIFICA
    # ------------------------------------------------------------------

    def _ov_box(self, p: QPainter, W: int, H: int) -> None:
        res = self._results
        if res is None:
            return
        tipo = res.get('tipo', '')
        ok   = res.get('verificata', False)

        if ok:
            stato = "✓  Verifica SODDISFATTA"
            bcol  = QColor(0,  210, 80);   tcol = QColor(0,  235, 100)
        else:
            stato = "✗  Verifica NON SODDISFATTA"
            bcol  = QColor(220, 48, 48);   tcol = QColor(245, 70, 70)

        det = [f"Analisi {tipo}"]
        if tipo == 'SLU':
            N   = res.get('N_Ed', 0.0)
            M   = abs(res.get('M_Ed', 0.0))
            Mr  = res.get('M_Rd', 0.0)
            rr  = res.get('rapporto_MEd_MRd', res.get('rapporto', 0.0))
            det += [f"N_Ed = {N:.1f} kN",
                    f"M_Ed = {M:.1f} kNm     M_Rd = {Mr:.1f} kNm",
                    f"M_Ed / M_Rd = {rr:.3f}"]
            if res.get('fuori_dominio', False):
                det.append("⚠  Fuori dal dominio resistente")
        elif tipo == 'SLE':
            sc_v = res.get('sigma_c_compr_max', res.get('sigma_c_max', 0.0))
            ss_v = res.get('sigma_s_traz_max', 0.0)
            lc   = res.get('sigma_c_limit', res.get('lim_cls'))
            ls   = res.get('sigma_s_limit', res.get('lim_acc'))
            det += [f"σ_c = {sc_v:.1f} MPa" + (f"  (lim {lc:.1f})" if lc else ""),
                    f"σ_s = {ss_v:.1f} MPa" + (f"  (lim {ls:.1f})" if ls else "")]
            for n in res.get('note', []):
                det.append(f"  {n}")

        f_sm = QFont(_FF, _FS - 1)
        f_ti = QFont(_FF, _FS); f_ti.setBold(True)
        p.setFont(f_sm)
        fm   = p.fontMetrics()
        lh   = fm.height() + 2
        bw   = max(fm.horizontalAdvance(t) for t in det) + 20
        bh   = lh * (len(det) + 1) + 12

        bx = 10;  by = H - bh - 10
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 15, 15, 190))
        p.drawRoundedRect(bx, by, bw, bh, 4, 4)

        p.setFont(f_ti)
        p.setPen(QPen(tcol))
        p.drawText(bx + 8, by + lh + 4, stato)

        p.setFont(f_sm)
        p.setPen(QPen(QColor(185, 185, 185)))
        for i, t in enumerate(det):
            p.drawText(bx + 8, by + lh * (i + 2) + 6, t)

    # ------------------------------------------------------------------
    # MOUSE
    # ------------------------------------------------------------------

    def mousePressEvent(self, e) -> None:
        if e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._last_pos = e.pos()
        self._cursor = e.pos()
        self.update()

    def mouseMoveEvent(self, e) -> None:
        self._cursor = e.pos()
        if self._last_pos is not None and (
                e.buttons() & (Qt.LeftButton | Qt.MiddleButton)):
            dx = e.x() - self._last_pos.x()
            dy = e.y() - self._last_pos.y()
            rx = self.wx_max - self.wx_min
            ry = self.wy_max - self.wy_min
            self.pan_x -= dx * rx / max(self.width(),  1)
            self.pan_y += dy * ry / max(self.height(), 1)
            self._last_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._last_pos = None
        self.update()

    def wheelEvent(self, e) -> None:
        delta = e.angleDelta().y()
        if delta == 0:
            return
        sx, sy   = e.pos().x(), e.pos().y()
        wx0, wy0 = self._s2w(sx, sy)
        factor   = 1.0 - math.copysign(0.10, delta)
        self.zoom = max(0.5, min(self.zoom * factor, 1e6))

        asp = self.width() / self.height() if self.height() > 0 else 1.0
        self.wx_min = self.pan_x - self.zoom * asp
        self.wx_max = self.pan_x + self.zoom * asp
        self.wy_min = self.pan_y - self.zoom
        self.wy_max = self.pan_y + self.zoom
        wx1, wy1 = self._s2w(sx, sy)
        self.pan_x += wx0 - wx1
        self.pan_y += wy0 - wy1
        self.update()

    # ------------------------------------------------------------------
    # UTILITÀ
    # ------------------------------------------------------------------

    def _s2w(self, sx: int, sy: int):
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return 0.0, 0.0
        return (self.wx_min + (sx / w) * (self.wx_max - self.wx_min),
                self.wy_min + (1.0 - sy / h) * (self.wy_max - self.wy_min))

    @staticmethod
    def _tick(rng: float) -> float:
        if rng <= 0:
            return 1.0
        rough = rng / 8.0
        mag   = 10 ** math.floor(math.log10(rough))
        ratio = rough / mag
        if   ratio >= 5: return 5 * mag
        elif ratio >= 2: return 2 * mag
        return mag