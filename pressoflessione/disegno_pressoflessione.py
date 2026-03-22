"""
disegno_pressoflessione.py  –  v3
==================================
Widget OpenGL 2D professionale – analisi pressoflessione sezioni RC.

Layout tre pannelli (coordinate ruotate, asse Y = d lungo n_vect):
    [Sezione discretizzata] | [Diagramma ε] | [Diagramma σ]

Correzioni v3
─────────────
• Diagramma tensioni completamente riscritto con GL_QUADS banda per banda:
  niente più GL_POLYGON → niente più linee spurie da max a min.
• Zero-crossing preciso inserito nel profilo σ → il diagramma si chiude
  esattamente all'asse neutro senza "gambette".
• Colormap JET simmetrica centrata a σ=0 (Blue=compressione, Green=0, Red=trazione).
  Usata sia per le fibre della sezione (modalità GRADIENTE) sia per i diagrammi σ.
• Legenda cromatica sul lato destro (visibile solo in modalità gradiente).
• Modalità NORMALE: CLS grigio chiaro/scuro, acciaio rosso (trazione) / blu (comp.).
• Sezione ruotata nelle coordinate (xr, yr=d): asse neutro sempre orizzontale
  a yr = d_na in tutti e tre i pannelli.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore    import Qt, QPoint
from PyQt5.QtGui     import (QColor, QFont, QLinearGradient,
                              QPainter, QPen, QBrush)
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_TEST,
    GL_LINE_LOOP, GL_LINE_SMOOTH, GL_LINES,
    GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_POINT_SMOOTH, GL_POINTS,
    GL_PROJECTION, GL_QUADS, GL_SRC_ALPHA, GL_TEXTURE_2D,
    GL_TRIANGLE_FAN,
    glBegin, glBlendFunc, glClear, glClearColor, glColor3f, glColor4f,
    glDisable, glEnable, glEnd, glLineWidth, glLoadIdentity, glMatrixMode,
    glOrtho, glPointSize, glVertex2f, glViewport,
)

# ══════════════════════════════════════════════════════════════════════════════
# COLORMAP JET  –  Blue(0) → Cyan → Green(0.5) → Yellow → Red(1)
# ══════════════════════════════════════════════════════════════════════════════
_CMAP_STOPS: List[Tuple[float, Tuple]] = [
    (0.00, (0.0,  0.0,  1.0)),   # Blue       – compressione massima
    (0.25, (0.0,  1.0,  1.0)),   # Cyan
    (0.50, (0.0,  1.0,  0.0)),   # Green      – σ = 0
    (0.75, (1.0,  1.0,  0.0)),   # Yellow
    (1.00, (1.0,  0.0,  0.0)),   # Red        – trazione massima
]


def _jet(t: float, alpha: float = 0.88) -> Tuple[float, float, float, float]:
    """Jet colormap  t ∈ [0,1] → RGBA."""
    t = max(0.0, min(1.0, t))
    for i in range(len(_CMAP_STOPS) - 1):
        t0, c0 = _CMAP_STOPS[i]
        t1, c1 = _CMAP_STOPS[i + 1]
        if t <= t1 + 1e-9:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (
                c0[0] + f * (c1[0] - c0[0]),
                c0[1] + f * (c1[1] - c0[1]),
                c0[2] + f * (c1[2] - c0[2]),
                alpha,
            )
    return (*_CMAP_STOPS[-1][1], alpha)


# ══════════════════════════════════════════════════════════════════════════════
# COSTANTI DI STILE
# ══════════════════════════════════════════════════════════════════════════════
_BG  = (40/255, 40/255, 40/255, 1.0)
_GRD = (0.18, 0.18, 0.18)
_SEP = (0.36, 0.36, 0.36)
_NA  = (1.00, 0.88, 0.08, 0.95)

# Modo NORMALE – sezione
_CN_REACT = (0.78, 0.78, 0.78, 0.90)   # cls chiaro = reagisce
_CN_INERT = (0.26, 0.26, 0.26, 0.75)   # cls scuro  = non reagisce / fessurato
_SN_TRAZ  = (1.00, 0.20, 0.08, 0.92)   # acciaio rosso = trazione
_SN_COMP  = (0.15, 0.42, 1.00, 0.92)   # acciaio blu  = compressione
_SN_ZERO  = (0.68, 0.68, 0.68, 0.85)

# Diagrammi (modo NORMALE)
_DC_F = (0.10, 0.35, 0.90, 0.38)   # fill  compressione
_DC_L = (0.18, 0.52, 1.00, 0.95)   # line  compressione
_DT_F = (0.90, 0.20, 0.06, 0.28)   # fill  trazione
_DT_L = (1.00, 0.28, 0.08, 0.95)   # line  trazione

_FF   = "Arial"
_FS   = 8
_FM   = 9


def _lerp(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(4))


# ══════════════════════════════════════════════════════════════════════════════
# WIDGET
# ══════════════════════════════════════════════════════════════════════════════
class OpenGLPressoflessioneWidget(QOpenGLWidget):
    """Visualizzatore 2D professionale per analisi pressoflessione."""

    def __init__(self, ui, parent=None) -> None:
        super().__init__(parent)
        self.ui = ui

        self._results: Optional[Dict] = None
        self._tipo      = "SLU"
        self._theta_deg = 0.0
        self._d_na      = 0.0
        self._n_vect    = (0.0, 1.0)
        self._gs        = 10.0

        # Fibre in coordinate ruotate
        self._fc: List[Tuple] = []    # (xr, yr, gs, eps, sig)  conc
        self._fa: List[Tuple] = []    # (xr, yr, r,  eps, sig)  barre

        # Profili
        self._sp: List[Tuple] = []    # [(d, eps)]  2 punti (lineare)
        self._tp: List[Tuple] = []    # [(d, sig_avg_conc)]  per banda
        self._bs: List[Tuple] = []    # [(yr, sig, xr)]  barre

        # Range sigma globale (per colormap)
        self._sig_min  = -1.0
        self._sig_max  = 1.0
        self._sig_mabs = 1.0   # max(|min|, |max|)  – per colormap simmetrica

        # Bounds sezione (coordinate ruotate)
        self._sx0 = self._sx1 = 0.0
        self._sy0 = self._sy1 = 0.0

        # Layout diagrammi
        self._xsc = 0.0   # x centro pannello deformazioni
        self._xtc = 0.0   # x centro pannello tensioni
        self._dhw = 50.0  # semi-larghezza diagrammi
        self._ssc = 1.0   # scala strain
        self._tsc = 1.0   # scala stress

        # Vista
        self.pan_2d   = [0.0, 0.0]
        self.zoom_2d  = 100.0
        self._lmp: Optional[QPoint] = None
        self.cursor_pos = QPoint(0, 0)
        self.wx_min = self.wx_max = 0.0
        self.wy_min = self.wy_max = 0.0

        # Modalità colore
        self._mode = 'normale'   # 'normale' | 'gradiente'

        self.setMouseTracking(True)

    # ──────────────────────────────────────────────────────────────────────────
    # API ESTERNA
    # ──────────────────────────────────────────────────────────────────────────

    def set_display_mode(self, mode: str) -> None:
        """Imposta la modalità di colorazione: 'normale' | 'gradiente'."""
        self._mode = mode
        self.update()

    def set_results(self, results: Dict) -> None:
        if results is None:
            self._results = None; self.update(); return

        self._results   = results
        self._tipo      = results.get('tipo', 'SLU')
        self._d_na      = results.get('d_na', 0.0)
        self._theta_deg = results.get('theta_deg', 0.0)
        nv              = results.get('n_vect', (0.0, 1.0))
        self._n_vect    = (float(nv[0]), float(nv[1]))
        self._gs        = results.get('grid_step', 10.0)

        gs   = self._gs
        nx, ny = self._n_vect
        ag   = gs * gs

        # ── Ruota coordinate fibre ────────────────────────────────────────────
        # x_rot = ny·x − nx·y   (tangenziale)
        # y_rot = nx·x + ny·y = d  (normale, asse verticale dei diagrammi)
        self._fc = []; self._fa = []
        all_sig: List[float] = []

        for x, y, area, eps, sig in results.get('fibre', []):
            all_sig.append(sig)
            xr = ny * x - nx * y
            yr = nx * x + ny * y
            if abs(area - ag) > ag * 0.30:
                r = math.sqrt(area / math.pi)
                self._fa.append((xr, yr, r, eps, sig))
            else:
                self._fc.append((xr, yr, gs, eps, sig))

        # Range sigma per colormap simmetrica centrata a 0
        if all_sig:
            self._sig_min  = min(all_sig)
            self._sig_max  = max(all_sig)
            self._sig_mabs = max(abs(self._sig_min), abs(self._sig_max), 1e-3)
        else:
            self._sig_min = -1.0; self._sig_max = 1.0; self._sig_mabs = 1.0

        # ── Bounds sezione (coordinate ruotate) ───────────────────────────────
        all_r = [(xr, yr) for xr, yr, *_ in self._fc]
        all_r += [(xr, yr) for xr, yr, *_ in self._fa]
        if all_r:
            xs = [c[0] for c in all_r]; ys = [c[1] for c in all_r]
            mg = gs * 0.65
            self._sx0, self._sx1 = min(xs) - mg, max(xs) + mg
            self._sy0, self._sy1 = min(ys) - mg, max(ys) + mg
        else:
            self._sx0 = self._sx1 = self._sy0 = self._sy1 = 0.0

        # ── Profilo deformazioni (2 punti – sempre lineare) ───────────────────
        self._sp = [
            (results.get('d_min', self._sy0), results.get('eps_bot', 0.0)),
            (results.get('d_max', self._sy1), results.get('eps_top', 0.0)),
        ]

        # ── Profilo tensioni calcestruzzo (media per banda d) ─────────────────
        # Usando le fibre della griglia: raggruppa per yr, calcola media sigma.
        # Le fibre fessurate hanno sig=0 e abbassano correttamente la media.
        ymap: dict = defaultdict(list)
        for xr, yr, _g, eps, sig in self._fc:
            key = round(yr / gs) * gs
            ymap[key].append(sig)
        self._tp = sorted(
            [(yk, float(np.mean(vs))) for yk, vs in ymap.items()],
            key=lambda t: t[0]
        )

        # ── Tensioni barre ────────────────────────────────────────────────────
        self._bs = [(yr, sig, xr) for xr, yr, r, eps, sig in self._fa]

        # ── Layout ────────────────────────────────────────────────────────────
        self._layout()

        # ── Vista iniziale ────────────────────────────────────────────────────
        x_end = self._xtc + self._dhw
        cx = (self._sx0 + x_end) / 2.0
        cy = (self._sy0 + self._sy1) / 2.0
        self.pan_2d  = [cx, cy]
        self.zoom_2d = max(x_end - self._sx0, self._sy1 - self._sy0) / 2.0 * 1.28
        self.update()

    # ──────────────────────────────────────────────────────────────────────────
    # LAYOUT
    # ──────────────────────────────────────────────────────────────────────────

    def _layout(self) -> None:
        sw        = max(self._sx1 - self._sx0, 10.0)
        self._dhw = sw * 0.62
        gap       = sw * 0.27
        self._xsc = self._sx1 + gap + self._dhw
        self._xtc = self._xsc + 2 * self._dhw + gap
        eps_max   = max(abs(self._sp[0][1]), abs(self._sp[-1][1]), 1e-9) if self._sp else 1e-9
        self._ssc = self._dhw * 0.87 / eps_max
        self._tsc = self._dhw * 0.87 / max(self._sig_mabs, 1e-6)

    # ──────────────────────────────────────────────────────────────────────────
    # OPENGL
    # ──────────────────────────────────────────────────────────────────────────

    def initializeGL(self) -> None:
        glClearColor(*_BG)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH); glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w, h) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_TEXTURE_2D); glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE)
        glClear(GL_COLOR_BUFFER_BIT)

        asp = self.width() / self.height() if self.height() > 0 else 1.0
        hw = self.zoom_2d * asp; hh = self.zoom_2d
        self.wx_min = self.pan_2d[0] - hw; self.wx_max = self.pan_2d[0] + hw
        self.wy_min = self.pan_2d[1] - hh; self.wy_max = self.pan_2d[1] + hh

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(self.wx_min, self.wx_max, self.wy_min, self.wy_max, -1, 1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        self._gl_grid()
        if self._results is None:
            self._ov_empty(); return

        self._gl_seps()
        self._gl_section()
        self._gl_na()
        self._gl_strain()
        self._gl_stress()
        self._ov_main()

    # ──────────────────────────────────────────────────────────────────────────
    # GL – GRIGLIA
    # ──────────────────────────────────────────────────────────────────────────

    def _gl_grid(self) -> None:
        dx = self.wx_max - self.wx_min; dy = self.wy_max - self.wy_min
        tx = self._tick(dx); ty = self._tick(dy)
        glColor3f(*_GRD); glLineWidth(0.6)
        glBegin(GL_LINES)
        x = math.floor(self.wx_min / tx) * tx
        while x <= self.wx_max + tx:
            glVertex2f(x, self.wy_min); glVertex2f(x, self.wy_max); x += tx
        y = math.floor(self.wy_min / ty) * ty
        while y <= self.wy_max + ty:
            glVertex2f(self.wx_min, y); glVertex2f(self.wx_max, y); y += ty
        glEnd()

    def _gl_seps(self) -> None:
        x1 = (self._sx1 + self._xsc - self._dhw) / 2.0
        x2 = (self._xsc + self._dhw + self._xtc - self._dhw) / 2.0
        glColor3f(*_SEP); glLineWidth(0.7)
        glBegin(GL_LINES)
        glVertex2f(x1, self.wy_min); glVertex2f(x1, self.wy_max)
        glVertex2f(x2, self.wy_min); glVertex2f(x2, self.wy_max)
        glEnd()

    # ──────────────────────────────────────────────────────────────────────────
    # GL – SEZIONE DISCRETIZZATA
    # ──────────────────────────────────────────────────────────────────────────

    def _gl_section(self) -> None:
        if not self._fc and not self._fa: return
        gs   = self._gs; h = gs * 0.90; mode = self._mode
        sm_a = max((abs(s) for *_, s in self._fa), default=1.0) or 1.0

        # ── Calcestruzzo (QUADS) ─────────────────────────────────────────────
        glBegin(GL_QUADS)
        for xr, yr, _g, eps, sig in self._fc:
            col = (self._jet_section(sig) if mode == 'gradiente'
                   else (_CN_REACT if abs(sig) > 0.3 else _CN_INERT))
            glColor4f(*col)
            glVertex2f(xr - h/2, yr - h/2); glVertex2f(xr + h/2, yr - h/2)
            glVertex2f(xr + h/2, yr + h/2); glVertex2f(xr - h/2, yr + h/2)
        glEnd()

        # ── Barre acciaio (cerchi) ────────────────────────────────────────────
        N = 36
        for xr, yr, r, eps, sig in self._fa:
            col_f = (self._jet_section(sig) if mode == 'gradiente'
                     else (_SN_TRAZ if sig > 1.0 else (_SN_COMP if sig < -1.0 else _SN_ZERO)))
            col_o = (col_f[0] * 0.62, col_f[1] * 0.62, col_f[2] * 0.62, 1.0)
            glColor4f(*col_f)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(xr, yr)
            for k in range(N + 1):
                t = 2 * math.pi * k / N
                glVertex2f(xr + r * math.cos(t), yr + r * math.sin(t))
            glEnd()
            glColor4f(*col_o); glLineWidth(1.2)
            glBegin(GL_LINE_LOOP)
            for k in range(N):
                t = 2 * math.pi * k / N
                glVertex2f(xr + r * math.cos(t), yr + r * math.sin(t))
            glEnd()

    def _jet_section(self, sig: float) -> Tuple:
        """Mappa sigma al jet simmetrico (α alta per la sezione)."""
        return _jet(self._s2t(sig), alpha=0.88)

    def _s2t(self, sig: float) -> float:
        """sigma → t ∈ [0,1]  con zero sempre a t=0.5."""
        ma = self._sig_mabs or 1.0
        return max(0.0, min(1.0, (sig + ma) / (2.0 * ma)))

    # ──────────────────────────────────────────────────────────────────────────
    # GL – ASSE NEUTRO
    # ──────────────────────────────────────────────────────────────────────────

    def _gl_na(self) -> None:
        """In coordinate ruotate l'asse neutro è sempre orizzontale a yr = d_na."""
        d   = self._d_na
        x0  = self._sx0 - 3
        x1  = self._xtc + self._dhw + 5
        L   = x1 - x0
        ns  = max(int(L / 10), 4)
        glColor4f(*_NA); glLineWidth(1.9)
        glBegin(GL_LINES)
        for k in range(ns):
            ta = k / ns; tb = min((k + 0.55) / ns, 1.0)
            glVertex2f(x0 + ta * L, d); glVertex2f(x0 + tb * L, d)
        glEnd()

    # ──────────────────────────────────────────────────────────────────────────
    # GL – DIAGRAMMA DEFORMAZIONI  (invariato – funziona correttamente)
    # ──────────────────────────────────────────────────────────────────────────

    def _gl_strain(self) -> None:
        if len(self._sp) < 2: return
        xc = self._xsc; sc = self._ssc
        d0, e0 = self._sp[0]; d1, e1 = self._sp[-1]

        # Asse zero
        glColor4f(0.58, 0.58, 0.58, 0.60); glLineWidth(0.9)
        glBegin(GL_LINES)
        glVertex2f(xc, d0 - 4); glVertex2f(xc, d1 + 4)
        glEnd()

        # Tick estremità sezione
        hw_t = self._dhw * 0.17
        glColor4f(0.48, 0.48, 0.48, 0.55); glLineWidth(0.8)
        glBegin(GL_LINES)
        glVertex2f(xc - hw_t, d0); glVertex2f(xc + hw_t, d0)
        glVertex2f(xc - hw_t, d1); glVertex2f(xc + hw_t, d1)
        glEnd()

        # Zero-crossing preciso
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
            if abs(va) < 1e-12 and abs(vb) < 1e-12: continue
            mid_v = (va + vb) / 2.0
            fill  = _DC_F if mid_v <= 0 else _DT_F
            line  = _DC_L if mid_v <= 0 else _DT_L

            # Riempimento
            glColor4f(*fill)
            glBegin(GL_QUADS)
            glVertex2f(xc,        da); glVertex2f(xc + va * sc, da)
            glVertex2f(xc + vb * sc, db); glVertex2f(xc,        db)
            glEnd()

            # Contorno (solo i tre lati "attivi", non il lato xc)
            glColor4f(*line); glLineWidth(1.9)
            glBegin(GL_LINES)
            # tick inferiore
            glVertex2f(xc,       da); glVertex2f(xc + va * sc, da)
            # profilo inclinato
            glVertex2f(xc + va * sc, da); glVertex2f(xc + vb * sc, db)
            # tick superiore
            glVertex2f(xc + vb * sc, db); glVertex2f(xc,       db)
            glEnd()

    # ──────────────────────────────────────────────────────────────────────────
    # GL – DIAGRAMMA TENSIONI  (riscritto con GL_QUADS)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _insert_zero_crossings(
            profile: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Inserisce punti interpolati dove il profilo taglia sigma=0."""
        out: List[Tuple[float, float]] = []
        for i, (d, s) in enumerate(profile):
            if i > 0:
                dp, sp = profile[i - 1]
                if sp * s < 0 and abs(s - sp) > 1e-15:
                    dz = dp + (-sp) / (s - sp) * (d - dp)
                    out.append((dz, 0.0))
            out.append((d, s))
        return out

    def _gl_stress(self) -> None:
        xc  = self._xtc
        sc  = self._tsc
        y0  = self._sy0
        y1  = self._sy1
        mode = self._mode

        # ── Asse zero ────────────────────────────────────────────────────────
        glColor4f(0.58, 0.58, 0.58, 0.60); glLineWidth(0.9)
        glBegin(GL_LINES)
        glVertex2f(xc, y0 - 4); glVertex2f(xc, y1 + 4)
        glEnd()

        # Tick estremità sezione
        hw_t = self._dhw * 0.17
        glColor4f(0.48, 0.48, 0.48, 0.55); glLineWidth(0.8)
        glBegin(GL_LINES)
        glVertex2f(xc - hw_t, y0); glVertex2f(xc + hw_t, y0)
        glVertex2f(xc - hw_t, y1); glVertex2f(xc + hw_t, y1)
        glEnd()

        # ── Profilo corpo sezione ─────────────────────────────────────────────
        if len(self._tp) >= 2:
            prof = self._insert_zero_crossings(self._tp)
            n    = len(prof)

            # Fill a bande (GL_QUADS – nessun GL_POLYGON, nessuna linea spuria)
            glBegin(GL_QUADS)
            for i in range(n - 1):
                d0, s0 = prof[i]
                d1, s1 = prof[i + 1]
                mid = (s0 + s1) / 2.0
                if abs(mid) < 0.05: continue

                if mode == 'gradiente':
                    col_f = _jet(self._s2t(mid), alpha=0.42)
                else:
                    col_f = _DC_F if mid < 0 else _DT_F

                glColor4f(*col_f)
                glVertex2f(xc,          d0)
                glVertex2f(xc + s0 * sc, d0)
                glVertex2f(xc + s1 * sc, d1)
                glVertex2f(xc,          d1)
            glEnd()

            # Contorno profilo  (solo la linea del profilo + ticks top/bot)
            # ─ profilo: segmento per segmento con colore adeguato
            glLineWidth(1.9)
            glBegin(GL_LINES)
            for i in range(n - 1):
                d0, s0 = prof[i]
                d1, s1 = prof[i + 1]
                mid = (s0 + s1) / 2.0
                if abs(mid) < 0.05: continue

                if mode == 'gradiente':
                    col_l = _jet(self._s2t(mid), alpha=0.96)
                else:
                    col_l = _DC_L if mid < 0 else _DT_L

                glColor4f(*col_l)
                glVertex2f(xc + s0 * sc, d0)
                glVertex2f(xc + s1 * sc, d1)
            glEnd()

            # tick inferiore e superiore (solo se sigma ≠ 0)
            d_bot, s_bot = prof[0]
            d_top, s_top = prof[-1]
            glLineWidth(1.9)
            glBegin(GL_LINES)
            if abs(s_bot) > 0.05:
                col_b = (_jet(self._s2t(s_bot), 0.96) if mode == 'gradiente'
                         else (_DC_L if s_bot < 0 else _DT_L))
                glColor4f(*col_b)
                glVertex2f(xc, d_bot); glVertex2f(xc + s_bot * sc, d_bot)
            if abs(s_top) > 0.05:
                col_t = (_jet(self._s2t(s_top), 0.96) if mode == 'gradiente'
                         else (_DC_L if s_top < 0 else _DT_L))
                glColor4f(*col_t)
                glVertex2f(xc + s_top * sc, d_top); glVertex2f(xc, d_top)
            glEnd()

        # ── Barre acciaio ─────────────────────────────────────────────────────
        sm_a = max((abs(s) for _, s, _ in self._bs), default=1.0) or 1.0

        glPointSize(8.0)
        glBegin(GL_POINTS)
        for yr, sig, _ in self._bs:
            col = (self._jet_section(sig) if mode == 'gradiente'
                   else (_SN_TRAZ if sig > 1.0 else (_SN_COMP if sig < -1.0 else _SN_ZERO)))
            glColor4f(*col)
            glVertex2f(xc + sig * sc, yr)
        glEnd()

        # Linee orizzontali barre → asse zero
        glLineWidth(1.4)
        glBegin(GL_LINES)
        for yr, sig, _ in self._bs:
            col = (self._jet_section(sig) if mode == 'gradiente'
                   else (_SN_TRAZ if sig > 1.0 else (_SN_COMP if sig < -1.0 else _SN_ZERO)))
            glColor4f(col[0], col[1], col[2], 0.60)
            glVertex2f(xc, yr); glVertex2f(xc + sig * sc, yr)
        glEnd()

    # ──────────────────────────────────────────────────────────────────────────
    # OVERLAY QPAINTER
    # ──────────────────────────────────────────────────────────────────────────

    def _ov_empty(self) -> None:
        p = QPainter(self)
        p.setFont(QFont(_FF, _FM))
        p.setPen(QColor(105, 105, 105))
        p.drawText(self.rect(), Qt.AlignCenter,
                   "Nessun risultato.\nEseguire prima la verifica.")
        p.end()

    def _ov_main(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        def ws(wx, wy):
            nx = (wx - self.wx_min) / (self.wx_max - self.wx_min)
            ny = (wy - self.wy_min) / (self.wy_max - self.wy_min)
            return int(nx * W), int((1 - ny) * H)

        def lbl(wx, wy, text, col=QColor(200, 200, 200),
                bold=False, size=_FS, anc='left'):
            sx, sy = ws(wx, wy)
            f = QFont(_FF, size); f.setBold(bold)
            painter.setFont(f)
            fm  = painter.fontMetrics()
            tw  = fm.horizontalAdvance(text)
            if   anc == 'right':  sx -= tw
            elif anc == 'center': sx -= tw // 2
            if not (-60 <= sx <= W + 60 and -20 <= sy <= H + 20): return
            painter.setPen(QPen(col))
            painter.drawText(sx, sy, text)

        # ── Titoli pannelli ───────────────────────────────────────────────────
        th_s   = f"  [θ={self._theta_deg:.1f}°]" if abs(self._theta_deg) > 0.5 else ""
        y_tit  = self._sy1 + (self.wy_max - self._sy1) * 0.38
        cx_sec = (self._sx0 + self._sx1) / 2.0
        lbl(cx_sec, y_tit, f"Sezione{th_s}  [{self._tipo}]",
            QColor(210, 210, 210), bold=True, size=_FM, anc='center')
        lbl(self._xsc, y_tit, "Deformazioni  ε",
            QColor(110, 165, 255), bold=True, size=_FM, anc='center')
        lbl(self._xtc, y_tit, "Tensioni  σ  [MPa]",
            QColor(255, 155, 65), bold=True, size=_FM, anc='center')

        # ── Asse neutro label ─────────────────────────────────────────────────
        lbl(self._sx0 + 3, self._d_na + 3, "A.N.", QColor(255, 220, 50), bold=True)

        # ── Valori ε ─────────────────────────────────────────────────────────
        if len(self._sp) == 2:
            d0, e0 = self._sp[0]; d1, e1 = self._sp[-1]
            xcs = self._xsc; scs = self._ssc; off = 4
            dz  = None
            if e0 * e1 < 0 and abs(e1 - e0) > 1e-15:
                dz = d0 + (-e0) / (e1 - e0) * (d1 - d0)
            for d_v, e_v in [(d0, e0), (d1, e1)]:
                if abs(e_v) < 1e-12: continue
                a = 'left' if e_v >= 0 else 'right'
                lbl(xcs + e_v * scs + (off if e_v >= 0 else -off), d_v,
                    f"{e_v * 1000:.2f}‰",
                    QColor(140, 190, 255) if e_v < 0 else QColor(255, 155, 90), anc=a)
            if dz is not None:
                lbl(xcs + 3, dz, "ε=0", QColor(255, 220, 50))

        # ── Valori σ corpo sezione ────────────────────────────────────────────
        if self._tp:
            xct = self._xtc; sct = self._tsc; off = 4
            reported: set = set()
            # Candidati: estremi + massima compressione + massima trazione
            cands = [self._tp[0], self._tp[-1]]
            if self._tp:
                mc = min(self._tp, key=lambda t: t[1])
                mt = max(self._tp, key=lambda t: t[1])
                if mc not in cands: cands.append(mc)
                if mt not in cands: cands.append(mt)
            for d_v, s_v in cands:
                if abs(s_v) < 0.3: continue
                k = round(d_v / max(self._gs, 1))
                if k in reported: continue
                reported.add(k)
                a = 'left' if s_v >= 0 else 'right'
                col = QColor(140, 190, 255) if s_v < 0 else QColor(255, 155, 90)
                lbl(xct + s_v * sct + (off if s_v >= 0 else -off), d_v,
                    f"{s_v:.1f}", col, anc=a)

        # ── Valori σ barre ────────────────────────────────────────────────────
        xct = self._xtc; sct = self._tsc; off = 4
        seen_y: set = set()
        for yr, sig, _ in self._bs:
            k = round(yr / max(self._gs, 1))
            if k in seen_y: continue
            seen_y.add(k)
            if abs(sig) < 0.1: continue
            a = 'left' if sig >= 0 else 'right'
            col = QColor(255, 100, 80) if sig > 0 else QColor(90, 160, 255)
            lbl(xct + sig * sct + (off if sig >= 0 else -off), yr,
                f"{sig:.1f}", col, anc=a)

        # ── Legenda colormap (solo modalità gradiente) ────────────────────────
        if self._mode == 'gradiente':
            self._draw_legend(painter, W, H)

        # ── Box verifica (basso sinistra) ─────────────────────────────────────
        self._ov_box(painter, W, H)

        # ── Tracker (basso destra) ────────────────────────────────────────────
        mx, my   = self.cursor_pos.x(), self.cursor_pos.y()
        wxc, wyc = self._stw(mx, my)
        painter.setFont(QFont(_FF, _FS))
        painter.setPen(QPen(QColor(200, 200, 200, 165)))
        painter.drawLine(mx - 7, my, mx + 7, my)
        painter.drawLine(mx, my - 7, mx, my + 7)
        trk = f"X: {wxc:.1f}   Y: {wyc:.1f}"
        tw  = painter.fontMetrics().horizontalAdvance(trk)
        painter.drawText(W - tw - 10, H - 9, trk)

        painter.end()

    # ──────────────────────────────────────────────────────────────────────────
    # LEGENDA COLORMAP (QPainter)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_legend(self, painter: QPainter, W: int, H: int) -> None:
        """Scala cromatica JET sul lato destro (visibile solo in modalità gradiente)."""
        bar_w   = 18
        bar_h   = min(260, H - 100)
        mrg_r   = 70    # margine dal bordo destro
        x_bar   = W - mrg_r - bar_w
        y_bar   = (H - bar_h) // 2

        # Sfondo riquadro
        bg_x = x_bar - 10
        bg_y = y_bar - 32
        bg_w = bar_w + mrg_r - 12
        bg_h = bar_h + 62
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(20, 20, 20, 180))
        painter.drawRoundedRect(bg_x, bg_y, bg_w, bg_h, 5, 5)

        # Titolo
        f_title = QFont(_FF, _FS); f_title.setBold(True)
        painter.setFont(f_title)
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(bg_x + 4, bg_y + 16, "σ [MPa]")

        # Gradiente  (bottom = Blue = -max_abs,  top = Red = +max_abs)
        gradient = QLinearGradient(x_bar, y_bar + bar_h, x_bar, y_bar)
        gradient.setColorAt(0.00, QColor(0,   0,   255))  # Blue  – compressione
        gradient.setColorAt(0.25, QColor(0,   255, 255))  # Cyan
        gradient.setColorAt(0.50, QColor(0,   255, 0))    # Green – σ = 0
        gradient.setColorAt(0.75, QColor(255, 255, 0))    # Yellow
        gradient.setColorAt(1.00, QColor(255, 0,   0))    # Red   – trazione
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(130, 130, 130), 1))
        painter.drawRect(x_bar, y_bar, bar_w, bar_h)

        # Tick marks e valori (5 livelli: ±max, ±max/2, 0)
        ma     = self._sig_mabs
        ticks  = [-ma, -ma / 2, 0.0, ma / 2, ma]
        f_tick = QFont(_FF, _FS - 1)
        painter.setFont(f_tick)
        for val in ticks:
            # t = (val + ma) / (2*ma)  →  y_tick da TOP (y_bar) a BOTTOM (y_bar+bar_h)
            t_norm = (val + ma) / (2.0 * ma)   # 0=bottom(Blue), 1=top(Red)
            y_tick = int(y_bar + bar_h - t_norm * bar_h)

            # tick mark
            painter.setPen(QPen(QColor(160, 160, 160), 1))
            painter.drawLine(x_bar + bar_w, y_tick, x_bar + bar_w + 5, y_tick)

            # label
            if abs(val) > 999 or (0 < abs(val) < 0.01):
                txt = f"{val:.1e}"
            else:
                txt = f"{val:.1f}" if val != 0 else "0"
            painter.setPen(QPen(QColor(210, 210, 210)))
            painter.drawText(x_bar + bar_w + 7, y_tick + 4, txt)

        # Labels MIN / MAX espliciti
        painter.setPen(QColor(170, 170, 170))
        painter.drawText(bg_x + 3, y_bar + bar_h + 16,
                         f"Min: {-ma:.1f}")
        painter.drawText(bg_x + 3, y_bar - 4,
                         f"Max: {ma:.1f}")

    # ──────────────────────────────────────────────────────────────────────────
    # BOX VERIFICA
    # ──────────────────────────────────────────────────────────────────────────

    def _ov_box(self, p: QPainter, W: int, H: int) -> None:
        res = self._results
        if res is None: return
        tipo = res.get('tipo', '')
        ok   = res.get('verificata', False)

        if ok:
            stato = "✓  Verifica SODDISFATTA"
            bcol  = QColor(0,  210, 80);  tcol = QColor(0,  235, 100)
        else:
            stato = "✗  Verifica NON SODDISFATTA"
            bcol  = QColor(220, 48, 48);  tcol = QColor(245, 70, 70)

        det = [f"Analisi {tipo}"]
        if tipo == 'SLU':
            N  = res.get('N_Ed', 0.0)
            M  = abs(res.get('M_Ed', 0.0))
            Mr = res.get('M_Rd', 0.0)
            rr = res.get('rapporto_MEd_MRd', 0.0)
            det += [f"N_Ed = {N:.1f} kN",
                    f"M_Ed = {M:.1f} kNm     M_Rd = {Mr:.1f} kNm",
                    f"M_Ed / M_Rd = {rr:.3f}"]
            if res.get('fuori_dominio', False):
                det.append("⚠  Fuori dal dominio resistente")
        elif tipo == 'SLE':
            sc_v = res.get('sigma_c_compr_max', 0.0)
            ss_v = res.get('sigma_s_traz_max',  0.0)
            lc   = res.get('sigma_c_limit')
            ls   = res.get('sigma_s_limit')
            det += [f"σ_c = {sc_v:.1f} MPa" + (f"  (lim {lc:.1f})" if lc else ""),
                    f"σ_s = {ss_v:.1f} MPa" + (f"  (lim {ls:.1f})" if ls else "")]
            for n in res.get('note', []):
                det.append(f"⚠  {n}")

        lh = 16; px = 10; py = 8
        bw = 295
        bh = py * 2 + 20 + len(det) * lh
        bx = 8; by = H - bh - 8

        p.fillRect(bx, by, bw, bh, QColor(16, 16, 16, 188))
        p.setPen(QPen(bcol, 1)); p.drawRect(bx, by, bw, bh)

        fb = QFont(_FF, _FM + 1); fb.setBold(True)
        p.setFont(fb); p.setPen(QPen(tcol))
        p.drawText(bx + px, by + py + 15, stato)

        p.setPen(QPen(bcol.darker(130), 1))
        p.drawLine(bx + 3, by + py + 21, bx + bw - 3, by + py + 21)

        fs = QFont(_FF, _FS); p.setFont(fs); p.setPen(QPen(QColor(188, 188, 188)))
        for i, line in enumerate(det):
            p.drawText(bx + px, by + py + 21 + (i + 1) * lh, line)

    # ──────────────────────────────────────────────────────────────────────────
    # UTILITÀ
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _tick(r: float) -> float:
        if r <= 0: return 1.0
        m = 10 ** math.floor(math.log10(r / 8.0))
        v = (r / 8.0) / m
        return (5 if v >= 5 else 2 if v >= 2 else 1) * m

    def _stw(self, sx, sy):
        W, H = self.width(), self.height()
        if not W or not H: return 0.0, 0.0
        return (self.wx_min + sx / W * (self.wx_max - self.wx_min),
                self.wy_min + (1 - sy / H) * (self.wy_max - self.wy_min))

    # ──────────────────────────────────────────────────────────────────────────
    # MOUSE
    # ──────────────────────────────────────────────────────────────────────────

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton: self._lmp = e.pos()
        self.cursor_pos = e.pos(); self.update()

    def mouseMoveEvent(self, e) -> None:
        self.cursor_pos = e.pos()
        if self._lmp:
            dx = e.x() - self._lmp.x(); dy = e.y() - self._lmp.y()
            self.pan_2d[0] -= dx * (self.wx_max - self.wx_min) / (self.width()  or 1)
            self.pan_2d[1] += dy * (self.wy_max - self.wy_min) / (self.height() or 1)
            self._lmp = e.pos()
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._lmp = None; self.update()

    def wheelEvent(self, e) -> None:
        d = e.angleDelta().y()
        if not d: return
        sx, sy   = e.pos().x(), e.pos().y()
        wx0, wy0 = self._stw(sx, sy)
        f        = 1.0 - math.copysign(0.10, d)
        self.zoom_2d = max(1.0, min(self.zoom_2d * f, 1e6))
        asp = self.width() / self.height() if self.height() > 0 else 1.0
        self.wx_min = self.pan_2d[0] - self.zoom_2d * asp
        self.wx_max = self.pan_2d[0] + self.zoom_2d * asp
        self.wy_min = self.pan_2d[1] - self.zoom_2d
        self.wy_max = self.pan_2d[1] + self.zoom_2d
        wx1, wy1 = self._stw(sx, sy)
        self.pan_2d[0] += wx0 - wx1; self.pan_2d[1] += wy0 - wy1
        self.update()