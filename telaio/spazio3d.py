"""
spazio3d.py  –  Widget OpenGL 3D per la visualizzazione del telaio.
Usa PyOpenGL + PyQt5 (QOpenGLWidget).

Dipendenze:  PyOpenGL, PyQt5, numpy, Pillow
"""

import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore    import Qt, QPoint
from PyQt5.QtGui     import QFont

import math
from OpenGL.GL  import *
from OpenGL.GLU import *

try:
    from PIL import Image, ImageDraw, ImageFont as PILFont
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ---------------------------------------------------------------------------
# Palette professionale (stile Blender dark theme)
# ---------------------------------------------------------------------------
BG_R, BG_G, BG_B          = 40/255, 40/255, 40/255

GRID_FINE_R  = 0.27; GRID_FINE_G  = 0.27; GRID_FINE_B  = 0.27
GRID_COARSE_R= 0.35; GRID_COARSE_G= 0.35; GRID_COARSE_B= 0.35

AXIS_X  = (0.862, 0.200, 0.200)   # rosso   X
AXIS_Y  = (0.310, 0.620, 0.165)   # verde   Y
AXIS_Z  = (0.161, 0.408, 0.784)   # blu     Z

NODE_R  = 0.20; NODE_G  = 0.85; NODE_B  = 0.95   # ciano brillante
BEAM_R  = 0.75; BEAM_G  = 0.75; BEAM_B  = 0.80
CONSTR_R= 0.15; CONSTR_G= 0.90; CONSTR_B= 0.40
LOAD_R  = 0.95; LOAD_G  = 0.20; LOAD_B  = 0.20
DIST_R  = 1.00; DIST_G  = 0.55; DIST_B  = 0.10

DEF_R   = 0.25; DEF_G   = 0.60; DEF_B   = 1.00
DIAG_FILL_R=1.0; DIAG_FILL_G=0.50; DIAG_FILL_B=0.05
DIAG_LINE_R=1.0; DIAG_LINE_G=0.82; DIAG_LINE_B=0.15


class Spazio3D(QOpenGLWidget):
    """
    Widget OpenGL che visualizza:
      - modello indeformato (aste, nodi, vincoli, carichi)
      - forma deformata amplificata
      - diagrammi delle sollecitazioni (N, Vy, Vz, My, Mz)

    Controllo camera:
      - tasto sinistro  → rotazione orbitale
      - tasto destro    → panning
      - rotella         → zoom
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Dati modello (visualizzazione indeformata) ---
        self.nodi:         dict = {}
        self.aste:         dict = {}
        self.vincoli:      dict = {}
        self.carichi_nodi: dict = {}
        self.carichi_aste: dict = {}

        # --- Dati risultati ---
        self.u_vec:   np.ndarray | None = None
        self.sforzi:  np.ndarray | None = None
        self.t_nodi:  list = []
        self.t_aste:  list = []

        # --- Stato vista ---
        self.view_mode:       str   = "Indeformata"
        self.scala_risultati: float = 1.0

        # --- Camera ---
        self.cam_dist:   float = 15.0
        self.rot_x:      float = 20.0
        self.rot_y:      float = -45.0
        self.pan_x:      float = 0.0
        self.pan_y:      float = 0.0
        self._ortho_mode: bool = False   # True nelle viste ortogonali x/y/z

        self._last_pos  = QPoint()
        self._mouse_btn = None

        # --- Cache testo OpenGL ---
        self._text_cache: dict = {}
        self._pil_font = None

        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # Inizializzazione OpenGL
    # ------------------------------------------------------------------

    def initializeGL(self):
        glClearColor(BG_R, BG_G, BG_B, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        if PIL_OK:
            try:
                self._pil_font = PILFont.truetype("arial.ttf", 13)
            except Exception:
                self._pil_font = PILFont.load_default()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Proiezione: ortografica nelle viste piano, prospettica in 3D
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        w_px   = self.width()  or 1
        h_px   = self.height() or 1
        aspect = w_px / h_px

        if self._ortho_mode:
            # Metà-ampiezza del frustum scalata con cam_dist (zoom tramite rotella)
            half_h = self.cam_dist * 0.5
            half_w = half_h * aspect
            glOrtho(-half_w - self.pan_x,  half_w - self.pan_x,
                    -half_h - self.pan_y,  half_h - self.pan_y,
                    -5000.0, 5000.0)
        else:
            gluPerspective(45.0, aspect, 0.05, 5000.0)

        # Modello-vista
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if self._ortho_mode:
            # In ortografica il pan è gestito direttamente in glOrtho
            pass
        else:
            # Prospettica: allontana la camera, poi applica pan nel piano camera
            glTranslatef(self.pan_x, self.pan_y, -self.cam_dist)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glRotatef(-90, 1, 0, 0)   # Z verso l'alto

        self._disegna_griglia()
        self._disegna_assi()

        if self.view_mode == "Indeformata":
            self._disegna_modello_base()
        elif self.view_mode == "Deformata":
            self._disegna_deformata()
        elif self.view_mode in ("N", "Vy", "Vz", "My", "Mz"):
            self._disegna_diagrammi()

    # ------------------------------------------------------------------
    # Aggiornamento dati
    # ------------------------------------------------------------------

    def aggiorna_geometria(self, nodi, aste, vincoli, carichi_nodi, carichi_aste):
        self.nodi          = nodi
        self.aste          = aste
        self.vincoli       = vincoli
        self.carichi_nodi  = carichi_nodi
        self.carichi_aste  = carichi_aste
        self.view_mode     = "Indeformata"
        self._text_cache.clear()
        self.update()

    def aggiorna_risultati(self, u_vec, sforzi, t_nodi, t_aste, view_mode, scala):
        self.u_vec            = u_vec
        self.sforzi           = sforzi
        self.t_nodi           = [np.array(n) for n in t_nodi]
        self.t_aste           = list(t_aste)
        self.view_mode        = view_mode
        self.scala_risultati  = scala
        self._text_cache.clear()
        self.update()

    def imposta_vista(self, preset: str):
        """Preset di vista: '3d', 'x', 'y', 'z'"""
        presets = {
            '3d': (20.0, -45.0, False),
            'x':  (0.0,  -90.0, True),
            'y':  (0.0,    0.0, True),
            'z':  (90.0,   0.0, True),
        }
        if preset in presets:
            self.rot_x, self.rot_y, self._ortho_mode = presets[preset]
            # In modalità ortografica azzera il pan per evitare offset residui
            if self._ortho_mode:
                self.pan_x = 0.0
                self.pan_y = 0.0
        self.update()

    # ------------------------------------------------------------------
    # Griglia infinita stile Blender
    # ------------------------------------------------------------------

    def _disegna_griglia(self):
        """
        Griglia in due livelli con alpha fade radiale sulla distanza:
          - griglia fine  ogni 1 unità  (grigio scuro)
          - griglia grossa ogni 5 unità (grigio più chiaro)
        """
        DIM   = 100     # estensione della griglia (unità)
        FADE  = 40    # distanza dal centro alla quale inizia il fade
        SEG_L = 2       # lunghezza dei "pezzetti" di linea per una sfumatura fluida

        glLineWidth(1.0)
        glDepthMask(GL_FALSE)   # non sporcare il depth buffer con le linee

        # Funzione helper per calcolare l'alpha radiale di un punto (x, y)
        def get_alpha(x, y):
            dist = math.hypot(x, y) / FADE
            return max(0.0, 1.0 - dist * dist * 0.55)

        for step, cr, cg, cb in (
            (1, GRID_FINE_R,   GRID_FINE_G,   GRID_FINE_B),
            (5, GRID_COARSE_R, GRID_COARSE_G, GRID_COARSE_B),
        ):
            glBegin(GL_LINES)
            for i in range(-DIM, DIM + 1, step):
                if step == 1 and (i % 5 == 0):
                    continue    # le linee di step=5 le disegna il secondo ciclo

                # Spezzettiamo la linea in piccoli segmenti per sfumarla progressivamente
                for j in range(-DIM, DIM, SEG_L):
                    
                    # --- LINEE VERTICALI (x = i fisso, y = j variabile) ---
                    y1 = j
                    y2 = j + SEG_L
                    a1_v = get_alpha(i, y1)
                    a2_v = get_alpha(i, y2)

                    # Disegniamo il segmentino solo se almeno uno dei due vertici è visibile
                    if a1_v > 0.01 or a2_v > 0.01:
                        glColor4f(cr, cg, cb, a1_v); glVertex3f(i, y1, 0)
                        glColor4f(cr, cg, cb, a2_v); glVertex3f(i, y2, 0)

                    # --- LINEE ORIZZONTALI (y = i fisso, x = j variabile) ---
                    x1 = j
                    x2 = j + SEG_L
                    a1_h = get_alpha(x1, i)
                    a2_h = get_alpha(x2, i)

                    if a1_h > 0.01 or a2_h > 0.01:
                        glColor4f(cr, cg, cb, a1_h); glVertex3f(x1, i, 0)
                        glColor4f(cr, cg, cb, a2_h); glVertex3f(x2, i, 0)

            glEnd()

        glDepthMask(GL_TRUE)

    # ------------------------------------------------------------------
    # Assi infiniti stile Blender
    # ------------------------------------------------------------------

    def _disegna_assi(self):
        """
        Assi X/Y/Z che si estendono quasi all'infinito.
        La parte positiva è opaca, la negativa semi-trasparente (stilema Blender).
        Le linee sottili percorrono l'intera estensione con fade.
        """
        EXT       = 50    # estensione praticamente infinita
        EXT_NEG   = -50.0     # parte negativa più breve e sfumata
        THICK_POS = 1.8
        THICK_NEG = 1.0

        glDepthMask(GL_FALSE)

        # -- parte positiva (opaca) --
        glLineWidth(THICK_POS)
        glBegin(GL_LINES)
        glColor4f(*AXIS_X, 1.0); glVertex3f(0,0,0); glVertex3f(EXT, 0, 0)
        glColor4f(*AXIS_Y, 1.0); glVertex3f(0,0,0); glVertex3f(0, EXT, 0)
        glColor4f(*AXIS_Z, 1.0); glVertex3f(0,0,0); glVertex3f(0, 0, EXT)
        glEnd()

        # -- parte negativa (tratteggiata visivamente via alpha ridotta) --
        glLineWidth(THICK_NEG)
        glBegin(GL_LINES)
        glColor4f(*AXIS_X, 0.30); glVertex3f(0,0,0); glVertex3f(EXT_NEG, 0, 0)
        glColor4f(*AXIS_Y, 0.30); glVertex3f(0,0,0); glVertex3f(0, EXT_NEG, 0)
        glColor4f(*AXIS_Z, 0.30); glVertex3f(0,0,0); glVertex3f(0, 0, EXT_NEG)
        glEnd()

        # -- puntino origine --
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glColor4f(0.9, 0.9, 0.9, 0.85)
        glVertex3f(0, 0, 0)
        glEnd()

        glDepthMask(GL_TRUE)

    # ------------------------------------------------------------------
    # Disegno testo via bitmap PIL (con cache)
    # ------------------------------------------------------------------

    def _disegna_testo(self, x: float, y: float, z: float,
                       testo: str, r=1.0, g=1.0, b=1.0, a=1.0):
        if not PIL_OK:
            return
        color_tuple = (int(r*255), int(g*255), int(b*255), int(a*255))
        key = (testo, color_tuple)

        if key not in self._text_cache:
            dummy = Image.new('RGBA', (1, 1))
            draw  = ImageDraw.Draw(dummy)
            try:
                bbox = draw.textbbox((0, 0), testo, font=self._pil_font)
                w = int(bbox[2] - bbox[0]) + 8
                h = int(bbox[3] - bbox[1]) + 6
            except Exception:
                w, h = max(len(testo) * 8, 10), 18

            img  = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            # ombra scura leggera per leggibilità
            for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                draw.text((2+ox, 2+oy), testo, font=self._pil_font,
                          fill=(0, 0, 0, 160))
            draw.text((2, 2), testo, font=self._pil_font, fill=color_tuple)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            self._text_cache[key] = (w, h, img.tobytes("raw", "RGBA", 0, 1))

        w, h, data = self._text_cache[key]
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glRasterPos3f(x, y, z)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)

    # ------------------------------------------------------------------
    # Vista INDEFORMATA
    # ------------------------------------------------------------------

    def _disegna_modello_base(self):
        # ---- Aste ----
        glColor3f(BEAM_R, BEAM_G, BEAM_B)
        glLineWidth(2.5)
        glBegin(GL_LINES)
        for a in self.aste.values():
            p1 = self.nodi.get(a['n1'])
            p2 = self.nodi.get(a['n2'])
            if p1 is not None and p2 is not None:
                glVertex3f(*p1); glVertex3f(*p2)
        glEnd()

        # Etichette aste (grigio tenue)
        for aid, a in self.aste.items():
            p1 = self.nodi.get(a['n1'])
            p2 = self.nodi.get(a['n2'])
            if p1 and p2:
                mx = (p1[0]+p2[0])/2
                my = (p1[1]+p2[1])/2
                mz = (p1[2]+p2[2])/2
                self._disegna_testo(mx, my, mz+0.18, f"A{aid}",
                                    0.62, 0.62, 0.65)

        # ---- Nodi ----
        glColor3f(NODE_R, NODE_G, NODE_B)
        glPointSize(9.0)
        glBegin(GL_POINTS)
        for p in self.nodi.values():
            glVertex3f(*p)
        glEnd()
        # alone più morbido
        glColor4f(NODE_R, NODE_G, NODE_B, 0.20)
        glPointSize(18.0)
        glBegin(GL_POINTS)
        for p in self.nodi.values():
            glVertex3f(*p)
        glEnd()

        for nid, p in self.nodi.items():
            self._disegna_testo(p[0]+0.15, p[1]+0.15, p[2]+0.18,
                                f"N{nid}", NODE_R, NODE_G, NODE_B)

        # ---- Vincoli ----
        glColor3f(CONSTR_R, CONSTR_G, CONSTR_B)
        glPointSize(14.0)
        glBegin(GL_POINTS)
        for v in self.vincoli.values():
            p = self.nodi.get(v['nodo'])
            if p: glVertex3f(*p)
        glEnd()
        glColor4f(CONSTR_R, CONSTR_G, CONSTR_B, 0.18)
        glPointSize(26.0)
        glBegin(GL_POINTS)
        for v in self.vincoli.values():
            p = self.nodi.get(v['nodo'])
            if p: glVertex3f(*p)
        glEnd()
        for vid, v in self.vincoli.items():
            p = self.nodi.get(v['nodo'])
            if p:
                self._disegna_testo(p[0]-0.45, p[1], p[2]-0.42,
                                    f"V{vid}", CONSTR_R, CONSTR_G, CONSTR_B)

        # ---- Carichi nodali (frecce rosse) ----
        glColor3f(LOAD_R, LOAD_G, LOAD_B)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        for c in self.carichi_nodi.values():
            p = self.nodi.get(c['nodo'])
            if p is None: continue
            f  = np.array(c['forze'][:3])
            fn = np.linalg.norm(f)
            if fn > 1e-8:
                d    = f / fn
                p_np = np.array(p, dtype=float)
                glVertex3f(*p_np)
                glVertex3f(*(p_np - d * 1.5))
        glEnd()
        for cid, c in self.carichi_nodi.items():
            p = self.nodi.get(c['nodo'])
            if p is None: continue
            f  = np.array(c['forze'][:3])
            fn = np.linalg.norm(f)
            if fn > 1e-8:
                d  = f / fn
                pt = np.array(p, dtype=float) - d * 1.8
                self._disegna_testo(*pt, f"C{cid}", LOAD_R, LOAD_G, LOAD_B)

        # ---- Carichi distribuiti (frecce arancioni) ----
        glColor3f(DIST_R, DIST_G, DIST_B)
        glLineWidth(1.8)
        glBegin(GL_LINES)
        for c in self.carichi_aste.values():
            asta = self.aste.get(c['asta'])
            if not asta: continue
            p1 = self.nodi.get(asta['n1'])
            p2 = self.nodi.get(asta['n2'])
            if p1 is None or p2 is None: continue
            q  = np.array(c['q'])
            qn = np.linalg.norm(q)
            if qn < 1e-8: continue
            dq = q / qn
            for k in range(9):
                t  = k / 8
                px = p1[0] + (p2[0]-p1[0])*t
                py = p1[1] + (p2[1]-p1[1])*t
                pz = p1[2] + (p2[2]-p1[2])*t
                glVertex3f(px, py, pz)
                glVertex3f(px - dq[0], py - dq[1], pz - dq[2])
        glEnd()

    # ------------------------------------------------------------------
    # Vista DEFORMATA
    # ------------------------------------------------------------------

    def _disegna_deformata(self):
        if not self.t_nodi:
            return

        # Indeformata fantasma
        glColor4f(0.55, 0.55, 0.55, 0.20)
        glLineWidth(1.2)
        glBegin(GL_LINES)
        for a in self.t_aste:
            glVertex3f(*self.t_nodi[a[0]])
            glVertex3f(*self.t_nodi[a[1]])
        glEnd()

        if self.u_vec is None:
            return

        nodi_def = [
            self.t_nodi[i] + self.u_vec[i*6: i*6+3] * self.scala_risultati
            for i in range(len(self.t_nodi))
        ]

        glColor3f(DEF_R, DEF_G, DEF_B)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        for a in self.t_aste:
            glVertex3f(*nodi_def[a[0]])
            glVertex3f(*nodi_def[a[1]])
        glEnd()

        # Nodi deformati
        glColor3f(1.0, 0.35, 0.35)
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for nd in nodi_def:
            glVertex3f(*nd)
        glEnd()
        glColor4f(1.0, 0.35, 0.35, 0.18)
        glPointSize(12.0)
        glBegin(GL_POINTS)
        for nd in nodi_def:
            glVertex3f(*nd)
        glEnd()

        for i, pos in enumerate(nodi_def):
            u = self.u_vec[i*6: i*6+3] * 1000.0   # → mm
            if np.linalg.norm(u) > 0.05:
                self._disegna_testo(
                    pos[0], pos[1], pos[2]+0.28,
                    f"[{u[0]:.1f}, {u[1]:.1f}, {u[2]:.1f}] mm",
                    1.0, 0.72, 0.15
                )

    # ------------------------------------------------------------------
    # Vista DIAGRAMMI SOLLECITAZIONI
    # ------------------------------------------------------------------

    def _disegna_diagrammi(self):
        if not self.t_nodi:
            return

        # Aste di sfondo
        glColor4f(0.60, 0.60, 0.60, 0.40)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        for a in self.t_aste:
            glVertex3f(*self.t_nodi[a[0]])
            glVertex3f(*self.t_nodi[a[1]])
        glEnd()

        if self.sforzi is None:
            return

        idx_map = {
            "N":  (0,  6,  "kN"),
            "Vy": (1,  7,  "kN"),
            "Vz": (2,  8,  "kN"),
            "My": (4,  10, "kNm"),
            "Mz": (5,  11, "kNm"),
        }
        idx_v1, idx_v2, unit = idx_map.get(self.view_mode, (0, 6, "kN"))

        posizioni_etichettate: set = set()

        for i, a in enumerate(self.t_aste):
            n1 = self.t_nodi[a[0]]
            n2 = self.t_nodi[a[1]]
            dx = n2 - n1
            L  = float(np.linalg.norm(dx))
            if L < 1e-10:
                continue

            e_x = dx / L
            D   = float(np.sqrt(e_x[0]**2 + e_x[1]**2))

            if D < 1e-10:
                e_y = np.array([0.0, 1.0, 0.0])
                e_z = np.array([-e_x[2], 0.0, 0.0])
            else:
                ref = np.array([0, 0, 1]) if D > 0.1 else np.array([0, 1, 0])
                e_z = np.cross(e_x, ref); e_z /= np.linalg.norm(e_z)
                e_y = np.cross(e_z, e_x); e_y /= np.linalg.norm(e_y)

            dir_vec = e_y if self.view_mode in ("N", "Vy", "Mz") else e_z

            v1_real = -float(self.sforzi[i, idx_v1])
            v2_real =  float(self.sforzi[i, idx_v2])

            p1_ext = n1 + dir_vec * v1_real * self.scala_risultati
            p2_ext = n2 + dir_vec * v2_real * self.scala_risultati

            # Riempimento semitrasparente
            glColor4f(DIAG_FILL_R, DIAG_FILL_G, DIAG_FILL_B, 0.30)
            glBegin(GL_QUADS)
            glVertex3f(*n1); glVertex3f(*p1_ext)
            glVertex3f(*p2_ext); glVertex3f(*n2)
            glEnd()

            # Contorno del diagramma
            glColor3f(DIAG_LINE_R, DIAG_LINE_G, DIAG_LINE_B)
            glLineWidth(1.8)
            glBegin(GL_LINE_STRIP)
            glVertex3f(*n1)
            glVertex3f(*p1_ext)
            glVertex3f(*p2_ext)
            glVertex3f(*n2)
            glEnd()

            # Etichetta picco
            v1_kn = v1_real / 1000.0
            v2_kn = v2_real / 1000.0
            if abs(v1_kn) >= abs(v2_kn):
                v_max, p_max = v1_real, p1_ext
            else:
                v_max, p_max = v2_real, p2_ext

            if abs(v_max / 1000.0) > 0.01:
                k = (round(p_max[0], 2), round(p_max[1], 2), round(p_max[2], 2))
                if k not in posizioni_etichettate:
                    off = dir_vec * (0.14 if v_max >= 0 else -0.14)
                    self._disegna_testo(
                        p_max[0]+off[0], p_max[1]+off[1], p_max[2]+off[2],
                        f"{v_max/1000.0:.2f} {unit}",
                        DIAG_LINE_R, DIAG_LINE_G, DIAG_LINE_B
                    )
                    posizioni_etichettate.add(k)

    # ------------------------------------------------------------------
    # Gestione mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        self._last_pos  = event.pos()
        self._mouse_btn = event.button()

    def mouseMoveEvent(self, event):
        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()
        if self._mouse_btn == Qt.LeftButton:
            self.rot_y += dx * 0.40
            self.rot_x += dy * 0.40
        elif self._mouse_btn == Qt.RightButton:
            speed = self.cam_dist * 0.0018
            self.pan_x += dx * speed
            self.pan_y -= dy * speed
        self._last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        delta  = event.angleDelta().y()
        factor = 0.88 if delta > 0 else 1.12
        self.cam_dist = max(0.5, self.cam_dist * factor)
        self.update()