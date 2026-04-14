"""
grafico_materiali.py
--------------------
Widget OpenGL per la visualizzazione dei legami costitutivi dei materiali.

Fix rispetto alla versione precedente:
  • _to_screen ora è un metodo d'istanza e tiene conto di pan_x/pan_y →
    assi X/Y e tacche si muovono insieme alla curva
  • Linee tratteggiate verticali (guides) agli estremi di ogni segmento,
    come in disegno_diagrammi.py
  • reset_vista() non richiama più set_segmenti: il pan/zoom si resetta
    senza ricalcolare le curve

Interazione:
  pan  : click sinistro + trascina
  zoom : rotellina
"""

import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui     import QPainter, QColor, QFont
from PyQt5.QtCore    import Qt, QPoint

from OpenGL.GL  import *
from OpenGL.GLU import *

_N_PUNTI = 400


class GraficoMateriali(QOpenGLWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._segmenti: list[dict] = []
        self._curve:    list       = []          # lista (x_arr, y_arr)
        self._colore:   tuple      = (1.0, 1.0, 1.0)

        self._range_x: float = 0.005
        self._range_y: float = 30.0

        self.pan_x:     float = 0.0
        self.pan_y:     float = 0.0
        self.zoom:      float = 1.0
        self._last_pos: QPoint | None = None
        self._cursor:   QPoint = QPoint(0, 0)

        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    #  API PUBBLICA
    # ------------------------------------------------------------------

    def set_segmenti(self, segmenti: list[dict],
                     colore: tuple = (1.0, 1.0, 1.0)):
        """Calcola le curve dai segmenti e aggiorna il disegno."""
        self._segmenti = segmenti
        self._colore   = colore
        self._curve.clear()

        all_x, all_y = [], []

        for seg in segmenti:
            formula = seg.get("formula", "0")
            try:
                eps_min = float(seg.get("eps_min", 0))
                eps_max = float(seg.get("eps_max", 0))
            except (TypeError, ValueError):
                continue

            if abs(eps_max - eps_min) < 1e-14:
                continue

            x = np.linspace(eps_min, eps_max, _N_PUNTI)
            try:
                y = eval(formula,
                         {"x": x, "np": np, "__builtins__": {}}, {})
                if not isinstance(y, np.ndarray):
                    y = np.full_like(x, float(y))
                y = np.where(np.isfinite(y), y, 0.0)
            except Exception as e:
                print(f"WARN  Grafico – formula non valida: {formula!r} → {e}")
                y = np.zeros_like(x)

            self._curve.append((x, y))
            all_x.append(x)
            all_y.append(y)

        # Calcola range dal contenuto
        if all_x:
            ax = np.concatenate(all_x)
            ay = np.concatenate(all_y)
            mx = max(abs(float(ax.min())), abs(float(ax.max())))
            my = max(abs(float(ay.min())), abs(float(ay.max())))
            self._range_x = mx * 1.35 if mx > 0 else 0.005
            self._range_y = my * 1.35 if my > 0 else 30.0
        else:
            self._range_x = 0.005
            self._range_y = 30.0

        # Reset vista senza ricalcolare
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom  = 1.0
        self.update()

    def reset_vista(self):
        """Resetta solo pan e zoom, NON ricalcola le curve."""
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom  = 1.0
        self.update()

    # ------------------------------------------------------------------
    #  OPENGL LIFECYCLE
    # ------------------------------------------------------------------

    def initializeGL(self):
        glClearColor(40/255, 40/255, 40/255, 1.0)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        rx = self._range_x * self.zoom
        ry = self._range_y * self.zoom

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-rx + self.pan_x,  rx + self.pan_x,
                -ry + self.pan_y,  ry + self.pan_y,
                -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self._disegna_griglia(rx, ry)
        self._disegna_guide()                       # tratteggiato verticale
        self._disegna_curve()
        # Layer QPainter (disegnato sopra OpenGL, coordinate schermo)
        self._disegna_assi_schermo()
        self._disegna_etichette()
        self._disegna_tracker()

    # ------------------------------------------------------------------
    #  GRIGLIA  (OpenGL)
    # ------------------------------------------------------------------

    def _disegna_griglia(self, rx: float, ry: float):
        wx_min = -rx + self.pan_x;  wx_max = rx + self.pan_x
        wy_min = -ry + self.pan_y;  wy_max = ry + self.pan_y
        tx = self._tick(wx_max - wx_min)
        ty = self._tick(wy_max - wy_min)

        glColor3f(0.17, 0.17, 0.17)
        glLineWidth(1.0)

        xv = np.floor(wx_min / tx) * tx
        while xv <= wx_max + tx * 0.01:
            glBegin(GL_LINES)
            glVertex2f(float(xv), wy_min)
            glVertex2f(float(xv), wy_max)
            glEnd()
            xv += tx

        yv = np.floor(wy_min / ty) * ty
        while yv <= wy_max + ty * 0.01:
            glBegin(GL_LINES)
            glVertex2f(wx_min, float(yv))
            glVertex2f(wx_max, float(yv))
            glEnd()
            yv += ty

    # ------------------------------------------------------------------
    #  GUIDE TRATTEGGIATE  (OpenGL)
    #  Linee verticali tratteggiate agli estremi di ogni segmento,
    #  dal punto della curva fino all'asse X (come in disegno_diagrammi.py)
    # ------------------------------------------------------------------

    def _disegna_guide(self):
        if not self._segmenti:
            return

        glEnable(GL_LINE_STIPPLE)
        glLineStipple(8, 0xAAAA)
        glColor3f(0.55, 0.55, 0.55)
        glLineWidth(1.0)

        for seg in self._segmenti:
            formula = seg.get("formula", "0")
            for xv in (seg.get("eps_min", 0), seg.get("eps_max", 0)):
                try:
                    xv = float(xv)
                except (TypeError, ValueError):
                    continue
                if abs(xv) < 1e-10:
                    continue
                try:
                    y0 = float(eval(formula,
                                    {"x": np.float64(xv), "np": np,
                                     "__builtins__": {}}, {}))
                    if not np.isfinite(y0):
                        continue
                except Exception:
                    continue
                glBegin(GL_LINES)
                glVertex2f(xv, y0)
                glVertex2f(xv, 0.0)
                glEnd()

        glDisable(GL_LINE_STIPPLE)

    # ------------------------------------------------------------------
    #  CURVE  (OpenGL)
    # ------------------------------------------------------------------

    def _disegna_curve(self):
        r, g, b = self._colore
        glColor3f(r, g, b)
        glLineWidth(2.2)
        for xs, ys in self._curve:
            glBegin(GL_LINE_STRIP)
            for xi, yi in zip(xs, ys):
                glVertex2f(float(xi), float(yi))
            glEnd()

    # ------------------------------------------------------------------
    #  ASSI SCHERMO  (QPainter)
    #  Passano per l'origine (0,0) nel sistema mondo → si muovono con pan
    # ------------------------------------------------------------------

    def _disegna_assi_schermo(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        ox, oy = self._to_screen(0.0, 0.0)

        pen = painter.pen()
        pen.setWidth(1)
        pen.setColor(QColor(200, 55, 55))
        painter.setPen(pen)
        painter.drawLine(0, oy, self.width(), oy)

        pen.setColor(QColor(55, 175, 75))
        painter.setPen(pen)
        painter.drawLine(ox, 0, ox, self.height())

        painter.end()

    # ------------------------------------------------------------------
    #  ETICHETTE  (QPainter)
    # ------------------------------------------------------------------

    def _disegna_etichette(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Segoe UI", 8))
        m = painter.fontMetrics()
        w, h = self.width(), self.height()

        rx = self._range_x * self.zoom
        ry = self._range_y * self.zoom
        wx_min = -rx + self.pan_x;  wx_max = rx + self.pan_x
        wy_min = -ry + self.pan_y;  wy_max = ry + self.pan_y
        tx = self._tick(wx_max - wx_min)
        ty = self._tick(wy_max - wy_min)

        ox, oy = self._to_screen(0.0, 0.0)

        # Tacche e valori asse X
        painter.setPen(QColor(200, 80, 80))
        xv = np.floor(wx_min / tx) * tx
        while xv <= wx_max + tx * 0.01:
            if abs(xv) > tx * 0.05:
                sx, _ = self._to_screen(float(xv), 0.0)
                painter.drawLine(sx, oy - 4, sx, oy + 4)
                lbl = f"{xv:.3g}"
                tw  = m.horizontalAdvance(lbl)
                painter.drawText(sx - tw // 2, oy + m.height() + 2, lbl)
            xv += tx

        # Tacche e valori asse Y
        painter.setPen(QColor(55, 200, 80))
        yv = np.floor(wy_min / ty) * ty
        while yv <= wy_max + ty * 0.01:
            if abs(yv) > ty * 0.05:
                _, sy = self._to_screen(0.0, float(yv))
                painter.drawLine(ox - 4, sy, ox + 4, sy)
                lbl = f"{yv:.4g}"
                tw  = m.horizontalAdvance(lbl)
                painter.drawText(ox - tw - 8, sy + m.height() // 2 - 2, lbl)
            yv += ty

        # Titoli assi (angoli fissi)
        painter.setPen(QColor(200, 80, 80))
        lbl = "Deformazione  ε"
        painter.drawText(w - m.horizontalAdvance(lbl) - 8, h - 8, lbl)

        painter.setPen(QColor(55, 200, 80))
        painter.save()
        painter.translate(14, 120)
        painter.rotate(-90)
        painter.drawText(0, 0, "Tensione  σ [MPa]")
        painter.restore()

        painter.end()

    # ------------------------------------------------------------------
    #  TRACKER  (QPainter)
    # ------------------------------------------------------------------

    def _disegna_tracker(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = self._cursor.x(), self._cursor.y()

        # Croce piccola
        pen = painter.pen()
        pen.setColor(QColor(255, 255, 255, 210))
        pen.setWidth(1)
        painter.setPen(pen)
        sz = 10
        painter.drawLine(cx - sz, cy, cx + sz, cy)
        painter.drawLine(cx, cy - sz, cx, cy + sz)

        # Linee guida scure complete
        pen.setColor(QColor(85, 85, 85, 120))
        painter.setPen(pen)
        painter.drawLine(cx, 0, cx, h)
        painter.drawLine(0, cy, w, cy)

        # Coordinate nel sistema mondo
        rx = self._range_x * self.zoom
        ry = self._range_y * self.zoom
        wx = (-rx + self.pan_x) + (cx / w) * (2 * rx) if w else 0.0
        wy = (-ry + self.pan_y) + (1 - cy / h) * (2 * ry) if h else 0.0

        pen.setColor(QColor(210, 210, 210, 185))
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(cx + 8, h - 8, f"ε: {wx:.5g}")
        painter.drawText(8, cy - 6, f"σ: {wy:.5g}")

        painter.end()

    # ------------------------------------------------------------------
    #  CONVERSIONE COORDINATE  (istanza → usa pan corrente)
    # ------------------------------------------------------------------

    def _to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """
        Converte coordinate mondo → coordinate schermo pixel.
        Tiene conto di pan_x, pan_y e zoom correnti.
        """
        rx = self._range_x * self.zoom
        ry = self._range_y * self.zoom
        w  = self.width()
        h  = self.height()
        if rx == 0 or ry == 0 or w == 0 or h == 0:
            return w // 2, h // 2
        nx = (wx - (-rx + self.pan_x)) / (2 * rx)
        ny = (wy - (-ry + self.pan_y)) / (2 * ry)
        return int(nx * w), int((1 - ny) * h)

    # ------------------------------------------------------------------
    #  TICK STEP  (griglia adattiva)
    # ------------------------------------------------------------------

    @staticmethod
    def _tick(rng: float) -> float:
        if rng <= 0:
            return 1.0
        rough = rng / 10
        mag   = 10 ** np.floor(np.log10(max(rough, 1e-300)))
        r     = rough / mag
        return (5 if r >= 5 else 2 if r >= 2 else 1) * float(mag)

    # ------------------------------------------------------------------
    #  MOUSE
    # ------------------------------------------------------------------

    def mousePressEvent(self, e):
        if e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._last_pos = e.pos()
        self._cursor = e.pos()
        self.update()

    def mouseMoveEvent(self, e):
        self._cursor = e.pos()
        if self._last_pos and self.width() and self.height():
            dx = e.x() - self._last_pos.x()
            dy = e.y() - self._last_pos.y()
            rx = self._range_x * self.zoom
            ry = self._range_y * self.zoom
            self.pan_x -= dx * (2 * rx) / self.width()
            self.pan_y += dy * (2 * ry) / self.height()
            self._last_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self._last_pos = None
        self._cursor   = e.pos()
        self.update()

    def wheelEvent(self, e):
        factor = 1.15
        self.zoom = (self.zoom / factor if e.angleDelta().y() > 0
                     else self.zoom * factor)
        self.update()