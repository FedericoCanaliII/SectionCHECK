"""
_draw_helpers.py – Funzioni condivise per disegno etichette e hint.
"""
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui  import QColor, QPen, QFont, QBrush

COL_LBL_BG  = QColor(30, 30, 32, 210)
COL_LBL_FG  = QColor(230, 230, 230)
COL_HINT    = QColor(180, 180, 180, 200)
FONT_LBL    = QFont("Consolas", 8)
FONT_HINT   = QFont("Segoe UI", 8)


def draw_label(painter, lx, ly, text, fm=None):
    """Disegna un'etichetta con sfondo e ritorna il QRectF."""
    if fm is None:
        fm = painter.fontMetrics()
    tw = fm.horizontalAdvance(text) + 10
    th = fm.height() + 6
    rect = QRectF(lx - tw / 2, ly - th / 2, tw, th)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(COL_LBL_BG))
    painter.drawRoundedRect(rect, 3, 3)
    painter.setPen(QPen(COL_LBL_FG))
    painter.drawText(rect, Qt.AlignCenter, text)
    return rect


def draw_hint(painter, widget, text):
    """Disegna hint in basso a sinistra."""
    painter.setPen(QPen(COL_HINT))
    painter.setFont(FONT_HINT)
    painter.drawText(10, widget.height() - 10, text)
