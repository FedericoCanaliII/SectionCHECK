"""
controller_lista_elementi.py – Controller per la lista degli elementi (index 5).

Gestisce:
  • I frame a scorrimento con i bottoni elemento (travi, pilastri, fondazioni, solai)
  • La creazione, selezione ed eliminazione degli elementi
  • La cattura e visualizzazione delle preview sui bottoni
  • Il pulsante laterale C/V per aprire il workspace carichi/vincoli (index 7)
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QAbstractButton, QMessageBox, QButtonGroup,
    QSizePolicy, QMenu, QLineEdit, QWidget, QHBoxLayout,
)
from PyQt5.QtCore  import Qt, pyqtSignal, QObject
from PyQt5.QtGui   import (
    QPainter, QColor, QPen, QBrush, QFont,
    QIcon, QPixmap,
)

from .modello_3d import Oggetto3D, Elemento
from .database_elementi import carica_database as _carica_db_elem


# ============================================================
#  HELPER NOMI
# ============================================================

def _prossimo_nome(nome_base: str, esistenti: set) -> str:
    """Genera nome_base.001, .002 ... non presente in esistenti."""
    n = 1
    while True:
        candidato = f"{nome_base}.{n:03d}"
        if candidato not in esistenti:
            return candidato
        n += 1


# ============================================================
#  STYLES / CONSTANTS
# ============================================================

_BTN_W, _BTN_H = 206, 160
_PREV_H = 110
_PREV_B = 180
_NAME_H = 28

_SHEET_CAT = {
    "trave": """
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(120,120,120);border-left:4px solid rgb(120,120,120);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(120,120,120);border-left:4px solid rgb(120,120,120)}
        QAbstractButton:checked{background-color:rgb(65,65,65);
            border:1px solid rgb(200,200,200);border-left:4px solid rgb(210,210,210)}""",
    "pilastro": """
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(80,110,150);border-left:4px solid rgb(80,110,150);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(80,110,150);border-left:4px solid rgb(80,110,150)}
        QAbstractButton:checked{background-color:rgb(28,40,62);
            border:1px solid rgb(100,145,200);border-left:4px solid rgb(120,165,215)}""",
    "fondazione": """
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(160,120,120);border-left:4px solid rgb(160,120,120);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(160,120,120);border-left:4px solid rgb(160,120,120)}
        QAbstractButton:checked{background-color:rgb(65,38,38);
            border:1px solid rgb(200,150,150);border-left:4px solid rgb(220,160,160)}""",
    "solaio": """
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(150,150,50);border-left:4px solid rgb(150,150,50);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(150,150,50);border-left:4px solid rgb(150,150,50)}
        QAbstractButton:checked{background-color:rgb(50,50,18);
            border:1px solid rgb(195,195,75);border-left:4px solid rgb(215,215,85)}""",
}

_FLOW_LAY_MARGIN = 8
_FLOW_LAY_SPACE  = 18

# Dimensioni pulsante laterale C/V
_CV_BTN_W, _CV_BTN_H = 122, 160
_CV_PREV_W, _CV_PREV_H = 100, 110   # area preview interna
_CV_NAME_H = 24                     # altezza riga testo "c-v"


# Stylesheet laterale – bordo spesso sul lato DESTRO (esterno della coppia)
def _cv_sheet(tipo: str) -> str:
    color_map = {
        "trave":      "rgb(120,120,120)",
        "pilastro":   "rgb(80,110,150)",
        "fondazione": "rgb(160,120,120)",
        "solaio":     "rgb(150,150,50)",
    }
    c = color_map.get(tipo, "rgb(120,120,120)")
    return (
        f"QAbstractButton{{background-color:rgb(35,35,35);"
        f"border:1px solid {c};border-right:4px solid {c};border-radius:6px}}"
        f"QAbstractButton:hover{{background-color:rgb(28,28,28);"
        f"border:1px solid {c};border-right:4px solid {c}}}"
    )


# ============================================================
#  FLOW LAYOUT
# ============================================================

class _FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, margin=8, spacing=8):
        super().__init__(parent)
        self._items, self._sp = [], spacing
        self.setContentsMargins(margin, margin, margin, margin)
    def addItem(self, i):  self._items.append(i)
    def count(self):       return len(self._items)
    def itemAt(self, i):   return self._items[i] if 0 <= i < len(self._items) else None
    def takeAt(self, i):   return self._items.pop(i) if 0 <= i < len(self._items) else None
    def expandingDirections(self): return Qt.Orientations(Qt.Orientation(0))
    def hasHeightForWidth(self): return True
    def heightForWidth(self, w): return self._do(QtCore.QRect(0,0,w,0), True)
    def setGeometry(self, r): super().setGeometry(r); self._do(r, False)
    def sizeHint(self): return self.minimumSize()
    def minimumSize(self):
        s = QtCore.QSize()
        for it in self._items: s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QtCore.QSize(m.left()+m.right(), m.top()+m.bottom())
    def _do(self, rect, test):
        m  = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, rh = eff.x(), eff.y(), 0
        for it in self._items:
            iw, ih = it.sizeHint().width(), it.sizeHint().height()
            nx = x + iw + self._sp
            if nx - self._sp > eff.right() and rh > 0:
                x, y = eff.x(), y + rh + self._sp; nx, rh = eff.x()+iw+self._sp, 0
            if not test: it.setGeometry(QtCore.QRect(QtCore.QPoint(x,y), it.sizeHint()))
            x, rh = nx, max(rh, ih)
        return y + rh - rect.y() + m.bottom()


# ============================================================
#  ELEMENTO BUTTON  (preview + name + delete X)
# ============================================================

class ElementoButton(QAbstractButton):
    deleteRequested = pyqtSignal()

    def __init__(self, nome, tipo, parent=None, standard=False):
        super().__init__(parent)
        self.nome     = nome
        self.tipo     = tipo
        self.standard = standard
        self._pixmap  = None
        self.setFixedSize(_BTN_W, _BTN_H)
        self.setCheckable(True)
        self.setStyleSheet(_SHEET_CAT.get(tipo, _SHEET_CAT["trave"]))

    def set_preview(self, pixmap: QPixmap):
        """Set the preview image captured from the 3D space."""
        self._pixmap = pixmap
        self.update()

    def _del_rect(self):
        return QtCore.QRect(_BTN_W - 20, 4, 16, 16)

    def initStyleOption(self, opt):
        opt.initFrom(self); opt.features = QtWidgets.QStyleOptionButton.None_
        opt.state |= (QtWidgets.QStyle.State_Sunken if self.isDown()
                      else QtWidgets.QStyle.State_Raised)
        if self.isChecked(): opt.state |= QtWidgets.QStyle.State_On
        opt.text = ""; opt.icon = QIcon()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        self.style().drawControl(QtWidgets.QStyle.CE_PushButton, opt, painter, self)

        # =========================================================
        # Preview area dimensionata con _PREV_B e _PREV_H
        # =========================================================
        pw = _PREV_B
        ph = _PREV_H
        
        # Centra orizzontalmente nel bottone (_BTN_W = 200, _PREV_B = 160 -> scarto diviso 2)
        ox = (_BTN_W - pw) // 2 
        oy = 6  # Piccolo margine dall'alto

        path = QtGui.QPainterPath()
        path.addRoundedRect(ox, oy, pw, ph, 5, 5)
        painter.setPen(Qt.NoPen)
        
        # Colore di sfondo a (50, 50, 50)
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.drawPath(path)

        painter.save()
        painter.setClipPath(path)
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(pw, ph, Qt.KeepAspectRatioByExpanding,
                                         Qt.SmoothTransformation)
            px_off = ox + (pw - scaled.width())  // 2
            py_off = oy + (ph - scaled.height()) // 2

            if isinstance(scaled, QtGui.QPixmap):
                painter.drawPixmap(px_off, py_off, scaled)
            elif isinstance(scaled, QtGui.QImage):
                painter.drawImage(px_off, py_off, scaled)
        # else: no preview -> lascia lo sfondo grigio (50,50,50) visibile
        painter.restore()

        # Name
        painter.setPen(QColor(210, 210, 210))
        painter.setFont(QFont("Segoe UI", 9))
        
        # Spostiamo il testo in base alla posizione effettiva e all'altezza della preview
        text_y = oy + ph + 2 
        painter.drawText(
            QtCore.QRect(4, text_y, _BTN_W - 8, _NAME_H),
            Qt.AlignCenter | Qt.TextWordWrap, self.nome
        )

        # Delete button (X) – solo per elementi non-standard
        if not self.standard:
            dr = self._del_rect()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(55, 55, 55, 220)))
            painter.drawEllipse(dr)
            pen = QPen(QColor(190, 190, 190, 230), 1.5)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            m = 4
            painter.drawLine(dr.left()+m, dr.top()+m, dr.right()-m, dr.bottom()-m)
            painter.drawLine(dr.right()-m, dr.top()+m, dr.left()+m, dr.bottom()-m)

        painter.end()

    def _draw_3d_icon(self, p, ox, oy, pw, ph):
        """Simple isometric-ish box icon."""
        cx, cy = ox + pw//2, oy + ph//2
        w2, h2 = pw//3, ph//3

        col_map = {
            "trave":      QColor(140, 148, 162),
            "pilastro":   QColor(80,  110, 165),
            "fondazione": QColor(165, 100,  90),
            "solaio":     QColor(140, 140,  60),
        }
        base_col = col_map.get(self.tipo, QColor(140, 148, 162))

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(base_col.lighter(120)))
        p.drawPolygon(QtGui.QPolygon([
            QtCore.QPoint(cx - w2, cy - 4),
            QtCore.QPoint(cx + w2, cy - 4),
            QtCore.QPoint(cx,      cy - h2 - 4),
        ]))
        p.setBrush(QBrush(base_col))
        p.drawRect(cx - w2, cy - 4, w2 * 2, h2)
        p.setBrush(QBrush(base_col.darker(115)))
        p.drawPolygon(QtGui.QPolygon([
            QtCore.QPoint(cx + w2, cy - 4),
            QtCore.QPoint(cx + w2, cy + h2 - 4),
            QtCore.QPoint(cx,      cy + h2),
            QtCore.QPoint(cx,      cy - h2 - 4),
        ]))

    def mousePressEvent(self, e):
        if (e.button() == Qt.LeftButton and not self.standard
                and self._del_rect().contains(e.pos())):
            self.deleteRequested.emit(); e.accept(); return
        super().mousePressEvent(e)

    def sizeHint(self): return QtCore.QSize(_BTN_W, _BTN_H)


# ============================================================
#  CARICHI/VINCOLI SIDE BUTTON  (50 × 160)
# ============================================================

class CariciVincoliButton(QAbstractButton):
    """
    Side button (72×160) displayed to the right of each ElementoButton.
    Opens the carichi/vincoli workspace (index 7) for the linked element.
    Shows a full-height preview pixmap of the CV objects (no text labels).
    """

    def __init__(self, tipo: str, parent=None):
        super().__init__(parent)
        self.tipo    = tipo
        self._pixmap = None
        self.setFixedSize(_CV_BTN_W, _CV_BTN_H)
        self.setCheckable(False)
        self.setStyleSheet(_cv_sheet(tipo))
        self.setToolTip("Apri workspace Carichi / Vincoli")

    def set_preview(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update()

    def initStyleOption(self, opt):
        opt.initFrom(self)
        opt.features = QtWidgets.QStyleOptionButton.None_
        opt.state |= (QtWidgets.QStyle.State_Sunken if self.isDown()
                      else QtWidgets.QStyle.State_Raised)
        opt.text = ""
        opt.icon = QIcon()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background frame (draws border via stylesheet)
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        self.style().drawControl(QtWidgets.QStyle.CE_PushButton, opt, painter, self)

        # ── Area preview 80×110 centrata orizzontalmente, margine 6 dall'alto ──
        ox = (_CV_BTN_W - _CV_PREV_W) // 2
        oy = 6

        path = QtGui.QPainterPath()
        path.addRoundedRect(ox, oy, _CV_PREV_W, _CV_PREV_H, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.drawPath(path)

        painter.save()
        painter.setClipPath(path)
        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                _CV_PREV_W, _CV_PREV_H,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            px_off = ox + (_CV_PREV_W - scaled.width())  // 2
            py_off = oy + (_CV_PREV_H - scaled.height()) // 2
            painter.drawPixmap(px_off, py_off, scaled)
        painter.restore()

        # ── Testo "c-v" sotto la preview ──
        text_y = oy + _CV_PREV_H + 2
        painter.setPen(QColor(180, 180, 180))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            QtCore.QRect(0, text_y, _CV_BTN_W, _CV_NAME_H),
            Qt.AlignCenter, "caric. e vinc.",
        )

        painter.end()

    def sizeHint(self): return QtCore.QSize(_CV_BTN_W, _CV_BTN_H)


# ============================================================
#  ELEMENTO BTN PAIR  (ElementoButton + CariciVincoliButton)
# ============================================================

class ElementoBtnPair(QWidget):
    """
    Container widget that holds an ElementoButton (200×160) and a
    CariciVincoliButton (50×160) side-by-side with 4 px spacing.
    Forwarding signals:
      • deleteRequested  – from ElementoButton
      • apri_extra       – from CariciVincoliButton (emits the Elemento)
    """

    deleteRequested = pyqtSignal()
    apri_extra      = pyqtSignal(object)   # Elemento

    def __init__(self, el: Elemento, tipo: str, parent=None, standard=False):
        super().__init__(parent)
        self._el = el

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.btn_elemento = ElementoButton(el.nome, tipo, self, standard=standard)
        self.btn_cv       = CariciVincoliButton(tipo, self)

        lay.addWidget(self.btn_elemento)
        lay.addWidget(self.btn_cv)

        self.setFixedSize(_BTN_W + 4 + _CV_BTN_W, _BTN_H)

        # Forward signals
        self.btn_elemento.deleteRequested.connect(self.deleteRequested)
        self.btn_cv.clicked.connect(lambda: self.apri_extra.emit(el))

    def set_preview(self, pixmap):
        self.btn_elemento.set_preview(pixmap)

    def set_cv_preview(self, pixmap):
        self.btn_cv.set_preview(pixmap)

    def sizeHint(self):
        return QtCore.QSize(_BTN_W + 4 + _CV_BTN_W, _BTN_H)


# ============================================================
#  CONTROLLER LISTA ELEMENTI
# ============================================================

class ControllerListaElementi(QObject):
    """Gestisce i frame con i bottoni elemento (index 5)."""

    elemento_selezionato = pyqtSignal(object)   # Elemento
    richiedi_salvataggio = pyqtSignal()
    apri_extra_elemento  = pyqtSignal(object)   # Elemento – apre index 7 (C/V)

    def __init__(self, ui, main_window):
        super().__init__(main_window)
        self._ui   = ui
        self._main = main_window

        self._db_elem = _carica_db_elem()

        self._elementi: dict[str, list] = {
            "trave": [], "pilastro": [], "fondazione": [], "solaio": [],
        }
        self._elem_corrente: Elemento | None = None
        self._bottoni: dict[int, ElementoButton] = {}   # id → ElementoButton
        self._pairs:   dict[int, ElementoBtnPair] = {}  # id → ElementoBtnPair
        self._btn_group = QButtonGroup(main_window)
        self._btn_group.setExclusive(True)

        self._setup_frame_elementi()

    # ================================================================
    #  SETUP
    # ================================================================

    def _setup_frame_elementi(self):
        mappa = {
            "trave":      (self._ui.elemento_aggiungi_trave,     self._ui.elemento_frame_trave),
            "pilastro":   (self._ui.elemento_aggiungi_pilastro,   self._ui.elemento_frame_pilastro),
            "fondazione": (self._ui.elemento_aggiungi_fondazione, self._ui.elemento_frame_fondazione),
            "solaio":     (self._ui.elemento_aggiungi_solaio,     self._ui.elemento_frame_solaio),
        }
        for tipo, (btn_add, frame) in mappa.items():
            self._init_frame(frame)
            btn_add.clicked.connect(lambda _c, t=tipo: self._aggiungi_elemento(t))
            for d in self._db_elem.get(tipo, []):
                el = Elemento.from_dict(d)
                self._elementi[tipo].append(el)
                self._crea_bottone_elemento(el)

    @staticmethod
    def _init_frame(frame):
        lay = frame.layout()
        if lay is None:
            frame.setLayout(_FlowLayout(margin=_FLOW_LAY_MARGIN, spacing=_FLOW_LAY_SPACE))
        else:
            while lay.count():
                it = lay.takeAt(0)
                w  = it.widget()
                if w:
                    w.setParent(None); w.deleteLater()

    # ================================================================
    #  ELEMENT MANAGEMENT
    # ================================================================

    def _aggiungi_elemento(self, tipo: str):
        if not self._main.ha_progetto():
            QMessageBox.warning(self._main, "Attenzione", "Crea o apri prima un progetto.")
            return

        el = Elemento(tipo)
        self._elementi[tipo].append(el)
        self._crea_bottone_elemento(el)
        self.richiedi_salvataggio.emit()
        print(f">> Elemento aggiunto: [{tipo}] {el.nome}")

    def _crea_bottone_elemento(self, el: Elemento):
        frame_map = {
            "trave":      self._ui.elemento_frame_trave,
            "pilastro":   self._ui.elemento_frame_pilastro,
            "fondazione": self._ui.elemento_frame_fondazione,
            "solaio":     self._ui.elemento_frame_solaio,
        }
        frame = frame_map.get(el.tipo)
        if frame is None:
            return

        std  = getattr(el, "standard", False)
        pair = ElementoBtnPair(el, el.tipo, frame, standard=std)
        btn  = pair.btn_elemento

        frame.layout().addWidget(pair)
        self._btn_group.addButton(btn)
        self._bottoni[el.id] = btn
        self._pairs[el.id]   = pair

        btn.clicked.connect(lambda _c, e=el: self._seleziona_elemento(e))
        if not std:
            pair.deleteRequested.connect(lambda e=el: self._elimina_elemento(e))
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        btn.customContextMenuRequested.connect(
            lambda pos, b=btn, e=el: self._ctx_elemento(b, e, pos))
        pair.apri_extra.connect(self.apri_extra_elemento)

    def _seleziona_elemento(self, el: Elemento):
        self._elem_corrente = el
        self.elemento_selezionato.emit(el)
        print(f">> Elemento selezionato: {el.nome}")

    def _elimina_elemento(self, el: Elemento):
        if getattr(el, "standard", False):
            return
        r = QMessageBox.question(
            self._main, "Elimina elemento",
            f"Eliminare «{el.nome}»?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if r != QMessageBox.Yes:
            return
        self._elementi[el.tipo] = [e for e in self._elementi[el.tipo] if e.id != el.id]
        btn  = self._bottoni.pop(el.id, None)
        pair = self._pairs.pop(el.id, None)
        if btn:
            self._btn_group.removeButton(btn)
        if pair:
            pair.setParent(None); pair.deleteLater()
        if self._elem_corrente is el:
            self._elem_corrente = None
            self.elemento_selezionato.emit(None)
        self.richiedi_salvataggio.emit()
        print(f">> Elemento eliminato: {el.nome}")

    # ================================================================
    #  PREVIEW
    # ================================================================

    def aggiorna_preview(self, el_id: int, pixmap):
        """Imposta la preview sul bottone dell'elemento indicato."""
        btn = self._bottoni.get(el_id)
        if btn and not pixmap.isNull():
            btn.set_preview(pixmap)

    def aggiorna_cv_preview(self, el_id: int, pixmap):
        """Imposta la preview sul bottone laterale C/V dell'elemento indicato."""
        pair = self._pairs.get(el_id)
        if pair and pixmap and not pixmap.isNull():
            pair.set_cv_preview(pixmap)

    # ================================================================
    #  PUBLIC ACCESSORS
    # ================================================================

    def get_elementi(self) -> dict:
        return self._elementi

    def get_elem_corrente(self) -> Elemento | None:
        return self._elem_corrente

    def get_bottoni(self) -> dict:
        return self._bottoni

    # ================================================================
    #  RINOMINA INLINE  (doppio click o voce menu)
    # ================================================================

    def _rinomina_inline(self, btn: ElementoButton, el: Elemento):
        """Apre un QLineEdit inline sopra il testo del bottone."""
        if getattr(el, "standard", False):
            return
        vec = el.nome
        le = QLineEdit(btn)
        text_y = 6 + _PREV_H + 2   # = 118
        le.setGeometry(4, text_y + 3, _BTN_W - 8, 22)
        le.setStyleSheet(
            "background:rgb(30,30,30);color:rgb(220,220,220);"
            "border:1px solid rgb(150,150,150);border-radius:3px;"
            "font:9pt 'Segoe UI';"
        )
        le.setText(vec)
        le.show(); le.raise_(); le.setFocus(); le.selectAll()

        _done = [False]

        def commit():
            if _done[0]: return
            _done[0] = True
            nuovo = le.text().strip()
            le.hide(); le.deleteLater()
            if nuovo and nuovo != vec:
                self._applica_rinomina_elem(btn, el, vec, nuovo)

        le.returnPressed.connect(commit)
        le.editingFinished.connect(commit)

    def _applica_rinomina_elem(self, btn: ElementoButton, el: Elemento,
                               vec: str, nuovo: str):
        tutti_nomi = {e.nome for lista in self._elementi.values() for e in lista
                      if e.id != el.id}
        if nuovo in tutti_nomi:
            return   # nome già in uso: ignora silenziosamente
        el.nome  = nuovo
        btn.nome = nuovo
        btn.update()
        self.richiedi_salvataggio.emit()
        print(f">> Elemento rinominato: {vec} → {nuovo}")

    # ================================================================
    #  CONTEXT MENU TASTO DESTRO
    # ================================================================

    def _ctx_elemento(self, btn: ElementoButton, el: Elemento, pos):
        menu = QMenu(btn)
        menu.setStyleSheet("QMenu{background:#252525;color:#ddd;border:1px solid #555}"
                           "QMenu::item:selected{background:#3a5080}")
        act_dup = menu.addAction("Duplica")
        act_ren = act_del = None
        if not getattr(el, "standard", False):
            act_ren = menu.addAction("Rinomina")
            act_del = menu.addAction("Elimina")
        action = menu.exec_(btn.mapToGlobal(pos))
        if action == act_dup:
            self._duplica_elemento(el)
        elif act_ren and action == act_ren:
            self._rinomina_inline(btn, el)
        elif act_del and action == act_del:
            self._elimina_elemento(el)

    # ================================================================
    #  DUPLICA ELEMENTO
    # ================================================================

    def _duplica_elemento(self, el: Elemento):
        tutti_nomi = {e.nome for lista in self._elementi.values() for e in lista}
        nuovo_nome = _prossimo_nome(el.nome, tutti_nomi)

        el_copy = Elemento.__new__(Elemento)
        Elemento._id_counter += 1
        el_copy.id       = Elemento._id_counter
        el_copy.tipo     = el.tipo
        el_copy.nome     = nuovo_nome
        el_copy.standard = False   # le copie sono sempre custom
        el_copy.oggetti  = [o.duplica() for o in el.oggetti]

        self._elementi[el.tipo].append(el_copy)
        self._crea_bottone_elemento(el_copy)
        self.richiedi_salvataggio.emit()
        print(f">> Elemento duplicato: {el.nome} → {nuovo_nome}")

    # ================================================================
    #  RELOAD FROM PROJECT
    # ================================================================

    def ricarica_da_progetto(self, sezione: dict):
        for btn in list(self._bottoni.values()):
            self._btn_group.removeButton(btn)
        self._bottoni.clear()
        self._pairs.clear()   # pairs destroyed by _init_frame below

        frames = {
            "trave":      self._ui.elemento_frame_trave,
            "pilastro":   self._ui.elemento_frame_pilastro,
            "fondazione": self._ui.elemento_frame_fondazione,
            "solaio":     self._ui.elemento_frame_solaio,
        }
        for tipo, frame in frames.items():
            self._init_frame(frame)
            self._elementi[tipo] = []
            # sempre re-inserisce gli standard dal database
            for d in self._db_elem.get(tipo, []):
                el = Elemento.from_dict(d)
                self._elementi[tipo].append(el)
                self._crea_bottone_elemento(el)
            # poi aggiunge gli elementi custom salvati nel progetto
            for d in sezione.get(tipo, []):
                if not d.get("standard", False):
                    el = Elemento.from_dict(d)
                    self._elementi[tipo].append(el)
                    self._crea_bottone_elemento(el)

        self._elem_corrente = None
