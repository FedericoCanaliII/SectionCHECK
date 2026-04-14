"""
gestione_struttura.py – Controller principale del modulo Struttura.

Gestisce:
  • Lista strutture con bottoni (index 8) – calcestruzzo, acciaio, personalizzate
  • Dettaglio struttura (index 9) – editor testo + spazio 3D OpenGL
  • Preview sui bottoni lista
  • Toggle layout (testo / split con 3D)
  • Bottoni vista 3D/X/Y/Z + centra
  • Info dialog
  • Integrazione con materiali e sezioni del progetto
"""

import copy
import math
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QAbstractButton, QButtonGroup, QMenu, QLineEdit,
    QMessageBox, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QIcon, QPixmap

from .database_struttura import carica_database, nuovo_struttura_vuota
from .testo_struttura import TextoStrutturaManager, parse_struttura
from .struttura_spazio_3d import StrutturaSpazio3D
from .struttura_info import StrutturaInfoDialog


# ================================================================
#  HELPERS
# ================================================================

def _prossimo_nome(nome_base: str, esistenti: set) -> str:
    n = 1
    while True:
        candidato = f"{nome_base}.{n:03d}"
        if candidato not in esistenti:
            return candidato
        n += 1

_NOME_BASE = {
    "calcestruzzo":   "struttura_rc",
    "acciaio":        "struttura_acc",
    "personalizzate": "struttura",
}


# ================================================================
#  FLOW LAYOUT
# ================================================================

class _FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, margin=8, spacing=12):
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


# ================================================================
#  STRUTTURA BUTTON
# ================================================================

_BTN_W, _BTN_H = 252, 180
_PREV_W, _PREV_H = 220, 140
_NAME_H = 28

_SHEET_CAT = {
    "calcestruzzo": """
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(120,120,120);border-left:4px solid rgb(120,120,120);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(120,120,120);border-left:4px solid rgb(120,120,120)}
        QAbstractButton:checked{background-color:rgb(65,65,65);
            border:1px solid rgb(200,200,200);border-left:4px solid rgb(210,210,210)}""",
    "acciaio": """
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(80,110,150);border-left:4px solid rgb(80,110,150);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(80,110,150);border-left:4px solid rgb(80,110,150)}
        QAbstractButton:checked{background-color:rgb(28,40,62);
            border:1px solid rgb(100,145,200);border-left:4px solid rgb(120,165,215)}""",
    "personalizzate": """
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(150,130,80);border-left:4px solid rgb(150,130,80);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(150,130,80);border-left:4px solid rgb(150,130,80)}
        QAbstractButton:checked{background-color:rgb(50,45,22);
            border:1px solid rgb(195,175,100);border-left:4px solid rgb(210,190,110)}""",
}


class StrutturaButton(QAbstractButton):
    deleteRequested = pyqtSignal()

    def __init__(self, nome, cat, parent=None, standard=False):
        super().__init__(parent)
        self.nome = nome
        self.cat = cat
        self.standard = standard
        self._pixmap = None
        self.setFixedSize(_BTN_W, _BTN_H)
        self.setCheckable(True)
        self.setStyleSheet(_SHEET_CAT.get(cat, _SHEET_CAT["calcestruzzo"]))

    def set_preview(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def _del_rect(self):
        return QtCore.QRect(_BTN_W - 20, 4, 16, 16)

    def initStyleOption(self, opt):
        opt.initFrom(self)
        opt.features = QtWidgets.QStyleOptionButton.None_
        opt.state |= (QtWidgets.QStyle.State_Sunken if self.isDown()
                      else QtWidgets.QStyle.State_Raised)
        if self.isChecked():
            opt.state |= QtWidgets.QStyle.State_On
        opt.text = ""
        opt.icon = QIcon()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        self.style().drawControl(QtWidgets.QStyle.CE_PushButton, opt, painter, self)

        # Preview area
        pw, ph = _PREV_W, _PREV_H
        ox = (_BTN_W - pw) // 2
        oy = 6

        path = QtGui.QPainterPath()
        path.addRoundedRect(ox, oy, pw, ph, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.drawPath(path)

        painter.save()
        painter.setClipPath(path)
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(pw, ph, Qt.KeepAspectRatioByExpanding,
                                         Qt.SmoothTransformation)
            px_off = ox + (pw - scaled.width()) // 2
            py_off = oy + (ph - scaled.height()) // 2
            if isinstance(scaled, QtGui.QPixmap):
                painter.drawPixmap(px_off, py_off, scaled)
            elif isinstance(scaled, QtGui.QImage):
                painter.drawImage(px_off, py_off, scaled)
        painter.restore()

        # Nome
        painter.setPen(QColor(210, 210, 210))
        painter.setFont(QFont("Segoe UI", 9))
        text_y = oy + ph + 2
        painter.drawText(
            QtCore.QRect(4, text_y, _BTN_W - 8, _NAME_H),
            Qt.AlignCenter | Qt.TextWordWrap, self.nome,
        )

        # Delete X
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

    def mousePressEvent(self, e):
        if (e.button() == Qt.LeftButton and not self.standard
                and self._del_rect().contains(e.pos())):
            self.deleteRequested.emit()
            e.accept()
            return
        super().mousePressEvent(e)

    def sizeHint(self):
        return QtCore.QSize(_BTN_W, _BTN_H)


# ================================================================
#  GESTIONE STRUTTURA (Controller principale)
# ================================================================

class GestioneStruttura:

    def __init__(self, ui, main_window):
        self._ui   = ui
        self._main = main_window

        QTimer.singleShot(0, self._ui.struttura_layout.click)

        self._db = carica_database()

        # Stato corrente
        self._cat_corrente:  str | None = None
        self._nome_corrente: str | None = None

        # Bottoni lista
        self._bottoni: dict[tuple, StrutturaButton] = {}   # (cat, nome) → btn
        self._btn_group = QButtonGroup(main_window)
        self._btn_group.setExclusive(True)

        # Spazio 3D
        self._spazio = StrutturaSpazio3D()
        self._installa_spazio_3d()

        # Testo manager
        self._testo = TextoStrutturaManager(ui, main_window)

        # Info dialog (lazy)
        self._info_dialog = None

        # Layout toggle state
        self._layout_espanso = True   # True = 1200px, False = 550px

        # Setup
        self._setup_frame_strutture()
        self._setup_connessioni()
        self._setup_viste()

        # Preview batch iniziale
        QTimer.singleShot(800, self._schedula_previews_tutti)

    # ================================================================
    #  SETUP
    # ================================================================

    def _installa_spazio_3d(self):
        """Installa il widget OpenGL dentro struttura_widget."""
        container = self._ui.struttura_widget
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(3,3,3,3)
        lay.addWidget(self._spazio)

    def _setup_frame_strutture(self):
        """Popola i frame della lista (index 8) con i bottoni dal database."""
        mappa = {
            "calcestruzzo":   (self._ui.struttura_aggiungi_calcestruzzo,
                               self._ui.struttura_frame_calcestruzzo),
            "acciaio":        (self._ui.struttura_aggiungi_acciaio,
                               self._ui.struttura_frame_acciaio),
            "personalizzate": (self._ui.struttura_aggiungi_personalizzate,
                               self._ui.struttura_frame_personalizzate),
        }
        for cat, (btn_add, frame) in mappa.items():
            self._init_frame(frame)
            btn_add.clicked.connect(lambda _c, c=cat: self._aggiungi(c))
            for nome, dati in self._db.get(cat, {}).items():
                std = dati.get("standard", False)
                self._crea_bottone(frame, cat, nome, standard=std)

    @staticmethod
    def _init_frame(frame):
        lay = frame.layout()
        if lay is None:
            frame.setLayout(_FlowLayout())
        else:
            while lay.count():
                it = lay.takeAt(0)
                w = it.widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()

    def _setup_connessioni(self):
        self._ui.struttura_aggiorna.clicked.connect(self._aggiorna_3d)
        self._ui.struttura_info.clicked.connect(self._mostra_info)
        self._ui.struttura_centra.clicked.connect(self._centra_vista)
        self._ui.struttura_layout.clicked.connect(self._toggle_layout)

    def _setup_viste(self):
        """Rende i bottoni vista checkable ed esclusivi."""
        btns = [
            self._ui.struttura_btn_vista_3d,
            self._ui.struttura_btn_vista_x,
            self._ui.struttura_btn_vista_y,
            self._ui.struttura_btn_vista_z,
        ]
        self._vista_group = QButtonGroup(self._main)
        self._vista_group.setExclusive(True)
        for btn in btns:
            btn.setCheckable(True)
            self._vista_group.addButton(btn)

        self._ui.struttura_btn_vista_3d.setChecked(True)

        self._ui.struttura_btn_vista_3d.clicked.connect(lambda: self._spazio.imposta_vista("3d"))
        self._ui.struttura_btn_vista_x.clicked.connect(lambda: self._spazio.imposta_vista("x"))
        self._ui.struttura_btn_vista_y.clicked.connect(lambda: self._spazio.imposta_vista("y"))
        self._ui.struttura_btn_vista_z.clicked.connect(lambda: self._spazio.imposta_vista("z"))

    # ================================================================
    #  BOTTONI LISTA (index 8)
    # ================================================================

    def _crea_bottone(self, frame, cat, nome, standard=False):
        btn = StrutturaButton(nome, cat, frame, standard=standard)
        frame.layout().addWidget(btn)
        self._btn_group.addButton(btn)
        self._bottoni[(cat, nome)] = btn

        btn.clicked.connect(lambda _c, c=cat, n=nome: self._on_bottone_cliccato(c, n))
        if not standard:
            btn.deleteRequested.connect(lambda c=cat, n=nome: self._elimina(c, n))
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        btn.customContextMenuRequested.connect(
            lambda pos, b=btn, c=cat, n=nome: self._ctx_menu(b, c, n, pos))

    def _on_bottone_cliccato(self, cat, nome):
        """Apre la struttura selezionata nell'editor (index 9)."""
        # Salva eventuale struttura precedente
        self._salva_corrente()

        self._cat_corrente = cat
        self._nome_corrente = nome

        # Carica il testo
        dati = self._leggi(cat, nome)
        testo = dati.get("testo", "") if dati else ""
        self._testo.set_testo(testo)

        # Aggiorna label
        self._ui.struttura_label.setText(f"Struttura: {nome}")

        # Naviga a index 9
        self._ui.stackedWidget_main.setCurrentIndex(9)

        # Aggiorna 3D
        self._aggiorna_3d()
        print(f">> Struttura selezionata: [{cat}] {nome}")

    def _aggiungi(self, cat):
        if not self._main.ha_progetto():
            QMessageBox.warning(self._main, "Attenzione",
                                "Crea o apri prima un progetto.")
            return

        dati_sez = self._main.get_sezione("strutture")
        cat_dict = dati_sez.setdefault(cat, {})
        esistenti = set(cat_dict.keys()) | set(self._db.get(cat, {}).keys())
        nome = _prossimo_nome(_NOME_BASE.get(cat, "struttura"), esistenti)

        cat_dict[nome] = nuovo_struttura_vuota(cat)
        self._main.push_undo("Aggiungi struttura", "lista_strutture")
        self._main.set_sezione("strutture", dati_sez)

        frame_map = {
            "calcestruzzo":   self._ui.struttura_frame_calcestruzzo,
            "acciaio":        self._ui.struttura_frame_acciaio,
            "personalizzate": self._ui.struttura_frame_personalizzate,
        }
        self._crea_bottone(frame_map[cat], cat, nome, standard=False)
        print(f">> Struttura aggiunta: [{cat}] {nome}")

    def _elimina(self, cat, nome):
        dati = self._leggi(cat, nome)
        if dati and dati.get("standard", False):
            return

        r = QMessageBox.question(
            self._main, "Elimina struttura",
            f"Eliminare «{nome}»?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if r != QMessageBox.Yes:
            return

        dati_sez = self._main.get_sezione("strutture")
        if cat in dati_sez and nome in dati_sez[cat]:
            del dati_sez[cat][nome]
            self._main.push_undo("Elimina struttura", "lista_strutture")
            self._main.set_sezione("strutture", dati_sez)

        btn = self._bottoni.pop((cat, nome), None)
        if btn:
            self._btn_group.removeButton(btn)
            btn.setParent(None)
            btn.deleteLater()

        if self._cat_corrente == cat and self._nome_corrente == nome:
            self._cat_corrente = None
            self._nome_corrente = None
            self._ui.struttura_label.setText("Struttura: -")

        print(f">> Struttura eliminata: [{cat}] {nome}")

    # ================================================================
    #  CONTEXT MENU
    # ================================================================

    def _ctx_menu(self, btn, cat, nome, pos):
        menu = QMenu(btn)
        menu.setStyleSheet("QMenu{background:#252525;color:#ddd;border:1px solid #555}"
                           "QMenu::item:selected{background:#3a5080}")
        act_dup = menu.addAction("Duplica")
        act_ren = act_del = None
        dati = self._leggi(cat, nome)
        is_std = dati.get("standard", False) if dati else False
        if not is_std:
            act_ren = menu.addAction("Rinomina")
            act_del = menu.addAction("Elimina")
        action = menu.exec_(btn.mapToGlobal(pos))
        if action == act_dup:
            self._duplica(cat, nome)
        elif act_ren and action == act_ren:
            self._rinomina_inline(btn, cat, nome)
        elif act_del and action == act_del:
            self._elimina(cat, nome)

    def _duplica(self, cat, nome):
        dati = self._leggi(cat, nome)
        if dati is None:
            return

        dati_sez = self._main.get_sezione("strutture")
        cat_dict = dati_sez.setdefault(cat, {})
        esistenti = set(cat_dict.keys()) | set(self._db.get(cat, {}).keys())
        nome_copia = _prossimo_nome(nome, esistenti)

        copia = copy.deepcopy(dati)
        copia["standard"] = False
        cat_dict[nome_copia] = copia
        self._main.push_undo("Duplica struttura", "lista_strutture")
        self._main.set_sezione("strutture", dati_sez)

        frame_map = {
            "calcestruzzo":   self._ui.struttura_frame_calcestruzzo,
            "acciaio":        self._ui.struttura_frame_acciaio,
            "personalizzate": self._ui.struttura_frame_personalizzate,
        }
        self._crea_bottone(frame_map[cat], cat, nome_copia, standard=False)
        print(f">> Struttura duplicata: {nome} → {nome_copia}")

    def _rinomina_inline(self, btn, cat, nome):
        dati = self._leggi(cat, nome)
        if dati and dati.get("standard", False):
            return

        le = QLineEdit(btn)
        text_y = 6 + _PREV_H + 2
        le.setGeometry(4, text_y + 3, _BTN_W - 8, 22)
        le.setStyleSheet(
            "background:rgb(30,30,30);color:rgb(220,220,220);"
            "border:1px solid rgb(150,150,150);border-radius:3px;"
            "font:9pt 'Segoe UI';"
        )
        le.setText(nome)
        le.show()
        le.raise_()
        le.setFocus()
        le.selectAll()

        _done = [False]

        def commit():
            if _done[0]:
                return
            _done[0] = True
            nuovo = le.text().strip()
            le.hide()
            le.deleteLater()
            if nuovo and nuovo != nome:
                self._applica_rinomina(btn, cat, nome, nuovo)

        le.returnPressed.connect(commit)
        le.editingFinished.connect(commit)

    def _applica_rinomina(self, btn, cat, vecchio, nuovo):
        # Controlla unicità
        dati_sez = self._main.get_sezione("strutture")
        cat_dict = dati_sez.get(cat, {})
        tutti = set(cat_dict.keys()) | set(self._db.get(cat, {}).keys())
        if nuovo in tutti:
            return

        # Aggiorna dati progetto
        if vecchio in cat_dict:
            cat_dict[nuovo] = cat_dict.pop(vecchio)
            self._main.push_undo("Rinomina struttura", "lista_strutture")
            self._main.set_sezione("strutture", dati_sez)

        # Aggiorna bottone
        self._bottoni.pop((cat, vecchio), None)
        btn.nome = nuovo
        self._bottoni[(cat, nuovo)] = btn
        btn.update()

        if self._cat_corrente == cat and self._nome_corrente == vecchio:
            self._nome_corrente = nuovo
            self._ui.struttura_label.setText(f"Struttura: {nuovo}")

        print(f">> Struttura rinominata: {vecchio} → {nuovo}")

    # ================================================================
    #  LETTURA DATI
    # ================================================================

    def _leggi(self, cat, nome) -> dict | None:
        """Legge i dati della struttura (progetto ha priorità su database)."""
        if self._main.ha_progetto():
            dati_sez = self._main.get_sezione("strutture")
            if cat in dati_sez and nome in dati_sez[cat]:
                return dati_sez[cat][nome]
        if cat in self._db and nome in self._db[cat]:
            return self._db[cat][nome]
        return None

    # ================================================================
    #  SALVATAGGIO
    # ================================================================

    def _salva_corrente(self):
        """Salva il testo corrente nel progetto."""
        if not self._main.ha_progetto():
            return
        if self._cat_corrente is None or self._nome_corrente is None:
            return

        cat, nome = self._cat_corrente, self._nome_corrente
        testo = self._testo.get_testo()

        dati_sez = self._main.get_sezione("strutture")
        cat_dict = dati_sez.setdefault(cat, {})

        if nome in cat_dict:
            if cat_dict[nome].get("testo", "") != testo:
                self._main.push_undo("Modifica struttura", "struttura")
                cat_dict[nome]["testo"] = testo
                self._main.set_sezione("strutture", dati_sez)
        else:
            # Struttura standard modificata → salva come custom nel progetto
            dati_base = self._leggi(cat, nome)
            if dati_base:
                nuovo = copy.deepcopy(dati_base)
                nuovo["testo"] = testo
                nuovo["standard"] = False
                self._main.push_undo("Modifica struttura", "struttura")
                cat_dict[nome] = nuovo
                self._main.set_sezione("strutture", dati_sez)

    # ================================================================
    #  AGGIORNA 3D
    # ================================================================

    def _aggiorna_3d(self):
        """Parsa il testo e aggiorna il viewer 3D + feedback errori."""
        dati, errori = self._testo.parse_e_valida()
        self._spazio.aggiorna_dati(dati)

        if dati.get("nodi"):
            self._spazio.centra_vista()

        # Salva anche
        self._salva_corrente()

        # Schedula cattura preview per il bottone
        QTimer.singleShot(200, self._cattura_preview_corrente)

    # ================================================================
    #  LAYOUT TOGGLE
    # ================================================================

    def _toggle_layout(self):
        """Toggle tra layout espanso (1200px) e compatto (550px)."""
        self._layout_espanso = not self._layout_espanso
        w = 1200 if self._layout_espanso else 550

        for widget in (self._ui.struttura_plainTextEdit,
                       self._ui.struttura_frame,
                       self._ui.text_control):
            widget.setMinimumWidth(w)
            widget.setMaximumWidth(w)

    # ================================================================
    #  VISTE
    # ================================================================

    def _centra_vista(self):
        self._spazio.centra_vista()

    # ================================================================
    #  INFO DIALOG
    # ================================================================

    def _mostra_info(self):
        if self._info_dialog is None:
            self._info_dialog = StrutturaInfoDialog(self._main)
        self._info_dialog.show()
        self._info_dialog.raise_()
        self._info_dialog.activateWindow()

    # ================================================================
    #  PREVIEW
    # ================================================================

    def _cattura_preview_corrente(self):
        """Cattura la preview isometrica per il bottone corrente."""
        if self._cat_corrente is None or self._nome_corrente is None:
            return
        key = (self._cat_corrente, self._nome_corrente)
        btn = self._bottoni.get(key)
        if btn is None:
            return

        dati, _ = parse_struttura(self._testo.get_testo())
        if not dati.get("nodi"):
            btn.set_preview(None)
            btn.update()
            return

        self._render_preview(btn, dati)

    def _render_preview(self, btn, dati):
        """Render isometrico per la preview del bottone."""
        # Salva stato camera
        old_rx   = self._spazio.rot_x
        old_ry   = self._spazio.rot_y
        old_px   = self._spazio.pan_x
        old_py   = self._spazio.pan_y
        old_dist = self._spazio.cam_dist
        old_ort  = self._spazio._ortho
        old_dati = self._spazio._dati

        px = None
        try:
            self._spazio._preview_mode = True
            self._spazio._ortho = False
            self._spazio.rot_x = 30.0
            self._spazio.rot_y = -45.0
            self._spazio._dati = dati

            # Centra
            if dati.get("nodi"):
                pts = np.array(list(dati["nodi"].values()), dtype=float)
                mn, mx = pts.min(axis=0), pts.max(axis=0)
                ctr = (mn + mx) / 2.0
                diag = float(np.linalg.norm(mx - mn))

                def Rx(deg):
                    a = math.radians(deg)
                    c, s = math.cos(a), math.sin(a)
                    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
                def Ry(deg):
                    a = math.radians(deg)
                    c, s = math.cos(a), math.sin(a)
                    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

                R = Rx(30.0) @ Ry(-45.0) @ Rx(-90.0)
                ctr_eye = R @ ctr
                self._spazio.pan_x = float(-ctr_eye[0])
                self._spazio.pan_y = float(-ctr_eye[1])
                self._spazio.cam_dist = max(
                    diag / (2.0 * math.tan(math.radians(22.5))) * 1.25, 2.0)

            if self._spazio.isValid():
                self._spazio.makeCurrent()
                self._spazio.paintGL()
                img = self._spazio.grabFramebuffer()
                if not img.isNull():
                    px = QPixmap.fromImage(img)

        except Exception as e:
            print(f"Errore render preview struttura: {e}")

        finally:
            self._spazio._preview_mode = False
            self._spazio.rot_x   = old_rx
            self._spazio.rot_y   = old_ry
            self._spazio.pan_x   = old_px
            self._spazio.pan_y   = old_py
            self._spazio.cam_dist = old_dist
            self._spazio._ortho  = old_ort
            self._spazio._dati   = old_dati
            if self._spazio.isValid():
                self._spazio.update()

        if px is not None and not px.isNull():
            btn.set_preview(px)
            btn.update()

    def _schedula_previews_tutti(self):
        """Genera preview per tutte le strutture che hanno testo."""
        self._coda_preview = []
        for (cat, nome), btn in list(self._bottoni.items()):
            dati = self._leggi(cat, nome)
            if dati and dati.get("testo", "").strip():
                self._coda_preview.append((btn, dati.get("testo", "")))
        if self._coda_preview:
            QTimer.singleShot(300, self._processa_prossima_preview)

    def _processa_prossima_preview(self):
        if not self._coda_preview:
            return
        btn, testo = self._coda_preview.pop(0)
        dati, _ = parse_struttura(testo)
        if dati.get("nodi"):
            self._render_preview(btn, dati)
        if self._coda_preview:
            QTimer.singleShot(100, self._processa_prossima_preview)

    # ================================================================
    #  PERSISTENZA (ricarica da progetto)
    # ================================================================

    def ricarica_da_progetto(self):
        """Ricarica la lista strutture dopo apertura/undo progetto."""
        # Svuota i frame
        frames = {
            "calcestruzzo":   self._ui.struttura_frame_calcestruzzo,
            "acciaio":        self._ui.struttura_frame_acciaio,
            "personalizzate": self._ui.struttura_frame_personalizzate,
        }
        for btn in list(self._bottoni.values()):
            self._btn_group.removeButton(btn)
        self._bottoni.clear()

        for cat, frame in frames.items():
            self._init_frame(frame)
            # Standard dal database
            for nome, dati in self._db.get(cat, {}).items():
                self._crea_bottone(frame, cat, nome, standard=True)
            # Custom dal progetto
            dati_sez = self._main.get_sezione("strutture")
            for nome, dati in dati_sez.get(cat, {}).items():
                if not dati.get("standard", False):
                    self._crea_bottone(frame, cat, nome, standard=False)

        self._cat_corrente = None
        self._nome_corrente = None
        self._ui.struttura_label.setText("Struttura: -")
        print(">> Modulo Struttura: progetto ricaricato.")

        self._schedula_previews_tutti()

    def ripristina_contesto(self, cat=None, nome=None):
        """Chiamata dopo undo/redo per riaprire la struttura in editing."""
        if cat and nome:
            dati = self._leggi(cat, nome)
            if dati:
                self._on_bottone_cliccato(cat, nome)
                return
        self._ui.stackedWidget_main.setCurrentIndex(8)
