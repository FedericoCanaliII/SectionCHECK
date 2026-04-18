"""
gestione_sezioni.py – Controller del modulo Sezioni.
Connette anche sezione_modifiche_lineEdit alle proprietà del tool/elemento.
"""
import copy
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QVBoxLayout, QAbstractButton, QMessageBox, QInputDialog,
    QStyledItemDelegate, QComboBox, QMenu, QHeaderView, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui  import QPainter, QColor, QPixmap, QPen, QFont, QBrush

from .database_sezioni import carica_database, nuova_sezione_vuota
from .spazio_disegno   import SpazioDisegno
from .tools_manager    import ToolsManager


# ================================================================
#  HELPERS NOMI
# ================================================================

def _prossimo_nome(nome_base: str, esistenti: set) -> str:
    """Genera nome_base.001, .002 ... non presente in esistenti."""
    n = 1
    while True:
        candidato = f"{nome_base}.{n:03d}"
        if candidato not in esistenti:
            return candidato
        n += 1

_NOME_BASE_SEZ: dict[str, str] = {
    "calcestruzzo_armato": "sezione_calcestruzzo",
    "profili":             "sezione_profili",
    "precompresso":        "sezione_precompresso",
    "personalizzate":      "sezione",
}


# ================================================================
#  FLOW LAYOUT
# ================================================================
class _FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, margin=8, spacing=8):
        super().__init__(parent)
        self._items, self._sp = [], spacing
        self.setContentsMargins(margin, margin, margin, margin)
    def addItem(self, i): self._items.append(i)
    def count(self): return len(self._items)
    def itemAt(self, i): return self._items[i] if 0 <= i < len(self._items) else None
    def takeAt(self, i): return self._items.pop(i) if 0 <= i < len(self._items) else None
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
        m = self.contentsMargins(); eff = rect.adjusted(m.left(),m.top(),-m.right(),-m.bottom())
        x, y, rh = eff.x(), eff.y(), 0
        for it in self._items:
            iw, ih = it.sizeHint().width(), it.sizeHint().height()
            nx = x + iw + self._sp
            if nx - self._sp > eff.right() and rh > 0:
                x, y = eff.x(), y + rh + self._sp; nx, rh = eff.x()+iw+self._sp, 0
            if not test: it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), it.sizeHint()))
            x, rh = nx, max(rh, ih)
        return y + rh - rect.y() + m.bottom()


# ================================================================
#  SECTION BUTTON  (200×100)
# ================================================================
_SHEET_CAT = {
    "calcestruzzo_armato": {"border":"rgb(120,120,120)","sheet":"""
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(120,120,120);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(120,120,120)}
        QAbstractButton:checked{background-color:rgb(65,65,65);
            border:1px solid rgb(200,200,200)}"""},
    "profili": {"border":"rgb(80,110,150)","sheet":"""
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(80,110,150);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(80,110,150)}
        QAbstractButton:checked{background-color:rgb(28,40,62);
            border:1px solid rgb(100,145,200)}"""},
    "precompresso": {"border":"rgb(160,120,120)","sheet":"""
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(160,120,120);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(160,120,120)}
        QAbstractButton:checked{background-color:rgb(65,38,38);
            border:1px solid rgb(200,150,150)}"""},
    "personalizzate": {"border":"rgb(150,150,50)","sheet":"""
        QAbstractButton{background-color:rgb(40,40,40);
            border:1px solid rgb(150,150,50);border-radius:6px}
        QAbstractButton:hover{background-color:rgb(30,30,30);
            border:1px solid rgb(150,150,50)}
        QAbstractButton:checked{background-color:rgb(50,50,18);
            border:1px solid rgb(195,195,75)}"""},
}
_BTN_W, _BTN_H = 200, 100; _PREV_H = 62; _NAME_H = 28

class SectionButton(QAbstractButton):
    deleteRequested = pyqtSignal()

    def __init__(self, nome, categoria, parent=None, standard=False):
        super().__init__(parent)
        self.nome = nome; self.categoria = categoria; self._preview = None
        self.standard = standard
        self.setFixedSize(_BTN_W, _BTN_H); self.setCheckable(True)
        self.setStyleSheet(_SHEET_CAT.get(categoria, _SHEET_CAT["personalizzate"])["sheet"])

    def _delete_btn_rect(self):
        """Rettangolo del bottone X in alto a destra (16×16 px)."""
        return QtCore.QRect(_BTN_W - 20, 4, 16, 16)
    def set_preview(self, px): self._preview = px; self.update()
    def initStyleOption(self, opt):
        opt.initFrom(self); opt.features = QtWidgets.QStyleOptionButton.None_
        opt.state |= (QtWidgets.QStyle.State_Sunken if self.isDown() else QtWidgets.QStyle.State_Raised)
        if self.isChecked(): opt.state |= QtWidgets.QStyle.State_On
        opt.text = ""; opt.icon = QtGui.QIcon()
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        self.style().drawControl(QtWidgets.QStyle.CE_PushButton, opt, painter, self)

        # --- PREVIEW AREA (100x50, arrotondata 4px, sfondo 50,50,50) ---
        prev_w, prev_h = 100, 50
        ox = (_BTN_W - prev_w) // 2
        oy = (_PREV_H - prev_h) // 2

        # Crea il tracciato con bordi arrotondati
        path = QtGui.QPainterPath()
        path.addRoundedRect(ox, oy, prev_w, prev_h, 4, 4)

        # Disegna lo sfondo grigio senza nessun bordo solido
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

        # Se c'è un'immagine, la disegna clippata all'interno del path arrotondato
        if self._preview:
            painter.save()
            painter.setClipPath(path)
            px = self._preview.scaled(prev_w, prev_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            px_x = ox + (prev_w - px.width()) // 2
            px_y = oy + (prev_h - px.height()) // 2
            painter.drawPixmap(px_x, px_y, px)
            painter.restore()

        # --- TESTO NOME SEZIONE ---
        painter.setPen(QColor(210, 210, 210))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(QtCore.QRect(4, _PREV_H + 4, _BTN_W - 8, _NAME_H), Qt.AlignCenter | Qt.TextWordWrap, self.nome)

        # --- BOTTONE ELIMINA (solo sezioni non-standard) ---
        if not self.standard:
            dr = self._delete_btn_rect()
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
                and self._delete_btn_rect().contains(e.pos())):
            self.deleteRequested.emit()
            e.accept()
            return
        super().mousePressEvent(e)

    def sizeHint(self): return QtCore.QSize(_BTN_W, _BTN_H)


# ================================================================
#  DELEGATE MATERIALE
# ================================================================
class _MatDelegate(QStyledItemDelegate):
    def __init__(self, get_fn, parent=None): super().__init__(parent); self._get = get_fn
    def createEditor(self, parent, opt, idx):
        c = QComboBox(parent); c.addItem("")
        for n in self._get(): c.addItem(n)
        return c
    def setEditorData(self, ed, idx): ed.setCurrentIndex(max(ed.findText(idx.data(Qt.EditRole) or ""), 0))
    def setModelData(self, ed, mod, idx): mod.setData(idx, ed.currentText(), Qt.EditRole)
    def updateEditorGeometry(self, ed, opt, idx): ed.setGeometry(opt.rect)


# ================================================================
#  TABELLE
# ================================================================
_STILE_TV = """
    /* --- BASE DELLA TABELLA --- */
QTableView {
    border: 1px solid rgb(120, 120, 120);
    border-top: 3px solid rgb(120, 120, 120);
    border-radius: 6px;
    background-color: rgb(50, 50, 50); /* Il tuo sfondo base */
    gridline-color: rgb(120, 120, 120); /* Linee sottili interne da 1px */
    font: 400 10pt "Segoe UI";
    color: rgb(255, 255, 255);
}

/* --- CELLE INTERNE --- */
QTableView::item {
    background-color: transparent; /* Nessun background aggiuntivo */
    border: none; /* I bordi sono gestiti dalla gridline-color del QTableView */
    padding: 4px; /* Un po' di respiro per il testo, modificabile a piacere */
}

/* --- CONTENITORE DEGLI HEADER --- */
QHeaderView {
    background-color: transparent;
    border: none;
}

/* --- HEADER ORIZZONTALI --- */
QHeaderView::section:horizontal {
    font: 400 12pt "Georgia";
    color: rgb(255, 255, 255);
    background-color: transparent;
    /* Metto il bordo SOLO a destra e in basso per evitare la somma di 1px + 1px = 2px */
    border: none;
    border-right: 1px solid rgb(120, 120, 120);
    border-bottom: 1px solid rgb(120, 120, 120);
    padding: 4px;
}

/* --- HEADER VERTICALI --- */
QHeaderView::section:vertical {
    font: 400 12pt "Georgia";
    color: rgb(255, 255, 255);
    background-color: transparent;
    /* Metto il bordo SOLO a destra e in basso per evitare l'effetto doppio */
    border: none;
    border-right: 1px solid rgb(120, 120, 120);
    border-bottom: 1px solid rgb(120, 120, 120);
    padding: 4px;
}

/* --- ANGOLO IN ALTO A SINISTRA --- */
QTableView QTableCornerButton::section {
    background-color: transparent;
    border: none;
    border-right: 1px solid rgb(120, 120, 120);
    border-bottom: 1px solid rgb(120, 120, 120);
}
"""
_COL_C = ["ID","Tipo","Proprietà geometriche","Materiale"]
_COL_B = ["ID","Proprietà geometriche","Materiale"]
_COL_S = ["ID","Proprietà geometriche","Materiale"]

def _item(t, editable=False, user_data=None):
    it = QtGui.QStandardItem(str(t))
    it.setForeground(QtGui.QBrush(QtGui.QColor(215,215,215)))
    it.setBackground(QtGui.QBrush(QtGui.QColor(30,30,30)))
    if not editable: it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if user_data is not None: it.setData(user_data, Qt.UserRole)
    return it

def _geom_text(el):
    """Testo proprietà geometriche nel formato lineEdit."""
    g = el["geometria"]; t = el["tipo"]
    if t in ("rettangolo", "foro_rettangolo"):
        return f"V1 ({g['x0']:.1f},{g['y0']:.1f}); V2 ({g['x1']:.1f},{g['y1']:.1f})"
    elif t in ("poligono", "foro_poligono"):
        return "; ".join(f"V{i+1} ({p[0]:.1f},{p[1]:.1f})" for i,p in enumerate(g["punti"]))
    elif t in ("cerchio", "foro_cerchio"):
        rx = g.get("rx", g.get("r", 10)); ry = g.get("ry", rx)
        return f"C({g['cx']:.1f},{g['cy']:.1f}); rx={rx:.1f}; ry={ry:.1f}"
    elif t == "barra":
        return f"Ø={g['r']*2:.1f}; C({g['cx']:.1f},{g['cy']:.1f})"
    elif t == "staffa":
        pts = g["punti"]; r = g.get("r", 4)
        parts = [f"Ø={r*2:.1f}"]
        for i,p in enumerate(pts):
            if i == len(pts)-1 and p == pts[0]: continue
            parts.append(f"P{i+1} ({p[0]:.1f},{p[1]:.1f})")
        return "; ".join(parts)
    return "—"

def _riga_carp(el):
    return [_item(el["id"], True, user_data=el["id"]),
            _item(el["tipo"]),
            _item(_geom_text(el), True),
            _item(el.get("materiale",""), True)]

def _riga_barra(el):
    return [_item(el["id"], True, user_data=el["id"]),
            _item(_geom_text(el), True),
            _item(el.get("materiale",""), True)]

def _riga_staffa(el):
    return [_item(el["id"], True, user_data=el["id"]),
            _item(_geom_text(el), True),
            _item(el.get("materiale",""), True)]


# ================================================================
#  CONTROLLER
# ================================================================
class GestioneSezioni:
    def __init__(self, ui, main_window):
        self._ui = ui; self._main = main_window
        self._cat_corrente = None; self._nome_corrente = None
        self._bottoni = {}; self._btn_group = QtWidgets.QButtonGroup(main_window)
        self._btn_group.setExclusive(True)
        self._db = carica_database()
        self._spazio = SpazioDisegno()
        self._spazio.elementi_modificati.connect(self._on_elementi_modificati)
        self._spazio.tool_preview_changed.connect(self._on_tool_preview)
        self._tools_mgr = ToolsManager(ui, self._spazio)
        self._mod_carp = QtGui.QStandardItemModel()
        self._mod_barre = QtGui.QStandardItemModel()
        self._mod_staf = QtGui.QStandardItemModel()
        self._setup_spazio_disegno(); self._setup_tabelle()
        self._setup_frame_sezioni(); self._setup_connessioni()
        self._setup_lineedit_modifiche()
        self._ui.sezione_btn_muovi.click()

    def _setup_spazio_disegno(self):
        c = self._ui.sezione_widget; lay = c.layout()
        if lay is None: lay = QVBoxLayout(c); lay.setContentsMargins(1,1,1,1)
        lay.addWidget(self._spazio)

    def _setup_tabelle(self):
        ET = QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.AnyKeyPressed
        # Carpenteria
        tv = self._ui.tableView_carpenteria
        self._mod_carp.setHorizontalHeaderLabels(_COL_C); tv.setModel(self._mod_carp)
        tv.setStyleSheet(_STILE_TV); tv.setEditTriggers(ET)
        tv.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tv.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        tv.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        tv.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        tv.setItemDelegateForColumn(3, _MatDelegate(self._get_mat_carp, tv))
        tv.setContextMenuPolicy(Qt.CustomContextMenu)
        tv.customContextMenuRequested.connect(lambda p: self._ctx(tv, self._mod_carp, "carpenteria", p))
        # Barre
        tv = self._ui.tableView_barre
        self._mod_barre.setHorizontalHeaderLabels(_COL_B); tv.setModel(self._mod_barre)
        tv.setStyleSheet(_STILE_TV); tv.setEditTriggers(ET)
        tv.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tv.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        tv.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        tv.setItemDelegateForColumn(2, _MatDelegate(self._get_mat_barre, tv))
        tv.setContextMenuPolicy(Qt.CustomContextMenu)
        tv.customContextMenuRequested.connect(lambda p: self._ctx(tv, self._mod_barre, "barre", p))
        # Staffe
        tv = self._ui.tableView_staffe
        self._mod_staf.setHorizontalHeaderLabels(_COL_S); tv.setModel(self._mod_staf)
        tv.setStyleSheet(_STILE_TV); tv.setEditTriggers(ET)
        tv.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tv.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        tv.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        tv.setItemDelegateForColumn(2, _MatDelegate(self._get_mat_barre, tv))
        tv.setContextMenuPolicy(Qt.CustomContextMenu)
        tv.customContextMenuRequested.connect(lambda p: self._ctx(tv, self._mod_staf, "staffe", p))
        for m in (self._mod_carp, self._mod_barre, self._mod_staf):
            m.itemChanged.connect(self._on_item_changed)

    # --- LINEEDIT MODIFICHE ---
    def _setup_lineedit_modifiche(self):
        le = self._ui.sezione_modifiche_lineEdit
        le.setStyleSheet("background-color: rgb(40, 40, 40);color: rgb(221, 221, 221);border: 1px solid rgb(120, 120, 120);border-left: 3px solid rgb(120,120,120);border-radius: 6px;padding:4px;font:9pt Consolas;")
        le.setPlaceholderText("Proprietà elemento...")
        le.returnPressed.connect(self._on_lineedit_apply)

    def _on_tool_preview(self, text):
        """Riceve le proprietà dal tool e le mostra nel lineEdit."""
        le = self._ui.sezione_modifiche_lineEdit
        le.blockSignals(True); le.setText(text); le.blockSignals(False)

    def _on_lineedit_apply(self):
        """Quando l'utente preme Enter nel lineEdit, applica le proprietà."""
        text = self._ui.sezione_modifiche_lineEdit.text()
        if not text.strip(): return
        tool = self._spazio.active_tool
        if tool:
            from .tools.tool_modifica import ToolModifica
            if isinstance(tool, ToolModifica):
                el = tool._get_selected(self._spazio)
                if el:
                    tool.apply_properties_on(el, text)
                    self._spazio.update()
                    self._spazio.elementi_modificati.emit()
            else:
                tool.apply_properties_text(text)
                self._spazio.update()

    # --- Frame sezioni ---
    def _setup_frame_sezioni(self):
        for btn in list(self._btn_group.buttons()): self._btn_group.removeButton(btn)
        self._bottoni.clear()
        prj = self._main.get_sezione("sezioni")
        mappa = {
            "calcestruzzo_armato": self._ui.sezione_frame_calcestruzzo_armato,
            "profili": self._ui.sezione_frame_profili,
            "precompresso": self._ui.sezione_frame_precompresso,
            "personalizzate": self._ui.sezione_frame_personalizzate,
        }
        for cat, frame in mappa.items():
            self._init_frame(frame)
            for nome in self._db.get(cat, {}): self._crea_bottone(frame, cat, nome, standard=True)
            for nome, dati in prj.get(cat, {}).items():
                if not dati.get("standard", True) and nome not in self._db.get(cat, {}):
                    self._crea_bottone(frame, cat, nome, standard=False)

    @staticmethod
    def _init_frame(f):
        lay = f.layout()
        if lay is None:
            # Prima volta: il frame non ha ancora un layout → lo creiamo
            f.setLayout(_FlowLayout(margin=8, spacing=8))
        else:
            # Chiamate successive: svuotiamo il layout esistente senza ricrearlo.
            # Chiamare setLayout() su un widget che ha già un layout genera il
            # warning "which already has a layout": Qt mantiene il puntatore
            # interno al vecchio layout anche dopo i tentativi di reparenting.
            while lay.count():
                it = lay.takeAt(0)
                w  = it.widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()

    def _crea_bottone(self, frame, cat, nome, standard=False):
        btn = SectionButton(nome, cat, frame, standard=standard)
        frame.layout().addWidget(btn)
        self._btn_group.addButton(btn)
        self._bottoni[(cat, nome)] = btn
        # Lambda dinamiche: leggono btn.nome/btn.categoria al momento del clic
        btn.clicked.connect(lambda _c, b=btn: self._seleziona_sezione(b.categoria, b.nome))
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        btn.customContextMenuRequested.connect(
            lambda pos, b=btn: self._ctx_sezione(b, pos))
        if not standard:
            btn.deleteRequested.connect(
                lambda b=btn: self._elimina_sezione(b.categoria, b.nome))
        self._aggiorna_preview(cat, nome)
        return btn

    # ---------------------------------------------------------------
    #  ICONE FRECCIA
    # ---------------------------------------------------------------
    _ICON_APERTO  = "interfaccia/icone/freccia_aperto.png"
    _ICON_CHIUSO  = "interfaccia/icone/freccia_chiuso.png"

    def _setup_connessioni(self):
        self._ui.sezione_aggiungi_calcestruzzo_armato.clicked.connect(lambda: self._aggiungi("calcestruzzo_armato"))
        self._ui.sezione_aggiungi_profili.clicked.connect(lambda: self._aggiungi("profili"))
        self._ui.sezione_aggiungi_precompresso.clicked.connect(lambda: self._aggiungi("precompresso"))
        self._ui.sezione_aggiungi_personalizzate.clicked.connect(lambda: self._aggiungi("personalizzate"))
        self._ui.btn_sezione_centra.clicked.connect(self._spazio.reset_view)

        # Bottoni visibilità tabelle
        _pairs = [
            (self._ui.sezione_visibile_carpenteria, self._ui.tableView_carpenteria),
            (self._ui.sezione_visibile_barre,       self._ui.tableView_barre),
            (self._ui.sezione_visibile_staffe,       self._ui.tableView_staffe),
        ]
        for btn, tv in _pairs:
            # Icona iniziale: tabella visibile → freccia aperto
            btn.setIcon(QtGui.QIcon(self._ICON_APERTO))
            btn.clicked.connect(lambda _checked, b=btn, t=tv: self._toggle_tabella(b, t))

    def _toggle_tabella(self, btn, tableview):
        """Mostra/nasconde la tableview e aggiorna l'icona del bottone."""
        visibile = tableview.isVisible()
        tableview.setVisible(not visibile)
        icon_path = self._ICON_CHIUSO if visibile else self._ICON_APERTO
        btn.setIcon(QtGui.QIcon(icon_path))

    def _seleziona_sezione(self, cat, nome):
        if self._nome_corrente: self._salva()
        self._cat_corrente = cat; self._nome_corrente = nome
        dati = self._leggi(cat, nome)
        if not dati: return
        self._spazio.carica_elementi(dati.get("elementi", {}))
        self._ricarica_tabelle()
        self._ui.label_id_sezione.setText(nome)
        self._ui.stackedWidget_main.setCurrentIndex(4)

    def _leggi(self, cat, nome):
        sez = self._main.get_sezione("sezioni")
        if nome in sez.get(cat, {}): return sez[cat][nome]
        src = self._db.get(cat, {}).get(nome)
        return copy.deepcopy(src) if src else None

    # --- Tabelle ---
    def _ricarica_tabelle(self):
        for m in (self._mod_carp, self._mod_barre, self._mod_staf):
            try: m.itemChanged.disconnect(self._on_item_changed)
            except: pass
        self._mod_carp.removeRows(0, self._mod_carp.rowCount())
        self._mod_carp.setHorizontalHeaderLabels(_COL_C)
        for el in self._spazio.get_elementi("carpenteria"): self._mod_carp.appendRow(_riga_carp(el))
        self._mod_barre.removeRows(0, self._mod_barre.rowCount())
        self._mod_barre.setHorizontalHeaderLabels(_COL_B)
        for el in self._spazio.get_elementi("barre"): self._mod_barre.appendRow(_riga_barra(el))
        self._mod_staf.removeRows(0, self._mod_staf.rowCount())
        self._mod_staf.setHorizontalHeaderLabels(_COL_S)
        for el in self._spazio.get_elementi("staffe"): self._mod_staf.appendRow(_riga_staffa(el))
        for m in (self._mod_carp, self._mod_barre, self._mod_staf):
            m.itemChanged.connect(self._on_item_changed)
        for tv in (self._ui.tableView_carpenteria, self._ui.tableView_barre, self._ui.tableView_staffe):
            tv.viewport().update()

    def _on_elementi_modificati(self):
        nc = len(self._spazio.get_elementi("carpenteria"))
        nb = len(self._spazio.get_elementi("barre"))
        ns = len(self._spazio.get_elementi("staffe"))
        print(f">> Elementi: carp={nc}, barre={nb}, staffe={ns}")
        self._ricarica_tabelle(); self._salva()
        self._aggiorna_preview(self._cat_corrente, self._nome_corrente)

    def _on_item_changed(self, item):
        mod = item.model(); row = item.row(); col = item.column()
        if mod is self._mod_carp:
            cat, col_id, col_prop, col_mat = "carpenteria", 0, 2, 3
        elif mod is self._mod_barre:
            cat, col_id, col_prop, col_mat = "barre", 0, 1, 2
        elif mod is self._mod_staf:
            cat, col_id, col_prop, col_mat = "staffe", 0, 1, 2
        else: return
        id_item = mod.item(row, col_id)
        if not id_item: return

        mod.blockSignals(True)
        try:
            if col == col_id:
                old_id = item.data(Qt.UserRole)
                new_id = item.text().strip()
                if not new_id or new_id == old_id: return
                # Verifica unicità ID
                all_ids = {e["id"] for c in ("carpenteria","barre","staffe")
                           for e in self._spazio.get_elementi(c)}
                if new_id in all_ids:
                    item.setText(old_id); return
                for el in self._spazio.get_elementi(cat):
                    if el["id"] == old_id:
                        el["id"] = new_id
                        # Aggiorna selected_id se era questo elemento
                        if self._spazio._selected_id == old_id:
                            self._spazio._selected_id = new_id
                        break
                item.setData(new_id, Qt.UserRole)
                self._spazio.update(); self._salva()
                self._aggiorna_preview(self._cat_corrente, self._nome_corrente)

            elif col == col_prop:
                eid = id_item.text(); text = item.text()
                from .tools.tool_modifica import ToolModifica
                _tm = ToolModifica()
                for el in self._spazio.get_elementi(cat):
                    if el["id"] == eid:
                        _tm.apply_properties_on(el, text)
                        # Normalizza testo dopo parsing
                        item.setText(_geom_text(el))
                        break
                self._spazio.update(); self._salva()
                self._aggiorna_preview(self._cat_corrente, self._nome_corrente)

            elif col == col_mat:
                eid = id_item.text(); mat = item.text()
                for el in self._spazio.get_elementi(cat):
                    if el["id"] == eid: el["materiale"] = mat; break
                self._salva()
        finally:
            mod.blockSignals(False)

    def _ctx(self, tv, mod, cat, pos):
        idx = tv.indexAt(pos)
        if not idx.isValid(): return
        id_item = mod.item(idx.row(), 0)
        if not id_item: return
        eid = id_item.text()
        menu = QMenu(tv)
        menu.setStyleSheet("QMenu{background:#252525;color:#ddd;border:1px solid #555}"
                           "QMenu::item:selected{background:#3a5080}")
        act_del = menu.addAction(f"Elimina «{eid}»")
        if menu.exec_(tv.viewport().mapToGlobal(pos)) == act_del:
            self._spazio.rimuovi_elemento(eid)

    def _salva(self):
        if not self._cat_corrente or not self._nome_corrente: return
        if not self._main.ha_progetto(): return
        sez = self._main.get_sezione("sezioni")
        cd = sez.setdefault(self._cat_corrente, {})
        base = cd.get(self._nome_corrente)
        if base is None:
            base = self._db.get(self._cat_corrente, {}).get(
                self._nome_corrente, nuova_sezione_vuota(self._cat_corrente))
        d = copy.deepcopy(base); d["elementi"] = self._spazio.get_tutti_elementi()
        self._main.push_undo(f"Modifica sezione [{self._nome_corrente}]")
        cd[self._nome_corrente] = d
        self._main.set_sezione("sezioni", sez)

    def _aggiungi(self, cat):
        if not self._main.ha_progetto():
            QMessageBox.warning(self._main, "Attenzione", "Crea o apri prima un progetto."); return
        sez = self._main.get_sezione("sezioni"); cd = sez.setdefault(cat, {})
        esistenti = set(cd.keys()) | set(self._db.get(cat, {}).keys())
        nome = _prossimo_nome(_NOME_BASE_SEZ.get(cat, "sezione"), esistenti)
        self._main.push_undo(f"Aggiungi sezione [{nome}]")
        cd[nome] = nuova_sezione_vuota(cat)
        self._main.set_sezione("sezioni", sez)
        fm = {"calcestruzzo_armato": self._ui.sezione_frame_calcestruzzo_armato,
              "profili": self._ui.sezione_frame_profili,
              "precompresso": self._ui.sezione_frame_precompresso,
              "personalizzate": self._ui.sezione_frame_personalizzate}.get(cat)
        if fm: self._crea_bottone(fm, cat, nome, standard=False)

    def _rinomina(self, btn):
        """Apre un QLineEdit inline per rinominare la sezione."""
        dati = self._leggi(btn.categoria, btn.nome)
        if dati and dati.get("standard", False):
            return  # standard: silenzioso

        vec = btn.nome
        le = QLineEdit(btn)
        le.setGeometry(4, _PREV_H + 4 + 3, _BTN_W - 8, 22)
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
                self._applica_rinomina_sez(btn, vec, nuovo)

        le.returnPressed.connect(commit)
        le.editingFinished.connect(commit)

    def _applica_rinomina_sez(self, btn, vec: str, nuovo: str):
        cat = btn.categoria
        sez = self._main.get_sezione("sezioni"); cd = sez.get(cat, {})
        if nuovo in cd or nuovo in self._db.get(cat, {}):
            QMessageBox.warning(self._main, "Attenzione", f"Esiste già «{nuovo}»."); return
        if vec in cd:
            self._main.push_undo(f"Rinomina sezione [{vec} → {nuovo}]")
            cd[nuovo] = cd.pop(vec); self._main.set_sezione("sezioni", sez)
        # Aggiorna bottone (le lambda leggono btn.nome dinamicamente)
        btn.nome = nuovo
        if (cat, vec) in self._bottoni:
            self._bottoni[(cat, nuovo)] = self._bottoni.pop((cat, vec))
        if self._nome_corrente == vec and self._cat_corrente == cat:
            self._nome_corrente = nuovo
            self._ui.label_id_sezione.setText(nuovo)
        btn.update()
        print(f">> Sezione rinominata: [{cat}] {vec} → {nuovo}")

    # ------------------------------------------------------------------
    #  CONTEXT MENU TASTO DESTRO
    # ------------------------------------------------------------------

    def _ctx_sezione(self, btn, pos):
        menu = QMenu(btn)
        menu.setStyleSheet("QMenu{background:#252525;color:#ddd;border:1px solid #555}"
                           "QMenu::item:selected{background:#3a5080}")
        act_dup = menu.addAction("Duplica")
        act_ren = act_del = None
        if not btn.standard:
            act_ren = menu.addAction("Rinomina")
            act_del = menu.addAction("Elimina")
        action = menu.exec_(btn.mapToGlobal(pos))
        if action == act_dup:
            self._duplica_sezione(btn.categoria, btn.nome)
        elif act_ren and action == act_ren:
            self._rinomina(btn)
        elif act_del and action == act_del:
            self._elimina_sezione(btn.categoria, btn.nome)

    # ------------------------------------------------------------------
    #  DUPLICA SEZIONE
    # ------------------------------------------------------------------

    def _duplica_sezione(self, cat: str, nome: str):
        dati = self._leggi(cat, nome)
        if dati is None:
            return
        sez = self._main.get_sezione("sezioni"); cd = sez.setdefault(cat, {})
        esistenti = set(cd.keys()) | set(self._db.get(cat, {}).keys())
        nuovo_nome = _prossimo_nome(nome, esistenti)
        copia = copy.deepcopy(dati)
        copia["standard"] = False
        self._main.push_undo(f"Duplica sezione [{nome} → {nuovo_nome}]")
        cd[nuovo_nome] = copia
        self._main.set_sezione("sezioni", sez)
        fm = {"calcestruzzo_armato": self._ui.sezione_frame_calcestruzzo_armato,
              "profili": self._ui.sezione_frame_profili,
              "precompresso": self._ui.sezione_frame_precompresso,
              "personalizzate": self._ui.sezione_frame_personalizzate}.get(cat)
        if fm:
            self._crea_bottone(fm, cat, nuovo_nome, standard=False)
        print(f">> Sezione duplicata: [{cat}] {nome} → {nuovo_nome}")

    def _elimina_sezione(self, cat, nome):
        dati = self._leggi(cat, nome)
        if dati and dati.get("standard", False):
            return
        reply = QMessageBox.question(
            self._main, "Elimina sezione",
            f"Eliminare la sezione «{nome}»?\nL'operazione non è reversibile.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        sez = self._main.get_sezione("sezioni"); cd = sez.get(cat, {})
        if nome in cd:
            self._main.push_undo(f"Elimina sezione [{nome}]")
            del cd[nome]; self._main.set_sezione("sezioni", sez)
        key = (cat, nome)
        if key in self._bottoni:
            btn = self._bottoni.pop(key)
            self._btn_group.removeButton(btn)
            btn.setParent(None); btn.deleteLater()
        if self._cat_corrente == cat and self._nome_corrente == nome:
            self._cat_corrente = None; self._nome_corrente = None
            self._spazio.reset_elementi(); self._ricarica_tabelle()

    def _aggiorna_preview(self, cat, nome):
        if not cat or not nome: return
        btn = self._bottoni.get((cat, nome))
        if not btn: return
        dati = self._leggi(cat, nome)
        if not dati: return
        tmp = SpazioDisegno(); tmp.carica_elementi(dati.get("elementi", {}))
        # Richiedi la thumbnail esattamente 100x50
        btn.set_preview(tmp.genera_thumbnail(100, 50))

    def _get_mat_carp(self):
        mat = self._main.get_sezione("materiali"); n = []
        for c in ("calcestruzzo","acciaio","personalizzati"): n += list(mat.get(c, {}).keys())
        return n

    def _get_mat_barre(self):
        mat = self._main.get_sezione("materiali"); n = []
        for c in ("barre","personalizzati"): n += list(mat.get(c, {}).keys())
        return n

    def ricarica_da_progetto(self):
        self._cat_corrente = None; self._nome_corrente = None
        self._setup_frame_sezioni(); self._spazio.reset_elementi(); self._ricarica_tabelle()
        print(">> Modulo Sezioni: progetto ricaricato.")

    def ripristina_contesto(self, cat: str, nome: str):
        """
        Chiamata da MainWindow dopo undo/redo per riaprire la sezione
        che era in editing al momento dello snapshot.
        """
        sez = self._main.get_sezione("sezioni")
        # La sezione deve ancora esistere dopo l'undo
        if nome not in sez.get(cat, {}) and nome not in self._db.get(cat, {}):
            return
        btn = self._bottoni.get((cat, nome))
        if btn:
            btn.setChecked(True)
        # _cat_corrente e _nome_corrente sono None dopo ricarica_da_progetto → ok
        self._seleziona_sezione(cat, nome)
