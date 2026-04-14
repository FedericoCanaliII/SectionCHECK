"""
controller_elementi.py – Controller per lo spazio 3D e il pannello (index 6).
"""

import re

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QVBoxLayout, QAbstractItemView, QInputDialog, QMenu,
    QStyledItemDelegate, QHeaderView, QTreeWidget, QTreeWidgetItem,
    QMessageBox,
)
from PyQt5.QtCore  import Qt, pyqtSignal, QObject
from PyQt5.QtGui   import (
    QPainter, QColor, QPen, QBrush, QFont,
    QStandardItemModel, QStandardItem,
)

from .modello_3d          import (Oggetto3D, Elemento, TIPO_STRUTTURALE, TIPO_ARMATURA,
                                  ruota_punto, detrasforma_punto)
from .elementi_spazio_3d  import ElementiSpazio3D
from .tools_manager       import ToolsManager


# ============================================================
#  STYLES / CONSTANTS
# ============================================================

_STILE_TV = """
QTableView {
    border:1px solid rgb(120,120,120); border-top:3px solid rgb(120,120,120);
    border-radius:6px; background-color:rgb(50,50,50);
    gridline-color:rgb(120,120,120); font:400 10pt "Segoe UI"; color:rgb(255,255,255);
}
QTableView::item { background-color:transparent; border:none; padding:4px; }
QHeaderView { background-color:transparent; border:none; }
QHeaderView::section:horizontal {
    font:400 11pt "Georgia"; color:rgb(255,255,255); background-color:transparent;
    border:none; border-right:1px solid rgb(120,120,120);
    border-bottom:1px solid rgb(120,120,120); padding:4px;
}
QHeaderView::section:vertical {
    font:400 10pt "Georgia"; color:rgb(255,255,255); background-color:transparent;
    border:none; border-right:1px solid rgb(120,120,120);
    border-bottom:1px solid rgb(120,120,120); padding:4px;
}
QTableView QTableCornerButton::section {
    background-color:transparent; border:none;
    border-right:1px solid rgb(120,120,120); border-bottom:1px solid rgb(120,120,120);
}
"""

_OUTLINER_SHEET = """
QTreeWidget {
    background-color: rgb(35,35,35);
    color: rgb(220,220,220);
    border: 1px solid rgb(120,120,120); /* Bordi esterni 120,120,120 */
    border-radius: 6px;                 /* Raggio a 6 */
    font: 9pt "Segoe UI";
    outline: none;
}
QTreeWidget::item {
    height: 24px;
    padding-left: 2px;
    border-bottom: 1px solid rgb(42,42,42);
}
QTreeWidget::item:selected {
    background-color: rgb(62,90,140);
    color: rgb(255,255,255);
}
QTreeWidget::item:hover:!selected {
    background-color: rgb(48,50,56);
}
QHeaderView {
    background-color: transparent;
    border: none;
}
QHeaderView::section {
    background-color: rgb(42,42,42);
    color: rgb(180,180,180);
    font: bold 9pt "Segoe UI";          /* Dimensione 9pt uguale a Carpenteria/Armature */
    border: none;
    border-right: 1px solid rgb(120,120,120);
    border-bottom: 1px solid rgb(120,120,120);
    padding: 2px;
}
/* FIX: Smussa l'header per non fargli "mangiare" i bordi del parent */
QHeaderView::section:first {
    border-top-left-radius: 5px;
}
QHeaderView::section:last {
    border-top-right-radius: 5px;
    border-right: none;
}
QTreeWidget::branch {
    background-color: transparent;
    image: none;
    border-image: none;
}
"""

_ROLE_OBJ_ID     = Qt.UserRole + 1
_ROLE_IS_FOLD    = Qt.UserRole + 2
_ROLE_VISIBLE    = Qt.UserRole + 3   # bool  – stored on col 0
_ROLE_SELECTABLE = Qt.UserRole + 4   # bool  – stored on col 1

_FOLD_STRUTTURALI = "Carpenteria"
_FOLD_ARMATURA_L  = "Armatura longitudinale"
_FOLD_ARMATURA_T  = "Armatura trasversale"

_FOLD_ORDER = [_FOLD_STRUTTURALI, _FOLD_ARMATURA_L, _FOLD_ARMATURA_T]

_TIPO_A_FOLDER = {
    "parallelepipedo": _FOLD_STRUTTURALI,
    "cilindro":        _FOLD_STRUTTURALI,
    "sfera":           _FOLD_STRUTTURALI,
    "barra":           _FOLD_ARMATURA_L,
    "staffa":          _FOLD_ARMATURA_T,
}

_GEOM_COLS = {
    "parallelepipedo": [("Base Y [m]",       "base"),
                        ("Altezza Z [m]",     "altezza"),
                        ("Lunghezza X [m]",   "lunghezza")],
    "cilindro":        [("Altezza Z [m]",     "altezza"),
                        ("Raggio R [m]",      "raggio")],
    "sfera":           [("Raggio R [m]",      "raggio")],
    "barra":           [("Diametro Φ [m]",    "diametro")],
    "staffa":          [("Diametro Φ [m]",    "diametro")],
}


# ============================================================
#  TABLE HELPERS
# ============================================================

def _make_item(text, editable=True):
    it = QStandardItem(str(text))
    it.setForeground(QtGui.QBrush(QColor(215, 215, 215)))
    it.setBackground(QtGui.QBrush(QColor(30, 30, 30)))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


# ============================================================
#  VERTEX TEXT PARSING
# ============================================================

def _parse_vertici_text(text: str) -> list | None:
    pattern = r'v\d+\s*\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    try:
        return [[float(x), float(y), float(z)] for x, y, z in matches]
    except ValueError:
        return None


# ============================================================
#  OUTLINER DELEGATE
# ============================================================

class _OutlinerDelegate(QStyledItemDelegate):
    """Gestisce il disegno personalizzato dei pallini (visibilità/selezione/cartelle)."""

    def __init__(self, tree_widget):
        super().__init__(tree_widget)
        self.tree = tree_widget

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        col = index.column()
        if col not in (0, 1):
            return

        is_fold = index.data(_ROLE_IS_FOLD)
        r = option.rect
        d = 10                                      # Diametro pallino
        x = r.x() + (r.width()  - d) // 2
        y = r.y() + (r.height() - d) // 2

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # Disegno per le CARTELLE
        if is_fold:
            if col == 0:
                item = self.tree.itemFromIndex(index)
                if item:
                    is_expanded = item.isExpanded()
                    # Colore violaceo: acceso se espanso, scuro se chiuso
                    fill   = QColor(140, 80, 200) if is_expanded else QColor(52, 52, 52)
                    border = QColor(170, 110, 230) if is_expanded else QColor(78, 78, 78)
                    painter.setPen(QPen(border, 1.0))
                    painter.setBrush(QBrush(fill))
                    painter.drawEllipse(x, y, d, d)
            # La colonna 1 (selezionabilità) per le cartelle resta vuota
            painter.restore()
            return

        # Disegno per i NORMALI OGGETTI
        if col == 0:
            on     = bool(index.data(_ROLE_VISIBLE))
            fill   = QColor(88, 195, 88)   if on else QColor(52, 52, 52)
            border = QColor(115, 225, 115) if on else QColor(78, 78, 78)
        else:
            on     = bool(index.data(_ROLE_SELECTABLE))
            fill   = QColor(65, 135, 215)  if on else QColor(52, 52, 52)
            border = QColor(95,  165, 245) if on else QColor(78, 78, 78)

        painter.setPen(QPen(border, 1.0))
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(x, y, d, d)
        painter.restore()

    def sizeHint(self, option, index):
        sh = super().sizeHint(option, index)
        if index.column() in (0, 1):
            return QtCore.QSize(28, sh.height())
        return sh


# ============================================================
#  OUTLINER
# ============================================================

class Outliner(QTreeWidget):
    """Albero oggetti stile Blender."""

    obj_selected    = pyqtSignal(int)
    obj_deleted     = pyqtSignal(int)
    obj_duplicated  = pyqtSignal(int)
    obj_renamed     = pyqtSignal(int, str)
    visibility_toggled    = pyqtSignal(int, bool)
    selectability_toggled = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        
        # --- FUNZIONE HELPER PER DIMENSIONE E POSIZIONE ICONE ---
        def prepara_icona(path, size, offset_x):
            # 1. Canvas ingrandito: 40 (larghezza colonna) x 36 (altezza sufficiente per l'icona da 30)
            canvas = QtGui.QPixmap(40, 36)
            canvas.fill(QtCore.Qt.transparent)
            
            painter = QtGui.QPainter(canvas)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            
            # Carichiamo e ridimensioniamo l'icona sorgente in alta qualità
            src = QtGui.QPixmap(path).scaled(
                size, size, 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
            
            # 2. Offset calcolato sulla nuova altezza del canvas (36)
            offset_y = (36 - src.height()) // 2
            
            # Disegniamo l'icona spostata verso destra del valore 'offset_x'
            painter.drawPixmap(offset_x, offset_y, src)
            painter.end()
            
            return QtGui.QIcon(canvas)
        # --------------------------------------------------------

        # --- NUOVO HEADER CON ICONE DA PATH ---
        header_item = QTreeWidgetItem()
        
        icona_visibile = prepara_icona("interfaccia\\icone/visibile.png", size=45, offset_x=4)
        icona_lucchetto = prepara_icona("interfaccia\\icone/lucchetto.png", size=30, offset_x=10)
        
        header_item.setIcon(0, icona_visibile)
        header_item.setIcon(1, icona_lucchetto)
        header_item.setText(2, " Oggetti")
        
        self.setHeaderItem(header_item)
        
        # 3. Sblocca il limite di Qt aggiornandolo alle nuove dimensioni massime
        self.header().setIconSize(QtCore.QSize(40, 36))
        # --------------------------------------

        # 4. Colonne allargate a 40 per contenere l'icona + l'offset senza tagli
        self.setColumnWidth(0, 40)
        self.setColumnWidth(1, 40)
        self.header().setSectionResizeMode(0, QHeaderView.Fixed)
        self.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self.header().setMinimumSectionSize(40) # Aggiornato a 40
        self.setIndentation(0) # Nasconde le freccette di sistema
        self.setRootIsDecorated(False)
        self.setStyleSheet(_OUTLINER_SHEET)
        
        # Passiamo l'istanza dell'albero al delegate
        self.setItemDelegate(_OutlinerDelegate(self))
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._ctx_menu)
        self.itemDoubleClicked.connect(self._on_double_click)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self._loading = False

        self._id_to_item: dict[int, QTreeWidgetItem] = {}
        self._item_to_id: dict[int, int]             = {}

        self.currentItemChanged.connect(self._on_current_changed)

        self._folders: dict[str, QTreeWidgetItem] = {}
        for f in _FOLD_ORDER:
            item = QTreeWidgetItem(["", "", f])
            item.setData(0, _ROLE_IS_FOLD, True)
            item.setFlags(Qt.ItemIsEnabled)
            font = item.font(2)
            font.setBold(True)
            font.setPointSize(9)
            item.setFont(2, font)
            item.setForeground(2, QBrush(QColor(200, 200, 200)))
            
            # Sfondo delle righe cartella a 40,40,40
            bg = QBrush(QColor(40, 40, 40)) 
            for col in range(3):
                item.setBackground(col, bg)
                
            self.addTopLevelItem(item)
            item.setExpanded(True)
            self._folders[f] = item

    # ------------------------------------------------------------------ public

    def ricarica(self, oggetti: list):
        """Ricostruisce l'albero garantendo la distruzione pulita della UI (anti-sovrapposizione)."""
        self._loading = True

        sel_id = self._current_id()

        self._id_to_item.clear()
        self._item_to_id.clear()
        
        # FIX SOVRAPPOSIZIONE: takeChildren() estirpa completamente i figli vecchi 
        # impedendo conflitti di rendering e "fantasmi" in memoria.
        for fi in self._folders.values():
            fi.takeChildren()

        item_to_restore = None
        for obj in oggetti:
            folder_name = _TIPO_A_FOLDER.get(obj.tipo, _FOLD_STRUTTURALI)
            parent_item = self._folders.get(folder_name)
            if parent_item is None:
                continue

            name_text = obj.nome
            if obj.materiale:
                name_text += f"  ({obj.materiale})"

            item = QTreeWidgetItem(["", "", name_text])
            item.setData(0, _ROLE_IS_FOLD,    False)
            item.setData(0, _ROLE_OBJ_ID,     obj.id)
            item.setData(0, _ROLE_VISIBLE,    bool(obj.visibile))
            item.setData(1, _ROLE_SELECTABLE, bool(obj.selezionabile))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            parent_item.addChild(item)

            self._id_to_item[obj.id]    = item
            self._item_to_id[id(item)]  = obj.id

            if obj.id == sel_id:
                item_to_restore = item

        if item_to_restore is not None:
            self.setCurrentItem(item_to_restore)
        else:
            self.setCurrentItem(None) # Pulisce la selezione residua

        self._loading = False

    def seleziona_oggetto(self, obj_id: int):
        it = self._id_to_item.get(obj_id)
        if it is None or it is self.currentItem():
            return
        self._loading = True
        self.setCurrentItem(it)
        self._loading = False

    def aggiorna_nome(self, obj_id: int, nuovo_nome: str, materiale: str = ""):
        it = self._id_to_item.get(obj_id)
        if it is not None:
            text = nuovo_nome + (f"  ({materiale})" if materiale else "")
            it.setText(2, text)

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, event):
        if self._loading:
            super().mousePressEvent(event)
            return

        item = self.itemAt(event.pos())
        if item is None:
            super().mousePressEvent(event)
            return

        # LOGICA ESPANSIONE CARTELLE TRAMITE PALLINO
        if item.data(0, _ROLE_IS_FOLD):
            col = self.header().logicalIndexAt(event.pos().x())
            if col == 0:
                item.setExpanded(not item.isExpanded())
                self.viewport().update()
                return
            # Cliccando sul nome della cartella espandiamo/chiudiamo uguale (opzionale ma comodo)
            item.setExpanded(not item.isExpanded())
            self.viewport().update()
            return

        obj_id = self._item_to_id.get(id(item))
        if obj_id is None:
            super().mousePressEvent(event)
            return

        col = self.header().logicalIndexAt(event.pos().x())

        if col == 0:
            vis = bool(item.data(0, _ROLE_VISIBLE))
            item.setData(0, _ROLE_VISIBLE, not vis)
            self.viewport().update()
            self.visibility_toggled.emit(obj_id, not vis)
            return

        if col == 1:
            sel = bool(item.data(1, _ROLE_SELECTABLE))
            item.setData(1, _ROLE_SELECTABLE, not sel)
            self.viewport().update()
            self.selectability_toggled.emit(obj_id, not sel)
            return

        if not bool(item.data(1, _ROLE_SELECTABLE)):
            return
        super().mousePressEvent(event)

    # ------------------------------------------------------------------ events

    def _on_current_changed(self, current: QTreeWidgetItem, previous: QTreeWidgetItem):
        if self._loading or current is None:
            return
        if current.data(0, _ROLE_IS_FOLD):
            return
        obj_id = self._item_to_id.get(id(current))
        if obj_id is None:
            obj_id = current.data(0, _ROLE_OBJ_ID)
        if obj_id is not None:
            self.obj_selected.emit(obj_id)

    def _on_double_click(self, item: QTreeWidgetItem, col: int):
        if col != 2 or item.data(0, _ROLE_IS_FOLD):
            return
        obj_id = self._item_to_id.get(id(item))
        if obj_id is None:
            obj_id = item.data(0, _ROLE_OBJ_ID)
        if obj_id is None:
            return
        full_text = item.text(2)
        nome = full_text.split("  (")[0].strip()
        nuovo, ok = QInputDialog.getText(self, "Rinomina", "Nuovo nome:", text=nome)
        if ok and nuovo.strip() and nuovo.strip() != nome:
            self.obj_renamed.emit(obj_id, nuovo.strip())

    def _ctx_menu(self, pos):
        item = self.itemAt(pos)
        if item is None or item.data(0, _ROLE_IS_FOLD):
            return
        obj_id = self._item_to_id.get(id(item))
        if obj_id is None:
            obj_id = item.data(0, _ROLE_OBJ_ID)
        if obj_id is None:
            return
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:rgb(50,50,50);color:#ddd;border:1px solid #666}"
            "QMenu::item:selected{background:rgb(70,100,150)}"
        )
        act_ren = menu.addAction("Rinomina")
        act_dup = menu.addAction("Duplica")
        menu.addSeparator()
        act_del = menu.addAction("Elimina")
        act = menu.exec_(self.viewport().mapToGlobal(pos))
        if act == act_ren:
            self._on_double_click(item, 2)
        elif act == act_dup:
            self.obj_duplicated.emit(obj_id)
        elif act == act_del:
            self.obj_deleted.emit(obj_id)

    def _current_id(self) -> int:
        cur = self.currentItem()
        if cur is None:
            return -1
        from_dict = self._item_to_id.get(id(cur), -1)
        if from_dict != -1:
            return from_dict
        role_id = cur.data(0, _ROLE_OBJ_ID)
        return role_id if role_id is not None else -1


# ============================================================
#  VERTEX EDIT WIDGET 
# ============================================================

class _VertexHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, doc, ref_idx: int = 0):
        super().__init__(doc)
        self._ref_idx = ref_idx
        self._fmt_ref = QtGui.QTextCharFormat()
        self._fmt_ref.setForeground(QColor(255, 80, 80))

    def set_ref_idx(self, idx: int):
        if idx != self._ref_idx:
            self._ref_idx = idx
            self.rehighlight()

    def highlightBlock(self, text: str):
        for m in re.finditer(r'v(\d+)\([^)]*\)', text):
            v_num = int(m.group(1)) - 1
            if v_num == self._ref_idx:
                self.setFormat(m.start(), m.end() - m.start(), self._fmt_ref)


class _VertexEdit(QtWidgets.QPlainTextEdit):
    returnPressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setStyleSheet(
            "QPlainTextEdit{"
            "background-color:rgb(40,40,40);color:rgb(221,221,221);"
            "border:1px solid rgb(120,120,120);"
            "border-left:3px solid rgb(120,120,120);"
            "border-radius:6px;padding:2px 4px;font:9pt Consolas;}"
        )
        self._highlighter = _VertexHighlighter(self.document(), ref_idx=0)

    def text(self) -> str:
        return self.toPlainText().replace("\n", " ").strip()

    def setText(self, text: str):
        self.blockSignals(True)
        self.setPlainText(text)
        self.blockSignals(False)

    def clear(self):
        self.blockSignals(True)
        super().clear()
        self.blockSignals(False)

    def set_ref_idx(self, idx: int):
        self._highlighter.set_ref_idx(idx)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.returnPressed.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


# ============================================================
#  CONTROLLER ELEMENTI (index 6)
# ============================================================

class ControllerElementi(QObject):
    richiedi_salvataggio = pyqtSignal()
    richiedi_preview     = pyqtSignal()

    def __init__(self, ui, main_window):
        super().__init__(main_window)
        self._ui   = ui
        self._main = main_window

        self._elem_corrente: Elemento | None = None
        self._loading = False
        self._vertex_edit: _VertexEdit | None = None

        self._spazio = ElementiSpazio3D()
        self._spazio.selection_changed.connect(self._on_selezione_cambiata)
        self._spazio.oggetto_modificato.connect(self._on_oggetto_modificato)

        self._outliner = Outliner()
        self._outliner.obj_selected.connect(self._seleziona_oggetto_da_outliner)
        self._outliner.visibility_toggled.connect(self._on_visibilita)
        self._outliner.selectability_toggled.connect(self._on_selezionabilita)
        self._outliner.obj_deleted.connect(self._elimina_oggetto)
        self._outliner.obj_duplicated.connect(self._duplica_oggetto)
        self._outliner.obj_renamed.connect(self._rinomina_oggetto)

        self._mod_mov  = QStandardItemModel(2, 3)
        self._mod_geom = QStandardItemModel(1, 1)

        self._setup_spazio()
        self._setup_outliner()
        self._setup_tabelle()
        self._setup_connessioni()
        self._setup_linedit()
        self._setup_combobox()
        self._svuota_pannello()

    def _setup_spazio(self):
        from PyQt5.QtWidgets import QVBoxLayout
        cont = self._ui.elemento_widget
        lay  = cont.layout()
        if lay is None:
            lay = QVBoxLayout(cont)
            lay.setContentsMargins(1, 1, 1, 1)
        lay.addWidget(self._spazio)
        self._tools_mgr = ToolsManager(self._ui, self._spazio)

    def _setup_outliner(self):
        from PyQt5.QtWidgets import QVBoxLayout
        cont = self._ui.elemento_widget_oggetti
        lay  = cont.layout()
        if lay is None:
            lay = QVBoxLayout(cont)
            lay.setContentsMargins(1, 1, 1, 1)
        lay.addWidget(self._outliner)

    def _setup_tabelle(self):
        ET = QAbstractItemView.DoubleClicked | QAbstractItemView.AnyKeyPressed

        tv = self._ui.tableView_movimento
        self._mod_mov.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self._mod_mov.setVerticalHeaderLabels(["Posizione", "Rotazione"])
        tv.setModel(self._mod_mov)
        tv.setStyleSheet(_STILE_TV)
        tv.setEditTriggers(ET)
        hh = tv.horizontalHeader()
        for c in range(3):
            hh.setSectionResizeMode(c, QHeaderView.Stretch)
        tv.verticalHeader().setDefaultSectionSize(28)
        self._mod_mov.itemChanged.connect(self._on_tabella_mov_cambiata)

        tv2 = self._ui.tableView_geometria
        tv2.setModel(self._mod_geom)
        tv2.setStyleSheet(_STILE_TV)
        tv2.setEditTriggers(ET)
        self._mod_geom.itemChanged.connect(self._on_tabella_geom_cambiata)

    def _setup_connessioni(self):
        self._ui.elemento_btn_parallelepipedo.clicked.connect(
            lambda: self._aggiungi_oggetto("parallelepipedo"))
        self._ui.elemento_btn_cilindro.clicked.connect(
            lambda: self._aggiungi_oggetto("cilindro"))
        self._ui.elemento_btn_sfera.clicked.connect(
            lambda: self._aggiungi_oggetto("sfera"))
        self._ui.elemento_btn_barra.clicked.connect(
            lambda: self._aggiungi_oggetto("barra"))
        self._ui.elemento_btn_staffa.clicked.connect(
            lambda: self._aggiungi_oggetto("staffa"))

        self._ui.btn_elemento_centra.clicked.connect(self._spazio.centra_vista)

        self._ui.elemento_combobox_materiale.currentTextChanged.connect(
            self._on_materiale_cambiato)

    def _setup_linedit(self):
        old_le = self._ui.elemento_modifiche_lineEdit
        parent = old_le.parent()
        self._vertex_edit = _VertexEdit(parent)
        self._vertex_edit.setPlaceholderText("v1(x,y,z), v2(x,y,z), ...")
        self._vertex_edit.returnPressed.connect(self._on_linedit_apply)

        layout = parent.layout()
        replaced = False
        if layout is not None:
            idx = layout.indexOf(old_le)
            if idx >= 0:
                layout.takeAt(idx)
                old_le.hide()
                old_le.setParent(None)
                layout.insertWidget(idx, self._vertex_edit)
                replaced = True
            else:
                from PyQt5.QtWidgets import QGridLayout
                if isinstance(layout, QGridLayout):
                    for r in range(layout.rowCount()):
                        for c in range(layout.columnCount()):
                            item = layout.itemAtPosition(r, c)
                            if item and item.widget() is old_le:
                                rs = layout.rowSpan(r, c)
                                cs = layout.columnSpan(r, c)
                                layout.removeWidget(old_le)
                                old_le.hide()
                                old_le.setParent(None)
                                layout.addWidget(self._vertex_edit, r, c, rs, cs)
                                replaced = True
                                break
                        if replaced:
                            break
        if not replaced:
            old_le.hide()
            self._vertex_edit.setGeometry(old_le.geometry())
            self._vertex_edit.show()

    def _setup_combobox(self):
        self._aggiorna_combobox()

    def carica_elemento(self, el: Elemento | None):
        self._elem_corrente = el
        if el is None:
            self._spazio.aggiorna_oggetti([])
            self._outliner.ricarica([])
            self._spazio.set_id_selezionato(-1) # Clean UI sync
            self._svuota_pannello()
            return
        self._spazio.aggiorna_oggetti(el.oggetti)
        self._spazio.set_id_selezionato(-1)
        self._outliner.ricarica(el.oggetti)
        self._aggiorna_combobox()
        self._svuota_pannello()
        self._spazio.centra_vista()
        self._ui.stackedWidget_main.setCurrentIndex(6)

    def svuota(self):
        self._elem_corrente = None
        self._spazio.aggiorna_oggetti([])
        self._outliner.ricarica([])
        self._spazio.set_id_selezionato(-1)
        self._svuota_pannello()

    def get_spazio(self) -> ElementiSpazio3D:
        return self._spazio

    def get_elem_corrente(self) -> Elemento | None:
        return self._elem_corrente

    def _aggiungi_oggetto(self, tipo: str):
        if self._elem_corrente is None:
            QMessageBox.information(self._main, "Info",
                                    "Seleziona prima un elemento dalla lista.")
            return

        self._aggiorna_combobox(tipo)
        mat = self._ui.elemento_combobox_materiale.currentText()
        obj = Oggetto3D(tipo, materiale=mat)
        
        # LOGICA NUMERAZIONE VERGINE PER SCENA
        # Calcoliamo il numero in base a quelli già presenti in QUESTO workspace
        tipo_formattato = tipo.capitalize()
        esistenti = [o.nome for o in self._elem_corrente.oggetti if o.nome.startswith(tipo_formattato)]
        numeri = []
        for nome_es in esistenti:
            match = re.search(r'(\d+)$', nome_es)
            if match:
                numeri.append(int(match.group(1)))
        
        prossimo_numero = max(numeri) + 1 if numeri else 1
        obj.nome = f"{tipo_formattato} {prossimo_numero}"

        self._elem_corrente.oggetti.append(obj)

        self._spazio.aggiorna_oggetti(self._elem_corrente.oggetti)
        self._outliner.ricarica(self._elem_corrente.oggetti)
        self._spazio.set_id_selezionato(obj.id)
        self.richiedi_salvataggio.emit()
        print(f">> Oggetto aggiunto: {obj.nome} → {self._elem_corrente.nome}")

    def _elimina_oggetto(self, obj_id: int):
        if self._elem_corrente is None:
            return
        r = QMessageBox.question(
            self._main, "Elimina oggetto", "Eliminare l'oggetto selezionato?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if r != QMessageBox.Yes:
            return
        self._elem_corrente.oggetti = [
            o for o in self._elem_corrente.oggetti if o.id != obj_id
        ]
        self._spazio.aggiorna_oggetti(self._elem_corrente.oggetti)
        self._spazio.set_id_selezionato(-1)
        self._outliner.ricarica(self._elem_corrente.oggetti)
        self._svuota_pannello()
        self.richiedi_salvataggio.emit()

    def _duplica_oggetto(self, obj_id: int):
        if self._elem_corrente is None:
            return
        obj = self._spazio.get_oggetto(obj_id)
        if obj is None:
            return
        nuovo = obj.duplica()
        
        # Facoltativo: Riassegna il nome vergine anche quando duplichi per evitare stringhe come "Parallelepipedo 1_copy"
        tipo_formattato = nuovo.tipo.capitalize()
        esistenti = [o.nome for o in self._elem_corrente.oggetti if o.nome.startswith(tipo_formattato)]
        numeri = [int(re.search(r'(\d+)$', n).group(1)) for n in esistenti if re.search(r'(\d+)$', n)]
        prox = max(numeri) + 1 if numeri else 1
        nuovo.nome = f"{tipo_formattato} {prox}"
        
        idx = next((i for i, o in enumerate(self._elem_corrente.oggetti) if o.id == obj_id), -1)
        if idx == -1:
            self._elem_corrente.oggetti.append(nuovo)
        else:
            self._elem_corrente.oggetti.insert(idx + 1, nuovo)
            
        self._spazio.aggiorna_oggetti(self._elem_corrente.oggetti)
        self._outliner.ricarica(self._elem_corrente.oggetti)
        self._spazio.set_id_selezionato(nuovo.id)
        self.richiedi_salvataggio.emit()
        print(f">> Oggetto duplicato: {nuovo.nome}")

    def _rinomina_oggetto(self, obj_id: int, nuovo_nome: str):
        obj = self._spazio.get_oggetto(obj_id)
        if obj is None:
            return
        obj.nome = nuovo_nome
        self._outliner.aggiorna_nome(obj_id, nuovo_nome, obj.materiale)
        self.richiedi_salvataggio.emit()

    def _on_selezione_cambiata(self, obj_id: int):
        self._outliner.seleziona_oggetto(obj_id)
        self._aggiorna_pannello(obj_id)

    def _seleziona_oggetto_da_outliner(self, obj_id: int):
        obj = self._spazio.get_oggetto(obj_id)
        if obj is not None and not obj.selezionabile:
            return
        self._spazio.set_id_selezionato(obj_id)
        self._aggiorna_pannello(obj_id)

    def _on_oggetto_modificato(self, obj_id: int):
        self._aggiorna_pannello(obj_id)
        self.richiedi_salvataggio.emit()
        self.richiedi_preview.emit()

    def _aggiorna_pannello(self, obj_id: int):
        if obj_id == -1:
            self._svuota_pannello(); return

        obj = self._spazio.get_oggetto(obj_id)
        if obj is None:
            self._svuota_pannello(); return

        # Aggiorna label nome oggetto selezionato
        self._ui.elemento_label_oggetto.setText(f"Proprietà: {obj.nome}")

        self._loading = True

        self._mod_mov.blockSignals(True)
        v1_world = obj.get_vertex_ref_world()
        rx, ry, rz = obj.rotazione
        for ci, val in enumerate(v1_world):
            self._mod_mov.setItem(0, ci, _make_item(f"{val:.4f}"))
        for ci, val in enumerate([rx, ry, rz]):
            self._mod_mov.setItem(1, ci, _make_item(f"{val:.4f}"))
        self._mod_mov.blockSignals(False)
        self._ui.tableView_movimento.viewport().update()

        self._mod_geom.blockSignals(True)
        tv2 = self._ui.tableView_geometria
        if obj.custom_geometry:
            self._mod_geom.setColumnCount(1)
            self._mod_geom.setRowCount(1)
            self._mod_geom.setHorizontalHeaderLabels(["Geometria personalizzata"])
            self._mod_geom.setItem(0, 0, _make_item("non disponibile", editable=False))
        else:
            col_defs = _GEOM_COLS.get(obj.tipo, [])
            n_cols   = max(len(col_defs), 1)
            self._mod_geom.setColumnCount(n_cols)
            self._mod_geom.setRowCount(1)
            self._mod_geom.setHorizontalHeaderLabels([c[0] for c in col_defs])
            for ci, (_, key) in enumerate(col_defs):
                val = obj.geometria.get(key, 0.0)
                self._mod_geom.setItem(0, ci, _make_item(f"{val:.4f}"))
        tv2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._mod_geom.blockSignals(False)
        tv2.viewport().update()

        self._update_vertex_edit(obj)

        cb = self._ui.elemento_combobox_materiale
        cb.blockSignals(True)
        self._aggiorna_combobox(obj.tipo)
        idx = cb.findText(obj.materiale)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        cb.blockSignals(False)

        self._loading = False

    def _update_vertex_edit(self, obj):
        if self._vertex_edit is None:
            return
        verts_world = obj.get_vertices_world()
        ref_idx     = getattr(obj, 'vertice_ref', 0)
        plain = ", ".join(
            f"v{i+1}({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})"
            for i, v in enumerate(verts_world)
        )
        self._vertex_edit.set_ref_idx(ref_idx)
        self._vertex_edit.setText(plain)

    def _svuota_pannello(self):
        self._ui.elemento_label_oggetto.setText("Proprietà: -")
        self._loading = True
        self._mod_mov.blockSignals(True)
        self._mod_geom.blockSignals(True)
        for r in range(2):
            for c in range(3):
                self._mod_mov.setItem(r, c, _make_item("—", editable=False))
        self._mod_geom.setColumnCount(1)
        self._mod_geom.setHorizontalHeaderLabels(["—"])
        self._mod_geom.setItem(0, 0, _make_item("—", editable=False))
        self._mod_mov.blockSignals(False)
        self._mod_geom.blockSignals(False)
        if self._vertex_edit is not None:
            self._vertex_edit.clear()
        self._loading = False

    def _on_tabella_mov_cambiata(self, item: QStandardItem):
        if self._loading:
            return
        obj_id = self._spazio.get_id_selezionato()
        obj    = self._spazio.get_oggetto(obj_id)
        if obj is None:
            return

        def _f(row, col):
            it = self._mod_mov.item(row, col)
            if it is None:
                return 0.0
            try:
                return float(it.text().replace(",", ".").strip())
            except ValueError:
                return 0.0

        v1_world_desired = [_f(0, 0), _f(0, 1), _f(0, 2)]
        new_rot          = [_f(1, 0), _f(1, 1), _f(1, 2)]

        self._loading = True
        obj.rotazione = new_rot

        verts_local = obj.get_vertices_local()
        if verts_local:
            v1_local = verts_local[0]
            v1_rotated = ruota_punto(v1_local, *new_rot)
            obj.posizione = [v1_world_desired[i] - v1_rotated[i] for i in range(3)]
        else:
            obj.posizione = v1_world_desired
        self._loading = False

        self._spazio.update()
        self._update_vertex_edit(obj)
        self.richiedi_salvataggio.emit()

    def _on_tabella_geom_cambiata(self, item: QStandardItem):
        if self._loading:
            return
        obj_id = self._spazio.get_id_selezionato()
        obj    = self._spazio.get_oggetto(obj_id)
        if obj is None:
            return

        col_defs = _GEOM_COLS.get(obj.tipo, [])
        col = item.column()
        if col >= len(col_defs):
            return

        key = col_defs[col][1]
        try:
            val = float(item.text().replace(",", ".").strip())
        except ValueError:
            return

        self._loading = True
        obj.geometria[key] = val
        obj.custom_geometry = False
        self._loading = False

        self._spazio.update()
        self._update_vertex_edit(obj)
        self.richiedi_salvataggio.emit()

    def _aggiorna_combobox(self, tipo_oggetto: str = None):
        cb = self._ui.elemento_combobox_materiale
        cb.blockSignals(True)
        old_text = cb.currentText()
        cb.clear()

        if tipo_oggetto is None:
            cb.blockSignals(False)
            return

        mats = self._main.get_sezione("materiali")
        is_armatura = tipo_oggetto in TIPO_ARMATURA

        if not is_armatura:
            for nome in mats.get("calcestruzzo", {}):
                cb.addItem(nome)
            for nome in mats.get("acciaio", {}):
                cb.addItem(nome)
        else:
            for nome in mats.get("barre", {}):
                cb.addItem(nome)

        for nome in mats.get("personalizzati", {}):
            cb.addItem(nome)

        idx = cb.findText(old_text)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        cb.blockSignals(False)

    def _on_materiale_cambiato(self, nome: str):
        if self._loading:
            return
        obj_id = self._spazio.get_id_selezionato()
        obj    = self._spazio.get_oggetto(obj_id)
        if obj is None:
            return
        obj.materiale = nome
        self._outliner.aggiorna_nome(obj_id, obj.nome, nome)
        self.richiedi_salvataggio.emit()

    def _on_linedit_apply(self):
        if self._vertex_edit is None:
            return
        text = self._vertex_edit.text()
        if not text:
            return
        obj_id = self._spazio.get_id_selezionato()
        obj    = self._spazio.get_oggetto(obj_id)
        if obj is None:
            return

        world_verts = _parse_vertici_text(text)
        if world_verts is None:
            QMessageBox.warning(
                self._main, "Formato non riconosciuto",
                "Usa il formato: v1(x,y,z), v2(x,y,z), ..."
            )
            return

        local_verts = [detrasforma_punto(w, obj.posizione, obj.rotazione)
                       for w in world_verts]

        if obj.tipo in TIPO_ARMATURA:
            obj.geometria["punti"] = local_verts
            self._spazio._emit_modificato(obj_id)
            return

        if not obj.custom_geometry:
            r = QMessageBox.question(
                self._main,
                "Conferma modifica geometrica",
                "Confermi la modifica geometrica?\n\n"
                "Perderai la possibilità di modificare la geometria tramite le "
                "proprietà standard. L'oggetto diventerà una geometria personalizzata "
                "definita unicamente dai suoi vertici.\n\n"
                "La traslazione e rotazione resteranno disponibili.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                self._update_vertex_edit(obj)
                return

        obj.set_vertices_custom(local_verts)
        self._spazio._emit_modificato(obj_id)
        self._mod_geom.blockSignals(True)
        self._mod_geom.setColumnCount(1)
        self._mod_geom.setHorizontalHeaderLabels(["Geometria personalizzata"])
        self._mod_geom.setItem(0, 0, _make_item("—", editable=False))
        self._mod_geom.blockSignals(False)

    def _on_visibilita(self, obj_id: int, vis: bool):
        obj = self._spazio.get_oggetto(obj_id)
        if obj:
            obj.visibile = vis
            self._spazio.update()

    def _on_selezionabilita(self, obj_id: int, sel: bool):
        obj = self._spazio.get_oggetto(obj_id)
        if obj:
            obj.selezionabile = sel