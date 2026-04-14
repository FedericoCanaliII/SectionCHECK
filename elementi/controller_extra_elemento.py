"""
controller_extra_elemento.py – Controller for the carichi/vincoli workspace (index 7).

Responsibilities:
  • Embed ExtraSpazio3D into extra_elemento_widget
  • Embed OutlinerCV into extra_widget_oggetti
  • Wire extra_tableView_movimento / geometria / caratteristiche
  • Wire extra_elemento_modifiche_lineEdit (vertex editor)
  • Wire tool buttons via ToolsManagerExtra
  • Wire view buttons + centra
  • Manage add/delete/duplicate/rename carichi-vincoli per elemento
  • Expose carica_elemento(el) to load the reference element and its CV objects
  • Expose get_tutti_carichi() for persistence
  • ricarica_da_progetto(dati) to reload from saved project
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

from .modello_carichi_vincoli import CaricoVincolo
from .extra_spazio_3d         import ExtraSpazio3D
from .tools_manager_extra     import ToolsManagerExtra
from .modello_3d              import ruota_punto, detrasforma_punto


# ============================================================
#  SHARED STYLES  (mirror controller_elementi.py)
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

# ============================================================
#  ITEM ROLES
# ============================================================

_ROLE_OBJ_ID     = Qt.UserRole + 1
_ROLE_IS_FOLD    = Qt.UserRole + 2
_ROLE_VISIBLE    = Qt.UserRole + 3
_ROLE_SELECTABLE = Qt.UserRole + 4
_ROLE_FOLD_COLOR = Qt.UserRole + 5   # QColor for folder indicator


# ============================================================
#  OUTLINER DELEGATE  (carichi/vincoli version)
# ============================================================

class _OutlinerDelegateCV(QStyledItemDelegate):
    def __init__(self, tree_widget):
        super().__init__(tree_widget)
        self.tree = tree_widget

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        col = index.column()
        if col not in (0, 1):
            return

        is_fold = index.data(_ROLE_IS_FOLD)
        r  = option.rect
        d  = 10
        x  = r.x() + (r.width()  - d) // 2
        y  = r.y() + (r.height() - d) // 2

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        if is_fold:
            if col == 0:
                item = self.tree.itemFromIndex(index)
                if item:
                    fold_col = item.data(0, _ROLE_FOLD_COLOR)
                    if isinstance(fold_col, QColor) and item.isExpanded():
                        fill   = fold_col
                        border = fold_col.lighter(130)
                    else:
                        fill   = QColor(52, 52, 52)
                        border = QColor(78, 78, 78)
                    painter.setPen(QPen(border, 1.0))
                    painter.setBrush(QBrush(fill))
                    painter.drawEllipse(x, y, d, d)
            painter.restore()
            return

        if col == 0:
            on     = bool(index.data(_ROLE_VISIBLE))
            fill   = QColor(88, 195, 88)  if on else QColor(52, 52, 52)
            border = QColor(115,225,115)  if on else QColor(78, 78, 78)
        else:
            on     = bool(index.data(_ROLE_SELECTABLE))
            fill   = QColor(65, 135, 215) if on else QColor(52, 52, 52)
            border = QColor(95, 165, 245) if on else QColor(78, 78, 78)

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
#  OUTLINER CV
# ============================================================

_FOLD_VINCOLI = "Vincoli"
_FOLD_CARICHI = "Carichi"
_FOLD_ORDER_CV = [_FOLD_VINCOLI, _FOLD_CARICHI]

_SOTTOTIPO_A_FOLDER = {
    "vincolo": _FOLD_VINCOLI,
    "carico":  _FOLD_CARICHI,
}

_FOLD_COLORS = {
    _FOLD_VINCOLI: QColor(140, 50, 210),   # purple (viola) – sia vincoli che carichi
    _FOLD_CARICHI: QColor(140, 50, 210),   # purple (viola)
}


class OutlinerCV(QTreeWidget):
    """Outliner dedicated to CaricoVincolo objects."""

    obj_selected          = pyqtSignal(int)
    obj_deleted           = pyqtSignal(int)
    obj_duplicated        = pyqtSignal(int)
    obj_renamed           = pyqtSignal(int, str)
    visibility_toggled    = pyqtSignal(int, bool)
    selectability_toggled = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        
        # --- FUNZIONE HELPER PER DIMENSIONE E POSIZIONE ICONE ---
        def prepara_icona(path, size, offset_x):
            # 1. Canvas ingrandito: 40 (larghezza colonna) x 36 (altezza sufficiente per l'icona da 30/45)
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
        self.setIndentation(0)
        self.setRootIsDecorated(False)
        self.setStyleSheet(_OUTLINER_SHEET)
        
        self.setItemDelegate(_OutlinerDelegateCV(self))
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._ctx_menu)
        self.itemDoubleClicked.connect(self._on_double_click)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self._loading = False
        
        self._id_to_item: dict[int, QTreeWidgetItem] = {}
        self._item_to_id: dict[int, int]             = {}

        self.currentItemChanged.connect(self._on_current_changed)

        self._folders: dict[str, QTreeWidgetItem] = {}
        for f in _FOLD_ORDER_CV:
            item = QTreeWidgetItem(["", "", f])
            item.setData(0, _ROLE_IS_FOLD, True)
            col = _FOLD_COLORS.get(f, QColor(120, 120, 120))
            item.setData(0, _ROLE_FOLD_COLOR, col)
            item.setFlags(Qt.ItemIsEnabled)
            font = item.font(2)
            font.setBold(True)
            font.setPointSize(9)
            item.setFont(2, font)
            item.setForeground(2, QBrush(QColor(200, 200, 200)))
            
            # ---> Sfondo delle righe cartella a 40,40,40 <---
            bg = QBrush(QColor(40, 40, 40)) 
            for c in range(3):
                item.setBackground(c, bg)
                
            self.addTopLevelItem(item)
            item.setExpanded(True)
            self._folders[f] = item

    # ------------------------------------------------------------------ public

    def ricarica(self, oggetti: list):
        self._loading = True
        sel_id = self._current_id()
        self._id_to_item.clear()
        self._item_to_id.clear()
        for fi in self._folders.values():
            fi.takeChildren()

        item_to_restore = None
        for cv in oggetti:
            folder_name = _SOTTOTIPO_A_FOLDER.get(cv.sottotipo, _FOLD_VINCOLI)
            parent_item = self._folders.get(folder_name)
            if parent_item is None:
                continue
            item = QTreeWidgetItem(["", "", cv.nome])
            item.setData(0, _ROLE_IS_FOLD,    False)
            item.setData(0, _ROLE_OBJ_ID,     cv.id)
            item.setData(0, _ROLE_VISIBLE,    bool(cv.visibile))
            item.setData(1, _ROLE_SELECTABLE, bool(cv.selezionabile))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            parent_item.addChild(item)
            self._id_to_item[cv.id]    = item
            self._item_to_id[id(item)] = cv.id
            if cv.id == sel_id:
                item_to_restore = item

        if item_to_restore is not None:
            self.setCurrentItem(item_to_restore)
        else:
            self.setCurrentItem(None)
        self._loading = False

    def seleziona_oggetto(self, cv_id: int):
        it = self._id_to_item.get(cv_id)
        if it is None or it is self.currentItem():
            return
        self._loading = True
        self.setCurrentItem(it)
        self._loading = False

    def aggiorna_nome(self, cv_id: int, nuovo_nome: str):
        it = self._id_to_item.get(cv_id)
        if it is not None:
            it.setText(2, nuovo_nome)

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, event):
        if self._loading:
            super().mousePressEvent(event)
            return
        item = self.itemAt(event.pos())
        if item is None:
            super().mousePressEvent(event)
            return
        if item.data(0, _ROLE_IS_FOLD):
            col = self.header().logicalIndexAt(event.pos().x())
            if col == 0:
                item.setExpanded(not item.isExpanded())
                self.viewport().update()
            else:
                item.setExpanded(not item.isExpanded())
                self.viewport().update()
            return
        cv_id = self._item_to_id.get(id(item))
        if cv_id is None:
            super().mousePressEvent(event)
            return
        col = self.header().logicalIndexAt(event.pos().x())
        if col == 0:
            vis = bool(item.data(0, _ROLE_VISIBLE))
            item.setData(0, _ROLE_VISIBLE, not vis)
            self.viewport().update()
            self.visibility_toggled.emit(cv_id, not vis)
            return
        if col == 1:
            sel = bool(item.data(1, _ROLE_SELECTABLE))
            item.setData(1, _ROLE_SELECTABLE, not sel)
            self.viewport().update()
            self.selectability_toggled.emit(cv_id, not sel)
            return
        if not bool(item.data(1, _ROLE_SELECTABLE)):
            return
        super().mousePressEvent(event)

    # ------------------------------------------------------------------ events

    def _on_current_changed(self, current: QTreeWidgetItem, _previous):
        if self._loading or current is None:
            return
        if current.data(0, _ROLE_IS_FOLD):
            return
        cv_id = self._item_to_id.get(id(current))
        if cv_id is None:
            cv_id = current.data(0, _ROLE_OBJ_ID)
        if cv_id is not None:
            self.obj_selected.emit(cv_id)

    def _on_double_click(self, item: QTreeWidgetItem, col: int):
        if col != 2 or item.data(0, _ROLE_IS_FOLD):
            return
        cv_id = self._item_to_id.get(id(item))
        if cv_id is None:
            cv_id = item.data(0, _ROLE_OBJ_ID)
        if cv_id is None:
            return
        nome    = item.text(2)
        nuovo, ok = QInputDialog.getText(self, "Rinomina", "Nuovo nome:", text=nome)
        if ok and nuovo.strip() and nuovo.strip() != nome:
            self.obj_renamed.emit(cv_id, nuovo.strip())

    def _ctx_menu(self, pos):
        item = self.itemAt(pos)
        if item is None or item.data(0, _ROLE_IS_FOLD):
            return
        cv_id = self._item_to_id.get(id(item))
        if cv_id is None:
            cv_id = item.data(0, _ROLE_OBJ_ID)
        if cv_id is None:
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
            self.obj_duplicated.emit(cv_id)
        elif act == act_del:
            self.obj_deleted.emit(cv_id)

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
#  VERTEX EDIT (reuse from controller_elementi)
# ============================================================

from .controller_elementi import _VertexEdit, _parse_vertici_text


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


# Geometry column defs for the parallelepipedo (only type used for CV)
_GEOM_COLS_CV = [
    ("Base Y [m]",     "base"),
    ("Altezza Z [m]",  "altezza"),
    ("Lunghezza X [m]","lunghezza"),
]


# ============================================================
#  CONTROLLER EXTRA ELEMENTO  (index 7)
# ============================================================

class ControllerExtraElemento(QObject):
    richiedi_salvataggio = pyqtSignal()
    richiedi_preview_cv  = pyqtSignal()   # emesso dopo ogni modifica CV

    def __init__(self, ui, main_window):
        super().__init__(main_window)
        self._ui   = ui
        self._main = main_window

        self._elem_rif    = None   # reference Elemento (read-only)
        self._cv_list: list[CaricoVincolo] = []
        self._loading = False
        self._vertex_edit: _VertexEdit | None = None

        # 3D space
        self._spazio = ExtraSpazio3D()
        self._spazio.selection_changed.connect(self._on_selezione_cambiata)
        self._spazio.oggetto_modificato.connect(self._on_cv_modificato)

        # Outliner
        self._outliner = OutlinerCV()
        self._outliner.obj_selected.connect(self._seleziona_cv_da_outliner)
        self._outliner.visibility_toggled.connect(self._on_visibilita)
        self._outliner.selectability_toggled.connect(self._on_selezionabilita)
        self._outliner.obj_deleted.connect(self._elimina_cv)
        self._outliner.obj_duplicated.connect(self._duplica_cv)
        self._outliner.obj_renamed.connect(self._rinomina_cv)

        # Table models
        self._mod_mov   = QStandardItemModel(2, 3)
        self._mod_geom  = QStandardItemModel(1, 3)
        self._mod_carat = QStandardItemModel(1, 3)

        self._setup_spazio()
        self._setup_outliner()
        self._setup_tabelle()
        self._setup_connessioni()
        self._setup_linedit()
        self._svuota_pannello()

    # ================================================================
    #  SETUP
    # ================================================================

    def _setup_spazio(self):
        cont = self._ui.extra_elemento_widget
        lay  = cont.layout()
        if lay is None:
            lay = QVBoxLayout(cont)
            lay.setContentsMargins(1, 1, 1, 1)
        lay.addWidget(self._spazio)
        self._tools_mgr = ToolsManagerExtra(self._ui, self._spazio)

    def _setup_outliner(self):
        cont = self._ui.extra_widget_oggetti
        lay  = cont.layout()
        if lay is None:
            lay = QVBoxLayout(cont)
            lay.setContentsMargins(1,1,1,1)
        lay.addWidget(self._outliner)

    def _setup_tabelle(self):
        ET = QAbstractItemView.DoubleClicked | QAbstractItemView.AnyKeyPressed

        # -- Movimento (posizione / rotazione) --
        tv = self._ui.extra_tableView_movimento
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

        # -- Geometria --
        tv2 = self._ui.extra_tableView_geometria
        self._mod_geom.setColumnCount(3)
        self._mod_geom.setRowCount(1)
        self._mod_geom.setHorizontalHeaderLabels(
            [c[0] for c in _GEOM_COLS_CV]
        )
        tv2.setModel(self._mod_geom)
        tv2.setStyleSheet(_STILE_TV)
        tv2.setEditTriggers(ET)
        tv2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tv2.verticalHeader().setDefaultSectionSize(28)
        self._mod_geom.itemChanged.connect(self._on_tabella_geom_cambiata)

        # -- Caratteristiche --
        tv3 = self._ui.extra_tableView_caratteristiche
        self._mod_carat.setColumnCount(3)
        self._mod_carat.setRowCount(1)
        tv3.setModel(self._mod_carat)
        tv3.setStyleSheet(_STILE_TV)
        tv3.setEditTriggers(ET)
        tv3.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tv3.verticalHeader().setDefaultSectionSize(28)
        self._mod_carat.itemChanged.connect(self._on_tabella_carat_cambiata)

    def _setup_connessioni(self):
        self._ui.extra_elemento_btn_vincolo.clicked.connect(
            lambda: self._aggiungi_cv("vincolo"))
        self._ui.extra_elemento_btn_carico.clicked.connect(
            lambda: self._aggiungi_cv("carico"))
        self._ui.extra_btn_elemento_centra.clicked.connect(
            self._spazio.centra_vista)

    def _setup_linedit(self):
        old_le = self._ui.extra_elemento_modifiche_lineEdit
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
        if not replaced:
            old_le.hide()
            self._vertex_edit.setGeometry(old_le.geometry())
            self._vertex_edit.show()

    # ================================================================
    #  PUBLIC API
    # ================================================================

    def carica_elemento(self, el):
        """
        Load a reference Elemento and its associated carichi/vincoli.
        Navigates to index 7.
        """
        self._elem_rif = el
        if el is None:
            self._spazio.aggiorna_rif([])
            self._spazio.aggiorna_oggetti([])
            self._outliner.ricarica([])
            self._spazio.set_id_selezionato(-1)
            self._cv_list = []
            self._svuota_pannello()
            return

        # Display reference element as ghost
        self._spazio.aggiorna_rif(el.oggetti)

        # Load CV objects for this element
        self._cv_list = self._get_cv_per_elemento(el.id)
        self._spazio.aggiorna_oggetti(self._cv_list)
        self._spazio.set_id_selezionato(-1)
        self._outliner.ricarica(self._cv_list)
        self._svuota_pannello()
        self._spazio.centra_vista()
        self._ui.stackedWidget_main.setCurrentIndex(7)

    def svuota(self):
        self._elem_rif = None
        self._cv_list  = []
        self._spazio.aggiorna_rif([])
        self._spazio.aggiorna_oggetti([])
        self._outliner.ricarica([])
        self._spazio.set_id_selezionato(-1)
        self._svuota_pannello()

    def get_spazio(self) -> ExtraSpazio3D:
        return self._spazio

    def get_tutti_carichi(self) -> dict:
        """
        Return all carichi/vincoli keyed by elemento ID (str) for persistence.
        """
        return self._carica_da_mem()

    # ================================================================
    #  PERSISTENCE HELPERS
    # ================================================================

    def _chiave(self, el_id: int) -> str:
        return str(el_id)

    def _get_cv_per_elemento(self, el_id: int) -> list:
        """Return CaricoVincolo list for this element from project data."""
        sezione = self._main.get_sezione("carichi")
        raw = sezione.get(self._chiave(el_id), [])
        return [CaricoVincolo.from_dict(d) for d in raw]

    def _salva_cv_correnti(self):
        """Save the current cv_list back to the project."""
        if self._elem_rif is None or not self._main.ha_progetto():
            return
        sezione = self._main.get_sezione("carichi")
        # Clone to avoid mutation issues
        import copy
        sezione = copy.deepcopy(sezione)
        sezione[self._chiave(self._elem_rif.id)] = [
            cv.to_dict() for cv in self._cv_list
        ]
        self._main.push_undo("Modifica carichi/vincoli")
        self._main.set_sezione("carichi", sezione)
        self.richiedi_preview_cv.emit()

    def _carica_da_mem(self) -> dict:
        """Return a fresh snapshot of all carichi/vincoli from project."""
        return self._main.get_sezione("carichi")

    def ricarica_da_progetto(self, dati: dict):
        """
        Called on project load. Resets ID counters and clears in-memory state.
        The per-element data remains in project; carica_elemento() loads lazily.
        """
        CaricoVincolo._id_counter = 0
        CaricoVincolo._nome_count = {}

        # Sync ID counter to max existing ID to prevent collisions
        max_id = 0
        for lista in dati.values():
            if isinstance(lista, list):
                for d in lista:
                    if isinstance(d, dict):
                        max_id = max(max_id, d.get("id", 0))
        if max_id:
            CaricoVincolo._id_counter = max_id

        self.svuota()
        print(">> Modulo Carichi/Vincoli: progetto ricaricato.")

    # ================================================================
    #  ADD / DELETE / DUPLICATE / RENAME
    # ================================================================

    def _aggiungi_cv(self, sottotipo: str):
        if self._elem_rif is None:
            QMessageBox.information(
                self._main, "Info",
                "Seleziona prima un elemento dalla lista (tasto C/V)."
            )
            return
        cv = CaricoVincolo(sottotipo)
        self._cv_list.append(cv)
        self._spazio.aggiorna_oggetti(self._cv_list)
        self._outliner.ricarica(self._cv_list)
        self._spazio.set_id_selezionato(cv.id)
        self._salva_cv_correnti()
        print(f">> CV aggiunto: {cv.nome} → {self._elem_rif.nome}")

    def _elimina_cv(self, cv_id: int):
        r = QMessageBox.question(
            self._main, "Elimina oggetto",
            "Eliminare il carico/vincolo selezionato?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if r != QMessageBox.Yes:
            return
        self._cv_list = [cv for cv in self._cv_list if cv.id != cv_id]
        self._spazio.aggiorna_oggetti(self._cv_list)
        self._spazio.set_id_selezionato(-1)
        self._outliner.ricarica(self._cv_list)
        self._svuota_pannello()
        self._salva_cv_correnti()

    def _duplica_cv(self, cv_id: int):
        cv = self._spazio.get_oggetto(cv_id)
        if cv is None:
            return
        nuovo = cv.duplica()
        idx = next((i for i, o in enumerate(self._cv_list) if o.id == cv_id), -1)
        if idx == -1:
            self._cv_list.append(nuovo)
        else:
            self._cv_list.insert(idx + 1, nuovo)
        self._spazio.aggiorna_oggetti(self._cv_list)
        self._outliner.ricarica(self._cv_list)
        self._spazio.set_id_selezionato(nuovo.id)
        self._salva_cv_correnti()
        print(f">> CV duplicato: {nuovo.nome}")

    def _rinomina_cv(self, cv_id: int, nuovo_nome: str):
        cv = self._spazio.get_oggetto(cv_id)
        if cv is None:
            return
        cv.nome = nuovo_nome
        self._outliner.aggiorna_nome(cv_id, nuovo_nome)
        self._salva_cv_correnti()

    # ================================================================
    #  SELECTION / MODIFICATION
    # ================================================================

    def _on_selezione_cambiata(self, cv_id: int):
        self._outliner.seleziona_oggetto(cv_id)
        self._aggiorna_pannello(cv_id)

    def _seleziona_cv_da_outliner(self, cv_id: int):
        cv = self._spazio.get_oggetto(cv_id)
        if cv is not None and not cv.selezionabile:
            return
        self._spazio.set_id_selezionato(cv_id)
        self._aggiorna_pannello(cv_id)

    def _on_cv_modificato(self, cv_id: int):
        self._aggiorna_pannello(cv_id)
        self._salva_cv_correnti()

    def _on_visibilita(self, cv_id: int, vis: bool):
        cv = self._spazio.get_oggetto(cv_id)
        if cv:
            cv.visibile = vis
            self._spazio.update()

    def _on_selezionabilita(self, cv_id: int, sel: bool):
        cv = self._spazio.get_oggetto(cv_id)
        if cv:
            cv.selezionabile = sel

    # ================================================================
    #  PANEL UPDATE
    # ================================================================

    def _aggiorna_pannello(self, cv_id: int):
        if cv_id == -1:
            self._svuota_pannello()
            return
        cv = self._spazio.get_oggetto(cv_id)
        if cv is None:
            self._svuota_pannello()
            return

        # Aggiorna label nome oggetto selezionato
        self._ui.extra_elemento_label_oggetto.setText(f"Proprietà: {cv.nome}")

        self._loading = True

        # ---- Movimento ----
        self._mod_mov.blockSignals(True)
        v1_world = cv.get_vertex_ref_world()
        rx, ry, rz = cv.rotazione
        for ci, val in enumerate(v1_world):
            self._mod_mov.setItem(0, ci, _make_item(f"{val:.4f}"))
        for ci, val in enumerate([rx, ry, rz]):
            self._mod_mov.setItem(1, ci, _make_item(f"{val:.4f}"))
        self._mod_mov.blockSignals(False)
        self._ui.extra_tableView_movimento.viewport().update()

        # ---- Geometria ----
        self._mod_geom.blockSignals(True)
        if cv.custom_geometry:
            self._mod_geom.setColumnCount(1)
            self._mod_geom.setHorizontalHeaderLabels(["Geometria personalizzata"])
            self._mod_geom.setItem(0, 0, _make_item("non disponibile", editable=False))
        else:
            self._mod_geom.setColumnCount(3)
            self._mod_geom.setHorizontalHeaderLabels([c[0] for c in _GEOM_COLS_CV])
            for ci, (_, key) in enumerate(_GEOM_COLS_CV):
                val = cv.geometria.get(key, 0.0)
                self._mod_geom.setItem(0, ci, _make_item(f"{val:.4f}"))
        self._ui.extra_tableView_geometria.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._mod_geom.blockSignals(False)
        self._ui.extra_tableView_geometria.viewport().update()

        # ---- Caratteristiche ----
        self._aggiorna_tabella_carat(cv)

        # ---- Vertex edit ----
        self._update_vertex_edit(cv)

        self._loading = False

    def _aggiorna_tabella_carat(self, cv: CaricoVincolo):
        self._mod_carat.blockSignals(True)
        self._mod_carat.setColumnCount(3)
        self._mod_carat.setRowCount(1)
        if cv.sottotipo == "vincolo":
            self._mod_carat.setHorizontalHeaderLabels(["X [m]", "Y [m]", "Z [m]"])
            self._mod_carat.setVerticalHeaderLabels(["Spostamento"])
            keys = ["sx", "sy", "sz"]
        else:
            self._mod_carat.setHorizontalHeaderLabels(["kN (X)", "kN (Y)", "kN (Z)"])
            self._mod_carat.setVerticalHeaderLabels(["Forza"])
            keys = ["fx", "fy", "fz"]
        for ci, key in enumerate(keys):
            val = cv.caratteristiche.get(key, 0.0)
            self._mod_carat.setItem(0, ci, _make_item(f"{val:.4f}"))
        self._ui.extra_tableView_caratteristiche.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._ui.extra_tableView_caratteristiche.verticalHeader().setDefaultSectionSize(28)
        self._mod_carat.blockSignals(False)
        self._ui.extra_tableView_caratteristiche.viewport().update()

    def _svuota_pannello(self):
        self._ui.extra_elemento_label_oggetto.setText("Proprietà: -")
        self._loading = True
        self._mod_mov.blockSignals(True)
        self._mod_geom.blockSignals(True)
        self._mod_carat.blockSignals(True)
        for r in range(2):
            for c in range(3):
                self._mod_mov.setItem(r, c, _make_item("—", editable=False))
        self._mod_geom.setColumnCount(1)
        self._mod_geom.setHorizontalHeaderLabels(["—"])
        self._mod_geom.setItem(0, 0, _make_item("—", editable=False))
        self._mod_carat.setColumnCount(1)
        self._mod_carat.setHorizontalHeaderLabels(["—"])
        self._mod_carat.setItem(0, 0, _make_item("—", editable=False))
        self._mod_mov.blockSignals(False)
        self._mod_geom.blockSignals(False)
        self._mod_carat.blockSignals(False)
        if self._vertex_edit is not None:
            self._vertex_edit.clear()
        self._loading = False

    def _update_vertex_edit(self, cv: CaricoVincolo):
        if self._vertex_edit is None:
            return
        verts_world = cv.get_vertices_world()
        ref_idx     = getattr(cv, "vertice_ref", 0)
        plain = ", ".join(
            f"v{i+1}({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})"
            for i, v in enumerate(verts_world)
        )
        self._vertex_edit.set_ref_idx(ref_idx)
        self._vertex_edit.setText(plain)

    # ================================================================
    #  TABLE CHANGE HANDLERS
    # ================================================================

    def _on_tabella_mov_cambiata(self, item: QStandardItem):
        if self._loading:
            return
        cv_id = self._spazio.get_id_selezionato()
        cv    = self._spazio.get_oggetto(cv_id)
        if cv is None:
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
        cv.rotazione = new_rot
        verts_local = cv.get_vertices_local()
        if verts_local:
            v1_local   = verts_local[0]
            v1_rotated = ruota_punto(v1_local, *new_rot)
            cv.posizione = [v1_world_desired[i] - v1_rotated[i] for i in range(3)]
        else:
            cv.posizione = v1_world_desired
        self._loading = False

        self._spazio.update()
        self._update_vertex_edit(cv)
        self._salva_cv_correnti()

    def _on_tabella_geom_cambiata(self, item: QStandardItem):
        if self._loading:
            return
        cv_id = self._spazio.get_id_selezionato()
        cv    = self._spazio.get_oggetto(cv_id)
        if cv is None:
            return
        col = item.column()
        if col >= len(_GEOM_COLS_CV):
            return
        key = _GEOM_COLS_CV[col][1]
        try:
            val = float(item.text().replace(",", ".").strip())
        except ValueError:
            return
        self._loading = True
        cv.geometria[key] = val
        cv.custom_geometry = False
        self._loading = False
        self._spazio.update()
        self._update_vertex_edit(cv)
        self._salva_cv_correnti()

    def _on_tabella_carat_cambiata(self, item: QStandardItem):
        if self._loading:
            return
        cv_id = self._spazio.get_id_selezionato()
        cv    = self._spazio.get_oggetto(cv_id)
        if cv is None:
            return
        col  = item.column()
        keys = (["sx", "sy", "sz"] if cv.sottotipo == "vincolo"
                else ["fx", "fy", "fz"])
        if col >= len(keys):
            return
        try:
            val = float(item.text().replace(",", ".").strip())
        except ValueError:
            return
        cv.caratteristiche[keys[col]] = val
        self._salva_cv_correnti()

    # ================================================================
    #  VERTEX EDIT APPLY
    # ================================================================

    def _on_linedit_apply(self):
        if self._vertex_edit is None:
            return
        text = self._vertex_edit.text()
        if not text:
            return
        cv_id = self._spazio.get_id_selezionato()
        cv    = self._spazio.get_oggetto(cv_id)
        if cv is None:
            return

        world_verts = _parse_vertici_text(text)
        if world_verts is None:
            QMessageBox.warning(
                self._main, "Formato non riconosciuto",
                "Usa il formato: v1(x,y,z), v2(x,y,z), ..."
            )
            return

        local_verts = [detrasforma_punto(w, cv.posizione, cv.rotazione)
                       for w in world_verts]

        if not cv.custom_geometry:
            r = QMessageBox.question(
                self._main,
                "Conferma modifica geometrica",
                "Confermi la modifica geometrica?\n\n"
                "Il carico/vincolo diventerà una geometria personalizzata "
                "definita unicamente dai suoi vertici.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                self._update_vertex_edit(cv)
                return

        cv.set_vertices_custom(local_verts)
        self._spazio._emit_modificato(cv_id)
        self._mod_geom.blockSignals(True)
        self._mod_geom.setColumnCount(1)
        self._mod_geom.setHorizontalHeaderLabels(["Geometria personalizzata"])
        self._mod_geom.setItem(0, 0, _make_item("—", editable=False))
        self._mod_geom.blockSignals(False)
        self._salva_cv_correnti()
