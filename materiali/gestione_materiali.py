"""
gestione_materiali.py
---------------------
Controller del modulo Materiali.

Fix rispetto alla versione precedente:
  • Bottoni lineare/SLU/SLE in QButtonGroup esclusivo →
    _aggiorna_grafico_live mantiene sempre la modalità attiva
  • Caricamento tabelle con flag _loading invece di blockSignals →
    le viste QTableView si aggiornano correttamente alla selezione
  • _mostra_lineare usa range molto largo (200× max SLU/SLE) →
    la retta sembra infinita
  • _to_screen rimossa: il grafico usa il proprio metodo d'istanza
  • _salva_dati_correnti: legge il progetto in modo più robusto,
    gestisce sia materiali nuovi che materiali standard già presenti;
    salva in _progetto_dati direttamente senza possibilità di alias persi
  • Tabelle SLU/SLE pre-popolate a ogni selezione senza bisogno di
    premere "+"
  • Nessuna chiamata a reset_vista() dopo ogni modifica, solo update()
"""

import copy
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QInputDialog, QMessageBox,
    QMenu, QLineEdit,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal

from .database_materiali import carica_database, nuovo_materiale_personalizzato
from .grafico_materiali  import GraficoMateriali


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

_NOME_BASE_MAT: dict[str, str] = {
    "calcestruzzo":   "calcestruzzo",
    "barre":          "acciaio_barra",
    "acciaio":        "acciaio",
    "personalizzati": "personalizzato",
}


# ================================================================
#  FLOW LAYOUT  (wrap orizzontale, multi-riga)
# ================================================================

class _FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, margin: int = 8, spacing: int = 8):
        super().__init__(parent)
        self._items:   list = []
        self._spacing: int  = spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):       self._items.append(item)
    def count(self):               return len(self._items)
    def itemAt(self, i):           return self._items[i] if 0 <= i < len(self._items) else None
    def takeAt(self, i):           return self._items.pop(i) if 0 <= i < len(self._items) else None
    def expandingDirections(self): return Qt.Orientations(Qt.Orientation(0))
    def hasHeightForWidth(self):   return True
    def heightForWidth(self, w):   return self._layout(QtCore.QRect(0, 0, w, 0), True)
    def setGeometry(self, r):      super().setGeometry(r); self._layout(r, False)
    def sizeHint(self):            return self.minimumSize()

    def minimumSize(self):
        s = QtCore.QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect: QtCore.QRect, test: bool) -> int:
        m   = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, rh = eff.x(), eff.y(), 0
        for it in self._items:
            iw, ih = it.sizeHint().width(), it.sizeHint().height()
            nx = x + iw + self._spacing
            if nx - self._spacing > eff.right() and rh > 0:
                x, y = eff.x(), y + rh + self._spacing
                nx, rh = eff.x() + iw + self._spacing, 0
            if not test:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), it.sizeHint()))
            x, rh = nx, max(rh, ih)
        return y + rh - rect.y() + m.bottom()


# ================================================================
#  STILI
# ================================================================

_BTN_SIZE = QSize(200, 40)

_STILE: dict[str, dict] = {
    "calcestruzzo": {
        "colore_grafico": (0.72, 0.72, 0.72),
        "sheet": """
            QPushButton{background-color:rgb(40,40,40);font:400 12pt "Segoe UI";
                color:#fff;padding-bottom:4px;
                border:1px solid rgb(120,120,120);
                border-left:4px solid rgb(120,120,120);border-radius:6px}
            QPushButton:hover{background-color:rgb(30,30,30);
                border:1px solid rgb(120,120,120);border-left:4px solid rgb(120,120,120)}
            QPushButton:checked{background-color:rgb(65,65,65);
                border:1px solid rgb(200,200,200);border-left:4px solid rgb(210,210,210)}
        """,
    },
    "barre": {
        "colore_grafico": (0.85, 0.52, 0.52),
        "sheet": """
            QPushButton{background-color:rgb(40,40,40);font:400 12pt "Segoe UI";
                color:#fff;padding-bottom:4px;
                border:1px solid rgb(160,120,120);
                border-left:4px solid rgb(160,120,120);border-radius:6px}
            QPushButton:hover{background-color:rgb(30,30,30);
                border:1px solid rgb(160,120,120);border-left:4px solid rgb(160,120,120)}
            QPushButton:checked{background-color:rgb(65,38,38);
                border:1px solid rgb(200,150,150);border-left:4px solid rgb(220,160,160)}
        """,
    },
    "acciaio": {
        "colore_grafico": (0.38, 0.63, 0.90),
        "sheet": """
            QPushButton{background-color:rgb(40,40,40);font:400 12pt "Segoe UI";
                color:#fff;padding-bottom:4px;
                border:1px solid rgb(80,110,150);
                border-left:4px solid rgb(80,110,150);border-radius:6px}
            QPushButton:hover{background-color:rgb(30,30,30);
                border:1px solid rgb(80,110,150);border-left:4px solid rgb(80,110,150)}
            QPushButton:checked{background-color:rgb(28,40,62);
                border:1px solid rgb(100,145,200);border-left:4px solid rgb(120,165,215)}
        """,
    },
    "personalizzati": {
        "colore_grafico": (0.85, 0.85, 0.28),
        "sheet": """
            QPushButton{background-color:rgb(40,40,40);font:400 12pt "Segoe UI";
                color:#fff;padding-bottom:4px;
                border:1px solid rgb(150,150,50);
                border-left:4px solid rgb(150,150,50);border-radius:6px}
            QPushButton:hover{background-color:rgb(30,30,30);
                border:1px solid rgb(150,150,50);border-left:4px solid rgb(150,150,50)}
            QPushButton:checked{background-color:rgb(50,50,18);
                border:1px solid rgb(195,195,75);border-left:4px solid rgb(215,215,85)}
        """,
    },
}

_MAPPA_LINEDIT: dict[str, str] = {
    "materiale_gamma":      "gamma",
    "materiale_alpha":      "alpha",
    "materiale_densita":    "densita",
    "materiale_poisson":    "poisson",
    "materiale_m_elastico": "m_elastico",
    "materiale_m_taglio":   "m_taglio",
}

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


# ================================================================
#  MATERIALE BUTTON
# ================================================================

class MaterialeButton(QPushButton):
    """Bottone materiale con tondino X per i custom (non-standard)."""
    deleteRequested = pyqtSignal()

    def __init__(self, nome: str, cat: str, parent=None, *, standard: bool = False):
        super().__init__(nome, parent)
        self.nome     = nome
        self.cat      = cat
        self.standard = standard
        self.setFixedSize(_BTN_SIZE)
        self.setCheckable(True)
        sheet = _STILE.get(cat, _STILE["personalizzati"])["sheet"]
        # Font uniforme a 9pt come sezioni ed elementi
        sheet = sheet.replace("12pt", "9pt")
        # Margine destro per il tondino X nei custom
        if not standard:
            sheet = sheet.replace("padding-bottom:4px;",
                                  "padding-bottom:4px;padding-right:20px;")
        self.setStyleSheet(sheet)

    def _del_rect(self) -> QtCore.QRect:
        h = _BTN_SIZE.height()
        return QtCore.QRect(_BTN_SIZE.width() - 20, (h - 16) // 2, 16, 16)

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.standard:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            dr = self._del_rect()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(55, 55, 55, 220)))
            painter.drawEllipse(dr)
            pen = QtGui.QPen(QtGui.QColor(190, 190, 190, 230), 1.5)
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

    def sizeHint(self): return _BTN_SIZE


# ================================================================
#  MODELLO TABELLA
# ================================================================

class _ModelloLegame(QtGui.QStandardItemModel):

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Funzione  f(x)", "ε  min", "ε  max"])

    def carica(self, segmenti: list[dict]):
        """Carica i segmenti SENZA bloccare i segnali verso la view."""
        self.beginResetModel()
        # Rimuoviamo le righe manualmente per non usare setRowCount(0)
        # (che non emette correttamente gli aggiornamenti in alcuni casi)
        while self.rowCount():
            self.removeRow(0)
        for seg in segmenti:
            self._aggiungi(
                str(seg.get("formula", "")),
                seg.get("eps_min", 0.0),
                seg.get("eps_max", 0.0),
            )
        self.endResetModel()

    def aggiungi_vuota(self):
        self._aggiungi("", 0.0, 0.0)

    def to_segmenti(self) -> list[dict]:
        out = []
        for r in range(self.rowCount()):
            formula = self.item(r, 0).text().strip() if self.item(r, 0) else ""
            if not formula:
                continue
            out.append({
                "formula":  formula,
                "eps_min":  self._float(r, 1),
                "eps_max":  self._float(r, 2),
            })
        return out

    def _aggiungi(self, formula: str, eps_min: float, eps_max: float):
        row = [
            QtGui.QStandardItem(formula),
            QtGui.QStandardItem(str(eps_min)),
            QtGui.QStandardItem(str(eps_max)),
        ]
        for it in row:
            it.setForeground(QtGui.QBrush(QtGui.QColor(215, 215, 215)))
            it.setBackground(QtGui.QBrush(QtGui.QColor(32, 32, 32)))
        self.appendRow(row)

    def _float(self, row: int, col: int) -> float:
        it = self.item(row, col)
        if it is None:
            return 0.0
        try:
            return float(it.text().replace(",", "."))
        except ValueError:
            return 0.0


# ================================================================
#  CONTROLLER
# ================================================================

class GestioneMateriali:

    # ------------------------------------------------------------------
    #  INIT
    # ------------------------------------------------------------------

    def __init__(self, ui, main_window):
        self._ui   = ui
        self._main = main_window

        self._cat_corrente:  str | None = None
        self._nome_corrente: str | None = None

        # Flag che impedisce a _aggiorna_grafico_live di fare cose
        # durante il caricamento iniziale del materiale
        self._loading: bool = False

        self._bottoni:   dict[tuple, QPushButton]   = {}
        self._btn_group: QtWidgets.QButtonGroup     = QtWidgets.QButtonGroup(main_window)
        self._btn_group.setExclusive(True)

        # Gruppo esclusivo per i 3 pulsanti di visualizzazione grafico
        self._btn_vista: QtWidgets.QButtonGroup = QtWidgets.QButtonGroup(main_window)
        self._btn_vista.setExclusive(True)

        self._modello_slu = _ModelloLegame()
        self._modello_sle = _ModelloLegame()
        self._grafico     = GraficoMateriali()

        # Cache del database standard
        self._db: dict = carica_database()

        self._setup_frame_materiali()
        self._setup_tabelle()
        self._setup_grafico()
        self._setup_connessioni()
        self._svuota_pannello()

    # ------------------------------------------------------------------
    #  SETUP: FRAME
    # ------------------------------------------------------------------

    def _setup_frame_materiali(self):
        for btn in list(self._btn_group.buttons()):
            self._btn_group.removeButton(btn)
        self._bottoni.clear()

        sezione_prj = self._main.get_sezione("materiali")

        mappa = {
            "calcestruzzo":   self._ui.materiale_frame_calcestruzzo,
            "barre":          self._ui.materiale_frame_barre,
            "acciaio":        self._ui.materiale_frame_acciaio,
            "personalizzati": self._ui.materiale_frame_personalizzati,
        }

        for cat, frame in mappa.items():
            self._init_frame(frame)
            # Materiali standard
            for nome in self._db.get(cat, {}):
                self._crea_bottone(frame, cat, nome, standard=True)
            # Materiali custom nel progetto (non nel db)
            for nome, dati in sezione_prj.get(cat, {}).items():
                if not dati.get("standard", True) and nome not in self._db.get(cat, {}):
                    self._crea_bottone(frame, cat, nome, standard=False)

    @staticmethod
    def _init_frame(frame: QtWidgets.QWidget):
        old = frame.layout()
        if old is not None:
            while old.count():
                item = old.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    item.widget().deleteLater()
            QtWidgets.QWidget().setLayout(old)
        frame.setLayout(_FlowLayout(margin=8, spacing=8))

    def _crea_bottone(self, frame: QtWidgets.QWidget,
                      cat: str, nome: str, standard: bool = False) -> "MaterialeButton":
        btn = MaterialeButton(nome, cat, frame, standard=standard)
        frame.layout().addWidget(btn)
        self._btn_group.addButton(btn)
        self._bottoni[(cat, nome)] = btn
        # Lambda che legge btn.nome/btn.cat al momento del clic → si aggiorna
        # automaticamente dopo un rename senza bisogno di riconnettere.
        btn.clicked.connect(lambda _c, b=btn: self._seleziona_materiale(b.cat, b.nome))
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        btn.customContextMenuRequested.connect(
            lambda pos, b=btn: self._ctx_materiale(b, pos))
        if not standard:
            btn.deleteRequested.connect(
                lambda b=btn: self._elimina_materiale(b.cat, b.nome))
        return btn

    # ------------------------------------------------------------------
    #  SETUP: TABELLE
    # ------------------------------------------------------------------

    def _setup_tabelle(self):
        for tv, mod in ((self._ui.tableView_slu, self._modello_slu),
                        (self._ui.tableView_sle, self._modello_sle)):
            tv.setModel(mod)
            tv.setStyleSheet(_STILE_TV)
            hh = tv.horizontalHeader()
            hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
            tv.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
            tv.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked |
                               QtWidgets.QAbstractItemView.SelectedClicked)

    # ------------------------------------------------------------------
    #  SETUP: GRAFICO
    # ------------------------------------------------------------------

    def _setup_grafico(self):
        container = self._ui.materiale_widget
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self._grafico)

    # ------------------------------------------------------------------
    #  SETUP: CONNESSIONI
    # ------------------------------------------------------------------

    def _setup_connessioni(self):
        # Aggiungi materiali custom
        self._ui.materiale_aggiungi_calcestruzzo.clicked.connect(
            lambda: self._aggiungi_custom("calcestruzzo"))
        self._ui.materiale_aggiungi_barre.clicked.connect(
            lambda: self._aggiungi_custom("barre"))
        self._ui.materiale_aggiungi_acciaio.clicked.connect(
            lambda: self._aggiungi_custom("acciaio"))
        self._ui.materiale_aggiungi_personalizzati.clicked.connect(
            lambda: self._aggiungi_custom("personalizzati"))

        # Aggiungi riga tabelle
        self._ui.materiale_aggiungi_slu.clicked.connect(
            lambda: (self._modello_slu.aggiungi_vuota(),
                     self._aggiorna_grafico_live()))
        self._ui.materiale_aggiungi_sle.clicked.connect(
            lambda: (self._modello_sle.aggiungi_vuota(),
                     self._aggiorna_grafico_live()))

        # ── Bottoni modalità grafico ──────────────────────────────────
        # Li mettiamo in un gruppo esclusivo così isChecked() è sempre affidabile
        for btn in (self._ui.materiale_btn_lineare,
                    self._ui.materiale_btn_slu,
                    self._ui.materiale_btn_sle):
            btn.setCheckable(True)
            self._btn_vista.addButton(btn)

        self._ui.materiale_btn_lineare.clicked.connect(self._mostra_lineare)
        self._ui.materiale_btn_slu.clicked.connect(self._mostra_slu)
        self._ui.materiale_btn_sle.clicked.connect(self._mostra_sle)

        # Reset vista grafico
        self._ui.btn_materiali_centra.clicked.connect(self._grafico.reset_vista)

        # Aggiorna (commit manuale)
        self._ui.btn_materiale_aggiorna.clicked.connect(self._aggiorna_e_salva)

        # ── Live update grafico ───────────────────────────────────────
        # itemChanged nelle tabelle → aggiorna grafico (NON fa reset vista)
        self._modello_slu.itemChanged.connect(self._on_tabella_cambiata)
        self._modello_sle.itemChanged.connect(self._on_tabella_cambiata)

        # textChanged delle line edit → aggiorna grafico live
        for nome_w in _MAPPA_LINEDIT:
            w = getattr(self._ui, nome_w, None)
            if w:
                w.textChanged.connect(self._on_linedit_cambiata)
                w.editingFinished.connect(self._salva_dati_correnti)

    # ------------------------------------------------------------------
    #  SELEZIONE MATERIALE
    # ------------------------------------------------------------------

    def _seleziona_materiale(self, cat: str, nome: str):
        # Salva il materiale precedente prima di cambiare
        if self._nome_corrente and self._nome_corrente != nome:
            self._salva_dati_correnti()

        self._cat_corrente  = cat
        self._nome_corrente = nome

        dati = self._leggi_dati(cat, nome)
        if dati is None:
            print(f"WARN  Materiale non trovato: [{cat}] {nome}")
            return

        # ── Blocca tutti i segnali live durante il caricamento ─────────
        self._loading = True

        # Aggiorna label nome
        try:
            self._ui.label_id_materiale.setText(nome)
        except AttributeError:
            pass

        # Popola line edit
        self._popola_linedit(dati)

        # Popola tabelle – usa beginResetModel/endResetModel (in carica())
        # così le view si aggiornano correttamente
        self._modello_slu.carica(dati.get("slu", []))
        self._modello_sle.carica(dati.get("sle", []))

        self._loading = False

        # ── Vai alla pagina dettaglio (index 2) ────────────────────────
        self._ui.stackedWidget_main.setCurrentIndex(2)

        # ── Grafico: mantieni la modalità attiva, default lineare ──────
        if not self._ui.materiale_btn_slu.isChecked() and \
           not self._ui.materiale_btn_sle.isChecked():
            self._ui.materiale_btn_lineare.setChecked(True)

        # Forza ridisegno con i nuovi dati SENZA resettare pan/zoom
        self._ridisegna_grafico_attivo()

    # ------------------------------------------------------------------
    #  LETTURA DATI
    # ------------------------------------------------------------------

    def _leggi_dati(self, cat: str, nome: str) -> dict | None:
        """
        Cerca il materiale nel progetto aperto (include le modifiche).
        Fallback al database standard se non ancora modificato.
        """
        sezione = self._main.get_sezione("materiali")
        prj_cat = sezione.get(cat, {})
        if nome in prj_cat:
            return prj_cat[nome]
        # Fallback db standard
        src = self._db.get(cat, {}).get(nome)
        return copy.deepcopy(src) if src else None

    # ------------------------------------------------------------------
    #  PANNELLO LINE EDIT
    # ------------------------------------------------------------------

    def _popola_linedit(self, dati: dict):
        for nome_w, chiave in _MAPPA_LINEDIT.items():
            w = getattr(self._ui, nome_w, None)
            if w:
                w.blockSignals(True)
                w.setText(str(dati.get(chiave, "")))
                w.blockSignals(False)

    def _svuota_pannello(self):
        try:
            self._ui.label_id_materiale.setText("—")
        except AttributeError:
            pass
        for nome_w in _MAPPA_LINEDIT:
            w = getattr(self._ui, nome_w, None)
            if w:
                w.blockSignals(True)
                w.clear()
                w.blockSignals(False)
        self._modello_slu.carica([])
        self._modello_sle.carica([])
        self._grafico.set_segmenti([], (0.5, 0.5, 0.5))

    # ------------------------------------------------------------------
    #  GRAFICO – 3 MODALITÀ
    # ------------------------------------------------------------------

    def _mostra_lineare(self):
        """
        Retta elastica lineare σ = E·ε.
        Range molto largo (200× il max SLU/SLE) così appare "infinita".
        """
        try:
            E = float(
                self._ui.materiale_m_elastico.text()
                         .replace(",", ".").strip() or "0"
            )
        except (ValueError, AttributeError):
            E = 0.0

        # Calcola range dai segmenti presenti nelle tabelle
        tutti = (self._modello_slu.to_segmenti() +
                 self._modello_sle.to_segmenti())
        if tutti:
            vals = ([abs(float(s.get("eps_min", 0))) for s in tutti] +
                    [abs(float(s.get("eps_max", 0))) for s in tutti])
            base_rng = max(vals) if max(vals) > 0 else 0.005
        else:
            base_rng = 0.005

        # 200× il range dei dati → la retta supera abbondantemente lo schermo
        rng = base_rng * 200.0

        self._grafico.set_segmenti(
            [{"formula": f"{E} * x", "eps_min": -rng, "eps_max": rng}],
            self._colore_corrente()
        )

    def _mostra_slu(self):
        """Legge direttamente dalla tabella SLU."""
        self._grafico.set_segmenti(
            self._modello_slu.to_segmenti(),
            self._colore_corrente()
        )

    def _mostra_sle(self):
        """Legge direttamente dalla tabella SLE."""
        self._grafico.set_segmenti(
            self._modello_sle.to_segmenti(),
            self._colore_corrente()
        )

    def _ridisegna_grafico_attivo(self):
        """
        Ridisegna il grafico nella modalità attualmente selezionata.
        NON resetta pan/zoom: l'utente mantiene la vista corrente.
        """
        if self._ui.materiale_btn_slu.isChecked():
            self._mostra_slu()
        elif self._ui.materiale_btn_sle.isChecked():
            self._mostra_sle()
        else:
            self._mostra_lineare()

    def _aggiorna_grafico_live(self):
        """
        Chiamato da eventi UI (modifica tabella, cambio linedit).
        Rispetta la modalità attiva, non fa mai reset vista.
        Ignorato durante il caricamento iniziale del materiale.
        """
        if self._loading:
            return
        self._ridisegna_grafico_attivo()

    def _colore_corrente(self) -> tuple:
        cat = self._cat_corrente or "calcestruzzo"
        return _STILE.get(cat, _STILE["personalizzati"])["colore_grafico"]

    # ------------------------------------------------------------------
    #  EVENTI LIVE (tabelle e line edit)
    # ------------------------------------------------------------------

    def _on_tabella_cambiata(self):
        """itemChanged delle tabelle → update grafico live."""
        self._aggiorna_grafico_live()

    def _on_linedit_cambiata(self):
        """textChanged delle line edit → update grafico live."""
        self._aggiorna_grafico_live()

    # ------------------------------------------------------------------
    #  SALVATAGGIO NEL PROGETTO
    # ------------------------------------------------------------------

    def _salva_dati_correnti(self):
        """
        Raccoglie i valori dalla UI e li scrive in _progetto_dati.
        Funziona per materiali standard (vengono "promossi" nel progetto
        con le modifiche) e custom.
        """
        if self._loading:
            return
        if self._cat_corrente is None or self._nome_corrente is None:
            return
        if not self._main.ha_progetto():
            return

        # Leggi la sezione materiali dal progetto
        sezione  = self._main.get_sezione("materiali")
        cat_dict = sezione.setdefault(self._cat_corrente, {})

        # Dati base: preferenza al progetto (già modificato),
        # fallback al db standard
        base = cat_dict.get(self._nome_corrente)
        if base is None:
            base = self._db.get(self._cat_corrente, {}).get(self._nome_corrente, {})
        dati = copy.deepcopy(base)

        # Aggiorna con le line edit
        for nome_w, chiave in _MAPPA_LINEDIT.items():
            w = getattr(self._ui, nome_w, None)
            if w:
                testo = w.text().replace(",", ".").strip()
                try:
                    dati[chiave] = float(testo)
                except ValueError:
                    pass   # campo vuoto o non numerico: mantieni valore precedente

        # Aggiorna con le tabelle
        dati["slu"] = self._modello_slu.to_segmenti()
        dati["sle"] = self._modello_sle.to_segmenti()

        # Salta il salvataggio (e lo snapshot undo) se i dati non sono cambiati
        current = cat_dict.get(self._nome_corrente)
        if current is not None and current == dati:
            return

        self._main.push_undo(f"Modifica materiale [{self._nome_corrente}]")
        cat_dict[self._nome_corrente] = dati
        self._main.set_sezione("materiali", sezione)

    def _aggiorna_e_salva(self):
        """btn_materiale_aggiorna: commit esplicito + refresh grafico."""
        if self._nome_corrente is None:
            return
        self._salva_dati_correnti()
        self._ridisegna_grafico_attivo()
        print(f">> Materiale aggiornato: [{self._cat_corrente}] {self._nome_corrente}")

    # ------------------------------------------------------------------
    #  AGGIUNGI MATERIALE CUSTOM  (nome default, senza dialog)
    # ------------------------------------------------------------------

    def _aggiungi_custom(self, cat: str):
        if not self._main.ha_progetto():
            QMessageBox.warning(self._main, "Attenzione",
                                "Crea o apri prima un progetto.")
            return

        sezione  = self._main.get_sezione("materiali")
        cat_dict = sezione.setdefault(cat, {})
        esistenti = set(cat_dict.keys()) | set(self._db.get(cat, {}).keys())
        nome = _prossimo_nome(_NOME_BASE_MAT.get(cat, cat), esistenti)

        dati = nuovo_materiale_personalizzato()
        dati["tipo"] = (cat.rstrip("i") if cat != "personalizzati"
                        else "personalizzato")
        self._main.push_undo(f"Aggiungi materiale [{nome}]")
        cat_dict[nome] = dati
        self._main.set_sezione("materiali", sezione)

        frame_mappa = {
            "calcestruzzo":   self._ui.materiale_frame_calcestruzzo,
            "barre":          self._ui.materiale_frame_barre,
            "acciaio":        self._ui.materiale_frame_acciaio,
            "personalizzati": self._ui.materiale_frame_personalizzati,
        }
        frame = frame_mappa.get(cat)
        if frame:
            btn = self._crea_bottone(frame, cat, nome, standard=False)
            btn.setChecked(True)

        print(f">> Materiale aggiunto: [{cat}] {nome}")

    # ------------------------------------------------------------------
    #  RINOMINA INLINE  (doppio click o voce menu)
    # ------------------------------------------------------------------

    def _rinomina_materiale(self, btn: "MaterialeButton"):
        """Apre un QLineEdit inline sul bottone per rinominare."""
        dati = self._leggi_dati(btn.cat, btn.nome)
        if dati and dati.get("standard", False):
            return  # standard: silenzioso

        vec = btn.nome
        le = QLineEdit(btn)
        h = _BTN_SIZE.height()
        le.setGeometry(4, (h - 22) // 2, _BTN_SIZE.width() - 28, 22)
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
                self._applica_rinomina_mat(btn, vec, nuovo)

        le.returnPressed.connect(commit)
        le.editingFinished.connect(commit)

    def _applica_rinomina_mat(self, btn: "MaterialeButton", vec: str, nuovo: str):
        cat      = btn.cat
        sezione  = self._main.get_sezione("materiali")
        cat_dict = sezione.get(cat, {})

        if nuovo in cat_dict or nuovo in self._db.get(cat, {}):
            QMessageBox.warning(self._main, "Attenzione",
                                f"Esiste già un materiale di nome «{nuovo}».")
            return

        if vec in cat_dict:
            self._main.push_undo(f"Rinomina materiale [{vec} → {nuovo}]")
            cat_dict[nuovo] = cat_dict.pop(vec)
            self._main.set_sezione("materiali", sezione)

        # Aggiorna bottone (le lambda leggono btn.nome dinamicamente)
        btn.nome = nuovo
        btn.setText(nuovo)

        if (cat, vec) in self._bottoni:
            self._bottoni[(cat, nuovo)] = self._bottoni.pop((cat, vec))

        if self._nome_corrente == vec and self._cat_corrente == cat:
            self._nome_corrente = nuovo
            try:
                self._ui.label_id_materiale.setText(nuovo)
            except AttributeError:
                pass

        print(f">> Materiale rinominato: [{cat}] {vec} → {nuovo}")

    # ------------------------------------------------------------------
    #  CONTEXT MENU TASTO DESTRO
    # ------------------------------------------------------------------

    def _ctx_materiale(self, btn: "MaterialeButton", pos):
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
            self._duplica_materiale(btn.cat, btn.nome)
        elif act_ren and action == act_ren:
            self._rinomina_materiale(btn)
        elif act_del and action == act_del:
            self._elimina_materiale(btn.cat, btn.nome)

    # ------------------------------------------------------------------
    #  ELIMINA MATERIALE CUSTOM
    # ------------------------------------------------------------------

    def _elimina_materiale(self, cat: str, nome: str):
        r = QMessageBox.question(
            self._main, "Elimina materiale",
            f"Eliminare il materiale «{nome}»?\nL'operazione non è reversibile.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        sezione  = self._main.get_sezione("materiali")
        cat_dict = sezione.get(cat, {})
        if nome in cat_dict:
            self._main.push_undo(f"Elimina materiale [{nome}]")
            del cat_dict[nome]
            self._main.set_sezione("materiali", sezione)
        key = (cat, nome)
        if key in self._bottoni:
            btn = self._bottoni.pop(key)
            self._btn_group.removeButton(btn)
            btn.setParent(None); btn.deleteLater()
        if self._cat_corrente == cat and self._nome_corrente == nome:
            self._cat_corrente = None; self._nome_corrente = None
            self._svuota_pannello()
        print(f">> Materiale eliminato: [{cat}] {nome}")

    # ------------------------------------------------------------------
    #  DUPLICA MATERIALE
    # ------------------------------------------------------------------

    def _duplica_materiale(self, cat: str, nome: str):
        dati = self._leggi_dati(cat, nome)
        if dati is None:
            return
        sezione  = self._main.get_sezione("materiali")
        cat_dict = sezione.setdefault(cat, {})
        esistenti = set(cat_dict.keys()) | set(self._db.get(cat, {}).keys())
        nuovo_nome = _prossimo_nome(nome, esistenti)

        copia = copy.deepcopy(dati)
        copia["standard"] = False
        self._main.push_undo(f"Duplica materiale [{nome} → {nuovo_nome}]")
        cat_dict[nuovo_nome] = copia
        self._main.set_sezione("materiali", sezione)

        frame_mappa = {
            "calcestruzzo":   self._ui.materiale_frame_calcestruzzo,
            "barre":          self._ui.materiale_frame_barre,
            "acciaio":        self._ui.materiale_frame_acciaio,
            "personalizzati": self._ui.materiale_frame_personalizzati,
        }
        frame = frame_mappa.get(cat)
        if frame:
            self._crea_bottone(frame, cat, nuovo_nome, standard=False)
        print(f">> Materiale duplicato: [{cat}] {nome} → {nuovo_nome}")

    # ------------------------------------------------------------------
    #  RICARICA DA PROGETTO
    # ------------------------------------------------------------------

    def ricarica_da_progetto(self):
        """
        Chiamata da MainWindow._imposta_progetto() dopo ogni
        caricamento o creazione progetto.
        """
        self._loading = True
        self._cat_corrente  = None
        self._nome_corrente = None
        self._setup_frame_materiali()
        self._svuota_pannello()
        self._loading = False
        print(">> Modulo Materiali: progetto ricaricato.")

    def ripristina_contesto(self, cat: str, nome: str):
        """
        Chiamata da MainWindow dopo undo/redo per riaprire il materiale
        che era in editing al momento dello snapshot.
        """
        sez = self._main.get_sezione("materiali")
        # Il materiale deve ancora esistere (potrebbe essere stato rimosso)
        if nome not in sez.get(cat, {}) and nome not in self._db.get(cat, {}):
            return
        btn = self._bottoni.get((cat, nome))
        if btn:
            btn.setChecked(True)
        self._seleziona_materiale(cat, nome)