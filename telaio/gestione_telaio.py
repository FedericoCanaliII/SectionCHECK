"""
gestione_telaio.py  -  Controller principale del modulo telaio FEM.

Unita di misura interfaccia:
  Posizioni nodi  -> m
  Carichi conc.   -> kN, kNm
  Carichi dist.   -> kN/m
  E, G            -> MPa   (x1e6  -> Pa)
  A               -> mm2   (x1e-6 -> m2)
  Iy, Iz, J       -> mm4   (x1e-12-> m4)

Nuove feature rispetto alla versione precedente:
  - Combobox sezioni direttamente nell'input aste
  - TableView EDITABILI: doppio clic su cella per modificare
  - Delegate ComboBox nella colonna Sezione delle aste
"""

import json
import copy
import numpy as np

from PyQt5.QtWidgets import (
    QButtonGroup, QMenu, QAction,
    QVBoxLayout, QMessageBox, QHeaderView,
    QComboBox, QStyledItemDelegate, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui  import QStandardItemModel, QStandardItem

from .spazio3d                 import Spazio3D
from .calcolo                  import TelaioFEM
from .telaio_materiali_sezioni import GestioneMaterialiTelaio, GestioneSezioniTelaio


# ======================================================================
#  Conversioni unita
# ======================================================================

def _ui_a_si(sezione: dict, materiale: dict) -> tuple:
    E  = float(materiale.get('E',   210_000.0)) * 1e6
    G  = float(materiale.get('G',    81_000.0)) * 1e6
    A  = float(sezione.get('A',      3_600.0))  * 1e-6
    Iy = float(sezione.get('Iy', 36_900_000.0)) * 1e-12
    Iz = float(sezione.get('Iz', 13_360_000.0)) * 1e-12
    J  = float(sezione.get('J',     59_100.0))  * 1e-12
    return E, G, A, Iy, Iz, J


# ======================================================================
#  Delegate - ComboBox per la colonna Sezione nella tabella aste
# ======================================================================

class SezioneDelegate(QStyledItemDelegate):
    """Mostra una QComboBox con le sezioni disponibili quando si edita
    la colonna 'Sezione' della tableView_aste."""

    def __init__(self, sezioni_getter, parent=None):
        super().__init__(parent)
        self._get_sezioni = sezioni_getter   # callable -> dict sezioni

    def createEditor(self, parent, option, index):
        cb = QComboBox(parent)
        for sid, sez in self._get_sezioni().items():
            cb.addItem(sez['nome'], userData=sid)
        return cb

    def setEditorData(self, editor, index):
        valore_corrente = index.data(Qt.UserRole) or index.data(Qt.DisplayRole)
        for i in range(editor.count()):
            if editor.itemData(i) == valore_corrente or editor.itemText(i) == valore_corrente:
                editor.setCurrentIndex(i)
                break

    def setModelData(self, editor, model, index):
        # Salva sia il testo visibile che l'id come UserRole
        model.setData(index, editor.currentText(), Qt.DisplayRole)
        model.setData(index, editor.currentData(),  Qt.UserRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


# ======================================================================
#  Worker thread
# ======================================================================

class CalcoloWorker(QObject):
    finito    = pyqtSignal(object, object, object, object)
    errore    = pyqtSignal(str)
    progresso = pyqtSignal(int)

    def __init__(self, dati: dict):
        super().__init__()
        self.dati = dati

    def esegui(self):
        try:
            d = self.dati
            self.progresso.emit(10)
            telaio = TelaioFEM()
            id_map = {}

            for g_id, coords in d['nodi'].items():
                id_map[g_id] = telaio.aggiungi_nodo_principale(*coords)
            self.progresso.emit(25)

            for aid, a in d['aste'].items():
                q_tot = [0.0, 0.0, 0.0]
                for c in d['carichi_aste'].values():
                    if c['asta'] == aid:
                        for k in range(3):
                            q_tot[k] += c['q'][k] * 1000.0   # kN/m -> N/m
                sezione   = d['sezioni'].get(a.get('sezione', ''), {})
                materiale = d['materiali'].get(
                    sezione.get('materiale', a.get('materiale', '')), {})
                E, G, A_si, Iy, Iz, J = _ui_a_si(sezione, materiale)
                sud = max(1, int(a.get('sud', 1)))
                telaio.aggiungi_asta(
                    id_map[a['n1']], id_map[a['n2']],
                    E, G, A_si, Iy, Iz, J,
                    num_suddivisioni=sud, q_distribuito=q_tot,
                )
            self.progresso.emit(50)

            for v in d['vincoli'].values():
                telaio.aggiungi_vincolo(id_map[v['nodo']], v['gradi'])

            for c in d['carichi_nodi'].values():
                telaio.aggiungi_carico(id_map[c['nodo']],
                                       [f * 1000.0 for f in c['forze']])
            self.progresso.emit(65)

            u_vec, sforzi = telaio.risolvi()
            self.progresso.emit(90)

            t_nodi = list(telaio._nodi_np[: telaio.num_nodi])
            t_aste = list(telaio._aste_np[: telaio.num_aste])
            self.progresso.emit(100)
            self.finito.emit(u_vec, sforzi, t_nodi, t_aste)

        except Exception:
            import traceback
            self.errore.emit(traceback.format_exc())


# ======================================================================
#  Controller principale
# ======================================================================

class GestioneTelaio:
    """
    Utilizzo nel MainWindow.__init__:
        from telaio.gestione_telaio import GestioneTelaio
        self.telaio = GestioneTelaio(self.ui)
    """

    # Indici colonne tabelle
    _COL_NODI    = {"id": 0, "x": 1, "y": 2, "z": 3}
    _COL_ASTE    = {"id": 0, "n1": 1, "n2": 2, "sezione": 3, "sud": 4}
    _COL_VINCOLI = {"id": 0, "nodo": 1,
                    "tx": 2, "ty": 3, "tz": 4, "rx": 5, "ry": 6, "rz": 7}
    _COL_CARICHI = {"id": 0, "tipo": 1, "rif": 2, "valori": 3}

    def __init__(self, ui):
        self.ui = ui

        self.materiali:    dict = {}
        self.sezioni:      dict = {}
        self.nodi:         dict = {}
        self.aste:         dict = {}
        self.vincoli:      dict = {}
        self.carichi_nodi: dict = {}
        self.carichi_aste: dict = {}

        self._cnt_nodi = self._cnt_aste = self._cnt_vinc = 0
        self._cnt_cn   = self._cnt_ca   = 0

        self._u_vec = self._sforzi = None
        self._t_nodi = []; self._t_aste = []
        self._risolto = False
        self._thread = self._worker = None

        # Viewer 3D
        self._viewer = Spazio3D()
        lyt = QVBoxLayout(self.ui.telaio_spazio)
        lyt.setContentsMargins(1, 1, 1, 1)
        lyt.addWidget(self._viewer)

        # Sub-controller materiali e sezioni
        self._ctrl_mat = GestioneMaterialiTelaio(
            ui, self.materiali, on_changed=self._on_materiali_changed)
        self._ctrl_sez = GestioneSezioniTelaio(
            ui, self.sezioni, self.materiali, on_changed=self._on_sezioni_changed)
        self._ctrl_mat.aggiorna_combobox(self.ui.combobox_teliaio_materiali)

        # Combobox sezioni nell'input aste (iniettata nel layout se non presente in UI)
        self._cb_sezione_asta = QComboBox()
        self._cb_sezione_asta.setMinimumWidth(150)
        self._aggiorna_cb_sezioni_asta()
        # Cerca di inserirla nella UI affiancata ai campi asta
        # (se il designer ha un widget placeholder chiamato 'widget_asta_sezione')
        try:
            lyt_asta = QVBoxLayout(self.ui.widget_asta_sezione)
            lyt_asta.setContentsMargins(0, 0, 0, 0)
            lyt_asta.addWidget(self._cb_sezione_asta)
        except AttributeError:
            # Fallback: la combobox e' accessibile via self._cb_sezione_asta
            # e viene usata internamente in _aggiungi_asta
            pass

        # Modelli tabelle
        self._model_nodi    = _crea_model(["ID", "X [m]", "Y [m]", "Z [m]"])
        self._model_aste    = _crea_model(["ID", "N1", "N2", "Sezione", "Subdiv."])
        self._model_vincoli = _crea_model(["ID", "Nodo", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz"])
        self._model_carichi = _crea_model(["ID", "Tipo", "Rif.", "Valori"])

        self.ui.tableView_nodi.setModel(self._model_nodi)
        self.ui.tableView_aste.setModel(self._model_aste)
        self.ui.tableView_vincoli.setModel(self._model_vincoli)
        self.ui.tableView_carichi.setModel(self._model_carichi)

        # Tutte le tabelle: header stretch + context menu + doppio-clic
        for tv in (self.ui.tableView_nodi, self.ui.tableView_aste,
                   self.ui.tableView_vincoli, self.ui.tableView_carichi):
            tv.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tv.setContextMenuPolicy(Qt.CustomContextMenu)
            # Abilita editing per doppio clic
            tv.setEditTriggers(QAbstractItemView.DoubleClicked |
                               QAbstractItemView.SelectedClicked)

        # Delegate combobox sezione nella colonna 3 di tableView_aste
        self._sez_delegate = SezioneDelegate(lambda: self.sezioni)
        self.ui.tableView_aste.setItemDelegateForColumn(
            self._COL_ASTE["sezione"], self._sez_delegate)

        # Context menu elimina
        self.ui.tableView_nodi.customContextMenuRequested.connect(
            lambda p: self._ctx_table(p, self.ui.tableView_nodi, 'nodo'))
        self.ui.tableView_aste.customContextMenuRequested.connect(
            lambda p: self._ctx_table(p, self.ui.tableView_aste, 'asta'))
        self.ui.tableView_vincoli.customContextMenuRequested.connect(
            lambda p: self._ctx_table(p, self.ui.tableView_vincoli, 'vincolo'))
        self.ui.tableView_carichi.customContextMenuRequested.connect(
            lambda p: self._ctx_table(p, self.ui.tableView_carichi, 'carico'))

        # Selezione riga -> popola campi input
        self.ui.tableView_nodi.selectionModel().selectionChanged.connect(
            self._carica_nodo_sel)
        self.ui.tableView_aste.selectionModel().selectionChanged.connect(
            self._carica_asta_sel)
        self.ui.tableView_vincoli.selectionModel().selectionChanged.connect(
            self._carica_vincolo_sel)

        # Segnale itemChanged -> sincronizza dati interni quando si edita in tabella
        self._model_nodi.itemChanged.connect(self._on_nodo_edited)
        self._model_aste.itemChanged.connect(self._on_asta_edited)
        self._model_vincoli.itemChanged.connect(self._on_vincolo_edited)

        self._setup_navigazione()
        self._setup_nodi()
        self._setup_aste()
        self._setup_vincoli()
        self._setup_carichi()
        self._setup_analisi()
        self._setup_viste()
        self._setup_scala()
        self._setup_salvataggio()

    # ------------------------------------------------------------------
    # Callback sub-controller
    # ------------------------------------------------------------------

    def _on_materiali_changed(self):
        if not hasattr(self, '_ctrl_mat'): return
        self._ctrl_mat.aggiorna_combobox(self.ui.combobox_teliaio_materiali)
        if hasattr(self, '_ctrl_sez'):
            self._ctrl_sez._ricostruisci_layout()
        if hasattr(self, '_cb_sezione_asta'):
            self._aggiorna_cb_sezioni_asta()

    def _on_sezioni_changed(self):
        if not hasattr(self, '_cb_sezione_asta'): return
        self._aggiorna_table_aste()
        self._aggiorna_cb_sezioni_asta()

    def _aggiorna_cb_sezioni_asta(self):
        """Aggiorna la QComboBox sezioni nell'input aste."""
        cb = self._cb_sezione_asta
        attuale = cb.currentData()
        cb.blockSignals(True)
        cb.clear()
        for sid, sez in self.sezioni.items():
            mat_nome = self.materiali.get(sez.get('materiale', ''), {}).get('nome', '?')
            cb.addItem(f"{sez['nome']}  [{mat_nome}]", userData=sid)
        # Ripristina selezione
        for i in range(cb.count()):
            if cb.itemData(i) == attuale:
                cb.setCurrentIndex(i); break
        cb.blockSignals(False)

    # ------------------------------------------------------------------
    # Navigazione
    # ------------------------------------------------------------------

    def _setup_navigazione(self):
        ui = self.ui
        g1 = QButtonGroup(ui.btn_telaio_sezioni)
        for btn, idx in [(ui.btn_telaio_sezioni, 0), (ui.btn_telaio_modello, 1)]:
            btn.setCheckable(True); g1.addButton(btn)
            btn.clicked.connect(lambda _, i=idx: ui.stackedWidget_struttura.setCurrentIndex(i))
        g1.setExclusive(True)

        g2 = QButtonGroup(ui.btn_telaio_nodi)
        for btn, idx in [(ui.btn_telaio_nodi, 0), (ui.btn_telaio_aste, 1),
                         (ui.btn_telaio_vincoli, 2), (ui.btn_telaio_carichi, 3)]:
            btn.setCheckable(True); g2.addButton(btn)
            btn.clicked.connect(lambda _, i=idx: ui.stackedWidget_modello.setCurrentIndex(i))
        g2.setExclusive(True)

        QTimer.singleShot(0,  ui.btn_telaio_sezioni.click)
        QTimer.singleShot(50, ui.btn_telaio_nodi.click)

    # ------------------------------------------------------------------
    # NODI
    # ------------------------------------------------------------------

    def _setup_nodi(self):
        self.ui.telaio_aggiungi_nodo.clicked.connect(self._aggiungi_nodo)

    def _aggiungi_nodo(self):
        try:
            x = _rf(self.ui.nodo_posizione_x)
            y = _rf(self.ui.nodo_posizione_y)
            z = _rf(self.ui.nodo_posizione_z)
        except ValueError:
            QMessageBox.warning(None, "Input non valido", "Inserire coordinate numeriche (m).")
            return
        nid = self._cnt_nodi; self._cnt_nodi += 1
        self.nodi[nid] = [x, y, z]
        self._aggiorna_table_nodi()
        self._aggiorna_viewer()
        for w in (self.ui.nodo_posizione_x, self.ui.nodo_posizione_y, self.ui.nodo_posizione_z):
            w.clear()

    def _aggiorna_table_nodi(self):
        self._model_nodi.itemChanged.disconnect(self._on_nodo_edited)
        m = self._model_nodi; m.setRowCount(0)
        for nid, c in self.nodi.items():
            id_item = _it(nid)
            m.appendRow([id_item, _it(f"{c[0]:.6g}"), _it(f"{c[1]:.6g}"), _it(f"{c[2]:.6g}")])
        self._model_nodi.itemChanged.connect(self._on_nodo_edited)

    def _carica_nodo_sel(self):
        row = self.ui.tableView_nodi.currentIndex().row()
        if row < 0: return
        try:
            nid = int(self._model_nodi.item(row, 0).text())
            c = self.nodi[nid]
            self.ui.nodo_posizione_x.setText(str(c[0]))
            self.ui.nodo_posizione_y.setText(str(c[1]))
            self.ui.nodo_posizione_z.setText(str(c[2]))
        except Exception: pass

    def _on_nodo_edited(self, item):
        """Sincronizza il dizionario nodi dopo una modifica in tabella."""
        row = item.row()
        try:
            # Leggi id dalla colonna 0
            nid_str = self._model_nodi.item(row, 0).text()
            # Se hanno modificato la colonna ID (rinomina) aggiorna la chiave
            old_nid = list(self.nodi.keys())[row]
            new_nid_candidate = int(nid_str)

            coords = self.nodi[old_nid]
            # Aggiorna coordinate se modificate
            for col, idx in [(1, 0), (2, 1), (3, 2)]:
                val_item = self._model_nodi.item(row, col)
                if val_item:
                    coords[idx] = float(val_item.text().replace(',', '.'))

            if new_nid_candidate != old_nid:
                # Rinomina chiave
                del self.nodi[old_nid]
                self.nodi[new_nid_candidate] = coords
            else:
                self.nodi[old_nid] = coords

            self._aggiorna_viewer()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ASTE
    # ------------------------------------------------------------------

    def _setup_aste(self):
        self.ui.telaio_asta_aggiungi.clicked.connect(self._aggiungi_asta)

    def _aggiungi_asta(self):
        try:
            n1  = int(self.ui.asta_nodo_a.text())
            n2  = int(self.ui.asta_nodo_b.text())
            sud = max(1, int(self.ui.telaio_asta_suddivisione.text() or '1'))
        except ValueError:
            QMessageBox.warning(None, "Input non valido",
                                "Controllare i campi N1, N2, Subdiv."); return
        if n1 not in self.nodi or n2 not in self.nodi:
            QMessageBox.warning(None, "Errore", f"Nodo {n1} o {n2} non esiste."); return

        # Sezione dalla combobox dedicata
        sid = self._cb_sezione_asta.currentData()
        if not sid:
            sid = self._ctrl_sez.id_attivo() or self._ctrl_sez.primo_id()
        if not sid:
            QMessageBox.warning(None, "Errore", "Definire almeno una sezione."); return
        mat_id = self.sezioni.get(sid, {}).get('materiale') or self._ctrl_mat.primo_id()

        aid = self._cnt_aste; self._cnt_aste += 1
        self.aste[aid] = {'n1': n1, 'n2': n2, 'sezione': sid,
                          'materiale': mat_id, 'sud': sud}
        self._aggiorna_table_aste()
        self._aggiorna_viewer()
        self.ui.asta_nodo_a.clear(); self.ui.asta_nodo_b.clear()

    def _aggiorna_table_aste(self):
        self._model_aste.itemChanged.disconnect(self._on_asta_edited)
        m = self._model_aste; m.setRowCount(0)
        for aid, a in self.aste.items():
            sez_nome = self.sezioni.get(a.get('sezione', ''), {}).get('nome', '-')
            # Colonna sezione: testo visibile + UserRole = sid per il delegate
            sez_item = _it(sez_nome)
            sez_item.setData(a.get('sezione', ''), Qt.UserRole)
            m.appendRow([_it(aid), _it(a['n1']), _it(a['n2']),
                         sez_item, _it(a.get('sud', 1))])
        self._model_aste.itemChanged.connect(self._on_asta_edited)

    def _carica_asta_sel(self):
        row = self.ui.tableView_aste.currentIndex().row()
        if row < 0: return
        try:
            aid = int(self._model_aste.item(row, 0).text())
            a = self.aste[aid]
            self.ui.asta_nodo_a.setText(str(a['n1']))
            self.ui.asta_nodo_b.setText(str(a['n2']))
            self.ui.telaio_asta_suddivisione.setText(str(a.get('sud', 1)))
            # Seleziona la sezione nella combobox input
            sid = a.get('sezione', '')
            for i in range(self._cb_sezione_asta.count()):
                if self._cb_sezione_asta.itemData(i) == sid:
                    self._cb_sezione_asta.setCurrentIndex(i); break
        except Exception: pass

    def _on_asta_edited(self, item):
        row = item.row()
        col = item.column()
        try:
            old_aid = list(self.aste.keys())[row]
            a = self.aste[old_aid]

            if col == self._COL_ASTE["id"]:
                new_aid = int(item.text())
                if new_aid != old_aid:
                    del self.aste[old_aid]
                    self.aste[new_aid] = a
            elif col == self._COL_ASTE["n1"]:
                a['n1'] = int(item.text())
            elif col == self._COL_ASTE["n2"]:
                a['n2'] = int(item.text())
            elif col == self._COL_ASTE["sezione"]:
                # UserRole contiene il sid (impostato dal delegate)
                sid = item.data(Qt.UserRole) or item.text()
                a['sezione'] = sid
                a['materiale'] = self.sezioni.get(sid, {}).get('materiale', '')
            elif col == self._COL_ASTE["sud"]:
                a['sud'] = max(1, int(item.text()))

            self._aggiorna_viewer()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # VINCOLI
    # ------------------------------------------------------------------

    def _setup_vincoli(self):
        self.ui.telaio_vincolo_aggiungi.clicked.connect(self._aggiungi_vincolo)

    def _aggiungi_vincolo(self):
        try:
            nodo = int(self.ui.vincolo_nodo.text())
        except ValueError:
            QMessageBox.warning(None, "Input non valido", "Inserire un ID nodo."); return
        if nodo not in self.nodi:
            QMessageBox.warning(None, "Errore", f"Nodo {nodo} non esiste."); return
        gradi = [1 if w.isChecked() else 0 for w in (
            self.ui.vincolo_tx, self.ui.vincolo_ty, self.ui.vincolo_tz,
            self.ui.vincolo_rx, self.ui.vincolo_ry, self.ui.vincolo_rz)]
        vid = self._cnt_vinc; self._cnt_vinc += 1
        self.vincoli[vid] = {'nodo': nodo, 'gradi': gradi}
        self._aggiorna_table_vincoli()
        self._aggiorna_viewer()
        self.ui.vincolo_nodo.clear()

    def _aggiorna_table_vincoli(self):
        self._model_vincoli.itemChanged.disconnect(self._on_vincolo_edited)
        m = self._model_vincoli; m.setRowCount(0)
        for vid, v in self.vincoli.items():
            g = v['gradi']
            m.appendRow([_it(vid), _it(v['nodo']),
                         *[_it("1" if x else "0") for x in g]])
        self._model_vincoli.itemChanged.connect(self._on_vincolo_edited)

    def _carica_vincolo_sel(self):
        row = self.ui.tableView_vincoli.currentIndex().row()
        if row < 0: return
        try:
            vid = int(self._model_vincoli.item(row, 0).text())
            v = self.vincoli[vid]
            self.ui.vincolo_nodo.setText(str(v['nodo']))
            chks = [self.ui.vincolo_tx, self.ui.vincolo_ty, self.ui.vincolo_tz,
                    self.ui.vincolo_rx, self.ui.vincolo_ry, self.ui.vincolo_rz]
            for i, w in enumerate(chks): w.setChecked(bool(v['gradi'][i]))
        except Exception: pass

    def _on_vincolo_edited(self, item):
        row = item.row(); col = item.column()
        try:
            old_vid = list(self.vincoli.keys())[row]
            v = self.vincoli[old_vid]
            if col == self._COL_VINCOLI["id"]:
                new_vid = int(item.text())
                if new_vid != old_vid:
                    del self.vincoli[old_vid]; self.vincoli[new_vid] = v
            elif col == self._COL_VINCOLI["nodo"]:
                v['nodo'] = int(item.text())
            elif col >= 2:
                gdl_idx = col - 2
                v['gradi'][gdl_idx] = 1 if item.text() in ('1', 'true', 'True', '✓') else 0
            self._aggiorna_viewer()
        except Exception: pass

    # ------------------------------------------------------------------
    # CARICHI
    # ------------------------------------------------------------------

    def _setup_carichi(self):
        self.ui.telaio_carico_concentrato_aggiungi.clicked.connect(self._aggiungi_carico_conc)
        self.ui.telaio_carico_distribuito_aggiungi.clicked.connect(self._aggiungi_carico_dist)

    def _aggiungi_carico_conc(self):
        try:
            nodo = int(self.ui.carico_concentrato_nodo.text())
        except ValueError:
            QMessageBox.warning(None, "Input non valido", "Inserire un ID nodo."); return
        if nodo not in self.nodi:
            QMessageBox.warning(None, "Errore", f"Nodo {nodo} non esiste."); return
        forze = [_rf(self.ui.carico_concentrato_fx), _rf(self.ui.carico_concentrato_fy),
                 _rf(self.ui.carico_concentrato_fz), _rf(self.ui.carico_concentrato_mx),
                 _rf(self.ui.carico_concentrato_my), _rf(self.ui.carico_concentrato_mz)]
        cid = self._cnt_cn; self._cnt_cn += 1
        self.carichi_nodi[cid] = {'nodo': nodo, 'forze': forze}
        self._aggiorna_table_carichi(); self._aggiorna_viewer()

    def _aggiungi_carico_dist(self):
        try:
            asta = int(self.ui.carico_distribuito_asta.text())
        except ValueError:
            QMessageBox.warning(None, "Input non valido", "Inserire un ID asta."); return
        if asta not in self.aste:
            QMessageBox.warning(None, "Errore", f"Asta {asta} non esiste."); return
        q = [_rf(self.ui.carico_distribuito_qx), _rf(self.ui.carico_distribuito_qy),
             _rf(self.ui.carico_distribuito_qz)]
        cid = self._cnt_ca; self._cnt_ca += 1
        self.carichi_aste[cid] = {'asta': asta, 'q': q}
        self._aggiorna_table_carichi(); self._aggiorna_viewer()

    def _aggiorna_table_carichi(self):
        m = self._model_carichi; m.setRowCount(0)
        for cid, c in self.carichi_nodi.items():
            f = c['forze']
            m.appendRow([_ro(f"CN{cid}"), _ro("Concentrato"), _ro(f"N{c['nodo']}"),
                         _ro(f"F=[{f[0]:.2f},{f[1]:.2f},{f[2]:.2f}]kN  "
                             f"M=[{f[3]:.2f},{f[4]:.2f},{f[5]:.2f}]kNm")])
        for cid, c in self.carichi_aste.items():
            q = c['q']
            m.appendRow([_ro(f"CA{cid}"), _ro("Distribuito"), _ro(f"A{c['asta']}"),
                         _ro(f"q=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f}]kN/m")])

    # ------------------------------------------------------------------
    # Context menu tabelle (elimina)
    # ------------------------------------------------------------------

    def _ctx_table(self, pos, tv, tipo):
        idx = tv.indexAt(pos)
        if not idx.isValid(): return
        id_str = tv.model().item(idx.row(), 0).text()
        menu = QMenu(); act_del = QAction("🗑️  Elimina", menu); menu.addAction(act_del)
        if menu.exec_(tv.viewport().mapToGlobal(pos)) != act_del: return
        try:
            if tipo == 'nodo':
                del self.nodi[int(id_str)]; self._aggiorna_table_nodi()
            elif tipo == 'asta':
                del self.aste[int(id_str)]; self._aggiorna_table_aste()
            elif tipo == 'vincolo':
                del self.vincoli[int(id_str)]; self._aggiorna_table_vincoli()
            elif tipo == 'carico':
                if id_str.startswith("CN"): del self.carichi_nodi[int(id_str[2:])]
                elif id_str.startswith("CA"): del self.carichi_aste[int(id_str[2:])]
                self._aggiorna_table_carichi()
        except Exception: pass
        self._aggiorna_viewer()

    # ------------------------------------------------------------------
    # ANALISI
    # ------------------------------------------------------------------

    def _setup_analisi(self):
        self.ui.analisi_telaio.clicked.connect(self._esegui_analisi)

    def _esegui_analisi(self):
        if len(self.nodi) < 2 or len(self.aste) < 1:
            QMessageBox.warning(None, "Errore", "Servono almeno 2 nodi e 1 asta."); return
        if not self.vincoli:
            QMessageBox.warning(None, "Errore", "Definire almeno un vincolo."); return
        self.ui.progressBar_telaio.setValue(0)
        self.ui.analisi_telaio.setEnabled(False)
        dati = {k: copy.deepcopy(v) for k, v in {
            'nodi': self.nodi, 'aste': self.aste, 'sezioni': self.sezioni,
            'materiali': self.materiali, 'vincoli': self.vincoli,
            'carichi_nodi': self.carichi_nodi, 'carichi_aste': self.carichi_aste,
        }.items()}
        self._thread = QThread()
        self._worker = CalcoloWorker(dati)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.esegui)
        self._worker.progresso.connect(self.ui.progressBar_telaio.setValue)
        self._worker.finito.connect(self._analisi_ok)
        self._worker.errore.connect(self._analisi_err)
        self._worker.finito.connect(self._thread.quit)
        self._worker.errore.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.ui.analisi_telaio.setEnabled(True))
        self._thread.start()

    def _analisi_ok(self, u_vec, sforzi, t_nodi, t_aste):
        self._u_vec = u_vec; self._sforzi = sforzi
        self._t_nodi = t_nodi; self._t_aste = t_aste
        self._risolto = True
        self.ui.btn_telaio_deformata.click()

    def _analisi_err(self, msg):
        QMessageBox.critical(None, "Errore nel calcolo", msg)
        self.ui.progressBar_telaio.setValue(0)

    # ------------------------------------------------------------------
    # VISTE
    # ------------------------------------------------------------------
    
    def _setup_viste(self):
        ui = self.ui

        # ── Gruppo 1: Vista (Indeformata / Deformata) ──────────────────
        # Esclusivo: uno sempre selezionato
        self._g_vista = QButtonGroup(ui.btn_telaio_indeformata)
        self._g_vista.setExclusive(True)
        for btn, mode in [
            (ui.btn_telaio_indeformata, "Indeformata"),
            (ui.btn_telaio_deformata,   "Deformata"),
        ]:
            btn.setCheckable(True)
            self._g_vista.addButton(btn)
            btn.clicked.connect(lambda _, m=mode: self._attiva_modo(m))

        # ── Gruppo 2: Diagrammi sollecitazioni (N, Vy, Vz, My, Mz) ────
        # Esclusivo: uno sempre selezionato.
        # Al click rimette anche "Indeformata" nel gruppo vista.
        self._g_diag = QButtonGroup(ui.btn_telaio_n)
        self._g_diag.setExclusive(True)
        for btn, mode in [
            (ui.btn_telaio_n,  "N"),
            (ui.btn_telaio_ty, "Vy"),
            (ui.btn_telaio_tz, "Vz"),
            (ui.btn_telaio_my, "My"),
            (ui.btn_telaio_mz, "Mz"),
        ]:
            btn.setCheckable(True)
            self._g_diag.addButton(btn)
            btn.clicked.connect(lambda _, m=mode: self._attiva_diagramma_con_reset(m))

        # ── Gruppo 3: Preset camera (3D, X, Y, Z) ──────────────────────
        # Esclusivo: uno sempre selezionato.
        self._g_cam = QButtonGroup(ui.btn_telaio_3d)
        self._g_cam.setExclusive(True)
        for btn, preset in [
            (ui.btn_telaio_3d, "3d"),
            (ui.btn_telaio_x,  "x"),
            (ui.btn_telaio_y,  "y"),
            (ui.btn_telaio_z,  "z"),
        ]:
            btn.setCheckable(True)
            self._g_cam.addButton(btn)
            btn.clicked.connect(lambda _, p=preset: self._viewer.imposta_vista(p))

        # Selezioni iniziali di default
        QTimer.singleShot(0, ui.btn_telaio_indeformata.click)            # gruppo vista
        QTimer.singleShot(0, lambda: ui.btn_telaio_n.setChecked(True))   # gruppo diagrammi
        QTimer.singleShot(0, lambda: ui.btn_telaio_3d.setChecked(True))  # gruppo camera

    def _attiva_modo(self, mode: str):
        """Punto unico di ingresso per tutti i bottoni vista/diagramma."""
        if not self._risolto and mode not in ("Indeformata",):
            QMessageBox.information(None, "Info", "Eseguire prima l'analisi.")
            # Rimette il check sull'indeformata
            self.ui.btn_telaio_indeformata.setChecked(True)
            return
        if mode == "Indeformata":
            self._viewer.aggiorna_geometria(self.nodi, self.aste, self.vincoli,
                                            self.carichi_nodi, self.carichi_aste)
        elif mode == "Deformata":
            self._viewer.aggiorna_risultati(self._u_vec, self._sforzi,
                                            self._t_nodi, self._t_aste,
                                            mode, self._scala_corrente())
        else:
            # Diagramma sollecitazione
            self._viewer.aggiorna_risultati(self._u_vec, self._sforzi,
                                            self._t_nodi, self._t_aste,
                                            mode, self._scala_auto(mode))

    def _attiva_diagramma_con_reset(self, mode: str):
        """
        Attiva un diagramma di sollecitazione.
        Rimette sempre il gruppo Vista su 'Indeformata' (i diagrammi si
        sovrappongono alla struttura indeformata, non a quella deformata).
        """
        self.ui.btn_telaio_indeformata.setChecked(True)
        self._attiva_modo(mode)

    # Mantieni alias per compatibilità chiamate interne esistenti
    def _cambia_vista(self, mode):
        self._attiva_modo(mode)

    def _attiva_diagramma(self, mode):
        self._attiva_modo(mode)

    def _scala_auto(self, mode):
        idx_map = {"N":[0,6],"Vy":[1,7],"Vz":[2,8],"My":[4,10],"Mz":[5,11]}
        i1, i2 = idx_map.get(mode, [0, 6])
        if self._sforzi is None: return 1.0
        max_v = float(np.max(np.abs(self._sforzi[:, [i1, i2]])))
        return 0.5 / max(max_v, 1e-8)

    # ------------------------------------------------------------------
    # SCALA DEFORMAZIONE
    # ------------------------------------------------------------------

    def _setup_scala(self):
        self._upd_scala = False
        self.ui.telaio_deformazione_scala.setText("10")
        self.ui.telaio_deformazione_slider.setMinimum(0)
        self.ui.telaio_deformazione_slider.setMaximum(1000)
        self.ui.telaio_deformazione_slider.setValue(10)
        self.ui.telaio_deformazione_scala.editingFinished.connect(self._scala_da_le)
        self.ui.telaio_deformazione_slider.valueChanged.connect(self._scala_da_slider)

    def _scala_da_le(self):
        if self._upd_scala: return
        self._upd_scala = True
        try:
            v = max(0.0, min(1000.0, float(self.ui.telaio_deformazione_scala.text())))
            self.ui.telaio_deformazione_slider.setValue(int(v))
            self._applica_scala(v)
        except Exception: pass
        self._upd_scala = False

    def _scala_da_slider(self, v):
        if self._upd_scala: return
        self._upd_scala = True
        self.ui.telaio_deformazione_scala.setText(str(v))
        self._applica_scala(float(v))
        self._upd_scala = False

    def _scala_corrente(self):
        try: return float(self.ui.telaio_deformazione_scala.text())
        except Exception: return 10.0

    def _applica_scala(self, scala):
        if self._risolto and self._viewer.view_mode == "Deformata":
            self._viewer.aggiorna_risultati(self._u_vec, self._sforzi,
                                            self._t_nodi, self._t_aste,
                                            "Deformata", scala)

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def _aggiorna_viewer(self):
        self._viewer.aggiorna_geometria(self.nodi, self.aste, self.vincoli,
                                        self.carichi_nodi, self.carichi_aste)

    # ------------------------------------------------------------------
    # SALVATAGGIO / CARICAMENTO
    # ------------------------------------------------------------------

    def _setup_salvataggio(self):
        # I bottoni btn_main_salva / btn_main_carica sono collegati centralmente
        # da main.py tramite get_dati_salvataggio() e carica_dati().
        pass

    # ------------------------------------------------------------------
    # API PUBBLICA SALVATAGGIO (usata da main.py)
    # ------------------------------------------------------------------

    def get_dati_salvataggio(self):
        """Restituisce un dict serializzabile con tutto il modello telaio."""
        return {
            'materiali':    self.materiali,
            'sezioni':      self.sezioni,
            'nodi':         {str(k): v for k, v in self.nodi.items()},
            'aste':         {str(k): v for k, v in self.aste.items()},
            'vincoli':      {str(k): v for k, v in self.vincoli.items()},
            'carichi_nodi': {str(k): v for k, v in self.carichi_nodi.items()},
            'carichi_aste': {str(k): v for k, v in self.carichi_aste.items()},
            'contatori': {
                'nodi': self._cnt_nodi, 'aste': self._cnt_aste,
                'vinc': self._cnt_vinc, 'cn':   self._cnt_cn, 'ca': self._cnt_ca,
            },
        }

    def carica_dati(self, dati):
        """Ripristina il modello telaio da un dict precedentemente salvato."""
        self.materiali.clear(); self.materiali.update(dati.get('materiali', {}))
        self.sezioni.clear();   self.sezioni.update(dati.get('sezioni', {}))
        self.nodi         = {int(k): v for k, v in dati.get('nodi', {}).items()}
        self.aste         = {int(k): v for k, v in dati.get('aste', {}).items()}
        self.vincoli      = {int(k): v for k, v in dati.get('vincoli', {}).items()}
        self.carichi_nodi = {int(k): v for k, v in dati.get('carichi_nodi', {}).items()}
        self.carichi_aste = {int(k): v for k, v in dati.get('carichi_aste', {}).items()}
        cnt = dati.get('contatori', {})
        self._cnt_nodi = cnt.get('nodi', 0); self._cnt_aste = cnt.get('aste', 0)
        self._cnt_vinc = cnt.get('vinc', 0); self._cnt_cn   = cnt.get('cn', 0)
        self._cnt_ca   = cnt.get('ca', 0)
        self._risolto = False; self._u_vec = self._sforzi = None
        self._ctrl_mat._ricostruisci_layout()
        self._ctrl_mat.aggiorna_combobox(self.ui.combobox_teliaio_materiali)
        self._ctrl_sez._ricostruisci_layout()
        self._aggiorna_cb_sezioni_asta()
        self._aggiorna_table_nodi(); self._aggiorna_table_aste()
        self._aggiorna_table_vincoli(); self._aggiorna_table_carichi()
        self._aggiorna_viewer()


# ======================================================================
#  Helper module-level
# ======================================================================

def _crea_model(headers):
    m = QStandardItemModel(); m.setHorizontalHeaderLabels(headers); return m

def _it(testo):
    """Item editabile."""
    return QStandardItem(str(testo))

def _ro(testo):
    """Item di sola lettura."""
    it = QStandardItem(str(testo))
    it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    return it

def _rf(widget, default=0.0):
    try: return float(widget.text().replace(',', '.'))
    except Exception: return default