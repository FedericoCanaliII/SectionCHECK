"""
telaio_materiali_sezioni.py
Controller dedicato alla gestione dei pannelli Materiali e Sezioni
del modulo telaio FEM.

Stile UI coerente con GestioneMateriali dell'app principale:
  - Bottoni rinominabili con doppio click
  - Eliminazione con tasto destro → menu contestuale
  - Selezione bottone → popola i campi
  - Salvataggio automatico dei campi quando si perde il focus

Unità di misura nei campi UI:
  E, G → MPa  |  rho → kg/m³  |  A → mm²  |  Iy, Iz, J → mm⁴
"""

import copy

from PyQt5.QtWidgets import (
    QButtonGroup, QPushButton, QMenu, QAction,
    QLineEdit, QInputDialog, QMessageBox,
)
from PyQt5.QtCore import Qt

from .telaio_dati_standard import MATERIALI_STANDARD, SEZIONI_STANDARD


# ══════════════════════════════════════════════════════════════════════
#  Stile bottone (coerente col pattern già usato nell'app)
# ══════════════════════════════════════════════════════════════════════
_BTN_STYLE = """
QPushButton {
    font: 400 12pt "Segoe UI";
    color: rgb(255, 255, 255);
    padding-bottom: 4px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: rgb(40, 40, 40);
    border: 1px solid rgb(120, 120, 120);
}
QPushButton:checked {
    background-color: rgb(30, 30, 30);
}
"""


class GestioneMaterialiTelaio:
    """
    Gestisce il pannello 'Materiali' nella sezione Sezioni del telaio.

    Parametri UI utilizzati:
      layout_telaio_materiali     QLayout (VBox)
      btn_aggiungi_materiali      QPushButton
      telaio_materiali_elastico   QLineEdit  [MPa]
      telaio_materiali_taglio     QLineEdit  [MPa]
      telaio_materiali_densita    QLineEdit  [kg/m³]
      combobox_teliaio_materiali  QComboBox  (usata anche in GestioneSezioniTelaio)
    """

    def __init__(self, ui, materiali_dict: dict, on_changed=None):
        """
        ui             : oggetto UI Qt Designer
        materiali_dict : dizionario condiviso  {mid: {...}}  (ref. esterna)
        on_changed     : callback chiamata ogni volta che la lista cambia
        """
        self.ui = ui
        self.materiali = materiali_dict
        self._on_changed = on_changed or (lambda: None)
        self._attivo: str | None = None
        self._updating = False

        self._btn_group = QButtonGroup()
        self._btn_group.setExclusive(True)
        self._btn_map: dict[str, QPushButton] = {}   # mid → btn

        # Carica i materiali standard se il dizionario è vuoto
        if not self.materiali:
            for mid, mat in MATERIALI_STANDARD.items():
                self.materiali[mid] = copy.deepcopy(mat)

        # Connessioni
        self.ui.btn_aggiungi_materiali.clicked.connect(self._aggiungi)
        # Salvataggio campi quando si perde il focus
        for w in (self.ui.telaio_materiali_elastico,
                  self.ui.telaio_materiali_taglio,
                  self.ui.telaio_materiali_densita):
            w.editingFinished.connect(self._salva_attivo)

        # Popola il layout
        self._ricostruisci_layout()

    # ──────────────────────────────────────────────────────────────────
    # Interfaccia pubblica
    # ──────────────────────────────────────────────────────────────────

    def aggiorna_combobox(self, combobox):
        """Aggiorna una QComboBox con i materiali correnti."""
        attuale = combobox.currentData()
        combobox.blockSignals(True)
        combobox.clear()
        for mid, mat in self.materiali.items():
            combobox.addItem(mat['nome'], userData=mid)
        # Ripristina la selezione precedente se esiste ancora
        for i in range(combobox.count()):
            if combobox.itemData(i) == attuale:
                combobox.setCurrentIndex(i)
                break
        combobox.blockSignals(False)

    def primo_id(self) -> str | None:
        """Restituisce il primo id materiale disponibile."""
        return next(iter(self.materiali), None)

    # ──────────────────────────────────────────────────────────────────
    # Aggiunta
    # ──────────────────────────────────────────────────────────────────

    def _aggiungi(self):
        # Genera un id univoco
        idx = 0
        while f"mat_{idx}" in self.materiali:
            idx += 1
        mid = f"mat_{idx}"

        # Legge i valori correnti dai campi (o usa default)
        self.materiali[mid] = {
            "nome": f"Materiale {len(self.materiali) + 1}",
            "E":   self._read_float(self.ui.telaio_materiali_elastico, 210_000.0),
            "G":   self._read_float(self.ui.telaio_materiali_taglio,    81_000.0),
            "rho": self._read_float(self.ui.telaio_materiali_densita,    7_850.0),
        }
        self._ricostruisci_layout()
        # Seleziona il nuovo bottone
        if mid in self._btn_map:
            self._btn_map[mid].click()
        self._on_changed()

    # ──────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────

    def _ricostruisci_layout(self):
        layout = self.ui.layout_telaio_materiali

        # Rimuovi solo i bottoni dinamici, lasciando intatti
        # label, btn_aggiungi, spacer fisso e spacer finale.
        for btn in list(self._btn_map.values()):
            layout.removeWidget(btn)
            btn.deleteLater()

        self._btn_group = QButtonGroup()
        self._btn_group.setExclusive(True)
        self._btn_map.clear()

        # Inserisci i bottoni subito prima dell'ultimo spacer espandibile.
        insert_pos = layout.count()
        for i in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(i)
            if item and item.spacerItem():
                insert_pos = i
                break

        for mid, mat in self.materiali.items():
            btn = self._crea_bottone(mat['nome'])
            btn.clicked.connect(lambda _, m=mid: self._seleziona(m))
            btn.customContextMenuRequested.connect(
                lambda pos, m=mid: self._ctx_menu(pos, m))
            btn.mouseDoubleClickEvent = lambda e, m=mid: self._rinomina_inline(m)
            self._btn_group.addButton(btn)
            self._btn_map[mid] = btn
            layout.insertWidget(insert_pos, btn)
            insert_pos += 1

        # Mantieni / ripristina la selezione corrente
        if self._attivo and self._attivo in self._btn_map:
            self._btn_map[self._attivo].setChecked(True)
        elif self._btn_map:
            primo = next(iter(self._btn_map))
            self._btn_map[primo].click()

        self._on_changed()

    def _crea_bottone(self, testo: str) -> QPushButton:
        btn = QPushButton(testo)
        btn.setCheckable(True)
        btn.setStyleSheet(_BTN_STYLE)
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        return btn

    # ──────────────────────────────────────────────────────────────────
    # Selezione
    # ──────────────────────────────────────────────────────────────────

    def _seleziona(self, mid: str):
        if mid not in self.materiali:
            return
        self._attivo = mid
        mat = self.materiali[mid]
        self._updating = True
        self.ui.telaio_materiali_elastico.setText(f"{mat['E']:.6g}")
        self.ui.telaio_materiali_taglio.setText(f"{mat['G']:.6g}")
        self.ui.telaio_materiali_densita.setText(f"{mat['rho']:.6g}")
        self._updating = False

    # ──────────────────────────────────────────────────────────────────
    # Salvataggio
    # ──────────────────────────────────────────────────────────────────

    def _salva_attivo(self):
        if self._updating or not self._attivo:
            return
        if self._attivo not in self.materiali:
            return
        self.materiali[self._attivo]['E']   = self._read_float(self.ui.telaio_materiali_elastico, 210_000.0)
        self.materiali[self._attivo]['G']   = self._read_float(self.ui.telaio_materiali_taglio,    81_000.0)
        self.materiali[self._attivo]['rho'] = self._read_float(self.ui.telaio_materiali_densita,    7_850.0)
        self._on_changed()

    # ──────────────────────────────────────────────────────────────────
    # Menu contestuale (tasto destro)
    # ──────────────────────────────────────────────────────────────────

    def _ctx_menu(self, pos, mid: str):
        btn = self._btn_map.get(mid)
        if not btn:
            return
        menu = QMenu()
        act_rin = QAction("✏️  Rinomina", menu)
        act_del = QAction("🗑️  Elimina", menu)
        menu.addAction(act_rin)
        menu.addSeparator()
        menu.addAction(act_del)
        chosen = menu.exec_(btn.mapToGlobal(pos))

        if chosen == act_rin:
            self._rinomina_dialog(mid)
        elif chosen == act_del:
            self._elimina(mid)

    def _rinomina_dialog(self, mid: str):
        nome_attuale = self.materiali[mid]['nome']
        nome, ok = QInputDialog.getText(
            None, "Rinomina materiale", "Nuovo nome:", text=nome_attuale)
        if ok and nome.strip():
            self.materiali[mid]['nome'] = nome.strip()
            self._ricostruisci_layout()
            self._on_changed()

    def _rinomina_inline(self, mid: str):
        """Rinomina con un QLineEdit sovrapposto al bottone (doppio click)."""
        btn = self._btn_map.get(mid)
        if not btn:
            return
        le = QLineEdit(btn.text(), btn.parent())
        le.setGeometry(btn.geometry())
        le.setFont(btn.font())
        le.selectAll()
        le.show()
        le.setFocus()
        btn.setEnabled(False)

        def conferma():
            nuovo = le.text().strip()
            if nuovo:
                self.materiali[mid]['nome'] = nuovo
            le.deleteLater()
            btn.setEnabled(True)
            self._ricostruisci_layout()
            self._on_changed()

        le.editingFinished.connect(conferma)

    def _elimina(self, mid: str):
        if len(self.materiali) <= 1:
            QMessageBox.warning(None, "Avviso",
                                "Deve esistere almeno un materiale.")
            return
        del self.materiali[mid]
        if self._attivo == mid:
            self._attivo = next(iter(self.materiali), None)
        self._ricostruisci_layout()
        self._on_changed()

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_float(widget: QLineEdit, default: float) -> float:
        try:
            return float(widget.text().replace(',', '.'))
        except Exception:
            return default


# ══════════════════════════════════════════════════════════════════════
#  GESTIONE SEZIONI
# ══════════════════════════════════════════════════════════════════════

class GestioneSezioniTelaio:
    """
    Gestisce il pannello 'Sezioni' nella sezione Sezioni del telaio.

    Parametri UI utilizzati:
      layout_telaio_sezioni       QLayout (VBox)
      btn_aggiungi_sezioni        QPushButton
      telaio_sezione_area         QLineEdit  [mm²]
      telaio_sezione_inerzia_y    QLineEdit  [mm⁴]
      telaio_sezione_inerzia_z    QLineEdit  [mm⁴]
      telaio_sezione_torsione     QLineEdit  [mm⁴]
      combobox_teliaio_materiali  QComboBox
    """

    def __init__(self, ui, sezioni_dict: dict, materiali_dict: dict,
                 on_changed=None):
        self.ui = ui
        self.sezioni   = sezioni_dict
        self.materiali = materiali_dict
        self._on_changed = on_changed or (lambda: None)
        self._attiva: str | None = None
        self._updating = False

        self._btn_group = QButtonGroup()
        self._btn_group.setExclusive(True)
        self._btn_map: dict[str, QPushButton] = {}

        # Carica sezioni standard se il dizionario è vuoto
        if not self.sezioni:
            for sid, sez in SEZIONI_STANDARD.items():
                self.sezioni[sid] = copy.deepcopy(sez)

        # Connessioni
        self.ui.btn_aggiungi_sezioni.clicked.connect(self._aggiungi)
        for w in (self.ui.telaio_sezione_area,
                  self.ui.telaio_sezione_inerzia_y,
                  self.ui.telaio_sezione_inerzia_z,
                  self.ui.telaio_sezione_torsione):
            w.editingFinished.connect(self._salva_attiva)
        self.ui.combobox_teliaio_materiali.currentIndexChanged.connect(
            self._salva_attiva)

        # Popola il layout
        self._ricostruisci_layout()

    # ──────────────────────────────────────────────────────────────────
    # Interfaccia pubblica
    # ──────────────────────────────────────────────────────────────────

    def id_attivo(self) -> str | None:
        return self._attiva

    def primo_id(self) -> str | None:
        return next(iter(self.sezioni), None)

    # ──────────────────────────────────────────────────────────────────
    # Aggiunta
    # ──────────────────────────────────────────────────────────────────

    def _aggiungi(self):
        idx = 0
        while f"sez_{idx}" in self.sezioni:
            idx += 1
        sid = f"sez_{idx}"
        mat_id = self.ui.combobox_teliaio_materiali.currentData()
        if not mat_id:
            mat_id = next(iter(self.materiali), None)

        self.sezioni[sid] = {
            "nome":      f"Sezione {len(self.sezioni) + 1}",
            "A":         self._read_float(self.ui.telaio_sezione_area,       3_600.0),
            "Iy":        self._read_float(self.ui.telaio_sezione_inerzia_y,  36_900_000.0),
            "Iz":        self._read_float(self.ui.telaio_sezione_inerzia_z,  13_360_000.0),
            "J":         self._read_float(self.ui.telaio_sezione_torsione,      59_100.0),
            "materiale": mat_id,
        }
        self._ricostruisci_layout()
        if sid in self._btn_map:
            self._btn_map[sid].click()
        self._on_changed()

    # ──────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────

    def _ricostruisci_layout(self):
        layout = self.ui.layout_telaio_sezioni

        # Rimuovi solo i bottoni dinamici, lasciando intatti
        # label, btn_aggiungi, spacer fisso e spacer finale.
        for btn in list(self._btn_map.values()):
            layout.removeWidget(btn)
            btn.deleteLater()

        self._btn_group = QButtonGroup()
        self._btn_group.setExclusive(True)
        self._btn_map.clear()

        # Inserisci i bottoni subito prima dell'ultimo spacer espandibile.
        insert_pos = layout.count()
        for i in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(i)
            if item and item.spacerItem():
                insert_pos = i
                break

        for sid, sez in self.sezioni.items():
            mat_nome = self.materiali.get(sez.get('materiale', ''), {}).get('nome', '?')
            etichetta = f"{sez['nome']}  [{mat_nome}]"
            btn = self._crea_bottone(etichetta)
            btn.clicked.connect(lambda _, s=sid: self._seleziona(s))
            btn.customContextMenuRequested.connect(
                lambda pos, s=sid: self._ctx_menu(pos, s))
            btn.mouseDoubleClickEvent = lambda e, s=sid: self._rinomina_inline(s)
            self._btn_group.addButton(btn)
            self._btn_map[sid] = btn
            layout.insertWidget(insert_pos, btn)
            insert_pos += 1

        if self._attiva and self._attiva in self._btn_map:
            self._btn_map[self._attiva].setChecked(True)
        elif self._btn_map:
            primo = next(iter(self._btn_map))
            self._btn_map[primo].click()

        self._on_changed()

    def _crea_bottone(self, testo: str) -> QPushButton:
        btn = QPushButton(testo)
        btn.setCheckable(True)
        btn.setStyleSheet(_BTN_STYLE)
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        return btn

    # ──────────────────────────────────────────────────────────────────
    # Selezione
    # ──────────────────────────────────────────────────────────────────

    def _seleziona(self, sid: str):
        if sid not in self.sezioni:
            return
        self._attiva = sid
        sez = self.sezioni[sid]
        self._updating = True
        self.ui.telaio_sezione_area.setText(f"{sez['A']:.6g}")
        self.ui.telaio_sezione_inerzia_y.setText(f"{sez['Iy']:.6g}")
        self.ui.telaio_sezione_inerzia_z.setText(f"{sez['Iz']:.6g}")
        self.ui.telaio_sezione_torsione.setText(f"{sez['J']:.6g}")
        # Imposta il materiale nella combobox
        cb = self.ui.combobox_teliaio_materiali
        for i in range(cb.count()):
            if cb.itemData(i) == sez.get('materiale'):
                cb.setCurrentIndex(i)
                break
        self._updating = False

    # ──────────────────────────────────────────────────────────────────
    # Salvataggio
    # ──────────────────────────────────────────────────────────────────

    def _salva_attiva(self):
        if self._updating or not self._attiva:
            return
        if self._attiva not in self.sezioni:
            return
        self.sezioni[self._attiva]['A']         = self._read_float(self.ui.telaio_sezione_area,       3_600.0)
        self.sezioni[self._attiva]['Iy']        = self._read_float(self.ui.telaio_sezione_inerzia_y,  36_900_000.0)
        self.sezioni[self._attiva]['Iz']        = self._read_float(self.ui.telaio_sezione_inerzia_z,  13_360_000.0)
        self.sezioni[self._attiva]['J']         = self._read_float(self.ui.telaio_sezione_torsione,      59_100.0)
        self.sezioni[self._attiva]['materiale'] = self.ui.combobox_teliaio_materiali.currentData()
        # Aggiorna l'etichetta del bottone
        self._ricostruisci_layout()
        self._on_changed()

    # ──────────────────────────────────────────────────────────────────
    # Menu contestuale
    # ──────────────────────────────────────────────────────────────────

    def _ctx_menu(self, pos, sid: str):
        btn = self._btn_map.get(sid)
        if not btn:
            return
        menu = QMenu()
        act_rin = QAction("✏️  Rinomina", menu)
        act_del = QAction("🗑️  Elimina", menu)
        menu.addAction(act_rin)
        menu.addSeparator()
        menu.addAction(act_del)
        chosen = menu.exec_(btn.mapToGlobal(pos))

        if chosen == act_rin:
            self._rinomina_dialog(sid)
        elif chosen == act_del:
            self._elimina(sid)

    def _rinomina_dialog(self, sid: str):
        nome, ok = QInputDialog.getText(
            None, "Rinomina sezione", "Nuovo nome:",
            text=self.sezioni[sid]['nome'])
        if ok and nome.strip():
            self.sezioni[sid]['nome'] = nome.strip()
            self._ricostruisci_layout()
            self._on_changed()

    def _rinomina_inline(self, sid: str):
        btn = self._btn_map.get(sid)
        if not btn:
            return
        le = QLineEdit(self.sezioni[sid]['nome'], btn.parent())
        le.setGeometry(btn.geometry())
        le.setFont(btn.font())
        le.selectAll()
        le.show()
        le.setFocus()
        btn.setEnabled(False)

        def conferma():
            nuovo = le.text().strip()
            if nuovo:
                self.sezioni[sid]['nome'] = nuovo
            le.deleteLater()
            btn.setEnabled(True)
            self._ricostruisci_layout()
            self._on_changed()

        le.editingFinished.connect(conferma)

    def _elimina(self, sid: str):
        if len(self.sezioni) <= 1:
            QMessageBox.warning(None, "Avviso",
                                "Deve esistere almeno una sezione.")
            return
        del self.sezioni[sid]
        if self._attiva == sid:
            self._attiva = next(iter(self.sezioni), None)
        self._ricostruisci_layout()
        self._on_changed()

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_float(widget: QLineEdit, default: float) -> float:
        try:
            return float(widget.text().replace(',', '.'))
        except Exception:
            return default
