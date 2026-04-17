"""
gestione_fem_struttura.py – Controller del modulo FEM Struttura.

Responsabilita':
  - Popola la combobox con le strutture del progetto
  - Gestisce la connessione tra interfaccia e moduli di analisi
  - Lancia l'analisi OpenSees in un thread separato
  - Gestisce la visualizzazione dei risultati (diagrammi, deformata, tensioni)
  - Gestisce la progress bar e i pulsanti esclusivi
"""
from __future__ import annotations

import os

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QVBoxLayout, QMessageBox, QButtonGroup
from PyQt5.QtCore import QThread, pyqtSignal

from .raccolta_dati_struttura import RaccoltaDatiStruttura
from .generatore_dati_materiali import GeneratoreDatiMateriali
from .generatore_dati_sezioni import GeneratoreDatiSezioni
from .generatore_mesh_struttura import GeneratoreMeshStruttura, MeshStruttura
from .analisi_fem_struttura import AnalisiStruttura
from .risultati_struttura import RisultatiFEMStruttura
from .disegno_fem_struttura import FEMStrutturaSpazio3D


# ==============================================================================
#  THREAD DI ANALISI
# ==============================================================================

class _AnalisiThread(QThread):
    """Esegue l'analisi FEM in background."""

    avanzamento = pyqtSignal(int)
    completato = pyqtSignal(object)       # RisultatiFEMStruttura
    errore = pyqtSignal(str)

    def __init__(self, mesh, materiali, sezioni, dati_struttura,
                 gravita, peso_proprio, parent=None):
        super().__init__(parent)
        self._mesh = mesh
        self._materiali = materiali
        self._sezioni = sezioni
        self._dati = dati_struttura
        self._gravita = gravita
        self._peso_proprio = peso_proprio

    def run(self):
        try:
            analisi = AnalisiStruttura()
            risultati = analisi.esegui(
                self._mesh,
                self._materiali,
                self._sezioni,
                self._dati,
                gravita=self._gravita,
                peso_proprio=self._peso_proprio,
                progress_cb=lambda p: self.avanzamento.emit(p),
            )

            # Salva risultati su disco
            cartella = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "analisi_svolte")
            nome = self._dati.get("_nome", "struttura")
            analisi.salva_risultati(risultati, nome, cartella)

            self.completato.emit(risultati)
        except Exception as e:
            self.errore.emit(str(e))


# ==============================================================================
#  CONTROLLER
# ==============================================================================

class GestioneFemStruttura:
    """Controller del modulo FEM Struttura."""

    def __init__(self, ui, main_window) -> None:
        self._ui = ui
        self._main = main_window
        self._raccolta = RaccoltaDatiStruttura(main_window)
        self._gen_mat = GeneratoreDatiMateriali(main_window)
        self._gen_sez = GeneratoreDatiSezioni(main_window)

        # Stato
        self._cat_corrente: str | None = None
        self._nome_corrente: str | None = None
        self._dati_correnti: dict | None = None
        self._mesh: MeshStruttura | None = None
        self._risultati: RisultatiFEMStruttura | None = None
        self._materiali_risolti: dict | None = None
        self._sezioni_risolte: dict | None = None
        self._thread: _AnalisiThread | None = None

        # Widget 3D
        self._spazio = FEMStrutturaSpazio3D()
        self._setup_widget_3d()
        self._setup_connessioni()
        self._setup_pulsanti_vista()
        self._setup_pulsanti_risultati()
        self._reset_ui()

    # ------------------------------------------------------------------
    #  SETUP
    # ------------------------------------------------------------------

    def _setup_widget_3d(self):
        container = self._ui.fem_struttura_widget
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(1, 1, 1, 1)
            lay.setSpacing(0)
        else:
            lay.setContentsMargins(1, 1, 1, 1)
            lay.setSpacing(0)
        lay.addWidget(self._spazio)

    def _setup_connessioni(self):
        ui = self._ui

        # Combobox struttura
        ui.fem_struttura_combobox.currentIndexChanged.connect(
            self._on_struttura_cambiata
        )

        # Pulsante analisi
        ui.fem_struttura_btn_analisi.clicked.connect(self._avvia_analisi)

        # Scala deformazione (reagisce sia durante la digitazione sia a fine editing)
        ui.fem_struttura_scala.textChanged.connect(self._on_scala_cambiata)
        ui.fem_struttura_scala.editingFinished.connect(self._on_scala_cambiata)

        # Tensioni: abilitato solo in modo deformata
        ui.fem_struttura_btn_tensioni.setCheckable(True)
        ui.fem_struttura_btn_tensioni.setChecked(False)
        ui.fem_struttura_btn_tensioni.setEnabled(False)
        ui.fem_struttura_btn_tensioni.clicked.connect(self._on_tensioni_toggle)

        # Centra vista
        if hasattr(ui, 'fem_struttura_centra'):
            ui.fem_struttura_centra.clicked.connect(self._centra_vista)

    def _setup_pulsanti_vista(self):
        """Configura i pulsanti vista (checkable, esclusivi)."""
        ui = self._ui
        self._vista_group = QButtonGroup(ui.fem_struttura_btn_vista_3d.parent())
        self._vista_group.setExclusive(True)

        for btn in (ui.fem_struttura_btn_vista_3d, ui.fem_struttura_btn_vista_x,
                    ui.fem_struttura_btn_vista_y, ui.fem_struttura_btn_vista_z):
            btn.setCheckable(True)
            self._vista_group.addButton(btn)

        ui.fem_struttura_btn_vista_3d.setChecked(True)
        ui.fem_struttura_btn_vista_3d.clicked.connect(
            lambda: self._spazio.imposta_vista("3d"))
        ui.fem_struttura_btn_vista_x.clicked.connect(
            lambda: self._spazio.imposta_vista("x"))
        ui.fem_struttura_btn_vista_y.clicked.connect(
            lambda: self._spazio.imposta_vista("y"))
        ui.fem_struttura_btn_vista_z.clicked.connect(
            lambda: self._spazio.imposta_vista("z"))

    def _setup_pulsanti_risultati(self):
        """Configura i pulsanti dei risultati (checkable, esclusivi)."""
        ui = self._ui
        self._risultati_group = QButtonGroup(
            ui.fem_struttura_btn_indeformata.parent())
        self._risultati_group.setExclusive(True)

        pulsanti = [
            (ui.fem_struttura_btn_indeformata, "indeformata"),
            (ui.fem_struttura_btn_N, "N"),
            (ui.fem_struttura_btn_Ty, "Vy"),
            (ui.fem_struttura_btn_Tz, "Vz"),
            (ui.fem_struttura_btn_My, "My"),
            (ui.fem_struttura_btn_Mz, "Mz"),
            (ui.fem_struttura_btn_deformata, "deformata"),
        ]

        for btn, modo in pulsanti:
            btn.setCheckable(True)
            self._risultati_group.addButton(btn)
            btn.clicked.connect(lambda _, m=modo: self._on_modo_cambiato(m))

        ui.fem_struttura_btn_indeformata.setChecked(True)

    def _reset_ui(self):
        ui = self._ui
        ui.fem_struttura_progressBar.setValue(0)
        # Sempre valore di default "pulito" per i campi numerici:
        if not ui.fem_struttura_definizione.text().strip():
            ui.fem_struttura_definizione.setText("5")
        if not ui.fem_struttura_gravita.text().strip():
            ui.fem_struttura_gravita.setText("9.81")

        # Scala deformazione: forza SEMPRE 1.0 (il default vuoto del .ui
        # altrimenti lasciava la scala a 0 in memoria)
        txt = ui.fem_struttura_scala.text().strip()
        try:
            val = float(txt)
            if val <= 0.0:
                raise ValueError
        except (ValueError, TypeError):
            ui.fem_struttura_scala.blockSignals(True)
            ui.fem_struttura_scala.setText("1.0")
            ui.fem_struttura_scala.blockSignals(False)
            val = 1.0
        # Sincronizza lo stato interno anche se il signal non è scattato
        self._spazio.set_scala_deformazione(val)

        ui.fem_struttura_pesoproprio_radioButton_si.setChecked(True)

    # ------------------------------------------------------------------
    #  COMBOBOX
    # ------------------------------------------------------------------

    def _popola_combobox(self):
        cb = self._ui.fem_struttura_combobox
        cb.blockSignals(True)
        cb.clear()

        strutture = self._raccolta.lista_strutture()
        for cat, nome in strutture:
            cb.addItem(f"[{cat}] {nome}", (cat, nome))

        if cb.count() > 0:
            # Prova a riselezionare la struttura precedente
            if self._cat_corrente and self._nome_corrente:
                trovato = False
                for i in range(cb.count()):
                    data = cb.itemData(i)
                    if data == (self._cat_corrente, self._nome_corrente):
                        cb.setCurrentIndex(i)
                        trovato = True
                        break
                if not trovato:
                    cb.setCurrentIndex(0)
                    cb.blockSignals(False)
                    self._on_struttura_cambiata(0)
                    return
            else:
                cb.setCurrentIndex(0)
                cb.blockSignals(False)
                self._on_struttura_cambiata(0)
                return

        cb.blockSignals(False)

    def _on_struttura_cambiata(self, index: int):
        cb = self._ui.fem_struttura_combobox
        if index < 0 or index >= cb.count():
            return

        data = cb.itemData(index)
        if data is None:
            return

        cat, nome = data
        self._cat_corrente = cat
        self._nome_corrente = nome

        # Reset stato
        self._risultati = None
        self._mesh = None
        self._materiali_risolti = None
        self._sezioni_risolte = None
        self._ui.fem_struttura_progressBar.setValue(0)

        # Parsa e carica struttura
        dati = self._raccolta.dati_struttura(cat, nome)
        self._dati_correnti = dati

        if dati:
            self._spazio.set_dati(dati)
            self._spazio.set_risultati(None)
            self._spazio.set_modo("indeformata")
            self._ui.fem_struttura_btn_indeformata.setChecked(True)
            self._aggiorna_stato_tensioni("indeformata")
            self._spazio.centra_vista()
            print(f">> FEM Struttura: caricata '{nome}' "
                  f"({len(dati.get('nodi', {}))} nodi, "
                  f"{len(dati.get('aste', {}))} aste, "
                  f"{len(dati.get('shell', {}))} shell)")
        else:
            self._spazio.set_dati(None)
            print(f"WARN  FEM Struttura: struttura '{nome}' vuota o non valida.")

    # ------------------------------------------------------------------
    #  ANALISI
    # ------------------------------------------------------------------

    def _avvia_analisi(self):
        if self._thread is not None and self._thread.isRunning():
            return

        if not self._dati_correnti:
            QMessageBox.warning(
                self._main, "Attenzione",
                "Seleziona una struttura prima di avviare l'analisi."
            )
            return

        if not self._main.ha_progetto():
            QMessageBox.warning(
                self._main, "Attenzione",
                "Crea o apri un progetto prima di avviare l'analisi."
            )
            return

        nodi = self._dati_correnti.get("nodi", {})
        aste = self._dati_correnti.get("aste", {})
        if not nodi or not aste:
            QMessageBox.warning(
                self._main, "Struttura incompleta",
                "La struttura deve avere almeno nodi e aste per l'analisi."
            )
            return

        # Leggi parametri
        n_div = self._leggi_n_divisioni()
        if n_div is None:
            return
        gravita = self._leggi_gravita()
        if gravita is None:
            return
        peso_proprio = self._ui.fem_struttura_pesoproprio_radioButton_si.isChecked()

        # ---- 1. Risolvi materiali e sezioni ----
        print(">> FEM Struttura: risoluzione materiali e sezioni...")

        mat_struttura = self._dati_correnti.get("materiali", {})
        sez_struttura = self._dati_correnti.get("sezioni", {})

        self._materiali_risolti = self._gen_mat.risolvi_materiali(mat_struttura)
        self._sezioni_risolte = self._gen_sez.risolvi_sezioni(
            sez_struttura, self._materiali_risolti
        )

        for sid, sd in self._sezioni_risolte.items():
            print(f"   Sezione {sid} '{sd['nome']}': "
                  f"A={sd['Area']:.6e} m², "
                  f"Iy={sd['Iy']:.6e} m⁴, "
                  f"Iz={sd['Iz']:.6e} m⁴")

        for mid, md in self._materiali_risolti.items():
            print(f"   Materiale {mid} '{md['nome']}': "
                  f"E={md['E']:.0f} MPa, "
                  f"G={md['G']:.0f} MPa, "
                  f"ρ={md['densita']:.0f} kg/m³")

        # ---- 2. Genera mesh ----
        print(f">> FEM Struttura: generazione mesh (N={n_div})...")
        gen_mesh = GeneratoreMeshStruttura(n_divisioni=n_div)
        self._mesh = gen_mesh.genera(self._dati_correnti)
        self._spazio.set_mesh(self._mesh)

        print(f"   Mesh: {self._mesh.n_nodi} nodi, "
              f"{len(self._mesh.elementi_beam)} beam, "
              f"{len(self._mesh.elementi_shell)} shell")

        # ---- 3. Lancia analisi in thread ----
        ui = self._ui
        ui.fem_struttura_progressBar.setValue(0)
        ui.fem_struttura_btn_analisi.setEnabled(False)

        self._thread = _AnalisiThread(
            mesh=self._mesh,
            materiali=self._materiali_risolti,
            sezioni=self._sezioni_risolte,
            dati_struttura=self._dati_correnti,
            gravita=gravita,
            peso_proprio=peso_proprio,
        )
        self._thread.avanzamento.connect(ui.fem_struttura_progressBar.setValue)
        self._thread.completato.connect(self._on_analisi_completata)
        self._thread.errore.connect(self._on_analisi_errore)
        self._thread.start()

        print(f">> FEM Struttura: analisi avviata (gravità={gravita}, "
              f"peso_proprio={peso_proprio})...")

    def _leggi_n_divisioni(self) -> int | None:
        try:
            val = int(self._ui.fem_struttura_definizione.text().strip())
            if val < 2:
                raise ValueError
            return val
        except (ValueError, AttributeError):
            QMessageBox.warning(
                self._main, "Parametro non valido",
                "Il numero di divisioni deve essere un intero >= 2."
            )
            return None

    def _leggi_gravita(self) -> float | None:
        try:
            val = float(self._ui.fem_struttura_gravita.text().strip())
            if val < 0:
                raise ValueError
            return val
        except (ValueError, AttributeError):
            QMessageBox.warning(
                self._main, "Parametro non valido",
                "Il valore di gravità deve essere un numero positivo."
            )
            return None

    def _on_analisi_completata(self, risultati: RisultatiFEMStruttura):
        self._risultati = risultati
        self._ui.fem_struttura_btn_analisi.setEnabled(True)
        self._ui.fem_struttura_progressBar.setValue(100)

        if risultati.successo:
            self._spazio.set_risultati(risultati)
            # Imposta modo indeformata (utente sceglie poi)
            self._spazio.set_modo("indeformata")
            self._ui.fem_struttura_btn_indeformata.setChecked(True)
            self._aggiorna_stato_tensioni("indeformata")

            print(f">> FEM Struttura: analisi completata con successo.")
            print(f"   Max spostamento: {risultati.max_spostamento:.6e} m")
            print(f"   Max N:  {risultati.max_sforzo('N'):.2f} kN")
            print(f"   Max Vy: {risultati.max_sforzo('Vy'):.2f} kN")
            print(f"   Max Vz: {risultati.max_sforzo('Vz'):.2f} kN")
            print(f"   Max My: {risultati.max_sforzo('My'):.2f} kNm")
            print(f"   Max Mz: {risultati.max_sforzo('Mz'):.2f} kNm")

            # Stampa reazioni
            if risultati.reazioni:
                print(f"   Reazioni vincolari:")
                for tag, r in risultati.reazioni.items():
                    nid_orig = self._mesh.mappa_nodi_originali.get(tag, tag)
                    print(f"     Nodo {nid_orig}: "
                          f"Fx={r[0]/1e3:.2f} kN, "
                          f"Fy={r[1]/1e3:.2f} kN, "
                          f"Fz={r[2]/1e3:.2f} kN, "
                          f"Mx={r[3]/1e3:.2f} kNm, "
                          f"My={r[4]/1e3:.2f} kNm, "
                          f"Mz={r[5]/1e3:.2f} kNm")
        else:
            QMessageBox.critical(
                self._main, "Analisi fallita",
                f"L'analisi non è riuscita:\n{risultati.messaggio}"
            )
            print(f"ERR  FEM Struttura: {risultati.messaggio}")

    def _on_analisi_errore(self, msg: str):
        self._ui.fem_struttura_btn_analisi.setEnabled(True)
        self._ui.fem_struttura_progressBar.setValue(0)
        QMessageBox.critical(
            self._main, "Errore analisi FEM",
            f"Si è verificato un errore:\n{msg}"
        )
        print(f"ERR  FEM Struttura: {msg}")

    # ------------------------------------------------------------------
    #  MODALITA' VISUALIZZAZIONE
    # ------------------------------------------------------------------

    def _on_modo_cambiato(self, modo: str):
        self._spazio.set_modo(modo)
        self._aggiorna_stato_tensioni(modo)

    def _aggiorna_stato_tensioni(self, modo: str):
        """Il pulsante tensioni e' abilitato solo in modo deformata.
        Fuori dalla deformata viene forzatamente disattivato."""
        btn = self._ui.fem_struttura_btn_tensioni
        if modo == "deformata":
            btn.setEnabled(True)
        else:
            if btn.isChecked():
                btn.setChecked(False)
                self._spazio.set_mostra_tensioni(False)
            btn.setEnabled(False)

    def _on_tensioni_toggle(self):
        attivo = self._ui.fem_struttura_btn_tensioni.isChecked()
        self._spazio.set_mostra_tensioni(attivo)

    def _on_scala_cambiata(self, *_args):
        txt = self._ui.fem_struttura_scala.text().strip()
        try:
            scala = float(txt)
        except (ValueError, AttributeError, TypeError):
            # Testo non valido: non resetto il campo, ma ignoro la modifica
            # (evito di sovrascrivere mentre l'utente sta digitando)
            return
        if scala < 0.0:
            scala = 0.0
        self._spazio.set_scala_deformazione(scala)

    def _centra_vista(self):
        self._spazio.centra_vista()

    # ------------------------------------------------------------------
    #  RICARICA DA PROGETTO
    # ------------------------------------------------------------------

    def aggiorna_combobox(self):
        self._popola_combobox()

    def ricarica_da_progetto(self):
        self._cat_corrente = None
        self._nome_corrente = None
        self._dati_correnti = None
        self._mesh = None
        self._risultati = None
        self._materiali_risolti = None
        self._sezioni_risolte = None

        self._spazio.set_dati(None)
        self._spazio.set_mesh(None)
        self._spazio.set_risultati(None)
        self._spazio.set_modo("indeformata")
        self._aggiorna_stato_tensioni("indeformata")

        self._popola_combobox()
        self._reset_ui()
        print(">> Modulo FEM Struttura: progetto ricaricato.")
