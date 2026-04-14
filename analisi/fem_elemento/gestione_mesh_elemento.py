"""
gestione_mesh_elemento.py – Controller del modulo FEM Elemento.

Responsabilita':
  - Popola la combobox con gli elementi strutturali del progetto
  - Gestisce la generazione della mesh (thread separato + progress bar)
  - Aggiorna lo spazio 3D (disegno_fem_elemento)
  - Gestisce visibilita' carpenteria / barre / staffe
  - Gestisce i pulsanti vista
  - Scrive i file .inp nella cartella mesh_create
  - Lancia le analisi CalculiX (lineare o nonlineare, una alla volta)
  - Gestisce la visualizzazione risultati (deformata, tensioni, animazione)
"""
from __future__ import annotations

import os

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QVBoxLayout, QMessageBox

from analisi.raccolta_dati_elemento_fem import RaccoltaDatiElementoFEM
from .disegno_fem_elemento import FEMSpazio3D
from .generatore_mesh import GeneratoreMesh, RisultatoMesh
from .scrittore_inp import ScrittoreINP
from .esecutore_analisi import AnalisiThread, ParametriAnalisi
from .lettore_frd import RisultatiFRD


# ==============================================================================
# THREAD DI GENERAZIONE MESH
# ==============================================================================

class _MeshThread(QtCore.QThread):
    """Esegue la generazione della mesh in background."""

    avanzamento = QtCore.pyqtSignal(int)
    completato  = QtCore.pyqtSignal(object)   # RisultatoMesh
    errore      = QtCore.pyqtSignal(str)

    def __init__(self, generatore: GeneratoreMesh, dati_fem: dict, parent=None):
        super().__init__(parent)
        self._gen = generatore
        self._dati = dati_fem

    def run(self):
        try:
            risultato = self._gen.genera(
                self._dati,
                progress_cb=lambda p: self.avanzamento.emit(p)
            )
            self.completato.emit(risultato)
        except Exception as e:
            self.errore.emit(str(e))


# ==============================================================================
# CONTROLLER
# ==============================================================================

class GestioneMeshElemento:
    """Controller del modulo FEM Elemento."""

    def __init__(self, ui, main_window) -> None:
        self._ui   = ui
        self._main = main_window
        self._dati = RaccoltaDatiElementoFEM(main_window)

        # Stato
        self._el_id_corrente: int | None = None
        self._risultato_mesh: RisultatoMesh | None = None
        self._thread: _MeshThread | None = None
        self._thread_analisi: AnalisiThread | None = None

        # Risultati analisi corrente
        self._risultati_corrente: RisultatiFRD | None = None
        self._tipo_analisi_corrente: str = ""  # "lineare" o "nonlineare"

        # Widget 3D
        self._spazio = FEMSpazio3D()
        self._setup_widget_3d()
        self._setup_connessioni()
        self._setup_connessioni_analisi()
        self._reset_ui()

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------

    def _setup_widget_3d(self):
        container = self._ui.fem_elemento_widget
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

        # Combobox elemento
        ui.fem_elemento_combobox.currentIndexChanged.connect(
            self._on_elemento_cambiato
        )

        # Pulsante genera mesh
        ui.fem_elemento_btn_mesh.clicked.connect(self._avvia_mesh)

        # Pulsanti visibilita'
        ui.fem_elemento_btn_carpenteria.setCheckable(True)
        ui.fem_elemento_btn_barre.setCheckable(True)
        ui.fem_elemento_btn_staffe.setCheckable(True)
        ui.fem_elemento_btn_carpenteria.setChecked(True)
        ui.fem_elemento_btn_barre.setChecked(True)
        ui.fem_elemento_btn_staffe.setChecked(True)

        ui.fem_elemento_btn_carpenteria.clicked.connect(self._aggiorna_visibilita)
        ui.fem_elemento_btn_barre.clicked.connect(self._aggiorna_visibilita)
        ui.fem_elemento_btn_staffe.clicked.connect(self._aggiorna_visibilita)

        # Pulsanti visibilita' nodi speciali
        ui.fem_elemento_btn_vincoli.setCheckable(True)
        ui.fem_elemento_btn_carichi.setCheckable(True)
        ui.fem_elemento_btn_crack.setCheckable(True)
        ui.fem_elemento_btn_vincoli.setChecked(True)
        ui.fem_elemento_btn_carichi.setChecked(True)
        ui.fem_elemento_btn_crack.setChecked(True)

        ui.fem_elemento_btn_vincoli.clicked.connect(self._aggiorna_visibilita_nodi)
        ui.fem_elemento_btn_carichi.clicked.connect(self._aggiorna_visibilita_nodi)
        ui.fem_elemento_btn_crack.clicked.connect(self._aggiorna_visibilita_nodi)

        # Pulsanti vista (checkable, esclusivi)
        from PyQt5.QtWidgets import QButtonGroup
        self._vista_group = QButtonGroup(ui.fem_elemento_btn_vista_3d.parent())
        self._vista_group.setExclusive(True)
        for btn in (ui.fem_elemento_btn_vista_3d, ui.fem_elemento_btn_vista_x,
                     ui.fem_elemento_btn_vista_y, ui.fem_elemento_btn_vista_z):
            btn.setCheckable(True)
            self._vista_group.addButton(btn)
        ui.fem_elemento_btn_vista_3d.setChecked(True)
        ui.fem_elemento_btn_vista_3d.clicked.connect(lambda: self._spazio.imposta_vista("3d"))
        ui.fem_elemento_btn_vista_x.clicked.connect(lambda: self._spazio.imposta_vista("x"))
        ui.fem_elemento_btn_vista_y.clicked.connect(lambda: self._spazio.imposta_vista("y"))
        ui.fem_elemento_btn_vista_z.clicked.connect(lambda: self._spazio.imposta_vista("z"))

    def _setup_connessioni_analisi(self):
        """Collega i widget dell'analisi e della visualizzazione risultati."""
        ui = self._ui

        # Pulsante lancia analisi
        ui.fem_elemento_btn_analisi.clicked.connect(self._avvia_analisi)

        # Scala deformazione
        ui.fem_elemento_scala.textChanged.connect(self._on_scala_cambiata)

        # Controlli animazione (sempre visibili, usati per lin e nlin)
        ui.fem_elemento_btn_play.setCheckable(True)
        ui.fem_elemento_btn_play.setChecked(False)
        ui.fem_elemento_btn_play.clicked.connect(self._on_btn_play)
        ui.fem_elemento_btn_replay.clicked.connect(self._on_btn_replay)
        ui.fem_elemento_velocita.textChanged.connect(self._on_velocita_cambiata)

        # Segnale animazione dallo spazio 3D
        self._spazio.animazione_step_changed.connect(self._on_anim_step_changed)

    def _reset_ui(self):
        self._ui.fem_elemento_progressBar.setValue(0)
        if not self._ui.fem_elemento_definizione.text().strip():
            self._ui.fem_elemento_definizione.setText("5")
        if not self._ui.fem_elemento_gravita.text().strip():
            self._ui.fem_elemento_gravita.setText("9.81")
        if not self._ui.fem_elemento_scala.text().strip():
            self._ui.fem_elemento_scala.setText("1.0")
        if not self._ui.fem_elemento_velocita.text().strip():
            self._ui.fem_elemento_velocita.setText("5")

        # Radio buttons default
        self._ui.fem_elemento_pesoproprio_radioButton_si.setChecked(True)
        self._ui.fem_elemento_collisioni_radioButton_no.setChecked(True)
        self._ui.fem_elemento_lineare_radioButton.setChecked(True)

    # ------------------------------------------------------------------
    # COMBOBOX
    # ------------------------------------------------------------------

    def _popola_combobox(self):
        cb = self._ui.fem_elemento_combobox
        cb.blockSignals(True)
        cb.clear()

        elementi = self._dati.lista_elementi()
        for tipo, nome, el_id in elementi:
            cb.addItem(f"[{tipo}] {nome}", el_id)

        if cb.count() > 0:
            if self._el_id_corrente is not None:
                trovato = False
                for i in range(cb.count()):
                    if cb.itemData(i) == self._el_id_corrente:
                        cb.setCurrentIndex(i)
                        trovato = True
                        break
                if not trovato:
                    cb.setCurrentIndex(0)
                    cb.blockSignals(False)
                    self._on_elemento_cambiato(0)
                    return
            else:
                cb.setCurrentIndex(0)
                cb.blockSignals(False)
                self._on_elemento_cambiato(0)
                return

        cb.blockSignals(False)

    def _on_elemento_cambiato(self, index: int):
        cb = self._ui.fem_elemento_combobox
        if index < 0 or index >= cb.count():
            return

        el_id = cb.itemData(index)
        if el_id is None:
            return

        self._el_id_corrente = el_id
        self._risultato_mesh = None
        self._risultati_corrente = None
        self._tipo_analisi_corrente = ""
        self._ui.fem_elemento_progressBar.setValue(0)

        # Ferma animazione
        self._spazio.anim_stop()
        self._ui.fem_elemento_btn_play.setChecked(False)

        # Carica oggetti nel viewer 3D
        el = self._dati.get_elemento_by_id(el_id)
        if el:
            self._spazio.set_oggetti(el.oggetti)
            self._spazio.set_mesh(None)
            self._spazio.set_risultati(None, None)
            self._spazio.set_modo_vista("mesh")
            self._spazio.centra_vista()

    # ------------------------------------------------------------------
    # GENERAZIONE MESH
    # ------------------------------------------------------------------

    def _avvia_mesh(self):
        if self._thread is not None and self._thread.isRunning():
            return

        if self._el_id_corrente is None:
            QMessageBox.warning(
                self._main, "Attenzione",
                "Seleziona un elemento prima di generare la mesh."
            )
            return

        if not self._main.ha_progetto():
            QMessageBox.warning(
                self._main, "Attenzione",
                "Crea o apri un progetto prima di generare la mesh."
            )
            return

        dati_fem = self._dati.dati_per_fem(self._el_id_corrente)
        if dati_fem is None:
            QMessageBox.warning(
                self._main, "Errore",
                "Impossibile caricare i dati dell'elemento."
            )
            return

        carpenteria = dati_fem.get("carpenteria", [])
        barre = dati_fem.get("barre", [])
        staffe = dati_fem.get("staffe", [])
        if not carpenteria and not barre and not staffe:
            QMessageBox.warning(
                self._main, "Elemento vuoto",
                "L'elemento selezionato non contiene oggetti geometrici."
            )
            return

        densita = self._leggi_densita()
        if densita is None:
            return

        # UI feedback
        ui = self._ui
        ui.fem_elemento_progressBar.setValue(0)
        ui.fem_elemento_btn_mesh.setEnabled(False)

        gen = GeneratoreMesh(densita=densita)

        self._thread = _MeshThread(gen, dati_fem)
        self._thread.avanzamento.connect(ui.fem_elemento_progressBar.setValue)
        self._thread.completato.connect(self._on_mesh_completata)
        self._thread.errore.connect(self._on_mesh_errore)
        self._thread.start()

        print(f">> FEM: generazione mesh avviata (densita={densita})...")

    def _leggi_densita(self) -> int | None:
        try:
            val = int(self._ui.fem_elemento_definizione.text().strip())
            if val < 1:
                raise ValueError
            return val
        except (ValueError, AttributeError):
            QMessageBox.warning(
                self._main, "Parametro non valido",
                "La densita' della mesh deve essere un intero positivo."
            )
            return None

    def _on_mesh_completata(self, risultato: RisultatoMesh):
        self._risultato_mesh = risultato
        self._ui.fem_elemento_btn_mesh.setEnabled(True)
        self._ui.fem_elemento_progressBar.setValue(100)

        # Aggiorna visualizzazione
        self._spazio.set_mesh(risultato)
        self._spazio.set_modo_vista("mesh")
        self._aggiorna_visibilita()
        self._spazio.update()

        print(f">> FEM: mesh completata. "
              f"Nodi: {risultato.n_nodi}, "
              f"Elementi hex: {len(risultato.elementi_hex)}, "
              f"Elementi truss: {len(risultato.elementi_beam)}, "
              f"TIE carp: {len(risultato.tie_constraints)}, "
              f"TIE arm: {len(risultato.tie_armatura)}, "
              f"Nodi vincolati: {len(risultato.nodi_vincolati)}, "
              f"Nodi caricati: {len(risultato.nodi_caricati)}")

        # Scrivi file .inp
        self._scrivi_inp(risultato)

    def _on_mesh_errore(self, msg: str):
        self._ui.fem_elemento_btn_mesh.setEnabled(True)
        self._ui.fem_elemento_progressBar.setValue(0)
        QMessageBox.critical(
            self._main, "Errore generazione mesh",
            f"Si e' verificato un errore:\n{msg}"
        )
        print(f"ERR  FEM mesh: {msg}")

    # ------------------------------------------------------------------
    # SCRITTURA .INP
    # ------------------------------------------------------------------

    def _scrivi_inp(self, risultato: RisultatoMesh):
        """Scrive il file .inp nella cartella mesh_create."""
        if self._el_id_corrente is None:
            return

        dati_fem = self._dati.dati_per_fem(self._el_id_corrente)
        if dati_fem is None:
            return

        el = dati_fem["elemento"]
        materiali = dati_fem.get("materiali", {})
        nome_safe = el.nome.replace(" ", "_").replace(".", "_")

        fem_dir = os.path.dirname(os.path.abspath(__file__))
        mesh_dir = os.path.join(fem_dir, "mesh_create")
        percorso = os.path.join(mesh_dir, f"{nome_safe}.inp")

        scrittore = ScrittoreINP(risultato, materiali, el.nome)
        if scrittore.scrivi(percorso):
            print(f">> FEM: file .inp scritto -> {percorso}")
        else:
            print(f"ERR  FEM: scrittura .inp fallita.")

    # ------------------------------------------------------------------
    # ANALISI
    # ------------------------------------------------------------------

    def _avvia_analisi(self):
        """Lancia l'analisi CalculiX (lineare O nonlineare, in base al radio)."""
        if self._thread_analisi is not None and self._thread_analisi.isRunning():
            return

        if self._risultato_mesh is None:
            QMessageBox.warning(
                self._main, "Attenzione",
                "Genera prima la mesh dell'elemento."
            )
            return

        # Raccogli parametri
        params = self._leggi_parametri_analisi()
        if params is None:
            return

        # Materiali
        dati_fem = self._dati.dati_per_fem(self._el_id_corrente)
        if dati_fem is None:
            return
        materiali = dati_fem.get("materiali", {})

        # Determina tipo analisi selezionato
        if self._ui.fem_elemento_nonlineare_radioButton.isChecked():
            params.analisi_lineare = False
            params.analisi_nonlineare = True
            self._tipo_analisi_corrente = "nonlineare"
        else:
            params.analisi_lineare = True
            params.analisi_nonlineare = False
            self._tipo_analisi_corrente = "lineare"

        # UI feedback
        ui = self._ui
        ui.fem_elemento_progressBar.setValue(0)
        ui.fem_elemento_btn_analisi.setEnabled(False)

        # Reset risultati precedenti
        self._risultati_corrente = None
        self._spazio.anim_stop()
        self._ui.fem_elemento_btn_play.setChecked(False)

        # Lancia thread
        self._thread_analisi = AnalisiThread(
            mesh=self._risultato_mesh,
            materiali=materiali,
            parametri=params,
        )
        self._thread_analisi.avanzamento.connect(ui.fem_elemento_progressBar.setValue)
        self._thread_analisi.log_message.connect(self._on_analisi_log)
        self._thread_analisi.completato.connect(self._on_analisi_completata)
        self._thread_analisi.errore.connect(self._on_analisi_errore)
        self._thread_analisi.start()

        print(f">> FEM: analisi {self._tipo_analisi_corrente} avviata "
              f"(grav={params.gravita}, peso={params.peso_proprio}, "
              f"coll={params.collisioni})...")

    def _leggi_parametri_analisi(self) -> ParametriAnalisi | None:
        """Legge i parametri analisi dalla UI."""
        ui = self._ui
        params = ParametriAnalisi()

        # Gravita'
        try:
            params.gravita = float(ui.fem_elemento_gravita.text().strip())
            if params.gravita < 0:
                raise ValueError
        except (ValueError, AttributeError):
            QMessageBox.warning(
                self._main, "Parametro non valido",
                "Il valore di gravita' deve essere un numero positivo."
            )
            return None

        # Peso proprio
        params.peso_proprio = ui.fem_elemento_pesoproprio_radioButton_si.isChecked()

        # Collisioni
        params.collisioni = ui.fem_elemento_collisioni_radioButton_si.isChecked()

        # Nome elemento
        dati_fem = self._dati.dati_per_fem(self._el_id_corrente)
        if dati_fem:
            params.nome_elemento = dati_fem["elemento"].nome

        return params

    def _on_analisi_log(self, msg: str):
        print(msg)

    def _on_analisi_completata(self, risultati: dict):
        """Callback quando l'analisi termina con successo."""
        self._ui.fem_elemento_btn_analisi.setEnabled(True)
        self._ui.fem_elemento_progressBar.setValue(100)

        # Recupera risultato per il tipo di analisi eseguita
        frd = risultati.get(self._tipo_analisi_corrente)
        self._risultati_corrente = frd

        # Calcola resistenze ultime per colorazione collasso
        sigma_ult = self._calcola_sigma_ult()

        # Passa risultati alla visualizzazione (sigma_ult incluso)
        if self._tipo_analisi_corrente == "nonlineare":
            self._spazio.set_risultati(None, frd, sigma_ult=sigma_ult)
            self._spazio.set_modo_vista("nonlineare")
        else:
            self._spazio.set_risultati(frd, None, sigma_ult=sigma_ult)
            self._spazio.set_modo_vista("lineare")

        # Scala deformazione
        self._on_scala_cambiata()

        n_steps = frd.n_steps if frd else 0
        print(f">> FEM: analisi {self._tipo_analisi_corrente} completata. "
              f"Steps: {n_steps}.")

        if frd and n_steps > 0:
            print(f"   Max spostamento: {frd.max_spostamento():.6e} m")
            print(f"   Max stress VM:   {frd.max_stress_vm() / 1e6:.2f} MPa")

    def _on_analisi_errore(self, msg: str):
        self._ui.fem_elemento_btn_analisi.setEnabled(True)
        self._ui.fem_elemento_progressBar.setValue(0)
        QMessageBox.critical(
            self._main, "Errore analisi FEM",
            f"Si e' verificato un errore:\n{msg}"
        )
        print(f"ERR  FEM analisi: {msg}")

    # ------------------------------------------------------------------
    # SIGMA ULTIME (per colorazione collasso in nero)
    # ------------------------------------------------------------------

    def _calcola_sigma_ult(self) -> dict:
        """
        Calcola la resistenza ultima per ogni materiale.
        Calcestruzzo: fcd = alpha * fck / gamma_c
        Acciaio barre: fyd = fyk / gamma_s
        Ritorna {nome_materiale: sigma_ult_MPa}.
        """
        sigma_ult = {}
        dati_fem = self._dati.dati_per_fem(self._el_id_corrente)
        if not dati_fem:
            return sigma_ult

        materiali = dati_fem.get("materiali", {})

        for nome_mat, dati in materiali.items():
            fck = dati.get("fck")
            fyk = dati.get("fyk")

            if fck is not None:
                alpha = float(dati.get("alpha", 0.85))
                gamma = float(dati.get("gamma", 1.5))
                fcd = float(fck) * alpha / gamma  # MPa
                sigma_ult[nome_mat] = fcd * 1e6    # -> Pa (coerente con output CalculiX)
            elif fyk is not None:
                gamma = float(dati.get("gamma", 1.15))
                fyd = float(fyk) / gamma           # MPa
                sigma_ult[nome_mat] = fyd * 1e6    # -> Pa

        return sigma_ult

    # ------------------------------------------------------------------
    # SCALA DEFORMAZIONE
    # ------------------------------------------------------------------

    def _on_scala_cambiata(self, *_args):
        try:
            scala = float(self._ui.fem_elemento_scala.text().strip())
        except (ValueError, AttributeError):
            scala = 1.0
        self._spazio.set_scala_deformazione(scala)

    # ------------------------------------------------------------------
    # CONTROLLI ANIMAZIONE (usati sia per lineare che nonlineare)
    # ------------------------------------------------------------------

    def _on_btn_play(self):
        """Play/Pause dell'animazione."""
        self._spazio.anim_play_pause()
        self._ui.fem_elemento_btn_play.setChecked(self._spazio.anim_is_playing)

    def _on_btn_replay(self):
        """Riavvia l'animazione dall'inizio."""
        self._spazio.anim_replay()
        self._ui.fem_elemento_btn_play.setChecked(True)

    def _on_velocita_cambiata(self, *_args):
        """Aggiorna la durata dell'animazione (in secondi)."""
        try:
            sec = float(self._ui.fem_elemento_velocita.text().strip())
        except (ValueError, AttributeError):
            sec = 5.0
        self._spazio.anim_set_durata(sec)

    def _on_anim_step_changed(self, step: int, total: int):
        """Callback dal widget 3D quando cambia lo step dell'animazione."""
        if step >= total - 1:
            self._ui.fem_elemento_btn_play.setChecked(False)

    # ------------------------------------------------------------------
    # VISIBILITA'
    # ------------------------------------------------------------------

    def _aggiorna_visibilita(self):
        carp = self._ui.fem_elemento_btn_carpenteria.isChecked()
        bar = self._ui.fem_elemento_btn_barre.isChecked()
        stf = self._ui.fem_elemento_btn_staffe.isChecked()
        self._spazio.set_visibilita(carp, bar, stf)

    def _aggiorna_visibilita_nodi(self):
        vinc = self._ui.fem_elemento_btn_vincoli.isChecked()
        caric = self._ui.fem_elemento_btn_carichi.isChecked()
        crack = self._ui.fem_elemento_btn_crack.isChecked()
        self._spazio.set_visibilita_nodi(vinc, caric, crack)

    # ------------------------------------------------------------------
    # RICARICA DA PROGETTO
    # ------------------------------------------------------------------

    def aggiorna_combobox(self):
        self._popola_combobox()

    def ricarica_da_progetto(self):
        self._el_id_corrente = None
        self._risultato_mesh = None
        self._risultati_corrente = None
        self._tipo_analisi_corrente = ""

        # Ferma animazione
        self._spazio.anim_stop()
        self._ui.fem_elemento_btn_play.setChecked(False)

        self._spazio.set_oggetti([])
        self._spazio.set_mesh(None)
        self._spazio.set_risultati(None, None)
        self._spazio.set_modo_vista("mesh")

        self._popola_combobox()
        self._reset_ui()
        print(">> Modulo FEM Elemento: progetto ricaricato.")
