"""
gestione_pressoflessione.py
============================
Controller del modulo Pressoflessione.

Responsabilità:
  - Popola la combobox con tutte le sezioni del progetto.
  - Aggiorna l'anteprima sezione al cambio di selezione.
  - Gestisce i parametri di analisi (grid_step, rotazione, N_Ed, M_Ed, SLU/SLE).
  - Lancia l'analisi su un thread separato con aggiornamento della progress bar.
  - Mostra il risultato nel widget 2D e nel campo di verifica.
  - Salva/carica le impostazioni nel progetto aperto.
  - Risponde a ricarica_da_progetto() quando cambia il file di progetto.
"""
from __future__ import annotations

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QVBoxLayout, QMessageBox, QButtonGroup

from analisi.raccolta_dati import RaccoltaDati
from .calcolo import (
    SezioneDiscretizzata,
    CalcoloPressoflessione,
)
from .disegno_pressoflessione_sezione import PressoflessioneSezioneWidget
from .disegno_pressoflessione           import PressoflessioneWidget


# ==============================================================================
# STILE BASE PER LABEL VERIFICA
# ==============================================================================
STILE_BASE_VERIFICA = """
    background-color: rgb(40, 40, 40);
    border: 1px solid rgb(120, 120, 120);
    border-left: 3px solid {colore_bordo};
    border-radius: 6px;
    color: {colore_testo};
    {font_weight}
"""

# ==============================================================================
# THREAD DI CALCOLO
# ==============================================================================

class _AnalisiThread(QtCore.QThread):
    """Esegue il calcolo in background e notifica il progresso."""

    avanzamento = QtCore.pyqtSignal(int)      # percentuale 0-100
    completato  = QtCore.pyqtSignal(dict)     # risultati
    errore      = QtCore.pyqtSignal(str)      # messaggio di errore

    def __init__(self,
                 calcolatore: CalcoloPressoflessione,
                 modo:        str,
                 N_Ed_kN:     float,
                 M_Ed_kNm:    float,
                 theta_deg:   float,
                 parent=None) -> None:
        super().__init__(parent)
        self._calc     = calcolatore
        self._modo     = modo
        self._N_Ed     = N_Ed_kN
        self._M_Ed     = M_Ed_kNm
        self._theta    = theta_deg

    def run(self) -> None:
        try:
            if self._modo == 'SLU':
                res = self._calc.analisi_slu(
                    self._N_Ed, self._M_Ed, self._theta,
                    progress_cb=lambda p: self.avanzamento.emit(p)
                )
            else:
                res = self._calc.analisi_sle(
                    self._N_Ed, self._M_Ed, self._theta,
                    progress_cb=lambda p: self.avanzamento.emit(p)
                )
            self.completato.emit(res)
        except Exception as e:
            self.errore.emit(str(e))


# ==============================================================================
# CONTROLLER
# ==============================================================================

class GestionePressoflessione:
    """
    Controller del modulo Pressoflessione.

    Connette gli elementi UI ai moduli di calcolo e di visualizzazione.
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, ui, main_window) -> None:
        self._ui   = ui
        self._main = main_window
        self._dati = RaccoltaDati(main_window)

        # Stato corrente
        self._sezione_corrente: str | None  = None
        self._risultati:        dict | None = None
        self._thread:           _AnalisiThread | None = None

        # Widgets custom inseriti nei placeholder dell'UI
        self._w_sezione = PressoflessioneSezioneWidget()
        self._w_analisi = PressoflessioneWidget()

        self._setup_widget_sezione()
        self._setup_widget_analisi()
        self._setup_connessioni()

        # Stato iniziale dell'UI
        self._reset_ui()

    # ------------------------------------------------------------------
    # SETUP – INSERIMENTO WIDGET CUSTOM NEI PLACEHOLDER
    # ------------------------------------------------------------------

    def _setup_widget_sezione(self) -> None:
        """Inserisce il widget OpenGL anteprima sezione nel placeholder."""
        container = self._ui.pressoflessione_widget_sezione
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(0)
        lay.addWidget(self._w_sezione)

    def _setup_widget_analisi(self) -> None:
        """Inserisce il widget OpenGL risultati nel placeholder."""
        container = self._ui.pressoflessione_widget
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(0)
        lay.addWidget(self._w_analisi)

    # ------------------------------------------------------------------
    # SETUP – CONNESSIONI UI
    # ------------------------------------------------------------------

    def _setup_connessioni(self) -> None:
        ui = self._ui

        # Combobox sezione
        ui.pressoflessione_combobox_sezioni.currentTextChanged.connect(
            self._on_sezione_cambiata
        )

        # Pulsante avvia analisi
        ui.pressoflessione_btn_analisi.clicked.connect(self._avvia_analisi)

        # Pulsanti vista – resi checkable ed esclusivi
        ui.pressoflessione_vista_oggetti.setCheckable(True)
        ui.pressoflessione_vista_gradiente.setCheckable(True)
        ui.pressoflessione_vista_oggetti.setChecked(True)   # default: normale

        self._btn_group_vista = QButtonGroup(ui.pressoflessione_vista_oggetti.parent())
        self._btn_group_vista.setExclusive(True)
        self._btn_group_vista.addButton(ui.pressoflessione_vista_oggetti)
        self._btn_group_vista.addButton(ui.pressoflessione_vista_gradiente)

        ui.pressoflessione_vista_oggetti.clicked.connect(
            lambda: self._imposta_vista('normale')
        )
        ui.pressoflessione_vista_gradiente.clicked.connect(
            lambda: self._imposta_vista('gradiente')
        )

        # Pulsante centra vista analisi
        ui.pressoflessione_centra.clicked.connect(self._w_analisi.reset_view)

        # Pulsante centra vista anteprima sezione
        ui.pressoflessione_sezione_centra.clicked.connect(
            self._w_sezione.reset_view
        )

    # ------------------------------------------------------------------
    # RESET UI
    # ------------------------------------------------------------------

    def _reset_ui(self) -> None:
        """Porta l'UI allo stato iniziale (nessuna analisi) con valori preset."""
        ui = self._ui
        ui.pressoflessione_progressBar.setValue(0)
        ui.pressoflessione_risultato_verifica.setText("Avviare analisi")
        ui.pressoflessione_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(120, 120, 120)",
                colore_testo="rgb(160, 160, 160)",
                font_weight="font-weight: normal;"
            )
        )
        ui.pressoflessione_label_sezione.setText("Anteprima sezione: —")
        self._w_analisi.set_results(None)

        # Preset valori di default (solo se i campi sono vuoti)
        if not ui.pressoflessione_definizione.text().strip():
            ui.pressoflessione_definizione.setText("10")
        if not ui.pressoflessione_rotazione.text().strip():
            ui.pressoflessione_rotazione.setText("0")
        if not ui.pressoflessione_sforzo_normale.text().strip():
            ui.pressoflessione_sforzo_normale.setText("0")
        if not ui.pressoflessione_sollecitazione_flettente.text().strip():
            ui.pressoflessione_sollecitazione_flettente.setText("100")
        # Default: SLU selezionato
        if (not ui.pressoflessione_radioButton_slu.isChecked() and
                not ui.pressoflessione_radioButton_sle.isChecked()):
            ui.pressoflessione_radioButton_slu.setChecked(True)

    # ------------------------------------------------------------------
    # POPOLA COMBOBOX
    # ------------------------------------------------------------------

    def _popola_combobox(self) -> None:
        cb = self._ui.pressoflessione_combobox_sezioni
        cb.blockSignals(True)
        cb.clear()
        nomi = self._dati.lista_sezioni()
        for nome in nomi:
            cb.addItem(nome)
        cb.blockSignals(False)

        if not nomi:
            # Nessuna sezione disponibile (database non trovato e progetto vuoto)
            self._ui.pressoflessione_label_sezione.setText(
                "Nessuna sezione disponibile"
            )
            return

        # Ripristina la selezione precedente se ancora presente
        if self._sezione_corrente:
            idx = cb.findText(self._sezione_corrente)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        elif cb.count() > 0:
            cb.setCurrentIndex(0)
            self._on_sezione_cambiata(cb.currentText())

    # ------------------------------------------------------------------
    # SELEZIONE SEZIONE
    # ------------------------------------------------------------------

    def _on_sezione_cambiata(self, nome: str) -> None:
        if not nome:
            return

        # Aggiorna label e anteprima sempre
        self._ui.pressoflessione_label_sezione.setText(
            f"Anteprima sezione: '{nome}'"
        )
        dati = self._dati.dati_sezione(nome)
        self._w_sezione.set_section_data(dati)

        # Resetta risultati SOLO se la sezione è davvero cambiata
        if nome != self._sezione_corrente:
            self._sezione_corrente = nome
            self._risultati        = None
            self._ui.pressoflessione_progressBar.setValue(0)
            self._ui.pressoflessione_risultato_verifica.setText("Avviare analisi")
            self._ui.pressoflessione_risultato_verifica.setStyleSheet(
                STILE_BASE_VERIFICA.format(
                    colore_bordo="rgb(120, 120, 120)",
                    colore_testo="rgb(160, 160, 160)",
                    font_weight="font-weight: normal;"
                )
            )
            self._w_analisi.set_results(None)

    # ------------------------------------------------------------------
    # LETTURA PARAMETRI
    # ------------------------------------------------------------------

    def _leggi_float(self, widget, default: float) -> float:
        try:
            return float(widget.text().replace(',', '.').strip())
        except (ValueError, AttributeError):
            return default

    def _leggi_parametri(self) -> dict | None:
        """Legge e valida i parametri dall'UI. Ritorna None se non validi."""
        ui = self._ui

        nome = ui.pressoflessione_combobox_sezioni.currentText()
        if not nome:
            QMessageBox.warning(
                self._main, "Parametri mancanti",
                "Seleziona una sezione prima di avviare l'analisi."
            )
            return None

        grid_step = self._leggi_float(ui.pressoflessione_definizione, 10.0)
        if grid_step <= 0:
            QMessageBox.warning(
                self._main, "Parametri non validi",
                "Il passo di discretizzazione deve essere un valore positivo."
            )
            return None

        theta   = self._leggi_float(ui.pressoflessione_rotazione, 0.0)
        N_Ed    = self._leggi_float(ui.pressoflessione_sforzo_normale, 0.0)
        M_Ed    = self._leggi_float(ui.pressoflessione_sollecitazione_flettente, 0.0)
        modo    = 'SLU' if ui.pressoflessione_radioButton_slu.isChecked() else 'SLE'

        return {
            'nome'      : nome,
            'grid_step' : grid_step,
            'theta'     : theta,
            'N_Ed'      : N_Ed,
            'M_Ed'      : M_Ed,
            'modo'      : modo,
        }

    # ------------------------------------------------------------------
    # AVVIO ANALISI
    # ------------------------------------------------------------------

    def _avvia_analisi(self) -> None:
        """Legge i parametri, costruisce la sezione e lancia il thread."""
        if self._thread is not None and self._thread.isRunning():
            return   # analisi già in corso

        params = self._leggi_parametri()
        if params is None:
            return

        dati_analisi = self._dati.dati_per_analisi(params['nome'])
        if dati_analisi is None:
            QMessageBox.warning(
                self._main, "Sezione non trovata",
                f"Impossibile caricare i dati della sezione «{params['nome']}»."
            )
            return

        # Verifica che ci siano elementi carpenteria o barre
        sez_dati  = dati_analisi['sezione']
        elementi  = sez_dati.get('elementi', {})
        n_carp    = len(elementi.get('carpenteria', []))
        n_barre   = len(elementi.get('barre', []))
        if n_carp + n_barre == 0:
            QMessageBox.warning(
                self._main, "Sezione vuota",
                "La sezione selezionata non contiene elementi geometrici."
            )
            return

        # Costruisce il resolver per i materiali
        mat_db = dati_analisi['materiali']
        def risolvi_mat(nome: str) -> dict | None:
            return mat_db.get(nome)

        # Discretizza la sezione
        grid_step = params['grid_step']
        try:
            sezione_disc = SezioneDiscretizzata(
                dati_sezione=sez_dati,
                risolvi_mat=risolvi_mat,
                grid_step=grid_step,
            )
        except Exception as e:
            QMessageBox.critical(
                self._main, "Errore discretizzazione",
                f"Errore durante la discretizzazione della sezione:\n{e}"
            )
            return

        if not sezione_disc.fibre:
            QMessageBox.warning(
                self._main, "Discretizzazione vuota",
                "Nessuna fibra generata. Controlla il passo di discretizzazione "
                "rispetto alle dimensioni della sezione."
            )
            return

        calcolatore = CalcoloPressoflessione(sezione_disc)

        # Aggiorna la progress bar e blocca il pulsante
        ui = self._ui
        ui.pressoflessione_progressBar.setValue(0)
        ui.pressoflessione_btn_analisi.setEnabled(False)
        ui.pressoflessione_risultato_verifica.setText("Calcolo in corso…")
        ui.pressoflessione_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(120, 120, 120)",
                colore_testo="rgb(160, 160, 160)",
                font_weight="font-weight: normal;"
            )
        )

        # Lancia il thread
        self._thread = _AnalisiThread(
            calcolatore  = calcolatore,
            modo         = params['modo'],
            N_Ed_kN      = params['N_Ed'],
            M_Ed_kNm     = params['M_Ed'],
            theta_deg    = params['theta'],
        )
        self._thread.avanzamento.connect(ui.pressoflessione_progressBar.setValue)
        self._thread.completato.connect(self._on_analisi_completata)
        self._thread.errore.connect(self._on_analisi_errore)
        self._thread.start()

        # Salva le impostazioni nel progetto
        self._salva_impostazioni(params)

    # ------------------------------------------------------------------
    # GESTIONE RISULTATI
    # ------------------------------------------------------------------

    def _on_analisi_completata(self, risultati: dict) -> None:
        self._risultati = risultati
        self._ui.pressoflessione_btn_analisi.setEnabled(True)
        self._ui.pressoflessione_progressBar.setValue(100)

        verificata     = risultati.get('verificata', False)
        fuori_dominio  = risultati.get('fuori_dominio', False)

        if fuori_dominio:
            testo   = "Verifica NON soddisfatta  (fuori dominio)"
            c_bordo = "rgb(220, 80, 80)"
            c_testo = "rgb(220, 80, 80)"
        elif verificata:
            testo   = "Verifica soddisfatta"
            c_bordo = "rgb(80, 200, 120)"
            c_testo = "rgb(80, 200, 120)"
        else:
            testo   = "Verifica NON soddisfatta"
            c_bordo = "rgb(220, 80, 80)"
            c_testo = "rgb(220, 80, 80)"

        self._ui.pressoflessione_risultato_verifica.setText(testo)
        self._ui.pressoflessione_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo=c_bordo,
                colore_testo=c_testo,
                font_weight="font-weight: bold;"
            )
        )

        # Aggiorna il widget 2D
        self._w_analisi.set_results(risultati)

        # Log sintetico
        tipo = risultati.get('tipo', '?')
        N    = risultati.get('N_Ed', 0.0)
        M    = risultati.get('M_Ed', 0.0)
        if tipo == 'SLU':
            MRd = risultati.get('M_Rd', 0.0)
            r   = risultati.get('rapporto_MEd_MRd', risultati.get('rapporto', 0.0))
            print(f">> Pressoflessione {tipo}: N={N:.1f} kN  M={M:.1f} kNm  "
                  f"MRd={MRd:.2f} kNm  MEd/MRd={r:.3f}  "
                  f"→ {'OK' if verificata else 'KO'}")
        else:
            # Supporta sia chiavi nuove che vecchie per retrocompatibilità
            sc  = risultati.get('sigma_c_compr_max', risultati.get('sigma_c_max', 0.0))
            ss  = risultati.get('sigma_s_traz_max', 0.0)
            lc  = risultati.get('sigma_c_limit', risultati.get('lim_cls'))
            ls  = risultati.get('sigma_s_limit', risultati.get('lim_acc'))
            print(f">> Pressoflessione {tipo}: N={N:.1f} kN  M={M:.1f} kNm  "
                  f"σ_c={sc:.1f}/{lc if lc else '?'} MPa  "
                  f"σ_s={ss:.1f}/{ls if ls else '?'} MPa  "
                  f"→ {'OK' if verificata else 'KO'}")
            for nota in risultati.get('note', []):
                print(f"   ! {nota}")

    def _on_analisi_errore(self, msg: str) -> None:
        self._ui.pressoflessione_btn_analisi.setEnabled(True)
        self._ui.pressoflessione_progressBar.setValue(0)
        self._ui.pressoflessione_risultato_verifica.setText("Errore di calcolo")
        self._ui.pressoflessione_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(220, 80, 80)",
                colore_testo="rgb(220, 80, 80)",
                font_weight="font-weight: bold;"
            )
        )
        QMessageBox.critical(
            self._main, "Errore calcolo pressoflessione",
            f"Si è verificato un errore durante il calcolo:\n{msg}"
        )
        print(f"ERR  Pressoflessione: {msg}")

    # ------------------------------------------------------------------
    # VISTA
    # ------------------------------------------------------------------

    def _imposta_vista(self, mode: str) -> None:
        """Cambia la modalità di colorazione del widget 2D."""
        self._w_analisi.set_display_mode(mode)

    # ------------------------------------------------------------------
    # SALVATAGGIO / CARICAMENTO IMPOSTAZIONI
    # ------------------------------------------------------------------

    def _salva_impostazioni(self, params: dict) -> None:
        if not self._main.ha_progetto():
            return
        self._dati.salva_impostazioni_pressoflessione({
            'sezione_selezionata' : params['nome'],
            'grid_step'           : params['grid_step'],
            'theta'               : params['theta'],
            'N_Ed'                : params['N_Ed'],
            'M_Ed'                : params['M_Ed'],
            'modo'                : params['modo'],
        })

    def _carica_impostazioni(self) -> None:
        imp = self._dati.carica_impostazioni_pressoflessione()
        if not imp:
            return

        ui = self._ui

        # Ripristina combobox
        nome = imp.get('sezione_selezionata', '')
        if nome:
            idx = ui.pressoflessione_combobox_sezioni.findText(nome)
            if idx >= 0:
                ui.pressoflessione_combobox_sezioni.setCurrentIndex(idx)

        # Grid step
        gs = imp.get('grid_step')
        if gs is not None:
            ui.pressoflessione_definizione.setText(str(gs))

        # Rotazione
        theta = imp.get('theta')
        if theta is not None:
            ui.pressoflessione_rotazione.setText(str(theta))

        # Carichi
        N_Ed = imp.get('N_Ed')
        if N_Ed is not None:
            ui.pressoflessione_sforzo_normale.setText(str(N_Ed))

        M_Ed = imp.get('M_Ed')
        if M_Ed is not None:
            ui.pressoflessione_sollecitazione_flettente.setText(str(M_Ed))

        # Modo SLU/SLE
        modo = imp.get('modo', 'SLU')
        if modo == 'SLE':
            ui.pressoflessione_radioButton_sle.setChecked(True)
        else:
            ui.pressoflessione_radioButton_slu.setChecked(True)

    # ------------------------------------------------------------------
    # RICARICA DA PROGETTO  (chiamata da MainWindow)
    # ------------------------------------------------------------------

    def aggiorna_combobox(self) -> None:
        """Aggiorna la combobox sezioni (da chiamare quando si naviga nel pannello)."""
        self._popola_combobox()

    def ricarica_da_progetto(self) -> None:
        """
        Chiamata da MainWindow._imposta_progetto() ogni volta che
        viene aperto o creato un progetto.
        """
        self._sezione_corrente = None
        self._risultati        = None
        self._w_analisi.set_results(None)
        self._w_sezione.set_section_data(None)
        self._popola_combobox()
        self._carica_impostazioni()
        self._reset_ui()
        print(">> Modulo Pressoflessione: progetto ricaricato.")