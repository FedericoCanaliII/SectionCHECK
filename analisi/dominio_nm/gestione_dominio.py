"""
gestione_dominio.py  –  analisi/dominio_nm/
=============================================
Controller del modulo Dominio di Interazione N-M.

Responsabilità:
  - Popola dominio_combobox_sezioni con tutte le sezioni del progetto.
  - Aggiorna l'anteprima sezione (dominio_widget_sezione) al cambio di selezione.
  - Legge e valida i parametri di calcolo (grid_step, theta_steps, k_steps).
  - Lancia il calcolo del dominio 3D su thread separato con progress bar.
  - Gestisce i quattro bottoni di vista (3D / N-Mx / N-My / Mx-My) in
    modo esclusivo, mostrando il widget corretto in dominio_widget.
  - Aggiorna il punto di verifica (N_Ed, Mx_Ed, My_Ed) in tempo reale
    sui widget 3D e 2D; aggiorna dominio_risultato_verifica.
  - Salva e ripristina tutte le impostazioni nel progetto (.scprj).
  - Espone ricarica_da_progetto() per essere notificato da MainWindow.
"""
from __future__ import annotations

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QButtonGroup, QMessageBox, QStackedWidget, QVBoxLayout

from analisi.raccolta_dati import RaccoltaDati
from analisi.pressoflessione.calcolo import SezioneDiscretizzata

from .calcolo               import CalcoloDominioNM, _DominioThread
from .disegno_dominio        import DominioWidget3D
from .disegno_dominio_2d     import DominioWidget2D
from .disegno_dominio_sezione import DominioSezioneWidget


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
# CONTROLLER
# ==============================================================================

class GestioneDominioNM:
    """
    Controller del modulo Dominio di Interazione N-M 3D.

    Connette gli elementi UI ai moduli di calcolo e di visualizzazione.
    """

    # Chiave usata nel dict analisi del progetto
    _CHIAVE_PROGETTO = 'dominio_nm'

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, ui, main_window) -> None:
        self._ui   = ui
        self._main = main_window
        self._dati = RaccoltaDati(main_window)

        # Stato
        self._sezione_corrente: str | None             = None
        self._domain_matrix:    object | None          = None   # np.ndarray
        self._thread:           _DominioThread | None  = None

        # Widget OpenGL
        self._w_sezione  = DominioSezioneWidget()
        self._w_3d       = DominioWidget3D()
        self._w_2d       = DominioWidget2D()

        self._setup_widget_sezione()
        self._setup_widget_dominio()
        self._setup_connessioni()
        self._reset_ui()

    # ------------------------------------------------------------------
    # SETUP – INSERIMENTO WIDGET NEI PLACEHOLDER
    # ------------------------------------------------------------------

    def _setup_widget_sezione(self) -> None:
        container = self._ui.dominio_widget_sezione
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4,4,4,4)
            lay.setSpacing(0)
        lay.addWidget(self._w_sezione)

    def _setup_widget_dominio(self) -> None:
        """
        Inserisce uno QStackedWidget in dominio_widget con:
          indice 0 → DominioWidget3D
          indice 1 → DominioWidget2D  (gestisce tutte e 3 le viste 2D)
        """
        container = self._ui.dominio_widget
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(0,0,0,0)
            lay.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._w_3d)    # idx 0
        self._stack.addWidget(self._w_2d)    # idx 1
        lay.addWidget(self._stack)

    # ------------------------------------------------------------------
    # SETUP – CONNESSIONI UI
    # ------------------------------------------------------------------

    def _setup_connessioni(self) -> None:
        ui = self._ui

        # Combobox sezione
        ui.dominio_combobox_sezioni.currentTextChanged.connect(
            self._on_sezione_cambiata
        )

        # Bottone avvia analisi
        ui.dominio_btn_analisi.clicked.connect(self._avvia_analisi)

        # Bottoni vista – resi checkable ed esclusivi
        for btn in (ui.dominio_btn_vista_3d,
                    ui.dominio_btn_vista_N_Mx,
                    ui.dominio_btn_vista_N_My,
                    ui.dominio_btn_vista_Mx_My):
            btn.setCheckable(True)

        self._btn_group_vista = QButtonGroup(ui.dominio_btn_vista_3d.parent())
        self._btn_group_vista.setExclusive(True)
        self._btn_group_vista.addButton(ui.dominio_btn_vista_3d)
        self._btn_group_vista.addButton(ui.dominio_btn_vista_N_Mx)
        self._btn_group_vista.addButton(ui.dominio_btn_vista_N_My)
        self._btn_group_vista.addButton(ui.dominio_btn_vista_Mx_My)

        ui.dominio_btn_vista_3d.setChecked(True)

        ui.dominio_btn_vista_3d.clicked.connect(
            lambda: self._cambia_vista('3d'))
        ui.dominio_btn_vista_N_Mx.clicked.connect(
            lambda: self._cambia_vista('N_Mx'))
        ui.dominio_btn_vista_N_My.clicked.connect(
            lambda: self._cambia_vista('N_My'))
        ui.dominio_btn_vista_Mx_My.clicked.connect(
            lambda: self._cambia_vista('Mx_My'))

        # Pulsanti centra
        ui.dominio_centra.clicked.connect(self._centra_dominio)
        ui.dominio_sezione_centra.clicked.connect(self._w_sezione.reset_view)

        # Sollecitazioni di verifica – aggiornamento in tempo reale
        ui.dominio_sollecitazione_N.textChanged.connect(
            self._aggiorna_punto_verifica)
        ui.dominio_sollecitazione_Mx.textChanged.connect(
            self._aggiorna_punto_verifica)
        ui.dominio_sollecitazione_My.textChanged.connect(
            self._aggiorna_punto_verifica)

        # Segnali di verifica dai widget
        self._w_3d.verifica_cambiata.connect(self._on_verifica_3d)
        self._w_2d.verifica_cambiata.connect(self._on_verifica_2d)

    # ------------------------------------------------------------------
    # RESET UI
    # ------------------------------------------------------------------

    def _reset_ui(self) -> None:
        ui = self._ui
        ui.dominio_progressBar.setValue(0)
        ui.dominio_risultato_verifica.setText("Avviare analisi")
        ui.dominio_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(120, 120, 120)",
                colore_testo="rgb(160, 160, 160)",
                font_weight="font-weight: normal;"
            )
        )

        # Valori di default (solo se i campi sono vuoti)
        defaults = [
            (ui.dominio_definizione,     "10"),
            (ui.dominio_step_rotazione,  "36"),
            (ui.dominio_step_traslazione,"30"),
            (ui.dominio_sollecitazione_N, "0"),
            (ui.dominio_sollecitazione_Mx,"0"),
            (ui.dominio_sollecitazione_My,"0"),
        ]
        for widget, val in defaults:
            if not widget.text().strip():
                widget.setText(val)

    # ------------------------------------------------------------------
    # POPOLA COMBOBOX
    # ------------------------------------------------------------------

    def _popola_combobox(self) -> None:
        cb = self._ui.dominio_combobox_sezioni
        cb.blockSignals(True)
        cb.clear()
        nomi = self._dati.lista_sezioni()
        for nome in nomi:
            cb.addItem(nome)
        cb.blockSignals(False)

        if not nomi:
            return

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

        dati = self._dati.dati_sezione(nome)
        self._w_sezione.set_section_data(dati)

        # Resetta risultati SOLO se la sezione è davvero cambiata
        if nome != self._sezione_corrente:
            self._sezione_corrente = nome
            self._domain_matrix    = None
            self._w_3d.set_points(None)
            self._w_2d.set_points(None)
            self._ui.dominio_progressBar.setValue(0)
            self._ui.dominio_risultato_verifica.setText("Avviare analisi")
            self._ui.dominio_risultato_verifica.setStyleSheet(
                STILE_BASE_VERIFICA.format(
                    colore_bordo="rgb(120, 120, 120)",
                    colore_testo="rgb(160, 160, 160)",
                    font_weight="font-weight: normal;"
                )
            )

    # ------------------------------------------------------------------
    # LETTURA PARAMETRI
    # ------------------------------------------------------------------

    def _leggi_float(self, widget, default: float) -> float:
        try:
            return float(widget.text().replace(',', '.').strip())
        except (ValueError, AttributeError):
            return default

    def _leggi_int(self, widget, default: int) -> int:
        try:
            return int(float(widget.text().replace(',', '.').strip()))
        except (ValueError, AttributeError):
            return default

    def _leggi_parametri(self) -> dict | None:
        ui = self._ui

        nome = ui.dominio_combobox_sezioni.currentText()
        if not nome:
            QMessageBox.warning(
                self._main, "Parametri mancanti",
                "Seleziona una sezione prima di avviare l'analisi."
            )
            return None

        grid_step = self._leggi_float(ui.dominio_definizione, 10.0)
        if grid_step <= 0:
            QMessageBox.warning(
                self._main, "Parametri non validi",
                "Il passo di discretizzazione deve essere positivo."
            )
            return None

        theta_steps = self._leggi_int(ui.dominio_step_rotazione, 36)
        k_steps     = self._leggi_int(ui.dominio_step_traslazione, 30)

        theta_steps = max(4,  theta_steps)
        k_steps     = max(5,  k_steps)

        N_Ed  = self._leggi_float(ui.dominio_sollecitazione_N,  0.0)
        Mx_Ed = self._leggi_float(ui.dominio_sollecitazione_Mx, 0.0)
        My_Ed = self._leggi_float(ui.dominio_sollecitazione_My, 0.0)

        return {
            'nome':         nome,
            'grid_step':    grid_step,
            'theta_steps':  theta_steps,
            'k_steps':      k_steps,
            'N_Ed':         N_Ed,
            'Mx_Ed':        Mx_Ed,
            'My_Ed':        My_Ed,
        }

    # ------------------------------------------------------------------
    # AVVIO ANALISI
    # ------------------------------------------------------------------

    def _avvia_analisi(self) -> None:
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

        sez_dati = dati_analisi['sezione']
        elementi = sez_dati.get('elementi', {})
        if (len(elementi.get('carpenteria', [])) +
                len(elementi.get('barre', []))) == 0:
            QMessageBox.warning(
                self._main, "Sezione vuota",
                "La sezione selezionata non contiene elementi geometrici."
            )
            return

        # Resolver materiali
        mat_db = dati_analisi['materiali']

        def risolvi_mat(nome: str) -> dict | None:
            return mat_db.get(nome)

        # Discretizzazione
        try:
            sezione_disc = SezioneDiscretizzata(
                dati_sezione = sez_dati,
                risolvi_mat  = risolvi_mat,
                grid_step    = params['grid_step'],
            )
        except Exception as e:
            QMessageBox.critical(
                self._main, "Errore discretizzazione",
                f"Errore durante la discretizzazione:\n{e}"
            )
            return

        if not sezione_disc.fibre:
            QMessageBox.warning(
                self._main, "Discretizzazione vuota",
                "Nessuna fibra generata. Riduci il passo di discretizzazione."
            )
            return

        calcolatore = CalcoloDominioNM(sezione_disc)

        # Stato UI
        ui = self._ui
        ui.dominio_progressBar.setValue(0)
        ui.dominio_btn_analisi.setEnabled(False)
        ui.dominio_risultato_verifica.setText("Calcolo in corso…")
        ui.dominio_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(120, 120, 120)",
                colore_testo="rgb(160, 160, 160)",
                font_weight="font-weight: normal;"
            )
        )

        # Avvio thread
        self._thread = _DominioThread(
            calcolatore   = calcolatore,
            theta_steps   = params['theta_steps'],
            neutral_steps = params['k_steps'],
        )
        self._thread.avanzamento.connect(ui.dominio_progressBar.setValue)
        self._thread.completato.connect(self._on_calcolo_completato)
        self._thread.errore.connect(self._on_calcolo_errore)
        self._thread.start()

        # Salva impostazioni
        self._salva_impostazioni(params)

    # ------------------------------------------------------------------
    # GESTIONE RISULTATI
    # ------------------------------------------------------------------

    def _on_calcolo_completato(self, matrix) -> None:
        self._domain_matrix = matrix
        ui = self._ui
        ui.dominio_btn_analisi.setEnabled(True)
        ui.dominio_progressBar.setValue(100)

        # Trasmette i dati ai widget
        self._w_3d.set_points(matrix)
        self._w_2d.set_points(matrix)

        # Aggiorna il punto di verifica corrente
        self._aggiorna_punto_verifica()

        print(f">> Dominio N-M: matrice ({matrix.shape[0]}θ "
              f"× {matrix.shape[1]}k) trasmessa ai widget.")

    def _on_calcolo_errore(self, msg: str) -> None:
        ui = self._ui
        ui.dominio_btn_analisi.setEnabled(True)
        ui.dominio_progressBar.setValue(0)
        ui.dominio_risultato_verifica.setText("Errore di calcolo")
        ui.dominio_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(220, 80, 80)",
                colore_testo="rgb(220, 80, 80)",
                font_weight="font-weight: bold;"
            )
        )
        QMessageBox.critical(
            self._main, "Errore calcolo dominio N-M",
            f"Si è verificato un errore durante il calcolo:\n{msg}"
        )
        print(f"ERR  Dominio N-M: {msg}")

    # ------------------------------------------------------------------
    # PUNTO DI VERIFICA
    # ------------------------------------------------------------------

    def _aggiorna_punto_verifica(self) -> None:
        """Legge N_Ed, Mx_Ed, My_Ed dall'UI e aggiorna entrambi i widget."""
        if self._domain_matrix is None:
            return

        try:
            N  = self._leggi_float(self._ui.dominio_sollecitazione_N,  0.0)
            Mx = self._leggi_float(self._ui.dominio_sollecitazione_Mx, 0.0)
            My = self._leggi_float(self._ui.dominio_sollecitazione_My, 0.0)
        except Exception:
            return

        # Aggiorna entrambi i widget; la verifica definitiva la fornisce il 2D
        self._w_3d.set_verification_point(N, Mx, My)
        inside_2d = self._w_2d.set_verification_point(N, Mx, My)

        if inside_2d is not None:
            self._aggiorna_label_verifica(inside_2d)

    def _on_verifica_3d(self, inside: bool) -> None:
        """Callback dal widget 3D – usato solo se il 2D non ha dati."""
        if self._domain_matrix is not None and self._w_2d._slice_polygon is None:
            self._aggiorna_label_verifica(inside)

    def _on_verifica_2d(self, inside: bool) -> None:
        """Callback dal widget 2D – autorità definitiva sulla verifica."""
        self._aggiorna_label_verifica(inside)

    def _aggiorna_label_verifica(self, inside: bool) -> None:
        ui = self._ui
        if inside:
            testo = "Verifica soddisfatta"
            c_bordo = "rgb(80, 200, 120)"
            c_testo = "rgb(80, 200, 120)"
        else:
            testo = "Verifica NON soddisfatta"
            c_bordo = "rgb(220, 80, 80)"
            c_testo = "rgb(220, 80, 80)"
            
        ui.dominio_risultato_verifica.setText(testo)
        ui.dominio_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo=c_bordo,
                colore_testo=c_testo,
                font_weight="font-weight: bold;"
            )
        )

    # ------------------------------------------------------------------
    # CAMBIO VISTA
    # ------------------------------------------------------------------

    def _cambia_vista(self, mode: str) -> None:
        if mode == '3d':
            self._stack.setCurrentIndex(0)
        else:
            self._stack.setCurrentIndex(1)
            self._w_2d.set_view_mode(mode)
            # Riaggiorna punto verifica nella nuova vista
            self._aggiorna_punto_verifica()

    def _centra_dominio(self) -> None:
        idx = self._stack.currentIndex()
        if idx == 0:
            self._w_3d.reset_view()
        else:
            self._w_2d.reset_view()

    # ------------------------------------------------------------------
    # SALVATAGGIO / CARICAMENTO IMPOSTAZIONI
    # ------------------------------------------------------------------

    def _salva_impostazioni(self, params: dict) -> None:
        if not self._main.ha_progetto():
            return
        analisi = self._main.get_sezione('analisi')
        analisi[self._CHIAVE_PROGETTO] = {
            'sezione_selezionata': params['nome'],
            'grid_step':           params['grid_step'],
            'theta_steps':         params['theta_steps'],
            'k_steps':             params['k_steps'],
            'N_Ed':                params['N_Ed'],
            'Mx_Ed':               params['Mx_Ed'],
            'My_Ed':               params['My_Ed'],
        }
        self._main.set_sezione('analisi', analisi)

    def _carica_impostazioni(self) -> None:
        analisi = self._main.get_sezione('analisi')
        imp     = analisi.get(self._CHIAVE_PROGETTO, {})
        if not imp:
            return

        ui = self._ui

        # Combobox sezione
        nome = imp.get('sezione_selezionata', '')
        if nome:
            idx = ui.dominio_combobox_sezioni.findText(nome)
            if idx >= 0:
                ui.dominio_combobox_sezioni.setCurrentIndex(idx)

        # Parametri di calcolo
        for chiave, widget, default in (
            ('grid_step',   ui.dominio_definizione,      '10'),
            ('theta_steps', ui.dominio_step_rotazione,   '36'),
            ('k_steps',     ui.dominio_step_traslazione, '30'),
            ('N_Ed',        ui.dominio_sollecitazione_N,  '0'),
            ('Mx_Ed',       ui.dominio_sollecitazione_Mx, '0'),
            ('My_Ed',       ui.dominio_sollecitazione_My, '0'),
        ):
            val = imp.get(chiave)
            if val is not None:
                widget.setText(str(val))

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
        # Ferma eventuale calcolo in corso
        if self._thread is not None and self._thread.isRunning():
            self._thread.richiedi_stop()
            self._thread.wait(2000)

        self._sezione_corrente = None
        self._domain_matrix    = None

        self._w_3d.set_points(None)
        self._w_2d.set_points(None)
        self._w_sezione.set_section_data(None)

        self._popola_combobox()
        self._carica_impostazioni()
        self._reset_ui()

        print(">> Modulo Dominio N-M: progetto ricaricato.")