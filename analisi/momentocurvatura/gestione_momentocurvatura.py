"""
gestione_momentocurvatura.py  –  analisi/momentocurvatura/
===========================================================
Controller del modulo Analisi Momento-Curvatura 3D (SLU).

Responsabilità:
  - Popola momentocurvatura_combobox_sezioni con tutte le sezioni.
  - Aggiorna l'anteprima sezione (momentocurvatura_widget_sezione).
  - Legge e valida i parametri (definizione, step_rotazione, step_punti,
    N, M).
  - Lancia il calcolo del diagramma M-χ 3D su thread separato con
    progress bar.
  - Gestisce i due bottoni di vista (3D / χ-M) in modo esclusivo,
    mostrando il widget corretto in momentocurvatura_widget.
  - Sincronizza slider ↔ campo angolo per la vista 2D.
  - Aggiorna l'anello/punto di verifica M_Ed in tempo reale senza
    ricalcolo; aggiorna momentocurvatura_risultato_verifica.
  - Salva e ripristina tutte le impostazioni nel progetto (.scprj).
  - Espone ricarica_da_progetto() per essere notificato da MainWindow.
"""
from __future__ import annotations

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QButtonGroup, QMessageBox, QStackedWidget, QVBoxLayout

from analisi.raccolta_dati import RaccoltaDati
from analisi.pressoflessione.calcolo import SezioneDiscretizzata

from .calcolo                          import CalcoloMomentoCurvatura, _MomentoCurvaturaThread
from .disegno_momentocurvatura         import MomentoCurvaturaWidget3D
from .disegno_momentocurvatura_2d      import MomentoCurvaturaWidget2D
from .disegno_momentocurvatura_sezione import MomentoCurvaturaSezioneWidget


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

class GestioneMomentoCurvatura:
    """
    Controller del modulo Momento-Curvatura 3D.

    Connette gli elementi UI ai moduli di calcolo e di visualizzazione.
    """

    _CHIAVE_PROGETTO = 'momentocurvatura'

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, ui, main_window) -> None:
        self._ui   = ui
        self._main = main_window
        self._dati = RaccoltaDati(main_window)

        # Stato
        self._sezione_corrente: str | None                   = None
        self._result_matrix:    object | None                = None
        self._thread:           _MomentoCurvaturaThread | None = None

        # Widget OpenGL
        self._w_sezione = MomentoCurvaturaSezioneWidget()
        self._w_3d      = MomentoCurvaturaWidget3D()
        self._w_2d      = MomentoCurvaturaWidget2D()

        self._setup_widget_sezione()
        self._setup_widget_output()
        self._setup_connessioni()
        self._reset_ui()

    # ------------------------------------------------------------------
    # SETUP – INSERIMENTO WIDGET
    # ------------------------------------------------------------------

    def _setup_widget_sezione(self) -> None:
        container = self._ui.momentocurvatura_widget_sezione
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(0)
        lay.addWidget(self._w_sezione)

    def _setup_widget_output(self) -> None:
        """
        Inserisce QStackedWidget in momentocurvatura_widget:
          indice 0 → 3D
          indice 1 → 2D (χ-M)
        """
        container = self._ui.momentocurvatura_widget
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._w_3d)   # idx 0
        self._stack.addWidget(self._w_2d)   # idx 1
        lay.addWidget(self._stack)

    # ------------------------------------------------------------------
    # SETUP – CONNESSIONI UI
    # ------------------------------------------------------------------

    def _setup_connessioni(self) -> None:
        ui = self._ui

        # Combobox sezione
        ui.momentocurvatura_combobox_sezioni.currentTextChanged.connect(
            self._on_sezione_cambiata
        )

        # Bottone avvia analisi
        ui.momentocurvatura_btn_analisi.clicked.connect(self._avvia_analisi)

        # Bottoni vista – checkable ed esclusivi
        for btn in (ui.momentocurvatura_btn_vista_3d,
                    ui.momentocurvatura_btn_vista_chi_M):
            btn.setCheckable(True)

        self._btn_group_vista = QButtonGroup(ui.momentocurvatura_btn_vista_3d.parent())
        self._btn_group_vista.setExclusive(True)
        self._btn_group_vista.addButton(ui.momentocurvatura_btn_vista_3d)
        self._btn_group_vista.addButton(ui.momentocurvatura_btn_vista_chi_M)

        ui.momentocurvatura_btn_vista_3d.setChecked(True)

        ui.momentocurvatura_btn_vista_3d.clicked.connect(
            lambda: self._cambia_vista('3d')
        )
        ui.momentocurvatura_btn_vista_chi_M.clicked.connect(
            lambda: self._cambia_vista('2d')
        )

        # Centra
        ui.momentocurvatura_centra.clicked.connect(self._centra_output)
        ui.momentocurvatura_sezione_centra.clicked.connect(
            self._w_sezione.reset_view
        )

        # Slider ↔ campo angolo (sincronia bidirezionale)
        ui.momentocurvatura_horizontalSlider.valueChanged.connect(
            self._on_slider_changed
        )
        ui.momentocurvatura_angolo.textChanged.connect(
            self._on_angolo_text_changed
        )

        # M_Ed → aggiorna verifica senza ricalcolo
        ui.momentocurvatura_sollecitazione_M.textChanged.connect(
            self._aggiorna_verifica_M
        )

        # Segnali verifica dai widget
        self._w_3d.verifica_cambiata.connect(self._on_verifica_3d)
        self._w_2d.verifica_cambiata.connect(self._on_verifica_2d)

    # ------------------------------------------------------------------
    # RESET UI
    # ------------------------------------------------------------------

    def _reset_ui(self) -> None:
        ui = self._ui
        ui.momentocurvatura_progressBar.setValue(0)
        ui.momentocurvatura_risultato_verifica.setText("Avviare analisi")
        ui.momentocurvatura_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(120, 120, 120)",
                colore_testo="rgb(160, 160, 160)",
                font_weight="font-weight: normal;"
            )
        )

        defaults = [
            (ui.momentocurvatura_definizione,       "10"),
            (ui.momentocurvatura_step_rotazione,    "36"),
            (ui.momentocurvatura_step_punti,        "50"),
            (ui.momentocurvatura_sollecitazione_N,  "0"),
            (ui.momentocurvatura_sollecitazione_M,  "0"),
            (ui.momentocurvatura_angolo,            "0"),
        ]
        for widget, val in defaults:
            if not widget.text().strip():
                widget.setText(val)

        # Slider range
        ui.momentocurvatura_horizontalSlider.setMinimum(0)
        ui.momentocurvatura_horizontalSlider.setMaximum(360)

    # ------------------------------------------------------------------
    # POPOLA COMBOBOX
    # ------------------------------------------------------------------

    def _popola_combobox(self) -> None:
        cb = self._ui.momentocurvatura_combobox_sezioni
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
            self._result_matrix    = None
            self._w_3d.set_points(None)
            self._w_2d.set_points(None)
            self._ui.momentocurvatura_progressBar.setValue(0)
            self._ui.momentocurvatura_risultato_verifica.setText("Avviare analisi")
            self._ui.momentocurvatura_risultato_verifica.setStyleSheet(
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

        nome = ui.momentocurvatura_combobox_sezioni.currentText()
        if not nome:
            QMessageBox.warning(
                self._main, "Parametri mancanti",
                "Seleziona una sezione prima di avviare l'analisi."
            )
            return None

        grid_step = self._leggi_float(ui.momentocurvatura_definizione, 10.0)
        if grid_step <= 0:
            QMessageBox.warning(
                self._main, "Parametri non validi",
                "Il passo di discretizzazione deve essere positivo."
            )
            return None

        n_angoli = max(6,  self._leggi_int(ui.momentocurvatura_step_rotazione, 36))
        n_punti  = max(10, self._leggi_int(ui.momentocurvatura_step_punti, 50))

        N_Ed = self._leggi_float(ui.momentocurvatura_sollecitazione_N, 0.0)
        M_Ed = self._leggi_float(ui.momentocurvatura_sollecitazione_M, 0.0)

        return {
            'nome':      nome,
            'grid_step': grid_step,
            'n_angoli':  n_angoli,
            'n_punti':   n_punti,
            'N_Ed':      N_Ed,
            'M_Ed':      M_Ed,
        }

    # ------------------------------------------------------------------
    # AVVIO ANALISI
    # ------------------------------------------------------------------

    def _avvia_analisi(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return

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

        mat_db = dati_analisi['materiali']

        def risolvi_mat(nome: str) -> dict | None:
            return mat_db.get(nome)

        try:
            sezione_disc = SezioneDiscretizzata(
                dati_sezione=sez_dati,
                risolvi_mat=risolvi_mat,
                grid_step=params['grid_step'],
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

        calcolatore = CalcoloMomentoCurvatura(sezione_disc)

        # Stato UI
        ui = self._ui
        ui.momentocurvatura_progressBar.setValue(0)
        ui.momentocurvatura_btn_analisi.setEnabled(False)
        ui.momentocurvatura_risultato_verifica.setText("Calcolo in corso…")
        ui.momentocurvatura_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(120, 120, 120)",
                colore_testo="rgb(160, 160, 160)",
                font_weight="font-weight: normal;"
            )
        )

        # Thread
        self._thread = _MomentoCurvaturaThread(
            calcolatore=calcolatore,
            n_angoli=params['n_angoli'],
            n_punti=params['n_punti'],
            N_target_kN=params['N_Ed'],
        )
        self._thread.avanzamento.connect(ui.momentocurvatura_progressBar.setValue)
        self._thread.completato.connect(self._on_calcolo_completato)
        self._thread.errore.connect(self._on_calcolo_errore)
        self._thread.start()

        self._salva_impostazioni(params)

    # ------------------------------------------------------------------
    # GESTIONE RISULTATI
    # ------------------------------------------------------------------

    def _on_calcolo_completato(self, matrix) -> None:
        self._result_matrix = matrix
        ui = self._ui
        ui.momentocurvatura_btn_analisi.setEnabled(True)
        ui.momentocurvatura_progressBar.setValue(100)

        # Trasmette ai widget
        self._w_3d.set_points(matrix)
        self._w_2d.set_points(matrix)

        # Sincronizza l'angolo corrente per il 2D
        self._on_slider_changed(ui.momentocurvatura_horizontalSlider.value())

        # Aggiorna verifica M_Ed
        self._aggiorna_verifica_M()

        print(f">> Momento-Curvatura: matrice ({matrix.shape[0]}θ "
              f"× {matrix.shape[1]}χ) trasmessa ai widget.")

    def _on_calcolo_errore(self, msg: str) -> None:
        ui = self._ui
        ui.momentocurvatura_btn_analisi.setEnabled(True)
        ui.momentocurvatura_progressBar.setValue(0)
        ui.momentocurvatura_risultato_verifica.setText("Errore di calcolo")
        ui.momentocurvatura_risultato_verifica.setStyleSheet(
            STILE_BASE_VERIFICA.format(
                colore_bordo="rgb(220, 80, 80)",
                colore_testo="rgb(220, 80, 80)",
                font_weight="font-weight: bold;"
            )
        )
        QMessageBox.critical(
            self._main, "Errore calcolo momento-curvatura",
            f"Si è verificato un errore durante il calcolo:\n{msg}"
        )
        print(f"ERR  Momento-Curvatura: {msg}")

    # ------------------------------------------------------------------
    # SLIDER ↔ ANGOLO
    # ------------------------------------------------------------------

    def _on_slider_changed(self, value: int) -> None:
        """Slider cambiato → aggiorna campo angolo e vista 2D."""
        angle_deg = float(value)
        self._ui.momentocurvatura_angolo.blockSignals(True)
        self._ui.momentocurvatura_angolo.setText(f"{angle_deg:.0f}")
        self._ui.momentocurvatura_angolo.blockSignals(False)

        self._w_2d.set_angle_deg(angle_deg)

        # Riaggiorna il punto M_Ed nella vista 2D
        if self._result_matrix is not None:
            M_Ed = self._leggi_float(self._ui.momentocurvatura_sollecitazione_M, 0.0)
            self._w_2d.set_M_Ed(M_Ed)

    def _on_angolo_text_changed(self, text: str) -> None:
        """Campo angolo cambiato → aggiorna slider."""
        try:
            val = float(text.replace(',', '.').strip())
        except (ValueError, AttributeError):
            return
        val = max(0, min(360, val))
        self._ui.momentocurvatura_horizontalSlider.blockSignals(True)
        self._ui.momentocurvatura_horizontalSlider.setValue(int(val))
        self._ui.momentocurvatura_horizontalSlider.blockSignals(False)

        self._w_2d.set_angle_deg(val)

        if self._result_matrix is not None:
            M_Ed = self._leggi_float(self._ui.momentocurvatura_sollecitazione_M, 0.0)
            self._w_2d.set_M_Ed(M_Ed)

    # ------------------------------------------------------------------
    # VERIFICA M_Ed
    # ------------------------------------------------------------------

    def _aggiorna_verifica_M(self) -> None:
        """Legge M_Ed dall'UI e aggiorna entrambi i widget."""
        if self._result_matrix is None:
            return

        M_Ed = self._leggi_float(self._ui.momentocurvatura_sollecitazione_M, 0.0)

        inside_3d = self._w_3d.set_M_Ed(M_Ed)
        inside_2d = self._w_2d.set_M_Ed(M_Ed)

        # La verifica definitiva è dal 3D (controlla tutti gli angoli)
        if inside_3d is not None:
            self._aggiorna_label_verifica(inside_3d)

    def _on_verifica_3d(self, inside: bool) -> None:
        self._aggiorna_label_verifica(inside)

    def _on_verifica_2d(self, inside: bool) -> None:
        # La verifica 2D è relativa al solo angolo corrente;
        # usiamo il 3D come autorità globale
        pass

    def _aggiorna_label_verifica(self, inside: bool) -> None:
        ui = self._ui
        if inside:
            testo   = "Verifica soddisfatta"
            c_bordo = "rgb(80, 200, 120)"
            c_testo = "rgb(80, 200, 120)"
        else:
            testo   = "Verifica NON soddisfatta"
            c_bordo = "rgb(220, 80, 80)"
            c_testo = "rgb(220, 80, 80)"

        ui.momentocurvatura_risultato_verifica.setText(testo)
        ui.momentocurvatura_risultato_verifica.setStyleSheet(
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
            # Aggiorna la slice 2D
            angle_deg = self._leggi_float(self._ui.momentocurvatura_angolo, 0.0)
            self._w_2d.set_angle_deg(angle_deg)
            if self._result_matrix is not None:
                M_Ed = self._leggi_float(self._ui.momentocurvatura_sollecitazione_M, 0.0)
                self._w_2d.set_M_Ed(M_Ed)

    def _centra_output(self) -> None:
        idx = self._stack.currentIndex()
        if idx == 0:
            self._w_3d.reset_view()
        else:
            self._w_2d.reset_view()

    # ------------------------------------------------------------------
    # SALVATAGGIO / CARICAMENTO
    # ------------------------------------------------------------------

    def _salva_impostazioni(self, params: dict) -> None:
        if not self._main.ha_progetto():
            return
        analisi = self._main.get_sezione('analisi')
        analisi[self._CHIAVE_PROGETTO] = {
            'sezione_selezionata': params['nome'],
            'grid_step':          params['grid_step'],
            'n_angoli':           params['n_angoli'],
            'n_punti':            params['n_punti'],
            'N_Ed':               params['N_Ed'],
            'M_Ed':               params['M_Ed'],
        }
        self._main.set_sezione('analisi', analisi)

    def _carica_impostazioni(self) -> None:
        analisi = self._main.get_sezione('analisi')
        imp     = analisi.get(self._CHIAVE_PROGETTO, {})
        if not imp:
            return

        ui = self._ui

        nome = imp.get('sezione_selezionata', '')
        if nome:
            idx = ui.momentocurvatura_combobox_sezioni.findText(nome)
            if idx >= 0:
                ui.momentocurvatura_combobox_sezioni.setCurrentIndex(idx)

        for chiave, widget, default in (
            ('grid_step', ui.momentocurvatura_definizione,      '10'),
            ('n_angoli',  ui.momentocurvatura_step_rotazione,   '36'),
            ('n_punti',   ui.momentocurvatura_step_punti,       '50'),
            ('N_Ed',      ui.momentocurvatura_sollecitazione_N, '0'),
            ('M_Ed',      ui.momentocurvatura_sollecitazione_M, '0'),
        ):
            val = imp.get(chiave)
            if val is not None:
                widget.setText(str(val))

    # ------------------------------------------------------------------
    # RICARICA DA PROGETTO
    # ------------------------------------------------------------------

    def aggiorna_combobox(self) -> None:
        """Aggiorna la combobox sezioni (da chiamare quando si naviga nel pannello)."""
        self._popola_combobox()

    def ricarica_da_progetto(self) -> None:
        """
        Chiamata da MainWindow._imposta_progetto() ogni volta che
        viene aperto o creato un progetto.
        """
        if self._thread is not None and self._thread.isRunning():
            self._thread.richiedi_stop()
            self._thread.wait(2000)

        self._sezione_corrente = None
        self._result_matrix    = None

        self._w_3d.set_points(None)
        self._w_2d.set_points(None)
        self._w_sezione.set_section_data(None)

        self._popola_combobox()
        self._carica_impostazioni()
        self._reset_ui()

        print(">> Modulo Momento-Curvatura: progetto ricaricato.")
