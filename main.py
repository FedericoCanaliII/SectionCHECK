import sys
import json
import os
import copy
from collections import deque
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon, QFont, QKeySequence
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QFileDialog,
                             QMessageBox, QShortcut, QLineEdit, QTextEdit,
                             QPlainTextEdit)

from interfaccia.main_interfaccia import Ui_MainWindow
from materiali import GestioneMateriali
from sezioni  import GestioneSezioni
from struttura.gestione_struttura import GestioneStruttura
from analisi.pressoflessione.gestione_pressoflessione import GestionePressoflessione
from analisi.dominio_nm.gestione_dominio import GestioneDominioNM
from analisi.momentocurvatura.gestione_momentocurvatura import GestioneMomentoCurvatura
from analisi.fem_elemento.gestione_mesh_elemento import GestioneMeshElemento
from analisi.fem_struttura.gestione_fem_struttura import GestioneFemStruttura
from ai.gestione_ai import GestioneAI
from elementi import GestioneElementi

# ============================================================
#  STREAM TERMINALE
# ============================================================

class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str, str)   # (testo, tipo)

    def __init__(self, original_stream):
        super().__init__()
        self._original = original_stream
        self._buffer   = ""

    def write(self, text: str):
        # FIX: Controlla se lo stream originale esiste (con Win32GUI è None)
        if self._original is not None:
            self._original.write(text)
            self._original.flush()
            
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.textWritten.emit(line, self._classifica(line))

    def flush(self):
        # FIX: Controlla se lo stream originale esiste prima del flush
        if self._original is not None:
            self._original.flush()
            
        if self._buffer.strip():
            self.textWritten.emit(self._buffer, self._classifica(self._buffer))
            self._buffer = ""

    @staticmethod
    def _classifica(line: str) -> str:
        l = line.upper()
        if any(k in l for k in ("ERR", "EXCEPTION", "TRACEBACK", "CRITICAL")):
            return "error"
        if any(k in l for k in ("WARN", "WARNING")):
            return "warning"
        return "info"

# ============================================================
#  STRUTTURA DATI PROGETTO
# ============================================================

def nuovo_progetto_template(nome: str) -> dict:
    """
    Crea il dizionario base di un nuovo progetto.
    La sezione 'materiali' viene pre-popolata con l'intero
    database standard (database_materiali.json) cosi ogni
    modifica viene tracciata nel .scprj.
    """
    try:
        from materiali.database_materiali import carica_database
        materiali_default = carica_database()
    except Exception as e:
        print(f"WARN  Database materiali non caricato: {e}")
        materiali_default = {
            "calcestruzzo": {}, "barre": {},
            "acciaio": {}, "personalizzati": {}
        }
    try:
        from elementi.database_elementi import carica_database as _carica_db_elem
        elementi_default = _carica_db_elem()
    except Exception as e:
        print(f"WARN  Database elementi non caricato: {e}")
        elementi_default = {"trave": [], "pilastro": [], "fondazione": [], "solaio": []}

    return {
        "metadata": {
            "nome_progetto":  nome,
            "versione_app":   "1.0.0",
            "data_creazione": datetime.now().isoformat(timespec="seconds"),
            "data_modifica":  datetime.now().isoformat(timespec="seconds"),
        },
        "materiali":  materiali_default,
        "sezioni":    {
            "calcestruzzo_armato": {},
            "profili":             {},
            "precompresso":        {},
            "personalizzate":      {},
        },
        "elementi":   elementi_default,
        "strutture":  {
            "calcestruzzo": {},
            "acciaio":      {},
            "personalizzate": {},
        },
        "geometria":  {},
        "analisi":    {},
        "carichi":    {},   # keyed by elemento id (str) → list of CaricoVincolo dicts
        "risultati":  {},
    }


# ============================================================
#  FINESTRA PRINCIPALE
# ============================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # --- Stato progetto ---
        self._progetto_path: str | None  = None
        self._progetto_dati: dict | None = None
        self._modificato:    bool        = False

        # --- Undo/Redo per modulo ---
        # Ogni modulo ha il proprio stack, così Ctrl+Z agisce solo
        # sulla pagina corrente senza interferire con gli altri moduli.
        self._undo_stacks: dict[str, deque] = {}
        self._redo_stacks: dict[str, deque] = {}

        # Mappa indice pagina → scope undo (ogni pagina ha il suo stack)
        self._PAGINA_MODULO: dict[int, str] = {
            1: "lista_materiali",   # lista materiali
            2: "materiale",         # dettaglio materiale
            3: "lista_sezioni",     # lista sezioni
            4: "sezione",           # disegno sezione (SpazioDisegno)
            5: "lista_elementi",    # lista elementi
            6: "elemento_3d",       # spazio 3D singolo elemento
            7: "extra_elemento",    # carichi / vincoli
            8: "lista_strutture",   # lista strutture
            9: "struttura",         # dettaglio struttura (testo + 3D)
        }
        # Mappa scope → chiavi di _progetto_dati da salvare nello snapshot
        self._MODULO_CHIAVI: dict[str, list[str]] = {
            "lista_materiali": ["materiali"],
            "materiale":       ["materiali"],
            "lista_sezioni":   ["sezioni"],
            "sezione":         ["sezioni"],
            "lista_elementi":  ["elementi"],
            "elemento_3d":     ["elementi"],
            "extra_elemento":  ["elementi", "carichi"],
            "lista_strutture": ["strutture"],
            "struttura":       ["strutture"],
        }

        # --- Finestra ---
        self.setWindowTitle("SectionCHECK")
        icon_path = os.path.join(os.path.dirname(__file__),
                                 "interfaccia", "icone", "logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.showMaximized()

        # --- Setup ---
        self.ui.frame_impostazioni.setParent(self)
        self.ui.frame_impostazioni.hide() # Assicurati che parta nascosta
        self.ui.btn_main_impostazioni.clicked.connect(self.mostra_nascondi_tendina) # Collega il click!
        self.ui.btn_main_tenda.clicked.connect(lambda: self.toggle_tende())
        self.ui.frame_tendina.setVisible(False)
        self.ui.frame_tendina_2.setVisible(True)
        self._setup_terminale()
        self._setup_navigazione()
        self._setup_pulsanti_progetto()
        self._setup_drag_drop()

        # --- Sotto-moduli ---
        self._materiali  = GestioneMateriali(self.ui, self)
        self._sezioni    = GestioneSezioni(self.ui, self)
        self._elementi   = GestioneElementi(self.ui, self)
        self._struttura  = GestioneStruttura(self.ui, self)
        self._pressoflessione = GestionePressoflessione(self.ui, self)
        self._dominio_nm      = GestioneDominioNM(self.ui, self)
        self._momentocurvatura = GestioneMomentoCurvatura(self.ui, self)
        self._fem_elemento    = GestioneMeshElemento(self.ui, self)
        self._fem_struttura   = GestioneFemStruttura(self.ui, self)

        # Aggiorna combobox sezioni quando si naviga nei pannelli di analisi
        self.ui.btn_main_pressoflessione.clicked.connect(
            self._pressoflessione.aggiorna_combobox
        )
        self.ui.btn_main_dominio.clicked.connect(
            self._dominio_nm.aggiorna_combobox
        )
        self.ui.btn_main_momentocurvatura.clicked.connect(
            self._momentocurvatura.aggiorna_combobox
        )
        self.ui.btn_main_fem_elemento.clicked.connect(
            self._fem_elemento.aggiorna_combobox
        )
        self.ui.btn_main_struttura.clicked.connect(
            self._fem_struttura.aggiorna_combobox
        )
        self.ui.btn_main_struttura_2.clicked.connect(
            self._fem_struttura.aggiorna_combobox
        )

        # --- Finestra AI (creata una volta, riutilizzata) ---
        self._ai_window: GestioneAI | None = None
        self.ui.btn_main_ai.clicked.connect(self._apri_ai)

        self.ui.btn_main_sc.click()
        print(">> SectionCHECK avviato.")

        self.ui.btn_main_stampa.clicked.connect(self._salva_screenshot)

    # ----------------------------------------------------------
    #  TERMINALE/IMPOSTAZIONI
    # ----------------------------------------------------------
    # Fuori dal costruttore (o come funzione separata)
    def toggle_tende(self):
        # Leggiamo lo stato attuale della prima
        nuovo_stato = not self.ui.frame_tendina.isVisible()
        
        # Applichiamo lo stato nuovo alla prima e l'opposto alla seconda
        self.ui.frame_tendina.setVisible(nuovo_stato)
        self.ui.frame_tendina_2.setVisible(not nuovo_stato)

    def mostra_nascondi_tendina(self):
        if self.ui.frame_impostazioni.isVisible():
            self.ui.frame_impostazioni.hide()
        else:
            # Usiamo QtCore.QPoint (assicurati che QtCore sia importato in alto, e lo è)
            punto = self.ui.btn_main_impostazioni.mapTo(self, QtCore.QPoint(-65, 45))
            
            x = punto.x()
            y = punto.y()

            # Sposta la tendina e mostrala
            self.ui.frame_impostazioni.move(x, y)
            self.ui.frame_impostazioni.show()
            self.ui.frame_impostazioni.raise_()

    def _setup_terminale(self):
        contenitore = self.ui.widget_terminale

        self._terminale = QtWidgets.QPlainTextEdit(contenitore)
        self._terminale.setReadOnly(True)
        self._terminale.setFont(QFont("Consolas", 9))
        self._terminale.setStyleSheet(
            "background-color:rgb(50,50,50);color:#d4d4d4;border:none;"
        )

        lay = contenitore.layout()
        if lay is None:
            lay = QtWidgets.QVBoxLayout(contenitore)
            lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self._terminale)

        self._stream_out = EmittingStream(sys.stdout)
        self._stream_out.textWritten.connect(self._scrivi_terminale)
        sys.stdout = self._stream_out

        self._stream_err = EmittingStream(sys.stderr)
        self._stream_err.textWritten.connect(self._scrivi_terminale)
        sys.stderr = self._stream_err

    def _scrivi_terminale(self, testo: str, tipo: str):
        colori = {"info": "#d4d4d4", "warning": "#ce9178", "error": "#f44747"}
        ts  = datetime.now().strftime("%H:%M:%S")
        txt = (testo.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
        html = (f'<span style="color:#569cd6;">[{ts}]</span> '
                f'<span style="color:{colori.get(tipo, "#d4d4d4")};">{txt}</span>')
        self._terminale.appendHtml(html)
        sb = self._terminale.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ----------------------------------------------------------
    #  NAVIGAZIONE
    # ----------------------------------------------------------

    def _setup_navigazione(self):
        mappatura = {
            self.ui.btn_main_sc:                 0,
            self.ui.btn_main_sc_2:               0,
            self.ui.btn_main_materiali:          1,
            self.ui.btn_main_materiali_2:        1,
            self.ui.btn_main_sezioni:            3,
            self.ui.btn_main_sezioni_2:          3,
            self.ui.btn_main_elementi:           5,
            self.ui.btn_main_elementi_2:         5,
            self.ui.btn_main_strutture:          8,
            self.ui.btn_main_strutture_2:        8,
            self.ui.btn_main_pressoflessione:    10,
            self.ui.btn_main_pressoflessione_2:  10,
            self.ui.btn_main_dominio:            11,
            self.ui.btn_main_dominio_2:          11,
            self.ui.btn_main_momentocurvatura:   12,
            self.ui.btn_main_momentocurvatura_2: 12,
            self.ui.btn_main_fem_elemento:       13,
            self.ui.btn_main_fem_elemento_2:     13,
            self.ui.btn_main_struttura:          14,
            self.ui.btn_main_struttura_2:        14,
        }
        
        # Creiamo DUE gruppi separati per evitare che i bottoni gemelli 
        # si deselezionino a vicenda essendo parte dello stesso exclusive group
        self.gruppo_base = QButtonGroup(self)
        self.gruppo_2 = QButtonGroup(self)
        
        self.gruppo_base.setExclusive(True)
        self.gruppo_2.setExclusive(True)

        visti = set()

        for btn, idx in mappatura.items():
            btn.setCheckable(True)
            
            # Smistiamo il primo bottone nel gruppo_base e il gemello nel gruppo_2
            if idx not in visti:
                self.gruppo_base.addButton(btn)
                visti.add(idx)
            else:
                self.gruppo_2.addButton(btn)

            # Colleghiamo il click al cambio pagina nello stackedWidget
            btn.clicked.connect(lambda _, i=idx: self.ui.stackedWidget_main.setCurrentIndex(i))
            # Sincronizziamo lo stato checked/unchecked del gemello
            btn.clicked.connect(lambda _, b=btn, i=idx: self._sincronizza_bottoni(b, i, mappatura))

    def _sincronizza_bottoni(self, btn_cliccato, idx, mappatura):
        """Sincronizza lo stato tra la tendina 1 e la tendina 2 senza loop infinito"""
        for gemello, gemello_idx in mappatura.items():
            if gemello is not btn_cliccato and gemello_idx == idx:
                gemello.blockSignals(True)   # Blocca il segnale per evitare loop
                gemello.setChecked(True)     # Seleziona il gemello
                gemello.blockSignals(False)  # Sblocca il segnale

        

    # ----------------------------------------------------------
    #  PULSANTI PROGETTO
    # ----------------------------------------------------------

    def _setup_pulsanti_progetto(self):
        self.ui.btn_main_nuovo.clicked.connect(self._nuovo_progetto)
        self.ui.btn_main_salva.clicked.connect(self._salva_progetto)
        self.ui.btn_main_carica.clicked.connect(self._carica_dialog)

        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)

    # ----------------------------------------------------------
    #  FINESTRA AI
    # ----------------------------------------------------------

    def _apri_ai(self):
        """Apre (o porta in primo piano) la finestra AI Agent."""
        if self._ai_window is None:
            self._ai_window = GestioneAI(self)
        self._ai_window.show()
        self._ai_window.raise_()
        self._ai_window.activateWindow()

    # ----------------------------------------------------------
    #  DRAG & DROP
    # ----------------------------------------------------------

    def _setup_drag_drop(self):
        btn = self.ui.btn_main_drop
        btn.setAcceptDrops(True)
        btn.installEventFilter(self)
        btn.clicked.connect(self._carica_dialog)

    def eventFilter(self, obj, event):
        if obj is self.ui.btn_main_drop:
            t = event.type()
            if t == QtCore.QEvent.DragEnter:
                if event.mimeData().hasUrls() and any(
                    u.toLocalFile().endswith(".scprj")
                    for u in event.mimeData().urls()
                ):
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return True
            if t == QtCore.QEvent.DragMove:
                event.acceptProposedAction()
                return True
            if t == QtCore.QEvent.Drop:
                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    if path.endswith(".scprj"):
                        if self._chiedi_salvataggio():
                            self._carica_da_path(path)
                        event.acceptProposedAction()
                        return True
                event.ignore()
                return True
        return super().eventFilter(obj, event)

    # ----------------------------------------------------------
    #  LOGICA PROGETTO
    # ----------------------------------------------------------

    def _nuovo_progetto(self):
        if not self._chiedi_salvataggio():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Crea nuovo progetto", "NuovoProgetto.scprj",
            "SectionCHECK Project (*.scprj)"
        )
        if not path:
            return
        nome = os.path.splitext(os.path.basename(path))[0]
        dati = nuovo_progetto_template(nome)
        if self._scrivi_json(path, dati):
            self._imposta_progetto(path, dati)
            print(f">> Nuovo progetto: {path}")

    def _salva_progetto(self):
        if self._progetto_path is None:
            self._nuovo_progetto()
            return
        self._progetto_dati["metadata"]["data_modifica"] = \
            datetime.now().isoformat(timespec="seconds")
        if self._scrivi_json(self._progetto_path, self._progetto_dati):
            self._modificato = False
            self._aggiorna_titolo()
            print(f">> Salvato: {self._progetto_path}")

    def _carica_dialog(self):
        if not self._chiedi_salvataggio():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Apri progetto", "",
            "SectionCHECK Project (*.scprj);;Tutti i file (*)"
        )
        if path:
            self._carica_da_path(path)

    def _carica_da_path(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                dati = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            QMessageBox.critical(self, "Errore", f"Impossibile aprire il file:\n{e}")
            print(f"ERR  {e}")
            return
        if "metadata" not in dati:
            QMessageBox.critical(
                self, "Errore",
                "File non riconosciuto come progetto SectionCHECK."
            )
            return
        self._imposta_progetto(path, dati)
        print(f">> Caricato: {path}")

    # ----------------------------------------------------------
    #  STATO INTERNO
    # ----------------------------------------------------------

    def _imposta_progetto(self, path: str, dati: dict):
        self._progetto_path = path
        self._progetto_dati = dati
        self._modificato    = False
        self._undo_stacks.clear()
        self._redo_stacks.clear()
        self._aggiorna_titolo()
        self._ricarica_sottomoduli()

    def _aggiorna_titolo(self):
        if self._progetto_path:
            # Estrae il nome del file (senza estensione)
            nome = os.path.splitext(os.path.basename(self._progetto_path))[0]
            # Gestisce l'asterisco se il file non è salvato
            indicatore_modifica = " *" if self._modificato else ""
            # 1. Aggiorna il titolo della finestra
            self.setWindowTitle(f"SectionCHECK – {nome}{indicatore_modifica}")
            # 2. AGGIORNA LA TUA LABEL (nome file + asterisco se modificato)
            self.ui.label_file_name.setText(f"{nome}{indicatore_modifica}")
        else:
            # Se non c'è nessun progetto aperto
            self.setWindowTitle("SectionCHECK")
            # Svuota o imposta un testo di default per la label
            self.ui.label_file_name.setText("Nessun progetto")

    # ----------------------------------------------------------
    #  API PER I SOTTO-MODULI
    # ----------------------------------------------------------

    def ha_progetto(self) -> bool:
        """True se esiste un progetto attivo in memoria."""
        return self._progetto_dati is not None

    def segna_modificato(self):
        """Chiamata dai sotto-moduli quando l'utente cambia un dato."""
        self._modificato = True
        self._aggiorna_titolo()

    def get_sezione(self, sezione: str) -> dict:
        """
        Lettura sicura di una sezione del progetto.
        Ritorna {} (non connesso) se nessun progetto e' aperto.
        """
        if self._progetto_dati is None:
            return {}
        return self._progetto_dati.setdefault(sezione, {})

    def set_sezione(self, sezione: str, dati: dict):
        """
        Scrittura sicura di una sezione del progetto.
        Marca automaticamente il progetto come modificato.
        Non crea snapshot undo: chiamare push_undo() prima se necessario.
        """
        if self._progetto_dati is None:
            print(f"WARN  Nessun progetto aperto – sezione '{sezione}' non scritta.")
            return
        self._progetto_dati[sezione] = dati
        self.segna_modificato()

    def _modulo_corrente(self) -> str | None:
        """Restituisce il nome del modulo attivo in base alla pagina visibile."""
        idx = self.ui.stackedWidget_main.currentIndex()
        return self._PAGINA_MODULO.get(idx)

    def push_undo(self, label: str = "", modulo: str | None = None):
        """
        Salva uno snapshot parziale (solo le chiavi del modulo) nello
        stack undo del modulo.  Va chiamato DAI SOTTOMODULI immediatamente
        PRIMA di modificare _progetto_dati, così lo snapshot cattura lo
        stato precedente alla modifica.

        *modulo*: scope esplicito (es. "sezione", "elemento_3d", "extra_elemento"…).
                  Se None, viene dedotto dalla pagina corrente.
        """
        if self._progetto_dati is None:
            return
        if modulo is None:
            modulo = self._modulo_corrente()
        if modulo is None:
            return

        chiavi = self._MODULO_CHIAVI.get(modulo, [])
        snapshot = {}
        for k in chiavi:
            if k in self._progetto_dati:
                snapshot[k] = copy.deepcopy(self._progetto_dati[k])

        stack = self._undo_stacks.setdefault(modulo, deque(maxlen=20))
        stack.append({
            "dati":    snapshot,
            "label":   label,
            "context": self._get_undo_context(),
        })
        self._redo_stacks.setdefault(modulo, deque(maxlen=20)).clear()

    # ----------------------------------------------------------
    #  UNDO / REDO  (per modulo – nessun cambio pagina)
    # ----------------------------------------------------------

    def _get_undo_context(self) -> dict:
        """Cattura quale pannello/elemento è aperto in questo momento."""
        ctx: dict = {"materiale": None, "sezione": None, "elemento_id": None,
                     "struttura": None,
                     "pagina": self.ui.stackedWidget_main.currentIndex()}
        if hasattr(self, "_materiali") and self._materiali._cat_corrente:
            ctx["materiale"] = (self._materiali._cat_corrente,
                                self._materiali._nome_corrente)
        if hasattr(self, "_sezioni") and self._sezioni._cat_corrente:
            ctx["sezione"] = (self._sezioni._cat_corrente,
                              self._sezioni._nome_corrente)
        if hasattr(self, "_elementi"):
            el = self._elementi._elem_ctrl.get_elem_corrente()
            if el is not None:
                ctx["elemento_id"] = el.id
            el_extra = self._elementi._extra_ctrl._elem_rif
            if el_extra is not None:
                ctx["extra_elemento_id"] = el_extra.id
        if hasattr(self, "_struttura") and self._struttura._cat_corrente:
            ctx["struttura"] = (self._struttura._cat_corrente,
                                self._struttura._nome_corrente)
        return ctx

    def _ripristina_contesto_locale(self, modulo: str, ctx: dict):
        """Dopo undo/redo: ripristina il contesto SOLO all'interno dello
        scope corrente, senza mai cambiare pagina cross-modulo."""
        if not ctx:
            return
        # Dettaglio sezione → riapri la sezione che era aperta
        if modulo == "sezione" and ctx.get("sezione") and hasattr(self, "_sezioni"):
            cat, nome = ctx["sezione"]
            self._sezioni.ripristina_contesto(cat, nome)
        # Dettaglio materiale → riapri il materiale che era aperto
        elif modulo == "materiale" and ctx.get("materiale") and hasattr(self, "_materiali"):
            cat, nome = ctx["materiale"]
            self._materiali.ripristina_contesto(cat, nome)
        # Spazio 3D elemento → riapri l'elemento che era in editing
        elif modulo == "elemento_3d" and hasattr(self, "_elementi"):
            eid = ctx.get("elemento_id")
            self._elementi.ripristina_contesto(eid, 6)
        # Spazio carichi/vincoli → riapri l'elemento nel workspace C/V
        elif modulo == "extra_elemento" and hasattr(self, "_elementi"):
            eid = ctx.get("extra_elemento_id") or ctx.get("elemento_id")
            self._elementi.ripristina_contesto_extra(eid)
        # Struttura → riapri la struttura che era in editing
        elif modulo in ("struttura", "lista_strutture") \
                and ctx.get("struttura") and hasattr(self, "_struttura"):
            cat, nome = ctx["struttura"]
            self._struttura.ripristina_contesto(cat, nome)

    def _ricarica_sottomoduli(self):
        """Ricarica TUTTI i sotto-moduli (usato al caricamento progetto)."""
        if hasattr(self, "_materiali"):
            self._materiali.ricarica_da_progetto()
        if hasattr(self, "_sezioni"):
            self._sezioni.ricarica_da_progetto()
        if hasattr(self, "_elementi"):
            self._elementi.ricarica_da_progetto()
        if hasattr(self, "_struttura"):
            self._struttura.ricarica_da_progetto()
        if hasattr(self, "_pressoflessione"):
            self._pressoflessione.ricarica_da_progetto()
        if hasattr(self, "_dominio_nm"):
            self._dominio_nm.ricarica_da_progetto()
        if hasattr(self, "_momentocurvatura"):
            self._momentocurvatura.ricarica_da_progetto()
        if hasattr(self, "_fem_elemento"):
            self._fem_elemento.ricarica_da_progetto()
        if hasattr(self, "_fem_struttura"):
            self._fem_struttura.ricarica_da_progetto()

    def _ricarica_modulo(self, modulo: str):
        """Ricarica SOLO il modulo interessato dallo scope (usato dopo undo/redo)."""
        if modulo in ("lista_materiali", "materiale") and hasattr(self, "_materiali"):
            self._materiali.ricarica_da_progetto()
        elif modulo in ("lista_sezioni", "sezione") and hasattr(self, "_sezioni"):
            self._sezioni.ricarica_da_progetto()
        elif modulo in ("lista_elementi", "elemento_3d", "extra_elemento") \
                and hasattr(self, "_elementi"):
            self._elementi.ricarica_da_progetto()
        elif modulo in ("lista_strutture", "struttura") \
                and hasattr(self, "_struttura"):
            self._struttura.ricarica_da_progetto()

    def _undo(self):
        # Se un widget di testo ha il focus, delega al suo undo nativo
        w = QApplication.focusWidget()
        if isinstance(w, QLineEdit):
            w.undo(); return
        if isinstance(w, (QTextEdit, QPlainTextEdit)):
            w.undo(); return

        modulo = self._modulo_corrente()
        if not modulo or self._progetto_dati is None:
            return
        stack = self._undo_stacks.get(modulo)
        if not stack:
            return

        snap = stack.pop()
        # Salva stato corrente per redo (solo chiavi del modulo)
        chiavi = self._MODULO_CHIAVI.get(modulo, [])
        corrente = {}
        for k in chiavi:
            if k in self._progetto_dati:
                corrente[k] = copy.deepcopy(self._progetto_dati[k])
        redo = self._redo_stacks.setdefault(modulo, deque(maxlen=20))
        redo.append({
            "dati":    corrente,
            "label":   snap["label"],
            "context": self._get_undo_context(),
        })

        # Ripristina solo le chiavi del modulo
        for k, v in snap["dati"].items():
            self._progetto_dati[k] = v

        self._modificato = True
        self._aggiorna_titolo()
        self._ricarica_modulo(modulo)
        self._ripristina_contesto_locale(modulo, snap.get("context", {}))
        print(f">> Undo [{modulo}]: {snap.get('label','–')} ({len(stack)} rimasti)")

    def _redo(self):
        # Se un widget di testo ha il focus, delega al suo redo nativo
        w = QApplication.focusWidget()
        if isinstance(w, QLineEdit):
            w.redo(); return
        if isinstance(w, (QTextEdit, QPlainTextEdit)):
            w.redo(); return

        modulo = self._modulo_corrente()
        if not modulo or self._progetto_dati is None:
            return
        stack = self._redo_stacks.get(modulo)
        if not stack:
            return

        snap = stack.pop()
        # Salva stato corrente per undo (solo chiavi del modulo)
        chiavi = self._MODULO_CHIAVI.get(modulo, [])
        corrente = {}
        for k in chiavi:
            if k in self._progetto_dati:
                corrente[k] = copy.deepcopy(self._progetto_dati[k])
        undo = self._undo_stacks.setdefault(modulo, deque(maxlen=20))
        undo.append({
            "dati":    corrente,
            "label":   snap["label"],
            "context": self._get_undo_context(),
        })

        # Ripristina solo le chiavi del modulo
        for k, v in snap["dati"].items():
            self._progetto_dati[k] = v

        self._modificato = True
        self._aggiorna_titolo()
        self._ricarica_modulo(modulo)
        self._ripristina_contesto_locale(modulo, snap.get("context", {}))
        print(f">> Redo [{modulo}]: {snap.get('label','–')} ({len(stack)} rimasti)")

    # ----------------------------------------------------------
    #  UTILITA'
    # ----------------------------------------------------------

    def _chiedi_salvataggio(self) -> bool:
        if not self._modificato:
            return True
        nome = os.path.basename(self._progetto_path or "Progetto senza nome")
        r = QMessageBox.question(
            self, "Modifiche non salvate",
            f"Il progetto '{nome}' ha modifiche non salvate.\n"
            "Vuoi salvare prima di continuare?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save
        )
        if r == QMessageBox.Save:
            self._salva_progetto()
            return True
        return r == QMessageBox.Discard

    @staticmethod
    def _scrivi_json(path: str, dati: dict) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dati, f, indent=4, ensure_ascii=False)
            return True
        except OSError as e:
            print(f"ERR  {e}")
            return False

    def _salva_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salva Screenshot", "Screenshot_SectionCHECK.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        if path:
            screen = QApplication.primaryScreen()
            if screen and screen.grabWindow(self.winId()).save(path):
                print(f">> Screenshot: {path}")
            else:
                print("ERR  Screenshot fallito.")

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self._chiedi_salvataggio():
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            event.accept()
        else:
            event.ignore()


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QToolTip {
            background-color: rgb(50, 50, 50);
            border: 1px solid rgb(120,120,120);
            border-radius:6px
        }""")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())