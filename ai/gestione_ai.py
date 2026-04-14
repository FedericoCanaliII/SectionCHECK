# -*- coding: utf-8 -*-
"""
gestione_ai.py
--------------
Controller della finestra AI Agent di SectionCHECK.
Si aggancia direttamente all'interfaccia generata in ai_interfaccia.py (Ui_btn_help).

Responsabilità:
  - Costruisce la chat all'interno di widget_chat
  - Rende il prompt dinamico (left + textEdit + right crescono assieme, max ~10 righe)
  - Imposta il placeholder nel textEdit_prompt
  - ai_btn_visibile  → toggle blur/chiaro sulla chiave API
  - ai_btn_help      → apre la finestrella di guida modelli/istruzioni
  - ai_btn_invia     → invia il messaggio (anche Enter senza Shift)
  - Loop agentico con tool_call embedded nel testo del modello
"""

import json
import os
import re

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QSize, QSettings
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QDialog, QScrollArea, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QLineEdit, QApplication,
)

_SETTINGS_ORG = "SectionCHECK"
_SETTINGS_APP = "AIAgent"

from .ai_interfaccia import Ui_btn_help
from .ai_worker import AIWorker
from .ai_strumenti import AIStrumenti, TOOLS_SCHEMA


# ── percorso icone ────────────────────────────────────────────────────
_ICO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "interfaccia", "icone",
)

def _ico(nome: str) -> QIcon:
    p = os.path.join(_ICO_DIR, nome)
    return QIcon(p) if os.path.exists(p) else QIcon()


# ================================================================
#  SYSTEM PROMPT
# ================================================================

def _build_system_prompt() -> str:
    righe = []
    for t in TOOLS_SCHEMA:
        p_str = ""
        if t["params"]:
            p_str = "\n" + "\n".join(
                f"      • {k}: {v}" for k, v in t["params"].items()
            )
        righe.append(f"  **{t['name']}** – {t['description']}{p_str}")
    tools_text = "\n\n".join(righe)

    return (
        "Sei SectionCHECK_Agent, assistente AI per la modellazione e verifica strutturale.\n\n"

        "## Capacità\n"
        "- **Materiali**: elenca, consulta, crea/modifica/elimina materiali personalizzati (σ-ε a tratti).\n"
        "- **Sezioni**: elenca, consulta, crea sezioni con carpenteria, armatura, staffe, fori.\n"
        "- **Elementi 3D**: elenca, consulta, crea elementi strutturali 3D (travi, pilastri, fondazioni, solai) "
          "con oggetti geometrici (parallelepipedo, cilindro, sfera) e armatura 3D (barre, staffe).\n"
        "- **Carichi/Vincoli**: aggiungi, modifica, elimina carichi e vincoli sugli elementi 3D. "
          "Ogni carico/vincolo è un parallelepipedo: i nodi mesh al suo interno ricevono forze/cedimenti.\n"
        "- **Manipolazione vertici**: ispeziona e modifica i vertici di qualsiasi oggetto 3D o carico/vincolo "
          "per alterare la forma (es. trasformare un parallelepipedo in tronco di piramide).\n"
        "- **Consulenza**: EC2, NTC2018, c.a., acciaio, pressoflessione, dominio N-M.\n\n"

        "## Strumenti – formato chiamata\n\n"
        "```tool_call\n"
        "{\"tool\": \"nome_strumento\", \"params\": {...}}\n"
        "```\n"
        "Più blocchi tool_call nella stessa risposta sono ammessi. "
        "Scrivi solo l'essenziale prima di ogni blocco. "
        "NON descrivere a testo ciò che stai disegnando: esegui direttamente con tool_call.\n\n"

        "## Strumenti disponibili\n\n"
        f"{tools_text}\n\n"

        "## Regole per il disegno di sezioni (unità: mm)\n"
        "- Coordinate in mm. Origine convenzionale al baricentro geometrico o all'angolo inferiore-sinistro.\n"
        "- **Cerchio/Ellisse**: usa `aggiungi_cerchio_carpenteria` con `r` (cerchio) o `rx`+`ry` (ellisse).\n"
        "- **Poligono**: usa `aggiungi_poligono` per carpenteria a forma libera (≥3 vertici).\n"
        "- **Fori**: usa `aggiungi_foro_rettangolo`, `aggiungi_foro_cerchio`, `aggiungi_foro_poligono`. "
          "I fori sottraggono area dalla carpenteria; devono essere interni ad essa.\n"
        "- **Staffe sezione**: DEVONO essere chiuse (l'ultimo punto = il primo). "
          "In ogni vertice interno della staffa va posizionata una barra longitudinale. "
          "I punti della staffa devono essere leggermente all'esterno delle barre (offset ≈ r_staffa): "
          "se la barra ha centro (cx,cy) e raggio rb, il vertice staffa sta a distanza ≈ rb + r_staffa dal centro barra.\n"
        "- **Ordine sezione**: (1) sezione vuota, (2) carpenteria/fori, (3) staffe, (4) barre.\n"
        "- Formule σ-ε: variabile 'x'.\n\n"

        "## Regole per il disegno di elementi 3D (unità: m)\n"
        "- Tutte le coordinate e dimensioni in **metri [m]**.\n"
        "- Sistema di riferimento locale: X = asse longitudinale (lunghezza), Y = larghezza (base), Z = altezza. "
          "Eccezione pilastro: asse longitudinale = Z (verticale).\n"
        "- **Ordine creazione**: (1) crea_elemento (vuoto), (2) aggiungi parallelepipedo (carpenteria), "
          "(3) aggiungi staffe 3D, (4) aggiungi barre longitudinali.\n"
        "- **Staffa 3D**: polyline chiusa (punti[0] == punti[-1]) nel piano della sezione trasversale, "
          "es. [[x, y0, z0],[x, y1, z0],[x, y1, z1],[x, y0, z1],[x, y0, z0]] dove x è la posizione sull'asse.\n"
        "- **Barra 3D**: polyline da [0, cy, cz] a [L, cy, cz] (trave/fondazione/solaio) "
          "o da [cx, cy, 0] a [cx, cy, H] (pilastro).\n"
        "- Centro barra Φ16: copriferro(0.030) + r_staffa(0.008) + r_barra(0.008) = 0.046 m dal lembo.\n"
        "- Centro barra Φ14: 0.030 + 0.008 + 0.007 = 0.045 m. Φ12: 0.025 + 0.006 = 0.031 m (solaio).\n"
        "- Elementi/sezioni/materiali **standard** non modificabili né eliminabili.\n"
        "- Se nessun progetto è aperto, comunicarlo subito senza eseguire strumenti.\n\n"

        "## Regole per carichi e vincoli\n"
        "- Ogni carico/vincolo è un parallelepipedo posizionato sull'elemento 3D.\n"
        "- **Vincolo**: impone cedimenti {sx, sy, sz} [m]. Valori 0.0 = bloccato in quella direzione.\n"
        "- **Carico**: applica forze {fx, fy, fz} [kN] distribuite sui nodi interni.\n"
        "- Posizionare il parallelepipedo in modo che contenga i vertici della mesh da vincolare/caricare.\n"
        "- Per vincolare un appoggio: creare un vincolo alle estremità dell'elemento (es. x=0 e x=L).\n"
        "- Per un carico distribuito: creare un carico che copra tutta la lunghezza dell'elemento.\n"
        "- La forma del parallelepipedo può essere alterata con modifica_vertici_oggetto (target='carico_vincolo').\n\n"

        "## Regole per la manipolazione dei vertici\n"
        "- Usare **get_vertici_oggetto** prima di modificare per conoscere la geometria attuale.\n"
        "- **modifica_vertici_oggetto** con 'vertici' (lista completa) per sostituzione totale, "
          "o con 'modifiche' (dict indice→[x,y,z]) per modifiche puntuali.\n"
        "- Parallelepipedo: 8 vertici – indici 0-3 faccia inferiore, 4-7 faccia superiore.\n"
        "  Ordine: [0,0,0],[L,0,0],[L,B,0],[0,B,0],[0,0,A],[L,0,A],[L,B,A],[0,B,A].\n"
        "- Per trasformare in tronco di piramide: restringere i vertici 4-7 (faccia superiore).\n"
        "- Cilindro: v0=centro basso, v1-v24=cerchio basso, v25=centro alto, v26-v49=cerchio alto.\n"
        "- Sfera: v0=centro, v1+ griglia latitudine/longitudine.\n"
        "- Barra/Staffa: vertici = punti della polyline (modificabili liberamente).\n"
        "- Il parametro 'target' distingue tra oggetti strutturali ('oggetto') e carichi/vincoli ('carico_vincolo').\n"
    )


# ================================================================
#  HELP DIALOG
# ================================================================

class _HelpDialog(QDialog):
    """
    Finestrella di guida: mix elegante con titoli in Georgia, 
    spaziature moderne e palette scura rilassante.
    """

    def __init__(self, on_select_modello=None, parent=None):
        super().__init__(parent)
        self._on_select = on_select_modello
        self.setWindowTitle("SectionCHECK Agent – Guida rapida")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self.setMinimumSize(560, 600)
        
        # Sfondo generale morbido
        self.setStyleSheet("QDialog { background-color: #242424; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 16)
        root.setSpacing(16)

        # Titolo principale - Georgia Bold
        titolo = QLabel("SectionCHECK Agent – Guida rapida")
        titolo.setStyleSheet(
            "color: #ffffff; font: 700 14pt 'Georgia'; border: none;"
        )
        root.addWidget(titolo)

        # Scroll area
        scroll = QScrollArea()
        scroll.setObjectName("help_scroll")
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea#help_scroll { border: none; background: transparent; }"
            "QScrollArea#help_scroll QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }"
            "QScrollArea#help_scroll QScrollBar::handle:vertical { background: #444444; border-radius: 3px; }"
            "QScrollArea#help_scroll QScrollBar::handle:vertical:hover { background: #666666; }"
            "QScrollArea#help_scroll QScrollBar::add-line:vertical, "
            "QScrollArea#help_scroll QScrollBar::sub-line:vertical { height: 0; }"
        )
        
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 16, 8)
        bl.setSpacing(10) # Respiro tra gli elementi

        # ── Modelli ──
        bl.addWidget(self._sep("Modelli supportati"))
        modelli = {
            "Anthropic (claude-*)": [
                "claude-opus-4-6", "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001", "claude-3-5-sonnet-20241022",
            ],
            "OpenAI (gpt-*)": [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
            ],
            "Google Gemini (gemini-*)": [
                "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash",
            ],
            "DeepSeek (deepseek-*)": [
                "deepseek-chat", "deepseek-reasoner",
            ],
        }
        
        for provider, lista in modelli.items():
            lp = QLabel(provider)
            # Provider in Georgia Regular
            lp.setStyleSheet(
                "color: #888888; font: 400 10pt 'Georgia'; "
                "padding-top: 10px; margin-bottom: -4px;"
            )
            bl.addWidget(lp)
            
            for m in lista:
                row = QWidget()
                row_lay = QHBoxLayout(row)
                row_lay.setContentsMargins(12, 0, 0, 0)
                row_lay.setSpacing(8)

                # Pallino classico testuale
                dot = QPushButton("●")
                dot.setFixedSize(18, 18)
                dot.setToolTip(f"Usa {m}")
                dot.setStyleSheet("""
                    QPushButton {
                        color: #666666; 
                        background: transparent;
                        border: none; 
                        font: 10pt 'Arial'; 
                        padding: 0;
                    }
                    QPushButton:hover {
                        color: #4da6ff;
                    }
                """)
                dot.setCursor(Qt.PointingHandCursor)
                dot.clicked.connect(lambda checked, nm=m: self._seleziona(nm))

                lbl_m = QLabel(m)
                lbl_m.setStyleSheet(
                    "color: #cccccc; font: 400 10.5pt 'Consolas'; border: none;"
                )
                
                row_lay.addWidget(dot)
                row_lay.addWidget(lbl_m)
                row_lay.addStretch()
                bl.addWidget(row)

        # ── Come iniziare ──
        bl.addSpacing(12)
        bl.addWidget(self._sep("Come avviare l'agente"))
        passi = [
            "<b>1.</b> Apri o crea un progetto (.scprj) in SectionCHECK.",
            "<b>2.</b> Clicca sul pallino ● accanto al modello (o digitalo in alto).",
            "<b>3.</b> Incolla la tua API key nell'apposito campo.",
            "<b>4.</b> Scrivi la richiesta nell'area di testo in basso.",
            "<b>5.</b> Premi Invio per inviare (Shift+Invio per andare a capo).",
            "<b>6.</b> Ogni azione svolta comparirà in chat con un marker colorato."
        ]
        for p in passi:
            l = QLabel(p)
            l.setWordWrap(True)
            # Corpo del testo mantenuto in Segoe UI per massima leggibilità
            l.setStyleSheet("color: #bbbbbb; font: 400 10pt 'Segoe UI'; line-height: 1.4;")
            bl.addWidget(l)

        # ── Capacità ──
        bl.addSpacing(12)
        bl.addWidget(self._sep("Cosa può fare l'agente"))
        caps = [
            "• Elencare e visualizzare materiali e sezioni del progetto.",
            "• Creare materiali personalizzati con legami σ-ε a tratti.",
            "• Creare sezioni con carpenteria rettangolare, circolare/ellittica e poligonale.",
            "• Aggiungere fori rettangolari, circolari/ellittici e poligonali.",
            "• Aggiungere barre di armatura e staffe chiuse.",
            "• Modificare ed eliminare materiali/sezioni personalizzati.",
            "• Fornire consulenza tecnica su EC2, NTC2018, c.a. e acciaio."
        ]
        for c in caps:
            l = QLabel(c)
            l.setWordWrap(True)
            l.setStyleSheet("color: #bbbbbb; font: 400 10pt 'Segoe UI'; line-height: 1.4;")
            bl.addWidget(l)

        bl.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # Pulsante di chiusura
        btn_ok = QPushButton("Chiudi")
        btn_ok.setCursor(Qt.PointingHandCursor)
        btn_ok.setFixedHeight(36)
        btn_ok.setStyleSheet("""
            QPushButton {
                background: #333333; 
                color: #ffffff;
                border: 1px solid #444444; 
                border-radius: 6px;
                font: 400 10pt 'Segoe UI';
            }
            QPushButton:hover {
                background: #404040;
                border: 1px solid #555555;
            }
            QPushButton:pressed {
                background: #2a2a2a;
            }
        """)
        btn_ok.clicked.connect(self.accept)
        root.addWidget(btn_ok)

    def _seleziona(self, nome_modello: str):
        if self._on_select:
            self._on_select(nome_modello)
        self.accept()

    @staticmethod
    def _sep(testo: str) -> QLabel:
        """Titoletto sezione in Georgia Regular."""
        l = QLabel(testo)
        l.setStyleSheet(
            "color: #e0e0e0; font: 400 11.5pt 'Georgia';"
            "padding-bottom: 4px; border-bottom: 1px solid #3a3a3a;"
        )
        return l

# ================================================================
#  WIDGET MESSAGGI NELLA CHAT
# ================================================================

class _MsgUtente(QWidget):
    """Messaggio utente – destra, cella scura."""
    def __init__(self, testo: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(80, 4, 14, 4)
        lbl = QLabel(testo)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        lbl.setStyleSheet(
            "background-color:rgb(52,52,52);"
            "color:rgb(230,230,230);"
            "border:1px solid rgb(75,75,75);"
            "border-radius:10px;"
            "padding:9px 14px;"
            "font:400 10pt 'Segoe UI';"
        )
        lay.addStretch()
        lay.addWidget(lbl)


class _MsgAI(QWidget):
    """Risposta AI – sinistra, nessuna cella (testo diretto sullo sfondo)."""
    def __init__(self, testo: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 4, 80, 4)
        lbl = QLabel(testo)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        lbl.setStyleSheet(
            "color:rgb(220,220,220);"
            "background:transparent;"
            "border:none;"
            "font:400 10pt 'Segoe UI';"
        )
        lay.addWidget(lbl)
        lay.addStretch()


class _MsgAzione(QWidget):
    """Riga azione/errore – colore verde (ok) o rosso (errore)."""
    def __init__(self, testo: str, ok: bool = True, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 1, 14, 1)
        colore = "rgb(95,195,115)" if ok else "rgb(210,90,70)"
        lbl = QLabel(testo)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setStyleSheet(
            f"color:{colore};background:transparent;border:none;"
            "font:400 9pt 'Consolas';"
        )
        lay.addWidget(lbl)


class _MsgStato(QWidget):
    """Riga di stato temporanea (corsivo grigio)."""
    def __init__(self, testo: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 1, 14, 1)
        self._lbl = QLabel(testo)
        self._lbl.setStyleSheet(
            "color:rgb(130,130,130);background:transparent;border:none;"
            "font:400 9pt 'Segoe UI';font-style:italic;"
        )
        lay.addWidget(self._lbl)
        lay.addStretch()

    def aggiorna(self, testo: str):
        self._lbl.setText(testo)


# ================================================================
#  GESTIONE AI  – controller principale
# ================================================================

_PROMPT_MIN_H = 64   # altezza minima (~2 righe con font 10-12pt + padding)
_PROMPT_MAX_H = 220  # altezza massima (~10 righe, poi scatta la scrollbar)


class GestioneAI(QWidget):
    """
    Finestra floating AI Agent.
    Istanziata una sola volta da MainWindow e riutilizzata con show/raise.
    """

    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        # ── UI generata ──────────────────────────────────────────
        self.ui = Ui_btn_help()
        self.ui.setupUi(self)

        # ── Stato ────────────────────────────────────────────────
        self._main           = main_window
        self._tools          = AIStrumenti(main_window)
        self._storia: list[dict] = []
        self._worker: AIWorker | None = None
        self._widget_stato: _MsgStato | None = None
        self._system_prompt  = _build_system_prompt()
        self._key_visibile   = False   # inizia con la chiave oscurata
        self._prompt_h       = _PROMPT_MIN_H

        # ── Setup ────────────────────────────────────────────────
        self._setup_finestra()
        self._setup_chat()
        self._setup_prompt_dinamico()
        self._setup_connessioni()
        self._carica_impostazioni()

    # ------------------------------------------------------------------
    #  SETUP FINESTRA
    # ------------------------------------------------------------------

    def _setup_finestra(self):
        self.setWindowTitle("SectionCHECK – AI Agent")
        self.setMinimumSize(900, 600)

        ico_path = os.path.join(_ICO_DIR, "logo.ico")
        if os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))

        # La chiave parte oscurata (Password)
        self.ui.ai_key.setEchoMode(QLineEdit.Password)

    # ------------------------------------------------------------------
    #  SETUP CHAT  (costruisce scroll area dentro widget_chat)
    # ------------------------------------------------------------------

    def _setup_chat(self):
        """Inserisce una QScrollArea con messaggi dentro widget_chat."""
        container = self.ui.widget_chat

        # Layout sul container
        c_lay = QVBoxLayout(container)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)

        # Scroll area – objectName usato per isolare il QSS dai widget fratelli
        self._scroll = QScrollArea()
        self._scroll.setObjectName("ai_chat_scroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            "QScrollArea#ai_chat_scroll{border:none;background:transparent;}"
            "QScrollArea#ai_chat_scroll QScrollBar:vertical{"
            "background:rgb(44,44,44);width:8px;margin:0;}"
            "QScrollArea#ai_chat_scroll QScrollBar::handle:vertical{"
            "background:rgb(78,78,78);min-height:30px;border-radius:4px;}"
            "QScrollArea#ai_chat_scroll QScrollBar::add-line:vertical,"
            "QScrollArea#ai_chat_scroll QScrollBar::sub-line:vertical,"
            "QScrollArea#ai_chat_scroll QScrollBar::up-arrow:vertical,"
            "QScrollArea#ai_chat_scroll QScrollBar::down-arrow:vertical,"
            "QScrollArea#ai_chat_scroll QScrollBar::add-page:vertical,"
            "QScrollArea#ai_chat_scroll QScrollBar::sub-page:vertical{"
            "background:none;height:0;}"
        )

        # Widget contenitore messaggi
        self._msg_widget = QWidget()
        self._msg_widget.setStyleSheet("background:transparent;")
        self._msg_layout = QVBoxLayout(self._msg_widget)
        self._msg_layout.setAlignment(Qt.AlignTop)
        self._msg_layout.setContentsMargins(0, 12, 0, 12)
        self._msg_layout.setSpacing(6)

        self._scroll.setWidget(self._msg_widget)
        c_lay.addWidget(self._scroll)

        # Messaggio di benvenuto
        self._aggiungi_msg_ai(
            "Ciao! Sono SectionCHECK Agent.\n"
            "Posso aiutarti a gestire materiali e sezioni del progetto corrente "
            "e rispondere a domande tecniche di ingegneria strutturale.\n"
            "Premi il pulsante ? per vedere i modelli disponibili e le istruzioni."
        )

    # ------------------------------------------------------------------
    #  SETUP PROMPT DINAMICO
    # ------------------------------------------------------------------

    def _setup_prompt_dinamico(self):
        """
        - Parte con _PROMPT_MIN_H (~2 righe)
        - Cresce fino a _PROMPT_MAX_H man mano che l'utente scrive
        """
        # Placeholder
        self.ui.textEdit_prompt.setPlaceholderText(
            "Cosa vuoi che faccia all'interno dell'applicazione?"
        )

        # Imposta l'altezza fissa iniziale (sovrascrive il layout del .ui)
        self.ui.frame_left_prompt.setFixedHeight(_PROMPT_MIN_H)
        self.ui.textEdit_prompt.setFixedHeight(_PROMPT_MIN_H)
        self.ui.frame_right_prompt.setFixedHeight(_PROMPT_MIN_H)
        self._prompt_h = _PROMPT_MIN_H

        # Applica la scrollbar custom
        self.ui.textEdit_prompt.verticalScrollBar().setStyleSheet(
            "QScrollBar:vertical{"
            "background:transparent;width:8px;margin:0px;}"
            "QScrollBar::handle:vertical{"
            "background-color:rgb(70,70,70);min-height:30px;border-radius:4px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical,"
            "QScrollBar::up-arrow:vertical,QScrollBar::down-arrow:vertical,"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{"
            "background:none;border:none;height:0px;}"
        )

        # Aggancia il ridimensionamento
        self.ui.textEdit_prompt.document().contentsChanged.connect(
            self._aggiorna_altezza_prompt
        )

        # Enter senza Shift → invia
        self.ui.textEdit_prompt.installEventFilter(self)

        # Forza un ricalcolo dell'altezza iniziale a vuoto
        self._aggiorna_altezza_prompt()

    # ------------------------------------------------------------------
    #  SETUP CONNESSIONI
    # ------------------------------------------------------------------

    def _setup_connessioni(self):
        self.ui.ai_btn_invia.clicked.connect(self._invia_messaggio)
        self.ui.ai_btn_help.clicked.connect(self._mostra_help)
        self.ui.ai_btn_visibile.clicked.connect(self._toggle_visibilita_key)

    # ------------------------------------------------------------------
    #  EVENT FILTER  (Enter nel textEdit_prompt)
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if (obj is self.ui.textEdit_prompt
                and event.type() == QtCore.QEvent.KeyPress):
            if (event.key() in (Qt.Key_Return, Qt.Key_Enter)
                    and not (event.modifiers() & Qt.ShiftModifier)):
                self._invia_messaggio()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    #  ALTEZZA PROMPT DINAMICA
    # ------------------------------------------------------------------

    def _aggiorna_altezza_prompt(self):
        # Prende l'altezza reale del testo contenuto
        doc_h = int(self.ui.textEdit_prompt.document().size().height())
        
        # Calcola la nuova altezza con un piccolo margine (+12px), limitata tra MIN e MAX
        new_h = max(_PROMPT_MIN_H, min(doc_h + 12, _PROMPT_MAX_H))
        
        # Applica la modifica solo se l'altezza è effettivamente cambiata
        if new_h != self._prompt_h:
            self._prompt_h = new_h
            self.ui.frame_left_prompt.setFixedHeight(new_h)
            self.ui.textEdit_prompt.setFixedHeight(new_h)
            self.ui.frame_right_prompt.setFixedHeight(new_h)

    # ------------------------------------------------------------------
    #  TOGGLE VISIBILITÀ CHIAVE
    # ------------------------------------------------------------------

    def _toggle_visibilita_key(self):
        self._key_visibile = not self._key_visibile
        if self._key_visibile:
            self.ui.ai_key.setEchoMode(QLineEdit.Normal)
            self.ui.ai_btn_visibile.setIcon(_ico("visibile.png"))
        else:
            self.ui.ai_key.setEchoMode(QLineEdit.Password)
            self.ui.ai_btn_visibile.setIcon(_ico("novisibile.png"))

    # ------------------------------------------------------------------
    #  PERSISTENZA IMPOSTAZIONI (modello + chiave)
    # ------------------------------------------------------------------

    def _carica_impostazioni(self):
        """Ripristina modello e chiave salvati alla chiusura precedente."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self.ui.ai_modello.setText(s.value("modello", "", type=str))
        self.ui.ai_key.setText(s.value("chiave", "", type=str))

    def _salva_impostazioni(self):
        """Persiste modello e chiave su QSettings (registro su Windows)."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue("modello", self.ui.ai_modello.text())
        s.setValue("chiave", self.ui.ai_key.text())

    def closeEvent(self, event):
        self._salva_impostazioni()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    #  AGGIUNTA MESSAGGI IN CHAT
    # ------------------------------------------------------------------

    def _aggiungi_msg_utente(self, testo: str):
        self._msg_layout.addWidget(_MsgUtente(testo))
        self._scroll_bottom()

    def _aggiungi_msg_ai(self, testo: str):
        self._msg_layout.addWidget(_MsgAI(testo))
        self._scroll_bottom()

    def _aggiungi_azione(self, testo: str, ok: bool = True):
        self._msg_layout.addWidget(_MsgAzione(f"◆ {testo}", ok=ok))
        self._scroll_bottom()

    def _aggiungi_stato(self, testo: str) -> _MsgStato:
        w = _MsgStato(testo)
        self._msg_layout.addWidget(w)
        self._scroll_bottom()
        return w

    def _rimuovi_stato(self):
        if self._widget_stato is not None:
            self._msg_layout.removeWidget(self._widget_stato)
            self._widget_stato.deleteLater()
            self._widget_stato = None

    def _scroll_bottom(self):
        QTimer.singleShot(
            60,
            lambda: self._scroll.verticalScrollBar().setValue(
                self._scroll.verticalScrollBar().maximum()
            ),
        )

    # ------------------------------------------------------------------
    #  INVIO MESSAGGIO E LOOP AGENTE
    # ------------------------------------------------------------------

    def _invia_messaggio(self):
        modello = self.ui.ai_modello.text().strip()
        key     = self.ui.ai_key.text().strip()
        testo   = self.ui.textEdit_prompt.toPlainText().strip()

        if not testo:
            return
        if not modello:
            self._aggiungi_azione("Specifica il modello nel campo 'Modello'.", ok=False)
            return
        if not key:
            self._aggiungi_azione("Inserisci la API key nel campo 'Key'.", ok=False)
            return

        self.ui.textEdit_prompt.clear()
        self._aggiungi_msg_utente(testo)
        self._storia.append({"role": "user", "content": testo})
        self._set_abilitato(False)
        self._avvia_worker()

    def _avvia_worker(self):
        modello = self.ui.ai_modello.text().strip()
        key     = self.ui.ai_key.text().strip()

        self._widget_stato = self._aggiungi_stato("In elaborazione…")

        worker = AIWorker(
            modello, key,
            list(self._storia),
            self._system_prompt,
        )
        worker.risposta_ricevuta.connect(self._on_risposta)
        worker.errore_ricevuto.connect(self._on_errore)
        worker.stato_aggiornato.connect(self._on_stato)
        self._worker = worker
        worker.start()

    # ── slot worker ──────────────────────────────────────────────

    def _on_stato(self, testo: str):
        if self._widget_stato:
            self._widget_stato.aggiorna(testo)

    def _on_risposta(self, risposta: str):
        self._rimuovi_stato()

        tool_calls   = self._estrai_tool_calls(risposta)
        testo_pulito = self._rimuovi_tool_calls(risposta).strip()

        # Mostra la parte testuale (se presente)
        if testo_pulito:
            self._aggiungi_msg_ai(testo_pulito)

        if tool_calls:
            # Aggiungi risposta completa alla storia
            self._storia.append({"role": "assistant", "content": risposta})

            # Esegui gli strumenti
            risultati = []
            for call in tool_calls:
                nome   = call.get("tool", "")
                params = call.get("params", {})
                ok, msg = self._tools.esegui(nome, params)
                self._aggiungi_azione(f"[{nome}] {msg}", ok=ok)
                stato_str = "SUCCESSO" if ok else "ERRORE"
                risultati.append(f"Tool '{nome}': {stato_str} – {msg}")

            # Rimanda i risultati al modello per la risposta finale
            self._storia.append({
                "role":    "user",
                "content": "[Risultati strumenti]\n" + "\n".join(risultati),
            })
            # Continua il loop
            self._avvia_worker()

        else:
            # Risposta finale, nessun tool call
            self._storia.append({"role": "assistant", "content": risposta})
            self._set_abilitato(True)

    def _on_errore(self, errore: str):
        self._rimuovi_stato()
        self._aggiungi_azione(errore, ok=False)
        self._set_abilitato(True)

    def _set_abilitato(self, val: bool):
        self.ui.textEdit_prompt.setEnabled(val)
        self.ui.ai_btn_invia.setEnabled(val)

    # ------------------------------------------------------------------
    #  PARSING TOOL CALLS
    # ------------------------------------------------------------------

    @staticmethod
    def _estrai_tool_calls(testo: str) -> list[dict]:
        calls = []
        for m in re.finditer(r"```tool_call\s*\n(.*?)\n?```", testo, re.DOTALL):
            raw = m.group(1).strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "tool" in obj:
                    obj.setdefault("params", {})
                    calls.append(obj)
            except json.JSONDecodeError:
                pass
        return calls

    @staticmethod
    def _rimuovi_tool_calls(testo: str) -> str:
        return re.sub(
            r"```tool_call\s*\n.*?\n?```", "", testo, flags=re.DOTALL
        ).strip()

    # ------------------------------------------------------------------
    #  HELP
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  HELP
    # ------------------------------------------------------------------

    def _mostra_help(self):
        # Creiamo una piccola funzione lambda che aggiorna la QLineEdit del modello.
        # Passiamo la lambda come callback e 'self' come parent.
        _HelpDialog(
            on_select_modello=lambda nome: self.ui.ai_modello.setText(nome), 
            parent=self
        ).exec_()
