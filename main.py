import sys
import json
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QVBoxLayout, QTextEdit, QFileDialog, QMessageBox
)

from interfaccia.main_interfaccia import Ui_MainWindow
from sezione.gestione_sezione import GestioneSezioni 
from output.gestione_output import GestioneOutput
from momentocurvatura.gestione_momentocurvatura import GestioneMomentocurvatura
from beam.gestione_beam import GestioneBeam
from telaio.gestione_telaio import GestioneTelaio
from pressoflessione.gestione_pressoflessione import GestionePressoflessione

# --- CLASSE PER CATTURARE L'OUTPUT DEL TERMINALE ---
class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


# --- QPUSHBUTTON CON SUPPORTO DRAG & DROP ---
class DropButton(QtWidgets.QPushButton):
    """
    QPushButton che accetta il trascinamento di file .scprj / .json sopra di sé.
    Emette il segnale fileDropped(str) con il percorso del file trascinato.
    Al click normale funziona come un pulsante standard (connesso esternamente).
    """
    fileDropped = QtCore.pyqtSignal(str)

    # Estensioni accettate
    ACCEPTED_EXTENSIONS = ('.scprj', '.json')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    # ---- Drag events ----

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if self._has_valid_file(event.mimeData()):
            # Feedback visivo: bordo evidenziato mentre il file è sopra il tasto
            self.setStyleSheet(self.styleSheet() + "border: 2px dashed #00ff00;")
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent):
        # Rimuove il feedback visivo quando il file lascia il bottone
        self._reset_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        self._reset_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(self.ACCEPTED_EXTENSIONS):
                self.fileDropped.emit(file_path)
                event.acceptProposedAction()
                return
        event.ignore()

    # ---- Helpers ----

    def _has_valid_file(self, mime: QtCore.QMimeData) -> bool:
        if not mime.hasUrls():
            return False
        return any(
            url.toLocalFile().lower().endswith(self.ACCEPTED_EXTENSIONS)
            for url in mime.urls()
        )

    def _reset_style(self):
        # Rimuove solo il bordo aggiunto durante il drag, mantenendo lo stile base
        current = self.styleSheet()
        cleaned = current.replace("border: 2px dashed #00ff00;", "").strip()
        self.setStyleSheet(cleaned)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self): 
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # LOGO APP
        self.setWindowTitle("SectionCHECK")
        icon_path = "C:\\Users\\canal\\Desktop\\SC\\interfaccia\\icone\\logo.ico"
        self.setWindowIcon(QIcon(icon_path))

        # SCHERMO INTERO
        self.showMaximized()

        # ---------------------------------------------------------
        # INTEGRAZIONE TERMINALE IN widget_terminale
        # ---------------------------------------------------------

        if self.ui.widget_terminale.layout() is None:
            self.terminal_layout = QVBoxLayout(self.ui.widget_terminale)
            self.terminal_layout.setContentsMargins(0, 0, 0, 0)
        else:
            self.terminal_layout = self.ui.widget_terminale.layout()

        if self.ui.widget_terminale_2.layout() is None:
            self.terminal_layout_2 = QVBoxLayout(self.ui.widget_terminale_2)
            self.terminal_layout_2.setContentsMargins(0, 0, 0, 0)
        else:
            self.terminal_layout_2 = self.ui.widget_terminale_2.layout()

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)

        self.console_output_2 = QTextEdit()
        self.console_output_2.setReadOnly(True)

        style_sheet_terminal = """
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: none;
            }
        """
        self.console_output.setStyleSheet(style_sheet_terminal)
        self.console_output_2.setStyleSheet(style_sheet_terminal)

        self.terminal_layout.addWidget(self.console_output)
        self.terminal_layout_2.addWidget(self.console_output_2)

        self.sys_stdout_backup = sys.stdout
        self.sys_stderr_backup = sys.stderr

        self.output_stream = EmittingStream()
        self.output_stream.textWritten.connect(self.append_terminal_text)

        sys.stdout = self.output_stream
        sys.stderr = self.output_stream

        print(">> Terminale Inizializzato correttamente...")
        print(">> Benvenuto in SectionCHECK")

        # ---------------------------------------------------------
        # SETUP BOTTONE DROP (deve avvenire PRIMA dei connect)
        # ---------------------------------------------------------
        # Sostituisce il btn_main_drop esistente nel layout con la versione
        # DropButton che supporta il drag & drop.
        # Se nel .ui il bottone è già presente come QPushButton standard,
        # lo "upgrade" in runtime tramite il metodo _upgrade_drop_button.
        self._upgrade_drop_button()

        # ---------------------------------------------------------
        # PULSANTI AUTOMATICI
        # ---------------------------------------------------------
        QtCore.QTimer.singleShot(0, self.ui.btn_main_sc.click)

        # GESTIONE DELLE PAGINE
        self.ui.btn_main_sc.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(0))
        self.ui.btn_main_materiali.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(1))
        self.ui.btn_main_input.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(2))
        self.ui.btn_main_output.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(3))
        self.ui.btn_main_momentocurvatura.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(4))
        self.ui.btn_main_beam.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(5))
        self.ui.btn_main_struttura.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(6))
        self.ui.btn_main_pressoflessione.clicked.connect(lambda: self.ui.stackedWidget_main.setCurrentIndex(7))

        # BTN GRUPPI MAIN
        btn_group_main = QButtonGroup(self)
        btn_group_main.addButton(self.ui.btn_main_sc)
        btn_group_main.addButton(self.ui.btn_main_materiali)
        btn_group_main.addButton(self.ui.btn_main_input)
        btn_group_main.addButton(self.ui.btn_main_output)
        btn_group_main.addButton(self.ui.btn_main_momentocurvatura)
        btn_group_main.addButton(self.ui.btn_main_beam)
        btn_group_main.addButton(self.ui.btn_main_struttura)
        btn_group_main.addButton(self.ui.btn_main_pressoflessione)
        btn_group_main.setExclusive(True)

        self.ui.btn_main_sc.setCheckable(True)
        self.ui.btn_main_materiali.setCheckable(True)
        self.ui.btn_main_input.setCheckable(True)
        self.ui.btn_main_output.setCheckable(True)
        self.ui.btn_main_momentocurvatura.setCheckable(True)
        self.ui.btn_main_beam.setCheckable(True)
        self.ui.btn_main_struttura.setCheckable(True)
        self.ui.btn_main_pressoflessione.setCheckable(True)

        # GESTIONE SEZIONI
        self.sezioni = GestioneSezioni(self.ui)
        self.output = GestioneOutput(self.ui, self.sezioni, self.sezioni.gestione_materiali)
        self.momentocurvatura = GestioneMomentocurvatura(self.ui, self.sezioni, self.sezioni.gestione_materiali)
        self.beam = GestioneBeam(self, self.ui, self.sezioni, self.sezioni.gestione_materiali)
        self.telaio = GestioneTelaio(self.ui)
        self.pressoflessione = GestionePressoflessione(self.ui, self.sezioni, self.sezioni.gestione_materiali)

        # TOOLTIP
        self.ui.btn_main_sc.setToolTip("SectionCHECK")
        self.ui.btn_main_lingua.setToolTip("Lingua")
        self.ui.btn_main_colore.setToolTip("Tema")
        self.ui.btn_main_salva.setToolTip("Salva progetto")
        self.ui.btn_main_carica.setToolTip("Carica progetto")
        self.ui.btn_main_stampa.setToolTip("Stampa")
        self.ui.btn_main_crea.setToolTip("Nuovo progetto")
        self.ui.btn_main_sfoglia.setToolTip("Apri progetto esistente")
        self.ui.btn_main_drop.setToolTip("Trascina un file .scprj qui, oppure clicca per sfogliare")

        self.ui.progressBar_verifica.setValue(0)
        self.ui.progressBar_verifica_MC.setValue(0)
        self.ui.progressBar_telaio.setValue(0)

        # COLLEGAMENTO TASTO STAMPA
        self.ui.btn_main_stampa.clicked.connect(self.salva_screenshot)

        # ==========================================================
        # SALVATAGGIO / CARICAMENTO PROGETTO GLOBALE
        # ==========================================================
        self.ui.btn_main_salva.clicked.connect(self._salva_progetto)
        self.ui.btn_main_carica.clicked.connect(self._carica_progetto)

        # --- NUOVI PULSANTI ---
        # Crea nuovo progetto da zero
        self.ui.btn_main_crea.clicked.connect(self._nuovo_progetto)

        # Sfoglia / Importa file esistente (identico a btn_main_carica)
        self.ui.btn_main_sfoglia.clicked.connect(self._carica_progetto)

        # Drop: click → sfoglia, drag → apre il file trascinato
        self.ui.btn_main_drop.clicked.connect(self._carica_progetto)
        self.ui.btn_main_drop.fileDropped.connect(self._carica_progetto_da_path)

    # ------------------------------------------------------------------
    # UPGRADE btn_main_drop → DropButton (drag & drop abilitato)
    # ------------------------------------------------------------------

    def _upgrade_drop_button(self):
        """
        Sostituisce il QPushButton standard btn_main_drop con una istanza
        di DropButton (sottoclasse con supporto drag & drop) mantenendo
        posizione, dimensione, testo e stile originali.
        """
        old_btn: QtWidgets.QPushButton = self.ui.btn_main_drop
        parent = old_btn.parent()
        layout = old_btn.parent().layout() if old_btn.parent() else None

        # Crea il nuovo bottone con le stesse proprietà
        new_btn = DropButton(parent)
        new_btn.setObjectName("btn_main_drop")
        new_btn.setText(old_btn.text())
        new_btn.setIcon(old_btn.icon())
        new_btn.setIconSize(old_btn.iconSize())
        new_btn.setToolTip(old_btn.toolTip())
        new_btn.setStyleSheet(old_btn.styleSheet())
        new_btn.setFixedSize(old_btn.size())
        new_btn.setGeometry(old_btn.geometry())
        new_btn.setFont(old_btn.font())

        # Sostituisce nel layout (se esiste)
        if layout is not None:
            idx = layout.indexOf(old_btn)
            if idx != -1:
                layout.removeWidget(old_btn)
                old_btn.hide()
                old_btn.deleteLater()
                layout.insertWidget(idx, new_btn)
            else:
                old_btn.hide()
                old_btn.deleteLater()
                new_btn.setParent(parent)
                new_btn.setGeometry(new_btn.geometry())
                new_btn.show()
        else:
            # Nessun layout: posizionamento assoluto
            old_btn.hide()
            old_btn.deleteLater()
            new_btn.setParent(parent)
            new_btn.show()

        # Aggiorna il riferimento nell'ui
        self.ui.btn_main_drop = new_btn

    # ------------------------------------------------------------------
    # NUOVO PROGETTO
    # ------------------------------------------------------------------

    def _nuovo_progetto(self):
        """
        Chiede conferma e resetta tutti i moduli allo stato iniziale,
        come se si aprisse l'applicazione per la prima volta.
        """
        risposta = QMessageBox.question(
            self,
            "Nuovo progetto",
            "Vuoi creare un nuovo progetto?\nTutti i dati non salvati andranno persi.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if risposta != QMessageBox.Yes:
            return

        errori = []

        # --- Reset Materiali ---
        try:
            self.sezioni.gestione_materiali.carica_dati({})
        except Exception as e:
            errori.append(f"Materiali: {e}")

        # --- Reset Sezioni ---
        try:
            self.sezioni.carica_dati({})
        except Exception as e:
            errori.append(f"Sezioni: {e}")

        # --- Reset Telaio ---
        try:
            self.telaio.carica_dati({})
        except Exception as e:
            errori.append(f"Telaio: {e}")

        self.ui.progressBar_verifica.setValue(0)
        self.ui.progressBar_verifica_MC.setValue(0)
        self.ui.progressBar_telaio.setValue(0)

        if errori:
            msg = "\n".join(errori)
            print(f">> Avvisi durante il reset:\n{msg}")
            QMessageBox.warning(self, "Nuovo progetto con avvisi",
                                f"Progetto resettato con alcuni avvisi:\n{msg}")
        else:
            print(">> Nuovo progetto creato.")
            QMessageBox.information(self, "Nuovo progetto", "Progetto resettato correttamente.")

    # ------------------------------------------------------------------
    # SALVATAGGIO PROGETTO GLOBALE
    # ------------------------------------------------------------------

    def _salva_progetto(self):
        """Raccoglie i dati da tutti i moduli e li salva in un unico file .scprj."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salva progetto SectionCHECK",
            "",
            "Progetto SectionCHECK (*.scprj);;JSON (*.json);;Tutti (*)"
        )
        if not path:
            return

        dati_progetto = {}

        try:
            dati_progetto['materiali'] = self.sezioni.gestione_materiali.get_dati_salvataggio()
        except Exception as e:
            print(f">> Avviso: impossibile salvare materiali: {e}")
            dati_progetto['materiali'] = {}

        try:
            dati_progetto['sezioni'] = self.sezioni.get_dati_salvataggio()
        except Exception as e:
            print(f">> Avviso: impossibile salvare sezioni: {e}")
            dati_progetto['sezioni'] = {}

        try:
            dati_progetto['telaio'] = self.telaio.get_dati_salvataggio()
        except Exception as e:
            print(f">> Avviso: impossibile salvare telaio: {e}")
            dati_progetto['telaio'] = {}

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dati_progetto, f, indent=2, ensure_ascii=False)
            print(f">> Progetto salvato: {path}")
            QMessageBox.information(self, "Salvato", f"Progetto salvato:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile salvare il progetto:\n{e}")

    # ------------------------------------------------------------------
    # CARICAMENTO PROGETTO (con dialogo file)
    # ------------------------------------------------------------------

    def _carica_progetto(self):
        """Apre il dialogo di selezione file e carica il progetto scelto."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Carica progetto SectionCHECK",
            "",
            "Progetto SectionCHECK (*.scprj);;JSON (*.json);;Tutti (*)"
        )
        if not path:
            return
        self._carica_progetto_da_path(path)

    # ------------------------------------------------------------------
    # CARICAMENTO PROGETTO DA PERCORSO DIRETTO (usato anche dal drop)
    # ------------------------------------------------------------------

    def _carica_progetto_da_path(self, path: str):
        """Legge un file .scprj/.json e ripristina tutti i moduli."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                dati_progetto = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile leggere il file:\n{e}")
            return

        errori = []

        if 'materiali' in dati_progetto:
            try:
                self.sezioni.gestione_materiali.carica_dati(dati_progetto['materiali'])
            except Exception as e:
                errori.append(f"Materiali: {e}")

        if 'sezioni' in dati_progetto:
            try:
                self.sezioni.carica_dati(dati_progetto['sezioni'])
            except Exception as e:
                errori.append(f"Sezioni: {e}")

        if 'telaio' in dati_progetto:
            try:
                self.telaio.carica_dati(dati_progetto['telaio'])
            except Exception as e:
                errori.append(f"Telaio: {e}")

        if errori:
            msg = "\n".join(errori)
            print(f">> Avvisi durante il caricamento:\n{msg}")
            QMessageBox.warning(self, "Caricato con avvisi",
                                f"Progetto caricato con alcuni avvisi:\n{msg}")
        else:
            print(f">> Progetto caricato: {path}")
            QMessageBox.information(self, "Caricato", f"Progetto caricato:\n{path}")

    # ------------------------------------------------------------------

    def append_terminal_text(self, text):
        """Funzione chiamata ogni volta che c'è un print()"""
        cursor = self.console_output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.console_output.setTextCursor(cursor)
        self.console_output.insertPlainText(text)
        sb = self.console_output.verticalScrollBar()
        sb.setValue(sb.maximum())

        cursor2 = self.console_output_2.textCursor()
        cursor2.movePosition(QtGui.QTextCursor.End)
        self.console_output_2.setTextCursor(cursor2)
        self.console_output_2.insertPlainText(text)
        sb2 = self.console_output_2.verticalScrollBar()
        sb2.setValue(sb2.maximum())

    def closeEvent(self, event):
        """Ripristina stdout alla chiusura per evitare errori in console IDE"""
        sys.stdout = self.sys_stdout_backup
        sys.stderr = self.sys_stderr_backup
        super().closeEvent(event)

    def salva_screenshot(self):
        """Cattura uno screenshot fedele della finestra usando lo schermo."""
        opzioni = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Salva Screenshot",
            "Screenshot_SectionCHECK.png",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)",
            options=opzioni
        )

        if file_path:
            screen = QtWidgets.QApplication.primaryScreen()
            if screen:
                screenshot = screen.grabWindow(self.winId())
                successo = screenshot.save(file_path)
                if successo:
                    print(f">> Screenshot salvato (metodo Screen): {file_path}")
                else:
                    print(">> Errore nel salvataggio.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())