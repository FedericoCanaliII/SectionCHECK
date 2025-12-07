# gestione_output.py
from PyQt5.QtWidgets import QVBoxLayout, QButtonGroup
from PyQt5.QtWidgets import QPushButton, QToolButton, QAbstractButton, QWidget, QComboBox
from PyQt5.QtCore import Qt, QTimer
import traceback
import numpy as np

# importa i moduli domain/verifica dal tuo progetto (come già facevi)
from output.verifica import Verifica
from output.calcolo import SezioneRinforzata, Materiale, DomainCalculator


class GestioneOutput:
    def __init__(self, ui, gestione_sezioni, gestione_materiali):
        """
        ui: oggetto UI (contenitore widget e controlli)
        gestione_sezioni: l'oggetto GestioneSezioni (o equivalente) che espone section_manager / UI
        gestione_materiali: istanza GestioneMateriali
        """
        self.ui = ui
        self.gestione_sezioni = gestione_sezioni
        self.gestione_materiali = gestione_materiali
        self.verifica = Verifica(self.gestione_sezioni, self.gestione_materiali)

        # punti calcolati (numpy array o None)
        self.punti = None

        # mapping combobox -> indice reale sezione
        self._output_section_map = []   # list: combobox index -> section_index
        self.selected_section_index = None

        # TRY creare widgets di visualizzazione (3D e 2D). Se fallisce lascialo None ma non crashare.
        self.widget3d = None
        self.widget2d = None
        try:
            from output.disegno_output import OpenGLDomainWidget
            self.widget3d = OpenGLDomainWidget(self.ui, parent=getattr(self.ui, 'widget_out', None))
        except Exception:
            self.widget3d = None
            # optional: print error for debugging
            print("Warning: non è stato possibile creare OpenGLDomainWidget.")
            traceback.print_exc()

        try:
            from output.disegno_output_2d import Domain2DWidget
            self.widget2d = Domain2DWidget(self.ui, parent=getattr(self.ui, 'widget_out', None))
        except Exception:
            self.widget2d = None
            print("Warning: non è stato possibile creare Domain2DWidget.")
            traceback.print_exc()

        # layout: aggiungi i widget se esistono
        try:
            container = getattr(self.ui, 'widget_out', None)
            if container is not None:
                layout = QVBoxLayout(container)
                layout.setContentsMargins(1, 1, 1, 1)
                if self.widget3d is not None:
                    layout.addWidget(self.widget3d)
                if self.widget2d is not None:
                    layout.addWidget(self.widget2d)
        except Exception:
            traceback.print_exc()

        # Default: mostra 3D (se esiste) e nascondi 2D
        try:
            if self.widget3d is not None:
                self.widget3d.show()
            if self.widget2d is not None:
                self.widget2d.hide()
        except Exception:
            pass

        # pulsante verifica -> avvia verifica completa
        try:
            btn_ver = getattr(self.ui, 'btn_out_verifica', None)
            if btn_ver is not None:
                btn_ver.clicked.connect(self.verifica_completa)
        except Exception:
            traceback.print_exc()

        # Group di bottoni per scelta view (se presenti in UI)
        try:
            btns = (
                getattr(self.ui, 'btn_out_N_Mx', None),
                getattr(self.ui, 'btn_out_N_My', None),
                getattr(self.ui, 'btn_out_Mx_My', None),
                getattr(self.ui, 'btn_out_3d', None),
            )
            for btn in btns:
                if btn is not None:
                    try:
                        btn.setCheckable(True)
                    except Exception:
                        pass

            self.btn_group = QButtonGroup()
            self.btn_group.setExclusive(True)
            for btn in btns:
                if btn is not None:
                    self.btn_group.addButton(btn)

            # connect buttons to view selection
            if getattr(self.ui, 'btn_out_N_Mx', None) is not None:
                self.ui.btn_out_N_Mx.clicked.connect(lambda: self.select_view("N_Mx"))
            if getattr(self.ui, 'btn_out_N_My', None) is not None:
                self.ui.btn_out_N_My.clicked.connect(lambda: self.select_view("N_My"))
            if getattr(self.ui, 'btn_out_Mx_My', None) is not None:
                self.ui.btn_out_Mx_My.clicked.connect(lambda: self.select_view("Mx_My"))
            if getattr(self.ui, 'btn_out_3d', None) is not None:
                self.ui.btn_out_3d.clicked.connect(lambda: self.select_view("3D"))

            # default checked 3D se presente (postpone to next loop)
            if getattr(self.ui, 'btn_out_3d', None) is not None:
                QTimer.singleShot(0, lambda: self.ui.btn_out_3d.setChecked(True))
        except Exception:
            traceback.print_exc()

        # thread di calcolo (DomainCalculator)
        self.calc_thread = None

        # collega il pulsante che popola la combobox per l'output
        try:
            btn_main = getattr(self.ui, 'btn_main_output', None)
            if btn_main is not None:
                btn_main.clicked.connect(self.populate_output_combobox)
        except Exception:
            traceback.print_exc()

        # collega la combobox per tracciare la selezione
        try:
            combo = getattr(self.ui, 'combobox_output_sezioni', None)
            if combo is not None and isinstance(combo, QComboBox):
                combo.currentIndexChanged.connect(self._on_output_combo_changed)
        except Exception:
            traceback.print_exc()

    # ---------------- View switching ----------------
    def select_view(self, mode):
        """
        Switch fra widget 3D e 2D e setta la modalità di visione.
        """
        try:
            if mode == "3D":
                if self.widget3d is not None:
                    self.widget3d.show()
                if self.widget2d is not None:
                    self.widget2d.hide()
                if self.widget3d is not None and hasattr(self.widget3d, 'set_view_mode'):
                    try: self.widget3d.set_view_mode(mode)
                    except Exception: pass
                if self.punti is not None and self.widget3d is not None and hasattr(self.widget3d, 'set_points'):
                    try: self.widget3d.set_points(self.punti)
                    except Exception: pass
            else:
                if self.widget3d is not None:
                    self.widget3d.hide()
                if self.widget2d is not None:
                    self.widget2d.show()
                if self.widget2d is not None and hasattr(self.widget2d, 'set_view_mode'):
                    try: self.widget2d.set_view_mode(mode)
                    except Exception: pass
                if self.punti is not None and self.widget2d is not None and hasattr(self.widget2d, 'set_points'):
                    try: self.widget2d.set_points(self.punti)
                    except Exception: pass
        except Exception:
            traceback.print_exc()

    # ---------------- Combobox population & mapping ----------------
    def populate_output_combobox(self):
        """
        Popola combobox_output_sezioni mappando ciascun item all'indice reale
        della sezione (UserRole). Usa section_manager.sections se disponibile,
        altrimenti usa get_section_names_from_buttons() come fallback.
        """
        combo = getattr(self.ui, 'combobox_output_sezioni', None)
        if combo is None:
            return
        try:
            combo.blockSignals(True)
            combo.clear()
            self._output_section_map = []
            self.selected_section_index = None

            # Preferisci usare SectionManager.sections (ordine stabile)
            sm = None
            if hasattr(self.gestione_sezioni, 'section_manager'):
                sm = getattr(self.gestione_sezioni, 'section_manager')
            elif hasattr(self.gestione_sezioni, 'sections'):
                sm = self.gestione_sezioni

            if sm is not None and hasattr(sm, 'sections'):
                secs = getattr(sm, 'sections') or []
                if not secs:
                    combo.addItem("Nessuna sezione")
                    combo.setEnabled(False)
                    combo.blockSignals(False)
                    return

                combo.setEnabled(True)
                for section_index, sec in enumerate(secs):
                    # attempt to read name
                    name = None
                    try:
                        name = getattr(sec, 'name', None)
                        if name is None and isinstance(sec, dict):
                            name = sec.get('name')
                    except Exception:
                        name = None
                    if not name:
                        name = f"Sezione {section_index+1}"
                    combo.addItem(str(name))
                    # save section_index in itemData
                    combo.setItemData(combo.count() - 1, section_index, Qt.UserRole)
                    self._output_section_map.append(section_index)

                combo.setCurrentIndex(0)
                # set selected_section_index
                if self._output_section_map:
                    self.selected_section_index = self._output_section_map[0]
                combo.blockSignals(False)
                return

            # Fallback: use names scraped from buttons / children
            names = self.get_section_names_from_buttons()
            if not names:
                combo.addItem("Nessuna sezione")
                combo.setEnabled(False)
                combo.blockSignals(False)
                return

            combo.setEnabled(True)
            for i, name in enumerate(names):
                combo.addItem(str(name))
                combo.setItemData(i, i, Qt.UserRole)
                self._output_section_map.append(i)

            combo.setCurrentIndex(0)
            self.selected_section_index = self._output_section_map[0] if self._output_section_map else None
            combo.blockSignals(False)
        except Exception:
            traceback.print_exc()
            try:
                combo.blockSignals(False)
            except Exception:
                pass

    def _on_output_combo_changed(self, comb_idx: int):
        """
        Aggiorna selected_section_index leggendo itemData (UserRole) o fallback sulla mappa.
        """
        combo = getattr(self.ui, 'combobox_output_sezioni', None)
        if combo is None:
            self.selected_section_index = None
            return
        try:
            data = combo.itemData(comb_idx, Qt.UserRole)
            if data is None:
                # fallback: usa lista mappata
                if 0 <= comb_idx < len(self._output_section_map):
                    self.selected_section_index = self._output_section_map[comb_idx]
                else:
                    # ultima risorsa: usa comb_idx come indice
                    try:
                        self.selected_section_index = int(comb_idx)
                    except Exception:
                        self.selected_section_index = None
            else:
                try:
                    self.selected_section_index = int(data)
                except Exception:
                    self.selected_section_index = data
        except Exception:
            traceback.print_exc()
            try:
                self.selected_section_index = int(comb_idx)
            except Exception:
                self.selected_section_index = None

    # ---------------- Verifica / avvio calcolo ----------------
    def verifica_completa(self):
        """
        Esegue la preparazione e parte il DomainCalculator sulla sezione scelta
        nella combobox (non sulla sezione corrente dell'editor).
        """
        try:
            tutte_sezioni = self.verifica.get_tutte_matrici_sezioni()
            tutte_materiali = self.verifica.get_tutte_matrici_materiali()
            # debug print (opzionale)
            # print(tutte_sezioni)
            # print(list(tutte_materiali.keys()))

            if not tutte_sezioni:
                print("Nessuna sezione trovata per la verifica.")
                return
            if not tutte_materiali:
                print("Nessun materiale trovato per la verifica.")
                return

            # determina indice selezionato (preferisci selected_section_index)
            sel_idx = getattr(self, 'selected_section_index', None)
            combo = getattr(self.ui, 'combobox_output_sezioni', None)
            # se non impostato, prova a leggere dalla combobox
            if sel_idx is None and combo is not None:
                try:
                    data = combo.itemData(combo.currentIndex(), Qt.UserRole)
                    sel_idx = int(data) if data is not None else combo.currentIndex()
                except Exception:
                    sel_idx = combo.currentIndex()

            # fallback a 0 se fuori range
            try:
                if sel_idx is None or not (0 <= int(sel_idx) < len(tutte_sezioni)):
                    sel_idx = 0
            except Exception:
                sel_idx = 0
            sel_idx = int(sel_idx)

            pagina = tutte_sezioni[sel_idx]
            # costruisci dizionario materiali atteso da SezioneRinforzata
            materiali_dict = {}
            try:
                for nome, matr in tutte_materiali.items():
                    try:
                        materiali_dict[nome] = Materiale(matr, nome)
                    except Exception:
                        # fallback: keep raw if Materiale construction fails
                        materiali_dict[nome] = Materiale(matr, str(nome))
            except Exception:
                traceback.print_exc()

            sezione_rc = SezioneRinforzata(pagina['elementi'], materiali_dict)

            # se esiste un thread precedente in esecuzione, prova a fermarlo in modo pulito
            try:
                if self.calc_thread is not None:
                    try:
                        # se DomainCalculator è un QThread e ha metodi per fermarsi / quit
                        if hasattr(self.calc_thread, 'isRunning') and self.calc_thread.isRunning():
                            try:
                                # preferisci una API di stop se esiste
                                if hasattr(self.calc_thread, 'requestInterruption'):
                                    self.calc_thread.requestInterruption()
                                if hasattr(self.calc_thread, 'quit'):
                                    self.calc_thread.quit()
                                # non sempre join è necessario in Qt; lasciamo che termini
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

            # crea e avvia nuovo thread di calcolo
            try:
                self.calc_thread = DomainCalculator(sezione_rc, self.ui)
                # connessioni: assicurati che update_points esista
                try:
                    self.calc_thread.calculation_done.connect(self.update_points)
                except Exception:
                    # versione alternativa del segnale: try any on_finished name
                    try:
                        self.calc_thread.finished.connect(self.update_points)
                    except Exception:
                        pass

                # connect progress bar se presente
                try:
                    if hasattr(self.calc_thread, 'progress') and getattr(self.ui, 'progressBar_verifica', None) is not None:
                        self.calc_thread.progress.connect(self.ui.progressBar_verifica.setValue)
                except Exception:
                    pass

                # start
                try:
                    self.calc_thread.start()
                except Exception:
                    # se DomainCalculator non è QThread ma funzione sync, prova a chiamarne run()
                    try:
                        self.calc_thread.run()
                    except Exception:
                        traceback.print_exc()
            except Exception:
                traceback.print_exc()

        except Exception:
            traceback.print_exc()

    # signal handler: aggiorna i punti e ripple ai widget
    def update_points(self, punti):
        """
        punti: lista di punti o numpy array (come emesso da DomainCalculator)
        """
        try:
            if punti is None:
                self.punti = None
            else:
                self.punti = np.array(punti)

            # aggiorna il widget attualmente visibile
            try:
                if getattr(self.ui, 'btn_out_3d', None) is not None and getattr(self.ui, 'btn_out_3d', None).isChecked():
                    if self.widget3d is not None and hasattr(self.widget3d, 'set_points'):
                        self.widget3d.set_points(self.punti)
                else:
                    if self.widget2d is not None and hasattr(self.widget2d, 'set_points'):
                        self.widget2d.set_points(self.punti)
            except Exception:
                traceback.print_exc()
        except Exception:
            traceback.print_exc()

    # ---------------- Utilities per estrarre nomi (fallback) ----------------
    def _extract_names_from_button_like(self, btn):
        try:
            if hasattr(btn, 'text') and callable(getattr(btn, 'text')):
                t = btn.text()
                if isinstance(t, str) and t.strip():
                    return t.strip()
        except Exception:
            pass
        try:
            on = getattr(btn, 'objectName', None)
            if callable(on):
                on = on()
            if isinstance(on, str) and on.strip():
                return on.strip()
        except Exception:
            pass
        try:
            return str(btn)
        except Exception:
            return ""

    def get_section_names_from_buttons(self):
        """
        Fa un best-effort per estrarre nomi di sezione da gestione_sezioni (bottoni/figli/attributi).
        Usato solo in fallback quando non è disponibile section_manager.sections.
        """
        gs = self.gestione_sezioni
        if gs is None:
            return []

        names = []
        from PyQt5.QtWidgets import QPushButton, QToolButton, QAbstractButton

        try:
            for attr in ('buttons', 'bottoni', 'button_list', 'btns', 'buttons_list', 'lista_bottoni'):
                if hasattr(gs, attr):
                    val = getattr(gs, attr)
                    if val is None:
                        continue
                    if isinstance(val, dict):
                        vals = list(val.values())
                        for v in vals:
                            if isinstance(v, (QPushButton, QToolButton, QAbstractButton)):
                                names.append(self._extract_names_from_button_like(v))
                            else:
                                # if values not button, append key names
                                try:
                                    names.append(str(v))
                                except Exception:
                                    pass
                    elif isinstance(val, (list, tuple)):
                        for item in val:
                            if isinstance(item, (QPushButton, QToolButton, QAbstractButton)):
                                names.append(self._extract_names_from_button_like(item))
                            elif isinstance(item, str):
                                names.append(item)
                    else:
                        if isinstance(val, (QPushButton, QToolButton, QAbstractButton)):
                            names.append(self._extract_names_from_button_like(val))
        except Exception:
            pass

        # findChildren fallback if gs is a QWidget-like
        try:
            if isinstance(gs, QWidget) or hasattr(gs, 'findChildren'):
                btns = []
                try:
                    btns.extend(gs.findChildren(QPushButton))
                except Exception:
                    pass
                try:
                    btns.extend(gs.findChildren(QToolButton))
                except Exception:
                    pass
                try:
                    btns.extend(gs.findChildren(QAbstractButton))
                except Exception:
                    pass
                for b in btns:
                    names.append(self._extract_names_from_button_like(b))
        except Exception:
            pass

        # explore attributes for buttons
        try:
            for k in dir(gs):
                if k.startswith('_'):
                    continue
                try:
                    v = getattr(gs, k)
                except Exception:
                    continue
                if isinstance(v, (QPushButton, QToolButton, QAbstractButton)):
                    names.append(self._extract_names_from_button_like(v))
                if isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, (QPushButton, QToolButton, QAbstractButton)):
                            names.append(self._extract_names_from_button_like(item))
        except Exception:
            pass

        # remove duplicates keeping order
        seen = set()
        out = []
        for n in names:
            if not isinstance(n, str):
                try:
                    n = str(n)
                except Exception:
                    continue
            n = n.strip()
            if not n:
                continue
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out
