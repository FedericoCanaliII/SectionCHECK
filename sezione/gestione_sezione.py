# GestioneSezioni integrata con SectionManager e StaffaTool
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QButtonGroup, QPushButton, QMenu, QAction, QVBoxLayout, QLineEdit
from PyQt5.QtGui import QIntValidator

from materiali.gestione_materiali import GestioneMateriali

# importa il manager e il tool dalla module generato
from sezione.manager import SectionManager
from sezione.valori_print import Valori

# importa gli altri tool
from sezione.tools import SelectionTool
from tools.rettangolo import RectangleTool
from tools.movimento import MoveTool
from tools.cerchio import CircleTool
from tools.poligono import PolygonTool
from tools.barre import BarTool
from tools.staffe import StaffaTool
from tools.rinforzo import ReinforcementTool


class GestioneSezioni:
    def __init__(self, ui):
        self.ui = ui

        # ----- Section manager e widget OpenGL -----
        self.section_manager = SectionManager()
        self.gestione_materiali = GestioneMateriali(self.ui)

        # Timer iniziale per aprire pannelli (simula click iniziali)
        QtCore.QTimer.singleShot(0, self.ui.btn_sezioni_sezione.click)
        QtCore.QTimer.singleShot(0, self.ui.btn_sezioni_barre.click)
        QtCore.QTimer.singleShot(0, self.ui.btn_sezioni_staffe.click)
        QtCore.QTimer.singleShot(0, self.ui.btn_sezioni_rinforzi.click)
        QtCore.QTimer.singleShot(0, self.ui.btn_sezioni_rinforzi.click)


        # crea una singola istanza Valori
        self.valori_printer = Valori(self.section_manager, self.ui, gestione_materiali=self.gestione_materiali)
        try:
            self.ui.valori_btn.clicked.connect(self.valori_printer.print_valori)
        except Exception:
            try:
                self.valori_printer.connect_button(self.ui.valori_btn)
            except Exception:
                pass

        # registra callback manager -> valori_printer
        try:
            self.section_manager.on_update = lambda: self.valori_printer.update_tables()
        except Exception:
            pass

        # Rendi pulsanti checkable
        self.ui.btn_sezioni_sezione.setCheckable(True)
        self.ui.btn_sezioni_barre.setCheckable(True)
        self.ui.btn_sezioni_staffe.setCheckable(True)
        self.ui.btn_sezioni_rinforzi.setCheckable(True)

        # Collega toggle visibilità tabelle
        self.ui.btn_sezioni_sezione.toggled.connect(self.ui.tableView_sezione.setVisible)
        self.ui.btn_sezioni_barre.toggled.connect(self.ui.tableView_barre.setVisible)
        self.ui.btn_sezioni_staffe.toggled.connect(self.ui.tableView_staffe.setVisible)
        self.ui.btn_sezioni_rinforzi.toggled.connect(self.ui.tableView_rinforzi.setVisible)

        # OpenGL widget
        from sezione.disegno_sezione import OpenGLSectionWidget
        self.section_widget = OpenGLSectionWidget()
        layout = QVBoxLayout(self.ui.widget_sezioni)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.section_widget)

        # bind widget al manager
        self.section_manager.bind_widget(self.section_widget)

        # ----- tool instances (singleton) -----
        self.tools = {
            'move': MoveTool(),
            'select': SelectionTool(),
            'rect': RectangleTool(),
            'circle': CircleTool(),
            'poly': PolygonTool(),
            'bar': BarTool(),
            'staffe': StaffaTool(),  # integrato correttamente
            'reinforcement': ReinforcementTool()
        }

        # bind tools al manager (manager gestirà set_confirmed_list)
        self.section_manager.bind_tools(self.tools)

        # ----- gruppi bottoni -----
        self.btn_group_sezioni = QButtonGroup()
        self.btn_group_sezioni.setExclusive(True)

        self.btn_group_strumenti = QButtonGroup()
        self.btn_group_strumenti.setExclusive(True)

        self.section_buttons = []

        # setup UI controls
        self.ui.btn_sezioni_centra.clicked.connect(self.section_widget.reset_view)

        # grid spacing QLineEdit
        DEFAULT_GRID = 10
        le = getattr(self.ui, 'sezioni_dimenzioni_griglia', None)
        if le is not None:
            le.setValidator(QIntValidator(1, 10000, le))
            if le.text().strip() == "":
                le.setText(str(DEFAULT_GRID))

            def _apply_grid_from_le():
                txt = le.text().strip()
                try:
                    val = int(txt) if txt != "" else DEFAULT_GRID
                    if val < 1:
                        val = DEFAULT_GRID
                except Exception:
                    val = DEFAULT_GRID
                self.section_widget.set_grid_spacing(val)

            _apply_grid_from_le()
            le.editingFinished.connect(_apply_grid_from_le)
            try:
                le.returnPressed.connect(_apply_grid_from_le)
            except Exception:
                pass
        else:
            self.section_widget.set_grid_spacing(DEFAULT_GRID)

        # grid & snap buttons
        self.ui.btn_sezioni_griglia.setCheckable(True)
        self.ui.btn_sezioni_griglia.setChecked(True)
        self.ui.btn_sezioni_griglia.toggled.connect(self.section_widget.set_show_grid)

        self.ui.btn_sezioni_snap.setCheckable(True)
        self.ui.btn_sezioni_snap.setChecked(True)
        self.ui.btn_sezioni_snap.toggled.connect(self.section_widget.set_snap_to_grid)
        self.section_widget.set_snap_to_grid(self.ui.btn_sezioni_snap.isChecked())

        # ----- setup strumenti buttons (UI) -----
        self._setup_strumenti_buttons()

        # ----- imposto bottoni e sections iniziali -----
        idx0 = self.section_manager.create_section("Sezione 1")
        btn0 = self._create_section_button_for_index(idx0)
        self._switch_to_section(idx0)

        # bottone "+" per aggiungere nuove sezioni
        self.ui.btn_sezioni_piu.clicked.connect(self.aggiungi_nuove_sezioni)

        # inizialmente nessun tool attivo
        self.section_widget.set_active_tool(None)


    # ---------------- Sezioni UI / CRUD ----------------
    def _create_section_button_for_index(self, index: int) -> QPushButton:
        model = self.section_manager.get_section(index)
        name = model.name if model else f"Sezione {index+1}"
        nuovo_bottone = QPushButton(name)
        nuovo_bottone.setCheckable(True)
        nuovo_bottone.setContextMenuPolicy(Qt.CustomContextMenu)
        nuovo_bottone.setStyleSheet("""
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
        """)
        # inserimento layout
        try:
            count = self.ui.layout_sezioni.count()
            self.ui.layout_sezioni.insertWidget(count - 1, nuovo_bottone)
        except Exception:
            try:
                self.ui.layout_sezioni.addWidget(nuovo_bottone)
            except Exception:
                pass

        self.btn_group_sezioni.addButton(nuovo_bottone)
        self.section_buttons.append(nuovo_bottone)

        # click -> switch section
        def _on_click(checked=False, btn=nuovo_bottone):
            try:
                idx = self.section_buttons.index(btn)
            except ValueError:
                return
            self._switch_to_section(idx)
        nuovo_bottone.clicked.connect(_on_click)

        # context menu: rinomina / rimuovi
        def mostra_menu(pos, btn=nuovo_bottone):
            menu = QMenu()
            rimuovi_azione = QAction("Rimuovi")
            rinomina_azione = QAction("Rinomina")
            menu.addAction(rinomina_azione)
            menu.addAction(rimuovi_azione)

            def rimuovi():
                try:
                    idx = self.section_buttons.index(btn)
                except ValueError:
                    return
                self.section_manager.remove_section(idx)
                self.btn_group_sezioni.removeButton(btn)
                self.section_buttons.pop(idx)
                try:
                    self.ui.layout_sezioni.removeWidget(btn)
                except Exception:
                    pass
                btn.deleteLater()
                if self.section_manager.current_index is not None:
                    self._switch_to_section(self.section_manager.current_index)

            def rinomina():
                line_edit = QLineEdit(btn.text(), btn.parent())
                line_edit.setGeometry(btn.geometry())
                line_edit.setFont(btn.font())
                line_edit.show()
                line_edit.setFocus()
                btn.setEnabled(False)
                def conferma_modifica():
                    new_name = line_edit.text().strip()
                    if new_name == "":
                        new_name = btn.text()
                    try:
                        idx = self.section_buttons.index(btn)
                    except ValueError:
                        idx = None
                    if idx is not None:
                        self.section_manager.rename_section(idx, new_name)
                        btn.setText(new_name)
                    line_edit.deleteLater()
                    btn.setEnabled(True)
                line_edit.editingFinished.connect(conferma_modifica)

            rimuovi_azione.triggered.connect(rimuovi)
            rinomina_azione.triggered.connect(rinomina)
            menu.exec_(btn.mapToGlobal(pos))

        nuovo_bottone.customContextMenuRequested.connect(mostra_menu)
        nuovo_bottone.mouseDoubleClickEvent = lambda event, b=nuovo_bottone: nuovo_bottone.customContextMenuRequested.emit(nuovo_bottone.mapFromGlobal(nuovo_bottone.mapToGlobal(event.pos())) )

        return nuovo_bottone

    def aggiungi_nuove_sezioni(self):
        idx = self.section_manager.create_section()
        btn = self._create_section_button_for_index(idx)
        self._switch_to_section(idx)

    def _switch_to_section(self, index: int):
        if index is None or index < 0 or index >= len(self.section_manager.sections):
            return
        self.section_manager.switch_section(index)
        for i, b in enumerate(self.section_buttons):
            try:
                b.setChecked(i == index)
            except Exception:
                pass
        try:
            self.section_widget.update()
        except Exception:
            pass
        try:
            if hasattr(self, 'valori_printer') and self.valori_printer is not None:
                self.valori_printer.update_tables()
        except Exception:
            pass

    # ---------------- Strumenti (buttons -> tools) ----------------
    def _setup_strumenti_buttons(self):
        bottoni_strumenti = [
            ('move', getattr(self.ui, 'btn_muovi', None)),
            ('select', getattr(self.ui, 'btn_selezione', None)),
            ('rect', getattr(self.ui, 'btn_sezione_rettangolo', None)),
            ('circle', getattr(self.ui, 'btn_sezione_cerchio', None)),
            ('poly', getattr(self.ui, 'btn_sezione_poligono', None)),
            ('bar', getattr(self.ui, 'btn_barra', None)),
            ('staffe', getattr(self.ui, 'btn_staffa', None)),  # bottone staffa
            ('reinforcement', getattr(self.ui, 'btn_rinforzo', None))
        ]

        for name, btn in bottoni_strumenti:
            if btn is None:
                continue
            btn.setObjectName(f"btn_{name}")
            btn.setCheckable(True)
            btn.setAutoExclusive(False)
            self.btn_group_strumenti.addButton(btn)
            btn.clicked.connect(lambda checked, n=name, b=btn: self._on_tool_button_toggled(n, b.isChecked()))
        self.btn_group_strumenti.setExclusive(True)

    def _on_tool_button_toggled(self, tool_name: str, checked: bool):
        if not checked:
            self.section_widget.set_active_tool(None)
            for b in self.btn_group_strumenti.buttons():
                try:
                    b.setChecked(False)
                except Exception:
                    pass
            return

        tool = self.tools.get(tool_name)
        if tool is None:
            self.section_widget.set_active_tool(None)
            return

        for b in self.btn_group_strumenti.buttons():
            try:
                if hasattr(b, 'objectName') and b.objectName() == f"btn_{tool_name}":
                    b.setChecked(True)
                else:
                    b.setChecked(False)
            except Exception:
                pass

        self.section_widget.set_active_tool(tool)

    # ----- utility / UI small helpers -----
    def toggle_frame_impostazioni(self, checked):
        try:
            self.ui.frame_sezioni_impostazioni.setVisible(checked)
        except Exception:
            pass

    # ----- import bulk shapes (rettangoli) -----
    def import_rects_to_section(self, index: int, raw_rects_list):
        from sezione.manager import convert_bulk_rects
        canon = convert_bulk_rects(raw_rects_list)
        self.section_manager.attach_rects_bulk(index, canon)
