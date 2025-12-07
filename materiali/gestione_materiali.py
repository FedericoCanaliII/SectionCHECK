#---------------------------------------------------------------------------------------- GESTIONE MATERIALI --------------------------------------------------------------------------------------
from PyQt5.QtWidgets import QButtonGroup, QPushButton, QMenu, QLineEdit, QAction, QVBoxLayout
from PyQt5.QtCore import Qt
from materiali.disego_diagrammi import OpenGLGraphWidget
from materiali.calcestruzzo import Calcestruzzo
from materiali.acciaio_barre import Acciaio_Barre
from materiali.acciaio_profili import Acciaio_Profili
from materiali.frp import Frp
from materiali.pagina_nuovomat import NuovoMaterialePage
from materiali.combobox import Combobox_matriali

class GestioneMateriali:
    def __init__(self, ui):
        self.ui = ui
        self.btn_group_materiali = QButtonGroup(self.ui.stackedWidget_materiali)
        self.btn_group_materiali.setExclusive(True)
        self.setup_materiali()
#--------------------------------------------------------------------------------------- FUNZIONAMNTO DIAGRAMMA --------------------------------------------------------------------------------------
        
        # Crea il widget OpenGL
        self.graph_widget = OpenGLGraphWidget()
        # Da qualche parte nella tua app principale
        ui.calcestruzzo = Calcestruzzo(ui)
        ui.acciaio_barre = Acciaio_Barre(ui)
        ui.acciaio_profili = Acciaio_Profili(ui)
        ui.frp = Frp(ui)

        self.ui.combobox_materiali = Combobox_matriali(ui)
        
        # Pulsanti per il grafico
        self.ui.btn_materiali_centra.clicked.connect(self.graph_widget.reset_view)
        self.ui.btn_materiali_rigenera.clicked.connect(self.rigenera_grafico_materiale_corrente)

        # Crea un layout per il contenitore QtDesigner esistente
        layout = QVBoxLayout(self.ui.widget_materiali)
        layout.setContentsMargins(1,1,1,1)
        layout.addWidget(self.graph_widget)

#------------------------------------------------------------------------------------ MECCANISMI PER L'INTERFACCIA -------------------------------------------------------------------------------------
    def setup_materiali(self):
        # Imposta i pulsanti iniziali come checkable e li aggiunge al button group
        self._setup_bottone_materiale(self.ui.btn_materiali_calcestruzzo, 1)
        self._setup_bottone_materiale(self.ui.btn_materiali_acciaiobarre, 2)
        self._setup_bottone_materiale(self.ui.btn_materiali_acciaioprofili, 3)
        self._setup_bottone_materiale(self.ui.btn_materiali_frp, 4)

        # Imposta la pagina iniziale
        self.ui.stackedWidget_materiali.setCurrentIndex(0)

        # Collega il pulsante "+" alla funzione di aggiunta nuovi materiali
        self.ui.btn_materiali_piu.clicked.connect(self.aggiungi_nuovo_materiale)

    def _setup_bottone_materiale(self, bottone, index_pagina):
        bottone.setCheckable(True)
        self.btn_group_materiali.addButton(bottone)
        
        def cambia_pagina_e_aggiorna():
            self.ui.stackedWidget_materiali.setCurrentIndex(index_pagina)
            self.rigenera_grafico_materiale_corrente()
            
        bottone.clicked.connect(cambia_pagina_e_aggiorna)


    def aggiungi_nuovo_materiale(self):
        # Crea il nuovo pulsante
        nuovo_bottone = QPushButton("Nuovo mat.")
        nuovo_bottone.setCheckable(True)
        nuovo_bottone.setContextMenuPolicy(Qt.CustomContextMenu)  # Attiva menu contestuale

        # Applica lo stylesheet
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

        # Inserisci il pulsante nella penultima posizione del layout
        count = self.ui.layout_materiali.count()
        self.ui.layout_materiali.insertWidget(count - 1, nuovo_bottone)

        # Aggiungi il pulsante al gruppo
        self.btn_group_materiali.addButton(nuovo_bottone)

        # Crea la nuova pagina e aggiungi la UI del nuovo materiale
        nuova_pagina = NuovoMaterialePage()
        self.ui.stackedWidget_materiali.addWidget(nuova_pagina)
        nuovo_index = self.ui.stackedWidget_materiali.count() - 1

        # Collega il pulsante alla nuova pagina
        nuovo_bottone.clicked.connect(
            lambda _, b=nuovo_bottone, idx=nuovo_index: (
                b.setChecked(True),
                self.ui.stackedWidget_materiali.setCurrentIndex(idx)
            )
        )

        # Definisci il menu contestuale
        def mostra_menu(pos):
            menu = QMenu()
            rimuovi_azione = QAction("Rimuovi")
            menu.addAction(rimuovi_azione)

            def rimuovi():
                self.btn_group_materiali.removeButton(nuovo_bottone)
                self.ui.layout_materiali.removeWidget(nuovo_bottone)
                nuovo_bottone.deleteLater()

                self.ui.stackedWidget_materiali.removeWidget(nuova_pagina)
                nuova_pagina.deleteLater()

            rimuovi_azione.triggered.connect(rimuovi)
            menu.exec_(nuovo_bottone.mapToGlobal(pos))

        # Collega il click destro al menu
        nuovo_bottone.customContextMenuRequested.connect(mostra_menu)

        def modifica_testo():
            line_edit = QLineEdit(nuovo_bottone.text(), nuovo_bottone.parent())
            line_edit.setGeometry(nuovo_bottone.geometry())
            line_edit.setFont(nuovo_bottone.font())
            line_edit.show()
            line_edit.setFocus()
            nuovo_bottone.setEnabled(False)  # disabilita temporaneamente il bottone

            def conferma_modifica():
                nuovo_bottone.setText(line_edit.text())
                line_edit.deleteLater()
                nuovo_bottone.setEnabled(True)

            # Conferma modifica alla pressione di invio o perdita focus
            line_edit.editingFinished.connect(conferma_modifica)

        nuovo_bottone.mouseDoubleClickEvent = lambda event: modifica_testo()

    def rigenera_grafico_materiale_corrente(self):
        indice_corrente = self.ui.stackedWidget_materiali.currentIndex()

        # Calcestruzzo (index 1)
        if indice_corrente == 1:
            matrice = self.ui.calcestruzzo.generatore_matrice_calcestruzzo()
            self.graph_widget.set_data(matrice)

        # Acciaio Barre (index 2)
        elif indice_corrente == 2:
            matrice = self.ui.acciaio_barre.generatore_matrice_acciaiobarre()
            self.graph_widget.set_data(matrice)

        # Acciaio Profili (index 3)
        elif indice_corrente == 3:
            matrice = self.ui.acciaio_profili.generatore_matrice_acciaioprofili()
            self.graph_widget.set_data(matrice)

        # FRP (index 4)
        elif indice_corrente == 4:
            matrice = self.ui.frp.generatore_matrice_frp()
            self.graph_widget.set_data(matrice)

        # Materiali dinamici (index >= 5)
        elif indice_corrente >= 5:
            pagina = self.ui.stackedWidget_materiali.widget(indice_corrente)
            if hasattr(pagina, 'generatore_matrice_diagramma'):
                matrice = pagina.generatore_matrice_diagramma()
                self.graph_widget.set_data(matrice)


        # Rigenera il disegno
        self.graph_widget.recalculate()
