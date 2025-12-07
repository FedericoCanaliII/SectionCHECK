from PyQt5 import QtWidgets, QtCore, QtGui

class NuovoMaterialePage(QtWidgets.QWidget):
    """
    Pagina custom per il nuovo materiale, con righe dinamiche per σ, ε_min, ε_max in colonne separate.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.contatore_righe = 0
        self.lista_sigma_extra = []
        self.lista_epsilon_min_extra = []
        self.lista_epsilon_max_extra = []
        # Stile uniforme per tutti i QLineEdit
        self.style_lineedit = (
            "background-color: rgb(30,30,30);"
            "font: 10pt \"Segoe UI\";"
            "color: rgb(255, 255, 255);"
            "padding-left: 5px;"
            "border-radius:4px;"
        )
        self.setup_ui()

    def setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(11, 11, 11, 11)

        # Titolo
        self.label_nuovomat = QtWidgets.QLabel("NUOVO MATERIALE", self)
        self.label_nuovomat.setFont(QtGui.QFont("Segoe UI", 12))
        self.label_nuovomat.setStyleSheet("color: white;")
        self.label_nuovomat.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label_nuovomat)
        self.layout.addSpacing(15)

        # Gamma input
        frame_gamma = QtWidgets.QFrame(self)
        frame_gamma.setStyleSheet("background-color: #282828; border-radius:4px;")
        h_gamma = QtWidgets.QHBoxLayout(frame_gamma)
        lbl_gamma = QtWidgets.QLabel("γ:", frame_gamma)
        lbl_gamma.setFont(QtGui.QFont("Segoe UI", 12))
        lbl_gamma.setStyleSheet("color: white;")
        self.nuovomat_gamma = QtWidgets.QLineEdit(frame_gamma)
        self.nuovomat_gamma.setText("1")
        self.nuovomat_gamma.setFixedSize(120, 32)
        # Mantiene lo stile esistente per gamma
        self.nuovomat_gamma.setStyleSheet(
            "background-color: rgb(30,30,30);"
            "font: 10pt \"Segoe UI\";"
            "color: rgb(255, 255, 255);"
            "padding-left: 5px;"
            "border-radius:4px;"
        )
        h_gamma.addWidget(lbl_gamma)
        h_gamma.addStretch()
        h_gamma.addWidget(self.nuovomat_gamma)
        self.layout.addWidget(frame_gamma)
        self.layout.addSpacing(15)

        # Frame principale contenente bottoni e input
        frame = QtWidgets.QFrame(self)
        frame.setStyleSheet("background-color: #282828; border-radius:4px;")
        v_main = QtWidgets.QVBoxLayout(frame)

        # Bottoni
        h_btn = QtWidgets.QHBoxLayout()
        h_btn.addStretch()
        self.btn_nuovomat_aggiungi = QtWidgets.QPushButton("Aggiungi", frame)
        self.btn_nuovomat_rimuovi = QtWidgets.QPushButton("Rimuovi", frame)
        for btn in (self.btn_nuovomat_aggiungi, self.btn_nuovomat_rimuovi):
            btn.setFixedSize(94, 34)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgb(50, 50, 50);
                    font: 400 12pt "Segoe UI";
                    color: rgb(255, 255, 255);
                    padding-bottom: 4px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: rgb(30, 30, 30);
                    border: 1px solid rgb(120, 120, 120);
                }
            """)

            h_btn.addWidget(btn)
        v_main.addLayout(h_btn)
        v_main.addSpacing(15)

        # Layout orizzontale per le 3 colonne
        h_cols = QtWidgets.QHBoxLayout()
        # Colonna Sigma
        v_sigma = QtWidgets.QVBoxLayout()
        lbl_sigma = QtWidgets.QLabel("σ (ε):", frame)
        lbl_sigma.setFont(QtGui.QFont("Segoe UI", 14))
        lbl_sigma.setStyleSheet("color:white;")
        self.nuovomat_sigma = QtWidgets.QLineEdit(frame)
        self.nuovomat_sigma.setFixedHeight(32)
        self.nuovomat_sigma.setMinimumWidth(240)
        self.nuovomat_sigma.setStyleSheet(self.style_lineedit)
        v_sigma.addWidget(lbl_sigma)
        v_sigma.addWidget(self.nuovomat_sigma)
        # container per righe extra sigma
        self.layout_extra_sigma = QtWidgets.QVBoxLayout()
        v_sigma.addLayout(self.layout_extra_sigma)
        h_cols.addLayout(v_sigma)

        # Colonna Epsilon Min
        v_eps_min = QtWidgets.QVBoxLayout()
        lbl_eps_min = QtWidgets.QLabel("ε min:", frame)
        lbl_eps_min.setFont(QtGui.QFont("Segoe UI", 14))
        lbl_eps_min.setStyleSheet("color:white;")
        self.nuovomat_epsilon_min = QtWidgets.QLineEdit(frame)
        self.nuovomat_epsilon_min.setFixedHeight(32)
        self.nuovomat_epsilon_min.setStyleSheet(self.style_lineedit)
        v_eps_min.addWidget(lbl_eps_min)
        v_eps_min.addWidget(self.nuovomat_epsilon_min)
        self.layout_extra_eps_min = QtWidgets.QVBoxLayout()
        v_eps_min.addLayout(self.layout_extra_eps_min)
        h_cols.addLayout(v_eps_min)

        # Colonna Epsilon Max
        v_eps_max = QtWidgets.QVBoxLayout()
        lbl_eps_max = QtWidgets.QLabel("ε max:", frame)
        lbl_eps_max.setFont(QtGui.QFont("Segoe UI", 14))
        lbl_eps_max.setStyleSheet("color:white;")
        self.nuovomat_epsilon_max = QtWidgets.QLineEdit(frame)
        self.nuovomat_epsilon_max.setFixedHeight(32)
        self.nuovomat_epsilon_max.setStyleSheet(self.style_lineedit)
        v_eps_max.addWidget(lbl_eps_max)
        v_eps_max.addWidget(self.nuovomat_epsilon_max)
        self.layout_extra_eps_max = QtWidgets.QVBoxLayout()
        v_eps_max.addLayout(self.layout_extra_eps_max)
        h_cols.addLayout(v_eps_max)

        v_main.addLayout(h_cols)
        self.layout.addWidget(frame)
        self.layout.addStretch()

        # Collego i pulsanti
        self.btn_nuovomat_aggiungi.clicked.connect(self.aggiungi_riga_nuovomat)
        self.btn_nuovomat_rimuovi.clicked.connect(self.rimuovi_riga_nuovomat)

    def crea_lineedit_personalizzato(self, object_name):
        le = QtWidgets.QLineEdit()
        le.setObjectName(object_name)
        le.setFixedHeight(32)
        if 'sigma' in object_name:
            le.setMinimumWidth(240)
        le.setStyleSheet(self.style_lineedit)
        return le

    def aggiungi_riga_nuovomat(self):
        self.contatore_righe += 1
        idx = self.contatore_righe + 1
        sigma_le = self.crea_lineedit_personalizzato(f"nuovomat_sigma_{idx}")
        eps_min_le = self.crea_lineedit_personalizzato(f"nuovomat_epsilon_min_{idx}")
        eps_max_le = self.crea_lineedit_personalizzato(f"nuovomat_epsilon_max_{idx}")
        self.layout_extra_sigma.addWidget(sigma_le)
        self.layout_extra_eps_min.addWidget(eps_min_le)
        self.layout_extra_eps_max.addWidget(eps_max_le)
        self.lista_sigma_extra.append(sigma_le)
        self.lista_epsilon_min_extra.append(eps_min_le)
        self.lista_epsilon_max_extra.append(eps_max_le)

    def rimuovi_riga_nuovomat(self):
        if self.contatore_righe > 0:
            sigma_le = self.lista_sigma_extra.pop()
            eps_min_le = self.lista_epsilon_min_extra.pop()
            eps_max_le = self.lista_epsilon_max_extra.pop()
            self.layout_extra_sigma.removeWidget(sigma_le)
            self.layout_extra_eps_min.removeWidget(eps_min_le)
            self.layout_extra_eps_max.removeWidget(eps_max_le)
            sigma_le.deleteLater()
            eps_min_le.deleteLater()
            eps_max_le.deleteLater()
            self.contatore_righe -= 1

    def generatore_matrice_diagramma(self):
        matrice = []
        try:
            base = [
                (self.nuovomat_sigma.text() or 0),
                float(self.nuovomat_epsilon_min.text() or 0),
                float(self.nuovomat_epsilon_max.text() or 0)
            ]
            if base[1] != base[2]:
                matrice.append(base)
        except ValueError:
            pass
        for s_le, mn_le, mx_le in zip(
            self.lista_sigma_extra,
            self.lista_epsilon_min_extra,
            self.lista_epsilon_max_extra
        ):
            try:
                vals = [(s_le.text() or 0), float(mn_le.text() or 0), float(mx_le.text() or 0)]
                if vals[1] != vals[2]:
                    matrice.append(vals)
            except ValueError:
                continue
        #print("Matrice Nuovo Materiale Generata:", matrice)
        return matrice
