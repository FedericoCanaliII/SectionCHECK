from PyQt5.QtWidgets import QLineEdit

class Acciaio_Barre:

    def __init__(self, ui):
        self.ui = ui

        # Liste per i QLineEdit dinamici
        self.lista_sigma_extra = []
        self.lista_epsilon_min_extra = []
        self.lista_epsilon_max_extra = []

        # Conta quante righe dinamiche sono state aggiunte
        self.contatore_righe = 0

        self.style_lineedit = """
            background-color: rgb(30,30,30);
            font: 10pt "Segoe UI";
            color: rgb(255, 255, 255);
            padding-left: 5px;
            border-radius:4px;  
        """

        # Connessione del pulsante
        self.ui.btn_acciaiobarre_aggiungi.clicked.connect(self.aggiungi_riga_acciaiobarre)
        self.ui.btn_acciaiobarre_rimuovi.clicked.connect(self.rimuovi_riga_acciaiobarre)

    
    def crea_lineedit_personalizzato(self, object_name):
        lineedit = QLineEdit()
        lineedit.setObjectName(object_name)
        lineedit.setStyleSheet(self.style_lineedit)
        lineedit.setMinimumHeight(32)
        return lineedit

    def aggiungi_riga_acciaiobarre(self):
        self.contatore_righe += 1
        indice = self.contatore_righe + 2  # Indici partono da 3

        # Crea i QLineEdit personalizzati
        nuovo_sigma = self.crea_lineedit_personalizzato(f"calcestruzzo_sigma_{indice}")
        nuovo_eps_min = self.crea_lineedit_personalizzato(f"calcestruzzo_epsilon_min_{indice}")
        nuovo_eps_max = self.crea_lineedit_personalizzato(f"calcestruzzo_epsilon_max_{indice}")

        # Aggiungi ai layout
        self.ui.layout_acciaiobarre_sigma.addWidget(nuovo_sigma)
        self.ui.layout_acciaiobarre_epsilon_min.addWidget(nuovo_eps_min)
        self.ui.layout_acciaiobarre_epsilon_max.addWidget(nuovo_eps_max)

        # Salva nelle liste
        self.lista_sigma_extra.append(nuovo_sigma)
        self.lista_epsilon_min_extra.append(nuovo_eps_min)
        self.lista_epsilon_max_extra.append(nuovo_eps_max)

    def rimuovi_riga_acciaiobarre(self):
        if self.contatore_righe > 0:
            # Rimuove gli ultimi QLineEdit dai layout e li elimina dalle liste
            ultimo_sigma = self.lista_sigma_extra.pop()
            ultimo_eps_min = self.lista_epsilon_min_extra.pop()
            ultimo_eps_max = self.lista_epsilon_max_extra.pop()

            self.ui.layout_acciaiobarre_sigma.removeWidget(ultimo_sigma)
            self.ui.layout_acciaiobarre_epsilon_min.removeWidget(ultimo_eps_min)
            self.ui.layout_acciaiobarre_epsilon_max.removeWidget(ultimo_eps_max)

            ultimo_sigma.deleteLater()
            ultimo_eps_min.deleteLater()
            ultimo_eps_max.deleteLater()

            self.contatore_righe -= 1

    def generatore_matrice_acciaiobarre(self):
        matrice = []

        # Prendi i due terzetti fissi (presumo acciaiobarre_1 e _2 esistano, ma non _3 come nella versione originale)
        valori_iniziali = [
            ((self.ui.acciaiobarre_sigma_1.text() or 0),
            float(self.ui.acciaiobarre_epsilon_min_1.text() or 0),
            float(self.ui.acciaiobarre_epsilon_max_1.text() or 0)),

            ((self.ui.acciaiobarre_sigma_2.text() or 0),
            float(self.ui.acciaiobarre_epsilon_min_2.text() or 0),
            float(self.ui.acciaiobarre_epsilon_max_2.text() or 0)),

            ((self.ui.acciaiobarre_sigma_3.text() or 0),
            float(self.ui.acciaiobarre_epsilon_min_3.text() or 0),
            float(self.ui.acciaiobarre_epsilon_max_3.text() or 0)
            )
        ]

        # Aggiungi solo le righe con valori validi
        for tripletta in valori_iniziali:
            if tripletta[1] != tripletta[2]:
                matrice.append(list(tripletta))

        # Aggiungi le righe dinamiche
        for sigma, eps_min, eps_max in zip(
            self.lista_sigma_extra,
            self.lista_epsilon_min_extra,
            self.lista_epsilon_max_extra
        ):
            try:
                riga = [
                    (sigma.text() or 0),
                    float(eps_min.text() or 0),
                    float(eps_max.text() or 0)
                ]
                if riga[1] != riga[2]:
                    matrice.append(riga)
            except ValueError:
                # Ignora righe con input non validi
                continue

        #print("Matrice Acciaio Barre Generata:", matrice)

        return matrice
