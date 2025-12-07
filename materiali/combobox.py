from PyQt5 import QtCore

class Combobox_matriali:
    def __init__(self, ui):
        self.ui = ui

        #---------------------------- CALCESTRUZZO
        #premi il pulsante c25/30 quando apro il programma dopo un secondo
        QtCore.QTimer.singleShot(1000, lambda: self.ui.calcestruzzo_combobox.setCurrentText("C 25/30"))

        self.ui.calcestruzzo_combobox.currentTextChanged.connect(self.aggiorna_calcestruzzo)
        self.ui.calcestruzzo_gamma.textChanged.connect(self.aggiorna_calcestruzzo)
        self.ui.calcestruzzo_alpha.textChanged.connect(self.aggiorna_calcestruzzo)

        #----------------------------- ACCIAIO BARRE
        self.ui.acciaiobarre_combobox.setCurrentText("B 500C")
        QtCore.QTimer.singleShot(1000, lambda: self.ui.acciaiobarre_combobox.setCurrentText("B 450C"))

        self.ui.acciaiobarre_combobox.currentTextChanged.connect(self.aggiorna_acciaio_barre)
        self.ui.acciaiobarre_gamma.textChanged.connect(self.aggiorna_acciaio_barre)

        #---------------------------- ACCIAIO PROFILI
        QtCore.QTimer.singleShot(1000, lambda: self.ui.acciaioprofili_combobox.setCurrentText("S 275"))

        self.ui.acciaioprofili_combobox.currentTextChanged.connect(self.aggiorna_acciaio_profili)
        self.ui.acciaioprofili_gamma.textChanged.connect(self.aggiorna_acciaio_profili)

        #---------------------------- FRP
        QtCore.QTimer.singleShot(1000, lambda: self.ui.frp_combobox.setCurrentText("Vetro Structural"))

        self.ui.frp_combobox.currentTextChanged.connect(self.aggiorna_frp)
        self.ui.frp_gamma.textChanged.connect(self.aggiorna_frp)



    def aggiorna_calcestruzzo(self, *_):  # Usa *_, per ignorare argomenti extra da segnali
        classe_resistenza = self.ui.calcestruzzo_combobox.currentText()
        if classe_resistenza == "C 16/20":
            f_ck = 16
        elif classe_resistenza == "C 20/25":
            f_ck = 20
        if classe_resistenza == "C 25/30":
            f_ck = 25
        elif classe_resistenza == "C 30/37":
            f_ck = 30
        elif classe_resistenza == "C 35/45":
            f_ck = 35
        elif classe_resistenza == "C 40/50":
            f_ck = 40
        elif classe_resistenza == "C 45/55":
            f_ck = 45
        elif classe_resistenza == "C 50/60":
            f_ck = 50
        
        gamma = float(self.ui.calcestruzzo_gamma.text() or 1)
        alpha = float(self.ui.calcestruzzo_alpha.text() or 1)
        f_cd = f_ck * alpha / gamma

        self.ui.calcestruzzo_sigma_1.setText(f'{f_cd:.2f} * (1 - (1 - x / 0.002)**2)')
        self.ui.calcestruzzo_sigma_2.setText(f'{f_cd:.2f}')

    def aggiorna_acciaio_barre(self, *_):
        classe_resistenza = self.ui.acciaiobarre_combobox.currentText()
        if classe_resistenza == "B 450C":
            f_yk = 450
            epsilon_max = 0.075
        elif classe_resistenza == "B 500B":
            f_yk = 500
            epsilon_max = 0.05
        elif classe_resistenza == "B 500C":
            f_yk = 500
            epsilon_max = 0.075


        gamma = float(self.ui.acciaiobarre_gamma.text() or 1)
        f_y = f_yk / gamma
        epsilon_snervamento=f_y / 210000

        self.ui.acciaiobarre_sigma_2.setText(f'{f_y:.2f}')
        self.ui.acciaiobarre_sigma_3.setText(f'{-f_y:.2f}')
        self.ui.acciaiobarre_epsilon_min_1.setText(f'{-epsilon_snervamento:.6f}')
        self.ui.acciaiobarre_epsilon_max_1.setText(f'{+epsilon_snervamento:.6f}')
        self.ui.acciaiobarre_epsilon_min_2.setText(f'{+epsilon_snervamento:.6f}')
        self.ui.acciaiobarre_epsilon_max_2.setText(f'{epsilon_max:.6f}')
        self.ui.acciaiobarre_epsilon_min_3.setText(f'{-epsilon_max:.6f}')
        self.ui.acciaiobarre_epsilon_max_3.setText(f'{-epsilon_snervamento:.6f}')

    def aggiorna_acciaio_profili(self, *_):
        classe_resistenza = self.ui.acciaioprofili_combobox.currentText()
        if classe_resistenza == "S 235":
            f_yk = 235
            f_uk = 360
        elif classe_resistenza == "S 275":
            f_yk = 275
            f_uk = 430
        elif classe_resistenza == "S 355":
            f_yk = 355
            f_uk = 510
        elif classe_resistenza == "S 420":
            f_yk = 420
            f_uk = 600
        elif classe_resistenza == "S 460":
            f_yk = 460
            f_uk = 690

        gamma = float(self.ui.acciaioprofili_gamma.text() or 1)
        epsilon_max = 0.15
        f_y = f_yk / gamma
        f_u = f_uk / gamma
        epsilon_snervamento = f_y / 210000

        #retta passante per due punti
        m= (f_u - f_y) / (epsilon_max - epsilon_snervamento)
        q= f_y - m * epsilon_snervamento


        self.ui.acciaioprofili_sigma_2.setText(f'{m:.2f} * x + {q:.2f}')
        self.ui.acciaioprofili_sigma_3.setText(f'{m:.2f} * x - {q:.2f}')
        self.ui.acciaioprofili_epsilon_min_1.setText(f'{-epsilon_snervamento:.6f}')
        self.ui.acciaioprofili_epsilon_max_1.setText(f'{+epsilon_snervamento:.6f}')
        self.ui.acciaioprofili_epsilon_min_2.setText(f'{+epsilon_snervamento:.6f}')
        self.ui.acciaioprofili_epsilon_max_3.setText(f'{-epsilon_snervamento:.6f}')

    def aggiorna_frp(self, *_):
        classe_resistenza = self.ui.frp_combobox.currentText()
        if classe_resistenza == "Vetro Electrical":
            modulo = 70000
            epsilon_max = -0.035

        elif classe_resistenza == "Vetro Structural":
            modulo = 85000
            epsilon_max = -0.045

        elif classe_resistenza == "Aramide Kevlar":
            modulo = 100000
            epsilon_max = -0.019

        elif classe_resistenza == "Carbono Alto Modulo":
            modulo = 390000
            epsilon_max = -0.005

        elif classe_resistenza == "Carbono Alta Resistenza":
            modulo = 230000
            epsilon_max = -0.016

        gamma = float(self.ui.frp_gamma.text() or 1)
        modulo_effettivo = modulo / gamma

        self.ui.frp_sigma_1.setText(f'{modulo_effettivo:.2f} * x')
        self.ui.frp_epsilon_max_1.setText(f'{epsilon_max:.6f}')

        

