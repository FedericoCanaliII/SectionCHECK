"""
calcolo_pressoflessione.py
==========================
Motore di calcolo per la verifica a flessione / pressoflessione di sezioni in c.a.

Supporta:
  - SLU (Stati Limite Ultimo):  approccio a fibre con legami costitutivi non-lineari
                                 da NTC 2018 / EC2 (parabola-rettangolo calcestruzzo,
                                 elastoplastico acciaio)
  - SLE (Stati Limite di Esercizio): sezione fessurata, comportamento elastico lineare
                                      con iterazione sulla zona fessurata

Si appoggia alle classi SezioneRinforzata e Materiale definite in output/calcolo.py.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# Importa le classi base già presenti nel progetto
from output.calcolo import Materiale, SezioneRinforzata


# ==============================================================================
# COSTANTI
# ==============================================================================
_CONC_TENSION_LIMIT = 5e-3   # ult_tension < questo → calcestruzzo
_EPS_REL_TOL        = 1e-6   # tolleranza relativa per bisection
_BISECT_ITER        = 80     # max iterazioni bisection SLU
_SLE_MAX_ITER       = 60     # max iterazioni schema fessurativo


# ==============================================================================
# CLASSE PRINCIPALE
# ==============================================================================
class CalcoloPressoflessione:
    """
    Esegue la verifica a pressoflessione di una sezione RC per SLU e SLE.

    Parametri
    ---------
    sezione   : SezioneRinforzata già costruita e NON ancora allineata al centro.
    grid_step : passo della griglia di integrazione [mm].
    """

    def __init__(self, sezione: SezioneRinforzata, grid_step: float = 10.0) -> None:
        self.sezione   = sezione
        self.grid_step = grid_step

        # Allinea la sezione al suo baricentro geometrico (side-effect: traslazione)
        self.sezione.allinea_al_centro()

        # Prepara e cachéa i punti di integrazione
        self.punti: List[Tuple[float, float, float, Materiale]] = (
            self.sezione._prepara_punti_integrazione(grid_step)
        )

        # Classificazione automatica dei materiali (usata per SLE)
        self._E_cache: Dict[int, float] = {}

    # --------------------------------------------------------------------------
    # UTILITÀ GEOMETRICHE
    # --------------------------------------------------------------------------

    def _n_vect(self, theta_deg: float) -> np.ndarray:
        """
        Restituisce il vettore direzione-flessione (gradiente della deformazione)
        corrispondente ad un asse neutro inclinato di theta_deg rispetto all'orizzontale.

        Convenzione:
          theta_deg =  0°  → NA orizzontale  → n_vect = (0, 1)  (flessione attorno a X)
          theta_deg = 90°  → NA verticale    → n_vect = (1, 0)  (flessione attorno a Y)
        """
        rad = math.radians(theta_deg)
        return np.array([math.sin(rad), math.cos(rad)], dtype=float)

    def _proietta(self, n_vect: np.ndarray) -> Tuple[float, float, float]:
        """
        Proietta tutti i punti di integrazione sul vettore n_vect.
        Restituisce (d_min, d_max, H).
        """
        if not self.punti:
            return -100.0, 100.0, 200.0
        ds = np.array([x * n_vect[0] + y * n_vect[1]
                       for x, y, _a, _m in self.punti])
        d_min, d_max = float(ds.min()), float(ds.max())
        H = d_max - d_min if d_max > d_min else 100.0
        return d_min, d_max, H

    def _is_calcestruzzo(self, mat: Materiale) -> bool:
        """Discrimina calcestruzzo (ult_tension bassa) da acciaio."""
        return mat.ult_tension < _CONC_TENSION_LIMIT

    # --------------------------------------------------------------------------
    # KERNEL SLU: calcolo N e M per posizione asse neutro k
    # --------------------------------------------------------------------------

    def _NM_a_k(self,
                k:      float,
                n_vect: np.ndarray
                ) -> Tuple[float, float, List[Tuple[float, float, float, float, float]]]:
        """
        Calcola la coppia (N [kN], M [kNm]) alla posizione k dell'asse neutro
        usando il legame costitutivo SLU completo.

        Restituisce anche la lista fibre: [(x, y, area, eps, sigma), ...].

        Algoritmo:
          1. Fattore di scala m_g porta la fibra più sollecitata al limite del materiale.
          2. eps_i = m_g * (d_i - k)
          3. Integra sigma_i * A_i per ottenere N, Mx, My.
          4. Scalare: M = n_vect · (My, Mx).
        """
        m_g = self.sezione._calcola_fattore_scala_robusto(n_vect, k)

        N_N = 0.0   # [N]
        Mx  = 0.0   # [N·mm]
        My  = 0.0   # [N·mm]
        fibre: List[Tuple[float, float, float, float, float]] = []

        for x, y, area, mat in self.punti:
            d   = x * n_vect[0] + y * n_vect[1]
            eps = m_g * (d - k) if m_g > 1e-14 else 0.0
            sig = mat.sigma(eps)
            f   = sig * area
            N_N += f
            Mx  += f * y
            My  += f * x
            fibre.append((x, y, area, eps, sig))

        N_kN  = N_N / 1e3
        M_kNm = (n_vect[0] * My + n_vect[1] * Mx) / 1e6
        return N_kN, M_kNm, fibre

    def _N_solo(self, k: float, n_vect: np.ndarray) -> float:
        """Versione leggera (solo N) usata durante la bisection."""
        m_g = self.sezione._calcola_fattore_scala_robusto(n_vect, k)
        N_N = 0.0
        for x, y, area, mat in self.punti:
            d   = x * n_vect[0] + y * n_vect[1]
            eps = m_g * (d - k) if m_g > 1e-14 else 0.0
            N_N += mat.sigma(eps) * area
        return N_N / 1e3  # kN

    # --------------------------------------------------------------------------
    # SLU
    # --------------------------------------------------------------------------

    def analisi_slu(self,
                    N_Ed_kN:   float,
                    M_Ed_kNm:  float,
                    theta_deg: float = 0.0
                    ) -> Dict:
        """
        Verifica SLU a pressoflessione / flessione semplice (N_Ed = 0).

        Algoritmo:
          Per ogni posizione k dell'asse neutro → (N_k, M_k).
          Trova k* (bisection) tale che N(k*) = N_Ed.
          Confronta M(k*) = M_Rd con M_Ed.

        Parametri
        ---------
        N_Ed_kN  : sforzo normale di progetto [kN]  (negativo = compressione).
        M_Ed_kNm : momento flettente di progetto [kNm] (valore assoluto usato).
        theta_deg: rotazione asse neutro [°] dalla orizzontale.

        Restituisce
        -----------
        dict con tutti i risultati (vedi codice).
        """
        n_vect            = self._n_vect(theta_deg)
        d_min, d_max, H   = self._proietta(n_vect)

        # --- Definizione dell'intervallo di bisection ---
        # k << d_min → tutto in trazione   → N fortemente positivo
        # k >> d_max → tutto in compressione → N fortemente negativo
        span   = max(H * 15.0, 500.0)
        k_left  = d_min - span
        k_right = d_max + span

        N_l = self._N_solo(k_left,  n_vect)
        N_r = self._N_solo(k_right, n_vect)

        # Controllo che N_Ed sia nel range fisico
        N_min = min(N_l, N_r)
        N_max = max(N_l, N_r)
        fuori_dominio = (N_Ed_kN < N_min * 1.05) or (N_Ed_kN > N_max * 1.05)

        # Bisection
        # N(k) è decrescente: aumentando k si sposta la sezione verso la compressione
        k_lo = k_left
        k_hi = k_right

        # Assicuro che N_l > N_Ed > N_r  (o viceversa se entrambi positivi)
        # In pratica N_l > N_r → bisect standard
        k_star = (k_lo + k_hi) / 2.0
        conv   = False

        tol_N  = max(abs(N_Ed_kN) * 1e-4, 0.05)   # kN

        for _ in range(_BISECT_ITER):
            k_mid  = (k_lo + k_hi) / 2.0
            N_mid  = self._N_solo(k_mid, n_vect)
            err    = N_mid - N_Ed_kN

            if abs(err) < tol_N:
                k_star = k_mid
                conv   = True
                break

            # N decrescente con k
            if err > 0:
                k_lo = k_mid   # serve più compressione → sposta k verso destra
            else:
                k_hi = k_mid   # serve più trazione    → sposta k verso sinistra

            k_star = k_mid

        if not conv:
            k_star = (k_lo + k_hi) / 2.0

        # --- Calcolo completo al k* ---
        N_rd, M_rd, fibre = self._NM_a_k(k_star, n_vect)
        M_rd_abs = abs(M_rd)
        M_Ed_abs = abs(M_Ed_kNm)

        # Deformazioni alle fibre estreme
        m_g     = self.sezione._calcola_fattore_scala_robusto(n_vect, k_star)
        eps_top = m_g * (d_max - k_star) if m_g > 1e-14 else 0.0
        eps_bot = m_g * (d_min - k_star) if m_g > 1e-14 else 0.0

        # Classifico le barre per report separato
        barre = [f for f in fibre
                 if not self._is_calcestruzzo(
                     self.punti[fibre.index(f)][3]
                 )] if fibre else []

        # --- Verifica ---
        verificata = (not fuori_dominio) and (M_rd_abs >= M_Ed_abs)
        rapporto   = M_Ed_abs / M_rd_abs if M_rd_abs > 1e-9 else float('inf')

        return {
            'tipo'          : 'SLU',
            'verificata'    : verificata,
            'fuori_dominio' : fuori_dominio,
            'N_Ed'          : N_Ed_kN,
            'M_Ed'          : M_Ed_kNm,
            'theta_deg'     : theta_deg,
            'M_Rd'          : M_rd_abs,
            'N_Rd'          : N_rd,
            'rapporto_MEd_MRd': rapporto,
            'k_star'        : k_star,
            'd_na'          : k_star,          # posizione NA lungo n_vect [mm]
            'n_vect'        : tuple(n_vect),
            'eps_top'       : eps_top,
            'eps_bot'       : eps_bot,
            'fibre'         : fibre,           # (x, y, area, eps, sigma)
            'grid_step'     : self.grid_step,
            'sezione_bounds': (self.sezione.x_min, self.sezione.x_max,
                               self.sezione.y_min, self.sezione.y_max),
            'd_min'         : d_min,
            'd_max'         : d_max,
        }

    # --------------------------------------------------------------------------
    # SLE – moduli elastici
    # --------------------------------------------------------------------------

    def _stima_E(self, mat: Materiale) -> float:
        """
        Stima il modulo elastico dal pendio iniziale della curva sigma-epsilon.
        Usa la cache per non ricalcolare lo stesso materiale più volte.
        """
        mid = id(mat)
        if mid in self._E_cache:
            return self._E_cache[mid]

        try:
            if self._is_calcestruzzo(mat):
                delta = 5e-5      # piccola deformazione in compressione
                E = abs(mat.sigma(-delta)) / delta
                if E < 1000:     # fallback su deformazione più grande
                    E = abs(mat.sigma(-5e-4)) / 5e-4
                E = max(E, 5_000.0)
            else:
                delta = 5e-5
                E = abs(mat.sigma(delta)) / delta
                if E < 1000:
                    E = abs(mat.sigma(5e-4)) / 5e-4
                E = max(E, 50_000.0)
        except Exception:
            E = 30_000.0 if self._is_calcestruzzo(mat) else 200_000.0

        self._E_cache[mid] = E
        return E

    # --------------------------------------------------------------------------
    # SLE
    # --------------------------------------------------------------------------

    def analisi_sle(self,
                    N_Ed_kN:   float,
                    M_Ed_kNm:  float,
                    theta_deg: float = 0.0
                    ) -> Dict:
        """
        Analisi SLE – sezione fessurata, comportamento elastico lineare.

        Ipotesi:
          - Calcestruzzo in trazione: tensione = 0 (sezione fessurata, fase II)
          - Piano delle sezioni rimane piano (Bernoulli-Navier)
          - eps_i = eps_0 + kappa * d_i   (d_i = proiezione lungo n_vect)

        Iterazione sull'insieme delle fibre fessurate fino a convergenza.

        Parametri
        ---------
        N_Ed_kN  : sforzo normale [kN]
        M_Ed_kNm : momento flettente [kNm]
        theta_deg: rotazione asse neutro [°]

        Restituisce
        -----------
        dict con tensioni, deformazioni e stato di verifica approssimativo.
        """
        n_vect          = self._n_vect(theta_deg)
        d_min, d_max, H = self._proietta(n_vect)

        # Conversioni in unità base [N, N·mm]
        N_Ed_N   = N_Ed_kN   * 1e3
        M_Ed_Nmm = M_Ed_kNm  * 1e6

        # Moduli elastici per ogni fibra (costanti durante l'iterazione)
        E_base: List[float] = [self._stima_E(mat) for _x, _y, _a, mat in self.punti]

        # Stato di fessurazione: True = fibra cls in trazione → contributo zero
        cracked: List[bool] = [False] * len(self.punti)

        eps0  = 0.0
        kappa = 0.0

        for it in range(_SLE_MAX_ITER):
            # ---- Costruzione sistema 2×2 ----
            # eps_i = eps0 + kappa * d_i
            # N_Ed = Σ E_i·A_i·(eps0 + kappa·d_i)
            # M_Ed = Σ E_i·A_i·(eps0 + kappa·d_i)·d_i
            #
            # → [SEA   SEAd ] [eps0 ] = [N_Ed]
            #   [SEAd  SEAd²] [kappa]   [M_Ed]

            SEA   = 0.0
            SEAd  = 0.0
            SEAd2 = 0.0

            for i, (x, y, area, mat) in enumerate(self.punti):
                is_conc = self._is_calcestruzzo(mat)
                if is_conc and cracked[i]:
                    continue        # fibra cls fessurata: contributo nullo
                Ei  = E_base[i]
                d   = x * n_vect[0] + y * n_vect[1]
                EiA = Ei * area
                SEA   += EiA
                SEAd  += EiA * d
                SEAd2 += EiA * d * d

            # Solve
            det = SEA * SEAd2 - SEAd * SEAd
            if abs(det) < 1e-8:
                # Sezione singolare (solo compressione pura o vuota)
                # Calcola stima grezza con solo eps0
                if abs(SEA) > 1e-8:
                    eps0  = N_Ed_N / SEA
                    kappa = 0.0
                else:
                    eps0  = 0.0
                    kappa = 0.0
            else:
                eps0  = ( SEAd2 * N_Ed_N - SEAd  * M_Ed_Nmm) / det
                kappa = (-SEAd  * N_Ed_N + SEA   * M_Ed_Nmm) / det

            # ---- Aggiornamento mappa fessure ----
            new_cracked = [False] * len(self.punti)
            changed     = False

            for i, (x, y, area, mat) in enumerate(self.punti):
                if not self._is_calcestruzzo(mat):
                    continue       # acciaio: non si fessura
                d       = x * n_vect[0] + y * n_vect[1]
                eps_i   = eps0 + kappa * d
                fess    = eps_i > 1e-8  # fibra in trazione → fessurata
                new_cracked[i] = fess
                if fess != cracked[i]:
                    changed = True

            cracked = new_cracked
            if not changed:
                break   # convergenza raggiunta

        # ---- Calcolo tensioni finali ----
        fibre:            List[Tuple] = []
        sigma_c_min:      float = 0.0   # max compressione cls (negativo)
        sigma_c_max:      float = 0.0   # min compressione cls (vicino a 0, negativo)
        sigma_s_max_traz: float = 0.0   # max tensione acciaio
        sigma_s_max_comp: float = 0.0   # max compressione acciaio (negativo)

        for i, (x, y, area, mat) in enumerate(self.punti):
            d     = x * n_vect[0] + y * n_vect[1]
            eps_i = eps0 + kappa * d
            is_conc = self._is_calcestruzzo(mat)

            if is_conc and cracked[i]:
                sig_i = 0.0
            else:
                sig_i = E_base[i] * eps_i

            fibre.append((x, y, area, eps_i, sig_i))

            if is_conc and not cracked[i]:
                sigma_c_min = min(sigma_c_min, sig_i)
                sigma_c_max = max(sigma_c_max, sig_i)
            elif not is_conc:
                sigma_s_max_traz = max(sigma_s_max_traz, sig_i)
                sigma_s_max_comp = min(sigma_s_max_comp, sig_i)

        # Posizione asse neutro (dove eps = 0)
        if abs(kappa) > 1e-15:
            d_na = -eps0 / kappa
        else:
            d_na = (d_min + d_max) / 2.0

        eps_top = eps0 + kappa * d_max
        eps_bot = eps0 + kappa * d_min

        # ---- Verifica SLE approssimativa (EC2 / NTC 2018) ----
        # Stima fck dal materiale calcestruzzo: fcd ≈ sigma a eps_cu, fck = fcd·γc/αcc
        mat_conc = next((m for _x, _y, _a, m in self.punti
                         if self._is_calcestruzzo(m)), None)
        mat_acc  = next((m for _x, _y, _a, m in self.punti
                         if not self._is_calcestruzzo(m)), None)

        sigma_c_limit  = None
        sigma_s_limit  = None
        fck_approx     = None
        fyk_approx     = None
        note_verifica: List[str] = []

        if mat_conc is not None:
            try:
                # Stima fcd dal valore di plateau della curva
                eps_cu = mat_conc.ult_compression * 0.95
                fcd_approx = abs(mat_conc.sigma(eps_cu))
                fck_approx = fcd_approx * 1.5 / 0.85   # invertendo gamma_c e alpha_cc
                sigma_c_limit = 0.6 * fck_approx        # combinazione rara EC2 7.2(2)
            except Exception:
                sigma_c_limit = None

        if mat_acc is not None:
            try:
                # Stima fyd: sigma all'inizio del plateau
                eps_half = mat_acc.ult_tension * 0.40
                fyd_approx   = mat_acc.sigma(eps_half)
                fyk_approx   = fyd_approx * 1.15        # γs = 1.15
                sigma_s_limit = 0.8 * fyk_approx        # combinazione rara EC2 7.2(5)
            except Exception:
                sigma_s_limit = None

        verificata = True

        if sigma_c_limit is not None and abs(sigma_c_min) > sigma_c_limit:
            verificata = False
            note_verifica.append(
                f"σ_c = {abs(sigma_c_min):.1f} MPa  >  0.6·fck = {sigma_c_limit:.1f} MPa"
            )

        if sigma_s_limit is not None and sigma_s_max_traz > sigma_s_limit:
            verificata = False
            note_verifica.append(
                f"σ_s = {sigma_s_max_traz:.1f} MPa  >  0.8·fyk = {sigma_s_limit:.1f} MPa"
            )

        return {
            'tipo'              : 'SLE',
            'verificata'        : verificata,
            'note'              : note_verifica,
            'N_Ed'              : N_Ed_kN,
            'M_Ed'              : M_Ed_kNm,
            'theta_deg'         : theta_deg,
            'eps0'              : eps0,
            'kappa'             : kappa,
            'eps_top'           : eps_top,
            'eps_bot'           : eps_bot,
            'd_na'              : d_na,
            'n_vect'            : tuple(n_vect),
            'sigma_c_compr_max' : abs(sigma_c_min),     # MPa (positivo)
            'sigma_c_compr_min' : sigma_c_min,           # MPa (negativo)
            'sigma_s_traz_max'  : sigma_s_max_traz,      # MPa
            'sigma_s_comp_max'  : sigma_s_max_comp,      # MPa (negativo)
            'fck_approx'        : fck_approx,
            'fyk_approx'        : fyk_approx,
            'sigma_c_limit'     : sigma_c_limit,
            'sigma_s_limit'     : sigma_s_limit,
            'fibre'             : fibre,                 # (x, y, area, eps, sigma)
            'grid_step'         : self.grid_step,
            'sezione_bounds'    : (self.sezione.x_min, self.sezione.x_max,
                                   self.sezione.y_min, self.sezione.y_max),
            'd_min'             : d_min,
            'd_max'             : d_max,
        }
