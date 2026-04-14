"""
calcolo.py  –  analisi/dominio_nm/
====================================
Motore di calcolo del dominio di interazione N-M 3D (SLU).

Calcola la superficie di interazione N-Mx-My ruotando l'asse neutro
di 360° (theta_steps angoli) e traslando la sua posizione da -∞ a +∞
(neutral_steps posizioni) per ogni angolo.

Strategia Ottimizzata:
  - Riusa SezioneDiscretizzata / LegameCostitutivo / Fibra dalla pressoflessione.
  - Centra le fibre sul baricentro geometrico per stabilità numerica.
  - Ricerca "robusta" della curvatura di collasso per evitare instabilità.
  - Calcolo isolato dei punti di trazione/compressione pura (chiusura del dominio)
    assegnati direttamente ai poli del "cilindro" 3D.
  - Calcolo parallelo via ThreadPoolExecutor applicato SOLO ai punti intermedi
    di flessione per massimizzare le performance.
"""
from __future__ import annotations

import concurrent.futures
import math
import time
from typing import List, Tuple

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from analisi.pressoflessione.calcolo import (
    Fibra,
    LegameCostitutivo,
    SezioneDiscretizzata,
)


# ==============================================================================
# CALCOLATORE DEL DOMINIO
# ==============================================================================

class CalcoloDominioNM:
    """
    Calcola il dominio di interazione N-Mx-My (SLU) su una sezione discretizzata.
    """

    def __init__(self, sezione: SezioneDiscretizzata) -> None:
        self.sezione = sezione

        # Centra le coordinate sul baricentro geometrico (stabilità numerica)
        cx, cy = sezione.centroide()
        fibre = sezione.fibre

        # Array numpy per operazioni vettorizzate nel loop interno
        self._fx    = np.array([f.x - cx for f in fibre], dtype=np.float64)
        self._fy    = np.array([f.y - cy for f in fibre], dtype=np.float64)
        self._fa    = np.array([f.area   for f in fibre], dtype=np.float64)
        self._eps_min = np.array([f.legame_slu.eps_min for f in fibre], dtype=np.float64)
        self._eps_max = np.array([f.legame_slu.eps_max for f in fibre], dtype=np.float64)
        self._legami: List[LegameCostitutivo] = [f.legame_slu for f in fibre]

    # ------------------------------------------------------------------
    # METODI DI CALCOLO INTERNI (STABILITÀ E PRECISIONE)
    # ------------------------------------------------------------------

    def _calcola_fattore_scala_robusto(self, n_vect: np.ndarray, k: float) -> float:
        """
        Determina la curvatura 'm' al collasso in modo robusto.
        Trova la prima fibra che raggiunge il proprio limite di deformazione.
        """
        d = self._fx * n_vect[0] + self._fy * n_vect[1]
        dist = d - k
        
        m_govern = float('inf')

        # Lato trazione (fibra si allunga, dist > 0)
        mask_t = (dist > 1e-6) & (self._eps_max > 1e-9)
        if np.any(mask_t):
            m_t = float(np.min(self._eps_max[mask_t] / dist[mask_t]))
            if m_t < m_govern:
                m_govern = m_t

        # Lato compressione (fibra si accorcia, dist < 0)
        mask_c = (dist < -1e-6) & (self._eps_min < -1e-9)
        if np.any(mask_c):
            m_c = float(np.min(self._eps_min[mask_c] / dist[mask_c]))
            if m_c < m_govern:
                m_govern = m_c

        return m_govern if m_govern != float('inf') else 0.0

    def _calcola_punto(self, params: Tuple) -> List[float]:
        """
        Calcola il punto (Mx, My, N) del dominio per un dato vettore normale
        e una data posizione dell'asse neutro. (Adattato per i Thread worker)
        """
        theta, k = params
        n_vect = np.array([math.cos(theta), math.sin(theta)])
        
        m_govern = self._calcola_fattore_scala_robusto(n_vect, k)
        
        # Se la curvatura è trascurabile, restituisce zero per evitare artefatti
        if m_govern <= 1e-9:
            return [0.0, 0.0, 0.0]

        d     = self._fx * n_vect[0] + self._fy * n_vect[1]
        eps_v = m_govern * (d - k)

        N = Mx = My = 0.0
        for eps, legame, area, fy, fx in zip(eps_v, self._legami,
                                             self._fa, self._fy, self._fx):
            sig = legame.sigma(float(eps))
            if sig != 0.0:
                f   = sig * area
                N  += f
                Mx += f * fy
                My += f * fx

        return [Mx / 1e6, My / 1e6, N / 1e3]   # kNm, kNm, kN

    def calcola_punto_assiale_puro(self, modo: str) -> List[float]:
        """
        Calcola N, Mx, My per deformazione uniforme (curvatura 0).
        Ricerca le deformazioni ultime massime, aggiungendo fallback di sicurezza.
        """
        epsilon_limite = 0.0
        
        if modo == 'compressione':
            limiti = [eps for eps in self._eps_min if eps < -1e-5]
            if limiti:
                epsilon_limite = max(limiti)  # Es. max(-0.0035, -0.01) = -0.0035
            else:
                epsilon_limite = -0.002       # Fallback di sicurezza
        else: # trazione
            limiti = [eps for eps in self._eps_max if eps > 1e-4]
            if limiti:
                epsilon_limite = min(limiti)
            else:
                epsilon_limite = 0.0          # Fallback se non c'è resistenza a trazione

        N = Mx = My = 0.0
        for legame, area, fx, fy in zip(self._legami, self._fa, self._fx, self._fy):
            sig = legame.sigma(epsilon_limite)
            if sig != 0.0:
                f   = sig * area
                N  += f
                Mx += f * fy
                My += f * fx

        return [Mx / 1e6, My / 1e6, N / 1e3]

    def _genera_step_asse_neutro_strutturato(self, n_vect: np.ndarray, n_steps: int) -> List[float]:
        """
        Genera posizioni 'k' per l'asse neutro con distribuzione ottimizzata
        sulla geometria della sezione (addensata internamente, logaritmica esternamente).
        """
        d = self._fx * n_vect[0] + self._fy * n_vect[1]
        
        if len(d) == 0:
            return np.linspace(-100, 100, n_steps).tolist()
            
        d_min = float(np.min(d))
        d_max = float(np.max(d))
        H     = d_max - d_min if d_max != d_min else 1e-3

        n_inner = max(3, int(round(0.70 * n_steps)))
        n_outer = max(0, (n_steps - n_inner) // 2)

        c_inner = np.linspace(d_min + 0.001 * H, d_max - 0.001 * H, n_inner)

        if n_outer > 0:
            factors  = np.geomspace(0.01, 50.0, n_outer)
            c_lower  = d_min - factors * H
            c_upper  = d_max + factors * H
            c_total  = np.concatenate([np.flip(c_lower), c_inner, c_upper])
        else:
            c_total = c_inner

        # Ricampionamento esatto
        if len(c_total) != n_steps:
            c_total = np.interp(
                np.linspace(0.0, 1.0, n_steps),
                np.linspace(0.0, 1.0, len(c_total)),
                c_total,
            )
            
        return list(c_total)


# ==============================================================================
# THREAD DI CALCOLO
# ==============================================================================

class _DominioThread(QThread):
    """Calcola il dominio 3D su un thread separato, con struttura ottimizzata."""

    avanzamento = pyqtSignal(int)           # 0-100 %
    completato  = pyqtSignal(object)        # np.ndarray (theta, steps, 3)
    errore      = pyqtSignal(str)

    def __init__(self,
                 calcolatore:   CalcoloDominioNM,
                 theta_steps:   int,
                 neutral_steps: int,
                 parent=None) -> None:
        super().__init__(parent)
        self._calc          = calcolatore
        self._theta_steps   = max(4, theta_steps)
        self._neutral_steps = max(3, neutral_steps) # Almeno 3 per flessione
        self._stop_flag     = False

    def richiedi_stop(self) -> None:
        self._stop_flag = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            t0      = time.perf_counter()
            calc    = self._calc
            t_steps = self._theta_steps
            n_steps = self._neutral_steps

            thetas  = np.linspace(0.0, 2.0 * math.pi, t_steps, endpoint=False)
            results = np.zeros((t_steps, n_steps, 3), dtype=np.float64)

            # ── 1. Punti assiali puri (Chiusura del dominio) ────────────
            # Computati una sola volta per ottimizzazione
            pt_traz = calc.calcola_punto_assiale_puro('trazione')
            pt_comp = calc.calcola_punto_assiale_puro('compressione')

            # ── 2. Assegnazione Poli alla Matrice ───────────────────────
            for i in range(t_steps):
                results[i, 0,  :] = pt_traz     # Faccia superiore del cilindro (Top)
                results[i, -1, :] = pt_comp     # Faccia inferiore del cilindro (Bottom)

            # ── 3. Setup Loop Parallelo (Solo Bending Intermedio) ───────
            indices_to_compute = range(1, n_steps - 1)
            total_bending_tasks = t_steps * len(indices_to_compute)
            completed = 0

            if total_bending_tasks > 0:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_map = {}
                    
                    for i, theta in enumerate(thetas):
                        n_vect = np.array([math.cos(theta), math.sin(theta)])
                        
                        # Generiamo gli step neutri e prendiamo solo quelli centrali
                        c_full = calc._genera_step_asse_neutro_strutturato(n_vect, n_steps)
                        c_bending = c_full[1:-1]
                        
                        for k_idx, k in enumerate(c_bending):
                            real_j_index = k_idx + 1 # Offset di 1 per i poli
                            
                            params = (theta, k)
                            future = executor.submit(calc._calcola_punto, params)
                            future_map[future] = (i, real_j_index)

                    for future in concurrent.futures.as_completed(future_map):
                        if self._stop_flag:
                            executor.shutdown(wait=False, cancel_futures=True)
                            return

                        i, j = future_map[future]
                        try:
                            results[i, j, :] = future.result()
                        except Exception as exc:
                            print(f"WARN  dominio NM: punto ({i},{j}) → {exc}")

                        completed += 1
                        if completed % 15 == 0 or completed == total_bending_tasks:
                            self.avanzamento.emit(
                                int(completed / total_bending_tasks * 100)
                            )

            print(f">> Dominio N-M calcolato in "
                  f"{time.perf_counter() - t0:.2f} s  "
                  f"({t_steps}θ × {n_steps}k = {t_steps * n_steps} punti)")
                  
            self.avanzamento.emit(100)
            self.completato.emit(results)

        except Exception as exc:
            self.errore.emit(str(exc))