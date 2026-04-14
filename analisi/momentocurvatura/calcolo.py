"""
calcolo.py  –  analisi/momentocurvatura/
=========================================
Motore di calcolo del diagramma Momento-Curvatura 3D (SLU).

Calcola la superficie M-χ ruotando la direzione di carico di 360°
(n_angoli step) e per ogni direzione incrementa la curvatura fino
a rottura (n_punti step per ramo).

Strategia:
  - Riusa SezioneDiscretizzata / LegameCostitutivo / Fibra dalla pressoflessione.
  - Centra le fibre sul baricentro geometrico per stabilità numerica.
  - Solver biassiale: itera su β (angolo asse neutro) finché il momento
    risultante è parallelo alla direzione di carico, e su ε₀ (deformazione
    al baricentro) per equilibrio assiale N = N_target.
  - Bisezione precisa del punto di rottura quando una fibra raggiunge il
    proprio limite di deformazione.
  - Calcolo parallelo via ThreadPoolExecutor sui rami angolari.
"""
from __future__ import annotations

import concurrent.futures
import math
import time
from typing import List

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from analisi.pressoflessione.calcolo import (
    Fibra,
    LegameCostitutivo,
    SezioneDiscretizzata,
)


# ==============================================================================
# CALCOLATORE MOMENTO-CURVATURA
# ==============================================================================

class CalcoloMomentoCurvatura:
    """
    Prepara i dati della sezione in forma vettoriale per il calcolo
    del diagramma momento-curvatura biassiale.
    """

    def __init__(self, sezione: SezioneDiscretizzata) -> None:
        self.sezione = sezione

        # Centra su baricentro
        cx, cy = sezione.centroide()
        fibre = sezione.fibre

        self._fx = np.array([f.x - cx for f in fibre], dtype=np.float64)
        self._fy = np.array([f.y - cy for f in fibre], dtype=np.float64)
        self._fa = np.array([f.area   for f in fibre], dtype=np.float64)
        self._legami: List[LegameCostitutivo] = [f.legame_slu for f in fibre]

        # Limiti di deformazione per ciascuna fibra
        self._eps_min = np.array([f.legame_slu.eps_min for f in fibre], dtype=np.float64)
        self._eps_max = np.array([f.legame_slu.eps_max for f in fibre], dtype=np.float64)

    # ------------------------------------------------------------------
    # SIGMA VETTORIZZATA
    # ------------------------------------------------------------------

    def _sigma_vec(self, strains: np.ndarray) -> np.ndarray:
        """Calcola σ per tutte le fibre dato il vettore di deformazioni."""
        sigmas = np.zeros_like(strains)
        for i, (eps, legame) in enumerate(zip(strains, self._legami)):
            sigmas[i] = legame.sigma(float(eps))
        return sigmas

    # ------------------------------------------------------------------
    # EQUILIBRIO ASSIALE (Newton-Raphson + Bisezione fallback)
    # ------------------------------------------------------------------

    def _solve_axial_equilibrium(self, chi: float, d_arr: np.ndarray,
                                  N_target: float, eps0_guess: float) -> float:
        """Trova ε₀ tale che N_totale = N_target per curvatura χ e proiezioni d."""
        def residual(e0):
            strains = e0 - chi * d_arr
            sigmas = self._sigma_vec(strains)
            return float(np.sum(sigmas * self._fa)) - N_target

        # Newton-Raphson
        e_curr = eps0_guess
        for _ in range(20):
            res = residual(e_curr)
            if abs(res) < 10.0:
                return e_curr
            delta = 1e-6
            stiff = (residual(e_curr + delta) - res) / delta
            if abs(stiff) < 1e-3:
                break
            e_curr -= res / stiff

        # Fallback bisezione
        low, high = eps0_guess - 0.1, eps0_guess + 0.1
        f_low, f_high = residual(low), residual(high)
        if f_low * f_high > 0:
            low, high = -0.5, 0.5
            f_low = residual(low)
        for _ in range(50):
            mid = (low + high) / 2.0
            f_mid = residual(mid)
            if abs(f_mid) < 10.0 or (high - low) < 1e-8:
                return mid
            if f_low * f_mid < 0:
                high = mid
            else:
                low, f_low = mid, f_mid
        return (low + high) / 2.0

    # ------------------------------------------------------------------
    # MOMENTI RISULTANTI
    # ------------------------------------------------------------------

    def _calculate_Mx_My(self, strains: np.ndarray):
        """Restituisce (Mx, My) in N·mm."""
        sigmas = self._sigma_vec(strains)
        forces = sigmas * self._fa
        Mx = float(np.sum(forces * self._fy))
        My = float(np.sum(forces * self._fx))
        return Mx, My

    # ------------------------------------------------------------------
    # SOLVER BIASSIALE
    # ------------------------------------------------------------------

    def _solve_biaxial_state(self, chi: float, target_angle: float,
                              N_target: float, guess_eps0: float,
                              guess_beta: float):
        """
        Trova (ε₀, β) tali che:
          - N = N_target  (equilibrio assiale)
          - Il momento risultante è parallelo alla direzione target_angle

        Restituisce (eps0, beta, M_kNm, strains).
        """
        def compute_error(beta_trial):
            cx = math.cos(beta_trial)
            cy = math.sin(beta_trial)
            d_arr = self._fx * cx + self._fy * cy

            eps0 = self._solve_axial_equilibrium(chi, d_arr, N_target, guess_eps0)
            strains = eps0 - chi * d_arr
            Mx, My = self._calculate_Mx_My(strains)

            # Errore perpendicolare alla direzione target
            t_sin = math.sin(target_angle)
            t_cos = math.cos(target_angle)
            error_perp = -Mx * t_sin + My * t_cos

            return error_perp, eps0, strains, math.sqrt(Mx**2 + My**2)

        # Metodo della secante sull'angolo β
        beta_curr = guess_beta
        err_curr, e0_curr, strains_curr, M_curr = compute_error(beta_curr)

        if abs(err_curr) < 1e-3:
            return e0_curr, beta_curr, M_curr / 1e6, strains_curr

        beta_prev = beta_curr - 0.01
        err_prev, _, _, _ = compute_error(beta_prev)

        for _ in range(30):
            if abs(err_curr - err_prev) < 1e-9:
                break

            beta_new = beta_curr - err_curr * (beta_curr - beta_prev) / (err_curr - err_prev)
            err_new, e0_new, s_new, M_new = compute_error(beta_new)

            if abs(err_new) < 1e-3:
                return e0_new, beta_new, M_new / 1e6, s_new

            beta_prev, err_prev = beta_curr, err_curr
            beta_curr, err_curr = beta_new, err_new
            e0_curr, strains_curr, M_curr = e0_new, s_new, M_new

        return e0_curr, beta_curr, M_curr / 1e6, strains_curr

    # ------------------------------------------------------------------
    # CHECK ROTTURA
    # ------------------------------------------------------------------

    def _check_rupture(self, strains: np.ndarray):
        """Verifica se una fibra ha superato il proprio limite di deformazione."""
        mask_tens = strains > 0
        mask_comp = strains < 0
        max_ratio = 0.0

        valid_t = mask_tens & (self._eps_max > 1e-6)
        if np.any(valid_t):
            max_ratio = max(max_ratio, float(np.max(
                strains[valid_t] / self._eps_max[valid_t]
            )))

        valid_c = mask_comp & (self._eps_min < -1e-6)
        if np.any(valid_c):
            max_ratio = max(max_ratio, float(np.max(
                strains[valid_c] / self._eps_min[valid_c]
            )))

        return max_ratio, (max_ratio >= 1.0)

    # ------------------------------------------------------------------
    # BISEZIONE PUNTO DI ROTTURA
    # ------------------------------------------------------------------

    def _bisect_failure(self, chi_safe: float, chi_fail: float,
                         load_angle: float, N_target: float,
                         state_safe: dict):
        """Trova con bisezione il χ esatto di rottura."""
        low = chi_safe
        high = chi_fail
        best_M = 0.0
        best_chi = low

        curr_eps0 = state_safe['eps0']
        curr_beta = state_safe['beta']

        for _ in range(20):
            mid = (low + high) / 2.0
            e0, beta, M_val, strs = self._solve_biaxial_state(
                mid, load_angle, N_target, curr_eps0, curr_beta
            )
            curr_eps0, curr_beta = e0, beta
            _, broken = self._check_rupture(strs)

            if broken:
                high = mid
            else:
                low = mid
                best_M = M_val
                best_chi = mid

        return {'M': best_M, 'chi': best_chi}

    # ------------------------------------------------------------------
    # CALCOLO DI UN RAMO COMPLETO
    # ------------------------------------------------------------------

    def calcola_ramo(self, load_angle: float, n_steps: int,
                      N_target: float):
        """
        Calcola un ramo del diagramma M-χ per una data direzione di carico.

        Restituisce array (n_steps, 3) con colonne [M_kNm, χ_1/m, θ_rad].
        """
        # Stima curvatura massima
        D_proj = ((np.max(self._fx) - np.min(self._fx)) * abs(math.cos(load_angle)) +
                  (np.max(self._fy) - np.min(self._fy)) * abs(math.sin(load_angle)))
        if D_proj < 1e-3:
            D_proj = 1.0

        max_limit = max(
            float(np.max(np.abs(self._eps_min))),
            float(np.max(self._eps_max))
        )
        chi_max = (max_limit * 0.8) / (D_proj * 0.1)

        # Distribuzione quadratica della curvatura (più punti vicino a 0)
        t = np.linspace(0, 1.0, n_steps)
        chis = chi_max * (t ** 2)

        branch = np.zeros((n_steps, 3))
        state = {'eps0': 0.0, 'beta': load_angle}
        prev_chi = 0.0
        prev_state = dict(state)
        rupture = False

        for k, chi in enumerate(chis):
            if rupture:
                branch[k] = branch[k - 1]
                continue

            if chi == 0:
                branch[k] = [0.0, 0.0, load_angle]
                continue

            try:
                eps0, beta, M_res, strains = self._solve_biaxial_state(
                    chi, load_angle, N_target,
                    state['eps0'], state['beta']
                )
                state['eps0'] = eps0
                state['beta'] = beta
            except Exception:
                rupture = True
                branch[k] = branch[k - 1]
                continue

            _, is_broken = self._check_rupture(strains)

            if not is_broken:
                branch[k] = [M_res, chi * 1000.0, load_angle]
                prev_chi = chi
                prev_state = {'eps0': eps0, 'beta': beta}
            else:
                rupture = True
                final = self._bisect_failure(
                    prev_chi, chi, load_angle, N_target, prev_state
                )
                branch[k] = [final['M'], final['chi'] * 1000.0, load_angle]
                if k + 1 < n_steps:
                    branch[k + 1:] = branch[k]
                break

        return branch


# ==============================================================================
# THREAD DI CALCOLO
# ==============================================================================

class _MomentoCurvaturaThread(QThread):
    """Esegue il calcolo del diagramma M-χ 3D su thread separato."""

    avanzamento = pyqtSignal(int)       # 0-100 %
    completato  = pyqtSignal(object)    # np.ndarray (n_angoli, n_punti, 3)
    errore      = pyqtSignal(str)

    def __init__(self,
                 calcolatore: CalcoloMomentoCurvatura,
                 n_angoli:    int,
                 n_punti:     int,
                 N_target_kN: float,
                 parent=None) -> None:
        super().__init__(parent)
        self._calc     = calcolatore
        self._n_angoli = max(6, n_angoli)
        self._n_punti  = max(10, n_punti)
        self._N_target = N_target_kN * 1000.0   # kN → N
        self._stop     = False

    def richiedi_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            calc = self._calc
            n_ang = self._n_angoli
            n_pts = self._n_punti
            N_target = self._N_target

            angles = np.linspace(0, 2 * math.pi, n_ang, endpoint=False)
            results = np.zeros((n_ang, n_pts, 3), dtype=np.float64)

            completed = 0

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_map = {}
                for i, angle in enumerate(angles):
                    future = executor.submit(
                        calc.calcola_ramo, angle, n_pts, N_target
                    )
                    future_map[future] = i

                for future in concurrent.futures.as_completed(future_map):
                    if self._stop:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return

                    idx = future_map[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        print(f"WARN  M-χ: ramo {idx} → {exc}")

                    completed += 1
                    self.avanzamento.emit(int(completed / n_ang * 100))

            dt = time.perf_counter() - t0
            print(f">> Momento-Curvatura calcolato in {dt:.2f} s  "
                  f"({n_ang}θ × {n_pts}χ = {n_ang * n_pts} punti)")

            self.avanzamento.emit(100)
            self.completato.emit(results)

        except Exception as exc:
            self.errore.emit(str(exc))
