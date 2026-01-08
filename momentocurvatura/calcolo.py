# calcolo.py
import math
import time
import concurrent.futures
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate
from PyQt5.QtCore import QThread, pyqtSignal

# ======================================================================
# MODELLO MATERIALE (Vettorializzato & Ottimizzato)
# ======================================================================
class Materiale:
    """
    Gestisce le leggi costitutive. 
    Accetta array NumPy per calcoli paralleli su tutte le fibre.
    Identico al Secondo Codice.
    """
    def __init__(self, matrice: List[Tuple[str, float, float]], nome: str = "") -> None:
        self.nome = nome
        self.intervalli = []
        
        all_strains = []
        for i, (expr_raw, a, b) in enumerate(matrice):
            low = float(min(a, b))
            high = float(max(a, b))
            
            expr_str = expr_raw.replace('^', '**')
            compiled = compile(expr_str, f'<mat_{nome}_{i}>', 'eval')
            
            self.intervalli.append({
                'expr_code': compiled,
                'low': low,
                'high': high,
                'raw': expr_str
            })
            all_strains.extend([low, high])

        if all_strains:
            self.ult_compression = min(all_strains)
            self.ult_tension = max(all_strains)
        else:
            self.ult_compression = -1e9
            self.ult_tension = 1e9

    def get_sigma_vectorized(self, eps_array: np.ndarray) -> np.ndarray:
        sigma_out = np.zeros_like(eps_array)
        for intervallo in self.intervalli:
            mask = (eps_array >= intervallo['low']) & (eps_array <= intervallo['high'])
            if np.any(mask):
                x = eps_array[mask]
                try:
                    res = eval(intervallo['expr_code'], {'__builtins__': None, 'np': np}, {'x': x})
                    sigma_out[mask] = res
                except Exception:
                    pass
        return sigma_out

# ======================================================================
# SEZIONE RINFORZATA (Gestione Geometria e Mesh)
# ======================================================================
class SezioneRinforzata:
    """
    Identica al Secondo Codice.
    """
    def __init__(self, elementi: List[Tuple], materiali_dict: Dict[str, Materiale]) -> None:
        self.geometria = []
        self.barre = []
        self.rinforzi = []
        self.materiali = materiali_dict

        for elem in elementi:
            tipo = elem[0]
            if tipo == 'shape':
                _, shape_type, _, mat_name, *params = elem
                mat = self.materiali.get(mat_name)
                if shape_type == 'rect':
                    self.geometria.append((self._crea_rettangolo(*params), mat))
                elif shape_type == 'poly':
                    self.geometria.append((Polygon(params[0]), mat))
                elif shape_type == 'circle':
                    self.geometria.append((Point(params[0]).buffer(params[1], 32), mat))

            elif tipo == 'bar':
                _, _, mat_name, diam, center = elem
                mat = self.materiali.get(mat_name)
                self.barre.append({'x': center[0], 'y': center[1], 'diam': diam, 'mat': mat})

            elif tipo == 'reinf':
                _, reinf_type, _, mat_name, *params = elem
                mat = self.materiali.get(mat_name)
                if reinf_type == 'rect':
                    poly = LineString([params[0], params[1]]).buffer(params[2]/2)
                    self.rinforzi.append((poly, mat))
                elif reinf_type == 'poly':
                    poly = Polygon(params[0]).buffer(params[1]/2)
                    self.rinforzi.append((poly, mat))
                elif reinf_type == 'circular':
                    poly = Point(params[0]).buffer(params[1])
                    self.rinforzi.append((poly, mat))

        self._calcola_limiti_box()
        self.area_totale = self._calcola_area_totale()

        self.mesh_x: Optional[np.ndarray] = None
        self.mesh_y: Optional[np.ndarray] = None
        self.mesh_area: Optional[np.ndarray] = None
        self.mesh_mat_indices: Optional[np.ndarray] = None
        self.materiali_list: List[Materiale] = [] 

    def _crea_rettangolo(self, p1, p2):
        return Polygon([(min(p1[0],p2[0]), min(p1[1],p2[1])), 
                        (max(p1[0],p2[0]), min(p1[1],p2[1])),
                        (max(p1[0],p2[0]), max(p1[1],p2[1])), 
                        (min(p1[0],p2[0]), max(p1[1],p2[1]))])

    def _calcola_limiti_box(self):
        coords = []
        for p, _ in self.geometria + self.rinforzi:
            coords.extend(p.exterior.coords)
        for b in self.barre:
            coords.append((b['x'], b['y']))
        if not coords:
            self.min_x = self.max_x = self.min_y = self.max_y = 0
            return
        pts = np.array(coords)
        self.min_x, self.min_y = np.min(pts, axis=0)
        self.max_x, self.max_y = np.max(pts, axis=0)

    def _calcola_area_totale(self):
        a = sum(p.area for p, _ in self.geometria + self.rinforzi)
        a += sum(math.pi*(b['diam']/2)**2 for b in self.barre)
        return a

    def centroide_sezione(self):
        Ax = Ay = A_tot = 0.0
        for p, _ in self.geometria + self.rinforzi:
            Ax += p.centroid.x * p.area
            Ay += p.centroid.y * p.area
            A_tot += p.area
        for b in self.barre:
            area = math.pi*(b['diam']/2)**2
            Ax += b['x'] * area
            Ay += b['y'] * area
            A_tot += area
        if A_tot == 0: return 0,0
        return Ax/A_tot, Ay/A_tot

    def allinea_al_centro(self):
        cx, cy = self.centroide_sezione()
        self.geometria = [(translate(p, -cx, -cy), m) for p, m in self.geometria]
        self.rinforzi = [(translate(p, -cx, -cy), m) for p, m in self.rinforzi]
        for b in self.barre:
            b['x'] -= cx
            b['y'] -= cy
        self._calcola_limiti_box()
        self.mesh_x = None

    def genera_mesh_vettoriale(self, grid_step: float):
        if self.mesh_x is not None:
            return 

        punti_x = []
        punti_y = []
        aree = []
        mat_idx = []
        
        self.materiali_list = list({m for _, m in self.geometria} | 
                                   {m for _, m in self.rinforzi} | 
                                   {b['mat'] for b in self.barre})
        if None in self.materiali_list: self.materiali_list.remove(None)
        mat_to_id = {m: i for i, m in enumerate(self.materiali_list)}

        w = self.max_x - self.min_x
        h = self.max_y - self.min_y
        nx = int(math.ceil(w / grid_step)) + 1
        ny = int(math.ceil(h / grid_step)) + 1
        
        xs = np.linspace(self.min_x, self.max_x, nx)
        ys = np.linspace(self.min_y, self.max_y, ny)
        dA = (w/max(1, nx-1)) * (h/max(1, ny-1))

        for b in self.barre:
            if b['mat'] in mat_to_id:
                punti_x.append(b['x'])
                punti_y.append(b['y'])
                aree.append(math.pi * (b['diam']/2)**2)
                mat_idx.append(mat_to_id[b['mat']])

        for y in ys:
            for x in xs:
                p = Point(x,y)
                found_mat = None
                for poly, mat in self.rinforzi:
                    if poly.contains(p):
                        found_mat = mat
                        break
                if found_mat is None:
                    for poly, mat in self.geometria:
                        if poly.contains(p):
                            found_mat = mat
                            break
                if found_mat and found_mat in mat_to_id:
                    punti_x.append(x)
                    punti_y.append(y)
                    aree.append(dA)
                    mat_idx.append(mat_to_id[found_mat])

        self.mesh_x = np.array(punti_x, dtype=np.float64)
        self.mesh_y = np.array(punti_y, dtype=np.float64)
        self.mesh_area = np.array(aree, dtype=np.float64)
        self.mesh_mat_indices = np.array(mat_idx, dtype=np.int32)
        print(f"Mesh generata: {len(self.mesh_x)} fibre.")

# ======================================================================
# THREAD CALCOLATORE (Moment-Curvature Biassiale con Strain-Ratio Check)
# ======================================================================
class MomentCurvatureCalculator(QThread):
    calculation_done = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, sezione: SezioneRinforzata, ui, parent=None):
        super().__init__(parent)
        self.sezione = sezione
        self.ui = ui
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        # ---------------- INPUT UTENTE (Struttura Secondo Codice) ----------------
        try:
            grid_step = float(self._get_ui_val('momentocurvatura_precisione', 5.0))
            n_angoli = int(self._get_ui_val('momentocurvatura_angoli', 18))
            n_step_curv = int(self._get_ui_val('momentocurvatura_step', 50))
            
            # Convenzione del Secondo Codice: Input Positivo = Compressione
            N_input_kN = float(self._get_ui_val('momentocurvatura_N', 0))
            N_target = N_input_kN * 1000.0 
            
        except Exception as e:
            print(f"Errore lettura UI: {e}")
            return

        # ---------------- PREPARAZIONE ----------------
        self.sezione.allinea_al_centro()
        self.sezione.genera_mesh_vettoriale(grid_step)
        
        fib_x = self.sezione.mesh_x
        fib_y = self.sezione.mesh_y
        fib_A = self.sezione.mesh_area
        fib_mat_idx = self.sezione.mesh_mat_indices
        materials = self.sezione.materiali_list

        # Ottimizzazione Strain Ratio (dal Secondo Codice)
        lim_comp_expanded = np.zeros_like(fib_x)
        lim_tens_expanded = np.zeros_like(fib_x)
        
        for m_id, mat in enumerate(materials):
            mask = (fib_mat_idx == m_id)
            if np.any(mask):
                lim_comp_expanded[mask] = mat.ult_compression
                lim_tens_expanded[mask] = mat.ult_tension

        load_angles = np.linspace(0, 2*math.pi, n_angoli, endpoint=False)
        results = np.zeros((n_angoli, n_step_curv, 3))
        total_iter = n_angoli * n_step_curv
        count = 0

        # ---------------- CALCOLO PARALLELO ----------------
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {}
            
            for i, angle in enumerate(load_angles):
                future = executor.submit(
                    self._calcola_ramo_robusto,
                    angle, n_step_curv, N_target,
                    fib_x, fib_y, fib_A, fib_mat_idx, materials,
                    lim_comp_expanded, lim_tens_expanded
                )
                future_to_idx[future] = i

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                if self._stop: break
                
                try:
                    branch_res = future.result()
                    results[idx, :, :] = branch_res
                except Exception as e:
                    print(f"Errore angolo {idx}: {e}")
                    import traceback
                    traceback.print_exc()

                count += n_step_curv
                self.progress.emit(int(count/total_iter * 100))

        self.calculation_done.emit(results)

    def _get_ui_val(self, name, default):
        try:
            widget = getattr(self.ui, name)
            val = widget.text() if hasattr(widget, 'text') else str(default)
            return float(val) if val else default
        except:
            return default

    # ---------------- CORE INTEGRATO (STRUTTURA COD. 2 + MATEMATICA COD. 1) ----------------
    def _calcola_ramo_robusto(self, load_angle_rad, n_steps, N_target, 
                              x_arr, y_arr, A_arr, mat_idx_arr, 
                              materials, lim_comp_expanded, lim_tens_expanded):
        """
        Struttura esterna: Strain-Ratio Driven (Secondo Codice).
        Motore interno: Biaxial Solver (Primo Codice).
        """
        
        # 1. Stima Curvatura Massima (Logica Ratio)
        D_proj = (np.max(x_arr) - np.min(x_arr)) * abs(math.cos(load_angle_rad)) + \
                 (np.max(y_arr) - np.min(y_arr)) * abs(math.sin(load_angle_rad))
        
        max_limit = max(np.max(np.abs(lim_comp_expanded)), np.max(lim_tens_expanded))
        chi_max_theoretical = (max_limit * 0.8) / (D_proj * 0.1) 
        
        t = np.linspace(0, 1.0, n_steps)
        chis_target = chi_max_theoretical * (t ** 2)
        
        branch_data = np.zeros((n_steps, 3)) 
        
        # Variabili di stato (inclusa beta_na per il solver del Primo Codice)
        state_guess = {'eps0': 0.0, 'beta_na': load_angle_rad}
        
        rupture_reached = False
        
        for k, chi in enumerate(chis_target):
            if rupture_reached:
                branch_data[k] = branch_data[k-1]
                continue
            
            if chi == 0:
                branch_data[k] = [0.0, 0.0, load_angle_rad]
                continue

            # A. SOLVER BIASSIALE (Iniezione logica Primo Codice)
            try:
                eps0, beta_na, M_res, strains = self._solve_biaxial_state(
                    chi, load_angle_rad, N_target, 
                    state_guess['eps0'], state_guess['beta_na'],
                    x_arr, y_arr, A_arr, mat_idx_arr, materials
                )
                state_guess['eps0'] = eps0
                state_guess['beta_na'] = beta_na
                
            except RuntimeError:
                rupture_reached = True
                branch_data[k] = branch_data[k-1]
                continue

            # B. CHECK ROTTURA (Logica Secondo Codice)
            ratio_val, is_rupture = self._check_rupture(strains, lim_comp_expanded, lim_tens_expanded)
            
            if not is_rupture:
                branch_data[k] = [M_res, chi * 1000.0, load_angle_rad]
                prev_chi = chi
                prev_state = {'eps0': eps0, 'beta_na': beta_na}
            else:
                # --- ROTTURA RILEVATA ---
                rupture_reached = True
                # Bisezione (Usando il solver biassiale per precisione)
                final_res = self._bisect_failure_point(
                    prev_chi, chi, load_angle_rad, N_target, 
                    prev_state, 
                    x_arr, y_arr, A_arr, mat_idx_arr, materials,
                    lim_comp_expanded, lim_tens_expanded
                )
                branch_data[k] = [final_res['M'], final_res['chi'] * 1000.0, load_angle_rad]
                
                if k + 1 < n_steps:
                    branch_data[k+1:] = branch_data[k]
                break

        return branch_data

    def _solve_biaxial_state(self, chi, target_angle, N_target, guess_eps0, guess_beta,
                             x_arr, y_arr, A_arr, mat_idx_arr, materials):
        """
        (Nuovo) Solver che itera sull'angolo beta (Asse neutro) finché il momento
        risultante non è parallelo al carico esterno.
        """
        def compute_moment_error(beta_trial):
            cx = math.cos(beta_trial)
            cy = math.sin(beta_trial)
            d_arr_rotated = x_arr * cx + y_arr * cy
            
            eps0_equil = self._solve_axial_equilibrium(
                chi, d_arr_rotated, N_target, guess_eps0, 
                A_arr, mat_idx_arr, materials
            )
            
            strains = eps0_equil - chi * d_arr_rotated
            Mx, My = self._calculate_Mx_My(strains, x_arr, y_arr, A_arr, mat_idx_arr, materials)
            
            # Errore direzionale
            t_sin = math.sin(target_angle)
            t_cos = math.cos(target_angle)
            error_perp = -Mx * t_sin + My * t_cos
            
            return error_perp, eps0_equil, strains, math.sqrt(Mx**2 + My**2)

        # Solver tipo Secante per l'angolo beta
        beta_curr = guess_beta
        err_curr, e0_curr, strains_curr, M_curr = compute_moment_error(beta_curr)
        
        if abs(err_curr) < 1e-3: 
            return e0_curr, beta_curr, M_curr / 1e6, strains_curr

        beta_prev = beta_curr - 0.01 
        err_prev, _, _, _ = compute_moment_error(beta_prev)
        
        for _ in range(24): 
            if abs(err_curr - err_prev) < 1e-9: break
            
            beta_new = beta_curr - err_curr * (beta_curr - beta_prev) / (err_curr - err_prev)
            err_new, e0_new, s_new, M_new = compute_moment_error(beta_new)
            
            if abs(err_new) < 1e-3:
                return e0_new, beta_new, M_new / 1e6, s_new
            
            beta_prev, err_prev = beta_curr, err_curr
            beta_curr, err_curr = beta_new, err_new
            e0_curr, strains_curr, M_curr = e0_new, s_new, M_new

        return e0_curr, beta_curr, M_curr / 1e6, strains_curr

    def _solve_axial_equilibrium(self, chi, d_arr, N_target, guess_eps0, 
                                 A_arr, mat_idx_arr, materials):
        """Newton-Raphson per equilibrio assiale (Adattato per input biassiale)"""
        def residual(e0):
            strains = e0 - chi * d_arr
            N_calc = 0.0
            for m_id, mat in enumerate(materials):
                mask = (mat_idx_arr == m_id)
                if np.any(mask):
                    sigs = mat.get_sigma_vectorized(strains[mask])
                    N_calc += np.sum(sigs * A_arr[mask])
            return N_calc - N_target

        e_curr = guess_eps0
        for _ in range(16): 
            res = residual(e_curr)
            if abs(res) < 10.0: return e_curr
            
            delta = 1e-6
            stiff = (residual(e_curr + delta) - res) / delta
            if abs(stiff) < 1e-3: break 
            e_curr = e_curr - res / stiff
            if abs(residual(e_curr)) < 10.0: return e_curr

        # Fallback Bisezione
        low, high = guess_eps0 - 0.1, guess_eps0 + 0.1
        f_low, f_high = residual(low), residual(high)
        if f_low * f_high > 0:
            low, high = -0.5, 0.5 
            f_low, f_high = residual(low), residual(high)
        
        for _ in range(40):
            mid = (low + high) / 2
            f_mid = residual(mid)
            if abs(f_mid) < 10.0 or (high - low) < 1e-7: return mid
            if f_low * f_mid < 0: high = mid
            else: low, f_low = mid, f_mid
        return (low + high) / 2

    def _calculate_Mx_My(self, strains, x_arr, y_arr, A_arr, mat_idx_arr, materials):
        """Helper per momenti vettoriali"""
        sigmas = np.zeros_like(strains)
        for m_id, mat in enumerate(materials):
            mask = (mat_idx_arr == m_id)
            if np.any(mask):
                sigmas[mask] = mat.get_sigma_vectorized(strains[mask])
        
        forces = sigmas * A_arr
        Mx = np.sum(forces * y_arr)
        My = np.sum(forces * x_arr)
        return Mx, My

    def _check_rupture(self, strains, lim_comp, lim_tens):
        """Identico al Secondo Codice"""
        mask_tens = strains > 0
        mask_comp = strains < 0
        max_ratio = 0.0
        
        valid_t = mask_tens & (lim_tens > 1e-6)
        if np.any(valid_t):
            max_ratio = max(max_ratio, np.max(strains[valid_t] / lim_tens[valid_t]))
            
        valid_c = mask_comp & (lim_comp < -1e-6)
        if np.any(valid_c):
            max_ratio = max(max_ratio, np.max(strains[valid_c] / lim_comp[valid_c]))
            
        return max_ratio, (max_ratio >= 1.0)

    def _bisect_failure_point(self, chi_start, chi_end, load_angle, N_target, 
                              state_start, x_arr, y_arr, A_arr, mat_idx, mats, l_comp, l_tens):
        """
        Bisezione precisa del punto di rottura (Struttura Secondo Codice, 
        ma chiama il solver Biassiale del Primo Codice).
        """
        low = chi_start
        high = chi_end
        best_M = 0.0
        best_chi = low
        
        curr_eps0 = state_start['eps0']
        curr_beta = state_start['beta_na']
        
        for _ in range(20):
            mid_chi = (low + high) / 2.0
            
            e0, beta, M_val, strs = self._solve_biaxial_state(
                mid_chi, load_angle, N_target, curr_eps0, curr_beta,
                x_arr, y_arr, A_arr, mat_idx, mats
            )
            curr_eps0, curr_beta = e0, beta
            
            _, broken = self._check_rupture(strs, l_comp, l_tens)
            
            if broken:
                high = mid_chi 
            else:
                low = mid_chi
                best_M = M_val
                best_chi = mid_chi
                
        return {'M': best_M, 'chi': best_chi}