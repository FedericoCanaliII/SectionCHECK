import math
import time
import concurrent.futures
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate
from PyQt5.QtCore import QThread, pyqtSignal

# ======================================================================
# MODELLO MATERIALE
# ======================================================================
class Materiale:
    """Rappresentazione del materiale con curve sigma-epsilon."""
    def __init__(self, matrice: List[Tuple[str, float, float]], nome: str = "") -> None:
        self.nome = nome
        self.intervalli: List[Dict[str, Any]] = []
        self.ult_compression = 0.0
        self.ult_tension = 0.0

        all_strains: List[float] = []
        for i, intervallo in enumerate(matrice):
            expr_raw, a, b = intervallo
            low = min(a, b)
            high = max(a, b)
            expr = expr_raw.replace('^', '**')
            compiled = compile(expr, f'<mat_{nome}_{i}>', 'eval')
            self.intervalli.append({'expr': compiled, 'low': low, 'high': high})
            all_strains.extend([low, high])

        if all_strains:
            # Definiamo i limiti ultimi assoluti (minimo negativo e massimo positivo)
            self.ult_compression = min(all_strains)
            self.ult_tension = max(all_strains)

    def sigma(self, eps: float) -> float:
        for intervallo in self.intervalli:
            if intervallo['low'] <= eps <= intervallo['high']:
                return eval(intervallo['expr'], {'__builtins__': None}, {'x': eps})
        return 0.0


# ======================================================================
# SEZIONE RINFORZATA
# ======================================================================
class SezioneRinforzata:
    def __init__(self, elementi: List[Tuple], materiali_dict: Dict[str, Materiale]) -> None:
        self.geometria: List[Tuple[Polygon, Materiale]] = []
        self.barre: List[Tuple[float, float, float, Materiale]] = [] # x, y, diam, mat
        self.rinforzi: List[Tuple[Polygon, float, Materiale]] = []   # poly, thick, mat
        self.materiali = materiali_dict

        # --- Parsing Input ---
        for elem in elementi:
            tipo = elem[0]
            if tipo == 'shape':
                _, shape_type, _, mat_name, *params = elem
                mat = self.materiali.get(mat_name)
                if mat:
                    if shape_type == 'rect':
                        poly = self._crea_rettangolo(params[0], params[1])
                        self.geometria.append((poly, mat))
                    elif shape_type == 'poly':
                        self.geometria.append((Polygon(params[0]), mat))
                    elif shape_type == 'circle':
                        self.geometria.append((Point(params[0]).buffer(params[1], 16), mat))

            elif tipo == 'bar':
                _, _, mat_name, diam, center = elem
                mat = self.materiali.get(mat_name)
                if mat:
                    self.barre.append((center[0], center[1], diam, mat))

            elif tipo == 'reinf':
                _, reinf_type, _, mat_name, *params = elem
                mat = self.materiali.get(mat_name)
                if mat:
                    if reinf_type == 'rect':
                        poly = LineString([params[0], params[1]]).buffer(params[2]/2)
                        self.rinforzi.append((poly, params[2], mat))
                    elif reinf_type == 'poly':
                        poly = Polygon(params[0]).buffer(params[1]/2)
                        self.rinforzi.append((poly, params[1], mat))

        self._aggiorna_proprieta_geometriche()
        self.area_totale = self._calcola_area_totale()
        
        # Cache
        self._cache_punti_integrazione: Optional[List[Tuple[float, float, float, Materiale]]] = None
        self._cache_grid_step: float = 0.0

    # ---------------------------- Geometria Base ----------------------------
    def _crea_rettangolo(self, p1, p2) -> Polygon:
        return Polygon([(min(p1[0], p2[0]), min(p1[1], p2[1])),
                        (max(p1[0], p2[0]), min(p1[1], p2[1])),
                        (max(p1[0], p2[0]), max(p1[1], p2[1])),
                        (min(p1[0], p2[0]), max(p1[1], p2[1]))])

    def _aggiorna_proprieta_geometriche(self) -> None:
        all_coords = []
        for p, _ in self.geometria: all_coords.extend(p.exterior.coords)
        for p, _, _ in self.rinforzi: all_coords.extend(p.exterior.coords)
        for b in self.barre: 
            r = b[2]/2
            all_coords.extend([(b[0]-r, b[1]-r), (b[0]+r, b[1]+r)])

        if not all_coords:
            self.x_min = self.y_min = self.x_max = self.y_max = self.dx = self.dy = 0.0
            return

        pts = np.array(all_coords)
        self.x_min, self.y_min = np.min(pts, axis=0)
        self.x_max, self.y_max = np.max(pts, axis=0)
        self.dx = self.x_max - self.x_min
        self.dy = self.y_max - self.y_min

    def _calcola_area_totale(self) -> float:
        area = sum(p.area for p, _ in self.geometria)
        area += sum(p.area for p, _, _ in self.rinforzi)
        area += sum(math.pi * (b[2]/2)**2 for b in self.barre)
        return area

    def centroide_sezione(self) -> Tuple[float, float]:
        if self.area_totale == 0: return 0.0, 0.0
        Sx = Sy = 0.0
        for p, _ in self.geometria:
            Sx += p.area * p.centroid.x; Sy += p.area * p.centroid.y
        for p, _, _ in self.rinforzi:
            Sx += p.area * p.centroid.x; Sy += p.area * p.centroid.y
        for b in self.barre:
            a = math.pi * (b[2]/2)**2
            Sx += a * b[0]; Sy += a * b[1]
        return Sx / self.area_totale, Sy / self.area_totale

    def allinea_al_centro(self) -> None:
        cx, cy = self.centroide_sezione()
        if abs(cx) < 1e-5 and abs(cy) < 1e-5: return

        self.geometria = [(translate(p, -cx, -cy), m) for p, m in self.geometria]
        self.rinforzi = [(translate(p, -cx, -cy), t, m) for p, t, m in self.rinforzi]
        self.barre = [(b[0]-cx, b[1]-cy, b[2], b[3]) for b in self.barre]
        
        self._aggiorna_proprieta_geometriche()
        self._cache_punti_integrazione = None

    # ------------------------ Preparazione Calcolo ------------------------

    def _prepara_punti_integrazione(self, grid_step: float) -> List[Tuple[float, float, float, Materiale]]:
        """Crea la griglia di fibre per l'integrazione delle forze (N, M)."""
        if self._cache_punti_integrazione and self._cache_grid_step == grid_step:
            return self._cache_punti_integrazione

        punti = []
        # Barre (esatte)
        for b in self.barre:
            area = math.pi * (b[2]/2)**2
            punti.append((b[0], b[1], area, b[3]))

        # Geometria e Rinforzi (griglia)
        nx = int(self.dx / grid_step) + 2
        ny = int(self.dy / grid_step) + 2
        
        geo_list = self.geometria
        rinf_list = self.rinforzi
        
        for i in range(nx):
            x = self.x_min + i * grid_step + grid_step/2
            if x > self.x_max: break
            for j in range(ny):
                y = self.y_min + j * grid_step + grid_step/2
                if y > self.y_max: break
                
                p_sh = Point(x, y)
                found = False
                for poly, _, mat in rinf_list:
                    if poly.contains(p_sh):
                        punti.append((x, y, grid_step**2, mat))
                        found = True
                        break
                if found: continue
                for poly, mat in geo_list:
                    if poly.contains(p_sh):
                        punti.append((x, y, grid_step**2, mat))
                        break

        self._cache_punti_integrazione = punti
        self._cache_grid_step = grid_step
        return punti

    # ------------------------ Calcolo Assiale Puro (Chiusura Dominio) ------------------------
    
    def calcola_punto_assiale_puro(self, modo: str, grid_step: float) -> List[float]:
        """
        Calcola N, Mx, My per deformazione uniforme (curvatura 0).
        modo: 'compressione' o 'trazione'
        """
        # Assicurati che i punti siano pronti
        punti = self._prepara_punti_integrazione(grid_step)
        
        # 1. Trova la deformazione limite uniforme della sezione
        # È determinata dal materiale che raggiunge per primo il suo limite.
        epsilon_limite = 0.0
        
        if modo == 'compressione':
            # Cerchiamo il max (più vicino a 0) dei limiti negativi.
            # Esempio: calcestruzzo -0.0035, acciaio -0.01 -> Rottura a -0.0035
            limiti = [m.ult_compression for m in self.materiali.values() if m.ult_compression < -1e-5]
            if limiti:
                epsilon_limite = max(limiti) # Es. max(-0.0035, -0.01) = -0.0035
            else:
                epsilon_limite = -0.002 # Fallback
        else: # Trazione
            # Cerchiamo il min dei limiti positivi. Ignoriamo materiali con trazione ~0 (cls).
            limiti = [m.ult_tension for m in self.materiali.values() if m.ult_tension > 1e-4]
            if limiti:
                epsilon_limite = min(limiti)
            else:
                epsilon_limite = 0.0 # Se c'è solo calcestruzzo non armato a trazione -> 0
        
        # 2. Integra le forze
        N = Mx = My = 0.0
        for x, y, area, mat in punti:
            sig = mat.sigma(epsilon_limite)
            f = sig * area
            N += f
            Mx += f * y
            My += f * x
            
        return [Mx / 1e6, My / 1e6, N / 1e3]

    # ------------------------ Calcolo Curvatura (Bending) ------------------------

    def _calcola_estremi_esatti_materiali(self, n_vect: np.ndarray) -> Dict[Materiale, Dict[str, float]]:
        """Proietta la geometria esatta (no griglia) per trovare i limiti."""
        limiti = {} 
        def agg(mat, val_min, val_max):
            if mat not in limiti: limits = {'min': float('inf'), 'max': float('-inf')}
            else: limits = limiti[mat]
            if val_min < limits['min']: limits['min'] = val_min
            if val_max > limits['max']: limits['max'] = val_max
            limiti[mat] = limits

        for poly, mat in self.geometria:
            coords = np.array(poly.exterior.coords)
            p = np.dot(coords, n_vect)
            agg(mat, np.min(p), np.max(p))
        for poly, _, mat in self.rinforzi:
            coords = np.array(poly.exterior.coords)
            p = np.dot(coords, n_vect)
            agg(mat, np.min(p), np.max(p))
        for x, y, d, mat in self.barre:
            c = x*n_vect[0] + y*n_vect[1]
            agg(mat, c - d/2, c + d/2)
        return limiti

    def _calcola_fattore_scala_robusto(self, n_vect: np.ndarray, k: float) -> float:
        """Determina la curvatura 'm' al collasso usando geometria esatta."""
        limiti_mat = self._calcola_estremi_esatti_materiali(n_vect)
        m_govern = float('inf')
        
        for mat, est in limiti_mat.items():
            d_min, d_max = est['min'], est['max']
            
            # Rottura a trazione (fibra più lontana positiva)
            dist_traz = d_max - k
            if dist_traz > 1e-6 and mat.ult_tension > 0:
                m = mat.ult_tension / dist_traz
                if m < m_govern: m_govern = m

            # Rottura a compressione (fibra più lontana negativa)
            dist_comp = d_min - k
            if dist_comp < -1e-6 and mat.ult_compression < 0:
                m = mat.ult_compression / dist_comp
                if m < m_govern: m_govern = m

        return m_govern if m_govern != float('inf') else 0.0

    def _genera_step_asse_neutro_strutturato(self, n_vect: np.ndarray, n_steps: int) -> List[float]:
        """Genera posizioni 'k' per l'asse neutro."""
        # Proiezione globale
        all_d = []
        for p, _ in self.geometria: all_d.extend(np.dot(np.array(p.exterior.coords), n_vect))
        for b in self.barre:
            p = b[0]*n_vect[0] + b[1]*n_vect[1]
            all_d.extend([p - b[2]/2, p + b[2]/2])
        
        if not all_d: return np.linspace(-100, 100, n_steps).tolist()
        d_min, d_max = min(all_d), max(all_d)
        H = d_max - d_min if d_max != d_min else 100.0

        # Creiamo steps concentrati nella sezione
        n_inner = int(0.7 * n_steps)
        n_outer = (n_steps - n_inner) // 2
        
        c_inner = np.linspace(d_min + 0.001*H, d_max - 0.001*H, n_inner)
        factors = np.geomspace(0.01, 50, n_outer) # Logaritmico verso esterno
        
        c_lower = d_min - factors * H
        c_upper = d_max + factors * H
        
        # Unione e sort
        c_total = np.concatenate([np.flip(c_lower), c_inner, c_upper])
        
        # Interpolazione per avere esattamente n_steps
        if len(c_total) != n_steps:
            c_total = np.interp(np.linspace(0,1,n_steps), np.linspace(0,1,len(c_total)), c_total)
        return list(c_total)

    def _decomponi_singolo_punto(self, params: Tuple) -> List[float]:
        """Worker per calcolo parallelo."""
        theta, k, grid_step, punti_integrazione = params
        n_vect = np.array([math.cos(theta), math.sin(theta)])
        
        m_govern = self._calcola_fattore_scala_robusto(n_vect, k)
        if m_govern <= 1e-9: return [0.0, 0.0, 0.0]

        N = Mx = My = 0.0
        for x, y, area, mat in punti_integrazione:
            d = x * n_vect[0] + y * n_vect[1]
            eps = m_govern * (d - k)
            sig = mat.sigma(eps)
            if sig != 0:
                f = sig * area
                N += f
                Mx += f * y
                My += f * x
                
        return [Mx / 1e6, My / 1e6, N / 1e3]


# ======================================================================
# THREAD CALCOLATORE
# ======================================================================
class DomainCalculator(QThread):
    calculation_done = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, sezione: SezioneRinforzata, ui, parent=None):
        super().__init__(parent)
        self.sezione = sezione
        self.ui = ui

    def run(self) -> None:
        try:
            try: grid_step = float(self.ui.out_precisione.text())
            except: grid_step = 10.0
            try: theta_steps = int(self.ui.out_angoli.text())
            except: theta_steps = 24
            try: neutral_steps = int(self.ui.out_step.text())
            except: neutral_steps = 30

            if grid_step <= 0: grid_step = 10.0
            if theta_steps < 4: theta_steps = 4
            # Minimo 3 step per includere Trazione, Compressione e almeno un punto intermedio
            if neutral_steps < 3: neutral_steps = 3 
            
        except Exception as e:
            print(f"Errore parametri: {e}")
            return

        start_time = time.time()
        self.sezione.allinea_al_centro()
        
        # 1. Punti Integrazione
        punti_integrazione = self.sezione._prepara_punti_integrazione(grid_step)
        
        # 2. Calcolo Punti Estremi (Chiusura dominio)
        # Questi punti sono identici per ogni angolo theta
        pt_trazione = self.sezione.calcola_punto_assiale_puro('trazione', grid_step)
        pt_compressione = self.sezione.calcola_punto_assiale_puro('compressione', grid_step)
        
        # 3. Setup Loop
        thetas = np.linspace(0, 2 * math.pi, theta_steps, endpoint=False)
        results_matrix = np.zeros((theta_steps, neutral_steps, 3))
        
        # Indici da calcolare (escludiamo 0 e -1 che sono gli estremi fissi)
        indices_to_compute = range(1, neutral_steps - 1)
        total_bending_tasks = theta_steps * len(indices_to_compute)
        completed = 0
        
        # 4. Assegnazione Punti Fissi alla Matrice
        # Riempie la prima "fetta" e l'ultima "fetta" del cilindro con i poli
        for i in range(theta_steps):
            results_matrix[i, 0, :] = pt_trazione      # Top
            results_matrix[i, -1, :] = pt_compressione # Bottom

        # 5. Calcolo Parallelo (solo parte centrale a flessione)
        if total_bending_tasks > 0:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_idx = {}
                
                for i, theta in enumerate(thetas):
                    n_vect = np.array([math.cos(theta), math.sin(theta)])
                    
                    # Generiamo solo gli step intermedi
                    # Richiediamo (neutral_steps - 2) punti interni
                    c_full = self.sezione._genera_step_asse_neutro_strutturato(n_vect, neutral_steps)
                    # Prendiamo solo quelli centrali da passare al calcolatore
                    c_bending = c_full[1:-1] 
                    
                    for k_idx, k in enumerate(c_bending):
                        real_j_index = k_idx + 1 # Offset di 1 perché 0 è occupato dalla trazione pura
                        
                        params = (theta, k, grid_step, punti_integrazione)
                        future = executor.submit(self.sezione._decomponi_singolo_punto, params)
                        future_to_idx[future] = (i, real_j_index)

                for future in concurrent.futures.as_completed(future_to_idx):
                    i, j = future_to_idx[future]
                    try:
                        res = future.result()
                        results_matrix[i, j, :] = res
                    except Exception as e:
                        print(f"Errore calcolo {i},{j}: {e}")
                    
                    completed += 1
                    if completed % 20 == 0:
                        self.progress.emit(int(completed / total_bending_tasks * 100))
        
        print(f"Calcolo completato in {time.time() - start_time:.2f}s")
        self.progress.emit(100)
        self.calculation_done.emit(results_matrix)