import math
import time
from typing import List, Tuple, Optional, Dict, Any
import concurrent.futures

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate
from PyQt5.QtCore import QThread, pyqtSignal

# ======================================================================
# MODELLO MATERIALE (INVARIATO)
# ======================================================================
class Materiale:
    """Rappresentazione semplice di un materiale con intervalli di
    comportamento (espressioni in funzione della deformazione 'x').
    """

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
            compiled = compile(expr, f'<material_expr_{i}>', 'eval')
            self.intervalli.append({'expr': compiled, 'low': low, 'high': high})
            all_strains.extend([low, high])

        if all_strains:
            self.ult_compression = min(all_strains)
            self.ult_tension = max(all_strains)

    def sigma(self, eps: float) -> float:
        for intervallo in self.intervalli:
            if intervallo['low'] <= eps <= intervallo['high']:
                return eval(intervallo['expr'], {'__builtins__': None}, {'x': eps})
        return 0.0


# ======================================================================
# SEZIONE RINFORZATA (OTTIMIZZATA E STRUTTURATA)
# ======================================================================
class SezioneRinforzata:
    def __init__(self, elementi: List[Tuple], materiali_dict: Dict[str, Materiale]) -> None:
        self.geometria: List[Tuple[Polygon, Optional[Materiale]]] = []
        self.barre: List[Tuple[float, float, float, Optional[Materiale]]] = []
        self.rinforzi: List[Tuple[Polygon, float, Optional[Materiale]]] = []
        self.materiali = materiali_dict

        # Caricamento elementi
        for elem in elementi:
            tipo = elem[0]
            if tipo == 'shape':
                _, shape_type, _, mat_name, *params = elem
                mat = self.materiali.get(mat_name)
                if shape_type == 'rect':
                    p1, p2 = params
                    poly = self._crea_rettangolo(p1, p2)
                    self.geometria.append((poly, mat))
                elif shape_type == 'poly':
                    verts = params[0]
                    poly = Polygon(verts)
                    self.geometria.append((poly, mat))
                elif shape_type == 'circle':
                    center, radius = params
                    poly = Point(center).buffer(radius, 32)
                    self.geometria.append((poly, mat))

            elif tipo == 'bar':
                _, _, mat_name, diam, center = elem
                mat = self.materiali.get(mat_name)
                self.barre.append((center[0], center[1], diam, mat))

            elif tipo == 'reinf':
                _, reinf_type, _, mat_name, *params = elem
                mat = self.materiali.get(mat_name)
                if reinf_type == 'rect':
                    p0, p1, th = params
                    poly = self._crea_rinforzo_rettangolare(p0, p1, th)
                    self.rinforzi.append((poly, th, mat))
                elif reinf_type == 'poly':
                    verts, th = params
                    poly = Polygon(verts).buffer(th / 2, cap_style=2, join_style=2)
                    self.rinforzi.append((poly, th, mat))
                elif reinf_type == 'circular':
                    center, th = params
                    poly = Point(center).buffer(th, 32)
                    self.rinforzi.append((poly, th, mat))

        self._aggiorna_proprieta_geometriche()
        self.area_totale = self._calcola_area_totale()
        self._recentrata = False
        
        # Cache dei punti di integrazione per velocizzare
        self._cache_punti_integrazione: Optional[List[Tuple[float, float, float, Materiale]]] = None
        self._cache_grid_step: float = 0.0

    # ---------------------------- Geometria di Base ----------------------------
    def _aggiorna_proprieta_geometriche(self) -> None:
        all_points = []
        for poly, _ in self.geometria:
            all_points.extend(poly.exterior.coords)
        for poly, _, _ in self.rinforzi:
            all_points.extend(poly.exterior.coords)
        for bar in self.barre:
            all_points.append((bar[0], bar[1]))

        if not all_points:
            self.x_min = self.y_min = self.x_max = self.y_max = 0.0
            self.dx = self.dy = self.diametro = 0.0
        else:
            all_points = np.array(all_points)
            self.x_min, self.y_min = np.min(all_points, axis=0)
            self.x_max, self.y_max = np.max(all_points, axis=0)
            self.dx = self.x_max - self.x_min
            self.dy = self.y_max - self.y_min
            self.diametro = math.hypot(self.dx, self.dy)

    def _crea_rettangolo(self, p1, p2) -> Polygon:
        x_coords = sorted([p1[0], p2[0]])
        y_coords = sorted([p1[1], p2[1]])
        return Polygon([(x_coords[0], y_coords[0]), (x_coords[1], y_coords[0]),
                        (x_coords[1], y_coords[1]), (x_coords[0], y_coords[1])])

    def _crea_rinforzo_rettangolare(self, p0, p1, th) -> Polygon:
        return LineString([p0, p1]).buffer(th / 2, cap_style=2, join_style=2)

    def _get_materiale_per_punto(self, x: float, y: float) -> Optional[Materiale]:
        p = Point(x, y)
        for poly, _, mat in self.rinforzi:
            if poly.contains(p): return mat
        for poly, mat in self.geometria:
            if poly.contains(p): return mat
        return None

    def _calcola_area_totale(self) -> float:
        area = 0.0
        for poly, _ in self.geometria: area += poly.area
        for poly, _, _ in self.rinforzi: area += poly.area
        for bar in self.barre: area += math.pi * (bar[2] / 2) ** 2
        return area

    def centroide_sezione(self) -> Tuple[float, float]:
        if self.area_totale == 0: return 0.0, 0.0
        Sx = Sy = 0.0
        for poly, _ in self.geometria:
            Sx += poly.area * poly.centroid.x
            Sy += poly.area * poly.centroid.y
        for poly, _, _ in self.rinforzi:
            Sx += poly.area * poly.centroid.x
            Sy += poly.area * poly.centroid.y
        for bar in self.barre:
            area = math.pi * (bar[2] / 2) ** 2
            Sx += area * bar[0]
            Sy += area * bar[1]
        return Sx / self.area_totale, Sy / self.area_totale

    def allinea_al_centro(self) -> None:
        cx, cy = self.centroide_sezione()
        if cx == 0 and cy == 0 and not any([self.geometria, self.rinforzi, self.barre]):
            return
        
        # Trasla tutto
        for i, (poly, mat) in enumerate(self.geometria):
            self.geometria[i] = (translate(poly, xoff=-cx, yoff=-cy), mat)
        for i, (poly, th, mat) in enumerate(self.rinforzi):
            self.rinforzi[i] = (translate(poly, xoff=-cx, yoff=-cy), th, mat)
        for i, bar in enumerate(self.barre):
            self.barre[i] = (bar[0] - cx, bar[1] - cy, bar[2], bar[3])

        self._aggiorna_proprieta_geometriche()
        self.area_totale = self._calcola_area_totale()
        self._recentrata = True
        # Invalida cache
        self._cache_punti_integrazione = None

    # ------------------------ Calcolo Avanzato (Ottimizzato) ------------------------
    
    def _prepara_punti_integrazione(self, grid_step: float) -> List[Tuple[float, float, float, Materiale]]:
        """Genera una lista piatta di punti di integrazione (fibre + barre).
        Usato per velocizzare il calcolo ciclico.
        """
        if self._cache_punti_integrazione is not None and self._cache_grid_step == grid_step:
            return self._cache_punti_integrazione

        punti = []
        # Griglia geometria base
        nx = max(1, int(self.dx / grid_step))
        ny = max(1, int(self.dy / grid_step))
        
        # Ottimizzazione: scansione bounding box
        for i in range(nx + 1):
            x = self.x_min + i * (self.dx / nx) if nx > 0 else self.x_min
            for j in range(ny + 1):
                y = self.y_min + j * (self.dy / ny) if ny > 0 else self.y_min
                mat = self._get_materiale_per_punto(x, y)
                if mat:
                    punti.append((x, y, grid_step ** 2, mat))

        # Barre
        for bar in self.barre:
            area = math.pi * (bar[2] / 2) ** 2
            punti.append((bar[0], bar[1], area, bar[3])) # x, y, area, mat

        # Rinforzi (centroidi approssimati o discretizzati se necessario)
        # Per semplicità qui usiamo il centroide, per alta precisione bisognerebbe discretizzare
        for poly, _, mat in self.rinforzi:
            c = poly.centroid
            punti.append((c.x, c.y, poly.area, mat))

        self._cache_punti_integrazione = punti
        self._cache_grid_step = grid_step
        return punti

    def _trova_estremi_proiezione(self, n_vect: np.ndarray) -> Tuple[float, float]:
        """Trova d_min e d_max della sezione proiettata sulla direzione n_vect."""
        all_d = []
        # Geometria
        for poly, _ in self.geometria:
            coords = np.array(poly.exterior.coords)
            proj = np.dot(coords, n_vect)
            all_d.extend([np.min(proj), np.max(proj)])
        # Rinforzi
        for poly, _, _ in self.rinforzi:
            coords = np.array(poly.exterior.coords)
            proj = np.dot(coords, n_vect)
            all_d.extend([np.min(proj), np.max(proj)])
        # Barre
        for bar in self.barre:
            proj = bar[0]*n_vect[0] + bar[1]*n_vect[1]
            all_d.append(proj)

        return (min(all_d), max(all_d)) if all_d else (0.0, 0.0)

    def _calcola_fattore_scala(self, n_vect: np.ndarray, k: float, punti: List) -> float:
        """Calcola la curvatura (o fattore scala 'm') che porta alla rottura (ultimate strain)."""
        m_lim = 0.0
        
        # Logica ottimizzata: invece di liste, calcolo al volo il minimo
        # m = eps_lim / (d - k)
        
        min_m_pos = float('inf') # Per delta_d > 0 (trazione)
        min_m_neg = float('inf') # Per delta_d < 0 (compressione)

        # Pre-calcola dot products se possibile, ma qui iteriamo la lista mista
        for (x, y, _, mat) in punti:
            if mat is None: continue
            
            d = x * n_vect[0] + y * n_vect[1]
            delta_d = d - k
            
            if delta_d == 0: continue

            if delta_d < 0: # Compressione
                # eps_lim è negativo per compressione
                m = mat.ult_compression / delta_d
                if m > 0 and m < min_m_neg:
                    min_m_neg = m
            else: # Trazione
                m = mat.ult_tension / delta_d
                if m > 0 and m < min_m_pos:
                    min_m_pos = m
        
        # Il fattore governante è il più piccolo che causa rottura
        m_final = min(min_m_pos, min_m_neg)
        return m_final if m_final != float('inf') else 0.0

    def _genera_step_asse_neutro_strutturato(self, n_vect: np.ndarray, n_steps: int) -> List[float]:
        """
        Genera una lista di posizioni 'c' (distanza asse neutro) strutturata ed esponenziale.
        Garantisce lo stesso numero di punti per ogni angolo per un plot pulito.
        """
        d_min, d_max = self._trova_estremi_proiezione(n_vect)
        H = d_max - d_min
        if H <= 0: H = 1.0
        
        # Definiamo i punti chiave relativi alla proiezione
        # range interno: da d_min a d_max
        # range esterno: verso +- infinito (limitato numericamente)
        
        # Strategia: 
        # Trazione pura: c -> -inf (numericamente -100*H)
        # Compressione pura: c -> +inf (numericamente +100*H)
        # Punti densi tra d_min e d_max
        
        # Dividiamo gli step in 3 zone:
        # 1. Zona Trazione spinta (sotto d_min): 15% dei punti
        # 2. Zona Sezione (tra d_min e d_max): 70% dei punti (alta risoluzione)
        # 3. Zona Compressione spinta (sopra d_max): 15% dei punti
        
        n_inner = int(0.6 * n_steps)
        n_outer = (n_steps - n_inner) // 2
        
        # Zona centrale (lineare o leggermente sigmoidale)
        c_inner = np.linspace(d_min + 0.01*H, d_max - 0.01*H, n_inner)
        
        # Zona inferiore (esponenziale decrescente da d_min)
        # c = d_min - (exp(k) - 1)
        factor = 50.0 # Fattore di espansione (quante volte H ci allontaniamo)
        t = np.linspace(0, 1, n_outer + 1)[:-1] # escludo 0 che coincide con d_min
        # t ribaltato per andare da lontano a vicino
        t = np.flip(t) 
        # Formula esponenziale: (e^(alpha*t) - 1)
        # Usiamo una scala logaritmica più semplice
        offsets = H * factor * (np.power(10.0, t*2) - 1) / 99.0
        c_lower = d_min - offsets
        c_lower = np.sort(c_lower) # Ordinamento crescente
        
        # Zona superiore (esponenziale crescente da d_max)
        t = np.linspace(0, 1, n_outer + 1)[1:] # escludo 0
        offsets = H * factor * (np.power(10.0, t*2) - 1) / 99.0
        c_upper = d_max + offsets
        
        # Unione
        c_total = np.concatenate([c_lower, c_inner, c_upper])
        
        # Resampling per garantire esattamente n_steps
        if len(c_total) != n_steps:
            c_total = np.interp(np.linspace(0, 1, n_steps), np.linspace(0, 1, len(c_total)), c_total)
            
        return list(c_total)

    def _decomponi_singolo_punto(self, params: Tuple) -> List[float]:
        """Wrapper statico/pickle-friendly o metodo istanza thread-safe."""
        theta, c, grid_step, punti_fissi = params
        n_vect = np.array([math.cos(theta), math.sin(theta)])
        
        # Calcolo k (equazione piano: d = x*nx + y*ny = k)
        # Nella definizione classica, c è la distanza dell'asse neutro dalla fibra più compressa
        # Qui c è la coordinata assoluta lungo l'asse n_vect
        k = c
        
        # Calcolo fattore scala (curvatura)
        m_govern = self._calcola_fattore_scala(n_vect, k, punti_fissi)
        
        if m_govern == 0:
            return [0.0, 0.0, 0.0]

        N = Mx = My = 0.0
        for (x, y, area, mat) in punti_fissi:
            d = x * n_vect[0] + y * n_vect[1]
            eps = m_govern * (d - k)
            sig = mat.sigma(eps)
            
            force = sig * area
            N += force
            Mx += force * y
            My += force * x
            
        return [Mx / 1e6, My / 1e6, N / 1e3]


# ======================================================================
# THREAD CALCOLATORE (PARALLELIZZATO E OTTIMIZZATO)
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
            grid_step = float(self.ui.out_precisione.text() or "10")
            theta_steps = max(3, int(self.ui.out_angoli.text() or "24")) # Minimo 3 per triangolare
            neutral_steps = max(5, int(self.ui.out_step.text() or "30"))
        except:
            grid_step = 10.0
            theta_steps = 24
            neutral_steps = 30

        # 1. Allineamento
        self.sezione.allinea_al_centro()
        
        # 2. Pre-calcolo punti integrazione (fibers)
        # Questo viene fatto una volta sola per tutti gli angoli
        punti_integrazione = self.sezione._prepara_punti_integrazione(grid_step)
        
        # 3. Setup angoli
        theta_increment = 2 * math.pi / theta_steps
        thetas = np.linspace(0, 2 * math.pi, theta_steps, endpoint=False) # 0..360 escluso 360 per chiudere loop
        
        # Matrice risultati: [n_angoli][n_steps][3]
        # Inizializziamo a zero
        results_matrix = np.zeros((theta_steps, neutral_steps, 3))
        
        total_tasks = theta_steps * neutral_steps
        completed_tasks = 0
        
        # 4. Esecuzione parallela
        # Usiamo ThreadPoolExecutor perché gli oggetti Materiale usano 'eval' e 'compile'
        # che non sono facilmente picklable per ProcessPoolExecutor, e NumPy rilascia il GIL.
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            future_to_idx = {}
            
            for i, theta in enumerate(thetas):
                n_vect = np.array([math.cos(theta), math.sin(theta)])
                # Genera step c strutturati per questo angolo
                c_values = self.sezione._genera_step_asse_neutro_strutturato(n_vect, neutral_steps)
                
                for j, c in enumerate(c_values):
                    # Impacchetta parametri
                    params = (theta, c, grid_step, punti_integrazione)
                    # Sottomette task
                    future = executor.submit(self.sezione._decomponi_singolo_punto, params)
                    future_to_idx[future] = (i, j)
            
            # Raccolta risultati
            for future in concurrent.futures.as_completed(future_to_idx):
                i, j = future_to_idx[future]
                try:
                    res = future.result()
                    results_matrix[i, j, :] = res
                except Exception as e:
                    print(f"Errore calcolo {i},{j}: {e}")
                    results_matrix[i, j, :] = [0, 0, 0]
                
                completed_tasks += 1
                if completed_tasks % 50 == 0: # Aggiorna GUI ogni tanto
                    self.progress.emit(int(completed_tasks / total_tasks * 100))

        # Aggiungiamo la chiusura del cerchio duplicando il primo angolo alla fine?
        # Per il rendering GL_QUADS è meglio avere la matrice esatta. 
        # Chiuderemo la mesh nel rendering.
        
        print(f"Calcolo completato in {time.time() - start_time:.2f}s")
        self.progress.emit(100)
        self.calculation_done.emit(results_matrix)