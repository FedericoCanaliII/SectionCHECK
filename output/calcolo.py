import math
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate

# ======================================================================
# MODELLO MATERIALE (LEGGERMENTE RIPULITO)
# ======================================================================
class Materiale:
    """Rappresentazione semplice di un materiale con intervalli di
    comportamento (espressioni in funzione della deformazione 'x').

    Manteniamo la logica di valutazione delle espressioni così com'era,
    usando compile/eval con builtins disabilitati.
    """

    def __init__(self, matrice: List[Tuple[str, float, float]], nome: str = "") -> None:
        self.nome = nome
        self.intervalli: List[Dict[str, Any]] = []
        print (List[Dict[str, Any]])
        self.ult_compression = 0.0
        self.ult_tension = 0.0

        # Analizza gli intervalli per determinare i limiti di deformazione
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
        """Valuta lo sforzo per una data deformazione eps cercando l'intervallo
        adatto. Se non trova intervallo restituisce 0.0 (stesso comportamento).
        """
        for intervallo in self.intervalli:
            if intervallo['low'] <= eps <= intervallo['high']:
                return eval(intervallo['expr'], {'__builtins__': None}, {'x': eps})
        return 0.0


# ======================================================================
# SEZIONE RINFORZATA (PULITA E ORDINATA)
# ======================================================================
class SezioneRinforzata:
    """Classe per rappresentare la sezione composta da forme, barre e
    rinforzi. Qui abbiamo riordinato i metodi e aggiunto la funzionalitÃ 
    di recentramento (allineamento al centroide) richiesta.

    Nota importante: nel metodo diagramma_interazione_3D il parametro
    `theta_step` ora rappresenta il **numero di passi di rotazione**. L'angolo
    incrementale effettivo in gradi sarÃ  calcolato come 360 / theta_step.
    """

    def __init__(self, elementi: List[Tuple], materiali_dict: Dict[str, Materiale]) -> None:
        self.geometria: List[Tuple[Polygon, Optional[Materiale]]] = []
        self.barre: List[Tuple[float, float, float, Optional[Materiale]]] = []
        self.rinforzi: List[Tuple[Polygon, float, Optional[Materiale]]] = []
        self.materiali = materiali_dict

        # Caricamento elementi (pulito e leggibile)
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

        # Calcola e memorizza proprietà geometriche iniziali
        self._aggiorna_proprieta_geometriche()

        # Calcola area totale e punti di compressione/trazione centrata
        self.area_totale = self._calcola_area_totale()
        self.punto_compressione_centrata = self._calcola_punto_compressione_centrata()
        self.punto_trazione_centrata = self._calcola_punto_trazione_centrata()

        # Flag che indica se la sezione Ã¨ stata recentrata
        self._recentrata = False

    # ---------------------------- Geometria ----------------------------
    def _aggiorna_proprieta_geometriche(self) -> None:
        """Ricalcola bounding box e dimensioni basandosi sull'attuale geometria.
        Deve essere chiamato ogni volta che le geometrie vengono traslate.
        """
        all_points = []
        for poly, _ in self.geometria:
            all_points.extend(poly.exterior.coords)
        for poly, _, _ in self.rinforzi:
            all_points.extend(poly.exterior.coords)
        for bar in self.barre:
            all_points.append((bar[0], bar[1]))

        if not all_points:
            self.x_min = self.y_min = 0.0
            self.x_max = self.y_max = 0.0
            self.dx = self.dy = self.diametro = 0.0
        else:
            all_points = np.array(all_points)
            self.x_min, self.y_min = np.min(all_points, axis=0)
            self.x_max, self.y_max = np.max(all_points, axis=0)
            self.dx = self.x_max - self.x_min
            self.dy = self.y_max - self.y_min
            self.diametro = math.hypot(self.dx, self.dy)

    def _crea_rettangolo(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Polygon:
        x_coords = sorted([p1[0], p2[0]])
        y_coords = sorted([p1[1], p2[1]])
        return Polygon([
            (x_coords[0], y_coords[0]),
            (x_coords[1], y_coords[0]),
            (x_coords[1], y_coords[1]),
            (x_coords[0], y_coords[1])
        ])

    def _crea_rinforzo_rettangolare(self, p0: Tuple[float, float], p1: Tuple[float, float], th: float) -> Polygon:
        line = LineString([p0, p1])
        return line.buffer(th / 2, cap_style=2, join_style=2)

    # ------------------------ Materiale per punto ------------------------
    def _get_materiale_per_punto(self, x: float, y: float) -> Optional[Materiale]:
        p = Point(x, y)

        # PrioritÃ  ai rinforzi
        for poly, _, mat in self.rinforzi:
            if poly.contains(p):
                return mat

        for poly, mat in self.geometria:
            if poly.contains(p):
                return mat

        return None

    # ------------------------ ProprietÃ  globali ------------------------
    def _calcola_area_totale(self) -> float:
        area = 0.0
        for poly, _ in self.geometria:
            area += poly.area
        for poly, _, _ in self.rinforzi:
            area += poly.area
        for bar in self.barre:
            area += math.pi * (bar[2] / 2) ** 2
        return area

    def centroide_sezione(self) -> Tuple[float, float]:
        """Calcola il centroide dell'intera sezione (area-weighted). Le barre
        sono considerate con la loro area circolare e contribuiscono con il
        loro baricentro pari alla posizione (x,y).
        """
        if self.area_totale == 0:
            return 0.0, 0.0

        Sx = 0.0
        Sy = 0.0

        for poly, mat in self.geometria:
            A = poly.area
            c = poly.centroid
            Sx += A * c.x
            Sy += A * c.y

        for poly, _, mat in self.rinforzi:
            A = poly.area
            c = poly.centroid
            Sx += A * c.x
            Sy += A * c.y

        for bar in self.barre:
            x, y, diam, mat = bar
            A = math.pi * (diam / 2) ** 2
            Sx += A * x
            Sy += A * y

        cx = Sx / self.area_totale
        cy = Sy / self.area_totale
        return cx, cy

    def allinea_al_centro(self) -> None:
        """Recentra tutte le geometrie e le barre sul centroide calcolato.
        Questo modifica le coordinate interne della sezione in modo che il
        nuovo centro di riferimento (0,0) corrisponda al centroide.
        """
        cx, cy = self.centroide_sezione()
        if cx == 0 and cy == 0 and not any([self.geometria, self.rinforzi, self.barre]):
            # Se la sezione Ã¨ vuota o il centroide Ã¨ (0,0) non fare nulla
            return

        # Trasla geometrie
        for idx, (poly, mat) in enumerate(self.geometria):
            self.geometria[idx] = (translate(poly, xoff=-cx, yoff=-cy), mat)

        for idx, (poly, th, mat) in enumerate(self.rinforzi):
            self.rinforzi[idx] = (translate(poly, xoff=-cx, yoff=-cy), th, mat)

        # Trasla barre
        for idx, bar in enumerate(self.barre):
            x, y, diam, mat = bar
            self.barre[idx] = (x - cx, y - cy, diam, mat)

        # Aggiorna proprietÃ  derivate
        self._aggiorna_proprieta_geometriche()
        self.area_totale = self._calcola_area_totale()
        self.punto_compressione_centrata = self._calcola_punto_compressione_centrata()
        self.punto_trazione_centrata = self._calcola_punto_trazione_centrata()
        self._recentrata = True

    # ------------------------ Calcoli di estremi/proiezioni ------------------------
    def _trova_estremi_proiezione(self, n_vect: np.ndarray) -> Tuple[float, float]:
        all_d = []

        for poly, _ in self.geometria:
            coords = np.array(poly.exterior.coords)
            proj = np.dot(coords, n_vect)
            all_d.extend([np.min(proj), np.max(proj)])

        for poly, _, _ in self.rinforzi:
            coords = np.array(poly.exterior.coords)
            proj = np.dot(coords, n_vect)
            all_d.extend([np.min(proj), np.max(proj)])

        for bar in self.barre:
            proj = np.dot([bar[0], bar[1]], n_vect)
            all_d.append(proj)

        return (min(all_d), max(all_d)) if all_d else (0.0, 0.0)

    def _calcola_fattore_scala(self, n_vect: np.ndarray, k: float, punti: List[Tuple[float, float, float, Optional[Materiale]]]) -> float:
        m_candidates: List[float] = []

        for (x, y, area, mat) in punti:
            if mat is None:
                continue

            d = float(np.dot([x, y], n_vect))
            delta_d = d - k

            if delta_d < 0:
                eps_lim = mat.ult_compression
                if delta_d == 0:
                    continue
                m_candidate = eps_lim / delta_d
                if m_candidate > 0:
                    m_candidates.append(m_candidate)

            elif delta_d > 0:
                eps_lim = mat.ult_tension
                if delta_d == 0:
                    continue
                m_candidate = eps_lim / delta_d
                if m_candidate > 0:
                    m_candidates.append(m_candidate)

        return min(m_candidates) if m_candidates else 0.0

    def _genera_posizioni_asse_neutro(self, n_vect: np.ndarray, n_punti: int = 50, fattore_scala: float = 50.0, growth_rate: float = 4.0) -> List[float]:
        d_min, d_max = self._trova_estremi_proiezione(n_vect)
        diametro_proiezione = d_max - d_min
        if diametro_proiezione <= 0:
            diametro_proiezione = max(self.dx, self.dy, 1.0)

        limite_inf = d_min - fattore_scala * diametro_proiezione
        limite_sup = d_max + fattore_scala * diametro_proiezione

        punti_interni = max(10, n_punti // 3)
        s_min = diametro_proiezione / max(1, (punti_interni - 1))

        interior = np.arange(d_min, d_max + s_min * 0.5, s_min)

        left = []
        x = d_min
        safety = 0
        while x > limite_inf and safety < 20000:
            dist = d_min - x
            step = s_min * (1.0 + growth_rate * (dist / max(diametro_proiezione, 1e-9)))
            x = x - step
            if x < limite_inf:
                x = limite_inf
            left.append(x)
            safety += 1

        right = []
        x = d_max
        safety = 0
        while x < limite_sup and safety < 20000:
            dist = x - d_max
            step = s_min * (1.0 + growth_rate * (dist / max(diametro_proiezione, 1e-9)))
            x = x + step
            if x > limite_sup:
                x = limite_sup
            right.append(x)
            safety += 1

        posizioni = np.concatenate([np.array(left[::-1]), interior, np.array(right)])
        posizioni = posizioni[(posizioni >= limite_inf) & (posizioni <= limite_sup)]
        posizioni = np.unique(np.round(posizioni, decimals=12))
        return list(posizioni)

    # ------------------------ Integrazione della sezione ------------------------
    def _decomponi_sezione(self, theta: float, c: float, grid_step: float) -> Tuple[float, float, float]:
        n_vect = np.array([math.cos(theta), math.sin(theta)])

        d_min, d_max = self._trova_estremi_proiezione(n_vect)

        compressione_lato = d_min if c >= 0 else d_max
        k = compressione_lato + c

        punti_integrazione: List[Tuple[float, float, float, Optional[Materiale]]] = []

        # Griglia regolare per la geometria di base
        nx = max(1, int(self.dx / grid_step))
        ny = max(1, int(self.dy / grid_step))

        for i in range(nx + 1):
            x = self.x_min + i * (self.dx / nx) if nx > 0 else self.x_min
            for j in range(ny + 1):
                y = self.y_min + j * (self.dy / ny) if ny > 0 else self.y_min
                mat = self._get_materiale_per_punto(x, y)
                if mat:
                    punti_integrazione.append((x, y, grid_step ** 2, mat))

        # Barre (punti discreti)
        for bar in self.barre:
            x, y, diam, mat = bar
            area = math.pi * (diam / 2) ** 2
            punti_integrazione.append((x, y, area, mat))

        # Centroidi dei rinforzi
        for poly, _, mat in self.rinforzi:
            centroid = poly.centroid
            punti_integrazione.append((centroid.x, centroid.y, poly.area, mat))

        # Calcola fattore di scala
        m_govern = self._calcola_fattore_scala(n_vect, k, punti_integrazione)

        # Integrazione
        N = Mx = My = 0.0
        for (x, y, area, mat) in punti_integrazione:
            d = float(np.dot([x, y], n_vect))
            eps = m_govern * (d - k)
            sig = mat.sigma(eps)
            N += sig * area
            Mx += sig * area * y
            My += sig * area * x

        return N, Mx, My

    # ------------------------ Punti di compressione/trazione centrata ------------------------
    def _calcola_punto_compressione_centrata(self) -> Tuple[float, float, float]:
        N = Mx = My = 0.0

        for poly, mat in self.geometria:
            if mat:
                centroid = poly.centroid
                sig = mat.sigma(mat.ult_compression)
                area = poly.area
                N += sig * area
                Mx += sig * area * centroid.y
                My += sig * area * centroid.x

        for poly, _, mat in self.rinforzi:
            if mat:
                centroid = poly.centroid
                sig = mat.sigma(mat.ult_compression)
                area = poly.area
                N += sig * area
                Mx += sig * area * centroid.y
                My += sig * area * centroid.x

        for bar in self.barre:
            x, y, diam, mat = bar
            if mat:
                sig = mat.sigma(mat.ult_compression)
                area = math.pi * (diam / 2) ** 2
                N += sig * area
                Mx += sig * area * y
                My += sig * area * x

        # Conserviamo la medesima scala di output (kNm, kNm, kN) come prima
        return (Mx / 1e6, My / 1e6, N / 1e3)

    def _calcola_punto_trazione_centrata(self) -> Tuple[float, float, float]:
        N = Mx = My = 0.0

        for poly, mat in self.geometria:
            if mat:
                centroid = poly.centroid
                sig = mat.sigma(mat.ult_tension)
                area = poly.area
                N += sig * area
                Mx += sig * area * centroid.y
                My += sig * area * centroid.x

        for poly, _, mat in self.rinforzi:
            if mat:
                centroid = poly.centroid
                sig = mat.sigma(mat.ult_tension)
                area = poly.area
                N += sig * area
                Mx += sig * area * centroid.y
                My += sig * area * centroid.x

        for bar in self.barre:
            x, y, diam, mat = bar
            if mat:
                sig = mat.sigma(mat.ult_tension)
                area = math.pi * (diam / 2) ** 2
                N += sig * area
                Mx += sig * area * y
                My += sig * area * x

        return (Mx / 1e6, My / 1e6, N / 1e3)

    # ------------------------ Diagramma di interazione ------------------------
    def diagramma_interazione_3D(self, theta_step: float = 15.0, punti_per_angolo: int = 40, grid_step: float = 20.0, verbose: bool = True) -> np.ndarray:
        """Calcola il dominio di interazione. Prima di iniziare il calcolo
        allinea la sezione al proprio centroide (comportamento richiesto).

        Nota: `theta_step` qui rappresenta **numero di passi**. L'angolo
        incrementale in gradi usato internamente viene calcolato come
        `theta_increment_deg = 360.0 / max(1, int(theta_step))` e convertito in
        radianti prima di creare l'array di thetas.
        """
        # Se non Ã¨ stata recentrata, lo facciamo ora
        if not self._recentrata:
            if verbose:
                print("Allineo la sezione al centroide prima della simulazione...")
            self.allinea_al_centro()

        # Interpretiamo theta_step come numero di passi
        try:
            steps = max(1, int(theta_step))
        except Exception:
            steps = 24
        theta_increment_deg = 360.0 / steps
        thetas = np.deg2rad(np.arange(0.0, 360.0, theta_increment_deg))

        results: List[List[float]] = []
        start_time = time.time()

        # Aggiungi i punti di compressione e trazione centrata
        results.append(list(self.punto_compressione_centrata))
        results.append(list(self.punto_trazione_centrata))

        for i, theta in enumerate(thetas):
            if verbose:
                print(f'Processing angle {i+1}/{len(thetas)} (theta={math.degrees(theta):.2f} deg)...')

            n_vect = np.array([math.cos(theta), math.sin(theta)])
            posizioni_c = self._genera_posizioni_asse_neutro(n_vect, punti_per_angolo)

            for c in posizioni_c:
                try:
                    N, Mx, My = self._decomponi_sezione(theta, c, grid_step)
                    results.append([Mx / 1e6, My / 1e6, N / 1e3])
                except Exception as e:
                    if verbose:
                        print(f"Errore per theta={theta:.2f}, c={c:.2f}: {e}")

        if verbose:
            print(f'Completed in {time.time()-start_time:.1f} seconds')
            print(f'Generated {len(results)} points')
            print(f'Compressione centrata: {self.punto_compressione_centrata}')
            print(f'Trazione centrata: {self.punto_trazione_centrata}')

        return np.array(results)


# ======================================================================
# THREAD PER CALCOLO ASINCRONO (VERSIONE RIPULITA)
# ======================================================================
from PyQt5.QtCore import QThread, pyqtSignal


class DomainCalculator(QThread):
    calculation_done = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, sezione: SezioneRinforzata, ui, parent=None):
        super().__init__(parent)
        self.sezione = sezione
        self.ui = ui

    def run(self) -> None:
        # Leggi parametri dall'interfaccia utente
        try:
            grid_step = float(self.ui.out_precisione.text() or "10")
        except Exception:
            grid_step = 10.0
            
        try:
            theta_steps = int(self.ui.out_angoli.text() or "18")
            theta_steps = max(1, theta_steps)
        except Exception:
            theta_steps = 18

        try:
            neutral_step = int(self.ui.out_step.text() or "30")
        except Exception:
            neutral_step = 30

        # Allineamento richiesto: recentriamo la sezione al centroide prima di iniziare
        print("DomainCalculator: allineamento sezione al centroide prima del calcolo...")
        self.sezione.allinea_al_centro()

        # Calcolo incremento angolare in gradi a partire dal numero di passi
        theta_increment_deg = 360.0 / theta_steps
        thetas = np.deg2rad(np.arange(0.0, 360.0, theta_increment_deg))

        # Pre-calcolo delle posizioni per ogni angolo per poter stimare il progresso
        posizioni_per_theta = []
        for theta in thetas:
            n_vect = np.array([math.cos(theta), math.sin(theta)])
            pos_list = self.sezione._genera_posizioni_asse_neutro(n_vect, neutral_step)
            posizioni_per_theta.append(pos_list)

        per_angle_totals = [len(pl) for pl in posizioni_per_theta]
        overall_total = 2 + sum(per_angle_totals)  # includiamo i due punti centrati

        count = 0
        results: List[List[float]] = []

        # Aggiungi i punti di compressione/trazione centrata
        results.append(list(self.sezione.punto_compressione_centrata))
        results.append(list(self.sezione.punto_trazione_centrata))
        count += 2

        # Emissione progresso iniziale
        self.progress.emit(int(count / overall_total * 100) if overall_total > 0 else 100)

        # Loop principale
        for i, theta in enumerate(thetas):
            posizioni_c = posizioni_per_theta[i]

            for c in posizioni_c:
                try:
                    N, Mx, My = self.sezione._decomponi_sezione(theta, c, grid_step)
                    results.append([Mx / 1e6, My / 1e6, N / 1e3])
                except Exception:
                    results.append([0, 0, 0])

                count += 1
                percent = int(count / overall_total * 100) if overall_total > 0 else 100
                self.progress.emit(percent)

        # Emissione finale
        self.progress.emit(100)
        self.calculation_done.emit(np.array(results))
