"""
generatore_dati_sezioni.py
--------------------------
Risolve le sezioni definite nella struttura (sia inline che per riferimento)
e le converte nel formato necessario per l'analisi FEM:

    {sid: {"nome", "Area", "Iy", "Iz", "J_torsione", "materiale_ref", "E_ref", "G_ref"}}

Per sezioni definite come 'riferimento' (solo il nome), cerca nel database
sezioni del programma, esegue la discretizzazione a fibre (come in
pressoflessione), omogenizza i materiali rispetto alla carpenteria, e calcola
le proprieta' risultanti (A, Iy, Iz) tenendo conto anche delle barre
longitudinali omogenizzate.
"""
from __future__ import annotations

import math
from typing import Optional, Callable

from analisi.raccolta_dati import RaccoltaDati


# ==============================================================================
#  FUNZIONI GEOMETRICHE (stesse di pressoflessione/calcolo.py)
# ==============================================================================

def _bbox_rettangolo(g: dict):
    return (min(g['x0'], g['x1']), min(g['y0'], g['y1']),
            max(g['x0'], g['x1']), max(g['y0'], g['y1']))

def _bbox_poligono(g: dict):
    pts = g['punti']
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def _bbox_cerchio(g: dict):
    cx, cy = g.get('cx', 0.0), g.get('cy', 0.0)
    r = g.get('rx', g.get('r', 1.0))
    ry = g.get('ry', g.get('r', 1.0))
    return cx - r, cy - ry, cx + r, cy + ry

def _in_rettangolo(px, py, g):
    x0, y0, x1, y1 = g['x0'], g['y0'], g['x1'], g['y1']
    return min(x0, x1) <= px <= max(x0, x1) and min(y0, y1) <= py <= max(y0, y1)

def _in_poligono(px, py, g):
    pts = g['punti']
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(pts[i][0]), float(pts[i][1])
        xj, yj = float(pts[j][0]), float(pts[j][1])
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside

def _in_cerchio(px, py, g):
    cx, cy = g.get('cx', 0.0), g.get('cy', 0.0)
    rx = g.get('rx', g.get('r', 1.0))
    ry = g.get('ry', g.get('r', 1.0))
    if rx <= 0 or ry <= 0:
        return False
    return (px - cx) ** 2 / rx ** 2 + (py - cy) ** 2 / ry ** 2 <= 1.0


_BBOX_FN = {'rettangolo': _bbox_rettangolo, 'poligono': _bbox_poligono,
            'cerchio': _bbox_cerchio}
_CONTA_FN = {'rettangolo': _in_rettangolo, 'poligono': _in_poligono,
             'cerchio': _in_cerchio}

def _shape_base(tipo: str) -> str:
    return tipo.replace('foro_', '')

def _is_foro(tipo: str) -> bool:
    return tipo.startswith('foro_')


# ==============================================================================
#  COSTANTE TORSIONALE (approssimazione per sezioni rettangolari)
# ==============================================================================

def _costante_torsionale_rettangolare(b: float, h: float) -> float:
    """
    Costante torsionale di Saint-Venant per sezione rettangolare.
    b >= h.  Formula di Roark.  Unita' coerenti con input (mm -> mm^4).
    """
    if b <= 0 or h <= 0:
        return 0.0
    if b < h:
        b, h = h, b
    rapporto = h / b
    return b * h ** 3 * (1.0 / 3.0 - 0.21 * rapporto *
                         (1.0 - rapporto ** 4 / 12.0))


# ==============================================================================
#  GENERATORE
# ==============================================================================

class GeneratoreDatiSezioni:
    """Risolve le sezioni della struttura per l'analisi FEM."""

    def __init__(self, main_window) -> None:
        self._raccolta = RaccoltaDati(main_window)

    def risolvi_sezioni(self, sezioni_struttura: dict,
                        materiali_analisi: dict) -> dict[int, dict]:
        """
        Converte le sezioni parsate dalla struttura nel formato analisi.

        Parametri
        ---------
        sezioni_struttura : dict
            {sid: {...}} dal parser della struttura.
        materiali_analisi : dict
            {mid: {"nome", "E", "G", ...}} gia' risolti dal
            GeneratoreDatiMateriali.

        Ritorna
        -------
        dict[int, dict]
            {sid: {"nome", "Area", "Iy", "Iz", "J_torsione",
                   "materiale_ref", "E_ref", "G_ref"}}
        """
        # Mappa nome_materiale -> dati analisi
        mat_per_nome: dict[str, dict] = {}
        for mid, md in materiali_analisi.items():
            mat_per_nome[md["nome"]] = md
            mat_per_nome[str(mid)] = md

        risultato: dict[int, dict] = {}

        for sid, sez_data in sezioni_struttura.items():
            tipo = sez_data.get("tipo", "inline")
            nome = sez_data.get("nome", f"Sez_{sid}")

            if tipo == "inline":
                # Sezione definita direttamente nel testo
                mat_ref_key = sez_data.get("materiale", "")
                mat_ref = mat_per_nome.get(mat_ref_key, {})
                E_ref = mat_ref.get("E", 30000.0)
                G_ref = mat_ref.get("G", E_ref / 2.5)
                Area = float(sez_data.get("Area", 0.0))
                Iy = float(sez_data.get("Iy", 0.0))
                Iz = float(sez_data.get("Iz", 0.0))

                # Stima J torsionale dal rapporto Iy/Iz
                J_tors = self._stima_J_da_inerzie(Area, Iy, Iz)

                risultato[sid] = {
                    "nome": nome,
                    "Area": Area,
                    "Iy": Iy,
                    "Iz": Iz,
                    "J_torsione": J_tors,
                    "materiale_ref": mat_ref_key,
                    "E_ref": E_ref,
                    "G_ref": G_ref,
                }
            else:
                # Riferimento: discretizza e omogenizza
                risolto = self._risolvi_riferimento(nome, mat_per_nome)
                if risolto is not None:
                    risultato[sid] = risolto
                else:
                    print(f"WARN  Sezione '{nome}' (id={sid}) non trovata "
                          f"nel database. Verranno usate proprieta' di default.")
                    risultato[sid] = self._sezione_default(nome)

        return risultato

    def _risolvi_riferimento(self, nome_sezione: str,
                             mat_per_nome: dict) -> Optional[dict]:
        """
        Cerca la sezione nel database, la discretizza a fibre, omogenizza
        i materiali rispetto alla carpenteria e calcola A, Iy, Iz.
        """
        dati_sez = self._raccolta.dati_sezione(nome_sezione)
        if dati_sez is None:
            return None

        elementi = dati_sez.get("elementi", {})
        carpenteria = elementi.get("carpenteria", [])
        barre = elementi.get("barre", [])

        if not carpenteria:
            return None

        # ---- Determina il materiale di riferimento (prima carpenteria) ----
        mat_rif_nome = ""
        E_ref = 30000.0
        G_ref = 12500.0

        for elem in carpenteria:
            mat_nome = elem.get("materiale", "")
            if not mat_nome:
                continue
            mat_rif_nome = mat_nome
            # Cerca nel database materiali dell'analisi o nel database programma
            if mat_nome in mat_per_nome:
                E_ref = mat_per_nome[mat_nome].get("E", 30000.0)
                G_ref = mat_per_nome[mat_nome].get("G", 12500.0)
            else:
                dati_mat = self._raccolta.dati_materiale(mat_nome)
                if dati_mat:
                    E_ref = float(dati_mat.get("m_elastico", 30000.0))
                    G_ref = float(dati_mat.get("m_taglio", 12500.0))
                    poisson = float(dati_mat.get("poisson", 0.2))
                    if G_ref <= 0 and E_ref > 0:
                        G_ref = E_ref / (2.0 * (1.0 + poisson))
            break

        # ---- Discretizzazione a fibre ----
        solidi, fori = self._classifica_elementi(carpenteria)

        if not solidi:
            return None

        # Bounding box
        bb_list = [_BBOX_FN[base](geom) for base, geom, _ in solidi]
        gx0 = min(b[0] for b in bb_list)
        gy0 = min(b[1] for b in bb_list)
        gx1 = max(b[2] for b in bb_list)
        gy1 = max(b[3] for b in bb_list)

        W = gx1 - gx0
        H = gy1 - gy0

        # Passo griglia ~ 2 mm per buona precisione
        gs = 2.0
        nx = max(1, int(round(W / gs)))
        ny = max(1, int(round(H / gs)))
        gs_x = W / nx
        gs_y = H / ny
        area_fibra = gs_x * gs_y

        # Accumula fibre omogenizzate
        fibre_x = []
        fibre_y = []
        fibre_a = []   # area omogenizzata

        for j in range(ny):
            py = gy0 + j * gs_y + gs_y / 2.0
            for i in range(nx):
                px = gx0 + i * gs_x + gs_x / 2.0

                # Controlla fori
                in_foro = False
                for base, geom, _ in fori:
                    if _CONTA_FN[base](px, py, geom):
                        in_foro = True
                        break
                if in_foro:
                    continue

                # Trova a quale solido appartiene
                for base, geom, mat_nome in solidi:
                    if _CONTA_FN[base](px, py, geom):
                        # Rapporto di omogenizzazione
                        E_fibra = self._E_materiale(mat_nome, mat_per_nome)
                        n = E_fibra / E_ref if E_ref > 0 else 1.0
                        fibre_x.append(px)
                        fibre_y.append(py)
                        fibre_a.append(area_fibra * n)
                        break

        # ---- Aggiungi barre longitudinali omogenizzate ----
        for elem in barre:
            if elem.get("tipo") != "barra":
                continue
            g = elem.get("geometria", {})
            cx = float(g.get("cx", 0.0))
            cy = float(g.get("cy", 0.0))
            r = float(g.get("r", 0.0))
            if r <= 0:
                continue
            mat_nome = elem.get("materiale", "")
            E_barra = self._E_materiale(mat_nome, mat_per_nome)
            n = E_barra / E_ref if E_ref > 0 else 1.0
            area_barra = math.pi * r * r
            fibre_x.append(cx)
            fibre_y.append(cy)
            fibre_a.append(area_barra * n)

        if not fibre_a:
            return None

        # ---- Calcolo proprieta' omogenizzate ----
        A_tot = sum(fibre_a)
        if A_tot < 1e-12:
            return None

        # Baricentro omogenizzato
        xc = sum(a * x for a, x in zip(fibre_a, fibre_x)) / A_tot
        yc = sum(a * y for a, y in zip(fibre_a, fibre_y)) / A_tot

        # Momenti di inerzia rispetto al baricentro (mm^4)
        Iy = sum(a * (y - yc) ** 2 for a, y in zip(fibre_a, fibre_y))
        Iz = sum(a * (x - xc) ** 2 for a, x in zip(fibre_a, fibre_x))

        # Costante torsionale (approssimazione dal bounding box)
        J_tors = _costante_torsionale_rettangolare(W, H)

        # Converti da mm^2/mm^4 a m^2/m^4 per l'analisi
        # Le coordinate nel database sezioni sono in mm, quindi:
        A_m2 = A_tot * 1e-6       # mm^2 -> m^2
        Iy_m4 = Iy * 1e-12        # mm^4 -> m^4
        Iz_m4 = Iz * 1e-12        # mm^4 -> m^4
        J_m4 = J_tors * 1e-12     # mm^4 -> m^4

        return {
            "nome": nome_sezione,
            "Area": A_m2,
            "Iy": Iy_m4,
            "Iz": Iz_m4,
            "J_torsione": J_m4,
            "materiale_ref": mat_rif_nome,
            "E_ref": E_ref,         # MPa
            "G_ref": G_ref,         # MPa
        }

    def _classifica_elementi(self, carpenteria: list):
        """Separa solidi e fori dalla carpenteria."""
        solidi = []
        fori = []
        for elem in carpenteria:
            tipo = elem.get("tipo", "")
            base = _shape_base(tipo)
            if base not in _CONTA_FN:
                continue
            geom = elem.get("geometria", {})
            mat = elem.get("materiale", "")
            if _is_foro(tipo):
                fori.append((base, geom, mat))
            else:
                solidi.append((base, geom, mat))
        return solidi, fori

    def _E_materiale(self, nome_mat: str, mat_per_nome: dict) -> float:
        """Restituisce il modulo elastico di un materiale per nome."""
        if nome_mat in mat_per_nome:
            return float(mat_per_nome[nome_mat].get("E", 30000.0))
        dati = self._raccolta.dati_materiale(nome_mat)
        if dati:
            return float(dati.get("m_elastico", 30000.0))
        return 30000.0

    @staticmethod
    def _stima_J_da_inerzie(A: float, Iy: float, Iz: float) -> float:
        """Stima la costante torsionale dalle inerzie."""
        if A <= 0:
            return 0.0
        # Stima b e h dal rapporto A e inerzie
        # Per un rettangolo: Iy = b*h^3/12, Iz = h*b^3/12, A = b*h
        # -> h = sqrt(12*Iy/A), b = sqrt(12*Iz/A)
        if Iy > 0 and Iz > 0:
            h_est = math.sqrt(12.0 * Iy / A) if A > 0 else 1.0
            b_est = math.sqrt(12.0 * Iz / A) if A > 0 else 1.0
            return _costante_torsionale_rettangolare(b_est, h_est)
        return (Iy + Iz) * 0.5

    @staticmethod
    def _sezione_default(nome: str) -> dict:
        """Sezione di fallback (rettangolare 30x30 cm)."""
        b, h = 0.30, 0.30
        A = b * h
        Iy = b * h ** 3 / 12.0
        Iz = h * b ** 3 / 12.0
        J = _costante_torsionale_rettangolare(b * 1e3, h * 1e3) * 1e-12
        return {
            "nome": nome,
            "Area": A,
            "Iy": Iy,
            "Iz": Iz,
            "J_torsione": J,
            "materiale_ref": "",
            "E_ref": 30000.0,
            "G_ref": 12500.0,
        }
