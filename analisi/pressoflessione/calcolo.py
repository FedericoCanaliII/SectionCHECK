"""
calcolo.py
----------
Motore di calcolo per la verifica a pressoflessione (SLU e SLE) di sezioni
generiche in calcestruzzo armato, acciaio, e sezioni miste.

Caratteristiche:
  - Discretizzazione a fibre ottimizzata (griglia adattiva al bounding box per copertura 100%)
  - Fori/vuoti trattati come zone non reagenti
  - Staffe escluse dal calcolo (non influiscono sulla resistenza)
  - SLU: approccio a fibra con legami costitutivi non-lineari, bisection sull'equilibrio
  - SLE: Newton-Raphson smorzato (damped) per stabilità sulle sezioni fessurate
  - Gestione rigorosa del floating-point sui limiti di deformazione
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np


# ==============================================================================
# COSTANTI
# ==============================================================================
_BISECT_ITER   = 100       # massimo iterazioni bisection SLU
_SLE_MAX_ITER  = 80        # massimo iterazioni schema fessurativo SLE
_EPS_CRACK     = 1e-8      # soglia di trazione per dichiarare fessurata una fibra
_SIGMA_TOL     = 0.01      # kN – tolleranza equilibrio normale (bisection)
_SAFE_MATH     = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
_SAFE_MATH['abs'] = abs


# ==============================================================================
# LEGAME COSTITUTIVO
# ==============================================================================

class LegameCostitutivo:
    """Legame sigma-epsilon definito come funzione a tratti."""

    def __init__(self, segmenti: List[dict]) -> None:
        self._segmenti: List[Tuple[float, float, Callable]] = []
        for seg in segmenti:
            try:
                eps_min = float(seg.get('eps_min', 0.0))
                eps_max = float(seg.get('eps_max', 0.0))
                formula = str(seg.get('formula', '0'))
                fn = self._compila(formula)
                self._segmenti.append((eps_min, eps_max, fn))
            except Exception as e:
                print(f"WARN  LegameCostitutivo: segmento ignorato ({e})")

        self._segmenti.sort(key=lambda s: s[0])

    @staticmethod
    def _compila(formula: str) -> Callable[[float], float]:
        safe_pattern = re.compile(r'\b(import|exec|eval|open|os|sys|__)\b')
        if safe_pattern.search(formula):
            raise ValueError(f"Formula non sicura: {formula}")
        code = compile(formula, '<legame>', 'eval')
        def _fn(x: float) -> float:
            return float(eval(code, {"__builtins__": {}}, {**_SAFE_MATH, 'x': x}))
        return _fn

    def sigma(self, eps: float) -> float:
        """Restituisce la tensione [MPa] per la deformazione eps data, con tolleranza."""
        tol = 1e-8  # Assorbe l'errore di macchina sui bordi estremi del dominio
        for eps_min, eps_max, fn in self._segmenti:
            if (eps_min - tol) <= eps <= (eps_max + tol):
                # Clamp di sicurezza per evitare crash matematici ai bordi delle radici
                e_safe = max(eps_min, min(eps, eps_max))
                try:
                    return fn(e_safe)
                except Exception:
                    return 0.0
        return 0.0

    @property
    def eps_min(self) -> float:
        return self._segmenti[0][0] if self._segmenti else -1.0

    @property
    def eps_max(self) -> float:
        return self._segmenti[-1][1] if self._segmenti else 1.0

    def modulo_tangente_iniziale(self, side: str = 'compression') -> float:
        delta = 1e-5 if side == 'tension' else -1e-5
        try:
            s = self.sigma(delta)
            if abs(s) > 1e-12:
                return abs(s / delta)
        except Exception:
            pass
        delta2 = 5e-4 if side == 'tension' else -5e-4
        try:
            s2 = self.sigma(delta2)
            if abs(s2) > 1e-12:
                return abs(s2 / delta2)
        except Exception:
            pass
        return 30_000.0


# ==============================================================================
# FIBRA E GEOMETRIA
# ==============================================================================

class Fibra:
    __slots__ = ('x', 'y', 'area', 'legame_slu', 'legame_sle', 'is_hole')

    def __init__(self, x: float, y: float, area: float,
                 legame_slu: LegameCostitutivo, legame_sle: LegameCostitutivo,
                 is_hole: bool = False) -> None:
        self.x = x
        self.y = y
        self.area = area
        self.legame_slu = legame_slu
        self.legame_sle = legame_sle
        self.is_hole = is_hole


def _bbox_rettangolo(g: dict) -> Tuple[float, float, float, float]:
    return (min(g['x0'], g['x1']), min(g['y0'], g['y1']),
            max(g['x0'], g['x1']), max(g['y0'], g['y1']))

def _bbox_poligono(g: dict) -> Tuple[float, float, float, float]:
    pts = g['punti']
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def _bbox_cerchio(g: dict) -> Tuple[float, float, float, float]:
    cx, cy = g.get('cx', 0.0), g.get('cy', 0.0)
    r = g.get('rx', g.get('r', 1.0))
    ry = g.get('ry', g.get('r', 1.0))
    return cx - r, cy - ry, cx + r, cy + ry

def _in_rettangolo(px: float, py: float, g: dict) -> bool:
    x0, y0, x1, y1 = g['x0'], g['y0'], g['x1'], g['y1']
    return min(x0, x1) <= px <= max(x0, x1) and min(y0, y1) <= py <= max(y0, y1)

def _in_poligono(px: float, py: float, g: dict) -> bool:
    pts = g['punti']
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(pts[i][0]), float(pts[i][1])
        xj, yj = float(pts[j][0]), float(pts[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside

def _in_cerchio(px: float, py: float, g: dict) -> bool:
    cx, cy = g.get('cx', 0.0), g.get('cy', 0.0)
    rx, ry = g.get('rx', g.get('r', 1.0)), g.get('ry', g.get('r', 1.0))
    if rx <= 0 or ry <= 0: return False
    return (px - cx) ** 2 / rx ** 2 + (py - cy) ** 2 / ry ** 2 <= 1.0

_BBOX_FN  = {'rettangolo': _bbox_rettangolo, 'poligono': _bbox_poligono, 'cerchio': _bbox_cerchio}
_CONTA_FN = {'rettangolo': _in_rettangolo,   'poligono': _in_poligono,   'cerchio': _in_cerchio}

def _shape_base(tipo: str) -> str: return tipo.replace('foro_', '')
def _is_foro(tipo: str) -> bool: return tipo.startswith('foro_')


# ==============================================================================
# SEZIONE DISCRETIZZATA
# ==============================================================================

class SezioneDiscretizzata:
    def __init__(self, dati_sezione: dict, risolvi_mat: Callable[[str], Optional[dict]], grid_step: float = 10.0) -> None:
        self._dati      = dati_sezione
        self._mat_fn    = risolvi_mat
        self.grid_step  = max(grid_step, 0.5)
        self._cache_slu: dict[str, LegameCostitutivo] = {}
        self._cache_sle: dict[str, LegameCostitutivo] = {}
        self._fibre: Optional[List[Fibra]] = None

    def _legame(self, nome_mat: str, tipo: str) -> LegameCostitutivo:
        cache = self._cache_slu if tipo == 'slu' else self._cache_sle
        if nome_mat not in cache:
            mat_dati = self._mat_fn(nome_mat)
            segmenti = mat_dati.get(tipo, []) if mat_dati else []
            if not segmenti and tipo == 'sle':
                segmenti = mat_dati.get('slu', []) if mat_dati else []
            cache[nome_mat] = LegameCostitutivo(segmenti)
        return cache[nome_mat]

    def _discretizza(self) -> List[Fibra]:
        """Adatta i passi griglia al Bounding Box per garantire la copertura al 100%."""
        elementi = self._dati.get('elementi', {})
        carpenteria = elementi.get('carpenteria', [])
        barre       = elementi.get('barre', [])

        solidi: list = []
        fori:   list = []
        for elem in carpenteria:
            tipo = elem.get('tipo', '')
            base = _shape_base(tipo)
            if base not in _CONTA_FN: continue
            if _is_foro(tipo): 
                fori.append((base, elem.get('geometria', {}), elem.get('materiale', '')))
            else: 
                solidi.append((base, elem.get('geometria', {}), elem.get('materiale', '')))

        fibre: List[Fibra] = []

        if solidi:
            bb_list = [_BBOX_FN[base](geom) for base, geom, _ in solidi]
            gx0 = min(b[0] for b in bb_list)
            gy0 = min(b[1] for b in bb_list)
            gx1 = max(b[2] for b in bb_list)
            gy1 = max(b[3] for b in bb_list)

            W = gx1 - gx0
            H = gy1 - gy0
            gs = self.grid_step
            
            # Suddivide l'area trovando un passo "reale" vicinissimo a gs
            # così non ci sono avanzi/resti di area sui bordi
            nx = max(1, int(round(W / gs)))
            ny = max(1, int(round(H / gs)))
            
            gs_x = W / nx
            gs_y = H / ny
            area = gs_x * gs_y

            for j in range(ny):
                py = gy0 + j * gs_y + gs_y / 2.0
                for i in range(nx):
                    px = gx0 + i * gs_x + gs_x / 2.0

                    in_foro = False
                    for base, geom, _ in fori:
                        if _CONTA_FN[base](px, py, geom):
                            in_foro = True
                            break
                    if in_foro: continue

                    for base, geom, mat_nome in solidi:
                        if _CONTA_FN[base](px, py, geom):
                            slu = self._legame(mat_nome, 'slu')
                            sle = self._legame(mat_nome, 'sle')
                            fibre.append(Fibra(float(px), float(py), area, slu, sle))
                            break

        for elem in barre:
            if elem.get('tipo') != 'barra': continue
            g  = elem.get('geometria', {})
            cx, cy, r = float(g.get('cx', 0.0)), float(g.get('cy', 0.0)), float(g.get('r', 0.0))
            if r <= 0: continue
            mat_nome = elem.get('materiale', '')
            slu = self._legame(mat_nome, 'slu')
            sle = self._legame(mat_nome, 'sle')
            fibre.append(Fibra(cx, cy, math.pi * r * r, slu, sle))

        return fibre

    @property
    def fibre(self) -> List[Fibra]:
        if self._fibre is None: self._fibre = self._discretizza()
        return self._fibre

    def forza_ridiscretizzazione(self) -> None:
        self._fibre = None

    def centroide(self) -> Tuple[float, float]:
        tot_a = sum(f.area for f in self.fibre)
        if tot_a < 1e-12: return 0.0, 0.0
        return sum(f.x * f.area for f in self.fibre)/tot_a, sum(f.y * f.area for f in self.fibre)/tot_a

    def bounds(self) -> Tuple[float, float, float, float]:
        if not self.fibre: return -100.0, 100.0, -100.0, 100.0
        xs = [f.x for f in self.fibre]; ys = [f.y for f in self.fibre]
        return min(xs), max(xs), min(ys), max(ys)


# ==============================================================================
# CALCOLATORE
# ==============================================================================

class CalcoloPressoflessione:
    def __init__(self, sezione: SezioneDiscretizzata) -> None:
        self.sezione = sezione

    @staticmethod
    def _nvect(theta_deg: float) -> np.ndarray:
        r = math.radians(theta_deg)
        return np.array([math.sin(r), math.cos(r)], dtype=float)

    def _proietta(self, nv: np.ndarray) -> Tuple[float, float, float]:
        if not self.sezione.fibre: return -100.0, 100.0, 200.0
        ds = np.array([f.x * nv[0] + f.y * nv[1] for f in self.sezione.fibre])
        return float(ds.min()), float(ds.max()), float(ds.max() - ds.min())

    def _scala_deformazione(self, nv: np.ndarray, k: float, fibre: List[Fibra], modo: str = 'slu') -> float:
        m_g = 1e12 
        for f in fibre:
            d_rel = f.x * nv[0] + f.y * nv[1] - k
            if abs(d_rel) < 1e-12: continue
            legame = f.legame_slu if modo == 'slu' else f.legame_sle
            if d_rel > 0:
                if legame.eps_max > 1e-12: m_g = min(m_g, legame.eps_max / d_rel)
            else:
                if legame.eps_min < -1e-12: m_g = min(m_g, legame.eps_min / d_rel) 
        return m_g if m_g < 1e11 else 1.0

    # ------------------------------------------------------------------
    # SLU
    # ------------------------------------------------------------------

    def _N_slu(self, nv: np.ndarray, k: float, fibre: List[Fibra]) -> float:
        m_g = self._scala_deformazione(nv, k, fibre, 'slu')
        N = sum(f.legame_slu.sigma(m_g * (f.x * nv[0] + f.y * nv[1] - k)) * f.area for f in fibre)
        return N / 1e3

    def _NM_slu(self, nv: np.ndarray, k: float, fibre: List[Fibra]) -> Tuple[float, float, list]:
        m_g = self._scala_deformazione(nv, k, fibre, 'slu')
        N = Mx = My = 0.0
        res = []
        for f in fibre:
            eps = m_g * (f.x * nv[0] + f.y * nv[1] - k)
            sig = f.legame_slu.sigma(eps)
            force = sig * f.area
            N += force; Mx += force * f.y; My += force * f.x
            res.append((f.x, f.y, f.area, eps, sig))
        return N / 1e3, (nv[0] * My + nv[1] * Mx) / 1e6, res

    def analisi_slu(self, N_Ed_kN: float, M_Ed_kNm: float, theta_deg: float = 0.0, progress_cb=None) -> dict:
        fibre = self.sezione.fibre
        nv = self._nvect(theta_deg)
        d_min, d_max, H = self._proietta(nv)

        span = max(H * 20.0, 500.0)
        k_lo, k_hi = d_min - span, d_max + span
        N_lo, N_hi = self._N_slu(nv, k_lo, fibre), self._N_slu(nv, k_hi, fibre)

        fuori_dominio = (N_Ed_kN < min(N_lo, N_hi) * 1.05 or N_Ed_kN > max(N_lo, N_hi) * 1.05)

        if N_lo < N_hi:
            k_lo, k_hi, N_lo, N_hi = k_hi, k_lo, N_hi, N_lo

        tol_N = max(abs(N_Ed_kN) * 1e-5, _SIGMA_TOL)
        k_star, conv = (k_lo + k_hi) / 2.0, False

        for i in range(_BISECT_ITER):
            if progress_cb: progress_cb(int(10 + 80 * i / _BISECT_ITER))
            k_mid = (k_lo + k_hi) / 2.0
            err = self._N_slu(nv, k_mid, fibre) - N_Ed_kN

            if abs(err) < tol_N:
                k_star, conv = k_mid, True
                break

            if err > 0: k_lo = k_mid
            else: k_hi = k_mid
            k_star = k_mid

        if progress_cb: progress_cb(90)

        N_rd, M_rd, fibre_res = self._NM_slu(nv, k_star, fibre)
        M_Ed_abs, M_Rd_abs = abs(M_Ed_kNm), abs(M_rd)

        m_g = self._scala_deformazione(nv, k_star, fibre, 'slu')
        if progress_cb: progress_cb(100)

        return {
            'tipo': 'SLU', 'verificata': (not fuori_dominio) and (M_Rd_abs >= M_Ed_abs),
            'fuori_dominio': fuori_dominio, 'convergenza': conv, 'N_Ed': N_Ed_kN, 'M_Ed': M_Ed_kNm,
            'theta_deg': theta_deg, 'N_Rd': N_rd, 'M_Rd': M_Rd_abs,
            'rapporto': M_Ed_abs / M_Rd_abs if M_Rd_abs > 1e-9 else float('inf'),
            'rapporto_MEd_MRd': M_Ed_abs / M_Rd_abs if M_Rd_abs > 1e-9 else float('inf'),
            'k_star': k_star, 'd_na': k_star, 'n_vect': tuple(nv.tolist()),
            'eps_top': m_g * (d_max - k_star), 'eps_bot': m_g * (d_min - k_star),
            'd_min': d_min, 'd_max': d_max, 'grid_step': self.sezione.grid_step, 'fibre': fibre_res,
        }

    # ------------------------------------------------------------------
    # SLE
    # ------------------------------------------------------------------

    def analisi_sle(self, N_Ed_kN: float, M_Ed_kNm: float, theta_deg: float = 0.0, progress_cb=None) -> dict:
        fibre = self.sezione.fibre
        nv = self._nvect(theta_deg)
        d_min, d_max, _ = self._proietta(nv)
        n_fib = len(fibre)

        N_target = N_Ed_kN  * 1e3
        M_target = -M_Ed_kNm * 1e6

        ds = [f.x * nv[0] + f.y * nv[1] for f in fibre]
        areas = [f.area for f in fibre]

        def _E_guess(f: Fibra) -> float:
            e = max(f.legame_sle.modulo_tangente_iniziale('compression'), 
                    f.legame_sle.modulo_tangente_iniziale('tension'))
            return max(e, 1_000.0)

        SEA = SEAd = SEAd2 = 0.0
        for i, f in enumerate(fibre):
            EiA = _E_guess(f) * areas[i]
            SEA += EiA; SEAd += EiA * ds[i]; SEAd2 += EiA * ds[i] * ds[i]

        det0 = SEA * SEAd2 - SEAd * SEAd
        if abs(det0) > 1e-8:
            eps0  = ( SEAd2 * N_target - SEAd  * M_target) / det0
            kappa = (-SEAd  * N_target + SEA   * M_target) / det0
        elif abs(SEA) > 1e-8:
            eps0, kappa = N_target / SEA, 0.0
        else:
            eps0 = kappa = 0.0

        # ── Newton-Raphson Damped ───────────────────────────────────────────
        _DE = 1e-6   # Tolleranza di derivazione aumentata per stabilità numerica
        tol_N = max(abs(N_target) * 1e-5, 1.0)
        tol_M = max(abs(M_target) * 1e-5, 1e3)

        for it in range(_SLE_MAX_ITER):
            if progress_cb: progress_cb(int(10 + 75 * it / _SLE_MAX_ITER))

            sigs = [fibre[i].legame_sle.sigma(eps0 + kappa * ds[i]) for i in range(n_fib)]
            
            R_N = sum(sigs[i] * areas[i] for i in range(n_fib)) - N_target
            R_M = sum(sigs[i] * areas[i] * ds[i] for i in range(n_fib)) - M_target

            if abs(R_N) < tol_N and abs(R_M) < tol_M:
                break

            Et = []
            for i in range(n_fib):
                sp = fibre[i].legame_sle.sigma((eps0 + kappa * ds[i]) + _DE)
                et_i = (sp - sigs[i]) / _DE
                # Pseudo-rigidezza per evitare Jacobiani nulli quando tutto fessura
                Et.append(max(et_i, 1e-4))

            J11 = sum(Et[i] * areas[i] for i in range(n_fib))
            J12 = sum(Et[i] * areas[i] * ds[i] for i in range(n_fib))
            J22 = sum(Et[i] * areas[i] * ds[i] * ds[i] for i in range(n_fib))

            det = J11 * J22 - J12 * J12
            if abs(det) < 1e-30:
                eps0 += 1e-6
                continue

            d_eps0  = ( J22 * R_N - J12 * R_M) / det
            d_kappa = (-J12 * R_N + J11 * R_M) / det

            # --- DAMPING / STEP LIMITER ---
            # Previene salti che fanno oscillare il solutore sul punto di fessurazione
            max_d_eps = 0.0005
            scale = 1.0
            
            worst_d_eps = max(abs(d_eps0 + d_kappa * d_min), abs(d_eps0 + d_kappa * d_max))
            if worst_d_eps > max_d_eps:
                scale = max_d_eps / worst_d_eps
                
            # Damping progressivo per forzare l'atterraggio nei casi più instabili
            if it > 20: scale *= 0.5
            if it > 40: scale *= 0.25

            eps0  -= d_eps0 * scale
            kappa -= d_kappa * scale

        if progress_cb: progress_cb(88)

        # ── Post-Processing e Analisi dei Limiti ────────────────────────────
        def _is_concrete(f: Fibra) -> bool: return f.legame_slu.eps_max < 0.005

        fibre_res: list = []
        sig_c_max_compr = sig_s_traz_max = sig_s_comp_max = 0.0

        for i, f in enumerate(fibre):
            eps_i = eps0 + kappa * ds[i]
            sig_i = f.legame_sle.sigma(eps_i)
            fibre_res.append((f.x, f.y, f.area, eps_i, sig_i))

            if _is_concrete(f):
                sig_c_max_compr = max(sig_c_max_compr, abs(min(sig_i, 0.0)))
            else:
                sig_s_traz_max = max(sig_s_traz_max, max(sig_i, 0.0))
                sig_s_comp_max = max(sig_s_comp_max, abs(min(sig_i, 0.0)))

        d_na = -eps0 / kappa if abs(kappa) > 1e-15 else (d_min + d_max) / 2.0
        
        fck_approx = fyk_approx = sigma_c_limit = sigma_s_limit = None
        conc_legame = next((f.legame_sle for f in fibre if _is_concrete(f)),  None)
        acc_legame  = next((f.legame_sle for f in fibre if not _is_concrete(f)), None)

        if conc_legame is not None:
            try:
                fcd = abs(conc_legame.sigma(conc_legame.eps_min * 0.95))
                fck_approx = fcd * 1.5 / 0.85
                sigma_c_limit = 0.6 * fck_approx
            except Exception: pass

        if acc_legame is not None:
            try:
                fyd_approx = abs(acc_legame.sigma(acc_legame.eps_max * 0.40))
                fyk_approx = fyd_approx * 1.15
                sigma_s_limit = 0.8 * fyk_approx
            except Exception: pass

        verificata = True
        note: List[str] = []

        if sigma_c_limit and sig_c_max_compr > sigma_c_limit + 1e-3:
            verificata = False
            note.append(f"σ_c = {sig_c_max_compr:.1f} MPa  >  0.6·fck = {sigma_c_limit:.1f} MPa")
        if sigma_s_limit and sig_s_traz_max > sigma_s_limit + 1e-3:
            verificata = False
            note.append(f"σ_s(traz.) = {sig_s_traz_max:.1f} MPa  >  0.8·fyk = {sigma_s_limit:.1f} MPa")
        if sigma_s_limit and sig_s_comp_max > sigma_s_limit + 1e-3:
            verificata = False
            note.append(f"σ_s(comp.) = {sig_s_comp_max:.1f} MPa  >  0.8·fyk = {sigma_s_limit:.1f} MPa")

        if progress_cb: progress_cb(100)

        return {
            'tipo': 'SLE', 'verificata': verificata, 'note': note,
            'N_Ed': N_Ed_kN, 'M_Ed': M_Ed_kNm, 'theta_deg': theta_deg,
            'eps0': eps0, 'kappa': kappa, 'eps_top': eps0 + kappa * d_max, 'eps_bot': eps0 + kappa * d_min,
            'd_na': d_na, 'n_vect': tuple(nv.tolist()),
            'sigma_c_compr_max': sig_c_max_compr, 'sigma_s_traz_max': sig_s_traz_max, 'sigma_s_comp_max': sig_s_comp_max,
            'fck_approx': fck_approx, 'fyk_approx': fyk_approx,
            'sigma_c_limit': sigma_c_limit, 'sigma_s_limit': sigma_s_limit,
            'd_min': d_min, 'd_max': d_max, 'grid_step': self.sezione.grid_step, 'fibre': fibre_res,
        }