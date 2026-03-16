"""
calcolo.py  –  Motore FEM 3D per telai spaziali (beam elements, 6 DOF/nodo)
Usa scipy per la soluzione del sistema lineare sparso.
NON dipende da taichi: funziona su CPU pura, compatibile con qualsiasi ambiente.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class TelaioFEM:
    """
    Assembla e risolve un telaio spaziale 3D con elementi trave di Eulero-Bernoulli.
    Ogni nodo ha 6 gradi di libertà: [ux, uy, uz, rx, ry, rz].

    Flusso d'uso:
        1. aggiungi_nodo_principale()   → restituisce l'indice interno
        2. aggiungi_asta()              → collega due nodi con proprietà sez./mat.
        3. aggiungi_vincolo()           → vincola i gdl di un nodo
        4. aggiungi_carico()            → forze/momenti nodali
        5. risolvi()                    → restituisce (u_vec, sforzi)
    """

    def __init__(self):
        self.num_nodi = 0
        self.num_aste = 0

        # Lista dei nodi "principali" inseriti dall'utente (indici interni)
        self.nodi_principali: list[int] = []
        # Ogni elemento = {'n1': int, 'n2': int, 'idx_inizio': int, 'idx_fine': int}
        self.aste_principali: list[dict] = []

        # Array numpy interni (pre-allocati a dimensione massima)
        self._MAX = 4000
        self._nodi_np    = np.zeros((self._MAX, 3))
        self._aste_np    = np.zeros((self._MAX, 2), dtype=int)
        self._props_np   = np.zeros((self._MAX, 6))   # E, G, A, Iy, Iz, J
        self._vincoli_np = np.zeros((self._MAX, 6), dtype=int)
        self._forze_np   = np.zeros(self._MAX * 6)

        self._u_vec: np.ndarray | None = None
        self._sforzi: np.ndarray | None = None
        self.max_disp: float = 0.0

    # ------------------------------------------------------------------
    # Aggiunta entità
    # ------------------------------------------------------------------

    def aggiungi_nodo_principale(self, x: float, y: float, z: float) -> int:
        idx = self.num_nodi
        self._nodi_np[idx] = [x, y, z]
        self.nodi_principali.append(idx)
        self.num_nodi += 1
        return idx

    def _aggiungi_nodo_interno(self, x: float, y: float, z: float) -> int:
        idx = self.num_nodi
        self._nodi_np[idx] = [x, y, z]
        self.num_nodi += 1
        return idx

    def aggiungi_carico(self, nodo_idx: int, carichi: list[float]):
        """Aggiunge forze/momenti [Fx,Fy,Fz,Mx,My,Mz] al nodo (si sommano)."""
        for i in range(6):
            self._forze_np[nodo_idx * 6 + i] += carichi[i]

    def aggiungi_asta(
        self,
        n1: int, n2: int,
        E: float, G: float, A: float,
        Iy: float, Iz: float, J: float,
        num_suddivisioni: int = 1,
        q_distribuito: list[float] | None = None
    ):
        """
        Aggiunge un'asta tra i nodi n1 e n2.
        Se num_suddivisioni > 1, divide l'asta in sotto-elementi e
        distribuisce il carico distribuito come forze nodali equivalenti
        (metodo degli elementi finiti / approccio delle forze tributo).
        """
        if q_distribuito is None:
            q_distribuito = [0.0, 0.0, 0.0]

        idx_inizio = self.num_aste
        self.aste_principali.append({
            'n1': n1, 'n2': n2,
            'idx_inizio': idx_inizio,
            'idx_fine': idx_inizio + num_suddivisioni - 1
        })

        coord1 = self._nodi_np[n1].copy()
        coord2 = self._nodi_np[n2].copy()
        L_tot  = np.linalg.norm(coord2 - coord1)
        q_vec  = np.array(q_distribuito, dtype=float)

        if num_suddivisioni <= 1:
            self._crea_elemento(n1, n2, E, G, A, Iy, Iz, J)
            if np.any(q_vec):
                self.aggiungi_carico(n1, list(q_vec * L_tot / 2) + [0, 0, 0])
                self.aggiungi_carico(n2, list(q_vec * L_tot / 2) + [0, 0, 0])
        else:
            delta      = (coord2 - coord1) / num_suddivisioni
            trib_ends  = L_tot / (2 * num_suddivisioni)
            trib_mid   = L_tot / num_suddivisioni

            if np.any(q_vec):
                self.aggiungi_carico(n1, list(q_vec * trib_ends) + [0, 0, 0])

            nodo_corrente = n1
            for i in range(1, num_suddivisioni):
                nuova_coord = coord1 + delta * i
                nuovo_nodo  = self._aggiungi_nodo_interno(*nuova_coord)
                if np.any(q_vec):
                    self.aggiungi_carico(nuovo_nodo, list(q_vec * trib_mid) + [0, 0, 0])
                self._crea_elemento(nodo_corrente, nuovo_nodo, E, G, A, Iy, Iz, J)
                nodo_corrente = nuovo_nodo

            if np.any(q_vec):
                self.aggiungi_carico(n2, list(q_vec * trib_ends) + [0, 0, 0])
            self._crea_elemento(nodo_corrente, n2, E, G, A, Iy, Iz, J)

    def _crea_elemento(self, n1: int, n2: int,
                       E: float, G: float, A: float,
                       Iy: float, Iz: float, J: float):
        idx = self.num_aste
        self._aste_np[idx]  = [n1, n2]
        self._props_np[idx] = [E, G, A, Iy, Iz, J]
        self.num_aste += 1

    def aggiungi_vincolo(self, nodo_idx: int, gradi_bloccati: list[int]):
        """gradi_bloccati: lista di 6 interi 0/1 per [Tx,Ty,Tz,Rx,Ry,Rz]."""
        self._vincoli_np[nodo_idx] = gradi_bloccati

    # ------------------------------------------------------------------
    # Soluzione
    # ------------------------------------------------------------------

    def risolvi(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Assembla la matrice di rigidezza globale e risolve il sistema.
        Restituisce (u_vec, sforzi_aste).

        u_vec    : array 1D di lunghezza num_nodi*6  [m, rad]
        sforzi   : array 2D (num_aste, 12)  [N, Nm]
                   Convenzione colonne: [N1, Vy1, Vz1, Mt1, My1, Mz1,
                                         N2, Vy2, Vz2, Mt2, My2, Mz2]
        """
        tot_dof = self.num_nodi * 6
        K = lil_matrix((tot_dof, tot_dof))

        for i in range(self.num_aste):
            n1, n2 = self._aste_np[i]
            E, G, A, Iy, Iz, J = self._props_np[i]
            L, T = self._matrice_trasformazione(self._nodi_np[n1], self._nodi_np[n2])

            K_loc = self._matrice_rigidezza_locale(L, E, G, A, Iy, Iz, J)
            K_g   = T.T @ K_loc @ T

            dof_map = [n1*6+j for j in range(6)] + [n2*6+j for j in range(6)]
            for r in range(12):
                for c in range(12):
                    K[dof_map[r], dof_map[c]] += K_g[r, c]

        # Metodo della penalità per i vincoli
        PENALTY = 1e15
        for i in range(self.num_nodi):
            for j in range(6):
                if self._vincoli_np[i, j] == 1:
                    K[i*6+j, i*6+j] += PENALTY

        K_csr = K.tocsr()
        f_vec = self._forze_np[:tot_dof].copy()

        self._u_vec = spsolve(K_csr, f_vec)
        self._sforzi = self._calcola_sforzi()

        disp_matrix = self._u_vec.reshape(-1, 6)
        self.max_disp = float(np.max(np.linalg.norm(disp_matrix[:, :3], axis=1)))

        return self._u_vec, self._sforzi

    def _calcola_sforzi(self) -> np.ndarray:
        sforzi = np.zeros((self.num_aste, 12), dtype=float)
        for i in range(self.num_aste):
            n1, n2 = self._aste_np[i]
            E, G, A, Iy, Iz, J = self._props_np[i]
            L, T = self._matrice_trasformazione(self._nodi_np[n1], self._nodi_np[n2])
            K_loc = self._matrice_rigidezza_locale(L, E, G, A, Iy, Iz, J)

            u_g = np.concatenate([
                self._u_vec[n1*6: n1*6+6],
                self._u_vec[n2*6: n2*6+6]
            ])
            sforzi[i] = K_loc @ (T @ u_g)
        return sforzi

    # ------------------------------------------------------------------
    # Matrici elemento
    # ------------------------------------------------------------------

    @staticmethod
    def _matrice_rigidezza_locale(L: float, E: float, G: float,
                                   A: float, Iy: float, Iz: float, J: float) -> np.ndarray:
        K = np.zeros((12, 12))
        # Trazione/compressione assiale
        K[0, 0] = K[6, 6] =  E*A/L
        K[0, 6] = K[6, 0] = -E*A/L
        # Torsione
        K[3, 3] = K[9, 9] =  G*J/L
        K[3, 9] = K[9, 3] = -G*J/L
        # Flessione nel piano xz (Iy → Vz, My)
        K[2, 2]  = K[8, 8]  =  12*E*Iy/L**3
        K[2, 8]  = K[8, 2]  = -12*E*Iy/L**3
        K[4, 4]  = K[10,10] =   4*E*Iy/L
        K[4, 10] = K[10, 4] =   2*E*Iy/L
        K[2, 4]  = K[4, 2]  = K[2, 10] = K[10, 2] = -6*E*Iy/L**2
        K[8, 4]  = K[4, 8]  = K[8, 10] = K[10, 8] =  6*E*Iy/L**2
        # Flessione nel piano xy (Iz → Vy, Mz)
        K[1, 1]  = K[7, 7]  =  12*E*Iz/L**3
        K[1, 7]  = K[7, 1]  = -12*E*Iz/L**3
        K[5, 5]  = K[11,11] =   4*E*Iz/L
        K[5, 11] = K[11, 5] =   2*E*Iz/L
        K[1, 5]  = K[5, 1]  = K[1, 11] = K[11, 1] =  6*E*Iz/L**2
        K[7, 5]  = K[5, 7]  = K[7, 11] = K[11, 7] = -6*E*Iz/L**2
        return K

    @staticmethod
    def _matrice_trasformazione(p1: np.ndarray, p2: np.ndarray) -> tuple[float, np.ndarray]:
        delta = p2 - p1
        L = float(np.linalg.norm(delta))
        if L < 1e-12:
            raise ValueError(f"Lunghezza asta nulla tra i nodi {p1} e {p2}.")
        Cx, Cy, Cz = delta / L
        D = float(np.sqrt(Cx**2 + Cy**2))

        if D < 1e-10:
            # Asta verticale (parallela a Z)
            sign = 1.0 if Cz > 0 else -1.0
            R = np.array([
                [ 0,    0,   sign],
                [ 0,    1,   0   ],
                [-sign, 0,   0   ]
            ], dtype=float)
        else:
            # Direzione assiale
            e_x = np.array([Cx, Cy, Cz])
            # Vettore di riferimento (Z globale se non parallela, altrimenti Y)
            ref = np.array([0, 0, 1]) if D > 0.1 else np.array([0, 1, 0])
            e_z = np.cross(e_x, ref)
            e_z /= np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
            e_y /= np.linalg.norm(e_y)
            R = np.array([e_x, e_y, e_z])

        T = np.zeros((12, 12))
        T[0:3,  0:3]  = R
        T[3:6,  3:6]  = R
        T[6:9,  6:9]  = R
        T[9:12, 9:12] = R
        return L, T
