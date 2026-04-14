"""
generatore_mesh.py – Generatore di mesh esaedriche (C3D8I) per elementi strutturali.

Genera mesh strutturate per:
  - Parallelepipedi: griglia regolare
  - Cilindri: griglia radiale-assiale
  - Sfere: cube-sphere mapping

Genera mesh 1D per:
  - Barre: elementi T3D2 (truss 2-nodi)
  - Staffe: elementi T3D2 (truss 2-nodi, loop chiuso)

Gestisce:
  - TIE constraints tra facce a contatto di oggetti diversi
  - TIE constraints tra armature (T3D2) e carpenteria (C3D8) – embedded rebar
  - Applicazione carichi (nodi nel volume -> forze distribuite)
  - Applicazione vincoli (nodi nel volume -> boundary conditions + cedimenti)
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional

try:
    from scipy.spatial import cKDTree as _KDTree  # type: ignore
    _HAS_KDTREE = True
except Exception:  # pragma: no cover - fallback se scipy non e' installato
    _KDTree = None  # type: ignore
    _HAS_KDTREE = False

from elementi.modello_3d import (
    Oggetto3D, TIPO_STRUTTURALE, TIPO_ARMATURA,
    trasforma_punto, ruota_punto,
)
from elementi.modello_carichi_vincoli import CaricoVincolo


# ---------------------------------------------------------------------------
# Risultato della mesh
# ---------------------------------------------------------------------------

class RisultatoMesh:
    """Container dei dati della mesh generata."""

    def __init__(self):
        # Nodi globali: id -> [x, y, z]
        self.nodi: dict[int, list[float]] = {}
        # Elementi hex C3D8I: id -> [n1..n8]
        self.elementi_hex: dict[int, list[int]] = {}
        # Elementi truss T3D2: id -> [n1, n2]
        self.elementi_beam: dict[int, list[int]] = {}
        # Diametro per oggetto armatura: obj_id -> diametro (m)
        self.diametro_oggetto: dict[int, float] = {}
        # Set di nodi per oggetto: obj_id -> set(node_ids)
        self.nodi_per_oggetto: dict[int, set[int]] = {}
        # Set di elementi per oggetto: obj_id -> set(elem_ids)
        self.elementi_per_oggetto: dict[int, set[int]] = {}
        # Tipo oggetto: obj_id -> "carpenteria"|"barra"|"staffa"
        self.tipo_oggetto: dict[int, str] = {}
        # Materiale per oggetto: obj_id -> nome_materiale
        self.materiale_oggetto: dict[int, str] = {}
        # TIE constraints tra carpenteria: list of dict
        # Ogni dict: {master_obj, slave_obj, pairs: [(master_nid, slave_nid), ...]}
        self.tie_constraints: list[dict] = []
        # TIE constraints armatura-carpenteria (embedded rebar)
        self.tie_armatura: list[dict] = []
        # Nodi vincolati: node_id -> {"sx": val, "sy": val, "sz": val}
        self.nodi_vincolati: dict[int, dict] = {}
        # Nodi caricati: node_id -> {"fx": val, "fy": val, "fz": val}
        self.nodi_caricati: dict[int, dict] = {}
        # Facce esterne per ogni oggetto (per visualizzazione e TIE)
        # obj_id -> list of [n1, n2, n3, n4] (quad faces)
        self.facce_esterne: dict[int, list[list[int]]] = {}
        # Identificativo CalculiX della faccia esterna su ogni elemento hex:
        # obj_id -> list of (elem_id, "S1".."S6"), parallelo a facce_esterne[obj_id].
        # Permette di scrivere superfici element-based corrette per *TIE/*CONTACT.
        self.facce_elem: dict[int, list[tuple[int, str]]] = {}

    @property
    def n_nodi(self) -> int:
        return len(self.nodi)

    @property
    def n_elementi(self) -> int:
        return len(self.elementi_hex) + len(self.elementi_beam)


# ---------------------------------------------------------------------------
# Generatore
# ---------------------------------------------------------------------------

class GeneratoreMesh:
    """
    Generatore di mesh per un singolo elemento strutturale.

    Parametri
    ---------
    densita : int
        Numero di suddivisioni lungo la dimensione piu' corta.
        Valori tipici: 3-10 per mesh grossolane, 15-30 per mesh fini.
    """

    def __init__(self, densita: int = 5):
        self._densita = max(2, densita)
        self._node_counter = 0
        self._elem_counter = 0
        self._risultato = RisultatoMesh()
        self._dim_elem_rif: float = 0.0  # dimensione elemento di riferimento (m)

    def _next_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def _next_elem_id(self) -> int:
        self._elem_counter += 1
        return self._elem_counter

    # ==================================================================
    # API PUBBLICA
    # ==================================================================

    def genera(self, dati_fem: dict, progress_cb=None) -> RisultatoMesh:
        """
        Genera la mesh completa per l'elemento.

        dati_fem: dizionario da RaccoltaDatiElementoFEM.dati_per_fem()
        progress_cb: callback(int) con percentuale 0-100
        """
        self._risultato = RisultatoMesh()
        self._node_counter = 0
        self._elem_counter = 0

        carpenteria = dati_fem.get("carpenteria", [])
        barre = dati_fem.get("barre", [])
        staffe = dati_fem.get("staffe", [])
        carichi = dati_fem.get("carichi", [])
        vincoli = dati_fem.get("vincoli", [])

        totale = len(carpenteria) + len(barre) + len(staffe) + 2  # +2 per CV e TIE
        fatto = 0

        def _progress():
            nonlocal fatto
            fatto += 1
            if progress_cb:
                progress_cb(min(int(fatto / totale * 100), 99))

        # 1. Mesh carpenteria (esaedri C3D8I)
        for obj in carpenteria:
            self._mesh_carpenteria(obj)
            _progress()

        # Calcola dimensione elemento di riferimento dalla carpenteria
        self._calcola_dim_elem_rif(carpenteria)

        # Calcola facce esterne element-based (per *SURFACE TYPE=ELEMENT
        # corretti). Questo ricostruisce facce_esterne in modo coerente con
        # gli identificatori S1..S6 di CalculiX, pronti per *TIE / *CONTACT.
        self._calcola_facce_elem(carpenteria)

        # 2. Mesh barre (beam B31)
        for obj in barre:
            self._mesh_barra(obj)
            _progress()

        # 3. Mesh staffe (beam B31 loop)
        for obj in staffe:
            self._mesh_staffa(obj)
            _progress()

        # 4. TIE constraints tra facce a contatto (carpenteria-carpenteria)
        self._genera_tie_constraints(carpenteria)
        # 4b. TIE constraints armatura-carpenteria (embedded rebar)
        self._genera_tie_armatura(barre + staffe, carpenteria)
        _progress()

        # 5. Applica carichi e vincoli
        self._applica_carichi_vincoli(carichi, vincoli)
        _progress()

        if progress_cb:
            progress_cb(100)

        return self._risultato

    def _calcola_dim_elem_rif(self, carpenteria: list):
        """Calcola la dimensione tipica di un elemento hex dalla carpenteria.
        Usata per dare alle armature la stessa discretizzazione."""
        if not self._risultato.elementi_hex:
            self._dim_elem_rif = 0.0
            return
        # Campiona le lunghezze degli spigoli di alcuni elementi hex
        lunghezze = []
        count = 0
        for eid, hn in self._risultato.elementi_hex.items():
            for a, b in ((0, 1), (1, 2), (0, 3), (0, 4)):
                pa = np.array(self._risultato.nodi[hn[a]])
                pb = np.array(self._risultato.nodi[hn[b]])
                lunghezze.append(np.linalg.norm(pb - pa))
            count += 1
            if count >= 20:
                break
        self._dim_elem_rif = float(np.median(lunghezze)) if lunghezze else 0.0

    # ------------------------------------------------------------------
    # FACCE ESTERNE element-based (S1..S6 di CalculiX)
    # ------------------------------------------------------------------
    # Mappa "indice locale faccia" -> tupla di 4 nodi locali del C3D8.
    # E' la convenzione standard CalculiX/Abaqus per l'elemento C3D8:
    #   S1 = 1-2-3-4   (k = 0)
    #   S2 = 5-6-7-8   (k = +)
    #   S3 = 1-2-6-5   (j = 0)
    #   S4 = 2-3-7-6   (i = +)
    #   S5 = 3-4-8-7   (j = +)
    #   S6 = 4-1-5-8   (i = 0)
    # Indici 0-based sul vettore [n0..n7] del nostro storage interno.
    _S_FACES = {
        "S1": (0, 1, 2, 3),
        "S2": (4, 5, 6, 7),
        "S3": (0, 1, 5, 4),
        "S4": (1, 2, 6, 5),
        "S5": (2, 3, 7, 6),
        "S6": (3, 0, 4, 7),
    }

    def _calcola_facce_elem(self, carpenteria: list):
        """
        Per ciascun oggetto di carpenteria, ricostruisce l'elenco delle facce
        esterne con il loro identificativo (elem_id, "Sn") usando la sola
        topologia degli elementi hex generati.

        Una faccia condivisa da due elementi e' interna; una faccia che
        compare una sola volta in tutto il corpo e' esterna. Il confronto
        usa la tupla ordinata dei 4 id-nodo (gestisce anche elementi
        degeneri/wedge dei cilindri).
        """
        for obj in carpenteria:
            elem_set = self._risultato.elementi_per_oggetto.get(obj.id, set())
            if not elem_set:
                continue

            face_count: dict[tuple, list] = {}
            for eid in elem_set:
                hn = self._risultato.elementi_hex.get(eid)
                if hn is None:
                    continue
                for sname, idxs in self._S_FACES.items():
                    nodi = (hn[idxs[0]], hn[idxs[1]],
                            hn[idxs[2]], hn[idxs[3]])
                    key = tuple(sorted(nodi))
                    face_count.setdefault(key, []).append((eid, sname, nodi))

            facce_elem: list[tuple[int, str]] = []
            facce_quad: list[list[int]] = []
            for lst in face_count.values():
                if len(lst) != 1:  # interna (compare 2 volte) -> scarta
                    continue
                eid, sname, nodi = lst[0]
                # Filtra facce degeneri (wedge): meno di 3 nodi distinti
                if len(set(nodi)) < 3:
                    continue
                facce_elem.append((eid, sname))
                facce_quad.append(list(nodi))

            self._risultato.facce_elem[obj.id] = facce_elem
            # Riallinea facce_esterne (usato per visualizzazione/TIE)
            self._risultato.facce_esterne[obj.id] = facce_quad

    # ==================================================================
    # MESH CARPENTERIA (C3D8I)
    # ==================================================================

    def _mesh_carpenteria(self, obj: Oggetto3D):
        """Genera mesh esaedrica per un oggetto strutturale."""
        if obj.tipo == "parallelepipedo":
            self._mesh_parallelepipedo(obj)
        elif obj.tipo == "cilindro":
            self._mesh_cilindro(obj)
        elif obj.tipo == "sfera":
            self._mesh_sfera(obj)

    # ── Suddivisione adattiva ──────────────────────────────────

    def _suddivisioni_adattive(self, lx: float, ly: float, lz: float) -> tuple[int, int, int]:
        """
        Calcola nx, ny, nz con aspect ratio ~1.

        Strategia: la densità utente d determina direttamente la dimensione
        target degli elementi. Il lato medio viene diviso in d parti, e tutte
        le direzioni usano la stessa dimensione target. Nessun cap rigido:
        più alto è d, più fitta è la mesh.
        """
        d = self._densita
        dims = sorted([lx, ly, lz])
        l_min = max(dims[0], 1e-8)
        l_mid = max(dims[1], 1e-8)

        # Dimensione target: il lato medio viene diviso in d parti.
        target = l_mid / max(1, d)

        # Clamp inferiore: non più grossolana del lato corto (almeno 1 elem).
        target = min(target, l_min)

        nx = max(1, int(round(lx / target)))
        ny = max(1, int(round(ly / target)))
        nz = max(1, int(round(lz / target)))

        # Garantisci almeno 2 suddivisioni per direzioni non trascurabili
        if lx > target * 0.5:
            nx = max(2, nx)
        if ly > target * 0.5:
            ny = max(2, ny)
        if lz > target * 0.5:
            nz = max(2, nz)

        return nx, ny, nz

    # ── Parallelepipedo ──────────────────────────────────────────

    def _mesh_parallelepipedo(self, obj: Oggetto3D):
        """Mesh strutturata per parallelepipedo."""
        verts_local = obj.get_vertices_local()
        if len(verts_local) < 8:
            return

        v = [np.array(p, dtype=float) for p in verts_local]
        # v[0..7]: 0=origin, 1=+X, 2=+XY, 3=+Y, 4=+Z, 5=+XZ, 6=+XYZ, 7=+YZ

        d = self._densita
        lx = float(np.linalg.norm(v[1] - v[0]))
        ly = float(np.linalg.norm(v[3] - v[0]))
        lz = float(np.linalg.norm(v[4] - v[0]))

        dim_user = obj.geometria.get("dim_mesh", None)
        if dim_user and float(dim_user) > 1e-8:
            target = float(dim_user)
            nx = max(1, int(round(lx / target)))
            ny = max(1, int(round(ly / target)))
            nz = max(1, int(round(lz / target)))
        else:
            nx, ny, nz = self._suddivisioni_adattive(lx, ly, lz)

        # Genera nodi con interpolazione trilineare
        node_grid = {}  # (i, j, k) -> node_id
        nodi_obj = set()
        elems_obj = set()

        for k in range(nz + 1):
            w = k / nz
            for j in range(ny + 1):
                v_val = j / ny
                for i in range(nx + 1):
                    u = i / nx
                    # Interpolazione trilineare tra gli 8 vertici
                    pt = (v[0] * (1-u)*(1-v_val)*(1-w) +
                          v[1] * u*(1-v_val)*(1-w) +
                          v[2] * u*v_val*(1-w) +
                          v[3] * (1-u)*v_val*(1-w) +
                          v[4] * (1-u)*(1-v_val)*w +
                          v[5] * u*(1-v_val)*w +
                          v[6] * u*v_val*w +
                          v[7] * (1-u)*v_val*w)
                    # Trasforma in coordinate world
                    pt_world = trasforma_punto(pt.tolist(), obj.posizione, obj.rotazione)
                    nid = self._next_node_id()
                    self._risultato.nodi[nid] = pt_world
                    node_grid[(i, j, k)] = nid
                    nodi_obj.add(nid)

        # Genera elementi esaedrici
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n = [
                        node_grid[(i,   j,   k)],
                        node_grid[(i+1, j,   k)],
                        node_grid[(i+1, j+1, k)],
                        node_grid[(i,   j+1, k)],
                        node_grid[(i,   j,   k+1)],
                        node_grid[(i+1, j,   k+1)],
                        node_grid[(i+1, j+1, k+1)],
                        node_grid[(i,   j+1, k+1)],
                    ]
                    eid = self._next_elem_id()
                    self._risultato.elementi_hex[eid] = n
                    elems_obj.add(eid)

        # Facce esterne (6 facce del parallelepipedo)
        facce = []
        # Faccia -Z (k=0)
        for j in range(ny):
            for i in range(nx):
                facce.append([node_grid[(i,j,0)], node_grid[(i+1,j,0)],
                              node_grid[(i+1,j+1,0)], node_grid[(i,j+1,0)]])
        # Faccia +Z (k=nz)
        for j in range(ny):
            for i in range(nx):
                facce.append([node_grid[(i,j,nz)], node_grid[(i+1,j,nz)],
                              node_grid[(i+1,j+1,nz)], node_grid[(i,j+1,nz)]])
        # Faccia -Y (j=0)
        for k in range(nz):
            for i in range(nx):
                facce.append([node_grid[(i,0,k)], node_grid[(i+1,0,k)],
                              node_grid[(i+1,0,k+1)], node_grid[(i,0,k+1)]])
        # Faccia +Y (j=ny)
        for k in range(nz):
            for i in range(nx):
                facce.append([node_grid[(i,ny,k)], node_grid[(i+1,ny,k)],
                              node_grid[(i+1,ny,k+1)], node_grid[(i,ny,k+1)]])
        # Faccia -X (i=0)
        for k in range(nz):
            for j in range(ny):
                facce.append([node_grid[(0,j,k)], node_grid[(0,j+1,k)],
                              node_grid[(0,j+1,k+1)], node_grid[(0,j,k+1)]])
        # Faccia +X (i=nx)
        for k in range(nz):
            for j in range(ny):
                facce.append([node_grid[(nx,j,k)], node_grid[(nx,j+1,k)],
                              node_grid[(nx,j+1,k+1)], node_grid[(nx,j,k+1)]])

        self._risultato.nodi_per_oggetto[obj.id] = nodi_obj
        self._risultato.elementi_per_oggetto[obj.id] = elems_obj
        self._risultato.tipo_oggetto[obj.id] = "carpenteria"
        self._risultato.materiale_oggetto[obj.id] = obj.materiale
        self._risultato.facce_esterne[obj.id] = facce

    # ── Cilindro ─────────────────────────────────────────────────

    def _mesh_cilindro(self, obj: Oggetto3D):
        """Mesh strutturata per cilindro (griglia radiale + assiale)."""
        A = float(obj.geometria.get("altezza", 3.0))
        R = float(obj.geometria.get("raggio", 0.2))

        d = self._densita
        # Dimensione target: basata sulla media tra raggio e altezza,
        # controllata dalla densità utente.
        l_mid = (R + A) / 2.0 if A > R else R
        target = max(min(R, A), 1e-8) / max(d, 2)
        # Non più fine di R/d, non più grossolana di R stesso
        target = min(target, R)

        # Suddivisioni radiali
        n_r = max(1, int(round(R / target)))
        # Suddivisioni angolari: arco ~target, arrotondato a multiplo di 4
        n_theta = max(8, int(round(2 * math.pi * R / max(target, 1e-8))))
        n_theta = max(8, ((n_theta + 3) // 4) * 4)
        # Suddivisioni assiali
        n_z = max(1, int(round(A / max(target, 1e-8))))

        node_grid = {}  # (ir, itheta, iz) -> node_id
        nodi_obj = set()
        elems_obj = set()

        # Genera nodi: per ogni livello z, per ogni anello radiale, per ogni angolo
        for iz in range(n_z + 1):
            z = A * iz / n_z
            for ir in range(n_r + 1):
                r = R * ir / n_r
                if ir == 0:
                    # Centro: un solo nodo per livello
                    pt_local = [0.0, 0.0, z]
                    pt_world = trasforma_punto(pt_local, obj.posizione, obj.rotazione)
                    nid = self._next_node_id()
                    self._risultato.nodi[nid] = pt_world
                    nodi_obj.add(nid)
                    for itheta in range(n_theta):
                        node_grid[(ir, itheta, iz)] = nid
                else:
                    for itheta in range(n_theta):
                        theta = 2 * math.pi * itheta / n_theta
                        x = r * math.cos(theta)
                        y = r * math.sin(theta)
                        pt_local = [x, y, z]
                        pt_world = trasforma_punto(pt_local, obj.posizione, obj.rotazione)
                        nid = self._next_node_id()
                        self._risultato.nodi[nid] = pt_world
                        node_grid[(ir, itheta, iz)] = nid
                        nodi_obj.add(nid)

        # Genera elementi hex
        facce = []
        for iz in range(n_z):
            for ir in range(n_r):
                for itheta in range(n_theta):
                    itheta_next = (itheta + 1) % n_theta
                    if ir == 0:
                        # Wedge degenerato come hex (nodi 0-3 collassati)
                        n0 = node_grid[(ir,   itheta,      iz)]
                        n1 = node_grid[(ir+1, itheta,      iz)]
                        n2 = node_grid[(ir+1, itheta_next, iz)]
                        n3 = node_grid[(ir,   itheta_next, iz)]
                        n4 = node_grid[(ir,   itheta,      iz+1)]
                        n5 = node_grid[(ir+1, itheta,      iz+1)]
                        n6 = node_grid[(ir+1, itheta_next, iz+1)]
                        n7 = node_grid[(ir,   itheta_next, iz+1)]
                    else:
                        n0 = node_grid[(ir,   itheta,      iz)]
                        n1 = node_grid[(ir+1, itheta,      iz)]
                        n2 = node_grid[(ir+1, itheta_next, iz)]
                        n3 = node_grid[(ir,   itheta_next, iz)]
                        n4 = node_grid[(ir,   itheta,      iz+1)]
                        n5 = node_grid[(ir+1, itheta,      iz+1)]
                        n6 = node_grid[(ir+1, itheta_next, iz+1)]
                        n7 = node_grid[(ir,   itheta_next, iz+1)]

                    eid = self._next_elem_id()
                    self._risultato.elementi_hex[eid] = [n0, n1, n2, n3, n4, n5, n6, n7]
                    elems_obj.add(eid)

                    # Facce esterne (solo bordo esterno e basi)
                    if ir == n_r - 1:
                        facce.append([n1, n2, n6, n5])
                    if iz == 0:
                        facce.append([n0, n1, n2, n3])
                    if iz == n_z - 1:
                        facce.append([n4, n5, n6, n7])

        self._risultato.nodi_per_oggetto[obj.id] = nodi_obj
        self._risultato.elementi_per_oggetto[obj.id] = elems_obj
        self._risultato.tipo_oggetto[obj.id] = "carpenteria"
        self._risultato.materiale_oggetto[obj.id] = obj.materiale
        self._risultato.facce_esterne[obj.id] = facce

    # ── Sfera ────────────────────────────────────────────────────

    def _mesh_sfera(self, obj: Oggetto3D):
        """Mesh strutturata per sfera con cube-sphere mapping."""
        R = float(obj.geometria.get("raggio", 0.2))
        # Densita = numero di suddivisioni per asse del cubo proiettato.
        # Mantiene gli elementi superficiali ~target = (2R)/densita.
        d = max(2, self._densita)

        nodi_obj = set()
        elems_obj = set()
        facce = []
        node_map = {}  # (face_idx, i, j, k) -> node_id

        for k in range(d + 1):
            for j in range(d + 1):
                for i in range(d + 1):
                    # Coordinate nel cubo [-1, 1]
                    u = -1 + 2 * i / d
                    v = -1 + 2 * j / d
                    w = -1 + 2 * k / d

                    # Cube-sphere mapping
                    x2 = u * u
                    y2 = v * v
                    z2 = w * w
                    sx = u * math.sqrt(max(0, 1 - y2/2 - z2/2 + y2*z2/3))
                    sy = v * math.sqrt(max(0, 1 - z2/2 - x2/2 + z2*x2/3))
                    sz = w * math.sqrt(max(0, 1 - x2/2 - y2/2 + x2*y2/3))

                    pt_local = [sx * R, sy * R, sz * R]
                    pt_world = trasforma_punto(pt_local, obj.posizione, obj.rotazione)
                    nid = self._next_node_id()
                    self._risultato.nodi[nid] = pt_world
                    node_map[(i, j, k)] = nid
                    nodi_obj.add(nid)

        # Genera elementi hex
        for k in range(d):
            for j in range(d):
                for i in range(d):
                    n = [
                        node_map[(i,   j,   k)],
                        node_map[(i+1, j,   k)],
                        node_map[(i+1, j+1, k)],
                        node_map[(i,   j+1, k)],
                        node_map[(i,   j,   k+1)],
                        node_map[(i+1, j,   k+1)],
                        node_map[(i+1, j+1, k+1)],
                        node_map[(i,   j+1, k+1)],
                    ]
                    eid = self._next_elem_id()
                    self._risultato.elementi_hex[eid] = n
                    elems_obj.add(eid)

                    # Facce esterne (bordo del cubo -> bordo della sfera)
                    if i == 0:
                        facce.append([n[0], n[3], n[7], n[4]])
                    if i == d - 1:
                        facce.append([n[1], n[2], n[6], n[5]])
                    if j == 0:
                        facce.append([n[0], n[1], n[5], n[4]])
                    if j == d - 1:
                        facce.append([n[3], n[2], n[6], n[7]])
                    if k == 0:
                        facce.append([n[0], n[1], n[2], n[3]])
                    if k == d - 1:
                        facce.append([n[4], n[5], n[6], n[7]])

        self._risultato.nodi_per_oggetto[obj.id] = nodi_obj
        self._risultato.elementi_per_oggetto[obj.id] = elems_obj
        self._risultato.tipo_oggetto[obj.id] = "carpenteria"
        self._risultato.materiale_oggetto[obj.id] = obj.materiale
        self._risultato.facce_esterne[obj.id] = facce

    # ==================================================================
    # MESH BARRE E STAFFE (B31)
    # ==================================================================

    def _mesh_barra(self, obj: Oggetto3D):
        """Genera elementi truss T3D2 per una barra longitudinale."""
        punti = obj.geometria.get("punti", [])
        if len(punti) < 2:
            return
        diametro = float(obj.geometria.get("diametro", 0.016))
        self._risultato.diametro_oggetto[obj.id] = diametro
        self._mesh_lineare(obj, punti, chiuso=False, tipo="barra")

    def _mesh_staffa(self, obj: Oggetto3D):
        """Genera elementi truss T3D2 per una staffa (loop chiuso)."""
        punti = obj.geometria.get("punti", [])
        if len(punti) < 3:
            return
        diametro = float(obj.geometria.get("diametro", 0.008))
        self._risultato.diametro_oggetto[obj.id] = diametro
        self._mesh_lineare(obj, punti, chiuso=True, tipo="staffa")

    def _mesh_lineare(self, obj: Oggetto3D, punti: list, chiuso: bool, tipo: str):
        """Mesh 1D comune per barre e staffe.

        La discretizzazione cerca di "allinearsi" con la mesh hex circostante:
        la lunghezza target di ciascun elemento truss e' pari alla dimensione
        media degli spigoli degli esaedri (calcolata da _dim_elem_rif).
        Se non e' disponibile alcuna mesh hex, si usa un target derivato da
        'densita' applicato alla lunghezza totale del tracciato dell'armatura.
        """
        nodi_obj = set()
        elems_obj = set()

        # Per ogni segmento, suddividi
        n_punti = len(punti)
        segmenti = []
        for i in range(n_punti - 1):
            segmenti.append((punti[i], punti[i + 1]))
        if chiuso:
            segmenti.append((punti[-1], punti[0]))

        # Lunghezza totale del tracciato (in coord. locali dell'oggetto)
        lunghezze_seg = []
        for p0, p1 in segmenti:
            lunghezze_seg.append(float(np.linalg.norm(np.array(p1) - np.array(p0))))
        l_totale = sum(lunghezze_seg)

        # Dimensione target dell'elemento truss
        if self._dim_elem_rif > 1e-8:
            l_target = self._dim_elem_rif
        elif l_totale > 1e-8:
            # Fallback: distribuisci ~ (densita * 4) elementi sull'intero tracciato
            l_target = l_totale / max(1, self._densita * 4)
        else:
            l_target = 0.0

        # Limiti di sicurezza: l_target almeno pari al lato piu' corto del
        # segmento minimo / 1, e al massimo pari al segmento massimo
        if l_target > 0 and lunghezze_seg:
            l_min = min(l for l in lunghezze_seg if l > 1e-8)
            l_max = max(lunghezze_seg)
            l_target = min(max(l_target, l_min / 50.0), l_max)

        nodi_segmento = []  # lista di node_ids per tutti i segmenti

        for seg_idx, (p0, p1) in enumerate(segmenti):
            p0 = np.array(p0, dtype=float)
            p1 = np.array(p1, dtype=float)
            lunghezza = lunghezze_seg[seg_idx]

            # Salta segmenti di lunghezza zero
            if lunghezza < 1e-8:
                continue

            # Almeno una suddivisione per segmento. Per segmenti corti
            # (es. lati corti di una staffa) si garantisce almeno 1 elemento.
            if l_target > 1e-8:
                n_sub = max(1, int(round(lunghezza / l_target)))
            else:
                n_sub = max(1, self._densita)

            for i in range(n_sub + 1):
                if seg_idx > 0 and i == 0:
                    continue  # Riusa l'ultimo nodo del segmento precedente

                # Per l'ultimo segmento di un loop chiuso, l'ultimo punto
                # coincide col primo nodo -> non creare nodo duplicato
                if chiuso and seg_idx == len(segmenti) - 1 and i == n_sub:
                    continue

                t = i / n_sub
                pt_local = (p0 * (1 - t) + p1 * t).tolist()
                pt_world = trasforma_punto(pt_local, obj.posizione, obj.rotazione)
                nid = self._next_node_id()
                self._risultato.nodi[nid] = pt_world
                nodi_obj.add(nid)
                nodi_segmento.append(nid)

        # Genera elementi truss, filtrando quelli di lunghezza zero
        min_len = 1e-8
        for i in range(len(nodi_segmento) - 1):
            pa = np.array(self._risultato.nodi[nodi_segmento[i]])
            pb = np.array(self._risultato.nodi[nodi_segmento[i + 1]])
            if np.linalg.norm(pb - pa) < min_len:
                continue
            eid = self._next_elem_id()
            self._risultato.elementi_beam[eid] = [nodi_segmento[i], nodi_segmento[i + 1]]
            elems_obj.add(eid)

        # Chiudi il loop per staffe: collega ultimo nodo al primo
        if chiuso and len(nodi_segmento) >= 2:
            pa = np.array(self._risultato.nodi[nodi_segmento[-1]])
            pb = np.array(self._risultato.nodi[nodi_segmento[0]])
            if np.linalg.norm(pb - pa) >= min_len:
                eid = self._next_elem_id()
                self._risultato.elementi_beam[eid] = [nodi_segmento[-1], nodi_segmento[0]]
                elems_obj.add(eid)

        self._risultato.nodi_per_oggetto[obj.id] = nodi_obj
        self._risultato.elementi_per_oggetto[obj.id] = elems_obj
        self._risultato.tipo_oggetto[obj.id] = tipo
        self._risultato.materiale_oggetto[obj.id] = obj.materiale

    # ==================================================================
    # TIE CONSTRAINTS
    # ==================================================================

    def _genera_tie_constraints(self, carpenteria: list[Oggetto3D]):
        """
        Genera vincoli di tipo TIE (superficie-superficie) tra oggetti di carpenteria.
        Identifica le facce degli elementi (es. S1..S6) che si trovano nelle zone 
        di contatto geometrico tra due corpi, consentendo l'incollaggio di mesh 
        non conformi (es. densità diverse).
        """
        if len(carpenteria) < 2:
            return

        tol_contact = 1e-4  # Tolleranza per considerare due facce "a contatto" (m)
        nodi_glob = self._risultato.nodi

        # 1. Preparazione Dati: Calcola il Bounding Box per ogni oggetto 
        #    e per ogni sua faccia esterna.
        cache: dict[int, dict] = {}
        for obj in carpenteria:
            facce_quad = self._risultato.facce_esterne.get(obj.id, [])
            facce_elem = self._risultato.facce_elem.get(obj.id, [])
            if not facce_quad or not facce_elem:
                continue

            # Calcola BB dell'intero oggetto
            nodi_bordo = set()
            for fnodes in facce_quad:
                nodi_bordo.update(fnodes)
            
            pos_arr = np.array([nodi_glob[int(n)] for n in nodi_bordo], dtype=float)
            obj_bb_min = pos_arr.min(axis=0)
            obj_bb_max = pos_arr.max(axis=0)

            # Calcola BB per ogni singola faccia
            facce_dati = []
            for i, f_nodi in enumerate(facce_quad):
                f_pos = np.array([nodi_glob[int(n)] for n in f_nodi], dtype=float)
                f_bb_min = f_pos.min(axis=0)
                f_bb_max = f_pos.max(axis=0)
                facce_dati.append({
                    "elem_info": facce_elem[i],  # Tupla: (elem_id, "Sn")
                    "bb_min": f_bb_min,
                    "bb_max": f_bb_max
                })

            cache[obj.id] = {
                "bb_min": obj_bb_min,
                "bb_max": obj_bb_max,
                "facce": facce_dati
            }

        ids_carp = list(cache.keys())

        # 2. Controllo intersezioni a coppie
        for i in range(len(ids_carp)):
            for j in range(i + 1, len(ids_carp)):
                id_a = ids_carp[i]
                id_b = ids_carp[j]
                ca = cache[id_a]
                cb = cache[id_b]

                # Check veloce: i due oggetti si toccano globalmente?
                if not np.all((ca["bb_max"] + tol_contact >= cb["bb_min"]) & 
                              (ca["bb_min"] - tol_contact <= cb["bb_max"])):
                    continue

                # Raccogli le facce dell'oggetto A a contatto con l'oggetto B
                faces_a = []
                for f in ca["facce"]:
                    if np.all((f["bb_max"] + tol_contact >= cb["bb_min"]) & 
                              (f["bb_min"] - tol_contact <= cb["bb_max"])):
                        faces_a.append(f["elem_info"])

                # Raccogli le facce dell'oggetto B a contatto con l'oggetto A
                faces_b = []
                for f in cb["facce"]:
                    if np.all((f["bb_max"] + tol_contact >= ca["bb_min"]) & 
                              (f["bb_min"] - tol_contact <= ca["bb_max"])):
                        faces_b.append(f["elem_info"])

                # Se abbiamo trovato un'interfaccia di contatto
                if faces_a and faces_b:
                    # Regola FEM d'oro: la mesh più grossolana fa da Master,
                    # la mesh più fitta fa da Slave.
                    # Qui usiamo il numero di facce come stima della fittezza.
                    if len(faces_a) <= len(faces_b):
                        m_obj, s_obj = id_a, id_b
                        m_faces, s_faces = faces_a, faces_b
                    else:
                        m_obj, s_obj = id_b, id_a
                        m_faces, s_faces = faces_b, faces_a

                    self._risultato.tie_constraints.append({
                        "master_obj": m_obj,
                        "slave_obj": s_obj,
                        "master_faces": m_faces,
                        "slave_faces": s_faces,
                    })

    # ==================================================================
    # TIE ARMATURA-CARPENTERIA (EMBEDDED REBAR)
    # ==================================================================

    def _genera_tie_armatura(self, armature: list[Oggetto3D],
                            carpenteria: list[Oggetto3D]):
        if not armature or not carpenteria:
            return

        # Pre-calcola bounding box per ogni carpenteria (con margine)
        carp_bb = {}
        for obj in carpenteria:
            nodi_set = self._risultato.nodi_per_oggetto.get(obj.id, set())
            if not nodi_set:
                continue
            coords = np.array([self._risultato.nodi[nid] for nid in nodi_set])
            carp_bb[obj.id] = (coords.min(axis=0) - 0.05,
                            coords.max(axis=0) + 0.05)

        for arm_obj in armature:
            arm_nodi = self._risultato.nodi_per_oggetto.get(arm_obj.id, set())
            if not arm_nodi:
                continue

            # Dizionario per raggruppare i nodi della barra in base al solido che li ospita
            # { carp_id: set(node_ids) }
            nodi_per_host = {}

            for nid in arm_nodi:
                pos = np.array(self._risultato.nodi[nid])
                
                # Cerca in quale solido cade questo specifico nodo
                for carp_obj in carpenteria:
                    bb = carp_bb.get(carp_obj.id)
                    if bb is None:
                        continue
                    bb_min, bb_max = bb
                    
                    # Se il nodo è dentro il Bounding Box di questa carpenteria
                    if np.all(pos >= bb_min) and np.all(pos <= bb_max):
                        nodi_per_host.setdefault(carp_obj.id, set()).add(nid)
                        break # Passa al nodo successivo (assumiamo non ci siano sovrapposizioni esatte)

            # Ora crea un TIE separato per ogni tratto di barra contenuto nei vari solidi
            for carp_id, nodi_slave in nodi_per_host.items():
                master_nodi = self._risultato.nodi_per_oggetto.get(carp_id, set())
                
                self._risultato.tie_armatura.append({
                    "slave_obj": arm_obj.id,
                    "master_obj": carp_id,
                    "slave_nodes": nodi_slave, # Solo i nodi che cadono in QUESTO solido
                    "master_nodes": master_nodi.copy(),
                })

    # ==================================================================
    # CARICHI E VINCOLI
    # ==================================================================

    def _applica_carichi_vincoli(self, carichi: list[CaricoVincolo],
                                  vincoli: list[CaricoVincolo]):
        """
        Per ogni carico/vincolo, trova i nodi della mesh che cadono
        all'interno del suo volume e applica le condizioni.
        """
        for cv in vincoli:
            nodi_interni = self._trova_nodi_nel_volume(cv)
            for nid in nodi_interni:
                self._risultato.nodi_vincolati[nid] = {
                    "sx": cv.caratteristiche.get("sx", 0.0),
                    "sy": cv.caratteristiche.get("sy", 0.0),
                    "sz": cv.caratteristiche.get("sz", 0.0),
                }

        for cv in carichi:
            nodi_interni = self._trova_nodi_nel_volume(cv)
            if not nodi_interni:
                continue
            # Spalma la forza totale su tutti i nodi nel volume
            n_nodi = len(nodi_interni)
            fx = cv.caratteristiche.get("fx", 0.0) / n_nodi
            fy = cv.caratteristiche.get("fy", 0.0) / n_nodi
            fz = cv.caratteristiche.get("fz", 0.0) / n_nodi
            for nid in nodi_interni:
                if nid in self._risultato.nodi_caricati:
                    self._risultato.nodi_caricati[nid]["fx"] += fx
                    self._risultato.nodi_caricati[nid]["fy"] += fy
                    self._risultato.nodi_caricati[nid]["fz"] += fz
                else:
                    self._risultato.nodi_caricati[nid] = {
                        "fx": fx, "fy": fy, "fz": fz,
                    }

    def _trova_nodi_nel_volume(self, cv: CaricoVincolo) -> list[int]:
        """
        Trova tutti i nodi della mesh che cadono all'interno del volume
        del parallelepipedo definito dal CaricoVincolo.
        Supporta vertici custom (8 vertici del parallelepipedo trasformato).
        """
        verts_world = cv.get_vertices_world()
        if len(verts_world) < 8:
            return []

        v = np.array(verts_world, dtype=float)

        # Calcola bounding box per pre-filtering veloce
        bb_min = v.min(axis=0) - 1e-6
        bb_max = v.max(axis=0) + 1e-6

        # Costruisci le 6 facce del parallelepipedo come piani
        # e verifica che ogni nodo sia "dentro" tutti i piani
        # Facce: bottom(0123), top(4567), front(0154), back(3267), left(0374), right(1265)
        face_indices = [
            [0, 3, 2, 1],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 4, 7, 3],  # left
            [1, 2, 6, 5],  # right
        ]

        # Centro del volume
        centro = v.mean(axis=0)

        # Calcola normali delle facce (orientate verso l'interno)
        piani = []
        for fi in face_indices:
            p0 = v[fi[0]]
            p1 = v[fi[1]]
            p2 = v[fi[2]]
            normale = np.cross(p1 - p0, p2 - p0)
            n_len = np.linalg.norm(normale)
            if n_len < 1e-12:
                continue
            normale /= n_len
            # Assicura che la normale punti verso l'interno
            if np.dot(normale, centro - p0) < 0:
                normale = -normale
            piani.append((normale, p0))

        if not piani:
            return []

        nodi_trovati = []
        for nid, pos in self._risultato.nodi.items():
            p = np.array(pos)
            # Pre-filter con bounding box
            if np.any(p < bb_min) or np.any(p > bb_max):
                continue
            # Verifica con tutti i piani
            dentro = True
            for normale, p0 in piani:
                if np.dot(normale, p - p0) < -1e-6:
                    dentro = False
                    break
            if dentro:
                nodi_trovati.append(nid)

        return nodi_trovati
