"""
generatore_mesh_struttura.py
----------------------------
Discretizza la struttura per l'analisi FEM:

  - Ogni asta (beam) viene suddivisa in N sotto-elementi, creando nodi
    intermedi lungo l'asta.
  - Le shell vengono suddivise in modo che i loro nodi sul bordo
    coincidano con i nodi intermedi delle aste su cui poggiano.
  - Tutti i nodi (originali + intermedi) vengono raccolti in un unico
    dizionario con tag univoci.

Il risultato e' un oggetto ``MeshStruttura`` pronto per essere passato
al motore di analisi OpenSees.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
#  DATI MESH
# ==============================================================================

@dataclass
class NodoMesh:
    """Nodo della mesh discretizzata."""
    tag: int
    x: float
    y: float
    z: float


@dataclass
class ElementoBeamMesh:
    """Sotto-elemento beam nella mesh discretizzata."""
    tag: int
    nodo_i: int       # tag nodo inizio
    nodo_j: int       # tag nodo fine
    asta_originale: int   # id dell'asta originale
    sezione: str      # riferimento sezione
    lunghezza: float  # lunghezza sotto-elemento


@dataclass
class ElementoShellMesh:
    """Elemento shell nella mesh discretizzata."""
    tag: int
    nodi: list[int]        # tag dei nodi (3 o 4)
    shell_originale: int   # id della shell originale
    spessore: float
    materiale: str


@dataclass
class MeshStruttura:
    """Mesh discretizzata completa della struttura."""
    nodi: dict[int, NodoMesh] = field(default_factory=dict)
    elementi_beam: list[ElementoBeamMesh] = field(default_factory=list)
    elementi_shell: list[ElementoShellMesh] = field(default_factory=list)
    vincoli: dict[int, list[int]] = field(default_factory=dict)
    carichi_nodali: list[tuple] = field(default_factory=list)
    carichi_distribuiti: list[tuple] = field(default_factory=list)
    carichi_shell: list[tuple] = field(default_factory=list)

    # Mappa: tag_nodo_mesh -> id_nodo_originale (per nodi originali)
    mappa_nodi_originali: dict[int, int] = field(default_factory=dict)
    # Mappa: id_asta_originale -> lista tag sotto-elementi
    mappa_aste: dict[int, list[int]] = field(default_factory=dict)
    # Mappa: id_asta_originale -> lista tag nodi (ordinati da i a j)
    mappa_nodi_aste: dict[int, list[int]] = field(default_factory=dict)

    @property
    def n_nodi(self) -> int:
        return len(self.nodi)

    @property
    def n_elementi(self) -> int:
        return len(self.elementi_beam) + len(self.elementi_shell)


# ==============================================================================
#  GENERATORE
# ==============================================================================

class GeneratoreMeshStruttura:
    """Genera la mesh discretizzata della struttura."""

    def __init__(self, n_divisioni: int = 5) -> None:
        self.n_divisioni = max(2, n_divisioni)
        self._next_nodo_tag: int = 1
        self._next_elem_tag: int = 1

    def genera(self, dati: dict, progress_cb=None) -> MeshStruttura:
        """
        Genera la mesh a partire dai dati parsati della struttura.

        Parametri
        ---------
        dati : dict
            Output di ``parse_struttura``.
        progress_cb : callable, optional
            Callback(int) con avanzamento 0-100.

        Ritorna
        -------
        MeshStruttura
        """
        mesh = MeshStruttura()
        self._next_nodo_tag = 1
        self._next_elem_tag = 1

        nodi_orig = dati.get("nodi", {})
        aste_orig = dati.get("aste", {})
        shell_orig = dati.get("shell", {})
        vincoli_orig = dati.get("vincoli", {})

        if progress_cb:
            progress_cb(5)

        # ---- 1. Crea nodi originali ----
        # Mappa: id_nodo_originale -> tag_nodo_mesh
        mappa_id_to_tag: dict[int, int] = {}
        for nid in sorted(nodi_orig.keys()):
            coords = nodi_orig[nid]
            tag = self._nuovo_nodo_tag()
            mesh.nodi[tag] = NodoMesh(tag, float(coords[0]),
                                      float(coords[1]), float(coords[2]))
            mappa_id_to_tag[nid] = tag
            mesh.mappa_nodi_originali[tag] = nid

        if progress_cb:
            progress_cb(15)

        # ---- 2. Discretizza aste ----
        for bid in sorted(aste_orig.keys()):
            asta = aste_orig[bid]
            ni_orig = asta["nodo_i"]
            nj_orig = asta["nodo_j"]
            sezione = asta.get("sezione", "")

            if ni_orig not in mappa_id_to_tag or nj_orig not in mappa_id_to_tag:
                print(f"WARN  Asta {bid}: nodi non trovati, saltata.")
                continue

            tag_i = mappa_id_to_tag[ni_orig]
            tag_j = mappa_id_to_tag[nj_orig]

            nodo_i = mesh.nodi[tag_i]
            nodo_j = mesh.nodi[tag_j]

            # Crea nodi intermedi
            N = self.n_divisioni
            nodi_asta = [tag_i]

            for k in range(1, N):
                t = k / N
                x = nodo_i.x + t * (nodo_j.x - nodo_i.x)
                y = nodo_i.y + t * (nodo_j.y - nodo_i.y)
                z = nodo_i.z + t * (nodo_j.z - nodo_i.z)
                tag = self._nuovo_nodo_tag()
                mesh.nodi[tag] = NodoMesh(tag, x, y, z)
                nodi_asta.append(tag)

            nodi_asta.append(tag_j)

            # Crea sotto-elementi beam
            L_tot = math.sqrt((nodo_j.x - nodo_i.x) ** 2 +
                              (nodo_j.y - nodo_i.y) ** 2 +
                              (nodo_j.z - nodo_i.z) ** 2)
            L_sub = L_tot / N

            elem_tags = []
            for k in range(N):
                etag = self._nuovo_elem_tag()
                elem = ElementoBeamMesh(
                    tag=etag,
                    nodo_i=nodi_asta[k],
                    nodo_j=nodi_asta[k + 1],
                    asta_originale=bid,
                    sezione=sezione,
                    lunghezza=L_sub,
                )
                mesh.elementi_beam.append(elem)
                elem_tags.append(etag)

            mesh.mappa_aste[bid] = elem_tags
            mesh.mappa_nodi_aste[bid] = nodi_asta

        if progress_cb:
            progress_cb(45)

        # ---- 3. Mappa nodi intermedi lungo le aste (per le shell) ----
        # Per ogni coppia di nodi originali connessi da un'asta,
        # salva i nodi intermedi
        edge_nodes: dict[tuple[int, int], list[int]] = {}
        for bid, nodi_asta in mesh.mappa_nodi_aste.items():
            asta = aste_orig[bid]
            ni = asta["nodo_i"]
            nj = asta["nodo_j"]
            key = (min(ni, nj), max(ni, nj))
            if ni < nj:
                edge_nodes[key] = nodi_asta
            else:
                edge_nodes[key] = list(reversed(nodi_asta))

        # ---- 4. Discretizza shell ----
        for sid in sorted(shell_orig.keys()):
            shell = shell_orig[sid]
            nodi_shell_orig = shell["nodi"]
            spessore = float(shell.get("spessore", 0.20))
            materiale = shell.get("materiale", "")

            n_nodi_shell = len(nodi_shell_orig)

            # Verifica che tutti i nodi shell esistano
            all_ok = all(n in mappa_id_to_tag for n in nodi_shell_orig)
            if not all_ok:
                print(f"WARN  Shell {sid}: nodi non trovati, saltata.")
                continue

            if n_nodi_shell == 4:
                self._discretizza_shell_quad(
                    mesh, sid, nodi_shell_orig, spessore, materiale,
                    mappa_id_to_tag, edge_nodes, nodi_orig
                )
            elif n_nodi_shell == 3:
                self._discretizza_shell_tri(
                    mesh, sid, nodi_shell_orig, spessore, materiale,
                    mappa_id_to_tag, edge_nodes, nodi_orig
                )
            else:
                print(f"WARN  Shell {sid}: {n_nodi_shell} nodi non supportati.")

        if progress_cb:
            progress_cb(70)

        # ---- 5. Vincoli ----
        for nid, vals in vincoli_orig.items():
            if nid in mappa_id_to_tag:
                tag = mappa_id_to_tag[nid]
                mesh.vincoli[tag] = vals[:6]

        # ---- 6. Carichi ----
        # Carichi nodali
        for carico in dati.get("carichi_nodali", []):
            nid, fx, fy, fz = carico[0], carico[1], carico[2], carico[3]
            if nid in mappa_id_to_tag:
                tag = mappa_id_to_tag[nid]
                mesh.carichi_nodali.append((tag, fx, fy, fz))

        # Carichi distribuiti sulle aste
        for carico in dati.get("carichi_distribuiti", []):
            bid, wx, wy, wz = carico[0], carico[1], carico[2], carico[3]
            mesh.carichi_distribuiti.append((bid, wx, wy, wz))

        # Carichi sulle shell
        for carico in dati.get("carichi_shell", []):
            s_id, qx, qy, qz = carico[0], carico[1], carico[2], carico[3]
            mesh.carichi_shell.append((s_id, qx, qy, qz))

        if progress_cb:
            progress_cb(100)

        return mesh

    # ------------------------------------------------------------------
    #  DISCRETIZZAZIONE SHELL QUAD
    # ------------------------------------------------------------------

    def _discretizza_shell_quad(self, mesh: MeshStruttura, sid: int,
                                nodi_orig_ids: list[int],
                                spessore: float, materiale: str,
                                mappa_id_to_tag: dict, edge_nodes: dict,
                                nodi_orig: dict) -> None:
        """
        Discretizza una shell quadrilatera in NxN sotto-shell, dove N
        e' il numero di divisioni.  I nodi lungo i bordi che coincidono
        con aste vengono allineati ai nodi discretizzati dell'asta.
        """
        N = self.n_divisioni
        n1, n2, n3, n4 = nodi_orig_ids

        # Coordinate dei 4 vertici originali
        c1 = np.array(nodi_orig[n1], dtype=float)
        c2 = np.array(nodi_orig[n2], dtype=float)
        c3 = np.array(nodi_orig[n3], dtype=float)
        c4 = np.array(nodi_orig[n4], dtype=float)

        # Griglia (N+1) x (N+1) di nodi
        grid_tags = [[0] * (N + 1) for _ in range(N + 1)]

        for j in range(N + 1):
            for i in range(N + 1):
                s = i / N
                t = j / N

                # Interpolazione bilineare
                pos = ((1 - s) * (1 - t) * c1 +
                       s * (1 - t) * c2 +
                       s * t * c3 +
                       (1 - s) * t * c4)

                # Vertici originali -> riusa tag esistenti
                is_corner = False
                if i == 0 and j == 0:
                    grid_tags[j][i] = mappa_id_to_tag[n1]; is_corner = True
                elif i == N and j == 0:
                    grid_tags[j][i] = mappa_id_to_tag[n2]; is_corner = True
                elif i == N and j == N:
                    grid_tags[j][i] = mappa_id_to_tag[n3]; is_corner = True
                elif i == 0 and j == N:
                    grid_tags[j][i] = mappa_id_to_tag[n4]; is_corner = True

                if is_corner:
                    continue

                # Bordi: cerca se coincide con un'asta discretizzata
                on_edge = False
                if j == 0:
                    on_edge = self._nodo_bordo_asta(
                        mesh, grid_tags, j, i, n1, n2, i, N,
                        edge_nodes, mappa_id_to_tag)
                elif j == N:
                    on_edge = self._nodo_bordo_asta(
                        mesh, grid_tags, j, i, n4, n3, i, N,
                        edge_nodes, mappa_id_to_tag)
                elif i == 0:
                    on_edge = self._nodo_bordo_asta(
                        mesh, grid_tags, j, i, n1, n4, j, N,
                        edge_nodes, mappa_id_to_tag)
                elif i == N:
                    on_edge = self._nodo_bordo_asta(
                        mesh, grid_tags, j, i, n2, n3, j, N,
                        edge_nodes, mappa_id_to_tag)

                if not on_edge:
                    # Nodo interno: crea nuovo
                    tag = self._nuovo_nodo_tag()
                    mesh.nodi[tag] = NodoMesh(tag, pos[0], pos[1], pos[2])
                    grid_tags[j][i] = tag

        # Crea sotto-elementi shell
        for j in range(N):
            for i in range(N):
                t1 = grid_tags[j][i]
                t2 = grid_tags[j][i + 1]
                t3 = grid_tags[j + 1][i + 1]
                t4 = grid_tags[j + 1][i]

                if t1 == 0 or t2 == 0 or t3 == 0 or t4 == 0:
                    continue

                etag = self._nuovo_elem_tag()
                mesh.elementi_shell.append(ElementoShellMesh(
                    tag=etag,
                    nodi=[t1, t2, t3, t4],
                    shell_originale=sid,
                    spessore=spessore,
                    materiale=materiale,
                ))

    def _nodo_bordo_asta(self, mesh, grid_tags, j, i,
                         na, nb, idx, N, edge_nodes, mappa_id_to_tag) -> bool:
        """
        Se il bordo (na->nb) corrisponde ad un'asta discretizzata, assegna
        il nodo intermedio corrispondente.  Ritorna True se assegnato.
        """
        key = (min(na, nb), max(na, nb))
        if key not in edge_nodes:
            return False

        nodi_asta = edge_nodes[key]
        n_asta_nodes = len(nodi_asta)

        # Mappa l'indice della shell all'indice dell'asta
        # Se na < nb i nodi vanno nello stesso verso, altrimenti invertiti
        if na == key[0]:
            pos_idx = idx
        else:
            pos_idx = N - idx

        # Calcola l'indice corrispondente nell'asta
        asta_idx = int(round(pos_idx * (n_asta_nodes - 1) / N))
        asta_idx = max(0, min(asta_idx, n_asta_nodes - 1))

        tag_asta = nodi_asta[asta_idx]
        if tag_asta in mesh.nodi:
            grid_tags[j][i] = tag_asta
            return True
        return False

    # ------------------------------------------------------------------
    #  DISCRETIZZAZIONE SHELL TRI
    # ------------------------------------------------------------------

    def _discretizza_shell_tri(self, mesh: MeshStruttura, sid: int,
                               nodi_orig_ids: list[int],
                               spessore: float, materiale: str,
                               mappa_id_to_tag: dict, edge_nodes: dict,
                               nodi_orig: dict) -> None:
        """Discretizza una shell triangolare in sotto-elementi."""
        N = self.n_divisioni
        n1, n2, n3 = nodi_orig_ids

        c1 = np.array(nodi_orig[n1], dtype=float)
        c2 = np.array(nodi_orig[n2], dtype=float)
        c3 = np.array(nodi_orig[n3], dtype=float)

        # Crea griglia di nodi con coordinate baricentriche
        # Nodi su griglia triangolare: (i, j) con i+j <= N
        tri_tags: dict[tuple[int, int], int] = {}

        for j in range(N + 1):
            for i in range(N + 1 - j):
                s = i / N
                t = j / N
                u = 1.0 - s - t

                pos = u * c1 + s * c2 + t * c3

                # Vertici
                if i == 0 and j == 0:
                    tri_tags[(i, j)] = mappa_id_to_tag[n1]
                    continue
                elif i == N and j == 0:
                    tri_tags[(i, j)] = mappa_id_to_tag[n2]
                    continue
                elif i == 0 and j == N:
                    tri_tags[(i, j)] = mappa_id_to_tag[n3]
                    continue

                # Bordi: controlla aste
                assigned = False
                if j == 0:  # bordo n1-n2
                    assigned = self._nodo_bordo_asta_tri(
                        mesh, tri_tags, i, j, n1, n2, i, N,
                        edge_nodes, mappa_id_to_tag)
                elif i == 0:  # bordo n1-n3
                    assigned = self._nodo_bordo_asta_tri(
                        mesh, tri_tags, i, j, n1, n3, j, N,
                        edge_nodes, mappa_id_to_tag)
                elif i + j == N:  # bordo n2-n3
                    assigned = self._nodo_bordo_asta_tri(
                        mesh, tri_tags, i, j, n2, n3, j, N,
                        edge_nodes, mappa_id_to_tag)

                if not assigned:
                    tag = self._nuovo_nodo_tag()
                    mesh.nodi[tag] = NodoMesh(tag, pos[0], pos[1], pos[2])
                    tri_tags[(i, j)] = tag

        # Crea sotto-elementi
        for j in range(N):
            for i in range(N - j):
                t1 = tri_tags.get((i, j), 0)
                t2 = tri_tags.get((i + 1, j), 0)
                t3 = tri_tags.get((i, j + 1), 0)
                if t1 and t2 and t3:
                    etag = self._nuovo_elem_tag()
                    mesh.elementi_shell.append(ElementoShellMesh(
                        tag=etag, nodi=[t1, t2, t3],
                        shell_originale=sid, spessore=spessore,
                        materiale=materiale))

                # Triangolo invertito
                if i + j + 1 < N:
                    t4 = tri_tags.get((i + 1, j + 1), 0)
                    if t2 and t4 and t3:
                        etag = self._nuovo_elem_tag()
                        mesh.elementi_shell.append(ElementoShellMesh(
                            tag=etag, nodi=[t2, t4, t3],
                            shell_originale=sid, spessore=spessore,
                            materiale=materiale))

    def _nodo_bordo_asta_tri(self, mesh, tri_tags, i, j,
                             na, nb, idx, N, edge_nodes,
                             mappa_id_to_tag) -> bool:
        """Come _nodo_bordo_asta ma per griglia triangolare."""
        key = (min(na, nb), max(na, nb))
        if key not in edge_nodes:
            return False
        nodi_asta = edge_nodes[key]
        n_asta = len(nodi_asta)
        if na == key[0]:
            pos_idx = idx
        else:
            pos_idx = N - idx
        asta_idx = int(round(pos_idx * (n_asta - 1) / N))
        asta_idx = max(0, min(asta_idx, n_asta - 1))
        tag_asta = nodi_asta[asta_idx]
        if tag_asta in mesh.nodi:
            tri_tags[(i, j)] = tag_asta
            return True
        return False

    # ------------------------------------------------------------------
    #  UTILITY
    # ------------------------------------------------------------------

    def _nuovo_nodo_tag(self) -> int:
        t = self._next_nodo_tag
        self._next_nodo_tag += 1
        return t

    def _nuovo_elem_tag(self) -> int:
        t = self._next_elem_tag
        self._next_elem_tag += 1
        return t
