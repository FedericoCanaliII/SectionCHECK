"""
scrittore_inp.py – Scrive file .inp compatibili con CalculiX.

Genera:
  - Nodi (*NODE)
  - Elementi C3D8 (*ELEMENT, TYPE=C3D8)
  - Elementi T3D2 (*ELEMENT, TYPE=T3D2) per barre e staffe
  - Set di nodi e elementi (*NSET, *ELSET)
  - Materiali (*MATERIAL, *ELASTIC)
  - Sezioni solide (*SOLID SECTION)
  - TIE constraints (*TIE, *SURFACE)
  - Boundary conditions (*BOUNDARY) con cedimenti
  - Carichi concentrati (*CLOAD)
  - Step di analisi (*STEP, *STATIC)
"""
from __future__ import annotations

import os
from typing import Optional

from .generatore_mesh import RisultatoMesh


class ScrittoreINP:
    """
    Scrive un file .inp per CalculiX a partire da un RisultatoMesh.
    """

    def __init__(self, risultato: RisultatoMesh, materiali: dict,
                 nome_elemento: str = "elemento"):
        self._mesh = risultato
        self._materiali = materiali  # {nome_mat: dict_materiale}
        self._nome = nome_elemento

    def scrivi(self, percorso_file: str) -> bool:
        """Scrive il file .inp. Ritorna True se riuscito."""
        try:
            os.makedirs(os.path.dirname(percorso_file), exist_ok=True)
            with open(percorso_file, "w", encoding="utf-8") as f:
                self._scrivi_header(f)
                self._scrivi_nodi(f)
                self._scrivi_elementi_hex(f)
                self._scrivi_elementi_beam(f)
                self._scrivi_nset_elset(f)
                self._scrivi_materiali(f)
                self._scrivi_sezioni(f)
                self._scrivi_tie(f)
                self._scrivi_step(f)
            return True
        except Exception as e:
            print(f"ERR  Scrittura .inp fallita: {e}")
            return False

    # ------------------------------------------------------------------

    def _scrivi_header(self, f):
        f.write(f"** CalculiX input file generato da SectionCHECK\n")
        f.write(f"** Elemento: {self._nome}\n")
        f.write(f"** Nodi: {self._mesh.n_nodi}  Elementi: {self._mesh.n_elementi}\n")
        f.write(f"**\n")

    def _scrivi_nodi(self, f):
        f.write("*NODE, NSET=NALL\n")
        for nid in sorted(self._mesh.nodi.keys()):
            x, y, z = self._mesh.nodi[nid]
            f.write(f"{nid}, {x:.8e}, {y:.8e}, {z:.8e}\n")

    def _scrivi_elementi_hex(self, f):
        if not self._mesh.elementi_hex:
            return
        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL_HEX\n")
        for eid in sorted(self._mesh.elementi_hex.keys()):
            nodi = self._mesh.elementi_hex[eid]
            nodi_str = ", ".join(str(n) for n in nodi)
            f.write(f"{eid}, {nodi_str}\n")

    def _scrivi_elementi_beam(self, f):
        if not self._mesh.elementi_beam:
            return
        f.write("*ELEMENT, TYPE=T3D2, ELSET=EALL_TRUSS\n")
        for eid in sorted(self._mesh.elementi_beam.keys()):
            nodi = self._mesh.elementi_beam[eid]
            f.write(f"{eid}, {nodi[0]}, {nodi[1]}\n")

    def _scrivi_nset_elset(self, f):
        """Scrive NSET e ELSET per ogni oggetto."""
        for obj_id, nodi_set in self._mesh.nodi_per_oggetto.items():
            tipo = self._mesh.tipo_oggetto.get(obj_id, "unknown")
            nome_set = f"OBJ_{obj_id}_{tipo.upper()}"

            # NSET
            f.write(f"*NSET, NSET=N_{nome_set}\n")
            self._scrivi_lista_ids(f, sorted(nodi_set))

            # ELSET
            elems = self._mesh.elementi_per_oggetto.get(obj_id, set())
            if elems:
                f.write(f"*ELSET, ELSET=E_{nome_set}\n")
                self._scrivi_lista_ids(f, sorted(elems))

        # Set per nodi vincolati
        if self._mesh.nodi_vincolati:
            f.write("*NSET, NSET=N_VINCOLATI\n")
            self._scrivi_lista_ids(f, sorted(self._mesh.nodi_vincolati.keys()))

        # Set per nodi caricati
        if self._mesh.nodi_caricati:
            f.write("*NSET, NSET=N_CARICATI\n")
            self._scrivi_lista_ids(f, sorted(self._mesh.nodi_caricati.keys()))

    def _scrivi_materiali(self, f):
        """Scrive le definizioni dei materiali."""
        materiali_scritti = set()

        for obj_id, nome_mat in self._mesh.materiale_oggetto.items():
            if not nome_mat or nome_mat in materiali_scritti:
                continue
            materiali_scritti.add(nome_mat)

            dati = self._materiali.get(nome_mat, {})
            nome_safe = nome_mat.replace(" ", "_").replace(".", "_").replace("/", "_")

            f.write(f"**\n")
            f.write(f"*MATERIAL, NAME=MAT_{nome_safe}\n")

            # Cerca proprieta' elastiche (database in MPa -> CalculiX in Pa)
            E_mpa = dati.get("m_elastico", dati.get("E", dati.get("modulo_elastico",
                             dati.get("Ecm", dati.get("Es", 30000.0)))))
            E_pa = float(E_mpa) * 1e6
            nu = dati.get("nu", dati.get("poisson", 0.2))

            f.write(f"*ELASTIC\n")
            f.write(f"{E_pa:.4e}, {float(nu):.4f}\n")

            # Densita' (se disponibile)
            rho = dati.get("rho", dati.get("densita", dati.get("peso_specifico")))
            if rho is not None:
                f.write(f"*DENSITY\n")
                f.write(f"{float(rho):.4e}\n")

        # Materiale di default per oggetti senza materiale
        f.write(f"**\n")
        f.write(f"*MATERIAL, NAME=MAT_DEFAULT\n")
        f.write(f"*ELASTIC\n")
        f.write(f"3.0000e+10, 0.2000\n")

    def _scrivi_sezioni(self, f):
        """Assegna sezioni solide agli ELSET di ogni oggetto."""
        import math as _math
        for obj_id in self._mesh.elementi_per_oggetto:
            tipo = self._mesh.tipo_oggetto.get(obj_id, "carpenteria")
            nome_set = f"OBJ_{obj_id}_{tipo.upper()}"
            nome_mat = self._mesh.materiale_oggetto.get(obj_id, "")

            if nome_mat:
                nome_mat_safe = f"MAT_{nome_mat.replace(' ', '_').replace('.', '_').replace('/', '_')}"
            else:
                nome_mat_safe = "MAT_DEFAULT"

            if tipo == "carpenteria":
                f.write(f"*SOLID SECTION, ELSET=E_{nome_set}, MATERIAL={nome_mat_safe}\n")
                f.write(f"\n")
            elif tipo in ("barra", "staffa"):
                # T3D2: sezione solida con area circolare
                diametro = self._mesh.diametro_oggetto.get(obj_id, 0.016)
                area = _math.pi * diametro**2 / 4.0
                f.write(f"*SOLID SECTION, ELSET=E_{nome_set}, MATERIAL={nome_mat_safe}\n")
                f.write(f"{area:.10f}\n")

    def _scrivi_tie(self, f):
        """
        Scrive i vincoli *TIE tra le superfici di oggetti di carpenteria contigui
        per incollare mesh non conformi.
        """
        if not self._mesh.tie_constraints:
            return

        f.write(f"**\n** TIE CONSTRAINTS - Superficie-Superficie tra carpenteria\n")
        
        for idx, tie in enumerate(self._mesh.tie_constraints):
            master_obj = tie.get("master_obj")
            slave_obj = tie.get("slave_obj")
            master_faces = tie.get("master_faces", [])
            slave_faces = tie.get("slave_faces", [])

            if not master_faces or not slave_faces:
                continue
            
            surf_master = f"SURF_TIE_M_{master_obj}_{idx}"
            surf_slave = f"SURF_TIE_S_{slave_obj}_{idx}"
            tie_name = f"TIE_CARP_{idx}"

            # Scrittura superficie Master
            f.write(f"*SURFACE, NAME={surf_master}, TYPE=ELEMENT\n")
            for eid, sname in master_faces:
                f.write(f"{eid}, {sname}\n")

            # Scrittura superficie Slave
            f.write(f"*SURFACE, NAME={surf_slave}, TYPE=ELEMENT\n")
            for eid, sname in slave_faces:
                f.write(f"{eid}, {sname}\n")

            # Vincolo TIE
            f.write(f"*TIE, NAME={tie_name}, POSITION TOLERANCE=0.001\n")
            f.write(f"{surf_slave}, {surf_master}\n")

    def _scrivi_step(self, f):
        """Scrive lo step di analisi statica con carichi e vincoli."""
        f.write(f"**\n")
        f.write(f"** ============================================\n")
        f.write(f"** STEP DI ANALISI\n")
        f.write(f"** ============================================\n")
        f.write(f"*STEP\n")
        f.write(f"*STATIC\n")

        # Boundary conditions (vincoli)
        if self._mesh.nodi_vincolati:
            f.write(f"**\n")
            f.write(f"** VINCOLI\n")
            f.write(f"*BOUNDARY\n")
            for nid, ced in self._mesh.nodi_vincolati.items():
                sx = ced.get("sx", 0.0)
                sy = ced.get("sy", 0.0)
                sz = ced.get("sz", 0.0)
                f.write(f"{nid}, 1, 1, {sx:.8e}\n")
                f.write(f"{nid}, 2, 2, {sy:.8e}\n")
                f.write(f"{nid}, 3, 3, {sz:.8e}\n")

        # Carichi concentrati
        if self._mesh.nodi_caricati:
            f.write(f"**\n")
            f.write(f"** CARICHI (kN -> N)\n")
            f.write(f"*CLOAD\n")
            for nid, forze in self._mesh.nodi_caricati.items():
                fx = forze.get("fx", 0.0) * 1e3  # kN -> N
                fy = forze.get("fy", 0.0) * 1e3
                fz = forze.get("fz", 0.0) * 1e3
                if abs(fx) > 1e-9:
                    f.write(f"{nid}, 1, {fx:.8e}\n")
                if abs(fy) > 1e-9:
                    f.write(f"{nid}, 2, {fy:.8e}\n")
                if abs(fz) > 1e-9:
                    f.write(f"{nid}, 3, {fz:.8e}\n")

        # Output richiesti
        f.write(f"**\n")
        f.write(f"*NODE FILE\n")
        f.write(f"U\n")
        f.write(f"*EL FILE\n")
        f.write(f"S, E\n")
        f.write(f"*END STEP\n")

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    @staticmethod
    def _scrivi_lista_ids(f, ids: list[int], per_riga: int = 16):
        """Scrive una lista di ID in formato CalculiX (max per_riga per riga)."""
        for i in range(0, len(ids), per_riga):
            chunk = ids[i:i + per_riga]
            f.write(", ".join(str(x) for x in chunk))
            if i + per_riga < len(ids):
                f.write(",\n")
            else:
                f.write("\n")
